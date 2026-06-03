import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, make_scorer, r2_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.multioutput import MultiOutputRegressor

from tabnado.tasks import encode_classification_target, json_safe, resolve_task


def _import_catboost():
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CatBoost backend requested but catboost is not installed. "
            "Install/sync TabNado with the catboost dependency first."
        ) from exc
    return CatBoostClassifier, CatBoostRegressor


def _default_best_hp(param_dist: dict) -> dict:
    best_hp = {}
    for key, values in param_dist.items():
        first = values[0]
        if isinstance(first, np.floating):
            first = float(first)
        elif isinstance(first, np.integer):
            first = int(first)
        best_hp[key.replace("estimator__", "")] = first
    return best_hp


def _stratified_fraction_sample(
    data: pd.DataFrame,
    target_col: str,
    frac: float,
    seed: int,
) -> pd.DataFrame:
    frac = min(max(float(frac), 0.0), 1.0)
    if frac >= 1.0:
        return data

    parts = []
    for _, group in data.groupby(target_col, sort=False, observed=False):
        n = max(1, int(round(len(group) * frac)))
        parts.append(group.sample(n=min(n, len(group)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed)


def _cv_for_tune_data(tune_data: pd.DataFrame, seed: int):
    if hasattr(tune_data.index, "get_level_values"):
        try:
            contigs = tune_data.index.get_level_values("contig").astype(str)
        except KeyError:
            contigs = tune_data.index.astype(str).str.split(":").str[0]
    else:
        contigs = tune_data.index.astype(str).str.split(":").str[0]
    groups = contigs.values

    n_unique_groups = len(np.unique(groups))
    if n_unique_groups >= 2:
        n_splits = min(3, n_unique_groups)
        return (
            GroupKFold(n_splits=n_splits),
            {"groups": groups},
            f"{n_splits}-fold GroupKFold by chromosome",
            n_splits,
        )

    n_splits = min(3, len(tune_data))
    return (
        KFold(n_splits=n_splits, shuffle=True, random_state=seed),
        {},
        f"{n_splits}-fold KFold",
        n_splits,
    )


def sweep_catboost(
    feature_cols: list[str],
    target_cols: list[str],
    train_data: pd.DataFrame,
    n_sweeps: int = 20,
    sweep_fraction: float = 0.1,
    RES_DIR: str = "results",
    seed: int = 42,
    n_jobs: int = 1,
    wandb_cfg=None,
    TASK: str = "regression",
    **kwargs,
) -> dict:
    """Randomised HP search for CatBoost."""
    CatBoostClassifier, CatBoostRegressor = _import_catboost()
    Path(RES_DIR).mkdir(parents=True, exist_ok=True)
    task = resolve_task(TASK, train_data, target_cols)

    common_params = {
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
    }
    param_dist = {
        "depth": [3, 4, 5, 6, 8],
        "learning_rate": np.logspace(np.log10(0.01), np.log10(0.2), 10),
        "iterations": [300, 600, 1000],
        "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
        "random_strength": [0.0, 0.5, 1.0],
    }

    if task == "classification":
        encoded = encode_classification_target(train_data, target_cols)
        tune_data = _stratified_fraction_sample(
            train_data,
            encoded.target_col,
            sweep_fraction,
            seed,
        )
        X_tune = tune_data[feature_cols]
        class_lookup = {label: idx for idx, label in enumerate(encoded.classes)}
        y_tune = np.array(
            [class_lookup[label] for label in tune_data[encoded.target_col].astype(str)]
        )
        estimator = CatBoostClassifier(
            **common_params,
            loss_function=(
                "Logloss" if encoded.problem_type == "binary" else "MultiClass"
            ),
        )
        scoring = make_scorer(f1_score, average="macro")
        metric_name = "macro-F1"
    else:
        tune_data = train_data.sample(
            frac=min(sweep_fraction, 1.0),
            random_state=seed,
        )
        X_tune = tune_data[feature_cols]
        base = CatBoostRegressor(**common_params, loss_function="RMSE")
        if len(target_cols) == 1:
            estimator = base
            y_tune = tune_data[target_cols].values.ravel()
        else:
            estimator = MultiOutputRegressor(base, n_jobs=1)
            param_dist = {f"estimator__{k}": v for k, v in param_dist.items()}
            y_tune = tune_data[target_cols].values
        scoring = make_scorer(
            r2_score,
            greater_is_better=True,
            multioutput="uniform_average",
        )
        metric_name = "R²"

    if len(X_tune) < 2 or n_sweeps <= 0:
        best_hp = _default_best_hp(param_dist)
        logger.warning("CatBoost HP sweep skipped: using deterministic defaults.")
        out_path = Path(RES_DIR) / "best_hyperparameters.json"
        with open(out_path, "w") as f:
            json.dump(json_safe(best_hp), f, indent=2)
        logger.info(f"Saved best hyperparameters to {out_path}")
        return best_hp

    if task == "classification":
        unique_classes, class_counts = np.unique(y_tune, return_counts=True)
        if len(unique_classes) < 2:
            best_hp = _default_best_hp(param_dist)
            logger.warning(
                "CatBoost classifier HP sweep skipped: tuning sample contains "
                "fewer than two classes. Using deterministic defaults."
            )
            out_path = Path(RES_DIR) / "best_hyperparameters.json"
            with open(out_path, "w") as f:
                json.dump(json_safe(best_hp), f, indent=2)
            logger.info(f"Saved best hyperparameters to {out_path}")
            return best_hp

        n_splits = min(3, int(class_counts.min()))
        if n_splits < 2:
            best_hp = _default_best_hp(param_dist)
            logger.warning(
                "CatBoost classifier HP sweep skipped: every class needs at least "
                "two samples for stratified CV. Using deterministic defaults."
            )
            out_path = Path(RES_DIR) / "best_hyperparameters.json"
            with open(out_path, "w") as f:
                json.dump(json_safe(best_hp), f, indent=2)
            logger.info(f"Saved best hyperparameters to {out_path}")
            return best_hp

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fit_kwargs = {}
        cv_desc = f"{n_splits}-fold StratifiedKFold"
    else:
        cv, fit_kwargs, cv_desc, n_splits = _cv_for_tune_data(tune_data, seed)
    if len(X_tune) < max(10, n_splits * 4):
        best_hp = _default_best_hp(param_dist)
        logger.warning(
            "CatBoost HP sweep skipped: {} samples is too few for stable {}. "
            "Using deterministic defaults.".format(len(X_tune), cv_desc)
        )
        out_path = Path(RES_DIR) / "best_hyperparameters.json"
        with open(out_path, "w") as f:
            json.dump(json_safe(best_hp), f, indent=2)
        logger.info(f"Saved best hyperparameters to {out_path}")
        return best_hp

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_sweeps,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        random_state=seed,
        refit=False,
    )
    logger.info(
        f"CatBoost HP sweep: {n_sweeps} iterations on {len(X_tune):,} regions "
        f"({sweep_fraction:.0%} of train), {cv_desc}"
    )
    search.fit(X_tune, y_tune, **fit_kwargs)
    best_hp = {
        k.replace("estimator__", ""): v for k, v in search.best_params_.items()
    }
    if np.isnan(search.best_score_):
        logger.warning("CatBoost sweep scores are NaN; using deterministic defaults.")
        best_hp = _default_best_hp(param_dist)
    else:
        logger.info(f"Best sweep {metric_name}={search.best_score_:.4f}")

    out_path = Path(RES_DIR) / "best_hyperparameters.json"
    with open(out_path, "w") as f:
        json.dump(json_safe(best_hp), f, indent=2)
    logger.info(f"Saved best hyperparameters to {out_path}")
    return best_hp
