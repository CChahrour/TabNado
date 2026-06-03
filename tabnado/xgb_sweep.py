import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.metrics import make_scorer, r2_score
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.multioutput import MultiOutputRegressor

from tabnado.data import load_data
from tabnado.params import PipelineParams
from tabnado.tasks import encode_classification_target, json_safe, resolve_task
from tabnado.utils import parse_params_arg, setup_logger


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


def sweep_xgboost(
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
    """
    Randomised HP search for XGBoost using GroupKFold on chromosomes.
    When LOGGING == 'wandb', each candidate run is logged to wandb after the
    search completes.

    Returns best_hp dict and saves best_hyperparameters.json to RES_DIR.
    """
    Path(RES_DIR).mkdir(parents=True, exist_ok=True)
    use_wandb = wandb_cfg is not None
    task = resolve_task(TASK, train_data, target_cols)

    if task == "classification":
        encoded = encode_classification_target(train_data, target_cols)
        objective = (
            "binary:logistic"
            if encoded.problem_type == "binary"
            else "multi:softprob"
        )
        base_kwargs = {
            "objective": objective,
            "tree_method": "hist",
            "max_bin": 256,
            "n_jobs": 1,
            "random_state": seed,
            "verbosity": 0,
            "eval_metric": "logloss"
            if encoded.problem_type == "binary"
            else "mlogloss",
        }
        if encoded.problem_type == "multiclass":
            base_kwargs["num_class"] = len(encoded.classes)

        estimator = xgb.XGBClassifier(**base_kwargs)
        param_dist = {
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": np.logspace(np.log10(0.01), np.log10(0.2), 10),
            "n_estimators": [300, 600, 1000],
            "min_child_weight": [1, 3, 5, 10],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 1e-3, 1e-2, 1e-1, 1.0],
            "reg_lambda": [0.1, 1.0, 5.0, 10.0],
        }

        tune_data = _stratified_fraction_sample(
            train_data,
            encoded.target_col,
            sweep_fraction,
            seed,
        )
        X_tune = tune_data[feature_cols].values
        class_lookup = {label: idx for idx, label in enumerate(encoded.classes)}
        y_tune = np.array(
            [class_lookup[label] for label in tune_data[encoded.target_col].astype(str)]
        )

        if len(X_tune) < 2 or n_sweeps <= 0:
            best_hp = _default_best_hp(param_dist)
            logger.warning(
                "XGBoost classifier HP sweep skipped: using deterministic defaults."
            )
            out_path = Path(RES_DIR) / "best_hyperparameters.json"
            with open(out_path, "w") as f:
                json.dump(json_safe(best_hp), f, indent=2)
            logger.info(f"Saved best hyperparameters to {out_path}")
            return best_hp

        unique_classes, class_counts = np.unique(y_tune, return_counts=True)
        if len(unique_classes) < 2:
            best_hp = _default_best_hp(param_dist)
            logger.warning(
                "XGBoost classifier HP sweep skipped: tuning sample contains "
                "fewer than two classes. Using deterministic defaults."
            )
            out_path = Path(RES_DIR) / "best_hyperparameters.json"
            with open(out_path, "w") as f:
                json.dump(json_safe(best_hp), f, indent=2)
            logger.info(f"Saved best hyperparameters to {out_path}")
            return best_hp

        n_splits = min(3, int(class_counts.min()))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            fit_kwargs = {}
            cv_desc = f"{n_splits}-fold StratifiedKFold"
        else:
            best_hp = _default_best_hp(param_dist)
            logger.warning(
                "XGBoost classifier HP sweep skipped: every class needs at least "
                "two samples for stratified CV. Using deterministic defaults."
            )
            out_path = Path(RES_DIR) / "best_hyperparameters.json"
            with open(out_path, "w") as f:
                json.dump(json_safe(best_hp), f, indent=2)
            logger.info(f"Saved best hyperparameters to {out_path}")
            return best_hp

        if len(X_tune) < max(10, n_splits * 4):
            best_hp = _default_best_hp(param_dist)
            logger.warning(
                "XGBoost classifier HP sweep skipped: {} samples is too few for "
                "stable {}. Using deterministic defaults.".format(
                    len(X_tune), cv_desc
                )
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
            scoring=make_scorer(f1_score, average="macro"),
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            random_state=seed,
            refit=False,
        )
        logger.info(
            f"XGBoost classifier HP sweep: {n_sweeps} iterations on "
            f"{len(X_tune):,} regions ({sweep_fraction:.0%} of train), {cv_desc}"
        )
        search.fit(X_tune, y_tune, **fit_kwargs)
        best_hp = {
            k.replace("estimator__", ""): v for k, v in search.best_params_.items()
        }
        if np.isnan(search.best_score_):
            logger.warning("XGBoost classifier sweep scores are NaN; using defaults.")
            best_hp = _default_best_hp(param_dist)
        else:
            logger.info(
                f"Best sweep macro-F1={search.best_score_:.4f}  params={best_hp}"
            )

        out_path = Path(RES_DIR) / "best_hyperparameters.json"
        with open(out_path, "w") as f:
            json.dump(json_safe(best_hp), f, indent=2)
        logger.info(f"Saved best hyperparameters to {out_path}")
        return best_hp

    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        max_bin=256,
        n_jobs=1,
        random_state=seed,
        verbosity=0,
    )

    if len(target_cols) == 1:
        estimator = base
        param_dist = {
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": np.logspace(np.log10(0.01), np.log10(0.2), 10),
            "n_estimators": [300, 600, 1000],
            "min_child_weight": [1, 3, 5, 10],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 1e-3, 1e-2, 1e-1, 1.0],
            "reg_lambda": [0.1, 1.0, 5.0, 10.0],
        }
    else:
        estimator = MultiOutputRegressor(base, n_jobs=1)
        param_dist = {
            "estimator__max_depth": [3, 4, 5, 6, 8],
            "estimator__learning_rate": np.logspace(np.log10(0.01), np.log10(0.2), 10),
            "estimator__n_estimators": [300, 600, 1000],
            "estimator__min_child_weight": [1, 3, 5, 10],
            "estimator__subsample": [0.6, 0.8, 1.0],
            "estimator__colsample_bytree": [0.6, 0.8, 1.0],
            "estimator__reg_alpha": [0.0, 1e-3, 1e-2, 1e-1, 1.0],
            "estimator__reg_lambda": [0.1, 1.0, 5.0, 10.0],
        }

    tune_data = train_data.sample(frac=min(sweep_fraction, 1.0), random_state=seed)
    X_tune = tune_data[feature_cols].values
    y_tune = tune_data[target_cols].values
    if len(target_cols) == 1:
        y_tune = y_tune.ravel()

    if len(X_tune) < 2:
        best_hp = _default_best_hp(param_dist)
        logger.warning(
            "XGBoost HP sweep skipped: need at least 2 samples, got {}. "
            "Using deterministic default hyperparameters.".format(len(X_tune))
        )
        out_path = Path(RES_DIR) / "best_hyperparameters.json"
        with open(out_path, "w") as f:
            json.dump(best_hp, f, indent=2)
        logger.info(f"Saved best hyperparameters to {out_path}")
        return best_hp

    if hasattr(tune_data.index, "get_level_values"):
        contigs = tune_data.index.get_level_values("contig").astype(str)
    else:
        contigs = tune_data.index.astype(str).str.split(":").str[0]
    groups = contigs.values

    n_unique_groups = len(np.unique(groups))
    if n_unique_groups >= 2:
        n_splits = min(3, n_unique_groups)
        cv = GroupKFold(n_splits=n_splits)
        fit_kwargs = {"groups": groups}
        cv_desc = f"{n_splits}-fold GroupKFold by chromosome"
    else:
        n_splits = min(3, len(X_tune))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fit_kwargs = {}
        cv_desc = f"{n_splits}-fold KFold (single chromosome group in sweep subset)"

    min_cv_samples = max(10, n_splits * 4)
    if len(X_tune) < min_cv_samples:
        best_hp = _default_best_hp(param_dist)
        logger.warning(
            "XGBoost HP sweep skipped: {} samples is too few for stable {}. "
            "Using deterministic default hyperparameters.".format(
                len(X_tune), cv_desc
            )
        )
        out_path = Path(RES_DIR) / "best_hyperparameters.json"
        with open(out_path, "w") as f:
            json.dump(best_hp, f, indent=2)
        logger.info(f"Saved best hyperparameters to {out_path}")
        return best_hp

    r2_macro = make_scorer(
        r2_score,
        greater_is_better=True,
        multioutput="uniform_average",
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_sweeps,
        scoring=r2_macro,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        random_state=seed,
        refit=False,
    )

    logger.info(
        f"XGBoost HP sweep: {n_sweeps} iterations on {len(X_tune):,} regions "
        f"({sweep_fraction:.0%} of train), {cv_desc}"
    )
    search.fit(X_tune, y_tune, **fit_kwargs)

    best_score = search.best_score_
    raw_params = search.best_params_
    best_hp = {k.replace("estimator__", ""): v for k, v in raw_params.items()}
    if np.isnan(best_score):
        logger.warning(
            "XGBoost sweep: all CV scores are NaN — check that sweep_fraction produces "
            "enough samples per fold. Falling back to default hyperparameters."
        )
        best_hp = _default_best_hp(param_dist)
    else:
        logger.info(f"Best sweep R²={best_score:.4f}  params={best_hp}")

    if use_wandb:
        import wandb

        sweep_ts = time.strftime("%Y-%m-%d_%H%M%S")
        results = search.cv_results_
        for i in range(len(results["params"])):
            hp = {
                k.replace("estimator__", ""): v for k, v in results["params"][i].items()
            }
            with wandb_cfg.init_run(
                name=f"{wandb_cfg.model_name}_sweep_{sweep_ts}_{i}",
                group="sweep",
                config=hp,
                reinit="finish_previous",
            ):
                score = float(results["mean_test_score"][i])
                if not np.isnan(score):
                    wandb.log(
                        {
                            "val_r2": score,
                            "val_r2_std": float(results["std_test_score"][i]),
                            "fit_time": float(results["mean_fit_time"][i]),
                        }
                    )

    out_path = Path(RES_DIR) / "best_hyperparameters.json"
    with open(out_path, "w") as f:
        json.dump(
            best_hp,
            f,
            indent=2,
            default=lambda x: (
                float(x)
                if isinstance(x, np.floating)
                else int(x)
                if isinstance(x, np.integer)
                else x
            ),
        )
    logger.info(f"Saved best hyperparameters to {out_path}")

    return best_hp


def main():
    params = PipelineParams.from_yaml(parse_params_arg())
    setup_logger(params["RES_DIR"], params["PROJECT"])

    _, _, target_cols, feature_cols, train_data, _, _ = load_data(**vars(params))

    wandb_cfg = None
    if params["LOGGING"] == "wandb":
        from tabnado.wandb import WandbConfig

        wandb_cfg = WandbConfig.from_params(params)
    sweep_xgboost(
        feature_cols=feature_cols,
        target_cols=target_cols,
        train_data=train_data,
        n_sweeps=params["N_SWEEPS"],
        sweep_fraction=params["SWEEP_FRACTION"],
        RES_DIR=params["RES_DIR"],
        wandb_cfg=wandb_cfg,
        TASK=params.TASK,
    )
