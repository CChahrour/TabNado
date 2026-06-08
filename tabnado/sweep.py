import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.multioutput import MultiOutputRegressor

from tabnado.utils import (
    encode_classification_target,
    json_safe,
    require_single_classification_target,
    resolve_task,
)

# ---------------------------------------------------------------------------
# Shared sampling helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


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


def _sweep_xgboost(
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
    import xgboost as xgb

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


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------


def _import_catboost():
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CatBoost backend requested but catboost is not installed. "
            "Install/sync TabNado with the catboost dependency first."
        ) from exc
    return CatBoostClassifier, CatBoostRegressor


def _import_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CatBoost sweeps now use Optuna. Install/sync TabNado with the "
            "optuna dependency first."
        ) from exc
    return optuna


def _default_catboost_best_hp() -> dict[str, Any]:
    return {
        "colsample_bylevel": 0.1,
        "depth": 4,
        "boosting_type": "Plain",
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.0,
        "learning_rate": 0.05,
        "iterations": 600,
        "l2_leaf_reg": 3.0,
        "random_strength": 0.5,
    }


def _save_best_hp(best_hp: dict[str, Any], RES_DIR: str) -> None:
    out_path = Path(RES_DIR) / "best_hyperparameters.json"
    with open(out_path, "w") as f:
        json.dump(json_safe(best_hp), f, indent=2)
    logger.info(f"Saved best hyperparameters to {out_path}")


def _save_trials(study, RES_DIR: str) -> None:
    records = []
    for trial in study.trials:
        record = {
            "number": trial.number,
            "state": str(trial.state),
            "value": trial.value,
        }
        record.update(trial.params)
        records.append(record)

    if not records:
        return

    out_path = Path(RES_DIR) / "catboost_optuna_trials.csv"
    pd.DataFrame.from_records(records).to_csv(out_path, index=False)
    logger.info(f"Saved CatBoost Optuna trial table to {out_path}")


def _fraction_sample(
    data: pd.DataFrame,
    frac: float,
    seed: int,
) -> pd.DataFrame:
    frac = min(max(float(frac), 0.0), 1.0)
    if frac >= 1.0:
        return data
    n = max(1, int(round(len(data) * frac)))
    return data.sample(n=min(n, len(data)), random_state=seed)


def _split_tune_data(
    tune_data: pd.DataFrame,
    target_cols: list[str],
    task: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(tune_data) < 4:
        raise ValueError("Need at least 4 tuning rows to create a validation split.")

    stratify = None
    if task == "classification":
        target = tune_data[target_cols[0]].astype(str)
        class_counts = target.value_counts()
        if len(class_counts) >= 2 and int(class_counts.min()) >= 2:
            stratify = target

    train_part, valid_part = train_test_split(
        tune_data,
        test_size=0.25,
        random_state=seed,
        stratify=stratify,
    )
    return train_part, valid_part


def _suggest_catboost_params(trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical(
            "boosting_type",
            ["Ordered", "Plain"],
        ),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type",
            ["Bayesian", "Bernoulli", "MVS"],
        ),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "iterations": trial.suggest_categorical("iterations", [300, 600, 1000]),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
    }

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature",
            0.0,
            10.0,
        )
    elif params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1.0)

    return params


def _class_indices(labels: pd.Series, classes: list[str]) -> np.ndarray:
    class_lookup = {label: idx for idx, label in enumerate(classes)}
    return np.array([class_lookup[label] for label in labels.astype(str)])


def _valid_classification_tune_data(
    tune_data: pd.DataFrame,
    target_col: str,
) -> bool:
    class_counts = tune_data[target_col].astype(str).value_counts()
    return len(class_counts) >= 2 and int(class_counts.min()) >= 2


def _classification_score(
    estimator,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> float:
    y_pred = np.asarray(estimator.predict(X_valid)).astype(int).ravel()
    return float(
        f1_score(
            y_valid,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )


def _regression_score(
    estimators: list[Any],
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> float:
    preds = np.column_stack([estimator.predict(X_valid) for estimator in estimators])
    if preds.shape[1] == 1:
        preds = preds.ravel()
    return float(r2_score(y_valid, preds, multioutput="uniform_average"))


def _sweep_catboost(
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
    early_stopping_rounds: int = 10,
    eval_data: pd.DataFrame | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Optuna hyperparameter search for CatBoost."""
    CatBoostClassifier, CatBoostRegressor = _import_catboost()
    optuna = _import_optuna()
    Path(RES_DIR).mkdir(parents=True, exist_ok=True)
    task = resolve_task(TASK, train_data, target_cols)

    if n_sweeps <= 0:
        best_hp = _default_catboost_best_hp()
        logger.warning("CatBoost Optuna sweep skipped: using deterministic defaults.")
        _save_best_hp(best_hp, RES_DIR)
        return best_hp

    common_params = {
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": 1,
    }

    if task == "classification":
        encoded = encode_classification_target(train_data, target_cols, eval_data)
        tune_data = _stratified_fraction_sample(
            train_data,
            encoded.target_col,
            sweep_fraction,
            seed,
        )
        if not _valid_classification_tune_data(tune_data, encoded.target_col):
            best_hp = _default_catboost_best_hp()
            logger.warning(
                "CatBoost classifier Optuna sweep skipped: tuning sample needs at "
                "least two classes with at least two rows each. Using defaults."
            )
            _save_best_hp(best_hp, RES_DIR)
            return best_hp

        if eval_data is not None and not eval_data.empty:
            fit_data = tune_data
            valid_data = eval_data
        else:
            fit_data, valid_data = _split_tune_data(
                tune_data,
                target_cols,
                task,
                seed,
            )

        X_fit = fit_data[feature_cols]
        y_fit = _class_indices(fit_data[encoded.target_col], encoded.classes)
        X_valid = valid_data[feature_cols]
        y_valid = _class_indices(valid_data[encoded.target_col], encoded.classes)
        loss_function = "Logloss" if encoded.problem_type == "binary" else "MultiClass"
        metric_name = "weighted-F1"

        def objective(trial) -> float:
            params = _suggest_catboost_params(trial)
            estimator = CatBoostClassifier(
                **common_params,
                **params,
                loss_function=loss_function,
                eval_metric="TotalF1",
            )
            estimator.fit(
                X_fit,
                y_fit,
                eval_set=(X_valid, y_valid),
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )
            return _classification_score(estimator, X_valid, y_valid)

    else:
        tune_data = _fraction_sample(train_data, sweep_fraction, seed)
        if len(tune_data) < 4:
            best_hp = _default_catboost_best_hp()
            logger.warning(
                "CatBoost Optuna sweep skipped: need at least 4 tuning rows, got {}. "
                "Using defaults.".format(len(tune_data))
            )
            _save_best_hp(best_hp, RES_DIR)
            return best_hp

        if eval_data is not None and not eval_data.empty:
            fit_data = tune_data
            valid_data = eval_data
        else:
            fit_data, valid_data = _split_tune_data(
                tune_data,
                target_cols,
                task,
                seed,
            )

        X_fit = fit_data[feature_cols]
        X_valid = valid_data[feature_cols]
        y_valid = valid_data[target_cols].values
        if len(target_cols) == 1:
            y_valid = y_valid.ravel()
        metric_name = "R2"

        def objective(trial) -> float:
            params = _suggest_catboost_params(trial)
            estimators = []
            for col in target_cols:
                estimator = CatBoostRegressor(
                    **common_params,
                    **params,
                    loss_function="RMSE",
                    eval_metric="RMSE",
                )
                estimator.fit(
                    X_fit,
                    fit_data[col],
                    eval_set=(X_valid, valid_data[col]),
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )
                estimators.append(estimator)
            return _regression_score(estimators, X_valid, y_valid)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    logger.info(
        "CatBoost Optuna sweep: {} trials on {:,} tuning regions "
        "({:.0%} of train), validation={} regions, metric={}".format(
            n_sweeps,
            len(fit_data),
            min(max(float(sweep_fraction), 0.0), 1.0),
            len(valid_data),
            metric_name,
        )
    )
    study.optimize(objective, n_trials=n_sweeps, n_jobs=max(1, int(n_jobs)))

    _save_trials(study, RES_DIR)
    if study.best_value is None or np.isnan(study.best_value):
        best_hp = _default_catboost_best_hp()
        logger.warning("CatBoost Optuna sweep scores are NaN; using defaults.")
    else:
        best_hp = dict(study.best_params)
        logger.info(
            "Best CatBoost Optuna {}={:.4f}  params={}".format(
                metric_name,
                study.best_value,
                best_hp,
            )
        )

    _save_best_hp(best_hp, RES_DIR)

    if wandb_cfg is not None:
        run = wandb_cfg.init_run(
            name=f"{wandb_cfg.model_name}_catboost_optuna_sweep",
            group="sweep",
            config={"n_trials": n_sweeps, "metric": metric_name, **best_hp},
            reinit="finish_previous",
        )
        run.log({f"best_{metric_name}": float(study.best_value)})
        run.finish()

    return best_hp


# ---------------------------------------------------------------------------
# GANDALF (PyTorch Tabular, wandb-driven sweeps)
# ---------------------------------------------------------------------------

_LOCAL_SWEEP_BEST_HP_CACHE: dict[str, dict] = {}


def _center_window_columns(columns: list[str]) -> list[str]:
    """Return only TSS-centered columns (suffix `_0`)."""
    center_cols = [c for c in columns if c.endswith("_0")]
    return center_cols if center_cols else columns


def _make_data_config(feature_cols, target_cols):
    from pytorch_tabular.config import DataConfig

    return DataConfig(
        continuous_cols=feature_cols,
        target=target_cols,
        validation_split=0,
        normalize_continuous_features=False,
        num_workers=0,
        pin_memory=False,
        dataloader_kwargs={"persistent_workers": False},
    )


def _sample_from_sweep_params(param_spec: dict) -> dict:
    sampled = {}
    for key, spec in param_spec.items():
        if "values" in spec:
            sampled[key] = spec["values"][np.random.randint(0, len(spec["values"]))]
            continue

        distribution = spec.get("distribution")
        if distribution == "uniform":
            sampled[key] = float(np.random.uniform(spec["min"], spec["max"]))
        elif distribution == "log_uniform_values":
            low = np.log(spec["min"])
            high = np.log(spec["max"])
            sampled[key] = float(np.exp(np.random.uniform(low, high)))
        else:
            raise ValueError(
                f"Unsupported sweep distribution for {key}: {distribution}"
            )
    return sampled


def _gandalf_sweep_train(
    train_data,
    eval_data,
    test_data,
    feature_cols,
    target_cols,
    FIG_DIR: str = "figures",
    RES_DIR: str = "results",
    SWEEP_FRACTION: float = 0.1,
    LOGGING_DIR: str | None = None,
    SWEEP_ID: str | None = None,
    hp_config: dict | None = None,
    LOGGING: str = "none",
    wandb_cfg=None,
    TASK: str = "regression",
):
    import torch
    import wandb
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import ExperimentConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.models import GANDALFConfig
    from sklearn.metrics import f1_score as _f1_score

    from tabnado.data import stratified_sample
    from tabnado.utils import LoguruProgressCallback, log_macro, seed_everything

    task = resolve_task(TASK, train_data, target_cols)
    if task == "classification":
        require_single_classification_target(target_cols)

    use_wandb = wandb_cfg is not None
    logging_dir = LOGGING_DIR or os.path.join(RES_DIR, "logging")
    use_tensorboard = LOGGING == "tensorboard"
    if use_tensorboard:
        os.makedirs(logging_dir, exist_ok=True)
    experiment_project = (
        logging_dir
        if use_tensorboard
        else (wandb_cfg.project if use_wandb else logging_dir)
    )

    run_name = f"GANDALF_sweep_{time.strftime('%Y-%m-%d_%H%M')}"
    sweep_root = (
        os.path.join(RES_DIR, "sweep", SWEEP_ID)
        if SWEEP_ID
        else os.path.join(RES_DIR, "sweep")
    )
    run_dir = os.path.join(sweep_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    config_dict = hp_config
    if config_dict is None and not use_wandb:
        config_dict = _sample_from_sweep_params(create_sweep_dict()["parameters"])

    sweep_target_cols = _center_window_columns(target_cols)
    selected_cols = feature_cols + sweep_target_cols

    train_view = train_data[selected_cols]
    eval_view = eval_data[selected_cols]
    test_view = test_data[selected_cols]

    logger.info(
        "Sweep uses all tiled columns: {} features, {} targets".format(
            len(feature_cols), len(sweep_target_cols)
        )
    )

    def _make_model_config(config_obj, sweep_target_cols):
        if task == "classification":
            return GANDALFConfig(
                learning_rate=config_obj.learning_rate,
                embedding_dropout=config_obj.embedding_dropout,
                gflu_stages=config_obj.gflu_stages,
                gflu_dropout=config_obj.gflu_dropout,
                gflu_feature_init_sparsity=config_obj.gflu_feature_init_sparsity,
                learnable_sparsity=False,
                head="LinearHead",
                loss="CrossEntropyLoss",
                metrics=["accuracy"],
                metrics_params=[{}],
                seed=42,
                task="classification",
            )

        return GANDALFConfig(
            learning_rate=config_obj.learning_rate,
            embedding_dropout=config_obj.embedding_dropout,
            gflu_stages=config_obj.gflu_stages,
            gflu_dropout=config_obj.gflu_dropout,
            gflu_feature_init_sparsity=config_obj.gflu_feature_init_sparsity,
            learnable_sparsity=False,
            head="LinearHead",
            loss="MSELoss",
            metrics=["r2_score", "mean_squared_error"],
            metrics_params=[{}] * 2,
            seed=42,
            target_range=[(0, 1)] * len(sweep_target_cols),
            task="regression",
        )

    def _fit_and_eval(config_obj):
        early_stopping = (
            "valid_accuracy" if task == "classification" else "valid_r2_score"
        )
        model = TabularModel(
            data_config=_make_data_config(feature_cols, sweep_target_cols),
            experiment_config=ExperimentConfig(
                exp_log_freq=1,
                exp_watch=None,
                log_logits=False,
                log_target=LOGGING,
                project_name=experiment_project,
                run_name=run_name,
            ),
            model_config=_make_model_config(config_obj, sweep_target_cols),
            optimizer_config=OptimizerConfig(
                optimizer_params={"weight_decay": config_obj.weight_decay},
            ),
            trainer_config=TrainerConfig(
                accelerator="mps"
                if torch.backends.mps.is_available()
                else "gpu"
                if torch.cuda.is_available()
                else "cpu",
                auto_lr_find=False,
                batch_size=2048,
                check_val_every_n_epoch=1,
                checkpoints_path=os.path.join(run_dir, "checkpoints"),
                early_stopping=early_stopping,
                early_stopping_mode="max",
                early_stopping_patience=3,
                load_best=True,
                max_epochs=12,
                gradient_clip_val=config_obj.gradient_clip_val,
                trainer_kwargs=dict(enable_model_summary=False),
                progress_bar="none",
            ),
            verbose=False,
            suppress_lightning_logger=True,
        )

        if task == "classification":
            sweep_train_data = train_view.sample(
                frac=min(SWEEP_FRACTION, 1.0), random_state=42
            )
        else:
            sweep_train_data = stratified_sample(
                train_view, sweep_target_cols, frac=SWEEP_FRACTION
            )
        logger.info(
            f"Sweep train size: {len(sweep_train_data):,} ({SWEEP_FRACTION:.0%} of full)"
        )

        model.fit(
            train=sweep_train_data,
            validation=eval_view,
            callbacks=[LoguruProgressCallback()],
        )
        # Log macro metrics to wandb if enabled
        if use_wandb and task == "regression":
            log_macro(model, sweep_target_cols)

        pred_df = model.predict(test_view)
        if task == "classification":
            target_col = sweep_target_cols[0]
            pred_col = f"{target_col}_prediction"
            y_true = test_view[target_col].astype(str).values
            y_pred = pred_df[pred_col].astype(str).values
        else:
            pred_df.columns = [c.replace("_prediction", "") for c in pred_df.columns]
            y_true = test_view[sweep_target_cols].values
            y_pred = pred_df[sweep_target_cols].values

        if task == "regression" and np.isnan(y_pred).any():
            logger.warning("NaN predictions - skipping run")
            if use_wandb:
                wandb.log({"failed": True})
            return None

        if task == "classification":
            test_metrics = {
                "test_accuracy": float(np.mean(y_true == y_pred)),
                "test_macro_f1": float(_f1_score(y_true, y_pred, average="macro")),
            }
        else:
            test_metrics = {
                "test_r2_macro": float(
                    r2_score(y_true, y_pred, multioutput="uniform_average")
                ),
                "test_mse_macro": float(mean_squared_error(y_true, y_pred)),
            }
            for i, col in enumerate(sweep_target_cols):
                test_metrics[f"test_r2_{col}"] = float(
                    r2_score(y_true[:, i], y_pred[:, i])
                )

        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=4)
        with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
            hp_keys_to_save = list(create_sweep_dict()["parameters"].keys())
            hp_dict = {
                k: getattr(config_obj, k)
                for k in hp_keys_to_save
                if hasattr(config_obj, k)
            }
            json.dump(hp_dict, f, indent=4)

        if use_wandb:
            wandb.log(test_metrics)
        logger.info(test_metrics)

        if hasattr(model, "callbacks"):
            model.callbacks = []
        model.save_model(os.path.join(run_dir, "model"), inference_only=False)
        return test_metrics

    seed_everything(42)
    logger.info(f"======== Fitting {run_name} ========")

    if not use_wandb:
        assert config_dict is not None
        return _fit_and_eval(SimpleNamespace(**config_dict))

    with wandb_cfg.init_run(
        name=run_name,
        group="sweep",
        reinit="finish_previous",
        dir_override=run_dir,
    ) as run:
        return _fit_and_eval(run.config)


def create_sweep_dict(
    MODEL_NAME: str = "GANDALF_Sweep",
    PROJECT: str = "PROJECT_NAME",
    TASK: str = "regression",
):
    metric = (
        {"name": "valid_accuracy", "goal": "maximize"}
        if str(TASK).lower() == "classification"
        else {"name": "valid_r2_score", "goal": "maximize"}
    )
    return {
        "name": MODEL_NAME,
        "method": "bayes",
        "metric": metric,
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-3,
                "max": 3e-2,
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 5e-4,
                "max": 5e-3,
            },
            "gradient_clip_val": {"distribution": "uniform", "min": 0.5, "max": 2.0},
            "embedding_dropout": {"distribution": "uniform", "min": 0.0, "max": 0.02},
            "gflu_dropout": {"distribution": "uniform", "min": 0.0, "max": 0.05},
            "gflu_feature_init_sparsity": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.4,
            },
            "gflu_stages": {"values": [6, 8, 10, 12]},
        },
    }


def _gandalf_start_sweep_and_run(
    train_data,
    eval_data,
    test_data,
    feature_cols,
    target_cols,
    count,
    RES_DIR: str = "results",
    SWEEP_FRACTION: float = 0.1,
    LOGGING: str = "none",
    LOGGING_DIR: str | None = None,
    wandb_cfg=None,
    TASK: str = "regression",
) -> str:
    import wandb

    task = resolve_task(TASK, train_data, target_cols)
    use_wandb = wandb_cfg is not None
    logging_dir = LOGGING_DIR or os.path.join(RES_DIR, "logging")
    if LOGGING == "tensorboard":
        os.makedirs(logging_dir, exist_ok=True)

    sweep_dir = os.path.join(RES_DIR, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)

    if not use_wandb:
        sweep_id = f"local-{time.strftime('%Y-%m-%d_%H%M')}"
        local_sweep_dir = os.path.join(sweep_dir, sweep_id)
        os.makedirs(local_sweep_dir, exist_ok=True)

        logger.info(f"Running local sweep: {sweep_id}")
        best_hp = None
        best_score = -np.inf
        sweep_params = create_sweep_dict(TASK=task)["parameters"]
        score_key = "test_macro_f1" if task == "classification" else "test_r2_macro"

        for _ in range(count):
            hp = _sample_from_sweep_params(sweep_params)
            metrics = _gandalf_sweep_train(
                train_data,
                eval_data,
                test_data,
                feature_cols,
                target_cols,
                RES_DIR=RES_DIR,
                SWEEP_FRACTION=SWEEP_FRACTION,
                LOGGING_DIR=logging_dir,
                SWEEP_ID=sweep_id,
                hp_config=hp,
                LOGGING=LOGGING,
                wandb_cfg=wandb_cfg,
                TASK=task,
            )
            if metrics is not None and metrics[score_key] > best_score:
                best_score = metrics[score_key]
                best_hp = hp

        if best_hp is None:
            raise RuntimeError("Local sweep produced no valid runs")

        _LOCAL_SWEEP_BEST_HP_CACHE[sweep_id] = best_hp
        return sweep_id

    sweep_id = wandb.sweep(
        sweep=create_sweep_dict(TASK=task),
        entity=wandb_cfg.entity,
        project=wandb_cfg.project,
    )
    logger.info(f"Created sweep: {sweep_id}")
    wandb.agent(
        sweep_id,
        function=lambda: _gandalf_sweep_train(
            train_data,
            eval_data,
            test_data,
            feature_cols,
            target_cols,
            RES_DIR=RES_DIR,
            SWEEP_FRACTION=SWEEP_FRACTION,
            LOGGING_DIR=logging_dir,
            SWEEP_ID=sweep_id,
            LOGGING=LOGGING,
            wandb_cfg=wandb_cfg,
            TASK=task,
        ),
        count=count,
    )
    return sweep_id


def _gandalf_get_best_hp_from_sweep(
    sweep_id: str,
    RES_DIR: str = "results",
    wandb_cfg=None,
) -> dict:
    """Fetch the hyperparameters of the best-performing run in a sweep."""
    import wandb

    use_wandb = wandb_cfg is not None
    if not use_wandb:
        cached = _LOCAL_SWEEP_BEST_HP_CACHE.get(sweep_id)
        if cached is not None:
            return cached

        root_hp_path = os.path.join(RES_DIR, "best_hyperparameters.json")
        if os.path.exists(root_hp_path):
            with open(root_hp_path) as f:
                return json.load(f)

        raise FileNotFoundError(
            f"No local best_hyperparameters found in cache and no file at {root_hp_path}"
        )

    hp_keys = list(create_sweep_dict()["parameters"].keys())

    # Try wandb API first; fall back to local sweep run directories.
    try:
        api = wandb.Api()
        entity = wandb_cfg.entity
        sweep_path = (
            f"{entity}/{wandb_cfg.project}/{sweep_id}"
            if entity
            else f"{wandb_cfg.project}/{sweep_id}"
        )
        sweep = api.sweep(sweep_path)
        best_run = sweep.best_run()
        if best_run is not None:
            hp = {k: best_run.config[k] for k in hp_keys if k in best_run.config}
            if hp:
                return hp
    except Exception as e:
        logger.warning(
            f"wandb API unavailable ({e}), falling back to local sweep files"
        )

    # Fall back: scan local run directories under RES_DIR/sweep/sweep_id/
    sweep_dir = os.path.join(RES_DIR, "sweep", sweep_id)
    if not os.path.isdir(sweep_dir):
        raise FileNotFoundError(f"No local sweep directory found at {sweep_dir}")

    best_hp = None
    best_score = -np.inf
    for run_dir in os.scandir(sweep_dir):
        if not run_dir.is_dir():
            continue
        metrics_path = os.path.join(run_dir.path, "metrics.json")
        hp_path = os.path.join(run_dir.path, "hyperparameters.json")
        if not os.path.exists(metrics_path) or not os.path.exists(hp_path):
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        score = metrics.get(
            "test_macro_f1",
            metrics.get("test_r2_macro", -np.inf),
        )
        if score > best_score:
            with open(hp_path) as f:
                all_hp = json.load(f)
            hp = {k: all_hp[k] for k in hp_keys if k in all_hp}
            if hp:
                best_score = score
                best_hp = hp

    if best_hp is None:
        raise RuntimeError(f"No valid local sweep runs found in {sweep_dir}")

    logger.info(f"Best local sweep run: test_r2_macro={best_score:.4f}  hp={best_hp}")
    return best_hp


def _sweep_gandalf(
    feature_cols: list[str],
    target_cols: list[str],
    train_data: pd.DataFrame,
    n_sweeps: int = 20,
    sweep_fraction: float = 0.1,
    RES_DIR: str = "results",
    wandb_cfg=None,
    TASK: str = "regression",
    eval_data: pd.DataFrame | None = None,
    test_data: pd.DataFrame | None = None,
    LOGGING: str = "none",
    LOGGING_DIR: str | None = None,
    **kwargs,
) -> dict:
    """Run a GANDALF hyperparameter sweep (wandb agent or local loop) and
    persist the best hyperparameters to ``RES_DIR/best_hyperparameters.json``."""
    task = resolve_task(TASK, train_data, target_cols)

    sweep_id = _gandalf_start_sweep_and_run(
        train_data,
        eval_data,
        test_data,
        feature_cols,
        target_cols,
        count=n_sweeps,
        RES_DIR=RES_DIR,
        SWEEP_FRACTION=sweep_fraction,
        LOGGING=LOGGING,
        LOGGING_DIR=LOGGING_DIR,
        wandb_cfg=wandb_cfg,
        TASK=task,
    )
    best_hp = _gandalf_get_best_hp_from_sweep(
        sweep_id,
        RES_DIR=RES_DIR,
        wandb_cfg=wandb_cfg,
    )
    logger.info(f"Best hyperparameters: {best_hp}")
    hp_path = Path(RES_DIR) / "best_hyperparameters.json"
    with open(hp_path, "w") as f:
        json.dump(best_hp, f, indent=4)
    logger.info(f"Saved best hyperparameters to {hp_path}")
    return best_hp


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

_SWEEP_BACKENDS = {
    "xgboost": _sweep_xgboost,
    "catboost": _sweep_catboost,
    "gandalf": _sweep_gandalf,
}


def sweep_model(model_type: str, *args, **kwargs) -> dict:
    """Dispatch the hyperparameter sweep to the configured backend.

    ``model_type`` is one of ``"xgboost"``, ``"catboost"``, or ``"gandalf"``
    (any other value falls back to GANDALF, matching the legacy backend
    selection in the pipeline API). Returns the best hyperparameters dict
    and persists it to ``best_hyperparameters.json`` under ``RES_DIR``.
    """
    sweep_fn = _SWEEP_BACKENDS.get(model_type, _sweep_gandalf)
    return sweep_fn(*args, **kwargs)


def main():
    from tabnado.data import load_data
    from tabnado.params import PipelineParams
    from tabnado.utils import parse_params_arg, setup_logger

    params = PipelineParams.from_yaml(parse_params_arg())
    setup_logger(params["RES_DIR"], params["PROJECT"])

    _, _, target_cols, feature_cols, train_data, eval_data, test_data = load_data(
        **vars(params)
    )

    wandb_cfg = None
    if params["LOGGING"] == "wandb":
        from tabnado.wandb import WandbConfig

        wandb_cfg = WandbConfig.from_params(params)

    sweep_model(
        params.MODEL_TYPE,
        feature_cols=feature_cols,
        target_cols=target_cols,
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        n_sweeps=params["N_SWEEPS"],
        sweep_fraction=params["SWEEP_FRACTION"],
        RES_DIR=params["RES_DIR"],
        LOGGING=params["LOGGING"],
        LOGGING_DIR=params["LOGGING_DIR"],
        wandb_cfg=wandb_cfg,
        TASK=params.TASK,
    )


if __name__ == "__main__":
    main()
