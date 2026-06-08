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
    StratifiedKFold,
    cross_val_score,
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
    Optuna hyperparameter search for XGBoost using cross-validation
    (StratifiedKFold for classification, GroupKFold-by-chromosome for
    regression). When LOGGING == 'wandb', the best trial is logged to wandb
    purely for record-keeping — wandb does not drive the search.

    Returns best_hp dict and saves best_hyperparameters.json to RES_DIR.
    """
    import xgboost as xgb

    optuna = _import_optuna()
    Path(RES_DIR).mkdir(parents=True, exist_ok=True)
    task = resolve_task(TASK, train_data, target_cols)

    def _suggest_xgboost_params(trial) -> dict[str, Any]:
        return {
            "max_depth": trial.suggest_categorical("max_depth", [3, 4, 5, 6, 8]),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "n_estimators": trial.suggest_categorical(
                "n_estimators", [300, 600, 1000]
            ),
            "min_child_weight": trial.suggest_categorical(
                "min_child_weight", [1, 3, 5, 10]
            ),
            "subsample": trial.suggest_categorical("subsample", [0.6, 0.8, 1.0]),
            "colsample_bytree": trial.suggest_categorical(
                "colsample_bytree", [0.6, 0.8, 1.0]
            ),
            "reg_alpha": trial.suggest_categorical(
                "reg_alpha", [0.0, 1e-3, 1e-2, 1e-1, 1.0]
            ),
            "reg_lambda": trial.suggest_categorical(
                "reg_lambda", [0.1, 1.0, 5.0, 10.0]
            ),
        }

    def _bail_out(best_hp: dict, reason: str) -> dict:
        logger.warning(f"XGBoost HP sweep skipped: {reason} Using deterministic defaults.")
        out_path = Path(RES_DIR) / "best_hyperparameters.json"
        with open(out_path, "w") as f:
            json.dump(json_safe(best_hp), f, indent=2)
        logger.info(f"Saved best hyperparameters to {out_path}")
        return best_hp

    param_dist_for_defaults = {
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": np.logspace(np.log10(0.01), np.log10(0.2), 10),
        "n_estimators": [300, 600, 1000],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 1e-3, 1e-2, 1e-1, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0, 10.0],
    }

    if task == "classification":
        encoded = encode_classification_target(train_data, target_cols)
        objective_name = (
            "binary:logistic"
            if encoded.problem_type == "binary"
            else "multi:softprob"
        )
        base_kwargs = {
            "objective": objective_name,
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
            return _bail_out(_default_best_hp(param_dist_for_defaults), "using deterministic defaults.")

        unique_classes, class_counts = np.unique(y_tune, return_counts=True)
        if len(unique_classes) < 2:
            return _bail_out(
                _default_best_hp(param_dist_for_defaults),
                "tuning sample contains fewer than two classes.",
            )

        n_splits = min(3, int(class_counts.min()))
        if n_splits < 2:
            return _bail_out(
                _default_best_hp(param_dist_for_defaults),
                "every class needs at least two samples for stratified CV.",
            )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv_desc = f"{n_splits}-fold StratifiedKFold"

        if len(X_tune) < max(10, n_splits * 4):
            return _bail_out(
                _default_best_hp(param_dist_for_defaults),
                f"{len(X_tune)} samples is too few for stable {cv_desc}.",
            )

        scoring = make_scorer(f1_score, average="macro")
        metric_name = "macro-F1"

        def objective(trial) -> float:
            params = _suggest_xgboost_params(trial)
            estimator = xgb.XGBClassifier(**base_kwargs, **params)
            scores = cross_val_score(
                estimator, X_tune, y_tune, scoring=scoring, cv=cv, n_jobs=n_jobs
            )
            return float(np.mean(scores))

        logger.info(
            f"XGBoost classifier Optuna sweep: {n_sweeps} trials on "
            f"{len(X_tune):,} regions ({sweep_fraction:.0%} of train), {cv_desc}"
        )

    else:
        tune_data = train_data.sample(frac=min(sweep_fraction, 1.0), random_state=seed)
        X_tune = tune_data[feature_cols].values
        y_tune = tune_data[target_cols].values
        if len(target_cols) == 1:
            y_tune = y_tune.ravel()

        if len(X_tune) < 2:
            return _bail_out(
                _default_best_hp(param_dist_for_defaults),
                f"need at least 2 samples, got {len(X_tune)}.",
            )

        if hasattr(tune_data.index, "get_level_values"):
            contigs = tune_data.index.get_level_values("contig").astype(str)
        else:
            contigs = tune_data.index.astype(str).str.split(":").str[0]
        groups = contigs.values

        n_unique_groups = len(np.unique(groups))
        if n_unique_groups >= 2:
            n_splits = min(3, n_unique_groups)
            cv = GroupKFold(n_splits=n_splits)
            fit_params = {"groups": groups}
            cv_desc = f"{n_splits}-fold GroupKFold by chromosome"
        else:
            n_splits = min(3, len(X_tune))
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            fit_params = {}
            cv_desc = f"{n_splits}-fold KFold (single chromosome group in sweep subset)"

        min_cv_samples = max(10, n_splits * 4)
        if len(X_tune) < min_cv_samples:
            return _bail_out(
                _default_best_hp(param_dist_for_defaults),
                f"{len(X_tune)} samples is too few for stable {cv_desc}.",
            )

        scoring = make_scorer(r2_score, greater_is_better=True, multioutput="uniform_average")
        metric_name = "R2"
        base_kwargs = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "max_bin": 256,
            "n_jobs": 1,
            "random_state": seed,
            "verbosity": 0,
        }
        multioutput = len(target_cols) > 1

        def objective(trial) -> float:
            params = _suggest_xgboost_params(trial)
            base = xgb.XGBRegressor(**base_kwargs, **params)
            estimator = MultiOutputRegressor(base, n_jobs=1) if multioutput else base
            scores = cross_val_score(
                estimator,
                X_tune,
                y_tune,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                params=fit_params,
            )
            return float(np.mean(scores))

        logger.info(
            f"XGBoost Optuna sweep: {n_sweeps} trials on {len(X_tune):,} regions "
            f"({sweep_fraction:.0%} of train), {cv_desc}"
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_sweeps, n_jobs=1)

    _save_trials(study, RES_DIR, backend="xgboost")
    if study.best_value is None or np.isnan(study.best_value):
        best_hp = _default_best_hp(param_dist_for_defaults)
        logger.warning("XGBoost Optuna sweep scores are NaN; using defaults.")
    else:
        best_hp = dict(study.best_params)
        logger.info(
            f"Best XGBoost Optuna {metric_name}={study.best_value:.4f}  params={best_hp}"
        )
        _log_optuna_best_to_wandb(wandb_cfg, "xgboost", study, best_hp, metric_name)

    out_path = Path(RES_DIR) / "best_hyperparameters.json"
    with open(out_path, "w") as f:
        json.dump(json_safe(best_hp), f, indent=2)
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


def _save_trials(study, RES_DIR: str, backend: str = "catboost") -> None:
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

    out_path = Path(RES_DIR) / f"{backend}_optuna_trials.csv"
    pd.DataFrame.from_records(records).to_csv(out_path, index=False)
    logger.info(f"Saved {backend.title()} Optuna trial table to {out_path}")


def _log_optuna_best_to_wandb(
    wandb_cfg,
    backend: str,
    study,
    best_hp: dict[str, Any],
    metric_name: str,
) -> None:
    """Log the best Optuna trial result to wandb purely for record-keeping."""
    if wandb_cfg is None:
        return

    run = wandb_cfg.init_run(
        name=f"{wandb_cfg.model_name}_{backend}_optuna_sweep",
        group="sweep",
        config={"n_trials": len(study.trials), "metric": metric_name, **best_hp},
        reinit="finish_previous",
    )
    run.log({f"best_{metric_name}": float(study.best_value)})
    run.finish()


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


def _suggest_catboost_params(trial, search_space: str = "extended") -> dict[str, Any]:
    """Suggest CatBoost hyperparameters for an Optuna trial.

    ``search_space="notebook"`` mirrors the narrower 4-parameter search used in
    ``06-model-sem-vs-rs411-catboost.ipynb`` (colsample_bylevel, depth,
    boosting_type, bootstrap_type [+ conditional bagging_temperature/subsample]),
    leaving learning_rate/iterations/l2_leaf_reg/random_strength at their
    CatBoost defaults. ``search_space="extended"`` (default) tunes those four
    extra parameters as well.
    """
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
    }

    if search_space == "extended":
        params["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.01, 0.2, log=True
        )
        params["iterations"] = trial.suggest_categorical(
            "iterations", [300, 600, 1000]
        )
        params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True)
        params["random_strength"] = trial.suggest_float("random_strength", 0.0, 1.0)

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
    catboost_search_space: str = "extended",
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
            params = _suggest_catboost_params(trial, catboost_search_space)
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
            params = _suggest_catboost_params(trial, catboost_search_space)
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

    _save_trials(study, RES_DIR, backend="catboost")
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
        _log_optuna_best_to_wandb(wandb_cfg, "catboost", study, best_hp, metric_name)

    _save_best_hp(best_hp, RES_DIR)
    return best_hp


# ---------------------------------------------------------------------------
# GANDALF (PyTorch Tabular, wandb-driven sweeps)
# ---------------------------------------------------------------------------

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

    if hp_config is None:
        raise ValueError("_gandalf_sweep_train requires hp_config (sampled by Optuna)")
    config_dict = hp_config

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
            json.dump(vars(config_obj), f, indent=4)

        if use_wandb:
            wandb.log(test_metrics)
        logger.info(test_metrics)

        if hasattr(model, "callbacks"):
            model.callbacks = []
        model.save_model(os.path.join(run_dir, "model"), inference_only=False)
        return test_metrics

    seed_everything(42)
    logger.info(f"======== Fitting {run_name} ========")

    config_obj = SimpleNamespace(**config_dict)
    if not use_wandb:
        return _fit_and_eval(config_obj)

    with wandb_cfg.init_run(
        name=run_name,
        group="sweep",
        config=config_dict,
        reinit="finish_previous",
        dir_override=run_dir,
    ):
        return _fit_and_eval(config_obj)


def _suggest_gandalf_params(trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 5e-4, 5e-3, log=True),
        "gradient_clip_val": trial.suggest_float("gradient_clip_val", 0.5, 2.0),
        "embedding_dropout": trial.suggest_float("embedding_dropout", 0.0, 0.02),
        "gflu_dropout": trial.suggest_float("gflu_dropout", 0.0, 0.05),
        "gflu_feature_init_sparsity": trial.suggest_float(
            "gflu_feature_init_sparsity", 0.1, 0.4
        ),
        "gflu_stages": trial.suggest_categorical("gflu_stages", [6, 8, 10, 12]),
    }


def _default_gandalf_best_hp() -> dict[str, Any]:
    return {
        "learning_rate": 5e-3,
        "weight_decay": 1e-3,
        "gradient_clip_val": 1.0,
        "embedding_dropout": 0.01,
        "gflu_dropout": 0.02,
        "gflu_feature_init_sparsity": 0.2,
        "gflu_stages": 8,
    }


def _sweep_gandalf(
    feature_cols: list[str],
    target_cols: list[str],
    train_data: pd.DataFrame,
    n_sweeps: int = 20,
    sweep_fraction: float = 0.1,
    RES_DIR: str = "results",
    seed: int = 42,
    wandb_cfg=None,
    TASK: str = "regression",
    eval_data: pd.DataFrame | None = None,
    test_data: pd.DataFrame | None = None,
    LOGGING: str = "none",
    LOGGING_DIR: str | None = None,
    **kwargs,
) -> dict:
    """Optuna hyperparameter search for GANDALF.

    Each trial fits a GANDALF model on a sampled fraction of the training set,
    validates on ``eval_data``, and is scored on ``test_data`` (macro-F1 for
    classification, macro-R2 for regression). When LOGGING == 'wandb', each
    trial run is logged to wandb purely for record-keeping — wandb does not
    drive the search.
    """
    optuna = _import_optuna()
    Path(RES_DIR).mkdir(parents=True, exist_ok=True)
    task = resolve_task(TASK, train_data, target_cols)
    score_key = "test_macro_f1" if task == "classification" else "test_r2_macro"

    sweep_id = f"optuna-{time.strftime('%Y-%m-%d_%H%M%S')}"
    logging_dir = LOGGING_DIR or os.path.join(RES_DIR, "logging")
    if LOGGING == "tensorboard":
        os.makedirs(logging_dir, exist_ok=True)

    if n_sweeps <= 0:
        best_hp = _default_gandalf_best_hp()
        logger.warning("GANDALF Optuna sweep skipped: using deterministic defaults.")
        _save_best_hp(best_hp, RES_DIR)
        return best_hp

    def objective(trial) -> float:
        hp = _suggest_gandalf_params(trial)
        metrics = _gandalf_sweep_train(
            train_data,
            eval_data,
            test_data,
            feature_cols,
            target_cols,
            RES_DIR=RES_DIR,
            SWEEP_FRACTION=sweep_fraction,
            LOGGING_DIR=logging_dir,
            SWEEP_ID=sweep_id,
            hp_config=hp,
            LOGGING=LOGGING,
            wandb_cfg=wandb_cfg,
            TASK=task,
        )
        if metrics is None:
            return float("nan")
        return float(metrics[score_key])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    logger.info(f"GANDALF Optuna sweep: {n_sweeps} trials, metric={score_key}")
    study.optimize(objective, n_trials=n_sweeps)

    _save_trials(study, RES_DIR, backend="gandalf")
    if study.best_value is None or np.isnan(study.best_value):
        best_hp = _default_gandalf_best_hp()
        logger.warning("GANDALF Optuna sweep scores are NaN; using defaults.")
    else:
        best_hp = dict(study.best_params)
        logger.info(
            f"Best GANDALF Optuna {score_key}={study.best_value:.4f}  params={best_hp}"
        )
        _log_optuna_best_to_wandb(wandb_cfg, "gandalf", study, best_hp, score_key)

    _save_best_hp(best_hp, RES_DIR)
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
