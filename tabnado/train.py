import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabnado.utils import (
    classification_metrics,
    classification_prediction_frame,
    encode_classification_target,
    flatten_metric_dict,
    json_safe,
    resolve_task,
)

# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


def _train_xgboost_classifier(
    best_hp: dict,
    feature_cols: list[str],
    target_cols: list[str],
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    RES_DIR: str,
    early_stopping_rounds: int,
    wandb_cfg=None,
) -> dict:
    """Train an XGBClassifier for binary or multiclass classification."""
    import xgboost as xgb

    model_dir = Path(RES_DIR) / "final_model"
    os.makedirs(model_dir, exist_ok=True)

    encoded = encode_classification_target(train_data, target_cols, eval_data)
    assert encoded.eval is not None

    X_train = train_data[feature_cols].values
    X_eval = eval_data[feature_cols].values

    objective = (
        "binary:logistic" if encoded.problem_type == "binary" else "multi:softprob"
    )
    model_kwargs = {
        **best_hp,
        "objective": objective,
        "tree_method": "hist",
        "max_bin": 256,
        "n_jobs": 1,
        "random_state": 42,
        "verbosity": 0,
        "early_stopping_rounds": early_stopping_rounds,
        "eval_metric": "logloss" if encoded.problem_type == "binary" else "mlogloss",
    }
    if encoded.problem_type == "multiclass":
        model_kwargs["num_class"] = len(encoded.classes)

    use_wandb = wandb_cfg is not None
    if use_wandb:
        run = wandb_cfg.init_run(
            name=f"{wandb_cfg.model_name}_final_{time.strftime('%Y-%m-%d_%H%M%S')}",
            group="final",
            config={
                "task": "classification",
                "problem_type": encoded.problem_type,
                "early_stopping_rounds": early_stopping_rounds,
                **best_hp,
            },
            reinit="finish_previous",
        )

    model = xgb.XGBClassifier(**model_kwargs)
    logger.info(
        "Training XGBoost classifier for target '{}' ({}, classes={}) on {:,} regions".format(
            encoded.target_col,
            encoded.problem_type,
            encoded.classes,
            len(X_train),
        )
    )
    model.fit(
        X_train,
        encoded.train,
        eval_set=[(X_train, encoded.train), (X_eval, encoded.eval)],
        verbose=False,
    )

    eval_pred_idx = model.predict(X_eval)
    eval_pred = np.asarray(encoded.classes, dtype=object)[eval_pred_idx.astype(int)]
    eval_proba = model.predict_proba(X_eval)
    metrics = classification_metrics(
        eval_data[encoded.target_col],
        eval_pred,
        probabilities=eval_proba,
        classes=encoded.classes,
    )
    logger.info(
        "Eval accuracy={accuracy:.4f} balanced_accuracy={balanced_accuracy:.4f} "
        "macro_f1={macro_f1:.4f}".format(**metrics)
    )

    if use_wandb:
        run.log(flatten_metric_dict(metrics, prefix="eval_", sep="_"))
        run.finish()

    artifact = {
        "task": "classification",
        "problem_type": encoded.problem_type,
        "target_col": encoded.target_col,
        "classes": encoded.classes,
        "model": model,
    }
    model_path = model_dir / "xgboost_model.joblib"
    joblib.dump(artifact, model_path)
    logger.info(f"Saved XGBoost classifier to {model_path}")

    with open(model_dir / "eval_metrics.json", "w") as f:
        json.dump(json_safe(metrics), f, indent=2)

    return artifact


def _train_xgboost(
    best_hp: dict,
    feature_cols: list[str],
    target_cols: list[str],
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    RES_DIR: str = "results",
    early_stopping_rounds: int = 20,
    wandb_cfg=None,
    TASK: str = "regression",
    **kwargs,
):
    """Train one XGBRegressor per target with early stopping and eval logging."""
    import xgboost as xgb

    task = resolve_task(TASK, train_data, target_cols)
    if task == "classification":
        return _train_xgboost_classifier(
            best_hp,
            feature_cols,
            target_cols,
            train_data,
            eval_data,
            RES_DIR,
            early_stopping_rounds,
            wandb_cfg=wandb_cfg,
        )

    model_dir = Path(RES_DIR) / "final_model"
    os.makedirs(model_dir, exist_ok=True)

    X_train = train_data[feature_cols].values
    X_eval = eval_data[feature_cols].values

    use_wandb = wandb_cfg is not None
    if use_wandb:
        run = wandb_cfg.init_run(
            name=f"{wandb_cfg.model_name}_final_{time.strftime('%Y-%m-%d_%H%M%S')}",
            group="final",
            config={"early_stopping_rounds": early_stopping_rounds, **best_hp},
            reinit="finish_previous",
        )

    models = []
    all_eval_preds = []

    for col in target_cols:
        y_train = train_data[col].values
        y_eval = eval_data[col].values

        model = xgb.XGBRegressor(
            **best_hp,
            objective="reg:squarederror",
            tree_method="hist",
            max_bin=256,
            n_jobs=1,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric=["rmse"],
        )

        logger.info(f"Training XGBoost for target '{col}' on {len(X_train):,} regions")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_eval, y_eval)],
            verbose=False,
        )

        evals = model.evals_result()
        train_rmse = evals["validation_0"]["rmse"]
        eval_rmse = evals["validation_1"]["rmse"]
        best_round = model.best_iteration
        logger.info(
            f"  '{col}': best_round={best_round}  "
            f"train_rmse={train_rmse[best_round]:.4f}  eval_rmse={eval_rmse[best_round]:.4f}"
        )

        if use_wandb:
            for rnd, (tr_rmse, ev_rmse) in enumerate(zip(train_rmse, eval_rmse)):
                run.log(
                    {
                        f"{col}/train_rmse": tr_rmse,
                        f"{col}/eval_rmse": ev_rmse,
                    },
                    step=rnd,
                )

        models.append(model)
        all_eval_preds.append(model.predict(X_eval))

    y_eval_mat = eval_data[target_cols].values
    y_pred_mat = np.column_stack(all_eval_preds)
    r2 = float(r2_score(y_eval_mat, y_pred_mat, multioutput="uniform_average"))
    mse = float(mean_squared_error(y_eval_mat, y_pred_mat))
    logger.info(f"Eval R²={r2:.4f}  MSE={mse:.4f}")

    if use_wandb:
        run.log({"eval_r2": r2, "eval_mse": mse})
        run.finish()

    model_path = model_dir / "xgboost_model.joblib"
    joblib.dump(models, model_path)
    logger.info(f"Saved XGBoost model to {model_path}")

    metrics = {"eval_r2": r2, "eval_mse": mse}
    with open(model_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return models


def load_xgboost_model(RES_DIR: str):
    """Load a saved XGBoost model."""
    import xgboost as xgb  # noqa: F401  (ensures backend is importable)

    model_path = Path(RES_DIR) / "final_model" / "xgboost_model.joblib"
    logger.info(f"Loading XGBoost model from {model_path}")
    return joblib.load(model_path)


def predict_xgboost(
    model,
    data: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
) -> pd.DataFrame:
    """Run predictions and return a DataFrame with target column names."""
    X = data[feature_cols].values
    if isinstance(model, dict) and model.get("task") == "classification":
        estimator = model["model"]
        classes = [str(cls) for cls in model["classes"]]
        target_col = model["target_col"]
        pred_idx = estimator.predict(X).astype(int)
        pred_labels = np.asarray(classes, dtype=object)[pred_idx]
        return classification_prediction_frame(
            pred_labels,
            estimator.predict_proba(X),
            target_col,
            classes,
            data.index,
        )

    preds = np.column_stack([m.predict(X) for m in model])
    return pd.DataFrame(preds, index=data.index, columns=target_cols)


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


def _train_catboost_classifier(
    best_hp: dict,
    feature_cols: list[str],
    target_cols: list[str],
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    RES_DIR: str,
    early_stopping_rounds: int,
    wandb_cfg=None,
) -> dict:
    CatBoostClassifier, _ = _import_catboost()
    model_dir = Path(RES_DIR) / "final_model"
    os.makedirs(model_dir, exist_ok=True)

    encoded = encode_classification_target(train_data, target_cols, eval_data)
    assert encoded.eval is not None

    X_train = train_data[feature_cols]
    X_eval = eval_data[feature_cols]

    use_wandb = wandb_cfg is not None
    if use_wandb:
        run = wandb_cfg.init_run(
            name=f"{wandb_cfg.model_name}_final_{time.strftime('%Y-%m-%d_%H%M%S')}",
            group="final",
            config={
                "task": "classification",
                "problem_type": encoded.problem_type,
                "early_stopping_rounds": early_stopping_rounds,
                **best_hp,
            },
            reinit="finish_previous",
        )

    loss_function = "Logloss" if encoded.problem_type == "binary" else "MultiClass"
    model = CatBoostClassifier(
        **best_hp,
        loss_function=loss_function,
        eval_metric="TotalF1",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    logger.info(
        "Training CatBoost classifier for target '{}' ({}, classes={}) on {:,} regions".format(
            encoded.target_col,
            encoded.problem_type,
            encoded.classes,
            len(X_train),
        )
    )
    model.fit(
        X_train,
        encoded.train,
        eval_set=(X_eval, encoded.eval),
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    eval_pred_idx = model.predict(X_eval).astype(int).ravel()
    eval_pred = np.asarray(encoded.classes, dtype=object)[eval_pred_idx]
    eval_proba = model.predict_proba(X_eval)
    metrics = classification_metrics(
        eval_data[encoded.target_col],
        eval_pred,
        probabilities=eval_proba,
        classes=encoded.classes,
    )
    logger.info(
        "Eval accuracy={accuracy:.4f} balanced_accuracy={balanced_accuracy:.4f} "
        "macro_f1={macro_f1:.4f}".format(**metrics)
    )

    if use_wandb:
        run.log(flatten_metric_dict(metrics, prefix="eval_", sep="_"))
        run.finish()

    artifact = {
        "task": "classification",
        "problem_type": encoded.problem_type,
        "target_col": encoded.target_col,
        "classes": encoded.classes,
        "model": model,
    }
    model_path = model_dir / "catboost_model.joblib"
    joblib.dump(artifact, model_path)
    logger.info(f"Saved CatBoost classifier to {model_path}")

    with open(model_dir / "eval_metrics.json", "w") as f:
        json.dump(json_safe(metrics), f, indent=2)

    return artifact


def _train_catboost(
    best_hp: dict,
    feature_cols: list[str],
    target_cols: list[str],
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    RES_DIR: str = "results",
    early_stopping_rounds: int = 20,
    wandb_cfg=None,
    TASK: str = "regression",
    **kwargs,
):
    """Train CatBoost for regression or classification."""
    _, CatBoostRegressor = _import_catboost()
    task = resolve_task(TASK, train_data, target_cols)
    if task == "classification":
        return _train_catboost_classifier(
            best_hp,
            feature_cols,
            target_cols,
            train_data,
            eval_data,
            RES_DIR,
            early_stopping_rounds,
            wandb_cfg=wandb_cfg,
        )

    model_dir = Path(RES_DIR) / "final_model"
    os.makedirs(model_dir, exist_ok=True)

    X_train = train_data[feature_cols]
    X_eval = eval_data[feature_cols]

    use_wandb = wandb_cfg is not None
    if use_wandb:
        run = wandb_cfg.init_run(
            name=f"{wandb_cfg.model_name}_final_{time.strftime('%Y-%m-%d_%H%M%S')}",
            group="final",
            config={"early_stopping_rounds": early_stopping_rounds, **best_hp},
            reinit="finish_previous",
        )

    models = []
    all_eval_preds = []
    for col in target_cols:
        model = CatBoostRegressor(
            **best_hp,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        logger.info(f"Training CatBoost for target '{col}' on {len(X_train):,} regions")
        model.fit(
            X_train,
            train_data[col],
            eval_set=(X_eval, eval_data[col]),
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
        models.append(model)
        all_eval_preds.append(model.predict(X_eval))

    y_eval_mat = eval_data[target_cols].values
    y_pred_mat = np.column_stack(all_eval_preds)
    r2 = float(r2_score(y_eval_mat, y_pred_mat, multioutput="uniform_average"))
    mse = float(mean_squared_error(y_eval_mat, y_pred_mat))
    logger.info(f"Eval R²={r2:.4f}  MSE={mse:.4f}")

    if use_wandb:
        run.log({"eval_r2": r2, "eval_mse": mse})
        run.finish()

    model_path = model_dir / "catboost_model.joblib"
    joblib.dump(models, model_path)
    logger.info(f"Saved CatBoost model to {model_path}")

    with open(model_dir / "eval_metrics.json", "w") as f:
        json.dump({"eval_r2": r2, "eval_mse": mse}, f, indent=2)

    return models


def load_catboost_model(RES_DIR: str):
    model_path = Path(RES_DIR) / "final_model" / "catboost_model.joblib"
    logger.info(f"Loading CatBoost model from {model_path}")
    return joblib.load(model_path)


def predict_catboost(
    model,
    data: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
) -> pd.DataFrame:
    X = data[feature_cols]
    if isinstance(model, dict) and model.get("task") == "classification":
        estimator = model["model"]
        classes = [str(cls) for cls in model["classes"]]
        target_col = model["target_col"]
        pred_idx = estimator.predict(X).astype(int).ravel()
        pred_labels = np.asarray(classes, dtype=object)[pred_idx]
        return classification_prediction_frame(
            pred_labels,
            estimator.predict_proba(X),
            target_col,
            classes,
            data.index,
        )

    preds = np.column_stack([m.predict(X) for m in model])
    return pd.DataFrame(preds, index=data.index, columns=target_cols)


# ---------------------------------------------------------------------------
# GANDALF (PyTorch Tabular)
# ---------------------------------------------------------------------------


def _train_gandalf(
    best_hp: dict,
    feature_cols: list[str],
    target_cols: list[str],
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    RES_DIR: str = "results",
    LOGGING_DIR: str | None = None,
    LOGGING: str = "none",
    wandb_cfg=None,
    TASK: str = "regression",
    **kwargs,
):
    import torch
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import ExperimentConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.models import GANDALFConfig

    from tabnado.sweep import _make_data_config
    from tabnado.utils import (
        LoguruProgressCallback,
        log_macro,
        require_single_classification_target,
    )

    logger.info("Training final GANDALF model with best hyperparameters")
    task = resolve_task(TASK, train_data, target_cols)
    if task == "classification":
        require_single_classification_target(target_cols)

    if eval_data is None or (hasattr(eval_data, "__len__") and len(eval_data) == 0):
        logger.warning(
            "eval_data is empty — deriving a validation split from train_data"
        )
        train_data, eval_data = _derive_validation_split(train_data, target_cols, task)

    logging_dir = LOGGING_DIR or os.path.join(RES_DIR, "logging")
    use_tensorboard = LOGGING == "tensorboard"
    if use_tensorboard:
        os.makedirs(logging_dir, exist_ok=True)
    use_wandb = wandb_cfg is not None
    run_name = (
        f"{wandb_cfg.model_name}_final_{time.strftime('%Y-%m-%d_%H%M')}"
        if use_wandb
        else f"GANDALF_final_{time.strftime('%Y-%m-%d_%H%M')}"
    )
    experiment_project = (
        logging_dir
        if use_tensorboard
        else (wandb_cfg.project if use_wandb else logging_dir)
    )
    if use_wandb:
        wandb_cfg.init_run(name=run_name, group="final", config=best_hp)
    if task == "classification":
        model_config = GANDALFConfig(
            learning_rate=best_hp.get("learning_rate", 1e-2),
            embedding_dropout=best_hp.get("embedding_dropout", 0.01),
            gflu_stages=best_hp.get("gflu_stages", 10),
            gflu_dropout=best_hp.get("gflu_dropout", 0.02),
            gflu_feature_init_sparsity=best_hp.get("gflu_feature_init_sparsity", 0.2),
            learnable_sparsity=False,
            head="LinearHead",
            loss="CrossEntropyLoss",
            metrics=["accuracy"],
            metrics_params=[{}],
            seed=42,
            task="classification",
        )
        early_stopping = "valid_accuracy"
        early_stopping_mode = "max"
    else:
        model_config = GANDALFConfig(
            learning_rate=best_hp.get("learning_rate", 1e-2),
            embedding_dropout=best_hp.get("embedding_dropout", 0.01),
            gflu_stages=best_hp.get("gflu_stages", 10),
            gflu_dropout=best_hp.get("gflu_dropout", 0.02),
            gflu_feature_init_sparsity=best_hp.get("gflu_feature_init_sparsity", 0.2),
            learnable_sparsity=False,
            head="LinearHead",
            loss="MSELoss",
            metrics=["r2_score", "mean_squared_error"],
            metrics_params=[{}] * 2,
            seed=42,
            target_range=[(0, 1)] * len(target_cols),
            task="regression",
        )
        early_stopping = "valid_r2_score"
        early_stopping_mode = "max"

    final_model = TabularModel(
        data_config=_make_data_config(feature_cols, target_cols),
        experiment_config=ExperimentConfig(
            exp_log_freq=1,
            exp_watch=None,
            log_logits=False,
            log_target="wandb" if use_wandb else LOGGING,
            project_name=experiment_project,
            run_name=run_name,
        ),
        model_config=model_config,
        optimizer_config=OptimizerConfig(
            optimizer_params={"weight_decay": best_hp.get("weight_decay", 1e-3)},
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
            checkpoints_path=os.path.join(RES_DIR, "final_model_checkpoints"),
            early_stopping=early_stopping,
            early_stopping_mode=early_stopping_mode,
            early_stopping_patience=5,
            load_best=True,
            max_epochs=100,
            gradient_clip_val=best_hp.get("gradient_clip_val", 1.0),
            trainer_kwargs=dict(enable_model_summary=False),
            progress_bar="none",
        ),
        verbose=False,
        suppress_lightning_logger=True,
    )
    final_model.fit(
        train=train_data, validation=eval_data, callbacks=[LoguruProgressCallback()]
    )
    assert final_model.model is not None
    final_model.save_model(os.path.join(RES_DIR, "final_model"), inference_only=False)
    logger.info(f"Final GANDALF model saved to {RES_DIR}/final_model")

    if use_wandb and task == "regression":
        log_macro(final_model, target_cols)
    return final_model


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

_TRAIN_BACKENDS = {
    "xgboost": _train_xgboost,
    "catboost": _train_catboost,
    "gandalf": _train_gandalf,
}


def _derive_validation_split(
    train_data: pd.DataFrame,
    target_cols: list[str],
    task: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a validation split out of ``train_data`` when ``eval_chr`` is blank.

    All backends require a non-empty eval set (for early stopping / validation
    loss tracking), so an empty ``eval_data`` (no ``eval_chr`` configured) needs
    a stand-in split — stratified by target for classification, where possible.
    """
    if len(train_data) < 4:
        raise ValueError(
            "Need at least 4 training rows to derive a validation split when "
            "eval_chr is blank."
        )

    stratify = None
    if task == "classification":
        target = train_data[target_cols[0]].astype(str)
        class_counts = target.value_counts()
        if len(class_counts) >= 2 and int(class_counts.min()) >= 2:
            stratify = target

    return train_test_split(
        train_data, test_size=0.2, random_state=seed, stratify=stratify
    )


def train_model(
    model_type: str,
    best_hp: dict,
    feature_cols: list[str],
    target_cols: list[str],
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    *,
    TASK: str = "auto",
    **kwargs,
):
    """Dispatch final-model training to the configured backend.

    ``model_type`` is one of ``"xgboost"``, ``"catboost"``, or ``"gandalf"``
    (any other value falls back to GANDALF, matching the legacy backend
    selection in the pipeline API). If ``eval_data`` is empty (``eval_chr``
    left blank), a validation split is carved out of ``train_data`` so the
    backends — which require a non-empty eval set — still have one to use.
    """
    if eval_data is None or eval_data.empty:
        task = resolve_task(TASK, train_data, target_cols)
        logger.warning(
            "eval split is empty (eval_chr left blank) — deriving a validation "
            "split from train_data for final-model training"
        )
        train_data, eval_data = _derive_validation_split(train_data, target_cols, task)

    train_fn = _TRAIN_BACKENDS.get(model_type, _train_gandalf)
    return train_fn(
        best_hp, feature_cols, target_cols, train_data, eval_data, TASK=TASK, **kwargs
    )


def main():
    from tabnado.data import load_data
    from tabnado.params import PipelineParams
    from tabnado.utils import parse_params_arg, setup_logger

    params = PipelineParams.from_yaml(parse_params_arg())
    setup_logger(params.RES_DIR, params.PROJECT)

    hp_path = f"{params.RES_DIR}/best_hyperparameters.json"
    if not os.path.exists(hp_path):
        raise FileNotFoundError(
            f"No best_hyperparameters.json found at {hp_path}. Run the sweep stage first."
        )
    with open(hp_path) as f:
        best_hp = json.load(f)
    logger.info(f"Loaded best hyperparameters from {hp_path}: {best_hp}")

    _, _, target_cols, feature_cols, train_data, eval_data, _ = load_data(
        **vars(params)
    )
    wandb_cfg = None
    if params.LOGGING == "wandb":
        from tabnado.wandb import WandbConfig

        wandb_cfg = WandbConfig.from_params(params)
    train_model(
        params.MODEL_TYPE,
        best_hp,
        feature_cols,
        target_cols,
        train_data,
        eval_data,
        RES_DIR=params.RES_DIR,
        LOGGING_DIR=params.LOGGING_DIR,
        LOGGING=params.LOGGING,
        wandb_cfg=wandb_cfg,
        TASK=params.TASK,
    )


if __name__ == "__main__":
    main()
