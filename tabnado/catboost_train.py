import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_squared_error, r2_score

from tabnado.tasks import (
    classification_metrics,
    classification_prediction_frame,
    encode_classification_target,
    json_safe,
    resolve_task,
)


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

    loss_function = (
        "Logloss" if encoded.problem_type == "binary" else "MultiClass"
    )
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
        run.log({f"eval_{k}": v for k, v in metrics.items()})
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


def train_catboost(
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
