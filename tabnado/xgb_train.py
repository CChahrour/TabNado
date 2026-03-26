import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor


def train_xgboost(
    best_hp: dict,
    feature_cols: list[str],
    target_cols: list[str],
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    RES_DIR: str = "results",
    LOGGING: str = "wandb",
    PROJECT: str = "PROJECT_NAME",
    MODEL_NAME: str = "XGBoost",
    **kwargs,
) -> MultiOutputRegressor | xgb.XGBRegressor:
    """Train XGBoost model with best hyperparameters from sweep."""
    model_dir = Path(RES_DIR) / "final_model"
    os.makedirs(model_dir, exist_ok=True)

    base = xgb.XGBRegressor(
        **best_hp,
        objective="reg:squarederror",
        tree_method="hist",
        max_bin=256,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )

    if len(target_cols) == 1:
        model = base
    else:
        model = MultiOutputRegressor(base, n_jobs=1)

    X_train = train_data[feature_cols].values
    y_train = train_data[target_cols].values
    X_eval = eval_data[feature_cols].values
    y_eval = eval_data[target_cols].values

    if len(target_cols) == 1:
        y_train = y_train.ravel()
        y_eval = y_eval.ravel()

    logger.info(
        f"Training XGBoost on {len(X_train):,} regions with {len(feature_cols)} features"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)
    if y_eval.ndim == 1:
        r2 = r2_score(y_eval, y_pred)
        mse = mean_squared_error(y_eval, y_pred)
    else:
        r2 = r2_score(y_eval, y_pred, multioutput="uniform_average")
        mse = mean_squared_error(y_eval, y_pred, multioutput="uniform_average")
    logger.info(f"Eval R²={r2:.4f}  MSE={mse:.4f}")

    if LOGGING == "wandb":
        import wandb

        with wandb.init(
            project=PROJECT,
            dir=RES_DIR,
            reinit="finish_previous",
            name=f"{MODEL_NAME}_final_{time.strftime('%Y-%m-%d_%H%M%S')}",
            tags=["xgb-train"],
        ):
            wandb.log({"eval_r2": float(r2), "eval_mse": float(mse)})

    model_path = model_dir / "xgboost_model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Saved XGBoost model to {model_path}")

    metrics = {"eval_r2": float(r2), "eval_mse": float(mse)}
    with open(model_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model


def load_xgboost_model(RES_DIR: str) -> MultiOutputRegressor | xgb.XGBRegressor:
    """Load a saved XGBoost model."""
    model_path = Path(RES_DIR) / "final_model" / "xgboost_model.joblib"
    logger.info(f"Loading XGBoost model from {model_path}")
    return joblib.load(model_path)


def predict_xgboost(
    model: MultiOutputRegressor | xgb.XGBRegressor,
    data: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
) -> pd.DataFrame:
    """Run predictions and return a DataFrame with target column names."""
    X = data[feature_cols].values
    preds = model.predict(X)
    if preds.ndim == 1:
        preds = preds[:, np.newaxis]
    return pd.DataFrame(preds, index=data.index, columns=target_cols)
