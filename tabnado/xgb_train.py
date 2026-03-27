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


def train_xgboost(
    best_hp: dict,
    feature_cols: list[str],
    target_cols: list[str],
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    RES_DIR: str = "results",
    early_stopping_rounds: int = 20,
    wandb_cfg=None,
    **kwargs,
) -> list[xgb.XGBRegressor]:
    """Train one XGBRegressor per target with early stopping and eval logging."""
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
            n_jobs=-1,
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


def main():
    import os

    from tabnado.data import load_data
    from tabnado.params import PipelineParams
    from tabnado.utils import parse_params_arg, setup_logger

    params = PipelineParams.from_yaml(parse_params_arg())
    setup_logger(params.RES_DIR, params.PROJECT)

    hp_path = f"{params.RES_DIR}/best_hyperparameters.json"
    if not os.path.exists(hp_path):
        raise FileNotFoundError(
            f"No best_hyperparameters.json found at {hp_path}. Run xgb_sweep first."
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
    train_xgboost(
        best_hp,
        feature_cols,
        target_cols,
        train_data,
        eval_data,
        RES_DIR=params.RES_DIR,
        wandb_cfg=wandb_cfg,
    )


def load_xgboost_model(RES_DIR: str) -> list[xgb.XGBRegressor]:
    """Load a saved XGBoost model."""
    model_path = Path(RES_DIR) / "final_model" / "xgboost_model.joblib"
    logger.info(f"Loading XGBoost model from {model_path}")
    return joblib.load(model_path)


def predict_xgboost(
    model: list[xgb.XGBRegressor],
    data: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
) -> pd.DataFrame:
    """Run predictions and return a DataFrame with target column names."""
    X = data[feature_cols].values
    preds = np.column_stack([m.predict(X) for m in model])
    return pd.DataFrame(preds, index=data.index, columns=target_cols)
