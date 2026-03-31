import json
import os
import time

import pandas as pd
import torch
from loguru import logger
from pytorch_tabular import TabularModel
from pytorch_tabular.config import (
    ExperimentConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.models import GANDALFConfig

from tabnado.data import load_data
from tabnado.gandalf_sweep import _make_data_config
from tabnado.params import PipelineParams
from tabnado.utils import (
    LoguruProgressCallback,
    log_macro,
    parse_params_arg,
    setup_logger,
)


def train_final_model(
    best_hp: dict,
    feature_cols: list[str],
    target_cols: list[str],
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    RES_DIR: str = "results",
    LOGGING_DIR: str | None = None,
    LOGGING: str = "none",
    wandb_cfg=None,
) -> TabularModel:
    logger.info("Training final GANDALF model with best hyperparameters")
    logging_dir = LOGGING_DIR or os.path.join(RES_DIR, "logging")
    os.makedirs(logging_dir, exist_ok=True)
    use_wandb = wandb_cfg is not None
    run_name = (
        f"{wandb_cfg.model_name}_final_{time.strftime('%Y-%m-%d_%H%M')}"
        if use_wandb
        else f"GANDALF_final_{time.strftime('%Y-%m-%d_%H%M')}"
    )
    experiment_project = (
        logging_dir
        if LOGGING == "tensorboard"
        else (wandb_cfg.project if use_wandb else logging_dir)
    )
    if use_wandb:
        wandb_cfg.init_run(name=run_name, group="final", config=best_hp)
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
        model_config=GANDALFConfig(
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
        ),
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
            early_stopping="valid_r2_score",
            early_stopping_mode="max",
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

    if use_wandb:
        log_macro(final_model, target_cols)
    return final_model


def main():
    params = PipelineParams.from_yaml(parse_params_arg())
    setup_logger(params.RES_DIR, params.PROJECT)
    run_start = time.perf_counter()
    logger.info("========== GANDALF TRAIN START ==========")
    logger.info(
        f"Train config: project={params.PROJECT} logging={params.LOGGING} model_name={params.MODEL_TYPE}"
    )
    wandb_cfg = None
    if params.LOGGING == "wandb":
        from tabnado.wandb import WandbConfig

        wandb_cfg = WandbConfig.from_params(params)

    hp_path = f"{params.RES_DIR}/best_hyperparameters.json"
    if not os.path.exists(hp_path):
        raise FileNotFoundError(
            f"No best_hyperparameters.json found at {hp_path}. Run gandalf_sweep.py first."
        )
    with open(hp_path) as f:
        best_hp = json.load(f)
    logger.info(f"Loaded best hyperparameters from {hp_path}: {best_hp}")

    _, _, target_cols, feature_cols, train_data, eval_data, _ = load_data(
        **vars(params)
    )
    train_final_model(
        best_hp,
        feature_cols,
        target_cols,
        train_data,
        eval_data,
        RES_DIR=params.RES_DIR,
        LOGGING=params.LOGGING,
        LOGGING_DIR=params.LOGGING_DIR,
        wandb_cfg=wandb_cfg,
    )
    logger.info(
        f"========== GANDALF TRAIN END ({time.perf_counter() - run_start:.2f}s total) =========="
    )


if __name__ == "__main__":
    main()
