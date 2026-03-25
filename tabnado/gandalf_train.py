import json
import os
from time import perf_counter

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

from tabnado.gandalf_sweep import _make_data_config
from tabnado.utils import LOAD_DATA_PARAMS, LoguruProgressCallback


def train_final_model(
    best_hp: dict,
    feature_cols: list[str],
    target_cols: list[str],
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    PROJECT: str = "GANDALF_PROJECT",
    MODEL_NAME: str = "GANDALF",
    RES_DIR: str = "results",
    LOGGING_DIR: str | None = None,
    date: str = "",
    LOGGING: str = "wandb",
) -> TabularModel:
    logger.info("Training final GANDALF model with best hyperparameters")
    logging_dir = LOGGING_DIR or os.path.join(RES_DIR, "logging")
    os.makedirs(logging_dir, exist_ok=True)
    experiment_project = logging_dir if LOGGING == "tensorboard" else PROJECT
    final_model = TabularModel(
        data_config=_make_data_config(feature_cols, target_cols),
        experiment_config=ExperimentConfig(
            exp_log_freq=1,
            exp_watch=None,
            log_logits=False,
            log_target=LOGGING,
            project_name=experiment_project,
            run_name=f"{MODEL_NAME}_final_{date}",
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
    return final_model


def main():
    from tabnado.data import load_data
    from tabnado.utils import load_params, parse_params_arg, setup_logger

    params = load_params(parse_params_arg())
    setup_logger(params["RES_DIR"], params["PROJECT"])
    run_start = perf_counter()
    logger.info("========== GANDALF TRAIN START ==========")
    logger.info(
        "Train config: project={} logging={} model_name={}".format(
            params["PROJECT"], params["LOGGING"], params["MODEL_NAME"]
        )
    )
    if params["LOGGING"] == "wandb":
        os.environ["WANDB_DIR"] = params["RES_DIR"]

    hp_path = f"{params['RES_DIR']}/best_hyperparameters.json"
    if not os.path.exists(hp_path):
        raise FileNotFoundError(
            f"No best_hyperparameters.json found at {hp_path}. Run gandalf_sweep.py first."
        )
    with open(hp_path) as f:
        best_hp = json.load(f)
    logger.info(f"Loaded best hyperparameters from {hp_path}: {best_hp}")

    _, _, target_cols, feature_cols, train_data, eval_data, _ = load_data(
        **{k: params[k] for k in LOAD_DATA_PARAMS}
    )
    train_final_model(
        best_hp,
        feature_cols,
        target_cols,
        train_data,
        eval_data,
        **{
            k: params[k]
            for k in (
                "PROJECT",
                "MODEL_NAME",
                "RES_DIR",
                "LOGGING_DIR",
                "date",
                "LOGGING",
            )
        },
    )
    logger.info(
        "========== GANDALF TRAIN END ({:.2f}s total) ==========".format(
            perf_counter() - run_start
        )
    )


if __name__ == "__main__":
    main()
