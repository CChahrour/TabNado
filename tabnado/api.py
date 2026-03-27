"""Public importable API for tabnado."""

import json
import os
import warnings
from pathlib import Path
from time import perf_counter

from loguru import logger

from tabnado.data import load_data
from tabnado.evaluate import compute_umap_embeddings, evaluate_model
from tabnado.params import PipelineParams
from tabnado.utils import (
    figure_style,
    setup_logger,
)


def run_pipeline(params_path: Path | str | None = None) -> None:
    """Run the full tabnado pipeline."""
    warnings.filterwarnings("ignore")
    figure_style()

    pipeline_start = perf_counter()
    params = PipelineParams.from_yaml(params_path)

    setup_logger(params.RES_DIR, params.PROJECT)
    logger.info("========== PIPELINE START ==========")
    logger.info(f"Loaded parameters: {params}")
    logger.info(
        f"Run summary: project={params.PROJECT} logging={params.LOGGING} target={params.TARGET} n_sweeps={params.N_SWEEPS} sweep_fraction={params.SWEEP_FRACTION}"
    )
    logger.info(f"Logging directory: {params.LOGGING_DIR}")
    if params.LOGGING == "wandb":
        os.environ["WANDB_DIR"] = params.RES_DIR
    elif params.LOGGING == "tensorboard":
        os.environ["TENSORBOARD_DIR"] = params.LOGGING_DIR

    stage_start = perf_counter()
    logger.info("[stage:data] START load_data")
    _, _, target_cols, feature_cols, train_data, eval_data, test_data = load_data(
        **vars(params)
    )
    logger.info(
        "[stage:data] END load_data in {:.2f}s".format(perf_counter() - stage_start)
    )

    model_type = params.MODEL_TYPE
    logger.info(f"Model backend: {model_type}")

    # Setup wandb config if needed
    wandb_cfg = None
    if params.LOGGING == "wandb":
        from tabnado.wandb import WandbConfig

        wandb_cfg = WandbConfig(
            project=params.PROJECT,
            entity=params.ENTITY,
            model_name=params.MODEL_TYPE,
            target=params.TARGET,
            res_dir=params.RES_DIR,
        )

    if model_type == "xgboost":
        from tabnado.xgb_sweep import sweep_xgboost
        from tabnado.xgb_train import train_xgboost

        stage_start = perf_counter()
        logger.info("[stage:sweep] START XGBoost hyperparameter sweep")
        best_hp = sweep_xgboost(
            feature_cols=feature_cols,
            target_cols=target_cols,
            train_data=train_data,
            n_sweeps=params.N_SWEEPS,
            sweep_fraction=params.SWEEP_FRACTION,
            RES_DIR=params.RES_DIR,
            wandb_cfg=wandb_cfg,
        )
        logger.info(
            "[stage:sweep] END XGBoost sweep in {:.2f}s".format(
                perf_counter() - stage_start
            )
        )

        stage_start = perf_counter()
        logger.info("[stage:train] START XGBoost final model training")
        final_model = train_xgboost(
            best_hp,
            feature_cols,
            target_cols,
            train_data,
            eval_data,
            RES_DIR=params.RES_DIR,
            wandb_cfg=wandb_cfg,
        )
        logger.info(
            "[stage:train] END XGBoost training in {:.2f}s".format(
                perf_counter() - stage_start
            )
        )
    else:
        from tabnado.gandalf_sweep import (
            get_best_hp_from_sweep,
            start_sweep_and_run,
        )
        from tabnado.gandalf_train import train_final_model

        stage_start = perf_counter()
        logger.info("[stage:sweep] START hyperparameter sweep")
        sweep_id = start_sweep_and_run(
            train_data,
            eval_data,
            test_data,
            feature_cols,
            target_cols,
            count=params.N_SWEEPS,
            RES_DIR=params.RES_DIR,
            SWEEP_FRACTION=params.SWEEP_FRACTION,
            LOGGING=params.LOGGING,
            LOGGING_DIR=params.LOGGING_DIR,
            wandb_cfg=wandb_cfg,
        )
        logger.info(
            "[stage:sweep] END hyperparameter sweep in {:.2f}s (sweep_id={})".format(
                perf_counter() - stage_start, sweep_id
            )
        )

        stage_start = perf_counter()
        logger.info("[stage:sweep] START best-hp selection")
        best_hp = get_best_hp_from_sweep(
            sweep_id,
            RES_DIR=params.RES_DIR,
            wandb_cfg=wandb_cfg,
        )
        logger.info(f"Best hyperparameters: {best_hp}")
        best_hp_path = f"{params.RES_DIR}/best_hyperparameters.json"
        with open(best_hp_path, "w") as f:
            json.dump(best_hp, f, indent=4)
        logger.info(
            "[stage:sweep] END best-hp selection in {:.2f}s (saved={})".format(
                perf_counter() - stage_start, best_hp_path
            )
        )

        stage_start = perf_counter()
        logger.info("[stage:train] START final model training")
        final_model = train_final_model(
            best_hp,
            feature_cols,
            target_cols,
            train_data,
            eval_data,
            RES_DIR=params.RES_DIR,
            LOGGING_DIR=params.LOGGING_DIR,
            LOGGING=params.LOGGING,
            wandb_cfg=wandb_cfg,
        )
        logger.info(
            "[stage:train] END final model training in {:.2f}s".format(
                perf_counter() - stage_start
            )
        )

    # Open a fresh wandb run for evaluation + SHAP (sweep/train stages finish any prior run)
    _eval_wandb_run = None
    if wandb_cfg is not None:
        _eval_wandb_run = wandb_cfg.init_run(
            name=f"{params.PROJECT}_eval",
            group=params.MODEL_TYPE,
            reinit="finish_previous",
        )

    # Evaluation
    stage_start = perf_counter()
    logger.info("[stage:evaluate] START evaluation/umap")
    evaluate_model(
        final_model,
        test_data,
        target_cols,
        feature_cols=feature_cols,
        FIG_DIR=params.FIG_DIR,
        RES_DIR=params.RES_DIR,
        model_type=model_type,
        wandb_run=_eval_wandb_run,
    )
    compute_umap_embeddings(
        final_model,
        test_data,
        feature_cols,
        target_cols,
        FIG_DIR=params.FIG_DIR,
        RES_DIR=params.RES_DIR,
        target=params.TARGET,
        model_type=model_type,
        wandb_run=_eval_wandb_run,
    )
    logger.info(
        "[stage:evaluate] END evaluation/umap in {:.2f}s".format(
            perf_counter() - stage_start
        )
    )

    # SHAP analysis
    stage_start = perf_counter()
    logger.info("[stage:shap] START shap analysis")
    if model_type == "xgboost":
        from tabnado.xgb_shap import compute_xgb_shap

        compute_xgb_shap(
            final_model,
            train_data,
            test_data,
            feature_cols,
            target_cols,
            RES_DIR=params.RES_DIR,
            FIG_DIR=params.FIG_DIR,
            wandb_run=_eval_wandb_run,
        )
    else:
        from tabnado.gandalf_shap import compute_gandalf_shap

        compute_gandalf_shap(
            final_model,
            train_data,
            test_data,
            feature_cols,
            target_cols,
            RES_DIR=params.RES_DIR,
            FIG_DIR=params.FIG_DIR,
            wandb_run=_eval_wandb_run,
        )
    logger.info(
        "[stage:shap] END shap analysis in {:.2f}s".format(perf_counter() - stage_start)
    )

    # W&B report
    if _eval_wandb_run is not None:
        _wandb_run_id = _eval_wandb_run.id
        _eval_wandb_run.finish()
        from tabnado.wandb import create_eval_report

        try:
            report_url = create_eval_report(
                wandb_cfg=wandb_cfg,
                run_id=_wandb_run_id,
                target_cols=target_cols,
            )
            logger.info(f"W&B report: {report_url}")
        except Exception as e:
            logger.warning(f"W&B report creation failed (skipping): {e}")

    logger.info(
        "========== PIPELINE END ({:.2f}s total) ==========".format(
            perf_counter() - pipeline_start
        )
    )


__all__ = [
    "run_pipeline",
    "setup_logger",
    "load_data",
    "evaluate_model",
    "compute_umap_embeddings",
]
