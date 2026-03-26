"""Public importable API for tabnado."""

import json
import os
import warnings
from pathlib import Path
from time import perf_counter

from loguru import logger

from tabnado.data import load_data
from tabnado.evaluate import compute_umap_embeddings, evaluate_model
from tabnado.utils import (
    load_params,
    setup_logger,
    figure_style,
    LOAD_DATA_PARAMS,
)


def run_pipeline(params_path: Path | str | None = None) -> None:
    """Run the full tabnado pipeline."""
    warnings.filterwarnings("ignore")
    figure_style()

    pipeline_start = perf_counter()
    params = load_params(params_path)

    setup_logger(params["RES_DIR"], params["PROJECT"])
    logger.info("========== PIPELINE START ==========")
    logger.info(f"Loaded parameters: {params}")
    logger.info(
        f"Run summary: project={params['PROJECT']} logging={params['LOGGING']} target={params['TARGET']} n_sweeps={params['N_SWEEPS']} sweep_fraction={params['SWEEP_FRACTION']}"
    )
    logger.info(f"Logging directory: {params['LOGGING_DIR']}")
    if params["LOGGING"] == "wandb":
        os.environ["WANDB_DIR"] = params["RES_DIR"]
    elif params["LOGGING"] == "tensorboard":
        os.environ["TENSORBOARD_DIR"] = params["LOGGING_DIR"]

    stage_start = perf_counter()
    logger.info("[stage:data] START load_data")
    _, _, target_cols, feature_cols, train_data, eval_data, test_data = load_data(
        **{k: params[k] for k in LOAD_DATA_PARAMS}
    )
    logger.info(
        "[stage:data] END load_data in {:.2f}s".format(perf_counter() - stage_start)
    )

    model_type = params.get("MODEL_TYPE", "gandalf")
    logger.info(f"Model backend: {model_type}")

    if model_type == "xgboost":
        from tabnado.xgb_sweep import sweep_xgboost
        from tabnado.xgb_train import train_xgboost

        stage_start = perf_counter()
        logger.info("[stage:sweep] START XGBoost hyperparameter sweep")
        best_hp = sweep_xgboost(
            feature_cols=feature_cols,
            target_cols=target_cols,
            train_data=train_data,
            n_sweeps=params["N_SWEEPS"],
            sweep_fraction=params["SWEEP_FRACTION"],
            RES_DIR=params["RES_DIR"],
            LOGGING=params["LOGGING"],
            PROJECT=params["PROJECT"],
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
            RES_DIR=params["RES_DIR"],
            LOGGING=params["LOGGING"],
            PROJECT=params["PROJECT"],
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
            count=params["N_SWEEPS"],
            RES_DIR=params["RES_DIR"],
            SWEEP_FRACTION=params["SWEEP_FRACTION"],
            LOGGING=params["LOGGING"],
            LOGGING_DIR=params["LOGGING_DIR"],
            PROJECT=params["PROJECT"],
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
            PROJECT=params["PROJECT"],
            RES_DIR=params["RES_DIR"],
            LOGGING=params["LOGGING"],
        )
        logger.info(f"Best hyperparameters: {best_hp}")
        best_hp_path = f"{params['RES_DIR']}/best_hyperparameters.json"
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
            **{
                k: params[k]
                for k in (
                    "PROJECT",
                    "MODEL_NAME",
                    "RES_DIR",
                    "LOGGING_DIR",
                    "LOGGING",
                )
            },
        )
        logger.info(
            "[stage:train] END final model training in {:.2f}s".format(
                perf_counter() - stage_start
            )
        )

    stage_start = perf_counter()
    logger.info("[stage:evaluate] START evaluation/umap")
    evaluate_model(
        final_model,
        test_data,
        target_cols,
        feature_cols=feature_cols,
        FIG_DIR=params["FIG_DIR"],
        RES_DIR=params["RES_DIR"],
        model_type=model_type,
    )
    compute_umap_embeddings(
        final_model,
        test_data,
        feature_cols,
        target_cols,
        FIG_DIR=params["FIG_DIR"],
        RES_DIR=params["RES_DIR"],
        target=params["TARGET"],
        model_type=model_type,
    )
    logger.info(
        "[stage:evaluate] END evaluation/umap in {:.2f}s".format(
            perf_counter() - stage_start
        )
    )

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
            RES_DIR=params["RES_DIR"],
            FIG_DIR=params["FIG_DIR"],
        )
    else:
        from tabnado.gandalf_shap import compute_gandalf_shap

        compute_gandalf_shap(
            final_model,
            train_data,
            test_data,
            feature_cols,
            target_cols,
            RES_DIR=params["RES_DIR"],
            FIG_DIR=params["FIG_DIR"],
        )
    logger.info(
        "[stage:shap] END shap analysis in {:.2f}s".format(perf_counter() - stage_start)
    )
    logger.info(
        "========== PIPELINE END ({:.2f}s total) ==========".format(
            perf_counter() - pipeline_start
        )
    )


__all__ = [
    "run_pipeline",
    "load_params",
    "setup_logger",
    "load_data",
    "evaluate_model",
    "compute_umap_embeddings",
]
