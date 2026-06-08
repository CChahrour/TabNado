"""Public importable API for tabnado."""

import json
import os
import warnings
from importlib import metadata
from pathlib import Path
from time import perf_counter

from loguru import logger

from tabnado.data import load_data
from tabnado.params import PipelineParams
from tabnado.utils import (
    figure_style,
    resolve_task,
    setup_logger,
)

ParamsPath = Path | str | None


def load_params(params_path: ParamsPath = None) -> PipelineParams:
    """Load tabnado pipeline parameters.

    Parameters
    ----------
    params_path
        Path to a params YAML file. Defaults to ``params.yaml`` to mirror the CLI.
    """
    resolved_path = Path("params.yaml") if params_path is None else params_path
    return PipelineParams.from_yaml(resolved_path)


def _setup_api_stage(params: PipelineParams, banner: str) -> None:
    params.create_directories()
    setup_logger(params.RES_DIR, params.PROJECT)
    logger.info(f"========== {banner} ==========")
    logger.info(
        f"Config: project={params.PROJECT} logging={params.LOGGING} "
        f"model_type={params.MODEL_TYPE} task={params.TASK} target={params.TARGET}"
    )
    if params.LOGGING == "wandb":
        os.environ["WANDB_DIR"] = params.RES_DIR
    elif params.LOGGING == "tensorboard":
        os.environ["TENSORBOARD_DIR"] = params.LOGGING_DIR


def _make_wandb_config(params: PipelineParams):
    if params.LOGGING != "wandb":
        return None

    from tabnado.wandb import WandbConfig

    return WandbConfig.from_params(params)


def _load_best_hyperparameters(params: PipelineParams) -> dict:
    hp_path = Path(params.RES_DIR) / "best_hyperparameters.json"
    if not hp_path.exists():
        raise FileNotFoundError(
            f"No best_hyperparameters.json found at {hp_path}. Run sweep first."
        )
    with open(hp_path) as f:
        best_hp = json.load(f)
    logger.info(f"Loaded best hyperparameters from {hp_path}: {best_hp}")
    return best_hp


def evaluate_model(*args, **kwargs):
    from tabnado.evaluate import evaluate_model as _evaluate_model

    return _evaluate_model(*args, **kwargs)


def compute_umap_embeddings(*args, **kwargs):
    from tabnado.evaluate import compute_umap_embeddings as _compute_umap_embeddings

    return _compute_umap_embeddings(*args, **kwargs)


def run_pipeline(params_path: Path | str | None = None) -> None:
    """Run the full tabnado pipeline."""
    warnings.filterwarnings("ignore")
    figure_style()

    pipeline_start = perf_counter()
    params = load_params(params_path)
    params.create_directories()

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
    task = resolve_task(params.TASK, train_data, target_cols)
    logger.info(f"Model backend: {model_type}  task: {task}")

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

    from tabnado.sweep import sweep_model
    from tabnado.train import train_model

    stage_start = perf_counter()
    logger.info(f"[stage:sweep] START {model_type} hyperparameter sweep")
    best_hp = sweep_model(
        model_type,
        feature_cols=feature_cols,
        target_cols=target_cols,
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        n_sweeps=params.N_SWEEPS,
        sweep_fraction=params.SWEEP_FRACTION,
        RES_DIR=params.RES_DIR,
        LOGGING=params.LOGGING,
        LOGGING_DIR=params.LOGGING_DIR,
        wandb_cfg=wandb_cfg,
        TASK=task,
    )
    logger.info(
        f"[stage:sweep] END {model_type} sweep in {{:.2f}}s".format(
            perf_counter() - stage_start
        )
    )

    stage_start = perf_counter()
    logger.info(f"[stage:train] START {model_type} final model training")
    final_model = train_model(
        model_type,
        best_hp,
        feature_cols,
        target_cols,
        train_data,
        eval_data,
        RES_DIR=params.RES_DIR,
        LOGGING_DIR=params.LOGGING_DIR,
        LOGGING=params.LOGGING,
        wandb_cfg=wandb_cfg,
        TASK=task,
    )
    logger.info(
        f"[stage:train] END {model_type} training in {{:.2f}}s".format(
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
        task=task,
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
        task=task,
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
    from tabnado.shap import compute_shap

    compute_shap(
        model_type,
        final_model,
        train_data,
        test_data,
        feature_cols,
        target_cols,
        eval_data=eval_data,
        RES_DIR=params.RES_DIR,
        FIG_DIR=params.FIG_DIR,
        task=task,
        wandb_run=_eval_wandb_run,
    )
    logger.info(
        "[stage:shap] END shap analysis in {:.2f}s".format(
            perf_counter() - stage_start
        )
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


def run_data(params_path: ParamsPath = None):
    """Run the data loading/build stage and return loaded split data."""
    params = load_params(params_path)
    _setup_api_stage(params, "MAKE DATASET")
    loaded = load_data(**vars(params))
    _, _, target_cols, feature_cols, train_data, eval_data, test_data = loaded
    logger.info(
        f"train={train_data.shape}  eval={eval_data.shape}  test={test_data.shape}"
    )
    logger.info(f"features={len(feature_cols)}  targets={target_cols}")
    return loaded


def run_sweep(params_path: ParamsPath = None) -> dict:
    """Run the backend-specific hyperparameter sweep and return best HP."""
    params = load_params(params_path)
    _setup_api_stage(params, f"{params.MODEL_TYPE.upper()} SWEEP START")
    run_start = perf_counter()
    wandb_cfg = _make_wandb_config(params)

    _, _, target_cols, feature_cols, train_data, eval_data, test_data = load_data(
        **vars(params)
    )
    task = resolve_task(params.TASK, train_data, target_cols)

    from tabnado.sweep import sweep_model

    best_hp = sweep_model(
        params.MODEL_TYPE,
        feature_cols=feature_cols,
        target_cols=target_cols,
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        n_sweeps=params.N_SWEEPS,
        sweep_fraction=params.SWEEP_FRACTION,
        RES_DIR=params.RES_DIR,
        LOGGING=params.LOGGING,
        LOGGING_DIR=params.LOGGING_DIR,
        wandb_cfg=wandb_cfg,
        TASK=task,
    )

    logger.info(
        "========== SWEEP END ({:.2f}s total) ==========".format(
            perf_counter() - run_start
        )
    )
    return best_hp


def run_train(params_path: ParamsPath = None):
    """Run final model training for the configured backend and return the model."""
    params = load_params(params_path)
    _setup_api_stage(params, f"{params.MODEL_TYPE.upper()} TRAIN START")
    run_start = perf_counter()
    wandb_cfg = _make_wandb_config(params)
    best_hp = _load_best_hyperparameters(params)

    _, _, target_cols, feature_cols, train_data, eval_data, _ = load_data(
        **vars(params)
    )
    task = resolve_task(params.TASK, train_data, target_cols)

    from tabnado.train import train_model

    final_model = train_model(
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
        TASK=task,
    )

    logger.info(
        "========== TRAIN END ({:.2f}s total) ==========".format(
            perf_counter() - run_start
        )
    )
    return final_model


def run_evaluate(params_path: ParamsPath = None) -> None:
    """Run evaluation and UMAP for the configured backend."""
    params = load_params(params_path)
    _setup_api_stage(params, "EVALUATE START")
    run_start = perf_counter()

    _, _, target_cols, feature_cols, _, _, test_data = load_data(**vars(params))
    task = resolve_task(params.TASK, test_data, target_cols)

    model_path = Path(params.RES_DIR) / "final_model"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}. Run train first.")

    if params.MODEL_TYPE == "xgboost":
        from joblib import load

        final_model = load(model_path / "xgboost_model.joblib")
    elif params.MODEL_TYPE == "catboost":
        from joblib import load

        final_model = load(model_path / "catboost_model.joblib")
    else:
        from pytorch_tabular import TabularModel

        final_model = TabularModel.load_model(model_path)
        if final_model.model is None:
            raise RuntimeError("Loaded model has no weights — check model directory")

    evaluate_model(
        final_model,
        test_data,
        target_cols,
        feature_cols=feature_cols,
        FIG_DIR=params.FIG_DIR,
        RES_DIR=params.RES_DIR,
        model_type=params.MODEL_TYPE,
        task=task,
    )
    compute_umap_embeddings(
        final_model,
        test_data,
        feature_cols,
        target_cols,
        FIG_DIR=params.FIG_DIR,
        RES_DIR=params.RES_DIR,
        target=params.TARGET,
        model_type=params.MODEL_TYPE,
        task=task,
    )
    logger.info(
        "========== EVALUATE END ({:.2f}s total) ==========".format(
            perf_counter() - run_start
        )
    )


def run_shap(params_path: ParamsPath = None) -> None:
    """Run backend-specific SHAP analysis."""
    params = load_params(params_path)
    _setup_api_stage(params, f"{params.MODEL_TYPE.upper()} SHAP START")
    run_start = perf_counter()

    model_path = Path(params.RES_DIR) / "final_model"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}. Run train first.")

    _, _, target_cols, feature_cols, train_data, eval_data, test_data = load_data(
        **vars(params)
    )
    task = resolve_task(params.TASK, train_data, target_cols)

    from tabnado.shap import _load_final_model, compute_shap

    final_model = _load_final_model(params.MODEL_TYPE, params.RES_DIR)
    compute_shap(
        params.MODEL_TYPE,
        final_model,
        train_data,
        test_data,
        feature_cols,
        target_cols,
        eval_data=eval_data,
        RES_DIR=params.RES_DIR,
        FIG_DIR=params.FIG_DIR,
        task=task,
    )

    logger.info(
        "========== SHAP END ({:.2f}s total) ==========".format(
            perf_counter() - run_start
        )
    )


def write_params_template(path: Path | str = "params.yaml", force: bool = False) -> Path:
    """Write a starter params YAML file, matching ``tabnado-init``."""
    from tabnado.cli import PARAMS_TEMPLATE

    output_path = Path(path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists. Pass force=True to overwrite it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(PARAMS_TEMPLATE, encoding="utf-8")
    logger.info(f"Wrote template params file to {output_path}")
    return output_path


run = run_pipeline
data = run_data
sweep = run_sweep
train = run_train
evaluate = run_evaluate
shap = run_shap
init = write_params_template


__all__ = [
    "PipelineParams",
    "load_params",
    "run_pipeline",
    "run_data",
    "run_sweep",
    "run_train",
    "run_evaluate",
    "run_shap",
    "write_params_template",
    "run",
    "data",
    "sweep",
    "train",
    "evaluate",
    "shap",
    "init",
    "setup_logger",
    "load_data",
    "evaluate_model",
    "compute_umap_embeddings",
    "__version__",
]

try:
    __version__ = metadata.version("tabnado")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"
