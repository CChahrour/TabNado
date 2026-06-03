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
from tabnado.tasks import resolve_task
from tabnado.utils import (
    figure_style,
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
            TASK=task,
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
            TASK=task,
        )
        logger.info(
            "[stage:train] END XGBoost training in {:.2f}s".format(
                perf_counter() - stage_start
            )
        )
    elif model_type == "catboost":
        from tabnado.catboost_sweep import sweep_catboost
        from tabnado.catboost_train import train_catboost

        stage_start = perf_counter()
        logger.info("[stage:sweep] START CatBoost hyperparameter sweep")
        best_hp = sweep_catboost(
            feature_cols=feature_cols,
            target_cols=target_cols,
            train_data=train_data,
            n_sweeps=params.N_SWEEPS,
            sweep_fraction=params.SWEEP_FRACTION,
            RES_DIR=params.RES_DIR,
            wandb_cfg=wandb_cfg,
            TASK=task,
        )
        logger.info(
            "[stage:sweep] END CatBoost sweep in {:.2f}s".format(
                perf_counter() - stage_start
            )
        )

        stage_start = perf_counter()
        logger.info("[stage:train] START CatBoost final model training")
        final_model = train_catboost(
            best_hp,
            feature_cols,
            target_cols,
            train_data,
            eval_data,
            RES_DIR=params.RES_DIR,
            wandb_cfg=wandb_cfg,
            TASK=task,
        )
        logger.info(
            "[stage:train] END CatBoost training in {:.2f}s".format(
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
            TASK=task,
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
            TASK=task,
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
    if model_type == "xgboost":
        from tabnado.xgb_shap import compute_xgb_shap

        compute_xgb_shap(
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
    elif model_type == "catboost":
        from tabnado.catboost_shap import compute_catboost_shap

        compute_catboost_shap(
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
    else:
        from tabnado.gandalf_shap import compute_gandalf_shap

        compute_gandalf_shap(
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

    if params.MODEL_TYPE == "xgboost":
        from tabnado.xgb_sweep import sweep_xgboost

        best_hp = sweep_xgboost(
            feature_cols=feature_cols,
            target_cols=target_cols,
            train_data=train_data,
            n_sweeps=params.N_SWEEPS,
            sweep_fraction=params.SWEEP_FRACTION,
            RES_DIR=params.RES_DIR,
            wandb_cfg=wandb_cfg,
            TASK=task,
        )
    elif params.MODEL_TYPE == "catboost":
        from tabnado.catboost_sweep import sweep_catboost

        best_hp = sweep_catboost(
            feature_cols=feature_cols,
            target_cols=target_cols,
            train_data=train_data,
            n_sweeps=params.N_SWEEPS,
            sweep_fraction=params.SWEEP_FRACTION,
            RES_DIR=params.RES_DIR,
            wandb_cfg=wandb_cfg,
            TASK=task,
        )
    else:
        from tabnado.gandalf_sweep import get_best_hp_from_sweep, start_sweep_and_run

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
            TASK=task,
        )
        best_hp = get_best_hp_from_sweep(
            sweep_id,
            RES_DIR=params.RES_DIR,
            wandb_cfg=wandb_cfg,
        )
        logger.info(f"Best hyperparameters: {best_hp}")
        hp_path = Path(params.RES_DIR) / "best_hyperparameters.json"
        with open(hp_path, "w") as f:
            json.dump(best_hp, f, indent=4)
        logger.info(f"Saved best hyperparameters to {hp_path}")

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

    if params.MODEL_TYPE == "xgboost":
        from tabnado.xgb_train import train_xgboost

        final_model = train_xgboost(
            best_hp,
            feature_cols,
            target_cols,
            train_data,
            eval_data,
            RES_DIR=params.RES_DIR,
            wandb_cfg=wandb_cfg,
            TASK=task,
        )
    elif params.MODEL_TYPE == "catboost":
        from tabnado.catboost_train import train_catboost

        final_model = train_catboost(
            best_hp,
            feature_cols,
            target_cols,
            train_data,
            eval_data,
            RES_DIR=params.RES_DIR,
            wandb_cfg=wandb_cfg,
            TASK=task,
        )
    else:
        from tabnado.gandalf_train import train_final_model

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

    if params.MODEL_TYPE == "xgboost":
        from joblib import load
        from tabnado.xgb_shap import compute_xgb_shap

        final_model = load(model_path / "xgboost_model.joblib")
        compute_xgb_shap(
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
    elif params.MODEL_TYPE == "catboost":
        from joblib import load
        from tabnado.catboost_shap import compute_catboost_shap

        final_model = load(model_path / "catboost_model.joblib")
        compute_catboost_shap(
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
    else:
        from pytorch_tabular import TabularModel
        from tabnado.gandalf_shap import compute_gandalf_shap

        final_model = TabularModel.load_model(model_path)
        if final_model.model is None:
            raise RuntimeError("Loaded model has no weights — check model directory")
        compute_gandalf_shap(
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
    from tabnado.init import PARAMS_TEMPLATE

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
