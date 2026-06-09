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
    """Load and validate tabnado pipeline parameters from a YAML file.

    Reads the YAML at ``params_path``, validates the required/optional keys
    (``logging``, ``model_name``, ``task``, ``catboost_search_space``, ...),
    derives the run's directory layout (``RES_DIR``/``FIG_DIR``/
    ``LOGGING_DIR``/``DATA_DIR`` from ``output_dir``/``model_name``/``target``),
    and returns the resulting :class:`~tabnado.params.PipelineParams`. This is
    the entry point every ``run_*`` function in this module uses to resolve
    its configuration — call it directly if you want to inspect or tweak
    parameters (e.g. ``params.RES_DIR``, ``params.MODEL_TYPE``) before driving
    the pipeline stages yourself.

    Parameters
    ----------
    params_path
        Path to a params YAML file (e.g. one written by
        :func:`write_params_template`/``tabnado-init``). Defaults to
        ``params.yaml`` in the current working directory, mirroring the CLI's
        ``--params``/``-p`` default.

    Returns
    -------
    PipelineParams
        The validated, fully-resolved parameter set (a frozen view of the
        YAML plus derived fields) used to configure every pipeline stage.

    Raises
    ------
    ValueError
        If required keys (``dataset``, ``model_name``) are missing/empty, or
        if ``logging``/``model_name``/``task``/``catboost_search_space`` hold
        values outside their respective allowed sets
        (:data:`~tabnado.params.VALID_LOGGING_BACKENDS`,
        :data:`~tabnado.params.VALID_MODEL_TYPES`,
        :data:`~tabnado.params.VALID_TASKS`,
        :data:`~tabnado.params.VALID_CATBOOST_SEARCH_SPACES`).
    FileNotFoundError
        If ``params_path`` does not exist.
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


def _balanced_train_data(
    params: PipelineParams,
    task: str,
    train_data,
    target_cols: list[str],
):
    """Resample ``train_data`` per ``params.CLASS_BALANCE`` for fitting/tuning.

    Mirrors the notebook's ``RandomUnderSampler`` step — applied only to the
    data the model fits/tunes on. Returns ``train_data`` unchanged for
    regression tasks or ``class_balance: none``.
    """
    if task != "classification" or params.CLASS_BALANCE == "none":
        return train_data

    from tabnado.data import balance_classes
    from tabnado.utils import require_single_classification_target

    target_col = require_single_classification_target(target_cols)
    return balance_classes(train_data, target_col, method=params.CLASS_BALANCE)


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
    """Compute test-set metrics and diagnostic plots for a trained model.

    Thin lazy-import wrapper around :func:`tabnado.evaluate.evaluate_model`,
    kept here so it can be imported from :mod:`tabnado.api` without pulling in
    the (heavier) evaluation dependencies at module import time.

    Parameters
    ----------
    *args, **kwargs
        Forwarded verbatim to :func:`tabnado.evaluate.evaluate_model`. In
        practice this is called with the trained backend model, the held-out
        ``test_data`` frame, ``target_cols``, and keyword arguments
        ``feature_cols``, ``FIG_DIR``, ``RES_DIR``, ``model_type``, ``task``,
        and (optionally) ``wandb_run``.

    Returns
    -------
    Whatever :func:`tabnado.evaluate.evaluate_model` returns — typically a
    dict of computed metrics. Regression runs get scatter/residual plots and
    regression metrics (R2, MSE, ...); classification runs get ROC curves,
    a confusion matrix, and classification metrics (accuracy, precision,
    recall, weighted F1). Figures are written under ``FIG_DIR`` and metrics
    are saved as JSON under ``RES_DIR``; if a ``wandb_run`` is supplied the
    metrics and figures are also logged there.
    """
    from tabnado.evaluate import evaluate_model as _evaluate_model

    return _evaluate_model(*args, **kwargs)


def compute_umap_embeddings(*args, **kwargs):
    """Compute and plot a UMAP projection of the test-set feature space.

    Thin lazy-import wrapper around
    :func:`tabnado.evaluate.compute_umap_embeddings`, kept here so it can be
    imported from :mod:`tabnado.api` without pulling in the UMAP/evaluation
    dependencies at module import time.

    Parameters
    ----------
    *args, **kwargs
        Forwarded verbatim to :func:`tabnado.evaluate.compute_umap_embeddings`.
        In practice this is called with the trained backend model, the
        held-out ``test_data`` frame, ``feature_cols``, ``target_cols``, and
        keyword arguments ``FIG_DIR``, ``RES_DIR``, ``target``, ``model_type``,
        ``task``, and (optionally) ``wandb_run``.

    Returns
    -------
    Whatever :func:`tabnado.evaluate.compute_umap_embeddings` returns —
    typically the embedding array/frame. A 2D UMAP scatter plot coloured by
    the target (or, for classification, by predicted class) is saved under
    ``FIG_DIR``; if a ``wandb_run`` is supplied the plot is also logged there.
    """
    from tabnado.evaluate import compute_umap_embeddings as _compute_umap_embeddings

    return _compute_umap_embeddings(*args, **kwargs)


def run_pipeline(params_path: Path | str | None = None) -> None:
    """Run the full tabnado pipeline end-to-end: data -> sweep -> train -> evaluate -> SHAP.

    This mirrors the ``tabnado-run`` CLI entry point and is the one-call way
    to reproduce a complete experiment from a params YAML file. It loads (or
    builds) the train/eval/test splits, runs a backend-specific Optuna
    hyperparameter sweep, trains the final model with the best
    hyperparameters, evaluates it on the held-out test set (metrics + UMAP),
    and computes SHAP feature-importance plots — logging timing and a summary
    for every stage. If ``logging: wandb`` is configured, a fresh run is
    opened for the evaluation/SHAP stages and an evaluation report is created
    at the end (failures there are logged as warnings, not raised).

    Parameters
    ----------
    params_path
        Path to a params YAML file (see :func:`load_params`). Defaults to
        ``params.yaml`` in the current working directory, mirroring the CLI.

    Returns
    -------
    None
        All artefacts (datasets, sweep trial tables, ``best_hyperparameters.json``,
        the trained model, figures, SHAP outputs, and logs) are written under
        ``RES_DIR``/``FIG_DIR``/``LOGGING_DIR`` as configured by the params file;
        nothing is returned to the caller.

    Notes
    -----
    Equivalent to running the individual stages in order:
    :func:`run_data`, :func:`run_sweep`, :func:`run_train`, :func:`run_evaluate`,
    and :func:`run_shap` — except the sweep/train results are passed directly
    in-memory to evaluation/SHAP rather than being reloaded from disk.
    """
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

    # Class rebalancing (mirrors the notebook's RandomUnderSampler step) is applied
    # only to the data the model fits/tunes on — SHAP/evaluation still use the
    # original `train_data`/`test_data`, matching how the manual notebook explains
    # on `X_train` rather than the resampled `X_train_sampled`.
    fit_train_data = _balanced_train_data(params, task, train_data, target_cols)

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
        train_data=fit_train_data,
        eval_data=eval_data,
        test_data=test_data,
        n_sweeps=params.N_SWEEPS,
        sweep_fraction=params.SWEEP_FRACTION,
        RES_DIR=params.RES_DIR,
        LOGGING=params.LOGGING,
        LOGGING_DIR=params.LOGGING_DIR,
        wandb_cfg=wandb_cfg,
        TASK=task,
        catboost_search_space=params.CATBOOST_SEARCH_SPACE,
        early_stopping_rounds=params.EARLY_STOPPING_ROUNDS,
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
        fit_train_data,
        eval_data,
        RES_DIR=params.RES_DIR,
        LOGGING_DIR=params.LOGGING_DIR,
        LOGGING=params.LOGGING,
        wandb_cfg=wandb_cfg,
        TASK=task,
        early_stopping_rounds=params.EARLY_STOPPING_ROUNDS,
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


def run_data(params_path: ParamsPath = None):
    """Build (or load cached) train/eval/test splits and report their shapes.

    Mirrors the ``tabnado-data`` CLI entry point. Useful for validating a
    dataset configuration — e.g. checking that the requested ``target`` has
    matching samples, that enough feature assays survive filtering, and that
    the chromosome-based eval/test split produces sensible region counts —
    before committing to a full sweep/train run.

    Parameters
    ----------
    params_path
        Path to a params YAML file (see :func:`load_params`). Defaults to
        ``params.yaml``.

    Returns
    -------
    tuple
        The 7-tuple returned by :func:`tabnado.data.load_data`:
        ``(ds, samples, target_cols, feature_cols, train_data, eval_data, test_data)``
        where ``ds`` is the underlying QuantNado dataset (``None`` for parquet
        datasets), ``samples``/``target_cols``/``feature_cols`` are the
        resolved sample/column names, and ``train_data``/``eval_data``/``test_data``
        are the scaled, chromosome-split feature/target frames (cached as
        parquet under ``DATA_DIR`` for reuse by later stages).

    Notes
    -----
    Logs the resulting split shapes and feature/target counts at INFO level.
    Subsequent calls reuse the cached parquet splits in ``DATA_DIR`` rather
    than rebuilding from the raw dataset/GTF.
    """
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
    """Run an Optuna hyperparameter sweep for the configured backend.

    Mirrors the ``tabnado-sweep`` CLI entry point. Loads (or builds) the
    train/eval/test splits, then dispatches to the backend-specific sweep
    implementation in :mod:`tabnado.sweep` — all three backends
    (``catboost``, ``xgboost``, ``gandalf``) search with an
    :class:`optuna.samplers.TPESampler`-driven study of ``n_sweeps`` trials
    over a fraction (``sweep_fraction``) of the training data, scored on
    ``eval_data``/test folds — weighted F1 for CatBoost classification, macro
    F1 for XGBoost/GANDALF classification, and R2 for regression tasks.
    ``logging: wandb`` does **not** drive the search — it only logs the best
    trial (and, for CatBoost/GANDALF, persists the per-trial table) for
    record-keeping. CatBoost additionally honours ``catboost_search_space``
    (``"extended"`` — the default 8-parameter space — or ``"notebook"``, the
    narrower 4-parameter space used in the original analysis notebook).

    Parameters
    ----------
    params_path
        Path to a params YAML file (see :func:`load_params`). Defaults to
        ``params.yaml``.

    Returns
    -------
    dict
        The best hyperparameters found, keyed by backend-native parameter
        names (e.g. ``depth``/``boosting_type`` for CatBoost,
        ``max_depth``/``learning_rate`` for XGBoost,
        ``learning_rate``/``gflu_stages`` for GANDALF). The same dict is
        persisted to ``RES_DIR/best_hyperparameters.json`` for
        :func:`run_train` to consume; a ``<backend>_optuna_trials.csv`` trial
        table is also written to ``RES_DIR``.

    Notes
    -----
    If a sweep cannot run safely (too few tuning rows, fewer than two classes
    in the tuning sample, NaN scores, or ``n_sweeps <= 0``), each backend
    falls back to deterministic default hyperparameters and logs a warning
    rather than raising.
    """
    params = load_params(params_path)
    _setup_api_stage(params, f"{params.MODEL_TYPE.upper()} SWEEP START")
    run_start = perf_counter()
    wandb_cfg = _make_wandb_config(params)

    _, _, target_cols, feature_cols, train_data, eval_data, test_data = load_data(
        **vars(params)
    )
    task = resolve_task(params.TASK, train_data, target_cols)
    fit_train_data = _balanced_train_data(params, task, train_data, target_cols)

    from tabnado.sweep import sweep_model

    best_hp = sweep_model(
        params.MODEL_TYPE,
        feature_cols=feature_cols,
        target_cols=target_cols,
        train_data=fit_train_data,
        eval_data=eval_data,
        test_data=test_data,
        n_sweeps=params.N_SWEEPS,
        sweep_fraction=params.SWEEP_FRACTION,
        RES_DIR=params.RES_DIR,
        LOGGING=params.LOGGING,
        LOGGING_DIR=params.LOGGING_DIR,
        wandb_cfg=wandb_cfg,
        TASK=task,
        catboost_search_space=params.CATBOOST_SEARCH_SPACE,
        early_stopping_rounds=params.EARLY_STOPPING_ROUNDS,
    )

    logger.info(
        "========== SWEEP END ({:.2f}s total) ==========".format(
            perf_counter() - run_start
        )
    )
    return best_hp


def run_train(params_path: ParamsPath = None):
    """Train the final model with the swept hyperparameters and persist it.

    Mirrors the ``tabnado-train`` CLI entry point. Loads
    ``RES_DIR/best_hyperparameters.json`` (raising if :func:`run_sweep` has
    not been run yet), loads the train/eval splits, and fits a single final
    model of the configured backend (``catboost``, ``xgboost``, or
    ``gandalf``) on the full training set with early stopping against
    ``eval_data``.

    Parameters
    ----------
    params_path
        Path to a params YAML file (see :func:`load_params`). Defaults to
        ``params.yaml``.

    Returns
    -------
    object
        The trained model, in the backend's native form:

        - ``catboost`` -> a fitted ``catboost.CatBoostClassifier``/``CatBoostRegressor``
        - ``xgboost`` -> a fitted ``xgboost.XGBClassifier``/``XGBRegressor``
          (or ``MultiOutputRegressor`` wrapper for multi-target regression)
        - ``gandalf`` -> a fitted ``pytorch_tabular.TabularModel``

        The model is also saved to ``RES_DIR/final_model/`` (as
        ``catboost_model.joblib``, ``xgboost_model.joblib``, or a
        PyTorch-Tabular model directory respectively) for later stages
        (:func:`run_evaluate`, :func:`run_shap`) to reload independently.

    Raises
    ------
    FileNotFoundError
        If no ``best_hyperparameters.json`` exists under ``RES_DIR`` —
        run :func:`run_sweep` (or ``tabnado-sweep``) first.
    """
    params = load_params(params_path)
    _setup_api_stage(params, f"{params.MODEL_TYPE.upper()} TRAIN START")
    run_start = perf_counter()
    wandb_cfg = _make_wandb_config(params)
    best_hp = _load_best_hyperparameters(params)

    _, _, target_cols, feature_cols, train_data, eval_data, _ = load_data(
        **vars(params)
    )
    task = resolve_task(params.TASK, train_data, target_cols)
    fit_train_data = _balanced_train_data(params, task, train_data, target_cols)

    from tabnado.train import train_model

    final_model = train_model(
        params.MODEL_TYPE,
        best_hp,
        feature_cols,
        target_cols,
        fit_train_data,
        eval_data,
        RES_DIR=params.RES_DIR,
        LOGGING_DIR=params.LOGGING_DIR,
        LOGGING=params.LOGGING,
        wandb_cfg=wandb_cfg,
        TASK=task,
        early_stopping_rounds=params.EARLY_STOPPING_ROUNDS,
    )

    logger.info(
        "========== TRAIN END ({:.2f}s total) ==========".format(
            perf_counter() - run_start
        )
    )
    return final_model


def run_evaluate(params_path: ParamsPath = None) -> None:
    """Reload the trained model and compute test-set metrics, plots, and a UMAP.

    Mirrors the ``tabnado-evaluate`` CLI entry point. Loads the held-out test
    split and the model previously saved by :func:`run_train` from
    ``RES_DIR/final_model/`` (raising if it doesn't exist), then runs
    :func:`evaluate_model` (test metrics + diagnostic plots — ROC/confusion
    matrix for classification, scatter/residuals + R2/MSE for regression) and
    :func:`compute_umap_embeddings` (a 2D UMAP projection of the test feature
    space) against it.

    Parameters
    ----------
    params_path
        Path to a params YAML file (see :func:`load_params`). Defaults to
        ``params.yaml``.

    Returns
    -------
    None
        Metrics are logged and saved as JSON under ``RES_DIR``; plots
        (ROC curves, confusion matrix, UMAP scatter, etc.) are written under
        ``FIG_DIR``. Nothing is returned to the caller.

    Raises
    ------
    FileNotFoundError
        If no trained model exists at ``RES_DIR/final_model`` —
        run :func:`run_train` (or ``tabnado-train``) first.
    RuntimeError
        If a saved GANDALF model loads with no weights (corrupt/incomplete
        checkpoint).

    Notes
    -----
    Unlike :func:`run_pipeline`, this stage does not open a fresh wandb run —
    metrics/plots are logged via the configured logging backend's default
    mechanism only.
    """
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
    """Reload the trained model and compute SHAP feature-importance analysis.

    Mirrors the ``tabnado-shap`` CLI entry point. Loads the train/eval/test
    splits and the model previously saved by :func:`run_train` (raising if it
    doesn't exist), then dispatches to the backend-specific SHAP computation
    in :mod:`tabnado.shap` (e.g. ``shap.TreeExplainer`` for CatBoost/XGBoost).

    Parameters
    ----------
    params_path
        Path to a params YAML file (see :func:`load_params`). Defaults to
        ``params.yaml``.

    Returns
    -------
    None
        SHAP summary/bar plots (overall, and per-class for classification
        tasks) are written under ``FIG_DIR``; any computed SHAP value arrays
        or tables are saved under ``RES_DIR``. Nothing is returned to the
        caller.

    Raises
    ------
    FileNotFoundError
        If no trained model exists at ``RES_DIR/final_model`` —
        run :func:`run_train` (or ``tabnado-train``) first.

    Notes
    -----
    Uses ``train_data`` (and ``eval_data`` where relevant) as the background/
    explanation set and ``test_data`` for held-out attributions — mirroring
    the data the model was trained and evaluated on. Unlike
    :func:`run_pipeline`, this stage does not open a fresh wandb run.
    """
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


def write_params_template(
    path: Path | str = "params.yaml", force: bool = False
) -> Path:
    """Write a starter params YAML template to ``path``.

    Mirrors the ``tabnado-init`` CLI entry point. Writes
    :data:`tabnado.cli.PARAMS_TEMPLATE` — a commented YAML skeleton covering
    every key understood by :meth:`PipelineParams.from_yaml` (``target``,
    ``model_name``, ``task``, ``sweep_fraction``, ``gtf_file``, ``eval_chr``/
    ``test_chr``, ``output_dir``, ``dataset``, ``windows_bed``, ``n_sweeps``,
    ``catboost_search_space``, ``logging``, ``min_target``, ``min_features``,
    ``exclude_ips``, ``prefixes``, ``window_size``/``step_size``/``tile_size``)
    — to give a new experiment a sensible, fully-annotated starting point that
    the user edits in place.

    Parameters
    ----------
    path
        Output path for the params file. Parent directories are created if
        missing. Defaults to ``params.yaml`` in the current working directory.
    force
        If ``True``, overwrite an existing file at ``path``. If ``False``
        (the default) and the file already exists, raise instead of
        clobbering an in-progress configuration.

    Returns
    -------
    pathlib.Path
        The path the template was written to (i.e. ``Path(path)``).

    Raises
    ------
    FileExistsError
        If ``path`` already exists and ``force`` is ``False``.
    """
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
