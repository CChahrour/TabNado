import os

# Keep native math runtimes from over-subscribing or colliding across Torch,
# XGBoost, CatBoost, SHAP, and plotting in one process. Users can override
# these by setting the environment before importing tabnado.
for _thread_env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_var, "1")

from .api import (
    PipelineParams,
    __version__,
    compute_umap_embeddings,
    evaluate_model,
    load_data,
    load_params,
    run,
    run_data,
    run_evaluate,
    run_pipeline,
    run_shap,
    run_sweep,
    run_train,
    setup_logger,
    write_params_template,
)

__all__ = [
    "PipelineParams",
    "__version__",
    "compute_umap_embeddings",
    "evaluate_model",
    "load_data",
    "load_params",
    "run",
    "run_data",
    "run_evaluate",
    "run_pipeline",
    "run_shap",
    "run_sweep",
    "run_train",
    "setup_logger",
    "write_params_template",
]
