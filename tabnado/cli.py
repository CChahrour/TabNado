"""CLI entry points for tabnado commands."""

from tabnado.api import PipelineParams


def run() -> None:
    """Run the full pipeline."""
    from tabnado.api import run_pipeline

    from tabnado.utils import parse_params_arg

    run_pipeline(parse_params_arg())


def data() -> None:
    """Run data loading/build stage."""
    from tabnado.data import main as data_main

    data_main()


def sweep() -> None:
    """Run hyperparameter sweep stage."""
    from tabnado.utils import parse_params_arg

    params = PipelineParams.from_yaml(parse_params_arg())
    model_type = params.MODEL_TYPE

    if model_type == "xgboost":
        from tabnado.xgb_sweep import main as sweep_main
    else:
        from tabnado.gandalf_sweep import main as sweep_main

    sweep_main()


def train() -> None:
    """Run final model training stage."""
    from tabnado.utils import parse_params_arg

    params = PipelineParams.from_yaml(parse_params_arg())
    model_type = params.MODEL_TYPE

    if model_type == "xgboost":
        from tabnado.xgb_train import main as train_main
    else:
        from tabnado.gandalf_train import main as train_main

    train_main()


def evaluate() -> None:
    """Run evaluation and UMAP stage."""
    from tabnado.evaluate import main as evaluate_main

    evaluate_main()


def shap() -> None:
    """Run SHAP analysis stage."""
    from tabnado.utils import parse_params_arg

    params = PipelineParams.from_yaml(parse_params_arg())
    model_type = params.MODEL_TYPE

    if model_type == "xgboost":
        from tabnado.xgb_shap import main as shap_main
    else:
        from tabnado.gandalf_shap import main as shap_main

    shap_main()


def init() -> None:
    """Create a template params YAML file for a new run."""
    from tabnado.init import main as init_main

    init_main()


main = run
