"""CLI entry points for tabnado commands."""


def run() -> None:
    """Run the full pipeline."""
    from tabnado.api import run as api_run
    from tabnado.utils import parse_params_arg

    api_run(parse_params_arg())


def data() -> None:
    """Run data loading/build stage."""
    from tabnado.api import data as api_data
    from tabnado.utils import parse_params_arg

    api_data(parse_params_arg())


def sweep() -> None:
    """Run hyperparameter sweep stage."""
    from tabnado.api import sweep as api_sweep
    from tabnado.utils import parse_params_arg

    api_sweep(parse_params_arg())


def train() -> None:
    """Run final model training stage."""
    from tabnado.api import train as api_train
    from tabnado.utils import parse_params_arg

    api_train(parse_params_arg())


def evaluate() -> None:
    """Run evaluation and UMAP stage."""
    from tabnado.api import evaluate as api_evaluate
    from tabnado.utils import parse_params_arg

    api_evaluate(parse_params_arg())


def shap() -> None:
    """Run SHAP analysis stage."""
    from tabnado.api import shap as api_shap
    from tabnado.utils import parse_params_arg

    api_shap(parse_params_arg())


def init() -> None:
    """Create a template params YAML file for a new run."""
    from tabnado.init import main as init_main

    init_main()


main = run
