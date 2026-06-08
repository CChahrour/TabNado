"""CLI entry points for tabnado commands."""

PARAMS_TEMPLATE = """# tabnado parameters
# Required keys
target: TARGET_NAME
model_name: GANDALF
task: auto
sweep_fraction: 0.2
gtf_file: data/gencode.vM25.annotation.gtf.gz
eval_chr: chr8  # blank, single name, or list (e.g. [\"chr8\", \"chr10\"]) — leave blank for no eval split
test_chr: [\"chr9\"]  # single name or list of chromosome names
output_dir: results
dataset: data/dataset
windows_bed: data/tss_windows.bed
n_sweeps: 10
catboost_search_space: extended  # or "notebook" for the narrower 4-param search
class_balance: none  # "none", "undersample", "oversample", or "smote" (classification only)
early_stopping: 10  # early_stopping_rounds passed to sweep AND final-model training (CatBoost/XGBoost)
logging: wandb
min_target: 1
min_features: 10
exclude_ips: [\"AF4C\", \"MLLC\"]
prefixes: [\"CAT\", \"ChIP\", \"CM\"]
window_size: 3000
step_size: 100
tile_size: 100
"""


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
    import argparse

    from tabnado.api import write_params_template

    parser = argparse.ArgumentParser(description="Create a tabnado params template")
    parser.add_argument(
        "path",
        nargs="?",
        default="params.yaml",
        help="Output path for params template (default: params.yaml)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite an existing file",
    )
    args = parser.parse_args()

    try:
        output_path = write_params_template(args.path, force=args.force)
    except FileExistsError as exc:
        parser.error(str(exc))

    print(f"Wrote template params file to {output_path}")


main = run
