"""Initialize template configuration files for tabnado."""

from pathlib import Path

PARAMS_TEMPLATE = """# tabnado parameters
# Required keys
target: TARGET_NAME
model_name: GANDALF
task: auto
sweep_fraction: 0.2
gtf_file: data/gencode.vM25.annotation.gtf.gz
eval_chr: chr8
test_chr: chr9
output_dir: results
dataset: data/dataset
windows_bed: data/tss_windows.bed
n_sweeps: 10
logging: wandb
min_target: 1
min_features: 10
exclude_ips: [\"AF4C\", \"MLLC\"]
prefixes: [\"CAT\", \"ChIP\", \"CM\"]
window_size: 3000
step_size: 100
tile_size: 100
"""


def main() -> None:
    """Create a template params YAML file for a new run."""
    import argparse

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

    from tabnado.api import write_params_template

    try:
        output_path = write_params_template(args.path, force=args.force)
    except FileExistsError as exc:
        parser.error(str(exc))

    print(f"Wrote template params file to {output_path}")


if __name__ == "__main__":
    main()
