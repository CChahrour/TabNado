"""Initialize template configuration files for tabnado."""

from pathlib import Path


PARAMS_TEMPLATE = """# tabnado parameters
# Required keys
target: MLLN
model_name: GANDALF
sweep_fraction: 0.2
gtf_file: data/gencode.vM25.annotation.gtf.gz
eval_chr: chr8
test_chr: chr9
output_dir: results

# Optional keys
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

    parser = argparse.ArgumentParser(
        description="Create a tabnado params template"
    )
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

    output_path = Path(args.path)
    if output_path.exists() and not args.force:
        parser.error(f"{output_path} already exists. Re-run with --force to overwrite.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(PARAMS_TEMPLATE, encoding="utf-8")
    print(f"Wrote template params file to {output_path}")


if __name__ == "__main__":
    main()
