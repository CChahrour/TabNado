import logging
import random
import sys
import time
from importlib import metadata
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from pytorch_lightning import Callback
from loguru import logger


LOAD_DATA_PARAMS = (
    "DATASET",
    "TARGET",
    "GTF_FILE",
    "WINDOWS_BED",
    "EVAL_CHR",
    "TEST_CHR",
    "FIG_DIR",
    "RES_DIR",
    "MIN_TARGET",
    "MIN_FEATURES",
    "EXCLUDE_IPS",
    "ASSAY_PREFIXES",
    "WINDOW_SIZE",
    "STEP_SIZE",
    "TILE_SIZE",
    "CHUNK_SIZE_ROWS",
)


class LoguruProgressCallback(Callback):
    """Log per-epoch train/val metrics via loguru so they appear in the log file."""

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        metrics = {k: v for k, v in trainer.callback_metrics.items()}
        if not metrics:
            return
        epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs
        parts = [f"epoch={epoch}/{max_epochs}"]
        for k, v in sorted(metrics.items()):
            try:
                parts.append(f"{k}={float(v):.4f}")
            except (TypeError, ValueError):
                pass
        logger.info("  ".join(parts))


def _package_version() -> str:
    try:
        return metadata.version("tabnado")
    except metadata.PackageNotFoundError:
        return "0.0.0"


def parse_params_arg() -> Path:
    """Parse CLI args for stage commands.

    Supports:
    - `--params` / `-p` to provide params YAML path
    - `--help` for usage
    - `--version` to print installed package version
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        "-p",
        type=Path,
        default=None,
        help="Path to params YAML file (defaults to params.yaml in project root)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )
    args, _ = parser.parse_known_args()
    return args.params


def load_params(params_path: Path | str | None = None) -> dict:
    """Load params.yaml and return a dict with raw + derived values."""
    if params_path is None:
        params_path = Path(__file__).parent.parent / "params.yaml"
    with open(params_path) as f:
        p = yaml.safe_load(f)

    required = [
        "target",
        "model_name",
        "sweep_fraction",
        "gtf_file",
        "eval_chr",
        "test_chr",
        "output_dir",
        "dataset",
        "n_sweeps",
        "logging",
        "min_target",
        "min_features",
        "exclude_ips",
        "prefixes",
        "window_size",
        "step_size",
        "tile_size",
    ]
    missing = [k for k in required if k not in p]
    for k in missing:
        logger.warning(f"Missing required parameter: '{k}'")
    if missing:
        raise KeyError(f"Missing required params: {missing}")

    logging_backend = str(p.get("logging", "wandb")).lower()
    if logging_backend not in {"wandb", "tensorboard"}:
        raise ValueError(
            f"Invalid logging backend '{logging_backend}'. Use 'wandb' or 'tensorboard'."
        )

    model_type = str(p.get("model_name", "gandalf")).lower()
    if model_type not in {"gandalf", "xgboost"}:
        raise ValueError(
            f"Invalid model_type '{model_type}'. Use 'gandalf' or 'xgboost'."
        )

    date = time.strftime("%Y-%m-%d")
    project = f"{p['model_name']}_{p['target']}_{date}"
    res_dir = f"{p['output_dir']}/{project}"
    fig_dir = f"{res_dir}/figures"
    logging_dir = f"{res_dir}/logging"

    data_dir = f"{res_dir}/dataset"
    for d in [fig_dir, res_dir, logging_dir, data_dir]:
        os.makedirs(d, exist_ok=True)

    return {
        "DATASET": p["dataset"],
        "TARGET": p["target"],
        "MODEL_NAME": p["model_name"],
        "SWEEP_FRACTION": p["sweep_fraction"],
        "N_SWEEPS": p["n_sweeps"],
        "LOGGING": logging_backend,
        "GTF_FILE": p["gtf_file"],
        "WINDOWS_BED": Path(p["windows_bed"])
        if "windows_bed" in p
        else Path(data_dir) / "regions.bed",
        "EVAL_CHR": p["eval_chr"],
        "TEST_CHR": p["test_chr"],
        "DATA_DIR": data_dir,
        "MIN_TARGET": p["min_target"],
        "MIN_FEATURES": p["min_features"],
        "EXCLUDE_IPS": p["exclude_ips"],
        "ASSAY_PREFIXES": p["prefixes"],
        "WINDOW_SIZE": p["window_size"],
        "STEP_SIZE": p["step_size"],
        "TILE_SIZE": p["tile_size"],
        "CHUNK_SIZE_ROWS": int(p["chunk_size_rows"])
        if "chunk_size_rows" in p and p["chunk_size_rows"] is not None
        else None,
        "date": date,
        "PROJECT": project,
        "RES_DIR": res_dir,
        "FIG_DIR": fig_dir,
        "LOGGING_DIR": logging_dir,
        "MODEL_TYPE": model_type,
    }


class _InterceptHandler(logging.Handler):
    """Route stdlib logging records into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno  # type: ignore[assignment]
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logger(res_dir: str, project: str) -> None:
    log_path = f"{res_dir}/{project}.log"
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_path, rotation="1 MB", level="INFO")
    logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO, force=True)
    logging.getLogger("shap").setLevel(logging.WARNING)
    logger.info(f"Project: {project}  Results: {res_dir}")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def figure_style():
    size = 16
    smaller = 10
    rc = {
        "figure.titlesize": size,
        "figure.titleweight": "bold",
        "axes.titlesize": size,
        "axes.titleweight": "bold",
        "axes.labelsize": size,
        "axes.labelweight": "bold",
        "xtick.labelsize": smaller,
        "ytick.labelsize": smaller,
        "legend.fontsize": smaller,
        "legend.title_fontsize": smaller,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "grid.color": "gray",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }
    plt.rcParams.update(rc)
    sns.set_theme(style="white", rc=rc)
