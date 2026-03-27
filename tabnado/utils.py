import logging
import random
import sys
from importlib import metadata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from loguru import logger
from pytorch_lightning import Callback


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


def log_macro(final_model, target_cols):
    """
    Helper function to log valid_r2-macro and train_r2-macro to wandb after GANDALF training.
    Logs both metrics per epoch (step).
    """
    import wandb

    if (
        not hasattr(final_model, "history")
        or "valid_r2_score" not in final_model.history
        or "train_r2_score" not in final_model.history
        or "valid_loss" not in final_model.history
        or "train_loss" not in final_model.history
    ):
        print(
            "No valid_r2_score or train_r2_score found in model history. Skipping r2_macro logging."
        )
        return

    valid_r2 = final_model.history["valid_r2_score"]
    train_r2 = final_model.history["train_r2_score"]
    valid_loss = final_model.history["valid_loss"]
    train_loss = final_model.history["train_loss"]
    if (
        isinstance(valid_r2, list)
        and isinstance(train_r2, list)
        and isinstance(valid_loss, list)
        and isinstance(train_loss, list)
    ):
        for step, (v_r2, t_r2, v_loss, t_loss) in enumerate(
            zip(valid_r2, train_r2, valid_loss, train_loss)
        ):
            log_dict = {
                "valid_r2_macro": v_r2 / len(target_cols),
                "train_r2_macro": t_r2 / len(target_cols),
                "valid_loss_macro": v_loss / len(target_cols),
                "train_loss_macro": t_loss / len(target_cols),
            }
            wandb.log(log_dict, step=step)
