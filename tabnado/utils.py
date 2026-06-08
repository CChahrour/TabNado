import logging
import random
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import LabelEncoder

VALID_TASKS = {"auto", "regression", "classification"}

_LoguruProgressCallback = None


class LoguruProgressCallback:
    """Create a Lightning callback without importing Torch on tree backends."""

    def __new__(cls):
        global _LoguruProgressCallback
        if _LoguruProgressCallback is not None:
            return _LoguruProgressCallback()

        from pytorch_lightning import Callback

        class _RegisteredLoguruProgressCallback(Callback):
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

        _RegisteredLoguruProgressCallback.__name__ = "_LoguruProgressCallback"
        _RegisteredLoguruProgressCallback.__qualname__ = "_LoguruProgressCallback"
        _RegisteredLoguruProgressCallback.__module__ = __name__
        _LoguruProgressCallback = _RegisteredLoguruProgressCallback
        globals()["_LoguruProgressCallback"] = _RegisteredLoguruProgressCallback
        return _LoguruProgressCallback()


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
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def figure_style():
    import matplotlib.pyplot as plt
    import seaborn as sns

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
        "axes.grid": False,
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


@dataclass
class EncodedTarget:
    target_col: str
    classes: list[str]
    problem_type: str
    train: np.ndarray
    eval: np.ndarray | None = None


def validate_task(task: str) -> str:
    normalized = str(task or "auto").lower()
    if normalized not in VALID_TASKS:
        raise ValueError(f"Invalid task '{task}'. Use one of {VALID_TASKS}.")
    return normalized


def resolve_task(
    task: str,
    train_data: pd.DataFrame,
    target_cols: list[str],
) -> str:
    """Resolve ``auto`` into regression or classification."""
    normalized = validate_task(task)
    if normalized != "auto":
        return normalized

    if len(target_cols) == 1:
        target = train_data[target_cols[0]]
        if (
            not pd.api.types.is_numeric_dtype(target)
            or pd.api.types.is_bool_dtype(target)
            or isinstance(target.dtype, pd.CategoricalDtype)
        ):
            return "classification"

    return "regression"


def require_single_classification_target(target_cols: list[str]) -> str:
    if len(target_cols) != 1:
        raise ValueError(
            "Classification currently expects exactly one target column; "
            f"received {len(target_cols)} targets."
        )
    return target_cols[0]


def encode_classification_target(
    train_data: pd.DataFrame,
    target_cols: list[str],
    eval_data: pd.DataFrame | None = None,
) -> EncodedTarget:
    target_col = require_single_classification_target(target_cols)
    encoder = LabelEncoder()

    fit_values = train_data[target_col].astype(str)
    if eval_data is not None:
        fit_values = pd.concat([fit_values, eval_data[target_col].astype(str)], axis=0)
    encoder.fit(fit_values)

    classes = [str(cls) for cls in encoder.classes_]
    if len(classes) < 2:
        raise ValueError(
            f"Classification target '{target_col}' needs at least two classes; "
            f"found {classes}."
        )

    return EncodedTarget(
        target_col=target_col,
        classes=classes,
        problem_type="binary" if len(classes) == 2 else "multiclass",
        train=encoder.transform(train_data[target_col].astype(str)),
        eval=(
            encoder.transform(eval_data[target_col].astype(str))
            if eval_data is not None
            else None
        ),
    )


def probability_columns(target_col: str, classes: list[str]) -> list[str]:
    return [f"{target_col}_{class_name}_probability" for class_name in classes]


def classification_shap_output_columns(
    target_col: str,
    classes: list[str],
    n_outputs: int,
) -> list[str]:
    """Return class-specific names for SHAP outputs."""
    if n_outputs < 1:
        raise ValueError("SHAP output count must be at least one.")

    class_names = [str(cls) for cls in classes]
    if n_outputs == len(class_names):
        return [f"{target_col}_{class_name}" for class_name in class_names]

    if n_outputs == 1 and len(class_names) == 2:
        return [f"{target_col}_{class_names[1]}"]

    if n_outputs == 1:
        return [target_col]

    return [f"{target_col}_output_{idx}" for idx in range(n_outputs)]


def classification_prediction_frame(
    pred_labels: np.ndarray,
    probabilities: np.ndarray | None,
    target_col: str,
    classes: list[str],
    index: pd.Index,
) -> pd.DataFrame:
    pred_df = pd.DataFrame({target_col: pred_labels.astype(str)}, index=index)
    if probabilities is not None:
        proba = np.asarray(probabilities)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        pred_df = pd.concat(
            [
                pred_df,
                pd.DataFrame(
                    proba,
                    columns=probability_columns(target_col, classes),
                    index=index,
                ),
            ],
            axis=1,
        )
    return pred_df


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    probabilities: np.ndarray | None = None,
    classes: list[str] | None = None,
) -> dict[str, Any]:
    y_true_arr = np.asarray(pd.Series(y_true).astype(str))
    y_pred_arr = np.asarray(pd.Series(y_pred).astype(str))
    labels = (
        [str(cls) for cls in classes]
        if classes
        else sorted(set(y_true_arr) | set(y_pred_arr))
    )
    per_class_f1 = f1_score(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average=None,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro")),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted")),
        "per_class_f1": {
            label: float(score) for label, score in zip(labels, per_class_f1)
        },
    }
    if probabilities is not None and classes is not None:
        try:
            metrics["log_loss"] = float(
                log_loss(y_true_arr, probabilities, labels=list(classes))
            )
        except ValueError:
            pass
    return metrics


def flatten_metric_dict(
    metrics: dict[str, Any],
    prefix: str = "",
    sep: str = "/",
) -> dict[str, Any]:
    """Flatten nested metric dictionaries for scalar loggers."""
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_metric_dict(value, prefix=f"{full_key}{sep}", sep=sep))
        else:
            flat[full_key] = value
    return flat


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value
