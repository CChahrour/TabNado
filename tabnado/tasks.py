"""Task helpers shared across TabNado backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import LabelEncoder

VALID_TASKS = {"auto", "regression", "classification"}


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
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro")),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted")),
    }
    if probabilities is not None and classes is not None:
        try:
            metrics["log_loss"] = float(
                log_loss(y_true_arr, probabilities, labels=list(classes))
            )
        except ValueError:
            pass
    return metrics


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
