from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger


def shap_values_to_output_list(shap_values: Any) -> list[np.ndarray]:
    """Normalize SHAP return values to one 2D array per model output."""
    if isinstance(shap_values, list):
        outputs: list[np.ndarray] = []
        for values in shap_values:
            outputs.extend(shap_values_to_output_list(values))
        return outputs

    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    arr = np.asarray(values)

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return [arr[:, :, 0]]
        return [arr[:, :, idx] for idx in range(arr.shape[-1])]

    if arr.ndim == 2:
        return [arr]

    if arr.ndim == 1:
        return [arr.reshape(-1, 1)]

    raise ValueError(f"Unsupported SHAP values shape: {arr.shape}")


def default_shap_output_columns(target_cols: list[str], n_outputs: int) -> list[str]:
    """Return stable output column names for non-classification SHAP values."""
    if n_outputs < 1:
        raise ValueError("SHAP output count must be at least one.")

    if len(target_cols) == n_outputs:
        return target_cols

    if len(target_cols) == 1:
        target_col = target_cols[0]
        if n_outputs == 1:
            return [target_col]
        return [f"{target_col}_output_{idx}" for idx in range(n_outputs)]

    return [f"output_{idx}" for idx in range(n_outputs)]


def strip_spatial_offset(col: str) -> str:
    parts = col.rsplit("_", 1)
    if len(parts) != 2:
        return col
    feature_name, offset = parts
    try:
        int(offset)
    except ValueError:
        return col
    return feature_name


def cofactor_shap_table(mean_abs_shap: pd.DataFrame) -> pd.DataFrame:
    cofactor_data = mean_abs_shap.copy()
    cofactor_data.index = [strip_spatial_offset(c) for c in cofactor_data.index]
    return cofactor_data.groupby(level=0).mean()


def plot_shap_stacked_bar(
    mean_abs_shap: pd.DataFrame,
    FIG_DIR: str,
    SHAP_DIR: str,
    max_features: int = 30,
    wandb_run=None,
) -> str:
    """Plot cofactor-level mean |SHAP| as stacked bars by output."""
    cofactor_data = cofactor_shap_table(mean_abs_shap)
    cofactor_data = cofactor_data.loc[
        cofactor_data.sum(axis=1).sort_values(ascending=False).index
    ]

    if len(cofactor_data) > max_features:
        top = cofactor_data.iloc[:max_features].copy()
        other = cofactor_data.iloc[max_features:].sum(axis=0)
        cofactor_data = pd.concat(
            [top, pd.DataFrame([other], index=["Other"])],
            axis=0,
        )

    csv_path = f"{SHAP_DIR}/shap_stacked_bar_data.csv"
    cofactor_data.to_csv(csv_path)
    logger.info(f"Saved SHAP stacked bar table: {csv_path}")

    n_rows, n_outputs = cofactor_data.shape
    fig_height = max(5.5, min(18, n_rows * 0.35 + 2))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    left = np.zeros(n_rows, dtype=float)
    colors = sns.color_palette("tab20", n_colors=max(n_outputs, 1))
    y = np.arange(n_rows)

    for idx, output_col in enumerate(cofactor_data.columns):
        values = cofactor_data[output_col].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            label=str(output_col),
            color=colors[idx % len(colors)],
            linewidth=0,
        )
        left += values

    ax.set_yticks(y, labels=cofactor_data.index)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("Feature")
    ax.set_title("Stacked SHAP importance by cofactor")
    ax.legend(
        title="Output",
        fontsize=8,
        title_fontsize=9,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
    )
    ax.margins(x=0.01)
    fig.tight_layout()
    path = f"{FIG_DIR}/shap_stacked_bar.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info(f"Saved SHAP stacked bar plot: {path}")

    if wandb_run is not None:
        import wandb

        wandb_run.log({"shap/stacked_bar": wandb.Image(path)})

    return path
