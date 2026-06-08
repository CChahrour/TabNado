import os
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap as shap_pkg
from loguru import logger
from matplotlib import cm, colors

from tabnado.params import PipelineParams
from tabnado.utils import (
    classification_shap_output_columns,
    figure_style,
    parse_params_arg,
    resolve_task,
    setup_logger,
)

# ---------------------------------------------------------------------------
# Shared SHAP value/output helpers (formerly shap_utils.py)
# ---------------------------------------------------------------------------


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
    colors_palette = sns.color_palette("tab20", n_colors=max(n_outputs, 1))
    y = np.arange(n_rows)

    for idx, output_col in enumerate(cofactor_data.columns):
        values = cofactor_data[output_col].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            label=str(output_col),
            color=colors_palette[idx % len(colors_palette)],
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


def _parse_offset_from_column(col: str) -> tuple[str, int] | None:
    """
    Parse sample and offset from reshaped column name like 'ChIP-CELL_MLLN_-1500'.
    Returns (sample_name, offset_bp) or None if not a spatial column.
    """
    parts = col.rsplit("_", 1)
    if len(parts) != 2:
        return None
    sample_name, offset_str = parts
    try:
        offset_bp = int(offset_str)
        return sample_name, offset_bp
    except ValueError:
        return None


def _plot_spatial_shap(
    shap_values_list: list[np.ndarray],
    feature_cols: list[str],
    target_cols: list[str],
    test_idx,
    SHAP_DIR: str = "results/shap",
    FIG_DIR: str = "figures",
    tile_size: int = 100,
    wandb_run=None,
):
    """
    Create spatial SHAP visualizations aggregating importance by TSS offset.
    Produces:
      - spatial_shap_by_offset.csv: Mean |SHAP| per cofactor per offset
      - shap_spatial_heatmap_*.png: Heatmap of SHAP importance across offsets
      - shap_offset_line_*.png: Line plots showing offset importance profile
    """
    half_tile = tile_size // 2
    # Parse feature columns to extract cofactor and offset
    offset_to_col = {}  # offset_bp -> list of column indices
    col_to_parsed = {}
    cofactor_names = set()

    for col_idx, col in enumerate(feature_cols):
        parsed = _parse_offset_from_column(col)
        if parsed:
            sample, offset_bp = parsed
            col_to_parsed[col_idx] = (sample, offset_bp)
            cofactor_names.add(sample)
            offset_to_col.setdefault(offset_bp, []).append(col_idx)

    if not col_to_parsed:
        logger.warning("No spatial columns found; skipping spatial SHAP analysis")
        return

    cofactor_names = sorted(cofactor_names)
    cofactor_idx = {name: i for i, name in enumerate(cofactor_names)}
    offsets_bp = sorted(offset_to_col.keys())
    centre_bp = [o + half_tile for o in offsets_bp]
    logger.info(
        f"Spatial SHAP: {len(cofactor_names)} cofactors × {len(offsets_bp)} offsets = {len(col_to_parsed)} spatial features"
    )

    # Pre-compute (cofactor_row, offset_idx) for each column index
    col_to_matrix_pos: dict[int, tuple[int, int]] = {}
    for offset_idx, offset_bp in enumerate(offsets_bp):
        for col_idx in offset_to_col[offset_bp]:
            if col_idx in col_to_parsed:
                cofactor, _ = col_to_parsed[col_idx]
                col_to_matrix_pos[col_idx] = (cofactor_idx[cofactor], offset_idx)

    # For each target, aggregate SHAP by cofactor and offset
    for target_idx, target_col in enumerate(target_cols):
        shap_target = shap_values_list[target_idx]  # (n_samples, n_features)

        # Build spatial aggregation: cofactor × offset -> mean |SHAP|
        spatial_shap = np.zeros((len(cofactor_names), len(offsets_bp)))
        counts = np.zeros_like(spatial_shap)
        for col_idx, (cofactor_row, offset_idx) in col_to_matrix_pos.items():
            spatial_shap[cofactor_row, offset_idx] += np.abs(
                shap_target[:, col_idx]
            ).mean()
            counts[cofactor_row, offset_idx] += 1

        spatial_shap = np.divide(
            spatial_shap, counts, where=counts > 0, out=np.zeros_like(spatial_shap)
        )

        # Create DataFrame for easier manipulation
        spatial_df = pd.DataFrame(spatial_shap, index=cofactor_names, columns=centre_bp)

        # Save spatial SHAP table
        spatial_csv = (
            f"{SHAP_DIR}/spatial_shap_by_offset_{target_col.replace('/', '_')}.csv"
        )
        spatial_df.to_csv(spatial_csv)
        logger.info(f"Saved spatial SHAP: {spatial_csv}")

        # === Heatmap: cofactor × offset ===
        fig, ax = plt.subplots(figsize=(12, max(4, len(cofactor_names) * 0.4 + 2)))
        sns.heatmap(
            spatial_df,
            annot=False,
            cmap="mako",
            ax=ax,
            cbar_kws={"label": "Mean |SHAP|"},
        )
        ax.set_xlabel("Offset from TSS (bp)")
        ax.set_ylabel("Cofactor")
        ax.set_title(f"Spatial SHAP Importance ({target_col})")
        heatmap_path = (
            f"{FIG_DIR}/shap_spatial_heatmap_{target_col.replace('/', '_')}.png"
        )
        fig.savefig(heatmap_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        logger.info(f"Saved spatial heatmap: {heatmap_path}")
        if wandb_run is not None:
            import wandb

            safe_col = target_col.replace("/", "_")
            wandb_run.log(
                {f"shap/spatial_heatmap_{safe_col}": wandb.Image(heatmap_path)}
            )

        # === Line plot: one line per cofactor, coloured by feature ===
        palette = sns.color_palette("tab20", n_colors=len(cofactor_names))
        fig, ax = plt.subplots(figsize=(10, 4))
        for cofactor, color in zip(cofactor_names, palette):
            row = spatial_df.loc[cofactor].to_numpy(dtype=float)
            ax.plot(
                centre_bp, row, linewidth=1.5, alpha=0.8, label=cofactor, color=color
            )
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="TSS center")
        ax.set_xlabel("Offset from TSS (bp)")
        ax.set_ylabel("Mean |SHAP|")
        ax.set_title(f"Genomic Distance Importance Profile ({target_col})")
        ax.grid(True, alpha=0.3)
        ax.legend(
            fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0
        )
        line_path = f"{FIG_DIR}/shap_offset_line_{target_col.replace('/', '_')}.png"
        fig.savefig(line_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        logger.info(f"Saved offset importance profile: {line_path}")
        if wandb_run is not None:
            import wandb

            safe_col = target_col.replace("/", "_")
            wandb_run.log({f"shap/offset_line_{safe_col}": wandb.Image(line_path)})


def _plot_clustermap(
    mean_abs_shap: pd.DataFrame,
    output_cols: list[str],
    FIG_DIR: str,
    wandb_run=None,
) -> None:
    """Aggregate over tile offsets and plot one row per cofactor."""

    def _strip_offset(col: str) -> str:
        parsed = _parse_offset_from_column(col)
        return parsed[0] if parsed else col

    clustermap_data = mean_abs_shap.copy()
    clustermap_data.index = [_strip_offset(c) for c in clustermap_data.index]
    clustermap_data = clustermap_data.groupby(level=0).mean()

    n_rows = len(clustermap_data)
    n_cols = len(output_cols)
    g = sns.clustermap(
        clustermap_data,
        row_cluster=n_rows > 1,
        col_cluster=False,
        cmap="Reds",
        figsize=(max(3, n_cols * 1.5 + 2), max(4, n_rows * 0.4 + 2)),
        yticklabels=True,
        linewidths=0.5,
        cbar_pos=None,
    )
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=9, rotation=0)
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), fontsize=9, rotation=30, ha="right"
    )
    g.figure.suptitle("Mean |SHAP| per cofactor", y=1.06, fontsize=11)
    # Add horizontal colorbar between title and heatmap using heatmap axes position
    g.figure.canvas.draw()
    hm_pos = g.ax_heatmap.get_position()
    cbar_ax = g.figure.add_axes(
        (hm_pos.x0 + hm_pos.width * 0.15, hm_pos.y1 + 0.14, hm_pos.width * 0.7, 0.03)
    )
    norm = colors.Normalize(
        vmin=clustermap_data.values.min(), vmax=clustermap_data.values.max()
    )
    cb = g.figure.colorbar(
        cm.ScalarMappable(norm=norm, cmap="Reds"),
        cax=cbar_ax,
        orientation="horizontal",
    )
    cb.set_label("Mean |SHAP|", fontsize=9)
    clustermap_path = f"{FIG_DIR}/shap_clustermap.png"
    g.savefig(clustermap_path, bbox_inches="tight", dpi=120)
    plt.close(g.figure)
    logger.info(f"Saved SHAP clustermap: {clustermap_path}")
    if wandb_run is not None:
        import wandb

        wandb_run.log({"shap/clustermap": wandb.Image(clustermap_path)})


# ---------------------------------------------------------------------------
# Backend-specific SHAP value computation
# ---------------------------------------------------------------------------


def _force_single_thread_xgboost(model) -> None:
    """Keep XGBoost SHAP calls on one native thread for macOS/OpenMP stability."""
    try:
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        if hasattr(booster, "set_param"):
            booster.set_param({"nthread": 1})
    except Exception:
        logger.debug("Could not force XGBoost model to single-threaded SHAP mode")


def _infer_gandalf_class_names(
    final_model,
    prediction_data: pd.DataFrame,
    train_data: pd.DataFrame,
    target_col: str,
) -> list[str]:
    try:
        pred_df = final_model.predict(prediction_data.reset_index())
        prob_cols = [
            col
            for col in pred_df.columns
            if col.startswith(f"{target_col}_") and col.endswith("_probability")
        ]
        if prob_cols:
            return [
                col.removeprefix(f"{target_col}_").removesuffix("_probability")
                for col in prob_cols
            ]
    except Exception as exc:
        logger.warning(f"Could not infer class names from predictions: {exc}")

    return sorted(train_data[target_col].astype(str).unique())


def _xgb_shap_values(
    final_model,
    train_data: pd.DataFrame,
    X_bg: pd.DataFrame,
    X_regions: pd.DataFrame,
    target_cols: list[str],
    resolved_task: str,
    is_classifier_artifact: bool,
) -> tuple[list[np.ndarray], str, list[str]]:
    """Compute SHAP values for XGBoost using TreeExplainer; returns (sv_list, target_col, classes)."""
    sv_list: list[np.ndarray] = []
    if is_classifier_artifact:
        estimators = [final_model["model"]]
    else:
        estimators = final_model if isinstance(final_model, list) else [final_model]

    for booster in estimators:
        _force_single_thread_xgboost(booster)
        explainer = shap_pkg.Explainer(booster, X_bg)
        ex = explainer(X_regions, check_additivity=False)
        sv_list.extend(shap_values_to_output_list(ex))

    target_col = (
        final_model.get("target_col", target_cols[0])
        if is_classifier_artifact
        else target_cols[0]
    )
    if resolved_task == "classification":
        if is_classifier_artifact:
            classes = [str(cls) for cls in final_model["classes"]]
        else:
            estimator = estimators[0]
            if hasattr(estimator, "classes_") and len(
                getattr(estimator, "classes_")
            ) == len(sv_list):
                classes = [str(cls) for cls in estimator.classes_]
            else:
                classes = sorted(train_data[target_col].astype(str).unique())
    else:
        classes = []
    return sv_list, target_col, classes


def _catboost_shap_values(
    final_model,
    train_data: pd.DataFrame,
    X_bg: pd.DataFrame,
    X_regions: pd.DataFrame,
    target_cols: list[str],
    resolved_task: str,
    is_classifier_artifact: bool,
) -> tuple[list[np.ndarray], str, list[str]]:
    """Compute SHAP values for CatBoost using TreeExplainer; returns (sv_list, target_col, classes)."""
    sv_list: list[np.ndarray] = []
    if is_classifier_artifact:
        estimators = [final_model["model"]]
    else:
        estimators = final_model if isinstance(final_model, list) else [final_model]

    for estimator in estimators:
        explainer = shap_pkg.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_regions, check_additivity=False)
        sv_list.extend(shap_values_to_output_list(shap_values))

    target_col = (
        final_model.get("target_col", target_cols[0])
        if is_classifier_artifact
        else target_cols[0]
    )
    if resolved_task == "classification":
        if is_classifier_artifact:
            classes = [str(cls) for cls in final_model["classes"]]
        else:
            classes = sorted(train_data[target_col].astype(str).unique())
    else:
        classes = []
    return sv_list, target_col, classes


def _gandalf_shap_values(
    final_model,
    train_data: pd.DataFrame,
    X_bg: pd.DataFrame,
    X_regions: pd.DataFrame,
    shap_data: pd.DataFrame,
    target_cols: list[str],
    resolved_task: str,
) -> tuple[list[np.ndarray], str, list[str]]:
    """Compute SHAP values for GANDALF (PyTorch) using GradientExplainer."""
    import torch

    pt_model = final_model.model
    if pt_model is None:
        raise RuntimeError("TabularModel has no underlying model — was it trained?")
    device = next(pt_model.parameters()).device
    pt_model.eval()

    # GradientExplainer requires a model that takes a plain tensor.
    # Wrap the pytorch-tabular model's dict interface.
    class _TensorWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model({"continuous": x})["logits"]

    wrapped = _TensorWrapper(pt_model).to(device)

    bg_tensor = torch.tensor(X_bg.values, dtype=torch.float32, device=device)
    test_tensor = torch.tensor(X_regions.values, dtype=torch.float32, device=device)

    explainer = shap_pkg.GradientExplainer(wrapped, bg_tensor)
    shap_values = explainer.shap_values(test_tensor)
    sv_list = shap_values_to_output_list(shap_values)

    target_col = target_cols[0]
    if resolved_task == "classification":
        classes = _infer_gandalf_class_names(final_model, shap_data, train_data, target_col)
    else:
        classes = []
    return sv_list, target_col, classes


_BACKEND_EXPLAINER_DESCRIPTIONS = {
    "xgboost": "XGBoost (TreeExplainer)",
    "catboost": "CatBoost (TreeExplainer via shap package)",
    "gandalf": "GANDALF (GradientExplainer)",
}


# ---------------------------------------------------------------------------
# Unified SHAP entry point
# ---------------------------------------------------------------------------


def compute_shap(
    model_type: str,
    final_model,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    eval_data: pd.DataFrame | None = None,
    RES_DIR: str = "results",
    FIG_DIR: str = "figures",
    tile_size: int = 100,
    task: str = "auto",
    wandb_run=None,
):
    """Compute and plot SHAP values for the configured model backend."""
    figure_style()
    SHAP_DIR = os.path.join(RES_DIR, "shap")
    os.makedirs(SHAP_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    if train_data.empty or not feature_cols:
        raise ValueError(
            "Cannot compute SHAP: empty training data or no feature columns"
        )

    is_classifier_artifact = (
        isinstance(final_model, dict) and final_model.get("task") == "classification"
    )
    resolved_task = (
        "classification"
        if is_classifier_artifact
        else resolve_task(task, train_data, target_cols)
    )

    X_bg = train_data[feature_cols]
    shap_data = (
        pd.concat([train_data, eval_data, test_data], axis=0)
        if eval_data is not None
        else pd.concat([train_data, test_data], axis=0)
    )
    if model_type == "catboost":
        # CatBoost's SHAP previously evaluated the training population only.
        X_regions = X_bg
    else:
        X_regions = shap_data[feature_cols]

    logger.info(
        "Computing SHAP values for {} (background={}, shap_regions={})".format(
            _BACKEND_EXPLAINER_DESCRIPTIONS.get(model_type, model_type),
            len(X_bg),
            len(X_regions),
        )
    )

    if model_type == "xgboost":
        sv_list, target_col, classes = _xgb_shap_values(
            final_model,
            train_data,
            X_bg,
            X_regions,
            target_cols,
            resolved_task,
            is_classifier_artifact,
        )
    elif model_type == "catboost":
        sv_list, target_col, classes = _catboost_shap_values(
            final_model,
            train_data,
            X_bg,
            X_regions,
            target_cols,
            resolved_task,
            is_classifier_artifact,
        )
    else:
        if resolved_task == "classification":
            from tabnado.utils import require_single_classification_target

            require_single_classification_target(target_cols)
        sv_list, target_col, classes = _gandalf_shap_values(
            final_model,
            train_data,
            X_bg,
            X_regions,
            shap_data,
            target_cols,
            resolved_task,
        )

    if not sv_list:
        raise ValueError("SHAP explainer returned no values")

    if resolved_task == "classification":
        output_cols = classification_shap_output_columns(
            target_col,
            classes,
            len(sv_list),
        )
    else:
        output_cols = default_shap_output_columns(target_cols, len(sv_list))

    mean_abs_shap = pd.DataFrame(
        np.stack([np.abs(sv).mean(axis=0) for sv in sv_list]).T,
        index=feature_cols,
        columns=output_cols,
    )
    shap_mean_csv = f"{SHAP_DIR}/shap_mean_abs.csv"
    mean_abs_shap.to_csv(shap_mean_csv)
    logger.info(
        "Saved SHAP summary table: {} (features_kept={})".format(
            shap_mean_csv, len(mean_abs_shap)
        )
    )

    _plot_clustermap(mean_abs_shap, output_cols, FIG_DIR, wandb_run=wandb_run)

    plot_shap_stacked_bar(
        mean_abs_shap,
        FIG_DIR,
        SHAP_DIR,
        wandb_run=wandb_run,
    )

    # === SPATIAL SHAP ANALYSIS ===
    # Aggregate SHAP values by genomic offset from TSS
    _plot_spatial_shap(
        sv_list,
        feature_cols,
        output_cols,
        X_regions.index,
        SHAP_DIR,
        FIG_DIR,
        tile_size,
        wandb_run=wandb_run,
    )


_MODEL_FILENAMES = {
    "xgboost": "xgboost_model.joblib",
    "catboost": "catboost_model.joblib",
}


def _load_final_model(model_type: str, RES_DIR: str):
    model_path = os.path.join(RES_DIR, "final_model")

    if model_type in _MODEL_FILENAMES:
        from joblib import load

        artifact_path = os.path.join(model_path, _MODEL_FILENAMES[model_type])
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"No model found at {artifact_path}. Run train first."
            )
        return load(artifact_path)

    from pytorch_tabular import TabularModel

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}. Run train first.")
    final_model = TabularModel.load_model(model_path)
    if final_model.model is None:
        raise RuntimeError("Loaded model has no weights — check model directory")
    return final_model


def main():
    """CLI entry point for SHAP analysis."""
    from tabnado.data import load_data

    params = PipelineParams.from_yaml(parse_params_arg())
    setup_logger(params["RES_DIR"], params["PROJECT"])
    run_start = perf_counter()
    model_type = params["MODEL_TYPE"]
    logger.info(f"========== {model_type.upper()} SHAP START ==========")
    logger.info(
        "SHAP config: project={} target={} res_dir={}".format(
            params["PROJECT"], params["TARGET"], params["RES_DIR"]
        )
    )

    final_model = _load_final_model(model_type, params["RES_DIR"])

    _, _, target_cols, feature_cols, train_data, eval_data, test_data = load_data(
        **vars(params)
    )
    task = resolve_task(params["TASK"], train_data, target_cols)

    compute_shap(
        model_type,
        final_model,
        train_data,
        test_data,
        feature_cols,
        target_cols,
        eval_data=eval_data,
        RES_DIR=params["RES_DIR"],
        FIG_DIR=params["FIG_DIR"],
        task=task,
    )
    logger.info(
        "========== {} SHAP END ({:.2f}s total) ==========".format(
            model_type.upper(), perf_counter() - run_start
        )
    )


if __name__ == "__main__":
    main()
