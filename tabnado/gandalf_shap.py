import os
from time import perf_counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
from loguru import logger
from pytorch_tabular import TabularModel

from tabnado.utils import LOAD_DATA_PARAMS, figure_style


def compute_gandalf_shap(
    final_model: TabularModel,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    RES_DIR: str = "results",
    FIG_DIR: str = "figures",
    tile_size: int = 100,
):
    """Compute SHAP values for GANDALF (PyTorch) model using KernelExplainer."""
    figure_style()
    SHAP_DIR = os.path.join(RES_DIR, "shap")
    os.makedirs(SHAP_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    if train_data.empty or not feature_cols:
        raise ValueError(
            "Cannot compute SHAP: empty training data or no feature columns"
        )

    X_bg = train_data[feature_cols].sample(min(1000, len(train_data)), random_state=42)
    X_test_sub = test_data[feature_cols].sample(
        n=min(1000, len(test_data)), random_state=42
    )
    logger.info(
        "Computing SHAP values for GANDALF (background={}, test_subset={})".format(
            len(X_bg), len(X_test_sub)
        )
    )

    pt_model = final_model.model
    if pt_model is None:
        raise RuntimeError("TabularModel has no underlying model — was it trained?")
    device = next(pt_model.parameters()).device
    # SHAP queries the model with tiny batches (often size 1); ensure inference mode.
    pt_model.eval()

    def predict_fn(x_np):
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        # KernelExplainer may call the model with one sample; duplicate to avoid
        # batchnorm failures in models that still expect batch_size > 1.
        single_row = x_np.shape[0] == 1
        if single_row:
            x_np = np.repeat(x_np, 2, axis=0)
        x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            outputs = pt_model({"continuous": x_tensor})
        logits = outputs["logits"].detach().cpu().numpy()
        return logits[:1] if single_row else logits

    n_unique = len(train_data[feature_cols].drop_duplicates())
    k_bg = min(50, n_unique)
    X_bg_kmeans = shap.kmeans(train_data[feature_cols].values, k=k_bg)
    explainer = shap.KernelExplainer(
        predict_fn,
        X_bg_kmeans,
        seed=42,
        feature_names=feature_cols,
        output_names=target_cols,
    )
    shap_values = explainer.shap_values(X_test_sub.values)

    # SHAP may return either a list (one array per target) or a single
    # 3D array shaped (n_samples, n_features, n_targets). Normalize both
    # cases to a list of 2D arrays: one (n_samples, n_features) per target.
    if isinstance(shap_values, list):
        sv_list = [
            np.asarray(sv).squeeze(-1)
            if np.asarray(sv).ndim == 3 and np.asarray(sv).shape[-1] == 1
            else np.asarray(sv)
            for sv in shap_values
        ]
    else:
        sv = np.asarray(shap_values)
        if sv.ndim == 3 and sv.shape[-1] == len(target_cols):
            sv_list = [sv[:, :, i] for i in range(sv.shape[-1])]
        elif sv.ndim == 3 and sv.shape[-1] == 1:
            sv_list = [sv.squeeze(-1)]
        else:
            sv_list = [sv]

    mean_abs_shap = pd.DataFrame(
        np.stack([np.abs(sv).mean(axis=0) for sv in sv_list]).T,
        index=feature_cols,
        columns=target_cols,
    )
    shap_mean_csv = f"{SHAP_DIR}/shap_mean_abs.csv"
    mean_abs_shap.to_csv(shap_mean_csv)
    logger.info(
        "Saved SHAP summary table: {} (features_kept={})".format(
            shap_mean_csv, len(mean_abs_shap)
        )
    )

    # Aggregate over tile offsets: one row per cofactor for a readable clustermap.
    def _strip_offset(col: str) -> str:
        parsed = _parse_offset_from_column(col)
        return parsed[0] if parsed else col

    clustermap_data = mean_abs_shap.copy()
    clustermap_data.index = [_strip_offset(c) for c in clustermap_data.index]
    clustermap_data = clustermap_data.groupby(level=0).mean()

    n_rows = len(clustermap_data)
    n_cols = len(target_cols)
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
        [hm_pos.x0 + hm_pos.width * 0.15, hm_pos.y1 + 0.14, hm_pos.width * 0.7, 0.03]
    )
    norm = mpl.colors.Normalize(
        vmin=clustermap_data.values.min(), vmax=clustermap_data.values.max()
    )
    cb = g.figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="Reds"),
        cax=cbar_ax,
        orientation="horizontal",
    )
    cb.set_label("Mean |SHAP|", fontsize=9)
    clustermap_path = f"{FIG_DIR}/shap_clustermap.png"
    g.savefig(clustermap_path, bbox_inches="tight", dpi=120)
    logger.info(f"Saved SHAP clustermap: {clustermap_path}")

    # === SPATIAL SHAP ANALYSIS ===
    # Aggregate SHAP values by genomic offset from TSS
    _plot_spatial_shap(
        sv_list,
        feature_cols,
        target_cols,
        X_test_sub.index,
        SHAP_DIR,
        FIG_DIR,
        tile_size,
    )


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
        fig, ax = plt.subplots(figsize=(12, 4))
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

        # === Line plot: aggregate across cofactors ===
        offset_importance = spatial_df.mean(axis=0)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(centre_bp, offset_importance.values, marker="o", linewidth=2)
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="TSS center")
        ax.set_xlabel("Offset from TSS (bp)")
        ax.set_ylabel("Mean |SHAP| (avg across cofactors)")
        ax.set_title(f"Genomic Distance Importance Profile ({target_col})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        line_path = f"{FIG_DIR}/shap_offset_line_{target_col.replace('/', '_')}.png"
        fig.savefig(line_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        logger.info(f"Saved offset importance profile: {line_path}")


def main():
    """CLI entry point for GANDALF SHAP analysis."""
    from tabnado.data import load_data
    from tabnado.utils import load_params, parse_params_arg, setup_logger

    params = load_params(parse_params_arg())
    setup_logger(params["RES_DIR"], params["PROJECT"])
    run_start = perf_counter()
    logger.info("========== GANDALF SHAP START ==========")
    logger.info(
        "SHAP config: project={} target={} res_dir={}".format(
            params["PROJECT"], params["TARGET"], params["RES_DIR"]
        )
    )

    model_path = os.path.join(params["RES_DIR"], "final_model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at {model_path}. Run gandalf_train first."
        )

    final_model = TabularModel.load_model(model_path)
    if final_model.model is None:
        raise RuntimeError("Loaded model has no weights — check model directory")

    _, _, target_cols, feature_cols, train_data, _, test_data = load_data(
        **{k: params[k] for k in LOAD_DATA_PARAMS}
    )

    compute_gandalf_shap(
        final_model,
        train_data,
        test_data,
        feature_cols,
        target_cols,
        RES_DIR=params["RES_DIR"],
        FIG_DIR=params["FIG_DIR"],
    )
    logger.info(
        "========== GANDALF SHAP END ({:.2f}s total) ==========".format(
            perf_counter() - run_start
        )
    )


if __name__ == "__main__":
    main()
