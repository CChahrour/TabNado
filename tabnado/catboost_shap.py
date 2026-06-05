import os
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from loguru import logger
from matplotlib import cm, colors

from tabnado.params import PipelineParams
from tabnado.shap_utils import (
    default_shap_output_columns,
    plot_shap_stacked_bar,
    shap_values_to_output_list,
)
from tabnado.tasks import classification_shap_output_columns, resolve_task
from tabnado.utils import figure_style, parse_params_arg, setup_logger
from tabnado.xgb_shap import _parse_offset_from_column, _plot_spatial_shap


def compute_catboost_shap(
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
    """Compute SHAP values for CatBoost models using the shap package."""
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
    is_multiclass_classifier = False
    if resolved_task == "classification":
        if is_classifier_artifact:
            is_multiclass_classifier = (
                final_model.get("problem_type") == "multiclass"
                or len(final_model.get("classes", [])) > 2
            )
        else:
            is_multiclass_classifier = (
                train_data[target_cols[0]].astype(str).nunique() > 2
            )

    X_bg = train_data[feature_cols]
    shap_data = (
        pd.concat([train_data, eval_data, test_data], axis=0)
        if eval_data is not None
        else pd.concat([train_data, test_data], axis=0)
    )
    X_regions = shap_data[feature_cols]
    background_label = "none" if is_multiclass_classifier else str(len(X_bg))
    logger.info(
        "Computing SHAP values for CatBoost with shap.Explainer "
        "(background={}, shap_regions={})".format(background_label, len(X_regions))
    )

    sv_list: list[np.ndarray] = []
    if is_classifier_artifact:
        estimators = [final_model["model"]]
    else:
        estimators = final_model if isinstance(final_model, list) else [final_model]

    for estimator in estimators:
        if is_multiclass_classifier:
            # SHAP's CatBoost multiclass TreeExplainer crashes with a background
            # dataset in the current supported stack, but the no-background
            # TreeExplainer returns per-class values correctly.
            explainer = shap.Explainer(estimator)
        else:
            explainer = shap.Explainer(estimator, X_bg)
        ex = explainer(X_regions, check_additivity=False)
        sv_list.extend(shap_values_to_output_list(ex))

    if not sv_list:
        raise ValueError("SHAP explainer returned no values")

    if resolved_task == "classification":
        target_col = (
            final_model.get("target_col", target_cols[0])
            if is_classifier_artifact
            else target_cols[0]
        )
        if is_classifier_artifact:
            classes = [str(cls) for cls in final_model["classes"]]
        else:
            classes = sorted(train_data[target_col].astype(str).unique())
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
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), fontsize=9, rotation=0
    )
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), fontsize=9, rotation=30, ha="right"
    )
    g.figure.suptitle("Mean |SHAP| per cofactor", y=1.06, fontsize=11)
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

    plot_shap_stacked_bar(
        mean_abs_shap,
        FIG_DIR,
        SHAP_DIR,
        wandb_run=wandb_run,
    )

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


def main():
    """CLI entry point for CatBoost SHAP analysis."""
    from joblib import load

    from tabnado.data import load_data

    params = PipelineParams.from_yaml(parse_params_arg())
    setup_logger(params["RES_DIR"], params["PROJECT"])
    run_start = perf_counter()
    logger.info("========== CATBOOST SHAP START ==========")
    logger.info(
        "SHAP config: project={} target={} res_dir={}".format(
            params["PROJECT"], params["TARGET"], params["RES_DIR"]
        )
    )

    model_path = os.path.join(
        params["RES_DIR"], "final_model", "catboost_model.joblib"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at {model_path}. Run catboost_train first."
        )

    final_model = load(model_path)

    _, _, target_cols, feature_cols, train_data, eval_data, test_data = load_data(
        **vars(params)
    )
    task = resolve_task(params["TASK"], train_data, target_cols)

    compute_catboost_shap(
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
        "========== CATBOOST SHAP END ({:.2f}s total) ==========".format(
            perf_counter() - run_start
        )
    )


if __name__ == "__main__":
    main()
