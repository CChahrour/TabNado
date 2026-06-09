import json
import os
import tempfile
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from tabnado.utils import (
    classification_metrics,
    figure_style,
    flatten_metric_dict,
    json_safe,
    parse_params_arg,
    resolve_task,
    setup_logger,
)

UMAP_CATEGORICAL_PALETTE_LIMIT = 20


def _get_umap_cls():
    os.environ.setdefault(
        "NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "tabnado-numba-cache")
    )
    from umap import UMAP

    return UMAP


def _limit_categorical_labels(
    labels: pd.Series,
    max_categories: int = UMAP_CATEGORICAL_PALETTE_LIMIT,
) -> pd.Categorical:
    """Return display labels capped to a fixed palette size."""
    labels = pd.Series(labels).astype(str)
    max_categories = max(1, int(max_categories))
    counts = labels.value_counts()

    if len(counts) <= max_categories:
        categories = sorted(counts.index.astype(str))
        return pd.Categorical(labels, categories=categories, ordered=True)

    if max_categories == 1:
        display_labels = pd.Series(["Other"] * len(labels), index=labels.index)
        categories = ["Other"]
    else:
        keep_labels = list(counts.head(max_categories - 1).index.astype(str))
        display_labels = labels.where(labels.isin(keep_labels), "Other")
        categories = keep_labels + ["Other"]

    logger.info(
        "UMAP categorical palette limited from {} labels to {} displayed groups".format(
            len(counts),
            len(categories),
        )
    )
    return pd.Categorical(display_labels, categories=categories, ordered=True)


def _plot_roc_curve(
    y_true: pd.Series,
    probabilities: np.ndarray | None,
    classes: list[str],
    target_col: str,
    FIG_DIR: str,
    EVAL_DIR: str,
    wandb_run=None,
    model_type: str = "gandalf",
) -> dict[str, float]:
    if probabilities is None or not classes:
        logger.info("Skipping ROC curve: probability columns are unavailable")
        return {}

    y_true_arr = pd.Series(y_true).astype(str).to_numpy()
    y_score = np.asarray(probabilities, dtype=float)
    if y_score.ndim == 1:
        y_score = np.column_stack([1.0 - y_score, y_score])

    if y_score.shape[1] != len(classes):
        logger.warning(
            "Skipping ROC curve: probability columns ({}) do not match classes ({})".format(
                y_score.shape[1],
                len(classes),
            )
        )
        return {}

    if len(np.unique(y_true_arr)) < 2:
        logger.warning("Skipping ROC curve: test set contains fewer than two classes")
        return {}

    fig, ax = plt.subplots(figsize=(6, 5))
    auc_rows: list[dict[str, float | str]] = []
    metrics: dict[str, float] = {}

    if len(classes) == 2:
        positive_label = classes[1]
        y_binary = (y_true_arr == positive_label).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_score[:, 1])
        roc_auc = float(auc(fpr, tpr))
        metrics["roc_auc"] = roc_auc
        auc_rows.append({"class": positive_label, "auc": roc_auc})
        ax.plot(fpr, tpr, linewidth=2, label=f"{positive_label} AUC={roc_auc:.3f}")
    else:
        y_bin = label_binarize(y_true_arr, classes=classes)
        fpr_by_class: dict[str, np.ndarray] = {}
        tpr_by_class: dict[str, np.ndarray] = {}

        for idx, class_name in enumerate(classes):
            y_class = y_bin[:, idx]
            if len(np.unique(y_class)) < 2:
                logger.warning(
                    f"Skipping ROC for class '{class_name}': no positive or negative samples"
                )
                continue
            fpr, tpr, _ = roc_curve(y_class, y_score[:, idx])
            roc_auc = float(auc(fpr, tpr))
            fpr_by_class[class_name] = fpr
            tpr_by_class[class_name] = tpr
            metrics[f"roc_auc_{class_name}"] = roc_auc
            auc_rows.append({"class": class_name, "auc": roc_auc})
            ax.plot(fpr, tpr, linewidth=1.5, label=f"{class_name} AUC={roc_auc:.3f}")

        if fpr_by_class:
            fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
            auc_micro = float(auc(fpr_micro, tpr_micro))
            metrics["roc_auc_micro"] = auc_micro
            auc_rows.append({"class": "micro", "auc": auc_micro})
            ax.plot(
                fpr_micro,
                tpr_micro,
                color="black",
                linestyle=":",
                linewidth=2,
                label=f"micro AUC={auc_micro:.3f}",
            )

            all_fpr = np.unique(np.concatenate(list(fpr_by_class.values())))
            mean_tpr = np.zeros_like(all_fpr)
            for class_name in fpr_by_class:
                mean_tpr += np.interp(
                    all_fpr,
                    fpr_by_class[class_name],
                    tpr_by_class[class_name],
                )
            mean_tpr /= len(fpr_by_class)
            auc_macro = float(auc(all_fpr, mean_tpr))
            metrics["roc_auc_macro"] = auc_macro
            auc_rows.append({"class": "macro", "auc": auc_macro})
            ax.plot(
                all_fpr,
                mean_tpr,
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"macro AUC={auc_macro:.3f}",
            )

    if not auc_rows:
        plt.close(fig)
        logger.warning("Skipping ROC curve: no valid ROC curves were computed")
        return {}

    auc_path = f"{EVAL_DIR}/roc_auc.csv"
    pd.DataFrame(auc_rows).to_csv(auc_path, index=False)
    logger.info(f"Saved ROC AUC table: {auc_path}")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{target_col} ROC ({model_type.upper()})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    roc_path = f"{FIG_DIR}/roc_curve_{target_col}.png"
    fig.savefig(roc_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info(f"Saved ROC curve: {roc_path}")

    if wandb_run is not None:
        import wandb

        wandb_run.log({f"eval/roc_{target_col}": wandb.Image(roc_path)})

    return metrics


def evaluate_model(
    final_model,
    test_data: pd.DataFrame,
    target_cols: list[str],
    feature_cols: list[str] | None = None,
    FIG_DIR: str = "figures",
    RES_DIR: str = "results",
    model_type: str = "gandalf",
    task: str = "regression",
    wandb_run=None,
):
    figure_style()
    logger.info("Evaluating final model on test set")
    EVAL_DIR = f"{RES_DIR}/evaluate"
    os.makedirs(EVAL_DIR, exist_ok=True)

    if model_type == "xgboost":
        from tabnado.train import predict_xgboost

        if feature_cols is None:
            raise ValueError("feature_cols required for xgboost evaluation")
        pred_df = predict_xgboost(final_model, test_data, feature_cols, target_cols)
    elif model_type == "catboost":
        from tabnado.train import predict_catboost

        if feature_cols is None:
            raise ValueError("feature_cols required for catboost evaluation")
        pred_df = predict_catboost(final_model, test_data, feature_cols, target_cols)
    else:
        pred_df = final_model.predict(test_data.reset_index())
        if task == "classification":
            target_col = target_cols[0]
            pred_col = f"{target_col}_prediction"
            prob_cols = [
                col
                for col in pred_df.columns
                if col.startswith(f"{target_col}_") and col.endswith("_probability")
            ]
            pred_df = pred_df[[pred_col] + prob_cols].rename(
                columns={pred_col: target_col}
            )
            pred_df.index = test_data.index
        else:
            pred_df.columns = [c.replace("_prediction", "") for c in pred_df.columns]

    if task == "classification":
        target_col = target_cols[0]
        y_true = test_data[target_col].astype(str)
        y_pred = pred_df[target_col].astype(str)
        prob_cols = [
            col
            for col in pred_df.columns
            if col.startswith(f"{target_col}_") and col.endswith("_probability")
        ]
        classes = [
            col.removeprefix(f"{target_col}_").removesuffix("_probability")
            for col in prob_cols
        ]
        probabilities = pred_df[prob_cols].values if prob_cols else None
        metrics = classification_metrics(
            y_true,
            y_pred,
            probabilities=probabilities,
            classes=classes if classes else None,
        )
        logger.info(
            "Accuracy={accuracy:.4f}  Balanced accuracy={balanced_accuracy:.4f}  "
            "Macro F1={macro_f1:.4f}".format(**metrics)
        )

        pred_save_path = f"{EVAL_DIR}/predictions.parquet"
        pred_save = pred_df.copy()
        pred_save.index = test_data.index
        pred_save[f"{target_col}_true"] = y_true.values
        pred_save.to_parquet(pred_save_path)

        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).T
        report_df.to_csv(f"{EVAL_DIR}/classification_report.csv")

        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            f"{EVAL_DIR}/confusion_matrix.csv"
        )

        metrics.update(
            _plot_roc_curve(
                y_true,
                probabilities,
                classes,
                target_col,
                FIG_DIR,
                EVAL_DIR,
                wandb_run=wandb_run,
                model_type=model_type,
            )
        )

        with open(f"{EVAL_DIR}/metrics.json", "w") as f:
            json.dump(json_safe(metrics), f, indent=4)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{target_col} ({model_type.upper()})")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        confusion_path = f"{FIG_DIR}/confusion_matrix_{target_col}.png"
        fig.savefig(confusion_path)
        plt.close(fig)
        logger.info(f"Saved confusion matrix: {confusion_path}")

        if wandb_run is not None:
            wandb_run.log(flatten_metric_dict(metrics, prefix="eval/"))
        return

    y_true = test_data[target_cols].values
    y_pred = pred_df[target_cols].values
    if np.isnan(y_pred).any():
        logger.warning(
            "NaN predictions detected in evaluate_model — results may be unreliable"
        )
    # save predictions alongside true values for downstream analysis
    pred_save_path = f"{EVAL_DIR}/predictions.parquet"
    pred_save = pred_df.copy()
    pred_save.index = test_data.index
    for col in target_cols:
        pred_save[f"{col}_true"] = test_data[col].values
    pred_save.to_parquet(pred_save_path)
    metrics = {
        "R2_macro": r2_score(y_true, y_pred, multioutput="uniform_average"),
        "MSE_macro": mean_squared_error(y_true, y_pred),
        "MAE_macro": mean_absolute_error(y_true, y_pred),
    }
    logger.info(f"R2  (macro): {metrics['R2_macro']:.4f}")
    logger.info(f"MSE (macro): {metrics['MSE_macro']:.4f}")
    logger.info(f"MAE (macro): {metrics['MAE_macro']:.4f}")
    for i, col in enumerate(target_cols):
        metrics[col] = {
            "R2": r2_score(y_true[:, i], y_pred[:, i]),
            "MSE": mean_squared_error(y_true[:, i], y_pred[:, i]),
            "MAE": mean_absolute_error(y_true[:, i], y_pred[:, i]),
            "Rho": spearmanr(y_true[:, i], y_pred[:, i])[0],
        }

    # Save metrics to a JSON file
    metrics_path = f"{EVAL_DIR}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    if wandb_run is not None:
        flat = {
            "eval/R2_macro": metrics["R2_macro"],
            "eval/MSE_macro": metrics["MSE_macro"],
            "eval/MAE_macro": metrics["MAE_macro"],
        }
        for col in target_cols:
            for stat in ("R2", "MSE", "MAE", "Rho"):
                flat[f"eval/{col}/{stat}"] = metrics[col][stat]
        wandb_run.log(flat)

    for col in target_cols:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(test_data[col], pred_df[col], s=2, alpha=0.3, rasterized=True)
        r2 = r2_score(test_data[col], pred_df[col])
        ax.set_title(f"{col} ({model_type.upper()})\nR2={r2:.3f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        fig.tight_layout()
        scatter_path = f"{FIG_DIR}/scatter_test_{col}.png"
        fig.savefig(scatter_path)
        plt.close(fig)
        logger.info(f"Saved evaluation scatter plot: {scatter_path}")
        if wandb_run is not None:
            import wandb

            wandb_run.log({f"eval/scatter_{col}": wandb.Image(scatter_path)})


def compute_umap_embeddings(
    final_model,
    test_data: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    FIG_DIR: str = "figures",
    RES_DIR: str = "results",
    target: str = "",
    model_type: str = "gandalf",
    task: str = "regression",
    wandb_run=None,
):
    if model_type in {"xgboost", "catboost"}:
        logger.info(f"Skipping UMAP: not supported for {model_type} backend")
        return

    import torch

    EVAL_DIR = f"{RES_DIR}/evaluate"
    os.makedirs(EVAL_DIR, exist_ok=True)
    emb_path = f"{EVAL_DIR}/embeddings_umap.parquet"

    compute_embeddings = True
    if os.path.exists(emb_path):
        logger.info(f"Loading cached UMAP embeddings from {emb_path}")
        proj_df = pd.read_parquet(emb_path)
        if task == "classification":
            compute_embeddings = not {"true_label", "true_label_code"}.issubset(
                proj_df.columns
            )
        else:
            compute_embeddings = "true_mean" not in proj_df.columns
        if compute_embeddings:
            logger.info("Cached UMAP table does not match task; recomputing")

    if compute_embeddings:
        logger.info("Computing UMAP embeddings from model backbone activations")
        pt_model = final_model.model
        if pt_model is None:
            raise RuntimeError("TabularModel has no underlying model — was it trained?")
        X = test_data[feature_cols].to_numpy()
        device = next(pt_model.parameters()).device
        emb_batches = []

        def _grab_backbone(module, inputs, output):
            emb_batches.append(output.detach().cpu())

        hook = pt_model.backbone.register_forward_hook(_grab_backbone)
        pt_model.eval()
        with torch.no_grad():
            for i in range(0, len(X), 1024):
                xb = torch.tensor(X[i : i + 1024], dtype=torch.float32, device=device)
                pt_model({"continuous": xb})
        hook.remove()

        E = torch.cat(emb_batches, dim=0).numpy()
        n_samples = E.shape[0]
        if n_samples < 3:
            # UMAP is unstable for extremely small sample counts; provide a deterministic fallback.
            proj = np.column_stack(
                [np.arange(n_samples, dtype=float), np.zeros(n_samples)]
            )
        else:
            n_neighbors = min(15, n_samples - 1)
            proj = _get_umap_cls()(
                n_neighbors=n_neighbors, min_dist=0.1, random_state=42
            ).fit_transform(E)  # type: ignore[assignment]
        proj_df = pd.DataFrame(np.asarray(proj), columns=["UMAP1", "UMAP2"])
        if task == "classification":
            target_col = target_cols[0]
            proj_df["true_label"] = test_data[target_col].astype(str).values
            categories = pd.Categorical(proj_df["true_label"])
            proj_df["true_label_code"] = categories.codes
        else:
            proj_df["true_mean"] = test_data[target_cols].mean(axis=1).values
        proj_df.to_parquet(emb_path)
        logger.info(f"Saved UMAP embeddings table: {emb_path}")

    fig, ax = plt.subplots(figsize=(6, 5))
    if task == "classification":
        display_labels = _limit_categorical_labels(proj_df["true_label"])
        labels = list(display_labels.categories)
        label_codes = display_labels.codes

        tab20_colors = list(plt.get_cmap("tab20").colors)
        if labels and labels[-1] == "Other":
            colors = tab20_colors[: max(len(labels) - 1, 0)] + [(0.45, 0.45, 0.45)]
        else:
            colors = tab20_colors[: len(labels)]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(-0.5, len(labels) + 0.5), cmap.N)

        sc = ax.scatter(
            proj_df.UMAP1,
            proj_df.UMAP2,
            c=label_codes,
            cmap=cmap,
            norm=norm,
            s=5,
            alpha=0.7,
        )
        handles = []
        for idx, label_name in enumerate(labels):
            legend_label = label_name
            if label_name == "Other":
                legend_label = f"Other ({int((display_labels == 'Other').sum())})"
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=5,
                    markerfacecolor=cmap(idx),
                    markeredgecolor="none",
                    label=legend_label,
                )
            )
        ax.legend(
            handles=handles,
            title=target or target_cols[0],
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=8,
            title_fontsize=9,
        )
    else:
        sc = ax.scatter(
            proj_df.UMAP1,
            proj_df.UMAP2,
            c=proj_df.true_mean,
            cmap="mako",
            s=5,
            alpha=0.7,
        )
        label = (
            f"Mean {target} (scaled RPKM)" if target else "Mean target (scaled RPKM)"
        )
        plt.colorbar(sc, ax=ax, label=label)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"{model_type.upper()} backbone embeddings")
    umap_fig_path = f"{FIG_DIR}/embeddings_umap.png"
    fig.savefig(umap_fig_path)
    plt.close(fig)
    logger.info(f"Saved UMAP figure: {umap_fig_path}")
    if wandb_run is not None:
        import wandb

        wandb_run.log({"eval/umap": wandb.Image(umap_fig_path)})


def main():
    from tabnado.data import load_data
    from tabnado.params import PipelineParams

    params = PipelineParams.from_yaml(parse_params_arg())
    setup_logger(params.RES_DIR, params.PROJECT)
    run_start = perf_counter()
    logger.info("========== EVALUATE START ==========")
    logger.info(
        f"Evaluate config: project={params.PROJECT} target={params.TARGET} res_dir={params.RES_DIR}"
    )

    model_path = os.path.join(params["RES_DIR"], "final_model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}. Run train.py first.")

    _, _, target_cols, feature_cols, train_data, _, test_data = load_data(
        **vars(params)
    )
    task = resolve_task(params.TASK, train_data, target_cols)

    if params.MODEL_TYPE == "xgboost":
        from joblib import load

        xgb_path = os.path.join(model_path, "xgboost_model.joblib")
        final_model = load(xgb_path)
        model_type = "xgboost"
    elif params.MODEL_TYPE == "catboost":
        from joblib import load

        catboost_path = os.path.join(model_path, "catboost_model.joblib")
        final_model = load(catboost_path)
        model_type = "catboost"
    else:
        from pytorch_tabular import TabularModel

        final_model = TabularModel.load_model(model_path)
        if final_model.model is None:
            raise RuntimeError("Loaded model has no weights — check model directory")
        model_type = "gandalf"

    evaluate_model(
        final_model,
        test_data,
        target_cols,
        feature_cols=feature_cols,
        FIG_DIR=params["FIG_DIR"],
        RES_DIR=params["RES_DIR"],
        model_type=model_type,
        task=task,
    )
    compute_umap_embeddings(
        final_model,
        test_data,
        feature_cols,
        target_cols,
        FIG_DIR=params["FIG_DIR"],
        RES_DIR=params["RES_DIR"],
        target=params["TARGET"],
        model_type=model_type,
        task=task,
    )
    logger.info(
        "========== EVALUATE END ({:.2f}s total) ==========".format(
            perf_counter() - run_start
        )
    )


if __name__ == "__main__":
    main()
