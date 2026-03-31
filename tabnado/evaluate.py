import json
import os
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from loguru import logger
from pytorch_tabular import TabularModel
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tabnado.utils import figure_style, parse_params_arg, setup_logger


def evaluate_model(
    final_model,
    test_data: pd.DataFrame,
    target_cols: list[str],
    feature_cols: list[str] | None = None,
    FIG_DIR: str = "figures",
    RES_DIR: str = "results",
    model_type: str = "gandalf",
    wandb_run=None,
):
    figure_style()
    logger.info("Evaluating final model on test set")
    EVAL_DIR = f"{RES_DIR}/evaluate"
    os.makedirs(EVAL_DIR, exist_ok=True)

    if model_type == "xgboost":
        from tabnado.xgb_train import predict_xgboost

        if feature_cols is None:
            raise ValueError("feature_cols required for xgboost evaluation")
        pred_df = predict_xgboost(final_model, test_data, feature_cols, target_cols)
    else:
        pred_df = final_model.predict(test_data.reset_index())
        pred_df.columns = [c.replace("_prediction", "") for c in pred_df.columns]

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
        ax.set_title(f"{col}\nR2={r2:.3f}")
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
    wandb_run=None,
):
    if model_type == "xgboost":
        logger.info("Skipping UMAP: not supported for XGBoost backend")
        return

    EVAL_DIR = f"{RES_DIR}/evaluate"
    os.makedirs(EVAL_DIR, exist_ok=True)
    emb_path = f"{EVAL_DIR}/embeddings_umap.parquet"

    if os.path.exists(emb_path):
        logger.info(f"Loading cached UMAP embeddings from {emb_path}")
        proj_df = pd.read_parquet(emb_path)
    else:
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
            proj = umap.UMAP(
                n_neighbors=n_neighbors, min_dist=0.1, random_state=42
            ).fit_transform(E)  # type: ignore[assignment]
        proj_df = pd.DataFrame(np.asarray(proj), columns=["UMAP1", "UMAP2"])
        proj_df["true_mean"] = test_data[target_cols].mean(axis=1).values
        proj_df.to_parquet(emb_path)
        logger.info(f"Saved UMAP embeddings table: {emb_path}")

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        proj_df.UMAP1,
        proj_df.UMAP2,
        c=proj_df.true_mean,
        cmap="mako",
        s=5,
        alpha=0.7,
    )
    label = f"Mean {target} (scaled RPKM)" if target else "Mean target (scaled RPKM)"
    plt.colorbar(sc, ax=ax, label=label)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("GANDALF backbone embeddings")
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

    xgb_path = os.path.join(model_path, "xgboost_model.joblib")
    if os.path.exists(xgb_path):
        from joblib import load

        final_model = load(xgb_path)
        model_type = "xgboost"
    else:
        final_model = TabularModel.load_model(model_path)
        if final_model.model is None:
            raise RuntimeError("Loaded model has no weights — check model directory")
        model_type = "gandalf"

    _, _, target_cols, feature_cols, _, _, test_data = load_data(**vars(params))

    evaluate_model(
        final_model,
        test_data,
        target_cols,
        feature_cols=feature_cols,
        FIG_DIR=params["FIG_DIR"],
        RES_DIR=params["RES_DIR"],
        model_type=model_type,
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
    )
    logger.info(
        "========== EVALUATE END ({:.2f}s total) ==========".format(
            perf_counter() - run_start
        )
    )


if __name__ == "__main__":
    main()
