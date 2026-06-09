# Outputs

All outputs are written under:

```text
<output_dir>/<model_name>_<target>/
```

## Core Artifacts

| Path | Contents |
|------|----------|
| `best_hyperparameters.json` | Best hyperparameters from the Optuna sweep |
| `final_model/` | Serialized trained model (`xgboost_model.joblib`, `catboost_model.joblib`, or GANDALF checkpoint directory) |
| `final_model_checkpoints/` | Training checkpoints (GANDALF only) |

## Evaluation Artifacts

| Path | Contents |
|------|----------|
| `evaluate/metrics.json` | Test-set evaluation metrics |
| `evaluate/predictions.parquet` | Model predictions on the test chromosome |
| `evaluate/embeddings_umap.parquet` | UMAP embedding coordinates |
| `figures/embeddings_umap.png` | UMAP embedding figure |

**Regression only:**

| Path | Contents |
|------|----------|
| `figures/scatter_test_<target>.png` | True vs predicted scatter per target |

**Classification only:**

| Path | Contents |
|------|----------|
| `figures/roc_curve_<target>.png` | ROC curve per target |
| `evaluate/roc_auc.csv` | ROC AUC summary |
| `evaluate/classification_report.csv` | Per-class precision, recall, and F1 |
| `evaluate/confusion_matrix.csv` | Confusion matrix |
| `figures/confusion_matrix_<target>.png` | Confusion matrix heatmap |

## SHAP Artifacts

| Path | Contents |
|------|----------|
| `shap/shap_mean_abs.csv` | Mean absolute SHAP by feature and target |
| `shap/shap_stacked_bar_data.csv` | Data for the stacked bar plot |
| `figures/shap_clustermap.png` | Mean \|SHAP\| heatmap across cofactors and targets |
| `figures/shap_stacked_bar.png` | Stacked mean \|SHAP\| bar by cofactor and output |

**Spatial SHAP** (requires tiled windows with offset information):

| Path | Contents |
|------|----------|
| `shap/spatial_shap_by_offset_<target>.csv` | Spatial SHAP by genomic offset per target |
| `figures/shap_spatial_heatmap_<target>.png` | Spatial SHAP heatmap (cofactor × offset) per target |
| `figures/shap_offset_line_<target>.png` | SHAP importance profile by genomic offset per target |

## Logs

| Path | Contents |
|------|----------|
| `<project>.log` | Run-level pipeline log |
| `logs/` | TensorBoard event files (when `logging: tensorboard`) |
