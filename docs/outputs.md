# Outputs

Outputs are organized by project run folder:

```text
<output_dir>/<MODEL_NAME>_<TARGET>/
```

## Core Artifacts

- `best_hyperparameters.json`: Best hyperparameters from sweep
- `final_model/`: Serialized trained model
- `final_model_checkpoints/`: Training checkpoints

## Evaluation Artifacts

- `evaluate/metrics.json`: Evaluation metrics
- `evaluate/predictions.parquet`: predictions over test chromosome
- `figures/scatter_test.png`: True vs predicted scatter
- `figures/roc_curve_<target>.png`: ROC curve for classification runs
- `evaluate/roc_auc.csv`: ROC AUC summary for classification runs
- `evaluate/embeddings_umap.parquet`: UMAP embedding table
- `figures/embeddings_umap.png`: UMAP figure

## SHAP Artifacts

- `shap/shap_mean_abs.csv`: Mean absolute SHAP by feature and target
- `figures/shap_clustermap.png`: Mean |SHAP| heatmap across cofactors and targets
- `figures/shap_stacked_bar.png`: Stacked mean |SHAP| bar plot by cofactor and output
- `shap/shap_stacked_bar_data.csv`: Data used for the stacked SHAP bar plot
- `shap/spatial_shap_by_offset_<target>.csv`: Spatial SHAP summary by genomic offset per target
- `figures/shap_spatial_heatmap_<target>.png`: Spatial SHAP heatmap (cofactor × offset) per target
- `figures/shap_offset_line_<target>.png`: SHAP importance profile by genomic offset per target

## Logs

- `<PROJECT>.log`: Run-level pipeline log
- `logs/`: TensorBoard files (if enabled)
