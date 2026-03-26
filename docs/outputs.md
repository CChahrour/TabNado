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
- `evaluate/embeddings_umap.parquet`: UMAP embedding table
- `figures/embeddings_umap.png`: UMAP figure

## SHAP Artifacts

- `shap/shap_mean_abs.csv`: Mean absolute SHAP by feature and target
- `figures/shap_clustermap.png`: Mean |SHAP| heatmap across cofactors and targets
- `shap/spatial_shap_by_offset_<target>.csv`: Spatial SHAP summary by genomic offset per target
- `figures/shap_spatial_heatmap_<target>.png`: Spatial SHAP heatmap (cofactor × offset) per target
- `figures/shap_offset_line_<target>.png`: SHAP importance profile by genomic offset per target

## Logs

- `<PROJECT>.log`: Run-level pipeline log
- `logging/`: TensorBoard files (if enabled)
