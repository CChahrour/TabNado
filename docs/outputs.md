# Outputs

Outputs are organized by project run folder:

```text
<output_dir>/<MODEL_NAME>_<TARGET>_<date>/
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

- `shap_mean_abs.csv`: Mean absolute SHAP by feature and target
- `figures/shap_clustermap.png`: SHAP clustermap
- `figures/shap_top20_<target>.png`: Top-20 SHAP features per target
- `figures/shap_beeswarm_<target>.png`: SHAP beeswarm per target

## Logs

- `<PROJECT>.log`: Run-level pipeline log
- `logging/`: TensorBoard files (if enabled)
