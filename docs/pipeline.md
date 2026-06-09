# Pipeline Overview

## Stages

### Data stage
- Loads the QuantNado xarray dataset
- Filters samples into targets and features based on `prefixes`, `exclude_ips`, `min_target`, `min_features`
- Tiles TSS windows (or loads a user-supplied BED file)
- Builds cached train/eval/test parquet tables split by chromosome

### Sweep stage
- Runs an Optuna hyperparameter search using the configured backend
- Each trial trains on a fraction of the training data (`sweep_fraction`) and scores on eval/test
- Saves `best_hyperparameters.json`; if `n_sweeps: 0` the sweep is skipped and built-in defaults are used

### Train stage
- Trains the final model with best hyperparameters on the full training set
- If `eval_chr` is blank, a validation split is derived automatically from training data
- Saves model and checkpoints to `final_model/`

### Evaluate stage
- Runs predictions on the test chromosome
- Computes and saves metrics, scatter plots (regression) or ROC/confusion matrix (classification)
- Generates UMAP embeddings

### SHAP stage
- Computes SHAP values for all features using the trained model
- Produces per-target summary plots and spatial SHAP analysis (when offset information is available)

## Logging

| Setting | Behaviour |
|---------|-----------|
| `logging: wandb` | Runs logged to W&B; `WANDB_DIR` set to `<RES_DIR>` |
| `logging: tensorboard` | TensorBoard events written to `<RES_DIR>/logs/` |

Per-run text log is always written to `<RES_DIR>/<PROJECT>.log`.
