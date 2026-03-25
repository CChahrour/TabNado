# Pipeline Overview

## Stages

### Data stage
- Loads dataset store
- Filters samples into targets and features
- Builds cached train/eval/test parquet tables

### Sweep stage
- Runs local or W&B sweeps
- Optimizes model hyperparameters
- Saves `best_hyperparameters.json`

### Train stage
- Trains final model with best hyperparameters
- Saves model and checkpoints

### Evaluate stage
- Computes regression metrics
- Saves scatter and UMAP outputs
- Runs SHAP analysis and plots

## Logging

- Per-run text log: `<RES_DIR>/<PROJECT>.log`
- TensorBoard logging directory: `<RES_DIR>/logging` when `logging: tensorboard`
- W&B local directory: `<RES_DIR>` when `logging: wandb`
