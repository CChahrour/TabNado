# API Overview

Core modules in this project:

## Data and Utilities

- `tabnado.data`: Data loading and split creation
- `tabnado.params`: `PipelineParams` dataclass and `PipelineParams.from_yaml()` for loading configuration
- `tabnado.utils`: Logging setup, CLI argument parsing, and training utilities
- `tabnado.evaluate`: Metrics, scatter plots, and UMAP embeddings

## GANDALF Backend

- `tabnado.gandalf_sweep`: Bayesian hyperparameter sweep via Weights and Biases
- `tabnado.gandalf_train`: Final GANDALF model training with best hyperparameters
- `tabnado.gandalf_shap`: SHAP analysis for GANDALF models

## XGBoost Backend

- `tabnado.xgb_sweep`: XGBoost hyperparameter sweep
- `tabnado.xgb_train`: Final XGBoost model training with best hyperparameters
- `tabnado.xgb_shap`: SHAP analysis for XGBoost models

## Orchestration

- `tabnado.api`: `run_pipeline()` orchestrates all stages
- `tabnado.cli`: CLI entry points for `tabnado-run`, `tabnado-data`, `tabnado-sweep`, `tabnado-train`, `tabnado-evaluate`, `tabnado-shap`, and `tabnado-init`

## Initialization

- `tabnado.init`: template parameter YAML generation
