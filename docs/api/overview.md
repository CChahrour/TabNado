# API Overview

Core modules in `tabnado`:

## Data and Configuration

- `tabnado.data`: Data loading, filtering, normalisation, and train/eval/test split creation
- `tabnado.params`: `PipelineParams` dataclass; use `PipelineParams.from_yaml(path)` to load and validate a config file
- `tabnado.utils`: Logging setup, CLI argument parsing, and shared training utilities
- `tabnado.evaluate`: Test-set metrics, scatter/ROC figures, UMAP embeddings, and classification reports

## Model Backends

All sweep, training, and SHAP logic is unified in three modules that dispatch internally based on `model_type`:

- `tabnado.sweep`: Optuna hyperparameter search for GANDALF, XGBoost, and CatBoost — entry point `sweep_model(model_type, ...)`
- `tabnado.train`: Final model training with best hyperparameters — entry point `train_model(model_type, ...)`
- `tabnado.shap`: SHAP feature importance and spatial SHAP analysis — entry point `compute_shap(model_type, ...)`

## Orchestration

- `tabnado.api`: High-level pipeline functions (`run`, `data`, `sweep`, `train`, `evaluate`, `shap`) called by the CLI
- `tabnado.cli`: CLI entry points for `tabnado-run`, `tabnado-data`, `tabnado-sweep`, `tabnado-train`, `tabnado-evaluate`, `tabnado-shap`, and `tabnado-init`
- `tabnado.wandb`: W&B run management (`WandbConfig`, `create_eval_report`)
