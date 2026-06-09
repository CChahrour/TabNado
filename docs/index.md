# TabNado
[![Docs](https://img.shields.io/badge/docs-cchahrour.github.io-blue)](https://cchahrour.github.io/TabNado/)
[![CI](https://github.com/CChahrour/TabNado/actions/workflows/python-tests.yml/badge.svg)](https://github.com/CChahrour/TabNado/actions/workflows/python-tests.yml)


<p align="center">
	<img src="assets/tabnado.png" alt="TabNado logo" width="220" />
</p>

Predicts binding from epigenomic cofactors (ChIP-seq, CUT&TAG, CUT&RUN) over tiled TSS windows or user-defined BED regions. Supports classification and regression tasks with three model backends: GANDALF (neural tabular), XGBoost, and CatBoost.

Uses datasets prepared with [QuantNado](https://github.com/Milne-Group/QuantNado) via [SeqNado](https://github.com/Milne-Group/SeqNado).

## What It Does

1. Builds train/eval/test tabular datasets from genomic signal split by chromosome.
2. Runs Optuna hyperparameter sweeps with the configured backend (GANDALF, XGBoost, or CatBoost).
3. Trains a final model with best hyperparameters.
4. Evaluates predictions and exports metrics and figures.
5. Computes SHAP feature importance and spatial SHAP outputs.

## Pipeline at a Glance

1. **Data** — normalise signal, filter samples, split by chromosome.
2. **Sweep** — Optuna search → `best_hyperparameters.json`.
3. **Train** — final fit → `final_model/`.
4. **Evaluate** — metrics, scatter/ROC, UMAP → `evaluate/` and `figures/`.
5. **SHAP** — feature importance and spatial SHAP → `shap/` and `figures/`.

## Main Entry Points

- Initialise parameters: `tabnado-init`
- Full pipeline: `tabnado-run --params params.yaml`
- Data only: `tabnado-data --params params.yaml`
- Sweep only: `tabnado-sweep --params params.yaml`
- Train only: `tabnado-train --params params.yaml`
- Evaluate only: `tabnado-evaluate --params params.yaml`
- SHAP only: `tabnado-shap --params params.yaml`

See the navigation for detailed setup and configuration.
