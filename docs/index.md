# TabNado

<p align="center">
	<img src="assets/tabnado.png" alt="TabNado logo" width="220" />
</p>

Predicts binding from epigenomic cofactors (ChIP-seq, CUT&TAG, CUT&RUN) over tiled TSS windows or user-defined BED regions, with support for both GANDALF (neural tabular) and XGBoost backends.

Uses datasets prepared with [QuantNado](https://github.com/Milne-Group/QuantNado) via [SeqNado](https://github.com/Milne-Group/SeqNado).

## What It Does

1. Builds train/eval/test tabular datasets from genomic signal.
2. Runs hyperparameter sweeps (GANDALF or XGBoost backend).
3. Trains a final model with best hyperparameters.
4. Evaluates predictions and exports metrics/figures.
5. Computes SHAP feature importance outputs.

## Pipeline at a Glance

1. Data prep and split: chr8 for eval, chr9 for test.
2. Sweep stage: identifies best hyperparameters.
3. Train stage: produces `final_model/`.
4. Evaluate stage: metrics, scatter, UMAP artifacts.
5. SHAP stage: feature importance and spatial SHAP artifacts.

## Main Entry Points

- Initialise the parameters: `tabnado-init`
- Full pipeline: `tabnado-run --params params.yaml`
- Data only: `tabnado-data --params params.yaml`
- Sweep only: `tabnado-sweep --params params.yaml`
- Train only: `tabnado-train --params params.yaml`
- Evaluate only: `tabnado-evaluate --params params.yaml`
- SHAP only: `tabnado-shap --params params.yaml`

See the navigation for detailed setup and configuration.
