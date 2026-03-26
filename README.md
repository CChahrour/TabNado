# TabNado
[![Docs](https://img.shields.io/badge/docs-cchahrour.github.io-blue)](https://cchahrour.github.io/TabNado/)
[![CI](https://github.com/CChahrour/TabNado/actions/workflows/python-tests.yml/badge.svg)](https://github.com/CChahrour/TabNado/actions/workflows/python-tests.yml)

<p align="center">
  <img src="docs/assets/tabnado.png" alt="TabNado logo" width="192">
</p>


Predicts binding from epigenomic cofactors (ChIP-seq, CUT&TAG, CUT&RUN) over tiled TSS windows or user-defined BED regions, with support for both GANDALF (neural tabular) and XGBoost backends.

Uses datasets prepared with [QuantNado](https://github.com/Milne-Group/QuantNado) via [SeqNado](https://github.com/Milne-Group/SeqNado).

## Overview

1. **Data** — signal is extracted from an xarray dataset, normalised to RPKM, log1p + MinMax scaled, and split by chromosome (chr8 = eval, chr9 = test).
2. **Sweep** — backend-specific hyperparameter sweep.
3. **Train** — final model trained with best HP on full training set.
4. **Evaluate** — test-set metrics and UMAP embeddings.
5. **SHAP** — feature importance and spatial SHAP analysis.

Backend selection is controlled by `model_name`:

- `GANDALF` for the GANDALF neural tabular backend
- `XGBoost` for the XGBoost backend

## Setup

```bash
apptainer pull tabnado.sif docker://ghcr.io/cchahrour/tabnado:latest
```

## Configuration

All parameters are set in `params.yaml`.

You can generate a starter template:

```bash
tabnado-init
# or write to a custom location
tabnado-init experiments/params_MLLN.yaml
```

Required keys:

- `target`
- `model_name` (`gandalf` or `xgboost`)
- `sweep_fraction`
- `gtf_file`
- `eval_chr`
- `test_chr`
- `output_dir`
- `dataset`
- `n_sweeps`
- `logging` (`wandb` or `tensorboard`)
- `min_target`
- `min_features`
- `prefixes`
- `window_size`
- `step_size`
- `tile_size`

Optional keys:

- `windows_bed` (auto-generated from GTF if omitted)

Example:

```yaml
target: MLLN
model_name: GANDALF
sweep_fraction: 0.2
gtf_file: data/regions/gencode.v49.annotation.gtf
eval_chr: chr8
test_chr: chr9
output_dir: results
windows_bed: data/tss_windows.bed
dataset: data/dataset
n_sweeps: 10
logging: wandb
```

Pass a custom config with `--params` / `-p` to run different experiments without editing the file.

## Usage

### Full pipeline

```bash
tabnado-run --params params.yaml
```

### Individual stages

```bash
tabnado-data     --params params.yaml   # build / validate train/eval/test splits
tabnado-sweep    --params params.yaml   # HP sweep → results/<project>/best_hyperparameters.json
tabnado-train    --params params.yaml   # train final model → results/<project>/final_model/
tabnado-evaluate --params params.yaml   # metrics + UMAP → results/<project>/figures/
tabnado-shap     --params params.yaml   # SHAP analysis → results/<project>/shap/
```

### SLURM

```bash
#!/bin/bash
#SBATCH --job-name=tabnado
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

set -euo pipefail

DATA_DIR=/path/to/dataset

apptainer exec \
    --bind "$PWD:$PWD" \
    --bind "$DATA_DIR:$DATA_DIR" \
    --pwd "$PWD" \
    tabnado.sif \
    tabnado-run --params params_MLLN.yaml
```

For multiple experiments, use a SLURM array:

```bash
#!/bin/bash
#SBATCH --job-name=tabnado
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log
#SBATCH --array=0-3

set -euo pipefail

DATA_DIR=/path/to/dataset

apptainer exec \
    --bind "$PWD:$PWD" \
    --bind "$DATA_DIR:$DATA_DIR" \
    --pwd "$PWD" \
    tabnado.sif \
    tabnado-run --params experiments/params_${SLURM_ARRAY_TASK_ID}.yaml
```

## Outputs

Results are written to `results/<MODEL_NAME>_<TARGET>_<date>/`:

| Path | Contents |
|------|----------|
| `dataset/` | Processed dataset |
| `best_hyperparameters.json` | Best hyperparameters from sweep |
| `final_model/` | Saved backend model (for example `xgboost_model.joblib` for XGBoost) |
| `figures/scatter_test.png` | True vs predicted scatter |
| `figures/embeddings_umap.png` | UMAP embedding output |
| `figures/shap_clustermap.png` | Mean \|SHAP\| heatmap across cofactors and targets |
| `shap/shap_mean_abs.csv` | Mean absolute SHAP by feature and target |
| `shap/spatial_shap_by_offset_<target>.csv` | Spatial SHAP summary by genomic offset per target |
| `figures/shap_spatial_heatmap_<target>.png` | Spatial SHAP heatmap (cofactor × offset) per target |
| `figures/shap_offset_line_<target>.png` | SHAP importance profile by genomic offset per target |

## Required files

To run the model you need:

```
<working_dir>/
└── params.yaml           # experiment config (edit this)
```

Processed dataset is written to `<data_dir>`
Results are written to `results/<MODEL_NAME>_<TARGET>_<date>/` relative to wherever you run the command.
