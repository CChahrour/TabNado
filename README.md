# TabNado
[![Docs](https://img.shields.io/badge/docs-cchahrour.github.io-blue)](https://cchahrour.github.io/TabNado/)
[![CI](https://github.com/CChahrour/TabNado/actions/workflows/python-tests.yml/badge.svg)](https://github.com/CChahrour/TabNado/actions/workflows/python-tests.yml)

<p align="center">
  <img src="docs/assets/tabnado.png" alt="TabNado logo" width="192">
</p>


Predicts binding from epigenomic cofactors (ChIP-seq, CUT&TAG, CUT&RUN) over tiled TSS windows or user-defined BED regions. Supports classification and regression tasks with three model backends: GANDALF (neural tabular), XGBoost, and CatBoost.

Uses datasets prepared with [QuantNado](https://github.com/Milne-Group/QuantNado) via [SeqNado](https://github.com/Milne-Group/SeqNado).

## Overview

1. **Data** — signal is extracted from an xarray dataset, normalised to RPKM, log1p + MinMax scaled, and split by chromosome.
2. **Sweep** — Optuna hyperparameter sweep using the configured backend.
3. **Train** — final model trained with best hyperparameters on the full training set.
4. **Evaluate** — test-set metrics, figures, and UMAP embeddings.
5. **SHAP** — feature importance and spatial SHAP analysis.

Backend is controlled by `model_name`:

- `gandalf` — GANDALF neural tabular backend
- `xgboost` — XGBoost gradient-boosted trees
- `catboost` — CatBoost gradient-boosted trees

## Setup

```bash
apptainer pull tabnado.sif docker://ghcr.io/cchahrour/tabnado:latest
```

## Configuration

All parameters are set in `params.yaml`.

Generate a starter template:

```bash
tabnado-init
# or write to a custom location
tabnado-init experiments/params_MLLN.yaml
```

**Required keys** (pipeline fails without these):

- `dataset` — path to the QuantNado xarray dataset
- `model_name` — `gandalf`, `xgboost`, or `catboost`
- `target` — target IP name (e.g. `MLLN`)
- `output_dir` — base output root

**Optional keys** (all have sensible defaults):

| Key | Default | Description |
|-----|---------|-------------|
| `logging` | `wandb` | `wandb` or `tensorboard` |
| `task` | `auto` | `auto`, `classification`, or `regression` |
| `eval_chr` | `chr8` | Chromosome(s) for validation; can be blank, a string, or a list |
| `test_chr` | `chr9` | Chromosome(s) for test; same format as `eval_chr` |
| `n_sweeps` | `0` | Number of Optuna trials (0 = use defaults) |
| `sweep_fraction` | `0.0` | Fraction of training data per sweep trial |
| `early_stopping` | `10` | Early stopping rounds (XGBoost/CatBoost) |
| `gtf_file` | — | GTF path; required if `windows_bed` is not provided |
| `windows_bed` | auto | BED file of TSS windows; auto-generated from GTF if omitted |
| `window_size` | `2000` | Genomic window size around TSS in bp |
| `step_size` | `250` | Sliding window step size in bp |
| `tile_size` | `1000` | Tile size in bp |
| `min_target` | `1.0` | Minimum target sample count |
| `min_features` | `1` | Minimum feature IP count |
| `prefixes` | `[]` | Assay prefixes to include (e.g. `[ChIP, CAT]`) |
| `exclude_ips` | `[]` | IP names to exclude from modelling |
| `class_balance` | `none` | `none`, `undersample`, `oversample`, or `smote` |
| `catboost_search_space` | `extended` | `extended` or `notebook` |
| `entity` | — | W&B entity (team/username); defaults to your W&B default |
| `chunk_size_rows` | `1000000` | Rows per chunk during signal extraction |

Example:

```yaml
target: MLLN
model_name: xgboost
output_dir: results
dataset: data/dataset
gtf_file: data/regions/gencode.v49.annotation.gtf
eval_chr: chr8
test_chr: chr9
logging: wandb
n_sweeps: 100
sweep_fraction: 0.2
task: classification
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
tabnado-evaluate --params params.yaml   # metrics + UMAP → results/<project>/evaluate/ and figures/
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

Results are written to `<output_dir>/<model_name>_<target>/`:

| Path | Contents |
|------|----------|
| `best_hyperparameters.json` | Best hyperparameters from sweep |
| `final_model/` | Saved model (`xgboost_model.joblib`, `catboost_model.joblib`, or GANDALF directory) |
| `evaluate/metrics.json` | Test-set evaluation metrics |
| `evaluate/predictions.parquet` | Model predictions on test chromosome |
| `figures/scatter_test_<target>.png` | True vs predicted scatter (regression) |
| `figures/roc_curve_<target>.png` | ROC curve (classification) |
| `evaluate/roc_auc.csv` | ROC AUC summary (classification) |
| `evaluate/classification_report.csv` | Per-class precision/recall/F1 (classification) |
| `evaluate/confusion_matrix.csv` | Confusion matrix (classification) |
| `figures/embeddings_umap.png` | UMAP embedding figure |
| `shap/shap_mean_abs.csv` | Mean absolute SHAP by feature and target |
| `figures/shap_clustermap.png` | Mean \|SHAP\| heatmap across cofactors and targets |
| `figures/shap_stacked_bar.png` | Stacked mean \|SHAP\| bar by cofactor and output |
| `shap/spatial_shap_by_offset_<target>.csv` | Spatial SHAP by genomic offset per target |
| `figures/shap_spatial_heatmap_<target>.png` | Spatial SHAP heatmap per target |
| `figures/shap_offset_line_<target>.png` | SHAP importance profile by offset per target |
