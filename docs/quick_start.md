# Quick Start

## 1. Prepare Parameters

Create or edit a params file:

```bash
tabnado-init
```

```yaml
# tabnado parameters
# Required keys
target: MLLN
model_name: GANDALF
sweep_fraction: 0.2
gtf_file: data/gencode.vM25.annotation.gtf.gz
eval_chr: chr8
test_chr: chr9
output_dir: results

# Optional keys
dataset: data/dataset
# windows_bed: data/tss_windows.bed
n_sweeps: 10
logging: wandb
min_target: 1
min_features: 10
exclude_ips: ["AF4C", "MLLC"]
prefixes: ["CAT", "ChIP", "CM"]
window_size: 3000
step_size: 100
```

## 2. Run Full Pipeline

```bash
tabnado-run --params params.yaml
```

## 3. Check Outputs

Look under:

```text
results/<MODEL_NAME>_<TARGET>_<date>/
```

You should see:

- `best_hyperparameters.json`
- `final_model/`
- `figures/scatter_test.png`
- `figures/embeddings_umap.png`
- `shap_mean_abs.csv`
- `figures/shap_*.png`

## 4. Run on Slurm with a Container

Pull the image once on the cluster login node:

```bash
apptainer pull tabnado.sif docker://ghcr.io/cchahrour/tabnado:latest
```

If your cluster uses `singularity` instead of `apptainer`, use the same commands with `singularity`.

Note on `quantnado`:

- If `data_dir` already contains cached parquet splits (`dataset_train.parquet`, `dataset_eval.parquet`, `dataset_test.parquet`), the pipeline can run from those cached files.
- If those parquet files do not exist, `tabnado-data` needs `quantnado` available to build them from the raw dataset store.

Example single-job submission script (`run_tabnado.sbatch`):

```bash
#!/bin/bash
#SBATCH --job-name=tabnado
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd /path/to/tabnado
DATA_DIR=/path/to/dataset
apptainer exec \
	--bind "$PWD:$PWD" \
	--bind "$DATA_DIR:$DATA_DIR" \
	--pwd "$PWD" \
	tabnado.sif \
	tabnado-run --params params.yaml
```

Submit with:

```bash
sbatch run_tabnado.sbatch
```

Example array job for multiple parameter files:

```bash
#!/bin/bash
#SBATCH --job-name=tabnado-array
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

cd /path/to/tabnado
DATA_DIR=/path/to/dataset
apptainer exec \
	--bind "$PWD:$PWD" \
	--bind "$DATA_DIR:$DATA_DIR" \
	--pwd "$PWD" \
	tabnado.sif \
	tabnado-run --params experiments/params_${SLURM_ARRAY_TASK_ID}.yaml
```

