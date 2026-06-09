# Configuration

Parameters are loaded from a YAML file passed via `--params` / `-p`.

## Required Keys

These four keys must be present — the pipeline will fail immediately if any is missing:

- `dataset`: Path to the QuantNado xarray dataset
- `model_name`: `gandalf`, `xgboost`, or `catboost` (case-insensitive)
- `target`: Target IP name (e.g. `MLLN`)
- `output_dir`: Base output root

## Optional Keys

All other keys have defaults and can be omitted.

### Model and task

| Key | Default | Values | Description |
|-----|---------|--------|-------------|
| `logging` | `wandb` | `wandb`, `tensorboard` | Experiment logging backend |
| `task` | `auto` | `auto`, `classification`, `regression` | `auto` infers from the target column dtype |
| `n_sweeps` | `0` | integer | Number of Optuna trials; `0` skips the sweep and uses built-in defaults |
| `sweep_fraction` | `0.0` | float 0–1 | Fraction of training rows used per sweep trial |
| `early_stopping` | `10` | integer | Early stopping rounds (XGBoost and CatBoost) |
| `class_balance` | `none` | `none`, `undersample`, `oversample`, `smote` | Resampling strategy applied to training data before fitting |
| `catboost_search_space` | `extended` | `extended`, `notebook` | CatBoost Optuna search space; `notebook` is smaller and faster |
| `entity` | — | string | W&B entity (team or username); defaults to your W&B account default |

### Genomic/Data Keys

| Key | Default | Description |
|-----|---------|-------------|
| `eval_chr` | `chr8` | Validation chromosome(s); can be blank, a string, or a list |
| `test_chr` | `chr9` | Test chromosome(s); same format as `eval_chr` |
| `gtf_file` | — | GTF annotation path; required if `windows_bed` is not provided |
| `windows_bed` | auto | BED file of TSS windows; auto-generated from GTF if omitted |
| `window_size` | `2000` | Genomic window size around TSS in bp |
| `step_size` | `250` | Sliding window step size in bp |
| `tile_size` | `1000` | Tile size in bp; set larger than `step_size` for overlapping tiles |
| `min_target` | `1.0` | Minimum target sample count |
| `min_features` | `1` | Minimum feature IP count |
| `prefixes` | `[]` | Assay prefixes to include (e.g. `[ChIP, CAT, CM]`); empty = all |
| `exclude_ips` | `[]` | IP names to exclude from modelling |
| `chunk_size_rows` | `1000000` | Rows per chunk during signal extraction; reduce if memory-constrained |

## Sample Naming Convention

Sample names in the QuantNado dataset must follow a specific format to be parsed correctly:

```
{ASSAY}-{CELL_TYPE}_{IP}_{REPLICATE}
```

**Assay types**: Only samples with prefixes listed in `prefixes` are included. Defaults are `CAT`, `ChIP`, and `CM`.
**Cell type**: Multi-part names can use dashes within the cell type (trailing digits are stripped).
**IP**: Immunoprecipitation target name (e.g. `MLLN`, `H3K4me3`).
**Replicate**: Numeric replicate identifier.

**Example valid names**:
- `ChIP-K562_MLLN_1` — ChIP assay, K562 cells, MLLN IP, rep 1
- `CAT-hESC-BMP4_H3K27ac_2` — CAT assay, hESC-BMP4 cells, H3K27ac IP, rep 2
- `CM-THP1_CEBPD_3` — CUT&RUN (CM) assay, THP1 cells, CEBPD IP, rep 3

IPs listed in `exclude_ips` (default: `["AF4C", "MLLC"]`) will be filtered out.

## Derived Runtime Paths

At runtime the following paths are computed from `output_dir`, `model_name`, and `target`:

- `PROJECT = <model_name>_<target>`
- `RES_DIR = <output_dir>/<PROJECT>`
- `FIG_DIR = <RES_DIR>/figures`
- `LOGGING_DIR = <RES_DIR>/logs`
- `DATA_DIR = <RES_DIR>/dataset`
- `EVAL_DIR = <RES_DIR>/evaluate`
