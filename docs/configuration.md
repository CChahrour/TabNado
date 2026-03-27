# Configuration

Parameters are loaded from a YAML file passed via `--params` / `-p`.

## Required Keys

- `target`: Target IP name (for example `MLLN`)
- `model_name`: Model label in output project naming
- `sweep_fraction`: Fraction of training rows used during sweep
- `gtf_file`: GTF path
- `eval_chr`: Evaluation chromosome
- `test_chr`: Test chromosome
- `output_dir`: Base output root

If any required key is missing, runtime will fail while loading params.

## Common Keys

- `n_sweeps`: Number of sweep runs
- `logging`: `wandb` or `tensorboard` — wandb is supported by both backends; tensorboard is GANDALF only

## Genomic/Data Keys

- `dataset`: QuantNado dataset path
- `gtf_file`: GTF annotation file path
- `windows_bed`: BED file of TSS windows — **optional**; auto-generated from GTF if omitted
- `eval_chr`: Evaluation chromosome
- `test_chr`: Test chromosome
- `min_target`: Minimum target sample count
- `min_features`: Minimum feature IP count
- `exclude_ips`: List of IP names to exclude from modelling
- `prefixes`: List of assay prefixes to include from sample names
- `window_size`: Genomic window size around TSS in bp
- `step_size`: Step size for sliding window in bp
- `tile_size`: Size of each tile in bp; set larger than `step_size` for overlapping tiles
- `chunk_size_rows`: **optional** — limit the number of rows processed per chunk during signal extraction; useful for memory-constrained environments. Defaults to the full dataset if omitted.

## Sample Naming Convention

Sample names in the QuantNado dataset must follow a specific format to be parsed correctly:

```
{ASSAY}-{CELL_TYPE}_{IP}_{REPLICATE}
```

**Assay types (required)**: Only samples with prefixes listed in `prefixes` are included. Defaults are `CAT`, `ChIP`, and `CM`.  
**Cell type**: Multi-part names can use dashes within the cell type (trailing digits are stripped).  
**IP**: Immunoprecipitation target name (e.g., `MLLN`, `H3K4me3`, etc.).  
**Replicate**: Numeric replicate identifier.

**Example valid names**:
- `ChIP-K562_MLLN_1` — ChIP assay, K562 cells, MLLN IP, rep 1
- `CAT-hESC-BMP4_H3K27ac_2` — CAT assay, hESC-BMP4 cells, H3K27ac IP, rep 2
- `CM-THP1_CEBPD_3` — CUT&RUN (CM) assay, THP1 cells, CEBPD IP, rep 3

IPs listed in `exclude_ips` (default: `["AF4C", "MLLC"]`) will be filtered out.

## Derived Runtime Paths

At runtime the code computes:

- `PROJECT = <model_name>_<target>`
- `RES_DIR = <output_dir>/<PROJECT>`
- `FIG_DIR = <RES_DIR>/figures`
- `LOGGING_DIR = <RES_DIR>/logs`
- `DATA_DIR = <RES_DIR>/data`
- `EVAL_DIR = <RES_DIR>/evaluate`