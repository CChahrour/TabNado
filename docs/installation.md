# Installation

## Prerequisites

- Apptainer/Singularity (for HPC/container execution)

## Container Setup (Recommended)

```bash
apptainer pull tabnado.sif docker://ghcr.io/cchahrour/tabnado:latest
```

Run any command inside the container:

```bash
apptainer exec tabnado.sif tabnado-run --help
```

If your cluster uses `singularity` instead of `apptainer`, use the same commands with `singularity`.

## Optional: Local Dev Environment (uv)

```bash
uv venv --python 3.12
uv pip install -e .
uv pip install pytorch-tabular==1.2.0 --no-deps
```

Install test dependencies when needed:

```bash
uv pip install -e ".[test]"
```

## Verify CLI Commands

```bash
tabnado-init --help
```

## Optional: Run Tests

```bash
pytest tests -vv
```
