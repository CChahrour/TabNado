# CLI Reference

The package exposes these commands:

- `tabnado-init`: Create a template params YAML file
- `tabnado-run`: Full pipeline
- `tabnado-data`: Data load/build stage
- `tabnado-sweep`: Hyperparameter sweep stage
- `tabnado-train`: Final training stage
- `tabnado-evaluate`: Evaluation stage (metrics, UMAP)
- `tabnado-shap`: SHAP analysis stage

## Shared Argument Pattern

All stage commands accept:

```bash
--params <path_to_yaml>
# or
-p <path_to_yaml>
```

All commands also support:

```bash
--help
--version
```

## Examples

```bash
tabnado-init
tabnado-init configs/params_MLLN.yaml
tabnado-init configs/params_MLLN.yaml --force

tabnado-sweep --params params_MLLN.yaml
tabnado-train --params params_MLLN.yaml
tabnado-evaluate --params params_MLLN.yaml
tabnado-shap --params params_MLLN.yaml
```
