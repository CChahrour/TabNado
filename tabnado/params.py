from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import os
import yaml
import logging

VALID_LOGGING_BACKENDS = {"wandb", "tensorboard"}
VALID_MODEL_TYPES = {"gandalf", "xgboost"}


@dataclass
class PipelineParams:
    # --- Required ---
    DATASET: str
    TARGET: str
    MODEL_TYPE: str
    LOGGING: str

    # --- Derived from output_dir + model + target ---
    PROJECT: str
    RES_DIR: str
    FIG_DIR: str
    LOGGING_DIR: str
    DATA_DIR: str
    WINDOWS_BED: Path

    # --- Optional with defaults ---
    SWEEP_FRACTION: float = 0.0
    N_SWEEPS: int = 0
    MIN_TARGET: float = 1.0
    MIN_FEATURES: int = 1
    WINDOW_SIZE: int = 2000
    STEP_SIZE: int = 250
    TILE_SIZE: int = 1000
    EVAL_CHR: str = "chr8"
    TEST_CHR: str = "chr9"
    CHUNK_SIZE_ROWS: int = 1_000_000
    GTF_FILE: Optional[str] = None
    ENTITY: Optional[str] = None
    EXCLUDE_IPS: list = field(default_factory=list)
    ASSAY_PREFIXES: list = field(default_factory=list)

    @classmethod
    def from_yaml(cls, params_path: Path | str) -> "PipelineParams":
        """Construct a PipelineParams by loading and validating a YAML file."""
        logging.debug(f"Loading params from: {params_path}")
        with open(params_path) as f:
            p = yaml.safe_load(f)
        logging.debug(f"Loaded params: {p}")

        dataset = p.get("dataset")
        if not dataset:
            raise ValueError("'dataset' is required but missing or empty.")

        model_name = p.get("model_name")
        if not model_name:
            raise ValueError("'model_name' is required but missing or empty.")

        logging_backend = str(p.get("logging", "wandb")).lower()
        model_type = str(model_name).lower()
        cls._validate_logging_backend(logging_backend)
        cls._validate_model_type(model_type)

        target = p.get("target")
        project = f"{model_name}_{target}"
        res_dir = f"{p['output_dir']}/{project}"
        data_dir = f"{res_dir}/data"

        windows_bed = (
            Path(p["windows_bed"])
            if "windows_bed" in p
            else Path(data_dir) / "regions.bed"
        )
        chunk_size_rows = (
            int(p["chunk_size_rows"])
            if p.get("chunk_size_rows") is not None
            else 1_000_000
        )

        return cls(
            DATASET=dataset,
            TARGET=target,
            MODEL_TYPE=model_type,
            LOGGING=logging_backend,
            PROJECT=project,
            RES_DIR=res_dir,
            FIG_DIR=f"{res_dir}/figures",
            LOGGING_DIR=f"{res_dir}/logs",
            DATA_DIR=data_dir,
            WINDOWS_BED=windows_bed,
            SWEEP_FRACTION=p.get("sweep_fraction", 0.0),
            N_SWEEPS=p.get("n_sweeps", 0),
            MIN_TARGET=p.get("min_target", 1.0),
            MIN_FEATURES=p.get("min_features", 1),
            WINDOW_SIZE=p.get("window_size", 2000),
            STEP_SIZE=p.get("step_size", 250),
            TILE_SIZE=p.get("tile_size", 1000),
            EVAL_CHR=p.get("eval_chr", "chr8"),
            TEST_CHR=p.get("test_chr", "chr9"),
            CHUNK_SIZE_ROWS=chunk_size_rows,
            GTF_FILE=p.get("gtf_file"),
            ENTITY=p.get("entity"),
            EXCLUDE_IPS=p.get("exclude_ips", []),
            ASSAY_PREFIXES=p.get("prefixes", []),
        )

    @staticmethod
    def _validate_logging_backend(logging_backend: str) -> None:
        if logging_backend not in VALID_LOGGING_BACKENDS:
            raise ValueError(
                f"Invalid logging backend '{logging_backend}'. Use one of {VALID_LOGGING_BACKENDS}."
            )

    @staticmethod
    def _validate_model_type(model_type: str) -> None:
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type '{model_type}'. Use one of {VALID_MODEL_TYPES}."
            )

    def create_directories(self) -> None:
        """Create all output directories for this pipeline run."""
        for directory in (self.RES_DIR, self.FIG_DIR, self.LOGGING_DIR, self.DATA_DIR):
            os.makedirs(directory, exist_ok=True)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if not hasattr(self, key):
            raise KeyError(f"Key '{key}' does not exist in PipelineParams.")
        setattr(self, key, value)
