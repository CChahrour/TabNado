from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

VALID_LOGGING_BACKENDS = {"wandb", "tensorboard"}
VALID_MODEL_TYPES = {"catboost", "gandalf", "xgboost"}
VALID_TASKS = {"auto", "classification", "regression"}
VALID_CATBOOST_SEARCH_SPACES = {"extended", "notebook"}
VALID_CLASS_BALANCE_METHODS = {"none", "undersample", "oversample", "smote"}


def _as_chr_list(value: Any) -> list[str]:
    """Normalise a YAML chromosome entry (blank, scalar, or list) to a list of names."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    value = str(value).strip()
    return [value] if value else []


@dataclass
class PipelineParams:
    # --- Required ---
    DATASET: str
    TARGET: str
    MODEL_TYPE: str
    LOGGING: str
    TASK: str

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
    EVAL_CHR: list[str] = field(default_factory=lambda: ["chr8"])
    TEST_CHR: list[str] = field(default_factory=lambda: ["chr9"])
    CHUNK_SIZE_ROWS: int = 1_000_000
    GTF_FILE: Optional[str] = None
    ENTITY: Optional[str] = None
    EXCLUDE_IPS: list = field(default_factory=list)
    ASSAY_PREFIXES: list = field(default_factory=list)
    CATBOOST_SEARCH_SPACE: str = "extended"
    CLASS_BALANCE: str = "none"
    EARLY_STOPPING_ROUNDS: int = 10
    SCALE_DATA: bool = True

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
        task = str(p.get("task", "auto")).lower()
        catboost_search_space = str(p.get("catboost_search_space", "extended")).lower()
        class_balance = str(p.get("class_balance", "none")).lower()
        cls._validate_logging_backend(logging_backend)
        cls._validate_model_type(model_type)
        cls._validate_task(task)
        cls._validate_catboost_search_space(catboost_search_space)
        cls._validate_class_balance(class_balance)

        target = p.get("target")
        project = f"{model_name}_{target}"
        res_dir = f"{p['output_dir']}/{project}"
        data_dir = f"{res_dir}/dataset"

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
            TASK=task,
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
            EVAL_CHR=_as_chr_list(p.get("eval_chr", "chr8")),
            TEST_CHR=_as_chr_list(p.get("test_chr", "chr9")),
            CHUNK_SIZE_ROWS=chunk_size_rows,
            GTF_FILE=p.get("gtf_file"),
            ENTITY=p.get("entity"),
            EXCLUDE_IPS=p.get("exclude_ips", []),
            ASSAY_PREFIXES=p.get("prefixes", []),
            CATBOOST_SEARCH_SPACE=catboost_search_space,
            CLASS_BALANCE=class_balance,
            EARLY_STOPPING_ROUNDS=int(p.get("early_stopping", 10)),
            SCALE_DATA=bool(p.get("scale_data", True)),
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

    @staticmethod
    def _validate_task(task: str) -> None:
        if task not in VALID_TASKS:
            raise ValueError(f"Invalid task '{task}'. Use one of {VALID_TASKS}.")

    @staticmethod
    def _validate_catboost_search_space(search_space: str) -> None:
        if search_space not in VALID_CATBOOST_SEARCH_SPACES:
            raise ValueError(
                f"Invalid catboost_search_space '{search_space}'. "
                f"Use one of {VALID_CATBOOST_SEARCH_SPACES}."
            )

    @staticmethod
    def _validate_class_balance(class_balance: str) -> None:
        if class_balance not in VALID_CLASS_BALANCE_METHODS:
            raise ValueError(
                f"Invalid class_balance '{class_balance}'. "
                f"Use one of {VALID_CLASS_BALANCE_METHODS}."
            )

    def create_directories(self) -> None:
        """Create all output directories for this pipeline run."""
        for directory in (self.RES_DIR, self.FIG_DIR, self.DATA_DIR):
            os.makedirs(directory, exist_ok=True)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if not hasattr(self, key):
            raise KeyError(f"Key '{key}' does not exist in PipelineParams.")
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
