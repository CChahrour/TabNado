import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import tabnado.api as api

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_params(
    tmp_path, logging="tensorboard", model_type="xgboost", task="classification"
):
    return SimpleNamespace(
        PROJECT="test",
        LOGGING=logging,
        MODEL_TYPE=model_type,
        TASK=task,
        TARGET="label",
        RES_DIR=str(tmp_path / "results"),
        FIG_DIR=str(tmp_path / "figures"),
        LOGGING_DIR=str(tmp_path / "logs"),
        DATA_DIR=str(tmp_path / "data"),
        N_SWEEPS=2,
        SWEEP_FRACTION=0.5,
        CATBOOST_SEARCH_SPACE="extended",
        EARLY_STOPPING_ROUNDS=10,
        CLASS_BALANCE="none",
        ENTITY=None,
    )


def _small_df():
    return pd.DataFrame({"f0": [0.1, 0.9, 0.5], "label": ["a", "b", "a"]})


# ---------------------------------------------------------------------------
# write_params_template
# ---------------------------------------------------------------------------


def test_write_params_template_creates_file(tmp_path):
    out = tmp_path / "params.yaml"
    result = api.write_params_template(out)
    assert result == out
    assert out.exists()
    text = out.read_text()
    assert "target: TARGET_NAME" in text


def test_write_params_template_raises_if_exists_no_force(tmp_path):
    out = tmp_path / "params.yaml"
    out.write_text("old content")
    with pytest.raises(FileExistsError):
        api.write_params_template(out, force=False)
    assert out.read_text() == "old content"


def test_write_params_template_force_overwrites(tmp_path):
    out = tmp_path / "params.yaml"
    out.write_text("old content")
    api.write_params_template(out, force=True)
    assert "target: TARGET_NAME" in out.read_text()


def test_write_params_template_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "deep" / "params.yaml"
    api.write_params_template(out)
    assert out.exists()


# ---------------------------------------------------------------------------
# _setup_api_stage env vars
# ---------------------------------------------------------------------------


def test_setup_api_stage_sets_wandb_env(monkeypatch, tmp_path):
    import os

    params = _fake_params(tmp_path, logging="wandb")
    (tmp_path / "results").mkdir(parents=True)

    monkeypatch.setattr(api, "setup_logger", lambda *a, **k: None)
    monkeypatch.delenv("WANDB_DIR", raising=False)
    monkeypatch.delenv("TENSORBOARD_DIR", raising=False)

    # create_directories may fail without a full PipelineParams, so mock it
    params.create_directories = lambda: None

    api._setup_api_stage(params, "TEST")
    assert os.environ.get("WANDB_DIR") == params.RES_DIR


def test_setup_api_stage_sets_tensorboard_env(monkeypatch, tmp_path):
    import os

    params = _fake_params(tmp_path, logging="tensorboard")
    params.create_directories = lambda: None
    monkeypatch.setattr(api, "setup_logger", lambda *a, **k: None)
    monkeypatch.delenv("WANDB_DIR", raising=False)
    monkeypatch.delenv("TENSORBOARD_DIR", raising=False)

    api._setup_api_stage(params, "TEST")
    assert os.environ.get("TENSORBOARD_DIR") == params.LOGGING_DIR


def test_setup_api_stage_sets_no_env_for_none_logging(monkeypatch, tmp_path):
    import os

    params = _fake_params(tmp_path, logging="none")
    params.create_directories = lambda: None
    monkeypatch.setattr(api, "setup_logger", lambda *a, **k: None)
    monkeypatch.delenv("WANDB_DIR", raising=False)
    monkeypatch.delenv("TENSORBOARD_DIR", raising=False)

    api._setup_api_stage(params, "TEST")
    assert "WANDB_DIR" not in os.environ
    assert "TENSORBOARD_DIR" not in os.environ


# ---------------------------------------------------------------------------
# _make_wandb_config
# ---------------------------------------------------------------------------


def test_make_wandb_config_returns_none_for_non_wandb(tmp_path):
    params = _fake_params(tmp_path, logging="tensorboard")
    assert api._make_wandb_config(params) is None


def test_make_wandb_config_returns_config_for_wandb(monkeypatch, tmp_path):
    params = _fake_params(tmp_path, logging="wandb")

    fake_cfg = object()
    monkeypatch.setattr(
        "tabnado.wandb.WandbConfig.from_params",
        classmethod(lambda cls, p: fake_cfg),
    )

    result = api._make_wandb_config(params)
    assert result is not None


# ---------------------------------------------------------------------------
# _balanced_train_data
# ---------------------------------------------------------------------------


def test_balanced_train_data_regression_returns_unchanged(tmp_path):
    params = _fake_params(tmp_path)
    df = _small_df()
    result = api._balanced_train_data(params, "regression", df, ["label"])
    assert result is df


def test_balanced_train_data_class_balance_none_returns_unchanged(tmp_path):
    params = _fake_params(tmp_path)
    params.CLASS_BALANCE = "none"
    df = _small_df()
    result = api._balanced_train_data(params, "classification", df, ["label"])
    assert result is df


def test_balanced_train_data_calls_balance_classes(monkeypatch, tmp_path):
    import tabnado.api as api_mod

    params = _fake_params(tmp_path)
    params.CLASS_BALANCE = "undersample"
    df = _small_df()
    balanced = df.copy()
    called = {}

    def fake_balance(data, target_col, method):
        called["method"] = method
        return balanced

    monkeypatch.setattr("tabnado.data.balance_classes", fake_balance)

    result = api_mod._balanced_train_data(params, "classification", df, ["label"])
    assert called.get("method") == "undersample"
    assert result is balanced


# ---------------------------------------------------------------------------
# _load_best_hyperparameters
# ---------------------------------------------------------------------------


def test_load_best_hp_returns_dict(tmp_path):
    params = _fake_params(tmp_path)
    res_dir = Path(params.RES_DIR)
    res_dir.mkdir(parents=True)
    hp = {"max_depth": 4, "learning_rate": 0.1}
    (res_dir / "best_hyperparameters.json").write_text(json.dumps(hp))

    result = api._load_best_hyperparameters(params)
    assert result == hp


def test_load_best_hp_raises_if_missing(tmp_path):
    params = _fake_params(tmp_path)
    Path(params.RES_DIR).mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        api._load_best_hyperparameters(params)


# ---------------------------------------------------------------------------
# run_evaluate — FileNotFoundError when model missing
# ---------------------------------------------------------------------------


def test_run_evaluate_raises_if_model_missing(monkeypatch, tmp_path):
    params = _fake_params(tmp_path, model_type="xgboost")

    monkeypatch.setattr(api, "load_params", lambda path: params)
    monkeypatch.setattr(api, "_setup_api_stage", lambda *a, **k: None)
    monkeypatch.setattr(
        api,
        "load_data",
        lambda **kw: (
            None,
            None,
            ["label"],
            ["f0"],
            _small_df(),
            _small_df(),
            _small_df(),
        ),
    )
    monkeypatch.setattr(api, "resolve_task", lambda task, data, cols: "classification")

    with pytest.raises(FileNotFoundError, match="Run train first"):
        api.run_evaluate("params.yaml")


# ---------------------------------------------------------------------------
# run_shap — FileNotFoundError when model missing
# ---------------------------------------------------------------------------


def test_run_shap_raises_if_model_missing(monkeypatch, tmp_path):
    params = _fake_params(tmp_path, model_type="xgboost")

    monkeypatch.setattr(api, "load_params", lambda path: params)
    monkeypatch.setattr(api, "_setup_api_stage", lambda *a, **k: None)
    monkeypatch.setattr(
        api,
        "load_data",
        lambda **kw: (
            None,
            None,
            ["label"],
            ["f0"],
            _small_df(),
            _small_df(),
            _small_df(),
        ),
    )

    with pytest.raises(FileNotFoundError, match="Run train first"):
        api.run_shap("params.yaml")


# ---------------------------------------------------------------------------
# run_evaluate — loads xgboost model when model_type is xgboost
# ---------------------------------------------------------------------------


def test_run_evaluate_loads_xgboost_joblib(monkeypatch, tmp_path):
    params = _fake_params(tmp_path, model_type="xgboost")
    model_dir = Path(params.RES_DIR) / "final_model"
    model_dir.mkdir(parents=True)

    # MagicMock isn't picklable; use a plain dict as the fake model
    fake_model = {"task": "classification", "model": "stub"}

    import joblib

    joblib.dump(fake_model, model_dir / "xgboost_model.joblib")

    monkeypatch.setattr(api, "load_params", lambda path: params)
    monkeypatch.setattr(api, "_setup_api_stage", lambda *a, **k: None)
    monkeypatch.setattr(
        api,
        "load_data",
        lambda **kw: (
            None,
            None,
            ["label"],
            ["f0"],
            _small_df(),
            _small_df(),
            _small_df(),
        ),
    )
    monkeypatch.setattr(api, "resolve_task", lambda task, data, cols: "classification")

    evaluate_calls = []
    monkeypatch.setattr(
        api, "evaluate_model", lambda *a, **k: evaluate_calls.append(a[0])
    )
    monkeypatch.setattr(api, "compute_umap_embeddings", lambda *a, **k: None)

    api.run_evaluate("params.yaml")

    assert len(evaluate_calls) == 1


# ---------------------------------------------------------------------------
# run_train — raises FileNotFoundError when best_hp missing
# ---------------------------------------------------------------------------


def test_run_train_raises_if_no_best_hp(monkeypatch, tmp_path):
    params = _fake_params(tmp_path, model_type="xgboost")
    Path(params.RES_DIR).mkdir(parents=True)

    monkeypatch.setattr(api, "load_params", lambda path: params)
    monkeypatch.setattr(api, "_setup_api_stage", lambda *a, **k: None)
    monkeypatch.setattr(api, "_make_wandb_config", lambda p: None)
    monkeypatch.setattr(
        api,
        "load_data",
        lambda **kw: (
            None,
            None,
            ["label"],
            ["f0"],
            _small_df(),
            _small_df(),
            _small_df(),
        ),
    )
    monkeypatch.setattr(api, "resolve_task", lambda task, data, cols: "classification")

    with pytest.raises(FileNotFoundError):
        api.run_train("params.yaml")
