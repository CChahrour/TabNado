import json
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest

from tabnado.train import (
    _derive_validation_split,
    predict_xgboost,
    train_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_binary_data(n=20):
    rng = np.random.default_rng(0)
    features = rng.normal(size=(n, 4))
    labels = np.where(features[:, 0] > 0, "hot", "cold")
    df = pd.DataFrame(features, columns=["f0", "f1", "f2", "f3"])
    df["label"] = labels
    return df


def _make_regression_data(n=20):
    rng = np.random.default_rng(0)
    features = rng.normal(size=(n, 4))
    df = pd.DataFrame(features, columns=["f0", "f1", "f2", "f3"])
    df["target"] = rng.normal(size=n)
    return df


FEATURE_COLS = ["f0", "f1", "f2", "f3"]


# ---------------------------------------------------------------------------
# _derive_validation_split
# ---------------------------------------------------------------------------


def test_derive_split_raises_on_too_few_rows():
    df = pd.DataFrame({"f": [1, 2, 3], "label": ["a", "b", "a"]})
    with pytest.raises(ValueError, match="at least 4"):
        _derive_validation_split(df, ["label"], "classification")


def test_derive_split_regression_produces_correct_sizes():
    df = _make_regression_data(20)
    train, val = _derive_validation_split(df, ["target"], "regression")
    assert len(train) + len(val) == 20
    assert len(val) == pytest.approx(4, abs=1)


def test_derive_split_classification_stratified():
    df = _make_binary_data(20)
    train, val = _derive_validation_split(df, ["label"], "classification")
    assert set(val["label"].unique()) == {"hot", "cold"}


def test_derive_split_single_class_falls_back_to_unstratified():
    df = pd.DataFrame({"f": range(10), "label": ["a"] * 10})
    train, val = _derive_validation_split(df, ["label"], "classification")
    assert len(train) + len(val) == 10


# ---------------------------------------------------------------------------
# predict_xgboost — classification artifact (dict)
# ---------------------------------------------------------------------------


def test_predict_xgboost_classification_uses_predict_proba():
    feature_cols = ["f0", "f1"]
    data = pd.DataFrame({"f0": [0.1, 0.9], "f1": [0.2, 0.8]})

    fake_estimator = MagicMock()
    fake_estimator.predict.return_value = np.array([0, 1])
    fake_estimator.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])

    artifact = {
        "task": "classification",
        "target_col": "label",
        "classes": ["cold", "hot"],
        "model": fake_estimator,
    }
    result = predict_xgboost(artifact, data, feature_cols, ["label"])

    fake_estimator.predict_proba.assert_called_once()
    assert "label" in result.columns


def test_predict_xgboost_regression_stacks_columns():
    feature_cols = ["f0", "f1"]
    data = pd.DataFrame({"f0": [0.1, 0.9], "f1": [0.2, 0.8]})

    m1 = MagicMock()
    m1.predict.return_value = np.array([1.0, 2.0])
    m2 = MagicMock()
    m2.predict.return_value = np.array([3.0, 4.0])

    result = predict_xgboost([m1, m2], data, feature_cols, ["t1", "t2"])
    assert list(result.columns) == ["t1", "t2"]
    assert result["t1"].tolist() == [1.0, 2.0]


# ---------------------------------------------------------------------------
# _train_xgboost_classifier — monkeypatched
# ---------------------------------------------------------------------------


class _FakeXGBClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y, eval_set=None, verbose=False):
        self._n_train = len(X)

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)) * 0.6, np.ones(len(X)) * 0.4])

    def evals_result(self):
        return {"validation_0": {"rmse": [0.5]}, "validation_1": {"rmse": [0.6]}}

    best_iteration = 0


class _FakeXGBRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y, eval_set=None, verbose=False):
        pass

    def predict(self, X):
        return np.zeros(len(X))

    def evals_result(self):
        return {"validation_0": {"rmse": [0.5]}, "validation_1": {"rmse": [0.6]}}

    best_iteration = 0


def test_train_xgboost_classifier_saves_model_and_metrics(monkeypatch, tmp_path):
    import sys

    class FakeXGB:
        XGBClassifier = _FakeXGBClassifier
        XGBRegressor = _FakeXGBRegressor

    original = sys.modules.get("xgboost")
    sys.modules["xgboost"] = FakeXGB()  # type: ignore[assignment]

    from tabnado.train import _train_xgboost_classifier

    train = _make_binary_data(16)
    eval_ = _make_binary_data(8)
    best_hp = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 10}

    artifact = _train_xgboost_classifier(
        best_hp=best_hp,
        feature_cols=FEATURE_COLS,
        target_cols=["label"],
        train_data=train,
        eval_data=eval_,
        RES_DIR=str(tmp_path),
        early_stopping_rounds=5,
        wandb_cfg=None,
    )

    sys.modules["xgboost"] = original  # type: ignore[assignment]

    assert artifact["task"] == "classification"
    assert artifact["target_col"] == "label"
    assert (tmp_path / "final_model" / "xgboost_model.joblib").exists()
    assert (tmp_path / "final_model" / "eval_metrics.json").exists()


def test_train_xgboost_classifier_calls_wandb_init_and_finish(monkeypatch, tmp_path):
    import sys

    class FakeXGB:
        XGBClassifier = _FakeXGBClassifier
        XGBRegressor = _FakeXGBRegressor

    original = sys.modules.get("xgboost")
    sys.modules["xgboost"] = FakeXGB()  # type: ignore[assignment]

    from tabnado.train import _train_xgboost_classifier

    mock_run = MagicMock()
    mock_wandb_cfg = MagicMock()
    mock_wandb_cfg.init_run.return_value = mock_run
    mock_wandb_cfg.model_name = "XGBoost"

    train = _make_binary_data(16)
    eval_ = _make_binary_data(8)

    _train_xgboost_classifier(
        best_hp={},
        feature_cols=FEATURE_COLS,
        target_cols=["label"],
        train_data=train,
        eval_data=eval_,
        RES_DIR=str(tmp_path),
        early_stopping_rounds=5,
        wandb_cfg=mock_wandb_cfg,
    )

    sys.modules["xgboost"] = original  # type: ignore[assignment]

    mock_wandb_cfg.init_run.assert_called_once()
    mock_run.log.assert_called_once()
    mock_run.finish.assert_called_once()


# ---------------------------------------------------------------------------
# _train_catboost_classifier — monkeypatched via _import_catboost
# ---------------------------------------------------------------------------


class _FakeCatBoostClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        pass

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return np.zeros(n, dtype=int)

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return np.column_stack([np.ones(n) * 0.6, np.ones(n) * 0.4])


class _FakeCatBoostRegressor:
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        pass

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return np.zeros(n)


def test_train_catboost_classifier_saves_model_and_metrics(monkeypatch, tmp_path):
    import tabnado.train as train_module

    monkeypatch.setattr(
        train_module,
        "_import_catboost",
        lambda: (_FakeCatBoostClassifier, _FakeCatBoostRegressor),
    )

    from tabnado.train import _train_catboost_classifier

    train = _make_binary_data(16)
    eval_ = _make_binary_data(8)

    artifact = _train_catboost_classifier(
        best_hp={},
        feature_cols=FEATURE_COLS,
        target_cols=["label"],
        train_data=train,
        eval_data=eval_,
        RES_DIR=str(tmp_path),
        early_stopping_rounds=5,
        wandb_cfg=None,
    )

    assert artifact["task"] == "classification"
    assert (tmp_path / "final_model" / "catboost_model.joblib").exists()
    metrics = json.loads((tmp_path / "final_model" / "eval_metrics.json").read_text())
    assert "accuracy" in metrics


# ---------------------------------------------------------------------------
# load_xgboost_model / load_catboost_model
# ---------------------------------------------------------------------------


def test_load_xgboost_model_round_trip(tmp_path):
    model_dir = tmp_path / "final_model"
    model_dir.mkdir()
    fake = {"task": "classification", "model": "fake"}
    joblib.dump(fake, model_dir / "xgboost_model.joblib")

    import sys
    sys.modules.setdefault("xgboost", MagicMock())

    from tabnado.train import load_xgboost_model

    loaded = load_xgboost_model(str(tmp_path))
    assert loaded["task"] == "classification"


def test_load_catboost_model_round_trip(tmp_path):
    model_dir = tmp_path / "final_model"
    model_dir.mkdir()
    fake = [{"model": "fake_regressor"}]
    joblib.dump(fake, model_dir / "catboost_model.joblib")

    from tabnado.train import load_catboost_model

    loaded = load_catboost_model(str(tmp_path))
    assert loaded == fake


# ---------------------------------------------------------------------------
# train_model dispatcher
# ---------------------------------------------------------------------------


def test_train_model_routes_xgboost():
    import tabnado.train as train_module

    called = {}
    original = train_module._TRAIN_BACKENDS["xgboost"]

    def fake_xgboost(best_hp, feature_cols, target_cols, train_data, eval_data, **kwargs):
        called["backend"] = "xgboost"
        return "xgb_model"

    train_module._TRAIN_BACKENDS["xgboost"] = fake_xgboost
    try:
        df = _make_binary_data(10)
        train_model("xgboost", {}, FEATURE_COLS, ["label"], df, df.copy(), TASK="classification")
    finally:
        train_module._TRAIN_BACKENDS["xgboost"] = original

    assert called["backend"] == "xgboost"


def test_train_model_unknown_type_falls_back_to_gandalf(monkeypatch):
    import tabnado.train as train_module

    called = {}

    def fake_gandalf(best_hp, feature_cols, target_cols, train_data, eval_data, **kwargs):
        called["backend"] = "gandalf"
        return "gandalf_model"

    # train_model uses `_train_gandalf` as the .get() default, so patch the name
    monkeypatch.setattr(train_module, "_train_gandalf", fake_gandalf)

    df = _make_regression_data(10)
    train_model("unknown_backend", {}, FEATURE_COLS, ["target"], df, df.copy(), TASK="regression")

    assert called.get("backend") == "gandalf"


def test_train_model_derives_split_when_eval_empty(monkeypatch):
    import tabnado.train as train_module

    splits_made = []
    original_split = train_module._derive_validation_split

    def tracking_split(train_data, target_cols, task, seed=42):
        splits_made.append(True)
        return original_split(train_data, target_cols, task, seed)

    monkeypatch.setattr(train_module, "_derive_validation_split", tracking_split)

    called = {}
    original_xgb = train_module._TRAIN_BACKENDS["xgboost"]

    def fake_xgboost(best_hp, feature_cols, target_cols, train_data, eval_data, **kwargs):
        called["eval_len"] = len(eval_data)
        return "model"

    train_module._TRAIN_BACKENDS["xgboost"] = fake_xgboost
    try:
        df = _make_binary_data(20)
        empty_eval = pd.DataFrame()
        train_model("xgboost", {}, FEATURE_COLS, ["label"], df, empty_eval, TASK="classification")
    finally:
        train_module._TRAIN_BACKENDS["xgboost"] = original_xgb

    assert splits_made, "Expected _derive_validation_split to be called"
    assert called["eval_len"] > 0
