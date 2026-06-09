from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xgboost as xgb

from tabnado.shap import compute_shap
from tabnado.utils import (
    classification_shap_output_columns,
    encode_classification_target,
)


def test_classification_shap_output_columns_binary_uses_positive_class():
    assert classification_shap_output_columns("label", ["cold", "hot"], 1) == [
        "label_hot"
    ]


def test_xgb_shap_supports_classification_artifact(tmp_path: Path):
    feature_cols = ["ChIP_A_-100", "ChIP_A_0", "ChIP_B_-100", "ChIP_B_0"]
    rng = np.random.default_rng(42)

    train_features = rng.normal(size=(24, len(feature_cols)))
    test_features = rng.normal(size=(10, len(feature_cols)))
    train_labels = np.where(
        train_features[:, 0] + train_features[:, 1] > 0,
        "hot",
        "cold",
    )
    test_labels = np.where(
        test_features[:, 0] + test_features[:, 1] > 0,
        "hot",
        "cold",
    )

    train_data = pd.DataFrame(train_features, columns=feature_cols)
    train_data["label"] = train_labels
    test_data = pd.DataFrame(test_features, columns=feature_cols)
    test_data["label"] = test_labels

    encoded = encode_classification_target(train_data, ["label"])
    model = xgb.XGBClassifier(
        n_estimators=8,
        max_depth=2,
        learning_rate=0.3,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=42,
        verbosity=0,
    )
    model.fit(train_data[feature_cols].values, encoded.train)

    artifact = {
        "task": "classification",
        "problem_type": encoded.problem_type,
        "target_col": encoded.target_col,
        "classes": encoded.classes,
        "model": model,
    }

    res_dir = tmp_path / "results"
    fig_dir = tmp_path / "figures"
    compute_shap(
        "xgboost",
        artifact,
        train_data,
        test_data,
        feature_cols,
        ["label"],
        RES_DIR=str(res_dir),
        FIG_DIR=str(fig_dir),
        task="classification",
    )

    mean_abs = pd.read_csv(res_dir / "shap" / "shap_mean_abs.csv", index_col=0)
    assert list(mean_abs.index) == feature_cols
    assert "label" not in mean_abs.columns
    assert all(col.startswith("label_") for col in mean_abs.columns)
    assert (fig_dir / "shap_clustermap.png").exists()
    assert (fig_dir / "shap_stacked_bar.png").exists()
    assert (res_dir / "shap" / "shap_stacked_bar_data.csv").exists()


def test_catboost_shap_supports_classification_artifact(
    monkeypatch,
    tmp_path: Path,
):
    import tabnado.shap as tabnado_shap

    class FakeModel:
        pass

    observed = {}

    class FakeExplainer:
        def __init__(self, model):
            observed["model"] = model

        def shap_values(self, regions, check_additivity=False):
            observed["regions_rows"] = len(regions)
            observed["regions_cols"] = list(regions.columns)
            observed["check_additivity"] = check_additivity
            n_samples, n_features = regions.shape
            values = np.arange(n_samples * n_features, dtype=float)
            return values.reshape(n_samples, n_features)

    monkeypatch.setattr(tabnado_shap.shap_pkg, "TreeExplainer", FakeExplainer)

    feature_cols = ["ChIP_A_-100", "ChIP_A_0", "ChIP_B_-100", "ChIP_B_0"]
    n_rows = 1005
    train_data = pd.DataFrame(
        np.arange(n_rows * len(feature_cols)).reshape(n_rows, len(feature_cols)),
        columns=feature_cols,
    )
    train_data["label"] = ["cold", "hot"] * (n_rows // 2) + ["cold"] * (n_rows % 2)
    eval_data = train_data.iloc[:2].copy()
    test_data = train_data.iloc[:3].copy()

    res_dir = tmp_path / "results"
    fig_dir = tmp_path / "figures"
    compute_shap(
        "catboost",
        {
            "task": "classification",
            "target_col": "label",
            "classes": ["cold", "hot"],
            "model": FakeModel(),
        },
        train_data,
        test_data,
        feature_cols,
        ["label"],
        eval_data=eval_data,
        RES_DIR=str(res_dir),
        FIG_DIR=str(fig_dir),
        task="classification",
    )

    mean_abs = pd.read_csv(res_dir / "shap" / "shap_mean_abs.csv", index_col=0)
    assert isinstance(observed["model"], FakeModel)
    assert observed["regions_rows"] == len(train_data)
    assert observed["regions_cols"] == feature_cols
    assert observed["check_additivity"] is False
    assert list(mean_abs.columns) == ["label_hot"]
    assert list(mean_abs.index) == feature_cols
    assert (fig_dir / "shap_clustermap.png").exists()
    assert (fig_dir / "shap_stacked_bar.png").exists()
    assert (res_dir / "shap" / "shap_stacked_bar_data.csv").exists()
    assert (res_dir / "shap" / "spatial_shap_by_offset_label_hot.csv").exists()


def test_catboost_multiclass_shap_uses_tree_explainer_on_training_data(
    monkeypatch,
    tmp_path: Path,
):
    import tabnado.shap as tabnado_shap

    class FakeModel:
        pass

    observed = {}

    class FakeExplainer:
        def __init__(self, model):
            observed["model"] = model

        def shap_values(self, regions, check_additivity=False):
            observed["regions_rows"] = len(regions)
            observed["check_additivity"] = check_additivity
            n_samples, n_features = regions.shape
            values = np.arange(n_samples * n_features * 3, dtype=float)
            return values.reshape(n_samples, n_features, 3)

    monkeypatch.setattr(tabnado_shap.shap_pkg, "TreeExplainer", FakeExplainer)

    feature_cols = ["ChIP_A_-100", "ChIP_A_0"]
    train_data = pd.DataFrame(
        {
            "ChIP_A_-100": [0.0, 1.0, 2.0],
            "ChIP_A_0": [0.5, 1.5, 2.5],
            "label": ["cold", "hot", "warm"],
        }
    )
    test_data = train_data.iloc[:2].copy()

    res_dir = tmp_path / "results"
    fig_dir = tmp_path / "figures"
    compute_shap(
        "catboost",
        {
            "task": "classification",
            "problem_type": "multiclass",
            "target_col": "label",
            "classes": ["cold", "hot", "warm"],
            "model": FakeModel(),
        },
        train_data,
        test_data,
        feature_cols,
        ["label"],
        RES_DIR=str(res_dir),
        FIG_DIR=str(fig_dir),
        task="classification",
    )

    mean_abs = pd.read_csv(res_dir / "shap" / "shap_mean_abs.csv", index_col=0)
    assert isinstance(observed["model"], FakeModel)
    assert observed["regions_rows"] == len(train_data)
    assert observed["check_additivity"] is False
    assert list(mean_abs.columns) == ["label_cold", "label_hot", "label_warm"]


def _patch_run_shap_dispatch(monkeypatch, tmp_path, model_type, called):
    import tabnado.api as api
    import tabnado.shap as tabnado_shap

    res_dir = tmp_path / "results"
    (res_dir / "final_model").mkdir(parents=True)
    params = SimpleNamespace(
        MODEL_TYPE=model_type,
        RES_DIR=str(res_dir),
        FIG_DIR=str(tmp_path / "figures"),
        TASK="classification",
        PROJECT="test",
        LOGGING="tensorboard",
        LOGGING_DIR=str(tmp_path / "logs"),
        TARGET="label",
    )
    feature_cols = ["feature_0"]
    train_data = pd.DataFrame({"feature_0": [0.1, 0.9], "label": ["cold", "hot"]})
    test_data = pd.DataFrame({"feature_0": [0.2, 0.8], "label": ["cold", "hot"]})

    def fake_compute_shap(dispatched_model_type, final_model, *args, **kwargs):
        called["model_type"] = dispatched_model_type
        called["task"] = kwargs["task"]

    monkeypatch.setattr(api, "load_params", lambda params_path: params)
    monkeypatch.setattr(api, "_setup_api_stage", lambda params, banner: None)
    monkeypatch.setattr(
        api,
        "load_data",
        lambda **kwargs: (
            None,
            None,
            ["label"],
            feature_cols,
            train_data,
            None,
            test_data,
        ),
    )
    monkeypatch.setattr(
        tabnado_shap, "_load_final_model", lambda model_type, res_dir: {"task": "classification"}
    )
    monkeypatch.setattr(tabnado_shap, "compute_shap", fake_compute_shap)
    return params


def test_run_shap_dispatches_xgboost_classification(monkeypatch, tmp_path: Path):
    import tabnado.api as api

    called = {}
    _patch_run_shap_dispatch(monkeypatch, tmp_path, "xgboost", called)

    api.run_shap("params.yaml")

    assert called == {"model_type": "xgboost", "task": "classification"}


def test_run_shap_dispatches_catboost(monkeypatch, tmp_path: Path):
    import tabnado.api as api

    called = {}
    _patch_run_shap_dispatch(monkeypatch, tmp_path, "catboost", called)

    api.run_shap("params.yaml")

    assert called == {"model_type": "catboost", "task": "classification"}
