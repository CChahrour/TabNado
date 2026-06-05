from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tabnado.catboost_shap import compute_catboost_shap


def test_catboost_shap_supports_classification_artifact(
    monkeypatch,
    tmp_path: Path,
):
    import tabnado.catboost_shap as catboost_shap

    class FakeModel:
        pass

    observed = {}

    class FakeExplanation:
        def __init__(self, values):
            self.values = values

    class FakeExplainer:
        def __init__(self, model, background):
            observed["model"] = model
            observed["background_rows"] = len(background)
            observed["background_cols"] = list(background.columns)

        def __call__(self, regions, check_additivity=False):
            observed["regions_rows"] = len(regions)
            observed["regions_cols"] = list(regions.columns)
            observed["check_additivity"] = check_additivity
            n_samples, n_features = regions.shape
            values = np.arange(n_samples * n_features, dtype=float)
            return FakeExplanation(values.reshape(n_samples, n_features))

    monkeypatch.setattr(catboost_shap.shap, "Explainer", FakeExplainer)

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
    compute_catboost_shap(
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
    assert observed["background_rows"] == len(train_data)
    assert observed["background_cols"] == feature_cols
    assert observed["regions_rows"] == len(train_data) + len(eval_data) + len(test_data)
    assert observed["regions_cols"] == feature_cols
    assert observed["check_additivity"] is False
    assert list(mean_abs.columns) == ["label_hot"]
    assert list(mean_abs.index) == feature_cols
    assert (fig_dir / "shap_clustermap.png").exists()
    assert (fig_dir / "shap_stacked_bar.png").exists()
    assert (res_dir / "shap" / "shap_stacked_bar_data.csv").exists()
    assert (res_dir / "shap" / "spatial_shap_by_offset_label_hot.csv").exists()


def test_catboost_multiclass_shap_uses_shap_explainer_without_background(
    monkeypatch,
    tmp_path: Path,
):
    import tabnado.catboost_shap as catboost_shap

    class FakeModel:
        pass

    observed = {}

    class FakeExplanation:
        def __init__(self, values):
            self.values = values

    class FakeExplainer:
        def __init__(self, model, *args):
            observed["model"] = model
            observed["n_background_args"] = len(args)

        def __call__(self, regions, check_additivity=False):
            observed["regions_rows"] = len(regions)
            observed["check_additivity"] = check_additivity
            n_samples, n_features = regions.shape
            values = np.arange(n_samples * n_features * 3, dtype=float)
            return FakeExplanation(values.reshape(n_samples, n_features, 3))

    monkeypatch.setattr(catboost_shap.shap, "Explainer", FakeExplainer)

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
    compute_catboost_shap(
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
    assert observed["n_background_args"] == 0
    assert observed["regions_rows"] == len(train_data) + len(test_data)
    assert observed["check_additivity"] is False
    assert list(mean_abs.columns) == ["label_cold", "label_hot", "label_warm"]


def test_run_shap_dispatches_catboost(monkeypatch, tmp_path: Path):
    import joblib
    import tabnado.api as api
    import tabnado.catboost_shap as catboost_shap

    res_dir = tmp_path / "results"
    (res_dir / "final_model").mkdir(parents=True)
    params = SimpleNamespace(
        MODEL_TYPE="catboost",
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
    called = {}

    def fake_compute_catboost_shap(*args, **kwargs):
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
    monkeypatch.setattr(joblib, "load", lambda path: {"task": "classification"})
    monkeypatch.setattr(
        catboost_shap,
        "compute_catboost_shap",
        fake_compute_catboost_shap,
    )

    api.run_shap("params.yaml")

    assert called == {"task": "classification"}
