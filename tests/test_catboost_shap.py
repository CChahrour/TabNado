import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd

from tabnado.catboost_shap import (
    _catboost_shap_to_output_list,
    compute_catboost_shap,
)


def test_catboost_shap_values_strip_expected_value_column():
    values = np.arange(2 * 5).reshape(2, 5)
    outputs = _catboost_shap_to_output_list(values, n_features=4)

    assert len(outputs) == 1
    np.testing.assert_array_equal(outputs[0], values[:, :4])


def test_catboost_multiclass_shap_values_support_catboost_shape_variants():
    class_first = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    feature_first = np.arange(2 * 5 * 3).reshape(2, 5, 3)

    class_first_outputs = _catboost_shap_to_output_list(class_first, n_features=4)
    feature_first_outputs = _catboost_shap_to_output_list(
        feature_first,
        n_features=4,
    )

    assert len(class_first_outputs) == 3
    assert len(feature_first_outputs) == 3
    np.testing.assert_array_equal(class_first_outputs[0], class_first[:, 0, :4])
    np.testing.assert_array_equal(feature_first_outputs[0], feature_first[:, :4, 0])


def test_catboost_shap_supports_classification_artifact(
    monkeypatch,
    tmp_path: Path,
):
    class FakePool:
        def __init__(self, data, feature_names=None):
            self.data = data
            self.feature_names = feature_names

    class FakeModel:
        def get_feature_importance(self, pool, type):
            assert type == "ShapValues"
            n_samples = len(pool.data)
            n_features = len(pool.feature_names)
            values = np.arange(n_samples * (n_features + 1), dtype=float)
            return values.reshape(n_samples, n_features + 1)

    fake_catboost = ModuleType("catboost")
    fake_catboost.Pool = FakePool
    monkeypatch.setitem(sys.modules, "catboost", fake_catboost)

    feature_cols = ["ChIP_A_-100", "ChIP_A_0", "ChIP_B_-100", "ChIP_B_0"]
    train_data = pd.DataFrame(
        np.arange(8 * len(feature_cols)).reshape(8, len(feature_cols)),
        columns=feature_cols,
    )
    train_data["label"] = ["cold", "hot"] * 4
    test_data = train_data.copy()

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
        RES_DIR=str(res_dir),
        FIG_DIR=str(fig_dir),
        task="classification",
    )

    mean_abs = pd.read_csv(res_dir / "shap" / "shap_mean_abs.csv", index_col=0)
    assert list(mean_abs.columns) == ["label_hot"]
    assert list(mean_abs.index) == feature_cols
    assert (fig_dir / "shap_clustermap.png").exists()
    assert (fig_dir / "shap_stacked_bar.png").exists()
    assert (res_dir / "shap" / "shap_stacked_bar_data.csv").exists()
    assert (res_dir / "shap" / "spatial_shap_by_offset_label_hot.csv").exists()


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
