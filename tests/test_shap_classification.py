from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xgboost as xgb

from tabnado.tasks import (
    classification_shap_output_columns,
    encode_classification_target,
)
from tabnado.xgb_shap import compute_xgb_shap


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
    compute_xgb_shap(
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


def test_run_shap_dispatches_xgboost_classification(monkeypatch, tmp_path: Path):
    import joblib
    import tabnado.api as api
    import tabnado.xgb_shap as xgb_shap

    res_dir = tmp_path / "results"
    (res_dir / "final_model").mkdir(parents=True)
    params = SimpleNamespace(
        MODEL_TYPE="xgboost",
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

    def fake_compute_xgb_shap(*args, **kwargs):
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
    monkeypatch.setattr(xgb_shap, "compute_xgb_shap", fake_compute_xgb_shap)

    api.run_shap("params.yaml")

    assert called == {"task": "classification"}
