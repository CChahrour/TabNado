"""Integration tests for the tabnado pipeline."""

import json
import sys
from pathlib import Path

import pytest

from tabnado.params import PipelineParams

# ============================================================
# XGBoost pipeline tests
# ============================================================

TEST_DIR = Path(__file__).parent
XGB_PARAMS_PATH = TEST_DIR / "data" / "params_test_xgboost.yaml"


@pytest.fixture(scope="module")
def xgb_params(coverage_path):
    return PipelineParams.from_yaml(XGB_PARAMS_PATH)


@pytest.fixture(scope="module")
def xgb_loaded_data(xgb_params):
    import tabnado

    _, _, target_cols, feature_cols, train, eval_, test = tabnado.load_data(
        **vars(xgb_params)
    )
    return xgb_params, target_cols, feature_cols, train, eval_, test


def test_xgboost_pipeline(coverage_path, xgb_params, xgb_loaded_data):
    """Test the full XGBoost pipeline via CLI tabnado-run."""
    from tabnado.cli import run as cli_run

    old_argv = sys.argv[:]
    try:
        sys.argv = ["tabnado-run", "-p", str(XGB_PARAMS_PATH)]
        cli_run()
    finally:
        sys.argv = old_argv

    data_dir = Path(xgb_params["RES_DIR"]) / "dataset"
    eval_dir = Path(xgb_params["RES_DIR"]) / "evaluate"
    shap_dir = Path(xgb_params["RES_DIR"]) / "shap"
    fig_dir = Path(xgb_params["FIG_DIR"])

    best_hp_path = Path(xgb_params["RES_DIR"]) / "best_hyperparameters.json"
    best_hp = json.loads(best_hp_path.read_text())
    expected_keys = {
        "max_depth",
        "learning_rate",
        "n_estimators",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    }
    model_path = Path(xgb_params["RES_DIR"]) / "final_model" / "xgboost_model.joblib"
    _, target_cols, _, _, _, _ = xgb_loaded_data

    assert coverage_path.exists()
    assert (data_dir / "dataset_train.parquet").exists()
    assert (data_dir / "dataset_eval.parquet").exists()
    assert (data_dir / "dataset_test.parquet").exists()
    assert best_hp_path.exists()
    assert set(best_hp.keys()) == expected_keys
    assert model_path.exists()
    assert any(fig_dir.glob("scatter_test_*.png"))
    assert (eval_dir / "metrics.json").exists()
    assert (eval_dir / "predictions.parquet").exists()
    assert not (eval_dir / "embeddings_umap.parquet").exists()
    assert (shap_dir / "shap_mean_abs.csv").exists()
    assert (fig_dir / "shap_clustermap.png").exists()

    for col in target_cols:
        safe_col = col.replace("/", "_")
        assert (shap_dir / f"spatial_shap_by_offset_{safe_col}.csv").exists()
        assert (fig_dir / f"shap_spatial_heatmap_{safe_col}.png").exists()
        assert (fig_dir / f"shap_offset_line_{safe_col}.png").exists()
