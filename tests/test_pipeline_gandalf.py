"""Integration tests for the tabnado pipeline."""

import json
import sys
from pathlib import Path


def test_gandalf_pipeline(coverage_path, params, loaded_data):
    """Test the full Gandalf pipeline: data loading, hyperparameter sweep, training, evaluation, and SHAP analysis."""
    from tabnado.cli import run as cli_run

    old_argv = sys.argv[:]
    try:
        params_path = Path(__file__).parent / "data" / "params_gandalf_test.yaml"
        sys.argv = ["tabnado-run", "-p", str(params_path)]
        cli_run()
    finally:
        sys.argv = old_argv

    data_dir = Path(params["RES_DIR"]) / "dataset"
    eval_dir = Path(params["RES_DIR"]) / "evaluate"
    shap_dir = Path(params["RES_DIR"]) / "shap"
    fig_dir = Path(params["FIG_DIR"])

    best_hp_path = Path(params["RES_DIR"]) / "best_hyperparameters.json"
    best_hp = json.loads(best_hp_path.read_text())
    expected_keys = {
        "learning_rate",
        "weight_decay",
        "gradient_clip_val",
        "embedding_dropout",
        "gflu_dropout",
        "gflu_feature_init_sparsity",
        "gflu_stages",
    }
    model_path = Path(params["RES_DIR"]) / "final_model"
    _, target_cols, _, _, _, _ = loaded_data

    assert coverage_path.exists()
    assert (data_dir / "dataset_train.parquet").exists()
    assert (data_dir / "dataset_eval.parquet").exists()
    assert (data_dir / "dataset_test.parquet").exists()
    assert best_hp_path.exists()
    assert set(best_hp.keys()) == expected_keys
    assert model_path.exists()
    assert (model_path / "model.ckpt").exists()
    assert any(fig_dir.glob("scatter_test_*.png"))
    assert (fig_dir / "embeddings_umap.png").exists()
    assert (eval_dir / "metrics.json").exists()
    assert (eval_dir / "embeddings_umap.parquet").exists()
    assert (eval_dir / "predictions.parquet").exists()
    assert (shap_dir / "shap_mean_abs.csv").exists()
    assert (fig_dir / "shap_clustermap.png").exists()

    for col in target_cols:
        safe_col = col.replace("/", "_")
        assert (shap_dir / f"spatial_shap_by_offset_{safe_col}.csv").exists()
        assert (fig_dir / f"shap_spatial_heatmap_{safe_col}.png").exists()
        assert (fig_dir / f"shap_offset_line_{safe_col}.png").exists()
