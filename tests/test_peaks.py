from pathlib import Path

import pytest

# ============================================================
# Peaks / BED region pathway test
# ============================================================

TEST_DIR = Path(__file__).parent
PEAKS_BED = TEST_DIR / "data" / "test_peaks.bed"
PEAKS_PARAMS_PATH = TEST_DIR / "data" / "params_test.yaml"


@pytest.fixture()
def peak_output_dir(request, worker_id):
    out_dir = Path("test_output") / "peak_pathway" / worker_id / request.node.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def test_peaks_pathway(coverage_path, peak_output_dir):
    """Test data loading using a BED peaks file instead of GTF."""
    from tabnado.data import load_data
    from tabnado.utils import load_params, LOAD_DATA_PARAMS

    params = load_params(PEAKS_PARAMS_PATH)
    # Override WINDOWS_BED to use the test peaks file
    params["WINDOWS_BED"] = PEAKS_BED

    # Use a temp output dir to avoid collisions across parallel test workers.
    peak_output_dir.mkdir(parents=True, exist_ok=True)
    params["RES_DIR"] = str(peak_output_dir)
    params["FIG_DIR"] = str(peak_output_dir / "figures")

    data_dir = peak_output_dir / "dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    Path(params["FIG_DIR"]).mkdir(parents=True, exist_ok=True)

    _, _, target_cols, feature_cols, train_data, eval_data, test_data = load_data(
        **{k: params[k] for k in LOAD_DATA_PARAMS}
    )

    assert len(train_data) > 0, "Expected non-empty training data for peaks pathway"
    assert len(eval_data) > 0, "Expected non-empty eval data for peaks pathway"
    assert len(test_data) > 0, "Expected non-empty test data for peaks pathway"
    assert len(feature_cols) > 0, "Expected feature columns for peaks pathway"
