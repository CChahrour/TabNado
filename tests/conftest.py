import os
import shutil
import time
from pathlib import Path

import pytest
import zarr


TEST_DIR = Path(__file__).parent
PARAMS_PATH = TEST_DIR / "data" / "params_test.yaml"


@pytest.fixture(scope="session")
def params():
    from tabnado.utils import load_params

    return load_params(PARAMS_PATH)


@pytest.fixture(scope="session")
def coverage_path(params):
    """Ensure the QuantNado zarr store exists before any test that needs it."""
    from tests.make_test_dataset import create_test_dataset

    path = Path(params["DATASET"]) / "coverage.zarr"

    def _ready(zarr_path: Path) -> bool:
        if not zarr_path.exists() or not (zarr_path / "zarr.json").exists():
            return False
        try:
            group = zarr.open_group(str(zarr_path), mode="r")
            _ = group.attrs["sample_names"]
            _ = group.attrs["chromsizes"]
            _ = group.attrs["chunk_len"]
            return True
        except Exception:
            return False

    if _ready(path):
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / ".coverage.zarr.lock"
    start = time.monotonic()
    timeout_s = 120
    lock_acquired = False

    while True:
        if _ready(path):
            break
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            lock_acquired = True
            break
        except FileExistsError:
            if time.monotonic() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for dataset lock: {lock_path}")
            time.sleep(0.1)

    if lock_acquired:
        try:
            if not _ready(path):
                if path.exists():
                    shutil.rmtree(path, ignore_errors=True)
                create_test_dataset(path)
        finally:
            lock_path.unlink(missing_ok=True)

    if not _ready(path):
        raise RuntimeError(f"Test dataset creation failed: {path}")

    return path


@pytest.fixture(scope="session")
def loaded_data(params):
    import tabnado
    from tabnado.utils import LOAD_DATA_PARAMS

    _, _, target_cols, feature_cols, train, eval_, test = tabnado.load_data(
        **{k: params[k] for k in LOAD_DATA_PARAMS}
    )
    return params, target_cols, feature_cols, train, eval_, test
