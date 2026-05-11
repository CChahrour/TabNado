import logging
import os
import shutil
import time
from pathlib import Path

import pytest
import zarr

from tabnado.params import PipelineParams

TEST_DIR = Path(__file__).parent


@pytest.fixture(scope="session")
def params():
    params_path = Path(__file__).parent / "data" / "params_test.yaml"
    logging.debug(f"Loading test parameters from: {params_path}")
    return PipelineParams.from_yaml(params_path)


@pytest.fixture
def worker_id(request):
    worker_input = getattr(request.config, "workerinput", None)
    if isinstance(worker_input, dict):
        return worker_input.get("workerid", "master")
    return "master"


@pytest.fixture(scope="session")
def coverage_path(params):
    """Ensure per-sample zarr stores exist in the dataset directory."""
    from tests.make_test_dataset import SAMPLE_NAMES, create_test_dataset

    dataset_dir = Path(params["DATASET"])

    def _ready(d: Path) -> bool:
        if not d.is_dir():
            return False
        try:
            for name in SAMPLE_NAMES:
                zp = d / f"{name}.zarr"
                if not zp.exists():
                    return False
                root = zarr.open_group(str(zp), mode="r")
                if "assay" not in root.attrs or "sample" not in root.attrs:
                    return False
            return True
        except Exception:
            return False

    if _ready(dataset_dir):
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)
    lock_path = dataset_dir / ".dataset.lock"
    start = time.monotonic()
    timeout_s = 120
    lock_acquired = False

    while True:
        if _ready(dataset_dir):
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
            if not _ready(dataset_dir):
                for p in dataset_dir.glob("*.zarr"):
                    shutil.rmtree(p, ignore_errors=True)
                create_test_dataset(dataset_dir)
        finally:
            lock_path.unlink(missing_ok=True)

    if not _ready(dataset_dir):
        raise RuntimeError(f"Test dataset creation failed: {dataset_dir}")

    return dataset_dir


@pytest.fixture(scope="session")
def loaded_data(params):
    import tabnado

    _, _, target_cols, feature_cols, train, eval_, test = tabnado.load_data(
        **vars(params)
    )
    return params, target_cols, feature_cols, train, eval_, test
