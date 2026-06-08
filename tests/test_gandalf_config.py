from tabnado.sweep import _make_data_config
from tabnado.utils import LoguruProgressCallback


def test_loguru_progress_callback_is_pickleable():
    import pickle

    callback = LoguruProgressCallback()

    pickle.loads(pickle.dumps(callback))


def test_gandalf_data_config_uses_single_process_loader():
    config = _make_data_config(["feature_0"], ["target"])

    assert config.num_workers == 0
    assert config.dataloader_kwargs["persistent_workers"] is False
