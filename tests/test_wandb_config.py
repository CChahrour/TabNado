"""Unit tests for WandbConfig — no network calls."""

from tabnado.wandb import WandbConfig


def test_from_params_constructs_correctly(params):
    cfg = WandbConfig.from_params(params)
    assert cfg.project == params.PROJECT
    assert cfg.entity == params.ENTITY
    assert cfg.model_name == params.MODEL_TYPE
    assert cfg.target == params.TARGET
    assert cfg.res_dir == params.RES_DIR
