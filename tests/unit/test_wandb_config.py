"""Unit tests for WandbConfig — no network calls."""

from unittest.mock import MagicMock, patch

import tabnado.wandb as wandb_module
from tabnado.wandb import WandbConfig, create_eval_report


def test_from_params_constructs_correctly():
    params = {
        "PROJECT": "test_project",
        "ENTITY": "test_entity",
        "MODEL_TYPE": "GANDALF",
        "TARGET": "TEST",
        "RES_DIR": "test_output/results",
    }
    cfg = WandbConfig.from_params(params)
    assert cfg.project == "test_project"
    assert cfg.entity == "test_entity"
    assert cfg.model_name == "GANDALF"
    assert cfg.target == "TEST"
    assert cfg.res_dir == "test_output/results"


def test_from_params_entity_defaults_to_none():
    params = {"PROJECT": "p", "TARGET": "t", "RES_DIR": "r"}
    cfg = WandbConfig.from_params(params)
    assert cfg.entity is None


def test_from_params_model_type_defaults_to_model():
    params = {"PROJECT": "p", "TARGET": "t", "RES_DIR": "r"}
    cfg = WandbConfig.from_params(params)
    assert cfg.model_name == "model"


# ---------------------------------------------------------------------------
# WandbConfig.init_run
# ---------------------------------------------------------------------------


def test_init_run_calls_wandb_init_with_correct_args(monkeypatch):
    mock_wandb = MagicMock()
    monkeypatch.setattr(wandb_module, "_wandb", mock_wandb)

    cfg = WandbConfig(
        project="proj",
        entity="ent",
        model_name="XGBoost",
        target="label",
        res_dir="/tmp/r",
    )
    cfg.init_run(name="run1", group="sweep")

    mock_wandb.init.assert_called_once()
    kwargs = mock_wandb.init.call_args.kwargs
    assert kwargs["project"] == "proj"
    assert kwargs["entity"] == "ent"
    assert kwargs["name"] == "run1"
    assert kwargs["group"] == "sweep"


def test_init_run_omits_reinit_when_none(monkeypatch):
    mock_wandb = MagicMock()
    monkeypatch.setattr(wandb_module, "_wandb", mock_wandb)

    cfg = WandbConfig(
        project="p", entity=None, model_name="m", target="t", res_dir="/r"
    )
    cfg.init_run(name="n", group="g", reinit=None)

    kwargs = mock_wandb.init.call_args.kwargs
    assert "reinit" not in kwargs


def test_init_run_includes_reinit_when_provided(monkeypatch):
    mock_wandb = MagicMock()
    monkeypatch.setattr(wandb_module, "_wandb", mock_wandb)

    cfg = WandbConfig(
        project="p", entity=None, model_name="m", target="t", res_dir="/r"
    )
    cfg.init_run(name="n", group="g", reinit="finish_previous")

    kwargs = mock_wandb.init.call_args.kwargs
    assert kwargs["reinit"] == "finish_previous"


def test_init_run_dir_override_used_when_provided(monkeypatch):
    mock_wandb = MagicMock()
    monkeypatch.setattr(wandb_module, "_wandb", mock_wandb)

    cfg = WandbConfig(
        project="p", entity=None, model_name="m", target="t", res_dir="/default"
    )
    cfg.init_run(name="n", group="g", dir_override="/override")

    kwargs = mock_wandb.init.call_args.kwargs
    assert kwargs["dir"] == "/override"


# ---------------------------------------------------------------------------
# create_eval_report
# ---------------------------------------------------------------------------


def test_create_eval_report_calls_save_and_returns_url(monkeypatch):
    mock_wandb = MagicMock()
    mock_wr = MagicMock()
    monkeypatch.setattr(wandb_module, "_wandb", mock_wandb)
    monkeypatch.setattr(wandb_module, "wr", mock_wr)

    mock_report = MagicMock()
    mock_report.url = "https://wandb.ai/report/123"
    mock_wr.Report.return_value = mock_report

    cfg = WandbConfig(
        project="proj", entity="ent", model_name="m", target="t", res_dir="/r"
    )
    url = create_eval_report(cfg, run_id="abc123", target_cols=["label"])

    mock_report.save.assert_called_once()
    assert url == "https://wandb.ai/report/123"


def test_create_eval_report_uses_api_default_entity_when_none(monkeypatch):
    mock_wandb = MagicMock()
    mock_wr = MagicMock()
    monkeypatch.setattr(wandb_module, "_wandb", mock_wandb)
    monkeypatch.setattr(wandb_module, "wr", mock_wr)

    mock_wandb.Api.return_value.default_entity = "default_entity"
    mock_report = MagicMock()
    mock_report.url = "https://wandb.ai/report/x"
    mock_wr.Report.return_value = mock_report

    cfg = WandbConfig(
        project="proj", entity=None, model_name="m", target="t", res_dir="/r"
    )
    create_eval_report(cfg, run_id="run1", target_cols=["label"])

    # Verify Report was constructed with the resolved entity
    report_kwargs = mock_wr.Report.call_args.kwargs
    assert report_kwargs["entity"] == "default_entity"


def test_create_eval_report_slash_in_target_replaced(monkeypatch):
    mock_wandb = MagicMock()
    mock_wr = MagicMock()
    monkeypatch.setattr(wandb_module, "_wandb", mock_wandb)
    monkeypatch.setattr(wandb_module, "wr", mock_wr)

    mock_wandb.Api.return_value.default_entity = "ent"
    mock_report = MagicMock()
    mock_report.url = "u"
    mock_wr.Report.return_value = mock_report
    mock_wr.MediaBrowser = MagicMock(side_effect=lambda **kw: kw)

    cfg = WandbConfig(
        project="p", entity="ent", model_name="m", target="t", res_dir="/r"
    )
    create_eval_report(cfg, run_id="r", target_cols=["H3K4me3/input"])

    # Check that MediaBrowser was called with "_" instead of "/"
    all_calls = mock_wr.MediaBrowser.call_args_list
    media_keys_used = [kw["media_keys"] for _, kw in all_calls if "media_keys" in kw]
    all_keys = [k for keys in media_keys_used for k in keys]
    assert any("H3K4me3_input" in k for k in all_keys)
