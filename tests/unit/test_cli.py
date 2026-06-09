import shutil
import sys
from pathlib import Path

import pytest

from tabnado import cli as cli_module


@pytest.fixture()
def cli_output_dir(request):
    worker = "master"
    worker_input = getattr(request.config, "workerinput", None)
    if isinstance(worker_input, dict):
        worker = worker_input.get("workerid", worker)
    out_dir = Path("test_output") / "cli" / worker / request.node.name
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def test_init_creates_template(cli_output_dir, monkeypatch, capsys):
    output = cli_output_dir / "params.yaml"
    monkeypatch.setattr(sys, "argv", ["tabnado-init", str(output)])

    cli_module.init()

    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "target: TARGET_NAME" in text
    assert "model_name: GANDALF" in text

    out = capsys.readouterr().out
    assert "Wrote template params file" in out


def test_init_refuses_overwrite_without_force(cli_output_dir, monkeypatch):
    output = cli_output_dir / "params.yaml"
    output.write_text("target: OLD\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["tabnado-init", str(output)])

    with pytest.raises(SystemExit):
        cli_module.init()

    assert output.read_text(encoding="utf-8") == "target: OLD\n"


def test_init_overwrites_with_force(cli_output_dir, monkeypatch):
    output = cli_output_dir / "params.yaml"
    output.write_text("target: OLD\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["tabnado-init", str(output), "--force"])

    cli_module.init()

    assert "target: TARGET_NAME" in output.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Delegation tests — each CLI function calls the matching api.* function
# ---------------------------------------------------------------------------

# Each CLI command imports parse_params_arg and api.* at call time, so we
# patch them on the already-imported modules before invoking the CLI function.


def _patch_delegation(monkeypatch, tmp_path, api_attr: str):
    import tabnado.api as api_mod
    import tabnado.utils as utils_mod

    params_path = tmp_path / "params.yaml"
    called = {}

    monkeypatch.setattr(utils_mod, "parse_params_arg", lambda: params_path)
    monkeypatch.setattr(api_mod, api_attr, lambda path: called.update({"path": path}))
    return called, params_path


def test_run_delegates_to_api(monkeypatch, tmp_path):
    called, params_path = _patch_delegation(monkeypatch, tmp_path, "run")
    cli_module.run()
    assert called["path"] == params_path


def test_data_delegates_to_api(monkeypatch, tmp_path):
    called, params_path = _patch_delegation(monkeypatch, tmp_path, "data")
    cli_module.data()
    assert called["path"] == params_path


def test_sweep_delegates_to_api(monkeypatch, tmp_path):
    called, params_path = _patch_delegation(monkeypatch, tmp_path, "sweep")
    cli_module.sweep()
    assert called["path"] == params_path


def test_train_delegates_to_api(monkeypatch, tmp_path):
    called, params_path = _patch_delegation(monkeypatch, tmp_path, "train")
    cli_module.train()
    assert called["path"] == params_path


def test_evaluate_delegates_to_api(monkeypatch, tmp_path):
    called, params_path = _patch_delegation(monkeypatch, tmp_path, "evaluate")
    cli_module.evaluate()
    assert called["path"] == params_path


def test_shap_delegates_to_api(monkeypatch, tmp_path):
    called, params_path = _patch_delegation(monkeypatch, tmp_path, "shap")
    cli_module.shap()
    assert called["path"] == params_path
