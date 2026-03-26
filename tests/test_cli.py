import sys
from pathlib import Path

import pytest

from tabnado import init as init_module


@pytest.fixture()
def cli_output_dir(request):
    worker = "master"
    worker_input = getattr(request.config, "workerinput", None)
    if isinstance(worker_input, dict):
        worker = worker_input.get("workerid", worker)
    out_dir = Path("test_output") / "cli" / worker / request.node.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def test_init_creates_template(cli_output_dir, monkeypatch, capsys):
    output = cli_output_dir / "params.yaml"
    monkeypatch.setattr(sys, "argv", ["tabnado-init", str(output)])

    init_module.main()

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
        init_module.main()

    assert output.read_text(encoding="utf-8") == "target: OLD\n"


def test_init_overwrites_with_force(cli_output_dir, monkeypatch):
    output = cli_output_dir / "params.yaml"
    output.write_text("target: OLD\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["tabnado-init", str(output), "--force"])

    init_module.main()

    assert "target: TARGET_NAME" in output.read_text(encoding="utf-8")
