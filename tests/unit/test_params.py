from tabnado.params import PipelineParams


def test_create_directories_does_not_create_redundant_logs_dir(tmp_path):
    params_path = tmp_path / "params.yaml"
    params_path.write_text(
        "\n".join(
            [
                "target: Differential",
                "model_name: GANDALF",
                "task: auto",
                "output_dir: tabnado_output",
                "dataset: data/dataset",
                "logging: wandb",
            ]
        ),
        encoding="utf-8",
    )

    params = PipelineParams.from_yaml(params_path)
    params.RES_DIR = str(tmp_path / params.RES_DIR)
    params.FIG_DIR = str(tmp_path / params.FIG_DIR)
    params.LOGGING_DIR = str(tmp_path / params.LOGGING_DIR)
    params.DATA_DIR = str(tmp_path / params.DATA_DIR)

    params.create_directories()

    assert (tmp_path / "tabnado_output" / "GANDALF_Differential").is_dir()
    assert (tmp_path / "tabnado_output" / "GANDALF_Differential" / "figures").is_dir()
    assert (tmp_path / "tabnado_output" / "GANDALF_Differential" / "dataset").is_dir()
    assert not (tmp_path / "tabnado_output" / "GANDALF_Differential" / "logs").exists()
