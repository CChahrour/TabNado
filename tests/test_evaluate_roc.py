from pathlib import Path

import numpy as np
import pandas as pd

from tabnado.evaluate import _plot_roc_curve


def test_plot_roc_curve_binary(tmp_path: Path):
    fig_dir = tmp_path / "figures"
    eval_dir = tmp_path / "evaluate"
    fig_dir.mkdir()
    eval_dir.mkdir()

    metrics = _plot_roc_curve(
        pd.Series(["cold", "cold", "hot", "hot"]),
        np.array(
            [
                [0.9, 0.1],
                [0.7, 0.3],
                [0.2, 0.8],
                [0.1, 0.9],
            ]
        ),
        ["cold", "hot"],
        "label",
        str(fig_dir),
        str(eval_dir),
    )

    assert metrics["roc_auc"] == 1.0
    assert (fig_dir / "roc_curve_label.png").exists()
    auc_df = pd.read_csv(eval_dir / "roc_auc.csv")
    assert list(auc_df["class"]) == ["hot"]


def test_plot_roc_curve_multiclass(tmp_path: Path):
    fig_dir = tmp_path / "figures"
    eval_dir = tmp_path / "evaluate"
    fig_dir.mkdir()
    eval_dir.mkdir()

    metrics = _plot_roc_curve(
        pd.Series(["a", "b", "c", "a", "b", "c"]),
        np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.1, 0.7],
            ]
        ),
        ["a", "b", "c"],
        "label",
        str(fig_dir),
        str(eval_dir),
    )

    assert metrics["roc_auc_micro"] == 1.0
    assert metrics["roc_auc_macro"] == 1.0
    assert (fig_dir / "roc_curve_label.png").exists()
    auc_df = pd.read_csv(eval_dir / "roc_auc.csv")
    assert set(auc_df["class"]) == {"a", "b", "c", "micro", "macro"}
