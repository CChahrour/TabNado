from pathlib import Path

import numpy as np
import pandas as pd

from tabnado.evaluate import (
    _limit_categorical_labels,
    _plot_roc_curve,
    compute_umap_embeddings,
)
from tabnado.utils import classification_metrics, flatten_metric_dict


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


def test_classification_metrics_includes_f1_per_class_label():
    metrics = classification_metrics(
        pd.Series(["a", "a", "b", "b", "c", "c"]),
        pd.Series(["a", "b", "b", "b", "c", "a"]),
        classes=["a", "b", "c"],
    )

    assert metrics["per_class_f1"]["a"] == 0.5
    assert metrics["per_class_f1"]["b"] == 0.8
    assert np.isclose(metrics["per_class_f1"]["c"], 2 / 3)

    flat = flatten_metric_dict(metrics, prefix="eval/")
    assert flat["eval/per_class_f1/a"] == 0.5
    assert flat["eval/per_class_f1/b"] == 0.8
    assert np.isclose(flat["eval/per_class_f1/c"], 2 / 3)


def test_limit_categorical_labels_groups_low_frequency_values():
    labels = pd.Series(["a"] * 5 + ["b"] * 4 + ["c"] * 3 + ["d"] * 2 + ["e"])

    display = _limit_categorical_labels(labels, max_categories=4)

    assert list(display.categories) == ["a", "b", "c", "Other"]
    assert list(pd.Series(display).value_counts().index)[:3] == ["a", "b", "c"]
    assert int((display == "Other").sum()) == 3


def test_compute_umap_embeddings_limits_classification_palette_from_cache(
    tmp_path: Path,
):
    fig_dir = tmp_path / "figures"
    eval_dir = tmp_path / "results" / "evaluate"
    fig_dir.mkdir()
    eval_dir.mkdir(parents=True)

    labels = [f"class_{idx:02d}" for idx in range(25)]
    pd.DataFrame(
        {
            "UMAP1": np.arange(len(labels), dtype=float),
            "UMAP2": np.zeros(len(labels), dtype=float),
            "true_label": labels,
            "true_label_code": np.arange(len(labels), dtype=int),
        }
    ).to_parquet(eval_dir / "embeddings_umap.parquet")

    compute_umap_embeddings(
        final_model=object(),
        test_data=pd.DataFrame({"label": labels}),
        feature_cols=["feature"],
        target_cols=["label"],
        FIG_DIR=str(fig_dir),
        RES_DIR=str(tmp_path / "results"),
        target="label",
        model_type="gandalf",
        task="classification",
    )

    assert (fig_dir / "embeddings_umap.png").exists()
