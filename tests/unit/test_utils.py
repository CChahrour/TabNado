import numpy as np
import pandas as pd
import pytest

from tabnado.utils import (
    classification_metrics,
    classification_prediction_frame,
    classification_shap_output_columns,
    encode_classification_target,
    flatten_metric_dict,
    json_safe,
    resolve_task,
    validate_task,
)

# ---------------------------------------------------------------------------
# validate_task / resolve_task
# ---------------------------------------------------------------------------


def test_validate_task_normalises_to_lowercase():
    assert validate_task("Regression") == "regression"
    assert validate_task("AUTO") == "auto"


def test_validate_task_raises_on_invalid():
    with pytest.raises(ValueError, match="Invalid task"):
        validate_task("unsupervised")


def test_resolve_task_explicit_regression():
    df = pd.DataFrame({"t": [1.0, 2.0]})
    assert resolve_task("regression", df, ["t"]) == "regression"


def test_resolve_task_explicit_classification():
    df = pd.DataFrame({"t": ["a", "b"]})
    assert resolve_task("classification", df, ["t"]) == "classification"


def test_resolve_task_auto_numeric_is_regression():
    df = pd.DataFrame({"t": [1.0, 2.0, 3.0]})
    assert resolve_task("auto", df, ["t"]) == "regression"


def test_resolve_task_auto_string_is_classification():
    df = pd.DataFrame({"t": ["hot", "cold"]})
    assert resolve_task("auto", df, ["t"]) == "classification"


def test_resolve_task_auto_bool_is_classification():
    df = pd.DataFrame({"t": [True, False, True]})
    assert resolve_task("auto", df, ["t"]) == "classification"


def test_resolve_task_auto_categorical_is_classification():
    df = pd.DataFrame({"t": pd.Categorical(["a", "b", "a"])})
    assert resolve_task("auto", df, ["t"]) == "classification"


def test_resolve_task_auto_multi_target_is_regression():
    df = pd.DataFrame({"t1": ["a", "b"], "t2": [1.0, 2.0]})
    assert resolve_task("auto", df, ["t1", "t2"]) == "regression"


# ---------------------------------------------------------------------------
# encode_classification_target
# ---------------------------------------------------------------------------


def test_encode_without_eval():
    df = pd.DataFrame({"label": ["a", "b", "a", "b"]})
    enc = encode_classification_target(df, ["label"])
    assert enc.target_col == "label"
    assert enc.classes == ["a", "b"]
    assert enc.problem_type == "binary"
    assert enc.eval is None
    assert list(enc.train) == [0, 1, 0, 1]


def test_encode_with_eval_fits_combined():
    train = pd.DataFrame({"label": ["a", "a"]})
    eval_ = pd.DataFrame({"label": ["b", "b"]})
    enc = encode_classification_target(train, ["label"], eval_data=eval_)
    assert "b" in enc.classes
    assert enc.eval is not None
    assert list(enc.eval) == [1, 1]


def test_encode_multiclass_problem_type():
    df = pd.DataFrame({"label": ["a", "b", "c", "a"]})
    enc = encode_classification_target(df, ["label"])
    assert enc.problem_type == "multiclass"
    assert len(enc.classes) == 3


def test_encode_single_class_raises():
    df = pd.DataFrame({"label": ["a", "a", "a"]})
    with pytest.raises(ValueError, match="at least two classes"):
        encode_classification_target(df, ["label"])


def test_encode_multiple_targets_raises():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="exactly one target"):
        encode_classification_target(df, ["a", "b"])


# ---------------------------------------------------------------------------
# classification_shap_output_columns
# ---------------------------------------------------------------------------


def test_shap_cols_zero_outputs_raises():
    with pytest.raises(ValueError, match="at least one"):
        classification_shap_output_columns("label", ["a", "b"], 0)


def test_shap_cols_outputs_equal_classes():
    result = classification_shap_output_columns("label", ["cold", "hot", "warm"], 3)
    assert result == ["label_cold", "label_hot", "label_warm"]


def test_shap_cols_binary_one_output_returns_positive_class():
    result = classification_shap_output_columns("label", ["cold", "hot"], 1)
    assert result == ["label_hot"]


def test_shap_cols_multiclass_one_output_returns_target():
    result = classification_shap_output_columns("label", ["a", "b", "c"], 1)
    assert result == ["label"]


def test_shap_cols_mismatch_returns_indexed():
    result = classification_shap_output_columns("label", ["a", "b"], 4)
    assert result == [
        "label_output_0",
        "label_output_1",
        "label_output_2",
        "label_output_3",
    ]


# ---------------------------------------------------------------------------
# classification_prediction_frame
# ---------------------------------------------------------------------------


def test_prediction_frame_2d_probabilities():
    labels = np.array(["hot", "cold"])
    proba = np.array([[0.2, 0.8], [0.7, 0.3]])
    df = classification_prediction_frame(
        labels, proba, "label", ["cold", "hot"], pd.RangeIndex(2)
    )
    assert list(df["label"]) == ["hot", "cold"]
    assert "label_hot_probability" in df.columns


def test_prediction_frame_1d_probability_expands_to_2d():
    labels = np.array(["hot", "cold"])
    proba_1d = np.array([0.8, 0.3])
    df = classification_prediction_frame(
        labels, proba_1d, "label", ["cold", "hot"], pd.RangeIndex(2)
    )
    assert df.shape[1] == 3  # label + 2 probability cols
    assert df["label_hot_probability"].iloc[0] == pytest.approx(0.8)


def test_prediction_frame_no_probabilities():
    labels = np.array(["hot"])
    df = classification_prediction_frame(
        labels, None, "label", ["cold", "hot"], pd.RangeIndex(1)
    )
    assert list(df.columns) == ["label"]


# ---------------------------------------------------------------------------
# classification_metrics
# ---------------------------------------------------------------------------


def test_metrics_includes_log_loss_when_probabilities_given():
    y_true = pd.Series(["a", "a", "b", "b"])
    y_pred = pd.Series(["a", "b", "b", "a"])
    proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.6, 0.4]])
    metrics = classification_metrics(
        y_true, y_pred, probabilities=proba, classes=["a", "b"]
    )
    assert "log_loss" in metrics


def test_metrics_omits_log_loss_without_probabilities():
    y_true = pd.Series(["a", "b"])
    y_pred = pd.Series(["a", "b"])
    metrics = classification_metrics(y_true, y_pred)
    assert "log_loss" not in metrics


# ---------------------------------------------------------------------------
# flatten_metric_dict
# ---------------------------------------------------------------------------


def test_flatten_nested_with_prefix():
    metrics = {"accuracy": 0.9, "per_class_f1": {"a": 0.8, "b": 0.7}}
    flat = flatten_metric_dict(metrics, prefix="eval/")
    assert flat["eval/accuracy"] == 0.9
    assert flat["eval/per_class_f1/a"] == 0.8
    assert flat["eval/per_class_f1/b"] == 0.7


def test_flatten_custom_sep():
    metrics = {"a": {"b": 1}}
    flat = flatten_metric_dict(metrics, sep=".")
    assert flat["a.b"] == 1


# ---------------------------------------------------------------------------
# json_safe
# ---------------------------------------------------------------------------


def test_json_safe_converts_numpy_scalar():
    val = np.float32(3.14)
    result = json_safe(val)
    assert isinstance(result, float)


def test_json_safe_converts_numpy_array():
    arr = np.array([1, 2, 3])
    assert json_safe(arr) == [1, 2, 3]


def test_json_safe_recurses_into_dict():
    d = {"k": np.int64(5)}
    result = json_safe(d)
    assert result == {"k": 5}
    assert isinstance(result["k"], int)
