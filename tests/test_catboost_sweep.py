import json
from pathlib import Path

import pandas as pd

from tabnado.sweep import _sweep_catboost


def test_catboost_sweep_uses_optuna_and_eval_data(monkeypatch, tmp_path: Path):
    import tabnado.sweep as sweep_module

    fit_calls = []

    class FakeClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(
            self,
            X,
            y,
            eval_set=None,
            early_stopping_rounds=None,
            verbose=False,
        ):
            fit_calls.append(
                {
                    "fit_rows": len(X),
                    "eval_rows": len(eval_set[0]),
                    "early_stopping_rounds": early_stopping_rounds,
                    "loss_function": self.kwargs["loss_function"],
                    "params": self.kwargs,
                }
            )
            return self

        def predict(self, X):
            return [0] * len(X)

    class FakeRegressor:
        pass

    monkeypatch.setattr(
        sweep_module,
        "_import_catboost",
        lambda: (FakeClassifier, FakeRegressor),
    )

    train_data = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.3, 1.1, 1.2, 1.3],
            "f2": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
            "label": ["a", "a", "a", "b", "b", "b"],
        }
    )
    eval_data = pd.DataFrame(
        {
            "f1": [0.4, 1.4],
            "f2": [1.4, 2.4],
            "label": ["a", "b"],
        }
    )

    best_hp = _sweep_catboost(
        feature_cols=["f1", "f2"],
        target_cols=["label"],
        train_data=train_data,
        eval_data=eval_data,
        n_sweeps=2,
        sweep_fraction=1.0,
        RES_DIR=str(tmp_path),
        TASK="classification",
    )

    assert len(fit_calls) == 2
    assert {call["fit_rows"] for call in fit_calls} == {len(train_data)}
    assert {call["eval_rows"] for call in fit_calls} == {len(eval_data)}
    assert {call["early_stopping_rounds"] for call in fit_calls} == {10}
    assert {call["loss_function"] for call in fit_calls} == {"Logloss"}
    assert {"colsample_bylevel", "depth", "bootstrap_type"}.issubset(best_hp)

    saved_hp = json.loads((tmp_path / "best_hyperparameters.json").read_text())
    assert saved_hp == best_hp

    trials = pd.read_csv(tmp_path / "catboost_optuna_trials.csv")
    assert len(trials) == 2
    assert {"number", "value", "colsample_bylevel", "depth"}.issubset(trials.columns)


def test_catboost_sweep_zero_trials_writes_default_hyperparameters(
    monkeypatch,
    tmp_path: Path,
):
    import tabnado.sweep as sweep_module

    class FakeClassifier:
        pass

    class FakeRegressor:
        pass

    monkeypatch.setattr(
        sweep_module,
        "_import_catboost",
        lambda: (FakeClassifier, FakeRegressor),
    )

    train_data = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 1.1, 1.2],
            "label": ["a", "a", "b", "b"],
        }
    )

    best_hp = _sweep_catboost(
        feature_cols=["f1"],
        target_cols=["label"],
        train_data=train_data,
        n_sweeps=0,
        RES_DIR=str(tmp_path),
        TASK="classification",
    )

    assert best_hp["bootstrap_type"] == "Bayesian"
    assert best_hp["bagging_temperature"] == 1.0
    saved_hp = json.loads((tmp_path / "best_hyperparameters.json").read_text())
    assert saved_hp == best_hp
