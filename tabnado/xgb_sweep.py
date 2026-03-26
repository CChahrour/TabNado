import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor


def sweep_xgboost(
    feature_cols: list[str],
    target_cols: list[str],
    train_data: pd.DataFrame,
    n_sweeps: int = 20,
    sweep_fraction: float = 0.1,
    RES_DIR: str = "results",
    seed: int = 42,
    LOGGING: str = "wandb",
    PROJECT: str = "PROJECT_NAME",
    **kwargs,
) -> dict:
    """
    Randomised HP search for XGBoost using GroupKFold on chromosomes.
    When LOGGING == 'wandb', each candidate run is logged to wandb after the
    search completes.

    Returns best_hp dict and saves best_hyperparameters.json to RES_DIR.
    """
    Path(RES_DIR).mkdir(parents=True, exist_ok=True)
    use_wandb = LOGGING == "wandb"

    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        max_bin=256,
        n_jobs=1,
        random_state=seed,
        verbosity=0,
    )

    if len(target_cols) == 1:
        estimator = base
        param_dist = {
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": np.logspace(np.log10(0.01), np.log10(0.2), 10),
            "n_estimators": [300, 600, 1000],
            "min_child_weight": [1, 3, 5, 10],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 1e-3, 1e-2, 1e-1, 1.0],
            "reg_lambda": [0.1, 1.0, 5.0, 10.0],
        }
    else:
        estimator = MultiOutputRegressor(base, n_jobs=1)
        param_dist = {
            "estimator__max_depth": [3, 4, 5, 6, 8],
            "estimator__learning_rate": np.logspace(np.log10(0.01), np.log10(0.2), 10),
            "estimator__n_estimators": [300, 600, 1000],
            "estimator__min_child_weight": [1, 3, 5, 10],
            "estimator__subsample": [0.6, 0.8, 1.0],
            "estimator__colsample_bytree": [0.6, 0.8, 1.0],
            "estimator__reg_alpha": [0.0, 1e-3, 1e-2, 1e-1, 1.0],
            "estimator__reg_lambda": [0.1, 1.0, 5.0, 10.0],
        }

    tune_data = train_data.sample(frac=min(sweep_fraction, 1.0), random_state=seed)
    X_tune = tune_data[feature_cols].values
    y_tune = tune_data[target_cols].values
    if len(target_cols) == 1:
        y_tune = y_tune.ravel()

    def _default_best_hp() -> dict:
        return {
            k.replace("estimator__", ""): (
                float(v[0]) if isinstance(v[0], np.floating) else v[0]
            )
            for k, v in param_dist.items()
        }

    if len(X_tune) < 2:
        best_hp = _default_best_hp()
        logger.warning(
            "XGBoost HP sweep skipped: need at least 2 samples, got {}. "
            "Using deterministic default hyperparameters.".format(len(X_tune))
        )
        out_path = Path(RES_DIR) / "best_hyperparameters.json"
        with open(out_path, "w") as f:
            json.dump(best_hp, f, indent=2)
        logger.info(f"Saved best hyperparameters to {out_path}")
        return best_hp

    if hasattr(tune_data.index, "get_level_values"):
        contigs = tune_data.index.get_level_values("contig").astype(str)
    else:
        contigs = tune_data.index.astype(str).str.split(":").str[0]
    groups = contigs.values

    n_unique_groups = len(np.unique(groups))
    if n_unique_groups >= 2:
        n_splits = min(3, n_unique_groups)
        cv = GroupKFold(n_splits=n_splits)
        fit_kwargs = {"groups": groups}
        cv_desc = f"{n_splits}-fold GroupKFold by chromosome"
    else:
        n_splits = min(3, len(X_tune))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fit_kwargs = {}
        cv_desc = f"{n_splits}-fold KFold (single chromosome group in sweep subset)"

    r2_macro = make_scorer(
        r2_score,
        greater_is_better=True,
        multioutput="uniform_average",
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_sweeps,
        scoring=r2_macro,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=seed,
        refit=False,
    )

    logger.info(
        f"XGBoost HP sweep: {n_sweeps} iterations on {len(X_tune):,} regions "
        f"({sweep_fraction:.0%} of train), {cv_desc}"
    )
    search.fit(X_tune, y_tune, **fit_kwargs)

    best_score = search.best_score_
    raw_params = search.best_params_
    best_hp = {k.replace("estimator__", ""): v for k, v in raw_params.items()}
    logger.info(f"Best sweep R²={best_score:.4f}  params={best_hp}")

    if use_wandb:
        import wandb

        results = search.cv_results_
        for i in range(len(results["params"])):
            hp = {
                k.replace("estimator__", ""): v for k, v in results["params"][i].items()
            }
            score = float(results["mean_test_score"][i])
            with wandb.init(
                project=PROJECT,
                dir=RES_DIR,
                reinit="finish_previous",
                name=f"sweep_{time.strftime('%Y-%m-%d')}_{i}",
                config=hp,
                tags=["xgb-sweep"],
            ):
                wandb.log({"val_r2": score})

    out_path = Path(RES_DIR) / "best_hyperparameters.json"
    with open(out_path, "w") as f:
        json.dump(
            best_hp,
            f,
            indent=2,
            default=lambda x: float(x) if isinstance(x, np.floating) else x,
        )
    logger.info(f"Saved best hyperparameters to {out_path}")

    return best_hp
