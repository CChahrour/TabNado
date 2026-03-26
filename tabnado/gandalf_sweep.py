import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import wandb
from loguru import logger
from pytorch_tabular import TabularModel
from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.models import GANDALFConfig
from sklearn.metrics import mean_squared_error, r2_score

from tabnado.data import stratified_sample
from tabnado.utils import (
    LOAD_DATA_PARAMS,
    LoguruProgressCallback,
    seed_everything,
    log_macro,
)

_LOCAL_SWEEP_BEST_HP_CACHE: dict[str, dict] = {}


def _center_window_columns(columns: list[str]) -> list[str]:
    """Return only TSS-centered columns (suffix `_0`)."""
    center_cols = [c for c in columns if c.endswith("_0")]
    return center_cols if center_cols else columns


def _make_data_config(feature_cols, target_cols):
    return DataConfig(
        continuous_cols=feature_cols,
        target=target_cols,
        validation_split=0,
        normalize_continuous_features=False,
        num_workers=os.cpu_count(),
        pin_memory=False,
        dataloader_kwargs={"persistent_workers": True},
    )


def _sample_from_sweep_params(param_spec: dict) -> dict:
    sampled = {}
    for key, spec in param_spec.items():
        if "values" in spec:
            sampled[key] = spec["values"][np.random.randint(0, len(spec["values"]))]
            continue

        distribution = spec.get("distribution")
        if distribution == "uniform":
            sampled[key] = float(np.random.uniform(spec["min"], spec["max"]))
        elif distribution == "log_uniform_values":
            low = np.log(spec["min"])
            high = np.log(spec["max"])
            sampled[key] = float(np.exp(np.random.uniform(low, high)))
        else:
            raise ValueError(
                f"Unsupported sweep distribution for {key}: {distribution}"
            )
    return sampled


def sweep_train(
    train_data,
    eval_data,
    test_data,
    feature_cols,
    target_cols,
    FIG_DIR: str = "figures",
    RES_DIR: str = "results",
    SWEEP_FRACTION: float = 0.1,
    PROJECT: str = "PROJECT_NAME",
    LOGGING_DIR: str | None = None,
    SWEEP_ID: str | None = None,
    hp_config: dict | None = None,
    LOGGING: str = "wandb",
):
    use_wandb = LOGGING == "wandb"
    logging_dir = LOGGING_DIR or os.path.join(RES_DIR, "logging")
    os.makedirs(logging_dir, exist_ok=True)
    experiment_project = logging_dir if LOGGING == "tensorboard" else PROJECT

    run_name = f"GANDALF_sweep_{time.strftime('%Y-%m-%d_%H%M')}"
    sweep_root = (
        os.path.join(RES_DIR, "sweep", SWEEP_ID)
        if SWEEP_ID
        else os.path.join(RES_DIR, "sweep")
    )
    run_dir = os.path.join(sweep_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    config_dict = hp_config
    if config_dict is None and not use_wandb:
        config_dict = _sample_from_sweep_params(create_sweep_dict()["parameters"])

    sweep_target_cols = _center_window_columns(target_cols)
    selected_cols = feature_cols + sweep_target_cols

    train_view = train_data[selected_cols]
    eval_view = eval_data[selected_cols]
    test_view = test_data[selected_cols]

    logger.info(
        "Sweep uses all tiled columns: {} features, {} targets".format(
            len(feature_cols), len(sweep_target_cols)
        )
    )

    def _fit_and_eval(config_obj):
        # Optionally initialize wandb with custom config if using wandb
        if LOGGING == "wandb":
            import wandb
            # Always convert config_obj to dict for wandb config
            if hasattr(config_obj, "__dict__"):
                wandb_config = vars(config_obj)
            elif isinstance(config_obj, dict):
                wandb_config = config_obj
            else:
                # fallback: try to convert to dict
                try:
                    wandb_config = dict(config_obj)
                except Exception:
                    wandb_config = {}
            wandb.init(
                project=experiment_project,
                name=run_name,
                group="sweep",
                config=wandb_config,
            )
        model = TabularModel(
            data_config=_make_data_config(feature_cols, sweep_target_cols),
            experiment_config=ExperimentConfig(
                exp_log_freq=1,
                exp_watch=None,
                log_logits=False,
                log_target=LOGGING,
                project_name=experiment_project,
                run_name=run_name,
            ),
            model_config=GANDALFConfig(
                learning_rate=config_obj.learning_rate,
                embedding_dropout=config_obj.embedding_dropout,
                gflu_stages=config_obj.gflu_stages,
                gflu_dropout=config_obj.gflu_dropout,
                gflu_feature_init_sparsity=config_obj.gflu_feature_init_sparsity,
                learnable_sparsity=False,
                head="LinearHead",
                loss="MSELoss",
                metrics=["r2_score", "mean_squared_error"],
                metrics_params=[{}] * 2,
                seed=42,
                target_range=[(0, 1)] * len(sweep_target_cols),
                task="regression",
            ),
            optimizer_config=OptimizerConfig(
                optimizer_params={"weight_decay": config_obj.weight_decay},
            ),
            trainer_config=TrainerConfig(
                accelerator="mps"
                if torch.backends.mps.is_available()
                else "gpu"
                if torch.cuda.is_available()
                else "cpu",
                auto_lr_find=False,
                batch_size=2048,
                check_val_every_n_epoch=1,
                checkpoints_path=os.path.join(run_dir, "checkpoints"),
                early_stopping="valid_r2_score",
                early_stopping_mode="max",
                early_stopping_patience=3,
                load_best=True,
                max_epochs=12,
                gradient_clip_val=config_obj.gradient_clip_val,
                trainer_kwargs=dict(enable_model_summary=False),
                progress_bar="none",
            ),
            verbose=False,
            suppress_lightning_logger=True,
        )

        sweep_train_data = stratified_sample(
            train_view, sweep_target_cols, frac=SWEEP_FRACTION
        )
        logger.info(
            f"Sweep train size: {len(sweep_train_data):,} ({SWEEP_FRACTION:.0%} of full)"
        )

        model.fit(
            train=sweep_train_data,
            validation=eval_view,
            callbacks=[LoguruProgressCallback()],
        )
        # Log macro metrics to wandb if enabled
        if use_wandb:
            log_macro(model, sweep_target_cols)

        pred_df = model.predict(test_view)
        pred_df.columns = [c.replace("_prediction", "") for c in pred_df.columns]
        y_true = test_view[sweep_target_cols].values
        y_pred = pred_df[sweep_target_cols].values

        if np.isnan(y_pred).any():
            logger.warning("NaN predictions - skipping run")
            if use_wandb:
                wandb.log({"failed": True})
            return None

        test_metrics = {
            "test_r2_macro": float(
                r2_score(y_true, y_pred, multioutput="uniform_average")
            ),
            "test_mse_macro": float(mean_squared_error(y_true, y_pred)),
        }
        for i, col in enumerate(sweep_target_cols):
            test_metrics[f"test_r2_{col}"] = float(r2_score(y_true[:, i], y_pred[:, i]))

        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=4)
        with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
            json.dump(vars(config_obj), f, indent=4)

        if use_wandb:
            wandb.log(test_metrics)
        logger.info(test_metrics)

        if hasattr(model, "callbacks"):
            model.callbacks = []
        model.save_model(os.path.join(run_dir, "model"), inference_only=False)
        return test_metrics

    seed_everything(42)
    logger.info(f"======== Fitting {run_name} ========")

    if not use_wandb:
        assert config_dict is not None
        return _fit_and_eval(SimpleNamespace(**config_dict))

    with wandb.init(
        dir=run_dir,
        reinit="finish_previous",
        project=PROJECT,
        name=run_name,
        group="sweep",
    ) as run:
        return _fit_and_eval(run.config)


def create_sweep_dict(
    MODEL_NAME: str = "GANDALF_Sweep",
    PROJECT: str = "PROJECT_NAME",
):
    return {
        "name": MODEL_NAME,
        "method": "bayes",
        "metric": {"name": "valid_r2_score", "goal": "maximize"},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-3,
                "max": 3e-2,
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 5e-4,
                "max": 5e-3,
            },
            "gradient_clip_val": {"distribution": "uniform", "min": 0.5, "max": 2.0},
            "embedding_dropout": {"distribution": "uniform", "min": 0.0, "max": 0.02},
            "gflu_dropout": {"distribution": "uniform", "min": 0.0, "max": 0.05},
            "gflu_feature_init_sparsity": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.4,
            },
            "gflu_stages": {"values": [6, 8, 10, 12]},
        },
    }


def start_sweep_and_run(
    train_data,
    eval_data,
    test_data,
    feature_cols,
    target_cols,
    count,
    RES_DIR: str = "results",
    SWEEP_FRACTION: float = 0.1,
    LOGGING: str = "wandb",
    LOGGING_DIR: str | None = None,
    entity: str | None = None,
    PROJECT: str = "PROJECT_NAME",
) -> str:
    use_wandb = LOGGING == "wandb"
    logging_dir = LOGGING_DIR or os.path.join(RES_DIR, "logging")
    os.makedirs(logging_dir, exist_ok=True)

    sweep_dir = os.path.join(RES_DIR, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", RES_DIR)

    if not use_wandb:
        sweep_id = f"local-{time.strftime('%Y-%m-%d_%H%M')}"
        local_sweep_dir = os.path.join(sweep_dir, sweep_id)
        os.makedirs(local_sweep_dir, exist_ok=True)

        logger.info(f"Running local sweep: {sweep_id}")
        best_hp = None
        best_score = -np.inf
        sweep_params = create_sweep_dict()["parameters"]

        for _ in range(count):
            hp = _sample_from_sweep_params(sweep_params)
            metrics = sweep_train(
                train_data,
                eval_data,
                test_data,
                feature_cols,
                target_cols,
                RES_DIR=RES_DIR,
                SWEEP_FRACTION=SWEEP_FRACTION,
                PROJECT=PROJECT,
                LOGGING_DIR=logging_dir,
                SWEEP_ID=sweep_id,
                hp_config=hp,
                LOGGING=LOGGING,
            )
            if metrics is not None and metrics["test_r2_macro"] > best_score:
                best_score = metrics["test_r2_macro"]
                best_hp = hp

        if best_hp is None:
            raise RuntimeError("Local sweep produced no valid runs")

        _LOCAL_SWEEP_BEST_HP_CACHE[sweep_id] = best_hp
        return sweep_id

    sweep_id = wandb.sweep(sweep=create_sweep_dict(), entity=entity, project=PROJECT)
    logger.info(f"Created sweep: {sweep_id}")
    wandb.agent(
        sweep_id,
        function=lambda: sweep_train(
            train_data,
            eval_data,
            test_data,
            feature_cols,
            target_cols,
            RES_DIR=RES_DIR,
            SWEEP_FRACTION=SWEEP_FRACTION,
            PROJECT=PROJECT,
            LOGGING_DIR=logging_dir,
            SWEEP_ID=sweep_id,
            LOGGING=LOGGING,
        ),
        count=count,
    )
    return sweep_id


def get_best_hp_from_sweep(
    sweep_id: str,
    entity: str | None = None,
    PROJECT: str = "PROJECT_NAME",
    RES_DIR: str = "results",
    LOGGING: str = "wandb",
) -> dict:
    """Fetch the hyperparameters of the best-performing run in a sweep."""
    if LOGGING != "wandb":
        cached = _LOCAL_SWEEP_BEST_HP_CACHE.get(sweep_id)
        if cached is not None:
            return cached

        root_hp_path = os.path.join(RES_DIR, "best_hyperparameters.json")
        if os.path.exists(root_hp_path):
            with open(root_hp_path) as f:
                return json.load(f)

        raise FileNotFoundError(
            f"No local best_hyperparameters found in cache and no file at {root_hp_path}"
        )

    hp_keys = list(create_sweep_dict()["parameters"].keys())

    # Try wandb API first; fall back to local sweep run directories.
    try:
        api = wandb.Api()
        sweep_path = (
            f"{entity}/{PROJECT}/{sweep_id}" if entity else f"{PROJECT}/{sweep_id}"
        )
        sweep = api.sweep(sweep_path)
        best_run = sweep.best_run()
        if best_run is not None:
            hp = {k: best_run.config[k] for k in hp_keys if k in best_run.config}
            if hp:
                return hp
    except Exception as e:
        logger.warning(
            f"wandb API unavailable ({e}), falling back to local sweep files"
        )

    # Fall back: scan local run directories under RES_DIR/sweep/sweep_id/
    sweep_dir = os.path.join(RES_DIR, "sweep", sweep_id)
    if not os.path.isdir(sweep_dir):
        raise FileNotFoundError(f"No local sweep directory found at {sweep_dir}")

    best_hp = None
    best_score = -np.inf
    for run_dir in os.scandir(sweep_dir):
        if not run_dir.is_dir():
            continue
        metrics_path = os.path.join(run_dir.path, "metrics.json")
        hp_path = os.path.join(run_dir.path, "hyperparameters.json")
        if not os.path.exists(metrics_path) or not os.path.exists(hp_path):
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        score = metrics.get("test_r2_macro", -np.inf)
        if score > best_score:
            with open(hp_path) as f:
                all_hp = json.load(f)
            hp = {k: all_hp[k] for k in hp_keys if k in all_hp}
            if hp:
                best_score = score
                best_hp = hp

    if best_hp is None:
        raise RuntimeError(f"No valid local sweep runs found in {sweep_dir}")

    logger.info(f"Best local sweep run: test_r2_macro={best_score:.4f}  hp={best_hp}")
    return best_hp


def main():
    from tabnado.data import load_data
    from tabnado.utils import load_params, parse_params_arg, setup_logger

    params = load_params(parse_params_arg())
    setup_logger(params["RES_DIR"], params["PROJECT"])
    run_start = time.perf_counter()
    logger.info("========== GANDALF SWEEP START ==========")
    logger.info(
        "Sweep config: project={} logging={} n_sweeps={} sweep_fraction={}".format(
            params["PROJECT"],
            params["LOGGING"],
            params["N_SWEEPS"],
            params["SWEEP_FRACTION"],
        )
    )
    if params["LOGGING"] == "wandb":
        os.environ["WANDB_DIR"] = params["RES_DIR"]

    _, _, target_cols, feature_cols, train_data, eval_data, test_data = load_data(
        **{k: params[k] for k in LOAD_DATA_PARAMS}
    )

    sweep_id = start_sweep_and_run(
        train_data,
        eval_data,
        test_data,
        feature_cols,
        target_cols,
        count=params["N_SWEEPS"],
        RES_DIR=params["RES_DIR"],
        SWEEP_FRACTION=params["SWEEP_FRACTION"],
        LOGGING=params["LOGGING"],
        LOGGING_DIR=params["LOGGING_DIR"],
        PROJECT=params["PROJECT"],
    )
    best_hp = get_best_hp_from_sweep(
        sweep_id,
        PROJECT=params["PROJECT"],
        RES_DIR=params["RES_DIR"],
        LOGGING=params["LOGGING"],
    )
    logger.info(f"Best hyperparameters: {best_hp}")
    hp_path = f"{params['RES_DIR']}/best_hyperparameters.json"
    with open(hp_path, "w") as f:
        json.dump(best_hp, f, indent=4)
    logger.info(f"Saved best hyperparameters to {hp_path}")
    logger.info(
        "========== GANDALF SWEEP END ({:.2f}s total) ==========".format(
            time.perf_counter() - run_start
        )
    )


if __name__ == "__main__":
    main()
