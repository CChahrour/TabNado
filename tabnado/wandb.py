import os
from dataclasses import dataclass

import wandb as _wandb
import wandb_workspaces.reports.v2 as wr


@dataclass
class WandbConfig:
    """Centralised W&B settings shared across all pipeline stages."""

    project: str
    entity: str | None
    model_name: str
    target: str
    res_dir: str

    @classmethod
    def from_params(cls, params: dict) -> "WandbConfig":
        return cls(
            project=params["PROJECT"],
            entity=params.get("ENTITY"),
            model_name=params.get("MODEL_TYPE", "model"),
            target=params["TARGET"],
            res_dir=params["RES_DIR"],
        )

    def init_run(
        self,
        name: str,
        group: str,
        config: dict | None = None,
        reinit: str | None = None,
        dir_override: str | None = None,
    ):
        """Call wandb.init with project/entity/dir from this config."""
        os.environ.setdefault("WANDB_DIR", self.res_dir)
        kwargs: dict = dict(
            project=self.project,
            entity=self.entity,
            name=name,
            group=group,
            config=config or {},
            dir=dir_override or self.res_dir,
        )
        if reinit is not None:
            kwargs["reinit"] = reinit
        return _wandb.init(**kwargs)


def create_eval_report(
    wandb_cfg: WandbConfig,
    run_id: str,
    target_cols: list[str],
) -> str:
    """Create a W&B report for a project, scoped to a specific eval run.

    Returns the report URL.
    """
    resolved_entity = wandb_cfg.entity or _wandb.Api().default_entity
    runset = wr.Runset(project=wandb_cfg.project, entity=resolved_entity)

    api_run = _wandb.Api().run(f"{resolved_entity}/{wandb_cfg.project}/{run_id}")
    run_name = api_run.name

    macro_panels = [
        wr.ScalarChart(title="R² (macro)", metric=wr.SummaryMetric("eval/R2_macro")),
        wr.ScalarChart(title="MSE (macro)", metric=wr.SummaryMetric("eval/MSE_macro")),
        wr.ScalarChart(title="MAE (macro)", metric=wr.SummaryMetric("eval/MAE_macro")),
    ]
    per_target_panels = []
    for col in target_cols:
        for stat in ("R2", "MSE", "MAE", "Rho"):
            per_target_panels.append(
                wr.ScalarChart(
                    title=f"{col} — {stat}",
                    metric=wr.SummaryMetric(f"eval/{col}/{stat}"),
                )
            )

    scatter_browser = wr.MediaBrowser(
        title="Scatter plots (true vs predicted)",
        media_keys=[f"eval/scatter_{col}" for col in target_cols],
        num_columns=min(len(target_cols), 3),
        mode="gallery",
    )
    umap_browser = wr.MediaBrowser(
        title="UMAP embeddings",
        media_keys=["eval/umap"],
        num_columns=1,
    )
    clustermap_browser = wr.MediaBrowser(
        title="SHAP clustermap",
        media_keys=["shap/clustermap"],
        num_columns=1,
    )
    spatial_heatmap_browser = wr.MediaBrowser(
        title="Spatial SHAP heatmaps",
        media_keys=[
            f"shap/spatial_heatmap_{col.replace('/', '_')}" for col in target_cols
        ],
        num_columns=min(len(target_cols), 2),
        mode="gallery",
    )
    offset_line_browser = wr.MediaBrowser(
        title="Genomic distance importance profiles",
        media_keys=[f"shap/offset_line_{col.replace('/', '_')}" for col in target_cols],
        num_columns=min(len(target_cols), 2),
        mode="gallery",
    )

    report = wr.Report(
        entity=resolved_entity,
        project=wandb_cfg.project,
        title=f"Evaluation: {run_name}",
        description=f"Targets: {', '.join(target_cols)}",
    )
    report.blocks = [
        wr.TableOfContents(),
        wr.H1(text="Metrics"),
        wr.H2(text="Macro"),
        wr.PanelGrid(runsets=[runset], panels=macro_panels),
        wr.H2(text="Per target"),
        wr.PanelGrid(runsets=[runset], panels=per_target_panels),
        wr.H1(text="Evaluation plots"),
        wr.PanelGrid(runsets=[runset], panels=[scatter_browser, umap_browser]),
        wr.H1(text="SHAP"),
        wr.PanelGrid(runsets=[runset], panels=[clustermap_browser]),
        wr.H2(text="Spatial analysis"),
        wr.PanelGrid(
            runsets=[runset], panels=[spatial_heatmap_browser, offset_line_browser]
        ),
    ]
    report.save()
    return report.url
