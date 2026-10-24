"""Typed registry for the eight Phase 3 scientific tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from st_agent.contracts import Admission, Effect
from st_agent.scientific import (
    apply_qc_filter,
    cluster_leiden_umap,
    compute_qc_metrics,
    inspect_dataset,
    preprocess_standard,
    rank_markers,
    spatial_cluster_data,
    spatial_gene_data,
)


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NoArguments(Arguments):
    pass


class QCFilterArguments(Arguments):
    min_genes: int = Field(default=200, ge=0)
    min_cells: int = Field(default=3, ge=1)
    max_mito_pct: float = Field(default=20.0, ge=0, le=100)
    min_counts: int | None = Field(default=None, ge=0)


class PreprocessArguments(Arguments):
    n_hvg: int = Field(default=2000, ge=2)
    n_pcs: int = Field(default=30, ge=2)
    target_sum: float = Field(default=10_000, gt=0)


class ClusterArguments(Arguments):
    seed: int = Field(ge=0, le=2**32 - 1)
    n_neighbors: int = Field(default=15, ge=2)
    resolution: float = Field(default=1.0, gt=0)


class MarkerArguments(Arguments):
    n_genes: int = Field(default=10, ge=1)
    method: str = Field(default="wilcoxon", pattern=r"^(wilcoxon|t-test)$")


class PlotClusterArguments(Arguments):
    max_points: int = Field(default=20_000, ge=1, le=100_000)


class PlotGeneArguments(PlotClusterArguments):
    gene: str = Field(min_length=1)


@dataclass(frozen=True)
class ScientificTool:
    name: str
    admission: Admission
    effect: Effect
    arguments: type[Arguments]
    implementation: Any


TOOLS = {
    item.name: item
    for item in (
        ScientificTool(
            "dataset.inspect", Admission.VERIFIED, Effect.READ_ONLY, NoArguments, inspect_dataset
        ),
        ScientificTool(
            "qc.compute_metrics",
            Admission.VERIFIED,
            Effect.READ_ONLY,
            NoArguments,
            compute_qc_metrics,
        ),
        ScientificTool(
            "qc.apply_filter",
            Admission.VERIFIED,
            Effect.DERIVED_REVISION,
            QCFilterArguments,
            apply_qc_filter,
        ),
        ScientificTool(
            "preprocess.standard",
            Admission.VERIFIED,
            Effect.DERIVED_REVISION,
            PreprocessArguments,
            preprocess_standard,
        ),
        ScientificTool(
            "cluster.leiden_umap",
            Admission.VERIFIED,
            Effect.DERIVED_REVISION,
            ClusterArguments,
            cluster_leiden_umap,
        ),
        ScientificTool(
            "markers.rank",
            Admission.VERIFIED,
            Effect.DERIVED_REVISION,
            MarkerArguments,
            rank_markers,
        ),
        ScientificTool(
            "plot.spatial_clusters",
            Admission.VERIFIED,
            Effect.READ_ONLY,
            PlotClusterArguments,
            spatial_cluster_data,
        ),
        ScientificTool(
            "plot.spatial_gene",
            Admission.VERIFIED,
            Effect.READ_ONLY,
            PlotGeneArguments,
            spatial_gene_data,
        ),
    )
}


def execute_scientific_tool(
    name: str,
    source: str | Path,
    *,
    output: str | Path | None = None,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a validated tool directly; Phase 4 can call the same registry."""
    from st_agent.tool_registry import execute_registered_tool
    return execute_registered_tool(name, source, output=output, arguments=arguments)
