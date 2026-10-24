"""Typed Phase 10 admission registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from st_agent.advanced_scientific import (
    build_spatial_neighbors,
    compare_stages,
    detect_sample_metadata,
    detect_spatial_domains,
    detect_expression_domains,
    merge_samples,
    morans_i,
)
from st_agent.contracts import Admission, Effect
from st_agent.knowledge_tools import (
    annotate_genes,
    compute_ligand_receptor_scores,
    run_enrichment,
    search_pubmed,
)


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NeighborArguments(Arguments):
    n_neighbors: int = Field(default=6, ge=1, le=100)
    groupby: str | None = Field(default=None, min_length=1)


class MoranArguments(Arguments):
    genes: tuple[str, ...] = Field(min_length=1, max_length=500)
    connectivity_key: str = Field(default="spatial_connectivities", min_length=1)
    expression_source: str = Field(default="X", pattern=r"^(X|raw/X|layers/[^/]+)$")


class DomainArguments(Arguments):
    n_domains: int = Field(ge=2, le=100)
    seed: int = Field(ge=0, le=2**32 - 1)
    groupby: str | None = Field(default=None, min_length=1)


class ExpressionDomainArguments(NeighborArguments):
    seed: int = Field(ge=0, le=2**32 - 1)
    resolution: float = Field(default=1.0, gt=0, le=100)
    representation: Literal["X_pca"] = "X_pca"


class SampleDetectArguments(Arguments):
    pass


class SampleInput(Arguments):
    sample_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    replicate: str | int
    source: str = Field(min_length=1)


class MergeArguments(Arguments):
    samples: tuple[SampleInput, ...] = Field(min_length=2)
    gene_strategy: Literal["intersection", "union"]


class CompareArguments(Arguments):
    stage_column: str = Field(default="stage", min_length=1, description=(
        "The obs column defining comparison groups, such as section, sample, or stage. "
        "Use the requested grouping column here; for comparison by chip, set stage_column='chip'."
    ))
    genes: tuple[str, ...] = Field(default=(), max_length=500)
    cluster_key: str | None = Field(default=None, min_length=1, description=(
        "Optional obs column of within-group cluster or cell-type labels, used for composition comparison. "
        "It cannot replace stage_column or define the comparison groups. Leave null for QC and gene-expression "
        "comparison without composition. Only select it when common labels across groups are explicitly "
        "confirmed with labels_aligned=true."
    ))
    labels_aligned: bool = False
    expression_source: str = Field(default="X", pattern=r"^(X|raw/X|layers/[^/]+)$")


class ProviderArguments(Arguments):
    query: dict[str, Any]


class EnrichmentArguments(Arguments):
    genes: tuple[str, ...] = Field(min_length=1, max_length=500)
    species: str = Field(min_length=1)
    database: str = Field(min_length=1)


class MyGeneArguments(Arguments):
    genes: tuple[str, ...] = Field(min_length=1, max_length=500)
    species: str = Field(min_length=1)


class PubMedArguments(Arguments):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)


class InteractionInput(Arguments):
    ligand: str = Field(min_length=1)
    receptor: str = Field(min_length=1)
    source: str = Field(min_length=1)


class LigandReceptorArguments(Arguments):
    cluster_key: str = Field(min_length=1)
    interactions: tuple[InteractionInput, ...] = Field(min_length=1, max_length=100_000)
    min_expression: float = Field(default=0.0, ge=0)


@dataclass(frozen=True)
class AdvancedTool:
    name: str
    admission: Admission
    effect: Effect
    arguments: type[Arguments]
    implementation: Any
    admission_reason: str


ADVANCED_TOOLS = {
    tool.name: tool
    for tool in (
        AdvancedTool("spatial.neighbors", Admission.VERIFIED, Effect.DERIVED_REVISION,
                     NeighborArguments, build_spatial_neighbors, "deterministic kernel tested"),
        AdvancedTool("spatial.morans_i", Admission.VERIFIED, Effect.READ_ONLY,
                     MoranArguments, morans_i, "formula and edge cases tested"),
        AdvancedTool("spatial.domains", Admission.VERIFIED, Effect.DERIVED_REVISION,
                     DomainArguments, detect_spatial_domains, "coordinate partition only; no expression inference"),
        AdvancedTool("samples.detect", Admission.VERIFIED, Effect.READ_ONLY,
                     SampleDetectArguments, detect_sample_metadata, "never auto-confirms groups"),
        AdvancedTool("samples.merge", Admission.VERIFIED, Effect.DERIVED_REVISION,
                     MergeArguments, merge_samples, "explicit join and provenance tested"),
        AdvancedTool("samples.compare_stages", Admission.VERIFIED, Effect.READ_ONLY,
                     CompareArguments, compare_stages, "confirmed metadata required"),
        AdvancedTool("report.generate", Admission.VERIFIED, Effect.DERIVED_REVISION,
                     ProviderArguments, None, "admitted through Phase 8 ReportStore"),
        AdvancedTool("knowledge.enrichment", Admission.CANDIDATE, Effect.EXTERNAL_IO,
                     EnrichmentArguments, run_enrichment, "provider corpus not selected"),
        AdvancedTool("knowledge.mygene", Admission.CANDIDATE, Effect.EXTERNAL_IO,
                     MyGeneArguments, annotate_genes, "live service gate pending"),
        AdvancedTool(
            "knowledge.pubmed", Admission.CANDIDATE, Effect.EXTERNAL_IO,
            PubMedArguments, search_pubmed, "RAG review intentionally deferred"
        ),
        AdvancedTool("communication.ligand_receptor", Admission.CANDIDATE, Effect.EXTERNAL_IO,
                     LigandReceptorArguments, compute_ligand_receptor_scores,
                     "curated database not selected"),
    )
}
