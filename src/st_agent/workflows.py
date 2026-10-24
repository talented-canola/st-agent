"""Deterministic Phase 3 workflows that do not require an LLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from st_agent.scientific_tools import execute_scientific_tool


def inspect_workflow(source: str | Path) -> dict[str, Any]:
    return {"inspect": execute_scientific_tool("dataset.inspect", source)}


def qc_workflow(
    source: str | Path, workdir: str | Path, *, filters: dict[str, Any]
) -> dict[str, Any]:
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    filtered = root / "02-qc-filtered.h5ad"
    return {
        "inspect": execute_scientific_tool("dataset.inspect", source),
        "metrics": execute_scientific_tool("qc.compute_metrics", source),
        "filter": execute_scientific_tool(
            "qc.apply_filter", source, output=filtered, arguments=filters
        ),
        "output": str(filtered),
    }


def standard_analysis_workflow(
    source: str | Path,
    workdir: str | Path,
    *,
    filters: dict[str, Any],
    preprocess: dict[str, Any],
    clustering: dict[str, Any],
    markers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(workdir)
    qc = qc_workflow(source, root, filters=filters)
    processed = root / "03-processed.h5ad"
    clustered = root / "04-clustered.h5ad"
    marked = root / "05-markers.h5ad"
    result = {"qc": qc}
    result["preprocess"] = execute_scientific_tool(
        "preprocess.standard", qc["output"], output=processed, arguments=preprocess
    )
    result["cluster"] = execute_scientific_tool(
        "cluster.leiden_umap", processed, output=clustered, arguments=clustering
    )
    result["markers"] = execute_scientific_tool(
        "markers.rank", clustered, output=marked, arguments=markers or {}
    )
    result["output"] = str(marked)
    return result


def spatial_review_workflow(
    source: str | Path,
    workdir: str | Path,
    *,
    gene: str,
    filters: dict[str, Any],
    preprocess: dict[str, Any],
    clustering: dict[str, Any],
) -> dict[str, Any]:
    result = standard_analysis_workflow(
        source,
        workdir,
        filters=filters,
        preprocess=preprocess,
        clustering=clustering,
    )
    result["spatial_clusters"] = execute_scientific_tool("plot.spatial_clusters", result["output"])
    result["spatial_gene"] = execute_scientific_tool(
        "plot.spatial_gene", result["output"], arguments={"gene": gene}
    )
    return result
