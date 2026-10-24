"""Phase 10 deterministic spatial and multi-sample scientific kernels."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from st_agent.scientific import _spatial_key, _write


def _groups(adata, groupby: str | None = None):
    """Never treat overlapping coordinates of independent sections as registered."""
    from st_agent.analysis_context import SECTION_KEYS
    if groupby is None:
        groupby = next((key for key in (*SECTION_KEYS, "sample_id") if key in adata.obs), None)
    if groupby is None:
        return None, [("all", np.arange(adata.n_obs))]
    if groupby not in adata.obs or adata.obs[groupby].isna().any():
        raise ValueError("Spatial grouping column is missing or contains null groups")
    values = adata.obs[groupby].astype(str).to_numpy()
    groups = [(value, np.flatnonzero(values == value)) for value in sorted(set(values))]
    # Choosing "stage" cannot grant coordinate registration across physical sections.
    boundaries = [key for key in (*SECTION_KEYS, "sample_id") if key in adata.obs]
    for key in boundaries:
        if adata.obs[key].isna().any():
            raise ValueError("Spatial section/sample metadata contains missing identifiers")
        for _, indexes in groups:
            if adata.obs[key].iloc[indexes].astype(str).nunique() > 1:
                raise ValueError("Spatial group combines independent sections or samples; provide a unique section grouping")
    return groupby, groups


def _expression(adata, source: str):
    if source == "X":
        return adata.X, adata.var_names
    if source == "raw/X" and adata.raw is not None:
        return adata.raw.X, adata.raw.var_names
    if source.startswith("layers/") and source[7:] in adata.layers:
        return adata.layers[source[7:]], adata.var_names
    raise ValueError(f"Requested expression source is unavailable: {source}")


def _knn(adata, n_neighbors: int, groupby: str | None):
    if not 1 <= n_neighbors <= 100:
        raise ValueError("n_neighbors must be between 1 and 100")
    key = _spatial_key(adata)
    coordinates = np.asarray(adata.obsm[key], dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2 or not np.isfinite(coordinates).all():
        raise ValueError("Spatial coordinates must be finite with at least two dimensions")
    groupby, groups = _groups(adata, groupby)
    rows, cols, distances_all = [], [], []
    group_sizes = {}
    for name, selected in groups:
        if len(selected) <= n_neighbors:
            raise ValueError(f"n_neighbors must be smaller than section size: {name}")
        # Explicit identity removal remains correct for coincident coordinates.
        model = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coordinates[selected, :2])
        distances, indices = model.kneighbors(coordinates[selected, :2])
        for local, (ids, ds) in enumerate(zip(indices, distances)):
            keep = ids != local
            ids, ds = ids[keep][:n_neighbors], ds[keep][:n_neighbors]
            rows.extend([int(selected[local])] * len(ids))
            cols.extend(selected[ids].tolist())
            distances_all.extend(ds.tolist())
        group_sizes[name] = int(len(selected))
    shape = (adata.n_obs, adata.n_obs)
    graph = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape)
    graph = graph.maximum(graph.T).tocsr()
    distances = sparse.csr_matrix((distances_all, (rows, cols)), shape=shape)
    distances = distances.maximum(distances.T).tocsr()
    return graph, distances, {"coordinate_key": key, "method": "symmetric_knn_within_section",
        "n_neighbors": n_neighbors, "groupby": groupby, "group_sizes": group_sizes,
        "coordinate_units": str(adata.uns.get("coordinate_units", "unknown"))}


def _validate_graph(adata, connectivity_key: str):
    if connectivity_key not in adata.obsp:
        raise ValueError("Spatial neighbors must be computed before Moran's I")
    graph = sparse.csr_matrix(adata.obsp[connectivity_key], dtype=float)
    if graph.shape != (adata.n_obs, adata.n_obs) or not np.isfinite(graph.data).all():
        raise ValueError("Invalid spatial graph shape or non-finite weights")
    if (graph.data < 0).any() or graph.diagonal().any() or (graph - graph.T).nnz:
        raise ValueError("Spatial graph must be nonnegative, symmetric and have no self edges")
    groupby, groups = _groups(adata, adata.uns.get("st_agent_spatial_neighbors", {}).get("groupby"))
    for _, indexes in groups:
        if graph[indexes].nnz != graph[indexes][:, indexes].nnz:
            raise ValueError("Spatial graph crosses independent sections; recompute neighbors")
    return graph, groupby, groups


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_spatial_neighbors(
    source: str | Path,
    output: str | Path,
    *,
    n_neighbors: int = 6,
    groupby: str | None = None,
) -> dict[str, Any]:
    adata = ad.read_h5ad(source)
    if adata.n_obs < 2:
        raise ValueError("Spatial neighbors require at least two observations")
    adjacency, distance_matrix, metadata = _knn(adata, n_neighbors, groupby)
    adata.obsp["spatial_connectivities"] = adjacency
    adata.obsp["spatial_distances"] = distance_matrix
    # AnnData/HDF5 cannot encode None in nested metadata portably.
    adata.uns["st_agent_spatial_neighbors"] = {k: v for k, v in metadata.items() if v is not None}
    _write(adata, output)
    return {"observations": int(adata.n_obs), "edges": int(adjacency.nnz), **metadata}


def morans_i(
    source: str | Path,
    *,
    genes: Sequence[str],
    connectivity_key: str = "spatial_connectivities",
    expression_source: str = "X",
) -> dict[str, Any]:
    adata = ad.read_h5ad(source)
    matrix, names = _expression(adata, expression_source)
    if not names.is_unique:
        raise ValueError("Expression gene identifiers must be unique")
    missing = sorted(set(genes) - set(map(str, names)))
    if missing:
        raise ValueError(f"Genes are not present: {', '.join(missing)}")
    weights, groupby, groups = _validate_graph(adata, connectivity_key)
    if weights.nnz == 0:
        raise ValueError("Spatial connectivity matrix has no edges")
    results = []
    for group, indexes in groups:
        graph = weights[indexes][:, indexes]
        total_weight = float(graph.sum())
        if total_weight <= 0:
            raise ValueError(f"Spatial group has no edges: {group}")
        for gene in genes:
            vector = matrix[indexes, names.get_loc(gene)]
            values = vector.toarray().ravel() if sparse.issparse(vector) else np.asarray(vector).ravel()
            if not np.isfinite(values).all():
                raise ValueError("Expression contains non-finite values")
            centered = values.astype(float) - float(np.mean(values))
            denominator = float(centered @ centered)
            score = None if denominator == 0 else float(len(indexes) / total_weight *
                (centered @ (graph @ centered)) / denominator)
            results.append({"gene": gene, "group": group, "observations": int(len(indexes)),
                "morans_i": score, "warning": "constant_expression" if denominator == 0 else None})
    results.sort(key=lambda item: (item["group"],
        -item["morans_i"] if item["morans_i"] is not None else float("inf"), item["gene"]))
    return {"connectivity_key": connectivity_key, "expression_source": expression_source,
        "groupby": groupby, "inference": "descriptive_no_significance_test", "genes": results}


def detect_spatial_domains(
    source: str | Path,
    output: str | Path,
    *,
    n_domains: int,
    seed: int,
    groupby: str | None = None,
) -> dict[str, Any]:
    adata = ad.read_h5ad(source)
    if n_domains < 2:
        raise ValueError("n_domains must be at least two")
    key = _spatial_key(adata)
    coordinates = np.asarray(adata.obsm[key], dtype=float)[:, :2]
    if not np.isfinite(coordinates).all():
        raise ValueError("Spatial coordinates contain non-finite values")
    groupby, groups = _groups(adata, groupby)
    labels = np.empty(adata.n_obs, dtype=object)
    for group, indexes in groups:
        if n_domains > len(indexes) or np.unique(coordinates[indexes], axis=0).shape[0] < n_domains:
            raise ValueError("n_domains cannot exceed observations or unique coordinates in a section")
        current = KMeans(n_clusters=n_domains, random_state=seed, n_init=10).fit_predict(coordinates[indexes])
        labels[indexes] = [f"{group}::{label}" if groupby else str(label) for label in current]
    adata.obs["spatial_domain"] = pd.Categorical(labels)
    adata.obs["coordinate_partition"] = pd.Categorical(labels)
    adata.uns["st_agent_spatial_domains"] = {
        "method": "coordinate_partition_kmeans",
        "n_domains": n_domains,
        "seed": seed,
        "coordinate_key": key,
    }
    _write(adata, output)
    return {"domains": int(len(set(labels))), "domains_per_group": n_domains,
        "observations": int(adata.n_obs), "seed": seed, "groupby": groupby,
        "method": "coordinate_partition_kmeans", "uses_expression": False}


def detect_expression_domains(source: str | Path, output: str | Path, *,
    seed: int, resolution: float = 1.0, n_neighbors: int = 6,
    groupby: str | None = None, representation: str = "X_pca") -> dict[str, Any]:
    """A distinct baseline: expression-weighted section-local spatial graph Leiden."""
    import scanpy as sc
    adata = ad.read_h5ad(source)
    if representation != "X_pca" or representation not in adata.obsm:
        raise ValueError("Expression domains require the selected revision's X_pca representation")
    if "st_agent_preprocess" not in adata.uns:
        raise ValueError("Expression domains require a provenance-recorded preprocessing revision")
    pca = np.asarray(adata.obsm[representation], dtype=float)
    if pca.ndim != 2 or pca.shape[0] != adata.n_obs or pca.shape[1] < 2 or not np.isfinite(pca).all():
        raise ValueError("PCA representation has invalid shape or non-finite values")
    graph, distances, metadata = _knn(adata, n_neighbors, groupby)
    rows, cols = graph.nonzero()
    squared = np.sum((pca[rows] - pca[cols]) ** 2, axis=1)
    positive = squared[squared > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    weights = np.exp(-np.minimum(squared / bandwidth, 50.))
    weighted = sparse.csr_matrix((weights, (rows, cols)), shape=graph.shape)
    if not np.isfinite(weighted.data).all() or weighted.nnz == 0:
        raise ValueError("Invalid expression-weighted spatial graph")
    adata.obsp["spatial_connectivities"] = graph
    adata.obsp["spatial_distances"] = distances
    adata.obsp["spatial_expression_connectivities"] = weighted
    adata.uns["st_agent_spatial_neighbors"] = {k: v for k, v in metadata.items() if v is not None}
    sc.tl.leiden(adata, adjacency=weighted, resolution=resolution, random_state=seed,
        flavor="igraph", n_iterations=2, directed=False, key_added="spatial_expression_domain")
    actual_groupby, groups = _groups(adata, metadata["groupby"])
    labels = adata.obs["spatial_expression_domain"].astype(str).to_numpy(dtype=object)
    if actual_groupby:
        for group, indexes in groups:
            labels[indexes] = [f"{group}::{value}" for value in labels[indexes]]
    adata.obs["spatial_expression_domain"] = pd.Categorical(labels)
    method = "pca_weighted_section_knn_leiden_v1"
    adata.uns["st_agent_expression_domains"] = {"method": method, "representation": representation,
        "seed": seed, "resolution": resolution, "expression_bandwidth_squared": bandwidth,
        "n_neighbors": n_neighbors}
    _write(adata, output)
    return {"method": method, "representation": representation,
        "expression_source": "preprocessed_X_via_X_pca", "domains": int(len(set(labels))),
        "observations": int(adata.n_obs), "edges": int(weighted.nnz), "seed": seed,
        "resolution": resolution, "groupby": actual_groupby,
        "group_domains": {g: int(len(set(labels[idx]))) for g, idx in groups}}


_SAMPLE_PATTERN = re.compile(
    r"^(?P<stage>[A-Za-z][A-Za-z0-9-]*)[_-](?:rep|r)(?P<replicate>\d+)$",
    re.IGNORECASE,
)


def detect_sample_metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    match = _SAMPLE_PATTERN.fullmatch(source.stem)
    if match is None:
        return {
            "status": "ambiguous",
            "sample_id": source.stem,
            "candidates": [],
            "reason": "Expected a name like stageA_rep1.h5ad; researcher confirmation required",
        }
    return {
        "status": "candidate",
        "sample_id": source.stem,
        "stage": match.group("stage"),
        "replicate": int(match.group("replicate")),
        "source": str(source.resolve()),
        "requires_confirmation": True,
    }


class SampleSpec(dict):
    """Runtime sample record: sample_id, stage, replicate, source."""


def merge_samples(
    samples: Sequence[dict[str, Any]],
    output: str | Path,
    *,
    gene_strategy: Literal["intersection", "union"],
) -> dict[str, Any]:
    if len(samples) < 2:
        raise ValueError("Multi-sample merge requires at least two samples")
    required = {"sample_id", "stage", "replicate", "source"}
    sample_ids = [str(item.get("sample_id", "")) for item in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values must be unique")
    loaded: dict[str, ad.AnnData] = {}
    provenance = []
    for item in samples:
        missing = required - set(item)
        if missing:
            raise ValueError(f"Sample metadata missing: {', '.join(sorted(missing))}")
        path = Path(str(item["source"])).resolve()
        current = ad.read_h5ad(path)
        current.obs["sample_id"] = str(item["sample_id"])
        current.obs["stage"] = str(item["stage"])
        current.obs["replicate"] = str(item["replicate"])
        current.obs["source"] = str(path)
        loaded[str(item["sample_id"])] = current
        provenance.append({"sample_id": str(item["sample_id"]), "sha256": _sha256(path)})
    join = "inner" if gene_strategy == "intersection" else "outer"
    merged = ad.concat(loaded, join=join, label=None, index_unique="::", fill_value=0)
    merged.uns["st_agent_merge"] = {
        "gene_strategy": gene_strategy,
        "sample_hashes": {item["sample_id"]: item["sha256"] for item in provenance},
        "gene_truncation": False,
    }
    _write(merged, output)
    return {
        "samples": len(samples),
        "observations": int(merged.n_obs),
        "genes": int(merged.n_vars),
        "gene_strategy": gene_strategy,
        "input_hashes": provenance,
    }


def compare_stages(source: str | Path, *, stage_column: str = "stage",
    genes: Sequence[str] = (), cluster_key: str | None = None,
    labels_aligned: bool = False, expression_source: str = "X") -> dict[str, Any]:
    adata = ad.read_h5ad(source)
    if not stage_column.strip() or stage_column not in adata.obs:
        raise ValueError("Comparison metadata missing: explicit grouping column")
    grouping = adata.obs[stage_column]
    if grouping.isna().any() or grouping.astype(str).str.strip().eq("").any():
        raise ValueError("Comparison metadata contains missing group values")
    if grouping.nunique() < 2:
        raise ValueError("Cross-stage comparison requires at least two recorded groups")
    # Section-level summaries need grouping, not verified donor provenance.
    # Optional identifiers describe recorded labels only, never independence.
    availability = {}
    for key in ("sample_id", "replicate", "source"):
        availability[key] = ("missing" if key not in adata.obs else "incomplete"
            if adata.obs[key].isna().any() or adata.obs[key].astype(str).str.strip().eq("").any()
            else "complete")
    if cluster_key is not None and (cluster_key not in adata.obs or not labels_aligned):
        raise ValueError("Composition comparison requires explicitly aligned cluster labels")
    from st_agent.scientific import _qc_vectors
    vectors, count_source, _ = _qc_vectors(adata)
    matrix, names = _expression(adata, expression_source)
    if set(genes) - set(names):
        raise ValueError("Comparison genes are unavailable in selected expression source")
    if not names.is_unique:
        raise ValueError("Expression gene identifiers must be unique")
    rows, composition, expression = [], [], []
    for stage, frame in adata.obs.groupby(stage_column, observed=True, sort=True):
        indexes = adata.obs_names.get_indexer(frame.index)
        qc = {"mean_total_counts": float(np.mean(vectors["total_counts"][indexes])),
            "mean_detected_genes": float(np.mean(vectors["n_genes_by_counts"][indexes]))}
        mito = vectors["pct_counts_mt"]
        qc["mean_mito_pct"] = None if mito is None else float(np.mean(mito[indexes]))
        rows.append(
            {
                "stage": str(stage),
                "observations": int(len(frame)),
                "samples": int(frame["sample_id"].nunique())
                    if availability["sample_id"] == "complete" else None,
                "replicates": int(frame["replicate"].nunique())
                    if availability["replicate"] == "complete" else None,
                "qc": qc,
            }
        )
        if cluster_key:
            if frame[cluster_key].isna().any():
                raise ValueError("Aligned cluster labels contain missing values")
            for label, count in frame[cluster_key].astype(str).value_counts().sort_index().items():
                composition.append({"stage": str(stage), "cluster": str(label),
                    "count": int(count), "fraction": float(count / len(frame))})
        for gene in genes:
            values = matrix[indexes, names.get_loc(gene)]
            values = values.toarray().ravel() if sparse.issparse(values) else np.asarray(values).ravel()
            if not np.isfinite(values).all():
                raise ValueError("Selected expression contains non-finite values")
            expression.append({"stage": str(stage), "gene": gene,
                "mean_expression": float(np.mean(values)), "fraction_expressing": float(np.mean(values > 0))})
    if len(rows) < 2:
        raise ValueError("Cross-stage comparison requires at least two confirmed stages")
    return {"stage_column": stage_column, "groups": rows, "count_source": count_source,
        "expression_source": expression_source, "composition": composition,
        "expression": expression, "cluster_key": cluster_key, "labels_aligned": labels_aligned,
        "metadata_availability": availability,
        "replicate_count_scope": "distinct_recorded_replicate_labels_not_verified_biological_independence",
        "inference": "descriptive_no_population_inference_or_batch_correction"}


class StructuredProvider(Protocol):
    def __call__(self, query: dict[str, Any]) -> list[dict[str, Any]]: ...


def query_structured_provider(
    query: dict[str, Any], provider: StructuredProvider, *, required_fields: Sequence[str]
) -> dict[str, Any]:
    """Validate enrichment, MyGene, PubMed, or ligand-receptor provider output."""
    try:
        records = provider(query)
    except (OSError, TimeoutError) as error:
        raise ConnectionError("External scientific provider is unavailable") from error
    if not isinstance(records, list):
        raise ValueError("Provider response must be a list")
    required = set(required_fields)
    for record in records:
        if not isinstance(record, dict) or not required <= set(record):
            raise ValueError("Provider response failed structural validation")
    return {"query": query, "records": records, "record_count": len(records)}
