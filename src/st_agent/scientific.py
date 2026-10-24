"""Sparse-safe, deterministic scientific kernel for the first admitted tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from st_agent.analysis_context import analysis_context, load_analysis_metadata


COUNT_INTEGER_ATOL = 1e-6
COUNT_BLOCK_VALUES = 1_000_000


def _validate_count_blocks(blocks, *, source: str, shape, stored_values_only: bool) -> dict[str, Any]:
    """Scan every stored value, using an absolute (not relative) integer tolerance."""
    checked = 0
    invalid = {"non_finite": 0, "negative": 0, "fractional": 0, "unsupported_dtype": 0}
    for block in blocks:
        values = np.asarray(block).reshape(-1)
        checked += int(values.size)
        if values.dtype.kind not in "biuf":
            invalid["unsupported_dtype"] += int(values.size)
            continue
        finite = np.isfinite(values)
        invalid["non_finite"] += int((~finite).sum())
        invalid["negative"] += int((finite & (values < 0)).sum())
        if values.dtype.kind == "f":
            finite_values = values[finite]
            invalid["fractional"] += int((np.abs(finite_values - np.rint(finite_values)) > COUNT_INTEGER_ATOL).sum())
    violations = [key for key, number in invalid.items() if number]
    return {
        "source": source, "valid": not violations,
        "shape": [int(value) for value in shape], "checked_values": checked,
        "stored_values_only": stored_values_only, "integer_atol": COUNT_INTEGER_ATOL,
        "finite": not invalid["non_finite"], "nonnegative": not invalid["negative"],
        "integer_like": not (invalid["fractional"] or invalid["unsupported_dtype"]),
        "invalid_values": invalid, "violations": violations,
        "scope": "value validation; integer-like values do not alone establish original count provenance",
    }


def validate_count_source_file(path: str | Path, count_source: str) -> dict[str, Any]:
    """Validate dense or encoded sparse H5AD counts without loading the matrix."""
    with h5py.File(path, "r") as h5:
        if count_source not in h5:
            raise ValueError(f"Selected count source is missing: {count_source}")
        item = h5[count_source]
        if isinstance(item, h5py.Group):
            if "data" not in item:
                raise ValueError(f"Unsupported count source encoding: {count_source}")
            values = item["data"]
            blocks = (values[start:start + COUNT_BLOCK_VALUES] for start in range(0, values.shape[0], COUNT_BLOCK_VALUES))
            return _validate_count_blocks(blocks, source=count_source, shape=item.attrs["shape"], stored_values_only=True)
        if item.ndim != 2:
            raise ValueError("Count matrix must have two dimensions")
        rows = max(1, COUNT_BLOCK_VALUES // max(1, item.shape[1]))
        blocks = (item[start:start + rows, :] for start in range(0, item.shape[0], rows))
        return _validate_count_blocks(blocks, source=count_source, shape=item.shape, stored_values_only=False)


def _validate_count_matrix(matrix, count_source: str) -> dict[str, Any]:
    if matrix is None or len(matrix.shape) != 2:
        raise ValueError("Selected count matrix is missing or not two-dimensional")
    if sparse.issparse(matrix):
        data = matrix.data
        blocks = (data[start:start + COUNT_BLOCK_VALUES] for start in range(0, data.size, COUNT_BLOCK_VALUES))
        return _validate_count_blocks(blocks, source=count_source, shape=matrix.shape, stored_values_only=True)
    rows = max(1, COUNT_BLOCK_VALUES // max(1, matrix.shape[1]))
    blocks = (matrix[start:start + rows, :] for start in range(0, matrix.shape[0], rows))
    return _validate_count_blocks(blocks, source=count_source, shape=matrix.shape, stored_values_only=False)


def _require_counts(matrix, count_source: str) -> None:
    validation = _validate_count_matrix(matrix, count_source)
    if not validation["valid"]:
        raise ValueError(
            f"Selected count source {count_source} must contain finite, nonnegative integer-like counts "
            f"(absolute integer tolerance {COUNT_INTEGER_ATOL}); violations: {','.join(validation['violations'])}"
        )

def _descriptor(h5: h5py.File, key: str) -> dict[str, Any] | None:
    if key not in h5:
        return None
    item = h5[key]
    encoding = item.attrs.get("encoding-type", "array")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    shape = item.attrs.get("shape", item.shape if hasattr(item, "shape") else ())
    dtype = item["data"].dtype if isinstance(item, h5py.Group) else item.dtype
    return {
        "encoding": str(encoding),
        "sparse": str(encoding) in {"csr_matrix", "csc_matrix"},
        "shape": [int(value) for value in shape],
        "dtype": str(dtype),
    }


def inspect_dataset(path: str | Path) -> dict[str, Any]:
    """Inspect an H5AD without materializing its expression matrix."""
    source = Path(path).resolve()
    with h5py.File(source, "r") as h5:
        matrix = _descriptor(h5, "X")
        counts = _descriptor(h5, "layers/counts")
        raw = _descriptor(h5, "raw/X")
    backed = load_analysis_metadata(source)
    try:
        count_source = "layers/counts" if counts else "raw/X" if raw else "X"
        return {
            "shape": [int(backed.n_obs), int(backed.n_vars)],
            "size_bytes": source.stat().st_size,
            "matrix": matrix,
            "count_source": count_source,
            "count_validation": validate_count_source_file(source, count_source),
            "analysis_context": analysis_context(backed),
            "layers": sorted(map(str, backed.layers.keys())),
            "obs_columns": sorted(map(str, backed.obs.columns)),
            "var_columns": sorted(map(str, backed.var.columns)),
            "obsm_keys": sorted(map(str, backed.obsm.keys())),
            "spatial_keys": sorted(
                k for k in map(str, backed.obsm.keys()) if "spatial" in k.lower()
            ),
            "raw_present": backed.raw is not None,
            "obs_names_unique": bool(backed.obs_names.is_unique),
            "var_names_unique": bool(backed.var_names.is_unique),
        }
    finally:
        del backed


def _count_matrix(adata: ad.AnnData) -> tuple[Any, np.ndarray, str]:
    """Return a count matrix without mutating or rebuilding the source AnnData.

    Rebuilding an AnnData from ``raw.X`` discards ``raw``, layers and graph
    metadata.  That was the source of a serious provenance bug in the live
    path: a QC-only request could make a processed matrix look like raw counts.
    """
    if "counts" in adata.layers:
        names = np.asarray(adata.var_names.astype(str), dtype=str)
        _require_counts(adata.layers["counts"], "layers/counts")
        return adata.layers["counts"], names, "layers/counts"
    if adata.raw is not None:
        _require_counts(adata.raw.X, "raw/X")
        return adata.raw.X, np.asarray(adata.raw.var_names.astype(str), dtype=str), "raw/X"
    if "log1p" in adata.uns or "st_agent_preprocess" in adata.uns:
        raise ValueError("Selected count source X is declared processed; a verified counts layer or raw count source is required")
    _require_counts(adata.X, "X")
    return adata.X, np.asarray(adata.var_names.astype(str), dtype=str), "X"


def _matrix_for_current_vars(adata: ad.AnnData) -> tuple[Any, str]:
    """Align the selected count source to the current feature space safely."""
    matrix, source_names, count_source = _count_matrix(adata)
    target_names = np.asarray(adata.var_names.astype(str), dtype=str)
    if len(source_names) == len(target_names) and np.array_equal(source_names, target_names):
        return matrix, count_source
    positions = {name: index for index, name in enumerate(source_names)}
    if len(positions) != len(source_names):
        raise ValueError("Count-source gene names must be unique before filtering")
    missing = [name for name in target_names if name not in positions]
    if missing:
        raise ValueError("Selected count source cannot be aligned to current genes")
    return matrix[:, [positions[name] for name in target_names]], count_source


def _mitochondrial_mask(adata: ad.AnnData, var_names: np.ndarray, count_source: str) -> tuple[np.ndarray, str]:
    annotation = adata.raw.var if count_source == "raw/X" else adata.var
    prefix = "raw.var" if count_source == "raw/X" else "var"
    columns = {str(name).lower(): str(name) for name in annotation.columns}
    if "mt" in columns:
        values = annotation[columns["mt"]]
        mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
        if pd.api.types.is_bool_dtype(values):
            mask = values.fillna(False).to_numpy(dtype=bool)
        else:
            def parse_flag(value):
                if pd.isna(value):
                    return False
                if isinstance(value, (bool, np.bool_, int, np.integer, float, np.floating)) and value in (0, 1):
                    return bool(value)
                return mapping.get(str(value).strip().lower())
            parsed = values.map(parse_flag)
            if parsed.isna().any():
                raise ValueError("Mitochondrial var.mt annotation must contain booleans or explicit 0/1 values")
            mask = parsed.to_numpy(dtype=bool)
        return mask, f"{prefix}.{columns['mt']}"
    for candidate in ("gene_symbol", "gene_symbols", "symbol", "gene_name", "feature_name"):
        if candidate in columns:
            symbols = annotation[columns[candidate]].fillna("").astype(str).str.lower()
            mask = symbols.str.startswith("mt-").to_numpy()
            if bool(mask.any()):
                return mask, f"{prefix}.{columns[candidate]}"
    for candidate in ("chromosome", "chrom", "chr"):
        if candidate in columns:
            chromosomes = annotation[columns[candidate]].fillna("").astype(str).str.upper().str.replace(r"^CHR", "", regex=True)
            mask = chromosomes.isin(["M", "MT"]).to_numpy()
            if bool(mask.any()):
                return mask, f"{prefix}.{columns[candidate]}"
    return np.char.startswith(np.char.lower(var_names), "mt-"), "gene_names.mt_prefix"


def _qc_vectors(adata: ad.AnnData) -> tuple[dict[str, np.ndarray | None], str, int]:
    matrix, var_names, count_source = _count_matrix(adata)
    mitochondrial, _ = _mitochondrial_mask(adata, var_names, count_source)
    total_counts = np.asarray(matrix.sum(axis=1, dtype=np.float64)).reshape(-1)
    if not bool(np.isfinite(total_counts).all()):
        raise ValueError("Count totals overflowed; QC cannot emit non-finite summaries")
    n_genes = np.asarray((matrix != 0).sum(axis=1)).reshape(-1)
    pct_mito = None
    if bool(mitochondrial.any()):
        mitochondrial_counts = np.asarray(matrix[:, mitochondrial].sum(axis=1, dtype=np.float64)).reshape(-1)
        pct_mito = np.divide(
            mitochondrial_counts * 100.0, total_counts,
            out=np.zeros_like(total_counts, dtype=float), where=total_counts > 0,
        )
    return (
        {
            "total_counts": total_counts,
            "n_genes_by_counts": n_genes,
            "pct_counts_mt": pct_mito,
        },
        count_source,
        int(mitochondrial.sum()),
    )


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("QC distributions require at least one observation")
    minimum, q1, median, q3, maximum = np.percentile(values, [0, 25, 50, 75, 100])
    return {
        "min": float(minimum),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(maximum),
        "iqr": float(q3 - q1),
    }


def _write(adata: ad.AnnData, output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.h5ad")
    # Interactive revisions favor latency. Archival compression is a separate concern.
    adata.write_h5ad(temporary)
    temporary.replace(destination)


def compute_qc_metrics(source: str | Path) -> dict[str, Any]:
    """Summarize QC from the selected count source without creating a revision."""
    adata = ad.read_h5ad(source)
    vectors, count_source, mitochondrial_genes = _qc_vectors(adata)
    _, var_names, _ = _count_matrix(adata)
    _, mitochondrial_annotation_source = _mitochondrial_mask(adata, var_names, count_source)
    context = analysis_context(adata)
    warnings = [] if mitochondrial_genes else [
        "No mitochondrial genes could be identified from the selected count-source annotations; mitochondrial percentage is unavailable, not measured zero."
    ]
    qc_by = {}
    grouping = list(context["sections"]["candidates"])
    grouping.extend(key for key in ("celltype", "cell_type") if key in adata.obs and key not in grouping)
    for key in grouping:
        series = adata.obs[key]
        labels = series.astype(object).where(series.notna(), "<missing>").astype(str).to_numpy()
        qc_by[key] = {}
        for label in sorted(set(labels)):
            selected = labels == label
            qc_by[key][label] = {
                "observations": int(selected.sum()),
                "n_genes_by_counts": _distribution(vectors["n_genes_by_counts"][selected]),
                "total_counts": _distribution(vectors["total_counts"][selected]),
                "pct_counts_mt": _distribution(vectors["pct_counts_mt"][selected]) if mitochondrial_genes else None,
            }
    return {
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "mitochondrial_genes": mitochondrial_genes,
        "mitochondrial_metric_available": bool(mitochondrial_genes),
        "mitochondrial_annotation_source": mitochondrial_annotation_source,
        "warnings": warnings, "analysis_context": context, "qc_by": qc_by,
        "zero_count_cells": int((vectors["total_counts"] == 0).sum()),
        "count_source": count_source,
        "n_genes_by_counts": _distribution(vectors["n_genes_by_counts"]),
        "total_counts": _distribution(vectors["total_counts"]),
        "pct_counts_mt": _distribution(vectors["pct_counts_mt"]) if mitochondrial_genes else None,
    }


def apply_qc_filter(
    source: str | Path,
    output: str | Path,
    *,
    min_genes: int,
    min_cells: int,
    max_mito_pct: float,
    min_counts: int | None = None,
) -> dict[str, Any]:
    adata = ad.read_h5ad(source)
    vectors, count_source, mitochondrial_genes = _qc_vectors(adata)
    if not mitochondrial_genes and max_mito_pct < 100:
        raise ValueError("Mitochondrial percentage is unavailable: annotate mitochondrial genes before applying max_mito_pct below 100")
    failures = {"below_min_genes": vectors["n_genes_by_counts"] < min_genes}
    failures["above_max_mito_pct"] = vectors["pct_counts_mt"] > max_mito_pct if mitochondrial_genes else np.zeros(adata.n_obs, dtype=bool)
    if min_counts is not None:
        failures["below_min_counts"] = vectors["total_counts"] < min_counts
    else:
        failures["below_min_counts"] = np.zeros(adata.n_obs, dtype=bool)
    failed_criteria = sum(values.astype(int) for values in failures.values())
    keep_cells = failed_criteria == 0
    if not bool(keep_cells.any()):
        raise ValueError("QC thresholds would remove every cell")
    filtered = adata[keep_cells].copy()
    count_matrix, aligned_source = _matrix_for_current_vars(filtered)
    detected = np.asarray((count_matrix != 0).sum(axis=0)).reshape(-1)
    keep_genes = detected >= min_cells
    if not bool(keep_genes.any()):
        raise ValueError("QC thresholds would remove every gene")
    filtered = filtered[:, keep_genes].copy()
    context = analysis_context(adata)
    filter_reasons = {key: int(values.sum()) for key, values in failures.items()}
    filter_reasons.update({
        "excluded_union": int((~keep_cells).sum()), "overlap_cells": int((failed_criteria > 1).sum()),
        "below_min_cells_genes": int((~keep_genes).sum()),
    })
    grouping = list(context["sections"]["candidates"])
    grouping.extend(key for key in ("celltype", "cell_type") if key in adata.obs and key not in grouping)
    retention_by = {}
    for key in grouping:
        series = adata.obs[key]
        labels = series.astype(object).where(series.notna(), "<missing>").astype(str).to_numpy()
        retention_by[key] = {}
        for label in sorted(set(labels)):
            selected = labels == label
            before, after = int(selected.sum()), int((selected & keep_cells).sum())
            retention_by[key][label] = {"before": before, "after": after, "retention_rate": after / before}
    audit = {
        "filter_reasons": filter_reasons, "retention_by": retention_by,
        "count_source": count_source, "aligned_count_source": aligned_source,
        "mitochondrial_metric_available": bool(mitochondrial_genes),
        "unapplied_criteria": [] if mitochondrial_genes else ["max_mito_pct"],
        "observation_audit": [
            {"observation_id": str(name), "retained": bool(keep_cells[index]),
             "reasons": [key for key, values in failures.items() if bool(values[index])]}
            for index, name in enumerate(adata.obs_names)
        ],
        "gene_audit": [
            {"gene_id": str(name), "detected_after_cell_filter": int(detected[index]),
             "retained": bool(keep_genes[index]), "reasons": [] if keep_genes[index] else ["below_min_cells"]}
            for index, name in enumerate(adata.var_names)
        ],
    }
    filtered.uns["st_agent_qc_filter"] = {
        "min_genes": min_genes,
        "min_cells": min_cells,
        "max_mito_pct": max_mito_pct,
        "min_counts": min_counts if min_counts is not None else -1,
        "count_source": count_source,
        "aligned_count_source": aligned_source,
        "audit_json": json.dumps(audit, ensure_ascii=False, allow_nan=False),
    }
    filtered.uns["st_agent_analysis_context_json"] = json.dumps(analysis_context(filtered), ensure_ascii=False, allow_nan=False)
    _write(filtered, output)
    return {
        "cells_before": int(adata.n_obs),
        "cells_after": int(filtered.n_obs),
        "genes_before": int(adata.n_vars),
        "genes_after": int(filtered.n_vars),
        "count_source": count_source, "filter_reasons": filter_reasons,
        "retention_by": retention_by, "analysis_context": context,
        "mitochondrial_metric_available": bool(mitochondrial_genes),
        "unapplied_criteria": audit["unapplied_criteria"],
    }


def preprocess_standard(
    source: str | Path,
    output: str | Path,
    *,
    n_hvg: int = 2000,
    n_pcs: int = 30,
    target_sum: float = 10_000,
) -> dict[str, Any]:
    adata = ad.read_h5ad(source)
    counts, count_source = _matrix_for_current_vars(adata)
    # Preserve a verified count layer before X is normalized/log-transformed.
    adata.layers["counts"] = counts.copy()
    adata.X = counts.copy()
    was_sparse = sparse.issparse(adata.X)
    if sparse.isspmatrix_csc(adata.X):
        adata.X = adata.X.tocsr()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    actual_hvg = min(n_hvg, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=actual_hvg, flavor="seurat")
    max_pcs = min(n_pcs, int(adata.var["highly_variable"].sum()) - 1, adata.n_obs - 1)
    if max_pcs < 1:
        raise ValueError("Centered PCA requires at least two observations and two selected variable genes")
    sc.pp.pca(
        adata,
        n_comps=max_pcs,
        mask_var="highly_variable",
        zero_center=True,
        svd_solver="arpack",
        random_state=0,
    )
    adata.uns["st_agent_preprocess"] = {
        "target_sum": target_sum,
        "n_hvg": actual_hvg,
        "n_pcs": max_pcs,
        "zero_center": True, "pca_solver": "arpack", "pca_seed": 0,
        "representation": "log1p_normalized",
        "count_source": count_source,
    }
    adata.uns["st_agent_analysis_context_json"] = json.dumps(analysis_context(adata), ensure_ascii=False, allow_nan=False)
    _write(adata, output)
    return {
        "hvg": int(adata.var["highly_variable"].sum()),
        "pcs": int(adata.obsm["X_pca"].shape[1]),
        "sparse_preserved": bool(was_sparse and sparse.issparse(adata.X)),
        "zero_center": True, "pca_solver": "arpack", "pca_seed": 0,
        "representation": "log1p_normalized", "count_source": count_source,
    }


def cluster_leiden_umap(
    source: str | Path,
    output: str | Path,
    *,
    seed: int,
    n_neighbors: int = 15,
    resolution: float = 1.0,
) -> dict[str, Any]:
    adata = ad.read_h5ad(source)
    if "X_pca" not in adata.obsm:
        raise ValueError("PCA must be computed before clustering")
    neighbors = min(n_neighbors, adata.n_obs - 1)
    sc.pp.neighbors(adata, n_neighbors=neighbors, use_rep="X_pca", random_state=seed)
    # UMAP can prune shared sparse graph data in place via COO conversion.
    # Its layout optimization must not change the graph subsequently used by Leiden.
    graph_key = adata.uns["neighbors"]["connectivities_key"]
    neighbor_graph = adata.obsp[graph_key].copy()
    try:
        sc.tl.umap(adata, random_state=seed)
    finally:
        adata.obsp[graph_key] = neighbor_graph
    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=seed,
        flavor="igraph",
        n_iterations=2,
        key_added="leiden",
    )
    adata.uns["st_agent_clustering"] = {
        "seed": seed,
        "n_neighbors": neighbors,
        "resolution": resolution,
        "flavor": "igraph",
        "n_iterations": 2,
        "analysis_type": "expression_clustering",
        "spatial_coordinates_used": False,
    }
    from st_agent.scientific_quality import summarize_revision
    quality = summarize_revision(adata)
    adata.uns["st_agent_quality_json"] = json.dumps(quality, ensure_ascii=False, allow_nan=False)
    _write(adata, output)
    return {
        "clusters": int(adata.obs["leiden"].nunique()), "cells": int(adata.n_obs),
        "analysis_type": "expression_clustering", "spatial_coordinates_used": False,
        "quality": quality,
    }


def rank_markers(
    source: str | Path,
    output: str | Path,
    *,
    n_genes: int = 10,
    method: str = "wilcoxon",
) -> dict[str, Any]:
    adata = ad.read_h5ad(source)
    if "leiden" not in adata.obs:
        raise ValueError("Leiden labels must exist before marker ranking")
    if "log1p" not in adata.uns:
        raise ValueError("Marker ranking requires an explicitly declared log1p-normalized X representation")
    representation = _validate_count_matrix(adata.X, "X")
    if not representation["finite"] or not representation["nonnegative"] or "unsupported_dtype" in representation["violations"]:
        raise ValueError("Marker X representation must contain finite, nonnegative log-expression, not scaled or invalid values")
    labels = adata.obs["leiden"]
    if labels.isna().any():
        raise ValueError("Marker group labels must not be missing")
    sizes = labels.astype(str).value_counts()
    if len(sizes) < 2:
        raise ValueError("Marker ranking requires at least two nonempty groups")
    if bool((sizes < 2).any()) or bool(((adata.n_obs - sizes) < 2).any()):
        raise ValueError("Marker ranking requires at least two observations in each group and its reference")
    # Remove unused categories rather than passing empty groups to Scanpy.
    adata.obs["leiden"] = labels.astype(str).astype("category")
    sc.tl.rank_genes_groups(
        adata,
        groupby="leiden",
        method=method,
        n_genes=min(n_genes, adata.n_vars),
        pts=True,
        use_raw=False,
    )
    adata.uns["st_agent_markers"] = {
        "groupby": "leiden",
        "method": method,
        "correction": "benjamini-hochberg",
        "representation": "X",
        "expression_scale": "log1p_normalized",
        "contrast": "each_expression_cluster_vs_rest",
        "interpretation_scope": "exploratory cluster markers; not a replicate-aware condition differential-expression test",
        "group_sizes_json": json.dumps({str(key): int(value) for key, value in sizes.items()}, allow_nan=False),
    }
    from st_agent.scientific_quality import summarize_revision
    quality = summarize_revision(adata)
    adata.uns["st_agent_quality_json"] = json.dumps(quality, ensure_ascii=False, allow_nan=False)
    _write(adata, output)
    return {
        "groups": int(adata.obs["leiden"].nunique()),
        "genes_per_group": len(adata.uns["rank_genes_groups"]["names"]),
        "expression_scale": "log1p_normalized", "contrast": "each_expression_cluster_vs_rest",
        "quality": quality,
    }


def _spatial_key(adata: ad.AnnData) -> str:
    keys = [str(key) for key in adata.obsm if "spatial" in str(key).lower()]
    if not keys:
        raise ValueError("Dataset has no spatial coordinates")
    return "spatial" if "spatial" in keys else sorted(keys)[0]


def _sample_indices(size: int, limit: int) -> np.ndarray:
    return np.arange(size) if size <= limit else np.linspace(0, size - 1, limit, dtype=int)


def spatial_cluster_data(source: str | Path, *, max_points: int = 20_000) -> dict[str, Any]:
    adata = ad.read_h5ad(source)
    if "leiden" not in adata.obs:
        raise ValueError("Leiden labels must exist before spatial cluster plotting")
    key = _spatial_key(adata)
    indexes = _sample_indices(adata.n_obs, max_points)
    coordinates = np.asarray(adata.obsm[key])[indexes, :2]
    labels = adata.obs["leiden"].astype(str).to_numpy()[indexes]
    return {
        "spatial_key": key,
        "sampled": len(indexes) < adata.n_obs,
        "points": [
            {"x": float(x), "y": float(y), "cluster": str(label)}
            for (x, y), label in zip(coordinates, labels, strict=True)
        ],
    }


def spatial_gene_data(source: str | Path, *, gene: str, max_points: int = 20_000) -> dict[str, Any]:
    adata = ad.read_h5ad(source)
    if gene not in adata.var_names:
        raise ValueError(f"Gene is not present: {gene}")
    key = _spatial_key(adata)
    indexes = _sample_indices(adata.n_obs, max_points)
    coordinates = np.asarray(adata.obsm[key])[indexes, :2]
    column = adata[indexes, gene].X
    expression = (
        column.toarray().reshape(-1) if sparse.issparse(column) else np.asarray(column).reshape(-1)
    )
    return {
        "spatial_key": key,
        "gene": gene,
        "sampled": len(indexes) < adata.n_obs,
        "points": [
            {"x": float(x), "y": float(y), "expression": float(value)}
            for (x, y), value in zip(coordinates, expression, strict=True)
        ],
    }
