"""Read-only, JSON-safe evidence for expression clustering and marker review.

These metrics describe structure; author labels, stability, and associations
are not biological ground truth.  This module never fits a model, changes an
AnnData, or invents a cell-type annotation.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _number(value: Any, *, fraction=False) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or (fraction and not 0 <= result <= 1):
        return None
    return result


def _valid_labels(values) -> np.ndarray:
    return np.asarray(pd.notna(values), dtype=bool)


def _contingency(left, right) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    table = pd.crosstab(pd.Series(left, name="cluster"), pd.Series(right, name="reference"))
    public = {str(cluster): {str(label): int(count) for label, count in row.items()}
              for cluster, row in table.iterrows()}
    return table, public


def compare_partitions(left_ids: Sequence, left_labels: Sequence,
                       right_ids: Sequence, right_labels: Sequence) -> dict[str, Any]:
    """Align by observation identity before comparing partitions; never by row."""
    def mapping(ids, labels):
        if len(ids) != len(labels):
            raise ValueError("Observation IDs and labels must have equal lengths")
        normalized = list(map(str, ids))
        if len(set(normalized)) != len(normalized):
            raise ValueError("Observation IDs must be unique")
        return {key: str(label) for key, label in zip(normalized, labels, strict=True)
                if pd.notna(label)}
    left, right = mapping(left_ids, left_labels), mapping(right_ids, right_labels)
    shared = sorted(left.keys() & right.keys())
    result = {"shared_observations": len(shared), "left_only": len(left.keys() - right.keys()),
        "right_only": len(right.keys() - left.keys()), "adjusted_rand_index": None,
        "normalized_mutual_information": None, "status": "insufficient_overlap"}
    if len(shared) >= 2:
        a, b = [left[key] for key in shared], [right[key] for key in shared]
        result.update(adjusted_rand_index=float(adjusted_rand_score(a, b)),
            normalized_mutual_information=float(normalized_mutual_info_score(a, b)), status="available")
    return result


def _label_metrics(labels, reference):
    valid = _valid_labels(labels) & _valid_labels(reference)
    left = np.asarray(labels)[valid].astype(str)
    right = np.asarray(reference)[valid].astype(str)
    table, contingency = _contingency(left, right)
    result = {"n_valid": int(valid.sum()), "contingency": contingency,
        "adjusted_rand_index": None, "normalized_mutual_information": None,
        "status": "insufficient_observations"}
    if len(left) >= 2:
        result.update(adjusted_rand_index=float(adjusted_rand_score(left, right)),
            normalized_mutual_information=float(normalized_mutual_info_score(left, right)),
            status="available")
    return result, table


def _qc_vectors(adata, warnings):
    # Import lazily: scientific workers can call us without a module cycle.
    from st_agent.scientific import _count_matrix, _qc_vectors as kernel_qc_vectors
    try:
        matrix, _, source = _count_matrix(adata)
        values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("nonfinite_or_negative_count_source")
        if np.issubdtype(np.asarray(values).dtype, np.floating):
            for start in range(0, values.size, 1_000_000):
                block = np.asarray(values).reshape(-1)[start:start + 1_000_000]
                if np.any(np.abs(block - np.rint(block)) > 1e-6):
                    raise ValueError("fractional_count_source")
        vectors, source, _ = kernel_qc_vectors(adata)
        unavailable = [name for name, vector in vectors.items() if vector is None]
        if unavailable:
            warnings.append({"code": "qc_annotation_unavailable",
                "message": "Some QC metrics cannot be evaluated because their annotation is unavailable; missing values are not zero.",
                "evidence": {"metrics": unavailable, "count_source": source}})
        return {name: np.asarray(vector) for name, vector in vectors.items()
                if vector is not None}, source
    except (ValueError, TypeError, IndexError) as error:
        warnings.append({"code": "qc_associations_unavailable",
            "message": "QC association metrics were not computed from an invalid or unavailable count source.",
            "evidence": {"reason": str(error)}})
        return {}, None


def _qc_associations(adata, labels, warnings):
    vectors, source = _qc_vectors(adata, warnings)
    pca_metrics, cluster_metrics = [], {}
    if "X_pca" in adata.obsm:
        pca = np.asarray(adata.obsm["X_pca"])
        if pca.ndim == 2 and pca.shape[0] == adata.n_obs:
            for component in range(pca.shape[1]):
                for name, vector in vectors.items():
                    valid = np.isfinite(pca[:, component]) & np.isfinite(vector)
                    x, y = pca[valid, component], vector[valid]
                    rho = None
                    if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
                        rho = _number(spearmanr(x, y).statistic)
                    item = {"pc": component + 1, "metric": name, "spearman_rho": rho,
                        "n_valid": int(valid.sum()), "status": "available" if rho is not None
                        else "constant_or_insufficient", "count_source": source}
                    pca_metrics.append(item)
                    if rho is not None and abs(rho) >= 0.6:
                        warnings.append({"code": "pca_qc_association",
                            "message": "A component of X_pca is associated with a QC metric; association alone does not establish a technical cause.",
                            "evidence": {"pc": component + 1, "metric": name,
                                         "spearman_rho": rho, "advisory_threshold_abs_rho": 0.6}})
    if labels is not None:
        label_values = np.asarray(labels)
        for name, vector in vectors.items():
            valid = _valid_labels(labels) & np.isfinite(vector)
            values, groups = np.asarray(vector)[valid], label_values[valid].astype(str)
            medians, eta = {}, None
            if len(values):
                mean = float(np.mean(values))
                total = float(np.square(values - mean).sum())
                between = 0.0
                for group in sorted(set(groups)):
                    selected = values[groups == group]
                    medians[group] = float(np.median(selected))
                    between += len(selected) * (float(np.mean(selected)) - mean) ** 2
                if total > 0:
                    eta = float(np.clip(between / total, 0, 1))
            cluster_metrics[name] = {"eta_squared": eta, "cluster_medians": medians,
                "n_valid": int(valid.sum()), "count_source": source,
                "status": "available" if eta is not None else "constant_or_insufficient",
                "interpretation": "association_not_causation"}
            if eta is not None and eta >= 0.5:
                warnings.append({"code": "cluster_qc_association",
                    "message": "Cluster membership is associated with a QC metric; inspect biology and sampling design before assigning a technical cause.",
                    "evidence": {"metric": name, "eta_squared": eta, "advisory_threshold_eta_squared": 0.5}})
    return pca_metrics, cluster_metrics


def _marker_values(adata, labels, warnings):
    ranking = adata.uns.get("rank_genes_groups", {})
    names = ranking.get("names")
    groups = getattr(getattr(names, "dtype", None), "names", None)
    if not groups or labels is None:
        return {}, False
    params = ranking.get("params", {})
    groupby = params.get("groupby", adata.uns.get("st_agent_markers", {}).get("groupby", "leiden"))
    if groupby != "leiden":
        warnings.append({"code": "marker_groupby_mismatch", "message": "Marker groups do not refer to the displayed Leiden partition.",
                         "evidence": {"groupby": str(groupby)}})
        return {}, False
    layer = params.get("layer")
    declared = adata.uns.get("st_agent_markers", {}).get("representation")
    use_raw = params.get("use_raw")
    if isinstance(use_raw, (bool, np.bool_)):
        use_raw = bool(use_raw)
    if use_raw is None and declared == "X":
        use_raw = False
    matrix, source, genes = None, None, []
    if layer is not None and str(layer) in adata.layers:
        matrix, source, genes = adata.layers[str(layer)], "layer:" + str(layer), adata.var_names
    elif use_raw and adata.raw is not None:
        matrix, source, genes = adata.raw.X, "raw.X", adata.raw.var_names
    elif use_raw is False and layer is None:
        matrix, source, genes = adata.X, "X", adata.var_names
    if matrix is None:
        warnings.append({"code": "marker_representation_unknown",
            "message": "Marker expression source is absent or ambiguous; detection fractions are unavailable, not inferred from another matrix.",
            "evidence": {"layer": layer, "use_raw": use_raw, "declared_representation": declared}})
    positions = {str(gene): i for i, gene in enumerate(genes)}
    labels_array = np.asarray(labels)
    valid_labels = _valid_labels(labels)
    labels_string = np.asarray([str(value) if valid else "" for value, valid in
                               zip(labels_array, valid_labels, strict=True)])
    detection_cache = {}
    output = {}
    for group in groups:
        inside = valid_labels & (labels_string == str(group))
        outside = valid_labels & ~inside
        if not inside.any():
            warnings.append({"code": "marker_group_missing", "message": "A ranked marker group has no observations in the displayed partition.",
                             "evidence": {"group": str(group)}})
            continue
        rows = []
        for rank, raw_gene in enumerate(names[group]):
            gene = str(raw_gene)
            pct_in = pct_out = None
            if matrix is not None and gene in positions:
                if gene not in detection_cache:
                    column = matrix[:, positions[gene]]
                    vector = column.toarray().reshape(-1) if sparse.issparse(column) else np.asarray(column).reshape(-1)
                    detection_cache[gene] = vector
                vector = detection_cache[gene]
                if np.isfinite(vector).all():
                    pct_in = float(np.mean(vector[inside] > 0))
                    pct_out = float(np.mean(vector[outside] > 0)) if outside.any() else None
            def statistic(key):
                values = ranking.get(key)
                fields = getattr(getattr(values, "dtype", None), "names", ()) or ()
                if values is not None and group in fields and rank < len(values):
                    return _number(values[group][rank], fraction=key == "pvals_adj")
                return None
            rows.append({"gene": gene, "rank": rank + 1, "logfoldchange": statistic("logfoldchanges"),
                "p_adjusted": statistic("pvals_adj"), "pct_in": pct_in, "pct_out": pct_out,
                "detection_difference": pct_in - pct_out if pct_in is not None and pct_out is not None else None,
                "detection_source": source if pct_in is not None else None})
        output[str(group)] = rows
    warnings.append({"code": "marker_test_scope", "message": "Cluster-versus-rest marker statistics are exploratory and do not establish replicated condition-level differential expression or cell identity.",
                     "evidence": {"groupby": "leiden", "method": str(params.get("method", adata.uns.get("st_agent_markers", {}).get("method", "unknown"))),
                                  "representation": source}})
    return output, bool(output)


def summarize_revision(adata) -> dict[str, Any]:
    """Return auditable metrics from this exact AnnData without changing it."""
    warnings = []
    labels = adata.obs["leiden"] if "leiden" in adata.obs else None
    cluster_sizes = {}
    if labels is not None:
        valid = _valid_labels(labels)
        groups, counts = np.unique(np.asarray(labels)[valid].astype(str), return_counts=True)
        cluster_sizes = {str(group): int(count) for group, count in zip(groups, counts, strict=True)}
    references, chip_associations = {}, {}
    if labels is not None:
        for field in ("celltype", "germ_layer"):
            if field in adata.obs:
                metrics, _ = _label_metrics(labels, adata.obs[field])
                references[field] = {"role": "author_reference_not_truth", **metrics}
        if "chip" in adata.obs:
            metrics, table = _label_metrics(labels, adata.obs["chip"])
            metrics.pop("adjusted_rand_index")
            metrics["cramers_v"] = None
            if min(table.shape, default=0) > 1 and table.to_numpy().sum() > 0:
                chi2 = float(chi2_contingency(table.to_numpy(), correction=False).statistic)
                metrics["cramers_v"] = float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))
            metrics["interpretation"] = "association_not_causation"
            chip_associations["chip"] = metrics
            if metrics["cramers_v"] is not None and metrics["cramers_v"] >= 0.5:
                warnings.append({"code": "cluster_chip_association",
                    "message": "Clusters and sections are associated; spatial biology and sampling can produce this pattern, so association alone does not establish a technical cause.",
                    "evidence": {"field": "chip", "cramers_v": metrics["cramers_v"], "advisory_threshold_cramers_v": 0.5}})
    pca_qc, cluster_qc = _qc_associations(adata, labels, warnings)
    markers, markers_available = _marker_values(adata, labels, warnings)
    zero_center = adata.uns.get("st_agent_preprocess", {}).get("zero_center")
    representation = {"key": "X_pca" if "X_pca" in adata.obsm else None,
        "zero_center": bool(zero_center) if zero_center is not None else None,
        "method": "unknown" if zero_center is None else
                  "centered_pca" if bool(zero_center) else "uncentered_svd"}
    return {"schema_version": "scientific_quality/1", "analysis_kind": "expression_clustering",
        "n_observations": int(adata.n_obs), "n_features": int(adata.n_vars),
        "clustering_available": bool(cluster_sizes), "cluster_key": "leiden" if labels is not None else None,
        "cluster_sizes": cluster_sizes, "reference_annotation": references,
        "embedding_representation": representation,
        "cluster_chip_association": chip_associations, "pca_qc_associations": pca_qc,
        "cluster_qc_associations": cluster_qc, "markers": markers,
        "markers_available": markers_available, "warnings": warnings}
