"""Validated adapters for external biological knowledge and communication evidence."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import anndata as ad
import numpy as np

from st_agent.rag import fetch_pubmed_batches


def _unique_genes(genes: Sequence[str]) -> list[str]:
    cleaned = sorted({str(gene).strip() for gene in genes if str(gene).strip()})
    if not cleaned:
        raise ValueError("At least one non-empty gene is required")
    return cleaned


def _source(provider: Any) -> str:
    value = str(getattr(provider, "source", "")).strip()
    if not value:
        raise ValueError("Scientific provider must expose a versioned source")
    return value


def _bh_adjust(values: Sequence[float]) -> list[float]:
    count = len(values)
    order = np.argsort(np.asarray(values, dtype=float))
    adjusted = np.empty(count, dtype=float)
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        running = min(running, float(values[original_index]) * count / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted.tolist()


class EnrichmentProvider(Protocol):
    source: str

    def enrich(
        self, genes: list[str], *, species: str, database: str
    ) -> list[dict[str, Any]]: ...


def run_enrichment(
    genes: Sequence[str],
    *,
    species: str,
    database: str,
    provider: EnrichmentProvider,
) -> dict[str, Any]:
    unique = _unique_genes(genes)
    try:
        records = provider.enrich(unique, species=species, database=database)
    except (OSError, TimeoutError) as error:
        raise ConnectionError("Enrichment provider is unavailable") from error
    required = {"term_id", "term_name", "p_value", "overlap"}
    if not isinstance(records, list) or any(
        not isinstance(record, dict) or not required <= set(record) for record in records
    ):
        raise ValueError("Enrichment response failed structural validation")
    p_values = []
    for record in records:
        p_value = float(record["p_value"])
        if not 0 <= p_value <= 1:
            raise ValueError("Enrichment p-values must be between zero and one")
        p_values.append(p_value)
    adjusted = _bh_adjust(p_values)
    terms = [
        {
            "term_id": str(record["term_id"]),
            "term_name": str(record["term_name"]),
            "p_value": p_value,
            "adjusted_p_value": adjusted_value,
            "overlap": sorted(map(str, record["overlap"])),
        }
        for record, p_value, adjusted_value in zip(records, p_values, adjusted, strict=True)
    ]
    terms.sort(key=lambda item: (item["adjusted_p_value"], item["term_id"]))
    return {
        "query_genes": unique,
        "species": species,
        "database": database,
        "correction": "benjamini-hochberg",
        "source": _source(provider),
        "terms": terms,
    }


class MyGeneProvider(Protocol):
    source: str

    def query_many(self, genes: list[str], *, species: str) -> list[dict[str, Any]]: ...


def annotate_genes(
    genes: Sequence[str], *, species: str, provider: MyGeneProvider
) -> dict[str, Any]:
    unique = list(dict.fromkeys(str(gene).strip() for gene in genes if str(gene).strip()))
    if not unique:
        raise ValueError("At least one non-empty gene is required")
    try:
        records = provider.query_many(unique, species=species)
    except (OSError, TimeoutError) as error:
        raise ConnectionError("MyGene provider is unavailable") from error
    if not isinstance(records, list):
        raise ValueError("MyGene response must be a list")
    by_query = {}
    for record in records:
        if not isinstance(record, dict) or not record.get("query"):
            raise ValueError("MyGene response failed structural validation")
        by_query[str(record["query"])] = record
    annotations = [by_query.get(gene, {"query": gene, "notfound": True}) for gene in unique]
    return {
        "genes": unique,
        "species": species,
        "source": _source(provider),
        "annotations": annotations,
    }


class PubMedProvider(Protocol):
    source: str

    def search(self, query: str, *, limit: int) -> Sequence[str]: ...

    def fetch(self, pmids: Sequence[str]) -> Sequence[dict[str, str]]: ...


def search_pubmed(query: str, *, limit: int, provider: PubMedProvider) -> dict[str, Any]:
    normalized = query.strip()
    if not normalized:
        raise ValueError("PubMed query cannot be empty")
    if not 1 <= limit <= 100:
        raise ValueError("PubMed limit must be between 1 and 100")
    try:
        pmids = list(dict.fromkeys(map(str, provider.search(normalized, limit=limit))))[:limit]
        records = fetch_pubmed_batches(pmids, provider)
    except (OSError, TimeoutError) as error:
        raise ConnectionError("PubMed provider is unavailable") from error
    prepared = [
        {
            **record,
            "pmid": str(record["pmid"]),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{record['pmid']}/",
        }
        for record in records
    ]
    return {"query": normalized, "source": _source(provider), "records": prepared}


def _column_means(matrix: Any, row_indices: np.ndarray) -> np.ndarray:
    subset = matrix[row_indices]
    means = subset.mean(axis=0)
    return np.asarray(means).reshape(-1)


def compute_ligand_receptor_scores(
    source: str | Path,
    *,
    cluster_key: str,
    interactions: Sequence[dict[str, str]],
    min_expression: float = 0.0,
) -> dict[str, Any]:
    adata = ad.read_h5ad(source)
    if cluster_key not in adata.obs:
        raise ValueError(f"Unknown cluster column: {cluster_key}")
    required = {"ligand", "receptor", "source"}
    if not interactions or any(required - set(item) for item in interactions):
        raise ValueError("Interactions require ligand, receptor, and source")
    gene_positions = {str(gene): index for index, gene in enumerate(adata.var_names)}
    clusters = sorted(map(str, adata.obs[cluster_key].astype(str).unique()))
    labels = adata.obs[cluster_key].astype(str).to_numpy()
    means = {
        cluster: _column_means(adata.X, np.flatnonzero(labels == cluster)) for cluster in clusters
    }
    results = []
    for interaction in interactions:
        ligand, receptor = str(interaction["ligand"]), str(interaction["receptor"])
        if ligand not in gene_positions or receptor not in gene_positions:
            continue
        ligand_index, receptor_index = gene_positions[ligand], gene_positions[receptor]
        for sender in clusters:
            ligand_mean = float(means[sender][ligand_index])
            if ligand_mean < min_expression:
                continue
            for receiver in clusters:
                receptor_mean = float(means[receiver][receptor_index])
                if receptor_mean < min_expression:
                    continue
                results.append(
                    {
                        "sender": sender,
                        "receiver": receiver,
                        "ligand": ligand,
                        "receptor": receptor,
                        "score": ligand_mean * receptor_mean,
                        "source": str(interaction["source"]),
                    }
                )
    results.sort(key=lambda item: (-item["score"], item["sender"], item["receiver"]))
    return {
        "cluster_key": cluster_key,
        "method": "product_of_cluster_mean_expression",
        "database_sources": sorted({str(item["source"]) for item in interactions}),
        "interactions": results,
        "warnings": ["Scores are exploratory associations, not causal communication evidence"],
    }
