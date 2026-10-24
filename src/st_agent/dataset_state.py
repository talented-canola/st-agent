"""Read-only resolution of the scientific state of one dataset revision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from st_agent.analysis_context import analysis_context, load_analysis_metadata
from st_agent.contracts import StrictRecord
from st_agent.ids import IdKind, validate_id


class DatasetState(StrictRecord):
    dataset_id: str
    raw_revision_id: str | None = None
    input_revision_id: str
    active_revision_id: str
    count_source: str
    count_validation: dict[str, Any] = Field(default_factory=dict)
    analysis_context: dict[str, Any] = Field(default_factory=dict)
    is_raw_count_matrix: bool
    has_qc_metrics: bool
    has_normalization: bool
    has_log1p: bool
    has_hvg: bool
    has_pca: bool
    has_neighbors: bool
    has_umap: bool
    has_leiden: bool
    has_markers: bool
    has_spatial_coordinates: bool = False
    has_spatial_neighbors: bool = False
    obs_columns: tuple[str, ...] = ()


class DatasetStateInspector:
    """Resolve revision identity and processing state without mutating data."""

    def __init__(self, copilot) -> None:
        self.copilot = copilot

    def inspect(self, dataset_id: str, *, input_revision_id: str | None = None) -> DatasetState:
        dataset = self.copilot.get_dataset(dataset_id)
        if dataset.active_revision_id is None:
            raise ValueError("Dataset has no active revision")
        selected_id = input_revision_id or dataset.active_revision_id
        validate_id(selected_id, IdKind.REVISION)
        revision = self.copilot.revisions.get(selected_id)
        if revision.dataset_id != dataset.id or not revision.relative_path:
            raise ValueError("Input revision is unavailable for this dataset")
        path = self._project_path(revision.relative_path)
        revisions = self.copilot.revisions.list(dataset.id)
        raw = next((item.id for item in revisions if item.kind == "raw"), None)

        backed = load_analysis_metadata(path)
        try:
            uns_keys = set(map(str, backed.uns.keys()))
            obs_keys = set(map(str, backed.obs.columns))
            var_keys = set(map(str, backed.var.columns))
            count_source = (
                "layers/counts"
                if "counts" in backed.layers
                else "raw/X"
                if backed.raw is not None
                else "X"
            )
            has_normalization = "st_agent_preprocess" in uns_keys
            has_log1p = "log1p" in uns_keys
            # The matrix is scanned in bounded HDF5 blocks, never materialized here.
            from st_agent.scientific import validate_count_source_file
            count_validation = validate_count_source_file(path, count_source)
            return DatasetState(
                dataset_id=dataset.id,
                raw_revision_id=raw,
                input_revision_id=revision.id,
                active_revision_id=dataset.active_revision_id,
                count_source=count_source,
                count_validation=count_validation,
                analysis_context=analysis_context(backed),
                is_raw_count_matrix=bool(count_validation["valid"] and (count_source != "X" or not (has_normalization or has_log1p))),
                has_qc_metrics={"total_counts", "n_genes_by_counts", "pct_counts_mt"}
                <= obs_keys,
                has_normalization=has_normalization,
                has_log1p=has_log1p,
                has_hvg="highly_variable" in var_keys,
                has_pca="X_pca" in backed.obsm,
                has_neighbors="neighbors" in uns_keys,
                has_umap="X_umap" in backed.obsm,
                has_leiden="leiden" in obs_keys,
                has_markers="rank_genes_groups" in uns_keys or "st_agent_markers" in uns_keys,
                has_spatial_coordinates=any(item["finite"] for item in analysis_context(backed)["coordinates"]["candidates"]),
                has_spatial_neighbors=self._has_graph(path, "spatial_connectivities"),
                obs_columns=tuple(sorted(obs_keys)),
            )
        finally:
            del backed

    def _project_path(self, relative_path: str) -> Path:
        root = self.copilot.layout.root.resolve()
        path = (root / relative_path).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("Revision data file is missing or outside the project")
        return path

    @staticmethod
    def _has_graph(path: Path, key: str) -> bool:
        import h5py
        with h5py.File(path, "r") as handle:
            return f"obsp/{key}" in handle
