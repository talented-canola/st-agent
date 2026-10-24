"""Explicit, read-only context for scientific analysis and visualization.

Field names identify candidate annotations, not biological truth.  In particular,
the presence of spatial coordinates does not establish physical units or that
multiple sections have been registered into one coordinate system.
"""

from __future__ import annotations

from typing import Any
from types import SimpleNamespace

import h5py
import numpy as np
from anndata.io import read_elem


SECTION_KEYS = ("chip", "section_id", "section", "slice_id", "slice", "slide_id", "slide")


def load_analysis_metadata(path):
    """Read annotation/coordinate metadata, never expression, graphs or images.

    AnnData backed mode still materializes layers and other slots. Slot presence
    is represented by None here; only small declared scientific metadata and
    spatial coordinate matrices are decoded. The file is closed before return.
    """
    explicit = {"observation_unit", "species", "organism", "tissue", "stage",
        "developmental_stage", "coordinate_system", "coordinate_units",
        "spatial_units", "spatial_registration", "st_agent_preprocess",
        "st_agent_markers", "log1p"}
    with h5py.File(path, "r") as h5:
        obs, var = read_elem(h5["obs"]), read_elem(h5["var"])
        uns = {key: read_elem(value) if key in explicit else None
               for key, value in h5.get("uns", {}).items()}
        obsm = {key: read_elem(value) if any(fragment in key.lower()
                for fragment in ("spatial", "coordinate")) else None
                for key, value in h5.get("obsm", {}).items()}
        raw = None
        if "raw/var" in h5:
            raw_var = read_elem(h5["raw/var"])
            raw = SimpleNamespace(var=raw_var, var_names=raw_var.index)
        return SimpleNamespace(obs=obs, var=var, obs_names=obs.index,
            var_names=var.index, n_obs=len(obs), n_vars=len(var), uns=uns,
            obsm=obsm, layers={key: None for key in h5.get("layers", {})},
            raw=raw, X=None, expression_loaded=False)


def _explicit(adata, *keys: str, default: str = "unknown") -> str:
    for key in keys:
        value = adata.uns.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def analysis_context(adata) -> dict[str, Any]:
    """Return JSON-safe metadata without inferring species/units from filenames."""
    lower_columns = {str(name).lower(): str(name) for name in adata.obs.columns}
    section_candidates = [lower_columns[name] for name in SECTION_KEYS if name in lower_columns]
    section_key = section_candidates[0] if section_candidates else None
    sizes: dict[str, int] = {}
    missing_count = 0
    if section_key:
        series = adata.obs[section_key]
        missing_count = int(series.isna().sum())
        sizes = {
            str(key): int(value)
            for key, value in series[series.notna()].astype(str).value_counts(sort=False).items()
        }
        sizes = dict(sorted(sizes.items()))
    coordinate_candidates = []
    for key in map(str, adata.obsm.keys()):
        if not any(fragment in key.lower() for fragment in ("spatial", "coordinate")):
            continue
        values = np.asarray(adata.obsm[key])
        if values.ndim != 2 or values.shape[1] < 2:
            continue
        try:
            finite = bool(np.isfinite(values).all())
        except TypeError:
            finite = False
        coordinate_candidates.append({
            "key": key, "shape": [int(value) for value in values.shape],
            "dimensions": int(values.shape[1]), "finite": finite,
        })
    priority = {"spatial": 0, "X_spatial": 1}
    coordinate_candidates.sort(key=lambda item: (priority.get(item["key"], 2), item["key"]))
    coordinate_key = coordinate_candidates[0]["key"] if coordinate_candidates else None
    return {
        "schema_version": 1,
        "observations": {"n_obs": int(adata.n_obs), "unit": _explicit(adata, "observation_unit")},
        "feature_space": {"n_vars": int(adata.n_vars)},
        "biology": {
            "species": _explicit(adata, "species", "organism"),
            "tissue": _explicit(adata, "tissue"),
            "developmental_stage": _explicit(adata, "developmental_stage", "stage"),
        },
        "sections": {
            "key": section_key, "candidates": section_candidates,
            "count": len(sizes), "values": list(sizes), "sizes": sizes,
            "missing_count": missing_count,
        },
        "coordinates": {
            "key": coordinate_key, "candidates": coordinate_candidates,
            "coordinate_system": _explicit(adata, "coordinate_system"),
            "units": _explicit(adata, "coordinate_units", "spatial_units"),
            "spatial_registration": _explicit(adata, "spatial_registration"),
        },
    }
