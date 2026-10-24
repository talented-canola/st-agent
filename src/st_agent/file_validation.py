"""Cheap, read-only container checks before scientific H5AD inspection."""

from __future__ import annotations

from pathlib import Path

from st_agent.errors import DataValidationError

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def validate_h5ad_container(path: str | Path) -> dict[str, int | bool]:
    candidate = Path(path)
    if candidate.suffix.lower() != ".h5ad":
        raise DataValidationError(
            "Expected a .h5ad file", code="h5ad.extension"
        )
    if not candidate.is_file():
        raise DataValidationError(
            "H5AD file does not exist", code="h5ad.missing"
        )
    size = candidate.stat().st_size
    if size < len(HDF5_MAGIC):
        raise DataValidationError(
            "H5AD file is shorter than the HDF5 signature", code="h5ad.truncated"
        )
    try:
        with candidate.open("rb") as stream:
            signature = stream.read(len(HDF5_MAGIC))
    except OSError as exc:
        raise DataValidationError(
            "H5AD file cannot be read", code="h5ad.unreadable"
        ) from exc
    if signature != HDF5_MAGIC:
        raise DataValidationError(
            "H5AD file does not have an HDF5 signature", code="h5ad.signature"
        )
    return {"size_bytes": size, "hdf5_signature": True}
