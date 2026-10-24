"""Small dependency-free helpers shared by the foundation layer."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp in stable ISO-8601 form."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
