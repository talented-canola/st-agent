"""Build reproducibility metadata for deterministic tool execution."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from st_agent.contracts import Provenance


def dependency_lock_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_package_versions(package_names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in sorted(set(package_names)):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def build_provenance(
    *,
    tool_version: str,
    dataset_hash: str | None,
    parameters: dict[str, Any],
    random_seed: int | None,
    package_names: Iterable[str],
    lock_path: str | Path,
    started_at: datetime,
    finished_at: datetime,
) -> Provenance:
    duration_ms = max(0, round((finished_at - started_at).total_seconds() * 1000))
    return Provenance(
        tool_version=tool_version,
        dataset_hash=dataset_hash,
        parameters=parameters,
        random_seed=random_seed,
        package_versions=collect_package_versions(package_names),
        dependency_lock_hash=dependency_lock_hash(lock_path),
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
    )
