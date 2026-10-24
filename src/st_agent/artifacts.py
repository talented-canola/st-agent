"""Atomic publication of project-owned artifact files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from st_agent.contracts import ArtifactRef
from st_agent.ids import IdKind, new_id, validate_id
from st_agent.models import ArtifactRecord
from st_agent.project import ProjectLayout
from st_agent.repositories import ArtifactRepository
from st_agent.utils import sha256_file, utc_now_iso


class ArtifactPublisher:
    def __init__(
        self,
        layout: ProjectLayout,
        project_id: str,
        repository: ArtifactRepository,
        staging_root: str | Path,
    ) -> None:
        self.layout = layout
        self.project_id = validate_id(project_id, IdKind.PROJECT)
        self.repository = repository
        self.staging_root = Path(staging_root).resolve()

    def publish(
        self,
        candidate: str | Path,
        *,
        kind: str,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        source = Path(candidate).resolve()
        if not source.is_file() or self.staging_root not in source.parents:
            raise ValueError("Artifact must be a file inside its assigned staging directory")
        if run_id is not None:
            validate_id(run_id, IdKind.RUN)
        artifact_id = new_id(IdKind.ARTIFACT)
        suffix = "".join(source.suffixes) or ".bin"
        group = run_id or "unscoped"
        destination = self.layout.artifacts_path / group / f"{artifact_id}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256_file(source)
        os.replace(source, destination)
        relative = destination.relative_to(self.layout.root).as_posix()
        record = ArtifactRecord(
            id=artifact_id,
            project_id=self.project_id,
            run_id=run_id,
            kind=kind,
            relative_path=relative,
            sha256=digest,
            size_bytes=destination.stat().st_size,
            metadata=metadata or {},
            created_at=utc_now_iso(),
        )
        try:
            self.repository.create(record)
        except Exception:
            os.replace(destination, source)
            raise
        return ArtifactRef(
            artifact_id=record.id,
            kind=record.kind,
            relative_path=record.relative_path,
            sha256=record.sha256,
            size_bytes=record.size_bytes,
            metadata=record.metadata,
        )
