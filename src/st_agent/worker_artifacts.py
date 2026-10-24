"""Validate Worker files and publish them exclusively in the parent process."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from st_agent.artifacts import ArtifactPublisher
from st_agent.contracts import ArtifactRef
from st_agent.utils import sha256_file


class WorkerArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    relative_path: str = Field(min_length=1)
    format: Literal["csv", "json", "html", "png"]
    kind: str = Field(pattern=r"^(table|figure|data)\.[a-zA-Z0-9_.-]+$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0, le=64 * 1024 * 1024)


def validate_worker_artifacts(workspace: Path, declarations: list[dict]):
    """Validate the whole set before registering anything; never follow escapes."""
    if len(declarations) > 32:
        raise ValueError("Worker returned too many artifacts")
    validated, seen, total = [], set(), 0
    root = workspace.resolve()
    for item in declarations:
        declaration = WorkerArtifact.model_validate(item)
        relative = Path(declaration.relative_path)
        if (relative.is_absolute() or PureWindowsPath(declaration.relative_path).drive
            or ".." in relative.parts or "\\" in declaration.relative_path):
            raise ValueError("Worker artifact must use a safe workspace-relative path")
        path = root / relative
        if path.is_symlink() or root not in path.resolve().parents or not path.is_file():
            raise ValueError("Worker artifact is missing or outside its workspace")
        path = path.resolve()
        if path in seen or path.suffix.lower() != "." + declaration.format:
            raise ValueError("Duplicate Worker artifact or mismatched file type")
        seen.add(path)
        size = path.stat().st_size
        total += size
        if size != declaration.size_bytes or total > 128 * 1024 * 1024:
            raise ValueError("Worker artifact size does not match its declaration")
        if sha256_file(path) != declaration.sha256:
            raise ValueError("Worker artifact hash mismatch")
        if declaration.format == "png":
            with path.open("rb") as stream:
                if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                    raise ValueError("Worker artifact is not PNG")
        else:
            content = path.read_text(encoding="utf-8")
            if declaration.format == "json":
                json.loads(content)
            elif declaration.format == "csv":
                rows = csv.reader(io.StringIO(content))
                header = next(rows, [])
                if not header or any(len(row) != len(header) for row in rows):
                    raise ValueError("Worker table has an invalid CSV structure")
            elif "<html" not in content.lower() or "</html>" not in content.lower():
                raise ValueError("Worker artifact is not a complete HTML document")
        validated.append((declaration, path))
    return validated


def publish_worker_artifacts(copilot, workspace, validated, *, call_id, dataset_id, revision_id,
    derived: bool = False):
    if not validated:
        return []
    publisher = ArtifactPublisher(copilot.layout, copilot.project.id, copilot.artifacts, workspace)
    saved = copilot.list_artifacts()
    result = []
    for declaration, path in validated:
        existing = next((item for item in saved if item.metadata.get("call_id") == call_id
            and item.metadata.get("worker_relative_path") == declaration.relative_path
            and item.sha256 == declaration.sha256), None)
        if existing is not None:
            target = (copilot.layout.root / existing.relative_path).resolve()
            if not target.is_file() or sha256_file(target) != existing.sha256:
                raise ValueError("Previously published Worker artifact is corrupt")
            result.append(ArtifactRef(artifact_id=existing.id, kind=existing.kind,
                relative_path=existing.relative_path, sha256=existing.sha256,
                size_bytes=existing.size_bytes, metadata=existing.metadata))
            continue
        result.append(publisher.publish(path, kind=declaration.kind, metadata={
            "call_id": call_id, "dataset_id": dataset_id, "input_revision_id": revision_id,
            "revision_id": None if derived else revision_id,
            "binding": "pending_output_commit" if derived else "input_revision",
            "worker_relative_path": declaration.relative_path,
            "format": declaration.format}))
        from st_agent.trace import emit
        emit(copilot, "artifact.exported", artifact_id=result[-1].artifact_id,
            kind=declaration.kind, call_id=call_id, sha256=declaration.sha256,
            input_revision_id=revision_id)
    return result


def bind_output_revision(copilot, references, revision_id):
    """Reconcile output artifact identity after a successful commit, also on recovery."""
    if not references:
        return []
    result = []
    with copilot.artifacts._connect() as connection:
        for ref in references:
            metadata = {**ref.metadata, "revision_id": revision_id,
                "output_revision_id": revision_id, "binding": "committed_output"}
            connection.execute("UPDATE artifacts SET metadata_json=? WHERE id=? AND project_id=?",
                (json.dumps(metadata, ensure_ascii=False), ref.artifact_id, copilot.project.id))
            result.append(ref.model_copy(update={"metadata": metadata}))
        connection.commit()
    return result


def validate_published_refs(copilot, references):
    result = []
    root = copilot.layout.root.resolve()
    for item in references:
        ref = ArtifactRef.model_validate(item)
        record = copilot.artifacts.get(ref.artifact_id)
        path = (root / ref.relative_path).resolve()
        if (record.project_id != copilot.project.id or record.relative_path != ref.relative_path
            or record.sha256 != ref.sha256 or root not in path.parents
            or not path.is_file() or path.stat().st_size != ref.size_bytes
            or sha256_file(path) != ref.sha256):
            raise ValueError("Published artifact identity or content mismatch")
        result.append(ref)
    return tuple(result)
