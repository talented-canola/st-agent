"""Research-project directory creation and metadata loading."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from st_agent.config import Settings
from st_agent.ids import IdKind, new_id, validate_id
from st_agent.migrations import LATEST_SCHEMA_VERSION, MigrationManager
from st_agent.models import ProjectRecord
from st_agent.repositories import ProjectRepository
from st_agent.utils import utc_now_iso

PROJECT_METADATA_VERSION = "project/0.1"
PROJECT_DIRECTORIES = (
    "datasets/raw",
    "datasets/revisions",
    "artifacts",
    "knowledge/sources",
    "knowledge/index",
    "reports",
    "logs",
    "cache",
    "tmp",
)


class ProjectError(RuntimeError):
    """Base class for local research-project failures."""


class ProjectExistsError(ProjectError):
    pass


class InvalidProjectError(ProjectError):
    pass


class ProjectMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    project_id: str
    name: str
    created_at: str
    database: str = "project.sqlite"


@dataclass(frozen=True)
class ProjectLayout:
    root: Path

    @property
    def metadata_path(self) -> Path:
        return self.root / "project.yaml"

    @property
    def database_path(self) -> Path:
        return self.root / "project.sqlite"

    @property
    def raw_datasets_path(self) -> Path:
        return self.root / "datasets" / "raw"

    @property
    def artifacts_path(self) -> Path:
        return self.root / "artifacts"

    def ensure_directories(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for relative in PROJECT_DIRECTORIES:
            (self.root / relative).mkdir(parents=True, exist_ok=True)


def _write_metadata_atomic(path: Path, metadata: ProjectMetadata) -> None:
    payload = yaml.safe_dump(
        metadata.model_dump(mode="json"),
        allow_unicode=True,
        sort_keys=False,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".project-", suffix=".yaml.tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def create_project(
    path: str | Path, name: str, settings: Settings
) -> tuple[ProjectLayout, ProjectRecord]:
    root = Path(path).resolve()
    layout = ProjectLayout(root)
    if layout.metadata_path.exists() or layout.database_path.exists():
        raise ProjectExistsError(f"A project already exists at {root}")
    layout.ensure_directories()

    project_id = new_id(IdKind.PROJECT)
    created_at = utc_now_iso()
    metadata = ProjectMetadata(
        schema_version=PROJECT_METADATA_VERSION,
        project_id=project_id,
        name=name,
        created_at=created_at,
    )
    _write_metadata_atomic(layout.metadata_path, metadata)
    MigrationManager(
        layout.database_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms
    ).upgrade()
    record = ProjectRecord(
        id=project_id,
        name=name,
        root_path=str(root),
        schema_version=LATEST_SCHEMA_VERSION,
        created_at=created_at,
        updated_at=created_at,
    )
    ProjectRepository(layout.database_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms).create(
        record
    )
    return layout, record


def open_project(path: str | Path, settings: Settings) -> tuple[ProjectLayout, ProjectRecord]:
    root = Path(path).resolve()
    layout = ProjectLayout(root)
    if not layout.metadata_path.is_file() or not layout.database_path.is_file():
        raise InvalidProjectError(f"Missing project.yaml or project.sqlite at {root}")
    try:
        metadata = ProjectMetadata.model_validate(
            yaml.safe_load(layout.metadata_path.read_text(encoding="utf-8"))
        )
        validate_id(metadata.project_id, IdKind.PROJECT)
    except Exception as exc:
        raise InvalidProjectError(f"Invalid project metadata at {root}: {exc}") from exc
    if metadata.schema_version != PROJECT_METADATA_VERSION:
        raise InvalidProjectError(
            f"Unsupported project metadata version: {metadata.schema_version}"
        )
    layout.ensure_directories()
    MigrationManager(
        layout.database_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms
    ).upgrade()
    record = ProjectRepository(
        layout.database_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms
    ).get(metadata.project_id)
    if Path(record.root_path).resolve() != root:
        raise InvalidProjectError("Project database root does not match project.yaml location")
    return layout, record
