"""Public Phase 1 Python API for local project persistence."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

from st_agent.config import Settings
from st_agent.events import EventDraft, SQLiteEventRepository
from st_agent.ids import IdKind, new_id, validate_id
from st_agent.memory import MemoryStore
from st_agent.models import (
    ArtifactRecord,
    DatasetRecord,
    MessageRecord,
    ProjectRecord,
    SessionRecord,
)
from st_agent.project import ProjectLayout, create_project, open_project
from st_agent.rag import KnowledgeManifest, KnowledgeStore
from st_agent.repositories import (
    ArtifactRepository,
    DatasetRepository,
    RevisionRepository,
    SessionRepository,
    ToolExecutionRepository,
    WorkerJobRepository,
)
from st_agent.revisions import RevisionManager
from st_agent.utils import sha256_file, utc_now_iso
from st_agent.workers import WorkerManager


class ClosedProjectError(RuntimeError):
    """Raised when a caller uses a closed copilot handle."""


class ResearchCopilot:
    """A handle to one isolated local research project.

    Phase 1 intentionally exposes persistence operations only. Agent execution,
    Tool Calling and scientific workflows are added in later gated phases.
    """

    def __init__(
        self,
        layout: ProjectLayout,
        project: ProjectRecord,
        settings: Settings,
    ) -> None:
        self.layout = layout
        self.project = project
        self.settings = settings
        timeout = settings.sqlite_busy_timeout_ms
        self.datasets = DatasetRepository(layout.database_path, busy_timeout_ms=timeout)
        self.revisions = RevisionRepository(layout.database_path, busy_timeout_ms=timeout)
        self.worker_jobs = WorkerJobRepository(layout.database_path, busy_timeout_ms=timeout)
        self.tool_executions = ToolExecutionRepository(
            layout.database_path, busy_timeout_ms=timeout
        )
        self.revision_manager = RevisionManager(layout, project.id, self.datasets, self.revisions)
        self.sessions = SessionRepository(layout.database_path, busy_timeout_ms=timeout)
        self.artifacts = ArtifactRepository(layout.database_path, busy_timeout_ms=timeout)
        self.events = SQLiteEventRepository(
            layout.database_path, project.id, busy_timeout_ms=timeout
        )
        self.worker_manager = WorkerManager(layout, project.id, self.worker_jobs)
        self.memory = MemoryStore(layout.database_path, project.id)
        self._closed = False

    @classmethod
    def create(
        cls,
        project_path: str | Path,
        *,
        name: str | None = None,
        settings: Settings | None = None,
    ) -> ResearchCopilot:
        active_settings = settings or Settings.load()
        path = Path(project_path).resolve()
        layout, project = create_project(path, name or path.name, active_settings)
        instance = cls(layout, project, active_settings)
        instance.events.append(
            EventDraft(
                event_type="project.created",
                payload={"schema_version": project.schema_version},
            )
        )
        return instance

    @classmethod
    def open(
        cls,
        project_path: str | Path,
        *,
        settings: Settings | None = None,
    ) -> ResearchCopilot:
        active_settings = settings or Settings.load()
        layout, project = open_project(project_path, active_settings)
        instance = cls(layout, project, active_settings)
        instance.worker_manager.recover_abandoned()
        instance.revision_manager.recover_incomplete()
        instance.revision_manager.verify_and_reconcile_all()
        return instance

    @property
    def is_closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClosedProjectError(f"Project is closed: {self.project.id}")

    def close(self) -> None:
        """Close this lightweight handle; persistent state remains on disk."""
        self._closed = True

    def knowledge_store(self, manifest: KnowledgeManifest) -> KnowledgeStore:
        """Open the project-local trusted knowledge index for an allowlist manifest."""
        self._ensure_open()
        return KnowledgeStore(self.layout.database_path, self.project.id, manifest)

    def __enter__(self) -> ResearchCopilot:
        self._ensure_open()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def add_dataset(
        self,
        file_path: str | Path,
        *,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetRecord:
        """Register an immutable raw file using hard-link-first materialization."""
        self._ensure_open()
        source = Path(file_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = sha256_file(source)
        existing = self.datasets.find_by_hash(self.project.id, source_hash)
        if existing is not None:
            self.revision_manager.ensure_initial_revision(existing)
            return self.datasets.get(existing.id)

        dataset_id = new_id(IdKind.DATASET)
        suffix = "".join(source.suffixes) or ".bin"
        destination = self.layout.raw_datasets_path / f"{dataset_id}{suffix}"
        storage_mode = self._materialize_raw_file(source, destination)
        destination_hash = sha256_file(destination)
        if destination_hash != source_hash:
            destination.unlink(missing_ok=True)
            raise OSError(f"Raw dataset hash mismatch after {storage_mode}: {source.name}")

        try:
            record_metadata = dict(metadata or {})
            record_metadata["storage_mode"] = storage_mode
            record = DatasetRecord(
                id=dataset_id,
                project_id=self.project.id,
                display_name=display_name or source.name,
                source_path=str(source),
                storage_path=str(destination.resolve()),
                sha256=source_hash,
                size_bytes=source.stat().st_size,
                metadata=record_metadata,
                created_at=utc_now_iso(),
            )
            self.datasets.create(record)
            self.revision_manager.ensure_initial_revision(record)
        except Exception:
            self.datasets.delete(dataset_id)
            destination.unlink(missing_ok=True)
            raise
        self.events.append(
            EventDraft(
                event_type="dataset.registered",
                payload={
                    "dataset_id": record.id,
                    "sha256": record.sha256,
                    "size_bytes": record.size_bytes,
                    "storage_mode": storage_mode,
                },
            )
        )
        return self.datasets.get(record.id)

    @staticmethod
    def _materialize_raw_file(source: Path, destination: Path) -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
                return "copy"
            finally:
                temporary.unlink(missing_ok=True)

    def get_dataset(self, dataset_id: str) -> DatasetRecord:
        self._ensure_open()
        validate_id(dataset_id, IdKind.DATASET)
        return self.datasets.get(dataset_id, project_id=self.project.id)

    def list_datasets(self) -> list[DatasetRecord]:
        self._ensure_open()
        return self.datasets.list(self.project.id)

    def create_session(self, *, title: str | None = None) -> SessionRecord:
        self._ensure_open()
        record = SessionRecord(
            id=new_id(IdKind.SESSION),
            project_id=self.project.id,
            title=title,
            created_at=utc_now_iso(),
        )
        self.sessions.create(record)
        self.events.append(
            EventDraft(
                event_type="session.created",
                session_id=record.id,
                payload={"title": title},
            )
        )
        return record

    def list_sessions(self) -> list[SessionRecord]:
        self._ensure_open()
        return self.sessions.list(self.project.id)

    def close_session(self, session_id: str) -> SessionRecord:
        self._ensure_open()
        validate_id(session_id, IdKind.SESSION)
        session = self.sessions.get(session_id)
        if session.project_id != self.project.id:
            raise KeyError(f"Session does not belong to project: {session_id}")
        closed = self.sessions.close(session_id, utc_now_iso())
        self.events.append(
            EventDraft(
                event_type="session.closed",
                session_id=session_id,
                payload={},
            )
        )
        return closed

    def add_message(
        self,
        session_id: str,
        *,
        role: Literal["user", "assistant", "system", "tool"],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MessageRecord:
        self._ensure_open()
        validate_id(session_id, IdKind.SESSION)
        session = self.sessions.get(session_id)
        if session.project_id != self.project.id:
            raise KeyError(f"Session does not belong to project: {session_id}")
        record = MessageRecord(
            id=new_id(IdKind.MESSAGE),
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata or {},
            created_at=utc_now_iso(),
        )
        self.sessions.add_message(record)
        self.events.append(
            EventDraft(
                event_type="message.recorded",
                session_id=session_id,
                payload={"message_id": record.id, "role": role},
            )
        )
        return record

    def list_messages(self, session_id: str) -> list[MessageRecord]:
        self._ensure_open()
        validate_id(session_id, IdKind.SESSION)
        session = self.sessions.get(session_id)
        if session.project_id != self.project.id:
            raise KeyError(f"Session does not belong to project: {session_id}")
        return self.sessions.list_messages(session_id)

    def register_artifact(
        self,
        path: str | Path,
        *,
        kind: str,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        self._ensure_open()
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            relative = source.relative_to(self.layout.root)
        except ValueError as exc:
            raise ValueError("Artifacts must be stored inside the research project") from exc
        if not relative.parts or relative.parts[0] != "artifacts":
            raise ValueError("Artifacts must be stored under the project artifacts directory")
        record = ArtifactRecord(
            id=new_id(IdKind.ARTIFACT),
            project_id=self.project.id,
            run_id=run_id,
            kind=kind,
            relative_path=relative.as_posix(),
            sha256=sha256_file(source),
            size_bytes=source.stat().st_size,
            metadata=metadata or {},
            created_at=utc_now_iso(),
        )
        self.artifacts.create(record)
        self.events.append(
            EventDraft(
                event_type="artifact.registered",
                run_id=run_id,
                payload={"artifact_id": record.id, "kind": kind},
            )
        )
        return record

    def list_artifacts(self, *, run_id: str | None = None) -> list[ArtifactRecord]:
        self._ensure_open()
        return self.artifacts.list(self.project.id, run_id=run_id)
