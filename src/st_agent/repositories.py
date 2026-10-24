"""Repositories for small persistent records; no scientific arrays live here."""

from __future__ import annotations

import json
from pathlib import Path

from st_agent.contracts import ToolResult, ToolStatus
from st_agent.db import connect_database
from st_agent.models import (
    ArtifactRecord,
    DatasetRecord,
    DatasetRevisionRecord,
    JobStatus,
    MessageRecord,
    ProjectRecord,
    SessionRecord,
    WorkerJobRecord,
)
from st_agent.utils import utc_now_iso


class _Repository:
    def __init__(self, database_path: str | Path, *, busy_timeout_ms: int):
        self.database_path = Path(database_path)
        self.busy_timeout_ms = busy_timeout_ms

    def _connect(self):
        return connect_database(self.database_path, busy_timeout_ms=self.busy_timeout_ms)


class ProjectRepository(_Repository):
    def create(self, record: ProjectRecord) -> ProjectRecord:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO projects(id, name, root_path, schema_version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.name,
                    record.root_path,
                    record.schema_version,
                    record.created_at,
                    record.updated_at,
                ),
            )
            connection.commit()
        return record

    def get(self, project_id: str) -> ProjectRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown project: {project_id}")
        return ProjectRecord(**dict(row))


class DatasetRepository(_Repository):
    def create(self, record: DatasetRecord) -> DatasetRecord:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO datasets(
                    id, project_id, display_name, source_path, storage_path,
                    sha256, size_bytes, status, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.project_id,
                    record.display_name,
                    record.source_path,
                    record.storage_path,
                    record.sha256,
                    record.size_bytes,
                    record.status,
                    json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                    record.created_at,
                ),
            )
            connection.commit()
        return record

    def find_by_hash(self, project_id: str, sha256: str) -> DatasetRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM datasets WHERE project_id = ? AND sha256 = ?",
                (project_id, sha256),
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def get(self, dataset_id: str, *, project_id: str | None = None) -> DatasetRecord:
        sql = "SELECT * FROM datasets WHERE id = ?"
        params: tuple[object, ...] = (dataset_id,)
        if project_id is not None:
            sql += " AND project_id = ?"
            params += (project_id,)
        with self._connect() as connection:
            row = connection.execute(sql, params).fetchone()
        if row is None:
            raise KeyError(f"Unknown dataset: {dataset_id}")
        return self._from_row(row)

    def list(self, project_id: str) -> list[DatasetRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM datasets WHERE project_id = ? ORDER BY created_at, id", (project_id,)
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def update_status(self, dataset_id: str, status: str) -> DatasetRecord:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE datasets SET status = ? WHERE id = ?", (status, dataset_id)
            )
            connection.commit()
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown dataset: {dataset_id}")
        return self.get(dataset_id)

    def delete(self, dataset_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            connection.commit()

    @staticmethod
    def _from_row(row) -> DatasetRecord:
        payload = dict(row)
        payload["metadata"] = json.loads(payload.pop("metadata_json"))
        return DatasetRecord(**payload)


class RevisionRepository(_Repository):
    def create(self, record: DatasetRevisionRecord) -> DatasetRevisionRecord:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO dataset_revisions(
                    id, dataset_id, parent_revision_id, status, relative_path, sha256,
                    metadata_json, created_at, call_id, kind, provenance_json,
                    validation_json, error_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._values(record),
            )
            connection.commit()
        return record

    def create_initial(self, record: DatasetRevisionRecord) -> DatasetRevisionRecord:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT active_revision_id FROM datasets WHERE id = ?", (record.dataset_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown dataset: {record.dataset_id}")
            if row["active_revision_id"] is not None:
                connection.rollback()
                return self.get(row["active_revision_id"])
            connection.execute(
                """
                INSERT INTO dataset_revisions(
                    id, dataset_id, parent_revision_id, status, relative_path, sha256,
                    metadata_json, created_at, call_id, kind, provenance_json,
                    validation_json, error_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._values(record),
            )
            connection.execute(
                "UPDATE datasets SET active_revision_id = ? WHERE id = ?",
                (record.id, record.dataset_id),
            )
            connection.commit()
        return record

    def get(self, revision_id: str) -> DatasetRevisionRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM dataset_revisions WHERE id = ?", (revision_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown revision: {revision_id}")
        return self._from_row(row)

    def get_active(self, dataset_id: str) -> DatasetRevisionRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT r.* FROM dataset_revisions r
                JOIN datasets d ON d.active_revision_id = r.id
                WHERE d.id = ?
                """,
                (dataset_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Dataset has no active revision: {dataset_id}")
        return self._from_row(row)

    def find_by_call(self, dataset_id: str, call_id: str) -> DatasetRevisionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM dataset_revisions WHERE dataset_id = ? AND call_id = ?",
                (dataset_id, call_id),
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def list_incomplete(self) -> list[DatasetRevisionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM dataset_revisions
                WHERE status IN ('draft', 'validating')
                ORDER BY created_at, id
                """
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def list(self, dataset_id: str) -> list[DatasetRevisionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM dataset_revisions
                WHERE dataset_id = ? ORDER BY created_at, id
                """,
                (dataset_id,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def committed_paths(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT relative_path FROM dataset_revisions
                WHERE status = 'committed' AND relative_path IS NOT NULL
                """
            ).fetchall()
        return {str(row[0]) for row in rows}

    def transition(
        self,
        revision_id: str,
        *,
        expected: tuple[str, ...],
        status: str,
        validation: dict | None = None,
        error: dict | None = None,
        updated_at: str,
    ) -> DatasetRevisionRecord:
        placeholders = ",".join("?" for _ in expected)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE dataset_revisions
                SET status = ?, validation_json = ?, error_json = ?, updated_at = ?
                WHERE id = ? AND status IN ({placeholders})
                """,
                (
                    status,
                    self._json_or_none(validation),
                    self._json_or_none(error),
                    updated_at,
                    revision_id,
                    *expected,
                ),
            )
            connection.commit()
        if cursor.rowcount != 1:
            raise ValueError(f"Illegal revision transition for {revision_id}")
        return self.get(revision_id)

    def commit(
        self,
        revision_id: str,
        *,
        relative_path: str,
        sha256: str,
        provenance: dict,
        updated_at: str,
    ) -> DatasetRevisionRecord:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM dataset_revisions WHERE id = ?", (revision_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(f"Unknown revision: {revision_id}")
            if row["status"] != "validating":
                connection.rollback()
                raise ValueError(f"Revision is not validating: {revision_id}")
            dataset = connection.execute(
                "SELECT active_revision_id FROM datasets WHERE id = ?", (row["dataset_id"],)
            ).fetchone()
            if dataset is None or dataset["active_revision_id"] != row["parent_revision_id"]:
                connection.rollback()
                raise RuntimeError("active revision changed")
            connection.execute(
                """
                UPDATE dataset_revisions
                SET status = 'committed', relative_path = ?, sha256 = ?,
                    provenance_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    relative_path,
                    sha256,
                    json.dumps(provenance, ensure_ascii=False, sort_keys=True),
                    updated_at,
                    revision_id,
                ),
            )
            connection.execute(
                "UPDATE datasets SET active_revision_id = ? WHERE id = ?",
                (revision_id, row["dataset_id"]),
            )
            connection.commit()
        return self.get(revision_id)

    @staticmethod
    def _values(record: DatasetRevisionRecord) -> tuple:
        return (
            record.id,
            record.dataset_id,
            record.parent_revision_id,
            record.status.value,
            record.relative_path,
            record.sha256,
            json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
            record.created_at,
            record.call_id,
            record.kind,
            json.dumps(record.provenance, ensure_ascii=False, sort_keys=True),
            RevisionRepository._json_or_none(record.validation),
            RevisionRepository._json_or_none(record.error),
            record.updated_at,
        )

    @staticmethod
    def _json_or_none(value: dict | None) -> str | None:
        return json.dumps(value, ensure_ascii=False, sort_keys=True) if value is not None else None

    @staticmethod
    def _from_row(row) -> DatasetRevisionRecord:
        payload = dict(row)
        payload["metadata"] = json.loads(payload.pop("metadata_json"))
        payload["provenance"] = json.loads(payload.pop("provenance_json"))
        for source, target in (("validation_json", "validation"), ("error_json", "error")):
            raw = payload.pop(source)
            payload[target] = json.loads(raw) if raw is not None else None
        return DatasetRevisionRecord(**payload)


class SessionRepository(_Repository):
    def create(self, record: SessionRecord) -> SessionRecord:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions(id, project_id, title, status, created_at, closed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.project_id,
                    record.title,
                    record.status,
                    record.created_at,
                    record.closed_at,
                ),
            )
            connection.commit()
        return record

    def get(self, session_id: str) -> SessionRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown session: {session_id}")
        return SessionRecord(**dict(row))

    def list(self, project_id: str) -> list[SessionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions WHERE project_id = ? ORDER BY created_at, id",
                (project_id,),
            ).fetchall()
        return [SessionRecord(**dict(row)) for row in rows]

    def close(self, session_id: str, closed_at: str) -> SessionRecord:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions
                SET status = 'closed', closed_at = ?
                WHERE id = ? AND status = 'open'
                """,
                (closed_at, session_id),
            )
            connection.commit()
        if cursor.rowcount == 0:
            existing = self.get(session_id)
            if existing.status != "closed":
                raise KeyError(f"Unknown open session: {session_id}")
            return existing
        return self.get(session_id)

    def add_message(self, record: MessageRecord) -> MessageRecord:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages(id, session_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.session_id,
                    record.role,
                    record.content,
                    json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                    record.created_at,
                ),
            )
            connection.commit()
        return record

    def list_messages(self, session_id: str) -> list[MessageRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at, id",
                (session_id,),
            ).fetchall()
        records = []
        for row in rows:
            payload = dict(row)
            payload["metadata"] = json.loads(payload.pop("metadata_json"))
            records.append(MessageRecord(**payload))
        return records


class ArtifactRepository(_Repository):
    def create(self, record: ArtifactRecord) -> ArtifactRecord:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts(
                    id, project_id, run_id, kind, relative_path,
                    sha256, size_bytes, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.project_id,
                    record.run_id,
                    record.kind,
                    record.relative_path,
                    record.sha256,
                    record.size_bytes,
                    json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                    record.created_at,
                ),
            )
            connection.commit()
        return record

    def get(self, artifact_id: str) -> ArtifactRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown artifact: {artifact_id}")
        return self._from_row(row)

    def list(self, project_id: str, *, run_id: str | None = None) -> list[ArtifactRecord]:
        if run_id is None:
            sql = "SELECT * FROM artifacts WHERE project_id = ? ORDER BY created_at, id"
            params = (project_id,)
        else:
            sql = (
                "SELECT * FROM artifacts WHERE project_id = ? AND run_id = ? "
                "ORDER BY created_at, id"
            )
            params = (project_id, run_id)
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row) -> ArtifactRecord:
        payload = dict(row)
        payload["metadata"] = json.loads(payload.pop("metadata_json"))
        return ArtifactRecord(**payload)


class WorkerJobRepository(_Repository):
    def create_queued(
        self,
        *,
        project_id: str,
        call_id: str,
        entrypoint: str,
        request: dict,
        created_at: str,
        revision_id: str | None = None,
    ) -> WorkerJobRecord:
        from st_agent.ids import IdKind, new_id

        record = WorkerJobRecord(
            id=new_id(IdKind.JOB),
            project_id=project_id,
            call_id=call_id,
            revision_id=revision_id,
            status=JobStatus.QUEUED,
            entrypoint=entrypoint,
            request=request,
            created_at=created_at,
            updated_at=created_at,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO worker_jobs(
                    id, project_id, call_id, revision_id, status, entrypoint,
                    request_json, result_json, error_json, pid, owner_pid, cancel_requested,
                    created_at, started_at, finished_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, 0, ?, NULL, NULL, ?)
                """,
                (
                    record.id,
                    record.project_id,
                    record.call_id,
                    record.revision_id,
                    record.status.value,
                    record.entrypoint,
                    json.dumps(record.request, ensure_ascii=False, sort_keys=True),
                    record.created_at,
                    record.updated_at,
                ),
            )
            connection.commit()
        return record

    def create_blocked(
        self,
        *,
        project_id: str,
        call_id: str,
        entrypoint: str,
        request: dict,
        error: dict,
        created_at: str,
        revision_id: str | None = None,
    ) -> WorkerJobRecord:
        from st_agent.ids import IdKind, new_id

        record = WorkerJobRecord(
            id=new_id(IdKind.JOB),
            project_id=project_id,
            call_id=call_id,
            revision_id=revision_id,
            status=JobStatus.BLOCKED,
            entrypoint=entrypoint,
            request=request,
            error=error,
            created_at=created_at,
            finished_at=created_at,
            updated_at=created_at,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO worker_jobs(
                    id, project_id, call_id, revision_id, status, entrypoint,
                    request_json, result_json, error_json, pid, owner_pid, cancel_requested,
                    created_at, started_at, finished_at, updated_at
                ) VALUES (?, ?, ?, ?, 'blocked', ?, ?, NULL, ?, NULL, NULL, 0, ?, NULL, ?, ?)
                """,
                (
                    record.id,
                    record.project_id,
                    record.call_id,
                    record.revision_id,
                    record.entrypoint,
                    json.dumps(record.request, ensure_ascii=False, sort_keys=True),
                    json.dumps(record.error, ensure_ascii=False, sort_keys=True),
                    record.created_at,
                    record.finished_at,
                    record.updated_at,
                ),
            )
            connection.commit()
        return record

    def get(self, job_id: str) -> WorkerJobRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM worker_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown worker job: {job_id}")
        return self._from_row(row)

    def find_by_call(self, call_id: str) -> WorkerJobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM worker_jobs WHERE call_id = ?", (call_id,)
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def has_running(self, project_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM worker_jobs
                WHERE project_id = ? AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (project_id,),
            ).fetchone()
        return row is not None

    def list_active(self, project_id: str) -> list[WorkerJobRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM worker_jobs
                WHERE project_id = ? AND status IN ('queued', 'running')
                ORDER BY created_at, id
                """,
                (project_id,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def mark_running(
        self, job_id: str, *, pid: int, owner_pid: int | None = None
    ) -> WorkerJobRecord:
        now = utc_now_iso()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE worker_jobs
                SET status = 'running', pid = ?, owner_pid = ?, started_at = ?, updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (pid, owner_pid, now, now, job_id),
            )
            connection.commit()
        if cursor.rowcount != 1:
            raise ValueError(f"Job is not queued: {job_id}")
        return self.get(job_id)

    def request_cancel(self, job_id: str) -> WorkerJobRecord:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                "UPDATE worker_jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                (now, job_id),
            )
            connection.commit()
        return self.get(job_id)

    def finish(
        self,
        job_id: str,
        *,
        status: JobStatus,
        result: dict | None = None,
        error: dict | None = None,
    ) -> WorkerJobRecord:
        if status not in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.BLOCKED,
        }:
            raise ValueError(f"Not a terminal job status: {status}")
        now = utc_now_iso()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE worker_jobs
                SET status = ?, result_json = ?, error_json = ?, finished_at = ?, updated_at = ?
                WHERE id = ? AND status IN ('queued', 'running')
                """,
                (
                    status.value,
                    self._json_or_none(result),
                    self._json_or_none(error),
                    now,
                    now,
                    job_id,
                ),
            )
            connection.commit()
        if cursor.rowcount == 0:
            existing = self.get(job_id)
            if existing.status in {
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.BLOCKED,
            }:
                return existing
            raise ValueError(f"Job cannot finish: {job_id}")
        return self.get(job_id)

    @staticmethod
    def _json_or_none(value: dict | None) -> str | None:
        return json.dumps(value, ensure_ascii=False, sort_keys=True) if value is not None else None

    @staticmethod
    def _from_row(row) -> WorkerJobRecord:
        payload = dict(row)
        for source, target in (
            ("request_json", "request"),
            ("result_json", "result"),
            ("error_json", "error"),
        ):
            raw = payload.pop(source)
            payload[target] = json.loads(raw) if raw is not None else None
        payload["cancel_requested"] = bool(payload["cancel_requested"])
        return WorkerJobRecord(**payload)


class ToolExecutionRepository(_Repository):
    def list_for_dataset(self, project_id: str, dataset_id: str) -> list[ToolResult]:
        """Persisted observations, scoped before interpretation by callers."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT result_json FROM tool_executions WHERE project_id=? AND dataset_id=? "
                "AND status IN (?,?) ORDER BY finished_at DESC,call_id DESC",
                (project_id, dataset_id, ToolStatus.SUCCESS.value, ToolStatus.WARNING.value),
            ).fetchall()
        return [ToolResult.model_validate_json(row[0]) for row in rows]

    def save(
        self,
        *,
        project_id: str,
        dataset_id: str | None,
        result: ToolResult,
        contract_version: str,
        cache_key: str | None,
    ) -> ToolResult:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tool_executions(
                    call_id, project_id, dataset_id, tool_name, contract_version,
                    input_revision_id, output_revision_id, status, cache_key,
                    result_json, result_path, created_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    result.call_id,
                    project_id,
                    dataset_id,
                    result.tool_name,
                    contract_version,
                    result.input_revision_id,
                    result.output_revision_id,
                    result.status.value,
                    cache_key,
                    result.model_dump_json(exclude_none=False),
                    result.provenance.started_at.isoformat(),
                    result.provenance.finished_at.isoformat(),
                ),
            )
            connection.commit()
        return result

    def find_by_call(self, call_id: str) -> ToolResult | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM tool_executions WHERE call_id = ?", (call_id,)
            ).fetchone()
        return ToolResult.model_validate_json(row[0]) if row is not None else None

    def find_success(self, project_id: str, cache_key: str) -> ToolResult | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT result_json FROM tool_executions
                WHERE project_id = ? AND cache_key = ? AND status = ?
                ORDER BY finished_at DESC LIMIT 1
                """,
                (project_id, cache_key, ToolStatus.SUCCESS.value),
            ).fetchone()
        return ToolResult.model_validate_json(row[0]) if row is not None else None
