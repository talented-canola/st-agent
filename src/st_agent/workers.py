"""Spawn-isolated local workers with persistent status and bounded cancellation."""

from __future__ import annotations

import importlib
import multiprocessing
import os
import re
import sqlite3
import time
from pathlib import Path
from queue import Empty
from typing import Any

import psutil
from pydantic import BaseModel, ConfigDict

from st_agent.contracts import ResourceRequest
from st_agent.errors import (
    CancelledAgentError,
    ErrorCategory,
    ErrorDetail,
    error_detail_from_exception,
)
from st_agent.ids import IdKind, validate_id
from st_agent.models import JobStatus, WorkerJobRecord
from st_agent.project import ProjectLayout
from st_agent.repositories import WorkerJobRepository
from st_agent.resources import ResourceAdmission
from st_agent.serialization import ensure_payload_size, validate_json_value
from st_agent.utils import utc_now_iso
from st_agent.trace import emit

ENTRYPOINT_PATTERN = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$"
)


class WorkerOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    job_id: str
    status: JobStatus
    result: dict[str, Any] | None = None
    error: ErrorDetail | None = None


def _resolve_entrypoint(reference: str):
    module_name, function_name = reference.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"Worker entrypoint is not callable: {reference}")
    return function


def _worker_bootstrap(
    entrypoint: str,
    request: dict[str, Any],
    workspace: str,
    cancel_event,
    result_queue,
) -> None:
    try:
        function = _resolve_entrypoint(entrypoint)
        result = function(request, Path(workspace), cancel_event.is_set)
        normalized = validate_json_value(result)
        if not isinstance(normalized, dict):
            raise TypeError("Worker result must be a JSON object")
        ensure_payload_size(normalized)
        result_queue.put({"status": JobStatus.SUCCEEDED.value, "result": normalized})
    except BaseException as error:
        detail = error_detail_from_exception(error)
        status = (
            JobStatus.CANCELLED
            if detail.category is ErrorCategory.CANCELLED
            else JobStatus.FAILED
        )
        result_queue.put(
            {
                "status": status.value,
                "error": detail.model_dump(mode="json"),
            }
        )


class WorkerHandle:
    def __init__(
        self,
        manager: WorkerManager,
        job_id: str,
        workspace: Path,
        process=None,
        result_queue=None,
        cancel_event=None,
    ) -> None:
        self.manager = manager
        self.job_id = job_id
        self.workspace = workspace
        self._process = process
        self._result_queue = result_queue
        self._cancel_event = cancel_event

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else self.manager.repository.get(
            self.job_id
        ).pid

    def is_alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    def wait(self, timeout_seconds: float | None = None) -> WorkerOutcome:
        return self.manager._wait(self, timeout_seconds=timeout_seconds)

    def cancel(self, grace_seconds: float = 2.0) -> WorkerOutcome:
        return self.manager._cancel(self, grace_seconds=grace_seconds)


class WorkerManager:
    def __init__(
        self,
        layout: ProjectLayout,
        project_id: str,
        repository: WorkerJobRepository,
        *,
        resource_admission: ResourceAdmission | None = None,
    ) -> None:
        self.layout = layout
        self.project_id = validate_id(project_id, IdKind.PROJECT)
        self.repository = repository
        self.resource_admission = resource_admission or ResourceAdmission()
        self._context = multiprocessing.get_context("spawn")
        self._handles: dict[str, WorkerHandle] = {}

    def start(
        self,
        *,
        call_id: str,
        entrypoint: str,
        request: dict[str, Any],
        resources: ResourceRequest,
        revision_id: str | None = None,
    ) -> WorkerHandle:
        validate_id(call_id, IdKind.TOOL_CALL)
        if revision_id is not None:
            validate_id(revision_id, IdKind.REVISION)
        if ENTRYPOINT_PATTERN.fullmatch(entrypoint) is None:
            raise ValueError(f"Invalid worker entrypoint: {entrypoint}")
        normalized_request = validate_json_value(request)
        ensure_payload_size(normalized_request)
        existing = self.repository.find_by_call(call_id)
        if existing is not None:
            return self._handles.get(existing.id) or WorkerHandle(
                self, existing.id, self._workspace(existing.id)
            )

        decision = self.resource_admission.check(
            resources,
            project_path=self.layout.root,
            heavy_job_running=resources.heavy and self.repository.has_running(self.project_id),
        )
        emit(self, "resource.decision", status="allowed" if decision.allowed else "blocked",
             error_code=decision.error.code if decision.error else None,
             snapshot=decision.snapshot, request=resources.model_dump(mode="json"), call_id=call_id)
        if not decision.allowed:
            if decision.error is None:
                raise RuntimeError("Blocked resource decision is missing an error")
            blocked = self.repository.create_blocked(
                project_id=self.project_id,
                call_id=call_id,
                entrypoint=entrypoint,
                request=normalized_request,
                revision_id=revision_id,
                error=decision.error.model_dump(mode="json"),
                created_at=utc_now_iso(),
            )
            handle = WorkerHandle(self, blocked.id, self._workspace(blocked.id))
            self._handles[blocked.id] = handle
            return handle

        try:
            record = self.repository.create_queued(
                project_id=self.project_id,
                call_id=call_id,
                entrypoint=entrypoint,
                request=normalized_request,
                revision_id=revision_id,
                created_at=utc_now_iso(),
            )
        except sqlite3.IntegrityError:
            replay = self.repository.find_by_call(call_id)
            if replay is not None:
                return self._handles.get(replay.id) or WorkerHandle(
                    self, replay.id, self._workspace(replay.id)
                )
            detail = ErrorDetail(
                code="resource.heavy_job_running",
                category=ErrorCategory.RESOURCE_LIMIT,
                message="Another heavy job acquired the project worker slot",
                retryable=True,
                details={},
            )
            blocked = self.repository.create_blocked(
                project_id=self.project_id,
                call_id=call_id,
                entrypoint=entrypoint,
                request=normalized_request,
                revision_id=revision_id,
                error=detail.model_dump(mode="json"),
                created_at=utc_now_iso(),
            )
            handle = WorkerHandle(self, blocked.id, self._workspace(blocked.id))
            self._handles[blocked.id] = handle
            return handle
        workspace = self._workspace(record.id)

        workspace.mkdir(parents=True, exist_ok=False)
        result_queue = self._context.Queue(maxsize=1)
        cancel_event = self._context.Event()
        process = self._context.Process(
            target=_worker_bootstrap,
            args=(entrypoint, normalized_request, str(workspace), cancel_event, result_queue),
            name=f"st-agent-{record.id}",
        )
        handle = WorkerHandle(
            self,
            record.id,
            workspace,
            process=process,
            result_queue=result_queue,
            cancel_event=cancel_event,
        )
        self._handles[record.id] = handle
        try:
            process.start()
            self.repository.mark_running(
                record.id, pid=process.pid, owner_pid=os.getpid()
            )
        except BaseException as error:
            detail = error_detail_from_exception(error)
            self.repository.finish(
                record.id,
                status=JobStatus.FAILED,
                error=detail.model_dump(mode="json"),
            )
            raise
        return handle

    def recover_abandoned(self) -> list[str]:
        detail = ErrorDetail(
            code="worker.abandoned_after_restart",
            category=ErrorCategory.TOOL_EXECUTION_FAILED,
            message="Worker was running when the application stopped",
            retryable=True,
            details={},
        )
        recovered: list[str] = []
        for job in self.repository.list_active(self.project_id):
            if job.owner_pid is not None and psutil.pid_exists(job.owner_pid):
                continue
            self.repository.finish(
                job.id,
                status=JobStatus.FAILED,
                error=detail.model_dump(mode="json"),
            )
            recovered.append(job.id)
        return recovered

    def _wait(
        self, handle: WorkerHandle, *, timeout_seconds: float | None
    ) -> WorkerOutcome:
        current = self.repository.get(handle.job_id)
        if current.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            return self._outcome(current)
        if handle._process is None:
            return self._outcome(current)
        deadline = None if timeout_seconds is None else time.monotonic() + max(0, timeout_seconds)
        payload = None
        # A child's Queue feeder cannot finish until the parent drains the
        # pipe. Joining first deadlocks valid results larger than its buffer.
        # Receive while it is alive, under the SAME total timeout, then join.
        while True:
            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            if remaining == 0:
                break
            try:
                payload = handle._result_queue.get(timeout=min(0.1, remaining) if remaining is not None else 0.1)
                break
            except Empty:
                if not handle._process.is_alive():
                    break
            except (EOFError, OSError):
                break
        remaining = None if deadline is None else max(0, deadline - time.monotonic())
        handle._process.join(remaining if payload is not None else 0)
        if handle._process.is_alive():
            detail = ErrorDetail(
                code="worker.timeout",
                category=ErrorCategory.RESOURCE_LIMIT,
                message="Worker exceeded its execution timeout",
                retryable=True,
                details={"timeout_seconds": timeout_seconds},
            )
            handle._process.terminate()
            handle._process.join(2)
            record = self.repository.finish(
                handle.job_id,
                status=JobStatus.FAILED,
                error=detail.model_dump(mode="json"),
            )
            return self._outcome(record)
        if payload is None:
            detail = ErrorDetail(
                code="worker.exited_without_result",
                category=ErrorCategory.TOOL_EXECUTION_FAILED,
                message="Worker exited without a structured result",
                retryable=True,
                details={"exit_code": handle._process.exitcode},
            )
            record = self.repository.finish(
                handle.job_id,
                status=JobStatus.FAILED,
                error=detail.model_dump(mode="json"),
            )
            return self._outcome(record)
        status = JobStatus(payload["status"])
        record = self.repository.finish(
            handle.job_id,
            status=status,
            result=payload.get("result"),
            error=payload.get("error"),
        )
        return self._outcome(record)

    def _cancel(self, handle: WorkerHandle, *, grace_seconds: float) -> WorkerOutcome:
        current = self.repository.get(handle.job_id)
        if current.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            return self._outcome(current)
        self.repository.request_cancel(handle.job_id)
        if handle._cancel_event is not None:
            handle._cancel_event.set()
        if handle._process is not None:
            handle._process.join(max(0, grace_seconds))
            if handle._process.is_alive():
                handle._process.terminate()
                handle._process.join(2)
        detail = error_detail_from_exception(CancelledAgentError())
        record = self.repository.finish(
            handle.job_id,
            status=JobStatus.CANCELLED,
            error=detail.model_dump(mode="json"),
        )
        return self._outcome(record)

    def _workspace(self, job_id: str) -> Path:
        return self.layout.root / "tmp" / "jobs" / job_id

    @staticmethod
    def _outcome(record: WorkerJobRecord) -> WorkerOutcome:
        error = None
        if record.error is not None:
            payload = dict(record.error)
            payload["category"] = ErrorCategory(payload["category"])
            error = ErrorDetail.model_validate(payload)
        return WorkerOutcome(
            job_id=record.id,
            status=record.status,
            result=record.result,
            error=error,
        )
