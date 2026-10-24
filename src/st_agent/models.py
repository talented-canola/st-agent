"""Small persistent records stored in SQLite."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from st_agent.serialization import ensure_payload_size, validate_json_value


class FrozenRecord(BaseModel):
    model_config = ConfigDict(frozen=True)


class ProjectRecord(FrozenRecord):
    id: str
    name: str
    root_path: str
    schema_version: int
    created_at: str
    updated_at: str


class DatasetRecord(FrozenRecord):
    id: str
    project_id: str
    display_name: str
    source_path: str
    storage_path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    status: Literal["registered", "missing", "invalid"] = "registered"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    active_revision_id: str | None = None

    @field_validator("metadata")
    @classmethod
    def _small_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_json_value(value)
        ensure_payload_size(normalized)
        return normalized


class RevisionStatus(StrEnum):
    DRAFT = "draft"
    VALIDATING = "validating"
    COMMITTED = "committed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class DatasetRevisionRecord(FrozenRecord):
    id: str
    dataset_id: str
    parent_revision_id: str | None = None
    status: RevisionStatus
    relative_path: str | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    call_id: str | None = None
    kind: Literal["raw", "derived"] = "derived"
    provenance: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    updated_at: str

    @field_validator("metadata", "provenance", "validation", "error")
    @classmethod
    def _small_json_fields(cls, value):
        if value is None:
            return None
        normalized = validate_json_value(value)
        ensure_payload_size(normalized)
        return normalized


class WorkerJobRecord(FrozenRecord):
    id: str
    project_id: str
    call_id: str
    revision_id: str | None = None
    status: JobStatus
    entrypoint: str
    request: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    pid: int | None = None
    owner_pid: int | None = None
    cancel_requested: bool = False
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str

    @field_validator("request", "result", "error")
    @classmethod
    def _small_json_fields(cls, value):
        if value is None:
            return None
        normalized = validate_json_value(value)
        ensure_payload_size(normalized)
        return normalized


class SessionRecord(FrozenRecord):
    id: str
    project_id: str
    title: str | None = None
    status: Literal["open", "closed"] = "open"
    created_at: str
    closed_at: str | None = None


class MessageRecord(FrozenRecord):
    id: str
    session_id: str
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str

    @field_validator("metadata")
    @classmethod
    def _small_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_json_value(value)
        ensure_payload_size(normalized)
        return normalized


class ArtifactRecord(FrozenRecord):
    id: str
    project_id: str
    run_id: str | None = None
    kind: str
    relative_path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str

    @field_validator("metadata")
    @classmethod
    def _small_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_json_value(value)
        ensure_payload_size(normalized)
        return normalized
