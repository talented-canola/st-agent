"""Strict, serializable contracts shared by the tool runtime."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from st_agent.errors import ErrorDetail
from st_agent.ids import IdKind, validate_id
from st_agent.serialization import ensure_payload_size, validate_json_value


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Admission(StrEnum):
    VERIFIED = "verified"
    CANDIDATE = "candidate"
    EXPERIMENTAL = "experimental"
    RETIRED = "retired"


class Risk(StrEnum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"


class Effect(StrEnum):
    READ_ONLY = "read_only"
    DERIVED_REVISION = "derived_revision"
    EXTERNAL_IO = "external_io"


class CostClass(StrEnum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class ToolStatus(StrEnum):
    SUCCESS = "success"
    WARNING = "warning"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationStatus(StrEnum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class ToolSpec(StrictRecord):
    name: str = Field(pattern=r"^[a-z][a-z0-9_.-]+$")
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    admission: Admission
    risk: Risk
    input_schema_ref: str = Field(min_length=1)
    output_schema_ref: str = Field(min_length=1)
    preconditions: tuple[str, ...]
    effects: Effect
    deterministic: bool
    seed_policy: str = Field(min_length=1)
    timeout_seconds: int = Field(ge=1)
    cost_class: CostClass
    executor: str = Field(min_length=3)
    validator: str = Field(min_length=3)


class Provenance(StrictRecord):
    tool_version: str
    dataset_hash: str | None
    parameters: dict[str, Any]
    random_seed: int | None = None
    package_versions: dict[str, str]
    dependency_lock_hash: str
    started_at: datetime
    finished_at: datetime
    duration_ms: int = Field(ge=0)

    @field_validator("parameters")
    @classmethod
    def _parameters_are_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_json_value(value)

    @model_validator(mode="after")
    def _time_order_is_valid(self) -> Provenance:
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("Provenance timestamps must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        return self


class ArtifactRef(StrictRecord):
    artifact_id: str
    kind: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("artifact_id")
    @classmethod
    def _artifact_id(cls, value: str) -> str:
        return validate_id(value, IdKind.ARTIFACT)

    @field_validator("relative_path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Artifact path must stay inside the project")
        return path.as_posix()

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_json_value(value)


class ToolResult(StrictRecord):
    schema_version: str = Field(default="tool-result/0.1", pattern=r"^tool-result/0\.1$")
    call_id: str
    tool_name: str = Field(min_length=1)
    status: ToolStatus
    summary: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    warnings: tuple[str, ...] = ()
    error: ErrorDetail | None = None
    input_revision_id: str | None = None
    output_revision_id: str | None = None
    citations: tuple[dict[str, Any], ...] = ()
    provenance: Provenance

    @field_validator("call_id")
    @classmethod
    def _call_id(cls, value: str) -> str:
        return validate_id(value, IdKind.TOOL_CALL)

    @field_validator("input_revision_id", "output_revision_id")
    @classmethod
    def _revision_id(cls, value: str | None) -> str | None:
        return validate_id(value, IdKind.REVISION) if value is not None else None

    @field_validator("metrics")
    @classmethod
    def _metrics_are_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_json_value(value)

    @field_validator("citations")
    @classmethod
    def _citations_are_json(
        cls, value: tuple[dict[str, Any], ...]
    ) -> tuple[dict[str, Any], ...]:
        return tuple(validate_json_value(item) for item in value)

    @model_validator(mode="after")
    def _status_matches_error(self) -> ToolResult:
        failure_statuses = {ToolStatus.BLOCKED, ToolStatus.FAILED, ToolStatus.CANCELLED}
        if self.status in failure_statuses and self.error is None:
            raise ValueError(f"{self.status.value} requires an error")
        if self.status not in failure_statuses and self.error is not None:
            raise ValueError(f"{self.status.value} cannot contain an error")
        if self.status is ToolStatus.WARNING and not self.warnings:
            raise ValueError("warning status requires at least one warning")
        ensure_payload_size(self.model_dump(mode="json", exclude_none=False))
        return self


class ValidationReport(StrictRecord):
    status: ValidationStatus
    summary: str
    checks: dict[str, Any] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: ErrorDetail | None = None

    @field_validator("checks")
    @classmethod
    def _checks_are_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_json_value(value)

    @model_validator(mode="after")
    def _failure_has_error(self) -> ValidationReport:
        if self.status is ValidationStatus.FAILED and self.error is None:
            raise ValueError("failed validation requires an error")
        if self.status is not ValidationStatus.FAILED and self.error is not None:
            raise ValueError("passing validation cannot contain an error")
        return self


class ResourceRequest(StrictRecord):
    estimated_ram_bytes: int = Field(ge=0)
    estimated_temp_bytes: int = Field(ge=0)
    minimum_free_ram_bytes: int = Field(default=0, ge=0)
    minimum_free_disk_bytes: int = Field(default=0, ge=0)
    timeout_seconds: int = Field(ge=1)
    heavy: bool = True


class ResourceDecision(StrictRecord):
    allowed: bool
    snapshot: dict[str, int]
    error: ErrorDetail | None = None

    @model_validator(mode="after")
    def _decision_matches_error(self) -> ResourceDecision:
        if self.allowed == (self.error is not None):
            raise ValueError("Allowed decisions cannot have errors; blocked decisions require one")
        return self
