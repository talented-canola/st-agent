"""Structured errors used by tools, workers, validators, and policy code."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from st_agent.serialization import ensure_payload_size, validate_json_value


class ErrorCategory(StrEnum):
    INVALID_INPUT = "InvalidInput"
    PRECONDITION_FAILED = "PreconditionFailed"
    DATA_VALIDATION_FAILED = "DataValidationFailed"
    RESOURCE_LIMIT = "ResourceLimit"
    TRANSIENT_EXTERNAL = "TransientExternal"
    TOOL_EXECUTION_FAILED = "ToolExecutionFailed"
    SCIENTIFIC_QUALITY_FAILED = "ScientificQualityFailed"
    POLICY_BLOCKED = "PolicyBlocked"
    CANCELLED = "Cancelled"
    INTERNAL_BUG = "InternalBug"


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    code: str = Field(min_length=1)
    category: ErrorCategory
    message: str = Field(min_length=1)
    retryable: bool
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("details")
    @classmethod
    def _small_json_details(cls, value: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_json_value(value)
        ensure_payload_size(normalized, maximum_bytes=32 * 1024)
        return normalized


class AgentError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        category: ErrorCategory,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.category = category
        self.retryable = retryable
        self.details = validate_json_value(details or {})

    def to_detail(self) -> ErrorDetail:
        return ErrorDetail(
            code=self.code,
            category=self.category,
            message=str(self),
            retryable=self.retryable,
            details=self.details,
        )


class InvalidInputError(AgentError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "input.invalid",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            category=ErrorCategory.INVALID_INPUT,
            details=details,
        )


class PreconditionFailedError(AgentError):
    def __init__(self, message: str, *, code: str = "precondition.failed") -> None:
        super().__init__(message, code=code, category=ErrorCategory.PRECONDITION_FAILED)


class DataValidationError(AgentError):
    def __init__(self, message: str, *, code: str = "data.invalid") -> None:
        super().__init__(message, code=code, category=ErrorCategory.DATA_VALIDATION_FAILED)


class ResourceLimitError(AgentError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "resource.insufficient",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            category=ErrorCategory.RESOURCE_LIMIT,
            details=details,
        )


class ToolExecutionError(AgentError):
    def __init__(self, message: str, *, code: str = "tool.execution_failed") -> None:
        super().__init__(message, code=code, category=ErrorCategory.TOOL_EXECUTION_FAILED)


class ScientificQualityError(AgentError):
    def __init__(self, message: str, *, code: str = "science.quality_failed") -> None:
        super().__init__(message, code=code, category=ErrorCategory.SCIENTIFIC_QUALITY_FAILED)


class CancelledAgentError(AgentError):
    def __init__(self, message: str = "Execution was cancelled") -> None:
        super().__init__(message, code="execution.cancelled", category=ErrorCategory.CANCELLED)


def error_detail_from_exception(error: BaseException) -> ErrorDetail:
    if isinstance(error, AgentError):
        return error.to_detail()
    return ErrorDetail(
        code="internal.unexpected",
        category=ErrorCategory.INTERNAL_BUG,
        message="An unexpected internal error occurred",
        retryable=False,
        details={"exception_type": type(error).__name__},
    )
