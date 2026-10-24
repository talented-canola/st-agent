"""Composable validators independent from tool execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from st_agent.contracts import ValidationReport, ValidationStatus


class Validator(Protocol):
    def validate(self, candidate: Path, context: Mapping[str, Any]) -> ValidationReport: ...


class CompositeValidator:
    def __init__(self, validators: Sequence[Validator]) -> None:
        self.validators = tuple(validators)

    def validate(self, candidate: Path, context: Mapping[str, Any]) -> ValidationReport:
        checks: dict[str, Any] = {}
        warnings: list[str] = []
        summaries: list[str] = []
        for validator in self.validators:
            report = validator.validate(candidate, context)
            summaries.append(report.summary)
            checks.update(report.checks)
            warnings.extend(report.warnings)
            if report.status is ValidationStatus.FAILED:
                return ValidationReport(
                    status=ValidationStatus.FAILED,
                    summary=report.summary,
                    checks=checks,
                    warnings=tuple(warnings),
                    error=report.error,
                )
        status = (
            ValidationStatus.PASSED_WITH_WARNINGS if warnings else ValidationStatus.PASSED
        )
        return ValidationReport(
            status=status,
            summary="; ".join(summaries) or "No validators were configured",
            checks=checks,
            warnings=tuple(warnings),
        )
