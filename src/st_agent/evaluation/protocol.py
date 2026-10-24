"""Validate executable single-turn cases and fail closed on incomparable evidence."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from st_agent.intent_runtime import OPERATION_TO_TOOL


class BaselineCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    id: str = Field(min_length=1)
    utterance: str = Field(min_length=1)
    expected_tools: list[str]
    expected_status: Literal["completed", "needs_clarification", "waiting_approval", "denied"] = (
        "completed"
    )
    expected_blocked_reason: str | None = None
    reject: bool = False
    forbidden_tools: list[str] = Field(default_factory=list)
    required_fact_keys: list[str] = Field(default_factory=list)
    read_only: bool = False
    approve: bool = False

    @model_validator(mode="after")
    def check_contract(self):
        expected, forbidden = set(self.expected_tools), set(self.forbidden_tools)
        if self.approve and self.reject:
            raise ValueError("Cannot approve and reject the same request")
        if self.expected_status != "completed" and expected:
            raise ValueError("Non-completed scenario must not expect successful tools")
        if self.expected_status == "completed" and not expected:
            raise ValueError("Completed scientific scenario must expect a tool")
        if self.reject and self.expected_status != "denied":
            raise ValueError("Rejected scenario must expect denied")
        if expected & forbidden:
            raise ValueError("Expected and forbidden tools overlap")
        if (expected | forbidden) - set(OPERATION_TO_TOOL.values()):
            raise ValueError("Unknown research tool")
        if self.read_only and expected - {"dataset.inspect", "qc.compute_metrics"}:
            raise ValueError("Read-only case expects a data transformation")
        if self.read_only and self.approve:
            raise ValueError("Read-only case cannot automatically approve mutations")
        return self


def load_scenarios(path: Path) -> list[dict]:
    cases = [
        BaselineCase.model_validate_json(line).model_dump(mode="json")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("Evaluation requires at least one scenario")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate scenario id")
    return cases


def compare_reports(baseline: dict, candidate: dict) -> dict:
    if baseline.get("max_requests") != candidate.get("max_requests"):
        raise ValueError("Incompatible max_requests; compare runs with the same budget")
    for field in ("protocol_version", "model", "dataset_sha256"):
        if not baseline.get(field) or baseline.get(field) != candidate.get(field):
            raise ValueError(f"Incompatible or missing {field}; rerun both reports")
    if baseline["protocol_version"] != 2:
        raise ValueError("Unsupported protocol_version")

    def index(report):
        rows = report.get("cases", [])
        if not rows:
            raise ValueError("Report has no cases")
        indexed = {}
        for row in rows:
            case = BaselineCase.model_validate(row["case"]).model_dump(mode="json")
            if case["id"] in indexed or type(row.get("passed")) is not bool:
                raise ValueError("Duplicate case id or invalid pass result")
            indexed[case["id"]] = (case, row["passed"])
        return indexed

    before, after = index(baseline), index(candidate)
    shared = before.keys() & after.keys()
    for key in shared:
        if before[key][0] != after[key][0]:
            raise ValueError(f"Changed scenario content: {key}")
    return {
        "fixed": sorted(key for key in shared if not before[key][1] and after[key][1]),
        "regressed": sorted(key for key in shared if before[key][1] and not after[key][1]),
        "missing": sorted(before.keys() - after.keys()),
        "added": sorted(after.keys() - before.keys()),
    }
