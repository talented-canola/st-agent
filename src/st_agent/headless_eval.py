"""Isolated, headless regression scenarios for the research runtime."""

from __future__ import annotations

import json
from pathlib import Path

from st_agent.api import ResearchCopilot
from st_agent.dataset_state import DatasetState, DatasetStateInspector
from st_agent.ids import IdKind, new_id
from st_agent.planning import DeterministicPlanner, IntentMode, TaskIntent, TaskOperation
from st_agent.project_tools import ProjectToolExecutor
from st_agent.safety import ApprovalDecision
from st_agent.scientific_execution import ScientificPlanExecutor
from st_agent.utils import sha256_file


def run_qc_read_only_scenario(dataset_path: str | Path, output_directory: str | Path) -> dict:
    """Prove that a QC request leaves both source data and active revision intact.

    The output directory must be new or empty.  The registered project beneath
    it is intentionally isolated from a researcher's working project and can be
    retained as a reproduction artifact.
    """
    source = Path(dataset_path).resolve()
    destination = Path(output_directory).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Headless evaluation output directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    before_hash = sha256_file(source)
    project_root = destination / "project"
    copilot = ResearchCopilot.create(project_root, name="qc-read-only-regression")
    try:
        dataset = copilot.add_dataset(source)
        before_state = DatasetStateInspector(copilot).inspect(dataset.id)
        before_active = dataset.active_revision_id
        metrics = ProjectToolExecutor(copilot, dataset.id)("qc.compute_metrics", {})
        refreshed = copilot.get_dataset(dataset.id)
        after_state = DatasetStateInspector(copilot).inspect(dataset.id)
        after_hash = sha256_file(source)
        passed = bool(
            before_hash == after_hash
            and before_active == refreshed.active_revision_id
            and before_state.input_revision_id == after_state.input_revision_id
            and metrics["output_revision_id"] is None
        )
        report = {
            "schema_version": "research-scenario/0.1",
            "scenario": "qc_read_only",
            "passed": passed,
            "source_sha256_before": before_hash,
            "source_sha256_after": after_hash,
            "dataset_id": dataset.id,
            "active_revision_before": before_active,
            "active_revision_after": refreshed.active_revision_id,
            "state_before": before_state.model_dump(mode="json"),
            "state_after": after_state.model_dump(mode="json"),
            "tool_result": metrics,
        }
        (destination / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (destination / "report.md").write_text(_markdown_report(report), encoding="utf-8")
        if not passed:
            raise AssertionError("QC read-only scenario failed; see report.json")
        return report
    finally:
        copilot.close()


def run_approved_filter_scenario(
    dataset_path: str | Path,
    output_directory: str | Path,
    *,
    min_genes: int = 1,
    min_cells: int = 1,
    max_mito_pct: float = 100.0,
) -> dict:
    """Exercise plan, approval, worker, validation and revision commit in isolation.

    The explicit approval here is test evidence, not an automatic permission
    mechanism for researcher projects.  The scenario directory must be a new
    disposable project.
    """
    source = Path(dataset_path).resolve()
    destination = Path(output_directory).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Headless evaluation output directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    before_hash = sha256_file(source)
    copilot = ResearchCopilot.create(destination / "project", name="approved-filter-regression")
    try:
        dataset = copilot.add_dataset(source)
        session = copilot.create_session(title="approved filter regression")
        before_active = dataset.active_revision_id
        state = DatasetStateInspector(copilot).inspect(dataset.id)
        plan = DeterministicPlanner().build(
            TaskIntent(
                operation=TaskOperation.QC_FILTER,
                mode=IntentMode.EXECUTE,
                parameters={
                    "min_genes": min_genes,
                    "min_cells": min_cells,
                    "max_mito_pct": max_mito_pct,
                },
                confidence=1.0,
                rationale="Isolated regression scenario with explicit filter thresholds",
            ),
            state,
        )
        runtime = ScientificPlanExecutor(copilot)
        run_id = new_id(IdKind.RUN)
        waiting = runtime.execute(
            plan,
            dataset_id=dataset.id,
            session_id=session.id,
            run_id=run_id,
        )
        if waiting.approval_id is None:
            raise AssertionError("Derived scientific filter did not request approval")
        runtime.store.decide(waiting.approval_id, ApprovalDecision.APPROVE)
        completed = runtime.execute(
            plan,
            dataset_id=dataset.id,
            session_id=session.id,
            run_id=run_id,
            approval_id=waiting.approval_id,
        )
        result = completed.tool_result
        after = copilot.get_dataset(dataset.id)
        after_hash = sha256_file(source)
        passed = bool(
            result is not None
            and result.status.value in {"success", "warning"}
            and result.input_revision_id == before_active
            and result.output_revision_id == after.active_revision_id
            and result.output_revision_id != before_active
            and before_hash == after_hash
            and runtime.store.load_checkpoint(run_id).next_node == "completed"
        )
        report = {
            "schema_version": "research-scenario/0.1",
            "scenario": "approved_qc_filter",
            "passed": passed,
            "source_sha256_before": before_hash,
            "source_sha256_after": after_hash,
            "dataset_id": dataset.id,
            "active_revision_before": before_active,
            "active_revision_after": after.active_revision_id,
            "approval_id": waiting.approval_id,
            "run_id": run_id,
            "plan": plan.model_dump(mode="json"),
            "tool_result": result.model_dump(mode="json") if result is not None else None,
        }
        (destination / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (destination / "report.md").write_text(_markdown_filter_report(report), encoding="utf-8")
        if not passed:
            raise AssertionError("Approved filter scenario failed; see report.json")
        return report
    finally:
        copilot.close()


def _markdown_report(report: dict) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    return (
        "# QC Read-only Scenario\n\n"
        f"Status: **{status}**\n\n"
        f"- Dataset: `{report['dataset_id']}`\n"
        f"- Active revision before: `{report['active_revision_before']}`\n"
        f"- Active revision after: `{report['active_revision_after']}`\n"
        "- Source hash unchanged: "
        f"`{report['source_sha256_before'] == report['source_sha256_after']}`\n"
        f"- QC output revision: `{report['tool_result']['output_revision_id']}`\n"
    )


def _markdown_filter_report(report: dict) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    result = report["tool_result"] or {}
    return (
        "# Approved QC Filter Scenario\n\n"
        f"Status: **{status}**\n\n"
        f"- Dataset: `{report['dataset_id']}`\n"
        f"- Approval: `{report['approval_id']}`\n"
        f"- Input revision: `{report['active_revision_before']}`\n"
        f"- Output revision: `{report['active_revision_after']}`\n"
        f"- Tool status: `{result.get('status')}`\n"
        "- Source hash unchanged: "
        f"`{report['source_sha256_before'] == report['source_sha256_after']}`\n"
    )


def state_payload(state: DatasetState) -> dict:
    """Small explicit helper for callers that need a serialized state contract."""
    return state.model_dump(mode="json")
