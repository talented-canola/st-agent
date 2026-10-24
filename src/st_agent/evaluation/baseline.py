"""Capture actual workbench behavior; failures are evidence, not waived passes."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from st_agent.api import ResearchCopilot
from st_agent.config import Settings
from st_agent.deepseek import DeepSeekClient
from st_agent.evaluation.protocol import load_scenarios
from st_agent.safety import ApprovalDecision
from st_agent.utils import sha256_file
from st_agent.workbench import WorkbenchService


def run_baseline(
    dataset: Path, output: Path, cases: Path, *, settings=None, max_requests=100
) -> dict:
    settings = settings or Settings.load()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evidence directory must be empty")
    if not settings.deepseek_api_key:
        raise ValueError("provider.secret_missing")
    scenarios = load_scenarios(cases)
    output.mkdir(parents=True, exist_ok=True)
    original_hash = sha256_file(dataset)
    results = []
    client = DeepSeekClient(
        settings.deepseek_api_key.get_secret_value(),
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        max_requests=max_requests,
    )
    try:
        for index, case in enumerate(scenarios):
            with ResearchCopilot.create(output / f"case-{index:03d}", settings=settings) as project:
                registered = project.add_dataset(dataset)
                service = WorkbenchService(project)
                session = service.start_chat(case["id"])
                started = perf_counter()
                request_start = len(client.request_metrics())
                row = {"case": case, "passed": False, "error_type": None}
                try:
                    result = service.run_analysis_message(
                        session.id,
                        dataset_id=registered.id,
                        content=case["utterance"],
                        client=client,
                    )
                    waiting_trace = result.state if result else {}
                    if result and result.status.value == "waiting_approval" and case.get("approve"):
                        approval_id = result.state["approval_id"]
                        service.decide_approval(approval_id, ApprovalDecision.APPROVE)
                        result = service.resume_analysis(approval_id, client=client)
                    elif (
                        result and result.status.value == "waiting_approval" and case.get("reject")
                    ):
                        approval_id = result.state["approval_id"]
                        service.decide_approval(approval_id, ApprovalDecision.REJECT)
                        result = service.resume_analysis(approval_id, client=client)
                    row["trace"] = result.state if result else waiting_trace
                    tools = [fact["tool"] for fact in result.facts] if result else []
                    row["result"] = result.model_dump(mode="json") if result else None
                    row["checks"] = {
                        "expected_status": result is not None
                        and result.status.value == case["expected_status"],
                        "expected_tools": sorted(case["expected_tools"]) == sorted(tools),
                        "forbidden_tools": not set(case.get("forbidden_tools", [])) & set(tools),
                        "raw_unchanged": sha256_file(dataset) == original_hash,
                    }
                    if case.get("expected_blocked_reason"):
                        row["checks"]["blocked_reason"] = (
                            result is not None
                            and result.state.get("plan", {}).get("blocked_reason")
                            == case["expected_blocked_reason"]
                        )
                    if case["expected_status"] != "completed":
                        row["checks"]["no_tool_execution"] = (
                            result is not None
                            and result.tool_calls == 0
                            and not result.state.get("tool_result")
                        )
                    facts = {record.key: record for record in result.fact_records} if result else {}
                    required = case.get("required_fact_keys", [])
                    row["checks"]["requested_facts_present"] = all(key in facts for key in required)
                    row["checks"]["requested_facts_cited"] = all(
                        key in facts and f"[fact:{facts[key].id}]" in result.answer
                        for key in required
                    )
                    if case.get("read_only") or case["expected_status"] != "completed":
                        row["checks"]["active_unchanged"] = (
                            project.get_dataset(registered.id).active_revision_id
                            == registered.active_revision_id
                        )
                    row["passed"] = all(row["checks"].values())
                except Exception as error:
                    row["error_type"] = type(error).__name__
                row["elapsed_seconds"] = perf_counter() - started
                row["provider_requests"] = client.request_metrics()[request_start:]
                results.append(row)
                # Flush each case so interruption preserves the completed evidence.
                (output / f"trace-{index:03d}.json").write_text(
                    json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                if not row["passed"]:
                    (output / f"replay-{index:03d}.jsonl").write_text(
                        json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8"
                    )
    finally:
        client._client.close()
    report = {
        "protocol_version": 2,
        "max_requests": max_requests,
        "provider_requests": client.request_metrics(),
        "dataset_sha256": original_hash,
        "model": settings.deepseek_model,
        "total": len(results),
        "passed": sum(row["passed"] for row in results),
        "cases": results,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "report.md").write_text(
        f"# Live Workbench Evaluation\n\nPassed: {report['passed']}/{report['total']}\n\n"
        + "\n".join(
            f"- {row['case']['id']}: {'PASS' if row['passed'] else 'FAIL'}" for row in results
        ),
        encoding="utf-8",
    )
    return report
