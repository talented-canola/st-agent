"""Scientific figures, vision checks, and provenance-complete HTML reports."""

from __future__ import annotations

import html
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field

from st_agent.contracts import StrictRecord
from st_agent.db import connect_database
from st_agent.ids import IdKind, new_id
from st_agent.serialization import canonical_json, validate_json_value
from st_agent.utils import utc_now_iso


class ReportStatus(StrEnum):
    DRAFT = "draft"
    FINALIZED = "finalized"


class PlotSpec(StrictRecord):
    data: tuple[dict[str, Any], ...]
    layout: dict[str, Any] = Field(default_factory=dict)

    def as_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


class PngRenderer(Protocol):
    def __call__(self, plot_json: str) -> bytes: ...


def render_png_on_demand(spec: PlotSpec, renderer: PngRenderer) -> bytes:
    rendered = renderer(spec.as_json())
    if not rendered.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Renderer did not return PNG data")
    return rendered


class VisionReview(StrictRecord):
    accepted_claims: tuple[str, ...]
    rejected_claims: tuple[str, ...]


class VisionGuard:
    NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")

    def prepare_upload(self, artifact_id: str, *, consent: bool) -> dict[str, str]:
        if not consent:
            raise PermissionError("Image upload requires researcher consent")
        return {"artifact_id": artifact_id, "classification": "research_figure"}

    def validate_claims(self, claims: list[str], structured_facts: dict[str, Any]) -> VisionReview:
        evidence = canonical_json(validate_json_value(structured_facts))
        accepted, rejected = [], []
        for claim in claims:
            numbers = self.NUMBER.findall(claim)
            (accepted if all(number in evidence for number in numbers) else rejected).append(claim)
        return VisionReview(accepted_claims=tuple(accepted), rejected_claims=tuple(rejected))


class ReportRecord(StrictRecord):
    id: str
    project_id: str
    run_id: str | None = None
    title: str
    status: ReportStatus
    html: str
    tool_call_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    provenance: dict[str, Any]
    created_at: str
    finalized_at: str | None = None


class ReportFactBinding(StrictRecord):
    call_id: str
    fact_id: str
    key: str
    value: str | int | float | bool | None
    revision_id: str | None


class ReportSection(StrictRecord):
    kind: Literal["facts", "claim"]
    title: str
    facts: list[ReportFactBinding] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    text: str | None = None


def report_tool_facts(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand exact persisted metrics, retaining the runtime's fact identities."""
    from st_agent.agent_runtime import AgentEngine

    metrics = result.get("metrics")
    if metrics is None:  # Safe adapter for the original flat result API.
        metrics = {key: value for key, value in result.items() if key not in {
            "call_id", "status", "tool_name", "input_revision_id", "output_revision_id"}}
    if not isinstance(metrics, dict):
        raise ValueError("Report tool metrics binding must be an object")
    records = AgentEngine._records_from_result(result.get("tool_name", "legacy"), {
        **metrics, "input_revision_id": result.get("input_revision_id"),
        "output_revision_id": result.get("output_revision_id")}, 0)
    return [{"call_id": result["call_id"], "fact_id": record.id,
        "key": record.key, "value": record.value, "revision_id": record.revision_id}
        for record in records]


def _validated_sections(sections, tool_results, claims):
    known_calls = {item["call_id"] for item in tool_results}
    known_claims = {item["id"]: item for item in claims if item.get("id")}
    facts = {(fact["call_id"], fact["fact_id"]): fact
             for result in tool_results for fact in report_tool_facts(result)}
    rendered, bindings = [], []
    if not sections:
        raise ValueError("Report has no evidence binding")
    for raw in sections:
        if "kind" in raw:
            section = ReportSection.model_validate(raw)
            if section.kind == "facts":
                if not section.facts or section.text is not None or section.claim_ids:
                    raise ValueError("Report fact section has an unsupported text binding")
                lines = []
                for binding in section.facts:
                    item = binding.model_dump(mode="json")
                    expected = facts.get((binding.call_id, binding.fact_id))
                    if expected is None or any(item[key] != expected[key] for key in item) or \
                            type(item["value"]) is not type(expected["value"]):
                        raise ValueError("Report fact binding is untraceable or references an unknown fact")
                    lines.append(f"{binding.key}：{binding.value} "
                        f"[call:{binding.call_id}; fact:{binding.fact_id}; revision:{binding.revision_id}]")
                text = "\n".join(lines)
            else:
                if section.facts or not section.claim_ids:
                    raise ValueError("Report claim section has no evidence binding")
                if any(cid not in known_claims or not known_claims[cid].get("text")
                       for cid in section.claim_ids):
                    raise ValueError("Report section references an unknown claim text")
                text = "\n".join(known_claims[cid]["text"] for cid in section.claim_ids)
                if section.text is not None and section.text != text:
                    raise ValueError("Report claim text binding is untraceable")
            bindings.append(section.model_dump(mode="json"))
            rendered.append((section.title, text))
            continue
        # Legacy free text can only repeat a supported claim verbatim or a
        # deterministic original scalar template. Sharing digits is not proof.
        calls = set(raw.get("tool_call_ids", []))
        claim_ids = set(raw.get("claim_ids", []))
        if not (calls or claim_ids) or not calls <= known_calls or not claim_ids <= set(known_claims):
            raise ValueError("Report section references an unknown evidence binding")
        allowed = {known_claims[cid]["text"] for cid in claim_ids if known_claims[cid].get("text")}
        for fact in facts.values():
            if fact["call_id"] in calls and fact["key"] in {"cells", "shape[0]"} and type(fact["value"]) is int:
                allowed.add(f"There are {fact['value']} cells.")
        if raw.get("text") not in allowed:
            raise ValueError("Report contains untraceable text without an exact supported binding")
        rendered.append((str(raw["title"]), raw["text"]))
        bindings.append({"kind": "legacy_exact", **raw})
    return rendered, bindings


class ReportStore:
    def __init__(self, database_path: str | Path, project_id: str):
        self.database_path = Path(database_path)
        self.project_id = project_id

    def _connect(self):
        return connect_database(self.database_path)

    def create(
        self,
        *,
        title: str,
        sections: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        claims: list[dict[str, Any]],
        provenance: dict[str, Any],
        run_id: str | None = None,
    ) -> ReportRecord:
        if not tool_results or any(
            item.get("status") not in {"success", "warning"} for item in tool_results
        ):
            raise ValueError("Report requires successful persisted tool results")
        if any(item.get("status") != "supported" for item in claims):
            raise ValueError("Report contains an unsupported claim")
        known_calls = {str(item["call_id"]) for item in tool_results if item.get("call_id")}
        if len(known_calls) != len(tool_results):
            raise ValueError("Report requires unique tool call bindings")
        known_claims = {str(item["id"]) for item in claims if item.get("id")}
        rendered, bindings = _validated_sections(sections, tool_results, claims)
        provenance = {**provenance, "report_contract": "bound-sections/2", "sections": bindings,
                      "claims": claims}
        body = [f"<h1>{html.escape(title)}</h1>"]
        for section_title, text in rendered:
            body.append(f"<h2>{html.escape(section_title)}</h2>")
            body.append("<p>" + html.escape(text).replace("\n", "<br>") + "</p>")
        body.append(
            '<script type="application/json" id="st-agent-provenance">'
            + canonical_json(provenance).replace("<", "\\u003c")
            + "</script>"
        )
        report = ReportRecord(
            id=new_id(IdKind.ARTIFACT),
            project_id=self.project_id,
            run_id=run_id,
            title=title,
            status=ReportStatus.DRAFT,
            html=("<!doctype html><html><head><meta charset=\"utf-8\">"
                  "<style>html{color-scheme:light;background:#fff}"
                  "body{background:#fff;color:#17202a;font-family:system-ui,sans-serif;"
                  "line-height:1.6;padding:16px;overflow-wrap:anywhere}</style>"
                  "</head><body>" + "".join(body) + "</body></html>"),
            tool_call_ids=tuple(sorted(known_calls)),
            claim_ids=tuple(sorted(known_claims)),
            provenance=validate_json_value(provenance),
            created_at=utc_now_iso(),
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO research_reports VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    report.id,
                    report.project_id,
                    report.run_id,
                    report.title,
                    report.status.value,
                    report.html,
                    canonical_json(report.tool_call_ids),
                    canonical_json(report.claim_ids),
                    canonical_json(report.provenance),
                    report.created_at,
                    None,
                ),
            )
            connection.commit()
        return report

    def finalize(self, report_id: str, *, researcher_confirmed: bool) -> ReportRecord:
        if not researcher_confirmed:
            raise PermissionError("Report finalization requires researcher confirmation")
        with self._connect() as connection:
            connection.execute(
                """UPDATE research_reports SET status='finalized',finalized_at=?
                WHERE id=? AND project_id=? AND status='draft'""",
                (utc_now_iso(), report_id, self.project_id),
            )
            connection.commit()
        return self.get(report_id)

    def get(self, report_id: str) -> ReportRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_reports WHERE id=? AND project_id=?",
                (report_id, self.project_id),
            ).fetchone()
        if row is None:
            raise KeyError(report_id)
        return ReportRecord(
            id=row["id"],
            project_id=row["project_id"],
            run_id=row["run_id"],
            title=row["title"],
            status=ReportStatus(row["status"]),
            html=row["html"],
            tool_call_ids=tuple(json.loads(row["tool_call_ids_json"])),
            claim_ids=tuple(json.loads(row["claim_ids_json"])),
            provenance=json.loads(row["provenance_json"]),
            created_at=row["created_at"],
            finalized_at=row["finalized_at"],
        )

    def list(self) -> list[ReportRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM research_reports WHERE project_id=? ORDER BY created_at,id",
                (self.project_id,),
            ).fetchall()
        return [self.get(row["id"]) for row in rows]


def write_report_file(report: ReportRecord, destination: str | Path) -> Path:
    target = Path(destination)
    target.write_text(report.html, encoding="utf-8")
    return target


def _shown_supported_claims(copilot, dataset_id, revision, session_id):
    """Reuse only previously screened units that actually reached the user."""
    from st_agent.message_trace import load_message_trace

    if session_id is None:
        return []
    claims = []
    for message in copilot.list_messages(session_id):
        if message.role != "assistant":
            continue
        state = load_message_trace(copilot, message)
        if state.get("dataset_id") != dataset_id:
            continue
        audit = (state.get("answer_audit") or state.get("knowledge", {}).get("answer_audit")
                 or state.get("answer_delivery", {}).get("audit") or {})
        identity = audit.get("identity", {})
        if not audit.get("accepted") or identity.get("dataset_id") != dataset_id or \
                identity.get("revision_id") != revision.id or identity.get("sha256") != revision.sha256:
            continue
        rendered_ids = set(audit.get("rendered_unit_ids", []))
        supported = {f"reasoning-{entry['claim']['original_index']}/{verdict['unit_id']}"
            for entry in audit.get("reasoning_audit", {}).get("reasoning", [])
            for verdict in entry.get("grounding", {}).get("unit_verdicts", [])
            if verdict.get("allowed") and "original_index" in entry.get("claim", {})}
        for unit in audit.get("rendered_units", []):
            if unit.get("id") not in rendered_ids.intersection(supported) or \
                    not unit.get("text") or unit["text"] not in message.content:
                continue
            claims.append({"id": message.id + "/" + unit["id"], "status": "supported",
                "text": unit["text"], "role": unit.get("role"), "message_id": message.id,
                "unit_id": unit["id"], "revision_id": revision.id,
                "evidence_refs": unit.get("evidence_refs", []),
                # Keep the actual unit attribution that was displayed and
                # verified; never synthesize parent citations at report time.
                "citation_ids": unit.get("citation_ids", []),
                "statistical_scope": unit.get("statistical_scope"),
                "verification": "persisted_fallible_model_screen_not_independent_expert"})
        for index, entry in enumerate(audit.get("reasoning_audit", {}).get("references", [])):
            claim, assessment = entry.get("claim", {}), entry.get("assessment", {})
            if assessment.get("judgment") != "supported" or not assessment.get("relevant") or \
                    not claim.get("text") or claim["text"] not in message.content:
                continue
            claims.append({"id": message.id + "/reference-" + str(index), "status": "supported",
                "text": claim["text"], "role": "reference", "message_id": message.id,
                "citation_id": claim.get("citation_id"), "revision_id": revision.id,
                "verification": "persisted_reference_entailment_screen"})
    return claims


def create_analysis_report(copilot, dataset_id: str, *, session_id: str | None = None) -> ReportRecord:
    """Export committed lineage and successful saved results; never run analysis."""
    from st_agent.utils import sha256_file

    dataset = copilot.get_dataset(dataset_id)
    if session_id is not None:
        copilot.list_messages(session_id)  # Project ownership before any report write.
    root, lineage, seen = copilot.layout.root.resolve(), [], set()
    revision_id = dataset.active_revision_id
    while revision_id is not None:
        if revision_id in seen:
            raise ValueError("report.revision_cycle")
        seen.add(revision_id)
        revision = copilot.revisions.get(revision_id)
        if revision.dataset_id != dataset_id or revision.status.value != "committed" or \
                not revision.relative_path:
            raise ValueError("report.uncommitted_or_foreign_revision")
        path = (root / revision.relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != revision.sha256:
            raise ValueError("report.revision_fingerprint_mismatch")
        lineage.append(revision)
        revision_id = revision.parent_revision_id
    if not lineage:
        raise ValueError("report.no_committed_revision")
    current, by_id = lineage[0], {item.id: item for item in lineage}
    tools = []
    for result in copilot.tool_executions.list_for_dataset(copilot.project.id, dataset_id):
        inp, out = result.input_revision_id, result.output_revision_id
        if inp not in by_id or result.provenance.dataset_hash != dataset.sha256:
            continue
        if out is not None and out != inp and (out not in by_id or
                by_id[out].call_id != result.call_id or by_id[out].parent_revision_id != inp):
            continue
        tools.append(result.model_dump(mode="json"))
    if not tools:
        raise ValueError("Report requires successful persisted tool results on the current lineage")
    sections = []
    for result in reversed(tools):
        facts = report_tool_facts(result)
        if facts:
            bound = result.get("output_revision_id") or result["input_revision_id"]
            scope = "当前版本" if bound == current.id else "上游版本"
            sections.append({"kind": "facts", "title": scope + " · " + result["tool_name"],
                             "facts": facts})
    claims = _shown_supported_claims(copilot, dataset_id, current, session_id)
    for claim in claims:
        sections.append({"kind": "claim", "title": "已核验并呈现的回答 · " + str(claim["role"]),
                         "claim_ids": [claim["id"]], "text": claim["text"]})
    provenance = {"dataset_id": dataset_id, "dataset_sha256": dataset.sha256,
        "revision_id": current.id, "revision_sha256": current.sha256, "session_id": session_id,
        "lineage": [{"revision_id": item.id, "parent_revision_id": item.parent_revision_id,
            "sha256": item.sha256, "call_id": item.call_id} for item in lineage],
        "tool_sources": [{"call_id": result["call_id"], "tool_name": result["tool_name"],
            "input_revision_id": result["input_revision_id"],
            "output_revision_id": result["output_revision_id"],
            "provenance": result["provenance"]} for result in tools],
        "claim_scope": "selected_session_current_revision" if session_id else "no_session_selected",
        "new_analysis_calls": 0, "new_model_calls": 0}
    if copilot.get_dataset(dataset_id).active_revision_id != current.id:
        raise ValueError("report.active_revision_changed")
    report = ReportStore(copilot.layout.database_path, copilot.project.id).create(
        title=dataset.display_name + " · 分析报告", sections=sections, tool_results=tools,
        claims=claims, provenance=provenance)
    try:
        directory = root / "artifacts" / "reports"
        directory.mkdir(parents=True, exist_ok=True)
        saved = write_report_file(report, directory / (report.id + ".html"))
        copilot.register_artifact(saved, kind="report.analysis.html", metadata={
            "report_id": report.id, "dataset_id": dataset_id, "revision_id": current.id,
            "revision_sha256": current.sha256, "contract": "bound-sections/2"})
    except Exception:
        # This new draft is not a delivered report until its export is registered.
        # Keep any written file for diagnosis; never remove analysis/source files.
        with connect_database(copilot.layout.database_path) as connection:
            connection.execute("DELETE FROM research_reports WHERE id=? AND project_id=? AND status='draft'",
                               (report.id, copilot.project.id))
            connection.commit()
        raise
    return report
