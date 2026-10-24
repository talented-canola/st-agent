"""One local context assembler. User goals, current facts and ancestry stay distinct.

No analysis, retrieval, image rendering or extra model call happens here. Persisted
tool observations are not promoted to current-version facts merely for convenience.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from st_agent.db import connect_database
from st_agent.ids import IdKind, new_id
from st_agent.utils import utc_now_iso


class ContextError(ValueError):
    """A context boundary failed; do not convert it into a scientific conclusion."""


class ResearchGoalUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["set", "refine", "replace", "clear"]
    quote: str = Field(min_length=1, max_length=1000)


class ResearchContext(BaseModel):
    """Shared assembly contract; facts keep the existing scientific claim schema."""
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["research-context/1"] = Field(alias="schema")
    identity: dict[str, Any]
    research_state: dict[str, Any]
    facts: list[dict[str, Any]]
    sample_context: dict[str, Any]
    lineage: list[dict[str, Any]]
    tool_observations: list[dict[str, Any]]
    figure_refs: list[dict[str, Any]]
    gaps: list[dict[str, Any]]
    evidence_boundary: str
    fingerprint: str


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def research_state(copilot, dataset_id, session_id=None):
    """Durable user-stated goal, never an inferred scientific fact/authorization."""
    copilot.get_dataset(dataset_id)
    if session_id is None:
        return {"goal_quotes": [], "first_user_anchor": None, "goal_status": "unknown"}
    messages = copilot.list_messages(session_id)  # project ownership checked by API
    with connect_database(copilot.layout.database_path,
                          busy_timeout_ms=copilot.settings.sqlite_busy_timeout_ms) as connection:
        rows = connection.execute(
            "SELECT id,payload_json FROM events WHERE project_id=? AND session_id=? "
            "AND event_type=? ORDER BY rowid",
            (copilot.project.id, session_id, "research.goal.updated"),
        ).fetchall()
    quotes, status, omitted = [], "unknown", 0
    for row in rows:
        payload = json.loads(row["payload_json"])
        if payload.get("dataset_id") != dataset_id:
            continue
        action = payload["action"]
        item = {"quote": payload["quote"], "source_event_id": row["id"]}
        if action == "clear":
            quotes, status, omitted = [], "cleared", 0
        elif action in {"replace", "set"}:
            if action == "set" and quotes:
                continue
            quotes, status, omitted = [item], "user_stated", 0
        elif action == "refine":
            quotes.append(item)
            status = "user_stated"
        if len(quotes) > 4:
            omitted += len(quotes) - 4
            quotes = [quotes[0], *quotes[-3:]]
    anchor = next((m for m in messages if m.role == "user"
                   and m.metadata.get("dataset_id") == dataset_id), None)
    return {"goal_quotes": quotes, "goal_status": status,
            "omitted_goal_refinements": omitted,
            "first_user_anchor": ({"message_id": anchor.id, "text": anchor.content[:1600],
                "truncated": len(anchor.content) > 1600, "classification": "unclassified_user_text"}
                if anchor else None),
            "scope": {"project_id": copilot.project.id, "dataset_id": dataset_id,
                      "session_id": session_id},
            "authority": "user_context_only_not_scientific_evidence_or_execution_permission"}


def record_goal(copilot, dataset_id, session_id, run_id, request, update):
    if update is None:
        return
    copilot.list_messages(session_id)
    copilot.get_dataset(dataset_id)
    update = ResearchGoalUpdate.model_validate(update)
    if update.quote not in request:
        raise ContextError("context.goal_quote_not_in_current_request")
    payload = {"dataset_id": dataset_id, "origin_run_id": run_id, **update.model_dump(),
               "request_sha256": hashlib.sha256(request.encode()).hexdigest()}
    from st_agent.trace import redact
    payload = redact(payload)
    # Serialize replay detection and insertion; a resumed turn cannot append twice.
    # Intent precedes creation of an FK-owned execution run. Keep correlation in
    # payload instead of inventing a run or attaching an invalid events.run_id.
    with connect_database(copilot.layout.database_path,
                          busy_timeout_ms=copilot.settings.sqlite_busy_timeout_ms) as connection:
        connection.execute("BEGIN IMMEDIATE")
        rows = connection.execute(
            "SELECT payload_json FROM events WHERE project_id=? AND session_id=? "
            "AND json_extract(payload_json,'$.origin_run_id')=? AND event_type=?",
            (copilot.project.id, session_id, run_id, "research.goal.updated"),
        ).fetchall()
        existing = [json.loads(row[0]) for row in rows]
        if any(p.get("dataset_id") == dataset_id for p in existing):
            if payload not in existing:
                raise ContextError("context.goal_replay_conflict")
            return
        connection.execute(
            "INSERT INTO events(id,project_id,event_type,session_id,run_id,payload_json,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (new_id(IdKind.EVENT), copilot.project.id, "research.goal.updated", session_id,
             None, json.dumps(payload, ensure_ascii=False), utc_now_iso()),
        )
        connection.commit()
    from st_agent.trace import emit
    emit(copilot, "context.goal.persisted", dataset_id=dataset_id, session_id=session_id,
         run_id=run_id, action=update.action)


def _metric_path_value(value, path):
    """Dereference a saved-record path without changing its string segments."""
    for part in path:
        if isinstance(value, (list, tuple)):
            # _saved_records records list positions as decimal strings.
            value = value[int(part)]
        elif isinstance(value, dict):
            # Numeric-looking dict keys are real string keys, not list indices.
            value = value[part]
        else:
            raise TypeError(f"metric path traverses a scalar before {part!r}")
    return value


def _selection_for_result(selection, tool_name):
    """Apply the source restriction used by read_saved_evidence before projection."""
    if selection is None:
        return None
    kind = (selection.get("kind") if isinstance(selection, dict)
            else getattr(selection, "kind", None))
    if kind is None:
        return None  # Legacy MarkerSelection has its dedicated reader.
    if kind == "spatial_results" and not (tool_name or "").startswith(
            ("spatial.", "samples.")):
        return None
    return selection


def _metric_leaves(metrics, limit=64, *, selection=None):
    """Bounded lossless scalar projection; omissions explicit, full result stays in DB."""
    leaves, omitted = [], 0
    priority = []
    if selection is not None:
        from st_agent.evidence_context import SavedEvidenceSelection
        selection = SavedEvidenceSelection.model_validate(selection)
        from st_agent.evidence_context import _saved_records
        rows = _saved_records(metrics, {}, selection)
        priority = [row["path"] for row in rows
            if (not selection.groupby or row["groupby"] == selection.groupby)
            and (not selection.groups or row["group"] in selection.groups)
            and (not selection.clusters or row.get("cluster") in selection.clusters)
            and (not selection.genes or row.get("gene") in selection.genes)
            and (not selection.fields or row["field"] in selection.fields)]
    prioritized = set(map(tuple, priority))
    def visit(value, path):
        nonlocal omitted
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, [*path, str(key)])
        elif isinstance(value, (list, tuple)) and path == ["unapplied_criteria"]:
            # This small categorical list is a safety boundary, not a count vector.
            for index, item in enumerate(value[:8]):
                if isinstance(item, str):
                    visit(item, [*path, str(index)])
            omitted += max(0, len(value) - 8)
        elif isinstance(value, (list, tuple)):
            # A vector is not a statistic: don't flood prompts with coordinates/counts.
            omitted += len(value)
        elif value is None or type(value) in (str, int, float, bool):
            if isinstance(value, float) and not math.isfinite(value):
                omitted += 1
            elif tuple(path) in prioritized:
                return  # Selected leaves were emitted before the general projection.
            elif len(leaves) < limit:
                leaves.append({"path": path, "value": value})
            else:
                omitted += 1
    for path in priority:
        value = _metric_path_value(metrics, path)
        if len(leaves) < limit:
            leaves.append({"path": path, "value": value})
        else:
            omitted += 1
    # Put global QC/filter statistics ahead of potentially large per-group records.
    delayed = {"qc_by", "quality", "analysis_context"}
    for key in sorted(metrics, key=lambda k: (k in delayed, k)):
        visit(metrics[key], [key])
    return leaves, omitted


def assemble_context(copilot, dataset_id, *, session_id=None, sample=None, evidence_selection=None):
    from st_agent.scientific_evidence import build_evidence_bundle
    sample = sample if sample is not None else build_evidence_bundle(copilot, dataset_id)
    identity = sample["identity"]
    if identity["dataset_id"] != dataset_id:
        raise ValueError("context.dataset_mismatch")
    chain, seen, revision_id = [], set(), identity["revision_id"]
    while revision_id:
        if revision_id in seen:
            raise ValueError("context.revision_cycle")
        seen.add(revision_id)
        revision = copilot.revisions.get(revision_id)
        if revision.dataset_id != dataset_id or revision.status.value != "committed":
            raise ValueError("context.uncommitted_or_foreign_revision")
        chain.append(revision)
        revision_id = revision.parent_revision_id
    if chain[0].sha256 != identity["sha256"]:
        raise ValueError("context.revision_fingerprint_mismatch")
    lineage = {revision.id: revision for revision in chain}
    dataset = copilot.get_dataset(dataset_id)
    observations, gaps, latest = [], [], set()
    current_facts = list(sample["facts"])
    is_figure = sample.get("figure") is not None
    for result in copilot.tool_executions.list_for_dataset(copilot.project.id, dataset_id):
        inp, out = result.input_revision_id, result.output_revision_id
        # Provenance.dataset_hash identifies the registered dataset, NOT input revision.
        if inp not in lineage:
            continue
        if result.provenance.dataset_hash != dataset.sha256:
            gaps.append({"code": "context.tool_provenance_mismatch", "call_id": result.call_id})
            continue
        transforming = out is not None and out != inp
        if transforming and (out not in lineage or lineage[out].call_id != result.call_id
                             or lineage[out].parent_revision_id != inp):
            continue
        bound = out if transforming else inp
        key = (result.tool_name, bound)
        if key in latest:
            continue
        latest.add(key)
        selected = _selection_for_result(evidence_selection, result.tool_name)
        metrics, omitted = _metric_leaves(result.metrics, selection=selected)
        observations.append({"tool_name": result.tool_name, "call_id": result.call_id,
            "input_revision_id": inp, "output_revision_id": out,
            "scope": "current" if bound == identity["revision_id"] else "upstream",
            "status": result.status.value, "metrics": metrics,
            "omitted_metric_items": omitted, "warnings": list(result.warnings),
            "parameters": result.provenance.parameters,
            "tool_version": result.provenance.tool_version,
            "finished_at": result.provenance.finished_at.isoformat()})
        # Only readonly QC of THIS revision can extend its selectable scalar facts.
        if not is_figure and result.tool_name == "qc.compute_metrics" and not transforming \
                and inp == identity["revision_id"]:
            for metric in metrics:
                if metric["value"] is None:
                    continue
                path = metric["path"]
                if path[0] in {"analysis_context", "qc_by", "warnings"}:
                    continue
                fact_key = "/".join(path)
                current_facts.append({"id": "tool_" + _digest([result.call_id, path])[:24],
                    "key": "qc/" + fact_key, "value": metric["value"], "label": "QC " + fact_key,
                    "revision_id": inp, "source_call_id": result.call_id, "show": True})
    if len(observations) > 16:
        gaps.append({"code": "context.tool_observations_truncated", "omitted": len(observations)-16})
        # Preserve requested tool families before limiting the historical index.
        kind = getattr(evidence_selection, "kind", None)
        if isinstance(evidence_selection, dict):
            kind = evidence_selection.get("kind")
        def relevant(observation):
            prefix = ["qc_by"] if kind == "qc_groups" else ["quality", "cluster_chip_association"]
            return kind in {"qc_groups", "section_composition"} and any(
                m["path"][:len(prefix)] == prefix for m in observation["metrics"])
        observations.sort(key=lambda o: (not relevant(o), o["scope"] != "current"))
        observations = observations[:16]
    if not any(o["tool_name"] == "qc.compute_metrics" and o["scope"] == "current"
               for o in observations):
        gaps.append({"code": "context.current_revision_qc_not_available",
                     "meaning": "Upstream QC does not describe the active population."})
    figures = []
    from st_agent.utils import sha256_file
    for artifact in copilot.list_artifacts():
        if artifact.kind.startswith("figure.full.") and \
                artifact.metadata.get("dataset_id") == dataset_id and \
                artifact.metadata.get("revision_id") == identity["revision_id"]:
            path = (copilot.layout.root / artifact.relative_path).resolve()
            if not path.is_relative_to(copilot.layout.root.resolve()) or not path.is_file() \
                    or sha256_file(path) != artifact.sha256:
                gaps.append({"code": "context.figure_artifact_unavailable",
                             "artifact_id": artifact.id})
                continue
            figures.append({"artifact_id": artifact.id, "kind": artifact.kind,
                "relative_path": artifact.relative_path, "sha256": artifact.sha256,
                "parameters": artifact.metadata.get("parameters", {}),
                "pixels_supplied": False})
    chosen_ids = (evidence_selection.get("artifact_ids", []) if isinstance(evidence_selection, dict)
                  else getattr(evidence_selection, "artifact_ids", []))
    if chosen_ids:
        figures.sort(key=lambda f: f["artifact_id"] not in chosen_ids)
    if len(figures) > 8:
        gaps.append({"code": "context.figure_refs_truncated", "omitted": len(figures)-8})
    context = {"schema": "research-context/1", "identity": identity,
        "research_state": research_state(copilot, dataset_id, session_id),
        "facts": current_facts, "sample_context": sample["context"],
        "lineage": [{"revision_id": r.id, "parent_revision_id": r.parent_revision_id,
                     "sha256": r.sha256, "call_id": r.call_id} for r in chain],
        "tool_observations": observations,
        "figure_refs": figures[:8] if chosen_ids else figures[-8:], "gaps": gaps,
        "evidence_boundary": "upstream observations describe their input/output revisions, not current cells"}
    context["fingerprint"] = _digest(context)
    context = ResearchContext.model_validate(context).model_dump(by_alias=True)
    from st_agent.trace import emit
    emit(copilot, "context.assembled", context=context)
    return context


def context_projection(context):
    """Supplement to caller's existing identity/facts; do not duplicate large fact tables."""
    return {key: context[key] for key in ("schema", "fingerprint", "research_state", "lineage",
            "tool_observations", "figure_refs", "gaps", "evidence_boundary")}
