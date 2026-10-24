"""Shared answer boundary for tool receipts, saved-result discussion and figures.

This contract describes delivered content, never tool authorization or scientific truth.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ResponseRequirements(BaseModel):
    """Requested delivered content; never tool authorization or source permission.

    Three independent axes: execution is described by operation/mode/new_analysis,
    ``mode`` here is the requested answer content, and ``retrieval_policy`` is only
    the user's explicit constraint on reading external or library method sources.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    mode: Literal["receipt", "interpretation", "method", "mixed"] = Field(
        default="receipt",
        description=(
            "Requested answer content. receipt: the turn only runs/checks data and asks "
            "for no explanatory content. interpretation: interpret saved sample results. "
            "method: explain a method or discuss a study/analysis design, including a "
            "future plan. mixed: both sample results and method/design content. This is "
            "content only: it never authorizes execution, recomputation or data changes, "
            "and method is not a source-access permission."
        ),
    )
    questions: list[str] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "Concrete questions copied from the current request for non-receipt content."
        ),
    )
    # Independent of answer style: a user may require or forbid external
    # literature/method-source retrieval while still asking for mixed prose.
    retrieval_policy: Literal["auto", "forbidden", "required"] = Field(
        default="auto",
        description=(
            "The user's explicit constraint on reading external or library method "
            "sources (literature, the local knowledge corpus, or similar). auto when the "
            "current request does not constrain source reading, including when it says "
            "nothing about sources. forbidden only when the user explicitly says not to "
            "add, read or search such sources, for example to answer only from existing "
            "or saved results. required only when the user explicitly demands source "
            "retrieval. Never infer this from mode, from the absence of a literature "
            "request, or from a prohibition on executing, recomputing or running new "
            "scientific methods. Reading and explaining saved results is not external "
            "retrieval."
        ),
    )


def retrieval_required_for(requirements: ResponseRequirements, *, auto_default: bool) -> bool:
    """Resolve the explicit request constraint; only ``auto`` uses the legacy route.

    Saved-result reading and scientific explanation are not external retrieval,
    so ``forbidden`` suppresses corpus search without suppressing local answers.
    """
    if requirements.retrieval_policy == "forbidden":
        return False
    if requirements.retrieval_policy == "required":
        return True
    return auto_default


class AnswerContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: str = "answer-context/2"
    request: str
    dataset_id: str
    revision_id: str
    source: Literal["tool", "workflow", "knowledge", "figure"]
    requirements: ResponseRequirements = Field(default_factory=ResponseRequirements)
    facts: list[dict[str, Any]] = Field(default_factory=list)
    evidence_identity: dict[str, Any] = Field(default_factory=dict)
    evidence_selection: dict[str, Any] | None = None

    def fingerprint(self) -> str:
        value = self.model_dump(mode="json")
        # Preserve pre-selection fingerprints when no selection exists, while
        # making every explicit selection part of delivery reuse identity.
        if value.get("evidence_selection") is None:
            value.pop("evidence_selection", None)
        return hashlib.sha256(json.dumps(value, sort_keys=True,
            ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def normalize_evidence_selection(selection) -> dict[str, Any] | None:
    """Return one JSON selection identity; never silently drop a supplied value."""
    if selection is None:
        return None
    if isinstance(selection, dict):
        return dict(selection)
    dump = getattr(selection, "model_dump", None)
    if not callable(dump):
        raise ValueError("answer_delivery.evidence_selection_invalid")
    return dump(mode="json")


class AnswerResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: str = "answer-result/2"
    answer: str
    mode: Literal["receipt", "interpretation", "method", "mixed"]
    source: Literal["tool", "workflow", "knowledge", "figure"]
    dataset_id: str
    revision_id: str
    context_fingerprint: str
    rendered_fact_ids: list[str] = Field(default_factory=list)
    rendered_unit_ids: list[str] = Field(default_factory=list)
    fulfilled_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    error_code: str | None = None
    audit: dict[str, Any] = Field(default_factory=dict)


def from_answer(context: AnswerContext, answer: str, audit: dict[str, Any], *,
                fulfilled=None, missing=None, error_code=None) -> AnswerResult:
    return AnswerResult(answer=answer, mode=context.requirements.mode, source=context.source,
        dataset_id=context.dataset_id, revision_id=context.revision_id,
        context_fingerprint=context.fingerprint(), audit=audit,
        rendered_fact_ids=list(dict.fromkeys(audit.get("rendered_fact_ids", []))),
        rendered_unit_ids=list(dict.fromkeys(audit.get("rendered_unit_ids", []))),
        fulfilled_requirements=list(fulfilled or []), missing_requirements=list(missing or []),
        error_code=error_code)


def local_receipt(context: AnswerContext) -> AnswerResult:
    from st_agent.answer_contract import validate_and_render
    from st_agent.receipt_rendering import render_structured_receipt

    structured = render_structured_receipt(context.facts)
    if structured is not None:
        answer, audit = structured
        return from_answer(context, answer, audit, fulfilled=["receipt"])

    claims = [{"fact_id": f["id"], "key": f["key"], "value": f["value"],
               "revision_id": f.get("revision_id")} for f in context.facts[:64]]
    answer, audit = validate_and_render(json.dumps({"claims": claims}, ensure_ascii=False),
                                       context.facts)
    return from_answer(context, answer, audit, fulfilled=["receipt"])


def _partial_support_rejection_applies(context: AnswerContext, knowledge: dict,
                                       audit: dict) -> bool:
    """Identify the explicit local rejection that leaves verified units deliverable."""
    if knowledge.get("status") != "degraded" or audit.get("accepted") is not True:
        return False
    if audit.get("error_code") != "answer.partial_support_rejected":
        return False
    if not context.requirements.questions or knowledge.get("error_code"):
        return False
    reasoning_audit = audit.get("reasoning_audit") or {}
    if reasoning_audit.get("batch_errors") or reasoning_audit.get("item_errors"):
        return False
    identity = audit.get("identity") or {}
    return identity.get("dataset_id") == context.dataset_id and \
        identity.get("revision_id") == context.revision_id


def _unit_verdict(unit: dict, verdicts: dict) -> dict | None:
    """Resolve the verdict for this unit using the reasoning-{index}/{unit_id} identity."""
    unit_id = unit.get("unit_id")
    full_id = unit.get("id")
    if not isinstance(unit_id, str) or not isinstance(full_id, str):
        return None
    suffix = "/" + unit_id
    if not full_id.endswith(suffix):
        return None
    prefix = full_id[:-len(suffix)]
    if not prefix.startswith("reasoning-"):
        return None
    try:
        index = int(prefix[len("reasoning-"):])
    except ValueError:
        return None
    return verdicts.get((index, unit_id))


def _verified_partial_coverage(context: AnswerContext, knowledge: dict,
                               audit: dict) -> tuple[set[str], bool]:
    """Only count rendered units independently allowed and actually shown in the answer."""
    required = {f"Q{i+1}" for i in range(len(context.requirements.questions))}
    answer = knowledge.get("answer")
    rendered_ids = set(audit.get("rendered_unit_ids") or [])
    verdicts = {}
    for item in (audit.get("reasoning_audit") or {}).get("reasoning") or []:
        index = (item.get("claim") or {}).get("original_index")
        for verdict in (item.get("grounding") or {}).get("unit_verdicts") or []:
            verdicts[(index, verdict.get("unit_id"))] = verdict

    covered: set[str] = set()
    useful = False
    for unit in audit.get("rendered_units") or []:
        full_id = unit.get("id")
        if full_id not in rendered_ids:
            continue
        text = unit.get("text")
        if not isinstance(text, str) or not text or not isinstance(answer, str) or \
                text not in answer:
            continue
        verdict = _unit_verdict(unit, verdicts)
        if verdict is None or verdict.get("allowed") is not True:
            continue
        matched = set(unit.get("requirement_ids") or []) & \
            set(verdict.get("fulfilled_requirement_ids") or []) & required
        if not matched:
            continue
        covered |= matched
        if unit.get("role") in {"interpretation", "hypothesis", "proposal", "method"}:
            useful = True
    return covered, useful


def interpretation_completion(context: AnswerContext, knowledge: dict) -> tuple[list[str], list[str]]:
    audit = knowledge.get("answer_audit") or {}
    units = audit.get("rendered_units", [])
    required = {f"Q{i+1}" for i in range(len(context.requirements.questions))}
    covered = {rid for u in units for rid in u.get("requirement_ids", [])}
    useful = any(u.get("role") in {"interpretation", "hypothesis", "proposal", "method"} for u in units)
    if context.requirements.mode == "method" and not required:
        useful = useful or any(item.get("assessment", {}).get("judgment") == "supported" and
            item.get("assessment", {}).get("relevant") and item.get("claim", {}).get("text") and
            item["claim"]["text"] in knowledge.get("answer", "")
            for item in audit.get("reasoning_audit", {}).get("references", []))
    complete = knowledge.get("status") == "grounded" and not audit.get("error_code") and \
        not knowledge.get("error_code") and useful and required <= covered
    if complete:
        return [context.requirements.mode], []
    if _partial_support_rejection_applies(context, knowledge, audit):
        verified, verified_useful = _verified_partial_coverage(context, knowledge, audit)
        if verified_useful and required <= verified:
            return [context.requirements.mode], []
        return [], sorted(required - verified) or [context.requirements.mode]
    return [], sorted(required - covered) or [context.requirements.mode]


def deliver_after_execution(copilot, client, context: AnswerContext, *, session_id, run_id,
                            history=None, evidence_selection=None) -> tuple[AnswerResult, dict | None]:
    """Preserve committed computation even when optional interpretation fails."""
    receipt = local_receipt(context)
    if context.requirements.mode == "receipt":
        return receipt, None
    from st_agent.knowledge_service import KnowledgeService

    try:
        knowledge = KnowledgeService(copilot, client).respond(context.request,
            dataset_id=context.dataset_id, run_id=run_id, session_id=session_id, history=history,
            evidence_selection=evidence_selection,
            response_requirements={"mode": context.requirements.mode,
                "retrieval_policy": context.requirements.retrieval_policy,
                "questions": [{"id": f"Q{i+1}", "text": text}
                              for i, text in enumerate(context.requirements.questions)]},
            retrieval_required=retrieval_required_for(context.requirements,
                auto_default=context.requirements.mode in {"method", "mixed"}))
        audit = knowledge.get("answer_audit") or {}
        identity = knowledge.get("identity") or audit.get("identity") or {}
        if identity.get("dataset_id") != context.dataset_id or \
                identity.get("revision_id") != context.revision_id:
            raise ValueError("answer_delivery.evidence_identity_mismatch")
        error = knowledge.get("error_code") or audit.get("error_code")
        fulfilled, missing = interpretation_completion(context, knowledge)
        text = receipt.answer + "\n\n" + knowledge.get("answer", "解释未完成。")
        return from_answer(context, text, {**audit,
            "receipt_audit": receipt.audit,
            "rendered_fact_ids": list(dict.fromkeys([*receipt.rendered_fact_ids,
                                                     *audit.get("rendered_fact_ids", [])]))},
            fulfilled=["receipt", *fulfilled], missing=missing,
            error_code=error), knowledge
    except Exception as error:
        code = "answer_delivery." + type(error).__name__
        return from_answer(context, receipt.answer + "\n\n结果已保存；本次请求的解释未完成。",
            receipt.audit, fulfilled=["receipt"], missing=[context.requirements.mode],
            error_code=code), None
