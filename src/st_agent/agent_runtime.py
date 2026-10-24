"""Bounded LangGraph Tool Calling loop for the local research copilot."""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import StrEnum
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from st_agent.deepseek import DeepSeekError, DeepSeekProtocol
from st_agent.rag import EvidenceBundle, format_evidence_context
from st_agent.safety import GateDisposition, SafetyInterruption
from st_agent.tool_exposure import INTERNAL_TO_PROVIDER, provider_tools, validate_provider_call


class AgentStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED_PROVIDER = "paused_provider"
    NEEDS_CLARIFICATION = "needs_clarification"
    WAITING_APPROVAL = "waiting_approval"
    DENIED = "denied"
    CANCELLED = "cancelled"


class AgentEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sequence: int = Field(ge=1)
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class FactRecord(BaseModel):
    """One scalar scientific fact emitted by a completed local tool."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str
    source_tool: str
    key: str
    value: str | int | float | bool | None
    revision_id: str | None = None
    evidence_scope: str = "dataset_revision"


class AgentState(TypedDict):
    request: str
    dataset_id: str
    messages: list[dict[str, Any]]
    facts: list[dict[str, Any]]
    fact_records: list[dict[str, Any]]
    events: list[dict[str, Any]]
    model_calls: int
    tool_calls: int
    replans: int
    max_model_calls: int
    max_tool_calls: int
    max_replans: int
    status: str
    answer: str
    error: str | None


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: AgentStatus
    answer: str = ""
    model_calls: int
    tool_calls: int
    replans: int
    facts: list[dict[str, Any]]
    fact_records: list[FactRecord]
    events: list[AgentEvent]
    state: dict[str, Any] = Field(exclude=True)


class AgentEngine:
    def __init__(
        self,
        client: DeepSeekProtocol,
        *,
        executor: Callable[[str, dict[str, Any]], dict[str, Any]],
        max_model_calls: int = 4,
        max_tool_calls: int = 8,
        max_replans: int = 2,
        retriever: Callable[[str], EvidenceBundle] | None = None,
        allowed_tools: tuple[str, ...] | None = None,
    ) -> None:
        self.client = client
        self.executor = executor
        self.limits = (max_model_calls, max_tool_calls, max_replans)
        self.retriever = retriever
        self.allowed_tools = allowed_tools
        graph = StateGraph(AgentState)
        graph.add_node("receive", self._receive)
        graph.add_node("hydrate", self._hydrate)
        graph.add_node("route", self._route_request)
        graph.add_node("agent_step", self._step)
        graph.add_edge(START, "receive")
        graph.add_edge("receive", "hydrate")
        graph.add_edge("hydrate", "route")
        graph.add_edge("route", "agent_step")
        graph.add_conditional_edges("agent_step", self._route, {"again": "agent_step", "end": END})
        self.graph = graph.compile()

    def run(
        self,
        request: str,
        *,
        dataset_id: str,
        history: list[dict[str, str]] | None = None,
    ) -> AgentResult:
        model_limit, tool_limit, replan_limit = self.limits
        initial: AgentState = {
            "request": request[:4000],
            "dataset_id": dataset_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a spatial transcriptomics research copilot. Reply in the "
                        "researcher's language. Use only provided tools. Separate statements "
                        "using [事实], [解释], [推断], or [待确认]. Every [事实] statement after a "
                        "tool call must cite one or more supplied [fact:<id>] entries. If QC "
                        "is requested without explicit filtering "
                        "thresholds, compute QC metrics only and never filter implicitly. "
                        "Run filtering, preprocessing, clustering, and marker analysis only "
                        "when the researcher explicitly asks. Respect the scientific order: "
                        "inspect, QC metrics, optional filtering, preprocessing, clustering, "
                        "then marker analysis."
                    ),
                },
                *self._bounded_history(history or []),
                {"role": "user", "content": request[:4000]},
            ],
            "facts": [],
            "fact_records": [],
            "events": [{"sequence": 1, "type": "request.received", "payload": {}}],
            "model_calls": 0,
            "tool_calls": 0,
            "replans": 0,
            "max_model_calls": model_limit,
            "max_tool_calls": tool_limit,
            "max_replans": replan_limit,
            "status": AgentStatus.RUNNING.value,
            "answer": "",
            "error": None,
        }
        return self._result(self.graph.invoke(initial))

    @staticmethod
    def _bounded_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
        allowed = []
        for message in history[-12:]:
            role = message.get("role")
            content = str(message.get("content", ""))[:4000]
            if role in {"user", "assistant"} and content:
                allowed.append({"role": role, "content": content})
        return allowed

    def resume(self, paused: AgentResult) -> AgentResult:
        state = dict(paused.state)
        state["status"] = AgentStatus.RUNNING.value
        state["error"] = None
        self._event(state, "run.resumed")
        return self._result(self.graph.invoke(state))

    def _receive(self, state: AgentState) -> dict[str, Any]:
        self._event(state, "request.normalized")
        return state

    def _hydrate(self, state: AgentState) -> dict[str, Any]:
        evidence_count = 0
        if self.retriever is not None:
            bundle = self.retriever(state["request"])
            evidence_count = len(bundle.items)
            if bundle.items:
                state["messages"].append(
                    {"role": "user", "content": format_evidence_context(bundle)}
                )
        self._event(
            state,
            "context.hydrated",
            {
                "dataset_id": state["dataset_id"],
                "verified_tools": len(provider_tools(self.allowed_tools)),
                "evidence_items": evidence_count,
            },
        )
        return state

    def _route_request(self, state: AgentState) -> dict[str, Any]:
        self._event(state, "request.routed", {"route": "tool_calling"})
        return state

    def _step(self, state: AgentState) -> dict[str, Any]:
        if state["model_calls"] >= state["max_model_calls"]:
            state["status"] = AgentStatus.FAILED.value
            state["error"] = "model_call_budget_exhausted"
            self._event(state, "run.failed", {"reason": state["error"]})
            return state
        self._event(state, "model.started")
        try:
            reply = self.client.complete(state["messages"], provider_tools(self.allowed_tools))
        except DeepSeekError as error:
            state["status"] = AgentStatus.PAUSED_PROVIDER.value
            state["error"] = str(error)
            self._event(state, "run.paused", {"reason": "provider"})
            return state
        state["model_calls"] += 1
        self._event(state, "model.completed", {"response_id": reply.response_id})
        if reply.tool_calls:
            call = reply.tool_calls[0]
            try:
                internal, arguments = validate_provider_call(
                    call.name, call.arguments, allowed=self.allowed_tools
                )
                if state["tool_calls"] >= state["max_tool_calls"]:
                    raise ValueError("tool_call_budget_exhausted")
                self._event(state, "tool.validated", {"tool": internal})
                result = self.executor(internal, arguments)
                state["tool_calls"] += 1
                fact = {"tool": internal, "call_id": call.id, "status": "success", "result": result}
                state["facts"].append(fact)
                records = self._records_from_result(internal, result, len(state["fact_records"]))
                state["fact_records"].extend(record.model_dump() for record in records)
                state["messages"].append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments,
                                },
                            }
                        ],
                    }
                )
                state["messages"].append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
                )
                if records:
                    from st_agent.answer_contract import ANSWER_INSTRUCTION

                    ledger = [record.model_dump() for record in records]
                    state["messages"].append(
                        {
                            "role": "user",
                            "content": (
                                ANSWER_INSTRUCTION + " Verified fact ledger: "
                                f"{json.dumps(ledger, ensure_ascii=False)}"
                            ),
                        }
                    )
                self._event(state, "tool.completed", {"tool": internal})
            except SafetyInterruption as interruption:
                disposition = interruption.outcome.disposition
                status = {
                    GateDisposition.WAITING_APPROVAL: AgentStatus.WAITING_APPROVAL,
                    GateDisposition.DENIED: AgentStatus.DENIED,
                    GateDisposition.CANCELLED: AgentStatus.CANCELLED,
                }.get(disposition, AgentStatus.FAILED)
                state["status"] = status.value
                state["error"] = disposition.value
                payload = {"reason": disposition.value}
                if interruption.outcome.approval is not None:
                    payload["approval_id"] = interruption.outcome.approval.id
                self._event(state, "tool.blocked", payload)
            except (ValueError, KeyError, TypeError) as error:
                state["replans"] += 1
                self._event(state, "tool.rejected", {"reason": str(error)})
                if state["replans"] > state["max_replans"]:
                    state["status"] = AgentStatus.FAILED.value
                    state["error"] = "replan_budget_exhausted"
                else:
                    state["messages"].append(
                        {"role": "user", "content": f"Tool call rejected: {error}. Choose safely."}
                    )
            return state
        required_tool = self._explicit_tool_request(state["request"])
        if required_tool and not any(fact.get("tool") == required_tool for fact in state["facts"]):
            state["replans"] += 1
            if state["replans"] > state["max_replans"]:
                state["status"] = AgentStatus.FAILED.value
                state["error"] = "explicit_tool_not_executed"
                self._event(state, "run.failed", {"reason": state["error"]})
            else:
                provider_name = required_tool.replace(".", "_")
                state["messages"].append(
                    {
                        "role": "user",
                        "content": (
                            f"The researcher explicitly requested execution. Call {provider_name} "
                            "now with strictly valid arguments; do not only describe it."
                        ),
                    }
                )
                self._event(state, "tool.execution_required", {"tool": required_tool})
            return state
        answer = self._validated_answer(reply.content or "", state["fact_records"])
        state["answer"] = answer
        state["status"] = AgentStatus.COMPLETED.value
        self._event(state, "run.completed")
        return state

    @staticmethod
    def _records_from_result(tool: str, result: dict[str, Any], offset: int) -> list[FactRecord]:
        revision_id = result.get("output_revision_id") or result.get("input_revision_id")
        records: list[FactRecord] = []

        def visit(key: str, value: Any) -> None:
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    visit(f"{key}.{child_key}" if key else str(child_key), child_value)
            elif isinstance(value, (list, tuple)):
                for index, child_value in enumerate(value):
                    visit(f"{key}[{index}]", child_value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                records.append(
                    FactRecord(
                        id=f"fact_{offset + len(records) + 1}",
                        source_tool=tool,
                        key=key,
                        value=value,
                        revision_id=(
                            result.get("input_revision_id")
                            if key.endswith("_before")
                            else str(revision_id)
                            if revision_id is not None
                            else None
                        ),
                    )
                )

        for key, value in result.items():
            if key not in {"input_revision_id", "output_revision_id"}:
                visit(str(key), value)
        return records

    @staticmethod
    def _validated_answer(answer: str, records: list[dict[str, Any]]) -> str:
        from st_agent.answer_contract import validate_and_render

        return validate_and_render(answer, records)[0]

    @staticmethod
    def _grounded_fallback(records: list[dict[str, Any]]) -> str:
        if not records:
            return "[待确认] 当前没有已执行工具产生的数据事实；请先检查数据状态或提出明确分析请求。"
        core_keys = {
            "shape[0]",
            "shape[1]",
            "cells",
            "genes",
            "count_source",
            "input_revision_id",
            "output_revision_id",
            "cells_before",
            "cells_after",
            "genes_before",
            "genes_after",
            "clusters",
            "groups",
            "genes_per_group",
            "hvg",
            "pcs",
        }
        # Metadata arrays can be long; preserve scientific summary and provenance first.
        core = [record for record in records if record["key"] in core_keys]
        other = [record for record in records if record["key"] not in core_keys]
        selected = core + other[: max(0, 12 - len(core))]
        rendered = "；".join(
            f"{record['key']}={record['value']} [fact:{record['id']}]" for record in selected
        )
        return f"[事实] 已完成本地工具执行。已验证结果：{rendered}。"

    def _explicit_tool_request(self, request: str) -> str | None:
        normalized = request.lower()
        allowed = self.allowed_tools or tuple(INTERNAL_TO_PROVIDER)
        for internal_name in allowed:
            provider_name = INTERNAL_TO_PROVIDER[internal_name]
            if provider_name in normalized or internal_name in normalized:
                return internal_name
        return None

    @staticmethod
    def _route(state: AgentState) -> str:
        return "again" if state["status"] == AgentStatus.RUNNING.value else "end"

    @staticmethod
    def _event(state: AgentState, kind: str, payload: dict[str, Any] | None = None) -> None:
        state["events"].append(
            {"sequence": len(state["events"]) + 1, "type": kind, "payload": payload or {}}
        )

    @staticmethod
    def _result(state: AgentState) -> AgentResult:
        return AgentResult(
            status=AgentStatus(state["status"]),
            answer=state["answer"],
            model_calls=state["model_calls"],
            tool_calls=state["tool_calls"],
            replans=state["replans"],
            facts=state["facts"],
            fact_records=[FactRecord.model_validate(item) for item in state["fact_records"]],
            events=[AgentEvent.model_validate(item) for item in state["events"]],
            state=state,
        )
