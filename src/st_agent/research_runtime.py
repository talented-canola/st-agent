"""Shared conversation runtime for workbench and headless research scenarios."""

from __future__ import annotations

import json

from pydantic import ValidationError

from st_agent.agent_runtime import AgentEngine, AgentResult, AgentStatus
from st_agent.answer_contract import ANSWER_INSTRUCTION, validate_and_render
from st_agent.contracts import ToolStatus
from st_agent.dataset_state import DatasetStateInspector
from st_agent.deepseek import DeepSeekBudgetExceeded, DeepSeekError, DeepSeekMalformedResponse
from st_agent.errors import PreconditionFailedError
from st_agent.ids import IdKind, new_id
from st_agent.intent_runtime import (IntentInterpreter, effective_delivery_requirements,
                                     effective_delivery_selection, merge_delivery_requirements)
from st_agent.memory_runtime import apply_directive, context_for, persist_verified_run
from st_agent.planning import DeterministicPlanner, ExecutionPlan
from st_agent.safety import ApprovalStatus, GateDisposition, RecoveryRequired
from st_agent.scientific_execution import ScientificPlanExecutor
from st_agent.research_context import ContextError
from st_agent.tool_registry import build_capability_context


def _public_table_layout(artifact):
    """Small directory hint only; never read or upload table content here."""
    if artifact.kind != "data.public.input" and not artifact.kind.startswith("table.public."):
        return None
    raw = artifact.metadata.get("table_layout")
    if not isinstance(raw, dict):
        return {"mode": "positional_unknown_header", "header": "unknown"}
    layout = {}
    if isinstance(raw.get("header"), bool):
        layout["header"] = raw["header"]
    if isinstance(raw.get("columns"), list) and all(
            isinstance(item, str) and item for item in raw["columns"]):
        layout["columns"] = list(raw["columns"])
    if isinstance(raw.get("types"), dict) and all(
            isinstance(column, str) and isinstance(kind, str)
            for column, kind in raw["types"].items()):
        layout["types"] = dict(raw["types"])
    delimiter = raw.get("delimiter")
    if isinstance(delimiter, str) and delimiter in {",", "\t", "comma", "tab"}:
        layout["delimiter"] = delimiter
    if not layout:
        return {"mode": "positional_unknown_header", "header": "unknown"}
    layout["mode"] = "declared"
    return layout


class ResearchAgentRuntime:
    def __init__(self, copilot, client):
        self.copilot = copilot
        self.client = client
        self.executor = ScientificPlanExecutor(copilot)
        from st_agent.trace import attach
        if getattr(copilot, "trace", None) is None:
            attach(copilot)
        self.client.trace = copilot.trace

    def run_turn(self, request, *, dataset_id, session_id, history=None, _run_id=None):
        self.copilot.list_messages(session_id)  # Validate project ownership before writing.
        dataset = self.copilot.get_dataset(dataset_id)
        history = (history or [])[-12:]
        from st_agent.basic_analysis import BasicAnalysisWorkflow
        basic = BasicAnalysisWorkflow(self.copilot)
        pending = basic.active(dataset_id, session_id)
        if pending is not None:
            self.client.trace = basic._trace(pending)
            history = [*history, {"role": "system", "content": json.dumps({
                "pending_goal": pending.get("workflow_kind", "basic_analysis"), "task_id": pending["task_id"],
                "status": pending["status"], "parameters": pending["parameters"],
                "missing_parameters": pending["missing_parameters"],
                # Preserve the user's prior output demand and selection so the
                # pending workflow goal cannot erase them.
                "request": pending.get("request"),
                "delivery_request": pending.get("delivery_request"),
                "response_requirements": pending.get("response_requirements"),
                "saved_evidence": pending.get("saved_evidence")}, ensure_ascii=False)}]
        state = {
            "run_id": _run_id or new_id(IdKind.RUN),
            "session_id": session_id,
            "dataset_id": dataset_id,
            "request": request,
            "history": history,
        }
        self.executor.store.save_checkpoint(
            state["run_id"], {"conversation": state}, "interpreting"
        )
        try:
            memory_context = context_for(self.copilot, dataset_id, session_id)
            state["memory_context"] = memory_context
            from st_agent.trace import emit
            active_revision = dataset.active_revision_id
            observed_executions = self.copilot.tool_executions.list_for_dataset(
                self.copilot.project.id, dataset_id)
            capability_context = build_capability_context(observations=[
                {"tool_name": execution.tool_name, "call_id": execution.call_id,
                 "input_revision_id": execution.input_revision_id,
                 "output_revision_id": execution.output_revision_id,
                 "scope": ("current" if active_revision in {
                     execution.input_revision_id, execution.output_revision_id}
                           else "upstream"),
                 "status": execution.status.value,
                 "metric_keys": sorted(execution.metrics)[:16]}
                for execution in observed_executions])
            task_context = {"dataset_id": dataset_id,
                "revision_id": active_revision,
                "capability_context": capability_context,
                "artifacts": [{"artifact_id": a.id, "kind": a.kind,
                    "title": a.metadata.get("title", a.kind),
                    "revision_id": a.metadata.get("revision_id"),
                    "table_layout": _public_table_layout(a)}
                    for a in sorted(reversed(self.copilot.list_artifacts()), key=lambda item:
                        not (item.kind in {"data.public.input", "data.source_metadata"} or
                             item.kind.startswith(("table.public.", "figure.public."))))
                    if a.metadata.get("dataset_id") == dataset_id and
                    a.metadata.get("revision_id") == active_revision][:16]}
            state["capability_context"] = capability_context
            emit(self.copilot, "intent.start", request=request,
                 capability_context=capability_context,
                 capability_catalog_source="SCIENTIFIC_TOOLS",
                 automatic_execution=False)
            interpreted = IntentInterpreter(self.client).interpret(
                request, history=history, memory_context=memory_context,
                task_context=task_context)
            intent = interpreted.intent
            state["intent_model_calls"] = interpreted.model_calls
            from st_agent.research_context import record_goal, research_state
            record_goal(self.copilot, dataset_id, session_id, state["run_id"], request,
                        intent.research_goal)
            state["research_state"] = research_state(self.copilot, dataset_id, session_id)
            emit(self.copilot, "intent.parsed", intent=intent.model_dump(mode="json"))
            if intent.operation.value == "research_goal":
                state["intent"] = intent.model_dump(mode="json")
                state["response_id"] = interpreted.response_id
                answer = ("已清除这次会话中当前样本的研究目标；已有分析结果未删除。"
                          if intent.research_goal.action == "clear" else
                          "已记录当前样本的研究目标，后续讨论会继续参考；尚未执行数据分析。")
                completed = self._result(AgentStatus.COMPLETED, state, answer)
                self.executor.store.save_checkpoint(state["run_id"],
                    {"conversation": state, "completed_result": completed.model_dump(mode="json")},
                    "completed")
                return completed
            if intent.operation.value == "basic_analysis":
                result = basic.submit(intent, dataset_id=dataset_id, session_id=session_id, request=request)
                result.model_calls = interpreted.model_calls
                result.state["intent"] = intent.model_dump(mode="json")
                result.state["response_id"] = interpreted.response_id
                result.state["research_state"] = state["research_state"]
                return self._deliver_basic(result, request=request,
                                           requirements=effective_delivery_requirements(intent))
            state["intent"] = intent.model_dump(mode="json")
            state["response_id"] = interpreted.response_id
            if intent.operation.value == "memory":
                if intent.memory is None:
                    return self._result(
                        AgentStatus.NEEDS_CLARIFICATION,
                        state,
                        "请明确要记录、查询或遗忘的过滤设置。",
                    )
                answer, ids = apply_directive(self.copilot, dataset_id, session_id, intent.memory)
                state["memory_action"] = intent.memory.action
                state["memory_ids"] = ids
                completed = self._result(AgentStatus.COMPLETED, state, answer)
                self.executor.store.save_checkpoint(
                    state["run_id"],
                    {"conversation": state, "completed_result": completed.model_dump(mode="json")},
                    "completed",
                )
                return completed
            if intent.use_memory:
                profile = memory_context["approved_filter_profile"]
                if intent.operation.value != "qc_filter" or profile is None:
                    return self._result(
                        AgentStatus.NEEDS_CLARIFICATION,
                        state,
                        "当前样本没有适用的已确认参数，请明确本次设置；未执行分析。",
                    )
                intent = intent.model_copy(update={"parameters": {**profile, **intent.parameters}})
                from st_agent.scientific_tools import TOOLS

                TOOLS["qc.apply_filter"].arguments.model_validate(intent.parameters)
                state["memory_used"] = memory_context["memory_ids"]
            state["intent"] = intent.model_dump(mode="json")
            state["response_id"] = interpreted.response_id
            observed = DatasetStateInspector(self.copilot).inspect(dataset_id)
            if intent.operation.value == "spatial_morans" and intent.mode.value == "execute" and (
                not observed.has_spatial_neighbors or pending and pending.get("workflow_kind") == "spatial_morans"):
                result = basic.submit_spatial_dependency(intent, dataset_id=dataset_id,
                    session_id=session_id, request=request)
                result.model_calls = interpreted.model_calls
                result.state["intent"] = intent.model_dump(mode="json")
                return self._deliver_basic(result, request=request,
                                           requirements=effective_delivery_requirements(intent))
            plan = DeterministicPlanner().build(intent, observed)
            state["plan"] = plan.model_dump(mode="json")
            if plan.blocked_reason:
                messages = {
                    "intent_requires_clarification": (
                        "请求仍有未明确的信息，请补充具体操作或参数；尚未执行分析。"
                    ),
                    "filter_requires_explicit_thresholds": (
                        "请确认最少检测基因数、基因最少表达点位数和最大线粒体比例；尚未过滤数据。"
                    ),
                    "cluster_requires_pca": (
                        "当前版本缺少 PCA，请先完成预处理，再运行聚类；尚未执行聚类。"
                    ),
                    "marker_requires_leiden": (
                        "当前版本没有 Leiden 聚类标签，请先完成聚类，再寻找 marker；尚未执行分析。"
                    ),
                    "selected_revision_has_no_trustworthy_count_matrix": (
                        "当前版本没有可确认的计数矩阵，请先检查数据来源；尚未执行数据变换。"
                    ),
                    "operation_is_explanation_only": "这项请求属于方法讨论，不会执行数据变换。",
                }
                return self._result(
                    AgentStatus.NEEDS_CLARIFICATION,
                    state,
                    "[待确认] " + messages.get(plan.blocked_reason, "当前分析前置条件尚未满足。"),
                )
            if not plan.steps:
                from st_agent.knowledge_service import KnowledgeService
                from st_agent.utils import utc_now_iso

                # Checkpoints alone do not create a row in the FK-owned runs table.
                # This records a read-only conversation, not tool authorization.
                with self.executor.store._connect() as connection:
                    connection.execute(
                        "INSERT OR IGNORE INTO runs(id,session_id,dataset_id,active_revision_id,"
                        "status,goal,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                        (state["run_id"], session_id, dataset_id, plan.input_revision_id,
                         "running", "knowledge_discussion", utc_now_iso(), utc_now_iso()))
                    connection.commit()

                from st_agent.answer_delivery import retrieval_required_for

                requirements = effective_delivery_requirements(intent)
                selection = effective_delivery_selection(intent)
                knowledge = KnowledgeService(self.copilot, self.client).respond(
                    request, dataset_id=dataset_id, run_id=state["run_id"], history=history,
                    session_id=session_id,
                    retrieval_required=retrieval_required_for(requirements,
                        auto_default=requirements.mode != "interpretation"),
                    response_requirements={"mode": requirements.mode,
                        "retrieval_policy": requirements.retrieval_policy,
                        "questions": [{"id": f"Q{i+1}", "text": text} for i, text
                                      in enumerate(requirements.questions)]},
                    **({"evidence_selection": selection} if selection is not None else {})
                )
                state["knowledge"] = knowledge
                from st_agent.answer_delivery import AnswerContext, from_answer, interpretation_completion
                answer_context = AnswerContext(request=request, dataset_id=dataset_id,
                    revision_id=plan.input_revision_id, source="knowledge",
                    requirements=requirements,
                    evidence_identity={"artifact_ref": knowledge.get("artifact_ref"),
                                       "identity": knowledge.get("identity")})
                fulfilled, missing = interpretation_completion(answer_context, knowledge)
                delivered = from_answer(answer_context, knowledge["answer"],
                    knowledge.get("answer_audit") or {},
                    fulfilled=fulfilled, missing=missing,
                    error_code=knowledge.get("error_code"))
                state["answer_delivery"] = delivered.model_dump(mode="json")
                completed = self._result(
                    AgentStatus.FAILED if knowledge["status"] == "failed" else AgentStatus.COMPLETED,
                    state, knowledge["answer"], model_calls=interpreted.model_calls + (knowledge["api_requests"] or 0),
                )
                self.executor.store.save_checkpoint(
                    state["run_id"],
                    {"conversation": state, "completed_result": completed.model_dump(mode="json")},
                    "failed" if knowledge["status"] == "failed" else "completed",
                )
                with self.executor.store._connect() as connection:
                    connection.execute("UPDATE runs SET status=?,updated_at=? WHERE id=?",
                        ("failed" if knowledge["status"] == "failed" else "completed",
                         utc_now_iso(), state["run_id"]))
                    connection.commit()
                return completed
            return self._execute(plan, state)
        except ContextError as error:
            state["failure_boundary"] = "research_context"
            state["error_code"] = str(error)
            self.executor.store.save_checkpoint(state["run_id"], {"conversation": state}, "failed")
            from st_agent.trace import emit
            emit(self.copilot, "context.error", status="failed", error_code=str(error))
            return self._result(AgentStatus.FAILED, state,
                "研究上下文发生冲突，本轮未执行工具。错误码：" + str(error))
        except ValidationError as error:
            state["failure_boundary"] = "memory_parameters"
            state["validation_errors"] = error.errors(include_input=False, include_context=False)
            self.executor.store.save_checkpoint(state["run_id"], {"conversation": state}, "failed")
            return self._result(
                AgentStatus.NEEDS_CLARIFICATION,
                state,
                "记忆中的设置或本次覆盖参数不符合工具契约，请补充有效参数；未执行分析。",
            )
        except DeepSeekError as error:
            if hasattr(error, "model_calls"):
                state["intent_model_calls"] = error.model_calls
            state["error_code"] = type(error).__name__
            invalid_contract = isinstance(error, DeepSeekMalformedResponse)
            budget_exceeded = isinstance(error, DeepSeekBudgetExceeded)
            state["failure_boundary"] = (
                "budget"
                if budget_exceeded
                else "intent_contract"
                if invalid_contract
                else "provider"
            )
            if hasattr(error, "intent_arguments"):
                state["rejected_intent"] = error.intent_arguments
                state["validation_errors"] = error.validation_errors
            self.executor.store.save_checkpoint(
                state["run_id"],
                {"conversation": state},
                "failed" if invalid_contract or budget_exceeded else "paused_provider",
            )
            return self._result(
                AgentStatus.FAILED
                if invalid_contract or budget_exceeded
                else AgentStatus.PAUSED_PROVIDER,
                state,
                (
                    "[待确认] 本轮模型请求预算已耗尽，本次未执行工具。"
                    if budget_exceeded
                    else "[待确认] 模型返回的意图或参数不符合执行契约，本次未执行工具。"
                    if invalid_contract
                    else f"[待确认] 模型调用未完成：{type(error).__name__}"
                ),
            )

    def resume_run(self, run_id):
        checkpoint = self.executor.store.load_checkpoint(run_id)
        if "basic_task" in checkpoint.state:
            from st_agent.basic_analysis import BasicAnalysisWorkflow
            return self._deliver_basic(BasicAnalysisWorkflow(self.copilot).resume(run_id))
        state = checkpoint.state.get("conversation", {})
        if not state or state.get("run_id") != run_id:
            raise ValueError("Missing conversation checkpoint")
        if state.get("approval_id"):
            return self.resume_approval(state["approval_id"])
        if checkpoint.next_node == "completed" and checkpoint.state.get("completed_result"):
            result = AgentResult.model_validate_json(
                json.dumps({**checkpoint.state["completed_result"], "state": state})
            )
            result.model_calls = result.tool_calls = 0
            result.state["replayed"] = True
            return result
        if checkpoint.next_node == "paused_provider" and not state.get("tool_result"):
            return self.run_turn(
                state["request"],
                dataset_id=state["dataset_id"],
                session_id=state["session_id"],
                history=state.get("history", []),
                _run_id=run_id,
            )
        if state.get("plan") and checkpoint.next_node in {
            "gate",
            "completed",
            "automatic",
            "recovery_uncertain",
        }:
            plan = ExecutionPlan.model_validate_json(json.dumps(state["plan"]))
            return self._execute(plan, state, model_calls=0)
        return self._result(
            AgentStatus.NEEDS_CLARIFICATION,
            state,
            "执行状态需要核对，系统未自动重复分析。",
            model_calls=0,
        )

    def resume_approval(self, approval_id):
        approval = self.executor.store.get_approval(approval_id)
        checkpoint = self.executor.store.load_checkpoint(approval.run_id)
        state = checkpoint.state.get("conversation")
        if state and state.get("basic_task_id"):
            from st_agent.basic_analysis import BasicAnalysisWorkflow
            return self._deliver_basic(BasicAnalysisWorkflow(self.copilot).resume(
                state["basic_task_id"], approval_id=approval_id))
        if not state or state["session_id"] != approval.session_id:
            raise ValueError("Approval has no matching conversation checkpoint")
        if approval.status is ApprovalStatus.REJECTED:
            self.executor.store.save_checkpoint(approval.run_id, {"conversation": state}, "denied")
            return self._result(
                AgentStatus.DENIED, state, "[解释] 已拒绝分析，未执行数据变换。", model_calls=0
            )
        if checkpoint.next_node == "completed" and checkpoint.state.get("completed_result"):
            result = AgentResult.model_validate_json(
                json.dumps({**checkpoint.state["completed_result"], "state": state})
            )
            result.model_calls = 0
            result.tool_calls = 0
            result.state["replayed"] = True
            return result
        plan = ExecutionPlan.model_validate_json(json.dumps(state["plan"]))
        return self._execute(plan, state, approval_id=approval_id, model_calls=0)

    def _execute(self, plan, state, *, approval_id=None, model_calls=None):
        if model_calls is None:
            model_calls = state.get("intent_model_calls", 1)
        try:
            response = self.executor.execute(
                plan,
                dataset_id=state["dataset_id"],
                session_id=state["session_id"],
                run_id=state["run_id"],
                approval_id=approval_id,
                conversation_state=state,
            )
        except RecoveryRequired:
            state["failure_boundary"] = "recovery_uncertain"
            self.executor.store.save_checkpoint(
                state["run_id"], {"conversation": state}, "recovery_uncertain"
            )
            return self._result(
                AgentStatus.NEEDS_CLARIFICATION,
                state,
                "存在未完成的执行记录；为避免重复修改数据，已暂停并等待核对执行状态。",
                model_calls=0,
            )
        except PreconditionFailedError as error:
            state["failure_boundary"] = "plan_precondition"
            state["error_code"] = error.code
            self.executor.store.save_checkpoint(state["run_id"], {"conversation": state}, "failed")
            return self._result(
                AgentStatus.NEEDS_CLARIFICATION,
                state,
                "数据版本或执行前置条件已变化，请基于当前版本重新提出分析请求；没有重复执行。",
                model_calls=0,
            )
        state["approval_id"] = response.approval_id
        self.executor.store.save_checkpoint(
            state["run_id"], {"conversation": state}, response.disposition.value
        )
        if response.tool_result is None:
            status = {
                GateDisposition.WAITING_APPROVAL: AgentStatus.WAITING_APPROVAL,
                GateDisposition.DENIED: AgentStatus.DENIED,
                GateDisposition.CANCELLED: AgentStatus.CANCELLED,
            }.get(response.disposition, AgentStatus.FAILED)
            return self._result(
                status,
                state,
                "[待确认] 已生成分析计划，请在审批面板确认参数后执行。",
                model_calls=model_calls,
            )
        result = response.tool_result
        state["tool_result"] = result.model_dump(mode="json")
        if result.status not in {ToolStatus.SUCCESS, ToolStatus.WARNING}:
            return self._result(
                AgentStatus.FAILED,
                state,
                f"[待确认] 工具未完成：{result.error.code}",
                model_calls=model_calls,
            )
        payload = {
            **result.metrics,
            "input_revision_id": result.input_revision_id,
            "output_revision_id": result.output_revision_id,
        }
        records = AgentEngine._records_from_result(result.tool_name, payload, 0)
        ledger = [record.model_dump(mode="json") for record in records]
        from st_agent.answer_delivery import (AnswerContext, ResponseRequirements,
                                              deliver_after_execution)
        intent_data = state.get("intent", {})
        deferred = intent_data.get("deferred_delivery") or {}
        action_requirements = ResponseRequirements.model_validate(
            intent_data.get("response_requirements") or {})
        deferred_requirements = (
            ResponseRequirements.model_validate(
                deferred.get("response_requirements") or {})
            if deferred else None)
        requirements = merge_delivery_requirements(
            action_requirements, deferred_requirements)
        selection = deferred.get("saved_evidence") or intent_data.get("saved_evidence")
        answer_context = AnswerContext(request=state["request"], dataset_id=state["dataset_id"],
                revision_id=result.output_revision_id or result.input_revision_id,
                source="tool", requirements=requirements, facts=ledger)
        state["answer_context"] = answer_context.model_dump(mode="json")
        delivered, knowledge = deliver_after_execution(self.copilot, self.client, answer_context,
            session_id=state["session_id"], run_id=state["run_id"],
            history=state.get("history", []),
            evidence_selection=selection)
        answer = delivered.answer
        state["answer_audit"] = delivered.audit
        state["answer_delivery"] = delivered.model_dump(mode="json")
        if delivered.error_code:
            state["explanation_error"] = delivered.error_code
        if knowledge is not None:
            state["knowledge"] = knowledge
        from st_agent.trace import emit
        emit(self.copilot, "answer.delivered", delivery=state["answer_delivery"])
        state["tool_replayed"] = response.replayed
        completed = self._result(AgentStatus.COMPLETED, state, answer,
            model_calls=model_calls + ((knowledge or {}).get("api_requests") or 0))
        completed.facts = [
            {
                "tool": result.tool_name,
                "call_id": result.call_id,
                "status": "success",
                "result": payload,
            }
        ]
        completed.fact_records = records
        completed.tool_calls = 0 if response.replayed else 1
        persist_verified_run(self.copilot, state, records)
        self.executor.store.save_checkpoint(
            state["run_id"],
            {"conversation": state, "completed_result": completed.model_dump(mode="json")},
            "completed",
        )
        return completed

    def _deliver_basic(self, result, *, request=None, requirements=None):
        """Adapt completed durable workflows without re-running their computations."""
        task = result.state.get("basic_task")
        if result.status is not AgentStatus.COMPLETED or not task:
            return result
        from st_agent.answer_delivery import (AnswerContext, ResponseRequirements,
                                              deliver_after_execution, from_answer,
                                              normalize_evidence_selection)
        if requirements is not None and not isinstance(requirements, ResponseRequirements):
            requirements = ResponseRequirements.model_validate(requirements)
        persisted_requirements = ResponseRequirements.model_validate(
            task.get("response_requirements") or {})
        incoming_nonreceipt = requirements is not None and requirements.mode != "receipt"
        context_changed = False
        if incoming_nonreceipt:
            chosen_requirements = requirements
            encoded_requirements = chosen_requirements.model_dump(mode="json")
            if task.get("response_requirements") != encoded_requirements:
                task["response_requirements"] = encoded_requirements
                context_changed = True
            if request and task.get("delivery_request") != request:
                task["delivery_request"] = request
                context_changed = True
        elif persisted_requirements.mode != "receipt":
            chosen_requirements = persisted_requirements
        else:
            chosen_requirements = requirements or persisted_requirements
        if incoming_nonreceipt and request:
            context_request = request
        else:
            # A routine receipt/continue turn must not replace the original
            # goal, the retained non-receipt question, or the explicit query
            # that a later approval/resume has to reuse.
            context_request = (task.get("delivery_request") or task.get("request")
                               or request or "基础分析结果")
        selection = normalize_evidence_selection(task.get("saved_evidence"))
        context = AnswerContext(request=context_request,
            dataset_id=task["dataset_id"], revision_id=task["last_revision_id"],
            source="workflow", requirements=chosen_requirements,
            evidence_selection=selection,
            facts=[r.model_dump(mode="json") for r in result.fact_records])
        previous = task.get("answer_deliveries", {}).get(context.fingerprint())
        if previous is None:
            legacy = task.get("answer_delivery") or {}
            if legacy.get("context_fingerprint") == context.fingerprint():
                previous = legacy
        if previous and not previous.get("missing_requirements") and not previous.get("error_code"):
            result.answer = previous["answer"]
            result.state["answer_delivery"] = previous
            if context_changed:
                from st_agent.basic_analysis import BasicAnalysisWorkflow
                BasicAnalysisWorkflow(self.copilot)._save(task)
            return result
        if chosen_requirements.mode == "receipt":
            if task.get("workflow_kind") == "spatial_morans":
                from st_agent.answer_delivery import local_receipt
                delivered = local_receipt(context)
            else:
                delivered = from_answer(context, result.answer,
                {"contract": "workflow-summary/2", "accepted": True,
                 "rendered_fact_ids": []}, fulfilled=["receipt"])
        else:
            delivered, knowledge = deliver_after_execution(self.copilot, self.client, context,
                session_id=task["session_id"], run_id=task["task_id"],
                evidence_selection=selection)
            if knowledge is not None:
                result.state["knowledge"] = knowledge
                result.model_calls += knowledge.get("api_requests") or 0
        result.answer = delivered.answer
        task["answer_delivery"] = delivered.model_dump(mode="json")
        task.setdefault("answer_deliveries", {})[context.fingerprint()] = task["answer_delivery"]
        result.state["answer_delivery"] = task["answer_delivery"]
        from st_agent.basic_analysis import BasicAnalysisWorkflow
        BasicAnalysisWorkflow(self.copilot)._save(task)
        return result

    @staticmethod
    def _result(status, state, answer, *, model_calls=None):
        return AgentResult(
            status=status,
            answer=answer,
            model_calls=state.get("intent_model_calls", 1) if model_calls is None else model_calls,
            tool_calls=0,
            replans=0,
            facts=[],
            fact_records=[],
            events=[],
            state=state,
        )
