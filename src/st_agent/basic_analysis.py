"""Durable sequential orchestration over the existing one-step safety boundary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from st_agent.agent_runtime import AgentEngine, AgentResult, AgentStatus
from st_agent.answer_delivery import normalize_evidence_selection
from st_agent.contracts import ToolStatus
from st_agent.dataset_state import DatasetStateInspector
from st_agent.ids import IdKind, new_id
from st_agent.intent_runtime import effective_delivery_requirements
from st_agent.planning import DeterministicPlanner, TaskIntent
from st_agent.safety import ApprovalStatus, GateDisposition, RecoveryRequired, SafetyStore
from st_agent.scientific_execution import ScientificPlanExecutor
from st_agent.scientific_tools import TOOLS
from st_agent.trace import attach

STEPS = ("inspect", "qc_summary", "qc_filter", "preprocess", "cluster", "marker")
TOOL_NAMES = ("dataset.inspect", "qc.compute_metrics", "qc.apply_filter",
              "preprocess.standard", "cluster.leiden_umap", "markers.rank")


class BasicAnalysisWorkflow:
    def __init__(self, copilot):
        self.copilot = copilot
        self.store = SafetyStore(copilot.layout.database_path, copilot.project.id)
        self.executor = ScientificPlanExecutor(copilot)

    def active(self, dataset_id, session_id, *, include_completed=False):
        with self.store._connect() as connection:
            rows = connection.execute("SELECT run_id FROM safety_checkpoints WHERE project_id=? "
                "AND json_extract(state_json,'$.basic_task.session_id')=? "
                "AND json_extract(state_json,'$.basic_task.dataset_id')=? "
                "GROUP BY run_id ORDER BY MAX(rowid) DESC", (self.copilot.project.id,
                    session_id, dataset_id)).fetchall()
        for row in rows:
            task = self.store.load_checkpoint(row["run_id"]).state["basic_task"]
            if task["status"] not in {"completed", "denied", "cancelled"} or (
                    include_completed and task["status"] == "completed"):
                return task
        return None

    def _trace(self, task):
        current = getattr(self.copilot, "trace", None)
        expected = self.copilot.layout.root / task["trace_directory"]
        if current is None or current.directory != expected:
            current = attach(self.copilot, self.copilot.layout.root / task["trace_directory"])
        current.context.update(task_id=task["task_id"], run_id=task["task_id"],
                               step_id=None, attempt_id=None)
        return current

    def _save(self, task):
        from st_agent.utils import utc_now_iso
        with self.store._connect() as connection:
            connection.execute("INSERT INTO runs(id,session_id,dataset_id,active_revision_id,"
                "status,goal,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET status=excluded.status,"
                "active_revision_id=excluded.active_revision_id,updated_at=excluded.updated_at",
                (task["task_id"], task["session_id"], task["dataset_id"],
                 task["last_revision_id"], task["status"], task.get("workflow_kind", "basic_analysis"),
                 utc_now_iso(), utc_now_iso()))
            connection.commit()
        self.store.save_checkpoint(task["task_id"], {"basic_task": task}, task["status"])

    @staticmethod
    def _remember_intent_context(task, intent, request=None):
        """Persist explicit delivery context without letting null clear it.

        Action continuation and requested output are independent. A deferred
        directive therefore persists even when the top-level action demand is a
        receipt, while an omitted selection never clears an earlier one.
        """
        changed = False
        delivery = intent.deferred_delivery
        selected = (delivery.saved_evidence if delivery is not None
                    and delivery.saved_evidence is not None else intent.saved_evidence)
        selection = normalize_evidence_selection(selected)
        if selection is not None and task.get("saved_evidence") != selection:
            task["saved_evidence"] = selection
            changed = True
        requirements_model = effective_delivery_requirements(intent)
        requirements = requirements_model.model_dump(mode="json")
        if delivery is not None or requirements_model.mode != "receipt":
            if task.get("response_requirements") != requirements:
                task["response_requirements"] = requirements
                changed = True
            # Keep the workflow goal distinct from the explicit delivery query:
            # later approval/resume must reuse the query that was actually asked.
            delivery_request = request or (delivery.quote if delivery is not None else None)
            if delivery_request and task.get("delivery_request") != delivery_request:
                task["delivery_request"] = delivery_request
                changed = True
        return changed

    def submit(self, intent, *, dataset_id, session_id, request=None):
        self.copilot.list_messages(session_id)
        dataset = self.copilot.get_dataset(dataset_id)
        task = self.active(dataset_id, session_id)
        if task is None and not intent.new_analysis:
            existing = self.active(dataset_id, session_id, include_completed=True)
            if existing is not None and existing["status"] == "completed":
                # Capture an explicit follow-up selection/requirement before
                # completed-result reuse returns through resume().
                if self._remember_intent_context(existing, intent, request):
                    self._save(existing)
                return self.resume(existing["task_id"])
        if task is None:
            identity = new_id(IdKind.RUN)
            task = {"task_id": identity, "session_id": session_id, "dataset_id": dataset_id,
                "request": request or "基础分析", "response_requirements": intent.response_requirements.model_dump(mode="json"),
                "saved_evidence": None, "delivery_request": None,
                "initial_revision_id": dataset.active_revision_id,
                "last_revision_id": dataset.active_revision_id, "parameters": {},
                "completed_steps": [], "steps": [], "status": "needs_clarification",
                "artifacts": [], "missing_parameters": [], "error_code": None,
                "trace_directory": "artifacts/traces/" + identity}
            current_trace = getattr(self.copilot, "trace", None)
            if current_trace is not None:
                task["trace_directory"] = current_trace.directory.relative_to(
                    self.copilot.layout.root).as_posix()
        self._remember_intent_context(task, intent, request)
        if request and (not task.get("request") or task["request"] == "基础分析"):
            task["request"] = request
        self._save(task)
        trace = self._trace(task)
        trace.emit("request", parameters=intent.parameters)
        if intent.mode.value != "execute":
            return self._result(task, AgentStatus.NEEDS_CLARIFICATION,
                                "这是基础分析计划，尚未执行；请明确是否开始。")
        if task["status"] not in {"needs_clarification", "parameters_invalid"}:
            changed = any(task["parameters"].get(group, {}).get(key) != value
                          for group, values in intent.parameters.items()
                          if isinstance(values, dict) for key, value in values.items())
            if changed:
                return self._result(task, AgentStatus.NEEDS_CLARIFICATION,
                    "原任务已有执行或审批记录；不能静默修改参数。请先拒绝当前计划，再开始新的分析。")
            return self.resume(task["task_id"])
        try:
            if set(intent.parameters) - set(STEPS[2:]):
                raise ValueError("Unexpected basic-analysis parameter group")
            for group, values in intent.parameters.items():
                if not isinstance(values, dict):
                    raise ValueError("Step parameters must be objects")
                merged = {**task["parameters"].get(group, {}), **values}
                # Validate supplied fields even while mandatory values are absent.
                tool = TOOLS[TOOL_NAMES[STEPS.index(group)]]
                if group == "cluster" and "seed" not in merged:
                    tool.arguments.model_validate({**merged, "seed": 0})
                else:
                    tool.arguments.model_validate(merged)
                task["parameters"][group] = merged
            missing = ["qc_filter." + key for key in ("min_genes", "min_cells", "max_mito_pct")
                       if key not in task["parameters"].get("qc_filter", {})]
            if "seed" not in task["parameters"].get("cluster", {}):
                missing.append("cluster.seed")
            task["missing_parameters"] = missing
            if missing:
                task["status"] = "needs_clarification"
                self._save(task)
                return self._result(task, AgentStatus.NEEDS_CLARIFICATION,
                    "基础分析任务已保存，尚未开始。请补充：" + "、".join(missing) +
                    "。其他工具默认参数会在对应审批中展示。")
            task["parameters"] = {group: TOOLS[TOOL_NAMES[STEPS.index(group)]].arguments
                .model_validate(task["parameters"].get(group, {})).model_dump(mode="json")
                for group in STEPS[2:]}
        except (ValueError, ValidationError) as error:
            task.update(status="parameters_invalid", error_code="basic.parameters_invalid")
            trace.emit("error", status="failed", error_code=task["error_code"],
                       error_type=type(error).__name__)
            self._save(task)
            return self._result(task, AgentStatus.NEEDS_CLARIFICATION,
                                "基础分析参数不符合工具契约；请修正参数，任务和原始数据均已保留。")
        task["plan_hash"] = hashlib.sha256(json.dumps(task["parameters"], sort_keys=True)
                                            .encode()).hexdigest()
        task["status"] = "running"
        self._save(task)
        return self.resume(task["task_id"])

    def submit_spatial_dependency(self, intent, *, dataset_id, session_id, request):
        """Reuse the durable step engine for Moran's deterministic graph dependency."""
        from st_agent.tool_registry import SCIENTIFIC_TOOLS
        task = self.active(dataset_id, session_id)
        if task is not None:
            if task.get("workflow_kind") != "spatial_morans":
                return self._result(task, AgentStatus.NEEDS_CLARIFICATION,
                    "当前有尚未完成的分析任务，请先恢复或取消该任务。")
            if self._remember_intent_context(task, intent, request):
                self._save(task)
            return self.resume(task["task_id"])
        observed = DatasetStateInspector(self.copilot).inspect(dataset_id)
        parameters = SCIENTIFIC_TOOLS["spatial.morans_i"].arguments.model_validate_json(
            json.dumps(intent.parameters)).model_dump(mode="json")
        graph = SCIENTIFIC_TOOLS["spatial.neighbors"].arguments.model_validate_json("{}").model_dump(mode="json")
        identity = new_id(IdKind.RUN)
        delivery = intent.deferred_delivery
        requirements_model = effective_delivery_requirements(intent)
        selected = (delivery.saved_evidence if delivery is not None
                    and delivery.saved_evidence is not None else intent.saved_evidence)
        task = {"task_id": identity, "session_id": session_id, "dataset_id": dataset_id,
            "workflow_kind": "spatial_morans", "operations": ["spatial_neighbors", "spatial_morans"],
            "request": request, "response_requirements": requirements_model.model_dump(mode="json"),
            "saved_evidence": normalize_evidence_selection(selected),
            "delivery_request": ((request or delivery.quote)
                if delivery is not None or requirements_model.mode != "receipt" else None),
            "initial_revision_id": observed.input_revision_id, "last_revision_id": observed.input_revision_id,
            "parameters": {"spatial_neighbors": graph, "spatial_morans": parameters},
            "completed_steps": [], "steps": [], "status": "running", "artifacts": [],
            "missing_parameters": [], "error_code": None,
            "trace_directory": "artifacts/traces/" + identity}
        task["plan_hash"] = hashlib.sha256(json.dumps(task["parameters"], sort_keys=True).encode()).hexdigest()
        self._save(task)
        return self.resume(identity)

    def resume(self, task_id, *, approval_id=None):
        task = self.store.load_checkpoint(task_id).state["basic_task"]
        self._save(task)
        trace = self._trace(task)
        trace.emit("recovery.request", approval_id=approval_id, previous_status=task["status"])
        if task["status"] == "completed":
            from st_agent.utils import sha256_file
            if not task["artifacts"] or any(not (self.copilot.layout.root / item["relative_path"]).is_file()
                or sha256_file(self.copilot.layout.root / item["relative_path"]) != item["sha256"]
                for item in task["artifacts"]):
                return self._fail(task, "basic.completed_artifact_missing",
                                  "已有分析的产物缺失或指纹不符；保留计算结果，请续接导出。")
            return self._result(task, AgentStatus.COMPLETED, "基础分析已完成，已复用保存的结果；未重新计算。",
                                replayed=True)
        if task["status"] in {"denied", "cancelled"}:
            return self._result(task, AgentStatus(task["status"]), "任务已停止，未继续执行。")
        if task["missing_parameters"] or task["status"] == "parameters_invalid":
            return self._result(task, AgentStatus.NEEDS_CLARIFICATION, "请补齐或修正原任务参数。")
        if self.store.is_cancelled(task_id):
            task["status"] = "cancelled"
            self._save(task)
            return self._result(task, AgentStatus.CANCELLED, "基础分析已取消。")
        operations = task.get("operations", STEPS)
        while len(task["completed_steps"]) < len(operations):
            index = len(task["completed_steps"])
            operation = operations[index]
            if len(task["steps"]) <= index:
                observed = DatasetStateInspector(self.copilot).inspect(task["dataset_id"])
                if observed.input_revision_id != task["last_revision_id"]:
                    return self._fail(task, "basic.input_revision_changed", "数据版本已变化，停止旧任务。")
                intent = TaskIntent.model_validate_json(json.dumps({"operation": operation,
                    "mode": "execute", "confidence": 1.0, "rationale": "确定性基础流程",
                    "parameters": task["parameters"].get(operation, {})}))
                plan = DeterministicPlanner().build(intent, observed)
                if plan.blocked_reason:
                    return self._fail(task, plan.blocked_reason, "分析前置条件不满足，已暂停。")
                step_run = new_id(IdKind.RUN)
                task["steps"].append({"step_id": task_id + ":" + operation,
                    "run_id": step_run, "attempt": 1, "operation": operation,
                    "plan": plan.model_dump(mode="json"), "approval_id": None,
                    "status": "pending", "result": None})
                self._save(task)
                trace.emit("planning", operation=operation, plan=plan.model_dump(mode="json"))
            step = task["steps"][index]
            failed_result = step.get("result")
            if (approval_id is None and failed_result and failed_result["status"] not in
                    {"success", "warning"}):
                committed = self.copilot.revisions.find_by_call(task["dataset_id"],
                                                               failed_result["call_id"])
                if committed is None or committed.status.value != "committed":
                    if self.copilot.worker_jobs.list_active(self.copilot.project.id):
                        return self._fail(task, "basic.recovery_worker_active", "仍有活动Worker，未重复执行。")
                    if self.copilot.get_dataset(task["dataset_id"]).active_revision_id != task["last_revision_id"]:
                        return self._fail(task, "basic.input_revision_changed", "失败后输入版本已变化。")
                    # Preserve the failed attempt. Fresh identity needs a fresh, normally bound approval.
                    step.setdefault("attempt_history", []).append({key: step[key] for key in
                        ("run_id", "attempt", "approval_id", "status", "result")})
                    step.update(run_id=new_id(IdKind.RUN), attempt=step["attempt"] + 1,
                                approval_id=None, status="pending", result=None)
                    task["status"] = "running"
                    self._save(task)
                    trace.emit("recovery.new_attempt", step_id=step["step_id"], attempt=step["attempt"])
            trace.context.update(step_id=step["step_id"], run_id=step["run_id"],
                                 attempt_id=step["attempt"])
            from st_agent.planning import ExecutionPlan
            plan = ExecutionPlan.model_validate_json(json.dumps(step["plan"]))
            selected_approval = approval_id or step["approval_id"]
            if selected_approval:
                approval = self.store.get_approval(selected_approval)
                if approval.run_id != step["run_id"]:
                    return self._fail(task, "basic.approval_step_mismatch", "审批不属于当前步骤。")
                trace.emit("approval.decision", approval_id=selected_approval,
                           decision=approval.status.value, argument_hash=approval.argument_hash)
                if approval.status is ApprovalStatus.REJECTED:
                    task["status"] = "denied"
                    self._save(task)
                    return self._result(task, AgentStatus.DENIED, "已拒绝该步骤，基础分析停止。")
            conversation = {"basic_task_id": task_id, "run_id": step["run_id"],
                "session_id": task["session_id"], "dataset_id": task["dataset_id"]}
            try:
                response = self.executor.execute(plan, dataset_id=task["dataset_id"],
                    session_id=task["session_id"], run_id=step["run_id"],
                    approval_id=selected_approval, conversation_state=conversation)
            except RecoveryRequired:
                return self._fail(task, "basic.recovery_uncertain",
                    "执行状态尚未核对，保留当前步骤，不重复提交。")
            except Exception as error:
                return self._fail(task, getattr(error, "code", type(error).__name__),
                                  "当前步骤执行失败，已保留完成的结果。")
            step["approval_id"] = response.approval_id
            approval_id = None
            if response.tool_result is None:
                task["status"] = response.disposition.value
                step["status"] = task["status"]
                self._save(task)
                trace.emit("approval.request", approval_id=response.approval_id,
                           parameters=plan.steps[0].parameters)
                status = (AgentStatus.WAITING_APPROVAL if response.disposition is
                          GateDisposition.WAITING_APPROVAL else AgentStatus.FAILED)
                return self._result(task, status,
                    f"已完成 {index}/{len(operations)} 步。下一步 {operation}，请确认展示的参数后继续。")
            result = response.tool_result
            step["result"] = result.model_dump(mode="json")
            trace.emit("tool.result", status=result.status.value, duration_ms=result.provenance.duration_ms,
                       result=step["result"],
                       replayed=response.replayed)
            if result.status not in {ToolStatus.SUCCESS, ToolStatus.WARNING}:
                return self._fail(task, result.error.code if result.error else "tool.failed",
                                  f"基础分析停在 {operation}；已完成步骤保留，可核对后恢复。")
            step["status"] = "success"
            task["completed_steps"].append(operation)
            task["last_revision_id"] = result.output_revision_id or result.input_revision_id
            task["status"] = "running"
            task["error_code"] = None
            self._save(task)
        if task.get("workflow_kind") == "spatial_morans":
            task["artifacts"] = [ref for step in task["steps"]
                for ref in step["result"].get("artifacts", [])]
            if not task["artifacts"]:
                return self._fail(task, "spatial.required_artifacts_missing", "空间结果表图未保存。")
            task["status"] = "completed"
            self._save(task)
            return self._result(task, AgentStatus.COMPLETED,
                "空间邻域与Moran's I已完成，完整统计表和图已保存；结果绑定版本 " + task["last_revision_id"] + "。")
        try:
            from st_agent.basic_exports import export_basic_results
            task["artifacts"] = export_basic_results(self.copilot, task)
        except Exception as error:
            return self._fail(task, "basic.export_failed", "分析计算已完成，导出未完成；可只重试导出。",
                              error_type=type(error).__name__)
        task["status"] = "completed"
        self._save(task)
        trace.emit("task.completed", revision_id=task["last_revision_id"], artifacts=task["artifacts"])
        summary = next(item for item in task["artifacts"] if item["kind"] == "summary.scientific")
        overview = (self.copilot.layout.root / summary["relative_path"]).read_text(encoding="utf-8")
        return self._result(task, AgentStatus.COMPLETED,
            "基础分析六步已完成，marker 表、QC 与 UMAP 图已保存。结果绑定最终版本 " +
            task["last_revision_id"] + "；这证明流程和产物可用，不代表已验证生物学结论。\n\n"
            + "完整图表已保存到：" + str(self.copilot.layout.root / 'artifacts' / 'full_figures')
            + "。同参数图会校验后复用，空间图按切片分别保存。\n\n" + overview)

    def _fail(self, task, code, message, **detail):
        task.update(status="failed", error_code=code)
        self._trace(task).emit("error", status="failed", error_code=code, **detail)
        self._save(task)
        return self._result(task, AgentStatus.FAILED, message + " 错误码：" + code)

    def _result(self, task, status, answer, *, replayed=False):
        facts, records = [], []
        for step in task["steps"]:
            result = step.get("result")
            if result and step["status"] == "success":
                facts.append({"tool": result["tool_name"], "call_id": result["call_id"],
                              "status": "success", "result": result["metrics"]})
                records.extend(AgentEngine._records_from_result(result["tool_name"],
                    {**result["metrics"], "input_revision_id": result["input_revision_id"],
                     "output_revision_id": result["output_revision_id"]}, len(records)))
        state = {"run_id": task["task_id"], "session_id": task["session_id"],
            "dataset_id": task["dataset_id"], "basic_task": task,
            "approval_id": task["steps"][-1]["approval_id"] if task["steps"] and
                           status is AgentStatus.WAITING_APPROVAL else None,
            "trace_degraded": self._trace(task).degraded}
        if replayed:
            state["replayed"] = True
        self._trace(task).emit("answer.rendered", status=status.value, answer=answer)
        return AgentResult(status=status, answer=answer, model_calls=0,
            tool_calls=0 if replayed else len(facts), replans=0, facts=facts, fact_records=records, events=[], state=state)
