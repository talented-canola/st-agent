"""Public facade used by the non-technical local workbench."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from pydantic import Field

from st_agent.agent_runtime import AgentResult, AgentStatus
from st_agent.api import ResearchCopilot
from st_agent.contracts import StrictRecord
from st_agent.db import connect_database
from st_agent.deepseek import DeepSeekClient, DeepSeekProtocol
from st_agent.doctor import DoctorReport, run_doctor
from st_agent.memory import MemoryPreference
from st_agent.models import ArtifactRecord, DatasetRecord, SessionRecord, WorkerJobRecord
from st_agent.reporting import ReportRecord, ReportStore
from st_agent.research_runtime import ResearchAgentRuntime
from st_agent.safety import ApprovalDecision, ApprovalRequest, ApprovalStatus, SafetyStore


class WorkbenchSnapshot(StrictRecord):
    project_id: str
    datasets: tuple[DatasetRecord, ...]
    sessions: tuple[SessionRecord, ...]
    artifacts: tuple[ArtifactRecord, ...]
    approvals: tuple[ApprovalRequest, ...]
    memories: tuple[MemoryPreference, ...]
    active_jobs: tuple[WorkerJobRecord, ...]
    reports: tuple[ReportRecord, ...]


class UserFacingError(StrictRecord):
    title: str
    explanation: str
    next_action: str
    technical_detail: str = Field(default="", repr=False)


def explain_error(error: BaseException) -> UserFacingError:
    if isinstance(error, PermissionError):
        return UserFacingError(
            title="需要研究者确认",
            explanation="该操作受安全策略保护，尚未获得确认。",
            next_action="检查审批卡，确认、修改或拒绝该操作。",
            technical_detail=str(error),
        )
    if isinstance(error, FileNotFoundError):
        return UserFacingError(
            title="找不到文件",
            explanation="数据文件已移动、删除或路径不可访问。",
            next_action="重新选择原始数据文件；系统不会覆盖它。",
            technical_detail=str(error),
        )
    if isinstance(error, ValueError):
        return UserFacingError(
            title="输入或状态不符合要求",
            explanation="系统拒绝了不完整或不安全的操作。",
            next_action="查看参数提示，修正后重新提交。",
            technical_detail=str(error),
        )
    return UserFacingError(
        title="任务未完成",
        explanation="系统遇到未预期问题，但没有把失败标记为成功。",
        next_action="打开本地诊断面板，并保留错误详情用于排查。",
        technical_detail=str(error),
    )


class WorkbenchService:
    def __init__(self, copilot: ResearchCopilot):
        self.copilot = copilot
        self.safety = SafetyStore(copilot.layout.database_path, copilot.project.id)
        self.reports = ReportStore(copilot.layout.database_path, copilot.project.id)

    @property
    def project_path(self) -> Path:
        return self.copilot.layout.root

    @classmethod
    def create(cls, path: str | Path, *, name: str | None = None) -> WorkbenchService:
        return cls(ResearchCopilot.create(path, name=name))

    @classmethod
    def open(cls, path: str | Path) -> WorkbenchService:
        return cls(ResearchCopilot.open(path))

    def snapshot(self) -> WorkbenchSnapshot:
        return WorkbenchSnapshot(
            project_id=self.copilot.project.id,
            datasets=tuple(self.copilot.list_datasets()),
            sessions=tuple(self.copilot.list_sessions()),
            artifacts=tuple(self.copilot.list_artifacts()),
            approvals=tuple(self.safety.list_approvals()),
            memories=tuple(self.copilot.memory.list(include_inactive=True)),
            active_jobs=tuple(self.copilot.worker_jobs.list_active(self.copilot.project.id)),
            reports=tuple(self.reports.list()),
        )

    def add_dataset(self, path: str | Path) -> DatasetRecord:
        return self.copilot.add_dataset(path)

    def start_chat(self, title: str | None = None) -> SessionRecord:
        return self.copilot.create_session(title=title)

    def ensure_chat_session(
        self, session_id: str | None, *, title: str = "科研分析"
    ) -> SessionRecord:
        """Return a session owned by this project, replacing stale browser state."""
        if session_id:
            try:
                session = self.copilot.sessions.get(session_id)
                if session.project_id == self.copilot.project.id:
                    return session
            except KeyError:
                pass
        return self.start_chat(title)

    def submit_message(
        self,
        session_id: str,
        content: str,
        *,
        runner: Callable[[str], str] | None = None,
        dataset_id: str | None = None,
        client: DeepSeekProtocol | None = None,
    ) -> str:
        if runner is None:
            result = self.run_analysis_message(
                session_id,
                dataset_id=dataset_id,
                content=content,
                client=client,
            )
            return self.conversation(session_id)[-1].content if result is None else result.answer
        events = list(self.stream_message(session_id, content, runner=runner))
        return str(events[-1]["answer"])

    def run_analysis_message(
        self,
        session_id: str,
        *,
        dataset_id: str | None,
        content: str,
        client: DeepSeekProtocol | None = None,
    ) -> AgentResult | None:
        """Run one real model/tool conversation and persist its user-visible result."""
        owned = client is None
        previous_messages = self.copilot.list_messages(session_id)
        self.copilot.add_message(
            session_id, role="user", content=content, metadata={"dataset_id": dataset_id}
        )
        if dataset_id is None:
            self.copilot.add_message(
                session_id,
                role="assistant",
                content="请先在数据列表中选择一个已登记的数据集，再开始分析。",
                metadata={"status": "needs_dataset"},
            )
            return None
        if client is None:
            secret = self.copilot.settings.deepseek_api_key
            if secret is None:
                self.copilot.add_message(
                    session_id,
                    role="assistant",
                    content="DeepSeek API Key 尚未配置，当前无法执行分析。",
                    metadata={"status": "provider_unconfigured"},
                )
                return None
            client = DeepSeekClient(
                secret.get_secret_value(),
                base_url=self.copilot.settings.deepseek_base_url,
                model=self.copilot.settings.deepseek_model,
            )
        history = [
            {"role": message.role, "content": message.content}
            for message in previous_messages
            if message.role in {"user", "assistant"}
            and (
                message.metadata.get("dataset_id")
                or message.metadata.get("trace", {}).get("dataset_id")
            )
            == dataset_id
        ]
        try:
            result = ResearchAgentRuntime(self.copilot, client).run_turn(
                content, dataset_id=dataset_id, session_id=session_id, history=history
            )
        finally:
            if owned and hasattr(client, "close"):
                client.close()
        answer = result.answer or self._status_message(result.status)
        self.copilot.add_message(
            session_id,
            role="assistant",
            content=answer,
            metadata={
                "status": result.status.value,
                "model_calls": result.model_calls,
                "tool_calls": result.tool_calls,
                "tools": [str(fact["tool"]) for fact in result.facts],
                "trace": self._message_trace(result.state),
            },
        )
        if answer != result.answer:
            result = result.model_copy(update={"answer": answer})
        return result

    def _message_trace(self, state):
        """Store one copy of large fact tables in the hash-checked checkpoint.

        The runtime result remains complete. Chat retains the audit and source
        references consumed by history, reports and the UI, without duplicating
        the execution input ledger inside bounded message metadata.
        """
        trace = dict(state)
        if "answer_context" in state:
            checkpoint = self.safety.load_checkpoint(state["run_id"])
            saved = checkpoint.state.get("conversation", {}).get("answer_context")
            if saved != state["answer_context"]:
                raise ValueError("message.answer_context_checkpoint_mismatch")
            trace.pop("answer_context")
            trace["answer_context_ref"] = {"checkpoint_id": checkpoint.id,
                "state_hash": checkpoint.state_hash, "run_id": state["run_id"],
                "field": "conversation.answer_context",
                "context_fingerprint": state.get("answer_delivery", {}).get("context_fingerprint")}
        delivery = state.get("answer_delivery", {})
        if "audit" in delivery:
            owner = ("answer_audit" if state.get("answer_audit") == delivery["audit"] else
                "knowledge.answer_audit" if state.get("knowledge", {}).get("answer_audit") == delivery["audit"]
                else None)
            if owner is not None:
                trace["answer_delivery"] = {key: value for key, value in delivery.items() if key != "audit"}
                trace["answer_delivery"]["audit_ref"] = {"scope": "same_message_trace", "field": owner}
        from st_agent.message_trace import pack_message_trace

        return pack_message_trace(self.copilot, state, trace)

    def message_trace(self, message):
        """Resolve an inline or registered trace for presentation consumers."""
        from st_agent.message_trace import load_message_trace

        return load_message_trace(self.copilot, message)

    @staticmethod
    def _status_message(status: AgentStatus) -> str:
        return {
            AgentStatus.PAUSED_PROVIDER: "DeepSeek 暂时不可用，本次分析未被标记为成功。",
            AgentStatus.WAITING_APPROVAL: "分析计划需要研究者确认，请前往“审批与进度”。",
            AgentStatus.NEEDS_CLARIFICATION: "还缺少必要参数，请补充分析要求。",
            AgentStatus.DENIED: "该分析操作已被拒绝。",
            AgentStatus.CANCELLED: "该分析操作已取消。",
            AgentStatus.FAILED: "分析未完成，请查看诊断信息后重试。",
        }.get(status, "分析已完成。")

    def stream_message(
        self,
        session_id: str,
        content: str,
        *,
        runner: Callable[[str], str] | None = None,
    ):
        self.copilot.add_message(session_id, role="user", content=content)
        yield {"type": "message.accepted", "session_id": session_id}
        answer = runner(content) if runner is not None else "未配置分析执行器。"
        self.copilot.add_message(session_id, role="assistant", content=answer)
        yield {"type": "message.completed", "session_id": session_id, "answer": answer}

    def conversation(self, session_id: str):
        return self.copilot.list_messages(session_id)

    def explain_figure(self, session_id, dataset_id, figure_id, *, gene=None, client=None,
                       section_key=None, section_value=None, coordinate_key=None,
                       markers_per_group=3, marker_display_mode="overview",
                       marker_page_index=0, marker_page_size=None):
        from st_agent.figure_gallery import revision_gallery
        from st_agent.scientific_evidence import (GeneratedScientificDraft,
            build_evidence_bundle, interpretation_instruction, interpretation_payload,
            validate_scientific_draft, MAX_SCIENTIFIC_CLAIMS)
        from st_agent.trace import attach, redact

        session = self.copilot.sessions.get(session_id)
        if session.project_id != self.copilot.project.id:
            raise ValueError("Conversation belongs to another project")
        bundle = revision_gallery(self.copilot, dataset_id, gene=gene,
            section_key=section_key, section_value=section_value, coordinate_key=coordinate_key,
            full_extent=True)
        figure = next(item for item in bundle["figures"] if item["id"] == figure_id)
        evidence = build_evidence_bundle(self.copilot, dataset_id, figure=figure)
        facts = {
            "identity": bundle["identity"],
            "title": figure["title"],
            "description": figure["caption"],
        }
        trace = attach(self.copilot)
        image_parts, image_error = [], None
        try:
            from st_agent.full_figures import prepare_figure_images
            image_evidence, image_parts = prepare_figure_images(self.copilot, figure)
            evidence['image_evidence'] = image_evidence
        except Exception as error:
            image_error = str(error) if isinstance(error, ValueError) else type(error).__name__
            trace.emit('scientific.interpretation.image_unavailable', status='degraded',
                       error_code=image_error)
        trace.emit("scientific.interpretation.request", figure_id=figure_id,
            revision_id=evidence["identity"]["revision_id"], evidence=evidence)
        owned = client is None
        if owned:
            settings = self.copilot.settings
            if not settings.deepseek_api_key:
                raise ValueError("DeepSeek API Key 尚未配置")
            client = DeepSeekClient(
                settings.deepseek_api_key.get_secret_value(),
                base_url=settings.deepseek_base_url,
                model=settings.deepseek_model,
                max_requests=None,
                vision_enabled=bool(image_parts),
                timeout_seconds=90 if image_parts else 30,
                retries=1 if image_parts else 2,
            )
        client.trace = trace
        from st_agent.research_context import assemble_context, context_projection
        research_context = assemble_context(self.copilot, dataset_id,
            session_id=session_id, sample=evidence)
        evidence["research_context"] = context_projection(research_context)
        interpretation_request = "请解读当前完整图，并结合研究目标提出需要核验的解释和下一步。"
        evidence["request"] = interpretation_request
        evidence["response_requirements"] = {"mode": "interpretation", "questions": [
            {"id": "Q1", "text": interpretation_request}]}
        reply = None
        provider_error = None
        generation_error = None
        contract_ok = False
        schema_repairs = 0
        content = json.dumps(redact(interpretation_payload(evidence)), ensure_ascii=False)
        messages = [{"role": "system", "content": interpretation_instruction(with_images=bool(image_parts)) +
                    ' Facts are rows under fact_columns; each row has id/key/value/label. '
                    'All rows are bound to fact_revision_id. Copy it exactly into claim.revision_id. '
                    'Every reasoning record must include at least one atomic AnswerUnit.'},
                    {"role": "user", "content": ([{'type': 'text', 'text': content}, *image_parts]
                                                  if image_parts else content)}]

        def _generation_problem(raw):
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError) as error:
                return "answer.invalid_schema", [{"type": "json_invalid",
                    "msg": str(error)[:300]}], 0
            claims = parsed.get("claims") if isinstance(parsed, dict) else None
            count = len(claims) if isinstance(claims, list) else 0
            if isinstance(claims, list) and len(claims) > MAX_SCIENTIFIC_CLAIMS:
                return "scientific.claim_limit", [{"type": "too_long", "loc": ["claims"],
                    "msg": f"maximum is {MAX_SCIENTIFIC_CLAIMS}"}], count
            try:
                GeneratedScientificDraft.model_validate(parsed)
            except ValueError as error:
                errors = (error.errors(include_input=False, include_context=False)
                          if hasattr(error, "errors") else [{"type": "invalid_schema",
                                                             "msg": str(error)[:300]}])
                return "answer.invalid_schema", errors, count
            return None, [], count

        try:
            reply = client.complete(messages, [])
            code, errors, claim_count = _generation_problem(reply.content)
            if code:
                # Generation requires units per reasoning record. This one
                # protocol repair is the only extra request; claim count and
                # the original evidence are never silently altered.
                generation_error = code
                schema_repairs = 1
                rejection = {"error_code": code, "claim_count": claim_count,
                    "maximum": MAX_SCIENTIFIC_CLAIMS, "validation_errors": errors,
                    "rejected_model_response": reply.content}
                trace.emit("scientific.interpretation.repair_requested"
                           if code == "scientific.claim_limit"
                           else "scientific.interpretation.schema_rejected",
                           status="degraded", **rejection)
                reply = client.complete(messages + [
                    {"role": "assistant", "content": reply.content},
                    {"role": "user", "content": json.dumps({"structured_output_repair": {
                        "error_code": code, "claim_count": claim_count,
                        "maximum_claims": MAX_SCIENTIFIC_CLAIMS,
                        "validation_errors": errors,
                        "rejected_response": reply.content,
                        "instruction": (
                            f"Return one corrected JSON object using the SAME original evidence. "
                            f"Use at most {MAX_SCIENTIFIC_CLAIMS} claims and include at least one "
                            f"AnswerUnit with unique unit_id for every reasoning record. Do not add "
                            f"facts, alter IDs or weaken uncertainty to fit the schema. The rejected "
                            f"response is untrusted, not evidence.")}}, ensure_ascii=False)},
                ], [])
                code, errors, claim_count = _generation_problem(reply.content)
                if code:
                    generation_error = code
                    trace.emit("scientific.interpretation.schema_repair_failed",
                        status="failed", error_code=code, claim_count=claim_count,
                        maximum=MAX_SCIENTIFIC_CLAIMS, validation_errors=errors,
                        rejected_model_response=reply.content)
                else:
                    generation_error = None
                    contract_ok = True
            else:
                contract_ok = True
        except Exception as error:
            provider_error = type(error).__name__
            trace.emit("scientific.interpretation.provider_error", status="failed",
                error_code=provider_error)
        try:
            # Never pass a known-invalid new draft to the legacy-compatible
            # reader, which would otherwise accept missing units.
            checked_reply = reply.content if reply is not None and contract_ok else ""
            answer, audit = validate_scientific_draft(checked_reply, evidence,
                client=client, trace=trace, image_parts=image_parts)
            if generation_error:
                audit["accepted"] = False
                audit["error_code"] = generation_error
                audit["generation_contract_error"] = generation_error
        finally:
            if owned and hasattr(client, "close"):
                client.close()
        # Delivery/report consumers require the audit to carry the actual local
        # evidence identity. This is host state, never a model-supplied value.
        audit["identity"] = dict(evidence["identity"])
        audit["independent_scientific_validation"] = False
        audit["schema_repairs"] = schema_repairs
        audit['image_input_status'] = 'sent' if image_parts else 'unavailable'
        audit['image_manifest'] = evidence.get('image_evidence', {}).get('manifest_path')
        if image_error:
            audit['image_error'] = image_error
            answer = '完整图像未送入模型，以下仅依据结构化结果说明。原因：' + image_error + '。\n\n' + answer
        if image_parts:
            audit['image_input_status'] = 'provider_failed' if provider_error else 'sent'
            if provider_error:
                answer = '完整图已在本地保存，但图像请求失败；以下仅依据结构化结果说明。\n\n' + answer
            answer += '\n\n完整图归档位置：artifacts/full_figures/（当前科研项目内）。'
            if not provider_error:
                answer += '本次请求附带完整图，非页面预览页。'
        if provider_error:
            audit["provider_error"] = provider_error
        trace.emit("scientific.interpretation.validated",
            status="ok" if audit["accepted"] else "degraded", audit=audit,
            model_response=reply.content if reply else None, answer=answer)
        from st_agent.answer_delivery import (AnswerContext, ResponseRequirements,
            from_answer, interpretation_completion)

        answer_context = AnswerContext(request=interpretation_request, dataset_id=dataset_id,
            revision_id=evidence["identity"]["revision_id"], source="figure",
            requirements=ResponseRequirements(mode="interpretation",
                questions=[interpretation_request]),
            facts=evidence.get("facts", []), evidence_identity=evidence.get("identity", {}))
        delivery_error = audit.get("error_code") or (
            "scientific.interpretation.provider_error" if provider_error else
            "scientific.interpretation.image_unavailable" if image_error else None)
        delivery_status = ("grounded" if audit.get("accepted") is True and
            not audit.get("error_code") and not provider_error and not image_error
            else "degraded")
        fulfilled, missing = interpretation_completion(answer_context, {"answer": answer,
            "status": delivery_status, "answer_audit": audit, "error_code": delivery_error})
        answer_result = from_answer(answer_context, answer, audit,
            fulfilled=fulfilled, missing=missing, error_code=delivery_error)
        trace.emit("scientific.interpretation.delivered",
            answer_context=answer_context.model_dump(mode="json"),
            answer_result=answer_result.model_dump(mode="json"), automatic_execution=False)
        payload = {"evidence": evidence, "validation": audit, "answer": answer,
            "answer_context": answer_context.model_dump(mode="json"),
            "answer_result": answer_result.model_dump(mode="json")}
        saved = trace.directory / "scientific_interpretation.json"
        evidence_path = None
        try:
            saved.write_text(json.dumps(redact(payload), ensure_ascii=False, indent=2), encoding="utf-8")
            evidence_path = saved.relative_to(self.copilot.layout.root).as_posix()
        except OSError:
            trace._degrade()
            trace.emit("trace_degraded", status="degraded",
                error_code="trace.interpretation_payload_unavailable")
        from st_agent.message_trace import pack_message_trace

        trace_directory = trace.directory.relative_to(self.copilot.layout.root).as_posix()
        context_state = answer_context.model_dump(mode="json")
        delivery_state = answer_result.model_dump(mode="json")
        # A presentation-only, stable binding: no execution row is invented.
        # The trace directory makes each figure explanation a distinct
        # presentation identity within one session/dataset.
        presentation_run_id = "run_" + hashlib.sha256(
            f"figure-presentation:{session_id}:{dataset_id}:{trace_directory}".encode()
        ).hexdigest()[:32]
        full_state = {"run_id": presentation_run_id, "session_id": session_id,
            "dataset_id": dataset_id, "answer_audit": audit,
            "answer_context": context_state, "answer_delivery": delivery_state,
            "scientific_interpretation": {"evidence_path": evidence_path,
                "trace_directory": trace_directory,
                "image_input_status": audit.get("image_input_status")}}
        compact_trace = {"figure": facts, "response_id": reply.response_id if reply else None,
            "run_id": presentation_run_id, "session_id": session_id,
            "dataset_id": dataset_id, "scientific_validation": audit,
            "evidence_path": evidence_path, "trace_directory": trace_directory,
            "answer_context": context_state, "answer_delivery": delivery_state}
        message_trace = pack_message_trace(self.copilot, full_state, compact_trace)
        self.copilot.add_message(session_id, role="user", content=f"请解读：{figure['title']}",
                                 metadata={"dataset_id": dataset_id})
        self.copilot.add_message(
            session_id,
            role="assistant",
            content=answer,
            metadata={"trace": message_trace},
        )
        return answer

    def decide_approval(
        self, approval_id: str, decision: ApprovalDecision, edited_arguments: dict | None = None
    ) -> ApprovalRequest:
        return self.safety.decide(approval_id, decision, edited_arguments=edited_arguments)

    def pending_approvals(self) -> list[ApprovalRequest]:
        return self.safety.list_approvals(status=ApprovalStatus.PENDING)

    def resume_analysis(self, approval_id: str, *, client=None) -> AgentResult:
        approval = self.safety.get_approval(approval_id)
        owned = client is None
        if client is None:
            settings = self.copilot.settings
            if not settings.deepseek_api_key:
                raise ValueError("provider.secret_missing")
            client = DeepSeekClient(
                settings.deepseek_api_key.get_secret_value(),
                base_url=settings.deepseek_base_url,
                model=settings.deepseek_model,
            )
        try:
            result = ResearchAgentRuntime(self.copilot, client).resume_approval(approval_id)
        finally:
            if owned and hasattr(client, "close"):
                client.close()
        return self._record_resumed_result(approval.session_id, result)

    def resume_run(self, run_id: str, *, client=None) -> AgentResult:
        owned = client is None
        if client is None:
            settings = self.copilot.settings
            if not settings.deepseek_api_key:
                raise ValueError("provider.secret_missing")
            client = DeepSeekClient(
                settings.deepseek_api_key.get_secret_value(),
                base_url=settings.deepseek_base_url,
                model=settings.deepseek_model,
            )
        try:
            result = ResearchAgentRuntime(self.copilot, client).resume_run(run_id)
        finally:
            if owned and hasattr(client, "close"):
                client.close()
        return self._record_resumed_result(result.state["session_id"], result)

    def _record_resumed_result(self, session_id, result):
        existing = self.copilot.list_messages(session_id)
        if any(
            message.role == "assistant"
            and message.content == result.answer
            and message.metadata.get("trace", {}).get("run_id") == result.state["run_id"]
            and message.metadata.get("status") == result.status.value
            for message in existing
        ):
            return result
        self.copilot.add_message(
            session_id,
            role="assistant",
            content=result.answer,
            metadata={"status": result.status.value, "trace": self._message_trace(result.state)},
        )
        return result

    def cancel_run(self, run_id: str, reason: str = "researcher") -> bool:
        return self.safety.cancel_run(run_id, reason)

    def cancel_job(self, job_id: str) -> WorkerJobRecord:
        return self.copilot.worker_jobs.request_cancel(job_id)

    def revisions(self, dataset_id: str):
        return self.copilot.revisions.list(dataset_id)

    def forget_memory(self, memory_id: str) -> bool:
        return self.copilot.memory.forget(memory_id)

    def decide_memory(self, memory_id: str, decision: str, edited_value=None):
        record = self.copilot.memory.get(memory_id)
        if record.key == "qc_filter_profile" and decision in {"approve", "edit_and_approve"}:
            from st_agent.memory_runtime import RememberedFilter

            value = edited_value if decision == "edit_and_approve" else record.value
            validated = (
                RememberedFilter.model_validate_json(value)
                if isinstance(value, str)
                else RememberedFilter.model_validate(value)
            )
            if decision == "edit_and_approve":
                edited_value = validated.model_dump()
        return self.copilot.memory.decide(memory_id, decision, edited_value)

    def create_analysis_report(self, dataset_id: str, *, session_id: str | None = None) -> ReportRecord:
        """Build a downloadable report from existing committed evidence only."""
        from st_agent.reporting import create_analysis_report

        return create_analysis_report(self.copilot, dataset_id, session_id=session_id)

    def finalize_report(self, report_id: str) -> ReportRecord:
        return self.reports.finalize(report_id, researcher_confirmed=True)

    def evidence_and_claims(self) -> dict[str, list[dict]]:
        with connect_database(self.copilot.layout.database_path) as connection:
            evidence = [
                dict(row)
                for row in connection.execute(
                    """SELECT id,query_text,corpus_version,created_at FROM evidence_bundles
                    WHERE project_id=? ORDER BY created_at DESC""",
                    (self.copilot.project.id,),
                ).fetchall()
            ]
            claims = [
                dict(row)
                for row in connection.execute(
                    """SELECT id,claim_text,strength,status,reason,created_at FROM claim_ledger
                    WHERE project_id=? ORDER BY created_at DESC""",
                    (self.copilot.project.id,),
                ).fetchall()
            ]
        return {"evidence": evidence, "claims": claims}

    def diagnostics(self) -> DoctorReport:
        return run_doctor(self.copilot.layout.root, settings=self.copilot.settings)
