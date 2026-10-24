"""Fail-closed Phase 5 policy, approval, checkpoint, and recovery controls."""

from __future__ import annotations

import hashlib
import json
import re
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field

from st_agent.contracts import Effect, Risk, StrictRecord
from st_agent.db import connect_database
from st_agent.ids import IdKind, new_id
from st_agent.serialization import canonical_json, validate_json_value
from st_agent.utils import utc_now_iso


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    EDIT_AND_APPROVE = "edit_and_approve"


class GateDisposition(StrEnum):
    AUTOMATIC = "automatic"
    WAITING_APPROVAL = "waiting_approval"
    AUTHORIZED = "authorized"
    DENIED = "denied"
    CANCELLED = "cancelled"


class PolicyDecision(StrictRecord):
    effective_risk: Risk
    automatic: bool
    requires_approval: bool
    denied: bool
    reasons: tuple[str, ...]
    parameter_diff: dict[str, dict[str, Any]] = Field(default_factory=dict)


SCIENTIFIC_FIELDS = {
    "qc.apply_filter": {"min_genes", "min_cells", "max_mito_pct", "min_counts"},
    "preprocess.standard": {"n_hvg", "n_pcs", "target_sum"},
    "cluster.leiden_umap": {"seed", "n_neighbors", "resolution"},
    "markers.rank": {"n_genes", "method"},
}


class RiskPolicy:
    def evaluate(
        self,
        declared: Risk,
        effect: Effect,
        tool_name: str,
        proposed: dict[str, Any],
        previously_approved: dict[str, Any] | None,
    ) -> PolicyDecision:
        proposed = validate_json_value(proposed)
        previous = (
            validate_json_value(previously_approved) if previously_approved is not None else None
        )
        fields = SCIENTIFIC_FIELDS.get(tool_name, set())
        diff: dict[str, dict[str, Any]] = {}
        if previous is not None:
            for key in fields:
                if previous.get(key) != proposed.get(key):
                    diff[key] = {"before": previous.get(key), "after": proposed.get(key)}
        reasons: list[str] = []
        effective = declared
        if effect is Effect.EXTERNAL_IO:
            effective = max(effective, Risk.R3)
            reasons.append("external_io")
        if tool_name in SCIENTIFIC_FIELDS and (previous is None or diff):
            effective = max(effective, Risk.R2)
            reasons.append("scientific_parameters")
        denied = effective is Risk.R3
        requires = effective is Risk.R2
        return PolicyDecision(
            effective_risk=effective,
            automatic=not denied and not requires,
            requires_approval=requires,
            denied=denied,
            reasons=tuple(reasons) or ("declared_risk",),
            parameter_diff=diff,
        )


class ApprovalRequest(StrictRecord):
    id: str
    project_id: str
    session_id: str
    run_id: str
    dataset_id: str
    revision_id: str
    tool_name: str
    tool_version: str
    status: ApprovalStatus
    risk: Risk
    proposed_arguments: dict[str, Any]
    final_arguments: dict[str, Any]
    argument_hash: str
    reasons: tuple[str, ...]
    decision: ApprovalDecision | None = None
    created_at: str
    decided_at: str | None = None


class Checkpoint(StrictRecord):
    id: str
    project_id: str
    run_id: str
    sequence: int = Field(ge=1)
    next_node: str
    state: dict[str, Any]
    state_hash: str
    created_at: str


class GateOutcome(StrictRecord):
    disposition: GateDisposition
    policy: PolicyDecision
    approval: ApprovalRequest | None = None
    arguments: dict[str, Any]
    execution_key: str


class SafetyInterruption(RuntimeError):
    def __init__(self, outcome: GateOutcome):
        self.outcome = outcome
        super().__init__(f"Execution blocked: {outcome.disposition.value}")


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class SafetyStore:
    def __init__(self, database_path: str | Path, project_id: str, busy_timeout_ms: int = 30_000):
        self.database_path = Path(database_path)
        self.project_id = project_id
        self.busy_timeout_ms = busy_timeout_ms

    def _connect(self):
        return connect_database(self.database_path, busy_timeout_ms=self.busy_timeout_ms)

    def request_approval(
        self,
        *,
        session_id: str,
        run_id: str,
        dataset_id: str,
        revision_id: str,
        tool_name: str,
        tool_version: str,
        arguments: dict[str, Any],
        risk: Risk,
        reasons: tuple[str, ...],
    ) -> ApprovalRequest:
        arguments = validate_json_value(arguments)
        argument_hash = _digest(arguments)
        now = utc_now_iso()
        record = ApprovalRequest(
            id=new_id(IdKind.APPROVAL),
            project_id=self.project_id,
            session_id=session_id,
            run_id=run_id,
            dataset_id=dataset_id,
            revision_id=revision_id,
            tool_name=tool_name,
            tool_version=tool_version,
            status=ApprovalStatus.PENDING,
            risk=risk,
            proposed_arguments=arguments,
            final_arguments=arguments,
            argument_hash=argument_hash,
            reasons=reasons,
            created_at=now,
        )
        with self._connect() as connection:
            existing = connection.execute(
                """SELECT * FROM safety_approvals
                WHERE project_id=? AND run_id=? AND dataset_id=? AND revision_id=?
                AND tool_name=? AND tool_version=? AND argument_hash=?""",
                (
                    self.project_id,
                    run_id,
                    dataset_id,
                    revision_id,
                    tool_name,
                    tool_version,
                    argument_hash,
                ),
            ).fetchone()
            if existing is not None:
                return self._approval(existing)
            connection.execute(
                """INSERT INTO safety_approvals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.id,
                    record.project_id,
                    record.session_id,
                    record.run_id,
                    record.dataset_id,
                    record.revision_id,
                    record.tool_name,
                    record.tool_version,
                    record.status.value,
                    record.risk.value,
                    canonical_json(record.proposed_arguments),
                    canonical_json(record.final_arguments),
                    record.argument_hash,
                    canonical_json(record.reasons),
                    None,
                    record.created_at,
                    None,
                ),
            )
            connection.commit()
        return record

    def decide(
        self,
        approval_id: str,
        decision: ApprovalDecision,
        *,
        edited_arguments: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        current = self.get_approval(approval_id)
        if current.status is not ApprovalStatus.PENDING:
            if current.decision is not decision:
                raise ValueError("Approval already has a different durable decision")
            return current
        if decision is ApprovalDecision.EDIT_AND_APPROVE:
            if edited_arguments is None:
                raise ValueError("Edited approval requires arguments")
            final = validate_json_value(edited_arguments)
        elif edited_arguments is not None:
            raise ValueError("Edited arguments require edit_and_approve")
        else:
            final = current.proposed_arguments
        status = (
            ApprovalStatus.REJECTED
            if decision is ApprovalDecision.REJECT
            else ApprovalStatus.APPROVED
        )
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """UPDATE safety_approvals SET status=?, final_json=?, argument_hash=?, decision=?,
                decided_at=? WHERE id=? AND status='pending'""",
                (
                    status.value,
                    canonical_json(final),
                    _digest(final),
                    decision.value,
                    now,
                    approval_id,
                ),
            )
            connection.commit()
        return self.get_approval(approval_id)

    def get_approval(self, approval_id: str) -> ApprovalRequest:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM safety_approvals WHERE id=? AND project_id=?",
                (approval_id, self.project_id),
            ).fetchone()
        if row is None:
            raise KeyError(approval_id)
        return self._approval(row)

    def list_approvals(self, *, status: ApprovalStatus | None = None) -> list[ApprovalRequest]:
        sql = "SELECT * FROM safety_approvals WHERE project_id=?"
        params: list[Any] = [self.project_id]
        if status is not None:
            sql += " AND status=?"
            params.append(status.value)
        sql += " ORDER BY created_at,id"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._approval(row) for row in rows]

    def is_authorized(self, approval_id: str, **binding: Any) -> bool:
        approval = self.get_approval(approval_id)
        return bool(
            approval.status is ApprovalStatus.APPROVED
            and approval.session_id == binding["session_id"]
            and approval.run_id == binding["run_id"]
            and approval.dataset_id == binding["dataset_id"]
            and approval.revision_id == binding["revision_id"]
            and approval.tool_name == binding["tool_name"]
            and approval.tool_version == binding["tool_version"]
            and approval.argument_hash == _digest(binding["arguments"])
        )

    @staticmethod
    def _approval(row) -> ApprovalRequest:
        item = dict(row)
        return ApprovalRequest(
            id=item["id"],
            project_id=item["project_id"],
            session_id=item["session_id"],
            run_id=item["run_id"],
            dataset_id=item["dataset_id"],
            revision_id=item["revision_id"],
            tool_name=item["tool_name"],
            tool_version=item["tool_version"],
            status=ApprovalStatus(item["status"]),
            risk=Risk(item["risk"]),
            proposed_arguments=json.loads(item["proposed_json"]),
            final_arguments=json.loads(item["final_json"]),
            argument_hash=item["argument_hash"],
            reasons=tuple(json.loads(item["reasons_json"])),
            decision=ApprovalDecision(item["decision"]) if item["decision"] else None,
            created_at=item["created_at"],
            decided_at=item["decided_at"],
        )

    def save_checkpoint(self, run_id: str, state: dict[str, Any], next_node: str) -> Checkpoint:
        state = validate_json_value(state)
        payload_hash = _digest({"state": state, "next_node": next_node})
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT COALESCE(MAX(sequence),0)+1 FROM safety_checkpoints
                WHERE project_id=? AND run_id=?""",
                (self.project_id, run_id),
            ).fetchone()
            sequence = int(row[0])
            checkpoint = Checkpoint(
                id=new_id(IdKind.CHECKPOINT),
                project_id=self.project_id,
                run_id=run_id,
                sequence=sequence,
                next_node=next_node,
                state=state,
                state_hash=payload_hash,
                created_at=utc_now_iso(),
            )
            connection.execute(
                "INSERT INTO safety_checkpoints VALUES (?,?,?,?,?,?,?,?)",
                (
                    checkpoint.id,
                    checkpoint.project_id,
                    checkpoint.run_id,
                    checkpoint.sequence,
                    checkpoint.next_node,
                    canonical_json(checkpoint.state),
                    checkpoint.state_hash,
                    checkpoint.created_at,
                ),
            )
            connection.commit()
        return checkpoint

    def load_checkpoint(self, run_id: str) -> Checkpoint:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM safety_checkpoints WHERE project_id=? AND run_id=?
                ORDER BY sequence DESC LIMIT 1""",
                (self.project_id, run_id),
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        item = dict(row)
        state = json.loads(item["state_json"])
        actual = _digest({"state": state, "next_node": item["next_node"]})
        if actual != item["state_hash"]:
            raise ValueError("Checkpoint hash mismatch; refusing recovery")
        return Checkpoint(
            id=item["id"],
            project_id=item["project_id"],
            run_id=item["run_id"],
            sequence=item["sequence"],
            next_node=item["next_node"],
            state=state,
            state_hash=item["state_hash"],
            created_at=item["created_at"],
        )

    def corrupt_checkpoint_for_test(self, checkpoint_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE safety_checkpoints SET state_hash='corrupt' WHERE id=?", (checkpoint_id,)
            )
            connection.commit()

    def cancel_run(self, run_id: str, reason: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO run_cancellations VALUES (?,?,?,?)",
                (self.project_id, run_id, reason, utc_now_iso()),
            )
            connection.commit()
            return cursor.rowcount == 1

    def is_cancelled(self, run_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM run_cancellations WHERE project_id=? AND run_id=?",
                (self.project_id, run_id),
            ).fetchone()
        return row is not None

    def completed_execution(self, execution_key: str) -> dict[str, Any] | None:
        """Return a durable completed result without reopening a side effect."""
        with self._connect() as connection:
            row = connection.execute(
                """SELECT result_json FROM safety_executions
                WHERE project_id=? AND idempotency_key=? AND status='completed'""",
                (self.project_id, execution_key),
            ).fetchone()
        return json.loads(row["result_json"]) if row is not None else None


class SafetyGate:
    """Single execution boundary for policy, approval, cancellation, and replay safety."""

    def __init__(self, store: SafetyStore, policy: RiskPolicy | None = None):
        self.store = store
        self.policy = policy or RiskPolicy()

    @staticmethod
    def execution_key(
        *,
        project_id: str,
        session_id: str,
        run_id: str,
        dataset_id: str,
        revision_id: str,
        tool_name: str,
        tool_version: str,
        arguments: dict[str, Any],
    ) -> str:
        return _digest(
            {
                "project": project_id,
                "session": session_id,
                "run": run_id,
                "dataset": dataset_id,
                "revision": revision_id,
                "tool": tool_name,
                "version": tool_version,
                "arguments": arguments,
            }
        )

    def inspect(
        self,
        *,
        session_id: str,
        run_id: str,
        dataset_id: str,
        revision_id: str,
        tool_name: str,
        tool_version: str,
        arguments: dict[str, Any],
        declared_risk: Risk,
        effect: Effect,
        previously_approved: dict[str, Any] | None = None,
        approval_id: str | None = None,
    ) -> GateOutcome:
        decision = self.policy.evaluate(
            declared_risk, effect, tool_name, arguments, previously_approved
        )
        execution_key = self.execution_key(
            project_id=self.store.project_id,
            session_id=session_id,
            run_id=run_id,
            dataset_id=dataset_id,
            revision_id=revision_id,
            tool_name=tool_name,
            tool_version=tool_version,
            arguments=arguments,
        )
        if self.store.is_cancelled(run_id):
            return GateOutcome(
                disposition=GateDisposition.CANCELLED,
                policy=decision,
                arguments=arguments,
                execution_key=execution_key,
            )
        if decision.denied:
            return GateOutcome(
                disposition=GateDisposition.DENIED,
                policy=decision,
                arguments=arguments,
                execution_key=execution_key,
            )
        if not decision.requires_approval:
            return GateOutcome(
                disposition=GateDisposition.AUTOMATIC,
                policy=decision,
                arguments=arguments,
                execution_key=execution_key,
            )
        if approval_id is not None:
            approval = self.store.get_approval(approval_id)
            binding = {
                "session_id": session_id,
                "run_id": run_id,
                "dataset_id": dataset_id,
                "revision_id": revision_id,
                "tool_name": tool_name,
                "tool_version": tool_version,
                "arguments": approval.final_arguments,
            }
            if self.store.is_authorized(approval_id, **binding):
                return GateOutcome(
                    disposition=GateDisposition.AUTHORIZED,
                    policy=decision,
                    approval=approval,
                    arguments=approval.final_arguments,
                    execution_key=self.execution_key(
                        project_id=self.store.project_id,
                        session_id=session_id,
                        run_id=run_id,
                        dataset_id=dataset_id,
                        revision_id=revision_id,
                        tool_name=tool_name,
                        tool_version=tool_version,
                        arguments=approval.final_arguments,
                    ),
                )
        approval = self.store.request_approval(
            session_id=session_id,
            run_id=run_id,
            dataset_id=dataset_id,
            revision_id=revision_id,
            tool_name=tool_name,
            tool_version=tool_version,
            arguments=arguments,
            risk=decision.effective_risk,
            reasons=decision.reasons,
        )
        return GateOutcome(
            disposition=GateDisposition.WAITING_APPROVAL,
            policy=decision,
            approval=approval,
            arguments=arguments,
            execution_key=execution_key,
        )

    def execute_once(
        self,
        outcome: GateOutcome,
        *,
        run_id: str,
        executor,
    ) -> dict[str, Any]:
        if outcome.disposition not in {GateDisposition.AUTOMATIC, GateDisposition.AUTHORIZED}:
            raise SafetyInterruption(outcome)
        if self.store.is_cancelled(run_id):
            raise PermissionError("Execution blocked: cancelled")
        approval_id = outcome.approval.id if outcome.approval else None
        key = outcome.execution_key
        with self.store._connect() as connection:
            existing = connection.execute(
                "SELECT status,result_json FROM safety_executions WHERE idempotency_key=?",
                (key,),
            ).fetchone()
            if existing is not None:
                if existing["status"] == "completed":
                    return json.loads(existing["result_json"])
                raise RecoveryRequired("Execution already claimed; refusing a duplicate side effect")
            connection.execute(
                "INSERT INTO safety_executions VALUES (?,?,?,?,?,?,?,?)",
                (
                    key,
                    self.store.project_id,
                    approval_id,
                    run_id,
                    "executing",
                    None,
                    utc_now_iso(),
                    None,
                ),
            )
            connection.commit()
        try:
            result = validate_json_value(executor(outcome.arguments))
        except Exception:
            with self.store._connect() as connection:
                connection.execute(
                    "UPDATE safety_executions SET status='failed' WHERE idempotency_key=?", (key,)
                )
                connection.commit()
            raise
        with self.store._connect() as connection:
            connection.execute(
                """UPDATE safety_executions SET status='completed', result_json=?, completed_at=?
                WHERE idempotency_key=?""",
                (canonical_json(result), utc_now_iso(), key),
            )
            connection.commit()
        return result


class CircuitOpenError(RuntimeError):
    pass


class RecoveryRequired(RuntimeError):
    """Execution ownership/commit outcome must be reconciled before any retry."""


class CircuitBreaker:
    def __init__(
        self, store: SafetyStore, service: str, *, threshold: int = 3, cooldown_seconds: int = 30
    ):
        self.store, self.service = store, service
        self.threshold, self.cooldown_seconds = threshold, cooldown_seconds

    def _record(self) -> tuple[str, int, float | None]:
        with self.store._connect() as connection:
            row = connection.execute(
                """SELECT state,failures,opened_at FROM circuit_breakers
                WHERE project_id=? AND service=?""",
                (self.store.project_id, self.service),
            ).fetchone()
        return (row["state"], row["failures"], row["opened_at"]) if row else ("closed", 0, None)

    def _save(self, state: str, failures: int, opened_at: float | None) -> None:
        with self.store._connect() as connection:
            connection.execute(
                """INSERT INTO circuit_breakers VALUES (?,?,?,?,?,?) ON CONFLICT(project_id,service)
                DO UPDATE SET state=excluded.state, failures=excluded.failures,
                opened_at=excluded.opened_at, updated_at=excluded.updated_at""",
                (self.store.project_id, self.service, state, failures, opened_at, utc_now_iso()),
            )
            connection.commit()

    def failure(self) -> None:
        _, failures, _ = self._record()
        failures += 1
        if failures >= self.threshold:
            self._save("open", failures, time.time())
        else:
            self._save("closed", failures, None)

    def success(self) -> None:
        self._save("closed", 0, None)

    def allow(self, *, now_offset_seconds: float = 0) -> str:
        state, failures, opened_at = self._record()
        if state == "open":
            now = time.time() + now_offset_seconds
            if opened_at is not None and now - opened_at > self.cooldown_seconds:
                self._save("half_open", failures, opened_at)
                return "half_open"
            raise CircuitOpenError(f"Circuit is open for {self.service}")
        return state


class OutboundGuard:
    ROOT_FIELDS = {"model", "messages", "tools", "tool_choice", "thinking", "temperature"}
    SECRET = re.compile(r"(?:sk-|api[_-]?key|bearer\s+)[A-Za-z0-9_.-]{6,}", re.I)
    PATH = re.compile(r"(?:[A-Za-z]:\\|/home/|/Users/|\.h5ad\b)", re.I)

    def validate(self, envelope: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_json_value(envelope)
        unknown = set(normalized) - self.ROOT_FIELDS
        if unknown:
            raise ValueError(f"Outbound fields are not allowlisted: {sorted(unknown)}")
        encoded = canonical_json(normalized)
        if self.SECRET.search(encoded) or self.PATH.search(encoded):
            raise ValueError("Outbound envelope contains a secret or local data reference")
        return normalized
