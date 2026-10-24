"""Project-isolated working, task, user, and procedural memory."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field

from st_agent.contracts import StrictRecord
from st_agent.db import connect_database
from st_agent.ids import IdKind, new_id
from st_agent.serialization import canonical_json, validate_json_value
from st_agent.utils import utc_now_iso


class MemoryStatus(StrEnum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"
    FORGOTTEN = "forgotten"


class MemoryPreference(StrictRecord):
    id: str
    project_id: str
    key: str
    value: Any
    status: MemoryStatus
    scope_dataset_id: str | None = None
    source_session_id: str
    reason: str
    created_at: str
    updated_at: str


class TaskDigest(StrictRecord):
    id: str
    project_id: str
    session_id: str
    run_id: str
    dataset_id: str | None = None
    goal: str
    completed: tuple[str, ...]
    pending: tuple[str, ...]
    failed: tuple[str, ...]
    facts: tuple[dict[str, Any], ...]
    created_at: str
    updated_at: str


class WorkingMemory(StrictRecord):
    current_instruction: str
    task_digest: TaskDigest | None = None
    recent_messages: tuple[dict[str, str], ...] = ()
    verified_facts: tuple[dict[str, Any], ...] = ()
    preferences: dict[str, Any] = Field(default_factory=dict)
    estimated_tokens: int = Field(ge=0)


class ContextBudget:
    def __init__(self, max_tokens: int = 8_000):
        if max_tokens < 256:
            raise ValueError("Context budget must be at least 256 tokens")
        self.max_tokens = max_tokens

    @staticmethod
    def estimate(value: Any) -> int:
        return max(1, len(canonical_json(value)) // 4)

    def build(
        self,
        *,
        current_instruction: str,
        task_digest: TaskDigest | None,
        recent_messages: list[dict[str, str]],
        verified_facts: list[dict[str, Any]],
        preferences: dict[str, Any],
    ) -> WorkingMemory:
        # Current instruction is mandatory and intentionally precedes remembered preferences.
        base = {"current_instruction": current_instruction, "preferences": preferences}
        if self.estimate(base) > self.max_tokens:
            preferences = {}
        kept_facts: list[dict[str, Any]] = []
        kept_messages: list[dict[str, str]] = []
        for fact in reversed(verified_facts):
            candidate = [fact, *kept_facts]
            if self.estimate({**base, "facts": candidate}) <= self.max_tokens:
                kept_facts = candidate
        for message in reversed(recent_messages):
            candidate = [message, *kept_messages]
            payload = {**base, "facts": kept_facts, "messages": candidate}
            if self.estimate(payload) <= self.max_tokens:
                kept_messages = candidate
        digest = task_digest
        payload = {
            **base,
            "facts": kept_facts,
            "messages": kept_messages,
            "digest": digest.model_dump(mode="json") if digest else None,
        }
        if self.estimate(payload) > self.max_tokens:
            digest = None
            payload["digest"] = None
        return WorkingMemory(
            current_instruction=current_instruction,
            task_digest=digest,
            recent_messages=tuple(kept_messages),
            verified_facts=tuple(kept_facts),
            preferences=preferences,
            estimated_tokens=self.estimate(payload),
        )


class MemoryStore:
    def __init__(self, database_path: str | Path, project_id: str):
        self.database_path = Path(database_path)
        self.project_id = project_id

    def _connect(self):
        return connect_database(self.database_path)

    def save_digest(
        self,
        *,
        session_id: str,
        run_id: str,
        goal: str,
        steps: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        dataset_id: str | None = None,
    ) -> TaskDigest:
        categories = {"completed": [], "pending": [], "failed": []}
        for step in steps:
            status = step.get("status")
            if status in categories and isinstance(step.get("label"), str):
                categories[status].append(step["label"])
        verified = [validate_json_value(fact) for fact in facts if fact.get("verified") is True]
        now = utc_now_iso()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id,created_at FROM task_digests WHERE project_id=? AND run_id=?",
                (self.project_id, run_id),
            ).fetchone()
            digest_id = row["id"] if row else new_id(IdKind.MEMORY)
            created_at = row["created_at"] if row else now
            connection.execute(
                """INSERT INTO task_digests VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(project_id,run_id) DO UPDATE SET
                goal=excluded.goal,completed_json=excluded.completed_json,
                pending_json=excluded.pending_json,failed_json=excluded.failed_json,
                facts_json=excluded.facts_json,updated_at=excluded.updated_at""",
                (
                    digest_id,
                    self.project_id,
                    session_id,
                    run_id,
                    dataset_id,
                    goal,
                    canonical_json(categories["completed"]),
                    canonical_json(categories["pending"]),
                    canonical_json(categories["failed"]),
                    canonical_json(verified),
                    created_at,
                    now,
                ),
            )
            connection.commit()
        return self.get_digest(run_id)

    def get_digest(self, run_id: str) -> TaskDigest:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM task_digests WHERE project_id=? AND run_id=?",
                (self.project_id, run_id),
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return TaskDigest(
            id=row["id"],
            project_id=row["project_id"],
            session_id=row["session_id"],
            run_id=row["run_id"],
            dataset_id=row["dataset_id"],
            goal=row["goal"],
            completed=tuple(json.loads(row["completed_json"])),
            pending=tuple(json.loads(row["pending_json"])),
            failed=tuple(json.loads(row["failed_json"])),
            facts=tuple(json.loads(row["facts_json"])),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def propose(
        self,
        key: str,
        value: Any,
        *,
        source_session_id: str,
        reason: str,
        dataset_id: str | None = None,
    ) -> MemoryPreference:
        value = validate_json_value(value)
        now = utc_now_iso()
        record_id = new_id(IdKind.MEMORY)
        with self._connect() as connection:
            connection.execute(
                """UPDATE memory_preferences SET status='rejected',updated_at=?
                WHERE project_id=? AND key=? AND scope_dataset_id IS ? AND status='candidate'""",
                (now, self.project_id, key, dataset_id),
            )
            connection.execute(
                "INSERT INTO memory_preferences VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    record_id,
                    self.project_id,
                    key,
                    canonical_json(value),
                    MemoryStatus.CANDIDATE.value,
                    dataset_id,
                    source_session_id,
                    reason,
                    now,
                    now,
                ),
            )
            connection.commit()
        return self.get(record_id)

    def decide(self, memory_id: str, decision: str, edited_value: Any = None) -> MemoryPreference:
        current = self.get(memory_id)
        if current.status is not MemoryStatus.CANDIDATE:
            return current
        if decision not in {"approve", "reject", "edit_and_approve"}:
            raise ValueError("Unsupported memory decision")
        value = current.value
        if decision == "edit_and_approve":
            if edited_value is None:
                raise ValueError("Edited memory requires a value")
            value = validate_json_value(edited_value)
        elif edited_value is not None:
            raise ValueError("Edited value requires edit_and_approve")
        status = MemoryStatus.REJECTED if decision == "reject" else MemoryStatus.APPROVED
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if status is MemoryStatus.APPROVED:
                connection.execute(
                    """UPDATE memory_preferences SET status='forgotten',updated_at=?
                    WHERE project_id=? AND key=? AND scope_dataset_id IS ? AND status='approved'""",
                    (now, self.project_id, current.key, current.scope_dataset_id),
                )
            connection.execute(
                "UPDATE memory_preferences SET status=?,value_json=?,updated_at=? WHERE id=?",
                (status.value, canonical_json(value), now, memory_id),
            )
            connection.commit()
        return self.get(memory_id)

    def get(self, memory_id: str) -> MemoryPreference:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_preferences WHERE id=? AND project_id=?",
                (memory_id, self.project_id),
            ).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return self._preference(row)

    def list(self, *, include_inactive: bool = False) -> list[MemoryPreference]:
        sql = "SELECT * FROM memory_preferences WHERE project_id=?"
        params: list[Any] = [self.project_id]
        if not include_inactive:
            sql += " AND status='approved'"
        sql += " ORDER BY updated_at DESC,id"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._preference(row) for row in rows]

    def forget(self, memory_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE memory_preferences SET status='forgotten',updated_at=?
                WHERE id=? AND project_id=? AND status!='forgotten'""",
                (utc_now_iso(), memory_id, self.project_id),
            )
            connection.commit()
        return cursor.rowcount == 1

    def resolved_preferences(
        self, *, dataset_id: str | None = None, current: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM memory_preferences
                WHERE project_id=? AND status='approved'
                AND (scope_dataset_id IS NULL OR scope_dataset_id IS ?)
                ORDER BY CASE WHEN scope_dataset_id IS NULL THEN 0 ELSE 1 END,updated_at""",
                (self.project_id, dataset_id),
            ).fetchall()
        result = {row["key"]: json.loads(row["value_json"]) for row in rows}
        result.update(current or {})
        return result

    def register_procedure(self, name: str, version: str, instructions: list[str]) -> str:
        normalized = validate_json_value(instructions)
        digest = hashlib.sha256(canonical_json(normalized).encode()).hexdigest()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT content_sha256 FROM procedural_memory WHERE name=? AND version=?",
                (name, version),
            ).fetchone()
            if existing:
                return str(existing["content_sha256"])
            connection.execute(
                "INSERT OR IGNORE INTO procedural_memory VALUES (?,?,?,?,?)",
                (name, version, canonical_json(normalized), digest, utc_now_iso()),
            )
            connection.commit()
        return digest

    @staticmethod
    def _preference(row) -> MemoryPreference:
        return MemoryPreference(
            id=row["id"],
            project_id=row["project_id"],
            key=row["key"],
            value=json.loads(row["value_json"]),
            status=MemoryStatus(row["status"]),
            scope_dataset_id=row["scope_dataset_id"],
            source_session_id=row["source_session_id"],
            reason=row["reason"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
