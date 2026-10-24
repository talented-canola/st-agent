"""Small event interface and SQLite-backed implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from st_agent.db import connect_database
from st_agent.ids import IdKind, new_id
from st_agent.utils import utc_now_iso


class EventDraft(BaseModel):
    """Caller-provided event content before persistent fields are assigned."""

    event_type: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    run_id: str | None = None


class EventRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    project_id: str
    event_type: str
    payload: dict[str, Any]
    session_id: str | None = None
    run_id: str | None = None
    created_at: str


class EventSink(Protocol):
    def append(self, draft: EventDraft) -> EventRecord:
        """Persist one event and return its assigned identity."""


class SQLiteEventRepository:
    def __init__(self, database_path: str | Path, project_id: str, *, busy_timeout_ms: int):
        self.database_path = Path(database_path)
        self.project_id = project_id
        self.busy_timeout_ms = busy_timeout_ms

    def append(self, draft: EventDraft) -> EventRecord:
        record = EventRecord(
            id=new_id(IdKind.EVENT),
            project_id=self.project_id,
            event_type=draft.event_type,
            payload=draft.payload,
            session_id=draft.session_id,
            run_id=draft.run_id,
            created_at=utc_now_iso(),
        )
        with connect_database(
            self.database_path, busy_timeout_ms=self.busy_timeout_ms
        ) as connection:
            connection.execute(
                """
                INSERT INTO events(
                    id, project_id, event_type, session_id, run_id, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.project_id,
                    record.event_type,
                    record.session_id,
                    record.run_id,
                    json.dumps(record.payload, ensure_ascii=False, sort_keys=True),
                    record.created_at,
                ),
            )
            connection.commit()
        return record

    def get(self, event_id: str) -> EventRecord:
        with connect_database(
            self.database_path, busy_timeout_ms=self.busy_timeout_ms
        ) as connection:
            row = connection.execute(
                "SELECT * FROM events WHERE id = ? AND project_id = ?",
                (event_id, self.project_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown event: {event_id}")
        return EventRecord(
            id=row["id"],
            project_id=row["project_id"],
            event_type=row["event_type"],
            payload=json.loads(row["payload_json"]),
            session_id=row["session_id"],
            run_id=row["run_id"],
            created_at=row["created_at"],
        )

    def list(self, *, limit: int = 100) -> list[EventRecord]:
        with connect_database(
            self.database_path, busy_timeout_ms=self.busy_timeout_ms
        ) as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE project_id = ? ORDER BY created_at, id LIMIT ?",
                (self.project_id, limit),
            ).fetchall()
        return [
            EventRecord(
                id=row["id"],
                project_id=row["project_id"],
                event_type=row["event_type"],
                payload=json.loads(row["payload_json"]),
                session_id=row["session_id"],
                run_id=row["run_id"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
