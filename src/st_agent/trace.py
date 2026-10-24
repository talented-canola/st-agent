"""Append-only, parent-process telemetry. Never an authority for execution."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path


def redact(value):
    if isinstance(value, dict):
        return {str(k): ("[REDACTED]" if any(s in str(k).lower() for s in
                ("secret", "api_key", "authorization", "password", "token_value"))
                else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        value = re.sub(
            r"(?i)Bearer\s+\S+|\bsk-[\w-]+|\bhf_[A-Za-z0-9]{10,}\b",
            "[REDACTED]", value,
        )
        value = re.sub(r"(?<![\w/])[A-Za-z]:[\\/][^\s\"'<>]*", "[ABS_PATH]", value)
        return re.sub(
            r"(?i)\b(?:api[_-]?key|password|secret|token)\s*[:=]\s*\S+",
            "[REDACTED]", value,
        )
    return value


class Trace:
    def __init__(self, directory, *, sink=None):
        self.directory = Path(directory)
        self.sink = sink
        self.context = {}
        self.degraded = False
        self.lock = threading.RLock()
        self.path = self.directory / "trace.jsonl"
        self.root_span = uuid.uuid5(uuid.NAMESPACE_URL, str(self.directory.resolve())).hex
        self.seq = 0
        try:
            if self.path.exists():
                with self.path.open(encoding="utf-8") as stream:
                    self.seq = sum(1 for _ in stream)
        except OSError:
            self._degrade()

    def _degrade(self):
        if not self.degraded:
            logging.getLogger(__name__).warning("trace_degraded: evidence write unavailable")
        self.degraded = True

    def emit(self, event, *, status="ok", error_code=None, duration_ms=None, **payload):
        with self.lock:
            self.seq += 1
            event_id = uuid.uuid4().hex
            row = dict.fromkeys(("task_id", "run_id", "step_id", "attempt_id", "turn_id"))
            row.update(self.context)
            row.update(schema_version=1, event_id=event_id, seq=self.seq,
                       ts=datetime.now(UTC).isoformat(), trace_id=self.directory.name,
                       span_id=(uuid.uuid5(uuid.NAMESPACE_URL, self.root_span + row["step_id"]).hex
                                if row.get("step_id") else self.root_span),
                       parent_span_id=self.root_span if row.get("step_id") else None,
                       event=event, status=status, error_code=error_code,
                       duration_ms=duration_ms, payload_ref=None, payload_sha256=None)
            try:
                safe = redact(payload)
                self.directory.mkdir(parents=True, exist_ok=True)
                if safe:
                    encoded = json.dumps(safe, ensure_ascii=False, sort_keys=True).encode("utf-8")
                    relative = f"payloads/{event_id}.json"
                    target = self.directory / relative
                    target.parent.mkdir(exist_ok=True)
                    with target.open("xb") as stream:
                        stream.write(encoded)
                        stream.flush()
                    row.update(payload_ref=relative,
                               payload_sha256=hashlib.sha256(encoded).hexdigest())
                if self.sink is not None:
                    from st_agent.events import EventDraft
                    try:
                        saved = self.sink.append(EventDraft(event_type="trace." + event,
                            run_id=row.get("run_id"), payload={"trace_event_id": event_id,
                                "seq": self.seq, "status": status, "error_code": error_code}))
                        row["database_event_id"] = saved.id
                    except Exception:
                        self._degrade()
                        row["trace_degraded"] = True
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
            except Exception:
                self._degrade()
            return row


def emit(owner, event, **payload):
    trace = getattr(owner, "trace", None)
    if trace is not None:
        trace.emit(event, **payload)


def attach(copilot, directory=None):
    if directory is None:
        directory = copilot.layout.root / "artifacts" / "traces" / uuid.uuid4().hex
    trace = Trace(directory, sink=copilot.events)
    copilot.trace = trace
    copilot.worker_manager.trace = trace
    return trace
