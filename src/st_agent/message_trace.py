"""Bounded chat references to complete, immutable presentation evidence.

These snapshots are display/report views, never execution or recovery authority.
Recovery continues to use SafetyStore and its original checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import PurePosixPath

from st_agent.ids import IdKind, validate_id
from st_agent.serialization import MAX_PROTOCOL_BYTES, canonical_json, ensure_payload_size
from st_agent.trace import redact

SCHEMA = "conversation.trace/1"
_BUDGET = MAX_PROTOCOL_BYTES - 4096
_LOCK = threading.RLock()
_IDENTITIES = {"run_id": IdKind.RUN, "session_id": IdKind.SESSION, "dataset_id": IdKind.DATASET}


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _registered_run(copilot, run_id):
    with sqlite3.connect(
        copilot.layout.database_path.resolve().as_uri() + "?mode=ro", uri=True
    ) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT r.session_id,r.dataset_id,s.project_id AS session_project,"
            "d.project_id AS dataset_project FROM runs r LEFT JOIN sessions s ON s.id=r.session_id "
            "LEFT JOIN datasets d ON d.id=r.dataset_id WHERE r.id=?",
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    if any(
        row[key] not in (None, copilot.project.id) for key in ("session_project", "dataset_project")
    ):
        raise ValueError("message.trace_run_project_mismatch")
    return {"run_id": run_id, "session_id": row["session_id"], "dataset_id": row["dataset_id"]}


def _binding(copilot, full_state, compact_trace):
    sources = [full_state, compact_trace]
    nested = full_state.get("basic_task")
    if isinstance(nested, dict):
        sources.append(nested)

    def collect():
        bound = {"project_id": copilot.project.id}
        for key in ("project_id", *_IDENTITIES):
            values = {source[key] for source in sources if source.get(key) is not None}
            if len(values) > 1 or (
                key == "project_id" and values and values != {copilot.project.id}
            ):
                raise ValueError("message.trace_binding_mismatch")
            if values:
                bound[key] = values.pop()
        return bound

    bound = collect()
    registered = _registered_run(copilot, bound.get("run_id"))
    if registered:
        sources.append(registered)
        bound = collect()
    if any(key not in bound for key in _IDENTITIES):
        if "run_id" not in bound:
            raise ValueError("message.trace_binding_missing")
        from st_agent.safety import SafetyStore

        store = SafetyStore(
            copilot.layout.database_path,
            copilot.project.id,
            copilot.settings.sqlite_busy_timeout_ms,
        )
        try:
            checkpoint = store.load_checkpoint(bound["run_id"])
        except KeyError as error:
            raise ValueError("message.trace_session_binding_missing") from error
        for key in ("conversation", "basic_task"):
            source = checkpoint.state.get(key)
            if isinstance(source, dict):
                sources.append(source)
        bound = collect()
    for key, kind in _IDENTITIES.items():
        if not isinstance(bound.get(key), str):
            raise ValueError("message.trace_binding_missing")
        validate_id(bound[key], kind)
    if copilot.sessions.get(bound["session_id"]).project_id != copilot.project.id:
        raise ValueError("message.trace_session_project_mismatch")
    copilot.get_dataset(bound["dataset_id"])
    return bound


def _path(copilot, relative, digest):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("message.trace_path_invalid")
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or ".." in path.parts
        or relative != f"artifacts/conversation_trace/{digest}.json"
    ):
        raise ValueError("message.trace_path_invalid")
    target = (copilot.layout.root / relative).resolve()
    if not target.is_relative_to(copilot.layout.root.resolve()):
        raise ValueError("message.trace_path_outside_project")
    return target


def _check_record(copilot, record, binding, relative, digest):
    if (
        record.project_id != copilot.project.id
        or record.kind != "conversation.trace"
        or record.run_id not in (None, binding["run_id"])
        or record.relative_path != relative
        or record.sha256 != digest
        or record.metadata.get("schema") != SCHEMA
        or any(record.metadata.get(key) != value for key, value in binding.items())
    ):
        raise ValueError("message.trace_artifact_binding_mismatch")


def _head(full_state, binding):
    head = {**binding}
    for key in (
        "error_code",
        "explanation_error",
        "status",
        "stage",
        "boundary",
        "trace_degraded",
        "tool_replayed",
        "approval_id",
        "failure",
    ):
        value = full_state.get(key)
        if value is None or type(value) in (str, int, float, bool):
            if key in full_state and len(canonical_json(value).encode("utf-8")) <= 512:
                head[key] = value
        elif key == "failure" and isinstance(value, dict):
            head[key] = {
                name: child
                for name, child in value.items()
                if name in {"error_code", "type", "status", "stage", "boundary", "retryable"}
                and (child is None or type(child) in (str, int, float, bool))
                and len(canonical_json(child).encode("utf-8")) <= 512
            }
    return head


def pack_message_trace(copilot, full_state, compact_trace) -> dict:
    """Return an inline trace or a bounded reference, preserving full evidence.

    Missing identity fields may come from the hash-checked SafetyStore checkpoint.
    No session is guessed, and neither input nor the checkpoint is changed.
    """
    if len(canonical_json({"trace": compact_trace}).encode("utf-8")) <= _BUDGET:
        return compact_trace
    binding = _binding(copilot, full_state, compact_trace)
    snapshot = {"schema": SCHEMA, "binding": binding, "state": full_state}
    safe = json.loads(copilot.settings.redact(canonical_json(redact(snapshot))))
    raw = canonical_json(safe).encode("utf-8")
    digest = _digest(raw)
    relative = f"artifacts/conversation_trace/{digest}.json"
    target = _path(copilot, relative, digest)
    with _LOCK:
        records = [
            record for record in copilot.list_artifacts() if record.relative_path == relative
        ]
        if len(records) > 1:
            raise ValueError("message.trace_artifact_ambiguous")
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError("message.trace_immutable_content_mismatch")
        else:
            if records:
                raise ValueError("message.trace_registered_file_missing")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
        if records:
            record = records[0]
            _check_record(copilot, record, binding, relative, digest)
            if record.size_bytes != len(raw):
                raise ValueError("message.trace_artifact_size_mismatch")
        else:
            # Some presentation-only runs have no row in the execution runs table.
            # Bind them in immutable metadata without inventing an execution row.
            run_id = binding["run_id"] if _registered_run(copilot, binding["run_id"]) else None
            record = copilot.register_artifact(
                target,
                kind="conversation.trace",
                run_id=run_id,
                metadata={
                    "schema": SCHEMA,
                    **binding,
                    "purpose": "presentation_evidence_not_execution_checkpoint",
                },
            )
    head = _head(safe["state"], binding)
    head["state_ref"] = {"artifact_id": record.id, "relative_path": relative, "sha256": digest}
    ensure_payload_size({"trace": head}, maximum_bytes=_BUDGET)
    return head


def load_message_trace(copilot, message) -> dict:
    """Read a registered presentation view after checking all identity bindings."""
    trace = message.metadata.get("trace", {})
    if not isinstance(trace, dict):
        raise ValueError("message.trace_invalid")
    if "state_ref" not in trace:
        return trace
    ref = trace["state_ref"]
    if not isinstance(ref, dict) or not all(
        isinstance(ref.get(key), str) for key in ("artifact_id", "relative_path", "sha256")
    ):
        raise ValueError("message.trace_reference_invalid")
    validate_id(ref["artifact_id"], IdKind.ARTIFACT)
    binding = _binding(copilot, trace, trace)
    if message.session_id != binding["session_id"]:
        raise ValueError("message.trace_message_session_mismatch")
    record = copilot.artifacts.get(ref["artifact_id"])
    _check_record(copilot, record, binding, ref["relative_path"], ref["sha256"])
    target = _path(copilot, ref["relative_path"], ref["sha256"])
    raw = target.read_bytes()
    if len(raw) != record.size_bytes or _digest(raw) != record.sha256:
        raise ValueError("message.trace_artifact_integrity_mismatch")
    snapshot = json.loads(raw)
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("schema") != SCHEMA
        or snapshot.get("binding") != binding
        or not isinstance(snapshot.get("state"), dict)
    ):
        raise ValueError("message.trace_snapshot_schema_or_binding_mismatch")
    state = snapshot["state"]
    if any(state.get(key, value) != value for key, value in binding.items()):
        raise ValueError("message.trace_snapshot_state_binding_mismatch")
    return state
