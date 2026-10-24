"""Deterministic, bounded JSON helpers for persistent protocol payloads."""

from __future__ import annotations

import json
import math
from typing import Any

MAX_PROTOCOL_BYTES = 256 * 1024


def validate_json_value(value: Any, *, path: str = "$") -> Any:
    """Reject values that cannot safely enter a small JSON protocol."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"JSON float at {path} must be finite")
        return value
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"JSON object key at {path} must be a string")
            normalized[key] = validate_json_value(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            validate_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(
        f"Unsupported protocol value at {path}: {type(value).__module__}.{type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    normalized = validate_json_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def ensure_payload_size(value: Any, *, maximum_bytes: int = MAX_PROTOCOL_BYTES) -> None:
    size = len(canonical_json(value).encode("utf-8"))
    if size > maximum_bytes:
        raise ValueError(f"Protocol payload exceeds {maximum_bytes} bytes")
