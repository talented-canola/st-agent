"""Opaque ID generation and validation for persistent Agent entities."""

from __future__ import annotations

import re
import uuid
from enum import StrEnum


class IdKind(StrEnum):
    PROJECT = "prj"
    DATASET = "dset"
    REVISION = "rev"
    SESSION = "sess"
    MESSAGE = "msg"
    RUN = "run"
    STEP = "step"
    TOOL_CALL = "call"
    APPROVAL = "appr"
    ARTIFACT = "art"
    EVENT = "evt"
    JOB = "job"
    KNOWLEDGE_SOURCE = "ksrc"
    KNOWLEDGE_CHUNK = "kchk"
    RETRIEVAL = "ret"
    EVIDENCE = "evd"
    CLAIM = "claim"
    MEMORY = "mem"
    CHECKPOINT = "ckpt"


class InvalidIdError(ValueError):
    """Raised when an ID is malformed or has the wrong entity prefix."""


_PATTERN = re.compile(r"^(?P<prefix>[a-z][a-z0-9]*)_(?P<value>[0-9a-f]{32})$")


def new_id(kind: IdKind) -> str:
    """Generate a random, opaque ID with a human-auditable entity prefix."""
    return f"{kind.value}_{uuid.uuid4().hex}"


def validate_id(value: str, kind: IdKind | None = None) -> str:
    """Validate syntax and, when supplied, the expected entity kind."""
    match = _PATTERN.fullmatch(value)
    if match is None:
        raise InvalidIdError(f"Malformed persistent ID: {value!r}")
    if kind is not None and match.group("prefix") != kind.value:
        raise InvalidIdError(f"Expected {kind.value}_ ID, received {value!r}")
    return value
