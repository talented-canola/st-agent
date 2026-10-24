"""Stable cache identity; cache storage is added by the execution repository."""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import Field, field_validator

from st_agent.contracts import StrictRecord
from st_agent.ids import IdKind, validate_id
from st_agent.serialization import canonical_json, validate_json_value

__all__ = ["CacheIdentity", "build_cache_key", "canonical_json"]


class CacheIdentity(StrictRecord):
    dataset_fingerprint: str = Field(min_length=1)
    input_revision_id: str | None
    tool_name: str = Field(min_length=1)
    tool_contract_version: str = Field(min_length=1)
    canonical_args: dict[str, Any]
    random_seed: int | None
    dependency_lock_hash: str = Field(min_length=1)
    upstream_artifact_ids: tuple[str, ...] = ()

    @field_validator("input_revision_id")
    @classmethod
    def _revision_id(cls, value: str | None) -> str | None:
        return validate_id(value, IdKind.REVISION) if value is not None else None

    @field_validator("canonical_args")
    @classmethod
    def _args_are_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_json_value(value)

    @field_validator("upstream_artifact_ids")
    @classmethod
    def _artifact_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_id(value, IdKind.ARTIFACT) for value in values)


def build_cache_key(identity: CacheIdentity) -> str:
    payload = identity.model_dump(mode="json")
    payload["upstream_artifact_ids"] = sorted(payload["upstream_artifact_ids"])
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
