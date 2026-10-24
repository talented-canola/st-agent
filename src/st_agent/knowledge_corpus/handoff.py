"""Versioned, verified consumer projection. Never rewrite archived SourceSpec v1."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from .contracts import CorpusError, Record, Snapshot, SourceSpec, sha256
from .store import encoded

UNIT_POLICY_VERSION = "units-v1-exclude-api-templates"
COLLECTIONS = {
    "method": "method_knowledge",
    "workflow_reference": "method_knowledge",
    "scientific_evidence": "scientific_evidence",
    "ontology": "scientific_evidence",
}


class Applicability(Record):
    species: str | None = Field(default=None, min_length=1)
    tissue: str | None = Field(default=None, min_length=1)
    developmental_stage: str | None = Field(default=None, min_length=1)
    platform: str | None = Field(default=None, min_length=1)
    software: str | None = Field(default=None, min_length=1)
    software_version: str | None = Field(default=None, min_length=1)


class KnowledgeUnit(Record):
    unit_id: str
    source_key: str
    source_version: str
    source_snapshot_id: str
    title: str
    canonical_uri: str
    license: str
    role: Literal["method", "workflow_reference", "scientific_evidence", "ontology"]
    collection: Literal["method_knowledge", "scientific_evidence"]
    raw_sha256: str
    parsed_sha256: str
    parser_version: str
    content_sha256: str
    locator: dict[str, Any]
    applicability: Applicability
    condition_provenance: Literal["source_metadata_not_inferred"] = "source_metadata_not_inferred"
    kind: str
    text: str
    quality_warnings: tuple[str, ...] = ()
    eligible_for_index: bool

    @model_validator(mode="after")
    def consistent(self):
        if self.collection != COLLECTIONS[self.role]:
            raise ValueError("Source role and logical collection disagree")
        if self.content_sha256 != sha256(self.text.encode("utf-8")):
            raise ValueError("Original unit text checksum mismatch")
        if self.eligible_for_index != (not self.quality_warnings):
            raise ValueError("Quality warnings cannot be bypassed")
        return self


def quality_warnings(source_format, text):
    if source_format == "python_docstrings" and re.search(r"\{[a-zA-Z_]\w*\}|%\([^)]+\)s", text):
        return ("unexpanded_doc_template",)
    return ()


def project_release(store, release):
    release = Path(release)
    verification = store.verify_release(release)
    if not verification["passed"]:
        raise CorpusError("units.release_not_usable")
    manifest = json.loads(release.read_bytes())
    if manifest["release_sha256"] != verification["release_sha256"]:
        raise CorpusError("units.release_changed_during_read")
    units = []
    for entry in manifest["sources"]:
        source = SourceSpec.model_validate(entry["source"])
        snapshot = Snapshot.model_validate(entry["snapshot"])
        registered = store._row(source)
        if registered is None:
            raise CorpusError("units.catalog_registration_missing")
        if registered["spec"] != encoded(source.model_dump(mode="json")).decode("utf-8"):
            raise CorpusError("units.catalog_spec_mismatch")
        if registered["withdrawn_reason"]:
            raise CorpusError("units.release_not_usable")
        if store._snapshot(snapshot.snapshot_id) != snapshot:
            raise CorpusError("units.catalog_snapshot_mismatch")
        try:
            conditions = Applicability.model_validate(source.conditions)
        except ValidationError as error:
            raise CorpusError("units.invalid_conditions") from error
        document = store._verify_snapshot(source, snapshot)
        for block in document.blocks:
            warnings = quality_warnings(source.format, block.text)
            unit_id = sha256(
                encoded(
                    {
                        "source": source.source_key,
                        "version": source.version,
                        "raw": snapshot.raw_sha256,
                        "text": block.text_sha256,
                        "locator": block.locator,
                    }
                )
            )
            units.append(
                KnowledgeUnit(
                    unit_id=unit_id,
                    source_key=source.source_key,
                    source_version=source.version,
                    source_snapshot_id=snapshot.snapshot_id,
                    title=source.title,
                    canonical_uri=source.canonical_uri,
                    license=source.license,
                    role=source.role,
                    collection=COLLECTIONS[source.role],
                    raw_sha256=snapshot.raw_sha256,
                    parsed_sha256=snapshot.parsed_sha256,
                    parser_version=snapshot.parser_version,
                    content_sha256=block.text_sha256,
                    locator=block.locator,
                    applicability=conditions,
                    kind=block.kind,
                    text=block.text,
                    quality_warnings=warnings,
                    eligible_for_index=not warnings,
                )
            )
    return manifest, tuple(units)


def load_units(store, release, *, collection=None, eligible_only=True):
    """Both collections preserve original role; unknown conditions are never inferred."""
    if collection not in (None, "method_knowledge", "scientific_evidence"):
        raise CorpusError("units.invalid_collection")
    _, units = project_release(store, release)
    selected = tuple(
        u
        for u in units
        if (collection is None or u.collection == collection)
        and (not eligible_only or u.eligible_for_index)
    )
    store._emit(
        "units.loaded",
        collection=collection,
        eligible_only=eligible_only,
        archived_count=len(units),
        selected_count=len(selected),
        excluded_count=sum(not u.eligible_for_index for u in units),
    )
    return selected
