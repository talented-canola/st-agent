"""Contracts distinguish identity, provenance, permission and scientific applicability."""

from __future__ import annotations

import hashlib
import ipaddress
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class CorpusError(ValueError):
    """Stable boundary code, without raw HTTP responses or credentials."""


def public_https(value: str) -> str:
    url = urlsplit(value)
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or url.port not in (None, 443)
        or url.query
        or url.fragment
    ):
        raise ValueError("Public HTTPS source URL without credentials/query required")
    if url.hostname == "localhost" or url.hostname.endswith((".local", ".localhost")):
        raise ValueError("Local addresses cannot be corpus sources")
    try:
        address = ipaddress.ip_address(url.hostname)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError("Private addresses cannot be corpus sources")
    return value


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceSpec(Record):
    source_key: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,119}$")
    version: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    canonical_uri: str
    fetch_url: str
    format: Literal["python_docstrings", "jats", "markdown", "html", "tsv", "obo"]
    role: Literal["method", "scientific_evidence", "workflow_reference", "ontology"]
    origin: Literal["external_original"] = "external_original"
    selectors: tuple[str, ...] = ()
    html_selector: str = "main"
    license: Literal["BSD-3-Clause", "MIT", "Apache-2.0", "CC-BY-4.0", "CC-BY-3.0", "CC0-1.0"]
    license_url: str
    license_marker: str = Field(min_length=8)
    expected_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    doi: str | None = None
    pmcid: str | None = Field(default=None, pattern=r"^PMC[0-9]+$")
    conditions: dict[str, str | None] = Field(default_factory=dict)
    topics: tuple[str, ...] = ()

    _urls = field_validator("canonical_uri", "fetch_url", "license_url")(public_https)

    @model_validator(mode="after")
    def license_matches_marker(self):
        markers = {
            "BSD-3-Clause": {"BSD 3-Clause License"},
            "MIT": {"MIT License"},
            "Apache-2.0": {"Apache License, Version 2.0"},
            "CC-BY-4.0": {"creativecommons.org/licenses/by/4.0"},
            "CC-BY-3.0": {"creativecommons.org/licenses/by/3.0"},
            "CC0-1.0": {"creativecommons.org/publicdomain/zero/1.0"},
        }
        if self.license_marker not in markers[self.license]:
            raise ValueError("Permission marker does not match the declared license")
        return self

    @field_validator("version")
    @classmethod
    def fixed_version(cls, value):
        if value.lower() in {"latest", "stable", "main", "master"}:
            raise ValueError("A fixed source version is required")
        return value


class ParsedBlock(Record):
    text: str = Field(min_length=1)
    kind: str
    locator: dict[str, Any]

    @property
    def text_sha256(self):
        return sha256(self.text.encode("utf-8"))


class ParsedDocument(Record):
    parser_version: str
    blocks: tuple[ParsedBlock, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)


class Snapshot(Record):
    snapshot_id: str
    source_key: str
    source_version: str
    spec_sha256: str
    raw_sha256: str
    raw_ref: str
    license_sha256: str
    license_ref: str
    parsed_sha256: str
    parsed_ref: str
    parser_version: str
    block_count: int
