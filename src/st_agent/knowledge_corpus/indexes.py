"""Immutable, model-bound index artifacts; retrieval belongs to component two."""

from __future__ import annotations

import json
import math
import numbers
import sqlite3
import tempfile
from pathlib import Path

from pydantic import Field, field_validator

from .contracts import CorpusError, Record, sha256
from .handoff import UNIT_POLICY_VERSION, project_release
from .store import encoded


class ChunkerSpec(Record):
    name: str = "original-block-char-window"
    version: str = Field(default="1", min_length=1)
    max_chars: int = Field(default=1200, ge=16, le=20000, strict=True)


class EmbeddingSpec(Record):
    model_name: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    encoding_version: str = Field(min_length=1)
    dimension: int = Field(gt=0, le=65536, strict=True)
    normalized: bool = True

    @field_validator("revision")
    @classmethod
    def pinned(cls, value):
        if value.strip().lower() in {"latest", "main", "master", "stable"}:
            raise ValueError("Embedding revision must be fixed")
        return value


class CacheIdentity(Record):
    release_sha256: str
    units_sha256: str
    parser_versions: tuple[str, ...]
    unit_policy_version: str = UNIT_POLICY_VERSION
    chunker: ChunkerSpec
    embedding: EmbeddingSpec | None
    backend: str = "sqlite-fts5-unicode61-" + sqlite3.sqlite_version

    @property
    def cache_key(self):
        return sha256(encoded(self.model_dump(mode="json")))


class IndexArtifact(Record):
    cache_key: str
    identity: CacheIdentity
    database_ref: str
    database_sha256: str
    units_ref: str
    units_sha256: str
    chunk_count: int
    archived_unit_count: int
    eligible_unit_count: int
    excluded_unit_count: int
    collections: dict[str, dict[str, int]]
    reused: bool = False


def split_units(units, chunker):
    if chunker.name != "original-block-char-window" or chunker.version != "1":
        raise CorpusError("index.unsupported_chunker")
    chunks = []
    for unit in units:
        if not unit.eligible_for_index:
            continue
        for start in range(0, len(unit.text), chunker.max_chars):
            text = unit.text[start : start + chunker.max_chars]
            identity = {
                "unit": unit.unit_id,
                "start": start,
                "end": start + len(text),
                "text": sha256(text.encode("utf-8")),
                "chunker": chunker.model_dump(),
            }
            chunks.append(
                {
                    "chunk_id": sha256(encoded(identity)),
                    "unit_id": unit.unit_id,
                    "text": text,
                    "text_sha256": identity["text"],
                    "char_start": start,
                    "char_end": start + len(text),
                }
            )
    return chunks


def encoder_identity(embedder):
    if embedder is None:
        return None
    identity = getattr(embedder, "identity", None)
    if not isinstance(identity, EmbeddingSpec):
        raise CorpusError("embedding.identity_required")
    return identity


class IndexCache:
    def __init__(self, store):
        self.store = store

    def prepare(self, release, *, chunker=None, embedder=None):
        chunker = chunker or ChunkerSpec()
        embedding = encoder_identity(embedder)
        manifest, units = project_release(self.store, Path(release))
        eligible = tuple(u for u in units if u.eligible_for_index)
        if not eligible:
            raise CorpusError("index.no_eligible_units")
        original = encoded(
            {
                "unit_policy_version": UNIT_POLICY_VERSION,
                "units": [u.model_dump(mode="json") for u in units],
            }
        )
        identity = CacheIdentity(
            release_sha256=manifest["release_sha256"],
            units_sha256=sha256(original),
            parser_versions=tuple(sorted({u.parser_version for u in units})),
            chunker=chunker,
            embedding=embedding,
        )
        key = identity.cache_key
        relative = "indexes/manifests/" + key + ".json"
        chunks = split_units(units, chunker)
        if self.store._path(relative).exists():
            saved = json.loads(self.store._path(relative).read_bytes())
            if saved.pop("descriptor_sha256") != sha256(encoded(saved)):
                raise CorpusError("index.descriptor_checksum_mismatch")
            artifact = IndexArtifact.model_validate(saved)
            if artifact.identity != identity or artifact.cache_key != key:
                raise CorpusError("index.identity_mismatch")
            self._verify(artifact, chunks, eligible)
            self.store._emit(
                "index.reused",
                cache_key=key,
                chunk_count=len(chunks),
                embedding_revision=embedding.revision if embedding else None,
            )
            return artifact.model_copy(update={"reused": True})
        vectors = embedder.encode([c["text"] for c in chunks]) if embedder else [None] * len(chunks)
        if encoder_identity(embedder) != embedding:
            raise CorpusError("embedding.identity_changed_during_encode")
        vectors = self._vectors(vectors, len(chunks), embedding)
        units_ref = self.store._put("units/" + sha256(original) + ".json", original)
        database = self._build_database(identity, chunks, vectors, eligible)
        digest = sha256(database)
        database_ref = self.store._put("indexes/data/" + digest + ".sqlite", database)
        collections = {
            c: {
                "archived": sum(u.collection == c for u in units),
                "eligible": sum(u.collection == c for u in eligible),
            }
            for c in ("method_knowledge", "scientific_evidence")
        }
        artifact = IndexArtifact(
            cache_key=key,
            identity=identity,
            database_ref=database_ref,
            database_sha256=digest,
            units_ref=units_ref,
            units_sha256=sha256(original),
            chunk_count=len(chunks),
            archived_unit_count=len(units),
            eligible_unit_count=len(eligible),
            excluded_unit_count=len(units) - len(eligible),
            collections=collections,
        )
        self._verify(artifact, chunks, eligible)
        payload = artifact.model_dump(mode="json")
        payload["descriptor_sha256"] = sha256(encoded(payload))
        self.store._put(relative, encoded(payload))
        self.store._emit(
            "index.published",
            cache_key=key,
            chunk_count=len(chunks),
            excluded_count=artifact.excluded_unit_count,
            embedding_revision=embedding.revision if embedding else None,
        )
        return artifact

    @staticmethod
    def _vectors(vectors, count, identity):
        if vectors is None or not hasattr(vectors, "__len__"):
            raise CorpusError("embedding.invalid_count")
        if len(vectors) != count:
            raise CorpusError("embedding.invalid_count")
        if identity is None:
            if any(vector is not None for vector in vectors):
                raise CorpusError("embedding.invalid_unexpected_dense_vector")
            return [None] * count
        result = []
        for vector in vectors:
            if vector is None or not hasattr(vector, "__len__"):
                raise CorpusError("embedding.invalid_vector")
            if len(vector) != identity.dimension or any(
                isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v)
                for v in vector
            ):
                raise CorpusError("embedding.invalid_vector")
            floats = [float(v) for v in vector]
            if identity.normalized and not math.isclose(
                sum(v * v for v in floats), 1.0, abs_tol=0.001
            ):
                raise CorpusError("embedding.invalid_normalization")
            result.append(floats)
        return result

    def _build_database(self, identity, chunks, vectors, units):
        folder = self.store.root / "indexes"
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, suffix=".sqlite", delete=False) as stream:
            temporary = Path(stream.name)
        db = sqlite3.connect(temporary)
        try:
            db.executescript("""
                CREATE TABLE metadata(identity TEXT NOT NULL);
                CREATE TABLE units(unit_id TEXT PRIMARY KEY, collection TEXT NOT NULL,
                                   role TEXT NOT NULL, record TEXT NOT NULL);
                CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, unit_id TEXT NOT NULL,
                                    record TEXT NOT NULL, embedding_json TEXT);
                CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED,text,
                                                          tokenize='unicode61');
            """)
            db.execute(
                "INSERT INTO metadata VALUES (?)", (encoded(identity.model_dump()).decode(),)
            )
            for unit in units:
                db.execute(
                    "INSERT INTO units VALUES (?,?,?,?)",
                    (
                        unit.unit_id,
                        unit.collection,
                        unit.role,
                        encoded(unit.model_dump(mode="json")).decode(),
                    ),
                )
            for chunk, vector in zip(chunks, vectors, strict=True):
                db.execute(
                    "INSERT INTO chunks VALUES (?,?,?,?)",
                    (
                        chunk["chunk_id"],
                        chunk["unit_id"],
                        encoded(chunk).decode(),
                        encoded(vector).decode() if vector else None,
                    ),
                )
                db.execute(
                    "INSERT INTO chunks_fts VALUES (?,?)", (chunk["chunk_id"], chunk["text"])
                )
            db.commit()
            db.close()
            return temporary.read_bytes()
        finally:
            db.close()
            temporary.unlink()

    def _verify(self, artifact, chunks, units):
        if artifact.units_sha256 != artifact.identity.units_sha256:
            raise CorpusError("index.units_identity_mismatch")
        original = json.loads(self.store._check_bytes(artifact.units_ref, artifact.units_sha256))
        archived = original["units"]
        expected_collections = {
            name: {
                "archived": sum(u["collection"] == name for u in archived),
                "eligible": sum(u.collection == name for u in units),
            }
            for name in ("method_knowledge", "scientific_evidence")
        }
        if (
            original["unit_policy_version"] != artifact.identity.unit_policy_version
            or artifact.archived_unit_count != len(archived)
            or artifact.eligible_unit_count != len(units)
            or artifact.excluded_unit_count != len(archived) - len(units)
            or artifact.collections != expected_collections
        ):
            raise CorpusError("index.content_mismatch")
        self.store._check_bytes(artifact.database_ref, artifact.database_sha256)
        db = sqlite3.connect(
            self.store._path(artifact.database_ref).as_uri() + "?mode=ro", uri=True
        )
        try:
            actual = db.execute("SELECT identity FROM metadata").fetchone()
            if actual is None or json.loads(actual[0]) != artifact.identity.model_dump(mode="json"):
                raise CorpusError("index.database_identity_mismatch")
            records = {
                row[0]: json.loads(row[1])
                for row in db.execute("SELECT chunk_id,record FROM chunks")
            }
            expected = {chunk["chunk_id"]: chunk for chunk in chunks}
            parent_links = dict(db.execute("SELECT chunk_id,unit_id FROM chunks"))
            source_units = {
                row[0]: json.loads(row[1]) for row in db.execute("SELECT unit_id,record FROM units")
            }
            unit_columns = {
                row[0]: (row[1], row[2])
                for row in db.execute("SELECT unit_id,collection,role FROM units")
            }
            fts = {row[0]: row[1] for row in db.execute("SELECT chunk_id,text FROM chunks_fts")}
            vectors = [
                json.loads(row[0]) if row[0] is not None else None
                for row in db.execute("SELECT embedding_json FROM chunks")
            ]
            self._vectors(vectors, len(chunks), artifact.identity.embedding)
            if (
                records != expected
                or parent_links != {c["chunk_id"]: c["unit_id"] for c in chunks}
                or db.execute("SELECT COUNT(*) FROM metadata").fetchone()[0] != 1
                or source_units != {u.unit_id: u.model_dump(mode="json") for u in units}
                or unit_columns != {u.unit_id: (u.collection, u.role) for u in units}
                or fts != {c["chunk_id"]: c["text"] for c in chunks}
                or db.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] != len(chunks)
                or db.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
                or artifact.chunk_count != len(chunks)
            ):
                raise CorpusError("index.content_mismatch")
        finally:
            db.close()
