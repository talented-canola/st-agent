"""Resumable acquisition and immutable, independently verifiable corpus releases."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

from st_agent.utils import utc_now_iso

from .contracts import CorpusError, Snapshot, SourceSpec, sha256
from .parsers import PARSER_VERSION, parse_source


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


class CorpusStore:
    def __init__(self, root, *, trace=None):
        self.root = Path(root).resolve()
        self.trace = trace
        self._license_cache = {}
        for name in ("registry", "raw", "parsed", "releases", "reports"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.database = self.root / "catalog.sqlite"
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sources(
                    source_key TEXT NOT NULL, version TEXT NOT NULL, spec TEXT NOT NULL,
                    status TEXT NOT NULL, raw_ref TEXT, latest_snapshot TEXT, error_code TEXT,
                    withdrawn_reason TEXT, updated_at TEXT NOT NULL,
                    PRIMARY KEY(source_key, version));
                CREATE TABLE IF NOT EXISTS snapshots(
                    snapshot_id TEXT PRIMARY KEY, record TEXT NOT NULL);
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def _emit(self, event, **payload):
        if self.trace:
            self.trace.emit("knowledge." + event, **payload)

    def _path(self, relative):
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise CorpusError("integrity.path_escape")
        return path

    def _put(self, relative, content):
        path = self._path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != content:
                raise CorpusError("integrity.immutable_file_conflict")
            return relative
        # Link atomically without replacing an existing object, including concurrent writers.
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != content:
                    raise CorpusError("integrity.immutable_file_conflict") from None
        finally:
            temporary.unlink()
        return relative

    def _check_bytes(self, relative, digest):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise CorpusError("integrity.invalid_digest")
        raw = self._path(relative).read_bytes()
        if sha256(raw) != digest:
            raise CorpusError("integrity.checksum_mismatch")
        return raw

    def register(self, spec):
        value = encoded(spec.model_dump(mode="json"))
        self._put("registry/" + sha256(value) + ".json", value)
        with self._connect() as db:
            existing = db.execute(
                "SELECT spec FROM sources WHERE source_key=? AND version=?",
                (spec.source_key, spec.version),
            ).fetchone()
            if existing and existing["spec"] != value.decode():
                raise CorpusError("source.spec_conflict_use_new_version")
            db.execute(
                "INSERT OR IGNORE INTO sources VALUES (?,?,?,'pending',NULL,NULL,NULL,NULL,?)",
                (spec.source_key, spec.version, value.decode(), utc_now_iso()),
            )

    def _row(self, spec):
        with self._connect() as db:
            return db.execute(
                "SELECT * FROM sources WHERE source_key=? AND version=?",
                (spec.source_key, spec.version),
            ).fetchone()

    def _update(self, spec, status, **values):
        if not set(values) <= {"raw_ref", "latest_snapshot", "error_code"}:
            raise ValueError("Unknown catalog fields")
        values.update(status=status, updated_at=utc_now_iso())
        clause = ",".join(key + "=?" for key in values)
        with self._connect() as db:
            db.execute(
                "UPDATE sources SET " + clause + " WHERE source_key=? AND version=?",
                (*values.values(), spec.source_key, spec.version),
            )

    def _snapshot(self, snapshot_id):
        with self._connect() as db:
            row = db.execute(
                "SELECT record FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
        if row is None:
            raise CorpusError("integrity.snapshot_missing")
        return Snapshot.model_validate_json(row["record"])

    def _verify_snapshot(self, spec, snapshot):
        if (snapshot.source_key, snapshot.source_version) != (spec.source_key, spec.version):
            raise CorpusError("integrity.snapshot_source_mismatch")
        identity = {
            "spec": snapshot.spec_sha256,
            "raw": snapshot.raw_sha256,
            "license": snapshot.license_sha256,
            "parsed": snapshot.parsed_sha256,
            "parser": snapshot.parser_version,
        }
        if snapshot.snapshot_id != sha256(encoded(identity)):
            raise CorpusError("integrity.snapshot_identity_mismatch")
        if sha256(encoded(spec.model_dump(mode="json"))) != snapshot.spec_sha256:
            raise CorpusError("integrity.source_spec_mismatch")
        raw = self._check_bytes(snapshot.raw_ref, snapshot.raw_sha256)
        license_raw = self._check_bytes(snapshot.license_ref, snapshot.license_sha256)
        self._license(spec, license_raw)
        if spec.expected_sha256 and snapshot.raw_sha256 != spec.expected_sha256:
            raise CorpusError("integrity.expected_source_mismatch")
        parsed = self._check_bytes(snapshot.parsed_ref, snapshot.parsed_sha256)
        document = parse_source(spec, raw)
        if (
            encoded(document.model_dump(mode="json")) != parsed
            or document.parser_version != snapshot.parser_version
            or len(document.blocks) != snapshot.block_count
        ):
            raise CorpusError("integrity.reparse_mismatch")
        return document

    @staticmethod
    def _license(spec, raw):
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeError as error:
            raise CorpusError("source.license_mismatch") from error
        if spec.license_marker not in text:
            raise CorpusError("source.license_mismatch")

    def collect(self, spec: SourceSpec, fetcher, *, refresh=False):
        self.register(spec)
        row = self._row(spec)
        if row["withdrawn_reason"]:
            raise CorpusError("source.withdrawn")
        self._emit("collect.start", source_key=spec.source_key, version=spec.version)
        try:
            if row["status"] == "ready" and row["latest_snapshot"] and not refresh:
                snapshot = self._snapshot(row["latest_snapshot"])
                self._verify_snapshot(spec, snapshot)
                if snapshot.parser_version == PARSER_VERSION:
                    self._emit(
                        "collect.reused",
                        source_key=spec.source_key,
                        snapshot_id=snapshot.snapshot_id,
                    )
                    return snapshot
            raw_ref = row["raw_ref"]
            if not refresh and raw_ref:
                digest = Path(raw_ref).stem
                raw = self._check_bytes(raw_ref, digest)
            else:
                raw = fetcher.get(spec.fetch_url)
                raw_ref = self._put("raw/" + sha256(raw) + ".bin", raw)
                self._update(spec, "fetched", raw_ref=raw_ref, error_code=None)
                self._emit("raw.saved", source_key=spec.source_key, sha256=sha256(raw))
            if spec.expected_sha256 and sha256(raw) != spec.expected_sha256:
                raise CorpusError("source.content_mismatch")
            if spec.license_url == spec.fetch_url:
                license_raw = raw
            elif spec.license_url in self._license_cache:
                license_raw = self._license_cache[spec.license_url]
            else:
                license_raw = fetcher.get(spec.license_url)
            license_ref = self._put("raw/" + sha256(license_raw) + ".bin", license_raw)
            self._license(spec, license_raw)
            self._license_cache[spec.license_url] = license_raw
            document = parse_source(spec, raw)
            parsed = encoded(document.model_dump(mode="json"))
            parsed_ref = self._put("parsed/" + sha256(parsed) + ".json", parsed)
            identity = {
                "spec": sha256(encoded(spec.model_dump(mode="json"))),
                "raw": sha256(raw),
                "license": sha256(license_raw),
                "parsed": sha256(parsed),
                "parser": document.parser_version,
            }
            snapshot = Snapshot(
                snapshot_id=sha256(encoded(identity)),
                source_key=spec.source_key,
                source_version=spec.version,
                spec_sha256=identity["spec"],
                raw_sha256=identity["raw"],
                raw_ref=raw_ref,
                license_sha256=identity["license"],
                license_ref=license_ref,
                parsed_sha256=identity["parsed"],
                parsed_ref=parsed_ref,
                parser_version=document.parser_version,
                block_count=len(document.blocks),
            )
            self._verify_snapshot(spec, snapshot)
            with self._connect() as db:
                db.execute(
                    "INSERT OR IGNORE INTO snapshots VALUES (?,?)",
                    (snapshot.snapshot_id, encoded(snapshot.model_dump(mode="json")).decode()),
                )
                db.execute(
                    "UPDATE sources SET status='ready',latest_snapshot=?,error_code=NULL,"
                    "updated_at=? WHERE source_key=? AND version=?",
                    (snapshot.snapshot_id, utc_now_iso(), spec.source_key, spec.version),
                )
            self._emit(
                "collect.ready",
                source_key=spec.source_key,
                snapshot_id=snapshot.snapshot_id,
                block_count=snapshot.block_count,
            )
            return snapshot
        except Exception as error:
            code = (
                str(error) if isinstance(error, CorpusError) else "storage." + type(error).__name__
            )
            status = "quarantined" if code.startswith("source.") else "failed"
            self._update(spec, status, error_code=code)
            self._emit("collect.failed", status=status, error_code=code, source_key=spec.source_key)
            raise CorpusError(code) from error

    def status(self):
        with self._connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT source_key,version,status,latest_snapshot,error_code,withdrawn_reason "
                    "FROM sources ORDER BY source_key,version"
                )
            ]

    def withdraw(self, source_key, version, reason):
        if not reason.strip():
            raise ValueError("Withdrawal reason required")
        with self._connect() as db:
            updated = db.execute(
                "UPDATE sources SET withdrawn_reason=?,status='withdrawn',updated_at=? "
                "WHERE source_key=? AND version=?",
                (reason, utc_now_iso(), source_key, version),
            )
            if updated.rowcount != 1:
                raise CorpusError("source.not_registered")
        self._emit("source.withdrawn", source_key=source_key, version=version, reason=reason)

    def publish(self, name, specs):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,119}", name):
            raise CorpusError("release.invalid_name")
        if not specs or len({(s.source_key, s.version) for s in specs}) != len(specs):
            raise CorpusError("release.empty_or_duplicate_sources")
        entries = []
        for spec in sorted(specs, key=lambda s: (s.source_key, s.version)):
            row = self._row(spec)
            if row is None or row["status"] != "ready" or row["withdrawn_reason"]:
                raise CorpusError("release.source_not_ready:" + spec.source_key)
            snapshot = self._snapshot(row["latest_snapshot"])
            self._verify_snapshot(spec, snapshot)
            entries.append(
                {
                    "source": spec.model_dump(mode="json"),
                    "snapshot": snapshot.model_dump(mode="json"),
                }
            )
        manifest = {
            "schema_version": 1,
            "name": name,
            "origin": "external_original",
            "sources": entries,
            "generated_knowledge_count": 0,
        }
        manifest["release_sha256"] = sha256(encoded(manifest))
        relative = "releases/" + name + ".json"
        try:
            self._put(relative, encoded(manifest))
        except CorpusError as error:
            raise CorpusError("release.immutable_release") from error
        self._emit(
            "release.published",
            name=name,
            source_count=len(entries),
            release_sha256=manifest["release_sha256"],
        )
        return self._path(relative)

    def verify_release(self, path, *, allow_withdrawn=False):
        report = {
            "passed": False,
            "source_count": 0,
            "block_count": 0,
            "errors": [],
            "verification": "offline_hash_and_deterministic_reparse",
            "allow_withdrawn": allow_withdrawn,
        }
        try:
            manifest = json.loads(Path(path).read_bytes())
            claimed = manifest.pop("release_sha256")
            if sha256(encoded(manifest)) != claimed or manifest["origin"] != "external_original":
                raise CorpusError("integrity.release_checksum_mismatch")
            identities = [
                (e["source"]["source_key"], e["source"]["version"]) for e in manifest["sources"]
            ]
            if len(set(identities)) != len(identities) or manifest.get("schema_version") != 1:
                raise CorpusError("release.invalid_content")
            for entry in manifest["sources"]:
                spec = SourceSpec.model_validate(entry["source"])
                snapshot = Snapshot.model_validate(entry["snapshot"])
                row = self._row(spec)
                if row is not None and row["withdrawn_reason"] and not allow_withdrawn:
                    raise CorpusError("source.withdrawn")
                document = self._verify_snapshot(spec, snapshot)
                report["source_count"] += 1
                report["block_count"] += len(document.blocks)
            if not manifest["sources"] or manifest.get("generated_knowledge_count") != 0:
                raise CorpusError("release.invalid_content")
            report["release_sha256"] = claimed
            report["passed"] = True
        except Exception as error:
            report["errors"].append(
                str(error)
                if isinstance(error, CorpusError)
                else "integrity." + type(error).__name__
            )
        self._emit("release.verified", status="passed" if report["passed"] else "failed", **report)
        return report

    def inventory(self, path):
        """Report verified source/fragment counts, not inferred knowledge or accuracy."""
        verification = self.verify_release(path)
        if not verification["passed"]:
            raise CorpusError("inventory.release_not_verified")
        manifest = json.loads(Path(path).read_bytes())
        categories = {
            role: {"source_count": 0, "block_count": 0, "topics": [], "sources": []}
            for role in ("method", "scientific_evidence", "workflow_reference", "ontology")
        }
        warning_count = 0
        for entry in manifest["sources"]:
            source, snapshot = entry["source"], entry["snapshot"]
            category = categories[source["role"]]
            category["source_count"] += 1
            category["block_count"] += snapshot["block_count"]
            category["topics"] = sorted(set(category["topics"]) | set(source["topics"]))
            parsed = json.loads(
                self._check_bytes(snapshot["parsed_ref"], snapshot["parsed_sha256"])
            )
            blocks = []
            for block in parsed["blocks"]:
                warnings = []
                if source["format"] == "python_docstrings" and re.search(
                    r"\{[a-zA-Z_]\w*\}|%\([^)]+\)s", block["text"]
                ):
                    warnings.append("unexpanded_doc_template")
                warning_count += bool(warnings)
                blocks.append(
                    {
                        "kind": block["kind"],
                        "locator": block["locator"],
                        "text_sha256": sha256(block["text"].encode("utf-8")),
                        "quality_warnings": warnings,
                    }
                )
            category["sources"].append(
                {
                    "source_key": source["source_key"],
                    "title": source["title"],
                    "version": source["version"],
                    "canonical_uri": source["canonical_uri"],
                    "license": source["license"],
                    "block_count": snapshot["block_count"],
                    "raw_ref": snapshot["raw_ref"],
                    "parsed_ref": snapshot["parsed_ref"],
                    "blocks": blocks,
                }
            )
        return {
            "release": manifest["name"],
            "release_sha256": manifest["release_sha256"],
            "source_count": verification["source_count"],
            "block_count": verification["block_count"],
            "categories": categories,
            "generated_knowledge_count": manifest["generated_knowledge_count"],
            "counts_are": "external_source_files_and_original_blocks_not_verified_claims",
            "quality": {
                "blocks_with_warnings": warning_count,
                "blocks_with_warnings_are_direct_answer_ready": False,
                "scientific_claim_validation": "not_performed",
                "workflow_references_are_scientific_evidence": False,
                "downloaded_code_execution_allowed": False,
            },
        }
