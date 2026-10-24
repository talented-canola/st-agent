"""One headless/CLI entrance to component-one's verified consumer handoff."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from st_agent.trace import Trace, redact

from .contracts import CorpusError, sha256
from .handoff import project_release
from .indexes import ChunkerSpec, IndexCache
from .store import CorpusStore


def source_fingerprint():
    base = Path(__file__).resolve().parents[1]
    payload = b"".join(
        p.relative_to(base).as_posix().encode() + bytes.fromhex(sha256(p.read_bytes()))
        for p in sorted(base.rglob("*.py"))
    )
    return sha256(payload)


def prepare_handoff(root, release, output, *, max_chars=1200, embedder=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    trace = Trace(output)
    report = {
        "passed": False,
        "evidence_type": "component_one_consumer_preparation",
        "model_request_count": 0,
        "retrieval_quality_evaluated": False,
        "scientific_analysis_run": False,
        "source_sha256": source_fingerprint(),
    }
    project_root = Path(__file__).resolve().parents[3]
    try:
        head = subprocess.run(
            [
                "git",
                "-c",
                "safe.directory=" + project_root.as_posix(),
                "-C",
                str(project_root),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        report["git_head"] = head.stdout.strip() if head.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        report["git_head"] = None
    (output / "run-metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    started = time.perf_counter()
    trace.emit("knowledge.prepare.start", source_sha256=report["source_sha256"])
    try:
        store = CorpusStore(root, trace=trace)
        index = IndexCache(store).prepare(
            release, chunker=ChunkerSpec(max_chars=max_chars), embedder=embedder
        )
        manifest, units = project_release(store, Path(release))
        collections = {}
        for name in ("method_knowledge", "scientific_evidence"):
            archived = [u for u in units if u.collection == name]
            eligible = [u for u in archived if u.eligible_for_index]
            collections[name] = {
                "archived_source_count": len({(u.source_key, u.source_version) for u in archived}),
                "eligible_source_count": len({(u.source_key, u.source_version) for u in eligible}),
                "archived_unit_count": len(archived),
                "eligible_unit_count": len(eligible),
                "excluded_unit_count": len(archived) - len(eligible),
            }
        handoff = {
            "schema_version": 1,
            "release_sha256": manifest["release_sha256"],
            "index": index.model_dump(mode="json"),
            "collections": collections,
            "excluded": [
                {
                    "unit_id": u.unit_id,
                    "source_key": u.source_key,
                    "locator": u.locator,
                    "reason": list(u.quality_warnings),
                }
                for u in units
                if not u.eligible_for_index
            ],
            "unknown_conditions_are_universal": False,
            "external_text_is_instruction": False,
            "bge_m3_model_validation": "not_performed",
        }
        (output / "handoff.json").write_text(
            json.dumps(handoff, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report.update(
            index=index.model_dump(mode="json"),
            collections=collections,
            embedding_validation="lexical_only_no_model"
            if embedder is None
            else "provided_encoder_contract_only",
            passed=True,
        )
    except Exception as error:
        report.update(
            passed=False,
            error_code=str(error)
            if isinstance(error, CorpusError)
            else "preparation." + type(error).__name__,
        )
        trace.emit("knowledge.prepare.error", status="failed", error_code=report["error_code"])
    report.update(
        duration_seconds=time.perf_counter() - started,
        source_unchanged=report["source_sha256"] == source_fingerprint(),
    )
    report["passed"] = report["passed"] and report["source_unchanged"]
    trace.emit(
        "knowledge.prepare.end",
        status="passed" if report["passed"] else "failed",
        source_unchanged=report["source_unchanged"],
        model_request_count=0,
    )
    report["trace_degraded"] = trace.degraded
    report["passed"] = report["passed"] and not trace.degraded
    (output / "report.json").write_text(
        json.dumps(redact(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
