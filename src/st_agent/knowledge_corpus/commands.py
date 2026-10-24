"""CLI consumes the same acquisition and verification APIs as headless evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer

from st_agent.trace import Trace, redact

from .contracts import CorpusError, SourceSpec
from .fetch import HttpFetcher
from .preparation import prepare_handoff
from .store import CorpusStore

app = typer.Typer(
    no_args_is_help=True,
    help="Acquire and audit external-original knowledge.",
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)


def load_registry(path):
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1 or registry.get("origin") != "external_original":
        raise CorpusError("registry.invalid_schema")
    specs = [SourceSpec.model_validate(s) for s in registry["sources"]]
    if not specs or len({(s.source_key, s.version) for s in specs}) != len(specs):
        raise CorpusError("registry.empty_or_duplicate")
    return specs


def acquire_registry(registry, root, output, release, *, refresh=False, fetcher=None):
    """One bounded run; existing successful items are audited/reused, failed items resumed."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    trace = Trace(output)
    store = CorpusStore(root, trace=trace)
    report = {
        "passed": False,
        "evidence_type": "external_original_corpus_acquisition",
        "model_request_count": 0,
        "sources": [],
        "errors": [],
    }
    started = time.perf_counter()
    trace.emit("knowledge.run.start", refresh=refresh, release=release)
    own_fetcher = fetcher is None
    fetcher = fetcher or HttpFetcher(trace=trace)
    initial_requests = getattr(fetcher, "request_count", 0)
    try:
        specs = load_registry(registry)
        for spec in specs:
            try:
                snapshot = store.collect(spec, fetcher, refresh=refresh)
                report["sources"].append(
                    {
                        "source_key": spec.source_key,
                        "version": spec.version,
                        "snapshot_id": snapshot.snapshot_id,
                        "block_count": snapshot.block_count,
                    }
                )
            except CorpusError as error:
                report["errors"].append({"source_key": spec.source_key, "error_code": str(error)})
        if not report["errors"]:
            path = store.publish(release, specs)
            verified = store.verify_release(path)
            report.update(release_ref=str(path.relative_to(store.root)), verification=verified)
            report["passed"] = verified["passed"]
            if report["passed"]:
                inventory = store.inventory(path)
                (output / "inventory.json").write_text(
                    json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                report["category_counts"] = {
                    role: {"source_count": data["source_count"], "block_count": data["block_count"]}
                    for role, data in inventory["categories"].items()
                }
    except Exception as error:
        report["passed"] = False
        report["errors"].append(
            {
                "error_code": str(error)
                if isinstance(error, CorpusError)
                else "run." + type(error).__name__
            }
        )
        trace.emit("knowledge.run.error", status="failed", error_code=type(error).__name__)
    finally:
        if own_fetcher:
            fetcher.client.close()
    report.update(
        duration_seconds=time.perf_counter() - started,
        http_request_count=getattr(fetcher, "request_count", 0) - initial_requests,
    )
    trace.emit(
        "knowledge.run.end",
        status="passed" if report["passed"] else "failed",
        model_request_count=0,
        http_request_count=report["http_request_count"],
    )
    report["trace_degraded"] = trace.degraded
    report["passed"] = report["passed"] and not trace.degraded
    (output / "report.json").write_text(
        json.dumps(redact(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


@app.command("acquire")
def acquire(
    registry: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    root: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option(help="Unique run directory; never overwrite evidence")],
    release: Annotated[str, typer.Option()],
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
):
    """Fetch approved originals and publish only if every required source verifies."""
    report = acquire_registry(registry, root, output, release, refresh=refresh)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise typer.Exit(1)


@app.command("verify")
def verify(
    root: Annotated[Path, typer.Option(exists=True)],
    release: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    allow_withdrawn: Annotated[
        bool, typer.Option("--allow-withdrawn", help="Historical audit only")
    ] = False,
):
    """Verify all hashes and re-extract every original block without any network or LLM."""
    report = CorpusStore(root).verify_release(release, allow_withdrawn=allow_withdrawn)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise typer.Exit(1)


@app.command("status")
def status(root: Annotated[Path, typer.Option(exists=True)]):
    typer.echo(json.dumps(CorpusStore(root).status(), ensure_ascii=False, indent=2))


@app.command("inventory")
def inventory(
    root: Annotated[Path, typer.Option(exists=True)],
    release: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
):
    """List actual category counts and provenance after offline integrity verification."""
    typer.echo(json.dumps(CorpusStore(root).inventory(release), ensure_ascii=False, indent=2))


@app.command("withdraw")
def withdraw(
    root: Annotated[Path, typer.Option(exists=True)],
    source_key: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    reason: Annotated[str, typer.Option()],
):
    """Withdraw a corrected/retracted source without deleting its historical evidence."""
    CorpusStore(root).withdraw(source_key, version, reason)
    typer.echo("Source withdrawn; historical files retained.")


@app.command("prepare")
def prepare(
    root: Annotated[Path, typer.Option(exists=True)],
    release: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(help="Unique evidence directory")],
    max_chars: Annotated[int, typer.Option(min=16, max=20000)] = 1200,
):
    """Prepare verified two-collection units and a lexical index; no retrieval or LLM."""
    report = prepare_handoff(root, release, output, max_chars=max_chars)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise typer.Exit(1)


@app.command("retrieve")
def retrieve(
    root: Annotated[Path, typer.Option(exists=True)],
    release: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(help="Unique evidence directory")],
    request_file: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    text: Annotated[str | None, typer.Option(help="Natural language; requires --live")] = None,
    conditions_file: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    live: Annotated[bool, typer.Option("--live")] = False,
    live_rerank: Annotated[bool, typer.Option("--live-rerank")] = False,
):
    """Read-only evidence search. Structured JSON or a real provider intent; no analysis."""
    from st_agent.config import Settings
    from st_agent.deepseek import DeepSeekClient

    from .handoff import Applicability
    from .retrieval_runtime import run_retrieval

    if (request_file is None) == (text is None) or (text is not None and not live):
        raise typer.BadParameter("Choose --request-file, or --text with --live")
    if live and request_file is not None:
        raise typer.BadParameter("--live applies only to natural-language --text")
    if conditions_file is not None and text is None:
        raise typer.BadParameter("Structured requests already contain conditions")
    client = None
    reranker = None
    if live or live_rerank:
        settings = Settings.load()
        if settings.deepseek_api_key is None:
            raise typer.BadParameter("Configured DeepSeek authentication is required")
        client = DeepSeekClient(
            settings.deepseek_api_key.get_secret_value(),
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            max_requests=None,
        )
        if live_rerank:
            from .provider_reranker import ProviderReranker

            reranker = ProviderReranker(client)
    try:
        report = run_retrieval(
            root,
            release,
            output,
            request=json.loads(request_file.read_bytes()) if request_file else None,
            text=text,
            conditions=Applicability.model_validate_json(conditions_file.read_bytes())
            if conditions_file
            else None,
            client=client,
            reranker=reranker,
        )
    finally:
        if client is not None:
            client.close()
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise typer.Exit(1)
