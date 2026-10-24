"""Typer command-line interface for local research-project operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from st_agent.api import ResearchCopilot
from st_agent.doctor import run_doctor
from st_agent.headless_eval import run_approved_filter_scenario, run_qc_read_only_scenario
from st_agent.knowledge_corpus.commands import app as knowledge_app

app = typer.Typer(
    name="stagent",
    no_args_is_help=True,
    help="Local ST Research Agent project manager.",
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
)
project_app = typer.Typer(
    no_args_is_help=True,
    help="Create or inspect research projects.",
    rich_markup_mode=None,
)
dataset_app = typer.Typer(
    no_args_is_help=True,
    help="Register and inspect project datasets.",
    rich_markup_mode=None,
)
evaluation_app = typer.Typer(
    no_args_is_help=True,
    help="Run isolated headless research-runtime regression scenarios.",
    rich_markup_mode=None,
)
app.add_typer(project_app, name="project")
app.add_typer(dataset_app, name="dataset")
app.add_typer(evaluation_app, name="eval")
app.add_typer(knowledge_app, name="knowledge")


def _print_json(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@project_app.command("init")
def project_init(
    path: Annotated[Path, typer.Argument(help="Directory for the research project.")],
    name: Annotated[str | None, typer.Option("--name", help="Human-readable project name.")] = None,
) -> None:
    """Create a new isolated research-project directory."""
    copilot = ResearchCopilot.create(path, name=name)
    typer.echo(f"Created {copilot.project.name}: {copilot.project.id}")
    typer.echo(str(copilot.layout.root))
    copilot.close()


@project_app.command("open")
def project_open(
    path: Annotated[Path, typer.Argument(help="Existing research-project directory.")],
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Validate and summarize an existing research project."""
    copilot = ResearchCopilot.open(path)
    payload = {
        "project_id": copilot.project.id,
        "name": copilot.project.name,
        "root_path": copilot.project.root_path,
        "schema_version": copilot.project.schema_version,
        "datasets": len(copilot.list_datasets()),
        "sessions": len(copilot.list_sessions()),
    }
    if json_output:
        _print_json(payload)
    else:
        typer.echo(
            f"{payload['name']} ({payload['project_id']})\n"
            f"schema={payload['schema_version']} datasets={payload['datasets']} "
            f"sessions={payload['sessions']}\n{payload['root_path']}"
        )
    copilot.close()


@dataset_app.command("add")
def dataset_add(
    file: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)],
    project: Annotated[Path | None, typer.Option("--project", "-p")] = None,
    name: Annotated[str | None, typer.Option("--name")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Register a dataset in one project without modifying the source file."""
    copilot = ResearchCopilot.open(project or Path.cwd())
    record = copilot.add_dataset(file, display_name=name)
    if json_output:
        _print_json(record.model_dump(mode="json"))
    else:
        typer.echo(f"Registered {record.display_name}: {record.id}")
        typer.echo(f"sha256={record.sha256}")
    copilot.close()


@dataset_app.command("inspect")
def dataset_inspect(
    dataset_id: Annotated[str, typer.Argument(help="Registered dataset ID.")],
    project: Annotated[Path | None, typer.Option("--project", "-p")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show registered path, hash and storage metadata for one dataset."""
    copilot = ResearchCopilot.open(project or Path.cwd())
    record = copilot.get_dataset(dataset_id)
    if json_output:
        _print_json(record.model_dump(mode="json"))
    else:
        typer.echo(f"{record.display_name} ({record.id})")
        typer.echo(f"sha256={record.sha256}")
        typer.echo(f"stored={record.storage_path}")
    copilot.close()


@evaluation_app.command("qc-read-only")
def evaluation_qc_read_only(
    dataset: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o", help="New/empty evidence directory.")],
) -> None:
    """Run isolated QC and prove it does not change input data or active revision."""
    _print_json(run_qc_read_only_scenario(dataset, output))


@evaluation_app.command("approved-filter")
def evaluation_approved_filter(
    dataset: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o", help="New/empty evidence directory.")],
    min_genes: Annotated[int, typer.Option(min=0)] = 1,
    min_cells: Annotated[int, typer.Option(min=1)] = 1,
    max_mito_pct: Annotated[float, typer.Option(min=0, max=100)] = 100.0,
) -> None:
    """Run an explicitly approved filter in an isolated project and save evidence."""
    _print_json(
        run_approved_filter_scenario(
            dataset,
            output,
            min_genes=min_genes,
            min_cells=min_cells,
            max_mito_pct=max_mito_pct,
        )
    )


@app.command("doctor")
def doctor(
    project: Annotated[Path | None, typer.Option("--project", "-p")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Check Python, SQLite, credentials and optional project health."""
    report = run_doctor(project)
    if json_output:
        _print_json(report.model_dump(mode="json"))
    else:
        for check in report.checks:
            typer.echo(f"[{check.status.upper()}] {check.name}: {check.detail}")
    if not report.healthy:
        raise typer.Exit(code=1)


@evaluation_app.command("preflight")
def evaluation_preflight(
    live: Annotated[bool, typer.Option("--live")] = False,
    env_file: Annotated[Path | None, typer.Option("--env-file")] = None,
) -> None:
    """Check real chat and structured intent using explicit provider configuration."""
    from st_agent.config import Settings
    from st_agent.evaluation.preflight import run_provider_preflight

    if not live:
        raise typer.BadParameter("Pass --live to perform provider requests")
    report = run_provider_preflight(Settings.load(env_file))
    _print_json(report.model_dump(mode="json"))
    if not report.passed:
        raise typer.Exit(code=1)


@evaluation_app.command("run")
def evaluation_run(
    dataset: Annotated[Path, typer.Option(exists=True)],
    output: Annotated[Path, typer.Option()],
    cases: Annotated[Path, typer.Option(exists=True)],
    live: Annotated[bool, typer.Option("--live")] = False,
    env_file: Annotated[Path | None, typer.Option("--env-file")] = None,
    max_requests: Annotated[int, typer.Option(min=1)] = 100,
) -> None:
    """Run live cases; pass a replay JSONL as --cases to reproduce a failure."""
    from st_agent.config import Settings
    from st_agent.evaluation.baseline import run_baseline

    if not live:
        raise typer.BadParameter("Pass --live to perform provider requests")
    report = run_baseline(
        dataset, output, cases, settings=Settings.load(env_file), max_requests=max_requests
    )
    _print_json({"passed": report["passed"], "total": report["total"]})
    if report["passed"] != report["total"]:
        raise typer.Exit(code=1)


@evaluation_app.command("compare")
def evaluation_compare(
    baseline: Annotated[Path, typer.Option(exists=True)],
    candidate: Annotated[Path, typer.Option(exists=True)],
) -> None:
    """Compare stable case identities, reporting additions and missing coverage."""
    from st_agent.evaluation.protocol import compare_reports

    try:
        comparison = compare_reports(
            json.loads(baseline.read_text(encoding="utf-8")),
            json.loads(candidate.read_text(encoding="utf-8")),
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    _print_json(comparison)
    if comparison["regressed"] or comparison["missing"]:
        raise typer.Exit(code=1)
