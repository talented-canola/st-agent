"""Plan-bound, approval-gated execution of admitted scientific tools.

This headless execution boundary is shared by the conversation runtime and
the evaluation harness; the UI consumes it through WorkbenchService.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from st_agent.contracts import (
    Admission,
    CostClass,
    Effect,
    ResourceRequest,
    Risk,
    StrictRecord,
    ToolResult,
    ToolSpec,
    ToolStatus,
    ValidationReport,
    ValidationStatus,
    ArtifactRef,
)
from st_agent.errors import PreconditionFailedError, error_detail_from_exception
from st_agent.execution import ExecutionCoordinator, ToolDefinition
from st_agent.ids import IdKind, validate_id
from st_agent.planning import ExecutionPlan
from st_agent.safety import GateDisposition, SafetyGate, SafetyStore
from st_agent.scientific import inspect_dataset
from st_agent.tool_registry import SCIENTIFIC_TOOLS as TOOLS, TOOL_VERSIONS, execute_registered_tool
from st_agent.trace import emit


class ScientificWorkerOutput(BaseModel):
    """Small contract returned by the isolated scientific worker."""

    model_config = ConfigDict(extra="forbid", strict=True)

    summary: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    revision_output: str | None = None
    artifact_outputs: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)


def _spatial_artifacts(tool_name, source, output, metrics, workspace):
    """Write artifacts only in the Worker workspace; the parent validates/publishes."""
    if tool_name not in TOOL_VERSIONS or tool_name == "samples.detect":
        return []
    import csv
    import plotly.graph_objects as go
    from st_agent.utils import sha256_file
    files = []
    data_file = workspace / "spatial_results.json"
    data_file.write_text(json.dumps(metrics, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    files.append((data_file, "json", "data.spatial.results"))
    from plotly.subplots import make_subplots
    series = [key for key in ("genes", "groups", "composition", "expression")
        if isinstance(metrics.get(key), list) and metrics[key]]
    figure = make_subplots(rows=max(1, len(series)), cols=1,
        subplot_titles=series or ["Results"])
    chart_row = 0
    for key in ("genes", "groups", "composition", "expression"):
        rows = metrics.get(key)
        if not isinstance(rows, list) or not rows:
            continue
        flat = []
        for row in rows:
            projected = {}
            for key_, value in row.items():
                if isinstance(value, dict):
                    projected.update({f"{key_}_{child}": item for child, item in value.items()})
                else:
                    projected[key_] = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
            flat.append(projected)
        path = workspace / f"spatial_{key}.csv"
        fields = list(dict.fromkeys(k for row in flat for k in row))
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)
        files.append((path, "csv", "table.spatial." + key))
        value_key = {"genes": "morans_i", "groups": "observations",
            "composition": "fraction", "expression": "mean_expression"}[key]
        chart_row += 1
        figure.add_bar(name=key, x=[" / ".join(str(row[k]) for k in
            ("group", "stage", "gene", "cluster") if k in row) for row in rows],
            y=[row.get(value_key) for row in rows], row=chart_row, col=1)
        figure.update_yaxes(title_text=value_key, row=chart_row, col=1)
    if output is not None:
        import numpy as np
        from st_agent.analysis_context import load_analysis_metadata
        from st_agent.scientific import _spatial_key
        metadata = load_analysis_metadata(output)
        coordinates = np.asarray(metadata.obsm[_spatial_key(metadata)])
        selected = np.linspace(0, metadata.n_obs - 1, min(metadata.n_obs, 10_000), dtype=int)
        groupby = metrics.get("groupby")
        groups = metadata.obs[groupby].astype(str).to_numpy() if groupby else np.full(metadata.n_obs, "all")
        label_key = "spatial_expression_domain" if tool_name == "spatial.expression_domains" else (
            "coordinate_partition" if tool_name == "spatial.domains" else None)
        labels = metadata.obs[label_key].astype(str).to_numpy() if label_key else groups
        # Display independent sections on separate axes; coordinates remain explicitly local.
        import pandas as pd
        import plotly.express as px
        displayed_groups = sorted(set(groups[selected]))[:12]
        selected = selected[np.isin(groups[selected], displayed_groups)]
        frame = pd.DataFrame({"x": coordinates[selected, 0], "y": coordinates[selected, 1],
            "section": groups[selected], "label": labels[selected]})
        figure = px.scatter(frame, x="x", y="y", color="label", facet_col="section", facet_col_wrap=3)
        figure.update_xaxes(matches=None)
        figure.update_yaxes(matches=None)
        for axis in figure.layout:
            if axis.startswith("yaxis"):
                suffix = axis[5:]
                figure.layout[axis].update(scaleanchor="x" + suffix, scaleratio=1)
        figure.update_layout(title=f"{tool_name}: section-local coordinates; displayed {len(selected)}/{metadata.n_obs}; {len(displayed_groups)} sections")
        plotdata = workspace / "spatial_plotdata.json"
        plotdata.write_text(json.dumps({"coordinate_key": _spatial_key(metadata),
            "displayed_observations": len(selected), "total_observations": metadata.n_obs,
            "displayed_sections": displayed_groups, "groupby": groupby,
            "data": frame.to_dict(orient="records")}, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        files.append((plotdata, "json", "data.spatial.plotdata"))
    else:
        figure.update_layout(title=f"{tool_name}: descriptive results; no significance inference")
        figure.update_layout(height=max(450, 300 * max(1, len(series))))
    html = workspace / "spatial_results.html"
    html.write_text(figure.to_html(include_plotlyjs=True, div_id="st-agent-spatial-results"), encoding="utf-8")
    files.append((html, "html", "figure.spatial.results"))
    return [{"relative_path": path.name, "format": format_, "kind": kind,
        "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path, format_, kind in files]


def _bounded_metrics(metrics):
    """Keep full tables in saved files; make projection loss explicit in the wire result."""
    from st_agent.serialization import canonical_json
    projected = dict(metrics)
    lists = {key: len(value) for key, value in metrics.items() if isinstance(value, list)}
    while len(canonical_json(projected).encode("utf-8")) > 96 * 1024:
        candidates = [key for key in lists if len(projected[key]) > 1]
        if not candidates:
            raise ValueError("Scientific metadata exceeds the bounded Worker result contract")
        key = max(candidates, key=lambda item: len(canonical_json(projected[item]).encode("utf-8")))
        projected[key] = projected[key][:max(1, len(projected[key]) // 2)]
    omitted = {key: {"provided_rows": len(projected[key]), "total_rows": size,
        "availability": "partial_saved_full_table", "full_artifact": "spatial_results.json"}
        for key, size in lists.items() if len(projected[key]) != size}
    if omitted:
        projected["evidence_projection"] = omitted
    return projected


class ScientificRevisionValidator:
    """Validate that a worker produced a readable H5AD revision."""

    def __init__(self, tool_name: str | None = None):
        self.tool_name = tool_name

    def validate(self, candidate: Path, _context: dict[str, Any]) -> ValidationReport:
        try:
            details = inspect_dataset(candidate)
            if self.tool_name and self.tool_name.startswith("spatial."):
                import h5py
                import numpy as np
                from anndata.io import read_elem
                from st_agent.analysis_context import load_analysis_metadata
                from st_agent.advanced_scientific import _validate_graph
                metadata = load_analysis_metadata(candidate)
                with h5py.File(candidate, "r") as handle:
                    if self.tool_name in {"spatial.neighbors", "spatial.expression_domains"}:
                        metadata.obsp = {"spatial_connectivities": read_elem(handle["obsp/spatial_connectivities"])}
                        metadata.uns["st_agent_spatial_neighbors"] = read_elem(handle["uns/st_agent_spatial_neighbors"])
                        _validate_graph(metadata, "spatial_connectivities")
                    if self.tool_name == "spatial.expression_domains":
                        graph = read_elem(handle["obsp/spatial_expression_connectivities"])
                        if not np.isfinite(graph.data).all() or graph.nnz == 0 or (graph.data <= 0).any():
                            raise ValueError("Expression-domain graph has invalid weights")
                        if "spatial_expression_domain" not in metadata.obs or metadata.obs["spatial_expression_domain"].isna().any():
                            raise ValueError("Expression-domain labels are missing")
                    if self.tool_name == "spatial.domains" and (
                        "coordinate_partition" not in metadata.obs or metadata.obs["coordinate_partition"].isna().any()):
                        raise ValueError("Coordinate-partition labels are missing")
        except BaseException as error:
            return ValidationReport(
                status=ValidationStatus.FAILED,
                summary="Scientific worker output is not a readable H5AD dataset",
                error=error_detail_from_exception(error),
            )
        return ValidationReport(
            status=ValidationStatus.PASSED,
            summary="Scientific worker output is a readable H5AD dataset",
            checks={"shape": details["shape"], "count_source": details["count_source"]},
        )


class ScientificWorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tool_name: str
    source: str
    arguments: dict[str, Any]


def run_scientific_tool(
    request: dict[str, Any], workspace: Path, cancelled: Callable[[], bool]
) -> dict[str, Any]:
    """Worker entrypoint.  It has no project database access or approval authority."""
    payload = ScientificWorkerRequest.model_validate(request)
    if cancelled():
        raise RuntimeError("Scientific worker was cancelled before execution")
    tool = TOOLS.get(payload.tool_name)
    if tool is None:
        raise ValueError(f"Unknown admitted scientific tool: {payload.tool_name}")
    source = Path(payload.source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = workspace / "revision.h5ad" if tool.effect is Effect.DERIVED_REVISION else None
    metrics = execute_registered_tool(
        tool.name,
        source,
        output=output,
        arguments=payload.arguments,
    )
    if cancelled():
        raise RuntimeError("Scientific worker was cancelled during execution")
    artifact_outputs = _spatial_artifacts(tool.name, source, output, metrics, workspace)
    return ScientificWorkerOutput(
        summary=f"{tool.name} completed",
        metrics=_bounded_metrics(metrics) if artifact_outputs else metrics,
        revision_output=output.name if output is not None else None,
        artifact_outputs=artifact_outputs,
    ).model_dump(mode="json")


def _worker_request(tool_name: str) -> Callable[[dict, Path], dict]:
    def build(arguments: dict, source: Path) -> dict:
        return {"tool_name": tool_name, "source": str(source), "arguments": arguments}

    return build


def scientific_tool_definition(tool_name: str) -> ToolDefinition:
    """Create the typed runtime contract for one admitted scientific tool."""
    tool = TOOLS[tool_name]
    seed_policy = "required" if "seed" in tool.arguments.model_fields else "forbidden"
    is_derived = tool.effect is Effect.DERIVED_REVISION
    advanced = tool_name in TOOL_VERSIONS
    timeout = 600 if tool_name == "spatial.expression_domains" else 300 if is_derived else 120
    memory = 2_000_000_000 if tool_name == "spatial.expression_domains" else (
        1_000_000_000 if is_derived or advanced else 256_000_000)
    return ToolDefinition(
        spec=ToolSpec(
            name=tool.name,
            version=TOOL_VERSIONS.get(tool_name, "scientific/0.3"),
            description=("Coordinate-only KMeans partition; does not use expression" if tool_name == "spatial.domains"
                else f"Isolated execution of {tool.name}"),
            admission=Admission.VERIFIED,
            # Every admitted scientific transformation needs concrete approval;
            # the safety policy's historical name list is not an authorization registry.
            risk=Risk.R2 if is_derived else Risk.R0,
            input_schema_ref=f"scientific:{tool.arguments.__name__}",
            output_schema_ref="scientific:ScientificWorkerOutput",
            preconditions=("dataset.registered", "input_revision.materialized"),
            effects=tool.effect,
            deterministic=True,
            seed_policy=seed_policy,
            timeout_seconds=timeout,
            cost_class=CostClass.MEDIUM if is_derived else CostClass.SMALL,
            executor="st_agent.scientific_execution:run_scientific_tool",
            validator="st_agent.scientific_execution:ScientificRevisionValidator",
        ),
        input_model=tool.arguments,
        output_model=ScientificWorkerOutput,
        validator=ScientificRevisionValidator(tool_name),
        resources=ResourceRequest(
            estimated_ram_bytes=memory,
            estimated_temp_bytes=1_000_000_000 if is_derived else 64_000_000,
            timeout_seconds=timeout,
            heavy=is_derived or advanced,
        ),
        package_names=("anndata", "scanpy", "numpy", "scipy", "scikit-learn", "igraph", "plotly"),
        worker_request_builder=_worker_request(tool.name),
    )


class PlanExecutionResponse(StrictRecord):
    disposition: GateDisposition
    approval_id: str | None = None
    tool_result: ToolResult | None = None
    checkpoint_id: str
    replayed: bool = False


class ScientificPlanExecutor:
    """One-step plan executor with durable approval and replay boundaries."""

    def __init__(self, copilot) -> None:
        self.copilot = copilot
        self.store = SafetyStore(copilot.layout.database_path, copilot.project.id)
        self.gate = SafetyGate(self.store)
        self.coordinator = ExecutionCoordinator(
            copilot,
            copilot.worker_manager,
            lock_path=copilot.layout.root.parent / "environment-win-64.lock.yml",
            dependency_lock_fallback="0" * 64,
        )

    def execute(
        self,
        plan: ExecutionPlan,
        *,
        dataset_id: str,
        session_id: str,
        run_id: str,
        approval_id: str | None = None,
        conversation_state: dict[str, Any] | None = None,
    ) -> PlanExecutionResponse:
        validate_id(dataset_id, IdKind.DATASET)
        validate_id(session_id, IdKind.SESSION)
        validate_id(run_id, IdKind.RUN)
        if plan.blocked_reason is not None or len(plan.steps) != 1:
            raise PreconditionFailedError(
                "Only one unblocked plan step can enter the execution boundary",
                code="plan.not_executable",
            )
        if plan.state.dataset_id != dataset_id:
            raise PreconditionFailedError(
                "Execution plan belongs to a different dataset",
                code="plan.dataset_mismatch",
            )
        step = plan.steps[0]
        from st_agent.utils import utc_now_iso
        with self.store._connect() as connection:
            connection.execute("INSERT OR IGNORE INTO runs(id,session_id,dataset_id,"
                "active_revision_id,status,goal,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (run_id, session_id, dataset_id, plan.input_revision_id, "planned",
                 step.tool_name, utc_now_iso(), utc_now_iso()))
            connection.commit()
        definition = scientific_tool_definition(step.tool_name)
        if definition.spec.effects is not step.effect:
            raise PreconditionFailedError(
                "Plan effect does not match the admitted tool contract",
                code="plan.effect_mismatch",
            )
        replay_arguments = step.parameters
        if approval_id is not None:
            approval = self.store.get_approval(approval_id)
            if approval.status.value == "approved":
                replay_arguments = approval.final_arguments
        execution_key = SafetyGate.execution_key(
            project_id=self.copilot.project.id,
            session_id=session_id,
            run_id=run_id,
            dataset_id=dataset_id,
            revision_id=plan.input_revision_id,
            tool_name=definition.spec.name,
            tool_version=definition.spec.version,
            arguments=replay_arguments,
        )
        replay = self.store.completed_execution(execution_key)
        # Stable call identity bridges tool persistence and safety-journal persistence.
        durable = self.copilot.tool_executions.find_by_call("call_" + execution_key[:32])
        if durable is None or durable.status not in {ToolStatus.SUCCESS, ToolStatus.WARNING}:
            restored = self.coordinator.recover_committed(dataset_id=dataset_id,
                call_id="call_" + execution_key[:32], definition=definition,
                arguments=replay_arguments, input_revision_id=plan.input_revision_id)
            if restored is not None:
                durable = restored
                replay = None
        if replay is None and durable is not None:
            if (
                durable.tool_name != step.tool_name
                or durable.input_revision_id != plan.input_revision_id
            ):
                raise ValueError("Durable execution identity mismatch")
            replay = durable.model_dump(mode="json")
            with self.store._connect() as connection:
                connection.execute(
                    "UPDATE safety_executions SET status='completed',result_json=?,completed_at=? "
                    "WHERE idempotency_key=? AND project_id=?",
                    (
                        json.dumps(replay),
                        durable.provenance.finished_at.isoformat(),
                        execution_key,
                        self.copilot.project.id,
                    ),
                )
                connection.commit()
        if replay is not None:
            result = ToolResult.model_validate_json(json.dumps(replay))
            checkpoint = self.store.save_checkpoint(
                run_id,
                self._checkpoint_state(
                    plan,
                    dataset_id,
                    step.tool_name,
                    "completed",
                    result,
                    conversation=conversation_state,
                ),
                "completed",
            )
            return PlanExecutionResponse(
                disposition=(
                    GateDisposition.AUTHORIZED
                    if definition.spec.effects is Effect.DERIVED_REVISION
                    else GateDisposition.AUTOMATIC
                ),
                approval_id=approval_id,
                tool_result=result,
                checkpoint_id=checkpoint.id,
                replayed=True,
            )
        current = self.copilot.get_dataset(dataset_id)
        if (
            definition.spec.effects is Effect.DERIVED_REVISION
            and current.active_revision_id != plan.input_revision_id
        ):
            raise PreconditionFailedError(
                "Dataset active revision changed after planning",
                code="plan.input_revision_stale",
            )
        checkpoint = self.store.save_checkpoint(
            run_id,
            self._checkpoint_state(
                plan, dataset_id, step.tool_name, "gate", conversation=conversation_state
            ),
            "gate",
        )
        outcome = self.gate.inspect(
            session_id=session_id,
            run_id=run_id,
            dataset_id=dataset_id,
            revision_id=plan.input_revision_id,
            tool_name=definition.spec.name,
            tool_version=definition.spec.version,
            arguments=step.parameters,
            declared_risk=definition.spec.risk,
            effect=definition.spec.effects,
            approval_id=approval_id,
        )
        if outcome.disposition not in {GateDisposition.AUTOMATIC, GateDisposition.AUTHORIZED}:
            checkpoint = self.store.save_checkpoint(
                run_id,
                self._checkpoint_state(
                    plan,
                    dataset_id,
                    step.tool_name,
                    outcome.disposition.value,
                    conversation=conversation_state,
                ),
                outcome.disposition.value,
            )
            return PlanExecutionResponse(
                disposition=outcome.disposition,
                approval_id=outcome.approval.id if outcome.approval else None,
                checkpoint_id=checkpoint.id,
            )
        call_id = "call_" + outcome.execution_key[:32]
        emit(self.copilot, "execution.authorized", disposition=outcome.disposition.value,
             approval_id=approval_id, call_id=call_id, parameters=outcome.arguments,
             input_revision_id=plan.input_revision_id, tool_name=step.tool_name)
        seed = outcome.arguments.get("seed") if definition.spec.seed_policy == "required" else None

        def invoke(arguments: dict[str, Any]) -> dict[str, Any]:
            result = self.coordinator.execute(
                dataset_id=dataset_id,
                call_id=call_id,
                definition=definition,
                arguments=arguments,
                random_seed=seed,
                input_revision_id=plan.input_revision_id,
            )
            return result.model_dump(mode="json")

        payload = self.gate.execute_once(outcome, run_id=run_id, executor=invoke)
        result = ToolResult.model_validate_json(json.dumps(payload))
        next_node = (
            "completed" if result.status in {ToolStatus.SUCCESS, ToolStatus.WARNING} else "failed"
        )
        checkpoint = self.store.save_checkpoint(
            run_id,
            self._checkpoint_state(
                plan, dataset_id, step.tool_name, next_node, result, conversation=conversation_state
            ),
            next_node,
        )
        return PlanExecutionResponse(
            disposition=outcome.disposition,
            approval_id=outcome.approval.id if outcome.approval else None,
            tool_result=result,
            checkpoint_id=checkpoint.id,
        )

    @staticmethod
    def _checkpoint_state(
        plan: ExecutionPlan,
        dataset_id: str,
        tool_name: str,
        phase: str,
        result: ToolResult | None = None,
        *,
        conversation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "plan": plan.model_dump(mode="json"),
            "dataset_id": dataset_id,
            "tool_name": tool_name,
            "phase": phase,
        }
        if result is not None:
            state["tool_result"] = result.model_dump(mode="json")
        if conversation is not None:
            state["conversation"] = conversation
        return state
