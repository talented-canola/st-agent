"""Typed intent and deterministic scientific-plan contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from st_agent.contracts import Effect, StrictRecord
from st_agent.dataset_state import DatasetState
from st_agent.memory_runtime import MemoryDirective
from st_agent.research_context import ResearchGoalUpdate
from st_agent.evidence_context import EvidenceSelection
from st_agent.answer_delivery import ResponseRequirements


class TaskOperation(StrEnum):
    RESEARCH_GOAL = "research_goal"
    BASIC_ANALYSIS = "basic_analysis"
    INSPECT = "inspect"
    QC_SUMMARY = "qc_summary"
    QC_FILTER = "qc_filter"
    PREPROCESS = "preprocess"
    CLUSTER = "cluster"
    MARKER = "marker"
    SPATIAL_NEIGHBORS = "spatial_neighbors"
    SPATIAL_MORANS = "spatial_morans"
    COORDINATE_PARTITION = "coordinate_partition"
    SPATIAL_DOMAINS = "spatial_domains"
    SAMPLE_COMPARISON = "sample_comparison"
    EXPLAIN = "explain"
    MEMORY = "memory"


class IntentMode(StrEnum):
    ASK = "ask"
    PROPOSE = "propose"
    EXECUTE = "execute"


class DeferredDelivery(StrictRecord):
    """Output requested after the current/pending action completes.

    This is independent of the action continuation: stating that a tool should
    run (or continue) does not erase a requested explanation, selection or
    advice. It carries no execution permission.
    """
    quote: str = Field(min_length=1, max_length=4000, description=(
        "Verbatim substring of the CURRENT user request that states the output "
        "wanted after execution. Never a summary."))
    response_requirements: ResponseRequirements = Field(description=(
        "Requested answer mode/questions after the action completes. Receipt is "
        "not a deferred delivery."))
    saved_evidence: EvidenceSelection | None = Field(default=None, description=(
        "Entities explicitly selected by the current delivery request. Omit when "
        "the user did not newly select entities."))


class TaskIntent(StrictRecord):
    operation: TaskOperation
    mode: IntentMode
    parameters: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    clarification_needed: bool = False
    rationale: str = Field(min_length=1)
    memory: MemoryDirective | None = None
    use_memory: bool = False
    new_analysis: bool = False
    research_goal: ResearchGoalUpdate | None = None
    saved_evidence: EvidenceSelection | None = None
    response_requirements: ResponseRequirements = Field(default_factory=ResponseRequirements)
    deferred_delivery: DeferredDelivery | None = Field(default=None, description=(
        "Set only when the current user asks for an explanation/selection/advice "
        "that can only be delivered after the pending or newly approved action. "
        "Operation/mode/parameters describe the action; this field describes the "
        "requested output and must not be collapsed into receipt."))


class PlanStep(StrictRecord):
    tool_name: str
    effect: Effect
    parameters: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool


class ExecutionPlan(StrictRecord):
    input_revision_id: str
    state: DatasetState
    steps: tuple[PlanStep, ...] = ()
    required_approval: bool = False
    expected_outputs: tuple[str, ...] = ()
    blocked_reason: str | None = None


class DeterministicPlanner:
    """Local authority for preconditions and mutation policy."""

    def build(self, intent: TaskIntent, state: DatasetState) -> ExecutionPlan:
        if intent.clarification_needed:
            return self._blocked(state, "intent_requires_clarification")
        if intent.operation is TaskOperation.EXPLAIN:
            return ExecutionPlan(input_revision_id=state.input_revision_id, state=state)
        if intent.mode is not IntentMode.EXECUTE:
            return ExecutionPlan(input_revision_id=state.input_revision_id, state=state)
        if intent.operation is TaskOperation.INSPECT:
            return self._read_only(state, "dataset.inspect")
        if intent.operation is TaskOperation.QC_SUMMARY:
            return self._read_only(state, "qc.compute_metrics")
        if intent.operation is TaskOperation.QC_FILTER:
            if not {"min_genes", "min_cells", "max_mito_pct"} <= intent.parameters.keys():
                return self._blocked(state, "filter_requires_explicit_thresholds")
            return self._transform(state, intent, "qc.apply_filter", needs_raw_counts=True)
        if intent.operation is TaskOperation.PREPROCESS:
            return self._transform(state, intent, "preprocess.standard", needs_raw_counts=True)
        if intent.operation is TaskOperation.CLUSTER:
            if not state.has_pca:
                return self._blocked(state, "cluster_requires_pca")
            return self._transform(state, intent, "cluster.leiden_umap")
        if intent.operation is TaskOperation.MARKER:
            if not state.has_leiden:
                return self._blocked(state, "marker_requires_leiden")
            return self._transform(state, intent, "markers.rank")
        spatial_tools = {
            TaskOperation.SPATIAL_NEIGHBORS: "spatial.neighbors",
            TaskOperation.SPATIAL_MORANS: "spatial.morans_i",
            TaskOperation.COORDINATE_PARTITION: "spatial.domains",
            TaskOperation.SPATIAL_DOMAINS: "spatial.expression_domains",
            TaskOperation.SAMPLE_COMPARISON: "samples.compare_stages",
        }
        if intent.operation in spatial_tools:
            name = spatial_tools[intent.operation]
            from st_agent.tool_registry import SCIENTIFIC_TOOLS
            from pydantic import ValidationError
            import json
            try:
                values = SCIENTIFIC_TOOLS[name].arguments.model_validate_json(
                    json.dumps(intent.parameters)).model_dump(mode="json")
            except ValidationError:
                return self._blocked(state, "spatial_parameters_invalid_or_missing")
            if name.startswith("spatial.") and not state.has_spatial_coordinates:
                return self._blocked(state, "spatial_requires_finite_coordinates")
            if name == "spatial.expression_domains" and not (state.has_pca and state.has_normalization):
                return self._blocked(state, "expression_domains_require_preprocessed_pca")
            if values.get("groupby") and values["groupby"] not in state.obs_columns:
                return self._blocked(state, "spatial_grouping_column_missing")
            if name in {"spatial.neighbors", "spatial.expression_domains"}:
                sections = state.analysis_context.get("sections", {})
                sizes = sections.get("sizes", {})
                if sizes and values.get("groupby") in {None, sections.get("key")} and (
                    values["n_neighbors"] >= min(sizes.values())):
                    return self._blocked(state, "spatial_neighbors_exceed_smallest_section")
            if name == "spatial.morans_i" and not state.has_spatial_neighbors:
                return self._blocked(state, "morans_requires_spatial_neighbors")
            if name == "samples.compare_stages":
                # Descriptive grouping does not require donor/replicate metadata.
                # Missing/empty groups are validated by the scientific kernel.
                if not values["stage_column"].strip() or values["stage_column"] not in state.obs_columns:
                    return self._blocked(state, "comparison_requires_confirmed_group_metadata")
                if values.get("cluster_key") and not values["labels_aligned"]:
                    return self._blocked(state, "comparison_requires_explicitly_aligned_cluster_labels")
                if values.get("cluster_key") and values["cluster_key"] not in state.obs_columns:
                    return self._blocked(state, "comparison_cluster_labels_missing")
            tool = SCIENTIFIC_TOOLS[name]
            return ExecutionPlan(input_revision_id=state.input_revision_id, state=state,
                steps=(PlanStep(tool_name=name, effect=tool.effect, parameters=values,
                    requires_approval=tool.effect is Effect.DERIVED_REVISION),),
                required_approval=tool.effect is Effect.DERIVED_REVISION,
                expected_outputs=("ToolResult", "derived_revision") if tool.effect is Effect.DERIVED_REVISION
                    else ("ToolResult",))
        return self._blocked(state, "operation_is_explanation_only")

    @staticmethod
    def _read_only(state: DatasetState, tool_name: str) -> ExecutionPlan:
        return ExecutionPlan(
            input_revision_id=state.input_revision_id,
            state=state,
            steps=(
                PlanStep(tool_name=tool_name, effect=Effect.READ_ONLY, requires_approval=False),
            ),
            expected_outputs=("ToolResult",),
        )

    def _transform(
        self,
        state: DatasetState,
        intent: TaskIntent,
        tool_name: str,
        *,
        needs_raw_counts: bool = False,
    ) -> ExecutionPlan:
        if needs_raw_counts and not state.is_raw_count_matrix:
            return self._blocked(state, "selected_revision_has_no_trustworthy_count_matrix")
        return ExecutionPlan(
            input_revision_id=state.input_revision_id,
            state=state,
            steps=(
                PlanStep(
                    tool_name=tool_name,
                    effect=Effect.DERIVED_REVISION,
                    parameters=intent.parameters,
                    requires_approval=True,
                ),
            ),
            required_approval=True,
            expected_outputs=("ToolResult", "derived_revision"),
        )

    @staticmethod
    def _blocked(state: DatasetState, reason: str) -> ExecutionPlan:
        return ExecutionPlan(
            input_revision_id=state.input_revision_id,
            state=state,
            blocked_reason=reason,
        )
