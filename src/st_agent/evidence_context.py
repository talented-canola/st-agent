"""Small, local evidence contracts and deterministic marker-record expansion.

Values remain in the original fact table. Groups connect facts without making
scientific judgments, inferring provenance or authorizing any computation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

Scalar = StrictStr | StrictInt | StrictFloat | StrictBool | None
Metric = Literal[
    "logfoldchange", "p_adjusted", "pct_in", "pct_out", "detection_difference", "score", "p_value"
]
METRICS = (
    "logfoldchange",
    "p_adjusted",
    "pct_in",
    "pct_out",
    "detection_difference",
    "score",
    "p_value",
)
SOURCE_METADATA_FIELDS = (
    "species", "tissue", "developmental_stage", "technology", "donor_count", "sample_id",
    "biological_setting", "processing", "runtime_input_kind", "runtime_count_layer", "primary_url",
)
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{64}$")]
_MARKER = re.compile(r"^marker\.([^.]+)\.([1-9][0-9]*)\.(gene|" + "|".join(METRICS) + r")$")


class EvidenceError(ValueError):
    """A global or local evidence contract failed; never repair scientific values."""


class StrictEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class StoredFact(StrictEvidence):
    id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    value: Scalar
    revision_id: str | None
    label: str
    show: bool = True
    scope: Literal["current", "upstream"] | None = None
    source_call_id: str | None = None
    input_revision_id: str | None = None
    calculation: str | None = None

    @field_validator("value")
    @classmethod
    def finite(cls, value):
        if isinstance(value, float) and not math.isfinite(value):
            raise EvidenceError("evidence.nonfinite_fact")
        return value


class FactProvenance(StrictEvidence):
    kind: Literal["tool_result", "registered_artifact", "revision_metadata"]
    dataset_id: str = Field(min_length=1)
    evidence_revision_id: str = Field(min_length=1)
    source_call_id: str | None = None
    tool_name: str | None = None
    tool_version: str | None = None
    input_revision_id: str | None = None
    output_revision_id: str | None = None
    artifact_id: str | None = None
    artifact_sha256: Sha256 | None = None
    source_payload_sha256: Sha256 | None = None
    record_key: str


class MarkerComparison(StrictEvidence):
    kind: Literal["cluster_vs_rest", "explicit_groups", "unknown"] = "unknown"
    target_group: str
    reference_groups: list[str] = Field(default_factory=list)


class MarkerSemantics(StrictEvidence):
    observation_unit: Literal["cell", "spot", "observation", "unknown"] = "unknown"
    test_method: str | None = None
    effect_expression_source: str | None = None
    detection_expression_source: str | None = None
    expression_transform: str | None = None
    effect_scale: str | None = None
    interpretation_scope: Literal[
        "exploratory_cluster_marker", "other_recorded_scope", "unknown"
    ] = "unknown"


class MarkerEvidenceGroup(StrictEvidence):
    group_id: str
    dataset_id: str = Field(min_length=1)
    revision_id: str = Field(min_length=1)
    scope: Literal["current", "upstream"]
    gene: str
    cluster_id: str
    rank: int = Field(gt=0)
    groupby: str | None = None
    comparison: MarkerComparison
    semantics: MarkerSemantics
    metric_fact_ids: dict[Metric, str]
    metric_semantics: dict[Metric, str]
    source: FactProvenance
    gene_fact_id: str


class MarkerSelection(StrictEvidence):
    """Model-selectable entities only; host binds project and revision separately."""

    genes: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list, max_length=32
    )
    clusters: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list, max_length=32
    )
    comparison_kind: Literal["cluster_vs_rest", "explicit_groups", "unknown"] = "unknown"
    fields: list[Metric] = Field(default_factory=lambda: list(METRICS), max_length=7)
    limit: int = Field(default=8, ge=1, le=32)

    @field_validator("genes", "clusters", "fields")
    @classmethod
    def unique_selection(cls, value):
        if len(set(value)) != len(value):
            raise EvidenceError("evidence.duplicate_selection")
        return value


class EvidenceRequest(MarkerSelection):
    dataset_id: str = Field(min_length=1)
    revision_id: str = Field(min_length=1)


def saved_evidence_contracts():
    """One source for local validation and provider-visible kind constraints."""
    qc = {"observations", "pct_counts_mt"} | {
        metric + "/" + statistic
        for metric in ("n_genes_by_counts", "total_counts", "pct_counts_mt")
        for statistic in ("min", "q1", "median", "q3", "max", "iqr")}
    spatial = {"morans_i", "mean_expression", "fraction_expressing", "count", "fraction",
        "observations", "samples", "replicates", "qc/mean_total_counts",
        "qc/mean_detected_genes", "qc/mean_mito_pct"}
    return {
        "section_composition": {"fields": {"count"}, "empty_entities": {"genes", "artifact_ids"}},
        "qc_groups": {"fields": qc, "empty_entities": {"genes", "artifact_ids", "clusters"}},
        "figure_refs": {"fields": set(), "empty_entities": {"genes", "groupby", "groups", "clusters"}},
        "spatial_results": {"fields": spatial, "empty_entities": {"artifact_ids"}},
        "source_metadata": {"fields": set(SOURCE_METADATA_FIELDS),
            "empty_entities": {"genes", "artifact_ids", "groupby", "groups", "clusters"}},
        # Public input tables have no fixed scientific column vocabulary. Only
        # registered artifact IDs and bounded literal column names are allowed.
        "public_table": {"fields": None, "empty_entities": {"genes", "clusters"},
            "require_artifact_ids": True},
    }


class SavedEvidenceSelection(StrictEvidence):
    """Entity selection only. Project, revision and file paths are host-bound."""

    kind: Literal["section_composition", "qc_groups", "figure_refs", "spatial_results",
        "source_metadata", "public_table"]
    groupby: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    groups: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(default_factory=list, max_length=32)
    clusters: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(default_factory=list, max_length=32)
    genes: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(default_factory=list, max_length=32)
    fields: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(default_factory=list, max_length=32)
    artifact_ids: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(default_factory=list, max_length=32)
    limit: int = Field(default=32, ge=1, le=128, description=(
        "Bound on emitted evidence. For public_table this is the total number of "
        "scalar facts (selected cell values plus group-count facts), so it is not "
        "a row cap. For other kinds it caps saved records or figure references."))

    @model_validator(mode="after")
    def validate_kind(self):
        for name in ("groups", "clusters", "genes", "fields", "artifact_ids"):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise EvidenceError("evidence.duplicate_selection")
        contract = saved_evidence_contracts()[self.kind]
        if contract["fields"] is not None and set(self.fields) - contract["fields"]:
            raise EvidenceError("evidence.selection_field_invalid")
        if any(getattr(self, name) for name in contract["empty_entities"]):
            raise EvidenceError("evidence.selection_entity_invalid")
        if self.kind == "public_table":
            if contract.get("require_artifact_ids") and not self.artifact_ids:
                raise EvidenceError("evidence.public_table_artifact_required")
            if not self.fields:
                raise EvidenceError("evidence.public_table_field_required")
            # Values require a grouping column; an empty value selection reads all groups.
            if self.groups and not self.groupby:
                raise EvidenceError("evidence.public_table_groups_pair")
        return self


def saved_evidence_provider_schema():
    """Expand after-validator constraints without changing the stored model API."""
    base = SavedEvidenceSelection.model_json_schema()
    descriptions = {
        "spatial_results": (
            "Read saved spatial.* or samples.* result records only. This kind is not "
            "for markers.rank rows; marker discussions use the separate no-kind "
            "MarkerSelection. Keep a requested spatial scope even when no matching "
            "saved result is supplied; never silently convert it to a marker selection."),
        "section_composition": "Read saved section/cell composition counts only.",
        "qc_groups": "Read saved QC group summaries only.",
        "figure_refs": "Read saved figure references only; pixels require a separate image selection.",
        "source_metadata": "Read registered source-description metadata only; not a measurement.",
        "public_table": "Read one registered public input table with declared columns only.",
    }
    branches = []
    for kind, contract in saved_evidence_contracts().items():
        branch = deepcopy(base)
        branch.update(title="SavedEvidenceSelection_" + kind,
            description=descriptions.get(kind, "Read existing " + kind + " evidence only; "
                        "fields and entities are kind-specific."))
        properties = branch["properties"]
        properties["kind"] = {"type": "string", "const": kind}
        if contract["fields"]:
            properties["fields"]["items"]["enum"] = sorted(contract["fields"])
        elif contract["fields"] is None:
            # Public-table columns are declared per artifact, not a global enum.
            properties["fields"]["minItems"] = 1
        else:
            properties["fields"]["maxItems"] = 0
        for name in ("groups", "clusters", "genes", "fields", "artifact_ids"):
            properties[name]["uniqueItems"] = True
        if contract.get("require_artifact_ids"):
            properties["artifact_ids"]["minItems"] = 1
        for name in contract["empty_entities"]:
            if name == "groupby":
                properties[name] = {"type": "null", "default": None}
            else:
                properties[name]["maxItems"] = 0
        branches.append(branch)
    return {"title": "SavedEvidenceSelection", "anyOf": branches}


EvidenceSelection = MarkerSelection | SavedEvidenceSelection


class EvidenceAvailability(StrictEvidence):
    evidence_key: str
    group_id: str | None = None
    metric: Metric | None = None
    status: Literal["supplied", "partial", "saved_not_selected", "not_computed", "unavailable", "unknown"]
    source_ref: str | None = None
    reason_code: str | None = None
    # Typed reason for a saved numeric null, only when the same validated saved
    # result carries an explicit machine-readable warning. Unknown stays None.
    null_reason: Literal["constant_expression"] | None = None


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build_fact_index(bundle) -> dict[str, dict]:
    """Validate both tables before indexing; do not overwrite duplicate IDs.

    The caller retains its committed-lineage/call checks. This boundary also
    checks historical revision membership when a lineage was supplied.
    """
    identity = bundle.get("identity", {})
    revision = identity.get("revision_id")
    if "sha256" in identity and (
        not isinstance(identity["sha256"], str)
        or not re.fullmatch(r"[0-9a-fA-F]{64}", identity["sha256"])
    ):
        raise EvidenceError("evidence.identity_hash_invalid")
    lineage = bundle.get("research_context", {}).get("lineage", [])
    allowed = {r["revision_id"] for r in lineage} | {revision}
    records = {}
    for table in ("facts", "historical_facts"):
        for raw in bundle.get(table, []):
            fact = StoredFact.model_validate(raw)
            if fact.id in records:
                raise EvidenceError("evidence.duplicate_fact")
            if table == "facts" and fact.revision_id != revision:
                raise EvidenceError("evidence.current_revision_mismatch")
            if table == "historical_facts" and lineage and fact.revision_id not in allowed:
                raise EvidenceError("evidence.historical_revision_mismatch")
            records[fact.id] = dict(raw)  # Preserve omitted fields and exact scalar types.
    return records


def _source(bundle, fact, record_key):
    identity = bundle.get("identity", {})
    explicit = bundle.get("fact_provenance", {}).get(fact["id"])
    if explicit is not None:
        source = FactProvenance.model_validate(explicit).model_dump()
        if (
            source["dataset_id"] != identity.get("dataset_id", "")
            or source["evidence_revision_id"] != fact["revision_id"]
        ):
            raise EvidenceError("evidence.provenance_identity_mismatch")
        if fact.get("source_call_id") and source["source_call_id"] != fact["source_call_id"]:
            raise EvidenceError("evidence.provenance_call_mismatch")
        if source["record_key"] == fact["key"]:
            source["record_key"] = record_key
        return source
    call = fact.get("source_call_id")
    observations = bundle.get("research_context", {}).get("tool_observations", [])
    observation = next((o for o in observations if call and o.get("call_id") == call), {})
    return FactProvenance(
        kind="tool_result" if call else "revision_metadata",
        dataset_id=identity.get("dataset_id", ""),
        evidence_revision_id=fact["revision_id"],
        source_call_id=call,
        tool_name=observation.get("tool_name"),
        tool_version=observation.get("tool_version"),
        input_revision_id=observation.get("input_revision_id", fact.get("input_revision_id")),
        output_revision_id=observation.get("output_revision_id"),
        record_key=record_key,
    ).model_dump()


_SEMANTICS = {
    "logfoldchange": (
        "Source-reported expression effect; not detection-frequency difference. "
        "Scale only as recorded."
    ),
    "p_adjusted": (
        "Source-reported adjusted P value; stored 0.0 is not proof of "
        "mathematical zero probability."
    ),
    "p_value": "Source-reported unadjusted P value; not an expression effect size.",
    "pct_in": "Detection frequency in the target group, not expression intensity.",
    "pct_out": "Detection frequency in the reference group, not expression intensity.",
    "detection_difference": (
        "Target minus reference detection frequency; cannot replace expression effect."
    ),
    "score": "Source-reported test statistic; meaning depends on the recorded test method.",
}


def project_marker_groups(groups):
    """Share identical metric definitions once; retain every source and fact link."""
    projected, semantics = [], {}
    for group in groups:
        for metric, meaning in group["metric_semantics"].items():
            if metric in semantics and semantics[metric] != meaning:
                raise EvidenceError("evidence.metric_semantics_conflict")
            semantics[metric] = meaning
        projected.append({key: value for key, value in group.items() if key != "metric_semantics"})
    return projected, semantics


def _group_metadata(bundle, source, cluster, rank, gene):
    metadata = bundle.get("marker_metadata", {})
    by_call = metadata.get("by_call", {})
    if source["source_call_id"] in by_call:
        metadata = by_call[source["source_call_id"]]
    # Global metadata applies only to its recorded revision, never upstream data.
    same_revision = (
        metadata.get("revision_id", bundle.get("identity", {}).get("revision_id"))
        == source["evidence_revision_id"]
    )
    if not same_revision:
        metadata = {}
    quality = bundle.get("quality", {}) if same_revision else {}
    warning_records = metadata.get("warnings", quality.get("warnings", []))
    scope = next(
        (
            w.get("evidence", {})
            for w in warning_records
            if isinstance(w, dict) and w.get("code") == "marker_test_scope"
        ),
        {},
    )
    markers = metadata.get("markers", quality.get("markers", {}))
    row = next(
        (r for r in markers.get(cluster, []) if r.get("rank") == rank and r.get("gene") == gene), {}
    )
    kind = metadata.get("comparison_kind", "unknown")
    if metadata.get("contrast") == "each_expression_cluster_vs_rest" or scope:
        kind = "cluster_vs_rest"
    semantics = MarkerSemantics(
        observation_unit=metadata.get("observation_unit", "unknown"),
        test_method=metadata.get("method", scope.get("method")),
        effect_expression_source=metadata.get("representation", scope.get("representation")),
        detection_expression_source=row.get("detection_source"),
        expression_transform=metadata.get("expression_scale"),
        effect_scale=metadata.get("effect_scale"),
        interpretation_scope=(
            "exploratory_cluster_marker"
            if scope or metadata.get("contrast") == "each_expression_cluster_vs_rest"
            else "unknown"
        ),
    )
    comparison = MarkerComparison(
        kind=kind, target_group=cluster, reference_groups=metadata.get("reference_groups", [])
    )
    return metadata.get("groupby", scope.get("groupby")), comparison, semantics


def build_marker_groups(bundle) -> list[dict]:
    """Group controlled marker keys by revision and exact source, never gene alone."""
    records = build_fact_index(bundle)
    rows = {}
    for fact in records.values():
        match = _MARKER.fullmatch(fact["key"])
        if not match:
            continue
        cluster, rank_text, field = match.groups()
        prefix = f"marker.{cluster}.{rank_text}"
        source = _source(bundle, fact, prefix)
        key = (fact["revision_id"], cluster, int(rank_text), _digest(source))
        row = rows.setdefault(key, {"source": source, "fields": {}})
        if field in row["fields"]:
            raise EvidenceError("evidence.duplicate_marker_field")
        value = fact["value"]
        if field == "gene":
            if not isinstance(value, str) or not value:
                raise EvidenceError("evidence.marker_gene_type")
        elif value is not None:
            if type(value) not in (int, float):
                raise EvidenceError("evidence.marker_metric_type")
            low, high = (
                (0, 1)
                if field in {"pct_in", "pct_out", "p_adjusted", "p_value"}
                else (-1, 1)
                if field == "detection_difference"
                else (None, None)
            )
            if low is not None and not low <= value <= high:
                raise EvidenceError("evidence.marker_metric_range")
        row["fields"][field] = fact
    groups = []
    for (revision, cluster, rank, _), row in rows.items():
        fields, source = row["fields"], row["source"]
        if "gene" not in fields:
            continue  # Never recover a gene by looking at another source's record.
        values = {field: f["value"] for field, f in fields.items()}
        if all(
            type(values.get(k)) in (int, float)
            for k in ("pct_in", "pct_out", "detection_difference")
        ) and not math.isclose(
            values["pct_in"] - values["pct_out"],
            values["detection_difference"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise EvidenceError("evidence.marker_detection_conflict")
        gene = values["gene"]
        groupby, comparison, semantics = _group_metadata(bundle, source, cluster, rank, gene)
        group_id = (
            "marker_group_"
            + _digest([source, revision, cluster, rank, gene, comparison.model_dump()])[:32]
        )
        groups.append(
            MarkerEvidenceGroup(
                group_id=group_id,
                dataset_id=source["dataset_id"],
                revision_id=revision,
                scope="current" if revision == bundle["identity"]["revision_id"] else "upstream",
                gene=gene,
                cluster_id=cluster,
                rank=rank,
                groupby=groupby,
                comparison=comparison,
                semantics=semantics,
                metric_fact_ids={k: f["id"] for k, f in fields.items() if k != "gene"},
                metric_semantics={k: _SEMANTICS[k] for k in fields if k != "gene"},
                source=source,
                gene_fact_id=fields["gene"]["id"],
            ).model_dump()
        )
    return sorted(groups, key=lambda g: g["group_id"])


def expand_candidate_facts(selected_ids, bundle, records) -> dict:
    """Add already validated siblings without changing the model's selected IDs."""
    validated = build_fact_index(bundle)
    for fid, fact in validated.items():
        if (
            fid not in records
            or records[fid] != fact
            or type(records[fid]["value"]) is not type(fact["value"])
        ):
            raise EvidenceError("evidence.fact_index_mismatch")
    selected = list(dict.fromkeys(selected_ids))
    if any(fid not in validated for fid in selected):
        raise EvidenceError("evidence.unknown_fact")
    groups, expanded, reasons = [], [], {}
    for group in build_marker_groups(bundle):
        members = [group["gene_fact_id"], *group["metric_fact_ids"].values()]
        if not set(selected).intersection(members):
            continue
        groups.append(group)
        for fid in members:
            if fid not in selected and fid not in expanded:
                expanded.append(fid)
                reasons[fid] = "same_marker_record:" + group["group_id"]
    return {
        "facts": [validated[fid] for fid in [*selected, *expanded]],
        "expanded_fact_ids": expanded,
        "evidence_group_ids": [g["group_id"] for g in groups],
        "evidence_groups": groups,
        "expansion_reasons": reasons,
    }


def _checked_path(root, relative, expected_hash, *, artifact=False):
    from st_agent.utils import sha256_file

    code = "artifact" if artifact else "revision"
    relative = Path(relative or "")
    if relative.is_absolute() or ".." in relative.parts:
        raise EvidenceError(f"evidence.{code}_path_invalid")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or (
        artifact and (not relative.parts or relative.parts[0] != "artifacts")
    ):
        raise EvidenceError(f"evidence.{code}_path_invalid")
    if not path.is_file():
        raise EvidenceError(f"evidence.{code}_unavailable")
    if (
        not isinstance(expected_hash, str)
        or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash)
        or sha256_file(path) != expected_hash
    ):
        raise EvidenceError(f"evidence.{code}_hash_mismatch")
    return path


def _value(value):
    return getattr(value, "value", value)


def _requested(row, cluster, request):
    return (not request.genes or row.get("gene") in request.genes) and (
        not request.clusters or cluster in request.clusters
    )


def _csv_rows(copilot, request, root):
    """Read only registered same-revision marker tables, never a model path."""
    columns = {
        "logfoldchanges": "logfoldchange",
        "pvals_adj": "p_adjusted",
        "scores": "score",
        "pvals": "p_value",
        "pct_nz_group": "pct_in",
        "pct_nz_reference": "pct_out",
    }
    output = []
    artifacts = sorted(copilot.list_artifacts(), key=lambda a: a.id)
    for artifact in artifacts:
        if (
            artifact.kind != "table.markers"
            or artifact.metadata.get("dataset_id") != request.dataset_id
            or artifact.metadata.get("revision_id") != request.revision_id
        ):
            continue
        if artifact.project_id != copilot.project.id:
            raise EvidenceError("evidence.artifact_project_mismatch")
        path = _checked_path(root, artifact.relative_path, artifact.sha256, artifact=True)
        if path.suffix.lower() != ".csv" or path.stat().st_size > 16 * 1024 * 1024:
            raise EvidenceError("evidence.artifact_read_limit")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != artifact.sha256:
            raise EvidenceError("evidence.artifact_hash_mismatch")
        with io.StringIO(payload.decode("utf-8-sig"), newline="") as stream:
            reader = csv.DictReader(stream)
            if (
                not reader.fieldnames
                or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not {"group", "names"} <= set(reader.fieldnames)
            ):
                raise EvidenceError("evidence.artifact_marker_schema")
            ranks = {}
            for number, raw in enumerate(reader):
                if None in raw or raw.get("group") is None or raw.get("names") is None:
                    raise EvidenceError("evidence.artifact_marker_schema")
                cluster = raw["group"]
                ranks[cluster] = ranks.get(cluster, 0) + 1
                # The registered export preserves rank order within each group.
                rank = ranks[cluster]
                if raw.get("rank"):
                    try:
                        rank = int(raw["rank"])
                    except ValueError as error:
                        raise EvidenceError("evidence.artifact_marker_rank") from error
                if rank < 1:
                    raise EvidenceError("evidence.artifact_marker_rank")
                row = {"gene": raw["names"], "rank": rank}
                if not _requested(row, cluster, request):
                    continue
                for column, field in columns.items():
                    if raw.get(column) not in (None, ""):
                        try:
                            row[field] = float(raw[column])
                        except ValueError as error:
                            raise EvidenceError("evidence.artifact_marker_numeric") from error
                source = FactProvenance(
                    kind="registered_artifact",
                    dataset_id=request.dataset_id,
                    evidence_revision_id=request.revision_id,
                    artifact_id=artifact.id,
                    artifact_sha256=artifact.sha256,
                    record_key=f"csv_row:{number}",
                ).model_dump()
                output.append((cluster, row, source))
    return output


def read_marker_evidence(copilot, request: EvidenceRequest | dict) -> dict:
    """Read committed results first, registered CSV only when required.

    ``fields`` describes desired information, not permission to split a saved
    record. Every returned record carries all already saved relevant metrics.
    The function never writes, computes expression statistics or starts workers.
    Different sources remain separate groups even when they mention one gene.
    """
    request = EvidenceRequest.model_validate(request)
    dataset = copilot.get_dataset(request.dataset_id)
    if dataset.project_id != copilot.project.id:
        raise EvidenceError("evidence.dataset_project_mismatch")
    active_before = dataset.active_revision_id
    revision = copilot.revisions.get(request.revision_id)
    if (
        revision.id != request.revision_id
        or revision.dataset_id != request.dataset_id
        or _value(revision.status) != "committed"
    ):
        raise EvidenceError("evidence.revision_identity_mismatch")
    root = copilot.layout.root.resolve()
    _checked_path(root, revision.relative_path, revision.sha256)
    result = copilot.tool_executions.find_by_call(revision.call_id) if revision.call_id else None
    metrics = revision.metadata.get("worker_product", {}).get("metrics", {})
    source = FactProvenance(
        kind="revision_metadata",
        dataset_id=request.dataset_id,
        evidence_revision_id=request.revision_id,
        record_key="quality.markers",
    ).model_dump()
    if result is not None:
        if (
            result.call_id != revision.call_id
            or result.output_revision_id != revision.id
            or result.input_revision_id != revision.parent_revision_id
            or _value(result.status) not in {"success", "warning"}
            or result.provenance.dataset_hash != dataset.sha256
        ):
            raise EvidenceError("evidence.tool_result_revision_mismatch")
        metrics = result.metrics
        source.update(
            kind="tool_result",
            source_call_id=result.call_id,
            tool_name=result.tool_name,
            tool_version=result.provenance.tool_version,
            input_revision_id=result.input_revision_id,
            output_revision_id=result.output_revision_id,
        )
    quality = metrics.get("quality", {})
    source["source_payload_sha256"] = _digest(quality)
    marker_metadata = {
        "revision_id": request.revision_id,
        "warnings": quality.get("warnings", []),
        "markers": quality.get("markers", {}),
    }
    # Only copy semantics actually recorded by the persisted result.
    for key in ("expression_scale", "contrast"):
        if key in metrics:
            marker_metadata[key] = metrics[key]
    if (
        result is not None
        and result.tool_name == "markers.rank"
        and "method" in result.provenance.parameters
    ):
        marker_metadata["method"] = result.provenance.parameters["method"]
    rows = []
    for cluster, markers in quality.get("markers", {}).items():
        for row in markers:
            if not _requested(row, cluster, request):
                continue
            rank = row.get("rank")
            if type(rank) is not int or rank < 1:
                raise EvidenceError("evidence.marker_rank_invalid")
            provenance = {**source, "record_key": f"quality.markers.{cluster}.{rank}"}
            rows.append((cluster, row, provenance))
    missing_fields = any(
        any(row.get(field) is None for field in request.fields) for _, row, _ in rows
    )
    matched_genes = {row["gene"] for _, row, _ in rows}
    if not rows or missing_fields or (request.genes and not set(request.genes) <= matched_genes):
        # Keep source records independent; no unproved quality/CSV field joining.
        rows.extend(_csv_rows(copilot, request, root))
    # A supplementary CSV record must not be pushed past the limit by the
    # incomplete overview record that triggered its lookup. Ties retain the
    # original quality-first order; no statistical values are changed.
    rows.sort(key=lambda record: -sum(record[1].get(field) is not None for field in request.fields))
    facts, provenance, availability = [], {}, []
    for index, (cluster, row, source) in enumerate(rows):
        key = f"marker.{cluster}.{row['rank']}"
        supplied = index < request.limit
        for field in ["gene", *METRICS]:
            if field not in row:
                continue
            fid = "artifact_fact_" + _digest([source, key, field])[:32]
            if supplied:
                fact = {
                    "id": fid,
                    "key": key + "." + field,
                    "value": row[field],
                    "revision_id": request.revision_id,
                    "label": f"{row['gene']} {field}",
                    "show": True,
                }
                if source["source_call_id"]:
                    fact["source_call_id"] = source["source_call_id"]
                facts.append(fact)
                provenance[fid] = source
            if field != "gene":
                status = "supplied" if supplied else "saved_not_selected"
                reason = None if supplied else "evidence.record_limit"
                if row[field] is None:
                    status, reason = "unknown", "evidence.metric_not_saved"
                availability.append(
                    EvidenceAvailability(
                        evidence_key=key + "." + field,
                        metric=field,
                        status=status,
                        source_ref=source["artifact_id"] or source["source_call_id"],
                        reason_code=reason,
                    ).model_dump()
                )
        if supplied:
            for field in request.fields:
                if field not in row:
                    availability.append(
                        EvidenceAvailability(
                            evidence_key=key + "." + field,
                            metric=field,
                            status="unknown",
                            source_ref=source["artifact_id"] or source["source_call_id"],
                            reason_code="evidence.metric_not_saved",
                        ).model_dump()
                    )
    if not rows:
        availability.append(
            EvidenceAvailability(
                evidence_key="markers", status="unknown", reason_code="evidence.request_not_found"
            ).model_dump()
        )
    bundle = {
        "identity": {
            "dataset_id": request.dataset_id,
            "revision_id": request.revision_id,
            "sha256": revision.sha256,
        },
        "facts": facts,
        "fact_provenance": provenance,
        "marker_metadata": marker_metadata,
    }
    groups = build_marker_groups(bundle)
    if request.comparison_kind != "unknown" and any(
        g["comparison"]["kind"] != request.comparison_kind for g in groups
    ):
        raise EvidenceError("evidence.comparison_not_verified")
    for item in availability:
        source_groups = [
            g
            for g in groups
            if item["evidence_key"].startswith(f"marker.{g['cluster_id']}.{g['rank']}.")
            and item["source_ref"] in (g["source"]["artifact_id"], g["source"]["source_call_id"])
        ]
        if len(source_groups) == 1:
            item["group_id"] = source_groups[0]["group_id"]
    if copilot.get_dataset(request.dataset_id).active_revision_id != active_before:
        raise EvidenceError("evidence.input_revision_changed")
    _checked_path(root, revision.relative_path, revision.sha256)
    return {
        "facts": facts,
        "evidence_groups": groups,
        "availability": availability,
        "fact_provenance": provenance,
        "marker_metadata": marker_metadata,
    }


def _saved_sources(copilot, dataset, revision):
    """Same-revision results only; persisted metadata is a fallback, not a join."""
    sources, seen = [], set()
    for result in copilot.tool_executions.list_for_dataset(copilot.project.id, dataset.id):
        transforming = result.output_revision_id not in (None, result.input_revision_id)
        bound = result.output_revision_id if transforming else result.input_revision_id
        if bound != revision.id or _value(result.status) not in {"success", "warning"}:
            continue
        if result.provenance.dataset_hash != dataset.sha256 or (
            transforming and (result.call_id != revision.call_id or
                              result.input_revision_id != revision.parent_revision_id)):
            raise EvidenceError("evidence.tool_result_revision_mismatch")
        if result.tool_name in seen:
            continue
        seen.add(result.tool_name)
        source = FactProvenance(kind="tool_result", dataset_id=dataset.id,
            evidence_revision_id=revision.id, source_call_id=result.call_id,
            tool_name=result.tool_name, tool_version=result.provenance.tool_version,
            input_revision_id=result.input_revision_id, output_revision_id=result.output_revision_id,
            source_payload_sha256=_digest(result.metrics), record_key="metrics").model_dump()
        sources.append((result.metrics, source))
    metadata = revision.metadata.get("worker_product", {}).get("metrics", {})
    if metadata and not any(s[1]["source_call_id"] == revision.call_id for s in sources):
        sources.append((metadata, FactProvenance(kind="revision_metadata", dataset_id=dataset.id,
            evidence_revision_id=revision.id, source_payload_sha256=_digest(metadata),
            record_key="worker_product.metrics").model_dump()))
    return sources


_SAVED_NULL_REASONS = {"constant_expression": "constant_expression"}

_SAVED_FIELD_LABELS = {
    "morans_i": "Moran's I",
    "mean_expression": "平均表达",
    "fraction_expressing": "表达比例",
    "count": "计数",
    "fraction": "比例",
    "observations": "观测数",
    "samples": "样本数",
    "replicates": "重复数",
    "n_genes_by_counts": "检测基因数",
    "pct_counts_mt": "线粒体计数比例",
    "qc/mean_total_counts": "平均总计数",
    "qc/mean_detected_genes": "平均检测基因数",
    "qc/mean_mito_pct": "平均线粒体比例",
}


def _saved_record_label(row):
    """Compose presentation text from the same saved row, never from the value.

    The machine ``key`` remains the audit path. This label only makes the
    existing gene/group/null-reason fields visible to readers and the model;
    it does not change the value, fact identity, revision or source binding.
    """
    parts = []
    if row.get("gene") is not None:
        parts.append(f"基因 {row['gene']}")
    if row.get("cluster") is not None:
        parts.append(f"簇 {row['cluster']}")
    if (row.get("groupby") not in (None, "all", "source_metadata")
            and row.get("group") is not None):
        parts.append(f"分组 {row['groupby']}={row['group']}")
    field = row["field"]
    metric = _SAVED_FIELD_LABELS.get(field, field)
    unit = row.get("unit")
    if unit and unit != field:
        metric += f"（{unit}）"
    parts.append(metric)
    label = "；".join(parts)
    if row.get("value") is None:
        if row.get("null_reason") == "constant_expression":
            label += "：未定义（恒定表达）"
        else:
            # A recorded null without an explicit reason is not evidence that
            # the statistic is mathematically undefined.
            label += "：空值（原因未知）"
    return label


def _saved_records(metrics, source, selection):
    """Enumerate only documented, persisted result shapes; never read expression."""
    rows = []
    if selection.kind == "section_composition":
        tables = metrics.get("quality", {}).get("cluster_chip_association", {})
        for groupby, table in tables.items():
            for cluster, counts in table.get("contingency", {}).items():
                for group, value in counts.items():
                    if type(value) is not int or value < 0:
                        raise EvidenceError("evidence.section_count_invalid")
                    rows.append({"path": ["quality", "cluster_chip_association", groupby,
                        "contingency", cluster, group], "value": value,
                        "groupby": groupby, "cluster": cluster, "group": group, "field": "count",
                        "unit": "observations", "source": source})
    elif selection.kind == "qc_groups":
        for groupby, groups in metrics.get("qc_by", {}).items():
            for group, record in groups.items():
                for field, value in record.items():
                    pairs = [(field + "/" + key, item) for key, item in value.items()] if isinstance(value, dict) else [(field, value)]
                    for name, item in pairs:
                        # Persisted result contracts allow scalar values only here.
                        if item is not None and (type(item) not in (int, float) or
                                                not math.isfinite(item)):
                            raise EvidenceError("evidence.qc_metric_invalid")
                        rows.append({"path": ["qc_by", groupby, group, *name.split("/")],
                            "value": item, "groupby": groupby, "group": group, "field": name,
                            "unit": "percent" if field == "pct_counts_mt" else
                                    "observations" if field == "observations" else
                                    "detected_genes" if field == "n_genes_by_counts" else "counts",
                            "source": source})
    elif selection.kind == "spatial_results":
        groupby = metrics.get("groupby") or metrics.get("stage_column") or "all"
        for table in ("genes", "groups", "composition", "expression"):
            for index, record in enumerate(metrics.get(table, [])):
                if not isinstance(record, dict):
                    raise EvidenceError("evidence.spatial_row_invalid")
                warning = record.get("warning")
                if warning is not None and not isinstance(warning, str):
                    raise EvidenceError("evidence.spatial_warning_invalid")
                null_reason = _SAVED_NULL_REASONS.get(warning)
                for field, value in record.items():
                    pairs = [(field + "/" + k, v) for k, v in value.items()] if isinstance(value, dict) else [(field, value)]
                    for name, item in pairs:
                        if name not in {"morans_i", "mean_expression", "fraction_expressing", "count", "fraction",
                            "observations", "samples", "replicates", "qc/mean_total_counts",
                            "qc/mean_detected_genes", "qc/mean_mito_pct"}:
                            continue
                        if item is not None and (type(item) not in (int, float) or not math.isfinite(item)):
                            raise EvidenceError("evidence.spatial_metric_invalid")
                        rows.append({"path": [table, str(index), *name.split("/")],
                            "value": item, "groupby": groupby,
                            "group": str(record.get("group", record.get("stage", "all"))),
                            "gene": record.get("gene"), "cluster": record.get("cluster"),
                            "field": name, "unit": "fraction" if name in {"fraction", "fraction_expressing"}
                                else "observations" if name in {"count", "observations"} else name,
                            "null_reason": null_reason if item is None else None,
                            "source": source})
    return rows


def _full_spatial_source(copilot, dataset, revision, metrics, source):
    """Use a registered full result only for the exact same committed tool call."""
    matches = [a for a in copilot.list_artifacts() if a.kind == "data.spatial.results" and
        a.metadata.get("dataset_id") == dataset.id and a.metadata.get("revision_id") == revision.id and
        a.metadata.get("call_id") == source.get("source_call_id")]
    if not matches:
        return metrics, source
    if len({a.sha256 for a in matches}) != 1:
        raise EvidenceError("evidence.spatial_artifact_conflict")
    artifact = matches[0]
    if artifact.project_id != copilot.project.id or artifact.size_bytes > 64 * 1024 * 1024:
        raise EvidenceError("evidence.spatial_artifact_invalid")
    path = _checked_path(copilot.layout.root.resolve(), artifact.relative_path, artifact.sha256, artifact=True)
    full = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(full, dict):
        raise EvidenceError("evidence.spatial_artifact_invalid")
    return full, {**source, "kind": "registered_artifact", "artifact_id": artifact.id,
        "artifact_sha256": artifact.sha256, "source_payload_sha256": _digest(full)}


def _source_metadata_records(copilot, dataset, revision):
    """Only registered public source cards; their assertions are not measurements."""
    rows, contexts = [], []
    for artifact in copilot.list_artifacts():
        if artifact.kind != "data.source_metadata" or (
            artifact.metadata.get("dataset_id") != dataset.id or
            artifact.metadata.get("revision_id") != revision.id):
            continue
        if artifact.project_id != copilot.project.id:
            raise EvidenceError("evidence.artifact_project_mismatch")
        if (artifact.metadata.get("public_source") is not True or
                artifact.metadata.get("source_sha256") != artifact.sha256):
            raise EvidenceError("evidence.source_metadata_provenance_invalid")
        path = _checked_path(copilot.layout.root.resolve(), artifact.relative_path,
                             artifact.sha256, artifact=True)
        if path.suffix.lower() != ".json" or path.stat().st_size > 1024 * 1024:
            raise EvidenceError("evidence.artifact_read_limit")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != artifact.sha256:
            raise EvidenceError("evidence.artifact_hash_mismatch")
        try:
            card = json.loads(payload.decode("utf-8-sig"))
        except (ValueError, UnicodeError) as error:
            raise EvidenceError("evidence.source_metadata_schema_invalid") from error
        if not isinstance(card, dict):
            raise EvidenceError("evidence.source_metadata_schema_invalid")
        # Compare the declared registered-input hash when present. It describes
        # the registered input, not a derived revision's current count state.
        if card.get("runtime_input_sha256") not in (None, dataset.sha256):
            raise EvidenceError("evidence.source_metadata_input_mismatch")
        source = FactProvenance(kind="registered_artifact", dataset_id=dataset.id,
            evidence_revision_id=revision.id, artifact_id=artifact.id,
            artifact_sha256=artifact.sha256, source_payload_sha256=_digest(card),
            record_key="source_metadata").model_dump()
        contexts.append({"artifact_id": artifact.id, "artifact_sha256": artifact.sha256,
            "source_payload_sha256": source["source_payload_sha256"],
            "evidence_role": "source_metadata_not_independent_measurement",
            "applies_to": "registered_input_description_not_recomputed_revision_state"})
        for field in SOURCE_METADATA_FIELDS:
            if field not in card:
                continue
            value = card[field]
            if (value is not None and type(value) not in (str, int, float, bool)) or (
                isinstance(value, float) and not math.isfinite(value)) or (
                field == "donor_count" and value is not None and
                (type(value) is not int or value < 0)):
                raise EvidenceError("evidence.source_metadata_value_invalid")
            rows.append({"path": ["source_metadata", field], "value": value,
                "field": field, "groupby": "source_metadata", "group": artifact.id,
                "unit": "source_metadata_not_independent_measurement", "source": source})
    return rows, contexts


_PUBLIC_TABLE_TYPES = {"string", "int", "float", "bool"}


def _is_public_table(artifact) -> bool:
    return artifact.kind == "data.public.input" or artifact.kind.startswith("table.public.")


def _public_table_layout(artifact) -> dict:
    """Read one optional declared layout; never infer biology from a filename."""
    raw = artifact.metadata.get("table_layout")
    if raw is None:
        return {"declared": False, "header": None, "columns": None,
                "types": {}, "delimiter": None}
    if not isinstance(raw, dict) or set(raw) - {"header", "columns", "types", "delimiter"}:
        raise EvidenceError("evidence.public_table_layout_invalid")
    header = raw.get("header")
    if not isinstance(header, bool):
        raise EvidenceError("evidence.public_table_layout_invalid")
    columns = raw.get("columns")
    if columns is not None:
        if (not isinstance(columns, list) or not columns or len(columns) > 128
                or any(not isinstance(item, str) or not item or len(item) > 200
                       for item in columns) or len(set(columns)) != len(columns)):
            raise EvidenceError("evidence.public_table_layout_invalid")
        columns = list(columns)
    types = raw.get("types") or {}
    if not isinstance(types, dict) or any(
            not isinstance(column, str) or not column or kind not in _PUBLIC_TABLE_TYPES
            for column, kind in types.items()):
        raise EvidenceError("evidence.public_table_layout_invalid")
    delimiter = raw.get("delimiter")
    if delimiter is not None:
        delimiter = {",": ",", "\t": "\t", "comma": ",", "tab": "\t"}.get(delimiter)
        if delimiter is None:
            raise EvidenceError("evidence.public_table_layout_invalid")
    return {"declared": True, "header": header, "columns": columns,
            "types": dict(types), "delimiter": delimiter}


def _public_table_value(raw: str, kind: str | None):
    """Preserve exact strings unless the artifact explicitly declares a type."""
    if kind in (None, "string"):
        return raw
    if kind == "int":
        try:
            return int(raw)
        except (TypeError, ValueError) as error:
            raise EvidenceError("evidence.public_table_value_type") from error
    if kind == "float":
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise EvidenceError("evidence.public_table_value_type") from error
        if not math.isfinite(value):
            raise EvidenceError("evidence.public_table_value_type")
        return value
    if kind == "bool":
        normalized = raw.strip().lower()
        if normalized not in {"true", "false"}:
            raise EvidenceError("evidence.public_table_value_type")
        return normalized == "true"
    raise EvidenceError("evidence.public_table_value_type")


def _read_public_table(copilot, dataset, revision, selection, root, active_before):
    """Read only explicitly registered, same-revision public input tables."""
    artifacts = {artifact.id: artifact for artifact in copilot.list_artifacts()}
    selected, missing = [], []
    availability, missing_entities = [], []
    for artifact_id in selection.artifact_ids:
        artifact = artifacts.get(artifact_id)
        if artifact is None:
            missing.append("artifact_ids/" + artifact_id)
            missing_entities.append("artifact_ids/" + artifact_id)
            availability.append(EvidenceAvailability(
                evidence_key="public_table/" + artifact_id, status="unknown",
                reason_code="evidence.public_table_artifact_missing").model_dump())
            continue
        if artifact.project_id != copilot.project.id:
            raise EvidenceError("evidence.artifact_project_mismatch")
        if not _is_public_table(artifact):
            raise EvidenceError("evidence.public_table_kind_invalid")
        if (artifact.metadata.get("dataset_id") != dataset.id or
                artifact.metadata.get("revision_id") != revision.id):
            raise EvidenceError("evidence.public_table_identity_mismatch")
        if (artifact.metadata.get("public_source") is not True or
                artifact.metadata.get("source_sha256") != artifact.sha256):
            raise EvidenceError("evidence.public_table_provenance_invalid")
        path = _checked_path(root, artifact.relative_path, artifact.sha256, artifact=True)
        if path.suffix.lower() not in {".csv", ".tsv"}:
            raise EvidenceError("evidence.public_table_format")
        selected.append((artifact, path))
    if sum(path.stat().st_size for _, path in selected) > 16 * 1024 * 1024:
        raise EvidenceError("evidence.public_table_read_limit")

    facts, provenance, coverage, source_context, layouts = [], {}, [], [], []
    scans = []
    supplied_records = matching_records = 0
    matching_cell_scalars = group_total = 0
    physical_records = 0
    for artifact, path in selected:
        layout = _public_table_layout(artifact)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != artifact.sha256:
            raise EvidenceError("evidence.artifact_hash_mismatch")
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeError as error:
            raise EvidenceError("evidence.public_table_encoding") from error
        delimiter = layout["delimiter"] or (
            "\t" if path.suffix.lower() == ".tsv" else ",")
        try:
            reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
            rows = []
            for row in reader:
                # One extra line is permitted here only so a header row does not
                # make an otherwise valid 100000-record table fail.
                if len(rows) > 100000:
                    raise EvidenceError("evidence.public_table_record_limit")
                rows.append(row)
        except csv.Error as error:
            raise EvidenceError("evidence.public_table_format") from error
        if not rows:
            raise EvidenceError("evidence.public_table_empty")
        if layout["header"]:
            header = rows.pop(0)
            if (not header or any(not item.strip() for item in header)
                    or len(set(header)) != len(header)):
                raise EvidenceError("evidence.public_table_header_invalid")
            columns = list(header)
            if layout["columns"] is not None and layout["columns"] != columns:
                raise EvidenceError("evidence.public_table_layout_mismatch")
        elif layout["columns"] is not None:
            columns = layout["columns"]
            if len(rows[0]) != len(columns):
                raise EvidenceError("evidence.public_table_row_width")
        else:
            columns = ["column_" + str(index + 1) for index in range(len(rows[0]))]
        if not columns:
            raise EvidenceError("evidence.public_table_empty")
        if len(columns) > 128:
            raise EvidenceError("evidence.public_table_column_limit")
        if any(len(row) != len(columns) for row in rows):
            raise EvidenceError("evidence.public_table_row_width")
        if set(layout["types"]) - set(columns):
            raise EvidenceError("evidence.public_table_layout_mismatch")
        physical_records += len(rows)
        if physical_records > 100000:
            raise EvidenceError("evidence.public_table_record_limit")
        layouts.append({"artifact_id": artifact.id,
            "header": layout["header"] if layout["declared"] else "unknown",
            "columns": columns, "declared": layout["declared"],
            "delimiter": "tab" if delimiter == "\t" else "comma",
            "physical_records": len(rows)})
        source_context.append({"artifact_id": artifact.id, "kind": artifact.kind,
            "artifact_sha256": artifact.sha256,
            "source_payload_sha256": artifact.sha256,
            "layout": layouts[-1]})
        missing_fields = [field for field in selection.fields if field not in columns]
        for field in missing_fields:
            entity = "fields/" + field
            missing.append(entity)
            missing_entities.append("artifact_ids/" + artifact.id + "/" + entity)
            availability.append(EvidenceAvailability(
                evidence_key="public_table/" + artifact.id + "/" + field, status="unknown",
                reason_code="evidence.public_table_field_missing").model_dump())
        groupby_index = None
        if selection.groupby is not None:
            if selection.groupby not in columns:
                entity = "groupby/" + selection.groupby
                missing.append(entity)
                missing_entities.append("artifact_ids/" + artifact.id + "/" + entity)
                availability.append(EvidenceAvailability(
                    evidence_key="public_table/" + artifact.id + "/groupby", status="unknown",
                    reason_code="evidence.public_table_groupby_missing").model_dump())
                continue
            groupby_index = columns.index(selection.groupby)
        if missing_fields:
            continue
        matching = [(record_index, row) for record_index, row in enumerate(rows, 1)
            if groupby_index is None or not selection.groups or row[groupby_index] in selection.groups]
        matching_records += len(matching)
        matching_cell_scalars += len(matching) * len(selection.fields)
        counts = {}
        if groupby_index is not None:
            # Every matching row contributes to its artifact-local raw group
            # count, independent of how many cell facts fit the scalar budget.
            for _, row in matching:
                group = row[groupby_index]
                counts[group] = counts.get(group, 0) + 1
            group_total += len(counts)
        scans.append({"artifact": artifact, "columns": columns,
            "types": layout["types"], "matching": matching, "counts": counts})

    # Group-count facts are referenceable facts, so they take the scalar budget
    # before optional cell projections. Artifacts remain separate sources.
    group_output, supplied_group_count_facts = [], 0
    for scan in scans:
        if len(facts) >= selection.limit:
            break
        artifact = scan["artifact"]
        records = []
        for group, count in scan["counts"].items():
            if len(facts) >= selection.limit:
                break
            fid = "public_table_group_count_fact_" + _digest(
                [artifact.id, artifact.sha256, selection.groupby, group])[:32]
            key = (f"public_table/{artifact.id}/group_count/"
                   f"{selection.groupby}/{group}/rows")
            fact = StoredFact(id=fid, key=key, value=count, revision_id=revision.id,
                label=(f"公开表 {artifact.id} 列 {selection.groupby} 组 {group} "
                       "匹配记录数"),
                show=True, source_call_id=None).model_dump()
            source = FactProvenance(kind="registered_artifact",
                dataset_id=dataset.id, evidence_revision_id=revision.id,
                source_call_id=None, artifact_id=artifact.id,
                artifact_sha256=artifact.sha256,
                source_payload_sha256=artifact.sha256,
                record_key=(f"public_table:{artifact.id}:group_count:"
                            f"{selection.groupby}:{group}")).model_dump()
            facts.append(fact)
            provenance[fid] = source
            coverage.append({"fact_id": fid, "artifact_id": artifact.id,
                "groupby": selection.groupby, "group": group, "value": count,
                "unit": "registered_table_row_count",
                "basis": "all_matching_table_records_not_H5AD_membership"})
            availability.append(EvidenceAvailability(evidence_key=key,
                status="supplied", source_ref=artifact.id,
                reason_code=None).model_dump())
            records.append({"artifact_id": artifact.id,
                "groupby": selection.groupby, "group": group,
                "value": count, "fact_id": fid})
            supplied_group_count_facts += 1
        if records:
            group_output.append({"artifact_id": artifact.id,
                "total_groups": len(scan["counts"]), "records": records})

    for scan in scans:
        if len(facts) >= selection.limit:
            break
        artifact, columns = scan["artifact"], scan["columns"]
        for record_index, row in scan["matching"]:
            if len(facts) >= selection.limit:
                break
            supplied_for_row = 0
            for field in selection.fields:
                if len(facts) >= selection.limit:
                    break
                raw = row[columns.index(field)]
                value = _public_table_value(raw, scan["types"].get(field))
                fid = "public_table_fact_" + _digest(
                    [artifact.id, artifact.sha256, record_index, field])[:32]
                key = f"public_table/{artifact.id}/row/{record_index}/{field}"
                fact = StoredFact(id=fid, key=key, value=value, revision_id=revision.id,
                    label=f"公开表 {artifact.id} 第{record_index}条记录列 {field}",
                    show=True, source_call_id=None).model_dump()
                source = FactProvenance(kind="registered_artifact",
                    dataset_id=dataset.id, evidence_revision_id=revision.id,
                    source_call_id=None, artifact_id=artifact.id,
                    artifact_sha256=artifact.sha256,
                    source_payload_sha256=artifact.sha256,
                    record_key=f"public_table:{artifact.id}:{record_index}:{field}").model_dump()
                facts.append(fact)
                provenance[fid] = source
                coverage.append({"fact_id": fid, "artifact_id": artifact.id,
                    "record_index": record_index, "field": field,
                    "unit": "registered_table_cell",
                    "coverage": "table_input_not_executed_analysis"})
                availability.append(EvidenceAvailability(evidence_key=key,
                    status="supplied", source_ref=artifact.id,
                    reason_code=None).model_dump())
                supplied_for_row += 1
            # A row is counted as supplied only when every requested field made
            # the scalar budget; partially cut rows stay in omitted_records.
            if supplied_for_row == len(selection.fields):
                supplied_records += 1

    supplied_scalars = len(facts)
    matching_scalars = matching_cell_scalars + group_total
    omitted_scalars = max(0, matching_scalars - supplied_scalars)
    omitted_records = max(0, matching_records - supplied_records)
    if omitted_scalars:
        availability.append(EvidenceAvailability(evidence_key="public_table",
            status="saved_not_selected", reason_code="evidence.record_limit").model_dump())
    status = "unknown"
    if not matching_records:
        availability.append(EvidenceAvailability(evidence_key="public_table",
            status="unknown", reason_code="evidence.request_not_found").model_dump())
    elif selected:
        status = "partial" if omitted_scalars or missing else "complete"
    if not facts:
        status = "unknown"
    summary = {"kind": "public_table", "dataset_id": dataset.id,
        "revision_id": revision.id, "revision_sha256": revision.sha256,
        "status": status, "artifact_ids": list(selection.artifact_ids),
        "fields": list(selection.fields), "layout": layouts,
        "columns": {entry["artifact_id"]: entry["columns"] for entry in layouts},
        "physical_records": physical_records, "matching_records": matching_records,
        "supplied_records": supplied_records, "omitted_records": omitted_records,
        "matching_cell_scalars": matching_cell_scalars,
        "matching_group_count_facts": group_total,
        "supplied_group_count_facts": supplied_group_count_facts,
        "supplied_cell_facts": supplied_scalars - supplied_group_count_facts,
        "matching_scalars": matching_scalars, "supplied_scalars": supplied_scalars,
        "omitted_scalars": omitted_scalars, "missing_entities": missing_entities,
        "coverage": coverage, "source_context": source_context,
        "boundary_note": ("Registered public-table input only; supplied rows are not "
            "verified H5AD barcode membership, cells, slices, or independent donors."),
        "join_status": "not_performed",
        "join_note": ("No H5AD barcode join, cell/slice/donor inference, statistical "
            "comparison, or revision change was performed."),
        "automatic_execution": False}
    if selection.groupby is not None:
        supplied_groups = supplied_group_count_facts
        omitted_groups = max(0, group_total - supplied_groups)
        group_status = ("unknown" if not group_total else
                        "partial" if omitted_groups else "complete")
        group_summary = {"groupby": selection.groupby,
            "unit": "registered_table_row_count",
            "basis": "all_matching_table_records_not_H5AD_membership",
            "total_groups": group_total, "supplied_groups": supplied_groups,
            "omitted_groups": omitted_groups, "status": group_status}
        if len(selection.artifact_ids) == 1:
            records = group_output[0]["records"] if group_output else []
            group_summary["counts"] = {item["group"]: item["value"] for item in records}
            group_summary["records"] = records
        else:
            group_summary["by_artifact"] = {
                entry["artifact_id"]: {
                    "counts": {item["group"]: item["value"] for item in entry["records"]},
                    "records": entry["records"], "total_groups": entry["total_groups"],
                    "supplied_groups": len(entry["records"]),
                    "omitted_groups": max(0, entry["total_groups"] - len(entry["records"])),
                    "status": ("partial" if entry["total_groups"] > len(entry["records"])
                               else "complete")}
                for entry in group_output}
        summary["group_counts"] = group_summary
    if copilot.get_dataset(dataset.id).active_revision_id != active_before:
        raise EvidenceError("evidence.input_revision_changed")
    _checked_path(root, revision.relative_path, revision.sha256)
    return {"facts": facts, "fact_provenance": provenance,
        "availability": availability, "figure_refs": [],
        "selection_summary": summary}


def read_saved_evidence(copilot, *, dataset_id: str, revision_id: str,
                        selection: SavedEvidenceSelection | dict) -> dict:
    """Bounded read-only lookup of exact saved tables or registered figure refs.

    Missing selection and missing computation are different: absence of a saved
    record is unknown unless the source explicitly records a missing prerequisite.
    Returned figure refs are not image observations and never claim pixel input.
    """
    selection = SavedEvidenceSelection.model_validate(selection)
    dataset = copilot.get_dataset(dataset_id)
    if dataset.project_id != copilot.project.id:
        raise EvidenceError("evidence.dataset_project_mismatch")
    active_before = dataset.active_revision_id
    revision = copilot.revisions.get(revision_id)
    if (revision.id != revision_id or revision.dataset_id != dataset_id or
            _value(revision.status) != "committed"):
        raise EvidenceError("evidence.revision_identity_mismatch")
    root = copilot.layout.root.resolve()
    _checked_path(root, revision.relative_path, revision.sha256)
    if selection.kind == "public_table":
        return _read_public_table(copilot, dataset, revision, selection, root, active_before)
    facts, provenance, figures, availability, coverage = [], {}, [], [], []
    source_context, partial_sources = [], []
    rows, explicit_uncomputed = [], False
    if selection.kind == "figure_refs":
        for artifact in copilot.list_artifacts():
            if not artifact.kind.startswith("figure.") or (
                artifact.metadata.get("dataset_id") != dataset_id or
                artifact.metadata.get("revision_id") != revision_id):
                continue
            if artifact.project_id != copilot.project.id:
                raise EvidenceError("evidence.artifact_project_mismatch")
            rows.append({"artifact": artifact})
    elif selection.kind == "source_metadata":
        rows, source_context = _source_metadata_records(copilot, dataset, revision)
    else:
        for metrics, source in _saved_sources(copilot, dataset, revision):
            if selection.kind == "spatial_results":
                if not (source.get("tool_name") or "").startswith(("spatial.", "samples.")):
                    continue
                metrics, source = _full_spatial_source(copilot, dataset, revision, metrics, source)
                if metrics.get("evidence_projection"):
                    partial_sources.append({"source_call_id": source["source_call_id"],
                        "tables": metrics["evidence_projection"]})
            rows.extend(_saved_records(metrics, source, selection))
            source_context.append({"source_call_id": source["source_call_id"],
                "source_payload_sha256": source["source_payload_sha256"],
                "revision_id": revision_id, "tool_name": source.get("tool_name"),
                "count_source": metrics.get("count_source"), "expression_source": metrics.get("expression_source"),
                "cluster_key": metrics.get("cluster_key", metrics.get("quality", {}).get("cluster_key")),
                "groupby": metrics.get("groupby") or metrics.get("stage_column"),
                "labels_aligned": metrics.get("labels_aligned"),
                "inference": metrics.get("inference")})
            if selection.kind == "section_composition" and metrics.get("quality", {}).get("clustering_available") is False:
                explicit_uncomputed = True

    def matches(row):
        if selection.kind == "figure_refs":
            return not selection.artifact_ids or row["artifact"].id in selection.artifact_ids
        return ((not selection.groupby or row["groupby"] == selection.groupby) and
            (not selection.groups or row["group"] in selection.groups) and
            (not selection.clusters or row.get("cluster") in selection.clusters) and
            (not selection.genes or row.get("gene") in selection.genes) and
            (not selection.fields or row["field"] in selection.fields))

    requested = [row for row in rows if matches(row)]
    chosen = requested[:selection.limit]
    omitted = len(rows) - len(chosen)
    for row in chosen:
        if selection.kind == "figure_refs":
            artifact = row["artifact"]
            _checked_path(root, artifact.relative_path, artifact.sha256, artifact=True)
            figures.append({"artifact_id": artifact.id, "kind": artifact.kind,
                "relative_path": artifact.relative_path, "sha256": artifact.sha256,
                "dataset_id": dataset_id, "revision_id": revision_id,
                "parameters": artifact.metadata.get("parameters", {}), "pixels_supplied": False})
            availability.append(EvidenceAvailability(evidence_key="figure/" + artifact.id,
                status="supplied", source_ref=artifact.id).model_dump())
            continue
        key = "/".join(row["path"])
        source = {**row["source"], "record_key": key}
        fid = "saved_fact_" + _digest([source, key])[:32]
        fact = StoredFact(id=fid, key=key, value=row["value"], revision_id=revision_id,
            label=_saved_record_label(row), source_call_id=source["source_call_id"]).model_dump()
        facts.append(fact)
        provenance[fid] = source
        coverage.append({"fact_id": fid, "groupby": row["groupby"], "group": row["group"],
            "gene": row.get("gene"), "cluster": row.get("cluster"),
            "field": row["field"], "unit": row["unit"],
            "null_reason": row.get("null_reason")})
        null_reason = row.get("null_reason")
        availability.append(EvidenceAvailability(evidence_key=key,
            status=("supplied" if row["value"] is not None else
                    "unavailable" if null_reason else "unknown"),
            source_ref=source["source_call_id"],
            reason_code=None if row["value"] is not None else "evidence.metric_saved_null",
            null_reason=null_reason).model_dump())
    for partial_source in partial_sources:
        availability.append(EvidenceAvailability(evidence_key=selection.kind,
            status="partial", source_ref=partial_source["source_call_id"],
            reason_code="evidence.full_spatial_artifact_unavailable").model_dump())
    if omitted:
        availability.append(EvidenceAvailability(evidence_key=selection.kind,
            status="saved_not_selected", reason_code="evidence.saved_records_outside_selection").model_dump())
    # Report each requested entity missing from the actual source, not the prefix.
    missing = []
    entities = (("artifact_ids", "artifact"),) if selection.kind == "figure_refs" else (
        ("groups", "group"), ("clusters", "cluster"), ("genes", "gene"))
    for name, key in entities:
        found = {row["artifact"].id if key == "artifact" else row.get(key) for row in requested}
        missing.extend(f"{name}/{entity}" for entity in getattr(selection, name) if entity not in found)
    if selection.kind != "figure_refs":
        found_fields = {row["field"] for row in requested}
        missing.extend("fields/" + field for field in selection.fields if field not in found_fields)
    for entity in missing:
        availability.append(EvidenceAvailability(evidence_key=selection.kind + "/" + entity,
            status="unknown", reason_code="evidence.requested_entity_not_saved").model_dump())
    if not requested:
        availability.append(EvidenceAvailability(evidence_key=selection.kind,
            status="not_computed" if explicit_uncomputed else "unknown",
            reason_code="evidence.clustering_not_available" if explicit_uncomputed else
                        "evidence.request_not_found").model_dump())
    partial = len(requested) > len(chosen) or bool(missing) or bool(partial_sources) or any(
        row.get("value") is None for row in chosen if "artifact" not in row)
    status = "partial" if chosen and partial else "complete" if chosen else "unknown"
    if len(requested) > len(chosen):
        availability.append(EvidenceAvailability(evidence_key=selection.kind,
            status="partial", reason_code="evidence.record_limit").model_dump())
    if copilot.get_dataset(dataset_id).active_revision_id != active_before:
        raise EvidenceError("evidence.input_revision_changed")
    _checked_path(root, revision.relative_path, revision.sha256)
    response = {"facts": facts, "fact_provenance": provenance, "availability": availability,
        "figure_refs": figures, "selection_summary": {"kind": selection.kind,
            "dataset_id": dataset_id, "revision_id": revision_id,
            "revision_sha256": revision.sha256, "status": status,
            "saved_records": len(rows), "matching_records": len(requested),
            "supplied_records": len(chosen), "omitted_records": omitted,
            "missing_entities": missing, "coverage": coverage,
            "source_context": source_context, "partial_sources": partial_sources,
            "automatic_execution": False}}
    if selection.kind == "source_metadata":
        response["selection_summary"]["evidence_role"] = "source_metadata_not_independent_measurement"
    return response
