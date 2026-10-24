"""Structured sample-premise review and deterministic evidence-link checks.

The model judges meaning. This module validates its proof links and the review
contract; it does not claim to prove arbitrary prose or biological correctness.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, create_model

from st_agent.knowledge_corpus.contracts import Record

DIMENSIONS = ("premise_coverage", "comparison_scope", "measurement_semantics", "inference_identifiability")
# One explicit display/interpretation policy for the bounded marker judgment.
# It is not a hypothesis-test alpha, an equivalence margin, a cell-identity
# threshold or a scientific Worker parameter.
MARKER_DESCRIPTIVE_LOG2FC_CUTOFF = 0.25
ReviewType = Literal["evidence_collection", "inference", "background"]
# The single obligation table for every review_type. A unit receives exactly
# the listed dimensions; everything else is explicitly N/A for that class.
# Legacy/unclassified units are not in this table and stay on all dimensions.
REVIEW_TYPE_ENFORCED_DIMENSIONS = {
    # A collection/readout step still needs true premises, an executable and
    # applicable comparison, and correct measurement caliber. It does not
    # itself adjudicate competing explanations, so identifiability is N/A.
    "evidence_collection": frozenset({
        "premise_coverage", "comparison_scope", "measurement_semantics"}),
    # A pure source/method background only has to support its real source and
    # attribution. Sample comparison, measurement and inference are N/A; a
    # mixed unit cannot use this class (see _effective_review_type).
    "background": frozenset({"premise_coverage"}),
    # An inference carries every obligation.
    "inference": frozenset(DIMENSIONS),
}
# Assertion natures that can genuinely be pure source/method background. Any
# sample observation, plan, biological claim or conditional rule is not pure.
BACKGROUND_SAFE_ASSERTION_NATURES = frozenset({
    "method_semantics", "metadata_or_quality_limitation", "non_biological_other"})


def _review_obligations(review_type):
    """Return (enforced, waived) dimensions for one effective review class.

    Both validate_units and validate_assessment use this helper so the local
    gate and its audit cannot drift. ``None`` is the legacy/unclassified case
    and keeps every dimension strict.
    """
    enforced = (frozenset(DIMENSIONS) if review_type is None
                else REVIEW_TYPE_ENFORCED_DIMENSIONS.get(review_type, frozenset(DIMENSIONS)))
    return ([d for d in DIMENSIONS if d in enforced],
            [d for d in DIMENSIONS if d not in enforced])


INFERENCE_INSTRUCTION = (
    "When the request or the reviewed unit explicitly claims to distinguish competing causal "
    "explanations, a validation design must state those explanations, comparison unit, required "
    "inputs, controls, measurements and a result that distinguishes their predictions. If the same "
    "result is compatible with both explanations, call the test inconclusive and request the "
    "additional independent controls; do not claim the favored explanation is disproved. "
    "A same-sample descriptive check or a first local reference step does not carry that causal-"
    "identification burden: check its resampling object, held-fixed quantities, statistic, "
    "exchangeability conditions and the question it can answer, and require it to state what it "
    "cannot establish. Merely naming a possible technical explanation while declining to attribute "
    "it, or saying a future comparison would need to distinguish explanations, is a limitation, "
    "not a promise that this design identifies the cause. "
    "Section/cluster association or clustering sensitivity alone cannot distinguish technical "
    "artifacts from genuine spatial biology; reject a unit that claims that distinction without an "
    "independent basis, but do not demand that basis for a unit that explicitly limits itself to a "
    "descriptive question. Without independent biological replicates, separate "
    "descriptive sensitivity from biological or technical causal attribution. To separate an "
    "expression effect into detection frequency and positive-observation intensity, request "
    "the corresponding per-group positive-observation intensity measurements and verified "
    "expression scale, not merely another ranking of detection proportions. A comparison of "
    "two measurement types is not a causal or equivalence test. Explain unresolved alternatives. "
    "The evidence inventory reports supplied partial tables. Partial/context-truncated is not "
    "absent or never-computed; preserve the actual supplied-versus-missing scope. "
    "Evaluate proposed future designs conditionally on their explicitly requested inputs. "
    "A unit's planned_inputs names those prerequisites; it is neither observed evidence nor "
    "proof of availability, and a pure future action does not need a fake current fact or a "
    "generic gap binding. If the same unit also states a current result, that result still "
    "needs its own exact current premises and checks. Missing future data/replicates are "
    "prerequisites, not evidence that the conditional "
    "design is invalid or already executed. A well-calibrated descriptive sensitivity "
    "check can be useful without causal identification if it states that limit. Do not "
    "demand independent biological replicates for a within-sample descriptive decomposition; "
    "require them for population-level inference. Expression-specific measurement checks "
    "are not applicable to a purely compositional or partition-sensitivity comparison. "
    "For any permutation/bootstrap design specify the resampling unit, quantities held fixed "
    "and test statistic. Check that resampling can actually change the statistic under the "
    "null: permuting labels within a section preserves its cluster counts, so it cannot "
    "provide a nondegenerate null for section-by-cluster count differences. "
    "Checklist column names, required future inputs/controls and conditional verification "
    "designs are plans, not current absence or current-result assertions; use proposal, "
    "verification or method units and do not force a gap binding for them. State an actual "
    "current gap only in a separate limitation with its exact canonical sentence; a plan may "
    "depend on that limitation without first proving biological identity. "
    "Classify each unit's review_type by what it does, not by role or label: a plan that only "
    "collects observations/readouts is evidence_collection; a conclusion or a decision rule "
    "that maps an observed pattern to support/rejection of an explanation is inference even "
    "when unexecuted and called descriptive; source/method background is background. "
    "Evidence_collection units do not themselves carry the causal-identification "
    "burden, so inference_identifiability is not applicable to them; their factual "
    "premises, executable/applicable comparison and measurement caliber still have "
    "to pass. Pure background (source/tool/literature attribution) has only the "
    "source-support obligation: comparison_scope, measurement_semantics and "
    "inference_identifiability are not applicable, while the real citation/source "
    "and correct attribution are still checked. A background unit that carries a "
    "current-sample fact, measurement, comparison or conclusion is mixed and must "
    "be classified as evidence_collection or inference, never used to avoid those "
    "checks. For an inference, "
    "assess identifiability from the combined necessary evidence steps and controls in the "
    "same plan and name them as required dependencies; a criterion that cannot separate the "
    "competing explanations is insufficient. A uniform, absent or negative pattern is not "
    "evidence of absence when the competing explanation can also produce that pattern. If the "
    "unit's own text says its criterion cannot distinguish the alternatives, then any decision "
    "that one explanation is or is not supported is not identifiable: mark "
    "inference_identifiability insufficient. "
)


def review_target(kind):
    """Host-defined review scope; declared future actions never become observations."""
    return {
        "mode": "prospective_design" if kind == "next_step" else "sample_interpretation",
        "factual_premises": "Validate all claims about existing data against supplied sources; history is not evidence.",
        "future_inputs": "For next_step, distinguish explicitly requested prerequisites (planned_inputs) "
                         "from claims that they exist or were observed. Judge design validity conditional "
                         "on those prerequisites, never claim they were executed.",
        "measurement_scope": "Use not_applicable for expression semantics in a composition-only design. "
                             "Descriptive checks need not identify a cause, but must not claim that they do.",
        "statistical_scope": "Return scope_check for every unit. If no statistical_scope is "
                             "supplied, use not_applicable for an ordinary factual/method/"
                             "qualification/proposal unit, descriptive_only only when the text "
                             "merely describes an observed statistic's magnitude/direction "
                             "without a test, and inferential_claim when the text asserts "
                             "detectability, absence, significance or a group/population "
                             "difference without that structure. When statistical_scope is "
                             "supplied, compare its declared question, comparison object, test/"
                             "resampling design, data_status and supported decision with the "
                             "text. data_status only says whether the data already exist; it "
                             "never upgrades what the test can support. A within-group null "
                             "cannot support a between-group or causal decision, including a "
                             "future collection plan. Do not turn a descriptive coefficient's "
                             "proximity to zero into proof of no detectable structure; the separate "
                             "marker display policy may state only 'near zero/no descriptive "
                             f"expression feature' for abs(log2FC)<{MARKER_DESCRIPTIVE_LOG2FC_CUTOFF}, not absence, significance, "
                             "equivalence or cell identity. A p-value "
                             "from another source_call_id cannot support this result.",
    }


class GroupMetadataRef(Record):
    group_id: str
    path: Literal["semantics.observation_unit", "semantics.test_method",
                  "semantics.effect_expression_source", "semantics.detection_expression_source",
                  "semantics.expression_transform", "semantics.effect_scale",
                  "semantics.interpretation_scope", "comparison.kind", "comparison.target_group",
                  "groupby", "cluster_id"]


class PremiseBinding(Record):
    field: Literal["text", "limitation", "verification"]
    quote: str = Field(min_length=1, max_length=600)
    fact_ids: tuple[str, ...] = Field()
    gap_ids: tuple[str, ...] = Field(default=(), max_length=8)
    group_refs: tuple[GroupMetadataRef, ...] = Field(default=(), max_length=8)


class InferenceCheck(Record):
    dimension: Literal["premise_coverage", "comparison_scope", "measurement_semantics", "inference_identifiability"]
    status: Literal["supported", "insufficient", "contradicted", "not_applicable"]
    reason: str = Field(min_length=1, max_length=320)


class MeasurementClaim(Record):
    field: Literal["text", "limitation", "verification"] = Field(
        description="Unit field that contains the quoted measurement conclusion.")
    measurement_status: Literal["observed", "planned"] = Field(
        default="observed", description=(
            "Whether this is a current, already executed measurement (observed) or a "
            "future readout the unit proposes to obtain (planned). Legacy canonical/"
            "history records without this axis are observed. data_status only says "
            "whether input data exist; it never substitutes for this field."))
    quote: str = Field(min_length=1, max_length=600, description=(
        "Exact substring of the current unit's conclusion, not parent prose. For an "
        "observed claim this is the current conclusion; for a planned claim it is the "
        "future readout the unit proposes."))
    evidence_group_id: str | None = Field(description=(
        "The gN evidence_group_id that supports this measurement conclusion. "
        "An observed claim must bind a real same-unit group. A purely future "
        "planned readout with no current saved measurement group may use "
        "explicit null; null proves no measurement occurred and cannot satisfy "
        "an observed obligation."))
    scope: Literal["detection_frequency", "expression_intensity",
                   "overall_expression"] = Field(description=(
        "Measurement caliber concluded or proposed by the unit."))
    conclusion: Literal["different", "equivalent", "undetermined"] = Field(description=(
        "Conclusion as worded for an observed measurement. A planned measurement must "
        "use undetermined; equal detection does not establish equal expression "
        "intensity or overall expression equivalence."))


class UnitWirePremiseBinding(Record):
    """Production unit-review binding without a repeated field.

    The unit identity is the single source of truth. This wire object is used
    only by the unit-review schema; aggregate/history readers keep the strict
    internal PremiseBinding with an explicit field.
    """
    quote: str = Field(min_length=1, max_length=600)
    fact_ids: tuple[str, ...] = Field()
    gap_ids: tuple[str, ...] = Field(default=(), max_length=8)
    group_refs: tuple[GroupMetadataRef, ...] = Field(default=(), max_length=8)


class UnitWireMeasurementClaim(Record):
    measurement_status: Literal["observed", "planned"] = Field(description=(
        "Required explicit time/purpose axis. observed = a current, already executed "
        "measurement; planned = a future readout the unit proposes to obtain. "
        "data_status says only whether input data exist and never selects this value."))
    quote: str = Field(min_length=1, max_length=600, description=(
        "Exact substring of the current unit's conclusion, not parent prose. For an "
        "observed claim this is the current conclusion; for a planned claim it is the "
        "future readout the unit proposes."))
    evidence_group_id: str | None = Field(description=(
        "The gN evidence_group_id that supports this measurement conclusion. "
        "observed requires a real same-unit group. A purely future planned "
        "readout with no current saved measurement group may use explicit null; "
        "null is not evidence of an existing measurement and never satisfies an "
        "observed requirement. A non-null planned ID must still be legal and in "
        "this unit's support closure, never a sibling's group."))
    scope: Literal["detection_frequency", "expression_intensity",
                   "overall_expression"] = Field(description=(
        "Measurement caliber concluded or proposed by the unit."))
    conclusion: Literal["different", "equivalent", "undetermined"] = Field(description=(
        "Conclusion as worded for an observed measurement. A planned measurement must "
        "use undetermined; equal detection does not establish equal expression "
        "intensity or overall expression equivalence."))


NumericRelation = Literal["maximum", "minimum", "less_than", "less_or_equal",
                          "greater_than", "greater_or_equal", "equal", "not_equal"]


class UnitNumericComparison(Record):
    """One host-checkable numeric relation asserted by a unit.

    The verifier states the exact comparison span, the single metric and the
    actual recorded operand facts, subject first. The host recomputes the
    relation from saved scalar values. It never parses arbitrary prose or
    guesses which numbers the sentence intended to compare.
    """
    quote: str = Field(min_length=1, max_length=600)
    metric: str = Field(min_length=1, max_length=64, description=(
        "Recorded metric name: the evidence group's metric_fact_ids key, or for an "
        "ungrouped scalar the key segment after any trailing positional index. "
        "pct_in/pct_out share the detection_frequency caliber; detection_difference "
        "and logfoldchange are different calibers."))
    relation: NumericRelation
    operand_fact_ids: tuple[str, ...] = Field(min_length=2)


class StatisticalScopeCheck(Record):
    """Explicit applicability and scope comparison for one unit.

    ``not_applicable`` means the unit contains no statistical interpretation.
    ``descriptive_only`` means it interprets an observed statistic's magnitude
    or direction without a test and does not assert detectability, absence,
    significance or a group/population difference. ``inferential_claim`` means
    it makes such an inference while supplying no ``statistical_scope``; the
    host rejects that missing structure. ``consistent``/``mismatch`` compare a
    supplied scope with the text.
    """
    status: Literal["consistent", "mismatch", "not_applicable",
                    "descriptive_only", "inferential_claim"]
    reason: str = Field(min_length=1, max_length=320)


class SampleAssessment(Record):
    index: int = Field(ge=0, lt=6, strict=True)
    judgment: Literal["supported", "contradicted", "insufficient"]
    relevant: bool = Field(strict=True)
    premise_bindings: tuple[PremiseBinding, ...] = Field(max_length=16)
    checks: tuple[InferenceCheck, ...] = Field(min_length=4, max_length=4)
    measurement_claims: tuple[MeasurementClaim, ...] = Field(max_length=8, description=(
        "At least one observed claim is required when an observed unit binds an "
        "evidence-group fact; a semantically reviewed unexecuted design may return "
        "planned claims with conclusion=undetermined or an empty list."))
    unit_verdicts: tuple["UnitVerdict", ...] = Field(default=(), max_length=12)


UnitAssertionNature = Literal["biological_identity", "biological_mechanism",
                             "reference_transfer", "local_observation",
                             "method_semantics", "metadata_or_quality_limitation",
                             "unexecuted_check", "conditional_alternative_hypothesis",
                             "non_biological_other"]
EvidenceApplicability = Literal["matched_scientific_evidence_required",
                                "matched_scientific_evidence_not_required"]
BIOLOGICAL_ASSERTION_NATURES = frozenset({
    "biological_identity", "biological_mechanism", "reference_transfer"})
# A conditional alternative explanation is a falsifiable future test, not an
# assertion that the current sample already has the proposed identity,
# mechanism or reference transfer. Local code cannot parse that distinction
# from prose; the verifier must declare it, and biological declarations remain
# subject to the same-unit matched-evidence gate below.
CONDITIONAL_ASSERTION_NATURES = frozenset({"conditional_alternative_hypothesis"})


class UnitVerdict(Record):
    """Internal/aggregate unit verdict: every binding has an explicit field.

    The assertion/applicability fields are optional only for legacy aggregate
    records created before B08. New unit wire verdicts require all three; a
    missing field is rejected by ``validate_units`` and never defaults to an
    exemption.
    """
    unit_id: str = Field(min_length=1, max_length=64)
    status: Literal["supported", "contradicted", "insufficient"]
    relevant: bool = Field(strict=True)
    premise_bindings: tuple[PremiseBinding, ...] = Field(max_length=16)
    checks: tuple[InferenceCheck, ...] = Field(min_length=4, max_length=4)
    measurement_claims: tuple[MeasurementClaim, ...] = Field(max_length=8, description=(
        "At least one observed claim is required when an observed unit binds an "
        "evidence-group fact; a semantically reviewed unexecuted design may return "
        "planned claims with conclusion=undetermined or an empty list."))
    required_dependencies: tuple[str, ...] = Field(default=(), max_length=12)
    fulfilled_requirement_ids: tuple[str, ...] = Field(default=(), max_length=8)
    citation_check: Literal["not_applicable", "supported", "unsupported"] | None = None
    # Legacy canonical/history verdicts may omit the semantic class and keep
    # the original all-dimension review. A new wire verdict must provide it.
    review_type: ReviewType | None = None
    scope_check: StatisticalScopeCheck | None = None
    assertion_nature: UnitAssertionNature | None = None
    evidence_applicability: EvidenceApplicability | None = None
    applicability_reason: str | None = Field(default=None, min_length=1, max_length=320)
    numeric_comparisons: tuple[UnitNumericComparison, ...] = Field(default=(), max_length=8)


class WireUnitVerdict(Record):
    """Production unit-review wire: no field is repeated from the unit.

    ``scope_check`` and the assertion/applicability decision are required here.
    Every new unit review must state the statistical applicability and whether
    the unit actually asserts current-cluster biology/reference transfer. Local
    code then enforces same-unit matched ``scientific_evidence``; legacy
    aggregate/history verdicts keep the optional fields and are not treated as
    new verification.
    """
    unit_id: str = Field(min_length=1, max_length=64)
    status: Literal["supported", "contradicted", "insufficient"]
    relevant: bool = Field(strict=True)
    premise_bindings: tuple[UnitWirePremiseBinding, ...] = Field(max_length=16)
    checks: tuple[InferenceCheck, ...] = Field(min_length=4, max_length=4)
    measurement_claims: tuple[UnitWireMeasurementClaim, ...] = Field(
        max_length=8, description=(
            "Every entry must declare measurement_status=observed/planned. An observed "
            "unit that binds an evidence-group fact needs at least one observed entry. "
            "A semantically reviewed unexecuted proposal/conditional design may return "
            "planned entries (conclusion=undetermined) or [] when it reports no planned "
            "readout. scope_check=consistent only means design/text agreement."))
    required_dependencies: tuple[str, ...] = Field(default=(), max_length=12)
    fulfilled_requirement_ids: tuple[str, ...] = Field(default=(), max_length=8)
    citation_check: Literal["not_applicable", "supported", "unsupported"] | None = None
    # Required: the verifier independently classifies what THIS unit does.
    # Omission is a protocol error, never an implicit safe category.
    review_type: ReviewType
    scope_check: StatisticalScopeCheck
    assertion_nature: UnitAssertionNature
    evidence_applicability: EvidenceApplicability
    applicability_reason: str = Field(min_length=1, max_length=320)
    # New unit wire must state the comparison list explicitly, including an
    # empty list when the unit asserts no numeric relation. A missing field is
    # a schema error rather than an implicit "no comparison" approval.
    numeric_comparisons: tuple[UnitNumericComparison, ...] = Field(max_length=8)


class SampleAssessmentResult(Record):
    assessments: tuple[SampleAssessment, ...] = Field(max_length=6)


class UnitSampleAssessment(Record):
    """Production unit-only assessment.

    No aggregate judgment, premise bindings, four-dimension checks or
    measurement claims are accepted here. Those fields belonged to the hidden
    parent prose and caused duplicate output plus schema truncation. The only
    authorization material is the complete per-unit verdict list; any optional
    display projection must be built from verified units, never from this wire.
    """
    index: int = Field(ge=0, lt=6, strict=True)
    unit_verdicts: tuple[WireUnitVerdict, ...] = Field(min_length=1, max_length=12)


class UnitSampleAssessmentResult(Record):
    assessments: tuple[UnitSampleAssessment, ...] = Field(max_length=6)


class FlatUnitVerdict(Record):
    """One unit-only verdict in the flat production wire.

    A flat batch has exactly one repeatable boundary: a host-generated
    ``target_id`` that already fixes the original reasoning index and unit
    identity. There is no second ``assessments -> unit_verdicts`` nesting for
    the model to duplicate, close early or misplace. The canonical
    ``UnitVerdict`` (with ``unit_id`` and explicit field) is restored by the
    host before the ordinary strict validator runs.
    """
    target_id: str = Field(min_length=1, max_length=24)
    status: Literal["supported", "contradicted", "insufficient"]
    relevant: bool = Field(strict=True)
    premise_bindings: tuple[UnitWirePremiseBinding, ...] = Field(max_length=16)
    checks: tuple[InferenceCheck, ...] = Field(min_length=4, max_length=4)
    measurement_claims: tuple[UnitWireMeasurementClaim, ...] = Field(
        max_length=8, description=(
            "Every entry must declare measurement_status=observed/planned. An observed "
            "unit that binds an evidence-group fact needs at least one observed entry. "
            "A semantically reviewed unexecuted proposal/conditional design may return "
            "planned entries (conclusion=undetermined) or [] when it reports no planned "
            "readout. scope_check=consistent only means design/text agreement."))
    required_dependencies: tuple[str, ...] = Field(default=(), max_length=12)
    fulfilled_requirement_ids: tuple[str, ...] = Field(default=(), max_length=8)
    citation_check: Literal["not_applicable", "supported", "unsupported"] | None = None
    # Required: the verifier independently classifies what THIS unit does.
    # Omission is a protocol error, never an implicit safe category.
    review_type: ReviewType
    scope_check: StatisticalScopeCheck
    assertion_nature: UnitAssertionNature
    evidence_applicability: EvidenceApplicability
    applicability_reason: str = Field(min_length=1, max_length=320)
    # Required even when empty: this is the flat production wire.
    numeric_comparisons: tuple[UnitNumericComparison, ...] = Field(max_length=8)


class FlatUnitAssessmentResult(Record):
    """Unit-only response shape: one flat verdict list for the whole call.

    The concrete schema is generated per request by
    :func:`flat_unit_response_schema`; the base class documents the shared
    contract and is not advertised to a live model with an open target set.
    """
    unit_verdicts: tuple[FlatUnitVerdict, ...] = Field(min_length=1, max_length=72)


def flat_unit_response_schema(target_ids):
    """Restrict a flat unit response to the actual host targets.

    ``target_id`` is a closed ``Literal`` enum, so an unknown or fabricated
    target is a schema error. The list length is fixed to the target count;
    duplicate/omitted coverage is still checked by the host because two entries
    can repeat one literal while omitting another.
    """
    ids = tuple(dict.fromkeys(str(target) for target in target_ids))
    if not ids:
        raise ValueError("flat unit response requires at least one target")
    target_type = Literal[ids]
    verdict = create_model(
        "FlatUnitVerdictForRequest", __base__=FlatUnitVerdict,
        target_id=(target_type, ...))
    return create_model(
        "FlatUnitAssessmentResultForRequest", __base__=FlatUnitAssessmentResult,
        unit_verdicts=(tuple[verdict, ...], Field(min_length=len(ids), max_length=len(ids))))


class MixedSampleAssessmentResult(Record):
    """Wire union for a batch containing both legacy and unit records.

    This is only an outbound parsing union. The host revalidates every parsed
    assessment against the strict model selected by the corresponding candidate.
    A unit-bearing candidate must satisfy ``UnitSampleAssessment`` and therefore
    cannot smuggle in an aggregate ``judgment`` as authorization; a legacy
    candidate must satisfy the unchanged aggregate ``SampleAssessment``. A
    legacy sibling neither erases a correct unit nor lends it evidence.
    """
    assessments: tuple[UnitSampleAssessment | SampleAssessment, ...] = Field(max_length=6)


def shared_fact_catalog(records, provenance):
    """Compact all validated facts once; revision/call aliases are explicit tables."""
    revisions, calls, rows, aliases = [], [], [], {}
    for fact in records.values():
        revision = fact["revision_id"]
        call = fact.get("source_call_id") or provenance.get(fact["id"], {}).get("source_call_id")
        if revision not in revisions:
            revisions.append(revision)
        if call not in calls:
            calls.append(call)
        alias = "f" + str(len(aliases))
        aliases[alias] = fact["id"]
        rows.append([alias, fact["key"], fact["value"], revisions.index(revision),
                     calls.index(call), fact.get("label", fact["key"])])
    return {"columns": ["id", "key", "value", "revision_index", "call_index", "label"],
            "revisions": revisions, "calls": calls, "rows": rows,
            "id_encoding": "Request-local aliases; copy fN exactly, never guess an original storage ID",
            "complete_for_supplied_bundle": True,
            "scope": "Saved supplied facts only; not a promise that all original tool metrics were projected"}, aliases


def alias_candidate(candidate, aliases, group_aliases, unit_support=None):
    """Project one candidate for outbound review without two competing bodies.

    A unit-bearing candidate has already been rendered from its authoritative
    units. Its parent ``text/limitation/verification`` fields are only a legacy
    aggregate summary and may be stale after a local unit patch, so the outbound
    view omits those three prose fields while retaining identity, kind/domain,
    facts, sources, requirements and units. The draft and original audit still
    store the parent fields. A candidate without units keeps the legacy
    aggregate body unchanged for historical readers.

    ``unit_support`` maps each unit to its host-side bindable fact/group
    closure. Unit-bearing output then states that boundary explicitly instead
    of leaving the full shared catalog as an implied premise for every unit.
    IDs that are not part of the request-local alias union are omitted from the
    outbound pointer lists; they remain in the original audit candidate.
    """
    fact_ids = {original: alias for alias, original in aliases.items()}
    group_ids = {original: alias for alias, original in group_aliases.items()}
    groups = [{**g, "group_id": group_ids[g["group_id"]], "gene_fact_id": fact_ids[g["gene_fact_id"]],
               "metric_fact_ids": {key: fact_ids[fid] for key, fid in g["metric_fact_ids"].items()}}
              for g in candidate["evidence_groups"]
              if g["group_id"] in group_ids
              and g["gene_fact_id"] in fact_ids
              and all(fid in fact_ids for fid in g["metric_fact_ids"].values())]
    units = []
    for unit in candidate.get("units", []):
        # The verifier must classify review_type independently from the unit
        # text. The generator's declared class stays in the host-side audit and
        # is not shown here, so it cannot anchor or authorize the review.
        projected_unit = {**{key: value for key, value in unit.items()
                             if key != "review_type"},
                          "evidence_refs": list(dict.fromkeys(
                              fact_ids.get(fid, "unknown_fact:" + fid)
                              for fid in unit["evidence_refs"]))}
        support = (unit_support or {}).get(unit["unit_id"])
        if support is not None:
            projected_unit["supported_fact_ids"] = [
                fact_ids[fid] for fid in support.get("fact_ids", ()) if fid in fact_ids]
            projected_unit["supported_group_ids"] = [
                group_ids[gid] for gid in support.get("group_ids", ()) if gid in group_ids]
        units.append(projected_unit)
    view = {**{k: v for k, v in candidate.items() if k not in {"facts", "expansion_reasons"}},
            "fact_ids": [fact_ids[fid] for fid in candidate["fact_ids"] if fid in fact_ids],
            "expanded_fact_ids": [fact_ids[fid] for fid in candidate["expanded_fact_ids"]
                                  if fid in fact_ids],
            "evidence_group_ids": [group_ids[gid] for gid in candidate["evidence_group_ids"]
                                   if gid in group_ids],
            "evidence_groups": groups,
            "units": units}
    if candidate.get("units"):
        for name in ("text", "limitation", "verification"):
            view.pop(name, None)
    return view


def normalize_unit_fields(verdicts, unit_fields):
    """Convert field-free unit wire verdicts to internal explicit-field verdicts.

    The unit identity is authoritative. Aggregate reviews never call this: their
    PremiseBinding/MeasurementClaim still require an explicit field. A wire
    response that repeats ``field`` is an explicit protocol failure, not a value
    to silently overwrite, so old wrong-field responses cannot be "repaired"
    by inference.
    """
    normalized = []
    for verdict in verdicts:
        unit_id = verdict.get("unit_id")
        if unit_id not in unit_fields:
            raise ValueError("answer.unit_wire_unknown_unit")
        field = unit_fields[unit_id]
        item = dict(verdict)
        for name in ("premise_bindings", "measurement_claims"):
            bindings = []
            for binding in verdict.get(name, ()):
                if "field" in binding:
                    raise ValueError("answer.unit_wire_field_forbidden")
                bindings.append({**binding, "field": field})
            item[name] = bindings
        normalized.append(item)
    return normalized


def _restore_bindings(assessment, aliases, group_aliases):
    """Restore aliases in either an aggregate or a unit-only assessment.

    A unit-only production assessment deliberately has no parent premise or
    measurement arrays. Missing arrays therefore mean "not part of this wire",
    not an empty aggregate approval; authorization remains the per-unit list.
    """
    restored = dict(assessment)
    if "premise_bindings" in assessment:
        restored["premise_bindings"] = [{**b,
            "fact_ids": [aliases.get(fid, "unknown_alias:" + fid) for fid in b["fact_ids"]],
            "group_refs": [{**r, "group_id": group_aliases.get(
                r["group_id"], "unknown_group:" + r["group_id"])}
                           for r in b.get("group_refs", [])]}
            for b in assessment["premise_bindings"]]
    if "measurement_claims" in assessment:
        restored["measurement_claims"] = [{**m,
            "evidence_group_id": (
                None if m.get("evidence_group_id") is None else
                group_aliases.get(m["evidence_group_id"],
                                  "unknown_group:" + m["evidence_group_id"]))}
            for m in assessment["measurement_claims"]]
    if "numeric_comparisons" in assessment:
        restored["numeric_comparisons"] = [{**c,
            "operand_fact_ids": [aliases.get(fid, "unknown_alias:" + fid)
                                 for fid in c["operand_fact_ids"]]}
            for c in assessment["numeric_comparisons"]]
    if assessment.get("unit_verdicts"):
        restored["unit_verdicts"] = [_restore_bindings(u, aliases, group_aliases)
                                     for u in assessment["unit_verdicts"]]
    return restored


def restore_assessment(assessment, aliases, group_aliases):
    """Compatibility wrapper used by aggregate and unit-only readers."""
    return _restore_bindings(assessment, aliases, group_aliases)


def _dependency_cycle_nodes(ids, dependencies):
    """Return only units lying on a directed cycle of the support graph.

    Bounded by the unit-count limit: for each start node, a simple reachability
    walk asks whether that node can reach itself. Units merely reachable *from*
    a cycle are not marked; they are handled by the ordinary failure closure.
    Unknown dependency IDs are not graph nodes and are skipped here.

    Qualification edges are deliberately excluded by the caller. A limitation
    that bounds a proposal and the proposal that must carry that limitation are
    not two proof premises; treating both directions as support would fabricate
    a cycle from a valid "proposal + qualifier" pair.
    """
    cycle = set()
    for start in ids:
        stack = list(dependencies.get(start, ()))
        seen = set()
        while stack:
            node = stack.pop()
            if node == start:
                cycle.add(start)
                break
            if node in seen or node not in dependencies:
                continue
            seen.add(node)
            stack.extend(dependencies[node])
    return cycle


def _qualification_edge(source_role, target_role, target_field):
    """A limitation relation is a qualifier, never a factual proof premise.

    The edge may point from a claim to the limitation it must carry, or from a
    limitation to the claim it bounds. Neither direction makes the limitation
    evidence for the claim's factual content. Only the latter two roles can
    carry numbers or sample facts; the ordinary support DAG still handles
    those.
    """
    return (source_role == "limitation" or target_role == "limitation"
            or target_field == "limitation")


def _gap_binding_error(binding, available_gaps, *, future_prerequisite=False):
    """Validate one known-gap binding against the host's canonical sentence.

    A gap code is a host-owned boundary statement. The verifier may quote that
    sentence exactly, but it may not narrow a generic truncation/projection gap
    into a claim that a specific field, table or result does not exist. The
    omission of material from this context is not evidence that the material
    was never computed or saved. If the host can establish a narrower boundary
    from real coverage, it must register a separate ``available_gaps`` entry
    with its own ID and canonical sentence; no such inference is made here.
    Multiple gap IDs and unknown gap IDs are rejected for a current boundary.

    A proposal/verification unit explicitly marked as unexecuted is different:
    its gaps name future data prerequisites, not a current absence. The quote
    must already occur in the unit text (checked by the caller), and every gap
    ID must still be a known host gap, but no canonical current-absence
    sentence is required. This exemption never applies to a factual,
    interpretive or limitation unit.
    """
    gaps = binding.get("gap_ids", ())
    if future_prerequisite:
        if any(gap_id not in (available_gaps or {}) for gap_id in gaps):
            return "answer.sample_binding_gap_unknown", None
        return None, "future_prerequisite"
    if len(gaps) != 1:
        return "answer.sample_gap_quote_not_exact", None
    gap_id = gaps[0]
    if gap_id not in (available_gaps or {}):
        return "answer.sample_binding_gap_unknown", None
    canonical = available_gaps[gap_id].rstrip("。.")
    quote = binding["quote"].rstrip("。.")
    if quote != canonical:
        return "answer.sample_gap_quote_not_exact", None
    return None, "exact"


# Explicit compatibility relations, not a linear strength ladder. A wider test
# may legitimately support a narrower descriptive conclusion.
_DECISION_TEST_SCOPES = {
    "between_group_difference": {"between_group", "population"},
    "population_inference": {"population"},
    "causal_inference": {"causal"},
}
_DECISION_INFERENCE_SCOPES = {
    "between_group_difference": {"between_group", "population", "causal"},
    "population_inference": {"population", "causal"},
    "causal_inference": {"causal"},
}
_QUESTION_DECISIONS = {
    "within_group_structure": {"descriptive_only", "inconclusive"},
    "between_group_difference": {"descriptive_only", "inconclusive",
                                 "between_group_difference"},
    "population_effect": {"descriptive_only", "inconclusive",
                          "between_group_difference", "population_inference"},
}


def _unit_scope_error(unit, verdict):
    """Check structure/text applicability and explicit test-decision compatibility.

    The semantic verifier must compare the structure with the actual unit text
    through ``scope_check``. With no supplied scope, it must explicitly choose
    ``not_applicable``, ``descriptive_only``, or ``inferential_claim``; the last
    rejects a statistical inference that omitted its structure. Local code never
    infers statistical scope from Chinese keywords; it only rejects
    contradictions visible in the declared structure. ``data_status`` says
    whether the data already exist; it is
    orthogonal to ``supported_decision`` and never bypasses the test/inference
    compatibility checks. A future collection plan must still use a test that
    can support its declared conclusion.
    """
    scope = unit.get("statistical_scope")
    check = verdict.get("scope_check")
    if not scope:
        # A new wire verdict must say whether the unit is non-statistical,
        # descriptive-only, or asserts an inference that lacks a declared
        # statistical scope. Legacy aggregate/history verdicts may omit the
        # field; that omission is not promoted to success for new unit reviews,
        # including when a mixed verifier batch falls back to the aggregate
        # response schema. Old answers remain readable, but old omission cannot
        # become a fresh verification success.
        if check is None:
            return "answer.unit_scope_check_missing"
        status = check.get("status")
        if status in {"consistent", "mismatch", "inferential_claim"}:
            return "answer.unit_statistical_scope_missing"
        if status in {"not_applicable", "descriptive_only"}:
            return None
        return "answer.unit_scope_check_missing"
    if check is None or check.get("status") == "not_applicable":
        return "answer.unit_scope_check_missing"
    if check.get("status") == "mismatch":
        return "answer.unit_scope_text_mismatch"
    if check.get("status") in {"descriptive_only", "inferential_claim"}:
        # Those statuses apply only when no bounded scope was supplied.
        return "answer.unit_scope_text_mismatch"
    decision = scope["supported_decision"]
    if decision not in {"descriptive_only", "between_group_difference",
                        "population_inference", "causal_inference", "inconclusive"}:
        return "answer.unit_inference_scope_mismatch"
    if scope.get("data_status") not in {"available", "future_required", "unknown"}:
        return "answer.unit_inference_scope_mismatch"
    if scope["question"] in _QUESTION_DECISIONS and \
            decision not in _QUESTION_DECISIONS[scope["question"]]:
        return "answer.unit_inference_scope_mismatch"
    tested = _DECISION_TEST_SCOPES.get(decision)
    inferred = _DECISION_INFERENCE_SCOPES.get(decision)
    if tested is not None and scope["tested_scope"] not in tested:
        return "answer.unit_inference_scope_mismatch"
    if inferred is not None and scope["inference_scope"] not in inferred:
        return "answer.unit_inference_scope_mismatch"
    return None


def _unit_citation_error(unit, verdict):
    """A unit that actually cites literature needs a semantic support verdict."""
    if not unit.get("citation_ids"):
        return None
    if verdict.get("citation_check") == "unsupported":
        return "answer.unit_citation_unsupported"
    if verdict.get("citation_check") != "supported":
        return "answer.unit_citation_check_missing"
    return None


def _unit_applicability_error(unit, verdict, records, applicable_reference_ids):
    """Enforce the verifier's per-unit assertion nature with local evidence.

    The semantic verifier decides what the unit actually asserts. This function
    does not parse prose or use a keyword table; it only enforces the declared
    contract:

    * a current-cluster identity, mechanism or reference-transfer assertion
      requires a same-unit sample fact and a same-unit condition-matched
      ``scientific_evidence`` citation;
    * another unit's fact/citation, a claim-level citation, a method reference
      or a harmless sibling never satisfies that requirement;
    * a non-biological/local/method/metadata/unexecuted-check unit may declare
      that matched scientific evidence is not required, but it still has to
      pass the ordinary fact/gap/measurement review;
    * missing or contradictory applicability fields fail closed.
    """
    nature = verdict.get("assertion_nature")
    applicability = verdict.get("evidence_applicability")
    if nature is None or applicability is None or not verdict.get("applicability_reason"):
        return "answer.unit_applicability_missing"
    biological = nature in BIOLOGICAL_ASSERTION_NATURES
    if biological and applicability != "matched_scientific_evidence_required":
        return "answer.unit_applicability_mismatch"
    if not biological and applicability == "matched_scientific_evidence_required":
        biological = True
    if not biological:
        return None
    same_unit_facts = {fid for fid in unit.get("evidence_refs", ()) if fid in records}
    same_unit_facts.update(fid for binding in verdict.get("premise_bindings", ())
                           for fid in binding.get("fact_ids", ()) if fid in records)
    if not same_unit_facts:
        return "answer.unit_biological_sample_evidence_missing"
    citations = unit.get("citation_ids", ())
    applicable = applicable_reference_ids or set()
    if not any(cid in applicable for cid in citations):
        return "answer.unit_applicable_biological_evidence_missing"
    return None


def _bind_host_gap_choice(verdict, unit):
    """Bind a returned gap ID to the host's canonical sentence for that unit.

    The verifier chooses a gap ID; it does not author the canonical prose. When
    this unit's host-provided ``host_gap_bindings`` contains exactly the chosen
    ID, an inexact/paraphrased quote is replaced by that registered sentence.
    The sentence already occurs verbatim in the unit text by construction, and
    ``_gap_binding_error`` still runs unchanged over the result. Unknown IDs or
    a gap not registered for this unit are never canonicalized.
    """
    known = {g["gap_id"]: g["quote"] for g in unit.get("host_gap_bindings", ())}
    if not known:
        return verdict
    field_text = unit.get(unit.get("field", "text"), "")
    changed = False
    bindings = []
    for binding in verdict["premise_bindings"]:
        gaps = binding.get("gap_ids", ())
        if (len(gaps) == 1 and gaps[0] in known
                and binding.get("quote") not in field_text):
            # Keep the verifier's returned wording next to the host canonical
            # sentence so the substitution stays reviewable in the audit.
            binding = {**binding, "quote": known[gaps[0]],
                       "quote_original": binding.get("quote")}
            changed = True
        bindings.append(binding)
    return {**verdict, "premise_bindings": tuple(bindings)} if changed else verdict


def _fact_metric(fact, group=None):
    """Recorded metric identity from the record contract, never inferred prose.

    An evidence group names each metric fact explicitly through
    ``metric_fact_ids``; that mapping is authoritative even when two metrics
    (``pct_in``/``pct_out``) differ only by their role. For an ungrouped
    scalar, key segments are machine identity, so a trailing positional index
    (``cluster_size.0``) is not a metric name.
    """
    if group:
        for metric, fact_id in (group.get("metric_fact_ids") or {}).items():
            if fact_id == fact.get("id"):
                return metric
    parts = [part for part in
             str(fact.get("key", "")).replace("/", ".").split(".") if part]
    while parts and parts[-1].isdigit():
        parts.pop()
    return parts[-1] if parts else ""


_DETECTION_FREQUENCY_METRICS = frozenset({"pct_in", "pct_out"})


def _comparison_family(metric):
    """Scientific caliber of a recorded metric, not just its column name.

    ``pct_in`` and ``pct_out`` are two sides of one detection-frequency
    measurement and are directly comparable within a record. A detection
    difference, a cluster count and an expression effect each answer a
    different question and cannot be pooled with that level.
    """
    if metric in _DETECTION_FREQUENCY_METRICS:
        return "detection_frequency"
    return metric


def _group_for_fact(groups, fact_id):
    for group in groups:
        if group.get("gene_fact_id") == fact_id:
            return group
        if fact_id in (group.get("metric_fact_ids") or {}).values():
            return group
    return None


def _group_source(group):
    source = group.get("source")
    if isinstance(source, dict):
        return source.get("source_call_id") or source.get("source")
    return source


def _comparison_domain(group):
    """Scope shared by comparable recorded rows.

    Rank and cluster label are row positions, not scientific calibers: the same
    source/revision can compare its rank-1 row with its rank-3 row. Dataset,
    revision and source are checked separately, so omitting position here does
    not permit cross-call borrowing.
    """
    comparison = group.get("comparison") or {}
    return (group.get("groupby"), comparison.get("kind"))


def _numeric_comparison_error(unit, verdict, records, groups, bound_ids,
                              unit_support_facts):
    """Recompute declared numeric relations from the same recorded values.

    The verifier chooses which recorded facts its sentence compares and states
    the metric and relation. The host rejects a declaration whose span is not
    in this unit, whose operands were not bound by the same unit, whose
    revision/source/metric/recorded comparison domain differ, or whose actual
    values satisfy a different relation. It does not inspect the rest of the
    prose for numbers or keywords.
    """
    comparisons = verdict.get("numeric_comparisons", ())
    if not comparisons:
        return None
    for comparison in comparisons:
        quote = comparison["quote"]
        if quote not in unit.get("text", ""):
            return "answer.unit_numeric_comparison_basis_invalid"
        operand_ids = list(comparison["operand_fact_ids"])
        if len(set(operand_ids)) != len(operand_ids) or any(
                fid not in records for fid in operand_ids):
            return "answer.unit_numeric_comparison_basis_invalid"
        if bound_ids is not None and not set(operand_ids) <= set(bound_ids):
            # A comparison may not borrow a fact that this unit did not state
            # as a premise, even if the fact exists elsewhere in the batch.
            return "answer.unit_numeric_comparison_basis_invalid"
        if unit_support_facts is not None and not set(operand_ids) <= set(unit_support_facts):
            return "answer.unit_numeric_comparison_basis_invalid"
        facts = [records[fid] for fid in operand_ids]
        located = [_group_for_fact(groups, fid) for fid in operand_ids]
        metrics = [_fact_metric(fact, group)
                   for fact, group in zip(facts, located)]
        families = {_comparison_family(metric) for metric in metrics}
        declared_family = _comparison_family(comparison["metric"])
        if (len(families) != 1 or declared_family not in families
                or comparison["metric"] not in metrics):
            # pct_in and pct_out are the two sides of one detection-frequency
            # caliber and may be compared. A detection level, a detection
            # difference and an expression effect answer different questions,
            # and the declared metric must name an actual operand metric.
            return "answer.unit_numeric_comparison_basis_invalid"
        values = []
        for fact in facts:
            value = fact.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                return "answer.unit_numeric_comparison_basis_invalid"
            values.append(value)
        if len({fact.get("revision_id") for fact in facts}) != 1:
            return "answer.unit_numeric_comparison_basis_invalid"
        if any(located):
            if not all(located):
                return "answer.unit_numeric_comparison_basis_invalid"
            if any(group.get("revision_id") != fact.get("revision_id")
                   for group, fact in zip(located, facts)):
                return "answer.unit_numeric_comparison_basis_invalid"
            if len({_group_source(group) for group in located}) != 1:
                return "answer.unit_numeric_comparison_basis_invalid"
            if len({_comparison_domain(group) for group in located}) != 1:
                return "answer.unit_numeric_comparison_basis_invalid"
        else:
            sources = {fact.get("source_call_id") for fact in facts}
            if len(sources) != 1:
                return "answer.unit_numeric_comparison_basis_invalid"
        relation = comparison["relation"]
        if relation in {"maximum", "minimum"}:
            target = values[0]
            others = values[1:]
            holds = (all(target >= value for value in others) if relation == "maximum"
                     else all(target <= value for value in others))
        else:
            if len(values) != 2:
                return "answer.unit_numeric_comparison_basis_invalid"
            left, right = values
            holds = {"less_than": left < right,
                     "less_or_equal": left <= right,
                     "greater_than": left > right,
                     "greater_or_equal": left >= right,
                     "equal": left == right,
                     "not_equal": left != right}[relation]
        if not holds:
            return "answer.unit_numeric_relation_contradicted"
    return None


def _background_mixed_reason(unit, verdict):
    """Return why a declared background is actually a sample/mixed unit.

    The host does not parse prose or use word lists. It uses the structured
    semantic fields the verifier already supplies. A pure source/method
    background has no current-sample evidence binding, measurement, numeric
    comparison, statistical scope, or sample/plan assertion nature. Anything
    else is mixed and must not inherit the background N/A set.
    """
    if (unit.get("evidence_refs") or unit.get("statistical_scope")
            or unit.get("planned_inputs")):
        return "sample_unit_fields"
    bindings = verdict.get("premise_bindings") or ()
    if any(binding.get("fact_ids") or binding.get("gap_ids")
           or binding.get("group_refs") for binding in bindings):
        return "sample_premise_binding"
    if verdict.get("measurement_claims"):
        return "measurement_claim"
    if verdict.get("numeric_comparisons"):
        return "numeric_comparison"
    scope_status = (verdict.get("scope_check") or {}).get("status")
    if scope_status != "not_applicable":
        return "scope_check:" + str(scope_status)
    nature = verdict.get("assertion_nature")
    if nature not in BACKGROUND_SAFE_ASSERTION_NATURES:
        return "assertion_nature:" + str(nature)
    return None


def _effective_review_type(unit, verdict):
    """Resolve the semantic class that governs this unit's obligations.

    The verifier's own classification is authoritative and is the only source
    that can waive a dimension. A missing verifier class on a legacy verdict
    keeps the original strict review; it never becomes an implicit safe class.
    A generator label can only escalate to the stricter inference class, and
    structured verifier declarations (biological/conditional assertion nature,
    an inferential scope declaration, or sample evidence/measurement/comparison
    attached to a declared background) also force stricter review. That keeps a
    collection/background label from hiding a current identity, mechanism,
    conditional decision rule, sample fact or measurement obligation.
    """
    declared = verdict.get("review_type")
    if declared is None:
        return None, "legacy_strict", []
    effective = declared
    escalations = []
    if unit.get("review_type") == "inference" and effective != "inference":
        effective = "inference"
        escalations.append("generator_declared_inference")
    nature = verdict.get("assertion_nature")
    if (nature in BIOLOGICAL_ASSERTION_NATURES
            or nature in CONDITIONAL_ASSERTION_NATURES) and effective != "inference":
        effective = "inference"
        escalations.append("assertion_nature:" + str(nature))
    scope_status = (verdict.get("scope_check") or {}).get("status")
    if scope_status in {"inferential_claim", "mismatch"} and effective != "inference":
        effective = "inference"
        escalations.append("scope_check:" + str(scope_status))
    if effective == "background":
        mixed_reason = _background_mixed_reason(unit, verdict)
        if mixed_reason is not None:
            # A verifier-declared background that carries sample evidence,
            # measurement, comparison or an inference structure is not pure.
            # Fail closed to the full inference obligation rather than letting
            # it escape through the source-only N/A set.
            effective = "inference"
            escalations.append("background_mixed:" + mixed_reason)
    source = "verifier" if effective == declared else "verifier_upgraded"
    return effective, source, escalations


def validate_units(candidate, verdicts, records, available_gaps=None, current_question_ids=None,
                   *, external_units=None, record_index=None, verified_reference_ids=None,
                   applicable_reference_ids=None, support_fact_ids=None,
                   support_group_ids=None):
    """Validate proof contracts per unit and then close rejection over dependencies.

    Legacy whole-record reviews are intentionally not promoted to unit reviews.
    Removing an independent bad limitation can retain a fact; removing a premise
    cannot retain any conclusion that requires that premise.

    Dependencies have two distinct meanings:

    * Factual/method support edges are a proof DAG. They retain cycle,
      unknown-reference and failed-premise propagation.
    * Limitation relations are qualifications. A claim that must carry a
      limitation and a limitation that bounds a claim are both valid, and the
      two directions are not a proof cycle. A failed qualification still blocks
      the unit that relies on it; limitations never substitute for factual
      evidence.

    Unit coverage itself remains strict and all-or-nothing. A local graph defect
    fails only the offending unit: every unit is still reviewed, cycle nodes and
    unknown-dependency nodes receive per-unit errors, and the ordinary failed-
    dependency closure blocks downstream units. Independent units that pass are
    retained, while all dependency edges remain in the verdict for audit.

    A cross-record dependency is resolved only to one unique, already reviewed
    unit in an earlier reasoning record. Ambiguous, forward or unresolved
    references are explicit graph defects rather than same-name guesses.

    Requirement coverage is the verifier's confirmed IDs intersected with the
    explicitly supplied current question IDs. Generator declarations remain
    advisory provenance and must not gate confirmed coverage. Without explicit
    current question IDs, coverage fails closed to empty.
    """
    units = candidate.get("units", [])
    current_questions = set(current_question_ids or ())

    def invalid(code):
        return {"allowed": False, "error_code": code, "bindings": [],
                "unit_verdicts": [], "rendered_unit_ids": [], "bound_fact_ids": []}

    ids = [u["unit_id"] for u in units]
    reviewed = [v["unit_id"] for v in verdicts]
    if not units or len(set(ids)) != len(ids) or len(set(reviewed)) != len(reviewed) or set(ids) != set(reviewed):
        return invalid("answer.unit_coverage_invalid")
    by_id = {v["unit_id"]: v for v in verdicts}
    units_by_id = {u["unit_id"]: u for u in units}
    dependencies = {uid: list(dict.fromkeys(
        [*units_by_id[uid]["depends_on"], *by_id[uid].get("required_dependencies", [])]))
        for uid in ids}
    known_ids = set(ids)

    def effective_support(uid, values, seen=()):
        """A unit may bind its own closure and named local dependencies' closures.

        Dependency edges are the ones already validated for this unit (author
        edges plus verifier-required local premises). Unrelated siblings are
        never merged merely because they share a reasoning record.
        """
        if values is None:
            return None
        result = set(values.get(uid, ()))
        for dep in dependencies[uid]:
            if dep in known_ids and dep not in seen and dep != uid:
                result.update(effective_support(dep, values, (*seen, uid)))
        return result
    support_deps, qualification_deps = {}, {}
    external_support, external_qualification, external_deps = {}, {}, {}
    unknown_units, cross_scope_units, cross_scope_ids = set(), set(), {}
    for uid in ids:
        source = units_by_id[uid]
        support, qualification, ext_support, ext_qualification = [], [], [], []
        unresolved, cross = [], []
        for dep in dependencies[uid]:
            if dep in known_ids:
                target = units_by_id[dep]
                if _qualification_edge(source["role"], target["role"], target["field"]):
                    qualification.append(dep)
                else:
                    support.append(dep)
                continue
            # A bare ID from another reasoning record is not a local miss. It
            # is acceptable only when exactly one earlier, already reviewed
            # unit owns that ID. Same-name ambiguity or a forward reference is
            # an explicit protocol defect, never a first-match guess.
            matches = []
            if external_units is not None:
                matches = [entry for entry in external_units.get(dep, ())
                           if record_index is None
                           or entry.get("record_index", -1) < record_index]
            if not matches:
                unresolved.append(dep)
            elif len(matches) > 1 or matches[0].get("allowed") is None:
                cross.append(dep)
            else:
                target = matches[0]
                external_deps[dep] = target
                if _qualification_edge(source["role"], target.get("role"), target.get("field")):
                    ext_qualification.append(dep)
                else:
                    ext_support.append(dep)
        support_deps[uid] = support
        qualification_deps[uid] = qualification
        external_support[uid] = ext_support
        external_qualification[uid] = ext_qualification
        if unresolved:
            unknown_units.add(uid)
        if cross:
            cross_scope_units.add(uid)
            cross_scope_ids[uid] = cross
    cycle_units = _dependency_cycle_nodes(ids, support_deps)
    decisions = {}
    for unit in units:
        verdict = _bind_host_gap_choice(by_id[unit["unit_id"]], unit)
        review_type, review_type_source, review_type_escalations = \
            _effective_review_type(unit, verdict)
        enforced_dimensions, waived_dimensions = _review_obligations(review_type)
        # A unit's bindable support is its own explicit references plus the
        # closures of local units it actually depends on. The shared catalog
        # may be wider, but another unit's facts/groups are not this unit's
        # premises.
        unit_support_facts = effective_support(unit["unit_id"], support_fact_ids)
        unit_support_groups = effective_support(unit["unit_id"], support_group_ids)
        all_groups = candidate.get("evidence_groups", [])
        groups = ([g for g in all_groups if g["group_id"] in unit_support_groups]
                  if unit_support_groups is not None else all_groups)
        # Use the existing strict fact/gap/measurement validator on precisely
        # this unit. A matching quote elsewhere cannot validate this unit.
        scoped = {**candidate, "text": "", "limitation": "", "verification": ""}
        scoped[unit["field"]] = unit["text"]
        if unit_support_groups is not None:
            scoped["evidence_groups"] = groups
            scoped["evidence_group_ids"] = [g["group_id"] for g in groups]
        measurements = verdict["measurement_claims"]
        observed_measurements = [
            m for m in measurements
            if m.get("measurement_status", "observed") == "observed"]
        # Units unrelated to expression (e.g. an execution-boundary limitation)
        # do not inherit a measurement obligation from another unit.
        referenced = set(unit["evidence_refs"])
        # A catalog row is context, not proof that THIS unit asserts an
        # expression difference. The verifier must separately declare an
        # observed descriptive/inferential claim (scope_check) or supply an
        # explicit observed measurement. Pure proposals, methods, metadata
        # limitations and conditional alternatives do not acquire a measurement
        # claim merely because a marker fact was bound as design context.
        scope_status = (verdict.get("scope_check") or {}).get("status")
        non_observed_natures = {
            "unexecuted_check", "conditional_alternative_hypothesis",
            "method_semantics", "metadata_or_quality_limitation",
            "non_biological_other"}
        # ``measurement_status`` on each supplied claim is the time/purpose axis.
        # ``data_status`` says only whether inputs exist and therefore never
        # decides whether a design was executed. ``scope_check=consistent``
        # likewise only says the text agrees with the design, not that a result
        # was observed. A semantically reviewed unexecuted proposal/conditional
        # design can be a plan with either available or future_required inputs;
        # for a unit that supplies no claim, that declaration is the only basis
        # for not requiring an observed measurement. Any observed
        # descriptive/inferential/mismatch declaration (or a legacy verdict with
        # no scope) still carries the ordinary obligation, including when an
        # unexecuted/method label is attached to a sentence reporting a current
        # statistic.
        # The time/purpose declaration governs, not the presentation role. A
        # conditional interpretation that proposes a readout is still a plan;
        # one that reports a current statistic is not.
        unexecuted_design = (
            scope_status == "consistent"
            and verdict.get("assertion_nature") in {
                "unexecuted_check", "conditional_alternative_hypothesis"})
        asserts_observed_measurement = (
            not unexecuted_design and (
                scope_status is None
                or scope_status in {"descriptive_only", "inferential_claim",
                                    "consistent", "mismatch"}))
        quantitative_bound_ids = {fid for binding in verdict["premise_bindings"]
                                  for fid in binding["fact_ids"]}
        # Planned claims do not satisfy an observed-measurement obligation.
        scoped["_measurement_required"] = bool(observed_measurements) or (
            asserts_observed_measurement and any(
                g["gene_fact_id"] in quantitative_bound_ids or
                quantitative_bound_ids.intersection(g["metric_fact_ids"].values())
                for g in groups))
        # Metadata refs remain resolvable even when this unit has no numerical
        # expression claim and therefore no measurement classification obligation.
        check = {**verdict, "judgment": verdict["status"]}
        # The no-fact shortcut belongs only to an explicitly unexecuted check.
        # A proposal role is not a blanket exemption: a conditional alternative
        # still needs its factual premises, and a proposal that actually asserts
        # current-sample biology is rejected by the applicability gate below.
        proposal = (unit["role"] == "proposal" and unit["field"] == "verification"
                    and verdict.get("assertion_nature") == "unexecuted_check")
        if proposal and not verdict["premise_bindings"] and not referenced and not measurements:
            checks = verdict["checks"]
            correct_dimensions = len(checks) == 4 and {c["dimension"] for c in checks} == set(DIMENSIONS)
            blocked = [c["dimension"] for c in checks
                       if c["status"] not in {"supported", "not_applicable"}
                       and c["dimension"] not in waived_dimensions]
            allowed = correct_dimensions and verdict["status"] == "supported" and verdict["relevant"] and all(
                c["status"] in {"supported", "not_applicable"}
                or c["dimension"] in waived_dimensions for c in checks) and next(
                c["status"] for c in checks if c["dimension"] == "premise_coverage"
                ) in {"supported", "not_applicable"}
            decision = {"allowed": allowed, "error_code": None if allowed else "answer.unit_proposal_rejected",
                        "blocked_dimensions": blocked, "bindings": [], "bound_fact_ids": [],
                        "waived_dimensions": waived_dimensions,
                        "enforced_dimensions": enforced_dimensions,
                        "basis": "unexecuted_verification_proposal"}
        else:
            decision = validate_assessment(
                scoped, check, records, available_gaps,
                unit_role=unit["role"], unit_field=unit["field"],
                review_type=review_type,
                referenced_evidence_ids=tuple(referenced),
                qualification_dependencies=tuple(
                    [*qualification_deps[unit["unit_id"]],
                     *external_qualification[unit["unit_id"]]]),
                verified_reference_ids=verified_reference_ids,
                unit_citation_ids=tuple(unit.get("citation_ids", ())))
        # Repeated identical IDs are one reference. An unknown ID is never
        # dropped to make the unit pass.
        if any(fid not in records for fid in referenced):
            decision = {"allowed": False, "error_code": "answer.unit_evidence_unknown",
                        "bindings": [], "bound_fact_ids": []}
        # A planned readout is expected to use not_applicable (no current
        # statistical interpretation) or consistent (the text matches the
        # design). An observed claim under a non-observed declaration is a
        # contradiction: it cannot be exempted by the unexecuted/method label.
        if (decision["allowed"] and observed_measurements
                and scope_status == "not_applicable"
                and verdict.get("assertion_nature") in non_observed_natures):
            decision = {**decision, "allowed": False,
                        "error_code": "answer.unit_measurement_scope_not_applicable"}
        # A unit that contains a real host-registered table-coverage sentence
        # must use that table-level gap for the table boundary. A generic
        # metric-projection gap may accompany it, but cannot be its only gap
        # premise: generic truncation does not establish a specific table's
        # coverage. This is keyed on actual registered metadata, not prose.
        table_gap_ids = {binding["gap_id"]
                         for binding in unit.get("host_gap_bindings", ())
                         if binding["gap_id"].startswith(
                             "evidence.section_cluster_table.")}
        used_gap_ids = {gap_id for binding in decision.get("bindings", ())
                        for gap_id in binding.get("gaps", {})}
        if (decision["allowed"] and table_gap_ids
                and "context.metric_projection_truncated" in used_gap_ids
                and not (used_gap_ids & table_gap_ids)):
            decision = {**decision, "allowed": False,
                        "error_code": "answer.sample_gap_not_table_specific"}
        if decision["allowed"] and unit_support_facts is not None:
            bound_ids = set(decision.get("bound_fact_ids", ()))
            used_group_ids = {r["group_id"] for b in decision.get("bindings", ())
                              for r in b.get("group_refs", ())}
            used_group_ids.update(m["evidence_group_id"] for m in verdict["measurement_claims"]
                                  if m.get("evidence_group_id") is not None)
            if not bound_ids <= unit_support_facts or (
                    unit_support_groups is not None
                    and not used_group_ids <= unit_support_groups):
                decision = {**decision, "allowed": False,
                            "error_code": "answer.unit_binding_out_of_scope"}
        if decision["allowed"]:
            numeric_error = _numeric_comparison_error(
                unit, verdict, records, groups,
                set(decision.get("bound_fact_ids", ())), unit_support_facts)
            if numeric_error:
                decision = {**decision, "allowed": False, "error_code": numeric_error}
        if decision["allowed"]:
            protocol_error = (_unit_citation_error(unit, verdict)
                              or ("answer.unit_citation_unknown"
                                  if verified_reference_ids is not None and any(
                                      cid not in verified_reference_ids
                                      for cid in unit.get("citation_ids", ()))
                                  else None)
                              or _unit_scope_error(unit, verdict)
                              or _unit_applicability_error(
                                  unit, verdict, records, applicable_reference_ids))
            if protocol_error:
                decision = {**decision, "allowed": False, "error_code": protocol_error}
        decisions[unit["unit_id"]] = {**decision, "unit_id": unit["unit_id"],
            "role": unit["role"], "field": unit["field"], "status": verdict["status"],
            # Same review frame returned to the revision step: planned inputs,
            # the actual host-registered coverage sentences, and this unit's
            # legal current fact/group boundary. None means the caller did not
            # supply a unit-scoped closure (legacy aggregate review).
            "planned_inputs": list(unit.get("planned_inputs", ())),
            "host_gap_bindings": [dict(g) for g in unit.get("host_gap_bindings", ())],
            "supported_fact_ids": (sorted(unit_support_facts)
                                   if unit_support_facts is not None else None),
            "supported_group_ids": (sorted(unit_support_groups)
                                    if unit_support_groups is not None else None),
            "review_type": review_type,
            "generated_review_type": unit.get("review_type"),
            "review_type_source": review_type_source,
            "review_type_escalations": review_type_escalations,
            "enforced_dimensions": enforced_dimensions,
            "waived_dimensions": waived_dimensions,
            # The actual units whose evidence/controls jointly support this
            # inference's identifiability. Only declared support dependencies
            # count; unrelated siblings in the same record do not.
            "identifiability_support_unit_ids": (
                list(dict.fromkeys([*support_deps[unit["unit_id"]],
                                    *external_support[unit["unit_id"]]]))
                if review_type == "inference" else []),
            "assertion_nature": verdict.get("assertion_nature"),
            "evidence_applicability": verdict.get("evidence_applicability"),
            "applicability_reason": verdict.get("applicability_reason"),
            "scope_check": verdict.get("scope_check"),
            "numeric_comparisons": [dict(c) for c in verdict.get("numeric_comparisons", ())],
            "depends_on": dependencies[unit["unit_id"]],
            "support_depends_on": support_deps[unit["unit_id"]],
            "qualification_depends_on": [*qualification_deps[unit["unit_id"]],
                                         *external_qualification[unit["unit_id"]]],
            "external_depends_on": [*external_support[unit["unit_id"]],
                                    *external_qualification[unit["unit_id"]]],
            "fulfilled_requirement_ids": list(dict.fromkeys(
                qid for qid in verdict.get("fulfilled_requirement_ids", ())
                if qid in current_questions))}
    # A locally valid unit can still be a graph defect. Only nodes actually on a
    # support cycle, declaring an unknown dependency, or using an unresolvable
    # cross-record dependency take the graph error codes; the rest of each
    # verdict (bindings, dimensions, edges) is preserved for audit.
    for uid in ids:
        if uid in cycle_units:
            decisions[uid].update(allowed=False, error_code="answer.unit_dependency_cycle")
        elif uid in unknown_units:
            decisions[uid].update(allowed=False, error_code="answer.unit_dependency_unknown")
        elif uid in cross_scope_units:
            decisions[uid].update(allowed=False, error_code="answer.unit_dependency_cross_scope",
                                  cross_scope_dependencies=cross_scope_ids[uid])
    # DAG closure is bounded by unit count; independent accepted units survive.
    # Unknown dependency IDs have no decision and must not raise KeyError.
    # Support failure and qualification failure remain distinguishable: a
    # missing proof premise cannot be repaired by a limitation, while a
    # rejected required limitation blocks the qualified claim.
    for _ in units:
        for uid, decision in decisions.items():
            if not decision["allowed"]:
                continue
            failed_support = [dep for dep in support_deps[uid]
                              if dep in decisions and not decisions[dep]["allowed"]]
            failed_support.extend(dep for dep in external_support[uid]
                                  if external_deps[dep].get("allowed") is False)
            if failed_support:
                decision.update(allowed=False, error_code="answer.unit_dependency_rejected",
                                failed_dependencies=list(dict.fromkeys(failed_support)))
                continue
            failed_qualification = [dep for dep in qualification_deps[uid]
                                    if dep in decisions and not decisions[dep]["allowed"]]
            failed_qualification.extend(dep for dep in external_qualification[uid]
                                        if external_deps[dep].get("allowed") is False)
            if failed_qualification:
                decision.update(allowed=False, error_code="answer.unit_qualification_rejected",
                                failed_dependencies=list(dict.fromkeys(failed_qualification)))
    # Only units that survive every check and dependency closure may claim
    # coverage; generator declarations never restore it.
    for decision in decisions.values():
        if not decision["allowed"]:
            decision["fulfilled_requirement_ids"] = []
    rendered = [uid for uid in ids if decisions[uid]["allowed"]]
    # Preserve the established cycle code as the primary graph error when both
    # graph defects occur; unknown-only failures still report their own code.
    if len(rendered) == len(ids):
        error_code = None
    elif cycle_units:
        error_code = "answer.unit_dependency_cycle"
    elif unknown_units:
        error_code = "answer.unit_dependency_unknown"
    elif cross_scope_units:
        error_code = "answer.unit_dependency_cross_scope"
    else:
        error_code = "answer.partial_units_rejected"
    return {"allowed": len(rendered) == len(ids), "error_code": error_code,
            "unit_verdicts": [decisions[uid] for uid in ids], "rendered_unit_ids": rendered,
            "bindings": [b for uid in rendered for b in decisions[uid]["bindings"]],
            "bound_fact_ids": list(dict.fromkeys(fid for uid in rendered
                for fid in decisions[uid].get("bound_fact_ids", []))),
            "semantic_judge": "fallible_provider", "independent_scientific_validation": False}


def validate_assessment(candidate, assessment, records, available_gaps=None, *,
                        unit_role=None, unit_field=None, review_type=None,
                        referenced_evidence_ids=(), qualification_dependencies=(),
                        verified_reference_ids=None, unit_citation_ids=()):
    """Return one local decision; a bad proof does not invalidate other candidates.

    Aggregate/legacy candidates keep the strict binding contract. Unit reviews
    additionally distinguish factual evidence bindings from three bounded
    structural bases:

    * an unexecuted future proposal;
    * a scope/qualification sentence whose supporting facts or qualifier are
      already present on the same unit;
    * a pure method reference backed by verified citations or known evidence.

    Those bases only avoid a fake sample fact or gap binding. They do not make
    the verifier's checks pass, and they never satisfy a factual premise.
    """
    checks = assessment["checks"]
    if len(checks) != len(DIMENSIONS) or {c["dimension"] for c in checks} != set(DIMENSIONS):
        return {"allowed": False, "error_code": "answer.sample_check_dimensions_invalid", "bindings": []}
    bindings = []
    groups = {g["group_id"]: g for g in candidate["evidence_groups"]}
    unit_context = unit_role is not None
    premise_status = next(c["status"] for c in checks
                          if c["dimension"] == "premise_coverage")
    nature = assessment.get("assertion_nature")
    # Nonfactual process/method, bookkeeping and qualification statements do
    # not invent a measured sample fact or a paper citation. This exemption is
    # semantic: the verifier must mark premise coverage not_applicable (or the
    # unit must carry an actual qualification dependency). A current-sample or
    # external factual premise still follows the ordinary binding checks.
    nonfactual_nature = nature in {
        "method_semantics", "non_biological_other",
        "metadata_or_quality_limitation"}
    raw_bindings = assessment["premise_bindings"]
    has_any_evidence_binding = any(
        b.get("fact_ids") or b.get("gap_ids") or b.get("group_refs") for b in raw_bindings)
    referenced_known = bool(referenced_evidence_ids)
    cited_verified = bool(unit_citation_ids) and verified_reference_ids is not None and all(
        cid in verified_reference_ids for cid in unit_citation_ids)
    for binding in raw_bindings:
        # Repeated identical IDs are one reference; order is preserved.
        ids = tuple(dict.fromkeys(binding["fact_ids"]))
        gaps = binding.get("gap_ids", [])
        group_evidence = []
        for ref in binding.get("group_refs", []):
            group = groups.get(ref["group_id"])
            if group is None:
                return {"allowed": False, "error_code": "answer.sample_binding_group_unknown", "bindings": []}
            value = group
            for part in ref["path"].split("."):
                if not isinstance(value, dict) or part not in value:
                    return {"allowed": False, "error_code": "answer.sample_binding_group_path_invalid", "bindings": []}
                value = value[part]
            group_evidence.append({**ref, "value": value, "revision_id": group["revision_id"],
                                   "source": group["source"]})
        if binding["quote"] not in candidate[binding["field"]]:
            return {"allowed": False, "error_code": "answer.sample_binding_span_invalid", "bindings": []}
        if any(fid not in records for fid in ids):
            return {"allowed": False, "error_code": "answer.sample_binding_fact_unknown", "bindings": []}
        gap_mode = None
        if not ids and not gaps and not group_evidence:
            if binding["field"] == "verification" and (
                    not unit_context or unit_role == "proposal"):
                basis = "unexecuted_verification_proposal"
            elif unit_context and unit_field == binding["field"] and (
                    (unit_role == "interpretation" and
                     (has_any_evidence_binding or referenced_known))
                    or (unit_role == "fact" and has_any_evidence_binding)):
                basis = "unit_scope_statement"
            elif unit_context and unit_field == binding["field"] and (
                    nature == "metadata_or_quality_limitation"
                    and qualification_dependencies):
                basis = "qualification_statement"
            elif unit_context and unit_role == "limitation" and \
                    unit_field == binding["field"] and (
                        has_any_evidence_binding or referenced_known or qualification_dependencies):
                basis = "qualification_statement"
            elif unit_context and unit_role == "method" and \
                    unit_field == binding["field"] and (referenced_known or cited_verified):
                basis = "method_reference"
            elif unit_context and unit_field == binding["field"] and \
                    premise_status == "not_applicable" and nonfactual_nature:
                basis = "nonfactual_scope_statement"
            else:
                return {"allowed": False, "error_code": "answer.sample_binding_gap_unknown", "bindings": []}
        else:
            basis = "evidence_binding"
        if gaps:
            future_prerequisite = (
                unit_context and unit_role == "proposal"
                and unit_field == "verification" and binding["field"] == unit_field
                and assessment.get("assertion_nature") == "unexecuted_check"
                and not ids and not group_evidence)
            error, gap_mode = _gap_binding_error(
                binding, available_gaps,
                future_prerequisite=future_prerequisite)
            if error:
                return {"allowed": False, "error_code": error, "bindings": []}
        bindings.append({**binding, "facts": [records[fid] for fid in ids],
                         "gaps": {gap: available_gaps[gap] for gap in gaps},
                         "group_evidence": group_evidence, "basis": basis,
                         **({"gap_mode": gap_mode} if gap_mode else {})})
    if not bindings:
        structural_basis = None
        if unit_context:
            if unit_role == "proposal" and unit_field == "verification":
                structural_basis = "unexecuted_verification_proposal"
            elif unit_role == "interpretation" and referenced_known:
                structural_basis = "unit_scope_statement"
            elif unit_role == "method" and (referenced_known or cited_verified):
                structural_basis = "method_reference"
            elif unit_role == "limitation" and (referenced_known or qualification_dependencies):
                structural_basis = "qualification_statement"
            elif premise_status == "not_applicable" and nonfactual_nature:
                structural_basis = "nonfactual_scope_statement"
        if structural_basis is None:
            return {"allowed": False, "error_code": "answer.sample_binding_missing", "bindings": []}
    elif not any(b["basis"] == "evidence_binding" for b in bindings):
        allowed_structural = False
        if unit_context:
            if unit_role == "proposal" and unit_field == "verification" and \
                    all(b["basis"] == "unexecuted_verification_proposal" for b in bindings):
                allowed_structural = True
            elif all(b["basis"] in {"unit_scope_statement", "method_reference",
                                    "qualification_statement",
                                    "nonfactual_scope_statement"} for b in bindings):
                allowed_structural = (
                    (unit_role == "interpretation" and referenced_known)
                    or (unit_role == "method" and (referenced_known or cited_verified))
                    or (unit_role == "limitation" and
                        (referenced_known or qualification_dependencies))
                    or (nature == "metadata_or_quality_limitation"
                        and qualification_dependencies)
                    or (premise_status == "not_applicable" and nonfactual_nature))
        if not allowed_structural:
            return {"allowed": False, "error_code": "answer.sample_binding_missing", "bindings": []}
    # A generic 'supported' cannot override a failed enforced dimension. The
    # obligation table decides which dimensions apply: collection keeps true,
    # executable/applicable premises and measurement caliber; pure background
    # keeps only source support; inference keeps all four; legacy/unclassified
    # keeps all four. Every current numeric/sample fact still passes the
    # ordinary local checks regardless of class.
    enforced_dimensions, waived_dimensions = _review_obligations(review_type)
    blocked = [c["dimension"] for c in checks if c["status"] in {"insufficient", "contradicted"}
               and c["dimension"] in enforced_dimensions]
    measurements = assessment["measurement_claims"]
    observed_measurements = [
        m for m in measurements
        if m.get("measurement_status", "observed") == "observed"]
    if ("measurement_semantics" in enforced_dimensions
            and candidate.get("_measurement_required", bool(candidate["evidence_groups"]))
            and not observed_measurements):
        blocked.append("measurement_scope_missing")
    for measurement in measurements:
        status = measurement.get("measurement_status", "observed")
        group_id = measurement.get("evidence_group_id")
        # Quote stays an exact current-unit substring. A non-null group must be
        # one of this scoped candidate's groups; a pure future planned readout
        # may explicitly use null because no current saved group exists. An
        # observed claim never accepts null or a sibling/unknown group.
        if measurement["quote"] not in candidate[measurement["field"]] or (
                group_id is not None and group_id not in candidate["evidence_group_ids"]):
            return {"allowed": False, "error_code": "answer.sample_measurement_binding_invalid", "bindings": bindings}
        if status == "planned":
            # A planned readout is allowed only as part of a semantically
            # reviewed unexecuted proposal/conditional design, and it must not
            # claim a current observed difference or equivalence. Null proves
            # no measurement occurred; it does not satisfy observed.
            # Presentation role is not the time/purpose axis. A reviewed
            # unexecuted or conditional design may carry a planned readout
            # regardless of role, but only with null current group and
            # conclusion=undetermined checked below.
            allowed_planned_context = (
                unit_context
                and assessment.get("assertion_nature") in {
                    "unexecuted_check", "conditional_alternative_hypothesis"})
            if not allowed_planned_context:
                return {"allowed": False,
                        "error_code": "answer.unit_measurement_planned_not_allowed",
                        "bindings": bindings}
            if measurement["conclusion"] != "undetermined":
                return {"allowed": False,
                        "error_code": "answer.unit_measurement_planned_conclusion_invalid",
                        "bindings": bindings}
            continue
        if group_id is None:
            return {"allowed": False, "error_code": "answer.sample_measurement_binding_invalid", "bindings": bindings}
        # Exploratory marker summaries contain no equivalence-test margin or
        # conditional intensity distribution. A non-significant/zero frequency
        # difference cannot establish overall/intensity equivalence. This is a
        # measurement-semantics obligation, so a pure background/source unit
        # with no measurement claim is not asked to satisfy it.
        if ("measurement_semantics" in enforced_dimensions
                and measurement["scope"] != "detection_frequency"
                and measurement["conclusion"] == "equivalent"):
            blocked.append("expression_equivalence_not_established")
    # The semantic obligation table permits not_applicable where there is no
    # factual premise. Only an actual failed coverage blocks here.
    if premise_status in {"insufficient", "contradicted"}:
        blocked = list(dict.fromkeys(["premise_coverage", *blocked]))
    allowed = assessment["judgment"] == "supported" and assessment["relevant"] and not blocked
    return {"allowed": allowed, "error_code": None if allowed else "answer.sample_inference_rejected",
            "blocked_dimensions": blocked, "bindings": bindings,
            "waived_dimensions": waived_dimensions,
            "enforced_dimensions": enforced_dimensions,
            "bound_fact_ids": list(dict.fromkeys(fid for b in bindings for fid in b["fact_ids"])),
            "semantic_judge": "fallible_provider", "independent_scientific_validation": False}


_MARKER_JOINT_FIELDS = (
    "pct_in", "pct_out", "detection_difference", "logfoldchange")
_MARKER_JOINT_LABELS = {
    "pct_in": "簇内检出率",
    "pct_out": "簇外检出率",
    "detection_difference": "检出率差",
    "logfoldchange": "log2FC",
}


def _marker_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _marker_pct(value):
    return format(value * 100, ".4g") + "%"


def _marker_effect(value):
    return format(value, ".4g")


def _marker_fact(records, group, field):
    """Return one same-record fact or a precise missing/conflict reason."""
    fact_id = group.get("metric_fact_ids", {}).get(field)
    if not fact_id:
        return None, {"kind": "missing", "field": field, "fact_id": None}
    fact = records.get(fact_id)
    if fact is None:
        return None, {"kind": "missing", "field": field, "fact_id": fact_id}
    source = group.get("source", {})
    if fact.get("revision_id") != group.get("revision_id"):
        return None, {"kind": "conflict", "field": field, "fact_id": fact_id,
                      "reason": "revision"}
    source_call = source.get("source_call_id")
    fact_call = fact.get("source_call_id")
    # Facts may inherit the call only from validated fact_provenance; an
    # omitted per-fact call is not a conflict. A present call must match.
    if source_call is not None and fact_call is not None and fact_call != source_call:
        return None, {"kind": "conflict", "field": field, "fact_id": fact_id,
                      "reason": "call"}
    dataset_id = group.get("dataset_id")
    if fact.get("dataset_id") is not None and fact["dataset_id"] != dataset_id:
        return None, {"kind": "conflict", "field": field, "fact_id": fact_id,
                      "reason": "dataset"}
    value = fact.get("value")
    if field == "logfoldchange":
        if not _marker_number(value):
            return None, {"kind": "missing", "field": field, "fact_id": fact_id}
        return value, None
    low, high = (-1, 1) if field == "detection_difference" else (0, 1)
    if not _marker_number(value) or not low <= value <= high:
        return None, {"kind": "conflict", "field": field, "fact_id": fact_id,
                      "reason": "scale"}
    return value, None


def render_joint_marker_measurement(group, records):
    """Render one marker record's detection and signed expression effect once.

    The group is already keyed by revision, cluster, rank and exact source, so
    the values come from the same gene/record/comparison. The function never
    borrows a sibling fact and never turns a descriptive cutoff into a test.
    """
    gene_fact = records.get(group.get("gene_fact_id"))
    source = group.get("source", {})
    if gene_fact is None or gene_fact.get("value") != group.get("gene") or \
            gene_fact.get("revision_id") != group.get("revision_id") or \
            (source.get("source_call_id") is not None and
             gene_fact.get("source_call_id") is not None and
             gene_fact.get("source_call_id") != source.get("source_call_id")) or \
            (gene_fact.get("dataset_id") is not None and
             gene_fact["dataset_id"] != group.get("dataset_id")):
        return {"text": "同一marker记录的基因、版本、调用或数据集标识不一致，无法合并判断。",
                "fact_ids": [group["gene_fact_id"]]}
    gene = gene_fact["value"]
    used = [group["gene_fact_id"]]
    values, problems = {}, []
    for field in _MARKER_JOINT_FIELDS:
        value, problem = _marker_fact(records, group, field)
        if problem:
            problems.append(problem)
        else:
            values[field] = value
        if problem is None or problem.get("fact_id"):
            fact_id = problem["fact_id"] if problem else group["metric_fact_ids"].get(field)
            if fact_id:
                used.append(fact_id)
    conflict = next((p for p in problems if p["kind"] == "conflict"), None)
    cluster = group["cluster_id"]
    if conflict:
        label = _MARKER_JOINT_LABELS[conflict["field"]]
        reason = {"revision": "版本", "call": "调用", "dataset": "数据集",
                  "scale": "数值范围"}[conflict["reason"]]
        return {"text": f"{gene}（簇{cluster}）：同一marker记录的{label}{reason}不一致，无法合并判断。",
                "fact_ids": list(dict.fromkeys(used))}
    missing = [p["field"] for p in problems]
    detection_missing = any(field in missing for field in ("pct_in", "pct_out"))
    if detection_missing or "logfoldchange" in missing:
        if "pct_in" in values and "pct_out" in values:
            if values["pct_in"] == values["pct_out"]:
                measurement = (f"{gene} 在簇{cluster}内外检出率均为"
                               f"{_marker_pct(values['pct_in'])}，检出率无区分力")
            else:
                measurement = (f"{gene} 在簇{cluster}内检出率为"
                               f"{_marker_pct(values['pct_in'])}，簇外为"
                               f"{_marker_pct(values['pct_out'])}")
        else:
            measurement = f"{gene}（簇{cluster}）"
        missing_label = ("簇内或簇外检出率" if detection_missing else "log2FC")
        return {"text": measurement + f"；同一marker记录缺少{missing_label}，无法给出完整联合判断。",
                "fact_ids": list(dict.fromkeys(used))}
    pct_in, pct_out = values["pct_in"], values["pct_out"]
    if pct_in == pct_out:
        detection = f"在簇{cluster}内外检出率均为{_marker_pct(pct_in)}，检出率无区分力；"
    else:
        detection = (f"在簇{cluster}内检出率为{_marker_pct(pct_in)}，"
                     f"簇外为{_marker_pct(pct_out)}")
        if "detection_difference" in values:
            detection += f"，检出率差为{_marker_pct(values['detection_difference'])}"
        detection += "；"
    effect = values["logfoldchange"]
    if abs(effect) < MARKER_DESCRIPTIVE_LOG2FC_CUTOFF:
        conclusion = (f"簇{cluster}表达与簇外接近"
                      f"（|log2FC|<{MARKER_DESCRIPTIVE_LOG2FC_CUTOFF}），不作为表达特征。")
    elif effect > 0:
        conclusion = f"簇{cluster}表达更高，可作为表达特征。"
    else:
        conclusion = f"簇{cluster}表达更低，可作为表达特征。"
    return {"text": f"{gene} {detection}log2FC≈{_marker_effect(effect)}，{conclusion}",
            "fact_ids": list(dict.fromkeys(used))}


def render_unit_marker_measurements(unit, groups, records, verdict,
                                    rendered_group_ids=None):
    """Apply the joint marker record policy to a supported inference unit.

    Replacement is limited to what the existing semantic review classified as
    a present local observation on an interpretation unit. Conditional future
    interpretations and biological hypotheses keep their reviewed text.
    Only units whose reviewed references actually select marker detection or
    expression metrics are replaced. A fact/method/proposal unit, an unrelated
    interpretation and a unit without marker metrics keep their reviewed text.
    """
    if unit.get("role") != "interpretation" or \
            unit.get("review_type") != "inference" or \
            verdict.get("assertion_nature") != "local_observation":
        return None
    refs = set(unit.get("evidence_refs", []))
    marker_fact_ids = {
        fact_id for group in groups
        for fact_id in [group.get("gene_fact_id"),
                        *group.get("metric_fact_ids", {}).values()]
        if fact_id}
    # A mixed unit may make a different claim (for example depth or quality).
    # Only a unit whose evidence is entirely this marker record is replaced.
    if refs - marker_fact_ids:
        return None
    selected = []
    for group in groups:
        metric_ids = set(group.get("metric_fact_ids", {}).values())
        if not refs.intersection(metric_ids):
            continue
        # p_adjusted alone is not a detection/expression measurement statement.
        if not refs.intersection({
                group.get("metric_fact_ids", {}).get(field)
                for field in _MARKER_JOINT_FIELDS} - {None}):
            continue
        selected.append(group)
    rendered_group_ids = rendered_group_ids if rendered_group_ids is not None else set()
    texts, used, group_ids = [], [], []
    for group in selected:
        group_id = group["group_id"]
        if group_id in rendered_group_ids:
            continue
        rendered = render_joint_marker_measurement(group, records)
        if rendered is None:
            continue
        rendered_group_ids.add(group_id)
        texts.append(rendered["text"])
        used.extend(rendered["fact_ids"])
        group_ids.append(group_id)
    if not selected:
        return None
    if not texts:
        return {"text": "", "fact_ids": [], "group_ids": [],
                "mode": "joint_marker_measurement_duplicate"}
    return {"text": "".join(texts), "fact_ids": list(dict.fromkeys(used)),
            "group_ids": group_ids, "mode": "joint_marker_measurement_interpretation"}


def render_marker_scope(candidate, records, rendered_group_ids=None):
    """Render a legacy equal-detection interpretation from same-record metrics."""
    groups = candidate["evidence_groups"]
    if candidate["kind"] != "interpretation":
        return None
    candidate_facts = set(candidate.get("fact_ids", []))
    selected = [group for group in groups
                if candidate_facts.intersection(group.get("metric_fact_ids", {}).values())]
    groups = selected or groups
    equal_detection = False
    for group in groups:
        ids = group["metric_fact_ids"]
        left, right = ids.get("pct_in"), ids.get("pct_out")
        left_fact, right_fact = records.get(left, {}), records.get(right, {})
        if left_fact.get("value") is not None and \
                left_fact.get("value") == right_fact.get("value"):
            equal_detection = True
    if not equal_detection:
        return None
    rendered_group_ids = rendered_group_ids if rendered_group_ids is not None else set()
    texts, used = [], []
    for group in groups:
        if group["group_id"] in rendered_group_ids:
            continue
        rendered = render_joint_marker_measurement(group, records)
        if rendered is None:
            continue
        rendered_group_ids.add(group["group_id"])
        texts.append(rendered["text"])
        used.extend(rendered["fact_ids"])
    if not texts:
        return {"text": "", "fact_ids": [],
                "mode": "local_marker_joint_measurement_duplicate"}
    return {"text": "".join(texts), "fact_ids": list(dict.fromkeys(used)),
            "mode": "local_marker_joint_measurement"}
