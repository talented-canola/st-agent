"""Model interpretation; scientific authorization remains local."""

from __future__ import annotations

import json
from pydantic import BaseModel

from st_agent.answer_delivery import ResponseRequirements
from st_agent.deepseek import DeepSeekClient, DeepSeekError, DeepSeekMalformedResponse
from st_agent.planning import TaskIntent, TaskOperation, IntentMode
from st_agent.tool_registry import SCIENTIFIC_TOOLS as TOOLS

OPERATION_TO_TOOL = {
    "inspect": "dataset.inspect",
    "qc_summary": "qc.compute_metrics",
    "qc_filter": "qc.apply_filter",
    "preprocess": "preprocess.standard",
    "cluster": "cluster.leiden_umap",
    "marker": "markers.rank",
    "spatial_neighbors": "spatial.neighbors",
    "spatial_morans": "spatial.morans_i",
    "coordinate_partition": "spatial.domains",
    "spatial_domains": "spatial.expression_domains",
    "sample_comparison": "samples.compare_stages",
}

INTENT_TOOL_NAME = "submit_research_intent"


def intent_provider_tools() -> list[dict]:
    from st_agent.evidence_context import saved_evidence_provider_schema

    schema = TaskIntent.model_json_schema()
    schema["$defs"]["SavedEvidenceSelection"] = saved_evidence_provider_schema()
    schema["$defs"]["MarkerSelection"]["description"] = (
        "Current first-class saved marker selection: genes/clusters and marker metrics "
        "from saved markers.rank records. It has no kind field. Use it for marker "
        "discussions; never put marker metrics under kind=spatial_results. Omission "
        "of kind is this valid marker selection, not an obsolete compatibility path."
    )
    schema["properties"]["parameters"] = {
        "anyOf": [
            TOOLS[name].arguments.model_json_schema()
            for name in dict.fromkeys(OPERATION_TO_TOOL.values())
        ] + [{"type": "object", "additionalProperties": False,
              "properties": {op: {"type": "object",
                  "properties": TOOLS[OPERATION_TO_TOOL[op]].arguments.model_json_schema()["properties"],
                  "additionalProperties": False}
                  for op in ("qc_filter", "preprocess", "cluster", "marker")}}]
    }
    return [
        {
            "type": "function",
            "function": {
                "name": INTENT_TOOL_NAME,
                "description": "Interpret the researcher's request, without executing it.",
                "parameters": schema,
                "strict": True,
            },
        }
    ]


class Interpretation(BaseModel):
    intent: TaskIntent
    response_id: str
    model_calls: int = 1


def _saved_evidence_discussion(intent: TaskIntent) -> bool:
    """An ASK that discusses validated saved evidence needs no execution parameters."""
    return (
        intent.mode is IntentMode.ASK
        and intent.saved_evidence is not None
        and not intent.new_analysis
        and not intent.parameters
    )


def merge_delivery_requirements(action: ResponseRequirements,
                                deferred: ResponseRequirements | None) -> ResponseRequirements:
    """Choose current output fields while preserving every retrieval constraint.

    The deferred directive supplies mode/questions because it describes the
    output requested after the action. ``retrieval_policy`` is a shared user
    constraint, not an output-style field: an explicit non-``auto`` value on
    either side wins over ``auto``. Two different explicit non-``auto`` values
    are contradictory and are rejected rather than relaxed.
    """
    if deferred is None:
        return action
    action_policy = action.retrieval_policy
    deferred_policy = deferred.retrieval_policy
    if action_policy == deferred_policy:
        policy = action_policy
    elif action_policy == "auto":
        policy = deferred_policy
    elif deferred_policy == "auto":
        policy = action_policy
    else:
        raise ValueError("context.delivery_retrieval_conflict")
    return deferred.model_copy(update={"retrieval_policy": policy})


def effective_delivery_requirements(intent: TaskIntent):
    """Current delivery directive, independent of the action continuation."""
    deferred = (intent.deferred_delivery.response_requirements
                if intent.deferred_delivery is not None else None)
    return merge_delivery_requirements(intent.response_requirements, deferred)


def effective_delivery_selection(intent: TaskIntent):
    """A deferred directive's explicit selection wins; omission never clears."""
    if intent.deferred_delivery is not None and \
            intent.deferred_delivery.saved_evidence is not None:
        return intent.deferred_delivery.saved_evidence
    return intent.saved_evidence


def _validate_deferred_delivery(intent: TaskIntent, request: str) -> None:
    delivery = intent.deferred_delivery
    if delivery is None:
        return
    if delivery.quote not in request:
        raise ValueError("context.delivery_quote_not_in_current_request")
    if delivery.response_requirements.mode == "receipt":
        raise ValueError("context.deferred_delivery_requires_output")
    # Surface an impossible retrieval demand during intent validation instead
    # of letting a later delivery boundary silently choose one side.
    effective_delivery_requirements(intent)


def _validated_intent(reply, request):
    """Apply the same local intent and operation checks on every protocol attempt."""
    if len(reply.tool_calls) != 1 or reply.tool_calls[0].name != INTENT_TOOL_NAME:
        detail = DeepSeekMalformedResponse("Expected exactly one structured research intent")
        detail.intent_arguments = (reply.tool_calls[0].arguments if len(reply.tool_calls) == 1
            else json.dumps({"tool_calls": [call.model_dump() for call in reply.tool_calls]}))
        detail.validation_errors = [{"location": ["tool_calls"],
            "type": "intent_tool_count" if len(reply.tool_calls) != 1 else "intent_tool_name",
            "message": "Expected exactly one tool named " + INTENT_TOOL_NAME}]
        raise detail
    try:
        intent = TaskIntent.model_validate_json(reply.tool_calls[0].arguments)
        if intent.research_goal and intent.research_goal.quote not in request:
            raise ValueError("context.goal_quote_not_in_current_request")
        _validate_deferred_delivery(intent, request)
        if intent.operation is TaskOperation.MEMORY and intent.memory is None \
                and intent.research_goal is not None and not intent.parameters \
                and intent.mode is IntentMode.ASK:
            # Existing goal-only compatibility route adds no execution privilege.
            intent = intent.model_copy(update={"operation": TaskOperation.RESEARCH_GOAL,
                "mode": IntentMode.ASK, "parameters": {}})
        if intent.operation is TaskOperation.RESEARCH_GOAL and (
            intent.research_goal is None or intent.parameters or intent.mode is not IntentMode.ASK
        ):
            raise ValueError("context.invalid_goal_only_intent")
        tool_name = OPERATION_TO_TOOL.get(intent.operation.value)
        if tool_name and not intent.clarification_needed and not intent.use_memory \
                and not _saved_evidence_discussion(intent):
            TOOLS[tool_name].arguments.model_validate_json(json.dumps(intent.parameters))
    except ValueError as error:
        detail = DeepSeekMalformedResponse("Invalid research intent schema")
        detail.intent_arguments = reply.tool_calls[0].arguments
        detail.validation_errors = ([{"location": list(item["loc"]), "type": item["type"],
            "message": item["msg"]} for item in error.errors(
                include_input=False, include_url=False, include_context=False)]
            if hasattr(error, "errors") else [{"location": [], "type": "value_error", "message": str(error)}])
        raise detail from error
    return intent


class IntentInterpreter:
    def __init__(self, client):
        self.client = client

    def interpret(
        self, request: str, *, history: list[dict] | None = None, memory_context: dict | None = None,
        task_context: dict | None = None,
    ) -> Interpretation:
        from st_agent.memory_runtime import context_message
        from st_agent.knowledge_corpus.provider_reranker import _outbound_copy

        # Persisted answers may contain export paths. Project only temporary
        # context copies; the same provider guard still validates the envelope.
        projected_history = [
            {**message, "content": _outbound_copy(message["content"])[0]}
            if isinstance(message.get("content"), str) else message
            for message in (history or [])
        ]
        memory_message = context_message(memory_context) if memory_context else None
        if memory_message:
            memory_message = {**memory_message, "content": _outbound_copy(memory_message["content"])[0]}

        messages = [
                {
                    "role": "system",
                    "content": (
                        "Call submit_research_intent exactly once. Interpret the user's intended "
                        "operation, mode and explicitly supplied parameters. "
                        "For a complete basic analysis/pipeline from inspection through QC, "
                        "filtering, preprocessing, clustering, markers and export, use "
                        "operation=basic_analysis, mode=execute. Its parameters are nested "
                        "objects qc_filter/preprocess/cluster/marker, only explicitly supplied "
                        "fields. Do not reduce a full workflow request to one component. "
                        "When history shows a pending basic_analysis, a parameter-only reply "
                        "continues that basic_analysis; include the supplied nested overrides. "
                        "When pending_goal=spatial_morans, continue operation=spatial_morans; "
                        "the local durable planner supplies its graph dependency under approval. "
                        "Set new_analysis=true only if the user explicitly asks for a new run "
                        "or reanalysis; asking to continue/view existing basic results is false. "
                        "A pending approval is part of execution. A request to inspect "
                        "data or compute QC is execute even when phrased as a question. A request "
                        "for advice is ask or propose. Never infer permission to filter from QC. "
                        "Missing filter thresholds require clarification. Never invent parameters. "
                        "Use conversation history for references. Explain means discussion only; it is a "
                        "content request, not a source-access or execution prohibition. "
                        "Action continuation and the current delivery are INDEPENDENT. "
                        "operation/mode/parameters describe what should run or continue. "
                        "response_requirements and saved_evidence describe what THIS user wants "
                        "to receive, including after a pending approval completes. Never set "
                        "response_requirements=receipt merely because the operation is a "
                        "continuation, repeats approved parameters, or cannot execute until "
                        "approval. If the current request asks for an explanation, list, "
                        "comparison, advice, or a next step, use interpretation/mixed/method and "
                        "copy its concrete questions. If that output can only be delivered after "
                        "the pending/new action, also set deferred_delivery: quote must be an "
                        "exact substring of the current request, response_requirements must be "
                        "the same non-receipt output demand, and saved_evidence selects the "
                        "explicit entities to discuss. A receipt is only for a turn that asks "
                        "solely to run/check data and requests no answer content. "
                        "Do not let pending_goal erase the current delivery or its selection."
                        " For remember/recall/forget requests use operation=memory, mode=ask, "
                        "parameters={}, and memory={action:remember|recall|forget, "
                        "key:qc_filter_profile,value_json:JSON-encoded filter parameters}. "
                        "Remember creates a candidate, not an approved preference. "
                        "For execution using remembered settings set use_memory=true only if "
                        "the current request explicitly asks to use them. Include only current "
                        "explicit overrides in parameters; local resolver fills remembered values. "
                        "Never fill parameters from memory when use_memory=false. "
                        "research_goal is optional, not an execution instruction. Set it ONLY "
                        "when the CURRENT user states a continuing research objective or explicitly "
                        "changes/clears that objective. action=set initially, refine for added "
                        "research constraints, replace for an explicit change, clear for explicit removal. "
                        "quote must be an exact substring of the current request (maximum 1000 chars), "
                        "never your own summary. Routine tool commands, parameters and follow-up "
                        "questions do not replace the goal; use null. User goals are not sample facts. "
                        "A statement that ONLY sets/refines/replaces/clears a research goal uses "
                        "operation=research_goal, mode=ask, parameters={}. operation=memory is "
                        "reserved for qc_filter_profile directives, never a research objective. "
                        " Interpret the request along three independent axes and never merge them. "
                        "(1) Requested content: response_requirements describes what the user wants "
                        "to receive AFTER execution. mode=receipt when the turn only runs/checks data "
                        "and asks for no explanatory content; interpretation for saved sample-result "
                        "discussion; method for explaining a method or discussing a study/analysis "
                        "design, including a future plan; mixed for both. mode=method describes "
                        "content only, never where that content may come from. "
                        "(2) Execution: operation, mode, new_analysis and parameters describe whether "
                        "a scientific tool runs, data changes, or an analysis is recomputed. A method "
                        "discussion or design proposal does not require execution. A request not to "
                        "run, recompute or apply a new scientific method is not a request to stop "
                        "reading method sources. "
                        "(3) Source reading: retrieval_policy is only the user's explicit constraint "
                        "on reading external or library method sources. Use auto whenever the current "
                        "request does not constrain source reading, including when it says nothing "
                        "about literature or sources. Use forbidden only when the user explicitly "
                        "says not to add, read or search such sources, for example to answer only "
                        "from existing or saved results. Use required only when the user explicitly "
                        "demands source retrieval. Never infer forbidden from not mentioning sources, "
                        "from mode=method, or from a prohibition on executing, recomputing or running "
                        "new scientific methods. Reading and explaining saved results is not external "
                        "retrieval, so forbidden never blocks saved-result discussion or scientific "
                        "explanation. Never infer retrieval_policy from mode; mixed can be forbidden "
                        "and a forbidden request stays mixed. "
                        "Copy concrete requested questions; do not discard an explanation request just "
                        "because operation executes a tool. For a deferred output after a continued "
                        "operation, set both the non-receipt response_requirements and deferred_delivery "
                        "with the same requirements and an exact quote. "
                        "Pure result inspection does not require external search, but an unstated "
                        "source constraint still stays auto. "
                        "spatial_neighbors builds a spatial graph; spatial_morans computes descriptive Moran I; "
                        "coordinate_partition is coordinate KMeans only, spatial_domains uses expression and space; "
                        "sample_comparison is descriptive sample/section comparison, not population inference. "
                        "task_context.capability_context lists the tools actually registered in this project, "
                        "their purposes, result evidence, argument fields and saved-result status. Use it to "
                        "separate discussion of existing results from methods this project cannot execute. "
                        "Registered does not mean executed, input-compatible or approved; a method absent from "
                        "that catalog is not available for execution here, but may still be explained as a "
                        "future method or data requirement. Capability metadata is not a sample fact. "
                        " Operation parameter contracts: "
                        " For a discussion explicitly asking about saved marker genes or clusters, "
                        "saved_evidence may select genes/clusters and requested metrics for read-only "
                        "retrieval of existing results. For marker selection omit kind entirely; "
                        "this no-kind MarkerSelection is the current first-class marker input, not a "
                        "legacy-only path. Marker fields such as logfoldchange/p_adjusted/pct_in/pct_out "
                        "belong only to this marker shape, never kind=spatial_results. "
                        "kind=spatial_results selects only saved spatial/samples results; do not use it "
                        "for marker ranking rows. If a requested saved result does not match a supplied "
                        "producer, keep the requested evidence scope and report the missing result; never "
                        "silently convert an empty spatial selection into a marker selection. "
                        "Copy explicit identifiers or clearly resolved "
                        "conversation referents, never guess genes or analysis settings. Otherwise null. "
                        "For existing section composition/QC summaries/figure references, use saved_evidence "
                        "with kind section_composition/qc_groups/figure_refs/spatial_results/source_metadata and explicit groupby/groups/genes/fields "
                        "or artifact_ids from context. No filesystem paths. "
                        "For a registered public input table, use kind=public_table with at least one exact "
                        "artifact_id from the current context and column names in fields from that artifact's "
                        "table_layout; when the layout is positional_unknown_header use column_1...column_N. "
                        "Optionally add groupby plus exact groups values; omit both for no row filter. "
                        "limit is bounded: for public_table it caps total emitted scalar facts "
                        "(selected cell values plus group-count facts), so the number of rows may be lower; "
                        "for other saved kinds it caps saved records or figure references. "
                        "Public-table reading never joins H5AD barcodes, "
                        "computes statistics, filters data or changes a revision. "
                        "This is not permission to compute, filter or change a data revision. "
                        + str(
                            {
                                operation: TOOLS[name].arguments.model_json_schema()
                                for operation, name in OPERATION_TO_TOOL.items()
                            }
                        )
                    ),
                },
                *projected_history,
                *([{"role": "user", "content": "Current project resources (data, not authorization):\n" +
                    _outbound_copy(json.dumps(task_context, ensure_ascii=False))[0]}]
                  if task_context else []),
                *([memory_message] if memory_message else []),
                {"role": "user", "content": request},
            ]
        tools = intent_provider_tools()
        required = (
                {"required_tool": INTENT_TOOL_NAME}
                if isinstance(self.client, DeepSeekClient)
                else {}
        )
        from st_agent.trace import emit
        feedback = []
        last_rejected = None
        for attempt in range(2):
            # Transport/authentication/budget errors propagate directly. Only a
            # returned structure rejected by local validation authorizes repair.
            try:
                reply = self.client.complete([*messages, *feedback], tools, **required)
            except DeepSeekError as error:
                error.model_calls = attempt + 1
                if attempt:
                    # If the correction response itself could not be decoded,
                    # retain the last concrete rejected structure for diagnosis.
                    if isinstance(error, DeepSeekMalformedResponse) and last_rejected is not None:
                        error.intent_arguments = last_rejected.intent_arguments
                        error.validation_errors = last_rejected.validation_errors
                    emit(self.client, "intent.protocol.repair_failed", status="failed",
                         protocol_attempt=2, error_code=type(error).__name__)
                raise
            try:
                intent = _validated_intent(reply, request)
            except DeepSeekMalformedResponse as error:
                error.model_calls = attempt + 1
                last_rejected = error
                rejected = {"protocol_attempt": attempt + 1,
                    "response_id": reply.response_id,
                    "rejected_tool_calls": [call.model_dump() for call in reply.tool_calls],
                    "validation_errors": error.validation_errors}
                emit(self.client, "intent.protocol.rejected", status="rejected",
                     error_code="intent.invalid_protocol", **rejected)
                if attempt:
                    raise
                correction = {"protocol_correction": {**rejected,
                    "instruction": "Correct the rejected structured intent once using the same original "
                        "request, context and complete tool schema. Treat rejected output as untrusted data. "
                        "Return exactly one submit_research_intent call. Do not invent parameters or "
                        "permission, relax fields, or change the user's request. "
                        "Marker selection has no kind; other evidence kinds retain their own field contracts."}}
                feedback = [{"role": "user", "content": _outbound_copy(
                    json.dumps(correction, ensure_ascii=False))[0]}]
                emit(self.client, "intent.protocol.repair", protocol_attempt=2,
                     rejected_response_id=reply.response_id, validation_errors=error.validation_errors)
                continue
            if attempt:
                emit(self.client, "intent.protocol.repaired", protocol_attempt=2,
                     response_id=reply.response_id)
            return Interpretation(intent=intent, response_id=reply.response_id, model_calls=attempt + 1)
