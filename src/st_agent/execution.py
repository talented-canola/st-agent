"""Fault-safe orchestration of one strongly typed local tool execution."""

from __future__ import annotations

import os
import json
import shutil
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from st_agent.cache import CacheIdentity, build_cache_key
from st_agent.contracts import (
    Effect,
    Provenance,
    ResourceRequest,
    ToolResult,
    ToolSpec,
    ToolStatus,
    ValidationStatus,
)
from st_agent.errors import (
    ErrorCategory,
    ErrorDetail,
    InvalidInputError,
    PreconditionFailedError,
    ToolExecutionError,
    error_detail_from_exception,
)
from st_agent.ids import IdKind, validate_id
from st_agent.models import JobStatus, RevisionStatus
from st_agent.provenance import build_provenance
from st_agent.serialization import validate_json_value
from st_agent.validation import Validator
from st_agent.workers import WorkerManager
from st_agent.trace import emit


@dataclass(frozen=True)
class ToolDefinition:
    spec: ToolSpec
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    validator: Validator
    resources: ResourceRequest
    package_names: tuple[str, ...] = ()
    worker_request_builder: Callable[[dict, Path], dict] | None = None


class ExecutionCoordinator:
    def __init__(
        self,
        copilot,
        workers: WorkerManager,
        *,
        lock_path: str | Path,
        dependency_lock_fallback: str,
        failpoint: Callable[[str], None] | None = None,
    ) -> None:
        self.copilot = copilot
        self.workers = workers
        self.lock_path = Path(lock_path)
        self.dependency_lock_fallback = dependency_lock_fallback
        self.failpoint = failpoint

    def execute(
        self,
        *,
        dataset_id: str,
        call_id: str,
        definition: ToolDefinition,
        arguments: dict,
        random_seed: int | None,
        input_revision_id: str | None = None,
    ) -> ToolResult:
        validate_id(dataset_id, IdKind.DATASET)
        validate_id(call_id, IdKind.TOOL_CALL)
        replay = self.copilot.tool_executions.find_by_call(call_id)
        if replay is not None:
            return replay
        started_at = datetime.now(UTC)
        dataset = self.copilot.get_dataset(dataset_id)
        if dataset.status != "registered":
            return self._persist_failure(
                call_id=call_id,
                dataset_id=dataset_id,
                definition=definition,
                arguments=arguments,
                random_seed=random_seed,
                started_at=started_at,
                input_revision_id=input_revision_id,
                error=error_detail_from_exception(
                    PreconditionFailedError(
                        "Dataset fingerprint is invalid", code="dataset.invalid"
                    )
                ),
                status=ToolStatus.BLOCKED,
            )
        input_id = input_revision_id or dataset.active_revision_id
        if input_id is None:
            raise PreconditionFailedError("Dataset has no active revision")
        input_revision = self.copilot.revisions.get(input_id)
        if input_revision.dataset_id != dataset_id:
            raise PreconditionFailedError("Input revision belongs to another dataset")
        try:
            json_arguments = validate_json_value(arguments)
        except ValueError:
            json_arguments = {}
            input_error = InvalidInputError(
                "Tool arguments must contain only small JSON values",
                code="tool.arguments_not_json",
            )
            return self._persist_failure(
                call_id=call_id,
                dataset_id=dataset_id,
                definition=definition,
                arguments=json_arguments,
                random_seed=random_seed,
                started_at=started_at,
                input_revision_id=input_id,
                error=error_detail_from_exception(input_error),
                status=ToolStatus.BLOCKED,
            )
        seed_error = self._seed_error(definition.spec.seed_policy, random_seed)
        if seed_error is not None:
            return self._persist_failure(
                call_id=call_id,
                dataset_id=dataset_id,
                definition=definition,
                arguments=json_arguments,
                random_seed=random_seed,
                started_at=started_at,
                input_revision_id=input_id,
                error=error_detail_from_exception(seed_error),
                status=ToolStatus.BLOCKED,
            )
        try:
            validated_input = definition.input_model.model_validate_json(json.dumps(json_arguments))
        except ValidationError as error:
            details = {
                "errors": [
                    {"location": list(item["loc"]), "type": item["type"]}
                    for item in error.errors(include_input=False, include_url=False)
                ]
            }
            input_error = InvalidInputError(
                "Tool arguments did not match the declared input schema",
                code="tool.input_schema_invalid",
                details=details,
            )
            return self._persist_failure(
                call_id=call_id,
                dataset_id=dataset_id,
                definition=definition,
                arguments=json_arguments,
                random_seed=random_seed,
                started_at=started_at,
                input_revision_id=input_id,
                error=error_detail_from_exception(input_error),
                status=ToolStatus.BLOCKED,
            )
        normalized_args = validate_json_value(validated_input.model_dump(mode="json"))
        source_path = self._revision_path(input_revision.relative_path)
        worker_request = (
            definition.worker_request_builder(normalized_args, source_path)
            if definition.worker_request_builder is not None
            else normalized_args
        )
        provenance_seed = self._provenance(
            definition,
            dataset.sha256,
            normalized_args,
            random_seed,
            started_at,
            started_at,
        )
        cache_identity = CacheIdentity(
            dataset_fingerprint=dataset.sha256,
            input_revision_id=input_id,
            tool_name=definition.spec.name,
            tool_contract_version=definition.spec.version,
            canonical_args=normalized_args,
            random_seed=random_seed,
            dependency_lock_hash=provenance_seed.dependency_lock_hash,
            upstream_artifact_ids=(),
        )
        cache_key = build_cache_key(cache_identity)
        cached = self.copilot.tool_executions.find_success(self.copilot.project.id, cache_key)
        if cached is not None:
            from st_agent.worker_artifacts import validate_published_refs
            try:
                cache_artifacts = validate_published_refs(self.copilot,
                    [ref.model_dump(mode="json") for ref in cached.artifacts])
            except (ValueError, KeyError, OSError) as error:
                return self._persist_failure(call_id=call_id, dataset_id=dataset_id,
                    definition=definition, arguments=normalized_args, random_seed=random_seed,
                    started_at=started_at, input_revision_id=input_id,
                    error=error_detail_from_exception(PreconditionFailedError(
                        "Cached artifacts are missing or corrupt; saved analysis remains unchanged",
                        code="cache.artifact_invalid")), status=ToolStatus.BLOCKED)
            finished = datetime.now(UTC)
            provenance = self._provenance(
                definition,
                dataset.sha256,
                normalized_args,
                random_seed,
                started_at,
                finished,
            )
            result = ToolResult(
                call_id=call_id,
                tool_name=cached.tool_name,
                status=ToolStatus.SUCCESS,
                summary=cached.summary,
                metrics={**cached.metrics, "cache_hit": True},
                artifacts=cache_artifacts,
                warnings=cached.warnings,
                input_revision_id=cached.input_revision_id,
                output_revision_id=cached.output_revision_id,
                citations=cached.citations,
                provenance=provenance,
            )
            return self._save(result, dataset_id, definition.spec.version, cache_key)

        handle = self.workers.start(
            call_id=call_id,
            entrypoint=definition.spec.executor,
            request=worker_request,
            resources=definition.resources,
        )
        emit(self.copilot, "worker.started" if handle.pid else "worker.not_started",
             call_id=call_id, job_id=handle.job_id, pid=handle.pid)
        outcome = handle.wait(timeout_seconds=definition.spec.timeout_seconds)
        emit(self.copilot, "worker.stopped", status=outcome.status.value, call_id=call_id,
             job_id=handle.job_id, error_code=outcome.error.code if outcome.error else None)
        if outcome.status is not JobStatus.SUCCEEDED:
            status = {
                JobStatus.BLOCKED: ToolStatus.BLOCKED,
                JobStatus.CANCELLED: ToolStatus.CANCELLED,
            }.get(outcome.status, ToolStatus.FAILED)
            error = outcome.error or ErrorDetail(
                code="worker.failed",
                category=ErrorCategory.TOOL_EXECUTION_FAILED,
                message="Worker failed without an error detail",
                retryable=False,
            )
            result = self._persist_failure(
                call_id=call_id,
                dataset_id=dataset_id,
                definition=definition,
                arguments=normalized_args,
                random_seed=random_seed,
                started_at=started_at,
                input_revision_id=input_id,
                error=error,
                status=status,
            )
            if handle.workspace.is_dir():
                shutil.rmtree(handle.workspace)
            return result

        draft = None
        try:
            try:
                validated_output = definition.output_model.model_validate(outcome.result or {})
            except ValidationError as error:
                raise ToolExecutionError(
                    "Worker result did not match the declared output schema",
                    code="tool.output_schema_invalid",
                ) from error
            product = validate_json_value(validated_output.model_dump(mode="json"))
            from st_agent.worker_artifacts import validate_worker_artifacts, publish_worker_artifacts
            # A Worker may declare local files, never authorize project artifact IDs.
            if product.get("artifacts"):
                raise ToolExecutionError("Worker cannot supply published artifact identities",
                    code="tool.worker_artifact_authority")
            validated_artifacts = validate_worker_artifacts(handle.workspace,
                product.get("artifact_outputs", []))
            published_artifacts = []
            revision_output = product.get("revision_output")
            candidate = (
                (handle.workspace / revision_output).resolve()
                if isinstance(revision_output, str)
                else None
            )
            workspace = handle.workspace.resolve()
            self._hit_failpoint("after_worker_before_validation")
            if definition.spec.effects is Effect.DERIVED_REVISION:
                if (
                    candidate is None
                    or not candidate.is_file()
                    or workspace not in candidate.parents
                ):
                    raise PreconditionFailedError(
                        "Derived tool did not produce a safe revision file",
                        code="tool.missing_revision_output",
                    )
                report = definition.validator.validate(
                    candidate,
                    {
                        "dataset_id": dataset_id,
                        "input_revision_id": input_id,
                        "arguments": normalized_args,
                    },
                )
                emit(self.copilot, "output.validated", status=report.status.value,
                     call_id=call_id, checks=report.checks)
                if report.status is ValidationStatus.FAILED:
                    return self._persist_failure(
                        call_id=call_id,
                        dataset_id=dataset_id,
                        definition=definition,
                        arguments=normalized_args,
                        random_seed=random_seed,
                        started_at=started_at,
                        input_revision_id=input_id,
                        error=report.error,
                        status=ToolStatus.FAILED,
                    )
                published_artifacts = publish_worker_artifacts(self.copilot, handle.workspace,
                    validated_artifacts, call_id=call_id, dataset_id=dataset_id, revision_id=input_id,
                    derived=True)
                if "artifacts" in product:
                    product["artifacts"] = [item.model_dump(mode="json") for item in published_artifacts]
                self._hit_failpoint("after_validation_before_draft")
                draft, created = self.copilot.revision_manager.begin_draft(
                    dataset_id,
                    call_id=call_id,
                    metadata={"tool_name": definition.spec.name, "worker_product": product,
                              "cache_key": cache_key},
                    expected_parent_revision_id=input_id,
                )
                if not created and draft.status is RevisionStatus.COMMITTED:
                    raise PreconditionFailedError(
                        "Committed revision exists without a persisted result",
                        code="execution.incomplete_persistence",
                    )
                draft_candidate = self.copilot.revision_manager.draft_path(
                    draft.id, candidate.name
                )
                os.replace(candidate, draft_candidate)
                self.copilot.revision_manager.mark_validating(
                    draft.id, validation=report.model_dump(mode="json", exclude_none=False)
                )
                commit_provenance = self._provenance(
                    definition,
                    dataset.sha256,
                    normalized_args,
                    random_seed,
                    started_at,
                    datetime.now(UTC),
                )
                committed = self.copilot.revision_manager.commit(
                    draft.id,
                    draft_candidate,
                    provenance=commit_provenance.model_dump(mode="json"),
                    failpoint=self.failpoint,
                )
                output_revision_id = committed.id
                from st_agent.worker_artifacts import bind_output_revision
                published_artifacts = bind_output_revision(self.copilot, published_artifacts, committed.id)
                emit(self.copilot, "revision.committed", revision_id=committed.id,
                     input_revision_id=input_id, call_id=call_id, sha256=committed.sha256)
                report_warnings = report.warnings
            else:
                output_revision_id = None
                report_warnings = ()
                published_artifacts = publish_worker_artifacts(self.copilot, handle.workspace,
                    validated_artifacts, call_id=call_id, dataset_id=dataset_id, revision_id=input_id)
            finished_at = datetime.now(UTC)
            warnings = tuple(product.get("warnings", ())) + tuple(report_warnings)
            status = ToolStatus.WARNING if warnings else ToolStatus.SUCCESS
            result = ToolResult(
                call_id=call_id,
                tool_name=definition.spec.name,
                status=status,
                summary=str(product.get("summary", "Tool completed")),
                metrics=product.get("metrics", {}),
                artifacts=tuple(published_artifacts),
                warnings=warnings,
                input_revision_id=input_id,
                output_revision_id=output_revision_id,
                provenance=(
                    commit_provenance
                    if definition.spec.effects is Effect.DERIVED_REVISION
                    else self._provenance(
                        definition,
                        dataset.sha256,
                        normalized_args,
                        random_seed,
                        started_at,
                        finished_at,
                    )
                ),
            )
            return self._save(
                result,
                dataset_id,
                definition.spec.version,
                cache_key if status is ToolStatus.SUCCESS else None,
            )
        except BaseException as error:
            if draft is not None and draft.status in {
                RevisionStatus.DRAFT,
                RevisionStatus.VALIDATING,
            }:
                with suppress(Exception):
                    self.copilot.revision_manager.rollback(
                        draft.id,
                        status=RevisionStatus.FAILED,
                        error=error_detail_from_exception(error).model_dump(mode="json"),
                    )
            return self._persist_failure(
                call_id=call_id,
                dataset_id=dataset_id,
                definition=definition,
                arguments=normalized_args,
                random_seed=random_seed,
                started_at=started_at,
                input_revision_id=input_id,
                error=error_detail_from_exception(error),
                status=ToolStatus.FAILED,
            )
        finally:
            if handle.workspace.is_dir():
                shutil.rmtree(handle.workspace)

    def recover_committed(self, *, dataset_id, call_id, definition, arguments, input_revision_id):
        """Repair result persistence from a verified commit; never start a worker."""
        import json
        from st_agent.events import EventDraft
        from st_agent.utils import sha256_file
        revision = self.copilot.revisions.find_by_call(dataset_id, call_id)
        if revision is None or revision.status is not RevisionStatus.COMMITTED:
            return None
        if revision.parent_revision_id != input_revision_id:
            raise PreconditionFailedError("Committed parent mismatch", code="recovery.parent_mismatch")
        provenance = Provenance.model_validate_json(json.dumps(revision.provenance))
        normalized = definition.input_model.model_validate_json(json.dumps(arguments)).model_dump(mode="json")
        if (revision.metadata.get("tool_name") != definition.spec.name or
                provenance.tool_version != definition.spec.version or provenance.parameters != normalized):
            raise PreconditionFailedError("Committed binding mismatch", code="recovery.binding_mismatch")
        product = revision.metadata.get("worker_product")
        if not product:
            raise PreconditionFailedError("Commit lacks recoverable worker result",
                                          code="recovery.missing_worker_product")
        product = definition.output_model.model_validate(product).model_dump(mode="json")
        candidate = self._revision_path(revision.relative_path)
        if sha256_file(candidate) != revision.sha256:
            raise PreconditionFailedError("Committed file mismatch", code="recovery.hash_mismatch")
        report = definition.validator.validate(candidate, {"dataset_id": dataset_id,
            "input_revision_id": input_revision_id, "arguments": normalized})
        if report.status is ValidationStatus.FAILED:
            raise PreconditionFailedError("Committed output invalid", code="recovery.output_invalid")
        warnings = tuple(product.get("warnings", ())) + tuple(report.warnings)
        from st_agent.worker_artifacts import validate_published_refs
        published_artifacts = validate_published_refs(self.copilot, product.get("artifacts", []))
        from st_agent.worker_artifacts import bind_output_revision
        published_artifacts = tuple(bind_output_revision(self.copilot, published_artifacts, revision.id))
        restored = ToolResult(call_id=call_id, tool_name=definition.spec.name,
            status=ToolStatus.WARNING if warnings else ToolStatus.SUCCESS,
            summary=product["summary"], metrics=product["metrics"], warnings=warnings,
            artifacts=published_artifacts,
            input_revision_id=input_revision_id, output_revision_id=revision.id, provenance=provenance)
        previous = self.copilot.tool_executions.find_by_call(call_id)
        # Preserve the previous failure in an append-only database event before repairing its row.
        self.copilot.events.append(EventDraft(event_type="execution.commit_reconciled", payload={
            "call_id": call_id, "revision_id": revision.id,
            "previous_result": previous.model_dump(mode="json") if previous else None}))
        if previous is None:
            self._save(restored, dataset_id, definition.spec.version, revision.metadata.get("cache_key"))
        else:
            with self.copilot.tool_executions._connect() as connection:
                connection.execute("UPDATE tool_executions SET status=?,output_revision_id=?,"
                    "result_json=?,cache_key=?,finished_at=? WHERE call_id=? AND project_id=?",
                    (restored.status.value, revision.id, restored.model_dump_json(),
                     revision.metadata.get("cache_key"), provenance.finished_at.isoformat(),
                     call_id, self.copilot.project.id))
                connection.commit()
        emit(self.copilot, "recovery.commit_reconciled", call_id=call_id, revision_id=revision.id)
        return restored

    def _persist_failure(
        self,
        *,
        call_id: str,
        dataset_id: str,
        definition: ToolDefinition,
        arguments: dict,
        random_seed: int | None,
        started_at: datetime,
        input_revision_id: str | None,
        error: ErrorDetail | None,
        status: ToolStatus,
    ) -> ToolResult:
        detail = error or ErrorDetail(
            code="internal.missing_error",
            category=ErrorCategory.INTERNAL_BUG,
            message="Failure did not include an error detail",
            retryable=False,
        )
        finished_at = datetime.now(UTC)
        result = ToolResult(
            call_id=call_id,
            tool_name=definition.spec.name,
            status=status,
            summary=detail.message,
            error=detail,
            input_revision_id=input_revision_id,
            provenance=self._provenance(
                definition,
                self.copilot.get_dataset(dataset_id).sha256,
                arguments,
                random_seed,
                started_at,
                finished_at,
            ),
        )
        return self._save(result, dataset_id, definition.spec.version, None)

    def _provenance(
        self,
        definition: ToolDefinition,
        dataset_hash: str,
        arguments: dict,
        random_seed: int | None,
        started_at: datetime,
        finished_at: datetime,
    ) -> Provenance:
        if self.lock_path.is_file():
            return build_provenance(
                tool_version=definition.spec.version,
                dataset_hash=dataset_hash,
                parameters=arguments,
                random_seed=random_seed,
                package_names=definition.package_names,
                lock_path=self.lock_path,
                started_at=started_at,
                finished_at=finished_at,
            )
        return Provenance(
            tool_version=definition.spec.version,
            dataset_hash=dataset_hash,
            parameters=arguments,
            random_seed=random_seed,
            package_versions={},
            dependency_lock_hash=self.dependency_lock_fallback,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=max(0, round((finished_at - started_at).total_seconds() * 1000)),
        )

    def _save(
        self,
        result: ToolResult,
        dataset_id: str,
        contract_version: str,
        cache_key: str | None,
    ) -> ToolResult:
        return self.copilot.tool_executions.save(
            project_id=self.copilot.project.id,
            dataset_id=dataset_id,
            result=result,
            contract_version=contract_version,
            cache_key=cache_key,
        )

    def _hit_failpoint(self, name: str) -> None:
        if self.failpoint is not None:
            self.failpoint(name)

    def _revision_path(self, relative_path: str | None) -> Path:
        if not relative_path:
            raise PreconditionFailedError(
                "Input revision has no materialized data file",
                code="revision.no_data_file",
            )
        root = self.copilot.layout.root.resolve()
        candidate = (root / relative_path).resolve()
        if root not in candidate.parents or not candidate.is_file():
            raise PreconditionFailedError(
                "Input revision data file is unavailable or outside the project",
                code="revision.data_path_invalid",
            )
        return candidate

    @staticmethod
    def _seed_error(seed_policy: str, random_seed: int | None):
        if seed_policy == "required" and random_seed is None:
            return InvalidInputError(
                "This deterministic tool requires a random seed",
                code="tool.seed_required",
            )
        if seed_policy == "forbidden" and random_seed is not None:
            return InvalidInputError(
                "This tool does not accept a random seed",
                code="tool.seed_forbidden",
            )
        return None
