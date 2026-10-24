"""Fault-safe dataset revision lifecycle and recovery."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from st_agent.errors import PreconditionFailedError
from st_agent.ids import IdKind, new_id, validate_id
from st_agent.models import DatasetRecord, DatasetRevisionRecord, RevisionStatus
from st_agent.project import ProjectLayout
from st_agent.repositories import DatasetRepository, RevisionRepository
from st_agent.utils import sha256_file, utc_now_iso


class RevisionManager:
    def __init__(
        self,
        layout: ProjectLayout,
        project_id: str,
        datasets: DatasetRepository,
        revisions: RevisionRepository,
    ) -> None:
        self.layout = layout
        self.project_id = validate_id(project_id, IdKind.PROJECT)
        self.datasets = datasets
        self.revisions = revisions

    def ensure_initial_revision(self, dataset: DatasetRecord) -> DatasetRevisionRecord:
        refreshed = self.datasets.get(dataset.id)
        if refreshed.active_revision_id is not None:
            return self.revisions.get(refreshed.active_revision_id)
        storage = Path(refreshed.storage_path).resolve()
        if not storage.is_file() or sha256_file(storage) != refreshed.sha256:
            self.datasets.update_status(refreshed.id, "invalid")
            raise PreconditionFailedError(
                f"Registered raw data failed fingerprint verification: {refreshed.id}",
                code="dataset.fingerprint_mismatch",
            )
        relative_path = storage.relative_to(self.layout.root).as_posix()
        now = utc_now_iso()
        record = DatasetRevisionRecord(
            id=new_id(IdKind.REVISION),
            dataset_id=refreshed.id,
            status=RevisionStatus.COMMITTED,
            relative_path=relative_path,
            sha256=refreshed.sha256,
            metadata={"source": "registered_raw"},
            created_at=now,
            kind="raw",
            updated_at=now,
        )
        return self.revisions.create_initial(record)

    def verify_and_reconcile_all(self) -> None:
        for dataset in self.datasets.list(self.project_id):
            storage = Path(dataset.storage_path)
            if not storage.is_file() or sha256_file(storage) != dataset.sha256:
                self.datasets.update_status(dataset.id, "invalid")
                continue
            if dataset.status != "registered":
                self.datasets.update_status(dataset.id, "registered")
            self.ensure_initial_revision(dataset)

    def begin_draft(
        self,
        dataset_id: str,
        *,
        call_id: str,
        metadata: dict[str, Any] | None = None,
        expected_parent_revision_id: str | None = None,
    ) -> tuple[DatasetRevisionRecord, bool]:
        validate_id(dataset_id, IdKind.DATASET)
        validate_id(call_id, IdKind.TOOL_CALL)
        if expected_parent_revision_id is not None:
            validate_id(expected_parent_revision_id, IdKind.REVISION)
        existing = self.revisions.find_by_call(dataset_id, call_id)
        if existing is not None:
            if (
                expected_parent_revision_id is not None
                and existing.parent_revision_id != expected_parent_revision_id
            ):
                raise PreconditionFailedError(
                    "Existing draft is bound to a different input revision",
                    code="revision.call_parent_mismatch",
                )
            return existing, False
        dataset = self.datasets.get(dataset_id)
        if dataset.status != "registered" or dataset.active_revision_id is None:
            raise PreconditionFailedError(
                f"Dataset is not ready for a derived revision: {dataset_id}",
                code="dataset.not_ready",
            )
        if (
            expected_parent_revision_id is not None
            and dataset.active_revision_id != expected_parent_revision_id
        ):
            raise PreconditionFailedError(
                "Dataset active revision changed after planning",
                code="revision.input_changed",
            )
        now = utc_now_iso()
        record = DatasetRevisionRecord(
            id=new_id(IdKind.REVISION),
            dataset_id=dataset_id,
            parent_revision_id=expected_parent_revision_id or dataset.active_revision_id,
            status=RevisionStatus.DRAFT,
            metadata=metadata or {},
            created_at=now,
            call_id=call_id,
            kind="derived",
            updated_at=now,
        )
        created = self.revisions.create(record)
        self._draft_directory(record.id).mkdir(parents=True, exist_ok=False)
        return created, True

    def draft_path(self, revision_id: str, filename: str) -> Path:
        validate_id(revision_id, IdKind.REVISION)
        if Path(filename).name != filename or filename in {"", ".", ".."}:
            raise ValueError("Draft filename must be a plain filename")
        directory = self._draft_directory(revision_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / filename

    def mark_validating(
        self, revision_id: str, *, validation: dict[str, Any]
    ) -> DatasetRevisionRecord:
        try:
            return self.revisions.transition(
                revision_id,
                expected=(RevisionStatus.DRAFT.value,),
                status=RevisionStatus.VALIDATING.value,
                validation=validation,
                updated_at=utc_now_iso(),
            )
        except ValueError as exc:
            raise PreconditionFailedError(str(exc), code="revision.illegal_transition") from exc

    def commit(
        self,
        revision_id: str,
        candidate: str | Path,
        *,
        provenance: dict[str, Any] | None = None,
        failpoint: Callable[[str], None] | None = None,
    ) -> DatasetRevisionRecord:
        revision = self.revisions.get(revision_id)
        if revision.status is not RevisionStatus.VALIDATING:
            raise PreconditionFailedError(
                f"Revision is not validating: {revision_id}",
                code="revision.not_validated",
            )
        candidate_path = Path(candidate).resolve()
        draft_directory = self._draft_directory(revision_id).resolve()
        if not candidate_path.is_file() or draft_directory not in candidate_path.parents:
            raise PreconditionFailedError(
                f"Candidate is missing or outside its draft: {revision_id}",
                code="revision.invalid_candidate",
            )
        digest = sha256_file(candidate_path)
        suffix = "".join(candidate_path.suffixes) or ".bin"
        final_path = self.layout.root / "datasets" / "revisions" / f"{revision_id}{suffix}"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(candidate_path, final_path)
        if failpoint is not None:
            failpoint("after_publish_before_database_commit")
        relative_path = final_path.relative_to(self.layout.root).as_posix()
        try:
            committed = self.revisions.commit(
                revision_id,
                relative_path=relative_path,
                sha256=digest,
                provenance=provenance or {},
                updated_at=utc_now_iso(),
            )
        except RuntimeError as exc:
            quarantine = self._quarantine_directory() / final_path.name
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            os.replace(final_path, quarantine)
            raise PreconditionFailedError(
                "Dataset active revision changed before commit",
                code="revision.active_changed",
            ) from exc
        self._remove_draft_directory(revision_id)
        return committed

    def rollback(
        self,
        revision_id: str,
        *,
        status: RevisionStatus = RevisionStatus.ROLLED_BACK,
        error: dict[str, Any] | None = None,
    ) -> DatasetRevisionRecord:
        if status not in {
            RevisionStatus.FAILED,
            RevisionStatus.CANCELLED,
            RevisionStatus.ROLLED_BACK,
        }:
            raise ValueError(f"Invalid rollback status: {status}")
        try:
            record = self.revisions.transition(
                revision_id,
                expected=(RevisionStatus.DRAFT.value, RevisionStatus.VALIDATING.value),
                status=status.value,
                error=error,
                updated_at=utc_now_iso(),
            )
        except ValueError as exc:
            raise PreconditionFailedError(str(exc), code="revision.illegal_transition") from exc
        self._remove_draft_directory(revision_id)
        return record

    def recover_incomplete(self) -> list[str]:
        recovered: list[str] = []
        for revision in self.revisions.list_incomplete():
            self.rollback(
                revision.id,
                status=RevisionStatus.ROLLED_BACK,
                error={"code": "revision.recovered_after_restart"},
            )
            recovered.append(revision.id)
        self._recover_orphan_publications()
        self._recover_unrecorded_drafts()
        return recovered

    def _draft_directory(self, revision_id: str) -> Path:
        return self.layout.root / "tmp" / "revisions" / revision_id

    def _quarantine_directory(self) -> Path:
        return self.layout.root / "tmp" / "quarantine"

    def _remove_draft_directory(self, revision_id: str) -> None:
        directory = self._draft_directory(revision_id)
        if directory.is_dir():
            shutil.rmtree(directory)

    def _recover_orphan_publications(self) -> None:
        revision_root = self.layout.root / "datasets" / "revisions"
        known = self.revisions.committed_paths()
        destination_root = self._quarantine_directory() / "orphan-revisions"
        for path in revision_root.iterdir():
            if not path.is_file():
                continue
            relative = path.relative_to(self.layout.root).as_posix()
            if relative in known:
                continue
            destination_root.mkdir(parents=True, exist_ok=True)
            os.replace(path, destination_root / path.name)

    def _recover_unrecorded_drafts(self) -> None:
        draft_root = self.layout.root / "tmp" / "revisions"
        if not draft_root.is_dir():
            return
        destination_root = self._quarantine_directory() / "unrecorded-drafts"
        for path in draft_root.iterdir():
            destination_root.mkdir(parents=True, exist_ok=True)
            os.replace(path, destination_root / path.name)
