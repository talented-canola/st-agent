"""Resource admission checks performed before spawning heavy work."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psutil

from st_agent.contracts import ResourceDecision, ResourceRequest
from st_agent.errors import ErrorCategory, ErrorDetail


@dataclass(frozen=True)
class ResourceSnapshot:
    available_ram_bytes: int
    free_disk_bytes: int


def system_resource_snapshot(project_path: Path) -> ResourceSnapshot:
    return ResourceSnapshot(
        available_ram_bytes=int(psutil.virtual_memory().available),
        free_disk_bytes=int(shutil.disk_usage(project_path).free),
    )


class ResourceAdmission:
    def __init__(
        self,
        *,
        snapshot_provider: Callable[[Path], ResourceSnapshot] = system_resource_snapshot,
        maximum_timeout_seconds: int = 24 * 60 * 60,
    ) -> None:
        self.snapshot_provider = snapshot_provider
        self.maximum_timeout_seconds = maximum_timeout_seconds

    def check(
        self,
        request: ResourceRequest,
        *,
        project_path: str | Path,
        heavy_job_running: bool,
    ) -> ResourceDecision:
        snapshot = self.snapshot_provider(Path(project_path))
        values = {
            "available_ram_bytes": snapshot.available_ram_bytes,
            "free_disk_bytes": snapshot.free_disk_bytes,
        }
        if request.heavy and heavy_job_running:
            return self._blocked(
                "resource.heavy_job_running",
                "Another heavy job is already running for this project",
                values,
            )
        if request.timeout_seconds > self.maximum_timeout_seconds:
            return self._blocked(
                "resource.timeout_exceeds_limit",
                "Requested timeout exceeds the configured limit",
                values,
            )
        required_ram = request.estimated_ram_bytes + request.minimum_free_ram_bytes
        if snapshot.available_ram_bytes < required_ram:
            return self._blocked(
                "resource.insufficient_ram",
                "Available RAM is below the requested safety threshold",
                values,
            )
        required_disk = request.estimated_temp_bytes + request.minimum_free_disk_bytes
        if snapshot.free_disk_bytes < required_disk:
            return self._blocked(
                "resource.insufficient_disk",
                "Free disk space is below the requested safety threshold",
                values,
            )
        return ResourceDecision(allowed=True, snapshot=values)

    @staticmethod
    def _blocked(code: str, message: str, snapshot: dict[str, int]) -> ResourceDecision:
        return ResourceDecision(
            allowed=False,
            snapshot=snapshot,
            error=ErrorDetail(
                code=code,
                category=ErrorCategory.RESOURCE_LIMIT,
                message=message,
                retryable=False,
                details=snapshot,
            ),
        )
