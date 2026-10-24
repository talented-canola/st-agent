"""Local environment and project health diagnostics."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from st_agent import __version__
from st_agent.api import ResearchCopilot
from st_agent.config import Settings
from st_agent.db import connect_database
from st_agent.migrations import LATEST_SCHEMA_VERSION, MigrationManager


class DoctorCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    status: Literal["pass", "warning", "fail"]
    detail: str


class DoctorReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    healthy: bool
    package_version: str
    checks: list[DoctorCheck]


def run_doctor(
    project_path: str | Path | None = None,
    *,
    settings: Settings | None = None,
) -> DoctorReport:
    active_settings = settings or Settings()
    checks: list[DoctorCheck] = []

    python_ok = sys.version_info[:2] == (3, 11)
    checks.append(
        DoctorCheck(
            name="python",
            status="pass" if python_ok else "fail",
            detail=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )
    sqlite_ok = sqlite3.sqlite_version_info >= (3, 35, 0)
    checks.append(
        DoctorCheck(
            name="sqlite",
            status="pass" if sqlite_ok else "fail",
            detail=sqlite3.sqlite_version,
        )
    )
    checks.append(
        DoctorCheck(
            name="deepseek_credentials",
            status="pass" if active_settings.deepseek_api_key else "warning",
            detail="configured"
            if active_settings.deepseek_api_key
            else "not configured; Agent execution unavailable",
        )
    )

    if project_path is not None:
        try:
            copilot = ResearchCopilot.open(project_path, settings=active_settings)
            version = MigrationManager(
                copilot.layout.database_path,
                busy_timeout_ms=active_settings.sqlite_busy_timeout_ms,
            ).current_version()
            with connect_database(
                copilot.layout.database_path,
                busy_timeout_ms=active_settings.sqlite_busy_timeout_ms,
            ) as connection:
                journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
                foreign_keys = int(connection.execute("PRAGMA foreign_keys").fetchone()[0])
            project_ok = (
                version == LATEST_SCHEMA_VERSION and journal_mode == "wal" and foreign_keys == 1
            )
            checks.append(
                DoctorCheck(
                    name="project",
                    status="pass" if project_ok else "fail",
                    detail=(
                        f"id={copilot.project.id}; schema={version}; "
                        f"journal={journal_mode}; foreign_keys={foreign_keys}"
                    ),
                )
            )
            copilot.close()
        except Exception as exc:
            checks.append(DoctorCheck(name="project", status="fail", detail=str(exc)))

    return DoctorReport(
        healthy=not any(check.status == "fail" for check in checks),
        package_version=__version__,
        checks=checks,
    )
