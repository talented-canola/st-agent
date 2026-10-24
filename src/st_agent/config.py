"""Central, typed configuration with secret-safe diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process configuration loaded from environment variables or ``.env``.

    Project metadata is deliberately not stored here: settings describe the
    running process, while each research project owns its own YAML and SQLite
    state. Secrets remain ``SecretStr`` values and are never returned by
    :meth:`safe_dump`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    deepseek_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ST_AGENT_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        validation_alias=AliasChoices("ST_AGENT_DEEPSEEK_BASE_URL", "DEEPSEEK_BASE_URL"),
    )
    deepseek_model: str = Field(
        default="deepseek-flash",
        validation_alias=AliasChoices("ST_AGENT_DEEPSEEK_MODEL", "DEEPSEEK_MODEL"),
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        validation_alias=AliasChoices("ST_AGENT_LOG_LEVEL", "LOG_LEVEL"),
    )
    sqlite_busy_timeout_ms: int = Field(
        default=30_000,
        ge=1_000,
        le=300_000,
        validation_alias="ST_AGENT_SQLITE_BUSY_TIMEOUT_MS",
    )
    project_schema_version: int = Field(default=6, ge=1)
    knowledge_root: Path = Field(
        default=Path(__file__).resolve().parents[2] / "knowledge",
        validation_alias="ST_AGENT_KNOWLEDGE_ROOT",
    )
    knowledge_release: str = Field(
        default="basic-spatial-evidence-20261003-v2.json",
        pattern=r"^[A-Za-z0-9_.-]+\.json$",
        validation_alias="ST_AGENT_KNOWLEDGE_RELEASE",
    )
    knowledge_generate_answers: bool = Field(
        default=True, validation_alias="ST_AGENT_KNOWLEDGE_GENERATE_ANSWERS",
    )
    knowledge_retrieval_profile: Literal["baseline_v1", "stable_full_v1", "stable_task_v1", "stable_task_v2"] = Field(
        default="stable_full_v1", validation_alias="ST_AGENT_KNOWLEDGE_RETRIEVAL_PROFILE",
    )
    knowledge_rerank_view: Literal["full_window_v1", "matched_neighbors_v1", "structured_full_v1", "shared_sources_full_v1"] = Field(
        default="shared_sources_full_v1", validation_alias="ST_AGENT_KNOWLEDGE_RERANK_VIEW",
    )
    configuration_source: str = "default configuration"

    @classmethod
    def load(cls, env_file: str | Path | None = None) -> Settings:
        path = (
            Path(env_file).resolve()
            if env_file else Path(__file__).resolve().parents[2] / ".env"
        )
        if env_file is not None and not path.is_file():
            raise FileNotFoundError(path)
        return cls(_env_file=path, configuration_source=str(path))

    def safe_dump(self) -> dict[str, object]:
        """Return configuration suitable for logs and diagnostics."""
        payload = self.model_dump(mode="json")
        payload["deepseek_api_key"] = "********" if self.deepseek_api_key else None
        return payload

    def redact(self, text: str) -> str:
        """Remove the configured API key from arbitrary diagnostic text."""
        if not self.deepseek_api_key:
            return text
        value = self.deepseek_api_key.get_secret_value()
        return text.replace(value, "********") if value else text
