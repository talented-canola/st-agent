"""Numbered, reversible SQLite migrations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from st_agent.db import connect_database
from st_agent.utils import utc_now_iso


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    up_sql: str
    down_sql: str


MIGRATION_0001 = Migration(
    version=1,
    name="foundation_schema",
    up_sql="""
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    root_path TEXT NOT NULL UNIQUE,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE datasets (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    display_name TEXT NOT NULL,
    source_path TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
    status TEXT NOT NULL CHECK(status IN ('registered','missing','invalid')),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(project_id, sha256)
);
CREATE TABLE dataset_revisions (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    parent_revision_id TEXT REFERENCES dataset_revisions(id),
    status TEXT NOT NULL,
    relative_path TEXT,
    sha256 TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT,
    status TEXT NOT NULL CHECK(status IN ('open','closed')),
    created_at TEXT NOT NULL,
    closed_at TEXT
);
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user','assistant','system','tool')),
    content TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    dataset_id TEXT REFERENCES datasets(id) ON DELETE SET NULL,
    active_revision_id TEXT REFERENCES dataset_revisions(id) ON DELETE SET NULL,
    status TEXT NOT NULL,
    goal TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE run_steps (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    status TEXT NOT NULL,
    step_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(run_id, ordinal)
);
CREATE TABLE tool_calls (
    id TEXT PRIMARY KEY,
    run_step_id TEXT NOT NULL REFERENCES run_steps(id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    status TEXT NOT NULL,
    arguments_json TEXT NOT NULL DEFAULT '{}',
    result_path TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE approvals (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    request_json TEXT NOT NULL DEFAULT '{}',
    response_json TEXT,
    created_at TEXT NOT NULL,
    answered_at TEXT
);
CREATE TABLE artifacts (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(project_id, relative_path)
);
CREATE TABLE knowledge_sources (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    uri TEXT NOT NULL,
    version TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE knowledge_chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES knowledge_sources(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text_path TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(source_id, ordinal)
);
CREATE TABLE retrieval_runs (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
    query_text TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    result_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE TABLE evidence (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
    source_kind TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    locator_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE claims (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
    claim_text TEXT NOT NULL,
    confidence TEXT NOT NULL,
    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE TABLE user_memories (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    value_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE agent_checkpoints (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    state_path TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE events (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX idx_datasets_project ON datasets(project_id);
CREATE INDEX idx_sessions_project ON sessions(project_id, created_at);
CREATE INDEX idx_messages_session ON messages(session_id, created_at);
CREATE INDEX idx_artifacts_project_run ON artifacts(project_id, run_id);
CREATE INDEX idx_events_project_created ON events(project_id, created_at);
""",
    down_sql="""
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS agent_checkpoints;
DROP TABLE IF EXISTS user_memories;
DROP TABLE IF EXISTS claims;
DROP TABLE IF EXISTS evidence;
DROP TABLE IF EXISTS retrieval_runs;
DROP TABLE IF EXISTS knowledge_chunks;
DROP TABLE IF EXISTS knowledge_sources;
DROP TABLE IF EXISTS artifacts;
DROP TABLE IF EXISTS approvals;
DROP TABLE IF EXISTS tool_calls;
DROP TABLE IF EXISTS run_steps;
DROP TABLE IF EXISTS runs;
DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS dataset_revisions;
DROP TABLE IF EXISTS datasets;
DROP TABLE IF EXISTS projects;
""",
)

MIGRATION_0002 = Migration(
    version=2,
    name="revision_and_tool_runtime",
    up_sql="""
ALTER TABLE datasets
ADD COLUMN active_revision_id TEXT REFERENCES dataset_revisions(id) ON DELETE SET NULL;

ALTER TABLE dataset_revisions ADD COLUMN call_id TEXT;
ALTER TABLE dataset_revisions ADD COLUMN kind TEXT NOT NULL DEFAULT 'derived';
ALTER TABLE dataset_revisions ADD COLUMN provenance_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE dataset_revisions ADD COLUMN validation_json TEXT;
ALTER TABLE dataset_revisions ADD COLUMN error_json TEXT;
ALTER TABLE dataset_revisions ADD COLUMN updated_at TEXT;

CREATE UNIQUE INDEX idx_revisions_dataset_call
ON dataset_revisions(dataset_id, call_id)
WHERE call_id IS NOT NULL;
CREATE INDEX idx_revisions_dataset_status
ON dataset_revisions(dataset_id, status, created_at);
CREATE INDEX idx_datasets_active_revision ON datasets(active_revision_id);

CREATE TABLE worker_jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    call_id TEXT NOT NULL UNIQUE,
    revision_id TEXT REFERENCES dataset_revisions(id) ON DELETE SET NULL,
    status TEXT NOT NULL CHECK(status IN (
        'queued','running','succeeded','failed','cancelled','blocked'
    )),
    entrypoint TEXT NOT NULL,
    request_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT,
    error_json TEXT,
    pid INTEGER,
    owner_pid INTEGER,
    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0, 1)),
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_worker_jobs_project_status
ON worker_jobs(project_id, status, created_at);
CREATE UNIQUE INDEX idx_worker_jobs_one_active_per_project
ON worker_jobs(project_id)
WHERE status IN ('queued', 'running');

CREATE TABLE tool_executions (
    call_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    dataset_id TEXT REFERENCES datasets(id) ON DELETE SET NULL,
    tool_name TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    input_revision_id TEXT REFERENCES dataset_revisions(id) ON DELETE SET NULL,
    output_revision_id TEXT REFERENCES dataset_revisions(id) ON DELETE SET NULL,
    status TEXT NOT NULL CHECK(status IN (
        'success','warning','blocked','failed','cancelled'
    )),
    cache_key TEXT,
    result_json TEXT,
    result_path TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    CHECK(result_json IS NOT NULL OR result_path IS NOT NULL)
);
CREATE INDEX idx_tool_executions_project_cache
ON tool_executions(project_id, cache_key, status);

UPDATE projects SET schema_version = 2;
""",
    down_sql="""
DROP INDEX IF EXISTS idx_tool_executions_project_cache;
DROP TABLE IF EXISTS tool_executions;
DROP INDEX IF EXISTS idx_worker_jobs_project_status;
DROP INDEX IF EXISTS idx_worker_jobs_one_active_per_project;
DROP TABLE IF EXISTS worker_jobs;
DROP INDEX IF EXISTS idx_datasets_active_revision;
DROP INDEX IF EXISTS idx_revisions_dataset_status;
DROP INDEX IF EXISTS idx_revisions_dataset_call;

ALTER TABLE dataset_revisions DROP COLUMN updated_at;
ALTER TABLE dataset_revisions DROP COLUMN error_json;
ALTER TABLE dataset_revisions DROP COLUMN validation_json;
ALTER TABLE dataset_revisions DROP COLUMN provenance_json;
ALTER TABLE dataset_revisions DROP COLUMN kind;
ALTER TABLE dataset_revisions DROP COLUMN call_id;
ALTER TABLE datasets DROP COLUMN active_revision_id;

UPDATE projects SET schema_version = 1;
""",
)

MIGRATION_0003 = Migration(
    version=3,
    name="human_gate_and_recovery",
    up_sql="""
CREATE TABLE safety_approvals (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected')),
    risk TEXT NOT NULL CHECK(risk IN ('R0','R1','R2','R3')),
    proposed_json TEXT NOT NULL,
    final_json TEXT NOT NULL,
    argument_hash TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    decision TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    UNIQUE(project_id, run_id, dataset_id, revision_id, tool_name, tool_version, argument_hash)
);
CREATE TABLE safety_checkpoints (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL CHECK(sequence > 0),
    next_node TEXT NOT NULL,
    state_json TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(project_id, run_id, sequence)
);
CREATE TABLE circuit_breakers (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    service TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('closed','open','half_open')),
    failures INTEGER NOT NULL CHECK(failures >= 0),
    opened_at REAL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(project_id, service)
);
CREATE TABLE run_cancellations (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    run_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(project_id, run_id)
);
CREATE TABLE safety_executions (
    idempotency_key TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    approval_id TEXT REFERENCES safety_approvals(id) ON DELETE RESTRICT,
    run_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('executing','completed','failed')),
    result_json TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX idx_safety_approvals_binding ON safety_approvals(
    project_id, dataset_id, revision_id, tool_name, status
);
CREATE INDEX idx_safety_checkpoints_latest ON safety_checkpoints(project_id, run_id, sequence);
UPDATE projects SET schema_version = 3;
""",
    down_sql="""
DROP INDEX IF EXISTS idx_safety_checkpoints_latest;
DROP INDEX IF EXISTS idx_safety_approvals_binding;
DROP TABLE IF EXISTS run_cancellations;
DROP TABLE IF EXISTS safety_executions;
DROP TABLE IF EXISTS circuit_breakers;
DROP TABLE IF EXISTS safety_checkpoints;
DROP TABLE IF EXISTS safety_approvals;
UPDATE projects SET schema_version = 2;
""",
)

MIGRATION_0004 = Migration(
    version=4,
    name="trusted_rag_and_evidence",
    up_sql="""
CREATE TABLE rag_sources (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_key TEXT NOT NULL,
    source_type TEXT NOT NULL,
    canonical_uri TEXT NOT NULL,
    version TEXT NOT NULL,
    license TEXT NOT NULL,
    trust_tier TEXT NOT NULL CHECK(trust_tier IN ('official','peer_reviewed','curated')),
    content_sha256 TEXT NOT NULL,
    manifest_version TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(project_id, source_key, version)
);
CREATE TABLE rag_chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES rag_sources(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    text TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    locator_json TEXT NOT NULL,
    embedding_json TEXT,
    UNIQUE(source_id, ordinal)
);
CREATE VIRTUAL TABLE rag_chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    tokenize='unicode61'
);
CREATE TABLE evidence_bundles (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    run_id TEXT,
    query_text TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE claim_ledger (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    run_id TEXT,
    claim_text TEXT NOT NULL,
    strength TEXT NOT NULL CHECK(strength IN ('observation','interpretation','strong')),
    evidence_ids_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('supported','insufficient','rejected')),
    reason TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_rag_sources_project ON rag_sources(project_id, source_key);
CREATE INDEX idx_rag_chunks_source ON rag_chunks(source_id, ordinal);
CREATE INDEX idx_claim_ledger_run ON claim_ledger(project_id, run_id);
UPDATE projects SET schema_version = 4;
""",
    down_sql="""
DROP INDEX IF EXISTS idx_claim_ledger_run;
DROP INDEX IF EXISTS idx_rag_chunks_source;
DROP INDEX IF EXISTS idx_rag_sources_project;
DROP TABLE IF EXISTS claim_ledger;
DROP TABLE IF EXISTS evidence_bundles;
DROP TABLE IF EXISTS rag_chunks_fts;
DROP TABLE IF EXISTS rag_chunks;
DROP TABLE IF EXISTS rag_sources;
UPDATE projects SET schema_version = 3;
""",
)

MIGRATION_0005 = Migration(
    version=5,
    name="layered_memory",
    up_sql="""
CREATE TABLE task_digests (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    dataset_id TEXT,
    goal TEXT NOT NULL,
    completed_json TEXT NOT NULL,
    pending_json TEXT NOT NULL,
    failed_json TEXT NOT NULL,
    facts_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, run_id)
);
CREATE TABLE memory_preferences (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('candidate','approved','rejected','forgotten')),
    scope_dataset_id TEXT,
    source_session_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX idx_memory_one_active_key
ON memory_preferences(project_id, key, COALESCE(scope_dataset_id,''))
WHERE status = 'approved';
CREATE TABLE procedural_memory (
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    instructions_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(name, version)
);
CREATE INDEX idx_task_digests_session ON task_digests(project_id, session_id, updated_at);
CREATE INDEX idx_memory_project_status ON memory_preferences(project_id, status, updated_at);
UPDATE projects SET schema_version = 5;
""",
    down_sql="""
DROP INDEX IF EXISTS idx_memory_project_status;
DROP INDEX IF EXISTS idx_task_digests_session;
DROP TABLE IF EXISTS procedural_memory;
DROP INDEX IF EXISTS idx_memory_one_active_key;
DROP TABLE IF EXISTS memory_preferences;
DROP TABLE IF EXISTS task_digests;
UPDATE projects SET schema_version = 4;
""",
)

MIGRATION_0006 = Migration(
    version=6,
    name="verified_reports",
    up_sql="""
CREATE TABLE research_reports (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    run_id TEXT,
    title TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('draft','finalized')),
    html TEXT NOT NULL,
    tool_call_ids_json TEXT NOT NULL,
    claim_ids_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    finalized_at TEXT
);
CREATE TABLE vision_reviews (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    run_id TEXT,
    artifact_id TEXT NOT NULL,
    consented INTEGER NOT NULL CHECK(consented IN (0,1)),
    structured_facts_json TEXT NOT NULL,
    accepted_claims_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_reports_project_run ON research_reports(project_id, run_id, created_at);
CREATE INDEX idx_vision_project_run ON vision_reviews(project_id, run_id, created_at);
UPDATE projects SET schema_version = 6;
""",
    down_sql="""
DROP INDEX IF EXISTS idx_vision_project_run;
DROP INDEX IF EXISTS idx_reports_project_run;
DROP TABLE IF EXISTS vision_reviews;
DROP TABLE IF EXISTS research_reports;
UPDATE projects SET schema_version = 5;
""",
)

MIGRATIONS = (
    MIGRATION_0001,
    MIGRATION_0002,
    MIGRATION_0003,
    MIGRATION_0004,
    MIGRATION_0005,
    MIGRATION_0006,
)
LATEST_SCHEMA_VERSION = MIGRATIONS[-1].version


class MigrationError(RuntimeError):
    """Raised when a requested migration target is invalid."""


class MigrationManager:
    def __init__(self, database_path: str | Path, *, busy_timeout_ms: int = 30_000):
        self.database_path = Path(database_path)
        self.busy_timeout_ms = busy_timeout_ms

    def _ensure_registry(self) -> None:
        with connect_database(
            self.database_path, busy_timeout_ms=self.busy_timeout_ms
        ) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def current_version(self) -> int:
        self._ensure_registry()
        with connect_database(
            self.database_path, busy_timeout_ms=self.busy_timeout_ms
        ) as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
            ).fetchone()
            return int(row[0])

    def upgrade(self, target: int = LATEST_SCHEMA_VERSION) -> int:
        if target < 0 or target > LATEST_SCHEMA_VERSION:
            raise MigrationError(f"Unsupported migration target: {target}")
        self._ensure_registry()
        current = self.current_version()
        if target < current:
            return self.downgrade(target)
        for migration in MIGRATIONS:
            if current < migration.version <= target:
                with connect_database(
                    self.database_path, busy_timeout_ms=self.busy_timeout_ms
                ) as connection:
                    try:
                        connection.executescript("BEGIN IMMEDIATE;\n" + migration.up_sql)
                        connection.execute(
                            """
                            INSERT INTO schema_migrations(version, name, applied_at)
                            VALUES (?, ?, ?)
                            """,
                            (migration.version, migration.name, utc_now_iso()),
                        )
                        connection.commit()
                    except Exception:
                        connection.rollback()
                        raise
                current = migration.version
        return current

    def downgrade(self, target: int = 0) -> int:
        if target < 0 or target > LATEST_SCHEMA_VERSION:
            raise MigrationError(f"Unsupported migration target: {target}")
        self._ensure_registry()
        current = self.current_version()
        for migration in reversed(MIGRATIONS):
            if target < migration.version <= current:
                with connect_database(
                    self.database_path, busy_timeout_ms=self.busy_timeout_ms
                ) as connection:
                    try:
                        connection.execute("PRAGMA foreign_keys = OFF")
                        connection.executescript("BEGIN IMMEDIATE;\n" + migration.down_sql)
                        connection.execute(
                            "DELETE FROM schema_migrations WHERE version = ?", (migration.version,)
                        )
                        connection.commit()
                    except Exception:
                        connection.rollback()
                        raise
                current = migration.version - 1
        return current
