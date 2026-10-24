"""Condition-aware retrieval from verified, immutable corpus releases.

Returned text is untrusted reference data. Applicability matching neither validates
scientific claims nor authorizes scientific tools or analysis revision changes.
"""

from __future__ import annotations

import json
import math
import numbers
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .contracts import CorpusError, Record, sha256
from .handoff import Applicability, KnowledgeUnit
from .indexes import ChunkerSpec, EmbeddingSpec, IndexCache, encoder_identity

Collection = Literal["method_knowledge", "scientific_evidence"]
SourceRole = Literal["method", "scientific_evidence", "workflow_reference", "ontology"]
ContextPolicy = Literal["original-unit-window-v1"]


class RetrievalRequest(Record):
    query: str = Field(min_length=1, max_length=12000)
    task_type: Literal["method_help", "scientific_interpretation", "mixed"]
    conditions: Applicability = Field(default_factory=Applicability)
    collections: tuple[Collection, ...] = ("method_knowledge", "scientific_evidence")
    limit: int = Field(default=5, ge=1, le=100, strict=True)
    recall_limit: int = Field(default=50, ge=1, le=1000, strict=True)
    unknown_policy: Literal["include_with_warning", "exclude"] = "include_with_warning"
    max_chunks_per_unit: int = Field(default=1, ge=1, le=10, strict=True)
    rerank_failure_policy: Literal["degrade", "fail"] = "degrade"
    search_terms: tuple[str, ...] = Field(default=(), max_length=64)
    source_roles: tuple[SourceRole, ...] = ()
    context_max_chars: int = Field(default=6000, ge=1200, le=20000, strict=True)
    context_policy: ContextPolicy = "original-unit-window-v1"

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value):
        if not value.strip():
            raise ValueError("Nonblank retrieval query required")
        return value

    @field_validator("search_terms")
    @classmethod
    def bounded_terms(cls, value):
        if any(not term.strip() or len(term) > 300 for term in value):
            raise ValueError("Search terms must be nonblank and bounded")
        return value

    @model_validator(mode="after")
    def consistent(self):
        if any(
            value is not None and not value.strip()
            for value in self.conditions.model_dump().values()
        ):
            raise ValueError("Sample conditions must be nonblank when supplied")
        if not self.collections or len(set(self.collections)) != len(self.collections):
            raise ValueError("Unique nonempty collections required")
        if len(set(self.source_roles)) != len(self.source_roles):
            raise ValueError("Source roles must be unique")
        if self.recall_limit < self.limit:
            raise ValueError("Recall limit must cover result limit")
        if self.task_type == "scientific_interpretation":
            if "scientific_evidence" not in self.collections:
                raise ValueError("Scientific interpretation requires scientific evidence")
            if self.source_roles and not set(self.source_roles) <= {
                "scientific_evidence",
                "ontology",
            }:
                raise ValueError("Workflow/method roles cannot support scientific interpretation")
        return self


class RetrievedEvidence(Record):
    unit: KnowledgeUnit
    chunk_id: str
    excerpt: str
    char_start: int
    char_end: int
    text_sha256: str
    matched_char_start: int = Field(ge=0, strict=True)
    matched_char_end: int = Field(gt=0, strict=True)
    matched_text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    context_policy: ContextPolicy = "original-unit-window-v1"
    condition_status: Literal["matched", "unknown", "not_assessed"]
    support_scope: Literal["background_only", "applicability_matched_reference"]
    condition_details: dict[str, str]
    channel_scores: dict[str, float]

    @model_validator(mode="after")
    def original_context_and_seed(self):
        if not (
            0
            <= self.char_start
            <= self.matched_char_start
            < self.matched_char_end
            <= self.char_end
            <= len(self.unit.text)
        ):
            raise ValueError("Context must contain the matched span within its original unit")
        if self.excerpt != self.unit.text[self.char_start : self.char_end]:
            raise ValueError("Context must be an exact original-unit substring")
        if self.text_sha256 != sha256(self.excerpt.encode()):
            raise ValueError("Context text checksum mismatch")
        matched = self.unit.text[self.matched_char_start : self.matched_char_end]
        if self.matched_text_sha256 != sha256(matched.encode()):
            raise ValueError("Matched original chunk text checksum mismatch")
        return self


class RejectedEvidence(Record):
    chunk_id: str
    unit_id: str
    source_key: str
    source_version: str
    collection: Collection
    reasons: tuple[str, ...]


class EvidenceBundle(Record):
    id: str
    request: RetrievalRequest
    release_sha256: str
    cache_key: str
    items: tuple[RetrievedEvidence, ...]
    rejected: tuple[RejectedEvidence, ...]
    warnings: tuple[str, ...]
    status: Literal["evidence_available", "no_evidence"]
    embedding: EmbeddingSpec | None
    reranker_used: bool
    reranker_identity: str | None = None
    duration_ms: float


FTS_TOKEN_LIMIT = 128


def _fts_tokens(request):
    # Semantic terms come from the retrieval planner; the raw query may also
    # carry a full request and multi-turn history. Spend the fixed lexical
    # budget on the search plan first, then fill any remainder from the query.
    tokens = dict.fromkeys(
        token
        for text in (*request.search_terms, request.query)
        for token in re.findall(r"\w+", text, flags=re.UNICODE)
    )
    return list(tokens)[:FTS_TOKEN_LIMIT]


def _fts_expression(request):
    # Quote every token; none of the request is executable SQL or free FTS syntax.
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in _fts_tokens(request))


def _condition_assessment(conditions, applicability):
    requested = {key: value for key, value in conditions.model_dump().items() if value is not None}
    if not requested:
        return "not_assessed", {}, ()
    details, conflicts = {}, []
    available = applicability.model_dump()
    for key, value in requested.items():
        source = available[key]
        if source is None or not source.strip():
            details[key] = "unknown"
        elif source.strip().casefold() == value.strip().casefold():
            details[key] = "matched"
        else:
            details[key] = "conflict"
            conflicts.append("condition_conflict:" + key)
    status = "unknown" if "unknown" in details.values() else "matched"
    return status, details, tuple(conflicts)


def _finite_scores(values, count):
    if values is None or not hasattr(values, "__len__") or len(values) != count:
        raise CorpusError("retrieval.invalid_reranker_scores")
    if any(
        isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value)
        for value in values
    ):
        raise CorpusError("retrieval.invalid_reranker_scores")
    return [float(value) for value in values]


def _parent_context(unit, chunk, budget):
    """Expand one ranked seed within its parent, never into another source block."""
    seed_start, seed_end = chunk["char_start"], chunk["char_end"]
    seed_length = seed_end - seed_start
    if seed_length > budget:
        raise CorpusError("retrieval.context_budget_smaller_than_match")
    if len(unit.text) <= budget:
        start, end = 0, len(unit.text)
    else:
        # Split additional context evenly; at either edge put unused capacity
        # on the opposite side while retaining the complete matched window.
        start = max(0, min(seed_start - (budget - seed_length) // 2, len(unit.text) - budget))
        end = start + budget
    return start, end, unit.text[start:end]


def _ranking_context(unit, chunk, answer_budget, ranking_budget=None):
    """Exact ranking substring, never a summary or the returned answer evidence.

    Preserve the entire seed and extend outward to sentence/newline boundaries.
    If boundaries are absent, retain the full parent window instead of slicing a
    prerequisite/qualification. This is a target, not a hard truncation limit.
    """
    full_start, full_end, full_text = _parent_context(unit, chunk, answer_budget)
    if ranking_budget is None:
        return full_start, full_end, full_text
    if (isinstance(ranking_budget, bool) or not isinstance(ranking_budget, int)
            or not 1200 <= ranking_budget <= 20000):
        raise CorpusError("retrieval.invalid_ranking_budget")
    seed_start, seed_end = chunk["char_start"], chunk["char_end"]
    if full_end - full_start <= ranking_budget or seed_end - seed_start >= ranking_budget:
        return full_start, full_end, full_text
    extra = ranking_budget - (seed_end - seed_start)
    desired_start = max(full_start, seed_start - extra // 2)
    desired_end = min(full_end, seed_end + extra - extra // 2)
    boundaries = [full_start + m.end() for m in re.finditer(r"[.!?。！？](?=\s|$)|\n", full_text)]
    start = max([full_start, *(p for p in boundaries if p <= desired_start)])
    end = min([full_end, *(p for p in boundaries if p >= desired_end)])
    return start, end, unit.text[start:end]


class ConditionalRetriever:
    def __init__(self, store, release, *, chunker=None, embedder=None, reranker=None):
        self.store = store
        self.release = Path(release)
        self.chunker = chunker or ChunkerSpec()
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(self, request: RetrievalRequest) -> EvidenceBundle:
        # Revalidate even instances constructed with model_copy/model_construct.
        request = RetrievalRequest.model_validate(
            request.model_dump() if isinstance(request, RetrievalRequest) else request
        )
        started = time.perf_counter()
        index = IndexCache(self.store).prepare(
            self.release, chunker=self.chunker, embedder=self.embedder
        )
        identity = index.identity.embedding
        if encoder_identity(self.embedder) != identity:
            raise CorpusError("retrieval.query_embedding_identity_mismatch")
        query_vector = None
        if self.embedder is not None:
            vectors = self.embedder.encode([request.query + "\n" + " ".join(request.search_terms)])
            if encoder_identity(self.embedder) != identity:
                raise CorpusError("embedding.identity_changed_during_query_encode")
            query_vector = IndexCache._vectors(vectors, 1, identity)[0]

        collections = (
            ("scientific_evidence",)
            if request.task_type == "scientific_interpretation"
            else request.collections
        )
        warnings = []
        if query_vector is None:
            warnings.append("lexical_only_cross_language_recall_not_verified")
        expression = _fts_expression(request)
        assessments, excluded_reasons, recall_reasons = {}, {}, {}
        db = sqlite3.connect(self.store._path(index.database_ref).as_uri() + "?mode=ro", uri=True)
        try:
            rows = db.execute(
                "SELECT c.record,c.embedding_json,u.record FROM chunks c "
                "JOIN units u ON u.unit_id=c.unit_id"
            ).fetchall()
            chunks, units, vectors = {}, {}, {}
            for chunk_raw, vector_raw, unit_raw in rows:
                chunk = json.loads(chunk_raw)
                chunks[chunk["chunk_id"]] = chunk
                units[chunk["chunk_id"]] = KnowledgeUnit.model_validate_json(unit_raw)
                if vector_raw is not None:
                    vectors[chunk["chunk_id"]] = json.loads(vector_raw)

            # Metadata is already verified by IndexCache. Apply it before any
            # channel's top-K truncation so conflicting sources cannot starve
            # applicable evidence as the corpus grows.
            eligible = set()
            for key, unit in units.items():
                if unit.collection not in collections:
                    continue
                status, details, conflicts = _condition_assessment(
                    request.conditions, unit.applicability
                )
                reasons = list(conflicts)
                if request.source_roles and unit.role not in request.source_roles:
                    reasons.append("source_role_not_requested")
                if status == "unknown" and request.unknown_policy == "exclude":
                    reasons.extend(
                        "condition_unknown:" + field
                        for field, state in details.items()
                        if state == "unknown"
                    )
                if reasons:
                    excluded_reasons[key] = tuple(reasons)
                else:
                    eligible.add(key)
                    assessments[key] = status, details

            def recall(ranking):
                selected, counts = [], {}
                for key, value, channel in ranking:
                    if key not in eligible:
                        continue
                    unit_id = units[key].unit_id
                    if counts.get(unit_id, 0) >= request.max_chunks_per_unit:
                        recall_reasons.setdefault(key, set()).add("unit_window_limit")
                    elif len(selected) >= request.recall_limit:
                        recall_reasons.setdefault(key, set()).add("recall_limit")
                    else:
                        selected.append((key, value, channel))
                        counts[unit_id] = counts.get(unit_id, 0) + 1
                return selected

            scores = {}
            for collection in collections:
                rankings = []
                if expression:
                    lexical = db.execute(
                        "SELECT f.chunk_id,bm25(chunks_fts) FROM chunks_fts f "
                        "JOIN chunks c ON c.chunk_id=f.chunk_id "
                        "JOIN units u ON u.unit_id=c.unit_id "
                        "WHERE chunks_fts MATCH ? AND u.collection=? "
                        "ORDER BY bm25(chunks_fts),f.chunk_id LIMIT ?",
                        (expression, collection, index.chunk_count),
                    ).fetchall()
                    rankings.append(
                        recall(
                            [(chunk_id, -float(score), "lexical") for chunk_id, score in lexical]
                        )
                    )
                if query_vector is not None:
                    dense = []
                    qnorm = math.sqrt(sum(value * value for value in query_vector))
                    if qnorm == 0:
                        raise CorpusError("embedding.invalid_zero_query_vector")
                    for chunk_id, vector in vectors.items():
                        if chunk_id not in eligible or units[chunk_id].collection != collection:
                            continue
                        norm = math.sqrt(sum(value * value for value in vector))
                        if norm == 0:
                            raise CorpusError("embedding.invalid_zero_index_vector")
                        similarity = sum(
                            a * b for a, b in zip(query_vector, vector, strict=True)
                        ) / (qnorm * norm)
                        if not math.isfinite(similarity):
                            raise CorpusError("embedding.invalid_similarity")
                        if similarity > 0:
                            dense.append((chunk_id, similarity, "dense"))
                    dense.sort(key=lambda candidate: (-candidate[1], candidate[0]))
                    rankings.append(recall(dense))
                for ranking in rankings:
                    for rank, (chunk_id, value, channel) in enumerate(ranking, 1):
                        channel_scores = scores.setdefault(chunk_id, {"rrf": 0.0})
                        channel_scores[channel] = value
                        channel_scores["rrf"] += 1.0 / (60 + rank)
        finally:
            db.close()

        rejected = []

        def reject(chunk_id, reasons):
            unit = units[chunk_id]
            rejected.append(
                RejectedEvidence(
                    chunk_id=chunk_id,
                    unit_id=unit.unit_id,
                    source_key=unit.source_key,
                    source_version=unit.source_version,
                    collection=unit.collection,
                    reasons=reasons,
                )
            )

        for key in sorted(excluded_reasons):
            reject(key, excluded_reasons[key])
        for key in sorted(recall_reasons):
            if key not in scores:
                reject(key, tuple(sorted(recall_reasons[key])))
        candidates = sorted(scores, key=lambda key: (-scores[key]["rrf"], key))

        reranker_used = False
        reranker_identity = None
        if self.reranker is not None and candidates:
            try:
                score_evidence = getattr(self.reranker, "score_evidence", None)
                if callable(score_evidence):
                    ranking_budget = getattr(self.reranker, "context_max_chars", None)
                    scoring_candidates, windows = [], []
                    for key in candidates:
                        unit, chunk = units[key], chunks[key]
                        start, end, excerpt = _ranking_context(
                            unit, chunk, request.context_max_chars, ranking_budget)
                        full_start, full_end, _ = _parent_context(unit, chunk, request.context_max_chars)
                        scoring_candidates.append({"source_title": unit.title, "source_role": unit.role,
                                                   "locator": unit.locator, "excerpt": excerpt})
                        windows.append({"chunk_id": key, "unit_id": unit.unit_id,
                                        "source_content_sha256": unit.content_sha256,
                                        "char_start": start, "char_end": end,
                                        "text_sha256": sha256(excerpt.encode()),
                                        "matched_char_start": chunk["char_start"],
                                        "matched_char_end": chunk["char_end"],
                                        "answer_char_start": full_start, "answer_char_end": full_end,
                                        "reduced": end-start < full_end-full_start})
                    self.store._emit("retrieval.ranking_windows", target_chars=ranking_budget,
                                     candidate_count=len(candidates), windows=windows)
                    raw_scores = score_evidence(request.query, scoring_candidates)
                else:
                    raw_scores = self.reranker.score(request.query, [chunks[key]["text"] for key in candidates])
                rerank = _finite_scores(raw_scores, len(candidates))
                declared = getattr(self.reranker, "identity", None)
                if hasattr(declared, "model_dump"):
                    declared = declared.model_dump(mode="json")
                unversioned = False
                if isinstance(declared, dict):
                    # Identity serialization is part of reranking's boundary:
                    # never publish NaN/Infinity or unstable object encodings.
                    candidate_identity = json.dumps(
                        declared,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    revision = declared.get("revision")
                    unversioned = not isinstance(revision, str) or revision.strip().lower() in {
                        "",
                        "main",
                        "master",
                        "stable",
                        "latest",
                    }
                else:
                    candidate_identity = "unversioned:" + (
                        type(self.reranker).__module__ + "." + type(self.reranker).__qualname__
                    )
                    unversioned = True
            except Exception:
                self.store._emit(
                    "retrieval.reranker_failed",
                    error_code="retrieval.reranker_failed",
                    policy=request.rerank_failure_policy,
                )
                if request.rerank_failure_policy == "fail":
                    raise CorpusError("retrieval.reranker_failed") from None
                warnings.append("reranker_failed_lexical_dense_ranking_retained")
            else:
                reranker_used = True
                reranker_identity = candidate_identity
                if unversioned:
                    warnings.append("reranker_identity_unversioned")
                for key, value in zip(candidates, rerank, strict=True):
                    scores[key]["rerank"] = value
                candidates.sort(key=lambda key: (-scores[key]["rerank"], -scores[key]["rrf"], key))

        # Restrict repeated windows of one original unit before filling the result.
        # Keep duplicate sources/versions auditable rather than rewriting provenance.
        unique, unit_counts, texts = [], {}, set()
        for key in candidates:
            chunk, unit = chunks[key], units[key]
            text_key = (unit.collection, unit.source_key, unit.source_version, chunk["text_sha256"])
            if text_key in texts:
                reject(key, ("duplicate_chunk_text",))
            elif unit_counts.get(unit.unit_id, 0) >= request.max_chunks_per_unit:
                reject(key, ("unit_window_limit",))
            else:
                unique.append(key)
                unit_counts[unit.unit_id] = unit_counts.get(unit.unit_id, 0) + 1
                texts.add(text_key)

        # Near-tied passages from one long paper must not occupy every result slot.
        # Only comparable reranker scores are balanced; weak unrelated sources are
        # not promoted. This is an engineering diversity heuristic, not validation.
        if reranker_used and callable(getattr(self.reranker, "score_evidence", None)) and unique:
            floor = max(scores[key]["rerank"] for key in unique) * 0.8
            per_source = max(1, math.ceil(request.limit / 2))
            balanced, deferred, weaker, source_counts = [], [], [], {}
            for key in unique:
                source = (units[key].source_key, units[key].source_version)
                if scores[key]["rerank"] < floor:
                    weaker.append(key)
                elif source_counts.get(source, 0) >= per_source:
                    deferred.append(key)
                else:
                    balanced.append(key)
                    source_counts[source] = source_counts.get(source, 0) + 1
            diversified = balanced + deferred + weaker
            if diversified[:request.limit] != unique[:request.limit]:
                warnings.append("near_tied_sources_diversified")
            unique = diversified

        # Preserve both independently recalled collections when capacity permits.
        selected = []
        if len(collections) > 1 and request.limit >= len(collections):
            for collection in collections:
                best = next((key for key in unique if units[key].collection == collection), None)
                if best is not None:
                    selected.append(best)
        for key in unique:
            if len(selected) >= request.limit:
                break
            if key not in selected:
                selected.append(key)
        selected.sort(key=lambda key: unique.index(key))
        for key in unique:
            if key not in selected:
                reject(key, ("result_limit",))

        items = []
        for key in selected:
            chunk = chunks[key]
            context_start, context_end, excerpt = _parent_context(
                units[key], chunk, request.context_max_chars
            )
            status, details = assessments[key]
            if status == "unknown":
                warnings.append("unknown_applicability_background_only")
            if status == "not_assessed":
                warnings.append("applicability_not_assessed_background_only")
            items.append(
                RetrievedEvidence(
                    unit=units[key],
                    chunk_id=key,
                    excerpt=excerpt,
                    char_start=context_start,
                    char_end=context_end,
                    text_sha256=sha256(excerpt.encode()),
                    matched_char_start=chunk["char_start"],
                    matched_char_end=chunk["char_end"],
                    matched_text_sha256=chunk["text_sha256"],
                    context_policy=request.context_policy,
                    condition_status=status,
                    support_scope=(
                        "applicability_matched_reference"
                        if status == "matched"
                        else "background_only"
                    ),
                    condition_details=details,
                    channel_scores=scores[key],
                )
            )
        warnings.append("retrieval_does_not_validate_scientific_claim_support")
        bundle = EvidenceBundle(
            id=uuid.uuid4().hex,
            request=request,
            release_sha256=index.identity.release_sha256,
            cache_key=index.cache_key,
            items=tuple(items),
            rejected=tuple(rejected),
            warnings=tuple(dict.fromkeys(warnings)),
            status="evidence_available" if items else "no_evidence",
            embedding=identity,
            reranker_used=reranker_used,
            reranker_identity=reranker_identity,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        self.store._emit(
            "retrieval.complete",
            retrieval_id=bundle.id,
            status=bundle.status,
            cache_key=bundle.cache_key,
            release_sha256=bundle.release_sha256,
            selected_count=len(items),
            rejected_count=len(rejected),
            reranker_used=reranker_used,
            warnings=bundle.warnings,
            context_policy=request.context_policy,
            context_max_chars=request.context_max_chars,
        )
        return bundle
