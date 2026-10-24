"""Revision-bound external knowledge for the production read-only discussion route."""

from __future__ import annotations

import importlib.metadata
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from st_agent.knowledge_corpus.contracts import CorpusError, sha256
from st_agent.knowledge_corpus.handoff import Applicability
from st_agent.knowledge_corpus.indexes import IndexCache
from st_agent.knowledge_corpus.preparation import source_fingerprint
from st_agent.knowledge_corpus.provider_reranker import DEFAULT_EVIDENCE_VIEW, ProviderReranker
from st_agent.knowledge_corpus.retrieval import EvidenceBundle
from st_agent.knowledge_corpus.retrieval_runtime import (
    DEFAULT_PERFORMANCE_PROFILE, api_attempt_count, conversation_context, run_retrieval,
)
from st_agent.knowledge_corpus.store import CorpusStore
from st_agent.scientific_evidence import build_evidence_bundle
from st_agent.research_context import assemble_context, context_projection
from st_agent.trace import Trace, redact

SERVICE_POLICY = "revision-source-preserving-knowledge-v1"

# Only source-card assertions that describe applicability conditions. Counts,
# processing details and biological interpretations stay facts, not conditions.
_SOURCE_BIOLOGY_FIELDS = {"species": "species", "tissue": "tissue",
    "developmental_stage": "developmental_stage", "technology": "platform",
    "platform": "platform"}


def _apply_source_biology(sample, provenance, dataset_id, revision_id, trace):
    """Fill or conflict-check biology from the same revision-bound source facts.

    The facts already passed read_saved_evidence's public-source, whitelist and
    hash gates. This function adds no evidence and never rewrites stored facts.
    """
    context = sample.get("context")
    if not isinstance(context, dict):
        context = {}
    current = context.get("biology")
    biology = dict(current) if isinstance(current, dict) else {}
    candidates = {}
    for fact in sample.get("facts", []):
        key = fact.get("key")
        if not isinstance(key, str):
            continue
        prefix, separator, source_field = key.partition("/")
        if prefix != "source_metadata" or not separator:
            continue
        field = _SOURCE_BIOLOGY_FIELDS.get(source_field)
        if field is None:
            continue
        source = provenance.get(fact.get("id"))
        if not isinstance(source, dict):
            continue
        if source.get("kind") != "registered_artifact":
            continue
        if (source.get("dataset_id") != dataset_id or
                source.get("evidence_revision_id") != revision_id or
                fact.get("revision_id") != revision_id):
            continue
        if not (source.get("artifact_id") and source.get("artifact_sha256") and
                source.get("source_payload_sha256")):
            continue
        value = fact.get("value")
        if not isinstance(value, str):
            continue
        value = value.strip()
        if not value or value.casefold() == "unknown":
            continue
        candidates.setdefault(field, []).append({"value": value,
            "normalized": value.casefold(), "fact_id": fact["id"],
            "artifact_id": source["artifact_id"]})
    if not candidates:
        return
    selected, conflicts = {}, []
    for field, entries in candidates.items():
        existing = biology.get(field)
        existing_text = existing.strip() if isinstance(existing, str) else None
        existing_known = bool(existing_text) and existing_text.casefold() != "unknown"
        values = list(dict.fromkeys(entry["value"] for entry in entries))
        normalized = {entry["normalized"] for entry in entries}
        if existing_known:
            normalized.add(existing_text.casefold())
        if len(normalized) > 1:
            if existing_known and existing_text not in values:
                values.insert(0, existing_text)
            biology[field] = None
            conflicts.append({"field": field, "values": values,
                "fact_ids": list(dict.fromkeys(entry["fact_id"] for entry in entries)),
                "artifact_ids": list(dict.fromkeys(entry["artifact_id"] for entry in entries)),
                "existing_sample_value": existing_text if existing_known else None})
            continue
        if existing_known:
            # Preserve the sample's own spelling rather than letting a card
            # silently replace an already known condition.
            biology[field] = existing
            basis = "existing_sample_matches_source_metadata"
        else:
            biology[field] = values[0]
            basis = "source_metadata"
        selected[field] = {"value": biology[field], "basis": basis,
            "fact_ids": list(dict.fromkeys(entry["fact_id"] for entry in entries)),
            "artifact_ids": list(dict.fromkeys(entry["artifact_id"] for entry in entries))}
    context["biology"] = biology
    sample["context"] = context
    trace.emit("knowledge.source_conditions.resolved", selected=selected,
        conflicts=conflicts, evidence_role="source_applicability_not_measurement",
        automatic_execution=False)


def references(bundle):
    """No conversion through the legacy lossy RAG EvidenceBundle."""
    return [
        {
            "citation_id": sha256(
                json.dumps(
                    {
                        "chunk": item.chunk_id,
                        "unit": item.unit.unit_id,
                        "text_sha256": item.text_sha256,
                        "start": item.char_start,
                        "end": item.char_end,
                    },
                    sort_keys=True,
                ).encode()
            ),
            "source_key": item.unit.source_key,
            "source_version": item.unit.source_version,
            "title": item.unit.title,
            "uri": item.unit.canonical_uri,
            "license": item.unit.license,
            "role": item.unit.role,
            "locator": item.unit.locator,
            "unit_id": item.unit.unit_id,
            "chunk_id": item.chunk_id,
            "char_start": item.char_start,
            "char_end": item.char_end,
            "matched_char_start": item.matched_char_start,
            "matched_char_end": item.matched_char_end,
            "matched_text_sha256": item.matched_text_sha256,
            "text_sha256": item.text_sha256,
            "condition_status": item.condition_status,
            "support_scope": item.support_scope,
            "conditions": item.unit.applicability.model_dump(mode="json"),
            "excerpt": item.excerpt,
        }
        for item in bundle.items
    ]


def reference_answer(refs):
    if not refs:
        return "当前知识发布中没有找到可用的相关证据；尚未执行分析，也不据此推断样本结论。"
    lines = ["已找到以下可追溯参考。原文用于方法或研究背景，不代表已验证当前样本结论。"]
    for index, item in enumerate(refs[:3], 1):
        lines.append(
            f"[{index}] {item['title']}\n\n> " + item["excerpt"][:450].replace("\n", "\n> ")
        )
    return "\n\n".join(lines)


def marker_context(sample, project):
    """Preserve recorded marker semantics without assigning the latest call to old facts."""
    identity = sample["identity"]
    call_id = identity.get("worker_call_id")
    result = project.tool_executions.find_by_call(call_id) if call_id else None
    quality = sample.get("quality", {})
    metadata = {"revision_id": identity["revision_id"], "markers": quality.get("markers", {}),
                "warnings": quality.get("warnings", []), "observation_unit": "observation"}
    provenance = {}
    if result is not None and result.tool_name == "markers.rank" and \
            result.output_revision_id == identity["revision_id"] and \
            result.status.value in {"success", "warning"}:
        source_hash = sha256(json.dumps(quality, sort_keys=True,
                                       ensure_ascii=False, allow_nan=False).encode())
        for fact in sample["facts"]:
            if fact["key"].startswith("marker."):
                provenance[fact["id"]] = {"kind": "tool_result",
                    "dataset_id": identity["dataset_id"],
                    "evidence_revision_id": fact["revision_id"], "source_call_id": result.call_id,
                    "tool_name": result.tool_name, "tool_version": result.provenance.tool_version,
                    "input_revision_id": result.input_revision_id,
                    "output_revision_id": result.output_revision_id, "artifact_id": None,
                    "artifact_sha256": None, "source_payload_sha256": source_hash,
                    "record_key": "quality.markers." + fact["key"].removeprefix("marker.").rsplit(".", 1)[0]}
    return metadata, provenance


class KnowledgeService:
    def __init__(self, copilot, client):
        self.copilot, self.client = copilot, client

    def _prepare_sample(self, dataset_id, session_id, selection, trace):
        from st_agent.evidence_context import (EvidenceRequest, MarkerSelection,
            SavedEvidenceSelection, read_marker_evidence, read_saved_evidence)
        project = self.copilot
        sample = build_evidence_bundle(project, dataset_id)
        metadata, provenance = marker_context(sample, project)
        availability, figure_refs, source_summary = [], [], {}

        def merge(extra, chosen, *, automatic=False):
            metadata.update(extra.get("marker_metadata", {}))
            known = {f["id"]: f for f in sample["facts"]}
            for fact in extra["facts"]:
                incoming = extra["fact_provenance"][fact["id"]]
                old = next((f for f in sample["facts"] if f["key"] == fact["key"]
                    and f["revision_id"] == fact["revision_id"] and
                    all(provenance.get(f["id"], {}).get(k) == incoming.get(k) for k in
                        ("kind", "source_call_id", "artifact_id", "source_payload_sha256"))), None)
                if old is not None:
                    if old["value"] != fact["value"] or type(old["value"]) is not type(fact["value"]):
                        raise CorpusError("evidence.saved_fact_conflict")
                    continue
                if fact["id"] in known:
                    raise CorpusError("evidence.duplicate_fact")
                sample["facts"].append(fact)
                known[fact["id"]] = fact
            provenance.update({fid: value for fid, value in extra["fact_provenance"].items() if fid in known})
            availability.extend(extra["availability"])
            trace.emit("scientific.evidence.read", selection=chosen.model_dump(),
                added_fact_ids=[f["id"] for f in extra["facts"]], availability=extra["availability"],
                host_selected=automatic,
                automatic_execution=False)

        # A registered input description is small, read-only context needed to
        # interpret counts and study units. It is not a tool measurement and
        # cannot silently follow the sample into a later revision.
        revision_id = sample["identity"]["revision_id"]
        if any(a.kind == "data.source_metadata" and a.metadata.get("dataset_id") == dataset_id
               and a.metadata.get("revision_id") == revision_id for a in project.list_artifacts()):
            source_selection = SavedEvidenceSelection(kind="source_metadata")
            source_extra = read_saved_evidence(project, dataset_id=dataset_id,
                revision_id=revision_id, selection=source_selection)
            merge(source_extra, source_selection, automatic=True)
            source_summary = source_extra.get("selection_summary", {})
        if selection is not None:
            raw = selection.model_dump() if hasattr(selection, "model_dump") else selection
            if raw.get("kind"):
                chosen = SavedEvidenceSelection.model_validate(raw)
                extra = read_saved_evidence(project, dataset_id=dataset_id,
                    revision_id=revision_id, selection=chosen)
                figure_refs = extra.get("figure_refs", [])
            else:
                chosen = MarkerSelection.model_validate(raw)
                request = EvidenceRequest(dataset_id=dataset_id,
                    revision_id=revision_id, **chosen.model_dump())
                extra = read_marker_evidence(project, request)
            merge(extra, chosen)
        # Both local generation and retrieval Applicability must see the same
        # source-derived conditions, not an older biology block.
        _apply_source_biology(sample, provenance, dataset_id, revision_id, trace)
        assembled = assemble_context(project, dataset_id, session_id=session_id, sample=sample,
                                     evidence_selection=selection)
        assembled["source_metadata"] = source_summary
        if figure_refs:
            assembled["figure_refs"] = figure_refs
        if selection is not None:
            assembled["saved_evidence_selection"] = extra.get("selection_summary", {})
        # The catalog is rebuilt from the registry and this turn's actual
        # observations/selection. It remains metadata, never a sample fact.
        from st_agent.tool_registry import build_capability_context
        prior_capability = sample.get("capability_context") or {}
        supply_summary = dict(assembled.get("saved_evidence_selection") or {})
        if selection is not None:
            raw_selection = (selection.model_dump() if hasattr(selection, "model_dump")
                             else dict(selection))
            supply_summary.setdefault("requested_fields", raw_selection.get("fields", []))
        assembled["capability_context"] = build_capability_context(
            observations=assembled.get("tool_observations", []),
            data_state=prior_capability.get("data_state"),
            selection_summary=supply_summary or None,
            availability=availability)
        assembled.pop("fingerprint", None)
        assembled["fingerprint"] = sha256(json.dumps(assembled, sort_keys=True,
            ensure_ascii=False).encode())
        return sample, metadata, provenance, availability, assembled

    def _prepare_selected_images(self, sample, assembled, evidence_selection, trace):
        """Materialize pixels only for explicitly selected, revision-bound image refs.

        HTML and other non-raster references stay available through the assembled
        figure refs, but they are never presented to the model as image pixels.
        """
        import base64
        import hashlib
        import io
        from PIL import Image

        selection_kind = (evidence_selection.get("kind") if isinstance(evidence_selection, dict)
                          else getattr(evidence_selection, "kind", None))
        refs = assembled.get("figure_refs", []) if selection_kind == "figure_refs" else []
        if not refs:
            return {}, []
        project = self.copilot
        root = project.layout.root.resolve()
        identity = sample["identity"]
        artifacts = {item.id: item for item in project.list_artifacts()}
        images, order, assets = {}, [], []
        selected = []
        for ref in refs:
            required = ("artifact_id", "kind", "relative_path", "sha256", "dataset_id", "revision_id")
            if any(not ref.get(key) for key in required) or \
                    ref["dataset_id"] != identity["dataset_id"] or \
                    ref["revision_id"] != identity["revision_id"]:
                raise CorpusError("evidence.image_reference_invalid")
            artifact = artifacts.get(ref["artifact_id"])
            if artifact is None or artifact.project_id != project.project.id or \
                    artifact.kind != ref["kind"] or \
                    artifact.relative_path != ref["relative_path"] or \
                    artifact.sha256 != ref["sha256"] or \
                    artifact.metadata.get("dataset_id") != identity["dataset_id"] or \
                    artifact.metadata.get("revision_id") != identity["revision_id"]:
                raise CorpusError("evidence.image_reference_invalid")
            selected.append((ref, artifact))

        total_bytes = 0

        def add_image(image_id, content, mime, meta, ref):
            nonlocal total_bytes
            current = images.get(image_id)
            if current is None:
                # Charge distinct outbound images before retaining their bytes.
                if len(images) >= 64 or total_bytes + len(content) > 24 * 1024 * 1024:
                    raise CorpusError("evidence.image_collection_budget_exceeded")
                total_bytes += len(content)
                images[image_id] = {"content": content, "mime": mime, "meta": meta,
                    "artifact_ids": [ref["artifact_id"]], "artifact_kinds": {ref["kind"]}}
                order.append(image_id)
                return
            if current["meta"].get("sha256") != meta.get("sha256") or current["content"] != content:
                raise CorpusError("evidence.image_identity_conflict")
            if ref["artifact_id"] not in current["artifact_ids"]:
                current["artifact_ids"].append(ref["artifact_id"])
            current["artifact_kinds"].add(ref["kind"])

        single_figure = None
        for ref, artifact in selected:
            suffix = Path(ref["relative_path"]).suffix.lower()
            image_ids = []
            if ref["kind"].startswith("figure.full."):
                from st_agent.figure_gallery import revision_gallery
                from st_agent.full_figures import prepare_figure_images

                allowed = {"gene", "section_key", "section_value", "coordinate_key",
                    "markers_per_group", "max_marker_genes", "marker_display_mode",
                    "marker_page_index", "marker_page_size"}
                options = {key: value for key, value in (ref.get("parameters") or {}).items()
                           if key in allowed}
                gallery = revision_gallery(project, identity["dataset_id"],
                    revision_id=identity["revision_id"], full_extent=True, **options)
                figure_id = ref["kind"].removeprefix("figure.full.")
                item = next((x for x in gallery.get("figures", []) if x.get("id") == figure_id), None)
                if item is None:
                    raise CorpusError("evidence.figure_reference_unavailable")
                evidence, parts = prepare_figure_images(project, item)
                files_by_id = {f["id"]: f for f in evidence.get("files", []) if f.get("id")}
                if len(parts) % 2:
                    raise CorpusError("evidence.image_reference_invalid")
                for index in range(0, len(parts), 2):
                    try:
                        meta = json.loads(parts[index]["text"])
                        url = parts[index + 1]["image_url"]["url"]
                    except (KeyError, TypeError, ValueError):
                        raise CorpusError("evidence.image_reference_invalid")
                    prefix = "data:image/png;base64,"
                    if not isinstance(url, str) or not url.startswith(prefix):
                        raise CorpusError("evidence.image_format_invalid")
                    content = base64.b64decode(url[len(prefix):], validate=True)
                    image_id = meta.get("id")
                    if not image_id or hashlib.sha256(content).hexdigest() != meta.get("sha256"):
                        raise CorpusError("evidence.image_fingerprint_changed")
                    with Image.open(io.BytesIO(content)) as rendered:
                        width, height, image_format = rendered.width, rendered.height, rendered.format
                        if image_format != "PNG" or \
                                (width, height) != (meta.get("width"), meta.get("height")):
                            raise CorpusError("evidence.image_format_invalid")
                        if width < 1 or height < 1 or width * height > 32_000_000:
                            raise CorpusError("evidence.image_budget_exceeded")
                        rendered.load()
                    source = files_by_id.get(image_id, {})
                    add_image(image_id, content, "image/png", {
                        **meta, "width": width, "height": height,
                        "relative_path": source.get("relative_path")}, ref)
                    image_ids.append(image_id)
                single_figure = item
            elif suffix in {".png", ".jpg", ".jpeg"}:
                path = (root / ref["relative_path"]).resolve()
                if not path.is_relative_to(root):
                    raise CorpusError("evidence.image_path_invalid")
                try:
                    if path.stat().st_size > 20_000_000:
                        raise CorpusError("evidence.image_budget_exceeded")
                    content = path.read_bytes()
                except OSError:
                    raise CorpusError("evidence.image_reference_invalid")
                if hashlib.sha256(content).hexdigest() != ref["sha256"]:
                    raise CorpusError("evidence.image_fingerprint_changed")
                if len(content) > 20_000_000:
                    raise CorpusError("evidence.image_budget_exceeded")
                with Image.open(io.BytesIO(content)) as provided:
                    width, height, image_format = provided.width, provided.height, provided.format
                    expected = "PNG" if suffix == ".png" else "JPEG"
                    if image_format != expected:
                        raise CorpusError("evidence.image_format_invalid")
                    if width < 1 or height < 1 or width * height > 32_000_000:
                        raise CorpusError("evidence.image_budget_exceeded")
                    provided.load()
                image_id = "image_" + ref["sha256"][:24]
                mime = "image/png" if expected == "PNG" else "image/jpeg"
                add_image(image_id, content, mime, {"id": image_id, "kind": "provided_public_image",
                    "region": "full", "width": width, "height": height,
                    "sha256": ref["sha256"], "relative_path": ref["relative_path"]}, ref)
                image_ids.append(image_id)
            # Ordinary HTML (and any other non-raster) references are not pixels.
            assets.append({"source_artifact_id": ref["artifact_id"], "kind": ref["kind"],
                "sha256": ref["sha256"], "dataset_id": ref["dataset_id"],
                "revision_id": ref["revision_id"],
                "pixels_supplied": bool(image_ids),
                "image_ids": list(dict.fromkeys(image_ids))})

        if len(order) > 64:
            raise CorpusError("evidence.image_collection_budget_exceeded")
        total_bytes = sum(len(image["content"]) for image in images.values())
        if total_bytes > 24 * 1024 * 1024:
            raise CorpusError("evidence.image_collection_budget_exceeded")
        manifest = {"assets": assets, "image_ids": list(order),
                    "selected_ref_count": len(assets), "sent_image_count": len(order),
                    "total_bytes": total_bytes}
        manifest_sha256 = sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode())
        trace.emit("knowledge.images.prepared", manifest=manifest,
                   manifest_sha256=manifest_sha256, selected_ref_count=len(assets),
                   sent_image_count=len(order), total_bytes=total_bytes,
                   image_ids=list(order), automatic_execution=False)
        if not order:
            return {}, []

        files, image_parts = [], []
        for image_id in order:
            image = images[image_id]
            meta = image["meta"]
            file = {"id": image_id, "kind": meta.get("kind", "provided_public_image"),
                "region": meta.get("region", "full"), "width": meta["width"],
                "height": meta["height"], "sha256": meta["sha256"],
                "source_artifact_id": image["artifact_ids"][0],
                "source_artifact_ids": list(image["artifact_ids"]),
                "artifact_kinds": sorted(image["artifact_kinds"]),
                "dataset_id": identity["dataset_id"], "revision_id": identity["revision_id"],
                "revision_sha256": identity.get("sha256"), "send_to_model": True}
            if meta.get("relative_path"):
                file["relative_path"] = meta["relative_path"]
            files.append(file)
            label = {key: file[key] for key in ("id", "source_artifact_id", "source_artifact_ids",
                "artifact_kinds", "kind", "sha256", "dataset_id", "revision_id", "revision_sha256")}
            image_parts.extend([
                {"type": "text", "text": json.dumps(label, ensure_ascii=False)},
                {"type": "image_url", "image_url": {
                    "url": "data:" + image["mime"] + ";base64," +
                           base64.b64encode(image["content"]).decode("ascii"),
                    "detail": "original"}}])
        context_extra = {"image_evidence": {"files": files,
            "manifest_sha256": manifest_sha256, "selected_ref_count": len(assets),
            "sent_image_count": len(files), "total_bytes": total_bytes}}
        if len(assets) == 1 and assets[0]["pixels_supplied"]:
            ref = selected[0][0]
            if single_figure is not None:
                context_extra["figure"] = {
                    "id": single_figure["id"], "title": single_figure["title"],
                    "caption": single_figure["caption"],
                    "metadata": {**(single_figure.get("metadata") or {}),
                        "source_artifact_id": ref["artifact_id"],
                        "dataset_id": identity["dataset_id"],
                        "revision_id": identity["revision_id"],
                        "revision_sha256": identity.get("sha256")}}
            else:
                context_extra["figure"] = {"id": ref["artifact_id"], "title": ref["kind"],
                    "caption": "用户提供的图像；像素可用于图中可见特征，不提供未测量的分子或细胞身份。",
                    "metadata": {"dataset_id": identity["dataset_id"],
                        "revision_id": identity["revision_id"],
                        "revision_sha256": identity.get("sha256"),
                        "source_artifact_id": ref["artifact_id"]}}
        else:
            context_extra["figure"] = {"id": "selected-figures", "title": "已选图像集合",
                "caption": "本轮选择 " + str(len(assets)) +
                    " 个已保存图像引用；只有 pixels_supplied=true 的像素送入模型。",
                "metadata": {"dataset_id": identity["dataset_id"],
                    "revision_id": identity["revision_id"],
                    "revision_sha256": identity.get("sha256"),
                    "source_artifact_ids": [a["source_artifact_id"] for a in assets],
                    "assets": assets}}
        return context_extra, image_parts

    def respond_local(self, query, *, dataset_id, run_id, history=None, session_id=None,
                      evidence_selection=None, response_requirements=None):
        """Explicit sample discussion consumes saved evidence without an external search."""
        project = self.copilot
        directory = project.layout.root / "artifacts" / "knowledge_runs" / uuid.uuid4().hex
        trace = Trace(directory, sink=project.events)
        trace.context.update(run_id=run_id, turn_id=run_id)
        previous_trace = getattr(self.client, "trace", None)
        self.client.trace = trace
        before = api_attempt_count(self.client)
        result = {"answer": "", "status": "failed", "references": [], "reused": False,
            "artifact_ref": None, "trace_ref": trace.path.relative_to(project.layout.root).as_posix()}
        trace.emit("knowledge.integration.start", dataset_id=dataset_id, retrieval_requested=False)
        try:
            sample, metadata, provenance, availability, assembled = self._prepare_sample(
                dataset_id, session_id, evidence_selection, trace)
            result["identity"] = sample["identity"]
            context = {"identity": sample["identity"], "facts": assembled["facts"],
                "research_context": context_projection(assembled), "references": [],
                "capability_context": assembled.get("capability_context", {}),
                "conditions": sample["context"].get("biology", {}),
                "conversation": conversation_context(history), "marker_metadata": metadata,
                "fact_provenance": provenance, "availability": availability,
                "response_requirements": response_requirements or {},
                "saved_evidence_selection": assembled.get("saved_evidence_selection", {}),
                "source_metadata": assembled.get("source_metadata", {}),
                "retrieval_state": {"external_knowledge_requested": False,
                                    "external_knowledge_available": None}}
            context_extra, image_parts = self._prepare_selected_images(
                sample, assembled, evidence_selection, trace)
            context.update(context_extra)
            (directory / "context.json").write_text(json.dumps(redact(context),
                ensure_ascii=False, indent=2), encoding="utf-8")
            trace.emit("knowledge.context.delivered", context=context,
                       automatic_execution=False, retrieval_skipped="sample_only_request")
            from st_agent.knowledge_answering import scientific_answer
            generated = scientific_answer(query, context, self.client, trace, image_parts=image_parts)
            current = project.get_dataset(dataset_id).active_revision_id
            if current != sample["identity"]["revision_id"]:
                raise CorpusError("knowledge.input_fingerprint_changed")
            result.update(answer=generated["answer"], answer_audit=generated["audit"],
                status=generated["status"], identity=sample["identity"])
            saved = directory / "answer.json"
            saved.write_text(json.dumps(redact({"context": context, "result": result}),
                ensure_ascii=False, indent=2), encoding="utf-8")
            artifact = project.register_artifact(saved, kind="knowledge.answer", run_id=run_id,
                metadata={"revision_id": current, "retrieval_requested": False})
            result["artifact_ref"] = artifact.relative_path
        except Exception as error:
            import traceback
            code = str(error) if isinstance(error, (CorpusError, ValueError)) else type(error).__name__
            result.update(status="failed", error_code=code,
                answer="本次保存结果的解释未完成，已有分析结果保留。错误码：" + code)
            trace.emit("knowledge.local.error", status="failed", error_code=code,
                message=project.settings.redact(str(error)),
                frames=[{"module": Path(f.filename).name, "function": f.name, "line": f.lineno}
                        for f in traceback.extract_tb(error.__traceback__)])
        finally:
            after = api_attempt_count(self.client)
            result["api_requests"] = after - before if before is not None and after is not None else None
            result["trace_degraded"] = trace.degraded
            self.client.trace = previous_trace
            trace.emit("knowledge.integration.end", status=result["status"],
                       api_requests=result["api_requests"], retrieval_requested=False)
        (directory / "report.json").write_text(json.dumps(redact(result),
            ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def respond(self, query, *, dataset_id, run_id, history=None, session_id=None,
                evidence_selection=None, retrieval_required=True, response_requirements=None):
        if not retrieval_required:
            return self.respond_local(query, dataset_id=dataset_id, run_id=run_id,
                history=history, session_id=session_id, evidence_selection=evidence_selection,
                response_requirements=response_requirements)
        project = self.copilot
        conversation = conversation_context(history)
        directory = project.layout.root / "artifacts" / "knowledge_runs" / uuid.uuid4().hex
        trace = Trace(directory, sink=project.events)
        trace.context.update(run_id=run_id, turn_id=run_id)
        previous_trace = getattr(self.client, "trace", None)
        self.client.trace = trace
        start_count = api_attempt_count(self.client)
        result = {
            "answer": "",
            "status": "failed",
            "references": [],
            "reused": False,
            "artifact_ref": None,
            "trace_ref": trace.path.relative_to(project.layout.root).as_posix(),
        }
        trace.emit("knowledge.integration.start", dataset_id=dataset_id, policy=SERVICE_POLICY)
        stage = "context"
        try:
            # Older in-memory page sessions can hold pre-profile Settings objects.
            # Rollback remains explicit on every new Settings load.
            retrieval_profile = getattr(project.settings, "knowledge_retrieval_profile", DEFAULT_PERFORMANCE_PROFILE)
            reranker = ProviderReranker(self.client, evidence_view=getattr(
                project.settings, "knowledge_rerank_view", DEFAULT_EVIDENCE_VIEW))
            sample, metadata, provenance, availability, assembled = self._prepare_sample(
                dataset_id, session_id, evidence_selection, trace)
            result["identity"] = sample["identity"]
            # Initialized here so a retrieval failure after preparation can reuse
            # the same pixels; actual materialization waits for a cache miss.
            context_extra, image_parts = ({}, [])
            selected_images_prepared = False
            biology = sample["context"]["biology"]
            # The metadata reader uses "unknown" as its missing-value sentinel.
            # It must not become a known species/tissue condition in retrieval.
            biology = {k: None if v == "unknown" else v for k, v in biology.items()}
            conditions = Applicability(
                species=biology.get("species"),
                tissue=biology.get("tissue"),
                developmental_stage=biology.get("developmental_stage"),
                platform=biology.get("platform"),
            )
            # The sample is not "Scanpy-only": the stack also uses AnnData.
            # A global software condition incorrectly excludes cross-library help.
            software_environment = {}
            for package in ("scanpy", "anndata"):
                try:
                    software_environment[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    software_environment[package] = None
            stage = "retrieval_setup"
            root = project.settings.knowledge_root.resolve()
            release = root / "releases" / project.settings.knowledge_release
            store = CorpusStore(root, trace=trace)
            verification = store.verify_release(release)
            if not verification["passed"]:
                raise CorpusError("knowledge.release_unusable")
            index = IndexCache(store).prepare(release)
            from st_agent.knowledge_answering import policy_identity

            key = sha256(
                json.dumps(
                    {
                        "request": query,
                        "response_requirements": response_requirements or {},
                        "conversation": conversation,
                        "identity": sample["identity"],
                        "research_context_fingerprint": assembled["fingerprint"],
                        "evidence_selection": (evidence_selection.model_dump()
                            if hasattr(evidence_selection, "model_dump") else evidence_selection),
                        "conditions": conditions.model_dump(mode="json"),
                        "software_environment": software_environment,
                        "index": index.cache_key,
                        "service_sha256": sha256(Path(__file__).read_bytes()),
                        "runtime_sha256": source_fingerprint(),
                        "model": getattr(self.client, "model", type(self.client).__name__),
                        "generate": project.settings.knowledge_generate_answers,
                        "retrieval_profile": retrieval_profile,
                        "answer_policy": policy_identity(),
                        "reranker_policy": reranker.identity,
                        # Provider aliases lack immutable weight revisions; bound reuse to one day.
                        "cache_epoch_utc": datetime.now(UTC).strftime("%Y-%m-%d"),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode()
            )
            cache = project.layout.root / "artifacts" / "knowledge_cache" / (key + ".json")
            stage = "cache"
            if cache.is_file():
                pointer = json.loads(cache.read_bytes())
                target = (project.layout.root / pointer["artifact_ref"]).resolve()
                if project.layout.root.resolve() not in target.parents:
                    raise CorpusError("knowledge.cache_path_invalid")
                artifacts = {item.relative_path: item.sha256 for item in project.list_artifacts()}
                relative = target.relative_to(project.layout.root).as_posix()
                if (
                    artifacts.get(relative) != pointer["sha256"]
                    or sha256(target.read_bytes()) != pointer["sha256"]
                ):
                    raise CorpusError("knowledge.cache_integrity_failed")
                cached = json.loads(target.read_bytes())
                if cached["cache_key"] != key:
                    raise CorpusError("knowledge.cache_identity_mismatch")
                replay_trace = result["trace_ref"]
                result.update(cached["result"], reused=True)
                result["original_trace_ref"] = result["trace_ref"]
                result["trace_ref"] = replay_trace
                trace.emit("knowledge.answer.reused", cache_key=key, artifact_ref=relative)
            else:
                # Cache misses only: a cache hit returns the previously reviewed
                # answer without re-rendering or re-encoding selected pixels.
                if project.settings.knowledge_generate_answers:
                    context_extra, image_parts = self._prepare_selected_images(
                        sample, assembled, evidence_selection, trace)
                    selected_images_prepared = True
                stage = "retrieval"
                retrieval = run_retrieval(
                    root,
                    release,
                    directory / "retrieval",
                    text=query,
                    conditions=conditions,
                    client=self.client,
                    reranker=reranker,
                    history=conversation,
                    research_context=context_projection(assembled),
                    performance_profile=retrieval_profile,
                )
                if not retrieval["passed"]:
                    raise CorpusError(retrieval.get("error_code", "knowledge.retrieval_failed"))
                raw = json.loads((directory / "retrieval" / "evidence.json").read_bytes())
                bundle = EvidenceBundle.model_validate(raw)
                refs = references(bundle)
                allowed_refs, excluded = [], []
                for ref in refs:
                    source_conditions = ref["conditions"]
                    package = source_conditions.get("software")
                    expected = source_conditions.get("software_version")
                    installed = software_environment.get(package)
                    if expected and installed and installed != expected:
                        excluded.append(ref["citation_id"])
                        continue
                    ref["software_compatibility"] = (
                        "matched" if expected and installed == expected else "unknown"
                    )
                    allowed_refs.append(ref)
                refs = allowed_refs
                stage = "answer"
                trace.emit(
                    "knowledge.software.assessed",
                    installed=software_environment,
                    excluded_citation_ids=excluded,
                )
                result.update(
                    references=refs,
                    status="references_only" if refs else "no_evidence",
                    answer=reference_answer(refs),
                    release_sha256=bundle.release_sha256,
                    index_cache_key=bundle.cache_key,
                    warnings=list(bundle.warnings),
                )
                context = {
                    "response_requirements": response_requirements or {},
                    "saved_evidence_selection": assembled.get("saved_evidence_selection", {}),
                    "source_metadata": assembled.get("source_metadata", {}),
                    "identity": sample["identity"],
                    "facts": assembled["facts"],
                    "research_context": context_projection(assembled),
                    "capability_context": assembled.get("capability_context", {}),
                    "conditions": conditions.model_dump(mode="json"),
                    "software_environment": software_environment,
                    "references": refs,
                    "release_sha256": bundle.release_sha256,
                    "conversation": conversation,
                    "marker_metadata": metadata,
                    "fact_provenance": provenance,
                    "availability": availability,
                    "retrieval_state": {"release_sha256": bundle.release_sha256,
                        "index_cache_key": bundle.cache_key,
                        "reranker_used": raw.get("reranker_used"),
                        "dense_used": raw.get("embedding") is not None,
                        "warnings": list(bundle.warnings)},
                    **context_extra,
                }
                (directory / "context.json").write_text(
                    json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                trace.emit(
                    "knowledge.context.delivered",
                    revision_id=sample["identity"]["revision_id"],
                    release_sha256=bundle.release_sha256,
                    citation_ids=[r["citation_id"] for r in refs],
                    reference_count=len(refs),
                    research_context_fingerprint=assembled["fingerprint"],
                    conditions=context["conditions"],
                )
                if project.settings.knowledge_generate_answers:
                    from st_agent.knowledge_answering import scientific_answer

                    generated = scientific_answer(query, context, self.client, trace,
                                                  image_parts=image_parts)
                    result.update(
                        answer=generated["answer"],
                        answer_audit=generated["audit"],
                        status=generated["status"],
                    )
                result["artifact_ref"] = (
                    (directory / "answer.json").relative_to(project.layout.root).as_posix()
                )
                document = {"cache_key": key, "context": context, "result": result}
                saved = directory / "answer.json"
                saved.write_text(
                    json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                artifact = project.register_artifact(
                    saved,
                    kind="knowledge.answer",
                    run_id=run_id,
                    metadata={
                        "revision_id": sample["identity"]["revision_id"],
                        "release_sha256": bundle.release_sha256,
                    },
                )
                # Exclusive creation: failed or competing attempts never overwrite evidence.
                cache.parent.mkdir(parents=True, exist_ok=True)
                if result["status"] != "degraded":
                    try:
                        with cache.open("x", encoding="utf-8") as stream:
                            json.dump(
                                {"artifact_ref": result["artifact_ref"], "sha256": artifact.sha256},
                                stream,
                            )
                    except FileExistsError:
                        trace.emit("knowledge.cache.concurrent_result_retained", cache_key=key)
                else:
                    trace.emit("knowledge.cache.degraded_not_reused", cache_key=key)
            stage = "revision_guard"
            dataset = project.get_dataset(dataset_id)
            revision = project.revisions.get(dataset.active_revision_id)
            if (
                revision.id != sample["identity"]["revision_id"]
                or revision.sha256 != sample["identity"]["sha256"]
            ):
                raise CorpusError("knowledge.input_revision_changed")
            from st_agent.utils import sha256_file

            if sha256_file(project.layout.root / revision.relative_path) != revision.sha256:
                raise CorpusError("knowledge.input_fingerprint_changed")
        except Exception as error:
            code = (
                str(error)
                if isinstance(error, CorpusError)
                else "knowledge." + type(error).__name__
            )
            result.update(
                status="failed",
                error_code=code,
                answer="知识检索或证据核验未完成，未生成科学结论，也未执行数据分析。错误码："
                + code,
            )
            trace.emit("knowledge.integration.error", status="failed", error_code=code)
            if stage in {"retrieval_setup", "retrieval"} and project.settings.knowledge_generate_answers:
                # Only external knowledge availability degrades. Integrity, version,
                # ownership and cache failures must NOT take this fallback path.
                try:
                    from st_agent.knowledge_answering import scientific_answer
                    from st_agent.utils import sha256_file
                    revision = project.revisions.get(project.get_dataset(dataset_id).active_revision_id)
                    if revision.id != sample["identity"]["revision_id"] or \
                            sha256_file(project.layout.root / revision.relative_path) != sample["identity"]["sha256"]:
                        raise CorpusError("knowledge.input_fingerprint_changed")
                    # A retrieval_setup failure can precede the cache-miss image
                    # preparation. Prepare once here; a no-image ({}, []) result
                    # is still a completed preparation and is never repeated.
                    if not selected_images_prepared:
                        context_extra, image_parts = self._prepare_selected_images(
                            sample, assembled, evidence_selection, trace)
                        selected_images_prepared = True
                    context = {"identity": sample["identity"], "facts": assembled["facts"],
                        "research_context": context_projection(assembled), "references": [],
                        "capability_context": assembled.get("capability_context", {}),
                        "conditions": conditions.model_dump(mode="json"), "conversation": conversation,
                        "marker_metadata": metadata, "fact_provenance": provenance,
                        "availability": availability,
                        "response_requirements": response_requirements or {},
                        "saved_evidence_selection": assembled.get("saved_evidence_selection", {}),
                        "source_metadata": assembled.get("source_metadata", {}),
                        "retrieval_state": {"warnings": [code], "external_knowledge_available": False},
                        **context_extra}
                    generated = scientific_answer(query, context, self.client, trace,
                                                  image_parts=image_parts)
                    current = project.get_dataset(dataset_id).active_revision_id
                    if current != revision.id or sha256_file(project.layout.root / revision.relative_path) != revision.sha256:
                        raise CorpusError("knowledge.input_fingerprint_changed")
                    result.update(status="degraded", retrieval_error=code, error_code=None,
                        references=[], answer_audit=generated["audit"],
                        answer=generated["answer"] + "\n\n外部知识本轮不可用；以上只依据已保存样本记录，"
                               "不据此确认生物学身份或机制。检索错误码：" + code)
                    saved = directory / "answer.json"
                    saved.write_text(json.dumps({"context": context, "result": result},
                        ensure_ascii=False, indent=2), encoding="utf-8")
                    artifact = project.register_artifact(saved, kind="knowledge.answer", run_id=run_id,
                        metadata={"revision_id": revision.id, "retrieval_degraded": True})
                    result["artifact_ref"] = artifact.relative_path
                    trace.emit("knowledge.sample_fallback", status="degraded", error_code=code,
                               artifact_ref=artifact.relative_path)
                except Exception as fallback_error:
                    result.update(status="failed", error_code="knowledge.sample_fallback_failed",
                        answer="样本回答的版本或证据核验未完成，本轮未执行分析。错误码：knowledge.sample_fallback_failed")
                    trace.emit("knowledge.sample_fallback_failed", status="failed",
                               error_code=type(fallback_error).__name__)
        finally:
            end_count = api_attempt_count(self.client)
            result["api_requests"] = (
                end_count - start_count
                if start_count is not None and end_count is not None
                else None
            )
            trace.emit(
                "knowledge.integration.end",
                status=result["status"],
                api_requests=result["api_requests"],
                reused=result["reused"],
            )
            result["trace_degraded"] = trace.degraded
            self.client.trace = previous_trace
        (directory / "report.json").write_text(
            json.dumps(redact(result), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
