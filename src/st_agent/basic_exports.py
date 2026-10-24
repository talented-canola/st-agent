"""Version-bound exports; never performs a scientific transformation."""
import json
import html
import shutil

import anndata as ad
import numpy as np
import plotly.graph_objects as go
import scanpy as sc

from st_agent.dataset_state import DatasetStateInspector
from st_agent.figure_gallery import revision_gallery
from st_agent.trace import emit
from st_agent.utils import sha256_file


def export_basic_results(copilot, task):
    directory = copilot.layout.root / "artifacts" / "basic_analysis" / task["task_id"]
    directory.mkdir(parents=True, exist_ok=True)
    existing_artifacts = {item.relative_path: item for item in copilot.list_artifacts(run_id=task['task_id'])}
    final = copilot.revisions.get(task["last_revision_id"])
    source = DatasetStateInspector(copilot)._project_path(final.relative_path)
    if sha256_file(source) != final.sha256:
        raise ValueError("Final revision fingerprint mismatch")
    data = ad.read_h5ad(source)
    if data.n_obs == 0 or data.n_vars == 0:
        raise ValueError("Empty result")
    if not {"leiden"} <= set(data.obs) or "X_umap" not in data.obsm:
        raise ValueError("Missing clustering output")
    if data.obsm["X_umap"].shape != (data.n_obs, 2) or not np.isfinite(data.obsm["X_umap"]).all():
        raise ValueError("Invalid UMAP")
    markers = sc.get.rank_genes_groups_df(data, group=None)
    if markers.empty or not set(markers["names"].astype(str)) <= set(data.var_names):
        raise ValueError("Invalid marker table")
    if not set(markers["group"].astype(str)) <= set(data.obs["leiden"].astype(str)):
        raise ValueError("Invalid marker groups")
    for column in ("scores", "pvals", "pvals_adj"):
        if column in markers and not np.isfinite(markers[column].to_numpy()).all():
            raise ValueError("Non-finite marker statistic: " + column)
    outputs = []
    markers.to_csv(directory / "markers.csv", index=False)
    outputs.append((directory / "markers.csv", "table.markers", final.id))
    final_copy = directory / "final.h5ad"
    if not final_copy.exists():
        shutil.copyfile(source, final_copy)
    if sha256_file(final_copy) != final.sha256:
        raise ValueError("Exported final H5AD mismatch")
    outputs.append((final_copy, "data.final", final.id))
    qc = task["steps"][1]["result"]
    filtering = task["steps"][2]["result"]
    summary = {"qc": qc["metrics"], "filter": filtering["metrics"],
               "qc_revision_id": qc["input_revision_id"],
               "filter_revision_id": filtering["output_revision_id"]}
    summary_path = directory / "qc_filter_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs.append((summary_path, "summary.qc_filter", final.id))
    bundles = [(qc["input_revision_id"], {"n_genes_by_counts", "total_counts", "pct_counts_mt"}),
               (final.id, {"X_umap", "spatial", "X_spatial", "spatial_xy", "markers"})]
    for revision_id, requested in bundles:
        bundle = revision_gallery(copilot, task["dataset_id"], revision_id=revision_id)
        selected = [item for item in bundle["figures"] if item["id"] in requested]
        if revision_id == final.id and not any(item["id"] == "X_umap" for item in selected):
            raise ValueError("Gallery did not produce UMAP")
        for item in selected:
            target = directory / (item["id"] + ".html")
            relative = target.relative_to(copilot.layout.root).as_posix()
            existing = existing_artifacts.get(relative)
            if existing is not None and target.is_file() and sha256_file(target) == existing.sha256:
                outputs.append((target, 'figure.' + item['id'], revision_id))
                continue
            document = go.Figure(item["figure"]).to_html(include_plotlyjs=True,
                div_id='basic-' + item['metadata']['figure_fingerprint'])
            caption = "<p>" + html.escape(item["caption"]) + "</p>"
            metadata = "<details><summary>数据版本与图表口径</summary><pre>" + html.escape(
                json.dumps(item.get("metadata", {}), ensure_ascii=False, indent=2)) + "</pre></details>"
            content = document.replace("</body>", caption + metadata + "</body>").encode('utf-8')
            if existing is not None:
                import hashlib
                from st_agent.figure_cache import preserve_invalid
                digest = hashlib.sha256(content).hexdigest()
                if digest != existing.sha256:
                    # An older random-ID export cannot be reconstructed byte for
                    # byte. Preserve its evidence and publish a distinct repair.
                    preserve_invalid(target)
                    target = directory / (item['id'] + '__recovered__' + digest + '.html')
            from st_agent.figure_cache import atomic_write
            atomic_write(target, content)
            outputs.append((target, "figure." + item["id"], revision_id))
    from st_agent.scientific_evidence import build_evidence_bundle
    evidence = build_evidence_bundle(copilot, task["dataset_id"], task=task)
    evidence_path = directory / "scientific_evidence.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs.append((evidence_path, "evidence.scientific", final.id))
    overview_path = directory / "scientific_summary.md"
    overview_path.write_text(evidence["summary"] + "\n\n" + "\n\n".join(evidence["limitations"]), encoding="utf-8")
    outputs.append((overview_path, "summary.scientific", final.id))
    from st_agent.full_figures import export_full_figures
    archive = export_full_figures(copilot, task['dataset_id'], revision_id=final.id)
    raw_archive = export_full_figures(copilot, task['dataset_id'], revision_id=qc['input_revision_id'])
    full_index = directory / 'full_figures' / 'manifest.json'
    from st_agent.figure_cache import atomic_write, encoded
    atomic_write(full_index, encoded({'final_revision': archive['manifest_path'],
                                     'qc_input_revision': raw_archive['manifest_path'],
                                     'directory': 'artifacts/full_figures'}))
    outputs.append((full_index, 'manifest.full_figures', final.id))
    saved = copilot.list_artifacts(run_id=task["task_id"])
    records = []
    for path, kind, revision_id in outputs:
        relative = path.relative_to(copilot.layout.root).as_posix()
        digest = sha256_file(path)
        record = next((x for x in saved if x.relative_path == relative and x.sha256 == digest), None)
        if record is None:
            record = copilot.register_artifact(path, kind=kind, run_id=task["task_id"],
                metadata={"revision_id": revision_id, "dataset_id": task["dataset_id"]})
        records.append(record.model_dump(mode="json"))
        emit(copilot, "artifact.exported", artifact_id=record.id, kind=kind,
             revision_id=revision_id, sha256=digest, relative_path=relative)
    # Shared full files remain bound to this task through its manifest and
    # checkpoint, without registering another copy under a different run ID.
    known = {record['id'] for record in records}
    for item in archive['files'] + raw_archive['files']:
        record = copilot.artifacts.get(item['artifact_id'])
        if record.id not in known:
            records.append(record.model_dump(mode='json'))
            known.add(record.id)
    return records
