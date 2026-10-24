"""Revision-bound scientific figures with explicit selection and provenance."""

import colorsys
import hashlib
import json
import re
from contextlib import ExitStack
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import sparse

from st_agent.analysis_context import analysis_context
from st_agent.dataset_state import DatasetStateInspector
from st_agent.scientific import _qc_vectors
from st_agent.utils import sha256_file
from st_agent.figure_cache import file_lock, fingerprint, load_checked, save_checked


def _natural_key(value):
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold())
                 for part in re.split(r"(\d+)", str(value)))


def _colors(groups):
    palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3",
               "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
    result = {}
    for index, group in enumerate(groups):
        if index < len(palette):
            result[group] = palette[index]
        else:
            rgb = colorsys.hsv_to_rgb((index * 0.61803398875) % 1, 0.7, 0.75)
            result[group] = "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb)
    return result


def _sample(data, population, labels, limit):
    """ID-stable proportional stratification retaining every observed cluster.

    Guaranteeing one point per cluster oversamples very small clusters. Counts
    are disclosed; sampled points never estimate population proportions.
    """
    population = np.asarray(population, dtype=int)
    groups = sorted(set(labels[population]), key=_natural_key)
    counts = {group: int(np.sum(labels[population] == group)) for group in groups}
    if len(groups) > limit:
        raise ValueError("max_points must allow at least one point per cluster")
    ids = np.asarray(data.obs_names.astype(str))
    if len(population) <= limit:
        chosen = population
    else:
        capacity = np.array([counts[group] - 1 for group in groups], dtype=int)
        quotas = (limit - len(groups)) * capacity / capacity.sum()
        additions = np.floor(quotas).astype(int)
        remaining = limit - len(groups) - int(additions.sum())
        ranking = sorted(range(len(groups)), key=lambda i: (-(quotas[i] - additions[i]), _natural_key(groups[i])))
        for position in ranking[:remaining]:
            additions[position] += 1
        chosen_list = []
        for group, extra in zip(groups, additions, strict=True):
            members = population[labels[population] == group]
            ordered = sorted(members, key=lambda i: (hashlib.sha256(ids[i].encode()).hexdigest(), ids[i]))
            chosen_list.extend(ordered[:1 + int(extra)])
        chosen = np.asarray(chosen_list, dtype=int)
    chosen = np.asarray(sorted(chosen, key=lambda i: _natural_key(ids[i])), dtype=int)
    return chosen, {
        "method": "id_stable_proportional_stratified" if len(chosen) < len(population) else "all",
        "max_points": limit,
        "population_n": int(len(population)),
        "displayed_n": int(len(chosen)),
        "population_counts": counts,
        "sampled_counts": {group: int(np.sum(labels[chosen] == group)) for group in groups},
        "rare_clusters_preserved": True,
        "proportions_from_sample_are_not_population_estimates": len(chosen) < len(population),
    }


def _section_selection(data, context, section_key, section_value):
    key = section_key or context["sections"]["key"]
    if key is None:
        if section_value is not None:
            raise ValueError("Cannot select section without a section annotation")
        return np.arange(data.n_obs), {"key": None, "value": None, "mode": "no_section_annotation", "n_obs": int(data.n_obs)}, []
    if key not in data.obs:
        raise ValueError("指定切片字段不存在: " + str(key))
    valid = ~pd.isna(data.obs[key]).to_numpy()
    values = data.obs[key].astype(str).to_numpy()
    options = sorted(set(values[valid]), key=_natural_key)
    if not options:
        raise ValueError("切片字段没有可选择的有效值")
    selected = str(section_value) if section_value is not None else options[0]
    if selected not in options:
        raise ValueError("指定切片值不存在: " + selected)
    indices = np.flatnonzero(valid & (values == selected))
    return indices, {"key": str(key), "value": selected, "mode": "single_section", "n_obs": int(len(indices))}, options


def _expression_semantics(data):
    representation = data.uns.get("st_agent_preprocess", {}).get("representation")
    # An arbitrary uns['log1p'] key can survive unrelated edits; only the
    # explicit representation recorded by our transformation describes X.
    if representation == "log1p_normalized":
        return "log1p_normalized_expression"
    return "unknown_X_scale"


def revision_gallery(copilot, dataset_id, **options):
    """One guarded entry for previews and full exports; no scientific execution."""
    for name, default in (('max_points', 20000), ('markers_per_group', 3), ('max_marker_genes', 30),
                          ('marker_page_size', None), ('marker_page_index', 0)):
        value = options.get(name, default)
        if name == 'marker_page_size' and value is None:
            continue
        lower = 0 if name == 'marker_page_index' else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(name + ' has invalid integer value')
    if options.get('marker_display_mode', 'overview') not in {'overview', 'complete'}:
        raise ValueError('marker_display_mode must be overview or complete')
    if not isinstance(options.get('full_extent', False), bool):
        raise ValueError('full_extent must be boolean')
    key = fingerprint({'dataset_id': dataset_id, 'options': options})
    directory = copilot.layout.root / 'artifacts' / 'figures'
    with file_lock(directory / 'locks' / (key + '.lock')), ExitStack() as locks:
        return _revision_gallery(copilot, dataset_id, _locks=locks, **options)


def _revision_gallery(copilot, dataset_id, *, revision_id=None, gene=None, section_key=None,
                     section_value=None, coordinate_key=None, max_points=20000,
                     markers_per_group=3, max_marker_genes=30, marker_display_mode="overview",
                     marker_page_index=0, marker_page_size=None, full_extent=False, _locks=None):
    """Read only. UMAP uses all sections; spatial plots select a single section.

    No coordinate units, registration, axis inversion or cell types are inferred.
    Rendering options are cache-bound and exported with each figure's metadata.
    """
    if not isinstance(full_extent, bool):
        raise ValueError('full_extent must be boolean')
    for name, value in (("max_points", max_points), ("markers_per_group", markers_per_group), ("max_marker_genes", max_marker_genes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(name + " must be a positive integer")
    if marker_display_mode not in {"overview", "complete"}:
        raise ValueError("marker_display_mode must be overview or complete")
    if isinstance(marker_page_index, bool) or not isinstance(marker_page_index, int) or marker_page_index < 0:
        raise ValueError("marker_page_index must be a nonnegative integer")
    if marker_page_size is None:
        marker_page_size = max_marker_genes  # Legacy option is now a page size, not truncation.
    if isinstance(marker_page_size, bool) or not isinstance(marker_page_size, int) or marker_page_size < 1:
        raise ValueError("marker_page_size must be a positive integer")
    if full_extent:
        marker_display_mode, marker_page_index = 'complete', 0
        max_points, marker_page_size, max_marker_genes, markers_per_group = 20000, 30, 30, 3
    if marker_display_mode == 'complete':
        markers_per_group = 3  # This parameter is not used in complete mode.
    dataset = copilot.get_dataset(dataset_id)
    revision = copilot.revisions.get(revision_id or dataset.active_revision_id)
    if revision.dataset_id != dataset_id or not revision.relative_path:
        raise ValueError("Figure revision does not belong to selected dataset")
    source = DatasetStateInspector(copilot)._project_path(revision.relative_path)
    digest = sha256_file(source)
    if digest != revision.sha256:
        raise ValueError("Revision content changed; refusing stale figures")
    generator = fingerprint({'gallery': sha256_file(Path(__file__)),
                             'cache': sha256_file(Path(__file__).with_name('figure_cache.py'))})
    identity = {"schema": 7, "generator": generator, "dataset_id": dataset_id, "revision_id": revision.id, "sha256": digest,
                "gene": gene, "section_key": section_key, "section_value": None if section_value is None else str(section_value),
                "coordinate_key": coordinate_key, "max_points": max_points,
                "markers_per_group": markers_per_group, "max_marker_genes": max_marker_genes,
                "marker_display_mode": marker_display_mode, "marker_page_index": marker_page_index,
                "marker_page_size": marker_page_size, "full_extent": full_extent}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = copilot.layout.root / "artifacts" / "figures"
    target = directory / f"{key}.json"
    saved = load_checked(target, identity)
    if saved is not None:
        return saved
    data = ad.read_h5ad(source)
    if data.n_obs < 1 or not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("Figures require observations and unique observation/gene IDs")
    context = analysis_context(data)
    if full_extent:
        max_points = data.n_obs
        marker_page_size = max(1, data.n_vars)
    spatial_population, selection, section_options = _section_selection(data, context, section_key, section_value)
    spatial_key = coordinate_key or context["coordinates"]["key"]
    if spatial_key is not None and spatial_key not in data.obsm:
        raise ValueError("指定空间坐标字段不存在: " + str(spatial_key))
    if spatial_key is not None:
        coordinates = np.asarray(data.obsm[spatial_key])
        if coordinates.ndim != 2 or coordinates.shape[0] != data.n_obs or coordinates.shape[1] < 2:
            raise ValueError("Spatial coordinates must align with observations and contain two axes")
        if not np.isfinite(coordinates).all():
            raise ValueError("Spatial coordinates must be finite（有限数值）")
    ids = np.asarray(data.obs_names.astype(str))
    labels = data.obs["leiden"].astype(str).to_numpy() if "leiden" in data.obs else np.repeat("all", data.n_obs)
    groups = sorted(set(labels), key=_natural_key)
    colors = _colors(groups)
    expression_semantics = _expression_semantics(data)
    warnings = []
    figures, requested_order, pending = [], [], {}
    base = {k: identity[k] for k in ('dataset_id', 'revision_id', 'sha256', 'generator')}

    def want(name):
        options = {}
        if name == 'X_umap' or name == spatial_key or name == 'gene':
            options['max_points'] = max_points
        if name == spatial_key or name == 'gene':
            options.update(coordinate_key=spatial_key, section_key=selection['key'],
                           section_value=selection['value'])
        if name == 'gene':
            options['gene'] = gene
        if name == 'markers':
            options.update(marker_display_mode=marker_display_mode,
                           marker_page_index=marker_page_index, marker_page_size=marker_page_size)
            if marker_display_mode == 'overview':
                options['markers_per_group'] = markers_per_group
            # Legacy page limit appears in metadata; retain its provenance.
            options['max_marker_genes'] = max_marker_genes
        item_identity = dict(base, figure_id=name, parameters=options)
        item_key = fingerprint(item_identity)
        path = directory / 'items' / (item_key + '.json')
        _locks.enter_context(file_lock(directory / 'locks' / ('item-' + item_key + '.lock')))
        requested_order.append(name)
        cached = load_checked(path, item_identity)
        if cached is not None:
            figures.append(cached)
            return False
        pending[name] = (path, item_identity)
        return True

    qc_names = ('n_genes_by_counts', 'total_counts', 'pct_counts_mt')
    needed_qc = [name for name in qc_names if want(name)]
    try:
        if needed_qc:
            vectors, count_source, mt_gene_count = _qc_vectors(data)
        else:
            vectors, mt_gene_count = None, 0
            count_source = figures[0]['metadata']['count_source']
    except ValueError:
        # Refuse unsupported QC, not independent valid embeddings. This does
        # not permit scientific transformations to use unverified counts.
        vectors, count_source, mt_gene_count = None, None, 0
        warnings.append({"code": "count_evidence_unavailable",
                         "message": "计数来源无法核验，QC图不可评估；有效降维图仍可查看。"})

    def add(name, title, figure, caption, **details):
        metadata = {"schema_version": 1, "dataset_id": dataset_id, "revision_id": revision.id,
                    "revision_sha256": digest, "total_obs": int(data.n_obs), "obs_ids": [],
                    "coordinate_source": None, "coordinate_units": "unknown",
                    "section_selection": None, "count_source": count_source,
                    "expression_source": "X", "expression_semantics": expression_semantics,
                    "label_colors": colors, "sampling": None, "statistics": {}}
        metadata.update(details)
        metadata['figure_fingerprint'] = fingerprint(pending[name][1])
        metadata['figure_parameters'] = pending[name][1]['parameters']
        if name == "markers":
            # Marker geometry is supplied by the paginated plot below.
            figure.update_layout(title={"text": title, "x": 0, "xanchor": "left", "y": 0.99,
                                        "font": {"size": 16}}, template="plotly_white", meta=metadata)
        else:
            figure.update_layout(title=title, template="plotly_white", height=440,
                                 margin={"l": 35, "r": 20, "t": 55, "b": 40}, meta=metadata)
        item = {"id": name, "title": title, "figure": figure.to_plotly_json(), "caption": caption, "metadata": metadata}
        path, item_identity = pending[name]
        save_checked(path, item_identity, item)
        # Consistent JSON representation on both initial construction and cache hits.
        figures.append(load_checked(path, item_identity))

    for key_name, title in (("n_genes_by_counts", "检测基因数分布"), ("total_counts", "总计数分布"), ("pct_counts_mt", "线粒体比例分布（%）")):
        if key_name not in needed_qc:
            continue
        if vectors is None:
            add(key_name, title, go.Figure(), "计数证据不可核验，该QC指标不可评估；不能用处理后的表达值代替原始计数。",
                statistics={"available": False, "scope": "all_observations", "reason": "count_evidence_unavailable"})
            continue
        if key_name == "pct_counts_mt" and (mt_gene_count == 0 or vectors[key_name] is None):
            add(key_name, title, go.Figure(), "线粒体比例不可评估：当前没有可识别的线粒体基因注释；不能把未知比例当作0%或据此判断质量。",
                statistics={"available": False, "scope": "all_observations", "reason": "mitochondrial_annotation_unavailable"})
            continue
        values = np.asarray(vectors[key_name], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("QC figures require finite values")
        counts, edges = np.histogram(values, bins=40)
        figure = go.Figure(go.Bar(x=((edges[1:] + edges[:-1]) / 2).tolist(), y=counts.tolist(), width=np.diff(edges).tolist()))
        figure.update_xaxes(title=title)
        figure.update_yaxes(title="观测数量")
        add(key_name, title, figure, f"基于全样本{len(values)}个观测，计数来源{count_source}；中位数{float(np.median(values)):.3g}。分布不能单独证明质量或决定过滤阈值。",
            statistics={"available": True, "scope": "all_observations", "n": int(len(values)), "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())})
    if "pca" in data.uns and "variance_ratio" in data.uns["pca"] and want('pca'):
        ratio = np.asarray(data.uns["pca"]["variance_ratio"], dtype=float)
        if not np.isfinite(ratio).all():
            raise ValueError("PCA variance ratios must be finite")
        add("pca", "主成分解释方差", go.Figure(go.Scatter(x=list(range(1, len(ratio) + 1)), y=ratio.tolist(), mode="lines+markers")),
            f"共{len(ratio)}个主成分，累计解释比例{ratio.sum():.1%}；拐点仅供选择维度参考。",
            statistics={"variance_ratio": ratio.tolist(), "cumulative_variance": float(ratio.sum()), "scope": "all_observations"})
    if {"means", "dispersions_norm", "highly_variable"} <= set(data.var) and want('hvg'):
        figure = go.Figure()
        for chosen, label in ((False, "其他基因"), (True, "高变基因")):
            rows = data.var[data.var["highly_variable"] == chosen]
            rows = rows[np.isfinite(rows["means"]) & np.isfinite(rows["dispersions_norm"])]
            figure.add_trace(go.Scattergl(x=rows["means"].tolist(), y=rows["dispersions_norm"].tolist(), mode="markers", name=label, marker={"size": 4}, customdata=[[str(oid)] for oid in rows.index]))
        figure.update_xaxes(title="平均表达")
        figure.update_yaxes(title="标准化离散度")
        add("hvg", "高变基因", figure, "高变基因用于降维，不等同于marker或差异表达证据；不可计算的离散度不绘制。")

    def embedding(key_name, title, population, section):
        coords = np.asarray(data.obsm[key_name])
        if coords.ndim != 2 or coords.shape[1] < 2 or not np.isfinite(coords).all():
            raise ValueError("Embedding coordinates must have two finite axes")
        indices, sampling = _sample(data, population, labels, max_points)
        figure = go.Figure()
        for group in groups:
            selected = indices[labels[indices] == group]
            if not len(selected):
                continue
            figure.add_trace(go.Scattergl(x=coords[selected, 0].tolist(), y=coords[selected, 1].tolist(), mode="markers", name=f"簇{group}",
                marker={"size": 4, "opacity": 0.7, "color": colors[group]},
                customdata=[[str(oid), group] for oid in ids[selected]],
                hovertemplate="观测=%{customdata[0]}<br>簇=%{customdata[1]}<br>x=%{x}<br>y=%{y}<extra></extra>"))
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
        figure.update_xaxes(title="UMAP1" if key_name == "X_umap" else "坐标第1列（单位未推断）")
        figure.update_yaxes(title="UMAP2" if key_name == "X_umap" else "坐标第2列（方向未推断）")
        scope = f"全样本{data.n_obs}个观测；"
        if section is not None and section["key"]:
            scope += f"仅显示切片{section['value']}（{section['key']}），本切片{len(population)}个观测；"
        scope += f"绘制{len(indices)}/{len(population)}个观测。"
        caption = scope + ("降维位置不是组织坐标；" if key_name == "X_umap" else f"位置来自obsm/{key_name}；坐标单位、方向与跨切片配准不作推断；")
        caption += "簇标签不是细胞类型，需要结合marker验证。" + ("抽样保留稀有簇，图中点数比例不是总体簇比例。" if sampling["method"] != "all" else "")
        add(key_name, title, figure, caption, obs_ids=ids[indices].tolist(), coordinate_source="obsm/" + key_name,
            coordinate_dimensions=int(coords.shape[1]), displayed_coordinate_columns=[0, 1], axis_inverted=False,
            coordinate_system="embedding" if key_name == "X_umap" else context["coordinates"]["coordinate_system"],
            section_selection=section, sampling=sampling,
            statistics={"scope": "all_observations" if section is None else "selected_section", "cluster_sizes": sampling["population_counts"]})

    if "leiden" in data.obs:
        if "X_umap" in data.obsm and want('X_umap'):
            embedding("X_umap", "UMAP聚类", np.arange(data.n_obs), None)
        if spatial_key is not None and want(spatial_key):
            embedding(spatial_key, "空间聚类", spatial_population, selection)
    if gene and want('gene'):
        if gene not in data.var_names or spatial_key is None:
            raise ValueError("基因不存在或当前数据没有支持的空间坐标")
        indices, sampling = _sample(data, spatial_population, labels, max_points)
        expression = data[indices, gene].X
        values = expression.toarray().ravel() if sparse.issparse(expression) else np.asarray(expression).ravel()
        if not np.isfinite(values).all():
            raise ValueError("Gene expression must be finite")
        coords = np.asarray(data.obsm[spatial_key])[indices, :2]
        figure = go.Figure(go.Scattergl(x=coords[:, 0].tolist(), y=coords[:, 1].tolist(), mode="markers",
            marker={"color": values.tolist(), "colorscale": "Viridis", "showscale": True, "size": 4, "colorbar": {"title": expression_semantics}},
            customdata=[[str(oid), float(value)] for oid, value in zip(ids[indices], values, strict=True)],
            hovertemplate="观测=%{customdata[0]}<br>X表达=%{customdata[1]}<extra></extra>"))
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
        caption = f"全样本{data.n_obs}个观测；" + (f"切片{selection['value']}共{len(spatial_population)}个观测；" if selection["key"] else "没有切片注释；")
        caption += f"显示{len(indices)}个观测；颜色为当前版本X（{expression_semantics}），不默认等同于原始计数。"
        add("gene", f"{gene} 空间表达", figure, caption, obs_ids=ids[indices].tolist(), coordinate_source="obsm/" + spatial_key,
            coordinate_dimensions=int(np.asarray(data.obsm[spatial_key]).shape[1]), displayed_coordinate_columns=[0, 1], axis_inverted=False,
            coordinate_system=context["coordinates"]["coordinate_system"],
            section_selection=selection, sampling=sampling,
            statistics={"gene": gene, "scope": "displayed_observations", "n": int(len(values)), "mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())})
    if "rank_genes_groups" in data.uns and "leiden" in data.obs and want('markers'):
        ranked = data.uns["rank_genes_groups"]["names"]
        selected_genes, available_genes, available_by_group = [], [], {}
        for group in sorted(ranked.dtype.names, key=_natural_key):
            valid = [str(item) for item in ranked[group] if str(item) in data.var_names]
            available_by_group[group] = len(valid)
            for name in valid:
                if name not in available_genes:
                    available_genes.append(name)
            chosen = valid if marker_display_mode == "complete" else valid[:markers_per_group]
            for item in chosen:
                name = str(item)
                if name not in selected_genes:
                    selected_genes.append(name)
        page_count = max(1, (len(selected_genes) + marker_page_size - 1) // marker_page_size)
        if marker_page_index >= page_count:
            raise ValueError("marker_page_index is outside available pages")
        start = marker_page_index * marker_page_size
        genes = selected_genes[start:start + marker_page_size]
        # A common expression scale prevents identical colors on different
        # pages from representing different values. Bounds include all saved
        # candidates, so overview/complete mode remain comparable as well.
        all_means = []
        for group in groups:
            values = np.asarray(data[labels == group, available_genes].X.mean(axis=0)).ravel()
            if not np.isfinite(values).all():
                raise ValueError("Marker mean expression must be finite")
            all_means.extend(values.tolist())
        color_min, color_max = (float(min(all_means)), float(max(all_means))) if all_means else (0.0, 0.0)
        xs, ys, means, fractions, custom, points = [], [], [], [], [], []
        for group in groups:
            matrix = data[labels == group, genes].X
            average = np.asarray(matrix.mean(axis=0)).ravel()
            fraction = np.asarray((matrix > 0).mean(axis=0)).ravel()
            if not np.isfinite(average).all():
                raise ValueError("Marker mean expression must be finite")
            for name, mean, frac in zip(genes, average, fraction, strict=True):
                xs.append(name)
                ys.append(group)
                means.append(float(mean))
                fractions.append(float(frac))
                custom.append([float(frac), float(mean), int(np.sum(labels == group))])
                points.append({"gene": name, "cluster": group, "mean": float(mean), "positive_fraction": float(frac)})
        size_ref = 2.0 / (24.0 ** 2)
        figure = go.Figure(go.Scatter(x=xs, y=ys, mode="markers", showlegend=False,
            marker={"color": means, "size": fractions, "colorscale": "Viridis", "showscale": True,
                    "cmin": color_min, "cmax": color_max,
                    "sizemode": "area", "sizeref": size_ref, "sizemin": 0,
                    "colorbar": {"title": {"text": "平均X表达", "side": "top"},
                                 "orientation": "h", "x": 0, "xanchor": "left",
                                 "y": 1.02, "yanchor": "bottom", "len": 0.95,
                                 "thickness": 10, "tickfont": {"size": 10}}},
            customdata=custom, hovertemplate="基因=%{x}<br>簇=%{y}<br>平均X表达=%{customdata[1]:.3g}<br>表达>0比例=%{customdata[0]:.1%}<br>簇观测数=%{customdata[2]}<extra></extra>"))
        for fraction in (0.25, 0.5, 1.0):
            figure.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"{fraction:.0%}",
                marker={"color": "#888888", "size": [fraction], "sizemode": "area", "sizeref": size_ref, "sizemin": 0}, hoverinfo="skip"))
        figure.update_xaxes(title="候选marker基因", categoryorder="array", categoryarray=genes,
                            tickmode="array", tickvals=genes, ticktext=genes, tickangle=-90,
                            tickfont={"size": 11}, automargin=True)
        figure.update_yaxes(title="表达簇", categoryorder="array", categoryarray=groups,
                            tickmode="array", tickvals=groups, ticktext=groups, automargin=True)
        plot_height = max(230, len(groups) * 30 + 30)
        label_height = max(80, min(350, max((len(g) for g in genes), default=1) * 6))
        bottom = label_height + 120
        figure.update_layout(legend={"orientation": "h", "x": 0, "xanchor": "left",
                                    "y": -(label_height + 65) / plot_height, "yanchor": "top",
                                    "title": {"text": "X>0比例"}, "font": {"size": 10}},
                             height=plot_height + 110 + bottom,
                             margin={"l": 55, "r": 15, "t": 110, "b": bottom})
        selection_rule = ("本次保存的全部marker排名（不是全部表达基因）" if marker_display_mode == "complete"
                          else f"各簇前{markers_per_group}个已保存marker的去重集合")
        add("markers", "Marker表达点图", figure,
            f"{selection_rule}；选择{len(selected_genes)}/{len(available_genes)}个不同基因，"
            f"当前第{marker_page_index + 1}/{page_count}页显示{len(genes)}个（第{start + 1 if genes else 0}–{start + len(genes)}个），"
            "其余所选基因可翻页查看，不丢弃。统计覆盖全样本，非空间切片筛选结果。"
            f"颜色为平均X（{expression_semantics}），点面积正比于X>0比例，0比例不画点。描述性汇总，不是显著性或细胞类型判定。",
            statistics={"scope": "all_observations", "genes": genes, "markers_per_group": markers_per_group,
                        "max_marker_genes": max_marker_genes, "points": points,
                        "display_mode": marker_display_mode, "available_gene_count": len(available_genes),
                        "available_genes": available_genes, "selected_gene_count": len(selected_genes),
                        "selected_genes": selected_genes, "displayed_genes": genes,
                        "page_index": marker_page_index, "page_size": marker_page_size, "page_count": page_count,
                        "omitted_count": len(available_genes) - len(selected_genes),
                        "color_range": [color_min, color_max], "color_range_scope": "all_saved_marker_genes",
                        "selection_rule": selection_rule, "available_markers_per_group": available_by_group})
    figures.sort(key=lambda item: requested_order.index(item['id']))
    if any(x['metadata']['statistics'].get('reason') == 'count_evidence_unavailable' for x in figures):
        warnings = [{"code": "count_evidence_unavailable", "message": "计数来源无法核验，QC图不可评估；有效降维图仍可查看。"}]
    bundle = {"identity": identity, "figures": figures, "analysis_context": context,
              "warnings": warnings,
              "section_options": {"key": selection["key"], "values": section_options, "selected": selection["value"]},
              "coordinate_options": context["coordinates"]}
    save_checked(target, identity, bundle)
    return load_checked(target, identity)
