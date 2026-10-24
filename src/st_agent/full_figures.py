"""Full version-bound HTML archive: per-figure reuse, no analysis or revisions."""
import copy
import base64
import html
import hashlib
import io
import json
from pathlib import Path
import re

import plotly
import plotly.graph_objects as go

from st_agent.figure_cache import (atomic_write, encoded, file_lock, fingerprint,
                                   load_checked, preserve_invalid, save_checked)
from st_agent.figure_gallery import revision_gallery
from st_agent.trace import attach, emit
from st_agent.utils import sha256_file


def prepare_figure_images(copilot, item, *, font_size=15):
    """Archive one complete figure and materialize reusable PNG vision inputs.

    The caller must use full_extent=True. No analysis/worker or data revision is
    created. Binary contents never enter trace payloads or the fact bundle.
    """
    from PIL import Image
    import importlib.metadata

    metadata = item['metadata']
    sampling = metadata.get('sampling')
    if sampling and sampling['displayed_n'] != sampling['population_n']:
        raise ValueError('figure.vision_requires_full_population')
    if metadata['statistics'].get('available') is False:
        raise ValueError('figure.vision_data_unavailable')
    directory = copilot.layout.root / 'artifacts' / 'full_figures'
    archive = _export_item(copilot, item, directory, font_size)
    payload_path = copilot.layout.root / archive['json_path']
    if sha256_file(payload_path) != archive['json_sha256']:
        raise ValueError('figure.vision_archive_changed')
    saved = json.loads(payload_path.read_bytes())
    layout = saved['figure']['layout']
    width, height = int(layout['width']), int(layout.get('height', 440))
    # Explicit refusal, not a reduced set of genes/points masquerading as full.
    if width < 1 or height < 1 or width * height > 32_000_000:
        raise ValueError('figure.vision_render_budget_exceeded')
    tile_size, overlap = 1280, 160
    identity = {'schema': 1, 'archive_sha256': archive['json_sha256'],
                'width': width, 'height': height, 'tile_size': tile_size,
                'overlap': overlap, 'plotly': plotly.__version__,
                'kaleido': importlib.metadata.version('kaleido'),
                'pillow': importlib.metadata.version('pillow'),
                'renderer_sha256': sha256_file(Path(__file__))}
    key = fingerprint(identity)
    index = directory / 'png' / (key + '.json')
    with file_lock(directory / 'locks' / ('png-' + key + '.lock')):
        record = load_checked(index, identity)
        valid = record is not None and all(
            (copilot.layout.root / image['relative_path']).resolve().is_relative_to(copilot.layout.root.resolve())
            and (copilot.layout.root / image['relative_path']).is_file()
            and sha256_file(copilot.layout.root / image['relative_path']) == image['sha256']
            for image in record['files'])
        if not valid:
            import kaleido
            xs = list(range(0, width, tile_size - overlap))
            ys = list(range(0, height, tile_size - overlap))
            large = max(width, height) > 1600
            if large and 1 + len(xs) * len(ys) > 64:
                raise ValueError('figure.vision_tile_budget_exceeded')
            rendered_figure = go.Figure(saved['figure'])
            rendered_figure.update_layout(title={'yanchor': 'top', 'yref': 'container'})
            png = kaleido.calc_fig_sync(rendered_figure,
                opts={'format': 'png', 'width': width, 'height': height, 'scale': 1},
                kopts={'timeout': 60, 'mathjax': False})
            with Image.open(io.BytesIO(png)) as rendered:
                rendered.load()
                if rendered.size != (width, height):
                    raise ValueError('figure.vision_dimensions_mismatch')
                files = []

                def store_image(image, suffix, kind, region):
                    path = directory / 'png' / (key + '-' + suffix + '.png')
                    if path.exists():
                        preserve_invalid(path)
                    output = io.BytesIO()
                    image.save(output, format='PNG')
                    atomic_write(path, output.getvalue())
                    files.append({'id': 'image_' + key[:16] + '_' + suffix, 'kind': kind,
                        'region': region, 'width': image.width, 'height': image.height,
                        'relative_path': path.relative_to(copilot.layout.root).as_posix(),
                        'sha256': sha256_file(path), 'send_to_model': kind != 'archive_only'})

                store_image(rendered, 'full', 'archive_only' if large else 'full', [0, 0, width, height])
                if large:
                    overview = rendered.copy()
                    overview.thumbnail((1600, 1600))
                    store_image(overview, 'overview', 'overview', [0, 0, width, height])
                    for row, top in enumerate(ys):
                        for col, left in enumerate(xs):
                            region = [left, top, min(left + tile_size, width), min(top + tile_size, height)]
                            store_image(rendered.crop(region), f'tile-{row}-{col}', 'tile', region)
            record = {'identity': identity, 'source_archive': archive, 'files': files,
                      'complete_coverage': True, 'native_width': width, 'native_height': height}
            save_checked(index, identity, record)
        # Revalidate immediately before encoding, including a reused cache.
        parts = []
        total_bytes = 0
        for image in record['files']:
            if not image['send_to_model']:
                continue
            content = (copilot.layout.root / image['relative_path']).read_bytes()
            if hashlib.sha256(content).hexdigest() != image['sha256']:
                raise ValueError('figure.vision_image_changed')
            total_bytes += len(content)
            if total_bytes > 24 * 1024 * 1024:
                raise ValueError('figure.vision_request_budget_exceeded')
            parts.extend([
                {'type': 'text', 'text': json.dumps({k: image[k] for k in
                    ('id', 'kind', 'region', 'width', 'height', 'sha256')})},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                    base64.b64encode(content).decode('ascii'), 'detail': 'original'}}])
        result = dict(record, manifest_path=index.relative_to(copilot.layout.root).as_posix(),
                      status='reused' if valid else 'generated', sent_image_count=len(parts) // 2)
        emit(copilot, 'figure.vision.prepared', result=result)
        return result, parts


def _slug(value):
    return re.sub(r'[^\w-]+', '-', str(value)).strip('-')[:50] or 'all'


def _valid(root, record):
    for key, digest in (('relative_path', 'sha256'), ('json_path', 'json_sha256')):
        path = (root / record[key]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            return False
        if sha256_file(path) != record[digest]:
            return False
    return True


def _export_item(copilot, item, directory, font_size):
    root = copilot.layout.root
    item = copy.deepcopy(item)
    stats = item['metadata']['statistics']
    if item['id'] == 'markers':
        if stats['displayed_genes'] != stats['available_genes'] or stats['omitted_count']:
            raise ValueError('figure.full_marker_incomplete')
        width = max(960, 120 + 28 * len(stats['displayed_genes']))
    else:
        width = 960
    identity = {'schema': 1, 'figure_fingerprint': item['metadata']['figure_fingerprint'],
                'plot_payload_sha256': fingerprint(item), 'font_size': font_size, 'width': width,
                'format': 'standalone_html', 'plotly': plotly.__version__,
                'exporter_sha256': sha256_file(Path(__file__))}
    key = fingerprint(identity)
    parameters = item['metadata']['figure_parameters']
    label = ('gene-' + _slug(stats['gene']) if item['id'] == 'gene' else
             'section-' + _slug(parameters['section_value']) if parameters.get('section_key') else
             'all-saved-markers' if item['id'] == 'markers' else 'all')
    basename = _slug(item['id']) + '__' + label + '__' + key
    target, payload = directory / (basename + '.html'), directory / (basename + '.json')
    index = directory / 'index' / (key + '.json')
    with file_lock(directory / 'locks' / (key + '.lock')):
        record = load_checked(index, identity)
        state = 'reused' if record is not None and _valid(root, record) else None
        if state is None:
            state = 'rebuilt' if target.exists() or payload.exists() or index.exists() else 'generated'
            if state == 'rebuilt':
                for path in (target, payload, index):
                    preserve_invalid(path)
            item['figure']['layout'].update(width=width, autosize=False, font={'size': font_size})
            document = go.Figure(item['figure']).to_html(include_plotlyjs=True, div_id='figure-' + key,
                                                       config={'responsive': False})
            details = ('<p>' + html.escape(item['caption']) + '</p><details><summary>数据版本、完整参数与范围</summary><pre>'
                       + html.escape(json.dumps(item['metadata'], ensure_ascii=False, indent=2)) + '</pre></details>')
            document = document.replace('</head>', '<style>body{margin:16px;overflow-x:auto}</style></head>')
            atomic_write(target, document.replace('</body>', details + '</body>').encode('utf-8'))
            atomic_write(payload, encoded(item))
            record = {'figure_id': item['id'], 'title': item['title'], 'fingerprint': key,
                      'figure_fingerprint': identity['figure_fingerprint'],
                      'relative_path': target.relative_to(root).as_posix(), 'sha256': sha256_file(target),
                      'json_path': payload.relative_to(root).as_posix(), 'json_sha256': sha256_file(payload),
                      'revision_id': item['metadata']['revision_id'],
                      'dataset_id': item['metadata']['dataset_id'], 'parameters': parameters,
                      'render_parameters': {'font_size': font_size, 'width': width, 'format': 'standalone_html'}}
            save_checked(index, identity, record)
        # Lock also serializes duplicate artifact registration across sessions.
        existing = next((x for x in copilot.list_artifacts() if
                         x.relative_path == record['relative_path']), None)
        if existing is not None and existing.sha256 != record['sha256']:
            raise ValueError('figure.registered_digest_mismatch')
        if existing is None:
            existing = copilot.register_artifact(target, kind='figure.full.' + item['id'],
                metadata={'revision_id': record['revision_id'], 'dataset_id': record['dataset_id'],
                          'fingerprint': key, 'parameters': parameters})
        record = dict(record, artifact_id=existing.id, status=state)
        emit(copilot, 'figure.export.' + state, figure_id=item['id'], fingerprint=key,
             relative_path=record['relative_path'], revision_id=record['revision_id'])
        return record


def export_full_figures(copilot, dataset_id, *, revision_id=None, gene=None, font_size=15):
    if isinstance(font_size, bool) or not isinstance(font_size, int) or not 8 <= font_size <= 32:
        raise ValueError('font_size must be an integer between 8 and 32')
    trace = getattr(copilot, 'trace', None)
    if trace is None:
        trace = attach(copilot)
    trace.emit('figure.export.request', dataset_id=dataset_id, revision_id=revision_id,
               gene=gene, font_size=font_size)
    try:
        result = _export_full_figures(copilot, dataset_id, revision_id=revision_id,
                                      gene=gene, font_size=font_size)
    except Exception as error:
        trace.emit('figure.export.failed', status='failed', error_code=type(error).__name__)
        raise
    trace.emit('figure.export.completed', generated=result['generated'], reused=result['reused'],
               rebuilt=result['rebuilt'], manifest_path=result['manifest_path'])
    result['trace_degraded'] = trace.degraded
    return result


def _export_full_figures(copilot, dataset_id, *, revision_id=None, gene=None, font_size=15):
    """All saved markers, all points, separate spatial sections; HTML only.

    A single slice selection or preview page never narrows the archive. Export
    options affect rendering only, and archives are shared by project sessions.
    """
    if isinstance(font_size, bool) or not isinstance(font_size, int) or not 8 <= font_size <= 32:
        raise ValueError('font_size must be an integer between 8 and 32')
    directory = copilot.layout.root / 'artifacts' / 'full_figures'
    bundle = revision_gallery(copilot, dataset_id, revision_id=revision_id, gene=gene, full_extent=True)
    files, unavailable = [], []
    def export(items):
        for item in items:
            stats = item['metadata']['statistics']
            if stats.get('available') is False:
                unavailable.append({'figure_id': item['id'], 'reason': stats['reason']})
                continue
            files.append(_export_item(copilot, item, directory, font_size))
    export(bundle['figures'])
    sections = bundle['section_options']
    for section in sections['values']:
        if section == sections['selected']:
            continue
        extra = revision_gallery(copilot, dataset_id, revision_id=bundle['identity']['revision_id'],
                                 gene=gene, full_extent=True,
                                 section_key=sections['key'], section_value=section)
        export([item for item in extra['figures'] if item['id'] == 'gene' or
                item['metadata']['section_selection'] is not None])
    identity = {'schema': 1, 'dataset_id': dataset_id, 'revision_id': bundle['identity']['revision_id'],
                'revision_sha256': bundle['identity']['sha256'],
                'files': sorted(item['fingerprint'] for item in files), 'unavailable': unavailable}
    key = fingerprint(identity)
    manifest_path = directory / 'manifests' / (key + '.json')
    manifest = {'identity': identity, 'files': [{k: v for k, v in item.items() if k != 'status'} for item in files],
                'unavailable': unavailable, 'warnings': bundle['warnings'],
                'scope': '全观测；marker为本次保存的全部排名；空间图按切片分别保存，单位与配准不推断。'}
    with file_lock(directory / 'locks' / ('manifest-' + key + '.lock')):
        if load_checked(manifest_path, identity) != manifest:
            save_checked(manifest_path, identity, manifest)
    return {'directory': str(directory.resolve()),
            'manifest_path': manifest_path.relative_to(copilot.layout.root).as_posix(),
            'files': files, 'unavailable': unavailable, 'warnings': bundle['warnings'],
            **{state: sum(x['status'] == state for x in files) for state in ('generated', 'reused', 'rebuilt')}}
