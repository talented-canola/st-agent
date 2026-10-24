"""Single registry for scientific tools actually supported by the local Worker.

Multi-input merge remains a direct kernel until registered-revision resolution
and multi-input approval/provenance exist; source paths are never model arguments.
"""
from __future__ import annotations

from st_agent.advanced_tools import ADVANCED_TOOLS, AdvancedTool, ExpressionDomainArguments
from st_agent.advanced_scientific import detect_expression_domains
from st_agent.contracts import Admission, Effect
from st_agent.scientific_tools import TOOLS

SCIENTIFIC_TOOLS = {
    **TOOLS,
    **{name: ADVANCED_TOOLS[name] for name in (
        "spatial.neighbors", "spatial.morans_i", "spatial.domains",
        "samples.detect", "samples.compare_stages")},
    "spatial.expression_domains": AdvancedTool(
        "spatial.expression_domains", Admission.VERIFIED, Effect.DERIVED_REVISION,
        ExpressionDomainArguments, detect_expression_domains,
        "expression-weighted section-local graph, independently versioned"),
}

TOOL_VERSIONS = {
    "spatial.neighbors": "spatial-neighbors/2.0",
    "spatial.morans_i": "spatial-moran/2.0",
    "spatial.domains": "coordinate-partition/2.0",
    "spatial.expression_domains": "expression-spatial-leiden/1.0",
    "samples.compare_stages": "sample-comparison/2.0",
    "samples.detect": "sample-metadata/1.0",
}

# Compact capability metadata for tools that are actually registered. This is
# not a scientific measurement, an execution permission or a claim that a
# result exists. Input flags are checkable against an already-inspected
# DatasetState-shaped mapping; absent keys stay "unchecked" rather than being
# inferred from a filename or an unrelated artifact.
_TOOL_CAPABILITY_SPECS = {
    "dataset.inspect": (
        "检查当前 revision 的基本维度、切片字段、计数来源和已有处理记录",
        "细胞/基因计数、切片字段数、计数来源、预处理记录", (), ()),
    "qc.compute_metrics": (
        "从当前计数来源计算观测级 QC 指标并按已记录分组汇总",
        "总计数、检测基因数、线粒体比例及其分组分布", ("计数矩阵",), ()),
    "qc.apply_filter": (
        "从当前计数来源按明确阈值过滤观测并生成派生 revision",
        "过滤前后观测数、各规则排除数、派生 revision", ("计数矩阵", "明确过滤阈值"), ()),
    "preprocess.standard": (
        "执行标准化、log1p、高变基因选择和PCA",
        "预处理派生 revision、PCA 坐标与高变基因记录", ("表达矩阵",), ()),
    "cluster.leiden_umap": (
        "在已有 PCA 上构建邻居图，运行 Leiden 聚类和 UMAP",
        "表达簇标签、UMAP、簇大小", ("PCA",), ("has_pca",)),
    "markers.rank": (
        "对每个表达簇相对其余观测做 marker 排序",
        "候选 marker 的 log2 效应量、校正 P 值、簇内/簇外检出比例和检出差",
        ("Leiden 簇标签", "log1p 归一化表达矩阵"), ("has_leiden", "has_log1p")),
    "plot.spatial_clusters": (
        "从已保存表达簇生成切片空间坐标图",
        "空间散点图引用与图中簇统计", ("空间坐标", "Leiden 聚类标签"),
        ("has_spatial_coordinates", "has_leiden")),
    "plot.spatial_gene": (
        "从已保存表达矩阵生成指定基因的空间表达图",
        "基因空间表达图与图中表达统计", ("空间坐标", "目标基因位于表达矩阵"),
        ("has_spatial_coordinates",)),
    "spatial.neighbors": (
        "按空间坐标构建切片内空间邻接图",
        "spatial_connectivities 空间图", ("有限空间坐标",),
        ("has_spatial_coordinates",)),
    "spatial.morans_i": (
        "计算指定基因的描述性 Moran I",
        "Moran I 系数；不提供显著性检验", ("空间邻接图", "目标基因位于表达矩阵"),
        ("has_spatial_neighbors",)),
    "spatial.domains": (
        "仅用空间坐标执行 KMeans 坐标分区",
        "坐标分区标签；不含表达推断", ("有限空间坐标", "分区数与 seed"),
        ("has_spatial_coordinates",)),
    "spatial.expression_domains": (
        "结合已记录预处理的 PCA 表达与切片内空间邻域生成表达-空间联合域",
        "表达-空间联合域标签", ("PCA", "预处理记录", "空间坐标"),
        ("has_pca", "has_normalization", "has_spatial_coordinates")),
    "samples.detect": (
        "从当前 revision 的文件名识别候选样本/阶段/重复，不读取 obs 分组真相，也不自动确认比较分组",
        "样本标识、候选阶段、候选重复与需研究者确认的候选状态", (), ()),
    "samples.compare_stages": (
        "按已确认分组列做描述性样本/切片比较与组成比较",
        "分组 QC/表达摘要、组成表", ("已确认分组列", "可选对齐的组成标签", "基因表达"),
        ("sample_metadata_confirmed",)),
}

_CAPABILITY_BOUNDARY = (
    "本目录只列出当前项目 SCIENTIFIC_TOOLS 实际注册的工具。注册不代表已在当前数据上"
    "执行、输入适配或获得审批；result_status=available 只表示有已保存结果引用。"
    "目录中没有的方法在本项目不可执行，可以作为未来方案讨论所需方法与数据，但不能说"
    "已经执行或已有专项结果。能力元数据不是样本事实或科学测量。")

_UNREGISTERED_POLICY = (
    "未出现在本目录的方法未被本项目注册；不得伪称其已执行或已有结果。"
    "仍可把它作为未来所需的未执行方法或数据前置提出。")


def _capability_input_status(flags, data_state, requirements=()):
    if not flags:
        # Requirements exist but no checked boolean state; keep them explicitly
        # unchecked instead of claiming the tool has no prerequisites.
        if requirements:
            return "unchecked", list(requirements)
        return "not_required", []
    if data_state is None:
        return "unchecked", []
    missing = [flag for flag in flags if data_state.get(flag) is False]
    if missing:
        return "checked_missing", missing
    unknown = [flag for flag in flags if flag not in data_state]
    return ("unchecked" if unknown else "checked_satisfied"), unknown


def _result_status(refs):
    statuses = [ref["status"] for ref in refs]
    if "available" in statuses:
        return "available"
    if statuses and all(status == "failed" for status in statuses):
        return "failed"
    if statuses:
        return "not_completed"
    return "not_executed"


def build_capability_context(*, observations=(), data_state=None,
                             selection_summary=None, availability=()):
    """Describe registered tools, checked prerequisites and saved results.

    ``observations`` are existing tool-observation/execution records. Only
    registered tool names are retained. No result values are copied here:
    result refs identify the call/revision and the recorded field names, while
    values continue through the existing fact/tool-observation path.
    """
    state = data_state if isinstance(data_state, dict) else None
    refs, seen = [], set()
    for observation in observations or ():
        if not isinstance(observation, dict):
            continue
        name = observation.get("tool_name")
        if name not in SCIENTIFIC_TOOLS:
            continue
        raw_status = str(observation.get("status", "unknown"))
        status = ("available" if raw_status in {"success", "warning"} else
                  "failed" if raw_status in {"failed", "cancelled"} else "not_completed")
        fields = observation.get("metric_keys")
        if fields is None:
            fields = []
            for metric in observation.get("metrics", []) or ():
                path = metric.get("path") if isinstance(metric, dict) else None
                if path and path[0] not in fields:
                    fields.append(path[0])
        ref = {
            "tool_name": name,
            "call_id": observation.get("call_id"),
            "revision_id": observation.get("output_revision_id") or observation.get("input_revision_id"),
            "scope": observation.get("scope", "unknown"),
            "status": status,
            "recorded_fields": [field for field in fields[:16] if isinstance(field, str)],
        }
        identity = (name, ref["call_id"], ref["revision_id"], ref["status"])
        if identity not in seen:
            seen.add(identity)
            refs.append(ref)
    tools = []
    for name in sorted(SCIENTIFIC_TOOLS):
        purpose, result_evidence, requirements, flags = _TOOL_CAPABILITY_SPECS[name]
        input_status, missing = _capability_input_status(flags, state, requirements)
        tool_refs = [ref for ref in refs if ref["tool_name"] == name]
        tools.append({
            "tool_name": name,
            "registered": True,
            "purpose": purpose,
            "result_evidence": result_evidence,
            "argument_fields": list(SCIENTIFIC_TOOLS[name].arguments.model_fields),
            "input_requirements": list(requirements),
            "input_status": input_status,
            "missing_or_unchecked_inputs": missing,
            "result_status": _result_status(tool_refs),
            "result_refs": tool_refs,
        })
    if not selection_summary:
        supply = {"status": "no_explicit_saved_evidence_selection"}
    else:
        selection_summary = selection_summary if isinstance(selection_summary, dict) else {}
        supply = {
            "kind": selection_summary.get("kind"),
            "status": selection_summary.get("status"),
            "saved_records": selection_summary.get("saved_records"),
            "matching_records": selection_summary.get("matching_records"),
            "supplied_records": selection_summary.get("supplied_records"),
            "omitted_records": selection_summary.get("omitted_records"),
            "requested_fields": list(selection_summary.get("requested_fields", []))[:32],
            "coverage": list(selection_summary.get("coverage", []))[:16],
            "missing_entities": list(selection_summary.get("missing_entities", []))[:32],
            "unmatched_requested_entities": list(selection_summary.get("missing_entities", []))[:32],
            "automatic_execution": False,
        }
    return {
        "schema": "capability-context/1",
        "catalog_source": "SCIENTIFIC_TOOLS",
        "registered_tool_count": len(tools),
        "tools": tools,
        "saved_result_references": refs,
        "selection_supply": supply,
        "availability": list(availability)[:16],
        "data_state": {key: value for key, value in (state or {}).items()
                       if isinstance(value, bool)},
        "execution_boundary": _CAPABILITY_BOUNDARY,
        "unregistered_method_policy": _UNREGISTERED_POLICY,
        "authority": "capability_metadata_only_not_execution_permission_or_measurement",
    }


def execute_registered_tool(name, source, *, output=None, arguments=None):
    tool = SCIENTIFIC_TOOLS[name]
    # JSON roundtrip accepts JSON arrays for tuple fields while retaining strict
    # scalar types; the same validated values are persisted in provenance.
    import json
    values = tool.arguments.model_validate_json(json.dumps(arguments or {})).model_dump()
    if tool.effect is Effect.DERIVED_REVISION:
        if output is None:
            raise ValueError(f"{name} requires an output revision path")
        return tool.implementation(source, output, **values)
    return tool.implementation(source, **values)
