"""Local rendering for the structured spatial tool contracts, without a model."""
from __future__ import annotations

import re


_TOOLS = {"spatial.neighbors", "spatial.morans_i", "spatial.domains",
          "spatial.expression_domains", "samples.compare_stages"}
_LABELS = {"observations": "观测数", "edges": "邻接边数", "n_neighbors": "邻居数",
    "domains": "分区数", "method": "方法", "groupby": "切片分组字段",
    "expression_source": "表达来源", "representation": "表达表示",
    "connectivity_key": "空间权重图", "stage_column": "比较分组字段"}
_ROW_LABELS = {"morans_i": "Moran's I", "observations": "观测数", "samples": "样本数",
    "replicates": "记录的重复数", "count": "观测数", "fraction": "组成比例",
    "mean_expression": "平均表达", "fraction_expressing": "检出比例",
    "qc.mean_total_counts": "平均计数", "qc.mean_detected_genes": "平均检测基因数",
    "qc.mean_mito_pct": "平均线粒体比例"}
_ROW = re.compile(r"^(genes|groups|composition|expression)\[(\d+)\]\.(.+)$")


def render_structured_receipt(records):
    tools = list(dict.fromkeys(f.get("source_tool") for f in records if f.get("source_tool") in _TOOLS))
    if not tools:
        return None
    paragraphs, used = [], []
    for tool in tools:
        facts = [f for f in records if f.get("source_tool") == tool]
        lines = ["已完成：" + tool]
        for fact in facts:
            if fact["key"] in _LABELS:
                lines.append(f"{_LABELS[fact['key']]}：{fact['value']} [fact:{fact['id']}]")
                used.append(fact["id"])
        rows = {}
        for fact in facts:
            match = _ROW.fullmatch(fact["key"])
            if match:
                rows.setdefault((match[1], int(match[2])), {})[match[3]] = fact
        # Each saved table needs its own preview. A long expression table must
        # not consume the entire display budget and hide requested group QC.
        shown = {}
        omitted = False
        for (kind, _), fields in rows.items():
            if shown.get(kind, 0) >= 8:
                omitted = True
                continue
            identity = [fields[k] for k in ("gene", "group", "stage", "cluster") if k in fields]
            values = [fields[k] for k in _ROW_LABELS if k in fields]
            if not values:
                continue
            shown[kind] = shown.get(kind, 0) + 1
            prefix = " / ".join(str(f["value"]) for f in identity) or "汇总"
            parts = [f"{_ROW_LABELS[f['key'].split('].', 1)[1]]}："
                     + ("未知/不可计算" if f["value"] is None else str(f["value"])) for f in values]
            refs = [f["id"] for f in [*identity, *values]]
            lines.append(prefix + "：" + "；".join(parts) + " " + " ".join(f"[fact:{fid}]" for fid in refs))
            used.extend(refs)
        if omitted:
            lines.append("此处每类结果最多展示前8行；完整结果保存在本次工具产物。")
        paragraphs.append("\n".join(lines))
    return "\n\n".join(paragraphs), {"contract": "tool-receipt/2", "accepted": True,
        "errors": [], "rendered_fact_ids": list(dict.fromkeys(used)),
        "free_text_semantics_verified": False, "renderer": "structured-spatial-contracts"}
