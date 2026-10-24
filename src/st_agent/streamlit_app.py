"""Thin Streamlit rendering layer; durable state lives behind WorkbenchService."""

from __future__ import annotations

import json
from pathlib import Path

from st_agent.presentation import public_answer
from st_agent.safety import ApprovalDecision
from st_agent.ui_layout import conversation_layout_css
from st_agent.workbench import WorkbenchService, explain_error


def render_app() -> None:
    import streamlit as st

    title = "空间转录组科研助手"
    st.set_page_config(page_title=title, page_icon="🧬", layout="wide")
    st.title(title)
    st.caption("关键科学决策由研究者确认；原始数据保持只读。")
    st.sidebar.subheader("研究项目")
    project_path = st.sidebar.text_input(
        "项目文件夹路径",
        key="project_path_input",
        placeholder=r"例如 F:\stagent\my_first_test",
        help="新建项目请选择一个新文件夹；打开项目请选择包含 project.yaml 的已有文件夹。",
    )
    create = st.sidebar.button("在该路径新建项目", use_container_width=True)
    open_existing = st.sidebar.button("打开该路径的已有项目", use_container_width=True)
    if create or open_existing:
        if not project_path.strip():
            st.sidebar.error("请先输入项目文件夹路径。空路径不会自动打开任何项目。")
            return
        try:
            path = Path(project_path).resolve()
            service = WorkbenchService.create(path) if create else WorkbenchService.open(path)
            previous_path = st.session_state.get("active_project_path")
            st.session_state["active_project_path"] = str(service.project_path)
            if previous_path != str(service.project_path):
                st.session_state.pop("session_id", None)
            st.rerun()
        except Exception as error:
            _show_error(st, error)
    active_path = st.session_state.get("active_project_path")
    if not active_path:
        st.info("请在左侧输入项目文件夹路径，然后新建项目或打开已有项目。")
        return
    st.sidebar.success("当前已打开")
    st.sidebar.code(str(active_path), language=None)
    if st.sidebar.button("关闭当前项目", use_container_width=True):
        st.session_state.pop("active_project_path", None)
        st.session_state.pop("session_id", None)
        st.rerun()
    try:
        service = WorkbenchService.open(active_path)
        snapshot = service.snapshot()
    except Exception as error:
        _show_error(st, error)
        return
    with st.container(key="analysis_workspace"):
        conversation, gallery = st.columns([1.5, 1], gap="medium")
    with st.sidebar:
        st.subheader("项目数据")
        st.caption("同一项目中，相同文件内容只登记一次。项目保存硬链接或副本，不移动或覆盖源文件。")
        data_path = st.text_input(
            "登记新的 H5AD 文件（可选）",
            placeholder=r"例如 F:\data\sample.h5ad",
        )
        if st.button("登记新的数据集") and data_path:
            try:
                service.add_dataset(data_path)
                st.rerun()
            except Exception as error:
                _show_error(st, error)
        st.dataframe(
            [
                {"名称": item.display_name, "状态": item.status, "数据集 ID": item.id}
                for item in snapshot.datasets
            ],
            use_container_width=True,
        )
        selected_dataset_id = None
        if snapshot.datasets:
            dataset_by_name = {
                f"{item.display_name} · {item.id[-8:]}": item.id for item in snapshot.datasets
            }
            selected_label = st.selectbox("当前分析数据集", tuple(dataset_by_name))
            selected_dataset_id = dataset_by_name[selected_label]
            selected_dataset = next(
                item for item in snapshot.datasets if item.id == selected_dataset_id
            )
            st.info(
                f"本轮对话将分析：{selected_dataset.display_name}（数据集 ID："
                f"{selected_dataset.id}）"
            )
            with st.expander("样本位置与当前版本"):
                st.write("最初登记的源文件位置")
                st.code(selected_dataset.source_path, language=None)
                st.write("项目管理的原始数据位置")
                st.code(selected_dataset.storage_path, language=None)
                st.write("当前分析版本", selected_dataset.active_revision_id)
                st.write("内容指纹（用于同项目去重）", selected_dataset.sha256)
        else:
            st.warning("当前项目还没有数据集。请先登记一个 H5AD 文件。")
        st.subheader("分析对话")
        if st.button("开始新对话", help="保留数据和历史记录，只创建一个新的空白对话。"):
            st.session_state["session_id"] = service.start_chat("科研分析").id
            st.rerun()
        sessions = list(snapshot.sessions)
        if sessions:
            session_ids = [item.id for item in sessions]
            labels = {
                item.id: f"{item.title or '科研分析'} · {item.created_at} · {item.id[-8:]}"
                for item in sessions
            }
            current = st.session_state.get("session_id")
            selected = st.selectbox(
                "历史分析对话",
                session_ids,
                index=session_ids.index(current) if current in session_ids else 0,
                format_func=labels.get,
            )
            st.session_state["session_id"] = selected
        session = service.ensure_chat_session(st.session_state.get("session_id"), title="科研分析")
        st.session_state["session_id"] = session.id
    with conversation, st.container(key="conversation_panel"):
        st.subheader("分析对话")
        follow = st.checkbox("跟随最新消息", value=True, key=f"follow_latest_{session.id}",
                             help="关闭后可以停留在历史消息；切换图表不会拉回对话底部。")
        history_key = f"chat_history_{session.id}"
        st.markdown(conversation_layout_css(history_key), unsafe_allow_html=True)
        with st.container(height=480, key=history_key, autoscroll=follow):
            _render_history(st, service, session.id, snapshot)
        pending = service.pending_approvals()
        st.caption(f"待确认 {len(pending)} 项 · 活动任务 {len(snapshot.active_jobs)} 项。"
                   "任务状态已保存，刷新不会重复分析。")
        prompt = st.chat_input("描述你的科研问题或分析目标")
        if prompt:
            with st.spinner("DeepSeek 正在规划并执行已验证工具…"):
                service.submit_message(session.id, prompt, dataset_id=selected_dataset_id)
            st.rerun()
    with gallery, st.container(key="figure_panel"):
        st.subheader("结果图窗")
        st.caption("独立浏览当前数据版本的图表；查看图片不会重新执行分析。")
        if selected_dataset_id:
            from st_agent.figure_gallery import revision_gallery

            gene = st.text_input("空间表达基因（可选）", key="gallery_gene")
            try:
                bundle = revision_gallery(
                    service.copilot, selected_dataset_id, gene=gene.strip() or None,
                    marker_page_size=3,
                )
                section_options = bundle.get("section_options", {})
                section_value = section_options.get("selected")
                if len(section_options.get("values", [])) > 1:
                    section_value = st.selectbox(
                        "空间图切片", section_options["values"],
                        key=f"gallery_section_{selected_dataset_id}",
                        help="仅改变空间图的显示范围；UMAP仍展示全样本，不重新执行分析。",
                    )
                    if section_value != section_options.get("selected"):
                        bundle = revision_gallery(
                            service.copilot, selected_dataset_id, gene=gene.strip() or None,
                            section_key=section_options["key"], section_value=section_value,
                            marker_page_size=3,
                        )
                figures = {item["id"]: item for item in bundle["figures"]}
                selected_figure = st.selectbox(
                    "查看图表", list(figures), format_func=lambda key: figures[key]["title"]
                )
                item = figures[selected_figure]
                display_options = {"marker_page_size": 3}
                read_options = {"gene": gene.strip() or None,
                                "section_key": section_options.get("key"),
                                "section_value": section_value}
                if selected_figure == "markers":
                    stats = item["metadata"]["statistics"]
                    mode = st.selectbox("Marker显示范围", ("overview", "complete"),
                        format_func=lambda value: "概览：每簇前N个" if value == "overview" else "本次完整marker",
                        key=f"marker_mode_{selected_dataset_id}_{bundle['identity']['revision_id']}")
                    top_n = 3
                    if mode == "overview":
                        maximum = max(stats["available_markers_per_group"].values(), default=1)
                        top_n = int(st.number_input("每簇前N个已保存marker", min_value=1,
                            max_value=max(1, maximum), value=min(3, max(1, maximum)), step=1,
                            key=f"marker_n_{selected_dataset_id}_{bundle['identity']['revision_id']}"))
                    display_options.update(marker_display_mode=mode, markers_per_group=top_n)
                    bundle, item, display_options = _marker_page(
                        st, service, selected_dataset_id, read_options, display_options, prefix="preview")
                    st.caption("窄窗每页3个基因；展开查看可一次显示更多。完整指已保存的marker排名，不是全部表达基因。")
                st.plotly_chart(item["figure"], width="stretch", key="gallery_plot")
                st.write(item["caption"])
                if st.button("展开查看", key="expand_figure"):
                    _open_figure_dialog(st, service, selected_dataset_id, selected_figure,
                                        read_options, display_options, item)
                if st.button("解读当前图并建议下一步", key="explain_current_figure"):
                    with st.spinner("正在根据图表统计摘要生成解读…"):
                        service.explain_figure(
                            session.id,
                            selected_dataset_id,
                            selected_figure,
                            gene=gene.strip() or None,
                            section_key=section_options.get("key"), section_value=section_value,
                            **display_options,
                        )
                    st.rerun()
                st.caption(f"数据版本：{bundle['identity']['revision_id']}")
                st.download_button(
                    "下载图表数据",
                    json.dumps(item, ensure_ascii=False),
                    file_name=f"{selected_figure}.json",
                    mime="application/json",
                )
                if selected_figure == "markers":
                    _marker_inventory(st, service, snapshot, item)
                with st.expander('完整图表保存位置', expanded=True):
                    st.caption('完整图为离线 HTML，可缩放；所有已保存 marker 在同一宽图中，空间图按切片分别保存。')
                    st.code(str(service.project_path / 'artifacts' / 'full_figures'), language=None)
                    if st.button('保存完整图表', key='export_full_figures'):
                        from st_agent.full_figures import export_full_figures
                        with st.spinner('校验已有图表并保存缺失的完整图…'):
                            result = export_full_figures(service.copilot, selected_dataset_id,
                                                        revision_id=bundle['identity']['revision_id'],
                                                        gene=gene.strip() or None)
                        st.success(f"新增 {result['generated']} 张 · 复用 {result['reused']} 张 · 校验后重建 {result['rebuilt']} 张")
                        if result['trace_degraded']:
                            st.warning('trace_degraded：图表已保存，但导出轨迹写入不完整。')
                        for missing in result['unavailable']:
                            st.warning(f"{missing['figure_id']} 未保存：{missing['reason']}")
                    available = [a for a in service.copilot.list_artifacts() if
                                 a.kind.startswith('figure.full.') and
                                 a.metadata.get('revision_id') == bundle['identity']['revision_id']]
                    if available:
                        st.dataframe([{'图表': a.kind, '参数': a.metadata.get('parameters', {}),
                                       '文件': a.relative_path} for a in available], width='stretch')
                    else:
                        st.caption('当前版本尚未归档完整图；点击上方按钮补保存，不会重新分析。')
            except Exception as error:
                st.warning(f"图表暂不可用：{error}。分析结果仍保留。")
        else:
            st.info("选择样本后在这里查看图表。")
    with st.expander("产物、版本与溯源"):
        st.subheader("产物")
        st.dataframe(
            [
                {"类型": item.kind, "路径": item.relative_path, "SHA-256": item.sha256}
                for item in snapshot.artifacts
            ],
            use_container_width=True,
        )
        st.subheader("Revision 时间线")
        for dataset in snapshot.datasets:
            st.write(dataset.display_name)
            st.dataframe(
                [
                    {"Revision": item.id, "状态": item.status, "父版本": item.parent_revision_id}
                    for item in service.revisions(dataset.id)
                ],
                use_container_width=True,
            )
        evidence = service.evidence_and_claims()
        st.subheader("Evidence / Citation")
        st.dataframe(evidence["evidence"], use_container_width=True)
        st.dataframe(evidence["claims"], use_container_width=True)
    with st.expander("记忆管理"):
        for memory in snapshot.memories:
            with st.container(border=True):
                st.write(f"{memory.key}: {memory.value} · {memory.status.value}")
                if memory.status.value == "candidate":
                    left, middle, right = st.columns(3)
                    if left.button("接受记忆", key=f"memory-approve-{memory.id}"):
                        service.decide_memory(memory.id, "approve")
                        st.rerun()
                    edited = middle.text_input("修改值", key=f"memory-edit-value-{memory.id}")
                    if middle.button("修改后接受", key=f"memory-edit-{memory.id}"):
                        service.decide_memory(memory.id, "edit_and_approve", edited)
                        st.rerun()
                    if right.button("拒绝记忆", key=f"memory-reject-{memory.id}"):
                        service.decide_memory(memory.id, "reject")
                        st.rerun()
                elif memory.status.value == "approved" and st.button(
                    "忘记", key=f"forget-{memory.id}"
                ):
                    service.forget_memory(memory.id)
                    st.rerun()
    with st.expander("报告管理"):
        st.info("报告必须由已验证 ToolResult 和受支持 Claim 生成，并由研究者确认定稿。")
        if st.button("从当前分析创建报告", disabled=selected_dataset_id is None):
            try:
                service.create_analysis_report(selected_dataset_id,
                    session_id=st.session_state.get("session_id"))
                st.rerun()
            except Exception as error:
                _show_error(st, error)
        for report_item in snapshot.reports:
            st.subheader(report_item.title)
            st.caption("草稿" if report_item.status.value == "draft" else "已定稿")
            st.components.v1.html(report_item.html, height=300, scrolling=True)
            st.download_button("下载 HTML 报告", data=report_item.html.encode("utf-8"),
                file_name=report_item.id + ".html", mime="text/html",
                key=f"download-report-{report_item.id}")
            if report_item.status.value == "draft" and st.button(
                "确认并定稿", key=f"finalize-{report_item.id}"
            ):
                service.finalize_report(report_item.id)
                st.rerun()
    with st.expander("系统诊断"):
        report = service.diagnostics()
        st.metric("系统状态", "正常" if report.healthy else "需要处理")
        st.dataframe(
            [
                {"检查": item.name, "状态": item.status, "详情": item.detail}
                for item in report.checks
            ],
            use_container_width=True,
        )


def _marker_page(st, service, dataset_id, read_options, display_options, *, prefix):
    """View projection only: never schedules scientific tools or creates revisions."""
    from st_agent.figure_gallery import revision_gallery

    options = dict(display_options, marker_page_index=0)
    bundle = revision_gallery(service.copilot, dataset_id, **read_options, **options)
    item = next(figure for figure in bundle["figures"] if figure["id"] == "markers")
    stats = item["metadata"]["statistics"]
    page_key = (f"marker_page_{prefix}_{dataset_id}_{bundle['identity']['revision_id']}_"
                f"{options.get('marker_display_mode', 'overview')}_{options.get('markers_per_group', 3)}_"
                f"{options['marker_page_size']}")
    page = st.selectbox("Marker页码" if prefix == "preview" else "宽图Marker页码",
        list(range(stats["page_count"])),
        format_func=lambda index: f"第 {index + 1} 页 / 共 {stats['page_count']} 页",
        key=page_key)
    options["marker_page_index"] = page
    if page:
        bundle = revision_gallery(service.copilot, dataset_id, **read_options, **options)
        item = next(figure for figure in bundle["figures"] if figure["id"] == "markers")
    return bundle, item, options


def _open_figure_dialog(st, service, dataset_id, figure_id, read_options, display_options, item):
    @st.dialog("展开图表", width="large")
    def show():
        current = item
        if figure_id == "markers":
            size = st.selectbox("宽图每页基因数", (5, 10, 15), index=1, key="wide_marker_page_size")
            options = dict(display_options, marker_page_size=size, marker_page_index=0)
            _, current, _ = _marker_page(st, service, dataset_id, read_options, options, prefix="wide")
        st.plotly_chart(current["figure"], width="stretch", key="wide_gallery_plot")
        st.write(current["caption"])
        st.download_button("下载展开图表数据", json.dumps(current, ensure_ascii=False),
                           file_name=f"{figure_id}-wide.json", mime="application/json")
    show()


def _marker_inventory(st, service, snapshot, item):
    stats = item["metadata"]["statistics"]
    with st.expander("完整基因清单与marker表"):
        st.write(f"本次可用 {stats['available_gene_count']} 个不同基因；当前选择 "
                 f"{stats['selected_gene_count']} 个，当前页 {len(stats['genes'])} 个。")
        st.dataframe([{"基因": g, "在当前选择中": g in stats["selected_genes"]}
                      for g in stats["available_genes"]], width="stretch")
        st.download_button("下载完整marker基因清单", "gene\n" + "\n".join(stats["available_genes"]),
                           file_name="marker-genes.csv", mime="text/csv")
        from st_agent.utils import sha256_file
        for artifact in snapshot.artifacts:
            if (artifact.kind == "table.markers" and
                artifact.metadata.get("revision_id") == item["metadata"]["revision_id"]):
                path = (service.project_path / artifact.relative_path).resolve()
                if not path.is_relative_to(service.project_path.resolve()):
                    raise ValueError("Marker table path is outside the project")
                if sha256_file(path) != artifact.sha256:
                    raise ValueError("Marker table fingerprint changed")
                st.download_button("下载本次完整marker结果表", path.read_bytes(),
                                   file_name="markers.csv", mime="text/csv", key=f"csv_{artifact.id}")
                break


def _render_history(st, service, session_id, snapshot):
    """Move controls into the native scroller without changing their callbacks."""
    for message in service.conversation(session_id):
        with st.chat_message(message.role):
            st.markdown(public_answer(message.content) if message.role == "assistant"
                        else message.content)
            if message.role == "assistant" and message.metadata.get("trace"):
                try:
                    trace = service.message_trace(message)
                except (ValueError, OSError, KeyError) as error:
                    st.error(f"执行详情读取失败：{error}")
                    continue
                knowledge = trace.get("knowledge", {})
                if knowledge.get("references"):
                    with st.expander("参考来源"):
                        for index, reference in enumerate(knowledge["references"], 1):
                            st.markdown(f"{index}. [{reference['title']}]({reference['uri']})")
                            st.caption(f"版本：{reference['source_version']} · {reference['support_scope']}")
                            st.text(reference["excerpt"])
                            st.code(json.dumps(reference["locator"], ensure_ascii=False), language="json")
                        st.caption("文献适用条件未核实的内容仅作背景，不证明当前样本结论。")
                with st.expander("来源与执行详情"):
                    st.json(trace)
                if message.metadata.get("status") == "paused_provider":
                    if st.button("重试本轮模型请求", key=f"resume-{message.id}"):
                        with st.spinner("恢复原请求与上下文…"):
                            service.resume_run(trace["run_id"])
                        st.rerun()
    for approval in service.pending_approvals():
        with st.container(border=True):
            st.subheader(approval.tool_name)
            st.json(approval.final_arguments)
            edited = st.text_area("修改参数（JSON）", value=json.dumps(approval.final_arguments),
                                  key=approval.id)
            left, middle, right = st.columns(3)
            if left.button("确认", key=f"approve-{approval.id}"):
                service.decide_approval(approval.id, ApprovalDecision.APPROVE)
                service.resume_analysis(approval.id)
                st.rerun()
            if middle.button("修改后确认", key=f"edit-{approval.id}"):
                service.decide_approval(approval.id, ApprovalDecision.EDIT_AND_APPROVE, json.loads(edited))
                service.resume_analysis(approval.id)
                st.rerun()
            if right.button("拒绝", key=f"reject-{approval.id}"):
                service.decide_approval(approval.id, ApprovalDecision.REJECT)
                service.resume_analysis(approval.id)
                st.rerun()
    for job in snapshot.active_jobs:
        left, right = st.columns([5, 1])
        left.write(f"{job.entrypoint} · {job.status}")
        if right.button("取消任务", key=f"cancel-{job.id}"):
            service.cancel_job(job.id)
            st.rerun()


def _show_error(st, error: BaseException) -> None:
    explained = explain_error(error)
    st.error(explained.title)
    st.write(explained.explanation)
    st.info(explained.next_action)
    with st.expander("技术详情"):
        st.code(explained.technical_detail)
