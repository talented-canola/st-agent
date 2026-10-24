# ST Agent V2 使用指南

[返回项目首页](README.md)

## 1. 安装和模型配置

使用Python 3.11，进入仓库根目录：

```powershell
conda create -n st_agent_v2 python=3.11 -y
conda activate st_agent_v2
python -m pip install -e ".[ui,knowledge]"
Copy-Item .env.example .env
```

编辑本机`.env`：

```dotenv
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

模型名必须与实际服务支持的名称一致；模板中的`deepseek-flash`是项目所用配置名。使用其他兼容服务时，同时配置对应地址和模型名。不要把密钥粘贴到对话或截图。

```powershell
python -m streamlit run app.py
```

Windows环境锁文件为`environment-win-64.lock.yml`。首次使用建议先完成上面的最小安装。

## 2. 准备数据和项目

- 输入格式：H5AD；矩阵为观测×基因。QC需要可信的计数来源，已归一化表达不能代替原始计数。
- 空间分析需要有限的空间坐标，以及切片分组信息。示例数据使用`X_spatial`和`chip`。
- 左侧输入新文件夹路径创建项目，登记H5AD，再选择“当前分析数据集”。之后可以打开原项目继续使用。
- 原始数据保持只读；执行变换时，程序在项目中保存派生版本和产物。

## 3. 按顺序完成一次分析

下面的参数是可复制的流程示例，应根据自己的QC结果调整。每条完成后再发送下一条；出现审批卡时，核对参数与输入版本后确认。

### 检查与QC

```text
先检查当前数据规模、计数来源、已有处理状态和空间坐标，不要过滤。
```

```text
计算总计数、检测基因数和线粒体比例，汇总当前QC，不过滤。
```

### 过滤与预处理

```text
按min_genes=500、min_cells=3、max_mito_pct=5过滤，保留原始版本，报告过滤前后数量。
```

```text
只在当前筛后版本做标准预处理：target_sum=10000、n_hvg=2000、n_pcs=30，不重复过滤。
```

### 聚类与marker

```text
在已有PCA上运行Leiden与UMAP：n_neighbors=15、resolution=0.8、seed=7。
```

```text
按当前簇找marker：method=wilcoxon、n_genes=10。保存候选基因及其统计。
```

### 空间分析

以下基因和分组列必须存在于自己的数据中。

```text
只构建当前版本的空间邻居：n_neighbors=6、groupby=chip，各切片内部连接。
```

```text
复用已有spatial_connectivities，计算Dnmt3b、Prl8a2、Apoe的Moran I，expression_source=X，不重建图。
```

```text
按chip分别做纯坐标KMeans分区：n_domains=3、seed=7、groupby=chip。
```

```text
用已有PCA生成表达与空间联合域：representation=X_pca、n_neighbors=6、groupby=chip、resolution=0.8、seed=7。
```

纯坐标分区使用位置；联合域结合表达表示和空间邻域。Moran I为描述性系数，当前工具不返回显著性P值。

## 4. 看图和导出

在右侧“结果图窗”选择图表。空间表达图先填写实际基因名，并选择相应切片。“展开查看”打开大图；marker支持分页、完整显示和表格下载。查看已保存图表不需要重新进行科学计算。

“产物、版本与溯源”可查看结果所属版本。“报告管理”中点击“从当前分析创建报告”，再“下载HTML报告”。报告先保存为草稿，研究者核对后再确认。图表说明可能沿用更早运行的布局样式，以当前页面控件为准。

## 5. 常见问题

| 情况 | 处理 |
|---|---|
| 无法启动或提示缺包 | 确认当前环境为Python 3.11，并安装`.[ui,knowledge]`；使用`python -m streamlit`启动 |
| API认证或模型报错 | 核对本机`.env`的密钥、地址和服务端模型名，修改后重启应用 |
| 资源检查拒绝执行 | 释放内存或关闭其他大任务后重试，保留资源检查 |
| 提示已有未完成任务 | 先检查原任务状态和审批；避免在缺参多步任务上继续叠加新工作流 |
| 提示缺少知识快照 | 本发行包未打包外部原文下载缓存，需准备与版本清单对应的本地知识材料 |
| 回答漏掉表格或分组结果 | 先在保存表格中核对；上游结果跨版本解释有已知缺口，必要时明确询问对应版本与分组 |

## 6. 文件保存

科研项目中的`project.yaml`、SQLite数据库、`datasets/`与`artifacts/`属于自己的分析工作区。移动或备份项目时保留完整目录。API密钥保存在仓库根目录`.env`，不属于结果报告，也不随发行包提供。
