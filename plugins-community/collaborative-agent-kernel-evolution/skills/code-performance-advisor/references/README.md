# references 索引指南（给人和 ClaudeCode 助手）

本目录用于统一管理性能分析相关参考资料。目标是：**你要做什么，就能立刻定位到该去哪找什么**。

---

## 1. 快速导航：按任务找资料

| 我要做什么 | 去哪个目录 | 优先看什么文件 |
|---|---|---|
| 了解整体架构与模块关系 | `architeture/` | 架构总览、模块边界、流程图类文档 |
| 查看规则、入库标准、规则写法 | `standards/` | 入库规则、规则模板、规则示例 |
| 查芯片硬件差异与平台特征（`externel_refs` **按需获取**，见 `externel_refs/README.md`） | `externel_refs/hardware/` | `Ascend*.md`、`CUDA_A100.md` |
| 查官方算子 API / 调用与开发文档（按需获取） | `externel_refs/official_operator_api_introduction/` | 各域 `docs/zh/op_api_list.md`、`op_list.md`、`context/*` |
| 查历史算子分析结论与专家建议（按需获取） | `externel_refs/IdeaPool/IdeaPool/<op_name>/` | `diff_analysis.md`、`expert_ideas.md`（及对应 `.json`） |
| **查 CSV profiling 数据解读方法** | `standards/` | `csv_systematic_analysis_framework.md`, `op_summary_header_guide.md` |
| **查流水图分析方法** | `standards/` + `../subskills/` | `msprof_usage_guide.md`, `deep_research.md` |

---

## 2. externel_refs 目录说明（当前）

### 2.1 `hardware/`
- 作用：硬件平台信息与能力边界对照。
- 文件模式：`<Platform>.md`（例如 `Ascend910B4.md`）。
- 使用场景：性能瓶颈定位、跨平台策略选择、约束判定。

### 2.2 `official_operator_api_introduction/`
- 作用：官方算子文档镜像，按算子域拆分。
- 当前域目录：
	- `ops-math-master-docs/`
	- `ops-nn-master-docs/`
	- `ops-transformer-master-docs/`
- 检索入口建议：
	- 先看 `docs/zh/op_api_list.md`（算子 API 索引）
	- 再看 `docs/zh/op_list.md`（算子清单）
	- 再看 `docs/zh/context/*`（数据类型、接口约束、构建/运行等）

### 2.3 `IdeaPool/IdeaPool/`
- 作用：按算子沉淀"差异分析 + 专家建议"。
- 目录模式：`<op_name>/`。
- 标准文件：
	- `diff_analysis.md` / `diff_analysis.json`
	- `expert_ideas.md` / `expert_ideas.json`
- 使用场景：复用已验证思路、减少重复分析。

---

## 3. standards 目录说明（性能分析核心方法论）

### 3.1 CSV Profiling 分析（Phase 1 轻量级）

| 文档 | 用途 | 何时使用 |
|------|------|----------|
| `op_summary_header_guide.md` | CSV 字段定义速查表 | 遇到不认识的 CSV 字段时 |
| `csv_systematic_analysis_framework.md` | **8 维度系统化分析框架**（仅 CSV） | Phase 1：快速定位宏观瓶颈（计算/访存/等待） |

**核心方法**：
- 8 维度分析（BasicInfo, ArithmeticUtilization, PipeUtilization, ResourceConflictRatio, Memory, MemoryL0, MemoryUB, L2Cache）
- 信号→行动决策表
- 典型瓶颈模式识别

**数据需求**：`op_summary*.csv` 及 8 维度 CSV

**成本**：低（秒级分析）

---

### 3.2 流水图综合分析（Phase 2 深度）

| 文档 | 用途 | 何时使用 |
|------|------|----------|
| `msprof_usage_guide.md` | msprof 工具使用指南 | 需要采集 profiling 数据（CSV + 流水图）时 |
| `../subskills/deep_research.md` | **流水图 + CSV 综合分析**（Phase 2） | Phase 1 建议无效，需要指令级流水线分析时 |

**核心方法**：
- 流水图模式识别（Wait-Wait, Starvation, Transfer-Compute Gap, Sync Overhead）
- CSV-流水图交叉验证
- 量化指标提取（Idle Cycles, Overlap Ratio, Sync Density）
- 代码级瓶颈映射

**数据需求**：`op_summary*.csv` + 流水图（SVG/PNG/HTML）

**成本**：高（需要额外 profiling，可能需小时）

---

### 3.3 其他标准文档

| 文档 | 用途 |
|------|------|
| `TAG_ADDITION_GUIDE.md` | 如何为规则添加新标签 |
| `tag_taxonony.md` | 标签分类体系 |
| `performance_threshold_guide.md` | 性能目标设置指南 |
| `ascendc_api_validation_reference.md` | AscendC API 验证参考 |

---

## 4. 给大模型友好的检索约定

为提升 ClaudeCode 与其他大模型检索命中率，新增内容建议遵守：

1. **目录名表达主题**：例如 `hardware`、`official_operator_api_introduction`、`<op_name>`。
2. **文件名表达用途**：优先使用 `*_list.md`、`diff_analysis.md`、`expert_ideas.md` 这类稳定命名。
3. **每个子目录保留入口文件**：建议放 `README.md`，写清"目录用途 + 文件清单 + 最近更新"。
4. **中英文混排可接受，但关键词要稳定**：如 `op_api_list`、`context`、`develop`。

---

## 5. 渐进式披露工作流（Phase 1 → Phase 2）

**核心设计理念**：先用轻量级方法（CSV），无效时再升级到深度方法（流水图）

```
Phase 0: Fast Triage
  ├─ 数据：op_summary.csv（基本信息）
  ├─ 工具：code_tag + rules_search
  └─ 决策：根据 rule score 选择路径

Phase 1: Lightweight Analysis（CSV 分析）
  ├─ 数据：op_summary + 8维度 CSV
  ├─ 工具：deep_research subskill
  ├─ 参考：csv_systematic_analysis_framework.md
  └─ 输出：Top-3 优化建议（基于 CSV 指标）

Phase 2: Medium-Depth Analysis（流水图 + CSV 综合分析）
  ├─ 触发：Phase 1 建议无效
  ├─ 数据：CSV + 流水图（SVG/PNG）
  ├─ 工具：deep_research subskill
  ├─ 参考：msprof_usage_guide.md, deep_research.md
  └─ 输出：指令级瓶颈诊断 + 代码映射建议

Phase 3: Expert Knowledge Base
  ├─ 触发：Phase 1 & 2 建议均无效
  ├─ 工具：suggest subskill（rule library pattern matching）
  └─ 降级策略

Phase 4: Knowledge Capture
  ├─ 触发：任意 Phase 的建议成功
  ├─ 工具：rule_update subskill
  └─ 目标：将验证过的优化转化为可复用规则
```

---

## 6. 后续扩展模板（直接复制）

新增一级目录时，在本 README 的"快速导航"补一行，并按下列模板创建子目录 README：

```md
# <目录名>

## 用途
- 一句话说明该目录解决什么问题。

## 何时使用
- 场景1：...
- 场景2：...

## 文件索引
- <文件名A>：用途...
- <文件名B>：用途...

## 检索关键词
- 关键词1, 关键词2, 关键词3
```

这样可以保证：**人类好找，助手也好检索，后续新增不会破坏结构一致性**。

---

## 7. 最近更新（Changelog）

**2026-02-24**：
- ✅ 新增 `csv_systematic_analysis_framework.md`：8 维度 CSV 分析框架（Phase 1）
- ✅ 新增 `msprof_usage_guide.md`：msprof 工具使用指南（数据采集）
- ✅ 整合自 triton-ascend-dev-main 的优秀实践：
  - CSV 解读方法论（csv-interpretation.md）
  - 流水图生成和分析模板（msprof-op.md）
  - 系统化的瓶颈模式识别
  - 信号→行动决策表

---

**维护者**：code-performance-advisor skill
**最后更新**：2026-02-24