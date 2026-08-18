# 单算子劣化实测优化点库骨架（`runbooks/operator-optimization/single-op-degradation.md`）

[`ops-knowledge-optimization-ingest`](SKILL.md) 的**编译阶段**（见 [`references/workflow.md`](references/workflow.md)「编译进知识库」）把本次挖出的劣化机制**增量合并**进这份跨算子单一共享的 runbook 时加载本骨架。字段映射、优化维度判定、增量合并纪律见 workflow.md。

**关键纪律**：扁平 `OPT-*`，跨算子单一共享、**增量合并**（`canonical_family` 命中→只在「已知实例」append 反链；新机制→新增 `OPT-N`，**ID 不复用、不重排**）；条目**算子无关**——原则与骨架只用占位名（`inputA`/`constOperand`/`unitBytes`/`KEY_A`…），**禁 golden/算子业务变量名、禁 mermaid**；标题瘦身 `## OPT-N <短标题>`（不带【标签】/破锚符号/「已知实例」）；**每条 OPT 必含「坏实践（反例）」字段**（本库该字段总能填实——即挖出的劣化机制本身）；`## OPT-N` 标题、各 `**字段**：` 块、列表项之间一律空一行（GFM 下相邻非空行会塌成一段）。runbook 头部 blockquote **只写一句「定位」**，不嵌生成-维护元规则（YAML frontmatter 允许）。

**与 vv-fusion-common 的区别**：本库优化点均由**受控劣化 + 远程 eval 实测**得出，故 `置信度` 统一为「已验证(独立eval)」，且每条都带实测 `before→after` 收益；vv 库多为 golden 冷启动。两库并列于 `runbooks/operator-optimization/`，互不覆盖。

```
---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: 单算子劣化实测优化点库（NPU 垂域）
description: <一句话：由受控劣化 + 实测 profile 对比沉淀的跨算子 NPU 优化点库>
tags: [single_op, degradation, optimization]
created_at: '<YYYY-MM-DD>T00:00:00Z'
updated_at: '<YYYY-MM-DD>T00:00:00Z'
---

# 单算子劣化实测优化点库（NPU 垂域）

> **本文件**：<一句「定位」——单算子受控劣化挖矿沉淀的实测泛化 NPU 优化点库，跨算子单一共享、增量合并>

---

## 核心决策清单

每条 OPT 一行，设计期据此速查选型（优先级 × 优化维度 × 触发 → 指向）：

| 优先级 | 优化维度 | 优化点一句话 | 触发条件 | 实测收益 | 指向 |
|--------|----------|--------------|----------|----------|------|
| 必做/重要/进阶 | 搬运·内存·计算·精度 | <一句话> | <触发条件> | 反向劣化慢 X% | OPT-N |

---

## OPT-N <准确、短的核心标题>

- **摘要**: <一句话：优化了什么 + 怎么做；信息须完整、禁过简口号>
- **触发**: <何时适用：场景/条件>
- **优化维度**: 搬运·内存·计算          （∈ {搬运,计算,内存,精度} 一或多；按 profile_delta 主导指标判定）
- **泛化层级**: 通用 | 条件性
- **优先级**: 必做 | 重要 | 进阶
- **置信度**: 已验证(独立eval)          （本库统一；来源为远程评估服务实测）
- **实测收益**: 反向劣化使算子慢 <X>%（<before_us> → <after_us>，round N <mode>）
- **关联**: OPT-x · AP-y · CT-z          （可空）

**原则**：<NPU 垂域优化点，1~2 句，what + why；由劣化根因取逆得来，须与实测 profile 变化一致>

**通用骨架/示意图**：<算子无关；过程/算法型→伪码骨架，结构/权衡型→ASCII 示意图；占位名，禁 golden 变量名>

**迁移条件**：

- 适用：<对哪些算子/场景成立>
- 前提：<可机械检查的硬条件>
- 失效：<何时不成立、换算子要重判的点（由 why_not_duplicate / canonical_family 推导）>

**坏实践（反例）**：<**必填·本库即挖出的劣化机制本身**>如此改动（<change_anchor>）→ 实测慢 <X>%，瓶颈由 <before 主导> 迁移到 <after 主导>〔来源：round N <mode>，experience_lib〕；✅ 正解即本条**原则**。

**已知实例**：

- `ops/<category>/<op>.md#锚点`（一句：该算子在何处实例化本条）
  （知识库无该算子卡时写「待补充（experience_lib: <路径> 第 k 条）」，勿留悬空锚）

（每条 OPT 之间空一行。全部 OPT-* 条目之后依次接 ↓ 约束陷阱区 →「反模式」专节（可选·有则写） →「# 相关」托管块）

---

## 约束与陷阱（Constraints & Pitfalls，可选区）

无对应正向优化点的**平台/API/DMA/精度约束陷阱**单列为 `## CT-N`（与 OPT/AP 同为扁平条目、ID 单调不复用）：

## CT-N <短标题：什么约束/陷阱>

- 标签块 `类别 / 适用平台 / 置信度` + `症状 / 根因 / 规避 / 预防 / 已知实例`。

---

## 反模式（AP-*）（可选·仅当有无法内联的反例时）

> 无法内联到具体 OPT 的纯坏写法 → 独立 `## AP-N`（注明"读者忽略变量名看模式"）。**无此类反例则不写本节**，不留空占位。可内联的反例进对应 OPT 的「坏实践」字段（本库劣化机制通常都能内联，故此节多为空）。

---

<!-- okf:related:start -->

# 相关

- 相关主题: <成员算子卡>（相对路径） — <一句：该卡实例化了哪些 OPT>

<!-- okf:related:end -->
```

## index.md 骨架（`runbooks/operator-optimization/index.md`）

section 级导航，`kind: index`；已存在则**追加**指向 `single-op-degradation.md` 的条目，不覆盖既有（如 `vv-fusion-common.md`）：

```
---
schema_version: okf.v1
kind: index
type: section_index
title: 算子优化点库索引
updated_at: '<YYYY-MM-DD>T00:00:00Z'
---

# 算子优化点库

- [单算子劣化实测优化点库](single-op-degradation.md) — 受控劣化 + 实测 profile 沉淀的跨算子优化点
- [VV 融合算子公共优化点库](vv-fusion-common.md) — 多模板 VV 融合算子 golden 沉淀（若存在）
```
