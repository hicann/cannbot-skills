# `kb/` — ascendc-port-orchestrator 的知识目录

本文件说明 `kb/` 下每个子目录**装什么、谁读它、什么格式、谁能写**。
读者不需要预先了解本插件。

> **要改 `kb/` 的内容？先读 [`CONVENTIONS.md`](CONVENTIONS.md)** —— 那是硬约束
> （只读层、语料范围、目录深度、bundle 命名、frontmatter 受控词表、门禁），
> 每条都标了依据与违反后果。本文件只讲「是什么」，约束在那边。

---

## 0. 先讲背景：这个插件为什么需要一个知识目录

`ascendc-port-orchestrator` 做两件事：把 AscendC 算子从一代昇腾架构**移植**到另一代
（arch22 → arch35），以及由正向算子**生成反向算子**。这两件事都由一个确定性状态机驱动，
状态机会拉起若干子 agent（写 kernel 的、调性能的、查精度的）去干活。

子 agent 是无状态的——每次拉起都是全新上下文。所以「上一个算子踩过的坑」「这块芯片的
UB bank 是多少」「这个错误码意味着什么」必须**写在某个地方**，在拉起子 agent 时注入它的
brief 里。`kb/` 就是这个地方。

### 三层知识（术语：a 层 / b 层 / c 层）

| 层 | 是什么 | 在哪 | 运行时可写？ |
|---|---|---|---|
| **a 层** | 社区通用方法论 skills | 本仓其他插件 | — |
| **b 层** | 本插件自带的知识，随插件分发 | **`kb/`（就是本目录）** | **否，只读** |
| **c 层** | 用户本地积累的知识，跑一次算子长一点 | `$ASCENDC_PORT_USER_KB`，默认 `~/.ascendc-port/user_kb` | 是 |

冲突时优先级 **c > b > a**：用户自己验证过的经验覆盖插件自带的。

> **b 层为什么必须只读**：插件会被升级/重装，运行时写进 `kb/` 的东西会在下次升级时被覆盖丢失。
> 所以所有运行期产生的新知识都必须落到 c 层。

---

## 1. 目录速查

```
kb/
├── okf/            ← 可检索的知识卡片库（b 层主体）
├── shared/         ← 给 agent 读的运行时纪律与协议（不是可检索知识）
└── (仅此三项 —— hardware/ 与 plugin-scope/ 已并入 okf/，见 §4 §5)
```

| 目录 | 装什么 | 谁读 | 格式 |
|---|---|---|---|
| `okf/` | 知识**卡片**：踩过的坑、可复用优化点、硬件事实、API 参考 | 检索引擎（见 §2） | OKF `okf.v1` frontmatter + markdown |
| `shared/` | 行为约束：始终加载的规则、闸门契约、写作纪律、复盘 | 子 agent 的 brief 直接整段注入 | 普通 markdown，无 frontmatter |


---

## 2. `okf/` — 知识卡片库

### 什么是 OKF

**OKF = Operator Knowledge Format**，一种知识卡片规范。一张卡 = 一个 `.md` 文件，
开头是 YAML frontmatter（机器读），后面是正文（人和 agent 读）：

```markdown
---
schema_version: okf.v1
kind: operator_optimization      # 受控词表，见下
type: optimization_runbook
source_family: curated
title: "一句话说清这张卡讲什么"
description: "两三句摘要，检索结果里显示这个"
tags: [ascendc, optimization, memory_access]
created_at: 2026-08-27T00:00:00Z
updated_at: 2026-08-27T00:00:00Z
---

# 正文
```

`kind` 只能取这 9 个值之一（由检索引擎的词表约束）：
`api` / `guide` / `example` / `operator` / `glossary` / `operator_optimization` /
`implementation_trap` / `debugging_journey` / `cross_skill_gap`。

### 谁来检索

检索引擎**不在本插件里**，是另一个社区插件 `cannbot-knowledge` 的 `knowledge-query`。
本插件通过包装脚本调用它：

```bash
engine/src/scripts/okf/okf_kb.sh build     # 建索引 → kb/okf/search/okf.index.json（已 gitignore）
engine/src/scripts/okf/okf_kb.sh search …  # 检索
engine/src/scripts/okf/okf_kb.sh lint      # 规范体检（维护者用）
```

引擎只索引三个子目录（它写死在 `retrieval/config.py` 的 `CONTENT_ROOTS`）：
`reference/`、`ops/`、`runbooks/`。**只收 `.md`，且每一级的 `index.md` 都被排除**。
放在这三个目录之外的任何东西（例如曾经的 `_migration/`）都不进语料。

### 三个板块

| 板块 | 装什么 | 现状 |
|---|---|---|
| `runbooks/` | 实战卡：踩过的坑 + 可复用优化点 + 硬件事实 | 768 份，760 份带完整 frontmatter ✅ |
| `reference/` | API 参考卡与长文指南，按溯源制度分 2 个 bundle | 546 份，**全部带 OKF frontmatter** ✅ |
| `ops/` | 按单个算子的设计卡 | 预留占位，当前 0 张卡 |

> 数字口径：份数 = `find kb/okf/<板块> -name '*.md' | wc -l`；带 frontmatter = `grep -rl '^schema_version:'`。

`reference/` 内部按 **溯源制度** 分 bundle —— bundle 名（= `reference/` 下第一级）决定
`knowledge_lint.py` 适用哪套溯源规则：

```
reference/
├── asc-devkit-vendored/     511   上游 asc-devkit 文档的搬运副本，source_family=asc_devkit
│   ├── api/                 490   逐 API 参考，23 个功能族
│   │                              （reg_vector 124 / vector_compute 101 / cube_datamove 48 /
│   │                                tensor_layout 37 / class_api 26 / sys_var 25 / …）
│   └── guide/                21   programming_model 9 / api_overview 7 / compat_migration 5
└── porter/                   35   本插件自产，source_family=curated
    ├── patterns/             11   代码模板与实测记录
    ├── playbook/              9   arch22→arch35 迁移方法论 L1–L5 + ops-nn A5 产物布局
    ├── precision/             7   精度标准与测试流程
    ├── handbook/              6   API 目录 / 语言参考 / Roofline / SIMD-SIMT 决策
    └── toolchain/             2   msprof / NPU UT
```

> **为什么不叫 `asc-devkit`**：`knowledge_lint.py:81` 的 `GIT_BUNDLES = {"asc-devkit", "ops"}`
> 要求该 bundle 每张卡的 `resource` 是 GitCode **40 位 sha** 的永久链接。而这批文档的确切
> 上游 commit 不可考（经 `Ascend/agent-skills` PR #103 二次搬运，该 PR 分支已删；
> `cann/asc-devkit` 当前 HEAD 的文档树已改版，最接近的 9.0.0 分支仍有 126 条路径缺失）。
> 命名为 `asc-devkit` 就必须编一个 sha。代价是失去 `relevance.py:131` 对 `asc-devkit/api/`
> 的 +0.08 加分（仅 `api_usage` 检索意图时生效）。

`runbooks/` 内部再分：

```
runbooks/
├── field-notes/            现象卡：「我遇到了什么」
│   ├── build/              编译/构建/运行期报错          kind: implementation_trap
│   ├── precision/          精度问题                      kind: implementation_trap
│   ├── perf/               性能/测量                     kind: operator_optimization  ← 见 §6
│   └── inferred/           未验证候选（CAND-*，status:stub，默认检索排除）
├── operator-optimization/  可复用优化点与反模式：「下次该怎么做」  kind: operator_optimization
└── hardware/               芯片规格与 probe 实测           kind: guide
```

> **`field-notes/` 与 `operator-optimization/` 的区别**：前者记录**一次具体观测**（现象、
> 错误码、复现条件），后者记录**跨算子可复用的做法**（判定准则、正确写法、反模式）。
> 同一次经历常常两边各留一张，靠 frontmatter 的交叉引用连起来。

---

## 3. `shared/` — 运行时纪律，不是可检索知识

这里的文件**不是**知识卡，不进 OKF 语料，也不该按关键词检索。它们是**行为约束**，
在构造子 agent 的 brief 时被整段注入：

| 文件 | 作用 |
|---|---|
| `ALWAYS_LOADED_RULES.md` | 每个子 agent 都必须遵守的硬规则 |
| `ANTI_PRESSURE_PROTOCOLS.md` | 防止 agent 在压力下走捷径（伪造结果、降级验证）的协议 |
| `GATE_CONTRACT.md` | 各阶段闸门的契约：什么条件算过、不过怎么办 |
| `KERNEL_AUTHORING_GUARDS.md` | 写 kernel 时的禁止事项 |
| `KB_WRITING_DISCIPLINE.md` | 往 KB 里写东西的纪律（例如新条目必须英文） |
| `REGRESSION_METHODOLOGY.md` / `BENCHMARK_METHODOLOGY.md` | 回归与基准测试方法论 |
| `OUTPUT_PROJECT_LAYOUT.md` | 产出工程的目录约定 |
| `exploration/` | 有界探索协议、结构化维度、证据链 |
| `retrospectives/` | 历史复盘记录 |

**`ANTI_PRESSURE_PROTOCOLS.md` 是安装期硬依赖**——`init.sh` 的 `REQUIRED_PACKAGED_KB`
会检查它存在，缺失则安装直接失败。

> **命名遗留问题**：`shared` 这个名字容易和 OKF 的 `reference/`（共享参考知识）混淆。
> 更准确的名字是 `protocols/` 或 `rules/`。改名涉及约 71 处引用，留待独立 PR。

---

## 4. 硬件事实卡在哪（`hardware/` 目录已撤销）

芯片规格与 probe 实测结论**全部是 OKF 卡**，在 `okf/runbooks/hardware/`：

| 卡 | 内容 |
|---|---|
| `target-ascend950pr.md` / `target-ascend910c.md` / `target-ascend910b.md` | 各代芯片规格。由 `briefs/op_taxonomy.py` 的 `TARGET_HW_SPEC_MAP` 按 target 分发进 brief 的 MANDATORY 清单 |
| `probe-<date>-q-*.md`（6 张） | `aog-hardware-probe` 的历史实测结论（UB bank / MTE2 并行 / 指令周期 / 标量广播 / L1 scratch / DataCopyPad 尾块） |

> **历史**：曾有 `kb/hardware/{target,probe_findings}/` 与这些卡**内容重复**（去 frontmatter 后差 1–2 行），
> 是首轮迁移「converted 但源文件未删」的遗留。2026-09-05 已删除原件，
> 所有引用（`TARGET_HW_SPEC_MAP`、UT 断言、agent/skill prompt、卡间交叉引用）改指 OKF 卡。

**新 probe 结果写哪**：写 **c 层**（`$ASCENDC_PORT_USER_KB/probe_findings/`），
**不要写进 `kb/`** —— 那是随插件分发的 b 层，只读，写进去会在下次升级时被覆盖丢失。
`agents/aog-hardware-probe.md` 的交接说明已按此更正。

`HIASCEND_DOC_URLS.md`（hiascend.com 已知有效文档 URL 注册表，agent 查公开文档的起点、
发现新页面时追加）已移至 `shared/`，与其它 MANDATORY 项同处一层。

---

## 5. 非卡片资产放哪

`.txt` / `.json` 等非 Markdown 文件**不进 OKF 语料**（索引器只收 `.md`，见 §2），
但它们常常是某张卡的**证据**或**数据表**。约定是：**与引用它们的卡同目录存放**，
不单开资产目录 —— 证据紧邻声称，改卡时不容易漏掉配套数据。

| 文件 | 与谁同目录 | 用途 |
|---|---|---|
| `runbooks/operator-optimization/cube_required_ops.txt` | `cann-op-family-arch-classification.md`（该卡正文自述 "Full op list ... in this dir"） | cube 必需算子清单，由 `kw_brief_pa3_phases.py:364` 按路径下发给 agent |
| `reference/porter/patterns/fa_a3_multicore_mc_{verify,perf}.json` | `fa_a3_multicore_result.md` | pb-56 卡「20/20 精度、disk-verifiable」那句声称的原始实测数据 |

> **历史**：曾有一个 `kb/plugin-scope/` 目录，是 `plugins/protocol.py` 的 `kb_subdirs()`
> 文档化的 Phase-2 槽位（按 mode 隔离加载知识）。该槽位至今未接线
> （`plugins/base.py:kb_subdirs()` 返回 `["."]`，`port_a3` 未覆盖），且只装了一份文档。
> 该文档已转为正式 OKF 卡
> （`reference/porter/playbook/ops_nn_a5_artifact_layout.md`），进入检索语料，目录随之删除。
> 将来若要启用按 mode 隔离，重建该目录即可，`protocol.py` 的设计说明仍然有效。

## 6. 已知问题（如实记录，勿当作现状即正确）

1. ~~**`okf/reference/` 不符合 OKF 卡片规范**~~（2026-09-05 已解决）：原 570 份为整树平移的
   **原文**、0 份带 frontmatter，`okf_kb.sh lint` 报 5774 个 blocker。现已重组为 2 个 bundle、
   545 张卡全部补齐 frontmatter，blocker 收敛到 **1**（见第 5 条）。
   **但正文仍是「照搬」而非「蒸馏」**——`ops-knowledge-reference-ingest/SKILL.md` §1 要求
   「蒸馏非照搬」，而 `knowledge_lint.py:14-15` 明说该原则 out of scope、只在
   `--sample N` 的语义深审里查。所以「lint 全绿」不等于「内容已蒸馏」。
   另注意 `okf_kb.sh lint` **当前不在任何 CI 里跑**。

5. **`okf_graph verify` 的「cards never judged」是唯一剩余 blocker，已论证排除出范围**：
   `okf_graph.py` 是 LLM 判定的知识图谱层。排除依据（均可复核）：
   ① porter 无消费方——0/1330 卡有 `# 相关` 托管块，全插件 0 处调用 `related`/`neighbors`；
   ② 图谱不影响检索——`retrieval/index.py:77,280` 建索引时 `strip_related(body)` 主动剥掉；
   ③ 属存量状态——base 上即报 468 never judged、`judgments: 0`，图谱层从未建过。
   若将来要启用多跳检索，需单列里程碑：本仓这一版 `okf_graph.py` **没有 `judge` 子命令**
   （只有 build/candidates/inject/viz/verify/related/explain），判定须 agent 侧自行实现，
   缓存写 `<kb>/graph/edge_judgments.json`（`graph/` 不在 .gitignore，可提交）。

2. **`runbooks/field-notes/perf/` 的 kind 与兄弟目录不一致**：`build`/`precision`/`inferred`
   都是 `implementation_trap`，唯独 `perf`（28 张）是 `operator_optimization`——与顶层
   `operator-optimization/` 同 kind，机器层面无法区分二者。

3. **`hardware/` 的 9 份硬件事实与 `okf/runbooks/hardware/` 重复**（见 §4）：同一份知识
   被两套机制各自消费（brief 按固定路径 / 检索按卡片），两边正文差异仅 2–3 行。
   源文件有运行时硬引用，不能直接删。

4. **知识积累回不到 OKF 卡**：运行期新知识落到 c 层，但走的是两种非 OKF 格式
   （`learn_extract.py` 写 legacy 的 `candidates.md` 块；`kb_tiering` 的 c 层适配器写 JSON），
   且 `init.sh` 不创建 `learn_extract.py` 要写的 `reference/patterns/unverified/` 目录，
   父目录不存在时该函数静默返回 False。CAND → 正式卡的自动晋升模块也已移除。
   `okf/runbooks/field-notes/inferred/` 里的 121 张 `cand-*` 卡是一次性历史转换的快照，
   不是新知识的落地格式。
