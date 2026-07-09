# Plan Dashboard 模板

Dashboard 记录从推理场景、baseline profiling 分析、候选发现，到 Plan 实施 / review / 最终验收的整条过程。它是**单一文件**，约定落在 `optimization-analysis/<case>/plan-dashboard.md`，候选发现完成后初始化；在那之前的场景和 baseline profiling 产物，也先记进本文件对应小节。

## 落盘布局

一次优化的所有产物固定按下面布局落到 `optimization-analysis/<case>/`：

```text
optimization-analysis/<case>/
├── plan-dashboard.md          编排报告总览（脊柱）：只放关键信息 + 状态 + 裁决；主 agent 唯一写
├── scenario.md                场景记录（scenario subagent 产出，§1 链接）
├── analysis/                  各候选来源的分析产物（一来源一文件，文件名取 §5 表的来源名）
│   ├── <source>.md             一来源一文件，来源名见 SKILL §5 表
│   └── …
├── plans/                     每个 Plan 一份完整记录（方案描述 / 细节 / 实施 / review / 派生），不内联到 dashboard
│   ├── plan-A.md
│   └── …
└── perf/                      perf-breakdown 的工作目录（采集 + 分析的 profile 与报告全在这；内部结构和文件名归 model-infer-perf-breakdown，本流程不感知）
```

此外，`progress.md`（共享状态文件）不在本 case 树内——位置由主 agent 解析、找不到就创建，是跨 agent 共享上下文 + subagent 实施 / 踩坑 / 验证过程日志（沿用本插件 `progress_template.md` 的常驻区 / 工作区 / 归档约定）；编排器读它取上下文、往工作区写过程。

引用关系（一份分析派生多个 Plan，不重复）：

```text
baseline 性能报告 ──证据──► analysis/<source>.md ──派生──► plans/plan-A.md / plan-B.md / …
（时间分布 + 理论 gap）        （一份分析，多 Plan 共享）       每 Plan 一份完整记录
```

`perf/` 的内部结构由 `model-infer-perf-breakdown` 自己管，本流程只把它当工作目录、读它产出的报告，不重定义它的契约。

## 分层与写者

分**总览**和**明细**两层，每个 Plan 一份独立明细文件，dashboard 只做总览：

- **dashboard（总览，主 agent 唯一写）**：本文件 §1–§7，只放关键信息 + 状态 + 裁决——场景与基线、Baseline Profiling、候选发现记录、Plan Dashboard 表（每 Plan 一行：plan_id / 来源 / 优化类型 / skill / 互斥组 / 可叠加 / 状态 / round / 证据摘要 / 淘汰·保留原因 / plan 文件链接）、当前采纳的实现、Round 记录、最终验收。结构固定、与方向无关。
- **明细（每个 Plan 一份 `plans/plan-<id>.md`，领域 subagent 产出）**：方案描述、方案细节（领域自由块）、实施记录、Review 记录、派生记录全写在这里，**不内联到 dashboard**。结构由对应单点 skill 定义、主 agent 不解析。candidate 落初稿（方案描述 + 方案细节），该 Plan 的 implementer / reviewer 在循环里追加记录。
- **方向级分析产物**：module/op DAG、候选等，一个来源一份 `analysis/<source>.md`，被该来源所有 Plan 共享，由 candidate 产出（见 §3）。

**边界契约**：plan 文件里写多细由领域 skill 自己发挥；但每个 Plan 必须把编排裁决需要的少数信息**上浮到 dashboard 表**——是否真实生效、收益（性能以分析报告为准）、精度、enable / 回退、副作用，汇成「证据摘要」+ 状态。

**写者规则**：dashboard（§1–§7）只有主 agent 写；`plans/plan-<id>.md` 由该 Plan 的 candidate / implementer / reviewer 产出与追加。主 agent 读 plan 文件做裁决、把关键信息 + 状态镜像进 dashboard 表，不改 plan 文件的内层结构；subagent 不写 dashboard。

**dashboard / progress.md / plan-<id>.md 分工**：dashboard 只管**状态指示**（Plan 状态、采纳实现、round 裁决），是 Plan 状态的单一真相源。跨 agent 共享上下文与 subagent 实施 / 踩坑 / 验证过程落 `progress.md`（共享状态文件，路径由主 agent 解析、找不到就创建，不在本 case 树内，沿用本插件 `progress_template.md` 的常驻区 / 工作区 / 归档约定）；每个 Plan 的方案 spec 与 round 级结论摘要落 `plans/plan-<id>.md`。三者各司其职、不互相复制：看状态查 dashboard、看过程与共享上下文查 progress.md、看单个 Plan 的方案查它的 plan 文件。

## 1. 场景与基线

- 模型 / case：
- 代码目录：
- 推理入口：
- 推理场景：
- 性能目标（可选）：
- 精度 / 功能口径：
- 场景记录文件：
- 输入样本 / 构造脚本：
- 基线运行命令：
- 基线精度 / 功能结果：

## 2. Baseline Profiling 与性能分析（round0）

本节对应 round0：采集由 profiling-instrumenter subagent 做（非交互）；分析由 profile-analyzer subagent 跑（非交互），其拆解 spec 由主 agent 在派发前与用户交互敲定。是后续所有 round 做同口径对照的基准。

- profiling 采集命令 / 产物路径：
- perf-breakdown 工作目录：
- 性能分析报告路径：
- 关键瓶颈摘要（时间分布 top 模块 + 实测/理论 gap top 算子）：
- 热点模块 / 文件 / 算子：
- 数据质量或采集风险：

## 3. 候选发现记录

候选由多个 candidate subagent 并行产出，来源在 SKILL §5 候选发现表注册。每个来源一行（`<source>` 取 §5 表的来源名）。候选之间的互斥 / 叠加关系留到第 4 节由主 agent 裁定；实施用哪个单点 skill 在第 7 步按 Plan 内容定，不在此列。

| 来源 | 分析产物 | 候选数量 | 摘要 |
| --- | --- | --- | --- |
| <source> | analysis/<source>.md |  |  |

## 4. Plan Dashboard

每个 Plan 一行，只放关键信息 + 状态；明细在 `plan 文件` 链接的 `plans/plan-<id>.md`。

| plan_id | 来源 | 优化类型 | 单点 skill | 互斥组 | 可叠加 | 状态 | round | 证据摘要 | 淘汰 / 保留原因 | plan 文件 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Plan-A | candidate-1 |  |  | group-1 | yes/no | 待实现 |  |  |  | plans/plan-A.md |

字段说明：

- `plan_id`：稳定 ID，派生新 Plan 时递增，不复用旧 ID。
- `来源`：候选发现 subagent，或 `derived-from Plan-X/roundN`。
- `优化类型`：多流、融合算子、prefetch、图模式、KVCache、量化、并行等。
- `单点 skill`：实施该 Plan 用的单点技术 skill，由主 agent 在第 7 步按 Plan 内容判断填入（多流 Plan 已知是 `model-infer-multi-stream`，不是编排器）；候选发现阶段可留空。
- `互斥组`：跨方向的互斥关系由主 agent 在初始化时裁定；同组内最终只保留一个或一组兼容的 `通过` Plan，互不影响的写不同组。
- `可叠加`：填 `yes` 或 `no`；标 `yes` 的 Plan 即便各自通过，仍要在验收时验证叠加效果。
- `状态`：只允许 `待实现` / `通过` / `淘汰`。
- `round`：当前状态对应的全局 round 编号。
- `证据摘要`：只写可回查的摘要（从 plan 文件上浮），不贴长日志或 raw 表。
- `淘汰 / 保留原因`：一句话写核心原因；若派生了新 Plan，写明派生目标。
- `plan 文件`：该 Plan 的完整记录 `plans/plan-<id>.md`。

## 5. 当前采纳的实现

当前 `通过`、构成最终方案的 Plan 集合（随通过 / 淘汰 / 替代实时更新；最终验收时即最终采用方案）。互斥组里只列最终保留的那个，可叠加的并列。

| plan_id | 优化类型 | 单点 skill | 采纳的实现（代码改动 + enable 开关） | 收益摘要 | plan 文件 |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

## 6. Round 记录

只做索引：每个 round 选了哪个 Plan、是否重采、裁决结论；本轮的实施 / review / profile 明细写在对应 `plans/plan-<id>.md` 里。

| round | plan_id | 是否重采 profiling | profile 产物目录 | 主 agent 裁决 |
| --- | --- | --- | --- | --- |
| round1 |  | 是 / 否（沿用 roundX） |  |  |

## 7. 最终验收

进入本节前，先对最终代码路径重采一次 profiling，与 round0（baseline）做同场景、同口径对照。

- 是否已无 `待实现` Plan：
- 是否再无可派生的、仍有潜在收益的新 Plan：
- 所有 `通过` Plan 是否都已应用到最终代码：
- 可叠加 Plan 的组合效果是否已实测验证：
- 同一互斥组是否只保留最终采用的 Plan：
- 所有 `淘汰` Plan 是否已回退或被开关关闭：
- 最终精度 / 功能是否仍满足第 1 节的场景口径：
- 最终性能是否按 baseline 同口径对照（以 profile-analyzer 报告为准）：
- 最终采用方案：
- 淘汰方案摘要：
- 剩余风险：

---

每个 Plan 的完整记录 `plans/plan-<id>.md` 结构见 [`plan-file-template.md`](plan-file-template.md)。
