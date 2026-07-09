# baseline 之上的探索式优化编排 · 工作流

本工作流由 `model-infer-sota-approach` plugin 的 primary agent 在每次收到「baseline 之上的推理优化」请求时强制 Read，并严格按其推进。它在一个**已可运行、可复现精度的 baseline** 之上，由 profiling 数据驱动，在多个不确定方向上并行发现候选，再用 Plan 自循环（实施 → 复核 → 派生 → 淘汰）逐步逼近最优；primary 只编排不替工，具体优化下沉调用 `model-infer-*` 单点 skill。

> 路径约定：本文档中 `cann-recipes-infer/...` 指 `init.sh` clone 到插件目录的参考仓；引用的模板见 `templates/`、操作细则见 `references/`。
>
> Subagent 映射：文中的 scenario / profiling-instrumenter / profile-analyzer / candidate / implementer / reviewer 分别对应本插件 subagent `model-infer-sota-scenario` / `model-infer-sota-profiling-instrumenter` / `model-infer-sota-profile-analyzer` / `model-infer-sota-candidate` / `model-infer-sota-implementer` / `model-infer-sota-reviewer`，由 primary 按对应 `subagent_type` 派发；各自的角色定义见 `agents/`。

## 流程总览

```text
+--------------------------------------------------------------+
| 1. 确认推理场景与性能目标                                     |
|    性能目标可选；必须锁定要优化的真实推理场景                 |
+-----------------------------+--------------------------------+
                              |
                              v
+--------------------------------------------------------------+
| 2. 构造推理输入并跑通精度基线                                 |
|    拉 scenario subagent，生成输入、跑通基线、记录精度口径     |
+-----------------------------+--------------------------------+
                              |
                              v
+--------------------------------------------------------------+
| 3. 采集 baseline profiling (round0)                          |
|    拉 profiling-instrumenter subagent（非交互），可关闭回退  |
+-----------------------------+--------------------------------+
                              |
                              v
+--------------------------------------------------------------+
| 4. 分析 baseline profiling                                   |
|    主 agent 敲定拆解 spec，profile-analyzer subagent 跑分析 |
+-----------------------------+--------------------------------+
                              |
                              v
+--------------------------------------------------------------+
| 5. 候选发现                                                  |
|    为 §5 表每个来源并行拉 candidate subagent                 |
+-----------------------------+--------------------------------+
                              |
                              v
+--------------------------------------------------------------+
| 6. 初始化 Plan Dashboard（主 agent 唯一写者）                |
|    归并候选、裁定跨方向互斥/叠加、定状态与验收口径           |
+-----------------------------+--------------------------------+
                              |
                              v
+--------------------------------------------------------------+
| 7. Plan 实施 / review / 派生循环                             |
|    每 Plan 分配新的全局 roundN；按需重采 profiling           |
|    implementer 用单点 skill 实施，reviewer 复核验收          |
|    两者均可建议派生，状态由主 agent 裁决                     |
+-----------------------------+--------------------------------+
                              |
            +-----------------+-----------------+
            |                                   |
            v                                   v
+--------------------------+        +--------------------------+
| 仍有待实现或可派生 Plan  |        | 所有 Plan 通过 / 淘汰     |
| 回到实施 / review 循环   |        | 进入最终验收             |
+--------------------------+        +--------------------------+
```

## 执行规则

进入本 skill 后，先按下面的"编排步骤"建立 TaskList，再逐阶段推进。每次拉 subagent 前，按 [`references/subagent-prompt-templates.md`](references/subagent-prompt-templates.md) 选好模板、替换占位符。

Plan Dashboard 使用 [`templates/plan-dashboard-template.md`](templates/plan-dashboard-template.md)；状态裁决、互斥/叠加判定、是否重采 profiling 和派生规则见 [`references/decision-rules.md`](references/decision-rules.md)。

### 性能分析栈

profiling 的采集与分析交给两个 skill，本流程只调用、不重复它们的内部契约：

- **`model-infer-profiling`**：采集 profiling 数据（`kernel_details.csv` 等）。
- **`model-infer-perf-breakdown`**：本流程唯一的性能分析入口。一份报告同时给出「时间分布」（哪个模块 / 哪层耗时、在抖）和「逐算子实测 / 理论 gap」（哪些算子离理论 bound 最远）两类证据，直接支撑候选发现。它怎么拆、怎么算理论是它内部的事。

**采集和分析都走 subagent，交互只发生在主 agent 派发前**：`model-infer-perf-breakdown` 需要用户敲定的拆解口径（拆解模块偏好、structure / cluster spec 等）由主 agent 在派 subagent 前先问清、作为 spec 传下去，subagent 据此非交互地跑分析——包括第一次 baseline。采集本身非交互、输出冗长，按"采集走 subagent、主 agent 控上下文"的原则，baseline 和重采 round 的采集都走 profiling-instrumenter subagent。baseline 敲定的 spec 在重采轮复用、不再问；重采轮还与 baseline 做同口径对照得出 Δ%。

### 全局约束（全文只声明一次）

- **subagent 配置继承主 agent**：所有 subagent 的 model、thinking、上下文强度等配置都与主 agent 一致，不降级、不缩短、不切配置。本约束对下文每一个 subagent 都生效，后续不再重复。
- **主 agent 只编排不替工**：主 agent 负责阶段推进、一切用户交互（场景确认、perf-breakdown 拆解 spec 的提问等）、prompt 组装、Dashboard 维护和最终验收，不亲自下场跑采集 / 分析 / 实施。凡 subagent 因不能与用户对话而缺的交互，主 agent 在派发前先问好、把结果作为 spec 传下去；采集与分析（含第一次 baseline）都走 subagent。
- **Dashboard 总览 + Plan 明细分离**：`plan-dashboard.md`（总览，约定路径 `optimization-analysis/<case>/plan-dashboard.md`）只放关键信息 + 状态 + 裁决，**只有主 agent 写**；每个 Plan 的完整记录（方案描述 / 方案细节 / 实施 / review / 派生）写在它自己的 `plans/plan-<id>.md`，由该 Plan 的 candidate / implementer / reviewer 产出，主 agent 不解析其结构、只把关键信息和状态镜像进 Dashboard 表。方向级分析另在 `analysis/<source>.md`。详见 [`templates/plan-dashboard-template.md`](templates/plan-dashboard-template.md) 的"分层与写者"。
- **状态、过程、方案三处分工**：dashboard 只管**状态指示**（Plan 状态机、采纳实现、round 裁决），是 Plan 状态的单一真相源；项目级 `progress.md`（共享状态文件，路径由主 agent 解析、不在本流程钉死）承接**跨 agent 共享上下文 + subagent 实施 / 踩坑 / 验证过程**——常驻区放干活要读的上下文（场景与精度口径、baseline 瓶颈、已通过 Plan 速览、当前 round 目标，主 agent 维护），工作区由各 round 的 subagent 追加过程，其常驻区 / 工作区 / 归档（`progress_history.md`）与读写规则沿用本插件 `templates/progress_template.md` 约定的常驻区 / 工作区 / 归档结构、本流程不重复；`plans/plan-<id>.md` 装每个 Plan 的方案 spec 与 round 级结论摘要。三者互不复制：看状态查 dashboard、看过程与共享上下文查 progress.md、看单个 Plan 的方案查它的 plan 文件。progress.md 找得到就续写，找不到就用本插件 `templates/progress_template.md` 创建一份。dispatch prompt 只给模板字段与路径、**不转述上下文**，subagent 靠读 progress.md 取共享上下文。这套文件也是主 agent 跨上下文压缩重建状态的依据，关键节点即时回写、不攒到收尾。
- **候选先发现，Dashboard 后初始化**：场景、精度、baseline profiling 和候选发现没全部完成前，不要初始化 Dashboard。
- **性能证据统一口径**：一切性能收益判断都以 `profile-analyzer` 产出的分析报告为准，不拿裸 wall-clock 数字当结论依据。
- **roundN 全局递增、不复用**：每一个 Plan、每一档强度、每一次尝试都用独立的全局递增 `roundN` 记录；派生出的 Plan 也分配新的 `roundN`，既不复用旧编号、也不与旧 round 合并。
- **状态只有三种**：`待实现` / `通过` / `淘汰`。
- **不轻易收尾**：必须所有 Plan 都转为 `通过` 或 `淘汰`，且确认再无可派生的、仍有潜在收益的新 Plan，才进入最终验收。

### 0. TaskList 骨架

进入 skill 后先建到候选与 Dashboard 为止的阶段任务。**不要建"进入 Plan 循环"这种一个任务覆盖整个循环的长驻任务**——它会一直停在 in_progress、不反映真实进度：

```text
T1. 确认推理场景与可选性能目标
T2. 拉 scenario subagent 构造推理输入并跑通精度基线
T3. 拉 profiling-instrumenter subagent 采集 baseline (round0) profiling（model-infer-profiling，非交互）
T4. 主 agent 先与用户敲定 perf-breakdown 拆解 spec（structure / cluster / 模块偏好），再拉 profile-analyzer subagent 跑 baseline (round0) 分析，标定 round0 性能结构
T5. 为 §5 表每个候选来源并行拉 candidate subagent 发现候选
T6. 初始化 Plan Dashboard，归并候选并裁定跨方向互斥/叠加
```

T6 之后进入 Plan 循环：**每选定一个 Plan 就新建一组 TR{N} 任务**（N 为全局递增 round 编号），本轮做完即逐项关闭，再为下一个 Plan 或派生 Plan 建下一组——用每轮的任务组表达循环，不用单个长驻任务覆盖：

```text
TR{N}.1  选定当前 Plan，分配 roundN，准备上下文；判断本轮是否需要重采 profiling
TR{N}.2  主 agent 按 Plan 内容定单点 skill，拉 implementer subagent 实施
TR{N}.3  （按需）重采并分析 round{N} profiling
TR{N}.4  拉 reviewer subagent 复核并验收 implementer 的工作
TR{N}.5  主 agent 更新 Plan 状态、记录证据、处理派生建议，决定下一步
```

无待实现 Plan 后，建最终验收任务：

```text
TF1. 确认所有 Plan 均为通过或淘汰
TF2. 确认再无可派生的、仍有潜在收益的新 Plan
TF3. 确认所有通过 Plan 已应用，且可叠加 Plan 的组合效果已验收
TF4. 确认所有淘汰 Plan 已回退或被开关关闭
TF5. 汇总最终方案、淘汰原因与剩余风险
```

### 1. 确认推理场景与性能目标

这一步由主 agent 直接做、是交互式的，详细操作与提问流程见 [`references/scenario-confirm.md`](references/scenario-confirm.md)：先从仓库勘察候选场景、检查前置条件（高阶流程需已有可运行 baseline），再带着候选用结构化提问跟用户敲定。只锁定真实优化场景，不写 Dashboard。需要明确：

- 模型 / case 名称、代码目录和推理入口；
- 要优化的推理阶段或 workload（prefill、decode、图像生成、长序列、batch、并发等）；
- 精度或功能的判定口径；
- 性能目标——用户没给就标为可选，不阻塞后续流程；
- 输出归档目录，例如 `optimization-analysis/<case>/`。

如果用户只给了一个泛化目标，由主 agent 先从仓库的推理脚本、配置和已有报告里找候选场景；确实定不下来再问用户。

### 2. 构造推理输入并跑通精度基线

拉 `scenario` subagent，使用模板里的场景模板，详细操作见 [`references/scenario-setup.md`](references/scenario-setup.md)。它产出可复现的推理输入（或输入构造脚本）、基线推理命令、精度/功能结果，以及场景记录文件路径（`scenario.md`）。主 agent 只读摘要和路径，确认场景能稳定复现后再进入 profiling。

### 3. 采集 baseline profiling（round0）

拉 `profiling-instrumenter` subagent，用 `model-infer-profiling` 为已跑通的场景插入或启用 profiling 并采集——采集是非交互的，和重采轮一样走 subagent，让冗长的采集输出留在 subagent 里。它产出采集入口/开关、采集命令、产物路径和回退方式；采集要保留关闭开关、不污染普通推理路径。**这一轮采到的就是 baseline（round0）profile**，是后续所有 round 做同口径对照的基准。

### 4. 分析 baseline profiling

先由**主 agent 交互问清** `model-infer-perf-breakdown` 需要的拆解口径——主 agent 读该 skill 了解它要哪些字段（拆解模块偏好、structure / cluster spec 等），再与用户敲定，结果作为分析 spec；然后拉 `profile-analyzer` subagent 用该 spec **非交互**跑 round0 分析，产出一份报告，含「时间分布」和「逐算子实测 / 理论 gap + need optimization 清单」两类证据。baseline 敲定的分析 spec 会被后续每个重采 round 复用。

这份 baseline 分析标定了性能结构和优化空间，既是候选发现的主要依据，也是最终验收做同场景、同口径对照的基准。

### 5. 候选发现

候选来源在下表注册。主 agent 为**每一行**并行拉一个 candidate subagent，prompt 都用 [`references/subagent-prompt-templates.md`](references/subagent-prompt-templates.md) 的通用 candidate 模板，按本行的「说明 / 输入 / 产物文件」填占位符。
若有候选来源当前没有挂载关联的工具/skill，则跳过。

| 候选来源（source） | 说明（方法） | 输入 | 产物文件 | 下游衔接 |
| --- | --- | --- | --- | --- |
| multi-stream | 用 `model-infer-multi-stream` 做整网 / 模块 / 算子 DAG 拆解、判并行性，每个并行点派 ≥2 种多流编排候选 | 代码目录、场景记录、baseline 分析 | `analysis/multi-stream.md`（DAG + 候选，结构见 module-decomposition 模板） | 各 Plan 引用本文件的 DAG，implement / review 据此实施 |
| wiki | 查 CANN-Infer-Wiki，找该模型 / 场景适用的优化手段（不限多流）作候选；没挂载就跳过 | 场景记录（模型 / 硬件 / 阶段）、baseline 瓶颈摘要 | `analysis/wiki.md`（候选 + 页面引用 + 适用条件） | 每个候选把相关 wiki 页面 ID 记到对应 Plan 的 md，implement / review 按需 get_page 参考 |
| perf-insight | 读 baseline perf-breakdown 的 insight 产物整理候选：连续 vector 区间 → 融合、冗余搬运 → prefetch、理论偏离 Top → 重点算子 | baseline 报告与 insight 目录、场景记录 | `analysis/perf-insight.md`（候选 + insight 证据） | 各 Plan 引用本文件的 insight 证据 |

所有来源共性（不随来源变，故不进表）：各写自己的产物文件、都**不写** `plan-dashboard.md`；候选草案以摘要回主 agent，第 6 步统一归并、裁定候选间互斥 / 叠加；每个草案含方案描述、预期收益、风险与验证口径、互斥 / 可叠加、推荐优先级。用户给的优化列表作为种子分给对应来源的 candidate，增删合并要说明原因。

### 6. 初始化 Plan Dashboard

候选发现完成后，主 agent 读取 `plan-dashboard-template.md` 初始化 Dashboard 总览（写入约定路径，且只有主 agent 写）。这一步主 agent 要做：

- 归并重复或等价的候选；
- 给每个候选分配稳定的 `plan_id`，初始状态统一为 `待实现`，并为每个 Plan 建一份 `plans/plan-<id>.md`（把 candidate 草案的方案描述、方案细节、参考 wiki 页面落进去）；
- 在 Dashboard 表里为每个 Plan 填一行（关键信息 + 状态 + 链到它的 plan 文件）；
- **裁定候选之间的互斥与叠加关系**（例如两个 Plan 切了同一段 attention），标好互斥组和可叠加性；
- 填写每个 Plan 的优化类型和验收口径；具体用哪个单点 skill 实施留到第 7 步由主 agent 按 Plan 内容定；
- 记录每个候选的来源（哪个 candidate subagent）和支撑它的 baseline profiling 证据。

Dashboard 只放关键信息 + 状态 + 裁决；每个 Plan 的方案描述、方案细节、实施 / review / 派生记录都写在它自己的 `plans/plan-<id>.md`，不内联到 dashboard。

### 7. Plan 实施 / review / 派生循环

从 `待实现` Plan 中选一个，分配新的全局 `roundN`，进入下面的循环：

```text
选定 Plan / 分配 roundN
  -> 主 agent 按 Plan 内容判断实施用哪个单点 skill
  -> 主 agent 判断本轮是否需要重采 profiling（见下）
  -> implementer subagent 用选定的 skill 实施当前 Plan
  -> （需要时）拉 profiling-instrumenter + profile-analyzer 产出 round{N} profile
  -> reviewer subagent 复核并验收 implementer 的工作
  -> 主 agent 按 decision-rules 更新状态、处理派生建议
  -> 仍有待实现或新派生的 Plan，就继续下一 round
```

**实施 skill 由主 agent 按 Plan 内容定**，作为 implementer subagent 的 `领域 skill` 传下去；多流 Plan 用 `model-infer-multi-stream`。reviewer 用同一个 skill 复核。

**是否重采 profiling 由主 agent 判断，不强制每轮都重采。** 在上一轮收尾、选下一个 Plan 时，按以下依据决定：

- 上一轮改动是否改变了**算子下发时序、计算图结构、dtype 或 layout**——多流、融合、图模式、量化通常会改，需要重采；纯开关或参数微调这类只做对照的改动，可沿用现有 profile；
- reviewer 是否反映现有 profile 已不足以支撑收益判断；
- 是否切到了新方向或新的热点模块；
- 是否要与 baseline 做同口径对照——**最终方案验收前必须重采一次**。

需要重采时，用 `profiling-instrumenter` + `profile-analyzer` 两个 subagent 产出本轮 profile——它们复用 baseline 的分析配置、不再和用户交互；不需要时直接沿用现有数据。无论是否重采，性能判断都以 profile-analyzer 的分析报告为准。详细判据见 [`references/decision-rules.md`](references/decision-rules.md) 的"round 之间是否重采 profiling"。

implementer 可以在发现更合理方案时**建议**派生，但不得覆盖当前 Plan；reviewer 负责复核验收，可以建议派生，但不改代码、不回退代码。派生出的新 Plan 一律分配新的 `roundN`、初始状态 `待实现`，追加而不合并。

每轮收尾，implementer / reviewer 把实施 / 踩坑 / 验证**过程**写进 `progress.md` 工作区，把 round 级**结论摘要**（含方案细节 spec、派生建议）落该 Plan 的 `plans/plan-<id>.md`；主 agent 读它做裁决，只把关键信息 + 状态镜像进 Dashboard 表（证据摘要、状态、round、淘汰/保留原因），并同步更新「当前采纳的实现」和「Round 记录」。Dashboard 不展开明细。

### 8. 最终验收

满足以下条件才进入最终验收：Dashboard 已无 `待实现` Plan；每个 Plan 都有 implementer/reviewer 记录或明确的跳过/淘汰证据；主 agent 判断再无可派生的、仍有潜在收益的新 Plan。

最终验收前先重采一次 profiling，与 round0（baseline）做同场景、同口径的对照，然后逐项确认：

- 所有 `通过` Plan 都已应用到最终代码路径；
- 可叠加 Plan 的组合效果已经实测验证；
- 同一互斥组内状态自洽（最终只保留一个或一组明确兼容的 `通过` Plan）；
- 所有 `淘汰` Plan 已回退或被开关关闭，不影响最终 profile；
- 最终精度/功能仍满足第 1 步确认的口径；
- 最终性能用与 baseline 同一场景、同一指标口径，并以 profile-analyzer 的报告为准。

## 参考文档索引

- **场景确认操作细则（主 agent 交互：勘察候选 + 前置检查 + 结构化提问）**：[`references/scenario-confirm.md`](references/scenario-confirm.md)
- **Subagent prompt 模板**：[`references/subagent-prompt-templates.md`](references/subagent-prompt-templates.md)
- **scenario 操作细则（构造输入 + 跑通基线 + 定判定口径）**：[`references/scenario-setup.md`](references/scenario-setup.md)
- **Plan Dashboard 模板**：[`templates/plan-dashboard-template.md`](templates/plan-dashboard-template.md)
- **Plan 文件模板（plans/plan-<id>.md）**：[`templates/plan-file-template.md`](templates/plan-file-template.md)
- **状态裁决与 round 推进规则**：[`references/decision-rules.md`](references/decision-rules.md)
