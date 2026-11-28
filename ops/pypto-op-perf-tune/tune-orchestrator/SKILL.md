---
name: tune-orchestrator
description: PyPTO 算子性能调优编排器。控制调优流程按固定顺序严格执行、迭代轮次完整执行、Todo 清单实时更新。负责流程推进、完成条件校验、状态机转移，不负责具体优化建议或代码修改。触发：编排 PyPTO 算子性能调优流程、推进调优迭代轮次时使用。
---

# PyPTO 算子性能调优编排器

## 调优入口

正式调优从性能分析 `S2_COLLECT → S3_ANALYZE` 开始，采集性能数据、建立基线并定位瓶颈。分析前完成 `INIT` 的目标和用例准备，并通过 `S1_SETUP` 的环境与精度门禁。分析后按[三级状态机](#三级状态机)和本次轮次决定是否进入 `S4_TUNE`，最后在 `S5_REPORT` 交付结果。

- **新任务**。先确认算子目录和量化目标。按 INIT 准备主测用例、覆盖集和 `run_cases.py`，再进入 S1a 检查环境。S1b 使用已准备的运行入口验证精度。用调用方提供的配置或技能默认值补齐可推导的选项。
- **续跑**。读取当前任务的 Todo、阶段交接和已有测试产物，核对验收记录后从首个尚未完成的阶段继续。记录不足时，确认影响下一步的进度和缺失输入。
- **方案讨论**。以用户确认的状态为起点，说明接下来各阶段要取得的结果、推进条件和最终交付物。把实际验证留给执行阶段。

### 编排与执行

阶段执行角色根据调优点清单提出改动、实施并实测。编排器按 [Todo 管理规范](#todo-管理规范)核验证据，在每轮 `ITER_RECORD` 和阶段转移时更新进度、判定下一步。

### 调优轮次

`tune_rounds` 由调用方传入，缺省为 3。设为 0 时，完成环境与精度检查、基线采集和分析后直接进入 S5 回归并出报告。正轮次在进入 S4 前准备代表用例集、Tile 边界清单和 A4 议程，按[结构变更台账门控](#phase_summary_f-结构变更台账门控)分配外层预算。

### ⛔ 禁止自主判断铁律

**agent 不得自主判断"该优化什么"或"是否可以停止了"，所有决策必须由以下三个依据驱动：**

| 决策 | 唯一依据 | 禁止的行为 |
|------|---------|-----------|
| 选择优化点 | **调优点清单**（Phase A/B 分析产出） | 禁止凭"感觉"或"经验"选择优化点 |
| 是否继续下一轮 | **退出条件表**（达标/连续无提升阈值/清单全尝试） | 禁止凭"觉得差不多了"提前终止 |
| 是否进入下一阶段 | **PHASE_SUMMARY 摘要完整性校验** | 禁止不做摘要直接推进 |
| 是否跳过某个优化点 | **调优点清单中的分析结论**（如"❌不适用"须标注原因） | 禁止跳过未尝试的优化点 |
| 是否开始下一轮外循环 | **未达标且有剩余预算** | 禁止跳过剩余轮次 |

**⛔ 所有流程推进必须基于状态机转移条件的硬性判断，禁止用"我认为"替代条件检查。**

---

## 三级状态机

调优流程按三级状态机推进。迭代回退和 FRONTEND 重审按各自的转移条件执行。

### 第一级：主流程状态机

```
INIT → S1_SETUP → S2_COLLECT → S3_ANALYZE → S4_TUNE（正轮次）→ S5_REPORT → DONE
```

### 第二级：S4_TUNE 子阶段状态机

```
FRONTEND → PHASE_SUMMARY_F → 按台账重审或开始下一轮
每轮 SWIMLANE_n → PHASE_SUMMARY_S_n → INCORE_n → PHASE_SUMMARY_I_n
任一摘要达标，或需要开始下一轮时预算已耗尽 → S5_REPORT
```

每个 PHASE_SUMMARY 都是交接点。校验本阶段的实测和台账，将剩余预算及下一阶段记入 `MEMORY.md`。

### 第三级：每个 PHASE 内部的迭代循环

```
每个 PHASE 内部反复执行以下循环，直到退出条件满足：

┌──────────────────────────────────────────────────────────┐
│  ITER_START（选择优化点）                                  │
│      │                                                   │
│      ▼                                                   │
│  ITER_MODIFY（修改代码，单参数）                            │
│      │                                                   │
│      ▼                                                   │
│  ITER_VERIFY（验证精度）                                   │
│      │                                                   │
│      ├─ 精度失败 → ITER_ROLLBACK（回退代码）               │
│      │                   │                                │
│      │                   ▼                                │
│      │               ITER_RECORD（记录失败）               │
│      │                   │                                │
│      │                   ▼                                │
│      │               ITER_JUDGE（判断退出条件）            │
│      │                   │                                │
│      ├───────────────────┘                                │
│      │                                                   │
│      ▼ （精度通过）                                        │
│  ITER_MEASURE（测试性能）                                  │
│      │                                                   │
│      ├─ 性能回退 → ITER_ROLLBACK → ITER_RECORD            │
│      │                   │                                │
│      │                   ▼                                │
│      │               ITER_JUDGE（判断退出条件）            │
│      │                   │                                │
│      ├───────────────────┘                                │
│      │                                                   │
│      ▼ （性能提升或持平）                                   │
│  ITER_RECORD（记录结果，更新Todo）                          │
│      │                                                   │
│      ▼                                                   │
│  ITER_JUDGE（判断退出条件）                                │
│      │                                                   │
│      ├─ 未达标且轮次未耗尽 → 回到 ITER_START               │
│      │                                                   │
│      └─ 达标或轮次耗尽 → 退出迭代循环                    │
│         （进入对应的 PHASE_SUMMARY 状态）                  │
└──────────────────────────────────────────────────────────┘

⛔ 所有 ROLLBACK 路径（精度失败/性能回退）统一经过：
  ITER_ROLLBACK → ITER_RECORD → ITER_JUDGE
  ITER_JUDGE 统一判定：失败是否超限、是否继续下一轮。
```

---

## 全状态定义与转移条件

### 第一级状态转移表

零轮使用已有测试完成精度检查，并采集和分析基线，直接进入 S5。下表的代表用例集、`run_cases.py`、S2 详细泳道采集和 S4 调优门禁适用于正轮次。

| 当前状态 | 完成条件 | 转移到 | 强制动作 | 独立编排的 state_transition 门控 |
|---------|---------|--------|---------|------------------------|
| INIT | 用户已提供目标（如"提升X倍"/"≤X us"）或用户确认 | S1_SETUP | 记录目标到 Todo，计算目标执行时间；⛔ **确定 tuning case + covering set + 创建 run_cases.py**（见下方「covering set + runner 创建」），写入 Todo 并传入后续所有 S4 dispatch | 完成时调用 `state_transition(opDir, "complete_stage", 0)` |
| S1_SETUP | 环境检查全部通过（见下方「环境检查清单」）+ 精度通过 | S2_COLLECT | ⛔ S1a 环境检查：逐项检查环境；⛔ S1b 精度校验：环境通过后运行 `run_cases.py --cases all`（精度模式，全 covering set）→ 验证 runner + 建立 per-case 精度基线；⛔ 全部通过后强制创建完整 Todo | S1a通过时调用 `state_transition(opDir, "start_stage", 1)`，全部通过后调用 `state_transition(opDir, "complete_stage", 1)` |
| S2_COLLECT | swimlane.json 存在 + ⛔ S-20 检查完成（max_workspace_kb 已按 NPU 推荐值设置，host_options 已配置） | S3_ANALYZE | ⛔ S2 阶段首次运行时检查 stdout 中 "Recommended: set max_workspace_kb near XXXX KB" 提示，提取推荐值设置到 runtime_options；同时设置 host_options={"compile_monitor_enable": 0}；**tuning case 全 swimlane merge（重量级，S3 详细分析用）；covering set 其余 case 用 `run_cases.py --cases <non-tuning> --perf`（按同一计时口径采集逐用例基线）** | 完成时调用 `state_transition(opDir, "start_stage", 2)` → 验证后调用 `complete_stage(2)` |
| S3_ANALYZE | 性能报告文件存在 | 正轮次进入 S4_TUNE，零轮进入 S5_REPORT | 记录基准性能：tuning case 详细 bubble/core/load 分析（from swimlane）+ covering set per-case E2E 基线表（tuning case from swimlane summary, 其余 from 各 case 的性能日志）→ 写入 MEMORY 作为 S4 ITer 比较基线 | 完成时调用 `state_transition(opDir, "start_stage", 3)` → 验证后调用 `complete_stage(3)` |
| S4_TUNE | 任一 PHASE_SUMMARY 实测达标，或外层预算耗尽 | S5_REPORT | ⛔ 各PHASE间通过SUMMARY强制交接；⛔ 每轮外循环从SWIMLANE_n入口重新独立采集性能数据 | 每个 PHASE 完成时调用 `state_transition(opDir, "complete_stage", 4)` |
| S5_REPORT | 报告文件已保存 | DONE | ⛔ 还原 debug_options | 完成时调用 `state_transition(opDir, "complete_stage", 5)` |

### 第二级状态转移表（S4_TUNE 内部）

**⛔ 正轮次的 covering set + runner 创建（INIT 时，与 tuning case 同时确定）**：

> 目的：tile/unroll 类优化（F-8/F-9/F-10/F-11/F-16/F-17/F-20/S-11/S-12）改变**编译期分支**——单 tuning case 实测无法覆盖其他分支取值的编译正确性与性能。covering set + run_cases.py runner 是这些优化的验证基线（分支验证 + 采纳决策见 basic-block-optimization.md §1b）。

**步骤 1：确定 covering set（规则驱动，最小集）**：

不从 cases.yaml 枚举"每个参数的每个取值"（过冗余）——按以下规则选出 **2-3 个精度 case + 编译期补漏**的最小集，覆盖结构极端值而非穷举：

1. **锚点 case** = tuning case（主性能 case，必须在集合内——perf 信号须在其上）
2. **边界 regime** = 与锚点 qualitatively 不同的 loop_count regime（如 decode 类（loop_count=1）vs prefill 类（loop_count 大））
3. **覆盖 2 个最内层 loop + 每个 if-statement 的分支**：选一个 case 覆盖锚点未覆盖的 layout/causal/case 标量分支值（结构补集 case，如锚点用 layout A + flag=True，补集 case 用 layout B + flag=False）
4. **编译期补漏（compile-only gap-fill）**：未被精度集覆盖的等价类分支（如某 case 标量=8, 某 case 标量=512, 某 loop_count=2/4）→ 用 `--compile-only` 快速检查（不做精度/性能实测），只验证编译正确性
5. **每次 INIT 从当前 impl 的分支结构重新推导**（规则，不硬编码 case 列表）——分支结构随调优变化时，集合相应扩展

**步骤 2：创建 run_cases.py（多 case 验证 runner）**：

在算子目录创建 `run_cases.py`，以该算子的精度测试入口为基础。

runner 规格：
- **导入** `make_case_inputs`, `<op>_wrapper`, `<op>_golden`, `detailed_tensor_compare` 从现有 `test_<op>.py`（复用 env/path/compare 逻辑）
- **CASES 表**：covering set 的 case_id → kwargs 字典（匹配 `_run` 签名；seed = SHA256(case_id)，对齐 `eval/test_inputs.py`）
- **模式**：
  - 默认（精度）：逐 case 调 `_run`（golden compare）；per-case `try/except`（不 raise）；输出 `[case N] layout=.. causal=.. -> PASS/FAIL` + `OVERALL: n/N` + exit code；`[PRECISION_PASS]` 仅全通过时打印
  - `--compile-only`：wrapper 调两次（t1=compile+exec, t2=cached exec, compile≈t1−t2）；跳过 golden；输出 `[case N] compile 63s exec 0.9s -> OK`
  - `--perf`：为每个 case 单独启动进程，在独立目录启用 PyPTO 设备性能采集。使用 [perf-analyzer](../perf-analyzer/SKILL.md) 读取新生成的 `bubble_analysis.log`，取所有核心的 `Core Total Work Time` 最大值。输出 `[case N] E2E XXXus` 并记录日志路径。采集缺失或 E2E 无效时，该次测量失败。
- **选择器**：`--cases <id1>,<id2>,<id3>`（默认 = all）；`--l1-only` 保留为 `--cases <tuning-case-id>` 别名（S2 依赖）

**步骤 3：验证 runner（INIT 退出条件）**：
1. `py_compile run_cases.py` 通过
2. `run_cases.py --cases <smallest> --compile-only` 通过（~2 min smoke）
3. **交叉验证（一次性）**：对 tuning case 运行 `run_cases.py --cases <tuning-case-id> --perf`，与同次采集的 `AICore Prof Summary` 中 `AICore End-to-End Time` 核对，按摘要输出精度比较，并将结果写入 MEMORY。

记录到 Todo + MEMORY：`tuning_case_set: [case_id, ...]`（精度集）+ `compile_only_gap_fill: [case_id, ...]` + runner 路径 + 交叉验证结果。每次 S4 dispatch 摘要须包含精度集 + 编译期补漏清单。

**各 stage 使用方式（S4/S5 — S1b/S2/S3 见上表强制动作列）**：
- **S4 迭代**：修改 tile/unroll 后，先对受影响的用例做精度验证，并对未覆盖的编译分支用 `--compile-only` 补漏。通过后用 `--perf` 逐用例对比 S3 基线，保存 `[case N]` 输出。按[采纳决策策略](../shared/optimization_catalog.md)处理改善和回退分歧。
- **S5 / regression**：正轮次运行 `run_cases.py --cases all` 回归代表用例集。零轮沿用原精度测试。

独立编排时，主会话负责表中 `state_transition` 调用，并在阶段交接前完成登记。具体派工方式见[独立编排的 Task 交接](references/phase-handoff.md#独立编排的-task-交接)。

| 当前 PHASE | 进入动作 | 退出条件 | 转移到 | 独立编排的 state_transition 门控 |
|-----------|---------|---------|--------|------------------------|
| PHASE_FRONTEND | 加载 tune-frontend 子技能 + ⛔ 执行阶段A(全局分析:Loop+常量+Reshape+TileShape) + 阶段B(局部分析:数据操作) + ⛔ 编排器核查分析制品 + 生成优化点清单 | ⛔ 见「退出条件详细定义」 | PHASE_SUMMARY_F | 退出时调用 `state_transition(opDir, "start_stage", 4)` |
| PHASE_SUMMARY_F | 生成并核验 FRONTEND 摘要、达标状态和结构变更台账 | 摘要及台账已校验 | 达标或预算耗尽→S5_REPORT / 未达标且有预算→按下方台账门控重审或进入 SWIMLANE | 派工前调用 `state_transition(opDir, "complete_stage", 4)` |
| PHASE_SWIMLANE_n | ⛔ **外循环第{n}轮**：独立采集性能数据(S2型) → 独立分析泳道图(S3型) → 加载 tune-swimlane 子技能 → ⛔ **核使用率分析** → 生成调优点清单 | ⛔ 见「退出条件详细定义」 | PHASE_SUMMARY_S_n | 退出时调用 `state_transition(opDir, "complete_stage", 4)` |
| PHASE_SUMMARY_S_n | 生成 SWIMLANE 阶段交接摘要 + 检查达标状态 + 按序列推进（见「阶段交接流程」） | 摘要已生成 | 达标→S5_REPORT / 未达标→PHASE_INCORE_n | - |
| PHASE_INCORE_n | ⛔ **外循环第{n}轮**：独立采集性能数据(S2型) → 独立分析性能数据(S3型) → 加载 tune-incore 子技能 → ⛔ 生成调优点清单 | ⛔ 见「退出条件详细定义」 | PHASE_SUMMARY_I_n | 退出时调用 `state_transition(opDir, "complete_stage", 4)` |
| PHASE_SUMMARY_I_n | 生成 INCORE 阶段交接摘要 + 检查达标状态 + 按序列推进（见「阶段交接流程」） | 摘要已生成 | 达标或预算耗尽→S5_REPORT / 未达标且有剩余预算→PHASE_SWIMLANE_{n+1} | - |

### PHASE_SUMMARY_F 结构变更台账门控

1. 根据实测耗时和性能目标判定是否达标。达标后进入 S5，保留最终精度回归。
2. 未达标时，检查 FRONTEND 台账是否记录了本轮采纳且尚未重审的结构级变更，以及首轮基线中尚未复核的结构标记。缺少台账或证据时，先补齐阶段交接。
3. 有待复核的结构变更时，使用一轮剩余预算重跑 FRONTEND。清除旧的 `DO-NOT-RETRY` 标记，重做 A1–A4 并重新评估 F-1 至 F-21。重审完成后标记 `audited`。
4. 待复核项清空后，有剩余预算才进入 SWIMLANE。一次 FRONTEND 重审和一轮 SWIMLANE→INCORE 各消耗一轮外层预算。预算耗尽时进入 S5 并记录未解决项。

### 第三级状态转移表（迭代轮次内部）

⛔ 这是调优过程中最核心的控制，每一步都是独立操作，必须逐步执行。

| 迭代子状态 | 执行内容 | 完成标志 | 转移到 | 失败处理 |
|-----------|---------|---------|--------|---------|
| ITER_START | 根据子技能阶段A/B分析结论和 [shared/optimization_catalog.md](../shared/optimization_catalog.md) 选择一个优化点。⛔ 必须按调优点清单选择下一个未尝试的优化点，禁止凭直觉选题。⛔ FRONTEND阶段：优化点必须引用阶段A或B分析表格中的具体行号。⛔ SWIMLANE/INCORE阶段：优化点必须引用前置分析结果中的具体数据。**无法写出"依据：XX表第N行"或"依据：[分析数据]" → 前置分析未完成 → 回退补充，禁止进入 ITER_MODIFY。** ⛔ 退出前提：只有调优点清单中所有项均已标记为 ✅已尝试 或 ❌已失败 或 ❌不适用(须注明原因) 时，才允许"无优化点可选→退出迭代循环"。**清单中存在未标记项时，禁止以"无优化点可选"为由退出，必须逐一尝试所有标记为 ⏳待尝试 的项。** | 确定了要改什么参数（含编号+分析依据） | ITER_MODIFY | 无优化点可选 **且 清单已全部标记** → 退出迭代循环（进入对应 PHASE_SUMMARY） |
| ITER_MODIFY | 修改代码，只改一个参数 | 代码已修改 | ITER_VERIFY | - |
| ITER_VERIFY | 按[主技能步骤 1.2](../SKILL.md)逐个上板运行所需用例，记录验证时间、命令、结果和关键输出 | 每个用例均正常退出，输出含 `passed` 或 `success`，且无 `failed`、`error` 或 `timeout` | 通过后进入 ITER_MEASURE | 按[精度失败流程](#迭代轮次精度失败iter_verify)简单分析，仍无法通过则立即回退该项 |
| ITER_MEASURE | 运行用例采集性能数据 | 新的性能数据已获取 | ITER_RECORD | → ITER_ROLLBACK |
| ITER_RECORD | 更新 Todo 性能记录表 | Todo 已更新 | ITER_JUDGE | - |
| ITER_JUDGE | 检查退出条件 | 判定结果 | ITER_START 或 退出迭代循环（进入对应 PHASE_SUMMARY） | - |
| ITER_ROLLBACK | 回退代码修改 | 代码已恢复 | ITER_RECORD | - |

### S1_SETUP 结构

S1_SETUP 包含两个子阶段：
1. **S1a：环境检查** — 执行主技能步骤 1.0 环境检查流程，⛔ 不可跳过
2. **S1b：精度校验** — 环境检查全部通过后，执行主技能步骤 1.1-1.2 精度校验流程

两个子阶段全部通过后，转移到 S2_COLLECT。

### ⛔ 通用原则：SWIMLANE/INCORE 进入时独立采集与分析

**⛔ ⛔ ⛔ 强制门控：PHASE_SUMMARY_F/S/I 之后的所有 PHASE（SWIMLANE_n、INCORE_n）进入时，必须重新采集性能数据并重新分析。禁止复用上一阶段或 S2_COLLECT 的数据！⛔ ⛔ ⛔**

**PHASE_FRONTEND 之后的 SWIMLANE 和 INCORE 每个外循环轮次进入时，都必须独立进行性能数据采集和泳道图分析。** PHASE_FRONTEND 直接复用 S2_COLLECT + S3_ANALYZE 的数据，无需重复采集。原因：

1. **性能特征逐轮变化**：SWIMLANE_n 基于 INCORE_{n-1} 修改后的代码运行，执行时间、核心利用率、气泡率等指标已显著改变，上一轮数据无法反映当前实现的实际表现
2. **决策依赖最新数据**：核使用率分析、瓶颈识别、合图策略等所有优化决策必须基于当前代码的泳道图数据
3. **NOT FULL 判定敏感**：TileShape 优化会改变任务数，进而改变核使用率。每次回到核使用率分析前都必须重新采集

**入口数据采集与分析的通用流程**（每个外循环轮次的 SWIMLANE_n 和 INCORE_n 入口执行）：

```
Step 0: 独立性能数据采集
  ├─ 确保 debug_options={"runtime_debug_mode": 1}
  ├─ 运行算子用例（python3 ... --run-mode npu）
  └─ 确认 swimlane.json 已基于当前代码生成

Step 0a: 独立性能数据分析
  ├─ 记录本轮入口基准：执行时间、核心利用率、气泡率等
  ├─ 识别当前瓶颈（最大耗时子图、最严重气泡等）
  └─ 将入口基准写入 Todo 作为本轮的 Baseline
```

> **与 S2_COLLECT / S3_ANALYZE 的关系**：S2+S3 提供全局初始基准（Initial Baseline），PHASE_FRONTEND 直接复用此基准。每轮的 SWIMLANE_n/INCORE_n 入口独立采集提供的是**当前轮次基准（Round_n Baseline）**，因每轮间代码已变化，不可复用历史数据。

### 各 PHASE 编排器职责

编排器不介入子技能的具体分析/优化过程，仅负责：加载子技能 → 核查产出制品 → 判断退出条件。

| PHASE | 加载子技能 | 编排器核查的产出制品 | 退出后进入 |
|-------|-----------|-------------------|-----------|
| PHASE_FRONTEND | `tune-frontend` | 阶段A 全局分析制品 + 阶段B 局部分析制品 + 最终优化点排序清单（具体清单见 `../tune-frontend/SKILL.md`）+ ⛔ **结构变更台账**（F-1~F-21 全表标记完整性 + 已采纳未回退优化点的 `结构级`/`配置级` 标注 + 首轮追加入口基线既有结构标记及 `audited` 状态）。⛔ **F-19 为必做项**（两种模式均须检查，详见 `../tune-frontend/SKILL.md` §1.2.4/§1.2.5） | PHASE_SUMMARY_F |
| PHASE_SWIMLANE_n | `tune-swimlane` | 核使用率分析 + 调优点清单（具体清单见 `../tune-swimlane/SKILL.md`）；⛔ 泳道图重采次数记录在 Todo 中 | PHASE_SUMMARY_S_n |
| PHASE_INCORE_n | `tune-incore` | 瓶颈 task 分析 + 调优点清单（具体清单见 `../tune-incore/SKILL.md`） | PHASE_SUMMARY_I_n |

⛔ **核查规则**：子技能声明的所有分析制品均已生成后，才允许进入 ITER 循环。缺少任何一项 → 要求补充分析，禁止跳过进入优化。

SWIMLANE 内因核使用率或负载均衡调整而重采泳道图时，继续当前阶段，退出阶段时再生成 PHASE_SUMMARY。

---

## ⛔ 每次回复前的强制自检

在生成任何回复之前，你必须完成以下自检。如果任何一项未通过，必须先处理该项，禁止继续执行其他操作。

```
□ Q1: 当前处于哪个状态？
      第一级: [INIT / S1 / S2 / S3 / S4 / S5 / DONE]
      第二级: [FRONTEND / SUMMARY_F / SWIMLANE_n / SUMMARY_S_n / INCORE_n / SUMMARY_I_n]（仅 S4 时）
      第三级: [ANALYZE_A / ANALYZE_B / SWIMLANE_STEP0 / SWIMLANE_CORE_USAGE / SWIMLANE_LOAD_BALANCE / START / MODIFY / VERIFY / MEASURE / RECORD / JUDGE / ROLLBACK]
              （FRONTEND: ANALYZE_A/B; SWIMLANE: SWIMLANE_STEP0/SWIMLANE_CORE_USAGE/SWIMLANE_LOAD_BALANCE/ITER; INCORE: START/...）

□ Q1a: 阶段交接时核验 PHASE_SUMMARY
        → 实测已达标：进入 S5_REPORT
        → FRONTEND 未达标：按结构变更台账和剩余预算决定下一阶段
        → SWIMLANE 未达标：继续本轮 INCORE
        → INCORE 未达标：有剩余预算则开始下一轮，否则进入 S5_REPORT

□ Q2: 当前子状态的完成条件是否满足？
       → 不满足 → 继续执行当前子状态，禁止推进

□ Q2a: （仅 FRONTEND 阶段的 ITER_START 子状态时检查）
        FRONTEND 有独立的第三级状态：ANALYZE_A → ANALYZE_B → ITER 循环。
        → ANALYZE_A 未完成（阶段A 分析表未填写）→ 当前状态应为 ANALYZE_A，禁止进入 ITER
        → ANALYZE_B 未完成（阶段B 分析表未填写）→ 当前状态应为 ANALYZE_B，禁止进入 ITER
        → 两者均已完成 → 允许进入 ITER_START

□ Q2b: （仅 SWIMLANE 阶段）当前处于哪个子阶段？
       → SWIMLANE_STEP0（泳道图采集/回流重采）: debug_options 已启用？泳道图重采次数已记录？
       → SWIMLANE_CORE_USAGE（核使用率分析）: analyze_core_usage.py 已运行？
       → SWIMLANE_LOAD_BALANCE（负载均衡分析）: 瓶颈子图已识别？
       → ITER（优化迭代）: 优化点清单已生成？

□ Q3: 本轮回复是否只推进了一个子状态？
       → ITER_START → ITER_MODIFY  ✅（一个）
       → ITER_START → ITER_MODIFY → ITER_VERIFY  ❌（三个，太多了）

□ Q4: 强制动作是否已完成？
        → S1→S2: Todo 已创建？
        → FRONTEND 阶段C 前置: 阶段A+B 分析制品已核查？
        → PHASE→SUMMARY: 阶段交接摘要已生成并校验？
        → S5→DONE: debug_options 已还原？
        → 独立编排时，在交接前核对状态转移表中的 state_transition 门控。

□ Q5: Todo 是否最新？（⛔ 必须按下方规则逐项验证，仅问"是否已更新"不够）
       → 规则1: 刚完成 ITER_MEASURE/ROLLBACK → 迭代轮次记录表中必须有对应轮次的新行
       → 规则2: 刚完成 ITER_JUDGE → 当前迭代状态中的"连续无提升""失败累计"必须已更新
       → 规则3: 刚完成 PHASE_SUMMARY → 阶段状态（✅/⏸️）必须已更新
       → ⛔ 不确定时 → 通过 `bash` 执行 `todowrite` 重新读取 Todo 验证
         如果 `todowrite` 返回的 Todo 中缺少本轮记录 → **必须先补充，禁止推进**

□ Q6: （仅 ITER_START 时）本次选择的优化点是否来源于阶段A/B的分析结论？
        → 无分析结论支撑 → 禁止执行，回到分析阶段补充
        → 有分析结论支撑 → 确认引用了分析表格的具体行号

□ Q6a: （仅 ITER_START 判定"无优化点可选"时）调优点清单是否已全部标记？
        → 检查优化点清单中所有项的"状态"列：不存在 ⏳待尝试 或 空状态
        → 存在未标记项 → **禁止退出！必须按清单逐一尝试未标记项**
        → 所有项已标记（均为 ✅已尝试 / ❌已失败 / ❌不适用）→ 允许退出
```

---

## 迭代轮次执行规范

### ⛔ 单次回复执行边界

每次回复最多执行 **1-2 个相邻子状态**，具体规则：

```
允许的回复粒度（紧耦合的组合允许合并）：

✅ ITER_START + ITER_MODIFY
   → 选优化点 + 改代码（必须知道改什么才能改）

✅ ITER_VERIFY（单独执行，需要等待命令执行结果）

✅ ITER_MEASURE（单独执行，需要等待性能数据）

✅ ITER_RECORD + ITER_JUDGE（记录 + 判断，紧耦合）

✅ ITER_ROLLBACK + ITER_RECORD（回退 + 记录失败，紧耦合）

禁止的回复粒度：

❌ ITER_START → ITER_MODIFY → ITER_VERIFY → ITER_MEASURE
   （一次回复跨越4步，信息量过大无法校验）

❌ 跳过 ITER_VERIFY 直接 ITER_MEASURE
   （精度都没验就测性能）

❌ 跳过 ITER_RECORD 直接下一轮 ITER_START
   （没有记录就继续，Todo 会过时）
```

### ⛔ 单参数原则

逐步推进的限制适用于实际运行和状态转移。评审方案或交接下一轮任务时，给出从修改到留证、采纳或回退的完整操作说明。每轮在同设备、同用例上验证，记录命令、运行时间、退出码和精度标记。精度失败经简单分析仍未解决时立即回退该项，通过后才采集耗时、与基准比较并登记差值。[单变量迭代与精度回归](../SKILL.md#单变量迭代与精度回归)

```
ITER_MODIFY 阶段，每次只允许修改一个优化参数：

✅ 只改 BLOCK_SIZE: 128 → 64
✅ 只改 TileShape: (1,1,128) → (1,1,256)
✅ 只加一个 runtime_options 参数

❌ 同时改 BLOCK_SIZE + TileShape
❌ 同时改 unroll_list + stitch_max_num
❌ 同时加两个新参数
```

---

## ⛔ 退出条件详细定义

### ⛔ 核心原则：禁止询问用户"是否继续"

**退出由本表自动判定，满足任一条件就退出，不满足就继续。禁止向用户询问"是否继续优化"。只有在用户主动说"停"时才触发"用户要求停止"条件。**

### 各 PHASE 的退出条件

**PHASE_FRONTEND 退出条件（满足任一）**：

| 条件 | 判定方法 | 动作 |
|------|---------|------|
| 性能达标 | 当前 AICore E2E Time ≤ 目标值 | → 退出迭代循环 → PHASE_SUMMARY_F |
| 调优点清单已全部尝试 且 连续5轮无提升 | 调优点清单中所有项已标记为 ✅已尝试 或 ❌已失败，且最近5轮无提升 | → 退出迭代循环 → PHASE_SUMMARY_F |
| 用户主动要求停止 | 用户明确说"停止" | → 退出迭代循环 → PHASE_SUMMARY_F |

**PHASE_SWIMLANE 退出条件（满足任一）**：

| 条件 | 判定方法 | 动作 |
|------|---------|------|
| 性能达标 | 当前 AICore E2E Time ≤ 目标值 | → 退出迭代循环 → PHASE_SUMMARY_S |
| 调优点清单已全部尝试 且 连续8轮无提升 | 调优点清单中所有项已标记，且最近8轮无提升 | → 退出迭代循环 → PHASE_SUMMARY_S |
| 用户主动要求停止 | 用户明确说"停止" | → 退出迭代循环 → PHASE_SUMMARY_S |

**⚠️ S-14 Mix合图的轮次特殊规则**：Mix合图（S-14）内部调优的轮次计算、最低尝试要求、退化诊断决策树详见 [merge-optimization.md §4.2.5](../tune-swimlane/references/merge-optimization.md)。Mix合图退化时，禁止跳过退化诊断决策树直接退出 Mix 转向其他优化（如 S-15 ooo_sched_mode / S-16 vf_options）。必须完成诊断决策树全部 4 步（scope 范围→CV 通路→spill→配套参数）且仍无收益，才允许退出 Mix 合图。

**PHASE_INCORE 退出条件（满足任一）**：

| 条件 | 判定方法 | 动作 |
|------|---------|------|
| 性能达标 | 当前 AICore E2E Time ≤ 目标值 | → 退出迭代循环 → PHASE_SUMMARY_I_n |
| 调优点清单已全部尝试 且 达到理论性能上限 | 清单全部尝试，核心利用率 > 80% 且 气泡率 < 10% | → 退出迭代循环 → PHASE_SUMMARY_I_n |
| 调优点清单已全部尝试 且 连续5轮无提升 | 清单全部尝试，最近5轮无提升 | → 检查是否可进入算法级优化（见下方） |
| 用户要求停止 | 用户明确说停止 | → 退出迭代循环 → PHASE_SUMMARY_I_n |

**算法级优化路由（INCORE 收敛后）**：当 INCORE 调优点清单全部尝试且连续5轮无提升、性能仍未达标时，不直接退出到 S5_REPORT，而是检查是否可进入算法级优化（对应 top-level SKILL.md §4.3）：
1. 确认当前性能瓶颈是否为 DDR 往返带宽瓶颈（AIC 等 AIV 产出 / AIV 等前驱时间占比 > 20% E2E）
2. 若是 DDR 瓶颈 → 执行 A-D1~A-D3（合并 gather / view 复用 / 消除中间 assemble），每次只改一个算法点，改后验证精度+测性能。⚠️ A-D1↔I-10、A-D2↔I-11 是同一优化的两个编号，若 INCORE 已尝试 I-10/I-11 则 A-D1/A-D2 标记"已尝试"不重复执行。A-D3（消除中间 assemble 回 GM）无对应 I- 编号。A-D1~A-D3 修复 CV 断点后须重新评估 S-14 Mix合图（回退至 SWIMLANE 下一轮重试 S-14）
3. 若非 DDR 瓶颈 → 执行 A-G1~A-G7（通用算法优化）
4. 算法级优化也收敛（所有 A-D/A-G 项已尝试且连续3轮无提升）→ 退出迭代循环 → PHASE_SUMMARY_I_n
5. 算法级优化有收益 → 继续迭代直到达标或收敛

### ITER_JUDGE 判定流程

**⛔ 此流程由条件自动判定。禁止在条件未触发时询问用户是否继续。满足继续条件就直接下一轮，满足退出条件就自动退出。**

```
ITER_JUDGE 执行时，按以下顺序判断：

0. ⛔ 本轮的终端状态是什么？
   → ITER_ROLLBACK 后进入（编译失败/运行错误/精度失败）: 计入"失败累计"，递增失败累计计数器。**失败不计入"连续无提升"计数器**，但会消耗优化点清单中的一项。
   → ITER_MEASURE 后进入（正常执行）: 继续下方 1-7 判断流程。

0.5. ⛔ TODO 完整性闸门（必须通过才能继续判定）
    刚完成 ITER_MEASURE（正常执行）时，检查迭代轮次记录表：
    → 检查: 最新一轮（轮次编号最大）的"状态"列是否为 ✅（性能提升）或 ❌回退（性能回退）
    → 通过条件: 最新轮次的状态列已填写 ✅ 或 ❌
    → ❌ 未通过（状态列为空或标记为 🔄 进行中）:
       ⛔ 禁止继续判定！必须先执行 ITER_RECORD 更新 Todo，再回到 ITER_JUDGE
       ⛔ 命令: `todowrite` 补充本轮记录

    刚完成 ITER_ROLLBACK（失败分支）时，检查迭代轮次记录表：
    → 检查: 最新一轮的"结果类型"列是否已标记失败原因
    → ❌ 未通过: 禁止继续！先补充记录再判定

    刚完成 ITER_VERIFY 后直接进入的 ROLLBACK 路径，没有 ITER_MEASURE 数据：
    → 结果类型: 编译失败 / 运行错误 / 精度失败
    → 禁止状态列为空就直接退出

1. 性能是否达标？
    当前 AICore E2E Time ≤ 目标值
    → 是: 输出"达标"，退出迭代循环，进入对应 PHASE_SUMMARY
    → 否: 继续

2. 当前轮次是否有提升？
   判定条件（必须同时满足才能接受本轮改动）：
   - 限定性能指标：仅允许使用 AICore E2E 耗时，单位 μs，取值从 `bubble_analysis.log` 文件的 `Core Total Work Time` 字段读取（或通过 `perf-analyzer/scripts/analyze_perf.py` 脚本获取）。
   - 性能正向收益：本轮耗时 < 上一轮耗时，上一轮耗时存储于 Todo 迭代轮次记录表。
   口径说明：本技能全部判定上下文中的"执行时间 / 耗时"均指上述 AICore E2E 耗时（下同，不再重复注明）。
   → 是（本轮耗时 < 上一轮耗时）: 采纳本轮改动，重置连续无提升计数器为 0
   → 否（本轮耗时 ≥ 上一轮耗时，即未提升或劣化）: 必须 ITER_ROLLBACK 回退本轮改动（不得保留），连续无提升计数器 +1，计入失败累计，再进入 ITER_JUDGE

3. 连续无提升次数是否达到阈值？
    FRONTEND: 5次, SWIMLANE: 8次, INCORE: 5次
    → 是: 进入步骤 3a 核验调优点清单
    → 否: 继续

3a. ⛔ 调优点清单完整性闸门（连续无提升达阈值时的强制核验）
    检查优化点清单状态列：**所有项已标记为 ✅已尝试 或 ❌已失败 或 ❌不适用**
    → 清单已全部标记:
      ⚠️ 若为 INCORE 阶段且性能未达标 → 不直接退出，转算法级优化路由（见「PHASE_INCORE 退出条件」中的算法级优化路由）
      → 算法级优化也收敛（所有 A-D/A-G 项已尝试且连续3轮无提升）→ 输出"阶段内优化耗尽"，退出迭代循环，进入对应 PHASE_SUMMARY
      → 其他 PHASE（FRONTEND/SWIMLANE）→ 输出"阶段内优化耗尽"，退出迭代循环，进入对应 PHASE_SUMMARY
    → **清单存在未标记项（⏳待尝试 或 空状态）: ⛔ 禁止退出！必须回到 ITER_START 继续尝试未标记的优化点**

3b. 优化点清单是否已全部尝试？（仅当 3 未触发时执行）
    检查优化点清单状态列：所有项已标记为 ✅已尝试 或 ❌已失败
    → 是 且 连续无提升已达阈值: 退出迭代循环
    → 是 但 连续无提升未达阈值: 继续下一轮（ITR_START 会因无优化点可选自动退出）
    → 否: 继续

3c. （仅 INCORE 阶段）是否达到理论性能上限？
    核心利用率 > 80% 且 气泡率 < 10%
    → 是: 输出"达到理论性能上限"，退出迭代循环，进入 PHASE_SUMMARY_I_n
    → 否: 继续

4. 是否达到状态看板输出时机？
    总轮次 % 5 == 0
    → 是: 输出状态看板

5. 是否超过调优时间限制？（默认 12 小时）
    → 是: 输出"调优超时"，退出迭代循环，进入对应 PHASE_SUMMARY
    → 否: 继续

6. 继续下一轮
    → 转移至 ITER_START
```

---

## 完成条件校验方法

每个状态的完成条件必须通过实际命令或文件检查验证，禁止凭记忆或假设判断。

### S1_SETUP

> S1_SETUP 包含两个子阶段：S1a 环境检查 + S1b 精度校验。环境检查通过后，继续精度校验。
> **独立编排的 S1 登记**：`complete_stage(1)` 必须在 S1a（环境检查全部通过）和 S1b（精度校验通过）**两个子阶段均完成后**才能调用。任一子阶段未通过，禁止调用 `complete_stage(1)`。
> 独立编排时，子阶段失败使用 `fail_stage(1)` 标记，并在 Todo 中记录失败原因（区分环境问题或精度问题）。
> 精度校验流程详见主技能步骤 1.2。

### S2_COLLECT

零轮核对指定用例在本次设备上的原始基线计时，正轮次检查以下采集文件。

**⚠️ output 目录位于执行算子命令时的工作目录下，非固定位置。** 需在正确的工作目录下执行验证命令。

```bash
# 在执行算子命令时的工作目录下验证
ls output/output_*/merged_swimlane.json   # 必须存在

# 如果不确定位置，从项目根目录搜索
find . -name "merged_swimlane.json" -type f
```

### S3_ANALYZE

**⚠️ 同 S2_COLLECT，需在正确的工作目录下验证。**

```bash
# 在执行算子命令时的工作目录下验证
ls output/output_*/performance_analysis_report.md   # 必须存在

# 如果不确定位置，从项目根目录搜索
find . -name "performance_analysis_report.md" -type f
```

### S4_TUNE 各 PHASE

SWIMLANE→INCORE 为一轮。进入新一轮前按[结构变更台账门控](#phase_summary_f-结构变更台账门控)核对剩余预算。

**回环说明（预期行为）**：INCORE 阶段执行 I-10/I-11（合并 gather / view 复用）修复 CV 通路 DDR 断点后，有剩余预算时回到下一轮 SWIMLANE重试 S-14 Mix合图。这种"合图未完成→进入 INCORE 修复→回退 SWIMLANE 重试"的回环是预期行为，不是异常——外循环机制天然支持此场景。

**算法级优化（INCORE 尾部子步骤）**：INCORE 调优点清单收敛但性能未达标时，不直接退出到 S5_REPORT，而是在 INCORE 阶段内继续执行算法级优化（A-D1~A-D3 / A-G1~A-G7，对应 top-level SKILL.md §4.3）。A-D1~A-D3 修复 CV 断点后有剩余预算时同样回到 SWIMLANE 重试 S-14。

**单 PHASE 退出条件**：⛔ 见「退出条件详细定义」

**提前终止条件**：任意 PHASE_SUMMARY（F/S_n/I_n）在生成摘要时检查全局性能目标，达标则直接路由到 S5_REPORT，不继续后续阶段。

### S5_REPORT

```
报告文件已保存到算子目录下
debug_options 已还原（runtime_debug_mode 移除或置 0）
```

**S5_REPORT TODO 管理**：
1. 更新 Todo 状态机进度：标记已完成的阶段，为跳过的阶段记录原因
2. 输出最终状态看板（见主技能步骤 5.1 的「最终调优状态」模板）
3. Todo 中记录最终性能结果（基准→最终执行时间、累计提升百分比）和迭代统计（总轮次、成功率、最佳优化）
4. 确认 debug_options 已还原的记录写入 Todo

---

## Todo 管理规范

### ⛔ TODO 创建与执行铁律

**⛔ 铁律 1：TODO 必须一次性创建完整，包含本次预算内的子阶段**

S1_SETUP 全部通过后立即创建本次任务的 Todo。正轮次按下列阶段展开，零轮将 S4 标为跳过。

```
S1 → S2(⛔含S-20检查) → S3 → S4 → S5
  S4 = FRONTEND → 按台账重审 → 按剩余预算展开 SWIMLANE_n → INCORE_n
  ⚠️ 任意 PHASE_SUMMARY 达标可提前跳到 S5，不要求全部跑完
  ⚠️ INCORE 收敛但未达标时，在 INCORE 内执行算法级优化（A-D1~A-D3 / A-G1~A-G7）
  ⚠️ A-D1~A-D3 修复 CV 断点后有剩余预算时在下一轮 SWIMLANE重试 S-14（预期回环行为）
```

TODO 中必须明确列出：
- ✅ 本次预算覆盖的 SWIMLANE_n/INCORE_n 子阶段，因 FRONTEND 重审或提前达标而跳过的阶段标注原因
- ✅ INCORE_n 内的算法级优化条目（A-D1~A-D3 / A-G1~A-G7），标注为 INCORE 子步骤
- ✅ 每个子阶段的轮次数、连续无提升阈值（FRONTEND:5, SWIMLANE:8, INCORE:5, 算法级:3）
- ✅ 失败累计计数器（所有 PHASE 共用）
- ✅ 基准性能值和目标性能值

**⛔ 铁律 2：编排器依据 TODO 核验门禁并逐阶段推进**

- 按对应的 `tune-frontend`、`tune-swimlane` 或 `tune-incore` 子技能回传调优点、分析依据和验证结果。
- 编排器核对这些证据，在状态转移时同步更新当前阶段、步骤完成状态和下一步，按门禁逐项推进。
- 每步操作前必须确认当前状态与 TODO 中的"当前阶段"一致
- 未在 TODO 中标记为 ✅ 的阶段，禁止进入
- 每个 PHASE 退出时必须更新阶段状态为 ✅
- PHASE 之间必须生成 PHASE_SUMMARY 交接摘要

**⛔ 铁律 3：每次 ITER_RECORD 后编排器立即将执行角色的证据写入 TODO**

ITER_RECORD 执行后，编排器通过 `todowrite` 更新以下内容（缺一不可）：
- 在当前轮次和阶段新增一行。填写提出并实施建议的**执行角色**、调优点编号、分析表行或性能数据等**分析依据**，记录该角色回传的改动、精度结果和实测耗时，以及本轮变化率、结果类型与状态。
- 当前迭代状态中的"当前最佳性能"和"累计提升%"
- "连续无提升"计数器
- "失败累计"计数器
- "当前轮次"编号

**⛔ 铁律 4：禁止在 TODO 未更新的情况下推进任何步骤**

任何时候对 TODO 状态不确定 → 必须通过 `todowrite` 读取确认 → 确认无未闭合行 → 才能推进。

### 创建时机

| 时机 | 动作 |
|------|------|
| S1_SETUP 全部通过后（S1a 环境检查 + S1b 精度校验） | ⛔ 立即创建完整 Todo（按本次轮次展开），不创建禁止进入 S2 |
| 每轮 ITER_RECORD 后 | ⛔ 编排器登记执行角色回传的调优点和证据，刷新当前最佳、连续无提升和失败累计 |
| 每个 PHASE 完成后 | ⛔ 更新阶段状态为 ✅ |
| PHASE_SUMMARY 后 | ⛔ 编排器附加交接摘要，按门禁更新当前阶段和下一步 |

### ⛔ TODO 完整性自检（所有阶段切换/退出前的强制门控）

以下任何情况发生时，**必须先执行 `todowrite` 工具读取当前 Todo 列表，逐项核验**，核验通过后才能推进：

| 时机 | 核验内容 | 未通过处理 |
|------|---------|-----------|
| ITER_MEASURE 完成 → 进入 ITER_RECORD 前 | 本轮执行时间是否已记录？ | 先记录再进 RECORD |
| ITER_ROLLBACK 完成 → 进入 ITER_RECORD 前 | 回退原因(类型:列)是否已注明（编译失败/精度失败/性能回退）？ | 先注明原因再记录 |
| ITER_JUDGE 判定退出 → 进入 PHASE_SUMMARY 前 | 迭代轮次记录表中最后一行状态列是否为 ✅ 或 ❌，无空状态行 | ⛔ 禁止退出，先补齐记录 |
| PHASE_SUMMARY 完成 → 进入下一阶段前 | 阶段状态 ✅/⏸️ 与实际进度一致？轮次记录表无未闭合行？ | ⛔ 暂停交接，先修正 Todo |
| S5_REPORT 完成 → DONE | debug_options 已还原？已执行阶段标记为 ✅，跳过的阶段已记明原因？ | ⛔ 禁止完成 |

**⛔ 核验命令**：

```bash
# 通过 todowrite 读取当前 Todo 列表，检查是否有未闭合的轮次记录
todowrite  # 读取当前 Todo 清单
```

- 检查返回的 Todo 列表中的 "迭代轮次记录" 部分
- 最后一行状态列如果为空 或 标记为 🔄 → **未闭合轮次，禁止推进**
- 必须通过 `todowrite` 补充完整后再继续

**⛔ 强制原则**：任何时候，如果对 Todo 状态不确定，必须通过 `todowrite` 工具读取确认，禁止凭记忆假设 Todo 是最新的。

### Todo 完整模板（按本次预算展开）

```markdown
## 📊 [算子名称] 性能调优进度

### 目标
- 算子: [名称]
- 基准性能: [值] us
- 目标性能: [值] us (提升X倍)
- 当前最佳性能: [值] us
- 当前设备: NPU 卡 X

### 状态机进度
- ✅ S1: 环境检查 + 精度校验
- 🔄 S2: 性能数据采集
- ⏸️ S3: 性能数据分析
- ⏸️ S4: 外循环调优 (外层预算剩余 [剩余轮数] / [tune_rounds])
  - ⏸️ FRONTEND (轮次: 0, 连续无提升: 0/5, 失败累计: 0)
  - ⏸️ SWIMLANE_n (轮次: 0, 连续无提升: 0/8, 失败累计: 0)
  - ⏸️ INCORE_n (轮次: 0, 连续无提升: 0/5, 失败累计: 0)
- ⏸️ S5: 生成调优报告
- 下一步: [编排器按门禁登记的待办动作]

### 当前迭代状态
- 外层预算剩余: [剩余轮数] / [tune_rounds]
- 当前阶段: [FRONTEND / SWIMLANE_n / INCORE_n]
- 当前第三级状态: [ANALYZE_A / ANALYZE_B / SWIMLANE_STEP0 / SWIMLANE_CORE_USAGE / SWIMLANE_LOAD_BALANCE / ITER_START / MODIFY / VERIFY / MEASURE / RECORD / JUDGE / ROLLBACK]
- 当前轮次: #N（全局递增 #1,#2,...，编号来自 catalog）
- 泳道图重采次数（SWIMLANE_n 内）: [m] 次
- 当前最佳性能: [XX] us（较基准提升 [XX]%）
- 优化点清单剩余: [X] 项（共 [Y] 项）
- 连续无提升: Z / [阈值]
- 失败累计: W 轮

### 迭代轮次记录
| 轮次 | 外循环 | 阶段 | 执行角色 | 编号 | 分析依据 | 优化内容 | 精度 | 执行时间(us) | 变化 | 结果类型 | 状态 |
|------|--------|------|---------|------|---------|---------|------|-------------|------|---------|------|
| 基准 | 0 | - | - | - | - | - | ✅ | XX | - | - | ✅ |
```

### 迭代轮次记录更新示例

```markdown
| 轮次 | 外循环 | 阶段 | 执行角色 | 编号 | 分析依据 | 优化内容 | 精度 | 执行时间(us) | 变化 | 结果类型 | 状态 |
|------|--------|------|---------|------|---------|---------|------|-------------|------|---------|------|
| 基准 | 0 | - | - | - | - | - | ✅ | 79.34 | - | - | ✅ |
| #1 | 1 | FRONTEND | 前端执行角色 | F-1 | 阶段 A 对应分析行 | BLOCK_SIZE 128→64 | ✅ | 68.54 | -13.6% | 性能提升 | ✅ |
| #2 | 1 | FRONTEND | 前端执行角色 | F-8 | 阶段 A 对应分析行 | unroll_list [8,4,2,1] | ✅ | 74.44 | +8.6% | 性能回退 | ❌回退 |
| #3 | 1 | FRONTEND | 前端执行角色 | F-19 | A1 Loop 结构分析行 | 内层批量化 unroll→M | ✅ | 52.10 | -30.1% | 性能提升 | ✅ |
| #4 | 1 | FRONTEND | 前端执行角色 | F-10 | A4 TileShape 审查行 | TileShape 2560 | ❌ | - | - | 编译失败 | ❌回退 |

### 当前迭代状态
- 外层预算剩余: 3 / 3（初次 FRONTEND，示例）
- 当前阶段: FRONTEND
- 当前第三级状态: ITER_START
- 下一步: 核对 FRONTEND 调优点清单中的下一项
- 当前轮次: #4
- 当前最佳性能: 52.10 us（较基准 79.34 us 提升 34.3%）
- 连续无提升: 0 / 5
- 失败累计: 1 轮（#4 编译失败）
```

### 状态看板（每5轮输出）

```markdown
## 📊 调优状态看板

外层预算剩余: [剩余轮数] / [tune_rounds] | 当前阶段: [阶段名] | 轮次: [N]
当前最佳: [值] us | 基准: [值] us | 累计提升: [%]% | 目标: [值] us
进度: ████████░░░░░░░░ [%]%

### 本阶段统计
- 连续无提升: Z / [阈值]
- 最佳优化: [优化项] (+[%]%)
- 当前瓶颈: [描述]
```

---

## 阶段交接摘要

生成 PHASE_SUMMARY 时，按[阶段交接要求](references/phase-handoff.md)填写模板并逐项校验。

---

---

## 异常处理

| 异常类型 | 处理流程 |
|---------|---------|
| S1_SETUP 精度失败 | 换卡尝试（最多 3 次）→ 仍失败 → 停止，不修复，告知用户，状态机停留在 S1_SETUP |
| 迭代性能回退（ITER_MEASURE） | ITER_ROLLBACK → ITER_RECORD（标记❌回退）→ ITER_JUDGE |
| 执行超时（>300s） | 标记失败 → ITER_ROLLBACK → ITER_RECORD（记录超时）→ ITER_JUDGE |
| 外层预算耗尽仍未达标 | 最终摘要标注"未达标" → 建议重新设计算子 → 正常进入 S5_REPORT |

### 迭代轮次精度失败（ITER_VERIFY）

```
1. 不要立即回退，先做简单分析
   - 是否只改了一个参数？（应该只改了一个）
   - 修改的参数值是否在合理范围内？
   - 错误信息是什么？

2. 如果能快速定位原因 → 尝试修正参数值 → 重新 ITER_VERIFY
   ⚠️ 最多尝试 1 次修正

3. 如果无法快速定位 → ITER_ROLLBACK
   - 回退代码修改
   - ITER_RECORD 记录失败（标记 ❌）
   - ITER_JUDGE 判断是否继续

⛔ 精度问题是实现问题，不是调优问题。禁止花大量时间修复，禁止不回退就继续下一个优化点。
