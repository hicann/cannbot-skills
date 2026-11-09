---
name: catlass-dsl-optimize
description: Iteratively optimize, speed up, benchmark, and profile one existing CATLASS DSL operator source file on Ascend NPU with a text-tracked round ledger, noise-aware best selection, and profiler-guided tuning. Use for CATLASS DSL performance optimization or repeated kernel tuning.
---

# CATLASS DSL 算子优化（门禁引导）

> **入口**：本阶段开始前先加载 `catlass-dsl-workflow` 并输出阶段卡。

> **入口门禁**：`python3 <workflow>/scripts/gate.py fulltest --workspace <ws>` 必须通过，
> 即当前 kernel 已有全量测试（full test #1 或 #2）`passed`；未通过不得开始优化，也不得用
> focused test 或 optimize 自测代替。

> **自动执行**：遇到问题只要有可行方案就逐步尝试，一个方案失败就换下一个，不停下来找
> 用户确认。等待远端任务、编译、benchmark、设备或长跑用例时直接等待并轮询，不结束回合，
> 也不用进展汇报代替继续执行。仅当缺少只有用户能提供的信息、或可行路径已穷尽时才提问。

> **超时纪律**：编译、算子运行、benchmark、profiling 与普通 shell 命令必须拆成独立命令块；
> 编译独立 300s，单例运行 `timeout -k 5 60s`，批量分片，禁止整批套一次性超时，禁止裸跑。
> 超时或卡住立即终止，不得死等或加大超时。终止后先做卡健康判定：同一张卡的小探针或
> 官方算子能过 → 判本算子死锁并转 debug；探针也失败 → 判设备问题并换卡重试一次。

驱动 profile → hypothesis → modify → correctness → benchmark → record 闭环，直到达到性能
目标或满足停止条件。**本版本没有 `optimize_state.py`**：轮次账本、best 恢复和停止条件由
`ITERATIONS.md` 的 `## Performance Rounds` 表与本文规则维护。

## First action

先浏览工作区和用户提示，识别：

- workspace 与主实现文件；
- `solution.json`、`definition.json`、`workload.jsonl`、`workload_dev.jsonl`；
- correctness 与 benchmark 命令；
- NPU device、性能指标、目标、迭代上限和 stall threshold；
- operator/optimization 知识、既有 `ITERATIONS.md`。

不要机械询问可从文件系统确定的信息。执行任何测试或修改前展示：

**📋 Resolved Plan**

- **Workspace / Kernel** — `<path>`
- **Benchmark inputs** — `<solution.json>, <definition.json>, <workload_dev.jsonl>`
- **Correctness / benchmark** — `<resolved argv>`
- **Device** — `<npu:index>`
- **Target** — `<metric direction threshold；用户未给数值门槛时写 explore（模式 B）>`
- **Iteration limit** — `<max_iterations>; stall=<stall_threshold>`
- **Knowledge** — `<operator family, keywords, adopted concept paths>`

用户未指定迭代上限时默认最多 30 轮。

## 两种模式

| 模式 | 触发条件 | 停止条件 |
|---|---|---|
| A 达标 | 用户明确给出的数值门槛（`DESIGN.md` 仅转述用户原话；agent 自定、知识/基线/roofline 推算值不算） | best candidate 达到 threshold 且至少完成 1 轮 |
| B 探索 | 没有明确性能目标 | 至少完成 3 轮已记录迭代 |

门槛只能来自用户提示或用户确认。由知识条目、基线实现或 roofline 推算出的数值只能作为
轮次内的参考目标，不得写入 `threshold` 触发模式 A；`DESIGN.md` 中由 agent 自定的数值
目标不构成模式 A 的依据。用户只提出“更快”“性能差”“需要优化”等相对诉求而未给具体
指标时，按模式 B 执行（`threshold: none`），性能采集要求按 `性能要求` 行单独判定。

唯一例外：`性能要求：none` 且用户用例平均 MFU 已达 90% 时，按模式 A 记录
`threshold: MFU ≥ 90%`，用 1 轮判定收口，见 Setup and baseline 第 5 步。

模式在开始时确定并写入 `ITERATIONS.md`，中途不得切换。模式 B 在无用户确认时也必须执行
至少三轮，不得以“感觉差不多了”提前收口；模式 A 即使 baseline 已经达标也必须记录至少
一轮，避免把未优化的首版实现直接当作达标交付。`性能要求：none` 且 dev 子集内用户用例
平均 MFU 已达 90% 的收口例外见 Setup and baseline 第 5 步。

停止条件只看账本：模式 B 未满 3 轮、模式 A 未满 1 轮或未达标时必须继续下一轮。不得中途
结束回合、不得汇报“优化完成”、不得询问用户是否继续。认为没有新方向时，仍要选一个可
证伪的优化轴完成本轮并如实记录 `rejected`。

## 轮次账本

`ITERATIONS.md` 中维护：

```markdown
## Optimization Policy
- mode: target | explore
- metric: performance.candidate.mean_ms (lower is better)
- threshold: <value | none>
- device: npu:<id>
- dev workload: workload_dev.jsonl (<n> cases)

## Performance Rounds
| Round | Axis | Bottleneck | Before (ms) | After (ms) | Decision | Evidence |
|-------|------|------------|-----------|----------|----------|----------|
| 1 | ... | ... | 1.20 | 1.05 | adopted | evidence/round-001/result.json |
```

每轮必须填全：

- `Axis`：本轮唯一改动轴；
- `Bottleneck`：从本轮 fresh profile 指出主导 pipe、占比、搬运还是计算受限、理论下限；
- `Before (ms)`：进入本轮时 best 的 candidate mean；
- `After (ms)`：本轮正确性通过且完成计时后的 candidate mean，未通过写 `null`；
- `Decision`：越过噪声门禁为 `adopted`，否则 `rejected`；
- `Evidence`：本轮结果 JSON 与 profile 目录的项目相对路径。

账本逐轮追加，禁止事后补写。模式 B 不足三轮、模式 A 不足一轮时不得进入 finalization。

## 评测口径（控温仅在 optimize 阶段执行）

- 性能计时只使用 design 冻结的 `workload_dev.jsonl`（用户用例 ≥30 条时为 30~40 条；
  不足 30 条时全选；无用户用例时按覆盖需要自定，总数 ≤40），所有轮次共用同一份样本。
- 逐例配对：同一用例先跑 candidate 再跑 reference，在同一张卡相邻时间完成，抵消温度
  漂移；有对标实现时不得只跑 candidate。测量契约见
  [`catlass-dsl-bench`](../catlass-dsl-bench/SKILL.md)。
- 控温：同用例内各实现温差 ≤3℃；执行前校验 AICore 频率为 1650 MHz；温度超过 85℃ 停机，
  降温后重新配对测量该例。full test 与 benchmark 不控温。
- 单例 60 秒硬超时；超时判定卡死，换卡重试一次，二次仍超时判定 kernel 问题，不得加大
  超时绕过。

## Setup and baseline

1. 按 `catlass-dsl-knowledge` 读取 operator 与 optimization 证据，读已有 `ITERATIONS.md`，
   避免重复已证伪方向。
2. 确认优化对象是一个现有、无符号链接的主实现文件，允许包含多个 DSL kernel。继承
   design/develop 的 `fusion.mode`：默认 performance，允许多 executor、GM 中间张量和
   变体分派；仅用户明确要求单融合时按 `single_kernel` 审计唯一声明与单 launch。
3. 运行全部 correctness 命令，再用已解析命令生成 baseline；确认结果的
   `solution.source_sha256` 与当前 kernel SHA-256 一致。
4. 在 `ITERATIONS.md` 写入 `## Optimization Policy` 与第 0 行 baseline。
5. **MFU 收口判定（仅 `性能要求：none`）**：baseline 完成后，只用 **`workload_dev.jsonl`
   中 `origin=user` 的行**计算平均 MFU（逐例 MFU = 实际 FLOPs ÷ (mean_ms × 峰值算力)，
   FLOPs 算式与峰值来源写入账本）。口径是"子集内筛选出的用户用例"，不是
   `workload.jsonl` 里全部用户用例：不得为判定额外采集子集外用例的性能（用户用例总数
   远超子集规模时，只算子集内选中的那些，禁止先跑全量）。子集内没有 `origin=user`
   用例时该例外不适用，按模式 B 执行。平均 MFU ≥ 90% 时不再进入三轮优化：
   `Optimization Policy` 按模式 A 记录 `threshold: MFU ≥ 90%`，只写 1 轮判定行
   （Decision 为 `rejected`，Reason 写明 MFU 数值、参与计算用例的 uuid 列表与峰值来源），
   随后直接进入 finalization。agent 补充用例只承担正确性覆盖，不参与该判定。

## Optimization loop

每轮只探索一个可证伪的优化轴，严格按以下顺序执行：

1. **Analyze.** 从保留的 `kernel_details.csv`、`step_trace_time.csv` 和历史指标识别当前
   瓶颈。不要盲目调参；先形成可用于检索的 operator family、瓶颈类型、相关 DSL API、
   shape/dtype/layout 和架构关键词。

   若 fresh profile 中 `mac` 或 `mte2` 单项达到墙钟的 0.7 以上，本轮轴按
   `knowledge/dsl/pattern-gemm-optimization-ladder` 的阶梯顺序选取（波次量化 → L1 双缓冲
   → L0 双侧双缓冲），不要在同一阶段反复微调参数。

   若 `kernel_details.csv` 缺失、目标 kernel 行 task duration 为 0，或该时长与配对计时
   相差一个数量级，先按 `knowledge/profiler/torch-npu-profiler-dsl-kernels.md` 的
   Level1 + AiCMetrics 配置重采（上限 2 次）；仍不可用时，在账本写明「profiling 不可用」
   及原因，本轮允许用 makespan 算术与单变量消融归因。benchmark 交付前仍须有 kernel 级
   产物，或写明「profiling 不可用」。

2. **Query knowledge.** 读 `knowledge/optimization/index.md` 与相关分类索引，按算子族、
   `c310`、瓶颈关键词定位候选条目，再完整读取实际采用的条目；检查适用条件、验证记录和
   版本新鲜度。不得把知识结论当作性能证据。把采用的条目路径写入本轮记录；零命中时明确
   记录，并仅在 profile 证据足以支撑 hypothesis 时继续。

3. **Pre-register.** 在 `## Performance Rounds` 增加一行草稿，填写 `Axis`、
   `Bottleneck`、预期收益和判伪条件；Expected 必须在 benchmark 前写入。
4. **Snapshot.** 修改前把当前 best kernel 复制到 `evidence/round-NNN/kernel-before.py`，
   便于 rejected 后恢复。
5. **Modify.** 只修改主实现文件，修改后同步更新 `solution.json` 的源码包；不改变
   reference、workload、容差或计时边界。默认允许多 launch、融合范围及显式 dispatch
   调整，以完整 `run` 的性能为准；严格模式禁止多 kernel、多 launch 或独立变体。
6. **Correctness.** 运行全部 required correctness 命令。失败时立即停止本轮，不得
   benchmark；记录 `After ms = null`、`Decision = rejected`。
7. **Benchmark.** 正确后使用同一 device、`workload_dev.jsonl`、环境和采样配置测量，逐例
   配对。默认 warmup 1 次、采集 2 次；需要更稳时取 5 次中位数。不得为得到更好数字临时
   改变配置。
8. **Record immediately.** benchmark 返回后立即补齐本轮行与 `evidence/round-NNN/result.json`；
   在关闭本轮前不得 profile、读新资料、修改 kernel 或规划下一轮。failed/partial 轮也必须
   记录。
9. **Decide.** 按下方噪声门禁判断 adopted / rejected。rejected 时从
   `evidence/round-NNN/kernel-before.py` 恢复 best；不要手工重建候选。
10. **Continue.** 只有本轮已记录后才分析下一优化轴。不得复用上一轮检索结果跳过本轮核对，
    但可以复用仍适用的条目。

正确候选只有越过噪声门禁才更新 best：

```text
required_improvement = max(
  min_improvement_fraction,
  best_std_ms / best_mean_ms,
  candidate_std_ms / candidate_mean_ms)
```

candidate 相对 best 的提升小于 `required_improvement` 时记 `rejected`，不得因为“看起来
更快”就更新 best。

## Stall 与停止条件

连续 `stall_threshold` 轮没有产生 adopted 结果时进入 stall 处理，不得直接收口：

1. 重新分析保留的 `kernel_details.csv`、`step_trace_time.csv`、best 与 rejected 结果；
2. 采集一次 fresh profile，保存到对应轮的 `evidence/round-NNN/profile/`；
3. 选择与上一轮不同的改动轴，写明新的机制与判伪条件后继续。

仅在以下条件停止迭代并进入 Finalization：

- 模式 A 达到 threshold 且至少完成 1 轮；
- 模式 B 至少完成 3 轮；
- 达到 `max_iterations`（默认 30）；
- 环境缺失导致无法继续，记录 `blocked` / `not_run` 与精确重跑命令。

达到 `max_iterations` 仍未达标时，如实记录未达标、剩余瓶颈和下一步建议，不得伪造成达标。

## 保持循环高效

- 每轮 warmup 1、采集 2 次，用 candidate latency 排序正确候选。
- 候选变化时只重新采集 candidate profile；不变的 Torch reference 可以复用已有 profile。
- 不改变 correctness oracle、workload、tolerance、reference 或已确认的 benchmark 语义来
  缩短时间。
- 把完整 correctness、同配置 benchmark 和 fresh profiling 留给最终验收；快速信号不能
  替代最终复测。

## Finalization

满足停止条件后，对当前 best kernel 做一次最终验收：

1. 运行全部 correctness 命令，记录退出码与逐 case 结果；
2. 用 `workload_dev.jsonl` 做一次 fresh 配对 benchmark；
3. 采集 fresh profiling，保留每例 `kernel_details.csv` 与 `step_trace_time.csv`；
4. 在 `ITERATIONS.md` 写 `## Final Verdict`：模式、轮次数、best 前后耗时、加速比、
   剩余瓶颈、下一步建议，并记录**交付 kernel SHA-256**（当前主实现文件的 SHA）；
   交付 kernel 与 best kernel 不同时（例如后续 review 修复了源码）说明差异原因；
   Final Verdict 引用的证据路径必须真实存在，不得引用未产出的文件；
5. 把 best kernel 保留为工作区主实现文件，并确认 `solution.json` 与之一致。

最终验收证据只服务 optimize 自身，不移交 develop；交回 develop 后先跑
`gate.py optimize` 确认账本完整，再按 final review → 全量测试验收：采纳改动导致 kernel
变化时补跑 `full test #2`，全部轮次 rejected、kernel 未变化时可复用 `full test #1`。
`DESIGN.md` 声明 `性能要求：requested` 时继续执行 benchmark，`gate.py benchmark` 通过后
才算 optimize 真正交付；声明 `none` 时当前 kernel 全量测试通过后直接 finish，不再做交付
benchmark 与 profiling。缺少当前 kernel 的全量测试记录时不得进入 benchmark 或 finish，
也不得把本阶段的 dev 子集结果当作全量正确性证据。若后续 review 修复了源码，只需补跑
focused test 与 `full test #2`；优化轮次按其 evidence 中的 kernel SHA 说明适用范围，
不要求对最终 kernel 重跑。

## 知识沉淀

final 验收通过后，对 adopted 方向做一次去重检查：按 operator family、改动轴、关键 DSL
API、shape/dtype/layout 和最终 profiler 瓶颈，读 `knowledge/optimization/index.md` 与
`knowledge/learned/index.md`；已有条目覆盖同一机制与适用条件时，只在 `ITERATIONS.md`
引用，不重复录入。

同时满足以下条件才写入 learned：

- final correctness、fresh benchmark 和 profiling 均通过；
- best 相对 baseline 越过噪声门限，且达到 Resolved Plan 的提升阈值（未指定时至少 5%）；
- 知识库未覆盖同一优化机制与适用条件；
- 能从 accepted 结果与 final 证据说明 hypothesis、实际修改、适用条件、性能前后和
  profiler 观察。

每个独立机制一条 learned concept，`kernel_sha256` 使用 final best，evidence 引用 final
correctness/benchmark，profiling 通过时附 CSV 路径。结束前一次性写入，不逐轮写入，也不
录入 rejected/failed 方向。写入后把 concept 路径追加到 `ITERATIONS.md` 的 Final Verdict。

## Gotchas

- 追求真实 latency 降低，不做 reward hacking：不得跳过计算、返回未初始化输出或规避计时。
- 不得为了调参方便把 tile、dtype、shape、params 或开关移到模块级，也不得用可变全局状态
  或 closure 生成隐式编译期特化。
- 异常大幅提升先核对 correctness、kernel hash、workload 和 profiler CSV，再下结论。
- 不把仅调整 benchmark 参数、仅停留在 Torch 或跳过 profiling 当成 kernel 优化。
- 不因为工具不可用而静默停止；按 stall/blocked 协议保留证据和重跑入口。

## 产物布局

```text
.catlass-dsl/optimize-runs/<run-id>/
├── ITERATIONS.md
├── baseline/{kernel.py,result.json}
└── evidence/
    ├── round-001/{kernel-before.py,result.json,profile/case-NNNN/*.csv}
    ├── round-002/...
    └── final/{kernel.py,result.json,profile/case-NNNN/*.csv}
```

长期只保留 `ITERATIONS.md`、每轮结果 JSON、best kernel 快照和最终 profile CSV；不保存
完整 profiler trace。
