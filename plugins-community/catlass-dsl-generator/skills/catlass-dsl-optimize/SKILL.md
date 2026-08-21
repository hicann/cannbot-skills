---
name: catlass-dsl-optimize
description: Iteratively optimize, speed up, benchmark, and profile one existing CATLASS DSL kernel on Ascend NPU with correctness gates, compact traces, noise-aware best selection, and resumable state. Use for CATLASS DSL performance optimization, repeated kernel tuning, profiler-guided latency reduction, or requests to make an existing operator faster.
---

# CATLASS DSL 算子优化

驱动 profile → hypothesis → modify → correctness → benchmark → record 闭环，直到达到性能
目标或满足停止条件。直接修改当前 workspace 中唯一 kernel；用
`optimize_state.py` 维护状态、恢复 best、执行噪声门禁和清理产物。

## First action

先浏览工作区和用户提示，识别：

- workspace 与唯一 kernel；
- `solution.json`、`definition.json`、`workload.jsonl`；
- correctness 与 benchmark argv；
- NPU device、性能指标、目标、迭代上限和 stall threshold；
- operator/optimization 知识、既有 `state.json` 与 `ITERATIONS.md`。

不要机械询问可从文件系统确定的信息。只有 kernel、benchmark inputs 或性能目标存在真实
歧义时才停止并询问。执行任何测试或修改前，先向用户展示：

**📋 Resolved Plan**

- **Workspace** — `<path>`
- **Kernel** — `<path>`
- **Benchmark inputs** — `<solution.json>, <definition.json>, <workload.jsonl>`
- **Correctness / benchmark** — `<resolved argv>`
- **Device** — `<npu:index>`
- **Target** — `<metric direction threshold>`
- **Iteration limit** — `<max_iterations>; stall=<stall_threshold>`
- **Knowledge** — `<operator family, query filters, relevant concept paths or none>`

任一字段仍不可信时先解决歧义；否则进入工作流。

用户未指定迭代上限时，默认最多迭代 30 轮。

## Setup and baseline

1. 查询 `catlass-dsl-knowledge` 中的 operator 与 optimization 证据，读取已有
   `ITERATIONS.md`，避免重复已证伪方向。
2. 确认优化对象是一个现有、无符号链接的 kernel。融合入口每次调用必须只 launch 一个
   `@tla.kernel`；禁止拆成多个 executor，也禁止用 Torch 计算 API 实现。
   同时枚举 decorated kernel 的自由名字：除 `tla` API 命名空间和 Python 内建符号外
   必须为空。若 kernel 依赖模块级 tile/shape/dtype/params/开关、可变全局状态或 closure，
   先重构为函数内局部值或显式 ABI/tensor metadata，并重新通过 correctness；不得把结构
   违规源码作为 baseline。同一 solution 只能声明一个 `@tla.kernel`，现有 shape/dtype/layout
   独立 dispatch kernel 必须合并到同一 kernel 后才能优化。
3. 运行全部 correctness 命令，再用已解析 argv 生成 baseline。要求 benchmark 的
   `solution.source_sha256` 与 kernel SHA-256 一致。
4. 初始化 controller；它保存 baseline、生成 `ITERATIONS.md` 并返回 `next_action`。

## Optimization loop

每轮只探索一个可证伪的优化轴。严格按以下顺序执行：

1. **Analyze.** 从保留的 `kernel_details.csv`、`step_trace_time.csv` 和历史指标识别当前
   瓶颈。不要盲目调参；先形成可用于查询的 operator family、瓶颈类型、相关 DSL API、
   shape/dtype/layout 和架构关键词。
2. **Query knowledge.** 每轮都调用 `catlass-dsl-knowledge query`，至少查询
   `optimization`，并按 operator family、`c310`、瓶颈关键词和相关 tag 缩小范围；需要
   代码模式时再查询 `operator` 或 `dsl`。只读取命中的 concept，不扫描整个知识库：

   ```bash
   python3 skills/catlass-dsl-knowledge/scripts/record_knowledge.py query \
     --project-root <workspace> --type optimization \
     --operator-family <family> --arch c310 --text "<bottleneck-or-api>" --compact
   ```

   根据摘要、评分和正文片段选择候选后，对实际采用的每条结果调用
   `record_knowledge.py get --project-root <workspace> --path <query-result-path>`；只读取这些
   完整 concept，检查适用条件、验证记录和版本新鲜度，不得把知识结论当作性能证据。把实际
   采用的项目相对路径写入 proposal 的 `knowledge_sources`，并保留规范化查询、匹配模式及
   score/matched_fields/matched_terms 作为选择依据。零命中建议不得自动采用；如需使用，
   必须显式重查。零命中时在本轮
   `Expected:` 中明确记录，并仅在 profile evidence 足以支撑 hypothesis 时继续。
3. **Pre-register.** 从 [proposal template](templates/proposal.json) 创建 proposal，填写
   hypothesis、单一 `axis_id`、`expected_effect` 和 `falsification_condition`。调用
   `begin-round`，确保 `Expected:` 在 benchmark 前写入 `ITERATIONS.md`；controller 会先
   恢复当前 best kernel。
4. **Modify.** 只修改唯一 kernel。不得修改 reference、workload、solution 语义、容差、
   benchmark 或计时边界。任何多 stage/multi-launch 或 Torch/vendor 算子替代方向直接
   判为无效 proposal，不进入性能比较。tile、dtype、shape、params 和编译期开关只能是
   kernel 内局部值或显式配置；不得新增模块级依赖、closure capture，也不得让 wrapper
   通过 `global`、模块属性或可变容器改变 kernel 语义。不得为编译期变体声明独立 kernel，
   也不得为优化候选新增 `@tla.kernel`；所有 workload 必须继续复用同一 decorated kernel。
5. **Correctness.** 运行全部 required correctness commands。失败时立即停止本轮，不得
   benchmark。
6. **Benchmark.** 正确后使用同一 device、输入、环境和采样配置测量。默认 warmup 1 次、
   采集 2 次；必须得到 `anti_hack.status=passed`，否则本轮失败且不能更新 best。不得为了
   得到更好数字临时改变配置。
7. **Record immediately.** benchmark 返回后，下一项动作必须是从
   [round result template](templates/round-result.json) 创建 submission 并调用
   `record-round`。在关闭当前 iteration 前，不得 profile、读新资料、修改 kernel 或规划
   下一轮。failed/partial 轮也必须记录。
8. **Decide.** 读取 controller 的 accepted/rejected、best、stall count 和
   `next_action`。rejected/failed 后 controller 会恢复 best；不要手工重建候选。
9. **Continue.** 只有当前轮已记录后，才分析下一优化轴并发起下一轮知识查询。不得复用
   上一轮查询结果来跳过查询；可以复用仍适用的 concept，但必须重新核对本轮瓶颈和适用条件。

正确候选只有越过噪声门禁才更新 best：

```text
required_improvement = max(
  min_improvement_fraction,
  best_std_ms / best_mean_ms,
  candidate_std_ms / candidate_mean_ms)
```

## Keep the loop fast

- 每轮保持 warmup 1、采集 2 次，用 candidate latency 排序正确候选。
- 复用项目级 Torch reference profile cache；候选变化时只重新采集 candidate。
- 不改变 correctness oracle、workload、tolerance、reference 或已确认的 benchmark 语义来
  缩短时间。
- profiler 原始目录中必须用 `anti_hack_manifest.json` 和逐 iteration CSV 完成单 launch
  校验；压缩审计目录只保留每个 candidate workload 的合并 `kernel_details.csv`，并可选
  保留 `step_trace_time.csv`，不保存 `anti_hack/` 或 manifest。
- 把完整 correctness、同配置 benchmark 和 fresh profiling 留给 final verdict；快速信号
  不能替代最终复测。

## Stall handling

达到 stall threshold 后暂停修改：

1. 重新分析保留的两个 CSV、best 与 rejected traces，并查询尚未尝试的 operator-specific
   知识。
2. 提供当前运行中未使用过的 fresh profile。
3. 选择与上一轮不同的 `axis_id`，写明新的机制和判伪条件后继续。

完成 fresh profile 后继续迭代；后续再次达到 stall threshold 时重复上述诊断流程。stall
本身不是进入 finalization 的条件。profiler 暂时不可用是诊断约束，不是通过依据；保存
精确重跑 argv，绝不伪造 profile 或性能结论。

## When to stop

仅在以下条件停止迭代并进入 finalization：

1. controller 报告性能目标已达到；
2. 达到已解析的最大迭代数；
3. 环境缺失导致无法继续，记录为 blocked/not_run 并给出重跑命令。

停止时必须调用 `begin-finalization`，让 controller 从 snapshot 恢复真实 best kernel。
重新运行全部 correctness、同配置 benchmark 和 profiling，再调用 `finalize`。禁止凭记忆
重写 best，也禁止把最后一轮误当成 best。finalization 前再次检查 kernel 自包含性和
wrapper 无隐式全局注入；结构违规不能 finalize。最终 kernel 以未提交修改留在工作区。

## Knowledge admission after finalization

`finalize` 通过后，对 accepted 方向执行一次精确去重查询。使用 operator family、axis、
关键 DSL API、shape/dtype/layout 和最终 profiler 瓶颈查询 `optimization` 与 `learned`；若
已有 concept 已覆盖同一机制与适用条件，只在 `ITERATIONS.md` 引用它，不重复录入。

仅当经验同时满足以下条件时录入：

- final correctness、fresh benchmark 和 profiling 均通过；
- final best 相对 baseline 越过噪声门禁，并达到 Resolved Plan 的突出提升阈值；未指定时
  使用至少 5% latency 改善；
- 知识库没有覆盖同一优化机制和适用条件；
- 能从 accepted trace 与 final 证据说明 hypothesis、实际修改、适用条件、性能前后和
  profiler 观察，不把推测写成结论。

为每个符合条件且机制独立的方向生成 learned candidate，`kernel_sha256` 使用 final best
SHA-256，evidence 至少引用 final correctness/benchmark，profiling 通过时还要引用保留的
CSV。结束时一次批量录入，禁止逐轮写入或录入 rejected/failed 方向：

```bash
python3 skills/catlass-dsl-knowledge/scripts/record_knowledge.py record \
  --project-root <workspace> --entry <temporary>/knowledge-candidates.json
```

录入成功后将生成的 learned concept 路径和录入判定追加到 `ITERATIONS.md` session
synthesis，然后删除临时 candidate。录入失败不改变优化性能结论，但必须在 synthesis 中
保留错误、candidate 摘要和精确重建/重跑命令，不能声称知识已写入，也不能把临时文件留在
optimize run 中。

## Controller protocol

按顺序使用控制器，所有写命令传入最新 `expected_revision`：

```bash
# 1. baseline
python3 skills/catlass-dsl-optimize/scripts/optimize_state.py initialize \
  --state <run>/state.json --repository-root <workspace> --kernel <kernel.py> \
  --run-id <operator>-YYYYMMDD-HHMMSS --expected-revision 0 \
  --metric-path performance.candidate.mean_ms --direction lower --threshold <target> \
  --max-iterations <count> --stall-threshold <count> \
  --min-improvement-fraction <fraction> --profiling-required <yes|no> \
  --required-command <command-id> \
  --baseline-result <run>/baseline/benchmark/result.json \
  --correctness-evidence <command-id>=<evidence-path>

# 2. every iteration
python3 skills/catlass-dsl-optimize/scripts/optimize_state.py begin-round \
  --state <run>/state.json \
  --expected-revision <revision> --proposal <proposal.json>
python3 skills/catlass-dsl-optimize/scripts/optimize_state.py record-round \
  --state <run>/state.json \
  --expected-revision <revision> --result <trace>/submission.json

# 3. final verdict
python3 skills/catlass-dsl-optimize/scripts/optimize_state.py begin-finalization \
  --state <run>/state.json --expected-revision <revision>
python3 skills/catlass-dsl-optimize/scripts/optimize_state.py finalize \
  --state <run>/state.json \
  --expected-revision <revision> --result <run>/final/submission.json

# resume / inspect
python3 skills/catlass-dsl-optimize/scripts/optimize_state.py status \
  --state <run>/state.json
```

初始化会把 workspace、唯一 kernel、性能策略和 required command IDs 固化进
`state.json`；后续命令只读取 state。`state.json` 是机器权威；`ITERATIONS.md` 是唯一人类
阅读入口。不得创建 Git branch、commit 或 worktree，也不得自动 merge 或 push。

## Artifacts Layout

长期只保留：

```text
.catlass-dsl/optimize-runs/<run-id>/
├── state.json
├── ITERATIONS.md
├── baseline/{kernel.py,result.json}
├── traces/
│   └── iter-NNN-<axis-id>/
│       ├── kernel.py
│       ├── proposal.json
│       ├── result.json
│       └── profile/case-NNNN/{kernel_details.csv,step_trace_time.csv}
└── final/
    ├── kernel.py
    ├── result.json
    └── profile/case-NNNN/{kernel_details.csv,step_trace_time.csv}
```

每个 NPU `profile/case-NNNN/` 包含合并 `kernel_details.csv`，另可包含
`step_trace_time.csv`；不保留 `anti_hack/` 或 `anti_hack_manifest.json`。不得在 run 中留下
worktrees、完整 profiler trace、编译缓存或重复源码树。

## Gotchas

- 追求真实 latency 降低，不做 reward hacking；不得跳过计算、返回未初始化输出或规避计时。
- 不得把融合计算拆成多个 kernel launch，也不得直接调用 Torch/Tensor/vendor 现有算子。
- 不得为了调参方便把 tile、dtype、shape、params 或开关移到模块级，也不得用可变全局
  状态或 closure 生成隐式编译期特化。
- 异常大幅提升先核对 correctness、kernel hash、workload 和 profiler CSV，再下结论。
- 不把仅调整 benchmark 参数、仅停留在 Torch 或跳过 profiling 当成 kernel 优化。
- 不因为工具不可用而静默停止；按 stall/blocked 协议保留证据和重跑入口。

## Reference files

- [proposal.json](templates/proposal.json) — 开始一轮前填写 hypothesis 与 Expected。
- [round-result.json](templates/round-result.json) — correctness/benchmark 后立即提交本轮。
- [final-result.json](templates/final-result.json) — final verdict、剩余瓶颈和下一步。
- [ITERATIONS.md](templates/ITERATIONS.md) — Summary、Expected、dead ends 与 session synthesis。
