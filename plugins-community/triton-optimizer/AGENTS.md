---
description: Triton Optimizer Workflows — 优化 Triton Ascend NPU 算子或转换 PyTorch 算子为 Triton NPU 实现
mode: primary
skills:
  - triton-npu-optimize
  - triton-npu-convert
---

# Triton Optimizer Workflows

本插件是 Triton Optimizer 在 CANNBot 中的 workflow 入口，覆盖两类场景：

- `triton-npu-optimizer`：优化已有 Triton Ascend NPU 算子，按 baseline/round 方式迭代。
- `triton-npu-converter`：把 PyTorch 算子转换为 PyTorch-facing、Triton Ascend NPU-backed 的实现，并完成验证。

## 入口职责

- 优化任务使用 `triton-npu-optimize` skill，按 baseline/round 方式迭代。
- 转换任务使用 `triton-npu-convert` skill，把 PyTorch 算子转换为 Triton Ascend NPU 实现并验证。
- 若当前工具不支持 Skill，则按本文件规则在当前会话中直接执行 workflow。
- 所有 Skills 来自仓库 `ops/` 目录，通过安装脚本或插件依赖注入；插件目录不保留重复 skill 副本。
- `.triton-agent/` 是优化 workflow 运行态目录，由 hooks 与状态 skill 管理；不要手动编辑或删除。

## 可用 Skills

| Skill | 用途 |
| --- | --- |
| `triton-npu-optimize` | 主优化 workflow，包含 pattern triage、profiling、IR 分析、benchmark、测试生成、状态管理等全部子功能 |
| `triton-npu-convert` | PyTorch 算子到 Triton Ascend NPU 实现的转换 workflow，依赖 `triton-npu-optimize` 中的 gen-test、run-eval、repair-guide 等功能 |

## 脚本索引 (Script Index)

> 每个任务开始前必须先查阅此表确认应使用哪个脚本。

核心操作命令（通过 `triton-npu-optimize` skill 调用，脚本路径相对于 skill 目录）：

**run-eval** (scripts/run-eval/cli.py)
```
triton-npu-optimize run-eval run-test-baseline     --test-file <path> --operator-file <path>
triton-npu-optimize run-eval run-test-optimize     --test-file <path> --operator-file <path>
triton-npu-optimize run-eval run-bench             --bench-file <path> --operator-file <path>
triton-npu-optimize run-eval compare-perf          --baseline-artifact <path> --candidate-artifact <path>
```

**optimize-state** (scripts/optimize-state/cli.py)
```
triton-npu-optimize optimize-state start-round     --round-dir <path> --round-strategy <strategy> --analysis-policy <policy> --reason <reason>
triton-npu-optimize optimize-state submit-baseline  --baseline-dir <path>
triton-npu-optimize optimize-state submit-round     --round-dir <path>
triton-npu-optimize optimize-state set-current-round-state --round-dir <path> --strategy <strategy> --analysis-policy <policy> --reason <reason>
```

全部可用脚本：

| Script | Description | Reference |
|---|---|---|
| Analyze Round Performance | Deep performance diagnosis from round-local profile and optional IR evidence — scalar/vector/cube imbalance, frequent data movement, weak pipeline overlap, and other signals traced back to current operator implementation problems. | [analyze-round-performance.md](triton-npu-analyze-round-performance/analyze-round-performance.md) |
| Gen Bench | Generate benchmark code for an Ascend NPU operator. Use when a new benchmark file is needed for a given operator. | [gen-bench.md](triton-npu-gen-bench/gen-bench.md) |
| Gen Test | Generate correctness test code for an Ascend NPU operator from source code and task context. Supports standalone and differential test styles with requested output location. | [gen-test.md](triton-npu-gen-test/gen-test.md) |
| Optimize State | Manage temporary optimize workflow state including baseline acceptance, round start, same-round strategy-state updates, and round submission. | [optimize-state.md](triton-npu-optimize-state/optimize-state.md) |
| Prepare Optimize Baseline | Establish a reusable canonical optimize baseline by reusing or generating harnesses, performing minimum repair, and passing submit-baseline validation. | [prepare-optimize-baseline.md](triton-npu-prepare-optimize-baseline/prepare-optimize-baseline.md) |
| Profile Operator | Get and analyze Ascend NPU operator performance data — profiling hot operators, identifying timing bottlenecks, summarizing performance evidence, comparing profiling results across runs, inspecting msprof outputs, op_statistic/op_summary CSV files, and Ascend profiler `.bin` data. | [profile-operator.md](triton-npu-profile-operator/profile-operator.md) |
| Run Eval | Execute and evaluate generated operator artifacts — run test cases, run benchmark cases, fast-screen candidate operators against baselines, profile benchmark harnesses, summarize profiling data, and compare performance artifacts. | [run-eval.md](triton-npu-run-eval/run-eval.md) |
| Analyze Compiler Source | Source-backed explanation for performance-related lowering symptoms, suspicious pass effects, or compiler-side behavior when profiler and IR evidence have narrowed but not fully explained the issue. | [analyze-compiler-source.md](triton-npu-analyze-compiler-source/analyze-compiler-source.md) |
| Analyze IR | Capture, archive, and inspect Triton Ascend compiler IR for operator workflows — analyze dumped Triton or Bisheng IR stages, reason about performance issues from generated IR artifacts, collect complete IR from local or remote execution. | [analyze-ir.md](triton-npu-analyze-ir/analyze-ir.md) |
| Optimize Knowledge | Generic reference-only optimize knowledge for pattern triage and evidence-backed symptom routing. Does not define optimize workflow or own round artifacts. | [optimize-knowledge.md](triton-npu-optimize-knowledge/optimize-knowledge.md) |
| Repair Guide | Heuristic fixes for Ascend Triton compile/JIT/kernel errors and numerical precision mismatches when editing or converting operators. | [repair-guide.md](triton-npu-repair-guide/repair-guide.md) |

## 状态文件字段

### `baseline/state.json` (baseline state)

Workflow 运行态由 `optimize-state` 脚本自动管理。以下字段出现在 `baseline/state.json` 中，理解其含义有助于诊断状态异常。

| 字段 | 说明 |
|---|---|
| `baseline_kind` | canonical baseline 的类型：原始算子 (`original`) 或经过最小修复的 prepared baseline (`prepared`) |
| `source_operator` | 相对于 `baseline/` 目录的原始算子路径，通常为 `../<operator>.py` |
| `baseline_operator` | 相对于 `baseline/` 目录的 baseline 算子快照路径，通常为 `<operator>.py` |
| `test_file` | 相对于 `baseline/` 目录的正确性 harness 路径，通常为 `../test_<operator>.py` |
| `test_mode` | 解析后的正确性模式 (`standalone` / `differential`) |
| `bench_file` | 相对于 `baseline/` 目录的 benchmark harness 路径，通常为 `../bench_<operator>.py` |
| `bench_mode` | 解析后的 benchmark 模式 (`torch-npu-profiler` / `perf-counter`) |
| `perf_artifact` | 相对于 `baseline/` 目录的 canonical baseline 性能结果路径，通常为 `<operator>_perf.txt` |
| `correctness_status` | baseline 正确性结果，`passed` 表示通过 |
| `benchmark_status` | baseline benchmark 结果，`passed` 表示通过 |
| `baseline_established` | 仅当 `correctness_status` 和 `benchmark_status` 均为 `passed` 且 artifact 已写入后置为 `true` |

### `opt-round-N/round-state.json` (round state)

| 字段 | 说明 |
|---|---|
| `round` | round 目录名，如 `opt-round-1` |
| `parent_round` | 直接上游：`baseline` 或前一个 round 名，如 `opt-round-2` |
| `hypothesis` | 本轮测试的优化假设 |
| `evidence_sources` | 证明 round 结论的证据来源列表，如 `["benchmark", "profile"]` |
| `correctness_status` | round 正确性结果，`passed` 表示通过 |
| `benchmark_status` | round benchmark 结果，`passed` 表示通过 |
| `perf_artifact` | 相对于 round 目录的 round perf artifact 路径，通常为 `opt_<operator>_perf.txt` |
| `comparison_target` | 相对于 round 目录的 baseline perf artifact 路径，用于 compare-perf，通常为 `../baseline/<operator>_perf.txt` |
| `effective_metric_source` | compare-perf 判定 round 结论的指标：`kernel`、`total-op` 或 `mixed` |
| `summary_path` | 相对于 round 目录的 round 总结 markdown 路径，通常为 `summary.md` |
| `opt_note_updated` | 顶层 `opt-note.md` 的本轮条目已更新后置为 `true` |

可选字段：`analysis_skipped_reason`、`profile_dir`、`ir_dir`、`perf_analysis_path`。

## 优化 Workflow

详细流程以 `triton-npu-optimize` skill 为准，按以下 Phase 执行：

```
Phase 0: 明确 operator、target mode、评测命令和运行环境
Phase 1: 建立或复用 canonical baseline
Phase 2: 创建 opt-round-N，并在首次代码修改前完成方向选择
Phase 3: 每轮只实施一个主要优化点
Phase 4: 依次执行 correctness、benchmark、compare-perf
Phase 5: 必要时按 pattern -> profile -> IR -> compiler-source 升级证据
Phase 6: submit-round、更新 opt-note.md，并决定停止或进入下一轮
```

Phase 2 方向选择不固定为 pattern-index 单次门禁，但也不是自由发挥。每轮首次代码修改前必须完成 workflow context review：读取 Phase 1 已接受的 baseline state、`opt-note.md`、当前/历史 `attempts.md`、目标 operator 的 wrapper/kernel 结构和用户目标；然后从允许的 direction source 中选择一个 concrete hypothesis，并写清 success criteria、验证命令、evidence、命中条件、预期修改代码区域和未选方向原因。方向优先按 `triton-npu-optimize/references/triton-npu-optimize-knowledge/pattern_index.md` 的现有结构推进：先检查 `High Priority Patterns`，再检查适用的 `Generated Pattern Summaries` 结构性 pattern，最后才进入 pattern 明确允许的 bounded tuning/cleanup。benchmark/compare-perf、profile、IR、compiler-source 只作为证据升级、问题归因或 pattern 轨道释放后的依据，不作为随意试错入口。若连续 3 轮及以上 kernel 内部优化收益低于 1.2x，或证据显示 wrapper/aclnn/TensorMove/launch overhead 主导，或 benchmark 总耗时与 kernel 耗时明显不匹配，必须暂停 kernel micro-optimization 并回到 pattern-index triage，优先寻找 architecture-level 或 structural pattern。Free exploration 只有在适用的 high-priority、结构性和 bounded tuning/cleanup pattern 都尝试、拒绝或释放后才允许。禁止把多轮微调作为优化方向，不允许连续多轮只调整 `BLOCK_SIZE`、tile size、grid、`num_warps`、launch flag 或阈值常量；pattern/autotune 明确要求的参数搜索必须是有界候选集或 helper workflow。

## 转换 Workflow

1. 明确原始 PyTorch 算子文件、转换输出路径、验证模式和运行环境。
2. 使用 `triton-npu-convert` 读取原始算子，但不修改原文件。
3. 将转换后的 PyTorch-facing Triton Ascend NPU 实现写入用户指定输出路径。
4. 保留源文件尾部的输入辅助函数块，便于后续 harness 生成和差分验证。
5. 使用 `triton-npu-optimize` 中的 gen-test 功能复用或生成测试，再用 run-eval 功能执行验证。
6. 遇到 Triton 编译、JIT、launch 或 kernel 结构问题时，使用 `triton-npu-optimize` 中的 repair-guide 功能修复后重跑验证。

## 约束

- 优化流程一次只保持一个 active round。
- 不跳过正确性验证直接比较性能。
- 不把测试、benchmark、profile 结果写成口头假设，必须保留可追溯 artifact。
- 转换流程不得覆盖原始 PyTorch 算子文件。
- 转换结果必须真实走 Triton Ascend NPU kernel 路径，不能用 PyTorch 计算路径伪装替代。
- 不把 `ops/` 下的 Skill 内容复制回插件目录；插件只保留 agent、hooks 和安装入口。
