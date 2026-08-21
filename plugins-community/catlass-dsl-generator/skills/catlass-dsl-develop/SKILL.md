---
name: catlass-dsl-develop
description: Develop, debug, review, test, benchmark, profile, and optionally optimize one CATLASS DSL kernel from an approved DESIGN.md with agent-led decisions, correctness gates, compact traces, knowledge queries, and resumable state. Use for new CATLASS DSL operator implementation or complete operator delivery.
---

# CATLASS DSL Develop

驱动 analyze/query knowledge → task or hypothesis → modify → focused correctness → record →
review/full test → benchmark/profile → optional optimize → finalize 闭环。Agent 负责实现和决策；
`develop_state.py` 只冻结配置、验证门禁、推进 `next_action` 并清理产物。

## First action

先从用户提示、获批 `DESIGN.md` 和工作区解析信息；可发现时不要询问。只有接口、唯一
kernel、验证命令或性能目标存在真实歧义时才停止。执行前固定展示：

**📋 Resolved Plan**

- **Workspace** — `<repository root>`
- **Design** — `<DESIGN.md and SHA-256>`
- **Kernel** — `<one repository-relative kernel_path>`
- **Benchmark inputs** — `<solution.json>, <definition.json>, <workload.jsonl>`
- **Commands / required cases** — `<resolved JSON argv and case IDs>`
- **Device** — `<idle npu:index>`
- **Risk** — `<standard|high and review focus>`
- **Performance** — `<metric, target or measurement-only>`
- **Knowledge** — `<query filters and relevant concepts or none>`

把获批计划写入 [state template](templates/state.json) 的 `config`。`kernel_path` 必须引用
`allowed_paths` 中唯一的主 kernel；state 路径固定为：

```text
.catlass-dsl/develop-runs/<run-id>/state.json
```

初始化本地知识 bundle，验证并启动 controller：

```bash
python3 skills/catlass-dsl-knowledge/scripts/record_knowledge.py initialize \
  --project-root <workspace>
python3 skills/catlass-dsl-develop/scripts/develop_state.py validate \
  --state <run>/state.json
python3 skills/catlass-dsl-develop/scripts/develop_state.py start \
  --state <run>/state.json
```

## Agent loop

每个 iteration 只完成一个任务或验证一个假设：

1. **Analyze and query.** 读取 DESIGN、当前 kernel、已有 `ITERATIONS.md` 和最近 result。
   按 operator family、DSL API、错误类型、shape/dtype/layout 查询最窄的 `operator`、`dsl`
   或 `debug` concept；性能问题查询 `optimization`。查询先使用 `--compact`，再对选中的路径
   调用 `record_knowledge.py get` 读取完整 concept，不扫描整个 bundle。记录 CLI 返回的规范化查询、匹配模式，
   以及所采用条目的 score/matched_fields/matched_terms；零命中与建议也记录，但建议必须由
   agent 明确发起新查询后才能采用。
2. **Task or hypothesis.** task breakdown 必须覆盖 config 中所有 allowed paths 和非性能
   required cases。debug 只写一个假设、Expected 和判伪条件，不在一轮混入多个修改轴。
3. **Modify.** 只修改当前任务的 allowed paths。融合算子每次入口调用必须只 launch 一个
   `@tla.kernel`；不得以 GM 临时张量串联多个 executor，也不得用 Torch 计算 API 实现。
   包装层只允许元数据检查和空输出分配。不得修改 oracle、workload、容差、benchmark
   或计时边界换取通过。每个 decorated kernel 必须自包含，只能读取形式参数、函数内
   局部值、Python 内建符号和 `tla` API；禁止模块级 tile/shape/dtype/params/开关、可变
   全局状态和 closure capture。固定值在 kernel 内定义，运行时值走显式 ABI 或 tensor
   metadata；同一 solution 只声明一个 `@tla.kernel`，所有 shape/dtype/layout 编译期变体
   必须复用它并由形式参数类型或 metadata 驱动。wrapper 不得通过 `global`、修改模块对象
   改变 kernel 语义；不得为编译期变体声明独立 kernel 或选择独立 dispatch kernel。
4. **Focused correctness.** 先运行受影响的 focused cases。失败时停止本轮，不执行后续
   benchmark；保存稳定错误类型和有界诊断摘要。
5. **Record immediately.** 测试、review 或 benchmark 返回后，下一动作必须是创建临时
   submission 并调用 `advance`。当前 iteration 关闭前不得继续修改、查询或规划下一轮。
6. **Decide.** 按 controller 的 status、next action 和失效门禁继续 implement/debug、
   targeted review、final review、full test、benchmark、optimize 或 finish。

```bash
python3 skills/catlass-dsl-develop/scripts/develop_state.py advance \
  --state <run>/state.json \
  --result <run>/traces/iter-NNN-<stage>/submission.json
```

Controller 解析后删除 submission 和成功原始日志。`result.json` 保存 stage、attempt、
kernel SHA-256、状态、摘要、结构化 command/review/benchmark 结果、knowledge queries、
decision 和 next action。failed/blocked trace 最多额外保留一个有大小上限的
`failure.txt`。

## Review, test, and benchmark

- `risk_level=high` 时 implement/debug 后先做 independent targeted review；所有任务都要
  做覆盖当前 kernel 的 independent final review。
- review 修复会失效受影响测试和 benchmark；修复后重新执行相关门禁。
- final review 通过后运行全部 correctness、boundary 和 safety required cases。
- full test 通过后始终调用 `catlass-dsl-bench`，即使没有性能门槛；benchmark 必须包含
  correctness、有限正数 candidate mean、candidate profiling 和 `anti_hack.status=passed`。
- targeted/final review 必须检查 host 入口只调用一个 executor，且没有 Torch/Tensor
  计算路径；anti-hack 失败一律回到 implement/debug，不能进入 optimize 或 finish。
- targeted/final review 必须枚举每个 `@tla.kernel` 的自由名字；除 `tla` API 命名空间和
  Python 内建符号外必须为空，并确认 wrapper 没有使用 `global`、模块属性或可变容器
  注入 tile、shape、dtype、params 或分支开关。还要确认 solution 只声明一个 kernel，
  编译期变体没有独立 kernel dispatch。任一违反都回到 implement/debug。
- 使用 `npu-smi info` 选择无进程 NPU，执行前 source 项目 `env.sh`；真实 NPU 算子不在
  沙箱运行。
- Torch reference profile 使用项目级共享缓存。candidate profiler 原始目录中完成
  anti-hack 校验；压缩审计目录只保留合并 `kernel_details.csv`，并可选保留
  `step_trace_time.csv`，不保存 `anti_hack/` 或 manifest。校验时证据缺失则失败关闭。
- 有性能目标且未达标时调用 `catlass-dsl-optimize`。返回 best kernel path/SHA 后重新运行
  benchmark、final review 和 full test，不能直接采用 optimize 的最后一次尝试。

## Finalization and knowledge

所有 submission、iteration 和知识查询均绑定当前 kernel SHA-256，不依赖 Git 仓库、HEAD、
分支或提交。`finish` 必须绑定当前 kernel SHA，并证明 full test、final review、fresh benchmark 和
profiling 均通过。Controller 把真实 kernel 快照写入 `final/kernel.py`，生成紧凑
`final/result.json`，并核对它们与 state 中最终 SHA 一致。最终 `ITERATIONS.md` 必须列出
fresh benchmark 中每个 workload 的 UUID、candidate/reference mean ms 和
`reference_mean / candidate_mean` 加速比，使用 `## Workload Speedups` 独立表格呈现；不得
只报告聚合均值。

只有完整测试或 profiling 直接支持的结论才批量交给 `catlass-dsl-knowledge record`，使用
kernel SHA-256 绑定；证伪假设、静态推断和 `not_run` 不得录入。知识录入成功路径、有效
方向、dead ends、最终性能、剩余风险和下一步写入 `ITERATIONS.md` 综合总结。

## Compact artifacts

```text
.catlass-dsl/develop-runs/<run-id>/
├── state.json
├── ITERATIONS.md
├── traces/
│   └── iter-NNN-<stage>/
│       ├── kernel.py                         # 该轮关闭时的源码快照
│       ├── result.json
│       ├── failure.txt                         # 仅 failed/blocked
│       └── profile/case-NNNN/
│           ├── kernel_details.csv
│           └── step_trace_time.csv
└── final/
    ├── kernel.py
    ├── result.json
    └── profile/case-NNNN/
        ├── kernel_details.csv
        └── step_trace_time.csv
```

`ITERATIONS.md` 是唯一人类入口，表头固定为：

```markdown
| Iter | Title | Score | Passed | Notes |
```

非性能阶段 Score 为 `—`；benchmark/final 使用 baseline 相对加速比。禁止留下根
`README.md`、`state/`、`stages/`、`artifacts/`、逐阶段 `SUMMARY.md`、`manifest.json`、
持久 submission、完整 profiler trace、编译缓存、额外 Git 工作目录、分支或提交产物。
每个 trace 必须保留且只保留一份 `kernel.py`，其 SHA-256 必须等于该轮 `result.json`
中的 `kernel_sha256`，使通过、失败和 blocked iteration 都有可复核的源码证据。

## Stop and failure rules

- skipped、xfailed、missing、timeout、hang 和 `not_run` 都不能成为通过。
- NPU benchmark 缺少显式 `anti_hack.status=passed` 时失败关闭；旧证据必须重新执行。
- correctness 失败回到 implement/debug；review 失败先修复；benchmark correctness 失败
  不能进入 optimize。
- kernel 读取用户定义模块级值或 closure、wrapper 改写 kernel 隐式配置时，结构审查必须
  失败；为编译期变体声明独立 kernel 同样失败。正确性或性能通过不能覆盖该失败。
- NPU、CANN 或 CATLASS 环境缺失时返回 blocked/not_run，记录精确重跑 argv，不伪造通过。
- `state.json` 是唯一机器权威；只接受 `catlass.dsl.workflow.v3`，不迁移旧 run。

## Reference files

- [state.json](templates/state.json) — 冻结 config 并启动 run。
- [task-breakdown-result.json](templates/task-breakdown-result.json) — task breakdown submission 示例。
