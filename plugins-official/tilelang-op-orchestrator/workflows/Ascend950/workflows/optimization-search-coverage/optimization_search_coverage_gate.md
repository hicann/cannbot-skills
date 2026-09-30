# 性能优化搜索覆盖门禁

本门禁防止性能搜索只记录“已经想到的候选”，却遗漏由 case 结构直接触发的高收益路径。它不规定优化答案；它把逐 case 事实、待裁决义务和实际候选分开。

## 1. 修改源码前建立记录

在调优目录创建 `schema_version: 3` 的 `optimization-search-coverage.json`，包含三个 append-only 集合：

- `cases`：每个性能 case 的结构诊断卡；
- `obligations`：由结构事实触发、停止前必须裁决的覆盖义务；
- `candidate_events`：候选的 `CREATED`/`RESULT` 事件。

每个 case 必须填写：`outer_items`、`inner_independent_chunks_per_item`、二者是否独立、`available_cores`、两种 task waves、`per_task_payload_bytes`、基线 `kernel_time_us`、搬运/计算下界及来源、主要 pipe、`source_window` 和关联义务。还必须把用户目标写成 `performance_target`，把当前最佳同口径结果写入 `best_metrics`/`best_metric_evidence`；每次晋升同步更新。目标可用 `ANY`/`ALL` 组合多个 `LE`/`GE` 条件，例如“耗时不超过 10 us 或语义 GM 利用率不低于 80%”。下界暂不可得时填 `null` 并说明缺失证据，禁止猜数。

用户未给数值目标的 case 使用 `performance_target: null`，不得填写示例阈值或把基线耗时当作用户目标；仍须记录真实 `best_metrics` 和证据。全部 case 都没有数值目标时，`--final` 要求全部结构义务闭环，并使用 `convergence_evidence.case_bounds` 记录每个 case 的当前最佳耗时、可核验下界、证据和剩余差距解释；字段结构与第 7 节的 `unmet_convergence_evidence.case_bounds` 相同。通过后输出 `CONVERGED_WITHOUT_NUMERIC_TARGET`，不能写成 `TARGET_MET`。部分 case 有目标时仍按有目标流程执行，未设目标的 case 也必须完成结构覆盖与下界审计。

除 obligation 总状态外，每个关联 case 还要有 `case_dispositions`。只有候选实际激活修改路径，且 RESULT 的对应 `case_results` 为 `TESTED_PATH` 并包含同口径 kernel `performance_measurement`，才算这个 case 已测试；只过编译/精度、其他 shape、未激活的 runtime 分支或同 factory 的另一条路径都不能代替它。

最小骨架如下；完整枚举值以校验器常量和报错为准：

```json
{
  "schema_version": 3,
  "cases": [{
    "case_id": "x", "outer_items": 1,
    "inner_independent_chunks_per_item": 1,
    "inner_chunks_independent": false, "available_cores": 1,
    "outer_task_waves": 1.0, "flattened_task_waves": 1.0,
    "per_task_payload_bytes": 1, "kernel_time_us": 1.0,
    "lower_bounds_us": {"movement": null, "compute": null},
    "lower_bound_evidence": "...", "main_pipes": ["..."],
    "performance_target": {"logic": "ANY", "conditions": [
      {"metric": "kernel_time_us", "operator": "LE", "threshold": 10.0}
    ]},
    "best_metrics": {"kernel_time_us": 20.0},
    "best_metric_evidence": ["..."],
    "source_window": {
      "access_pattern": "INDEXED_OR_REORDERED", "bounded": true,
      "span_bytes": 128, "consumed_bytes": 96, "density": 0.75,
      "window_vregs": 2, "register_window_feasible": true,
      "window_scope": "smallest repeatable hot-loop source window",
      "decomposition_checked": true,
      "baseline_route_kind": "SCALAR_OR_SIMT",
      "baseline_writeback_kind": "SCALAR",
      "index_regularity": "periodic", "evidence": "..."
    },
    "obligation_ids": ["O1"]
  }],
  "obligations": [{
    "obligation_id": "O1", "kind": "...", "case_ids": ["x"],
    "status": "PENDING", "candidate_ids": [],
    "case_dispositions": [{
      "case_id": "x", "status": "PENDING", "candidate_ids": []
    }]
  }],
  "candidate_events": [{
    "candidate_id": "C1", "event": "CREATED",
    "identity": {
      "work_granularity": "...",
      "granularity_facts": {
        "unit": "...", "boundary_kinds": ["SIMD"],
        "values_by_case": {"x": 1}
      },
      "task_mapping": "...",
      "physical_dataflow": {
        "description": "...", "route_kind": "SCALAR_OR_SIMT",
        "source_access": "SCALAR", "writeback_kind": "SCALAR",
        "full_payload_materialization": false,
        "register_lane_reorder": false, "indexed_lane_access": false,
        "route_experiment": false
      },
      "precision": "...", "storage_pipeline": "...",
      "pipeline_facts": {"stages": 1, "realization": "SINGLE_STAGE"},
      "tail_strategy": "..."
    },
    "applicable_cases": ["x"]
  }]
}
```

可复制的未决模板见 [optimization_search_coverage.example.json](optimization_search_coverage.example.json)，完整 RESULT 形状见 [optimization_search_coverage.final.example.json](optimization_search_coverage.final.example.json)。示例值只是 schema 演示，必须替换为当前算子的事实和证据。

候选结束时追加同 identity/`applicable_cases` 的 RESULT，填写 `result_status`、总 `evidence` 和逐 applicable case 的 `case_results`。`TESTED_PATH` 还必须给出含 `kernel_time_us` 与 profiling `artifact` 的 `performance_measurement`；只过精度尚未采到性能时仍记 `NOT_RUN_WITH_EVIDENCE`。编译失败也要逐 case 记 `NOT_RUN_WITH_EVIDENCE`，不能伪装成实测覆盖。若 `REJECTED_WITH_EVIDENCE` 没有任何 `TESTED_PATH`，还必须填写结构化 `rejection_evidence`：`failure_stage`（编译、kernel 执行、精度或 profiling 有效性）、`observed_error`、健康环境/基线 `control_check` 和只归属于该候选的 `candidate_finding`。只要提供了 `rejection_evidence` 就必须满足该结构，不能用自由文本阶段名绕过校验。缺少候选归因对照时不能写 rejected，只能写 blocked。

设备、profiling 服务、外部资源或环境工具链在候选取得有效证据前失效时，不能把该候选记为性能/实现失败，也不能用故障文字关闭搜索义务。候选 RESULT 使用 `BLOCKED_BY_ENVIRONMENT`，未执行 case 使用 `NOT_RUN_BLOCKED`，并记录结构化 `blocking_evidence`：`kind`、`failure_stage`、`observed_error`、已做的 `control_check` 和可恢复验证的 `resume_condition`。关联 obligation、逐 case disposition 和 plan item 保持 `BLOCKED`，引用这个候选；普通校验允许保存该状态，`--final` 必须拒绝。

只有健康环境中的复现或对照检查能把故障归到候选本身。例如候选在直调中稳定触发 kernel 编译/执行错误，而基线和设备健康检查正常，可以作为候选失败；设备在候选启动前不可用、已知正确入口也无法启动，或 profiler 自身插桩/导出失败，只能作为环境阻塞。恢复后从阻塞候选重试，取得有效结果后再更新 obligation 状态。

```json
{
  "candidate_id": "C3", "event": "RESULT",
  "result_status": "BLOCKED_BY_ENVIRONMENT",
  "blocking_evidence": {
    "kind": "DEVICE_UNAVAILABLE",
    "failure_stage": "before kernel launch",
    "observed_error": "device retain failed",
    "control_check": "known-good entry also cannot start the device",
    "resume_condition": "device health check passes, then retry C3"
  },
  "case_results": [{
    "case_id": "x", "disposition": "NOT_RUN_BLOCKED",
    "evidence": ["No candidate kernel launch occurred."]
  }]
}
```

上例仍须保留与 CREATED 完全相同的 `identity` 和 `applicable_cases`；这里只省略未变化字段。关联 obligation/plan 的 `status` 填 `BLOCKED`、`candidate_ids` 引用 `C3` 并记录 `evidence`，obligation 另记录 `blocking_reasons` 字符串列表。

至少应用这些通用触发器：

1. 外层任务不足一波、内层存在多个独立 chunk：建立 `inner_parallel_flatten`；
2. 存在相邻可合并工作、连续边界或显著固定调度/访存成本：建立 `work_granularity_search`；
3. 热路径含逐元素/逐 lane 离散访存，而所需源窗口可界定：建立 `physical_dataflow_routes`；
4. 同一 case 同时需要工作粒度和数据流裁决：建立 `granularity_dataflow_interaction`；
5. 每核有足够独立迭代，且 MTE 与 Compute 均有可重叠时间：建立 `pipeline_realization`。

触发器只建立待裁决项，不保证方案更快。最多四个候选只是当前执行队列，不能限制或删除 coverage obligations。

## 2. 工作粒度义务

`work_granularity_search` 必须有 `boundary_plan`。对每个关联 case，分别记录并裁决：

- `FULL_CONTIGUOUS_UNIT`：不跨语义/元数据边界的最大连续单元；
- `SIMD`：有效 lane/load/select/scatter 的整数边界；
- `DMA`：burst、二维整行、对齐或 stride 边界；
- `CAPACITY`：计入全部常驻、临时和多版本 storage 的有效容量边界；
- `PARALLEL`：一波、两波和稳态多波的任务边界。

每个 plan item 用 `case_values` 写出推导值。TESTED 候选的 `granularity_facts` 必须含相同 boundary kind 和逐 case 值；不适用或淘汰则给出当前布局、容量、lowering 或同 payload 成本证据。测试一个小 group 只能覆盖该数值，不能覆盖取值不同的完整动态 segment。

## 3. 物理数据流义务

`physical_dataflow_routes` 必须有 `route_plan`，并对每个关联 case 分别裁决四个物理族：

- `SCALAR_OR_SIMT`；
- `MEMORY_INDEXED_VECTOR`；
- `CONTIGUOUS_LOAD_REGISTER_REORDER`；
- `MATERIALIZED_TRANSFORM`。

“直接 register reorder”严格定义为：从原始或已分段源窗口连续 Vector load，随后在寄存器内 select/shuffle/pack，期间不把完整重排 payload 写成 UB/L1 中间布局。先写 planar、transpose scratch 或其他完整临时布局，再连续 load 计算，必须标为 `MATERIALIZED_TRANSFORM`；不能因为最后的算术发生在 Vector register 中就标成 direct route。

`route_plan` 的一个条目不是笼统的“输入路线”，而是精确的 `route_kind × writeback_kind × cases` 组合。输入侧至少区分上述四族，输出侧至少区分 scalar、contiguous、strided、indexed scatter、materialized copy 和 mixed。TESTED item 只接受输入 route 与输出 writeback 都相同的候选；一个 gather 索引格式失败不能关闭 direct register route，一个 scatter/synchronization 失败也不能关闭同一输入路线配其他写回方式。

目标未达到时，仍与瓶颈匹配的输入/输出组合须保持 `PENDING`，或用实测候选/硬不可实施证据裁决。门禁不要求机械穷举所有笛卡尔积；但若候选已经改善某一轴、最终却因另一轴失败，必须保留已改善轴并为主导失败轴建立至少一个独立替代组合，不能把整个结构族一起关闭。

每个候选还必须写 `physical_dataflow.route_experiment`：只改粒度/任务/流水并完整继承父版本物理路线时为 `false`；新增或替换 gather、寄存器重排、完整物化等路线时为 `true`。对于 `register_window_feasible=true` 且尚未由基线或确定证据关闭 direct route 的 case，第一个 `route_experiment=true` 的候选必须走 `CONTIGUOUS_LOAD_REGISTER_REORDER`。这条顺序门禁只约束物理路线实验，不阻止先验证独立的任务映射或粒度候选。

`source_window` 把路由条件变成机器事实：访问类型、有界性、span、实际消费字节、density、所需寄存器数、寄存器窗口是否可实施、基线输入/写回路线、索引规律和证据。这里的窗口必须是“生成一个 Vector 输出 chunk 所需的最小可重复热循环窗口”，不是带 row stride 空洞的整张 patch、整行 group 或完整语义 segment；若能逐 row/chunk 流式处理，就按分解后的窗口计算 span、density 和寄存器数，并写 `decomposition_checked=true`。`INDEXED_OR_REORDERED` 会自动触发物理路线义务。

`register_window_feasible=true` 的含义不是“理论上也许放得下”，而是已经结合 dtype、单/多寄存器选择范围、live register 压力和当前 PTO lowering 判断存在可实现代表。填 `false` 必须附 `register_infeasibility`，使用与关闭义务相同的结构化容量/lowering/语义硬证据，并证明最小可重复窗口也不可实施；若尚未查清，应保持 direct route 未决并先查 API/lowering，不能用完整 segment 太大或 stride 空洞绕开顺序门禁。

`route_experiment` 表示该候选的 `route_kind × writeback_kind` 是否偏离对应 case 的 `baseline_route_kind × baseline_writeback_kind`，由校验器交叉核对，不是 agent 可自由声明的标签。可实施 direct register route 尚未关闭时，第一个偏离基线的物理路线候选必须测试它。

## 4. 粒度与数据流交互

同一 case 同时建立工作粒度和物理数据流义务时，必须建立 `granularity_dataflow_interaction`。每个 interaction item 明确记录 `boundary_kind × route_kind × writeback_kind`；高收益组合要么由同一个实际激活的候选测试，要么保持未决，或提供下节定义的硬不可实施证据。

`group + SIMT` 与 `baseline group + register Vector` 的两个单独结果不能代替组合。组合改变 buffer 生命周期或每核稳态迭代数后，还要重新做流水准入；旧粒度/旧数据流上的流水失败只否决旧结构。

## 5. 候选身份和关闭证据

候选身份仍由六轴组成：work granularity、task mapping、physical dataflow、precision、storage/pipeline、tail strategy。`granularity_facts` 是第一轴的机器事实，结构化 `physical_dataflow` 同时记录输入 `route_kind` 与输出 `writeback_kind`。任一轴、逐 case 粒度值、输入路线或输出写回变化都分配新 ID，不能改义复用。修复若改变 copy 路线、中间物化、同步/依赖、buffer 生命周期或调度方式，也属于新候选；只有不改变六轴含义的语法或实现缺陷修复才能沿用 ID。

候选失败天然只否决它自己的精确身份。`CLOSED_WITH_EVIDENCE` 不能靠解释性文字成立，必须提供结构化 `hard_infeasibility`，且声明 `exact_scope_only=true`。允许的硬证据只有：

- `CAPACITY`：给出确切容量、完整 storage footprint，并确认 streaming/chunking 仍不可分解；
- `LOWERING`：给出最小复现、实际查询 symbol 和编译/lowering 产物；
- `SEMANTIC`：给出不可保持语义的约束与证明；
- `STRICT_COST_DOMINANCE`：在同一有效 payload 上量化参考与候选的同单位成本，候选成本必须严格更高。

源码较长、实现困难、API 搜索文字、一个相邻组合失败、malformed ring 或单个候选无收益，都不是关闭同族其他组合的硬证据。它们只能留下该精确候选的 RESULT，并据新瓶颈重排下一组合。

“严格受支配”必须比较同一有效 payload 的具体读写层级、指令/转换/物化次数与可实施性。仅凭源码更长、指令看起来更多、一个相邻组合失败或某候选不便实现，不能关闭义务。

## 6. 流水义务

流水一旦由结构和 profiling 准入，完整性与数值性能目标无关：

- 自动版本化完整覆盖全部跨迭代存活 buffer，且 stage-1/2 配对证明生效，可把 `automatic.status` 记为 `COMPLETE`，并引用实际测试候选；
- 自动 not eligible、lowering 失败、仅部分 buffer 版本化或 latency/overlap 未生效，只否决该自动组合；随后建立手动 input/output ping-pong 候选，必要时再改变 full-wave/static-stage 结构。`manual.status=TESTED` 必须绑定一个真正运行到目标 case 的 MANUAL 多 stage 候选；编译或同步环失败不算已测试；
- 自动完整时也要明确把手动路径记为 `NOT_NEEDED_AUTO_COMPLETE`，不能保持未决；
- 工作粒度、task mapping 或物理数据流变化后，原流水结论不自动继承；对仍满足准入条件的 case 新建义务或重新实测。

候选同时用 `pipeline_facts` 记录 stage 数和 `SINGLE_STAGE`/`AUTOMATIC`/`MANUAL`。`pipeline_plan.automatic/manual` 都是含 `status` 与 `candidate_ids` 的对象。流水义务用 `basis_candidate_id` 绑定它所裁决的结构，初始结构写 `B0`。若一个新的单 stage 候选在已有流水 case 上晋升，最终门禁要求新增以该候选为 basis 的流水义务；已经形成完整多 stage 的候选不递归触发。

## 7. 校验与停止

首次修改前和创建下一候选前运行普通校验；它只检查记录结构与引用一致性并输出 `VALID_RECORD`，允许保留 `PENDING`，不构成候选实施的收口门禁。准备结束调优时才运行 `--final`。先解析本文件真实路径，将其上两级目录（`Ascend950/`）设为 `WORKFLOW_DIR`；该目录是插件资源目录，JSON 记录仍位于用户项目的调优输出目录，不依赖当前工作目录或目标工程的 Git 布局：

```bash
python3 "$WORKFLOW_DIR/scripts/validate_optimization_search_coverage.py" \
  <optimization_search_coverage_json>

python3 "$WORKFLOW_DIR/scripts/validate_optimization_search_coverage.py" \
  <optimization_search_coverage_json> --final
```

普通 `--final` 只在全部 case 均有数值目标且当前最佳指标达到其目标时输出 `TARGET_MET`。达到目标后允许未再探索的路线保持 `PENDING`，但所有已经创建的候选必须有 RESULT；门禁不会逼迫 agent 为无须继续的路线编造关闭理由。全部 case 未设数值目标时走第 1 节的严格收敛路径，仍不允许未决或阻塞义务。

目标未达到时，普通 `--final` 输出 `SEARCH_INCOMPLETE`。这不是停止授权：没有用户明确资源上限或外部阻塞时，必须返回候选分析/实施阶段，不能进入最终验收、归档或结束。只有确实无法继续且希望声明“证据收敛”时，才可额外运行：

```bash
python3 "$WORKFLOW_DIR/scripts/validate_optimization_search_coverage.py" \
  <optimization_search_coverage_json> --final --allow-unmet-convergence
```

这个例外路径要求所有 obligation、逐 case disposition、route/boundary/interaction plan 和流水链均由实际测试或上述硬不可实施证据闭环，并为每个未达标 case 提供 `unmet_convergence_evidence.case_bounds`：当前最佳 kernel time、可核验下界、证据和剩余差距解释。通过时输出 `UNMET_BUT_CONVERGED`。任何 `PENDING`、`BLOCKED`、未运行候选或文字性推断都会失败。机器校验只证明记录一致；是否漏建高收益组合仍需结合源码、逐 case 下界差距和生成代码做结构审计。
