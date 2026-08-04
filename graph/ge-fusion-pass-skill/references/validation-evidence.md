# 验证证据（validation-evidence）

本文件对应 skill 的内部验证证据里程碑 R4；R4 是 workflow 里程碑，不是版本号。 `"$SKILL_ROOT/scripts/validate_evidence.py"` 将真实 baseline/optimized 运行产生的证据汇总为可审计 JSON 和 Markdown。它不运行模型、不安装或加载 pass，也不从缺失文件推断成功；缺环境、缺 dump、缺输出或缺 profiling 时如实写 `NOT_RUN`。

## 一、统一状态与报告

每个子命令的顶层都输出 `status: PASSED | FAILED | NOT_RUN`：

- `PASSED`：已检查的契约全部满足。
- `FAILED`：证据可读但出现不一致、输出误差或规则违规。
- `NOT_RUN`：缺输入、工具、可靠图输出签名或运行产物，无法作出该项结论。

`report` 额外产生 `gate_status`。验证结论 `FAILED` 与环境可用性分离：所有步骤都运行过时 gate_status 为 `PASS`，但报告的 `status` 仍为 `FAILED`；任一项 `NOT_RUN` 时 gate_status 为 `DEGRADED`。因此失败不会被 G3 门禁掩盖。

每个用户 case 都在 `$CASE_ROOT/artifacts/case-matrix.json` 写一份仅含一个 case 的矩阵，作为 G3 `result_status` 的聚合输入：

```json
{
  "schema": "ge-fusion-pass-case-matrix/v1",
  "cases": [{
    "id": "<stable-case-id>",
    "steps": [
      {"name": "events", "status": "PASSED", "command": "<actual argv>", "evidence": ["artifacts/dfx-summary.json"]},
      {"name": "outputs", "status": "NOT_RUN", "command": null, "evidence": [], "reason": "缺 baseline/optimized OM"}
    ]
  }]
}
```

`cases` 长度必须为 1；每个适用步骤都必须有 `PASSED` / `FAILED` / `NOT_RUN`、实际命令（未运行可为 `null`）、证据路径和失败/未运行原因。`N/A` 步骤可以省略，不进入聚合。

```bash
python3 "$SKILL_ROOT/scripts/validate_evidence.py" report artifacts/dfx.json artifacts/graph.json \
  artifacts/outputs.json artifacts/performance.json \
  --out-json artifacts/validation-report.json \
  --out-md artifacts/validation-report.md
```

## 二、DFX 事件与官方统计

pass 可把一行一个 JSON 写到 `pass-events.jsonl`：

```json
{"event":"candidate","pass":"MatMulAddPass","stage":"kAfterInferShape"}
{"event":"matched","pass":"MatMulAddPass","stage":"kAfterInferShape"}
{"event":"guard_passed","pass":"MatMulAddPass","stage":"kAfterInferShape"}
{"event":"applied","pass":"MatMulAddPass","stage":"kAfterInferShape"}
{"event":"skip","pass":"MatMulAddPass","stage":"kAfterInferShape","reason":"bias_shape_unsupported"}
```

合法 `event` 是 `candidate`、`matched`、`guard_passed`、`guard_rejected`、`applied`、`replacement_failed`、`skip`、`pass_begin`、`pass_end`。拒绝、skip 与 replacement 失败必须带非空 `reason`。工具核验 `matched <= candidate`、`guard_passed <= matched`、`applied + replacement_failed <= guard_passed`，并按 reason 聚合。

若 ATC 生成 `fusion_result.json`，同时传入。工具递归读取 CANN 的 `match_times`/`effect_times`，核验 `effect_times <= match_times`；老版本没有官方上报时该小节为 `NOT_RUN`，自有事件仍可作为证据。

```bash
python3 "$SKILL_ROOT/scripts/validate_evidence.py" events \
  --events artifacts/pass-events.jsonl \
  --fusion-result artifacts/fusion_result.json \
  --pass-name MatMulAddPass --require-applied \
  --out-json artifacts/dfx-summary.json --out-md artifacts/dfx-summary.md
```

`GraphFuseInspectorUtils::CanFuse/ReportFuse` 与带 `CustomPassContext` 的 `Replace(..., ctx)` 可用时，应把其真实回调结果写成上述事件；当前 CANN/接口不具备时不得伪造官方计数。
传 `--pass-name --require-applied` 后，缺少目标 pass 或 `applied/effect_times` 为 0 会被明确标成失败/未运行，不会把其它 pass 的统计借来充数。

## 三、图结构正确性

`normalize` 将 ONNX 或 GE dump pbtxt 转为输入适配阶段（内部 R3）的 `normalized-graph.json`；JSON 输入直接复用。图比较检查节点、带端口数据边、控制边、节点属性/shape/dtype/format、悬空边、新消费者和图输出签名。

```bash
python3 "$SKILL_ROOT/scripts/validate_evidence.py" normalize baseline_PreRunBegin.pbtxt --out-json artifacts/baseline-graph.json
python3 "$SKILL_ROOT/scripts/validate_evidence.py" normalize optimized_RunCustomPass_AfterInferShape.pbtxt --out-json artifacts/optimized-graph.json
python3 "$SKILL_ROOT/scripts/validate_evidence.py" graph \
  --baseline artifacts/baseline-graph.json --optimized artifacts/optimized-graph.json \
  --rules requirements-graph-rules.json \
  --out-json artifacts/graph-comparison.json --out-md artifacts/graph-comparison.md
```

规则是需求分析文档 §4 的机器可读副本。默认严格：未声明的节点、边或字段变化都失败。示例：

```json
{
  "allow": {
    "removed_nodes": ["MatMul_0", "Add_1"],
    "added_nodes": [{"name":"FusedGemm", "op_type":"ge:Gemm"}],
    "data_edge_changes": true,
    "control_edge_changes": false,
    "node_field_changes": ["*.format"]
  },
  "require": {
    "preserve_outputs": true,
    "keep_nodes": ["x", "y"],
    "remove_nodes": ["MatMul_0", "Add_1"],
    "add_nodes": [{"name":"FusedGemm", "op_type":"ge:Gemm"}]
  }
}
```

`outputs` 缺失时，工具不会把末端节点猜为图输出，而是把输出检查和总结果标为 `NOT_RUN`。ATC 成功只证明可编译，不能替代该比较。

## 四、整网最终输出

运行程序分别保存 baseline/optimized 的每个**最终输出**为 `.npy`，并写 manifest。`context` 必须在两侧完全一致，且必须包含 `source_model_sha256`、`input_sha256`、`seed`、`preprocess`、`soc_version`、`compile_parameters`、`run_parameters`、`environment`。除此之外，完整验收必须保留 `execution.source_model`、`execution.source_model_sha256`、`execution.input_files[]` 和 `execution.input_sha256`；只复制相同 context 不能证明真的使用了同一输入。

```json
{
  "context": {"source_model_sha256":"...", "input_sha256":"...", "seed":7, "preprocess":"...", "soc_version":"Ascend910B3", "compile_parameters":"...", "run_parameters":"...", "environment":"..."},
  "execution": {"source_model":"model.onnx", "source_model_sha256":"...", "input_hash_scheme":"single-file-sha256-v1", "input_sha256":"...", "input_files":[{"index":0,"path":"input.bin","sha256":"...","size":1024}]},
  "outputs": [
    {"name":"logits", "path":"baseline_logits.npy"},
    {"name":"labels", "path":"baseline_labels.npy"}
  ]
}
```

```bash
python3 "$SKILL_ROOT/scripts/validate_evidence.py" outputs \
  --baseline artifacts/baseline-outputs.json --optimized artifacts/optimized-outputs.json \
  --out-json artifacts/output-comparison.json --out-md artifacts/output-comparison.md
```

输出名称与顺序、shape、dtype 默认必须一致。若 GE 只重命名或重排了外部输出，必须提供一对一的显式映射文件，不能只按下标比较：

```json
{"logits":"fused_logits"}
```

```bash
python3 "$SKILL_ROOT/scripts/validate_evidence.py" outputs \
  --baseline artifacts/baseline-outputs.json --optimized artifacts/optimized-outputs.json \
  --allow-output-name-diff --output-map artifacts/output-map.json \
  --out-json artifacts/output-comparison.json
```

整数/布尔精确比较；浮点使用 `allclose`，默认 FP16/BF16 为 `1e-3`、FP32 为 `1e-5`、FP64 为 `1e-8` 的 atol/rtol，可显式覆盖。结果记录最大绝对/相对误差、NaN/Inf 数量和失败元素数。不接入逐算子精度工具。

## 五、宿主 ACL 运行器

本 skill 不捆绑 ACL runner；case 必须显式提供与当前 CANN 兼容的 `--runner` 二进制及其构建证据。运行器只负责证据采集，不实现模型逻辑，也不加载 pass。

`"$SKILL_ROOT/scripts/run_om.py"` 调用该运行器，保存每个最终输出为 `.npy`，并生成 `outputs.json`、`performance.json`、原始二进制和 stdout/stderr。完整调用必须显式提供 `--source-model`；脚本会重新计算源模型和每个输入文件的 SHA256，并在不匹配时拒绝运行。单输入兼容 `input_sha256=该文件 hash`；多输入使用有序 `index:sha256` 行的 SHA256 聚合指纹。两次运行使用同一个 `context-json`；运行器路径、OM hash、源模型 hash、输入文件 hash 和设备写在 `execution`，不污染可比对的 context。`msprof --application` 没有参数传递时，可通过 `ACL_OM_MODEL`、`ACL_OM_OUTPUT_DIR`、`ACL_OM_INPUTS`、`ACL_OM_WARMUP`、`ACL_OM_RUNS` 环境变量驱动同一运行器。

`validate_evidence.py outputs/performance` 会重新核验 `execution` 中记录的源模型和输入文件；证据文件已被移动导致源/输入不可访问时，结果为 `NOT_RUN`，不会降级成“通过”。

## 六、性能

端到端 manifest 记录已去除 warmup 的 `latencies_ms` 与同一份 `context`，并额外要求 `warmup` 与 `runs`。工具报告样本数、median、P95、绝对差和相对变化。编译时间不得写进 `latencies_ms`。

```bash
python3 "$SKILL_ROOT/scripts/validate_evidence.py" performance \
  --baseline artifacts/baseline-performance.json --optimized artifacts/optimized-performance.json \
  --baseline-profile artifacts/prof-base --optimized-profile artifacts/prof-opt \
  --operator-groups artifacts/operator-groups.json \
  --out-json artifacts/performance-comparison.json --out-md artifacts/performance-comparison.md
```

`--operator-groups` 是一个 JSON 文件，明确 baseline 原算子组与 optimized 融合/替换算子组；工具据 `op_statistic` 汇总各组的 count、总耗时和差值。没有该文件、profiling 或要求的算子类型时，性能总项为 `NOT_RUN` 或 `FAILED`，不会外推性能结论。L0 图层收益也应由 graph comparison 记录，但不能等同于端到端性能提升。

```json
{"baseline": ["MatMul", "Add"], "optimized": ["FusedGemm"]}
```

## 七、历史证据的可追溯性

本仓当前没有可复核、已版本化的历史 CANN 运行证据包，因此不把过去在临时目录中的命令、计数、数值误差或性能数字作为本 skill 的验收结论。任何只剩 `/tmp/...` 路径的旧记录都视为 `NOT_RUN`，不得转述为已通过。

真实 case 要保留以下可追溯证据，写入 case 的 `artifacts/evidence/` 或不可变的外部制品库，并在验证报告中记录稳定 URI/相对路径、SHA256、GE revision、CANN 版本、soc、命令和采集时间：

- baseline/optimized OM、输入、context、`run_om.py` 生成的 manifests 和 runner stdout/stderr；
- ATC/pyatc 命令、GE dump、pass stdout 与 `fusion_result.json` / JSONL events；
- profiling 原始 CSV/SQLite 与由 `validate_evidence.py` 生成的报告。

证据无法随交付保存时，报告只能给出复跑步骤并标 `NOT_RUN`；不能以历史口述、环境日志摘要或消失的临时路径替代。
