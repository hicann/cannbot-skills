# 输出目录约束（Step 1）

`ops-perf-routing-plan` 只在 `<out-root>/<op>_VIP_routing_plan/<timestamp>/` 下产出一个 work 目录，不创建 delivery/final code 目录。

```text
<out-root>/<op>_VIP_routing_plan/<timestamp>/
├── code/
│   └── ori_code/<op>/                  # 只读分析基底
├── inputs/
│   ├── cases.csv
│   ├── hardware.json
│   ├── pipeline_meta.json
│   ├── base_code.json
│   ├── precision_test.py               # 探测到 golden 时生成
│   ├── perf_test.sh
│   └── golden/
│       ├── golden_flow.json
│       ├── golden_flow_review.md
│       └── golden_source.json
├── profiling/
│   └── baseline/performance.json
├── theoretical/
│   ├── shape_limits.json
│   └── theoretical_review.md
├── routing/
│   ├── coverage/
│   │   ├── state.json                  # 串行归类工作态
│   │   └── plan.json                   # freeze 后的只读计划
│   ├── templates/<name>/
│   │   ├── template_spec.json          # 模板提案
│   │   └── proposal_result.json        # 谓词命中结果
│   ├── tiling_routes.json
│   ├── tiling_table.csv
│   ├── coverage_report.json
│   ├── classification_report.md
│   └── classification_report.html
└── logs/
    ├── base_check.json
    ├── detect_repo.json
    ├── build.log                       # 现场构建时生成
    └── install_check.log               # 现场装包时生成
```

## 1. 冻结计划

`routing/coverage/plan.json` 是 Step 1 的核心交付，必须满足：

- `stage == "frozen"`；
- `remaining == []`；
- 所有 case 恰好属于一个模板；
- 模板 `name`、`tiling_key`、`priority` 唯一；
- 每个模板含 `representative_case_id` 和 `cases`；
- 每个模板含 `tiling_plan`（或可由 `tiling_intent` 解析）及 `tiling_validation.ok=true`。

后续 workflow/Agent 只读该计划，不应修改它。需要调整分类时重新运行 Step 1。

## 2. 路由表

`route_report.py` 生成：

- `tiling_routes.json`：`routes[]` 按 priority first-match；`case_to_template` 是实际分类结果；
- `tiling_table.csv`：逐 case 的分类模板、tiling_key、核数、tile、尾块策略；
- `coverage_report.json`：覆盖、未路由 case 和解析错误。

`original` 是路由表默认兜底，不表示优化失败。冻结计划不得依赖该兜底来掩盖未覆盖 shape。

## 3. 模板提案

`routing/templates/<name>/template_spec.json` 至少包含：

```json
{
  "template_name": "large_n_ub_full",
  "worst_case_id": "case_0042",
  "predicate": "numel > 1048576 and align_ok",
  "derived": {"align_ok": "(shape[-1] * elem_bytes) % 32 == 0"},
  "tiling_intent": {"num_cores": "aiv_core_num"},
  "tiling_plan": {},
  "expected_bottleneck": "MTE2",
  "evidence": "profiling + theoretical bound",
  "kernel_file": "op_kernel/example_kernel_large_n_ub_full.h"
}
```

`kernel_file` 只是 Step 2 的规划提示，本 skill 不创建任何 kernel 文件。

## 4. 禁止产物

Step 1 不应出现：

- `optimizer_templates/`、`optimizer.json`、`optimizer_result.json`
- `parallel/`、子 agent prompt/session 文件
- `code/shape_XX_code/`、`code/fin_code/`
- `final_code/`、`final_build/`、`final_code_manifest.json`
- `final_report.md/.html`、`assembly_report.json`
