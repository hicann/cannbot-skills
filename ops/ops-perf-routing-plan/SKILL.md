---
name: ops-perf-routing-plan
description: 对 AscendC 算子执行 Shape 级 tiling 路由的 Step 1：用 golden/计算流估算每个 shape 的理论下限并按 headroom 排序，以 worst-first 贪心划分 shape 组、选择典型 shape、校验 tiling 可行性，冻结 shape→模板计划并生成路由表与分类报告。仅做 Shape 分析、分组、路由和计划冻结，不执行 kernel 优化、子 agent 优化或 final code 拼装；可被性能优化 workflow、Agent 或人工单独调用。使用场景：当需要做 shape 分组、shape 分类、tiling 路由、tiling 模板规划、最差 shape 优先、典型 shape 选择、shape 特征路由、冻结路由计划或生成分类报告时使用。触发：shape 分组、shape 分类、tiling 路由、tiling 模板规划、最差 shape 优先、典型 shape 选择、shape 特征路由、冻结路由计划、分类报告。
---

# AscendC Shape 级 Tiling 路由（Step 1）

本 skill 是一个独立、可组合的分析能力，只完成 Shape 级 tiling 路由的 **Step 1：分析 → 排序 → 分组 → 冻结计划 / 路由表**。它不改 kernel，不优化模板，不合并代码，也不执行优化后的精度或性能验收。

调用方（workflow、Agent 或人工）可以在得到冻结计划后，自行进入 Step 2（按模板优化）和 Step 3（路由合码）；这些后续阶段不属于本 skill。

## 0. 定位 Skill 根目录

```bash
SKILL_DIR="$(cd "$(dirname "<本文件路径>")" && pwd)"
SK="$SKILL_DIR/scripts"
```

源码位于 `ops/ops-perf-routing-plan/`；安装后通常位于
`<project>/.agents/skills/ops-perf-routing-plan/`（claude 为 `.claude/skills/`）。

## 1. 脚本入口

脚本固定为五个，避免同一动作散落在多个包装脚本中：

| 脚本 | 用途 |
|---|---|
| `prepare.py` | `init` 建目录/落账/识别仓/golden；`build` 编译安装；`profile` 归档或采集基线 |
| `shape_limits.py` | 计算每个 shape 的理论下限和 headroom 排名 |
| `greedy_route.py` | `init/next/validate/commit/freeze/status` 串行归类状态机 |
| `route_report.py` | 解析冻结计划，生成路由表、覆盖报告和分类报告 |
| `common.py` | 共享库：表达式、cases、硬件、日志、tiling 解析与校验 |

## 2. 输入与前置条件

| 输入 | 必填 | 说明 |
|---|---|---|
| `--op-dir` | 是 | AscendC 算子目录，用于识别算子语义、golden 和基线构建 |
| `--op-name` | 是 | 算子名，用于命名工作目录和人工可读产物 |
| `--cases` | 是 | cases.csv / JSON / JSONL，至少含 `case_id`、`shape`、`dtype`、`attrs` |
| `--hardware` | 否 | 硬件档案；缺省根据 `--chip` 使用内置档案 |
| `--golden` | 否 | 显式 golden；缺省从 `<op-dir>/tests/st/` 探测 |
| `performance.json` | 是 | 每个 case 的 msprof 实测耗时；可由本 skill 基线采集，也可由调用方直接提供 |

`performance.json` 每个 case 必须有 `elapsed_us` / `task_us` / `asc_us` / `duration_us` 之一，且 `case_id` 必须与输入 cases 完全一致。没有实测耗时就无法计算 headroom，不能完成排序。

## 3. 输出契约

工作目录固定为：

```text
<out-root>/<op>_VIP_routing_plan/<timestamp>/
├── inputs/
│   ├── cases.csv, hardware.json
│   ├── pipeline_meta.json
│   ├── base_code.json
│   ├── precision_test.py, perf_test.sh      # 探测到 golden 时生成
│   └── golden/{golden_flow.json,golden_flow_review.md,golden_source.json}
├── code/ori_code/<op>/                      # 只读分析基底
├── profiling/baseline/performance.json
├── theoretical/{shape_limits.json,theoretical_review.md}
├── routing/
│   ├── coverage/{state.json,plan.json}      # 归类工作态 / 冻结计划
│   ├── templates/<name>/
│   │   ├── template_spec.json
│   │   └── proposal_result.json
│   ├── tiling_routes.json
│   ├── tiling_table.csv
│   ├── coverage_report.json
│   └── classification_report.md/.html
└── logs/
```

Step 2/3 的调用方应以 `routing/coverage/plan.json` 和 `routing/tiling_routes.json` 为输入；不需要读取本 skill 的内部日志。

`plan.json` 中每个模板至少包含：

- `name`、`priority`、`tiling_key`
- `predicate`、`derived`
- `representative_case_id`（该组首次选中的 worst shape）
- `cases`（本组覆盖的全部 shape）
- `tiling_plan` / `tiling_intent`
- `expected_bottleneck`、`evidence`
- `is_single_case_group`（单 shape 组提示）
- `tiling_validation`（commit 时的可行性摘要）

完整目录约束见 `references/output_layout.md`。

## 4. 初始化与基线

```bash
SKILL_DIR=<ops-perf-routing-plan 根目录>
SK="$SKILL_DIR/scripts"
OUT_ROOT=<输出根目录>
TILING_KEY_BASE=<可选，tiling key 起始值，默认 1>
WORK=$(python3 "$SK/prepare.py" init \
  --op-dir <算子目录> --op-name <op> \
  --cases <cases.csv> --out-root "$OUT_ROOT" --chip 910b2 \
  [--soc ascend910b] [--golden <golden>] [--hardware <hardware.json>] \
  [--tiling-key-base <n>] [--no-commit])
```

init 会建立工作目录、记录基底、复制只读分析代码、归一化 cases、选择硬件档案、探测 golden，并生成测试脚本骨架。默认会把算子目录的未提交改动落账为 `base_commit`；不希望修改 Git 历史时传 `--no-commit`。

需要现场采集基线时：

```bash
# 可选：编译并安装到 <WORK>/cann_opp
python3 "$SK/prepare.py" build --work "$WORK"

# 归档已有结果；或改用 --command 现场采集
python3 "$SK/prepare.py" profile --work "$WORK" --perf-json <performance.json>
```

如果调用方已有 `performance.json`，可跳过 `build`，直接 `profile --perf-json` 归档。`profile --command` 按 argv 列表执行，不经过 shell 解析；需要环境初始化时请传入可执行脚本，而不是 `source ... && ...`。编译与隔离规则见 `references/repo_build_install.md`。

## 5. 理论分析与排序

先检查 `inputs/golden/golden_flow.json`，记录输入/输出字节数、vector 指令和可选 Cube 计算；不支持的项必须显式报错，不能静默按 0 处理。schema 与公式见 `references/golden_flow_schema.md` 和 `references/theoretical_perf_model.md`。

```bash
python3 "$SK/shape_limits.py" \
  --cases "$WORK/inputs/cases.csv" \
  --perf "$WORK/profiling/baseline/performance.json" \
  --flow "$WORK/inputs/golden/golden_flow.json" \
  --hardware "$WORK/inputs/hardware.json" \
  --operator <op> -o "$WORK/theoretical/shape_limits.json"
```

排序规则固定为：

1. `headroom = T_candidate / T_hw` 降序；
2. headroom 相同按 `T_candidate` 降序；
3. 再按 `case_id` 字典序。

`headroom` 只用于排序，不代表最终加速比。

## 6. 串行归类并冻结

归类必须串行执行；不要并行调用 `next` / `validate` / `commit`。

```bash
python3 "$SK/greedy_route.py" init \
  --cases "$WORK/inputs/cases.csv" \
  --limits "$WORK/theoretical/shape_limits.json" \
  --hardware "$WORK/inputs/hardware.json" --operator <op> \
  --tiling-key-base "${TILING_KEY_BASE:-1}" \
  -o "$WORK/routing/coverage/state.json"

python3 "$SK/greedy_route.py" next \
  --state "$WORK/routing/coverage/state.json"
```

对 `next` 返回的当前最差 shape，在 `routing/templates/<name>/template_spec.json` 中写一个模板 proposal：

- `template_name` 必须体现 shape 特征，不能使用 `template1` 这类无语义名称；
- `worst_case_id` 必须是 `remaining[0]`；
- `predicate` 只使用 host 侧可静态计算的 2~4 个强特征；
- `derived` 可把重复子表达式命名；
- `tiling_plan` / `tiling_intent` 必须能通过核数、tile、UB、对齐和尾块检查；
- `kernel_file` 只作为后续 Step 2 的规划提示，本 skill 不创建该文件。

```bash
D="$WORK/routing/templates/<name>"
python3 "$SK/greedy_route.py" validate \
  --state "$WORK/routing/coverage/state.json" \
  --proposal "$D/template_spec.json" -o "$D/proposal_result.json"
python3 "$SK/greedy_route.py" commit \
  --state "$WORK/routing/coverage/state.json" \
  --proposal "$D/template_spec.json"
```

`commit` 在写 state 前会对本组全部 shape 执行 tiling 校验：

1. 核数 `num_cores` 在 `[1, aiv_core_num]`；
2. `tile_elems >= 1`，且 UB 占用不超过 `0.8 * ub_size`；
3. 末维对齐满足 `alignment.require_inner_align` 要求；
4. 尾块：`tail_elems = total_elems % tile_elems`，非零时 `tail_policy` 必须为 `mask` / `pad` / `handle` 等显式处理策略，不能为 `none` / `forbid`；
5. 表达式可安全求值。

任一校验失败都会直接拒绝，state 不变。重复 `next → proposal → validate → commit`，直到 `remaining` 为空，然后冻结：

```bash
python3 "$SK/greedy_route.py" freeze \
  --state "$WORK/routing/coverage/state.json" \
  -o "$WORK/routing/coverage/plan.json"
```

冻结检查：100% 覆盖、case 不重叠、模板名/tiling_key/priority 唯一、每个模板有代表 shape。允许形成单 shape 模板，但产物会标记 `is_single_case_group=true`，调用方不得将单点结果外推到其他 shape。

## 7. 生成路由表与分类报告

```bash
python3 "$SK/route_report.py" \
  --state "$WORK/routing/coverage/plan.json" \
  --limits "$WORK/theoretical/shape_limits.json"
```

`route_report.py` 只消费冻结计划，一次生成 `tiling_routes.json`、`tiling_table.csv`、`coverage_report.json` 和 `classification_report.md/.html`。命中模板的 shape 会解析出 `num_cores`、`tile_elems`、`tail_policy`；没有模板命中的 shape 记为 `unrouted`，属于冻结计划错误。`original` 仅作为路由表的默认兜底，不再表示“优化失败回退”。

## 8. 门控

| 门控 | 检查 |
|---|---|
| cases / profiling | case_id 对齐，每个 shape 有实测耗时 |
| golden flow | 表达式可求值，`T_hw > 0`，不支持项显式报错 |
| 理论模型 | 实测不得低于理论下限（允许测量误差） |
| 归类顺序 | 每轮锚点必须是当前 `remaining[0]`，谓词必须命中锚点 |
| 谓词 | 非空、可求值、只命中 remaining 中的 shape |
| tiling | 核数、tile、UB、对齐、尾块（`tail_elems` 与 `tail_policy`）检查通过 |
| 冻结计划 | 全覆盖、不重叠、name/key/priority 唯一、包含代表 shape |
| 路由表 | 冻结计划全覆盖，无 `unrouted`，tiling 可解析 |

失败时停止并报告具体 case/模板；不要通过放宽阈值、静默跳过 case 或伪造 profile 继续。

## 9. 明确不做什么

本 skill **不执行**以下工作：

- 不启动优化子 agent，不修改 kernel、host tiling 或构建文件；
- 不选择/调用外部 optimizer skill；
- 不做模板级并行优化、结果回填或性能 A/B 优化；
- 不拼装 `final_code`，不生成路由头文件，不改 CMake/plugin 注册；
- 不删除、覆盖调用方的原始算子代码。

Step 1 完成后，调用方拿到的是可复核的 shape 分类、代表 shape、模板谓词、逐 shape tiling 参数和冻结计划。
