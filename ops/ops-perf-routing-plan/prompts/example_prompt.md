# MoeInitRoutingV3 Shape 分级与 tiling 路由（Step 1）

使用 `ops-perf-routing-plan` skill，对 `MoeInitRoutingV3` 完成 Shape 级 tiling 路由的 Step 1：
理论分析、worst-first 分组、典型 shape 选择、tiling 可行性校验和冻结计划。不要修改 kernel，
不要启动优化子 agent，不要拼装 final code。详细流程以 `$TR_SKILL/SKILL.md` 为准。

## 目标

- 只处理用例文件中属于 `MoeInitRoutingV3` / `moe_init_routing_v3` 的 shape。
- 产出 100% 覆盖、互不重叠、可复核的 shape→模板冻结计划。
- 每个模板明确代表 shape、谓词、tiling 参数和预期瓶颈，供后续 Step 2 使用。

## 环境

```bash
# 替换所有 <...> 占位符为实际路径
export CANNBOT_SKILLS_ROOT="<cannbot-skills>"
export TR_SKILL="$CANNBOT_SKILLS_ROOT/ops/ops-perf-routing-plan"

export OP_NAME="moe_init_routing_v3"
export OP_CLASS="MoeInitRoutingV3"
export BR="<ops-transformer 仓>"
export OP_REL="moe/moe_init_routing_v3"
export CASES="<网络用例 xlsx/csv 提取后的 cases.csv>"
export PERF_JSON="<已采集的 performance.json>"

export OUT_ROOT="<输出根目录>"
export OP_WORK="$BR/$OP_REL"
export CHIP="910b2"
export SOC="ascend910b"
export TILING_KEY_BASE="2400000"
```

## 要求

1. 先完整读取 `$TR_SKILL/SKILL.md`，严格只执行 Step 1。
2. 用 `prepare.py init` 初始化：

   ```bash
   WORK=$(python3 "$TR_SKILL/scripts/prepare.py" init \
     --op-dir "$OP_WORK" --build-root "$BR" --op-rel "$OP_REL" \
     --op-name "$OP_NAME" --cases "$CASES" --out-root "$OUT_ROOT" \
     --chip "$CHIP" --soc "$SOC" --tiling-key-base "$TILING_KEY_BASE")
   ```

3. 将已有 profiling 结果归档为 baseline：

   ```bash
   python3 "$TR_SKILL/scripts/prepare.py" profile \
     --work "$WORK" --perf-json "$PERF_JSON"
   ```

4. 检查并按算子语义补全 `$WORK/inputs/golden/golden_flow.json`，然后执行
   `shape_limits.py`。不支持的理论项必须显式报告。
5. 串行执行 `greedy_route.py init/next/validate/commit`，直到 `remaining` 为空；
   commit 会校验当前组全部 shape 的 tiling 可行性。
6. 执行 `freeze` 和 `route_report.py`。
7. 不修改共享仓 `$BR`，不修改算子 kernel、host tiling、CMake 或 plugin 注册。

## 交付

- `$WORK/routing/coverage/plan.json`：冻结的 shape 分类计划
- `$WORK/routing/tiling_routes.json`：模板路由和逐 case 映射
- `$WORK/routing/tiling_table.csv`：逐 case tiling 参数
- `$WORK/routing/coverage_report.json`：覆盖检查
- `$WORK/routing/classification_report.md/.html`：理论排名、分组、代表 shape 和单 shape 组提示

完成后停止。不要继续做模板优化、性能 A/B 或代码合并。
