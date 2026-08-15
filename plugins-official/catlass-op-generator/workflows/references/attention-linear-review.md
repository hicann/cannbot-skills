# Attention / State Recurrence Review Gate

本文是 Linear Attention / GDN / KDA / retention / RWKV / state recurrence 场景的专项审查门禁。公共审查清单只负责按 operator type 路由到本文；具体 LA-OS 证据和逐项检查在本文维护。

## 触发条件

需求、DESIGN.md 或代码命中以下任一信号时启用：

```text
linear attention, flash linear attention, GDN, gated delta rule, KDA,
Kimi Delta Attention, retention, RWKV, state recurrence, chunk_gdn,
cumsum/scan inside attention
```

非此类算子跳过本文，不影响公共 catlass C1-C11。

## 快速证据命令

```bash
test -f operators/{operator_name}/docs/OPEN_SOURCE_ALIGNMENT.md
grep -n "reference_source" operators/{operator_name}/docs/OPEN_SOURCE_ALIGNMENT.md
grep -n "OPEN_SOURCE\\|USER_LOCAL" operators/{operator_name}/docs/OPEN_SOURCE_ALIGNMENT.md
grep -n "evaluation_baseline\\|仅评测\\|禁止作为实现参考" operators/{operator_name}/docs/OPEN_SOURCE_ALIGNMENT.md
grep -n "clone_status\\|CLONED\\|UNAVAILABLE" operators/{operator_name}/docs/OPEN_SOURCE_ALIGNMENT.md
grep -n "scale\\|mask\\|clamp\\|cast\\|layout\\|varlen\\|workspace\\|tiling" operators/{operator_name}/docs/OPEN_SOURCE_ALIGNMENT.md
```

## LA-OS 逐项门禁

LA-OS 是 PASS 前置条件，不能降级为 PASS WITH NOTES。REVIEW.md 必须逐项列出 LA1-LA18 状态、证据路径和失败原因。

| # | 检查项 | 证据要求 |
|---|---|---|
| LA1 | 已区分 full-flow / stage operator，并冻结同语义 baseline | DESIGN.md 对应章节、baseline/evaluation_baseline 说明 |
| LA2 | 已画 dependency graph，stage 切分来自依赖而非公式顺序 | DESIGN.md dependency graph / stage table |
| LA3 | 每个中间量有 producer、consumer、pipe、memory、lifetime | DESIGN.md 中间量表 |
| LA4 | GM workspace slot 和 flag 协议稳定，consumer 在 stage 入口 wait | DESIGN.md workspace/flag 章节、op_kernel 证据 |
| LA5 | L1 resident 与 scratch 生命周期分离 | DESIGN.md L1 预算和地址布局 |
| LA6 | L0/L0C/fixpipe 边界正确，split accumulation 只在末 subtile 写回 | DESIGN.md、op_kernel 或自定义组件证据 |
| LA7 | UB input/compute/output buffer 未混用 | DESIGN.md UB 预算、自定义 Tile 证据 |
| LA8 | GQA/GVA 的 K-side cache 按 HK 作用域管理 | DESIGN.md head mapping、shape case 证据 |
| LA9 | Shape 覆盖对齐 `shape-constraints.md` Δ5，smoke 不冒充完整覆盖 | PLAN.md、gen_data.py、报告 case 表 |
| LA10 | 精度报告使用 mixed tolerance 固定字段，阈值来源对齐 `ops-precision-standard` | verify_result.py、precision report |
| LA11 | 性能报告包含 baseline、launch count、workspace peak 和 profiler path | perf summary / profiler path |
| LA12 | 多 stage / A2/A3 场景已读取并执行 A2/A3 stage checklist | DESIGN.md/PLAN.md 引用和映射 |
| LA13 | GDN/KDA 需求已读取 open-source-linear-attention-map.md，并记录参考路径 | OPEN_SOURCE_ALIGNMENT.md |
| LA14 | KDA dAv varlen/partial 已区分物理 `BT` 与 `validRows` 掩码/writeback | DESIGN.md、tiling/op_kernel |
| LA15 | 近零输出由 `atol` 兜底，并记录 matched_ratio/max_abs | precision report |
| LA16 | 用户数学 contract 已冻结到 golden、verify、README 和报告 | 相关文件路径 |
| LA17 | 非 GEMM 逻辑已封装为 Catlass-style 自定义 Block/Tile 或明确 stage 化，device 主路径不为空且 host 不做真实计算 | 自定义组件路径、kernel 入口、host grep 证据 |
| LA18 | `reference_source` 合法；evaluation_baseline 不作为实现参考；OPEN_SOURCE clone 失败时有降级依据 | OPEN_SOURCE_ALIGNMENT.md |

## 判定规则

- 任一 LA 项失败，LA-OS 总判定为 FAIL。
- `OPEN_SOURCE` 下单纯 clone 失败不是 FAIL，前提是记录 `clone_status=UNAVAILABLE`、失败原因和降级依据。
- 用户未显式给本地实现参考路径时，任何开发机本地实现、历史算子目录或当前工作区外同名实现作为 primary reference 都是 FAIL。
- 用户给出的 baseline / 评测 / 性能对比路径只能作为 `evaluation_baseline`，不能作为 source-of-truth 或实现 pipeline。
