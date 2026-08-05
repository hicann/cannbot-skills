---
name: cannbotdsl-perf-optimize
description: "对已功能正确的 CANNBotDSL kernel 做系统性性能优化时使用，作为跨多层次优化的统一入口和方法论。当用户要求优化 kernel 性能、或对已有 kernel 独立触发调优时触发。提供 4 层优化栈（自底向上：Tiling 优化 → 核内流水 cube-pipeline+vec-pipeline → 宏级 Channel depth-N 流水 → 系统级 AOT 缓存+多核负载均衡），性能分析方法（msprof 采集 + op_summary CSV 解析 + pipe utilization 瓶颈诊断），优化决策树（cube bound / vec bound / memory bound / scalar bound 路由到对应策略），Task Duration 取 min 原则，性能基线建立规范。本 skill 是方法论总入口，具体流水优化下钻到 cube-pipeline/vec-pipeline/channel。Triggers: cannbotdsl 性能优化, perf, 4 层优化栈, pipe utilization, cube bound, vec bound, memory bound, 瓶颈诊断, baseline。Perf-Tuner sub-agent 独立工作流调用。"
---

# cannbotdsl-perf-optimize

CANNBotDSL 性能优化统一方法论。是优化工作的**总入口**：建基线 → 诊断瓶颈 → 按瓶颈路由到具体优化 skill → 复测。Perf-Tuner sub-agent 独立工作流调用。**前置：kernel 必须已功能正确**（精度过关），性能优化不改变数值语义。**需要 NPU**（msprof 采集）。

## 触发条件

- 用户要求优化已有 CANNBotDSL kernel 性能
- Perf-Tuner sub-agent 独立工作流触发

## 优化前提

1. 有一个精度已验证的 baseline kernel（否则先回 Stage 3/4）。
2. 有对齐 shape 的可跑 test。
3. 明确优化目标（延迟下限 / 吞吐 / 特定 shape）。

## 4 层优化栈（自底向上）

| 层 | 优化对象 | 下钻参考 |
|----|----------|---------|
| **1 Tiling** | tile shape 选择、多核拆分、tail block 处理 | `../cannbotdsl-tiling-design/SKILL.md` |
| **2 核内流水** | Cube L0 double buffer；Vec UB double buffer + VF 融合 | `references/channel-pipeline.md`（Cube）、`references/vf-deep.md`（Vec）、`../cannbotdsl-vf-fusion/SKILL.md` |
| **3 宏级流水** | preload：warmup / steady / drain 三阶段、Channel depth-N 环形、DelayLineGroup + stage-gate | `references/depth-design.md`、`references/fa-pipeline-optimization.md`（FA 实战） |
| **4 系统级** | AOT 缓存、多核负载均衡、通信/计算重叠 | `../cannbotdsl-api-reference/SKILL.md` |

## 性能分析方法（`../../debug-skills/cannbotdsl-msprof-compare/`）

- **采集**：msprof 上板，`--task-time=on` 必开；**每个 kernel 重复跑多次**（PipeUtilization 在单次短任务上抓不到 pipe 计数器）；`rm -rf PROF_*` 清旧数据免拼接。
- **解析**：解析 op_summary CSV（见 `cannbotdsl-msprof-compare` skill），输出 Task Duration n/min/mean/max。
- **Task Duration 取 min 原则**：min 才是 kernel 固有成本，mean 会被 HBM cache state 拉偏。

## 瓶颈诊断 → 决策树

看 pipe utilization 与 mac_ratio 判定瓶颈类型，路由到对应层：

| 瓶颈 | 征兆 | 主攻层 |
|------|------|--------|
| **cube bound** | `mac_ratio > 0.9`，Cube pipe 满 | 层 1 tiling（增大 tile 摊薄）、层 2 cube-pipeline L0 DB；已近下限则接受 |
| **vec bound** | Vec pipe 满、Cube 空等 | 层 2 vec-pipeline + VF 融合（减少 vec op 数）、Cube/Vec 负载再平衡 |
| **memory bound** | MTE2/MTE3 满、compute pipe 空 | 层 3 preload（搬运/计算重叠）、层 1 减少重复搬运、提高 L1 复用 |
| **scalar bound** | scalar pipe 占比高 | 减少标量控制、把标量循环改 `range_constexpr` 静态展开 |

## 基线与复测规范

1. **建基线**：记录 baseline Task Duration min + 瓶颈类型 + pipe util，作为对比锚点。
2. **单变量优化**：一次只改一层/一个决策，改完复测。
3. **对比**：用 `cannbotdsl-msprof-compare` 双跑 baseline vs 优化版，看 min 倍数（`1.99x` = 慢 2×）。
4. **精度回归**：每次优化后重跑精度 test，确认数值语义未变（`cannbotdsl-op-test`）。

## 门禁

- 优化不得改变数值语义 —— 每轮优化后必须过精度回归。
- 结论用 Task Duration **min**，并说明瓶颈类型与所在层，不用 mean、不空泛说"更快了"。
- 单变量迭代：一次改一处，可归因；避免多处齐改无法定位收益来源。
- 无 NPU 时 msprof 不可用，明确标 blocked，只能做静态的 tiling/流水设计建议（层 1-2 的设计部分）。

## 参考

- `../../debug-skills/cannbotdsl-msprof-compare/SKILL.md`（采集/解析、min 原则）
- `../cannbotdsl-tiling-design/SKILL.md`（层 1）、`references/channel-pipeline.md` + `references/vf-deep.md`（层 2）、`references/depth-design.md` + `references/fa-pipeline-optimization.md`（层 3）
- `../cannbotdsl-vf-fusion/SKILL.md`（VF 融合减少 vec op）、`../cannbotdsl-op-test/SKILL.md`（精度回归 + L3 性能测试）
- `references/fa-pipeline-optimization.md`（FA 4-stage pipeline 实战：从 318.8us 到 51.5us 的 6.2x 优化路径，含 DelayLineGroup/stage-gate/scf.if 限制/fused vmadd/两遍 softmax 等关键技术）
