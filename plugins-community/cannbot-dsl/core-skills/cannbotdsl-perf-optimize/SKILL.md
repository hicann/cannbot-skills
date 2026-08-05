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
| **3 宏级流水** | preload：warmup / steady / drain 三阶段、Channel depth-N 多级缓冲、DelayLineGroup + stage-gate | `references/depth-design.md`、`references/fa-pipeline-optimization.md`（FA 实战） |
| **4 系统级** | AOT 缓存、多核负载均衡、通信/计算重叠 | `../cannbotdsl-api-reference/SKILL.md` |

## 性能分析方法（`../../debug-skills/cannbotdsl-msprof-compare/`）

- **采集**：msprof 上板，`--task-time=on` 必开；**每个 kernel 重复跑多次**（PipeUtilization 在单次短任务上抓不到 pipe 计数器）；`rm -rf PROF_*` 清旧数据免拼接。
- **解析**：解析 op_summary CSV（见 `cannbotdsl-msprof-compare` skill），输出 Task Duration n/min/mean/max。
- **Task Duration 取 min 原则**：min 才是 kernel 固有成本，mean 会被 HBM cache state 拉偏。

## 瓶颈诊断 → 决策树

### 第 0 步（先做，否则下面整张表可能指错方向）：核对每核负载

**msprof 的 pipe ratio 描述的是「被 profile 的那一个核」，不是整个核阵。** `aiv_vec_ratio = 0.85` 的意思是"这个核 85% 时间在忙"，它**无法区分**两件事：

- 该单元本身吃满了（→ 该去减这个 pipe 的工作量）；
- **这个核被分到了远超平均的活，其他核在等它**（→ 该去改分发，别动 pipe）。

kernel 的墙钟由最慢的核决定，两种情况下"忙核"都会被 profile 到、都显示高 ratio，**计数器长得一模一样**。

先用几行 Python 从分发算式直接算，纯 host 侧算术、不用上板：

```python
# dispatch 形如 for tidx in range(get_block_idx(), TOTAL, get_block_num())
load = [sum(cost_of_tile(t) for t in range(c, TOTAL, GRID)) for c in range(GRID)]
print(max(load) / (sum(load) / GRID))     # > 1.2 → 先修分发，ratio 先别信
```

`cost_of_tile` 要用**真实代价**（如 causal 下该 tile 实际迭代的 kv 块数），不是 tile 计数 —— 均分 tile 数不等于均分工作量。

**什么时候特别容易中招**：tile 代价沿某个轴变化（causal 的 `nkb = mb+1`、变长序列、稀疏块数），**且**该轴 extent 与 round-robin 步长有公因子。极端情形是 `idx2crd` 把该轴放在**最内层**且 `extent | GRID` —— 该轴取值**每核恒定**，最贵的核永远最贵。

**修法通常是把该轴挪到 `idx2crd` 维度表的最外层**（成为最慢变化轴）。这是**纯排列**：同一批 tile、同样的坐标、只换核归属，数值逐位不变，只需确认精度未动、无需重验算法。

> **实测**（GQA causal，`NMB=8`、`GRID=32`）：`idx2crd(tidx, [B, NKV, G, NMB])` 把 `mb` 放最内层 → 0 号核永远拿 1 个 kv 块的 tile、7 号核永远拿 8 个 → max/mean = **1.78**。挪到最外层后 max == mean，**净得 1.61×，一条向量指令没改**。当时 pipe ratio 是 `VEC 0.854 / mac 0.082`，按下表会直接判成 "vec bound → 去减 vec op"，方向就错了。

### 第 1 步：按 pipe utilization 判定瓶颈类型

看 pipe utilization 与 mac_ratio 判定瓶颈类型，路由到对应层：

| 瓶颈 | 征兆 | 主攻层 |
|------|------|--------|
| **cube bound** | `mac_ratio > 0.9`，Cube pipe 满 | 层 1 tiling（增大 tile 摊薄）、层 2 cube-pipeline L0 DB；已近下限则接受 |
| **vec bound** | Vec pipe 满、Cube 空等（**先过第 0 步**） | 层 2 vec-pipeline + VF 融合（减少 vec op 数）、Cube/Vec 负载再平衡 |
| **memory bound** | MTE2/MTE3 满、compute pipe 空 | 层 3 preload（搬运/计算重叠）、层 1 减少重复搬运、提高 L1 复用 |
| **scalar bound** | scalar pipe 占比高 | 减少标量控制、把标量循环改 `range_constexpr` 静态展开 |

### 第 2 步（判为 compute bound 后）：先测吞吐受限还是延迟受限，别靠推算

下一个岔路是"减少指令数"还是"重排指令让依赖链重叠"。两条路收益天差地别，且**不能靠 `cycles/op vs 理想 IPC` 推断** —— "理想 1 IPC" 通常没有依据，用它算出的"理论下限"会把单指令固有代价误读成可回收的延迟缺陷。

**直接测，两种办法都很便宜**：

1. **看历史收益比例**：之前几次删指令的收益若与删除比例大致成正比（删 20% 省 20%），就是**吞吐受限**；延迟受限的段删指令回报会明显低于指令占比。
2. **直接改并行度**：把独立单元（如逐行循环）展开并交错，测 UNROLL 1 / 2 / 4。时间基本不变 = 吞吐受限。

> **实测**（同一 GQA vec 段）：UNROLL 1/2/4 = 717.8 / 715.6 / 716.6 µs —— **并行度变 4 倍，时间差 0.3%**。此前基于 "3.8 cycles/op vs 理想 1 IPC" 得出的"延迟受限、应交错"结论是错的。
>
> **判定结果直接决定策略**：吞吐受限下**只有删掉工作才有用** —— 重排、交错、软件流水化行循环全部无效。该找的是冗余指令（重复 load/broadcast、可被融合指令替代的组合、可整体跳过的分支路径），不是调度。


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
