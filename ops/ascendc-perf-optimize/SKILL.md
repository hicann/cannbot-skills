---
name: ascendc-perf-optimize
description: Ascend C 算子性能优化策略制定。结合 Tiling 建模与流水分析（仿真图 + profiling 数据），按卡间/核间/核内三层流水制定性能优化策略，并回修 Tiling 参数。触发：算子性能调优、流水分析、Tiling 修正、bound 诊断时。
---

# Ascend C 算子性能优化流水分析

> **⚠️ 重要**：本技能定义了 **4 步分层流水**优化流程（Step 1 → Step 2 → Step 3 → Step 4），而非简化的 3 步流程。Step 2（卡间流水）、Step 3（核间流水）、Step 4（单核流水）是三个独立的流水分析步骤，各有明确的适用条件、输入输出要求。请勿将它们合并为单一的"流水分析"步骤。

## 优化流程总览

| 步骤 | 名称 | 适用条件 | 关注点 |
|------|------|---------|--------|
| **Step 1** | Tiling 理论建模 | 所有算子 | 确定理想 tiling data |
| **Step 2** | 卡间流水优化 | 通信类算子 | 通算演算、卡间通信瓶颈 |
| **Step 3** | 核间流水优化 | 多核间同步算子 | 核间并行效率、同步开销 |
| **Step 4** | 单核流水优化 | 所有算子 | 核内流水、bound 诊断 |

**关键特征**：
- ✅ **4 步分层流水**，不是 3 步
- ✅ Step 2/3 有明确的**适用条件**，非适用算子跳过
- ✅ 每步都有独立的**仿真图分析 + profiling 报告 + 优化策略**输出
- ✅ Step 2/3/4 都会输出对 Step 1 tiling 策略的**修正建议**

## 优化流程

```
给定：算子类型、计算流程（kernel 代码/伪代码）、profiling 数据、仿真图（可选）

Step 1 — Tiling 理论建模 → 输出理想 tiling data
Step 2 — 通信类算子？ → 通算演算优化策略（卡间流水分析 + tiling 修正）
Step 3 — 多核间同步算子？ → 核间流水优化策略（核间流水分析 + tiling 修正）
Step 4 — 单核流水优化策略（核内流水分析 + tiling 修正）
```

每步的"流水"分析包含**仿真图解读**和 **profiling 数据**分析报告。Step 2/3/4 的策略输出均包含对 Step 1 tiling 策略的修正建议。

---

## Step 1 — Tiling 理论建模

**输入**：算子类型、Shape、dtype、计算流程

**过程**：根据算子 pattern 路由到对应的 Tiling 理论模型目录（详见 `references/tiling/`，入口为 `references/tiling/index.md`），输出理想 tiling data。

**输出**：
- [ ] 卡间切分方案（切分维度、通信内算子涉及）
- [ ] 多核切分方案（切分维度、单核任务量、核数）
- [ ] 单核切分方案：
  - Cube/融合类：L1 split（baseM/baseN/baseK、L1 ping-pong）+ L0 split（mL0/nL0/kL0）
  - Vec 类：UB split（block_size、repeat）
- [ ] Buffer 规划（各 buffer 用途与大小，区分 L1/L0/UB 层级）
- [ ] 分支场景覆盖（dtype、shape 大小、对齐）

---

## Step 2 — 通算演算优化策略（卡间流水）

**适用条件**：通信类算子（如 AllReduce、AllGather、ReduceScatter 等）。

非通信类算子**跳过**此步骤。

**输入**：Step 1 的 tiling data + 计算流程 + 仿真图 + profiling 数据

**过程**：加载 `references/comm-compute/`，分析卡间通信与计算的流水重叠，识别通信瓶颈。

**输出**：
- [ ] 卡间流水仿真图分析
- [ ] 通信/profiling 数据报告
- [ ] 卡间流水优化策略
- [ ] 对 Step 1 tiling 策略的修正建议

> 当前 `references/comm-compute/` 内容为空，此步骤返回「通算演算优化策略暂未收录，跳过卡间流水分析」。

---

## Step 3 — 核间流水优化策略

**适用条件**：涉及多核间同步的算子（如跨核同步、核间数据依赖等）。

非多核同步算子**跳过**此步骤。

**输入**：Step 1 的 tiling data + 计算流程 + 仿真图 + profiling 数据

**过程**：加载 `references/inter-core-pipeline/`，分析多核间的流水并行效率。

**输出**：
- [ ] 核间流水仿真图分析
- [ ] 核间 profiling 数据报告
- [ ] 核间流水优化策略
- [ ] 对 Step 1 tiling 策略的修正建议

> 当前 `references/inter-core-pipeline/` 内容为空，此步骤返回「核间流水优化策略暂未收录，跳过核间流水分析」。

---

## Step 4 — 单核流水优化策略

**适用条件**：所有算子。

**输入**：Step 1 的 tiling data + 计算流程 + 仿真图 + profiling 数据 + 前几步的策略输出

**过程**：加载 `references/single-core-pipeline/`，通过仿真图数据和 profiling 数据判定 bound 类型，按 bound 展开优化。

**输出**：
- [ ] 核内流水仿真图分析
- [ ] 核内 profiling 数据报告（bound 诊断结论）
- [ ] 核内流水优化策略
- [ ] 对 Step 1 tiling 策略的修正建议
- [ ] 最终优化方案汇总

---

## ⚠️ 常见误解澄清

### 误解1：将流程简化为 3 步
❌ **错误理解**：Tiling 建模 → 流水分析 → Tiling 回修  
✅ **正确理解**：Tiling 建模（Step 1）→ 卡间流水（Step 2，条件）→ 核间流水（Step 3，条件）→ 单核流水（Step 4）

**说明**：Step 2/3/4 不是一个"流水分析"步骤的不同维度，而是三个独立的步骤，各有明确的适用条件和输出要求。

### 误解2：将 Step 2/3/4 合并为"流水分析"
❌ **错误**：Step 2/3/4 是同一个"流水分析"步骤的卡间、核间、核内三个维度  
✅ **正确**：Step 2/3/4 是三个独立的分析步骤，按条件选择性执行，各自产生独立的优化策略输出

**说明**：
- Step 2 仅适用于通信类算子（如 AllReduce）
- Step 3 仅适用于多核间同步算子
- Step 4 适用于所有算子
- 非适用算子会跳过对应步骤

### 误解3：混淆卡间通信和核内搬运的概念
❌ **错误**：认为卡间流水只关注计算，不关注数据搬运  
✅ **正确**：Step 2 卡间流水关注卡间数据通信与计算的重叠（通算演算）

**说明**：
- 卡间通信涉及的 DDR→UB 是指**远端卡的数据**通过卡间链路（HCCS/NVLINK）传输到**本卡 UB**
- 这与 Step 4 核内流水中的 DDR→UB（**本卡** HBM/DDR → 本卡 L1 → 本卡 UB）是不同的数据路径
- Step 2 关注：远端数据通过卡间链路到达本卡，与本卡计算的流水重叠（通算演算优化）
- Step 4 关注：本卡内部的数据搬运（本卡 GM→L1→UB）与本卡计算的流水重叠

### 误解4：所有算子都执行全部 4 个步骤
❌ **错误**：每个算子都要经过 Step 1 → Step 2 → Step 3 → Step 4  
✅ **正确**：Step 1 和 Step 4 是必选，Step 2/3 根据算子类型选择性执行

**典型流程示例**：
- **普通计算算子**（如 MatMul）：Step 1 → Skip Step 2 → Skip Step 3 → Step 4
- **通信类算子**（如 AllReduce）：Step 1 → Step 2 → Skip Step 3 → Step 4
- **多核同步算子**：Step 1 → Skip Step 2 → Step 3 → Step 4
