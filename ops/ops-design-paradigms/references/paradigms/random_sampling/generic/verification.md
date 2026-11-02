<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式验证方法

> **层级**：通用知识层（generic/），语言中立。

## 1 三段式精度验收（范式标准模板，R13）

| 段 | 方法 | 判据 |
|----|------|------|
| ① **capture-replay 恒等** | 以算子写回的随机量为**已知量**（capture），独立重放主计算 `out ≡ f(x, noiseOut)` 逐位互验 | 排除随机性后主计算链路的正确性 |
| ② **固定 (seed, offset) 双跑**（R11） | 同输入/同设备/同 dtype 两次执行 | bitwise 一致（确定性建模 `nondeterministic_seeded`） |
| ③ **统计校验** | 值域逐点（[lower,upper)）+ 分布拟合（K-S / 卡方，α=0.01）+ 独立性 | 统计容差内 |

三段缺一不可：① 验主计算、② 验流确定性、③ 验分布质量。per-dtype 容差分档随 ③ 声明。

**多输出恒等判据（R16 泛化）**：随机量写回型算子（noise/mask 写回供 backward 复用），`out ≡ f(x, noiseOut)` 的数学恒等作为范式级验收项（① 的判据本体）。

**oracle 缺席规则（R12）**：随机流与竞品不逐位对齐（流协议不同）时，formula 值等价**禁止逐位对拍**——spec 声明 `reference_oracle.absent`，验收走三段式（①②③），竞品比对降级为统计等价。

**样本量契约（R14）**：单调用样本量 < 10^5 时，统计检验按**跨调用累积**执行（offset 按推进量契约递增的批调用序列）。

## 1.1 双轨验证（按流确定模式判定）

| 轨道 | 条件（**流确定模式判定**，非字面 seed≠0） | 判定 |
|------|------|------|
| **逐位轨** | 流确定模式：A/B/D 族恒确定（含 seed=0 用例——裸 key，合法且可复现）；C 族 seed≠0 | 输出与 golden **逐位一致**；跨切分方式（改单元数/块划分/布局）结果不变——不变量 I1/I2 的直接验证 |
| **统计轨** | 流非确定模式（仅 C 族 seed=0 且 seed2=0 → 时间熵） | 只验分布正确性：均值/方差/矩/直方图拟合（如 normal 的 KS 检验）；不逐位比对 |

逐位轨是主力：确定模式下随机算子退化为**确定性算子**，可用常规 golden 流程。统计轨只用于 C 族 seed=0 冒烟与分布回归。分型依据见 [prng-model.md](prng-model.md) §1.1 / [state-management.md](state-management.md) §1。

## 2 golden 构造

- golden 用**同源实现**（与 kernel 相同的算法常数：轮数、变换公式、开闭域、eps 保护、预留 R）——跨实现（如竞品 CPU 库）存在 ULP 差异，只能统计等价
- 竞品逐位对齐验证单独做一轮：以竞品输出为 golden 的逐位比对，暴露开闭域/eps/拒绝边界的差异点（记录为已知差异或修复）

## 3 边界用例（必覆盖）

| 用例 | 验证点 |
|------|--------|
| **固定 seed 确定性** | 同 seed 两次调用逐位一致；不同 seed 结果不同 |
| **空输出**（L=0 / shape 空） | 不崩溃、单元数下限路径、**流零消耗**（offset 不推进，prng-model §8）、offset 递进仍正确（L·R=0） |
| **标量输出**（shape 描述为空/标量） | 规模 1 路径 |
| **大输出超索引位宽** | 块协议触发、报错路径（超块数上限）、块间流不重叠（抽样逐位比对） |
| **offset 连续调用链** | 调用两次（同一 offset 张量），两段输出不重叠（拼接后做分布检验 + 抽样比对 golden 的流推进） |
| **变长消耗预留** | 截断正态拒绝路径：尾部概率场景（限制域小时拒绝率高）不产生重样 |
| **seed=0 冒烟** | 跑通 + 输出非全 0/非全同值 |
| **dtype 全档** | fp32/fp16/bf16（含 cast 路径）；整数输出档（uniform_int） |
| **概率边界** | bernoulli p=0/p=1；multinomial 概率含 0、未归一化 |
| **区间端点** | uniform [a,b) 的 a、b−ulp；uniform_int 上下闭域 |
| **多输出恒等** | `out ≡ f(x, noiseOut)` 逐位互验（写回型算子，§1 R16） |
| **跨切分不变** | 改核数/块划分后同 (seed,offset) 输出 bitwise 一致（切分不变性） |

## 4 精度标准

- 逐位轨：逐位一致（无容差）——同源 golden
- 统计轨：均值/方差误差 < 统计容差（按样本量给 σ 界，如 |mean − μ| < 5σ/√L）
- dtype 降精度（fp16/bf16 输出）：与同源 golden 逐位一致（cast 确定性）

## 5 性能验证要点

- 吞吐基准：元素/秒，随 L 缩放曲线（小 L 看调度开销，大 L 看带宽/流推进）
- **切分敏感性**：改执行单元数/块划分不改逐位结果（正确性）但改性能——性能报告须固定切分配置
- 变换函数是热点（对数/三角）：SIMT 标量库调用 vs 向量化实现的分档报告

## 6 验证输出清单

- [ ] 双轨验证策略声明（逐位主力 + 统计辅助）
- [ ] golden 同源性说明（算法常数清单）
- [ ] 边界用例表全覆盖
- [ ] offset 链与块不重叠的专项用例
- [ ] "待验证"项（跨实现 ULP、竞品逐位差异点）列入验证义务
