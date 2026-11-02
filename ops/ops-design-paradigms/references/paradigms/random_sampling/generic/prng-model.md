<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式随机流模型

> **层级**：通用知识层（generic/），语言中立。
>
> 核心问题：counter-based PRNG 的状态代数、跳越与多执行单元的不重叠划分。本文是本范式的数学根基。

## 1 流代数

随机流是状态机 `S = (key, counter)`：

```
step(S) → (S', u[0..k-1])     # 一次推进：产出 k 个随机数，counter 前进 1
```

- **确定性**：`step` 是纯函数——固定 `(key, counter₀)` 与推进路径，产出序列逐位可复现（与执行单元数、切分方式无关）
- **k（步产出）**：单次推进产出 k 个随机数（Philox4x32 族 k=4）
- **轮数硬约束（R1）**：流的轮数是流身份的一部分。实现接口的**默认轮数可能与流基准不一致**（如模板默认 7 轮 vs 原生流 10 轮）——必须显式指定并逐位核对；漏显式 → 流错且**逐位验收无法发现**（自身复现一致，但与流基准/竞品不匹配）
- **流身份与位置的分离**：key 决定"哪条流"，counter 决定"流内位置"；同一 key 不同 counter 区间产出不重叠的序列

## 1.1 接口族统一登记表（R2/R3 + seed 分型 + offset 单位）

`(seed/offset/key/counter)` 到流状态 `(key, counter)` 的映射**按接口族逐位登记**，四族并存不可混用。本表是 key/counter 布局（R2/R3）、offset 单位、seed=0 语义（见 P3 分型）、状态形态与回写的**唯一登记源**，[state-management.md](state-management.md) §1/§2 引用本表：

| 族 | 输入形态 | key 布局 | counter/offset 布局 | offset 单位与换算 | seed=0 语义 | 状态形态/回写 | **实现路线（R20）** | 实例 |
|----|---------|---------|---------------------|------------------|------------|--------------|------|
| **A 计数器直给** | key/counter 输入张量 | key_data[0] → key[0:2] | counter_data[0] → counter[0:1]（低 64 位）、counter_data[1] → counter[2:3]（高 64 位） | counter 步数（64 位直拆） | 无 seed 概念（key 直给，恒确定） | stateless / 无回写 | **SIMD**（TQue 流水；key/counter 张量接口族先例均纯 SIMD） | stateless_random_uniform_v2、TF V2 接口 |
| **B seed+offset 属性** | seed/offset 属性 | seed 低/高 32 位 → key[0:2] | offset 经 `⌈offset/k⌉` 换算后 SkipLo 进 counter[0:1]（**position 字段**） | 元素数（÷k 换算为步数；offset 须对齐 k） | **裸 key，0 合法且确定** | 带状态（offset 张量族有回写） | **SIMT**（GM 直写；torch 语义 self 填充，PhiloxSimtKernel 模板族） | sim_thread_exponential、random_common SIMT 模板（PhiloxAlgParsInit） |
| **C seed+seed2 属性（TF）** | seed/seed2 属性 | seed 低/高 32 位 → key[0:2] | seed2 → counter[2:3]，counter[0:1] 从零起算 | —（seed2 即计数高段） | **seed=0 且 seed2=0 → 时间熵非确定模式**；否则确定 | 无状态（每次调用重派生）/ 无回写 | **SIMD**（TQue 流水；TF 属性族先例均纯 SIMD） | random_uniform_v2 |
| **D torch seed/offset 桥接** | seed/offset 张量（philox_engine_inputs 桥接） | seed 低/高 32 位 → key[0:2] | offset → counter[2:3]（**subsequence 字段，不得写入 position 字段 counter[0:1]**） | 原始计数（直拆入 subsequence） | **裸 key，0 合法且确定** | 带状态：offset 由调用方管理；新 kernel host 透传零推进，组合链 legacy 按契约推进（见 §7） | **SIMT**（GM 直写；torch 对标硬指定，禁引 TF/SIMD 系参照） | rrelu_with_noise |

- **错位后果**：字段放错（如 D 族 offset 入 position）→ 流整体错位/错流——不报错，只能靠与流基准的逐位对拍发现
- **seed=0 语义分型（硬规则）**：只有 **C 族**存在"seed=0 → 时间熵"语义；B/D 族 seed 是裸 key 值（0 合法且确定）；A 族无 seed 概念。验证与公共实现**禁止**把 C 族语义泛化到其他族（[verification.md](verification.md) §1.1 的双轨条件按"流确定模式判定"，非字面 seed≠0）
- 设计义务：落位前确定接口族 → 对照本表逐位核对映射与 offset 单位换算

## 2 跳越算子 Skip(n)

从 counter₀ 直接前进 n 步等价于丢弃 n·k 个随机数：

```
Skip(n): counter₀ += n        # 带进位链（counter 为多字整数）
```

- 跳越是 O(1) 状态运算，不消耗计算——用于对齐流起点
- 进位链语义：低位字溢出向高位字进位（多字 counter 的加减法）

## 3 多执行单元的流划分（正确性不变量）

设并行单元数为 B、每输出消耗 c、布局系数 g（见 [splitting.md](splitting.md) §2），单元 p 处理输出索引 i 时的流位置：

```
连续布局：  group(i) = ⌊i / (c·B)⌋            # 单元 p 先用完一组，再进下一组
           counter_p(i) = counter₀ + ⌊i/(c·B)⌋·1 + 单元起始跳越
离散布局：  counter_p(i) = counter₀ + ⌊i/(c·B)⌋·1，组内取第 (i mod B) 路
```

**不变量 I1（区间不重叠）**：任意两个执行单元（或数据块）在同一时刻占有的 counter 区间互不重叠。违反后果：不同输出拿到相同随机数（肉眼不可见的错误，统计检验才能暴露）。

**不变量 I2（全覆盖）**：所有输出的流位置覆盖 `[counter₀, counter₀ + ⌈L/(c·B)⌉·B)`，无遗漏。

**切分不变性（I1/I2 的精确表述，R5）**：**逐元素全局线性下标 keying**——元素 i 的样本 ≡ 第 `⌊i/k⌋` 次推进的第 `(i mod k)` 个产出（lane），与切分方式（核数/块划分/布局）**无关**。任何切分方案都必须化归到这一恒等式；不满足则多核切分破坏流一致性（输出值仍"像随机"但与单核基准不同）。

**组对齐必要条件（R6）**：单元处理量 numPerCore 必须对齐到组粒度 k（如 k=4：`(n+3)&~3`），否则跨核流全线偏移；更强的块对齐（如 32B 粒度）隐含满足之。

## 4 预留律（变长消耗）

固定消耗（c 为常数）的算子流位置可精确推导；**变长消耗**（拒绝式采样，如截断正态：`|z|≥2 则重采样`）无法静态确定每个输出的消耗数。

```
预留律：counter_p(i) = counter₀ + ⌊i / B⌋ · R        # R ≥ 最大重采样数
```

- 每输出预留 R 个随机数配额（实测基准 R=256），组内第 (i mod B) 路
- R 的下界：使拒绝概率下溢（`P(连续拒绝 ≥ R/c 次) < 2^-40` 量级）；R 过小会导致**重样**（配额耗尽回绕到已用区间）
- 状态递进量同步放大：`offset' = offset + L·R`（见 [state-management.md](state-management.md) §3）

## 5 流分割协议（数据块间）

大输出空间切分为多个数据块独立执行时（触发条件见 [splitting.md](splitting.md) §3）：

```
块 b 的流起点：offset_b = offset_{b-1} + rounds(b-1)
rounds(b) = ⌈ numel(b) / (totalThreads(b) · c) ⌉       # 该块的总推进轮数
```

- 每块持有独立的 (流起点, 线程布局)，块间流区间由 offset 累进保证不重叠——不变量 I1 在块维度的扩展
- offset 累进发生在 host 侧切分阶段，kernel 只读本块的起点

## 6 精度与数值注意

- 随机数→目标域的映射区间要明确开闭（`(0,1)` vs `[0,1)`），两竞品接口可不同——数值域差异直接影响逐位对齐
- 变换中的保护边界：`u < eps` 时抬升到 eps（对数/Box-Muller 的下溢保护），是否保护须与竞品对齐（保护则一致确定，不保护则更贴竞品）
- 浮点函数（对数/三角）在不同实现下的 ULP 差异：固定 seed 逐位验证以**本侧 golden 同源实现**为基准，跨实现只保证统计等价

## 7 推进量契约（R4）

每次调用的 offset 推进量是**算子级调用契约**，范式只约束其性质：

| 算子实例 | 推进量 | 契约归属 |
|---------|--------|---------|
| rrelu_with_noise | 12 / 调用 | **组合链 legacy**（torch 桥接 philox_engine_inputs 的调用侧递增；新 kernel 为 host 透传零推进——同算子两代契约不同，按本代实现归属消费） |
| random_uniform_v2 | L × 256 | 新 kernel（底座回写路径） |

约束：① 推进量与流布局/预留一致（跨调用不重叠不跳变）② 写进接口文档 ③ 调用方按契约递增 offset ④ **表格消费前核对契约归属列**——误消费 legacy 契约会使新设计的 offset 链错位。

## 8 随机流不消耗契约（R19）

> 任何**不下发 kernel** 的路径不得推进随机流（counter/offset 状态不变）。

覆盖路径：aclnn 短路、host 侧 fallback、图模式空 tensor 防御。违反后果：流跨调用错位，固定 (seed, offset) 复现链断裂。host 侧对 seed/offset 只做**原样透传**（不推进、不变换），空 tensor 短路时直接返回（流零消耗）。
