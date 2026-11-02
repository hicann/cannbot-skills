<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式场景模型

> **层级**：通用知识层（generic/），语言中立。
>
> 来源：ops-math/random 全家桶（22 个算子 + random_common 公共底座）实测蒸馏（ascend950 / arch35）。

## 1 准入条件

本范式覆盖的算子需同时满足：

1. **输出值由 PRNG 驱动**——输出元素值来自伪随机数生成器的采样（均匀/正态/指数/截断/伯努利/多项/排列等），而非输入张量的确定性函数
2. **随机流为 counter-based**——随机数由 `(key, counter)` 状态经固定轮数的确定性变换产生（Philox 族），流的位置可由偏移量精确跳越
3. **输出规模由 spec 决定**——输出 shape 来自 shape 描述张量（值依赖）或属性，与输入数据值无关

准入判据的边界：输入驱动掩码（mask 由输入比较产生）归 MaskPredicate；洗牌已有数据（值来自输入、仅顺序随机）若不管理随机流状态则不属本范式。

## 2 随机流三要素

| 术语 | 含义 |
|------|------|
| **流 S** | `(key, counter)` 二元组；一次状态推进产出 k 个随机数 |
| **key** | 流身份；由 seed 派生 |
| **counter** | 流内位置；产出第 i 组随机数前的推进量 |
| **offset** | 跨调用/跨执行单元的流起点偏移 |
| **seed 语义** | **按接口族分型**（prng-model §1.1）：C 族（TF）seed=0 → 时间熵非确定；B/D 族 seed 是裸 key（0 合法且确定）；A 族无 seed 概念 |
| **R（预留数）** | 变长消耗算子每个输出预留的随机数配额 |
| **c（消耗数）** | 每个输出消耗的随机数个数（固定 1/2/4 或变长） |

## 3 场景三路由与算子族分类（R17）

**场景路由**（复合算子归类）：

| 路由 | 场景 | 归属 |
|------|------|------|
| (a) 纯随机生成 | 无主计算，输出即随机量 | 单范式本范式 |
| (b) 随机 + 逐元素复合 | 逐元素计算 × 随机采样（rrelu_with_noise 型） | 单范式本范式（**吸收 Elementwise 语义**） |
| (c) 随机 + 规约等复合 | 随机量参与非逐元素主计算 | 多范式并集 |

路由 (b) 的判据：随机采样是主语义（流管理占据设计主体），逐元素计算是随机量的消费者——此时 Elementwise 语义被吸收，单范式归属避免双范式交叉引用。随机噪声写回载体（供 backward 复用的 noise 输出）是 (b) 类的常见输出契约。

**算子族**（路由 (a)/(c) 的生成语义分类）：

| 算子族 | 语义 | 每输出消耗 c | 实例 |
|------|------|------------|------|
| **Distribution** | `y[i] = T(u[i..])`，T 为分布变换 | 1~4 或变长 | uniform / normal / exponential / truncated_normal |
| **MaskGen** | 按概率 p 生成 0/1 或 bool 掩码 | 1 | dropout gen_mask / bernoulli |
| **DiscreteSampling** | 按概率向量选离散结果 | 1~变长 | multinomial / choice_with_mask |
| **Permutation** | 生成 1..n 的随机排列 | 变长（拒绝式） | randperm |

- 掩码族与 MaskPredicate 的边界：掩码的**生成**（随机流驱动）属本范式；掩码的**应用**（掩码乘输入）是 elementwise
- 排列/选择族含排序成分（SortSelect）时，排序是手段，随机流管理仍是主体

与 **Stateful** 的边界：`(seed, offset)` 无状态参数化属本范式；全局 Generator 态（跨算子共享可变 RNG 状态对象）属 Stateful 复合。

## 4 数据流骨架

```
[段 1: 流初始化]   解析 seed/shape/offset → (key, counter₀)，输出规模 L
[段 2: 流切分]     为各执行单元/各数据块推导互不重叠的 counter 区间
[段 3: 采样]       每单元推进流、产出随机数、经分布变换写输出
[段 4: 状态回写]   （带状态接口）offset' = offset + L·R，唯一写者 + 缓存可见
```

- 段 1 的 shape 描述张量是**值依赖输入**（host 侧读值推导规模与输出 shape，契约见 [../../common/value-depend-process.md](../../common/value-depend-process.md)）
- 段 4 只有带状态接口需要；stateless 接口（key/counter 由输入给定）跳过

## 5 能力边界

1. **随机数质量与流语义**——本范式管理流的位置与正确性，不负责分布拟合优度（统计检验属验证手段，见 [verification.md](verification.md)）
2. **真熵源**——硬件真随机不在本范式内（`seed=0` 的时间熵只是"非确定模式"的实现选择，不是密码学熵）
3. **确定性基准**——同一 `(key, counter)` 序列在任何执行单元排布下产出相同随机数序列（不重叠区间划分是正确性不变量，见 [prng-model.md](prng-model.md)）

## 6 依赖谓词

| 谓词 | 用途 |
|------|------|
| `CAP_PRNG_ALG` | 流算法与步产出 k（段 3 粒度；契约表唯一定义源） |
| `CAP_PRNG_VECTORIZED` | 流推进可向量化批量执行 |
| `CAP_INDEX_BIT_WIDTH` | 线性索引位宽（大输出空间切分的触发条件） |
| `CAP_STATE_WRITE_BACK` | 带状态接口的回写（唯一写者 + 缓存可见性） |

谓词定义见 [capability-contract.md](capability-contract.md)；各语言真值见对应适配层 capability-mapping。
