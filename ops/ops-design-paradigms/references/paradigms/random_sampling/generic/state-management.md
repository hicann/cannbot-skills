<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式状态管理

> **层级**：通用知识层（generic/），语言中立。
>
> 核心问题：seed 语义、跨调用状态递进与接口形态分类。

## 1 seed 语义（按接口族分型）

**seed=0 语义不是全局规则**——按 [prng-model.md](prng-model.md) §1.1 接口族分型：

| 族 | seed=0 语义 | 验证口径 |
|------|------------|---------|
| C seed+seed2 属性（TF） | **时间熵非确定模式**（seed=0 且 seed2=0 触发） | 统计轨 |
| B seed+offset 属性 / D torch 桥接 | **裸 key，0 合法且确定** | 逐位轨（含 seed=0 用例） |
| A 计数器直给 | 无 seed 概念（key 直给，恒确定） | 逐位轨 |

- 熵源在 host 侧解析（每次调用重新取熵），device 侧无感知
- 设计含义：**按族判定验证轨道**（"流确定模式判定"）；只有 C 族需要"测试固定 seed≠0"约束，B/D 族的 seed=0 是合法确定性用例（错杀即假失败）

## 2 接口形态分类（流来源）

形态分类与 [prng-model.md](prng-model.md) §1.1 接口族统一登记表**同源**（族 → 状态形态/回写必要性列），不再单独维护对照：

| 形态 | 对应族 | 状态回写 |
|------|--------|---------|
| **stateless** | A 计数器直给 | 无 |
| **seed 属性 + offset 张量** | B seed+offset 属性（offset 张量变体） | **有**（offset 递进，§3） |
| **seed/seed2 属性** | C seed+seed2 属性（TF） | 无 |
| **torch 桥接 seed/offset 张量** | D torch 桥接 | 调用方管理（新 kernel host 透传零推进；legacy 按契约推进，见 prng-model §7） |

判定法则：接口含可变 offset 张量 → 带状态（需回写）；key/counter 为输入张量 → stateless。

## 3 offset 递进协议（带状态接口）

```
offset' = offset + L · R          # L=本次输出规模，R=每输出预留数
```

| 约束 | 说明 |
|------|------|
| **唯一写者** | 恰好一个执行单元（如 0 号核或尾核）执行回写，多写者竞争产生丢失更新 |
| **时序** | 回写必须发生在所有执行单元完成 offset 读取之后（否则后续单元读到新值，流区间错位） |
| **缓存可见性** | 回写后须保证下次调用（可能落在其他执行单元）读到新值——需要缓存清理/失效动作 |
| **递进量** | 推进量是**算子级调用契约**（固定消耗可用 `L·c`、预留口径用 `L·R`、低频小量契约可为常量，实例见 [prng-model.md](prng-model.md) §7）——保证跨调用不重叠不跳变 |

- 协议目的：连续两次调用产出**不重叠**的随机序列（训练场景正确性依赖）
- 违反后果：两次调用产出重复随机数——功能测试难以发现，长序列统计检验可暴露

## 3.1 host 侧透传契约（R15）

- host 侧（tiling/adapter 层）对 seed/offset 做 **int64 原样透传**：不推进、不变换、不做 dtype 窄化——任何 host 侧"顺手"推进都会破坏流复现链
- **空 tensor 短路不下发 kernel**（图模式直调兜底路径同理）：随机流零消耗（见 [prng-model.md](prng-model.md) §8）

## 4 shape 的值依赖

输出 shape 来自描述张量（int32/int64 一维）：

- host 侧推导须**读值**：infershape（输出 shape 推导）与 tiling（输出规模 L、切分）两侧都需要——三处声明契约见 [../../common/value-depend-process.md](../../common/value-depend-process.md)
- 描述张量为空（rank 0 或 size 0）时输出为标量/空 tensor，不允许在推导侧短路报错
- shape 值可为负（-1）的接口（自动推断维）在 host 侧解析后校验合法性

## 5 检查清单

- [ ] seed=0 / seed≠0 两模式的验证口径已声明
- [ ] 接口形态归类（stateless / 带状态）与回写必要性判定
- [ ] 回写的唯一写者、时序、缓存可见性三约束落实到同步设计
- [ ] 递进量公式（L·c 或 L·R）与预留口径一致
- [ ] shape 张量值依赖三处声明齐备
