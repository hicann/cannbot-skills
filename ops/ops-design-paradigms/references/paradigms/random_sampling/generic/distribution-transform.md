<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式分布变换

> **层级**：通用知识层（generic/），语言中立。
>
> 核心问题：均匀随机数到目标分布的变换、每输出消耗数与精度边界。

## 1 变换目录

| 目标分布 | 变换 | 每输出消耗 c | 公式 |
|---------|------|------------|------|
| **uniform [a,b)** | 线性映射 | 1 | `y = u·(b−a) + a` |
| **normal** | Box-Muller | 2（成对） | `z0 = √(−2·ln u1)·cos(2π·u2)`，`z1 = √(−2·ln u1)·sin(2π·u2)` |
| **exponential(λ)** | 逆变换 | 1 | `x = −ln(u)/λ`（或 `−ln(1−u)/λ`，开闭域随竞品） |
| **truncated_normal(±t)** | Box-Muller + 拒绝 | 变长 | 生成 z 后 `|z| < t` 接受，否则重采样 |
| **bernoulli(p)** | 阈值比较 | 1 | `y = (u < p) ? 1 : 0` |
| **multinomial** | 归一化 + 逆 CDF / 前缀和查找 | 1~n | 见 §3 |
| **permutation** | 拒绝式填充 | 变长 | 逐位取 `u·(剩余空间)`，冲突则重取 |

- 均匀→目标域的随机数映射区间（开闭）由竞品接口决定，直接决定逐位对齐结果
- Box-Muller 成对消耗：c=2 且**成对绑定**——流位置推导按对齐（见 [prng-model.md](prng-model.md) §3 的 g 布局系数）

## 2 变长消耗的判定

```
固定 c：变换可写成 u[0..c-1] → y 的纯函数           → 流位置可静态推导
变长 c：变换含条件重试（拒绝式）                      → 必须走预留律（R 配额）
```

设计检查：能改写为固定消耗的等价算法时优先固定（如 Box-Muller 替代拒绝式正态）；只有拒绝式显著更优（截断正态的短尾）才引入变长。

## 3 离散采样（multinomial）

- 输入概率向量 p（可能未归一化），输出选中类别
- 有放回：每个样本独立走逆 CDF（前缀和 + 二分/线性查找），消耗 1
- 无放回：选中后修改剩余分布，消耗与实现相关（变长）
- 与 SortSelect 的关系：前缀和是前缀扫描（归约族手段），但采样主体（随机数驱动 + 概率语义）属本范式

## 4 精度边界

| 边界 | 处理 | 竞品对齐义务 |
|------|------|------------|
| `u → 0`（对数下溢） | `u < eps` 抬升 | 是否保护、eps 取值须对齐（保护=确定一致，不保护=贴竞品） |
| 尾部截断 | 拒绝域边界 `|z| ≥ t` 的开闭 | 与竞品逐位差异点 |
| dtype 降精度 | fp32 计算后 cast 到 fp16/bf16 | cast 舍入模式声明 |

**u 构造路径一致性（R7，硬规则）**：uint32 → 目标域存在两条**实存口径**，**同流不同值**（同 `(key, counter)` 下 u 值不同）：

| 路径 | 公式 | 值域 | 实存口径 |
|------|------|------|---------|
| **IEEE 754 位重解释** | `(u & 0x7fffff) \| (127<<23)` 再 −1.0 | **[0,1)（含 0 不含 1）** | SIMD 族 U32Conversion、rrelu_with_noise 流基准 |
| **乘法 + 半步** | `u × 2^-32 + 2^-33` | **(0,1)（不含 0 不含 1）** | SIMT 模板族（PhiloxRandomSimt float 版）、curand 兼容 |

规则：
1. **u 构造路径必须与流基准/golden 完全同路径**——两路径混用即"同流不同值"，同源 golden 定义破裂，对拍必红且难定位
2. 值域开闭随路径登记（两条路径端点分布本就不同：位重解释含 0，乘法不含 0）
3. 新建流**推荐位重解释**（完美均匀：2^32 输入满射到 2^23 尾数值，每值恰命中 2^9 次）；乘法路径存在 2^-24 量级舍入偏差，作为既有流（SIMT 模板族/curand）兼容口径保留，不禁止但须声明

**仿射精度口径（R8，二选一 + 默认值）**：`y = u·(b−a)+a` 的舍入路径——

| 口径 | 路径 | 适用 |
|------|------|------|
| **torch CUDA 型（范式默认）** | u 先舍入到工作精度 d → 仿射在 d 内完成 → 一次舍入 | 对标 torch 语义 |
| CANN 组合型 | 高精度（fp32）仿射后 Cast 降精度到 d | 对标 CANN 既有组合 |

- 偏离默认口径时**声明义务**：spec 与 golden 同步标注；golden 与 kernel 口径不一致时对拍必红
- fp16/bf16 输出档的 Cast 模式（如 CAST_RINT）前置声明
| 浮点函数 ULP | 对数/三角的各实现差异 | 逐位验证用同源 golden；跨实现只做统计等价 |

## 5 检查清单

- [ ] 变换公式与竞品逐项核对（含开闭域、eps、拒绝边界）
- [ ] 每输出消耗数 c 已定（固定/变长），变长时 R 预留已论证
- [ ] 成对消耗的布局对齐（Box-Muller 的 g 系数）
- [ ] dtype 降精度路径的 cast 模式声明
