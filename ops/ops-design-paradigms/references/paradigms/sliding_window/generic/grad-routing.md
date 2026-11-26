<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# SlidingWindow 范式反向梯度路由

> **层级**：通用知识层（generic/），语言中立。
>
> 核心问题：反向梯度如何回传——三分支决策树、零原子论证与 argmask 契约。

## 1 反向输入形态

反向算子的输入：`grad_output`（正向输出同形梯度）+ 路由信息（按分支可选）：`argmax mask`（正向伴随输出）、`orig_input/orig_output`（重算路由）或无。

## 2 路由决策树

```
正向聚合携带可复用的路由信息？
├─ 是：argmax mask（正向伴随输出）→【B argmax 路由】
│      路由形态二选一（§4）：反窗口 + mask 检查（输入驱动，零原子）
│                            或 scatter 到 winner（输出驱动，零冲突——每窗口唯一 winner）
├─ 无 mask 但可重算（orig_input/orig_output 在手）→ 重算型：
│      重算代价可接受？→【B 变体：重算 argmax 后同 B】
│      否 →【C 原子散射】
└─ 否（聚合均匀贡献 / 无路由概念）：
    贡献规则可由窗口数学反推？→【A 反窗口枚举】（输入驱动，零原子）
    否 →【C 原子散射】
```

- **A/B 优先于 C**：A/B 均零原子零栅栏（单段 kernel 直写），C 需零初始化 + 散射两段 launch + 原子（同步骨架同 Interpolation grad / Sparse 散射族，此处不重复定义）
- B 的两形态等价性：scatter-to-winner 依赖"每窗口唯一 winner"（平局 tie-break 保证，[window-math.md](window-math.md) §3）；反窗口 + mask 检查则天然容忍 mask 重复——**两种形态不得混用于同一算子**（梯度守恒要求每个 grad_output 元素恰好路由一次）

## 3 分支 A：反窗口枚举（均匀贡献）

- 适用：贡献规则可由窗口边界数学反推的均匀贡献池化聚合——avg（含 divisorOverride / countIncludePad 变体）、lp
- 每个输入位置独立：反窗口区间（window-math §4）→ 枚举覆盖窗口 → `gradient += grad_output[窗口位] / 除子` → 直写
- **除子口径必须与正向一致**：`divisorOverride / countIncludePad(裁剪前 poolSize) / 有效面积`三分档按正向同款取值（正反向除子不一致是精度事故高发点）
- 梯度守恒验证：`sum(grad_input) == sum(grad_output)`（均匀贡献下成立）

## 4 分支 B：argmax 路由

- 输入驱动形态（max_pool_grad_with_argmax_v3 实测）：对每个输入位置反窗口枚举，`if (topMask[输出位] == 本输入线性坐标) gradient += grad_output[输出位]`——mask 命中即本位置为 winner 的窗口，恰好覆盖全部路由
- 输出驱动形态（scatter-to-winner）：对每个 grad_output 元素，`grad_input[mask[位]] += grad`——winner 唯一性由正向 tie-break 保证，无原子
- **mask 索引口径契约**：线性索引 `h·W + w`（批次内），不含 n/c 偏移；正反向实现必须同口径（int32/int64 分档一致）
- 稀疏命中密度：输入位置被路由命中的期望次数 = 窗口重叠度/窗口面积比；反窗口枚举的无效扫描占比高（大多数窗口位 mask 不命中），线程数可上调隐藏分支开销（[splitting.md](splitting.md) §4）

## 5 分支 C：原子散射

- 适用：无路由信息且窗口贡献不可由边界数学反推的池化反向（重算代价高者）、输出驱动形态且 winner 可能重复的退化场景
- 骨架：输出零初始化段 + 散射段两段顺序 launch（同 stream 免栅栏，无法保证时 `CAP_SYNC_ALL` + `CAP_SYNC_SCHEDULE_MODE` 显式栅栏——**多核散射缺 SyncAll 是实测事故**，max_pool_v3_grad 生成记录 design_peer HIGH）
- 原子 dtype 域与确定性路径：复用 Interpolation grad / Sparse 散射族结论（浮点原子非确定、与竞品 GPU 对拍用容差；要求确定性分桶预归约）
- 重算型 max_pool_grad（orig_input/orig_output 重推 winner）归本分支或 B 变体——按重算代价决策，产出须声明路由信息清单

## 6 设计输出清单

- [ ] 反向分支判定（决策树路径）与路由信息清单
- [ ] A/B 分支的零原子论证（坐标系 / 唯一 winner 契约）
- [ ] 除子口径与正向一致性声明（分支 A）
- [ ] mask 索引口径与 tie-break 契约（分支 B）
- [ ] 分支 C 的两段骨架与同步声明
- [ ] 梯度守恒验证义务登记
