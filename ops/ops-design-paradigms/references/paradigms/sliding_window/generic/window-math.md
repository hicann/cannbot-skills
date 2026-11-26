<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# SlidingWindow 范式窗口数学

> **层级**：通用知识层（generic/），语言中立。
>
> 核心问题：窗口边界与反窗口区间的推导规则、pad 裁剪语义、dilation 起点推进、空窗与聚合初始化。

## 1 窗口参数契约

- kernel/stride/pad/dilation 均为 **host 侧已知标量**（属性或 shape 推导），kernel 侧只接收数值，不做属性字符串解析（padding_mode 归一化在 host tiling 完成）
- 有效窗口跨度：`span = (k − 1) · d + 1`（dilation=1 时退化为 k）；end 的推导必须用 span 而非 k
- 输出尺寸公式（ceil/floor 模式差异在 host 侧归一）：`out = floor((in + padL + padR − span) / s) + 1`
- 窗口参数 ≤ `CAP_VF_PARAM_MAX` 时标量入参；快除参数表经片上共享内存广播

## 2 正向窗口边界（输出驱动）

对输出位置 `(ph, pw)`（单轴，轴间独立同构）：

```
start = p · s − pad                       # 可为负（pad 区）
end   = min(start + span, extent + pad)    # 右侧含 pad 上限
# 双向裁剪（两序不可交换——先定 end 再裁 start）：
start' = max(start, 0)
end'   = min(end, extent)
空窗:  start' ≥ end' → 空窗短路
```

- **dilation 下负起点的推进**：`while (start < 0) start += d`——**不是 clamp 到 0**。dilation > 1 时窗口格点为 `start + i·d`，clamp 到 0 会把起点移到非格点，引入窗口外的元素；逐格推进保证枚举仍在格点上
- dilation = 1 时 `while 推进` 与 `max(·,0)` 等价（avg_pool 直用 max 的简化形态）
- **空窗可达性反例**（`pads ≤ k/2` 约束不排除空窗）：`inH=2, kH=2, padH=1, dilH=3, sH=1`（padH=1 ≤ kH//2=1，合法域）→ hspan=4、outH=1；窗口起点 −1 推进至 2、end′=2，枚举空。设计须按本节公式核算空窗可达性并显式声明空窗语义（max: −Inf 直写；mask 初值口径与竞品核对），不得由输入域约束推定「空窗不可达」而跳过论证（max_pool_with_argmax_v3 生成记录评审 D3 实测缺口）
- **空窗核算义务（强制）**：可达性核算须**附完整过程**——逐轴列出 out 尺寸、窗口 start/end、裁剪后区间与枚举空判，不得只给结论；自检须把上述反例参数代入核算流程复核，且必须得出「枚举空」结论（核算流程对反例得出「不可达」即流程本身有误）。**「空窗不发生 / 每窗口 ≥1 个有效采样」类保证成文前须通过该核算**：`span ≤ L+2p`、`pads ≤ k/2` 等输入域约束不构成不可达证明（反例自身满足全部约束），由约束直接推定不可达＝未核算（max_pool_with_argmax_v3 第二轮评审 R2-2 实测：以 C7「span ≤ L+2p」推定空窗不发生、虚假保证 ≥8 处成文）。空窗语义须**三处同值核对**——接口值域声明 / golden oracle 公式 / kernel 初值：同一反例代入三处，空窗下的输出与 mask/argmax 初值结论须一致且落在接口声明的值域内（第二轮评审实测三方分叉：oracle −3、kernel +6、接口 [0, H·W)，越界互异）
- pad 区语义：窗口可延伸到 pad 区，**裁剪发生在聚合统计之前**——`countIncludePad` 决定除子用裁剪前面积还是有效面积（[scene.md](scene.md) §3.1 除子行）

## 3 聚合初始化与 NaN 传播

| ⊕ | 初始化 | NaN 语义 |
|---|--------|---------|
| max | **−Inf 哨兵**（依赖 `CAP_FP_NEG_INF`） | **NaN 优先传播**：`if (val > acc || isnan(val)) 更新 winner`——窗口含 NaN 时 winner 指向 NaN 位置，与竞品一致；不能只写 `val > acc`（NaN 比较恒 false 会漏传播） |
| avg / sum | 0 | NaN 照常累加传播（输出 NaN） |
| 平局（max 相等） | — | **首个 winner 保留**（严格大于才替换）；竞品 tie-break 规则须核对（首遇 / 末遇 / 任意），argmax mask 的反向路由依赖该契约 |

- winner 索引口径：**线性索引**（`h · W + w`，不含通道/样本偏移——批次内坐标），反向路由按同口径重算比对
- max 的初始化哨兵与"全空窗"输出：−Inf 直写（竞品语义，不转 0）

## 4 反窗口区间（输入驱动，反向）

对输入位置 `(h, w)`（单轴），输出维 extent 为 `poolSize`：

```
pStart = (pos + pad < span) ? 0 : (pos + pad − span) / s + 1      # 首个覆盖窗口
pEnd   = min((pos + pad) / s + 1, poolSize)                       # 末个覆盖窗口 +1
```

- **pStart 的除法在整数域**（反窗口区间端点是格点计算，无浮点）；`/ s` 用快除（`CAP_INT_DIV_COST`）
- 推导来源：覆盖条件 `start ≤ pos < end` 的不等式解——与正向窗口公式**互为逆运算**，两处推导必须同源核对（窗口参数一致性检查项）
- `pos + pad < span` 分支：窗口起点在 pad 区时不存在覆盖窗口，pStart = 0
- pEnd 由 `poolSize` 封顶：输出维越界不计入贡献

## 5 自适应池化的窗口换算

adaptive pool 的输出位置 → 窗口边界按输出/输入比例换算（每输出位置窗口不同）：

```
start = floor(pos · extent_in / extent_out)
end   = ceil((pos + 1) · extent_in / extent_out)     # 恒有 end > start（非空窗）
```

- 换算只进**边界公式**（整数窗口），不作邻居权重——与 Interpolation 的本质区别（那边 lerp 权重由浮点小数决定）
- host 侧可预算每输出位置的边界表（规则可推导时不需表）；kernel 侧逐位置重算用整数除法

## 6 设计输出清单

- [ ] 窗口参数清单（k/s/pad/dilation × 轴）与 host 归一化项（padding_mode / ceil_mode）
- [ ] 空窗语义声明（max: −Inf 直写；avg: 0 / 除子口径）与可达性核算（`pads ≤ k/2` 不排除空窗，反例见 §2）——核算过程完整随附、反例代入自检、「空窗不发生」类保证凭核算成文、接口 / oracle / kernel 三处空窗语义同值核对
- [ ] 负起点推进方式（dilation > 0 时 while 推进）与依据
- [ ] NaN 传播与平局 tie-break 的竞品核对结论
- [ ] 反窗口公式与正向窗口公式的同源核对
- [ ] adaptive 换算公式（如适用）与整数域声明
