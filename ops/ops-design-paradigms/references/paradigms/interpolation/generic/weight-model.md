<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式权重与累加精度模型

> **层级**：通用知识层（generic/），语言中立。
>
> 核心问题：邻居权重如何计算、有限项加权和如何选累加口径，使 kernel 与 golden（常为 fp64 顺序累加）精度对齐。

## 1 权重计算

| 来源 | 计算 | 说明 |
|------|------|------|
| **lerp 分离**（RegularResize / RoiSample） | 轴分离：`w_h = {1-lerp_h, lerp_h}`，`w_w = {1-lerp_w, lerp_w}`，四象限权重 `w_h[i] × w_w[j]` | 双线性 4 项；`(1 - lerp)` 的舍入路径与 golden 同序（golden 在 fp64 中精确，fp32 直减在 lerp < 0.5 时产生 ~2^-25 舍入） |
| **显式权重**（IndexedInterpolate） | `w[k]` 由权重张量直接读入 | **不做归一化校验**（竞品契约：权重和不必为 1，直接使用）；值依赖读入 |

- 权重与样本乘积 `w × x` 各自独立舍入（除非走补偿路线）；乘积**不预先除以权重和**
- 最近邻模式无权重段（w=1 退化）
- bicubic 16 项的权重见 [coordinate-mapping.md](coordinate-mapping.md) §6

## 2 加权 gather 的访存形态

- 每输出元素 k ≤ 4 次（bicubic 16 次）离散 GM 读 + 1 次直写；邻居地址无局部性保证（RoiSample 的框间离散尤甚）
- 邻居取数**直访 GM**是默认选择（`CAP_SIMT_GM_DIRECT`）：每次只读单元素搬入片上缓冲无收益；小体量元数据（scale、快除参数表）经片上共享内存传递给全部线程复用
- 反向的写侧为 scatter-add：目标地址按正向映射计算，冲突模型与原子选择同 Sparse 范式散射累积族（语义允许重复累积、浮点原子非确定与竞品 GPU 一致）

## 3 累加精度判定树

```
golden 的累加语义是什么？
├─ 整型 / 幂等 → 逐位比对（无精度问题）
├─ fp32 顺序累加（竞品 kernel 同口径）→ fp32 累加器 + 项序一致即可
└─ fp64 顺序累加（竞品 CPU/参考实现常态）→ 平台原生 fp64？（CAP_FP64_AICORE）
    ├─ 是 → fp64 累加，末位一次性舍回输出 dtype
    └─ 否 → 值域含极值 / golden 近零点相对误差敏感？
        ├─ 否（常规值域、容差宽松）→ fp32 累加器，float32 输入直累、half/bf16 输入升 fp32 累加
        └─ 是 → 补偿累加模拟 fp64 语义（§4）
```

- **fp32 累加器的项数上限**：k=3/4 项常规值域下 fp32 顺序累加与 fp64 的偏差 ~2^-24 量级，通常满足容差；大 C 维度逐通道复用同一权重做多次加权时**通道间不共享累加器**（three_interpolate：C 维并行展开，每输出元素独立 3 项累加，fp32 累加器即可收敛）
- **项序以当前算子 golden 的实际求值序为准**：双线性可分离两阶段（先 W 轴两段、后 H 轴一段）与四项平铺（((p0+p1)+p2)+p3）是**不同舍入路径**；范式样例序（three_interpolate 的 ((p0+p1)+p2)、补偿路线的 p00→p01→p10→p11）是所属算子的事实，禁止误引为当前算子序。spec 公式、Kernel、HostTiling、Overview、分支文档的项序表述必须与 golden 求值序逐字一致（实测评审一般项：HostTiling/Overview 误引 k=3 样例序为双线性项序）
- **相对误差陷阱**：验证指标用 `|a-g| / (|g| + eps)` 时，golden 近零元素（异号相消、边界外插）对绝对舍入差极敏感——命中即升级补偿路线，不要放宽整体容差迁就

## 4 补偿累加（模拟 fp64 语义）

平台无 fp64（`CAP_FP64_AICORE` = false）而 golden 为 fp64 顺序累加时的对齐路线（sync_resize_bilinear_v2 五轮修复实测收敛）：

1. **项序复现**：按 golden 的固定项序（如 p00 → p01 → p10 → p11）顺序累加，禁止重排
2. **double-fp32（df32）表示**：每项 `w×q` 用 TwoProdFMA 分解为 `(hi, lo)`——`hi = w×q`（单次舍入）、`lo = fmaf(w, q, -hi)`（精确误差项，依赖 `CAP_FMA_ERROR_TERM`）；`(1-lerp)` 类残差用精确补恒等式（Knuth TwoSum 误差项）消除 fp32 直减舍入
3. **吸收截断**：每步累加将 `|lo| < 2^-53 × |hi|` 的低位字截断为 0——复现 fp64 舍入把小分量吸收丢弃的语义；不做此步则 golden 相消得 0 时 kernel 残存小项，相对误差爆炸（实测 mare 4.9e28 → 截断后 7.1e-8）
4. **非有限值分支**：任一邻居样本非有限（inf/nan）时**禁用补偿链**，退回简单乘加——补偿链在 q=±inf 时误差项 `inf - inf = nan` 产生伪 NaN，且异号 inf 相消的 nan 本身就是 golden 语义，不可屏蔽
5. **溢出防护**：极值值域（±FLT_MAX）下对样本做 2 的幂预缩（如 ×2^-4），末尾还原——2 的幂缩放精确，防中间积溢出 inf
6. **精度余量声明**：df32（48bit 有效位）与 fp64（53bit）在 2^-53~2^-48 残余带存在差异，命中概率极低（需 ~2^-48 级精确相消）但须在验证方案中声明

## 5 inf / NaN 传播语义

- **正向**：非有限样本按 IEEE 754 传播（NaN 输入 → NaN 输出；`0 × inf = nan`）；权重精确零时按竞品语义（部分竞品以 `w=0` 短路 `0×inf=0`，须核对 golden 实现是乘加还是短路）
- **契约不校验**：weight 和不为 1、idx 越界（契约保证类）不在 kernel 校验；NaN 参与加权求和直接传播（three_interpolate 契约）
- **验证义务**：非有限用例必须覆盖"权重恰为零 × inf 样本"分支，比对本范式 kernel 与 golden 的逐位行为

## 6 设计输出清单

- [ ] golden 累加语义判定（fp32 / fp64 顺序）与平台 fp64 真值引用（判定结论全文档唯一：Validation/Kernel/分支/spec 一致）
- [ ] 累加口径选择（判定树路径）与依据
- [ ] 项序声明与 golden 实际求值序逐字一致（全文档唯一，不误引范式样例序）
- [ ] 补偿路线启用时的项序声明、吸收阈值、非有限分支
- [ ] half/bf16 输入的升位策略与输出舍回
- [ ] 权重零值 × 非有限样本分支的竞品核对结论
- [ ] 极值值域的溢出核算（是否需 2 的幂预缩）
