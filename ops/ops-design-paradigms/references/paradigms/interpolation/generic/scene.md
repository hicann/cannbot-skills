<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式场景模型

> **层级**：通用知识层（generic/），语言中立。
>
> 来源：three_interpolate、sync_resize_bilinear_v2、crop_and_resize 三个 AscendC SIMT 算子的实测蒸馏（ascend950 / arch35），并参照 upsample_bilinear2d/bicubic2d_grad、roi_align_rotated、deformable_roi_pool 的生成记录。

## 1 准入条件

本范式覆盖的算子需同时满足：

1. **输出值由输入在浮点坐标处的加权采样决定**——每个输出位置先确定输入域的浮点源坐标（坐标映射计算或显式索引+权重给出），再取离散邻居做加权求和；即插值 / 重采样 / 区域采样类语义
2. **计算核心是索引驱动的离散访存**——邻居地址在运行前不可知，按运行时计算的坐标（或读入的索引值）离散 gather；无法用连续块搬运或规则向量计算覆盖
3. **正向输出逐位独立**——输出元素间无数据依赖、无写冲突，逐位直写；梯度反向是 scatter-add 散射累积，冲突与同步知识复用 ScatterUpdate / Sparse 范式，不在本范式重复定义

## 2 术语与坐标系

| 术语 | 含义 |
|------|------|
| **源坐标 src** | 输出位置经坐标映射得到的输入域浮点坐标（如 `h_src`） |
| **缩放尺度 scale** | 输出坐标系到输入坐标系的比率，可由 shape 推导（resize）或由 ROI 框计算（crop） |
| **对齐模式** | scale 与 src 的计算约定：align_corners / half_pixel（见 [coordinate-mapping.md](coordinate-mapping.md) §2） |
| **邻居 (i0, i1)** | 源坐标 floor 后的整数邻居索引对；高阶核（bicubic）为 4 邻居 |
| **lerp 小数** | `src - floor(src)`，决定邻居间的线性权重 |
| **权重 w** | 邻居样本的加权系数，由 lerp 分离（bilinear）或显式输入（IndexedInterpolate） |
| **外插值 extrapolation_value** | 源坐标越出输入域时的替代输出常量（RoiSample 常用） |
| **totalOut** | 输出元素总数（正向切分主坐标） |
| **范式规范名** | 本范式名称一律写作 **Interpolation**（文档、产物、交叉引用统一；禁止 Interactive / Interpolated 等误写——实测生成物曾误写 Interactive） |

## 3 算子族分类

按坐标映射来源与邻居组织分为三族：

| 算子族 | 坐标映射 | 邻居与权重 | 实例 |
|------|---------|-----------|------|
| **RegularResize 规则网格重采样** | 输出坐标 × scale（对齐模式决定公式） | floor/lerp 分离的 2 邻居（线性）或 4 邻居（bicubic） | sync_resize_bilinear_v2、upsample_bilinear2d / upsample_bicubic2d（正反向） |
| **IndexedInterpolate 索引加权插值** | 索引张量显式给出 | 权重张量显式给出，k 项加权求和 | three_interpolate（k=3，PointNet++） |
| **RoiSample 区域采样** | ROI 框坐标变换（归一化框 → 绝对坐标）+ 网格 scale | 双线性 4 邻居；越界用外插值或贡献丢弃 | crop_and_resize、roi_align_rotated、deformable_roi_pool、img_warp_resize |

- RegularResize 与 RoiSample 同为"计算坐标再双线性采样"，差异在坐标映射的复杂度：前者是逐轴独立的线性映射，后者含 ROI 框仿射 / 旋转 / 偏移量叠加
- IndexedInterpolate 的索引与权重均为输入张量（值依赖访存），坐标映射退化为"读值即得"；三邻居地址天然离散，是本范式的最小样例形态
- 各族通用的反向：`grad_input[邻居] += grad_output[out] × w`，散射冲突处理与零初始化同步同 Sparse 范式散射累积族

## 4 数据流骨架

三族共享"输出坐标 → 坐标映射 → 邻居与权重 → 加权 gather → 直写"五段式：

```
[段 1: 输出坐标]     由输出线性索引分解多维坐标 (n, y, x, c)（快除，见 splitting §2）
[段 2: 坐标映射]     RegularResize: src = f(scale, out)（对齐模式分派）
                     RoiSample:     框参数 y1/x1/y2/x2 仿射 + 采样点偏移（含旋转/变形时叠加）
                     IndexedInterpolate: 读 idx[b,n,k]（值驱动）
[段 3: 邻居与权重]   i0 = floor(src)，lerp = src - i0；权重按核阶分离（或读显式权重）
[段 4: 加权 gather]  按邻居地址离散取数 → 有限项加权和（k ≤ 4 项，见 weight-model）
[段 5: 输出]         正向逐位直写；反向 scatter-add（原子）+ 输出零初始化两段 launch
```

- 段 2 的浮点计算**顺序与舍入路径须与 golden 严格同序**（索引数学禁融合重排，见 [coordinate-mapping.md](coordinate-mapping.md) §5）
- 段 4 是精度风险集中区：累加口径（fp32 / 补偿）按 [weight-model.md](weight-model.md) §3 判定树选择
- RoiSample 的段 2 可能读入框坐标张量（值依赖），与段 4 的 gather 同为离散访存

## 5 能力边界

1. **滑动窗口池化**（max_pool / avg_pool 类）归 SlidingWindow 范式——窗口坐标为规则整数偏移，无浮点源坐标；但其 grad（如 dilation2d_backprop_input、col2im 类散射）与本范式 grad 路径技术互通
2. **纯整数规则坐标变换**（transpose / reshape / strided_slice 类）归 LayoutTransform——无浮点坐标映射与加权采样
3. **写散射为语义主体**（scatter_update / index_put 类）归 ScatterUpdate 范式；本范式的 grad 反向虽为散射，语义主体仍是"正向插值"，只复用其原子与同步知识
4. **稀疏数据面**的插值（按 COO 索引采样）归 Sparse 范式——本范式以 dense 输入张量为数据面
5. **最近邻模式**（method=nearest / bilinear_v2 中的整数化）属本范式 RegularResize 的退化路径：坐标映射后取整直接 gather，无加权求和段
6. **affine_grid / 网格生成**类只产坐标不采样，不进本范式；grid_sample 归 RoiSample 族（坐标由 grid 张量显式给出）

## 6 依赖谓词

| 谓词 | 用途 |
|------|------|
| `CAP_SIMT_GM_DIRECT` | 邻居 gather 直访 GM、不经片上缓冲转译的前提 |
| `CAP_FP64_AICORE` | 累加精度路线选择（无 fp64 时需补偿累加模拟，weight-model §3） |
| `CAP_FMA_CONTROL` | 索引数学禁 FMA 融合的可行性（coordinate-mapping §5） |
| `CAP_FMA_ERROR_TERM` | 误差项提取（补偿累加 TwoProdFMA 的前提） |
| `CAP_SIMT_ATOMIC_ADD_DTYPE_SET` | grad 散射累积支持的 dtype 集合 |
| `CAP_SYNC_ALL` | grad 零初始化与散射两阶段间的全核同步 |
| `CAP_SIMT_THREAD_MAX` / `CAP_STACK_PER_THREAD` | 线程数与栈预算 |

谓词定义见 [capability-contract.md](capability-contract.md)；各语言真值见对应适配层 capability-mapping。
