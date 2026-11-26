<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式算子档案

Interpolation（插值与重采样）范式覆盖浮点坐标驱动的插值 / 重采样 / 区域采样算子：输出值由输入在浮点源坐标处的加权采样决定，计算核心是坐标映射 + 索引驱动的离散访存。

## 典型算子

- three_interpolate：PointNet++ 3 邻居索引加权插值（IndexedInterpolate 族），范式最小样例
- sync_resize_bilinear_v2：双线性重采样（RegularResize 族），对齐模式分派与精度深修基准
- crop_and_resize：ROI 框采样 + 双线性（RoiSample 族），值依赖框坐标与外插值
- upsample_bilinear2d / upsample_bicubic2d（正反向）：上采样与高阶核
- roi_align_rotated / deformable_roi_pool：旋转 / 变形区域采样
- three_interpolate_backward / resize_grad：梯度散射（复用 ScatterUpdate / Sparse 冲突与同步知识）

## 典型生成耗时与性能

| 算子名称 | 模型类型 | 平均性能 | 最差性能 | 算子生成时长 |
|---------|---------|---------|---------|-------------|
| three_interpolate | GLM-5.3 | 0.6 | 0.3 | 约 40 小时（不含性能优化阶段） |

总体情况：IndexedInterpolate 族（three_interpolate）结构最简（9 轮 TTK 收敛 323 用例全通过，生成约 40 小时）；RegularResize 族（sync_resize_bilinear_v2 约 15 小时，但 golden fp64 语义对齐的补偿累加链路经历 5 轮精度修复）；RoiSample 族（crop_and_resize 约 49 小时，值依赖框坐标与 method 分档实例收敛论证最重）。SIMT 离散邻居 gather 吞吐受 totalOut 与框 / 索引分布影响；精度链路（FMA 同序、df32 补偿累加）是本范式返工主因。平均/最差性能字段按用户提供的数值填入（0.6 / 0.3）；本表字段未注明统计口径与单位，以数据提供方口径为准，待实测口径明确后复核。

## 支持范围

- **开发语言**：AscendC（`__simt_vf__` 浮点坐标驱动离散 gather + 标量直写，grad 路径标量原子）；CANNBotDSL（SIMT 数据面 + 原子 + AI CPU metadata 调度）
- **芯片**：Ascend 950PR / 950DT（arch35）及以上；目标芯片不支持 SIMT 时不进入本范式，改走通用方法论论证非 SIMT 路线
- **版本**：DSL 0.8.0；SIMT（含混合）需 CANN ≥ 9.2.0（cannbot-dsl 仓配套 9.2.1）
- **覆盖算子族**：RegularResize / IndexedInterpolate / RoiSample（含各自 grad 散射路径）
- **能力边界**：滑动窗口池化归 SlidingWindow；纯整数规则坐标变换归 LayoutTransform；写散射为语义主体归 ScatterUpdate；稀疏数据面插值归 Sparse；稀疏注意力块跳过属 FA 组合范式

## 未来支持计划

暂无
