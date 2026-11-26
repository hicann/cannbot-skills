<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# SlidingWindow 范式算子档案

SlidingWindow（滑动窗口聚合）范式覆盖池化类算子（整数规则窗口聚合）：正向输出驱动枚举窗口聚合（max/avg/lp/sum），反向按梯度路由三分支回传（反窗口枚举 / argmax 路由 / 原子散射），两方向均为零原子逐位直写为首选路线。

## 典型算子

- max_pool_with_argmax_v3 / max_pool_grad_with_argmax_v3：max 聚合正反向配对（B 分支 argmax 路由），范式内嵌模板的算法语义来源
- avg_pool / avg_pool_v2_grad：avg 聚合正反向配对（A 分支反窗口枚举，零原子）
- max_pool3d_with_argmax_v2 / max_pool3d_grad：3D 池化正反向
- adaptive_max/avg_pool 系：自适应池化（整数边界换算）

## 典型生成耗时与性能

| 算子名称 | 模型类型 | 平均性能 | 最差性能 | 算子生成时长 |
|---------|---------|---------|---------|-------------|
| avg_pool_update | GLM-5.3 | 0.5 | 0.3 | 约 16.6 小时（不含性能优化阶段） |

总体情况：本范式正反向模板（max_pool_with_argmax / max_pool_grad_with_argmax，SIMT 单路线自含组织）算法语义蒸馏自 ops-nn 成熟包的实测 SIMT kernel，模板本身未独立编译上板（DRAFT，验证义务见各模板 README）；生成管线记录中 avg_pool_update 约 16.6 小时、max_pool_v3_grad 生成耗时未登记。正反向 SIMT 均为输出/输入驱动逐位直写，性能 ∝ 输出（正向）或输入（反向）元素数 × 窗口格点数；反向 argmax 路由的无效扫描占比与窗口重叠度强相关。平均/最差性能字段按用户提供的数值填入（0.5 / 0.3）；本表字段未注明统计口径与单位，以数据提供方口径为准，待实测口径明确后复核。

## 支持范围

- **开发语言**：AscendC（`__simt_vf__` 正向窗口聚合 + 反向输入驱动路由，分支 C 标量原子）；dsl 适配层见下方未来支持计划
- **芯片**：Ascend 950PR / 950DT（arch35）及以上；目标芯片不支持 SIMT 时不进入本范式，改走通用方法论论证非 SIMT 路线
- **版本**：SIMT 需 CANN ≥ 9.2.0
- **覆盖算子族**：正向 WindowAggregate（池化聚合 max/avg/lp/sum/adaptive）；反向 GradRoute 三分支（A 反窗口枚举 / B argmax 路由 / C 原子散射）
- **布局**：覆盖 NCHW 与 NHWC 两种 4D 布局（知识层差异见 generic/scene.md 布局维度一节；SIMT 示例模板以 NCHW 为准，NHWC 路线由多路线包的 TilingKey 分派承接）

## 未来支持计划

- CANNBotDSL 适配层（`vf(mode="simt")` 正反向路线 + 同算子草稿）：按双语言范式惯例补充，接入前 DSL 侧能力真值未登记
