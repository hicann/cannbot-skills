<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式 AscendC 语言适配层

> **层级**：AscendC 语言适配层（adapters/ascendc/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。
>
> 本适配层基于 **Ascend C SIMT 编程模型**（浮点坐标驱动离散采样场景的标准选择）。regbase（SIMD 向量）路线的通用开发方法见 [../../../general-methodology.md](../../../general-methodology.md)。

## 1 定位

本目录承载 Interpolation 范式中**强耦合 AscendC SIMT** 的知识：坐标分解与采样落地、对齐模式分派、累加精度实现、能力真值与陷阱。开发语言为 ascendc（由调用方明确提供）时，与通用知识层组合消费：

```
通用算法/模型（generic/）× AscendC SIMT 实现细节（本目录）→ SIMT 设计与代码产出
```

SIMT 基础 API（线程模型、访存函数、数学函数、调试接口）的官方文档位于 `$ASC_DEVKIT_DIR/docs/zh/api/SIMT-API/`；`ascendc-simt-best-practices` skill 提供 API 分类导航与实践总结，本适配层只展开 Interpolation 范式特有的内容。

## 2 能力映射

本语言对通用层能力谓词的真值见 [capability-mapping.md](capability-mapping.md)。通用层算法中引用的谓词（fp64 可用性、FMA 阻断、线程上限等）一律以该文件填值为准。

## 3 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| Kernel 入口与 tiling | [kernel-entrance.md](kernel-entrance.md) | 三段式入口、32/64 位索引双路径、TilingData/TilingKey 组织、实例数预算、值依赖声明 |
| 坐标分解与采样 | [sampling-ops.md](sampling-ops.md) | UintDiv 快除、grid-stride 循环、对齐模式分派（MapCoord）、volatile 阻断 FMA、df32 补偿累加、grad 散射段 |
| 陷阱与定位 | [pitfalls.md](pitfalls.md) | FMA 系统性偏差、坐标映射次序、索引 OOB、基址溢出、伪 NaN、吸收语义等错题集 |

## 4 实测基准算子

本适配层结论蒸馏自以下 ascend950（arch35）实测算子，编码前可对照源码组织方式：

| 算子 | 算子族 | 关键特征 |
|------|--------|---------|
| three_interpolate | IndexedInterpolate | grid-stride + UintDiv、显式索引 clamp 防御、volatile 阻断 FMA、32/64 位双路径 |
| sync_resize_bilinear_v2 | RegularResize | MapCoord 双对齐模式、快除降维、df32 补偿累加 + 吸收截断、非有限分支 |
| crop_and_resize | RoiSample | 框坐标值依赖、method 分档 TilingKey（33 实例）、外插值、TF 契约对标 |
| upsample_bilinear2d_grad | RegularResize（grad） | 零初始化 + 散射两段、贡献丢弃、索引 dtype 分档线程数 |
| three_interpolate_backward | IndexedInterpolate（grad） | 散射累积、原子路径 |

**内嵌样例包**：three_interpolate（IndexedInterpolate 族最小样例）已整包镜像至 [example/three_interpolate_package/](example/three_interpolate_package/)（op_graph/op_host/op_kernel/tests/examples 完整结构；provenance 见 [example/README.md](example/README.md)）；sync_resize_bilinear_v2 / crop_and_resize / upsample 系按算子名在 CANN 组织开源算子仓（https://gitcode.com/cann，如 [ops-cv](https://gitcode.com/cann/ops-cv)、[ops-math](https://gitcode.com/cann/ops-math)）检索阅读。

## 5 公共知识依赖

| 主题 | 位置 |
|------|------|
| 快除（magic/shift）参数与布局 | [../../../common/uintdiv-rules.md](../../../common/uintdiv-rules.md) |
| SIMT 硬件模型 | [../../../common/simt-hardware-model.md](../../../common/simt-hardware-model.md) |
| 同步与一致性 | [../../../common/sync-and-consistency.md](../../../common/sync-and-consistency.md) |
| 值依赖处理 | [../../../common/value-depend-process.md](../../../common/value-depend-process.md) |

## 6 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载其他语言适配层
2. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题
3. 能力谓词以 [capability-mapping.md](capability-mapping.md) 为真值；标注"待验证"的项作为假设输出并附验证义务
4. 编码前对照 §4 基准算子核实 API 用法；参考实现的具体参数须按当前算子规格重新核定
