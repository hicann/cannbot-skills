<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details.
You should not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Convolution 范式算子档案

Convolution（卷积前向/推理）范式覆盖输出元素由输入在多维窗口内与卷积核滑窗互相关乘累加得到的算子：存在 im2col 意义下的 M/N/K 三元组，计算主体为 Cube 乘加。通过泛化 tiling 五步推导、fullLoad 四态与 K 分块，按空间维数、stride/dilation/pad 与片上资源选择切分与绑核方案。

设计入口见 [Convolution 范式](patterns.md)，具体语言的模板与实现约束由对应适配层提供。

## 典型算子

- Conv2D：标准 2D 前向卷积，范式参考实现（`adapters/ascendc/example/conv2d/`，覆盖 tiling 分核/单核切分与 kernel 主体；`<<<>>>` 直调与 aclnn 注册两种拉起接入由上层工作流负责）。
- Conv1D / Conv3D：1D 由升维（H=1）承接；3D 含 D 轴外层循环 + kd 切片归约。
- 空洞 / 带步长 / 非对称 pad 卷积：任意 stride（1~63）、dilation（1~255）、非对称 pad（< kernel）。
- 量化反量化卷积（conv + scale·out + bias）：见 [generic/scene.md](generic/scene.md) 场景扩展。

## 典型生成耗时与性能

| 算子名称 | 模型类型 | 平均性能 | 最差性能 | 算子生成时长 |
|---------|---------|---------|---------|-------------|
| Conv2D | —（范式自带参考工程） | 待实测回填 | 待实测回填 | 按设计难度 Level 4 口径预估 5~8 天（不含性能优化阶段） |

总体情况：本范式首版随参考工程（tiling + kernel）交付，非经算子生成工作流产出，性能倍数与生成耗时尚无实测数据，待首个经工作流生成的卷积算子完成后按实测口径回填。参考实现覆盖 1D/2D/3D、stride/dilation/非对称 pad、多核与 K 分块等规格组合（用例设计与验证方法见 [generic/verification.md](generic/verification.md)）；拉起执行（`<<<>>>` / aclnn）与精度比对工程由上层直调/验证工作流负责。

## 支持范围

- **开发语言**：AscendC（纯 Cube 与 Cube+Vector 协同；参考工程见 `adapters/ascendc/example/conv2d/`；拉起接入（直调/aclnn）由上层工作流负责）；CANNBotDSL 适配层暂缺，语言缺失时按 DSL 通用方法论兜底。
- **芯片与版本**：参考工程面向 Ascend 950（arch35 / DAV_3510）；L1/L0 容量、指令上限与版本差异以 [adapters/ascendc/buffer-and-types.md](adapters/ascendc/buffer-and-types.md) 为准。
- **规格边界**：fp16、NCHW/NCDHW/NCW、groups=1、pad<kernel、stride 1~63、dilation 1~255；空输出为合法规格（tiling 返回 1，跳过执行）。
- **能力边界**：纯 Vector 滑窗（无 Mmad）与 Winograd 变体不在范式范围；分组卷积、pad≥kernel、NHWC、C04 小通道等变体按 [generic/scene.md](generic/scene.md) 的差异方式扩展，主干文档不重复。

## 未来支持计划

- 补充 CANNBotDSL 适配层与参考实现。
- 经算子生成工作流产出首个卷积算子后，回填生成耗时与性能实测数据。
