<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# SlidingWindow 范式内嵌样例

本目录承载 SlidingWindow 范式**正反向两个 SIMT 单路线模板包**（★ 基准模板），范式知识自含闭环——阅读与对照不需要任何外部仓依赖。

## 样例清单

| 模板包 | 算子类型 / 分支 | 范式对应 |
|--------|---------------|---------|
| [max_pool_with_argmax_simt_package/](max_pool_with_argmax_simt_package/) | 正向 WindowAggregate（max 聚合） | 输出驱动窗口枚举、−Inf 哨兵 + NaN 优先、argmax mask 双输出、NCHW/NHWC 双布局 |
| [max_pool_grad_with_argmax_simt_package/](max_pool_grad_with_argmax_simt_package/) | 反向 GradRoute·B（argmax 路由） | 输入驱动反窗口枚举（PStart/PEnd）、mask 命中检查、零原子直写、UintDiv 快除 |

## 来源与组织约定

- **状态**：均为 DRAFT 参考模板（未编译、未上板）——算法语义逐段蒸馏自 ops-nn 仓 `pooling/max_pool_with_argmax_v3` / `max_pool_grad_with_argmax_v3` 的 SIMT kernel（ascend950 实测），每个包的 README 登记对照关系与首例验证义务
- **自含组织**：SIMT 单路线形态——TilingData 自包含定义、TilingKey 用 `ASCENDC_TPL_ARGS_DECL` 模板编程（范式规则 §5.5.4）、无 common 库依赖、无多路线分派。**新算子设计不需要预先建 common**；成熟仓的多算子共享 common / 多路线 TilingKey 组织是工程演进形态，阅读既有算子时识别即可（[../simt-path.md](../simt-path.md) §4），设计产出不复制该形态
- **包结构**：op_graph（proto）+ op_host（infershape / tiling）+ op_kernel（apt + arch35 simt kernel + tiling data/key）+ tests（golden）+ README（I/O 契约 / 展示点 / 验证义务）；op 定义注册样板与编译 CMake 按 ascendc 通用技能补齐，不属范式知识
- 语义级差异声明：模板算子名去掉 `_v3` 后缀（`MaxPoolWithArgmax` / `MaxPoolGradWithArgmax`），与 ops-nn 成熟包的 ceil_mode / 自适应窗口等变体无关；对标实测时以 README 验证义务为准

其余基准算子（avg_pool / avg_pool_v2_grad 的 A 分支正反向配对、max_pool3d 系、dilation2_d / col2_im_v2 / extract_image_patches）按算子名在 [ops-nn](https://gitcode.com/cann/ops-nn) 仓 `pooling/` 检索阅读；基准登记见适配层 [index.md §4](../index.md)。

> 模板不随源仓演进追新；算法语义以蒸馏时的实测 kernel 为锚点。
