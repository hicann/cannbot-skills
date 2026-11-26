<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式内嵌样例

本目录整包镜像 Interpolation 范式的最小实测算子（★ 基准算子），供设计/编码前对照源码组织方式。

## 样例清单

| 样例包 | 源仓 / commit | 范式对应 |
|--------|--------------|---------|
| [three_interpolate_package/](three_interpolate_package/) | `ops-cv` 仓 `image/three_interpolate`，commit `4ccbb4c4` | IndexedInterpolate 族：grid-stride + UintDiv 坐标分解、显式索引 clamp 防御、volatile 阻断 FMA（issue #35 根因修复）、32/64 位索引双路径；对照 [sampling-ops.md](../sampling-ops.md) §1/§5.1、[kernel-entrance.md](../kernel-entrance.md) 双路径分派 |

包结构（op_graph / op_host / op_kernel / tests / examples / README）与源仓一致；除**本仓 pre-commit 与 CodeCheck 规则适配**（行尾空白与文件末尾换行、目录组织对齐）外未做任何语义改动，语义与源仓逐行对应；编译依赖（公共 CMake、SIMT 基础库）在源仓，本镜像用于阅读对照，不独立构建。其余基准算子（sync_resize_bilinear_v2 / crop_and_resize / upsample_bilinear2d_grad / three_interpolate_backward）按算子名在 CANN 组织开源算子仓（[ops-cv](https://gitcode.com/cann/ops-cv)、[ops-math](https://gitcode.com/cann/ops-math) 等，https://gitcode.com/cann）检索；基准登记见适配层 [index.md §4](../index.md)。

> 源仓后续演进时以源仓为准；本镜像不追新，仅锚定蒸馏版本。
