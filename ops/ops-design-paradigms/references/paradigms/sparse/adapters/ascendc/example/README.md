<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式内嵌样例

本目录整包镜像 Sparse 范式的最小实测算子（★ 基准算子），供设计/编码前对照源码组织方式。

## 样例清单

| 样例包 | 源仓 / commit | 范式对应 |
|--------|--------------|---------|
| [sparse_bincount_package/](sparse_bincount_package/) | `ops-math` 仓 `math/sparse_bincount`，commit `852dfe3ea` | Count 族：atomic 累积、值域裁剪、binary_output 幂等直写；对照 [atomic-ops.md](../atomic-ops.md) §3 幂等直写、[kernel-entrance.md](../kernel-entrance.md) 三段式入口 |

包结构（op_graph / op_host / op_kernel / tests / examples / README）与源仓一致；除**本仓 pre-commit 与 CodeCheck 规则适配**（clang-format 对齐、行尾空白与文件末尾换行、注释符后补空格、def 文件按文件名-类名一致性改名、测试 fixture 按文件名对齐）外未做任何语义改动，语义与源仓逐行对应；编译依赖（公共 CMake、SIMT 基础库）在源仓，本镜像用于阅读对照，不独立构建。其余 4 个基准算子（sparse_segment_sum_grad / sparse_apply_adadelta / sparse_apply_ftrl / sparse_reshape）按算子名在 CANN 组织开源算子仓（[ops-math](https://gitcode.com/cann/ops-math)、[ops-nn](https://gitcode.com/cann/ops-nn) 等，https://gitcode.com/cann）检索；基准登记见适配层 [index.md §4](../index.md)。

> 源仓后续演进时以源仓为准；本镜像不追新，仅锚定蒸馏版本。
