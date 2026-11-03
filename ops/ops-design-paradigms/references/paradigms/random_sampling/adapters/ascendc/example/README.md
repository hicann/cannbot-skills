<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式内嵌样例

本目录整包镜像 RandomSampling 范式的最小样例（★ 路线 a 纯生成形态），供设计/编码前对照源码组织方式。

## 样例清单

| 样例包 | 源仓 / commit | 范式对应 |
|--------|--------------|---------|
| [random_uniform_v2_package/](random_uniform_v2_package/) | `ops-math` 仓 `random/random_uniform_v2`，commit `852dfe3ea` | TF 对标 SIMD 路线（A/C 族）：`PhiloxRandom<10>` 批量产出、U32Conversion 尾数拼接、offset 回写；对照 [simd-path.md](../simd-path.md)、[common-template.md](../common-template.md) OpTilingConfig 四回调 |

包结构（op_graph / op_host / op_kernel / tests / examples / README）与源仓一致；除**本仓 pre-commit 与 CodeCheck 规则适配**外未做任何语义改动：clang-format 对齐、行尾空白与文件末尾换行、注释符后补空格、`.UnknownShapeFormat` 参数行缩进（clang-format off/on 保护）、两个文件按文件名-类名一致性规则重命名（`op_host/random_uniform_v2_def.cpp`→`random_uniform_v2.cpp`、`op_kernel/arch35/random_uniform_v2.h`→`random_uniform_v2_op.h`，OP_ADD/REG_OP 注册名不变）、tiling 类名与测试 fixture 按文件名对齐。算法与注册语义与源仓逐行对应。该算子继承公共底座 `random_common/`（源仓 `ops-math/random/random_common/`，本镜像未含——接入新算子时底座在源仓读取）。

> 源仓后续演进时以源仓为准；本镜像不追新，仅锚定蒸馏版本。
