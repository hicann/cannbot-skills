<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 范式内嵌样例

本目录整包镜像 ControlFlow 范式的设备核路线最小算子（★ 基准算子），供设计/编码前对照源码组织方式。

## 样例清单

| 样例包 | 源仓 / commit | 范式对应 |
|--------|--------------|---------|
| [npu_get_float_status_package/](npu_get_float_status_package/) | `ops-nn` 仓 `control/npu_get_float_status`，commit `bb42322c3` | 设备核路线：SIMT 空实现恒写零（降级态 I5）、退化 tiling 四步契约、`config/ascend950/*_binary.json` 预编译二进制单实例形态；对照 [status-kernel.md](../status-kernel.md) §2/§6 |

包结构（op_graph / op_host / op_kernel / tests / examples / README）与源仓一致；除**本仓 pre-commit 与 CodeCheck 规则适配**（clang-format 对齐、行尾空白与文件末尾换行、注释符后补空格、def 文件按文件名-类名一致性改名、测试 fixture 按文件名对齐）外未做任何语义改动，语义与源仓逐行对应；本镜像用于阅读对照，不独立构建。同族 npu_alloc/clear_float_status 源仓为 [cann/ops-nn](https://gitcode.com/cann/ops-nn) 的 `control/` 目录；assert / case_condition / sleep / update_tensor_desc 为语义基准（源仓实现非 regbase AICore 形态，不作编码参照），见适配层 [index.md §4](../index.md)。

> 源仓后续演进时以源仓为准；本镜像不追新，仅锚定蒸馏版本。
