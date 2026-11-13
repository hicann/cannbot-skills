/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file conv3dtranspose_tiling.cpp
 * \brief conv3dtranspose Host 侧 Tiling 编译入口（ascend950 目标唯一 tiling 编译单元适配层）
 *
 * 生成工程以 arch35/conv3dtranspose_tiling.cpp 作为 ascend950 目标的唯一
 * tiling 编译单元（npu_op_code_gen 与 npu_op_library 均以本文件为源），故本文件是
 * IMPL_OP_OPTILING 注册与 TilingFunc 的编译载体。write-tiling-ut 任务已将样例工程的工作
 * TilingFunc 缩减为 TDD 红相位 stub，落位于同目录 conv3dtranspose_tiling_arch35.cpp
 * （本文件转发编译，单一注册点不变）；真实实现由 implement-tiling 任务在
 * conv3dtranspose_tiling_arch35.cpp 内替换 stub（绿相位见 op/tests/tiling/ UT）。
 */

#include "conv3dtranspose_tiling_arch35.cpp"
