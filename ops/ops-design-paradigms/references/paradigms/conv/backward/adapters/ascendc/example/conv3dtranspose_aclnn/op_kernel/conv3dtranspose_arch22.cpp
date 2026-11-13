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
 * \file conv3dtranspose_arch22.cpp
 * \brief Conv3DTranspose 算子 kernel 入口（arch22 架构）
 *
 * 入口签名对齐 design/Kernel.md「1. Kernel 入口」：入口名 conv3dtranspose，
 * 参数声明序 input_size、x、filter、bias、offset_w、y、workspace、tiling
 * （输入按 Interface.md 声明序，输出 y，再 workspace、tiling）。
 *
 * 本文件当前为接口对齐的空跑桩：仅保留 SOC 状态初始化、tiling 读取与
 * 流水屏障，不含计算实现（计算由后续 kernel 任务按 Kernel.md 写入）。
 *
 * 模板参数说明（与 conv3dtranspose_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 定义对应，
 * 由 ASCENDC_TPL_SEL 实例化，本桩体内不消费）：
 *   - D_T_X: 数据类型，由 ASCENDC_TPL_DATATYPE_DECL 定义
 *   - BUFFER_MODE: 缓冲模式（0=单缓冲, 1=双缓冲），由 ASCENDC_TPL_UINT_DECL 定义
 *
 * 模板参数带默认值：SEL 组合经 `conv3dtranspose<TypeFromId<..>::type, N>(...)` 显式
 * 实参实例化；SEL 无匹配组合回落到 DEFAULT_TILING_KEY，生成侧以无实参形式调用
 * 入口——默认模板参数保证两种调用形式均可编译。空跑桩不消费任何模板参数，
 * 默认取值无语义影响。
 */

#include "kernel_operator.h"
#include "arch22/conv3dtranspose_tiling_data.h"
#include "arch22/conv3dtranspose_tiling_key.h"

template <typename D_T_X = int32_t, int BUFFER_MODE = 0>
__global__ __aicore__ void conv3dtranspose(GM_ADDR input_size, GM_ADDR x, GM_ADDR filter, GM_ADDR bias,
                                           GM_ADDR offset_w, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(Conv3DTransposeTilingData);
    GET_TILING_DATA_WITH_STRUCT(Conv3DTransposeTilingData, tilingData, tiling);

    // 空跑桩：计算未实现，显式弃置全部未消费参数，避免未使用告警。
    (void)input_size;
    (void)x;
    (void)filter;
    (void)bias;
    (void)offset_w;
    (void)y;
    (void)workspace;
    (void)tilingData;
    // F-07 整改：删除 PipeBarrier<PIPE_ALL>()——空跑桩无任何需排序的指令依赖
    // （全流水阻塞无场景豁免，命中即 FAIL）。
}
