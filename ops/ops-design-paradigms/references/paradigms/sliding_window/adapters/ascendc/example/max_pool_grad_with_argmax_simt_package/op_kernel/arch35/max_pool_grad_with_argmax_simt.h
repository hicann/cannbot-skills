/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* DRAFT — 范式参考模板，未编译、未上板 */

/*!
 * \file max_pool_grad_with_argmax_simt.h
 * \brief SlidingWindow 反向 GradRoute·B（argmax 路由）SIMT kernel 模板
 *
 * 输入驱动 grid-stride：每个输入位置独立枚举覆盖它的输出窗口（反窗口区间
 * 见范式 window-math.md §4），按正向 argmax mask 命中检查累加梯度——零原子
 * 逐位直写（grad-routing.md §4）。坐标分解与反窗口端点均走 UintDiv 快除
 * （CAP_INT_DIV_COST 反向档，8 参数经 UB 广播）。
 *
 * 算法语义对照 ops-nn pooling/max_pool_grad_with_argmax_v3 的 SIMT 路线
 * kernel（ascend950 实测，NCHW 布局）。
 */

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_SIMT_H_
#define MAX_POOL_GRAD_WITH_ARGMAX_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "max_pool_grad_with_argmax_tiling_data.h"

namespace NsMaxPoolGradWithArgmax {
using namespace AscendC;

static constexpr uint32_t THREAD_DIM = 1024; // 反向反窗口枚举线程档（splitting.md §4：高档隐藏无效扫描）

// 反窗口首端：首个覆盖 size 位置的输出窗口（window-math.md §4，整数域快除）
template <typename T>
__simt_callee__ __aicore__ inline T PStart(T size, T pad, T kernel, T dilation, T magicStride, T shiftStride)
{
    if (size + pad < ((kernel - 1) * dilation + 1)) {
        return 0; // 无覆盖窗口
    }
    T pStart = size + pad - ((kernel - 1) * dilation + 1);
    pStart = Simt::UintDiv<T>(pStart, magicStride, shiftStride);
    pStart += 1;
    return pStart;
}

// 反窗口末端（+1 开区间），由 poolSize 封顶
template <typename T>
__simt_callee__ __aicore__ inline T PEnd(T size, T pad, T poolSize, T magicStride, T shiftStride)
{
    T pEnd = size + pad;
    pEnd = Simt::UintDiv<T>(pEnd, magicStride, shiftStride);
    pEnd += 1;
    return pEnd > poolSize ? poolSize : pEnd;
}

template <typename VALUE_T, typename IDX_T, typename INDEX_SIZE_T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_DIM) inline void MaxPoolGradWithArgmaxNchw(
    __ubuf__ INDEX_SIZE_T* divParams, __gm__ const VALUE_T* grad, INDEX_SIZE_T count, int32_t inH, int32_t inW,
    int32_t outH, int32_t outW, int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t padH, int32_t padW,
    int32_t dilH, int32_t dilW, __gm__ VALUE_T* y, __gm__ const IDX_T* mask)
{
    const INDEX_SIZE_T magicHW = divParams[0]; // H*W 平面快除（nc = index / (H*W)）
    const INDEX_SIZE_T shiftHW = divParams[1];
    const INDEX_SIZE_T magicW = divParams[2]; // W 快除（h = r / W）
    const INDEX_SIZE_T shiftW = divParams[3];
    const INDEX_SIZE_T magicSH = divParams[4]; // strideH 快除（反窗口端点）
    const INDEX_SIZE_T shiftSH = divParams[5];
    const INDEX_SIZE_T magicSW = divParams[6]; // strideW 快除
    const INDEX_SIZE_T shiftSW = divParams[7];

    for (INDEX_SIZE_T index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index = index + blockDim.x * gridDim.x) {
        // 坐标分解（快除）：index -> (nc, h, w)
        INDEX_SIZE_T nc = Simt::UintDiv<INDEX_SIZE_T>(index, magicHW, shiftHW);
        INDEX_SIZE_T rem = index - nc * static_cast<INDEX_SIZE_T>(inH) * inW;
        INDEX_SIZE_T h = Simt::UintDiv<INDEX_SIZE_T>(rem, magicW, shiftW);
        INDEX_SIZE_T w = rem - h * static_cast<INDEX_SIZE_T>(inW);

        // 反窗口区间（window-math.md §4：与正向窗口公式互为逆运算，同源核对）
        INDEX_SIZE_T phstart = PStart<INDEX_SIZE_T>(h, static_cast<INDEX_SIZE_T>(padH), static_cast<INDEX_SIZE_T>(kH),
                                                    static_cast<INDEX_SIZE_T>(dilH), magicSH, shiftSH);
        INDEX_SIZE_T phend =
            PEnd<INDEX_SIZE_T>(h, static_cast<INDEX_SIZE_T>(padH), static_cast<INDEX_SIZE_T>(outH), magicSH, shiftSH);
        INDEX_SIZE_T pwstart = PStart<INDEX_SIZE_T>(w, static_cast<INDEX_SIZE_T>(padW), static_cast<INDEX_SIZE_T>(kW),
                                                    static_cast<INDEX_SIZE_T>(dilW), magicSW, shiftSW);
        INDEX_SIZE_T pwend =
            PEnd<INDEX_SIZE_T>(w, static_cast<INDEX_SIZE_T>(padW), static_cast<INDEX_SIZE_T>(outW), magicSW, shiftSW);

        // argmax 路由检查 + 贡献累加（grad-routing.md §4：mask 命中即本位置为 winner 的窗口）
        const __gm__ VALUE_T* gradSlice = grad + nc * static_cast<INDEX_SIZE_T>(outH) * outW;
        const __gm__ IDX_T* maskSlice = mask + nc * static_cast<INDEX_SIZE_T>(outH) * outW;
        const IDX_T srcIdx = static_cast<IDX_T>(h * static_cast<INDEX_SIZE_T>(inW) + w); // 口径与正向一致
        float gradient = 0;
        for (INDEX_SIZE_T ph = phstart; ph < phend; ++ph) {
            for (INDEX_SIZE_T pw = pwstart; pw < pwend; ++pw) {
                INDEX_SIZE_T idx = ph * static_cast<INDEX_SIZE_T>(outW) + pw;
                if (maskSlice[idx] == srcIdx) {
                    gradient += static_cast<float>(gradSlice[idx]);
                }
            }
        }
        y[index] = static_cast<VALUE_T>(gradient); // 零原子逐位直写
    }
}

template <typename VALUE_T, typename IDX_T, typename INDEX_SIZE_T>
__aicore__ inline void Process(GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxTilingData, tilingData, tiling);
    if (tilingData.count == 0) {
        return; // 空输入短路
    }

    // 快除参数表：每维 2 项（magic/shift），写 UB 后同步再进 kernel（kernel-entrance.md §4）
    LocalMemAllocator<AscendC::Hardware::UB> ubAlloc;
    LocalTensor<INDEX_SIZE_T> divParams = ubAlloc.Alloc<INDEX_SIZE_T>(8);
    INDEX_SIZE_T magicHW = 0;
    INDEX_SIZE_T shiftHW = 0;
    INDEX_SIZE_T magicW = 0;
    INDEX_SIZE_T shiftW = 0;
    INDEX_SIZE_T magicSH = 0;
    INDEX_SIZE_T shiftSH = 0;
    INDEX_SIZE_T magicSW = 0;
    INDEX_SIZE_T shiftSW = 0;
    GetUintDivMagicAndShift<INDEX_SIZE_T>(magicHW, shiftHW, tilingData.inH * tilingData.inW);
    GetUintDivMagicAndShift<INDEX_SIZE_T>(magicW, shiftW, tilingData.inW);
    GetUintDivMagicAndShift<INDEX_SIZE_T>(magicSH, shiftSH, tilingData.sH);
    GetUintDivMagicAndShift<INDEX_SIZE_T>(magicSW, shiftSW, tilingData.sW);
    divParams.SetValue(0, magicHW);
    divParams.SetValue(1, shiftHW);
    divParams.SetValue(2, magicW);
    divParams.SetValue(3, shiftW);
    divParams.SetValue(4, magicSH);
    divParams.SetValue(5, shiftSH);
    divParams.SetValue(6, magicSW);
    divParams.SetValue(7, shiftSW);
    DataSyncBarrier<MemDsbT::UB>();

    __gm__ VALUE_T* gradGm = (__gm__ VALUE_T*)grad;
    __gm__ IDX_T* maskGm = (__gm__ IDX_T*)argmax;
    __gm__ VALUE_T* yGm = (__gm__ VALUE_T*)y;
    asc_vf_call<MaxPoolGradWithArgmaxNchw<VALUE_T, IDX_T, INDEX_SIZE_T>>(
        dim3(THREAD_DIM), (__ubuf__ INDEX_SIZE_T*)divParams.GetPhyAddr(), gradGm,
        static_cast<INDEX_SIZE_T>(tilingData.count), tilingData.inH, tilingData.inW, tilingData.outH, tilingData.outW,
        tilingData.kH, tilingData.kW, tilingData.sH, tilingData.sW, tilingData.padH, tilingData.padW, tilingData.dilH,
        tilingData.dilW, yGm, maskGm);
}

} // namespace NsMaxPoolGradWithArgmax

#endif // MAX_POOL_GRAD_WITH_ARGMAX_SIMT_H_
