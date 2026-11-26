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
 * \file max_pool_with_argmax_simt.h
 * \brief SlidingWindow 正向 WindowAggregate（max 聚合）SIMT kernel 模板
 *
 * 输出驱动 grid-stride：每个输出位置独立枚举自己的窗口（窗口数学见范式
 * window-math.md §2），−Inf 哨兵初始化 + CycleUpdate NaN 优先更新，双输出
 * 聚合值 y 与 winner 线性索引 mask（反向 argmax 路由依据，口径 h*W+w）。
 *
 * 算法语义对照 ops-nn pooling/max_pool_with_argmax_v3 的 SIMT 路线 kernel
 * （ascend950 实测）；坐标分解为直除形态（CAP_INT_DIV_COST 正向档）。
 */

#ifndef MAX_POOL_WITH_ARGMAX_SIMT_H_
#define MAX_POOL_WITH_ARGMAX_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "max_pool_with_argmax_tiling_data.h"

namespace NsMaxPoolWithArgmax {
using namespace AscendC;

static constexpr uint32_t THREAD_NUM = 256; // 正向窗口聚合线程档（splitting.md §4：中低档，实测口径）

static constexpr uint32_t F32_NEG_INF = 0xff800000u; // fp32 -Inf 位模式（CAP_FP_NEG_INF）

// 聚合更新：严格大于才替换（平局首遇保留），isnan 分支保证 NaN 优先传播（window-math.md §3）
__simt_callee__ __aicore__ inline void CycleUpdate(float val, int64_t idxOffset, float* maxval, int64_t* maxidx)
{
    if ((val > *maxval) || isnan(val)) {
        *maxidx = idxOffset;
        *maxval = val;
    }
}

template <typename X_T, typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void MaxPoolForwardNchw(
    int64_t count, __gm__ const X_T* x, int32_t inH, int32_t inW, int32_t outH, int32_t outW, int32_t kH, int32_t kW,
    int32_t sH, int32_t sW, int32_t padH, int32_t padW, int32_t dilH, int32_t dilW, __gm__ X_T* y, __gm__ IDX_T* mask)
{
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x) {
        // 坐标分解（直除）：index -> (nc, ph, pw)；nc = n*C + c 合并轴（splitting.md §2 降维）
        int64_t pw = index % outW;
        int64_t ph = (index / outW) % outH;
        int64_t nc = index / outW / outH;

        // 窗口边界（window-math.md §2）：span = (k-1)*d+1；end 先定再裁 start
        int64_t hstart = ph * sH - padH;
        int64_t wstart = pw * sW - padW;
        int64_t hspan = (kH - 1) * dilH + 1;
        int64_t wspan = (kW - 1) * dilW + 1;
        int64_t hend = (hstart + hspan) < inH ? (hstart + hspan) : inH;
        int64_t wend = (wstart + wspan) < inW ? (wstart + wspan) : inW;
        while (hstart < 0) {
            hstart += dilH; // 负起点按 dilation 推进（非 clamp 到 0，window-math.md §2）
        }
        while (wstart < 0) {
            wstart += dilW;
        }

        float maxval = *reinterpret_cast<const float*>(&F32_NEG_INF);
        int64_t maxidx = hstart * inW + wstart;       // winner 线性索引口径：批次内 h*W+w
        const __gm__ X_T* plane = x + nc * inH * inW; // nc 已含 n*C+c，通道内平面寻址
        for (int64_t h = hstart; h < hend; h += dilH) {
            for (int64_t w = wstart; w < wend; w += dilW) {
                float val = static_cast<float>(plane[h * inW + w]);
                CycleUpdate(val, h * inW + w, &maxval, &maxidx);
            }
        }
        y[index] = static_cast<X_T>(maxval); // 空窗时 -Inf 直写（window-math.md §3）
        mask[index] = static_cast<IDX_T>(maxidx);
    }
}

template <typename X_T, typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void MaxPoolForwardNhwc(
    int64_t count, __gm__ const X_T* x, int32_t channels, int32_t inH, int32_t inW, int32_t outH, int32_t outW,
    int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t padH, int32_t padW, int32_t dilH, int32_t dilW,
    __gm__ X_T* y, __gm__ IDX_T* mask)
{
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x) {
        // NHWC 分解顺序：c 最内（capability-mapping §2 NCHW/NHWC 双布局）
        int64_t c = index % channels;
        int64_t pw = (index / channels) % outW;
        int64_t ph = (index / channels / outW) % outH;
        int64_t n = index / channels / outW / outH;

        int64_t hstart = ph * sH - padH;
        int64_t wstart = pw * sW - padW;
        int64_t hspan = (kH - 1) * dilH + 1;
        int64_t wspan = (kW - 1) * dilW + 1;
        int64_t hend = (hstart + hspan) < inH ? (hstart + hspan) : inH;
        int64_t wend = (wstart + wspan) < inW ? (wstart + wspan) : inW;
        while (hstart < 0) {
            hstart += dilH;
        }
        while (wstart < 0) {
            wstart += dilW;
        }

        float maxval = *reinterpret_cast<const float*>(&F32_NEG_INF);
        int64_t maxidx = hstart * inW + wstart;
        const __gm__ X_T* plane = x + n * inH * inW * channels;
        for (int64_t h = hstart; h < hend; h += dilH) {
            for (int64_t w = wstart; w < wend; w += dilW) {
                float val = static_cast<float>(plane[(h * inW + w) * channels + c]);
                CycleUpdate(val, h * inW + w, &maxval, &maxidx);
            }
        }
        y[index] = static_cast<X_T>(maxval);
        mask[index] = static_cast<IDX_T>(maxidx);
    }
}

template <typename X_T, typename IDX_T, uint32_t FORMAT_T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, GM_ADDR argmax, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxTilingData, tilingData, tiling);
    if (tilingData.count == 0) {
        return; // 空输出短路（splitting.md §1：下限 1 核，kernel 侧短路）
    }
    __gm__ X_T* xGm = (__gm__ X_T*)x;
    __gm__ X_T* yGm = (__gm__ X_T*)y;
    __gm__ IDX_T* maskGm = (__gm__ IDX_T*)argmax;
    if constexpr (FORMAT_T == 0) { // NCHW
        asc_vf_call<MaxPoolForwardNchw<X_T, IDX_T>>(dim3(THREAD_NUM), tilingData.count, xGm, tilingData.inH,
                                                    tilingData.inW, tilingData.outH, tilingData.outW, tilingData.kH,
                                                    tilingData.kW, tilingData.sH, tilingData.sW, tilingData.padH,
                                                    tilingData.padW, tilingData.dilH, tilingData.dilW, yGm, maskGm);
    } else { // NHWC
        asc_vf_call<MaxPoolForwardNhwc<X_T, IDX_T>>(
            dim3(THREAD_NUM), tilingData.count, xGm, tilingData.channels, tilingData.inH, tilingData.inW,
            tilingData.outH, tilingData.outW, tilingData.kH, tilingData.kW, tilingData.sH, tilingData.sW,
            tilingData.padH, tilingData.padW, tilingData.dilH, tilingData.dilW, yGm, maskGm);
    }
}

} // namespace NsMaxPoolWithArgmax

#endif // MAX_POOL_WITH_ARGMAX_SIMT_H_
