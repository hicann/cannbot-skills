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

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_TILING_DATA_H_
#define MAX_POOL_GRAD_WITH_ARGMAX_TILING_DATA_H_

/*!
 * \file max_pool_grad_with_argmax_tiling_data.h
 * \brief 反向 TilingData：host 可算标量包（kernel-entrance.md §2 规则）
 *
 * count = N*C*H*W（反向主坐标，输入驱动）；快除 magic/shift 不入（宿主段
 * 算好经 UB 广播）；线程数不入（编译期常量）。
 */
struct MaxPoolGradWithArgmaxTilingData {
    int64_t count;       // 输入元素总数 = N*C*H*W（grad_input 同形）
    int32_t inH;         // 原输入高（y 形状源）
    int32_t inW;         // 原输入宽
    int32_t outH;        // 正向输出高（grad/argmax 平面维）
    int32_t outW;        // 正向输出宽
    int32_t kH;          // 窗口高（与正向同源）
    int32_t kW;          // 窗口宽
    int32_t sH;          // 步长高
    int32_t sW;          // 步长宽
    int32_t padH;        // 对称填充高
    int32_t padW;        // 对称填充宽
    int32_t dilH;        // 空洞高
    int32_t dilW;        // 空洞宽
    int32_t needCoreNum; // 实际用核数（两步法，splitting.md §3）
};

#endif // MAX_POOL_GRAD_WITH_ARGMAX_TILING_DATA_H_
