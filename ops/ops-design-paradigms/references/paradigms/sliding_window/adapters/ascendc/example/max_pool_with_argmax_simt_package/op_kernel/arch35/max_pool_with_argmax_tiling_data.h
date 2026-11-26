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

#ifndef MAX_POOL_WITH_ARGMAX_TILING_DATA_H_
#define MAX_POOL_WITH_ARGMAX_TILING_DATA_H_

/*!
 * \file max_pool_with_argmax_tiling_data.h
 * \brief 正向 TilingData：host 可算标量包（kernel-entrance.md §2 规则）
 *
 * 只装 host 侧可算的窗口参数与切分参数；线程数不入（编译期常量，与
 * __launch_bounds__ 共用）；快除参数本正向模板不使用（直除形态）。
 */
struct MaxPoolWithArgmaxTilingData {
    int64_t count;       // 输出元素总数 = N*C*outH*outW（正向主坐标）
    int32_t channels;    // C（NHWC 分解用；NCHW 合并进 nc 不单独使用）
    int32_t inH;         // 输入高
    int32_t inW;         // 输入宽
    int32_t outH;        // 输出高（host 由窗口参数推导）
    int32_t outW;        // 输出宽
    int32_t kH;          // 窗口高
    int32_t kW;          // 窗口宽
    int32_t sH;          // 步长高
    int32_t sW;          // 步长宽
    int32_t padH;        // 对称填充高（4 边填充的对称特例，window-math.md §2）
    int32_t padW;        // 对称填充宽
    int32_t dilH;        // 空洞高
    int32_t dilW;        // 空洞宽
    int32_t needCoreNum; // 实际用核数（两步法，splitting.md §3）
};

#endif // MAX_POOL_WITH_ARGMAX_TILING_DATA_H_
