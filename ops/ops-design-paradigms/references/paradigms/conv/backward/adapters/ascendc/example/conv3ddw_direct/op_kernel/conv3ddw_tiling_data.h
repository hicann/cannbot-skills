/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV3DDW_TILING_DATA_H
#define CONV3DDW_TILING_DATA_H

#include <stdint.h>

/* Host/Device 共用的 MN_StreamK 参数。Bound 为元素数，*Bytes 为字节数。
 * cin1G/cout1G 沿用发布字段的含义：实际组内通道数，不是 C0 分形个数。
 */
struct Conv3DdwTilingData {
    uint32_t batch, cin, cout, cin1G, cout1G;
    uint32_t di, hi, wi, dout, ho, wo, dk, hk, wk;
    uint32_t group, realGroup;
    uint32_t strideD, strideH, strideW;
    uint32_t dilationD, dilationH, dilationW;
    uint32_t padFront, padBack, padUp, padDown, padLeft, padRight;
    uint32_t baseM, baseN, baseK;
    uint32_t stepKa, stepKb;
    uint32_t al1Pbuffer, bl1Pbuffer, al0Pbuffer, bl0Pbuffer, cl0Pbuffer;
    uint32_t al1Bound, bl1Bound;
    uint32_t singleCoreM, singleCoreN, singleCoreK;
    uint32_t singleCoreCin, singleCoreCout, singleCoreHo;
    uint32_t usedCoreNum, splitWo, iterateOrder, streamkType;
    uint32_t xFormat, yFormat, dtype;
    uint64_t singleCoreBatchDout;
};

#endif
