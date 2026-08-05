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
 * \file wp_util.h
 * \brief Coherent whole-port of arch35 common/include/op_kernel/util.h
 *        (LayOutTypeEnum, math helpers, BoolCopyIn). BODY COPIED per port_a3
 *        RED LINE (no #include "arch35/...", we compile our own copy).
 *        Source: ~/workspace/cann/ops-transformer/common/include/op_kernel/util.h
 *        Symbol-name adapted: header guard only; bodies verbatim.
 */
#ifndef WP_FLASH_ATTENTION_UTIL_H
#define WP_FLASH_ATTENTION_UTIL_H

constexpr int32_t blockBytes = 32;
constexpr int32_t byteBitRatio = 8;
constexpr int64_t prefixAttenMaskDownHeight = 1024;
constexpr static int32_t blockSize = blockBytes / 4; // 4 means sizeof(T)
constexpr static int32_t repeatMaxBytes = 256;
constexpr static int32_t repeatMaxTimes = 255;
constexpr static int32_t repeatMaxSize = repeatMaxBytes / 4; // 4 means sizeof(T)

using AscendC::LocalTensor;
using AscendC::GlobalTensor;
using AscendC::DataFormat;
using AscendC::ShapeInfo;
using AscendC::DataCopyParams;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyPadExtParams;
using AscendC::BinaryRepeatParams;
using AscendC::IsSameType;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::WaitFlag;

enum class LayOutTypeEnum {
    None = 0,
    LAYOUT_BSH = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
    LAYOUT_NTD_TND = 5,
    LAYOUT_NTD = 6,
    LAYOUT_NBSD = 7
};

enum class TransposeLayoutEnum : uint32_t {
    None = 0,
    BNSD_BSND = 1,
    BSND_BNSD = 2,
    BSH_BNSD = 3,
    BNSD_NBSD = 4,
    BSND_NBSD = 5,
    BSH_NBSD = 6,
    NTD_TND = 7,
    TND_NTD = 8
};

namespace math {
template <typename T> __aicore__ inline T Ceil(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T> __aicore__ inline T Align(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

__aicore__ inline int32_t Align(int32_t shape)
{
    int32_t alignFactor = 16;
    int32_t alignedSize = CeilDiv<int32_t, int32_t>(shape, alignFactor) * alignFactor;
    return alignedSize;
}

#endif // WP_FLASH_ATTENTION_UTIL_H
