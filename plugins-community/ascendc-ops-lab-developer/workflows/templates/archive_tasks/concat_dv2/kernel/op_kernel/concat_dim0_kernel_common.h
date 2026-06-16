/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"

#include "concat_dim0_tiling.h"
#include "concat_dv2_common.h"

template <int INPUT_COUNT>
class ConcatDim0KernelCommon {
public:
    __aicore__ inline void Init(
        GM_ADDR x0,
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR x3,
        GM_ADDR y,
        GM_ADDR tilingGM,
        AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        x0GM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x0), tiling_.M0 * tiling_.N);
        if constexpr (INPUT_COUNT >= 2) {
            x1GM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x1), tiling_.M1 * tiling_.N);
        }
        if constexpr (INPUT_COUNT >= 3) {
            x2GM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x2), tiling_.M2 * tiling_.N);
        }
        if constexpr (INPUT_COUNT >= 4) {
            x3GM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x3), tiling_.M3 * tiling_.N);
        }
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), tiling_.totalM * tiling_.N);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            subBlockRows_ = tiling_.blockM / AscendC::GetSubBlockNum();
            pipe_->InitBuffer(xInQueue_, 1, AlignedRowBytes());
            pipe_->InitBuffer(yOutQueue_, 1, AlignedRowBytes());
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int32_t subBlockIdx = AscendC::GetSubBlockIdx();
            for (int32_t localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int32_t bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= tiling_.blockNum) {
                    continue;
                }
                const int32_t rowBase = bx * tiling_.blockM + subBlockIdx * subBlockRows_;
                for (int32_t row = 0; row < subBlockRows_; ++row) {
                    const int32_t rowIdx = rowBase + row;
                    if (rowIdx < tiling_.totalM) {
                        ProcessRow(rowIdx);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline uint32_t RowBytes() const
    {
        return static_cast<uint32_t>(tiling_.N * static_cast<int32_t>(sizeof(float)));
    }

    __aicore__ inline uint32_t AlignedRowBytes() const
    {
        const uint32_t rowBytes = RowBytes();
        return ((rowBytes + 31U) / 32U) * 32U;
    }

    __aicore__ inline void CopyGmToUbRow(AscendC::LocalTensor<float> &dst, AscendC::GlobalTensor<float> src)
    {
        const uint32_t padElems = (AlignedRowBytes() - RowBytes()) / sizeof(float);
        AscendC::DataCopyExtParams copyParams{1, RowBytes(), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(padElems), 0.0f};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    __aicore__ inline void CopyUbToGmRow(AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> &src)
    {
        AscendC::DataCopyExtParams copyParams{1, RowBytes(), 0, 0, 0};
        AscendC::DataCopyPad(dst, src, copyParams);
    }

    __aicore__ inline void CopyInputRow(int32_t rowIdx, AscendC::LocalTensor<float> &dst)
    {
        if constexpr (INPUT_COUNT == 1) {
            CopyGmToUbRow(dst, x0GM_[rowIdx * tiling_.N]);
            return;
        }

        if (rowIdx < tiling_.M0) {
            CopyGmToUbRow(dst, x0GM_[rowIdx * tiling_.N]);
            return;
        }

        if constexpr (INPUT_COUNT >= 2) {
            const int32_t prefix01 = tiling_.M0 + tiling_.M1;
            if (rowIdx < prefix01) {
                CopyGmToUbRow(dst, x1GM_[(rowIdx - tiling_.M0) * tiling_.N]);
                return;
            }

            if constexpr (INPUT_COUNT >= 3) {
                const int32_t prefix012 = prefix01 + tiling_.M2;
                if (rowIdx < prefix012) {
                    CopyGmToUbRow(dst, x2GM_[(rowIdx - prefix01) * tiling_.N]);
                    return;
                }

                if constexpr (INPUT_COUNT >= 4) {
                    CopyGmToUbRow(dst, x3GM_[(rowIdx - prefix012) * tiling_.N]);
                    return;
                }
            }
        }
    }

    __aicore__ inline void ProcessRow(int32_t rowIdx)
    {
        xInQueue_.AllocTensor<float>(xLocal_);
        yOutQueue_.AllocTensor<float>(yLocal_);
        CopyInputRow(rowIdx, xLocal_);
        xInQueue_.EnQue(xLocal_);

        xInQueue_.DeQue<float>(xLocal_);
        AscendC::Adds(yLocal_, xLocal_, 0.0f, tiling_.N);
        xInQueue_.FreeTensor(xLocal_);
        yOutQueue_.EnQue(yLocal_);

        yOutQueue_.DeQue<float>(yLocal_);
        CopyUbToGmRow(yGM_[rowIdx * tiling_.N], yLocal_);
        yOutQueue_.FreeTensor(yLocal_);
    }

private:
    ConcatDim0Tiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int32_t subBlockRows_{0};

    AscendC::GlobalTensor<float> x0GM_;
    AscendC::GlobalTensor<float> x1GM_;
    AscendC::GlobalTensor<float> x2GM_;
    AscendC::GlobalTensor<float> x3GM_;
    AscendC::GlobalTensor<float> yGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> yOutQueue_;
    AscendC::LocalTensor<float> xLocal_;
    AscendC::LocalTensor<float> yLocal_;
};
