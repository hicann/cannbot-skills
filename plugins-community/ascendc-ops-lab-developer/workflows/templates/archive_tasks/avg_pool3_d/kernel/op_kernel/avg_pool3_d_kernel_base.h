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

#include "kernel_common.h"
#include "avg_pool3_d_tiling.h"

template <typename Derived>
class AvgPool3DKernelBase {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        InitCore(x, y, pipe, tiling_.vectorLen);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int workerId = static_cast<int>(AscendC::GetBlockIdx());
            int workerCount = tiling_.launchBlocks;
            if (workerCount <= 0) {
                workerCount = 1;
            }
            const int totalRows = tiling_.N * tiling_.outSpatial;
            for (int outRow = workerId; outRow < totalRows; outRow += workerCount) {
                ProcessOneRow(outRow);
            }
        }
    }

    __aicore__ inline void InitAccumulator(int cLen)
    {
        accLocal_ = accBuf_.Get<float>();
        AscendC::Duplicate(accLocal_, 0.0f, cLen);
    }

    __aicore__ inline void AccumulateOne(int inRow, int cBase, int cLen)
    {
        inQueue_.AllocTensor<float>(inLocal_);
        AscendC::DataCopy(inLocal_, xGM_[static_cast<uint64_t>(inRow) * tiling_.C + cBase], cLen);
        inQueue_.EnQue(inLocal_);
        inQueue_.DeQue<float>(inLocal_);
        AscendC::Add(accLocal_, accLocal_, inLocal_, cLen);
        inQueue_.FreeTensor(inLocal_);
    }

    __aicore__ inline void AccumulateKW(int nIdx, int idVal, int ihVal, int ow, int cBase, int cLen,
                                         int wTile = -1, int kwGroups = -1)
    {
        if (wTile < 0) {
            wTile = tiling_.kW;
        }
        if (kwGroups < 0) {
            kwGroups = static_cast<int>(CeilDivU32(static_cast<uint32_t>(tiling_.kW), static_cast<uint32_t>(wTile)));
        }
        for (int kwBase = 0; kwBase < kwGroups; ++kwBase) {
            for (int kwLocal = 0; kwLocal < wTile; ++kwLocal) {
                const int kw = kwBase * wTile + kwLocal;
                if (kw >= tiling_.kW) {
                    continue;
                }
                const int iwVal = ow * tiling_.sW - tiling_.pW + kw;
                if (iwVal < 0 || iwVal >= tiling_.W) {
                    continue;
                }
                const int inRow = nIdx * tiling_.inSpatial + idVal * tiling_.hw + ihVal * tiling_.W + iwVal;
                AccumulateOne(inRow, cBase, cLen);
            }
        }
    }

    __aicore__ inline void PoolKdKhLoop(int nIdx, int od, int oh, int ow, int cBase, int cLen,
                                        int wTile = -1, int kwGroups = -1)
    {
        for (int kd = 0; kd < tiling_.kD; ++kd) {
            const int idVal = od * tiling_.sD - tiling_.pD + kd;
            if (idVal < 0 || idVal >= tiling_.D) {
                continue;
            }
            for (int kh = 0; kh < tiling_.kH; ++kh) {
                const int ihVal = oh * tiling_.sH - tiling_.pH + kh;
                if (ihVal < 0 || ihVal >= tiling_.H) {
                    continue;
                }
                AccumulateKW(nIdx, idVal, ihVal, ow, cBase, cLen, wTile, kwGroups);
            }
        }
    }

    __aicore__ inline void ProcessOneRowChunk(int nIdx, int od, int oh, int ow, int outRow, int cBase, int cLen)
    {
        InitAccumulator(cLen);
        PoolKdKhLoop(nIdx, od, oh, ow, cBase, cLen);
        AvgPool3DStoreOutput(outQueue_, outLocal_, accLocal_, yGM_,
                              outRow, cBase, cLen, od, oh, ow, tiling_);
    }

    __aicore__ inline void ProcessOneRow(int outRow)
    {
        const int nIdx = outRow / tiling_.outSpatial;
        const int outRem = outRow - nIdx * tiling_.outSpatial;
        const int od = outRem / (tiling_.OH * tiling_.OW);
        const int odRem = outRem - od * (tiling_.OH * tiling_.OW);
        const int oh = odRem / tiling_.OW;
        const int ow = odRem - oh * tiling_.OW;

        for (int cBase = 0; cBase < tiling_.C; cBase += vecLen_) {
            const int remain = tiling_.C - cBase;
            const int cLen = remain < vecLen_ ? remain : vecLen_;
            static_cast<Derived *>(this)->ProcessOneRowChunk(nIdx, od, oh, ow, outRow, cBase, cLen);
        }
    }

protected:
    __aicore__ inline void InitCore(GM_ADDR x, GM_ADDR y, AscendC::TPipe *pipe, int vectorLen)
    {
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), static_cast<uint64_t>(tiling_.N) * tiling_.inSpatial * tiling_.C);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), static_cast<uint64_t>(tiling_.N) * tiling_.outSpatial * tiling_.C);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            vecLen_ = vectorLen > 0 ? vectorLen : tiling_.C;
            if (vecLen_ > tiling_.C) {
                vecLen_ = tiling_.C;
            }

            pipe_->InitBuffer(inQueue_, 1, static_cast<uint32_t>(vecLen_ * sizeof(float)));
            pipe_->InitBuffer(outQueue_, 1, static_cast<uint32_t>(vecLen_ * sizeof(float)));
            pipe_->InitBuffer(accBuf_, static_cast<uint32_t>(vecLen_ * sizeof(float)));
        }
    }

    __aicore__ inline int ComputeDivisor(int od, int oh, int ow) const
    {
        return ComputeDivisorHelper(od, oh, ow, tiling_);
    }

protected:
    AvgPool3DKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int vecLen_{0};

    AscendC::GlobalTensor<float> xGM_;
    AscendC::GlobalTensor<float> yGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf_;

    AscendC::LocalTensor<float> inLocal_;
    AscendC::LocalTensor<float> outLocal_;
    AscendC::LocalTensor<float> accLocal_;
};
