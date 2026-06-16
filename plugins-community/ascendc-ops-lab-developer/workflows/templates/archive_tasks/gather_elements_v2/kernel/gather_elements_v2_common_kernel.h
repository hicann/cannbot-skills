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
#include "gather_elements_v2_tiling.h"

template <typename DataType>
class GatherElementsV2CommonKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR index,
        GM_ADDR rowMap,
        GM_ADDR y,
        GM_ADDR tilingGM,
        AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(x), tiling_.XRows * tiling_.XStride);
        indexGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(index), tiling_.M * tiling_.YStride);
        if (tiling_.useRowMap != 0) {
            rowMapGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rowMap), tiling_.M);
        }
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(y), tiling_.M * tiling_.YStride);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            subBlockRows_ = tiling_.subBlockM;
            pipe_->InitBuffer(xInQueue_, 1, AlignedXRowBytes());
            pipe_->InitBuffer(indexInQueue_, 1, AlignedIndexRowBytes());
            pipe_->InitBuffer(yOutQueue_, 1, AlignedYRowBytes());
            if (tiling_.useRowMap != 0) {
                pipe_->InitBuffer(rowMapInQueue_, 1, AlignBytes(sizeof(int32_t)));
            }
        }
    }

protected:
    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const GatherElementsV2KernelTiling &tiling = this->Tiling();
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();

            for (int localIdx = 0; localIdx < tiling.tasksPerCore; ++localIdx) {
                const int bx = coreIdx * tiling.tasksPerCore + localIdx;
                if (bx >= this->BlockCount()) {
                    continue;
                }

                const int rowBase = bx * tiling.blockM + subBlockIdx * tiling.subBlockM;
                for (int row = 0; row < this->SubBlockRows(); ++row) {
                    const int rowIdx = rowBase + row;
                    if (rowIdx < tiling.M) {
                        const int32_t xRowIdx = this->ResolveXRowFromMap(rowIdx);
                        this->ProcessGatherRow(rowIdx, xRowIdx);
                    }
                }
            }
        }
    }

    __aicore__ inline const GatherElementsV2KernelTiling &Tiling() const
    {
        return tiling_;
    }

    __aicore__ inline int32_t SubBlockRows() const
    {
        return subBlockRows_;
    }

    __aicore__ inline int32_t BlockCount() const
    {
        return (tiling_.M + tiling_.blockM - 1) / tiling_.blockM;
    }

    __aicore__ inline uint32_t XRowBytes() const
    {
        return static_cast<uint32_t>(tiling_.XStride * sizeof(DataType));
    }

    __aicore__ inline uint32_t IndexRowBytes() const
    {
        return static_cast<uint32_t>(tiling_.YStride * sizeof(int32_t));
    }

    __aicore__ inline uint32_t YRowBytes() const
    {
        return static_cast<uint32_t>(tiling_.YStride * sizeof(DataType));
    }

    __aicore__ inline uint32_t AlignBytes(uint32_t bytes) const
    {
        return ((bytes + 31U) / 32U) * 32U;
    }

    __aicore__ inline uint32_t AlignedXRowBytes() const
    {
        return AlignBytes(XRowBytes());
    }

    __aicore__ inline uint32_t AlignedIndexRowBytes() const
    {
        return AlignBytes(IndexRowBytes());
    }

    __aicore__ inline uint32_t AlignedYRowBytes() const
    {
        return AlignBytes(YRowBytes());
    }

    __aicore__ inline void CopyXRowToUb(AscendC::LocalTensor<DataType> &dst, AscendC::GlobalTensor<DataType> src)
    {
        const uint32_t padElems = (AlignedXRowBytes() - XRowBytes()) / sizeof(DataType);
        AscendC::DataCopyExtParams copyParams{1, XRowBytes(), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DataType> padParams{
            true, 0, static_cast<uint8_t>(padElems), static_cast<DataType>(0)};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    __aicore__ inline void CopyIndexRowToUb(AscendC::LocalTensor<int32_t> &dst, AscendC::GlobalTensor<int32_t> src)
    {
        const uint32_t padElems = (AlignedIndexRowBytes() - IndexRowBytes()) / sizeof(int32_t);
        AscendC::DataCopyExtParams copyParams{1, IndexRowBytes(), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, static_cast<uint8_t>(padElems), 0};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    __aicore__ inline void CopyYRowToGm(AscendC::GlobalTensor<DataType> dst, AscendC::LocalTensor<DataType> &src)
    {
        AscendC::DataCopyExtParams copyParams{1, YRowBytes(), 0, 0, 0};
        AscendC::DataCopyPad(dst, src, copyParams);
    }

    __aicore__ inline int32_t ResolveXRowFromMap(int rowIdx)
    {
        if (tiling_.useRowMap == 0) {
            return rowIdx;
        }

        rowMapInQueue_.AllocTensor<int32_t>(rowMapLocal_);
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        AscendC::DataCopyPad(rowMapLocal_, rowMapGM_[rowIdx], copyParams, padParams);
        rowMapInQueue_.EnQue(rowMapLocal_);
        rowMapInQueue_.DeQue<int32_t>(rowMapLocal_);
        const int32_t mappedRow = rowMapLocal_.GetValue(0);
        rowMapInQueue_.FreeTensor(rowMapLocal_);
        return mappedRow;
    }

    __aicore__ inline void ProcessGatherRow(int rowIdx, int32_t xRowIdx)
    {
        xInQueue_.AllocTensor<DataType>(xLocal_);
        indexInQueue_.AllocTensor<int32_t>(indexLocal_);
        yOutQueue_.AllocTensor<DataType>(yLocal_);

        CopyXRowToUb(xLocal_, xGM_[xRowIdx * tiling_.XStride]);
        CopyIndexRowToUb(indexLocal_, indexGM_[rowIdx * tiling_.YStride]);
        xInQueue_.EnQue(xLocal_);
        indexInQueue_.EnQue(indexLocal_);

        xInQueue_.DeQue<DataType>(xLocal_);
        indexInQueue_.DeQue<int32_t>(indexLocal_);

        AscendC::Duplicate(yLocal_, static_cast<DataType>(0), tiling_.YStride);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int col = 0; col < tiling_.IG; ++col) {
            const int32_t gatherIdx = indexLocal_.GetValue(col);
            const DataType value = xLocal_.GetValue(gatherIdx);
            yLocal_.SetValue(col, value);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        xInQueue_.FreeTensor(xLocal_);
        indexInQueue_.FreeTensor(indexLocal_);
        yOutQueue_.EnQue(yLocal_);

        yOutQueue_.DeQue<DataType>(yLocal_);
        CopyYRowToGm(yGM_[rowIdx * tiling_.YStride], yLocal_);
        yOutQueue_.FreeTensor(yLocal_);
    }

private:
    GatherElementsV2KernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int subBlockRows_{0};

    AscendC::GlobalTensor<DataType> xGM_;
    AscendC::GlobalTensor<int32_t> indexGM_;
    AscendC::GlobalTensor<int32_t> rowMapGM_;
    AscendC::GlobalTensor<DataType> yGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> indexInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> rowMapInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> yOutQueue_;

    AscendC::LocalTensor<DataType> xLocal_;
    AscendC::LocalTensor<int32_t> indexLocal_;
    AscendC::LocalTensor<int32_t> rowMapLocal_;
    AscendC::LocalTensor<DataType> yLocal_;
};
