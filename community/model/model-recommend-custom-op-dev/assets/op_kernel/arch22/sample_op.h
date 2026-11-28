/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SAMPLE_OP_KERNEL_H__
#define __SAMPLE_OP_KERNEL_H__

#include "kernel_operator.h"
#include "sample_op_tiling_data.h"

namespace NsSampleOp {

using namespace AscendC;

template <typename D_T, int SCH_MODE>
class SampleOp {
public:
    __aicore__ inline SampleOp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, SampleOpTilingData* tilingData)
    {
        totalRows = tilingData->totalRows;
        cols = tilingData->cols;
        rowsPerCore = tilingData->rowsPerCore;

        const uint32_t blockIndex = GetBlockIdx();
        startRow = static_cast<uint64_t>(blockIndex) * rowsPerCore;
        endRow = startRow + rowsPerCore;
        if (endRow > totalRows) {
            endRow = totalRows;
        }
        if (startRow > totalRows) {
            startRow = totalRows;
        }

        gmX.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(x), totalRows * cols);
        gmY.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(y), totalRows * cols);
    }

    __aicore__ inline void Process()
    {
        if (startRow >= endRow || cols == 0) {
            return;
        }

        const uint32_t bytes = static_cast<uint32_t>(cols * sizeof(D_T));
        pipe.InitBuffer(bufX, bytes);
        pipe.InitBuffer(bufY, bytes);
        LocalTensor<D_T> xLocal = bufX.Get<D_T>();
        LocalTensor<D_T> yLocal = bufY.Get<D_T>();

        for (uint64_t row = startRow; row < endRow; ++row) {
            const int64_t offset = static_cast<int64_t>(row * cols);
            DataCopy(xLocal, gmX[offset], static_cast<int32_t>(cols));
            pipe_barrier(PIPE_ALL);

            // TODO(<op>): 真正的逐行计算逻辑，结果写入 yLocal
            DataCopy(yLocal, xLocal, static_cast<int32_t>(cols));
            pipe_barrier(PIPE_ALL);

            DataCopy(gmY[offset], yLocal, static_cast<int32_t>(cols));
            pipe_barrier(PIPE_ALL);
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufX;
    TBuf<TPosition::VECCALC> bufY;
    GlobalTensor<D_T> gmX;
    GlobalTensor<D_T> gmY;
    uint64_t totalRows = 0;
    uint64_t cols = 0;
    uint64_t rowsPerCore = 0;
    uint64_t startRow = 0;
    uint64_t endRow = 0;
};

} // namespace NsSampleOp

#endif // __SAMPLE_OP_KERNEL_H__
