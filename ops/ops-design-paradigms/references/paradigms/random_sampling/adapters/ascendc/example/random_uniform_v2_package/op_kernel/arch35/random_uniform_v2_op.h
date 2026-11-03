/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANDOM_UNIFORM_V2_H
#define RANDOM_UNIFORM_V2_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "../../random_common/arch35/random_kernel_base.h"

namespace RandomUniformV2 {
using namespace AscendC;
using namespace RandomKernelBase;

template <typename T>
class RandomUniformV2Op : public RandomKernelBaseOp {
public:
    __aicore__ inline RandomUniformV2Op(TPipe* pipe, const RandomUnifiedTilingDataStruct* __restrict tilingData)
        : RandomKernelBaseOp(tilingData),
          pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR offset);
    __aicore__ inline void Process();

private:
    TPipe* pipe_;
    static constexpr uint16_t BUFFER_NUM = 2;
    static constexpr uint16_t RESULT_ELEMENT_CNT = 4;
    static constexpr uint64_t K_RESERVEED_PER_OUTPUT = 256;

    GlobalTensor<T> outputGm_;
    GlobalTensor<int64_t> offsetGm_;
    TBuf<QuePosition::VECCALC> philoxQueBuf_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueY_;
};

template <typename T>
__aicore__ inline void RandomUniformV2Op<T>::Init(GM_ADDR y, GM_ADDR offset)
{
    VarsInit();

    outputGm_.SetGlobalBuffer((__gm__ T*)y);
    offsetGm_.SetGlobalBuffer((__gm__ int64_t*)offset);
    pipe_->InitBuffer(outQueY_, BUFFER_NUM, tiling_->singleBufferSize * sizeof(T));
    pipe_->InitBuffer(philoxQueBuf_, tiling_->singleBufferSize * sizeof(uint32_t));
}

template <typename T>
__aicore__ inline void RandomUniformV2Op<T>::Process()
{
    if (blockIdx_ > tiling_->usedCoreNum) {
        return;
    }

    auto offsetValue = offsetGm_.GetValue(0);
    if (offsetValue > 0) {
        Skip(offsetValue);
    }

    SyncAll();
    auto blockOffSet = tiling_->normalCoreProNum * blockIdx_;
    auto groupCnt = (blockOffSet + RESULT_ELEMENT_CNT - 1) / RESULT_ELEMENT_CNT;

    Skip(groupCnt);
    for (auto idx = 0; idx < ubRepeatimes_; idx++) {
        int64_t philoxNumPro = idx == (ubRepeatimes_ - 1) ?
                                   curCoreProNum_ - (ubRepeatimes_ - 1) * tiling_->singleBufferSize :
                                   tiling_->singleBufferSize;
        int64_t philoxNumOffset = idx * tiling_->singleBufferSize;

        LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
        GenRandomSIMD(philoxRes, philoxNumPro);
        LocalTensor<T> yOutput = outQueY_.AllocTensor<T>();
        U32Conversion(yOutput, philoxRes, philoxNumPro);
        outQueY_.EnQue(yOutput);

        yOutput = outQueY_.DeQue<T>();
        int64_t yOffset = blockOffSet + philoxNumOffset;
        CopyOut(yOutput, outputGm_, 1, philoxNumPro * sizeof(T), yOffset);
        outQueY_.FreeTensor(yOutput);
        groupCnt = (philoxNumPro + RESULT_ELEMENT_CNT - 1) / RESULT_ELEMENT_CNT;
        Skip(groupCnt);
    }

    if (blockIdx_ == 0) {
        offsetValue = offsetValue + tiling_->outputSize * K_RESERVEED_PER_OUTPUT;
        offsetGm_.SetValue(0, offsetValue);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ATOMIC>(offsetGm_);
    }
}
} // namespace RandomUniformV2
#endif
