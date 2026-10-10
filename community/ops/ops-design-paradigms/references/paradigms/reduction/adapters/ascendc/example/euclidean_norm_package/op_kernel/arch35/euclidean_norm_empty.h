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
 * \file euclidean_norm_empty.h
 * \brief EuclideanNorm 空 tensor 模板 kernel 类实现（新范式）。
 */
#ifndef OPS_NORM_EUCLIDEAN_NORM_EMPTY_H_
#define OPS_NORM_EUCLIDEAN_NORM_EMPTY_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "euclidean_norm_tiling_data.h"
#include "euclidean_norm_tiling_key.h"

namespace NsEuclideanNorm {

using namespace AscendC;

constexpr uint32_t VL_BYTES_EMPTY = Ops::Base::GetVRegSize();
constexpr float EMPTY_R_OUTPUT_VALUE = 0.0f; // empty_r_output_value：EMPTY_R 输出固化值

// EMPTY_R 输出填充 VF：Duplicate 固化值 sqrt(0)=0 到 outBuf，供后续 CopyOut 循环复用。
template <typename DType>
__simd_vf__ inline void DuplicateEmptyROutputVfImpl(__ubuf__ DType* outPtr, DType value, uint32_t totalElems,
                                                    uint16_t repeatTime)
{
    constexpr uint32_t repPerVf = VL_BYTES_EMPTY / sizeof(DType);
    AscendC::Reg::RegTensor<DType> dReg;
    AscendC::Reg::Duplicate(dReg, value);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        const int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(repPerVf);
        mask = AscendC::Reg::UpdateMask<DType>(remaining);
        AscendC::Reg::StoreAlign(outPtr + off, dReg, mask);
    }
}

template <typename DType>
class EuclideanNormEmptyKernel {
public:
    using DT = DType;

    __aicore__ inline EuclideanNormEmptyKernel() {}

    // Empty 初始化：缓存 EmptyTilingData 指针（切分字段经 td_ 直读，不逐字段拷贝到成员），
    // 绑定输出 GM，仅分配 outBuf 一个 UB buffer。
    __aicore__ inline void Init(GM_ADDR y, const EuclideanNormEmptyTilingData* td, TPipe* pipe)
    {
        td_ = td;

        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(y));
        pipe_ = pipe;
        pipe_->InitBuffer(outBuf_, td_->postBufSize);
        evVtoMTE3_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    }

    // Empty 主流程：EMPTY_A 时 usedCoreNum=0 全核直接返回；EMPTY_R 时按大小核均分输出区间，
    // outBuf 填 0 后循环 CopyOut（每个输出 = sqrt(Σ over 空轴) = sqrt(0) = 0）。
    __aicore__ inline void Process()
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if (blockIdx >= static_cast<int64_t>(td_->usedCoreNum)) {
            return;
        }

        int64_t aStart = 0;
        int64_t aEnd = 0;
        if (blockIdx < static_cast<int64_t>(td_->aBigCoreCnt)) {
            aStart = blockIdx * td_->aBigCoreLoopCnt * td_->aUbFactor;
            aEnd = aStart + td_->aBigCoreLoopCnt * td_->aUbFactor;
        } else {
            aStart = static_cast<int64_t>(td_->aBigCoreCnt) * td_->aBigCoreLoopCnt * td_->aUbFactor +
                     (blockIdx - static_cast<int64_t>(td_->aBigCoreCnt)) * td_->aSmallCoreLoopCnt * td_->aUbFactor;
            aEnd = aStart + td_->aSmallCoreLoopCnt * td_->aUbFactor;
        }
        if (aEnd > td_->aTotal) {
            aEnd = td_->aTotal;
        }
        if (aStart >= aEnd) {
            return;
        }

        DuplicateEmptyROutputVf();

        for (int64_t aOff = aStart; aOff < aEnd; aOff += td_->aUbFactor) {
            const int64_t aLen = (aOff + td_->aUbFactor > aEnd) ? (aEnd - aOff) : td_->aUbFactor;
            CopyOut(aOff, aLen);
        }
    }

private:
    // 调 DuplicateEmptyROutputVfImpl 按 td_->aUbFactor 填 outBuf 并做 V→MTE3 事件交接（V 写完才允许 MTE3 搬出）。
    __aicore__ inline void DuplicateEmptyROutputVf()
    {
        __ubuf__ DT* outPtr = reinterpret_cast<__ubuf__ DT*>(outBuf_.Get<DT>().GetPhyAddr());
        const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor);
        const uint32_t repDType = VL_BYTES_EMPTY / sizeof(DT);
        const uint16_t repeatTime = static_cast<uint16_t>(Ops::Base::CeilDiv(totalElems, repDType));

        asc_vf_call<DuplicateEmptyROutputVfImpl<DT>>(outPtr, static_cast<DT>(EMPTY_R_OUTPUT_VALUE), totalElems,
                                                     repeatTime);

        SetFlag<HardEvent::V_MTE3>(evVtoMTE3_);
        WaitFlag<HardEvent::V_MTE3>(evVtoMTE3_);
    }

    __aicore__ inline void CopyOut(int64_t outOff, int64_t aLen)
    {
        auto outLocal = outBuf_.Get<DT>();

        DataCopyExtParams outParams;
        outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(DT)));
        outParams.blockCount = 1;
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad(yGm_[outOff], outLocal, outParams);
    }

    const EuclideanNormEmptyTilingData* td_ = nullptr;

    GlobalTensor<DT> yGm_;
    TPipe* pipe_ = nullptr;
    TBuf<QuePosition::VECCALC> outBuf_;
    event_t evVtoMTE3_ = 0;
};

} // namespace NsEuclideanNorm

#endif // OPS_NORM_EUCLIDEAN_NORM_EMPTY_H_
