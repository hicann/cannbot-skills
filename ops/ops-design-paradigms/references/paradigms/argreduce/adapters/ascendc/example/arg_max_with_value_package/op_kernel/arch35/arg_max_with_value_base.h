/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file arg_max_with_value_base.h
 * \brief arg_max_with_value_base
 */

#ifndef ARG_MAX_WITH_VALUE_BASE_H
#define ARG_MAX_WITH_VALUE_BASE_H

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace ArgMaxWithValue {
using namespace AscendC;
constexpr int64_t NDDMA_DIM_NUM = 3;
constexpr uint16_t VALUE_TWO = 2;
constexpr uint16_t COMPACT_CHAIN_NUM = 2;
constexpr uint16_t VALUE_ONE = 1;
constexpr uint16_t VALUE_ZERO = 0;
constexpr uint32_t BLOCK_SIZE = 32;

constexpr float MAX_VALUE_FLOAT = INFINITY;
constexpr int32_t MAX_VALUE_INT32 = INT32_MAX;
constexpr int64_t MAX_VALUE_INT64 = INT64_MAX;
constexpr float MIN_VALUE_FLOAT = -INFINITY;
constexpr int32_t MIN_VALUE_INT32 = INT32_MIN;
constexpr int64_t MIN_VALUE_INT64 = INT64_MIN;

template <typename T1, typename T2, typename T3, bool isMin>
class ArgMaxWithValueBase {
public:
    __aicore__ inline ArgMaxWithValueBase(){};

protected:
    template <typename T4>
    __aicore__ inline T4 GetMaxOrMinValue()
    {
        T4 minValue = 0;
        // isMin时取最大值，isMax时取最小值
        if constexpr (IsSameType<T4, float>::value) {
            minValue = isMin ? MAX_VALUE_FLOAT : MIN_VALUE_FLOAT;
        } else if constexpr (IsSameType<T4, half>::value) {
            half MAX_VALUE_HALF = INFINITY;
            half MIN_VALUE_HALF = -INFINITY;
            minValue = isMin ? MAX_VALUE_HALF : MIN_VALUE_HALF;
        } else if constexpr (IsSameType<T4, int32_t>::value) {
            minValue = isMin ? MAX_VALUE_INT32 : MIN_VALUE_INT32;
        } else if constexpr (IsSameType<T4, int64_t>::value) {
            minValue = isMin ? MAX_VALUE_INT64 : MIN_VALUE_INT64;
        }
        return minValue;
    };

    __aicore__ inline uint16_t GetVLSize()
    {
        uint16_t vlSize = 1;
        if constexpr (IsSameType<T1, int64_t>::value) {
            vlSize = Ops::Base::GetVRegSize() / sizeof(T1) * VALUE_TWO;
        } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
            vlSize = Ops::Base::GetVRegSize() / sizeof(float);
        } else {
            vlSize = Ops::Base::GetVRegSize() / sizeof(T1);
        }
        return vlSize;
    };
    template <typename T4, typename T5, bool needAlign = false>
    __aicore__ inline void ArgMaxV1(__ubuf__ void* valueDst, __ubuf__ void* indexDst, __ubuf__ void* src, uint16_t dimA,
                                    uint16_t dimR, uint32_t startR = 0);
    template <typename T4, typename T5, typename T6, bool needAlign = false>
    __aicore__ inline void ArgMaxGatherV1(__ubuf__ void* valueDst, __ubuf__ void* indexDst, __ubuf__ void* src,
                                          uint16_t dimA, uint16_t dimR, uint16_t dimNextA, uint32_t startR = 0);
    template <typename T4, typename T5, typename T6, bool needAlign = false>
    __aicore__ inline void ArgMaxGatherV1ForRLessThanVL64(__ubuf__ void* valueDst, __ubuf__ void* indexDst,
                                                          __ubuf__ void* src, uint16_t dimA, uint16_t dimR,
                                                          uint16_t dimNextA, uint32_t startR = 0);
    template <typename T4, typename T5, typename T6>
    __aicore__ inline void ArgMaxGatherRa(__ubuf__ void* valueDst, __ubuf__ T2* indexDst, __ubuf__ void* src,
                                          uint16_t dimA, uint16_t dimR);
    template <typename T4, typename T5, bool isBfloat16Casted = true>
    __aicore__ inline void ArgMaxRa(__ubuf__ void* valueDst, __ubuf__ T2* indexDst, __ubuf__ void* src, uint16_t dimA,
                                    uint16_t alignA, uint16_t dimR, uint32_t startR = 0);
    template <typename T4, typename T5>
    __aicore__ inline void ArgMaxAra(__ubuf__ T4* valueDst, __ubuf__ T2* indexDst, __ubuf__ T4* src, uint16_t dimA0,
                                     uint16_t dimR, uint16_t dimA1, uint32_t startR);
    template <typename T4, typename T5>
    __aicore__ inline void ArgMaxAraInt64(__ubuf__ T4* valueDst, __ubuf__ T2* indexDst, __ubuf__ T4* src,
                                          uint16_t dimA0, uint16_t dimR, uint16_t dimA1, uint32_t startR);
    template <typename T4, typename T5>
    __aicore__ inline void ArgMaxAraUnAlign(__ubuf__ T4* valueDst, __ubuf__ T2* indexDst, __ubuf__ T4* src,
                                            uint16_t dimA0, uint16_t dimR, uint16_t dimA1, uint16_t dimA1Src,
                                            uint16_t dimA1Dst, uint32_t startR);
    template <typename T4, typename T5>
    __aicore__ inline void ArgMaxAraUnAlignCompact(__ubuf__ T4* valueDst, __ubuf__ T2* indexDst, __ubuf__ T4* src,
                                                   uint16_t dimA0, uint16_t dimR, uint16_t dimA1, uint16_t dimA1Src,
                                                   uint16_t dimA1Dst, uint32_t startR);
    template <typename T4, typename T5>
    __aicore__ inline void ArgMaxAraUnAlignInt64(__ubuf__ T4* valueDst, __ubuf__ T2* indexDst, __ubuf__ T4* src,
                                                 uint16_t dimA0, uint16_t dimR, uint16_t dimA1, uint16_t dimA1Src,
                                                 uint16_t dimA1Dst, uint32_t startR);
    template <typename T4, bool isBfloat16Casted = true>
    __aicore__ inline void UpdateResult(__ubuf__ void* srcIndice, __ubuf__ void* srcValue, __ubuf__ void* dstIndice,
                                        __ubuf__ void* dstValue, uint64_t num);
#if ORIG_DTYPE_X == DT_INT64
    template <bool needAlign = false>
    __aicore__ inline void ArgMaxV1Int64(__ubuf__ T1* valueDst, __ubuf__ T2* indexDst, __ubuf__ T1* src, uint16_t dimA,
                                         uint16_t dimR, uint32_t startR = 0);
    template <bool needAlign = false>
    __aicore__ inline void ArgMaxV1Int64ForRLessThanVL64(__ubuf__ T1* valueDst, __ubuf__ T2* indexDst, __ubuf__ T1* src,
                                                         uint16_t dimA, uint16_t dimR, uint32_t startR = 0);
    template <typename T4, typename T5>
    __aicore__ inline void ArgMaxRaInt64(__ubuf__ void* valueDst, __ubuf__ T2* indexDst, __ubuf__ void* src,
                                         uint16_t dimA, uint16_t alignA, uint16_t dimR, uint32_t startR = 0);
    template <typename T4, typename T5, typename T6>
    __aicore__ inline void ArgMaxGatherRaInt64(__ubuf__ void* valueDst, __ubuf__ T2* indexDst, __ubuf__ void* src,
                                               uint16_t dimA, uint16_t dimR);
    template <typename T4, typename T5, typename T6, bool needAlign = false>
    __aicore__ inline void ArgMaxGatherV1Int64(__ubuf__ void* valueDst, __ubuf__ void* indexDst, __ubuf__ void* src,
                                               uint16_t dimA, uint16_t dimR, uint16_t dimNextA, uint32_t startR = 0);
#endif
    __aicore__ inline void CastToBF16(LocalTensor<T1>& outValues, LocalTensor<float> outValues32, uint64_t copyNum);
    __aicore__ inline void CopyOutValues(GlobalTensor<T1>& valuesGm, LocalTensor<T1>& outValues, uint64_t offset,
                                         uint64_t copyNum);
    __aicore__ inline void CopyOutIndice(GlobalTensor<T3>& indiceGm, LocalTensor<T2>& outIndice, uint64_t offset,
                                         uint64_t copyNum, LocalTensor<T3>& castUb);
};

template <typename T1, typename T2, typename T3, bool isMin>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::CastToBF16(LocalTensor<T1>& outValues,
                                                                          LocalTensor<float> outValues32,
                                                                          uint64_t copyNum)
{
    Cast(outValues, outValues32, RoundMode::CAST_ROUND, copyNum);
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
}

template <typename T1, typename T2, typename T3, bool isMin>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::CopyOutValues(GlobalTensor<T1>& valuesGm,
                                                                             LocalTensor<T1>& outValues,
                                                                             uint64_t offset, uint64_t copyNum)
{
    DataCopyExtParams copyOutValuesParams{1, 0, 0, 0, 0};
    copyOutValuesParams.blockCount = 1;
    copyOutValuesParams.blockLen = copyNum * sizeof(T1);
    copyOutValuesParams.srcStride = 0;
    copyOutValuesParams.dstStride = 0;
    DataCopyPad(valuesGm[offset], outValues, copyOutValuesParams);
}

template <typename T1, typename T2, typename T3, bool isMin>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::CopyOutIndice(GlobalTensor<T3>& indiceGm,
                                                                             LocalTensor<T2>& outIndice,
                                                                             uint64_t offset, uint64_t copyNum,
                                                                             LocalTensor<T3>& castUb)
{
    DataCopyExtParams copyOutIndiceParams{1, 0, 0, 0, 0};
    copyOutIndiceParams.blockCount = 1;
    copyOutIndiceParams.blockLen = copyNum * sizeof(T3);
    copyOutIndiceParams.srcStride = 0;
    copyOutIndiceParams.dstStride = 0;
    if constexpr (IsSameType<T1, int64_t>::value && !IsSameType<T3, int64_t>::value) {
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Cast(castUb, outIndice, RoundMode::CAST_NONE, copyNum);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad(indiceGm[offset], castUb, copyOutIndiceParams);
    } else if constexpr (!IsSameType<T1, int64_t>::value && IsSameType<T3, int64_t>::value) {
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Cast(castUb, outIndice, RoundMode::CAST_NONE, copyNum);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad(indiceGm[offset], castUb, copyOutIndiceParams);
    } else {
        DataCopyPad(indiceGm[offset], outIndice, copyOutIndiceParams);
    }
}

#if ORIG_DTYPE_X == DT_INT64
template <typename T1, typename T2, typename T3, bool isMin>
template <bool needAlign>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxV1Int64(__ubuf__ T1* valueDst,
                                                                             __ubuf__ T2* indexDst, __ubuf__ T1* src,
                                                                             uint16_t dimA, uint16_t dimR,
                                                                             uint32_t startR)
{
    uint16_t VL = GetVLSize();
    if (dimR < VL) {
        ArgMaxV1Int64ForRLessThanVL64<needAlign>(valueDst, indexDst, src, dimA, dimR, startR);
        return;
    }
    uint16_t mainLoop = dimR / VL;
    uint32_t tailSize = dimR - mainLoop * VL;
    uint16_t loopTime = mainLoop - 1;
    uint16_t tailLoop = tailSize > 0 ? 1 : 0;

    uint32_t offset1 = VL;
    uint32_t offsetTail = tailSize;
    if constexpr (needAlign) {
        uint32_t alignNum = BLOCK_SIZE / sizeof(T1);
        offset1 = CeilDivision(offset1, alignNum) * alignNum;
        offsetTail = CeilDivision(offsetTail, alignNum) * alignNum;
    }
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T1, AscendC::Reg::RegTraitNumTwo> value0, value1, valueResult;
        AscendC::Reg::RegTensor<T2, AscendC::Reg::RegTraitNumTwo> level0, level1, indexResult, helpIndex,
            indexMulResult, indexAddResult;

        AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg tailMask = AscendC::Reg::UpdateMask<uint32_t>(tailSize);
        AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::MaskReg cmpMask;

        AscendC::Reg::Duplicate(helpIndex, VALUE_ONE, fullMask);
        vdup((AscendC::Reg::RegTensor<uint32_t>&)helpIndex.reg[0], VL, oneMask, MODE_MERGING);

        // 用于加法的寄存器
        AscendC::Reg::RegTensor<T2, AscendC::Reg::RegTraitNumTwo> vregIndicesAdd1, vregIndicesAddStartR;
        AscendC::Reg::Duplicate(vregIndicesAdd1, VALUE_ONE, fullMask);
        AscendC::Reg::Duplicate(vregIndicesAddStartR, startR, fullMask);

        // 主块, 载入i片数据，初始化i片索引
        AscendC::Reg::UnalignRegForLoad uValue, uIndex, uSrc;
        AscendC::Reg::LoadUnAlignPre(uSrc, (__ubuf__ T1*&)src);
        for (uint16_t a = 0; a < dimA; a++) {
            AscendC::Reg::LoadUnAlign<T1, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(value0, uSrc, (__ubuf__ T1*&)src,
                                                                                       offset1);
            AscendC::Reg::Duplicate(level0, VALUE_ZERO, fullMask);
            AscendC::Reg::Duplicate(level1, VALUE_ZERO, fullMask);
            // 载入i+1片数据，初始化i+1片索引，比较 i and i+1
            for (uint16_t r = 0; r < loopTime; r++) {
                AscendC::Reg::LoadUnAlign<T1, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(value1, uSrc,
                                                                                           (__ubuf__ T1*&)src, VL);
                // 微指令 AscendC::Reg::Duplicate(level1, r + 1, fullMask);
                Add(level1, level1, vregIndicesAdd1, fullMask);
                if constexpr (isMin) {
                    AscendC::Reg::Compare<T1, CMPMODE::LE>(cmpMask, value0, value1, fullMask);
                } else {
                    AscendC::Reg::Compare<T1, CMPMODE::GE>(cmpMask, value0, value1, fullMask);
                }
                AscendC::Reg::Select(value0, value0, value1, cmpMask);
                AscendC::Reg::Select(level0, level0, level1, cmpMask);
            }

            // 载入尾快，与主块比较
            for (uint16_t r = 0; r < tailLoop; r++) {
                AscendC::Reg::LoadUnAlign<T1, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                    value1, uSrc, (__ubuf__ T1*&)src, offsetTail);
                AscendC::Reg::Duplicate(level1, mainLoop, fullMask);
                if constexpr (isMin) {
                    AscendC::Reg::Compare<T1, CMPMODE::LT>(cmpMask, value1, value0, tailMask);
                } else {
                    AscendC::Reg::Compare<T1, CMPMODE::GT>(cmpMask, value1, value0, tailMask);
                }
                AscendC::Reg::Select(value0, value1, value0, cmpMask);
                AscendC::Reg::Select(level0, level1, level0, cmpMask);
            }

            // 输出最终值 + 相对索引
            if constexpr (isMin) {
                AscendC::Reg::Reduce<Reg::ReduceType::MIN>(valueResult, value0, fullMask);
            } else {
                AscendC::Reg::Reduce<Reg::ReduceType::MAX>(valueResult, value0, fullMask);
            }
            AscendC::Reg::StoreUnAlign<T1, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>((__ubuf__ T1*&)valueDst,
                                                                                        valueResult, uValue, 1);
            // 相对索引回溯绝对索引 idx = idx + level_Y[idx] * A
            AscendC::Reg::Duplicate(valueResult, valueResult, fullMask);
            AscendC::Reg::Compare<T1, CMPMODE::EQ>(cmpMask, valueResult, value0, fullMask);
            AscendC::Reg::Reduce<Reg::ReduceType::MIN>(indexResult, level0, cmpMask);
            AscendC::Reg::Mul(indexMulResult, helpIndex, indexResult, fullMask);
            AscendC::Reg::DeInterleave(indexResult, indexAddResult, indexMulResult, helpIndex);
            Add(indexResult, indexResult, indexAddResult, oneMask);
            Add(indexResult, indexResult, vregIndicesAddStartR, oneMask);
            AscendC::Reg::StoreUnAlign<T2, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ T2*&)indexDst, (AscendC::Reg::RegTensor<T2>&)indexResult, uIndex, 1);
        }
        // 非对齐搬出后处理
        AscendC::Reg::StoreUnAlignPost((__ubuf__ T1*&)valueDst, uValue, 0);
        AscendC::Reg::StoreUnAlignPost((__ubuf__ T2*&)indexDst, uIndex, 0);
    }
}

template <typename T1, typename T2, typename T3, bool isMin>
template <bool needAlign>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxV1Int64ForRLessThanVL64(
    __ubuf__ T1* valueDst, __ubuf__ T2* indexDst, __ubuf__ T1* src, uint16_t dimA, uint16_t dimR, uint32_t startR)
{
    uint16_t VL = GetVLSize();
    uint32_t offset1 = dimR;
    if constexpr (needAlign) {
        uint32_t alignNum = BLOCK_SIZE / sizeof(T1);
        offset1 = CeilDivision(offset1, alignNum) * alignNum;
    }
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T1, AscendC::Reg::RegTraitNumTwo> value0, valueResult;
        AscendC::Reg::RegTensor<T2, AscendC::Reg::RegTraitNumTwo> level0, indexResult, helpIndex, indexMulResult,
            indexAddResult;

        uint32_t tailSize = dimR;
        AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg tailMask = AscendC::Reg::UpdateMask<uint32_t>(tailSize);
        AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::MaskReg cmpMask;

        AscendC::Reg::Duplicate(helpIndex, VALUE_ONE, fullMask);
        vdup((AscendC::Reg::RegTensor<uint32_t>&)helpIndex.reg[0], VL, oneMask, MODE_MERGING);

        // 用于加法的寄存器
        AscendC::Reg::RegTensor<T2, AscendC::Reg::RegTraitNumTwo> vregIndicesAdd1, vregIndicesAddStartR;
        AscendC::Reg::Duplicate(vregIndicesAdd1, 1, fullMask);
        AscendC::Reg::Duplicate(vregIndicesAddStartR, startR, fullMask);

        // 主块, 载入i片数据，初始化i片索引
        AscendC::Reg::UnalignRegForLoad uValue, uIndex, uSrc;
        AscendC::Reg::LoadUnAlignPre(uSrc, (__ubuf__ T1*&)src);
        for (uint16_t a = 0; a < dimA; a++) {
            AscendC::Reg::LoadUnAlign<T1, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(value0, uSrc, (__ubuf__ T1*&)src,
                                                                                       offset1);
            vdup((AscendC::Reg::RegTensor<uint32_t>&)level0.reg[0], 0, tailMask, MODE_ZEROING);
            vdup((AscendC::Reg::RegTensor<uint32_t>&)level0.reg[1], 0, tailMask, MODE_ZEROING);

            // 输出最终值 + 相对索引
            if constexpr (isMin) {
                AscendC::Reg::Reduce<Reg::ReduceType::MIN>(valueResult, value0, tailMask);
            } else {
                AscendC::Reg::Reduce<Reg::ReduceType::MAX>(valueResult, value0, tailMask);
            }
            AscendC::Reg::StoreUnAlign<T1, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>((__ubuf__ T1*&)valueDst,
                                                                                        valueResult, uValue, 1);
            // 相对索引回溯绝对索引 idx = idx + level_Y[idx] * A
            AscendC::Reg::Duplicate(valueResult, valueResult, tailMask);
            AscendC::Reg::Compare<T1, CMPMODE::EQ>(cmpMask, valueResult, value0, tailMask);
            AscendC::Reg::Reduce<Reg::ReduceType::MIN>(indexResult, level0, cmpMask);
            AscendC::Reg::Mul(indexMulResult, helpIndex, indexResult, fullMask);
            AscendC::Reg::DeInterleave(indexResult, indexAddResult, indexMulResult, helpIndex);
            Add(indexResult, indexResult, indexAddResult, oneMask);
            Add(indexResult, indexResult, vregIndicesAddStartR, oneMask);
            AscendC::Reg::StoreUnAlign<T2, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ T2*&)indexDst, (AscendC::Reg::RegTensor<T2>&)indexResult, uIndex, 1);
        }
        // 非对齐搬出后处理
        AscendC::Reg::StoreUnAlignPost((__ubuf__ T1*&)valueDst, uValue, 0);
        AscendC::Reg::StoreUnAlignPost((__ubuf__ T2*&)indexDst, uIndex, 0);
    }
}

// RA场景VF接口，R < VL && A >= coreNum * VL
template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxRaInt64(__ubuf__ void* valueDst,
                                                                             __ubuf__ T2* indexDst, __ubuf__ void* src,
                                                                             uint16_t dimA, uint16_t alignA,
                                                                             uint16_t dimR, uint32_t startR)
{
    uint16_t VL = GetVLSize();
    // 向上对齐算上尾块次数
    uint16_t loopA = CeilDivision(dimA, VL);
    uint32_t processA = dimA;
    uint32_t sregMask = dimA;
    uint16_t loopR = dimR > 1 ? (dimR - 1) : 0;
    uint16_t offsetA = dimA >= VL ? VL : dimA;

    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<T4, AscendC::Reg::RegTraitNumTwo> vregValue0, vregValue1;
        AscendC::Reg::RegTensor<T5, AscendC::Reg::RegTraitNumTwo> vregIndices0, vregIndices1;

        AscendC::Reg::RegTensor<T5, AscendC::Reg::RegTraitNumTwo> vregIndicesAdd1, vregIndicesAddStartR;
        AscendC::Reg::Duplicate(vregIndicesAdd1, VALUE_ONE, fullMask);
        AscendC::Reg::Duplicate(vregIndicesAddStartR, startR, fullMask);

        AscendC::Reg::UnalignRegForLoad uSrc, tSrc;

        AscendC::Reg::MaskReg mask, cmpMask;

        for (uint16_t i = 0; i < loopA; i++) {
            auto tmpPtr = (__ubuf__ T4*&)src + alignA;
            mask = AscendC::Reg::UpdateMask<uint32_t>(processA);
            AscendC::Reg::LoadUnAlignPre(uSrc, (__ubuf__ T4*&)src);
            AscendC::Reg::LoadUnAlign<T4, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vregValue0, uSrc,
                                                                                       (__ubuf__ T4*&)src, offsetA);
            AscendC::Reg::Duplicate(vregIndices0, VALUE_ZERO, fullMask);
            AscendC::Reg::Duplicate(vregIndices1, VALUE_ZERO, fullMask);

            // 从第二行的R开始计算比较
            for (uint16_t j = 0; j < loopR; j++) {
                Add(vregIndices1, vregIndices1, vregIndicesAdd1, mask);
                AscendC::Reg::LoadUnAlignPre(tSrc, (__ubuf__ T4*&)tmpPtr);
                AscendC::Reg::LoadUnAlign<T4, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vregValue1, tSrc,
                                                                                           (__ubuf__ T4*&)tmpPtr, VL);
                tmpPtr += (alignA - VL);
                if constexpr (isMin) {
                    AscendC::Reg::Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, mask);
                } else {
                    AscendC::Reg::Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, mask);
                }
                AscendC::Reg::Select(vregValue0, vregValue0, vregValue1, cmpMask);
                AscendC::Reg::Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
            }

            MicroAPI::StoreAlign((__ubuf__ T4*)valueDst + i * VL, vregValue0, mask);

            Add(vregIndices0, vregIndices0, vregIndicesAddStartR, mask);
            MicroAPI::StoreAlign((__ubuf__ T5*)indexDst + i * VL, vregIndices0, mask);
        }
    }
}

// AR场景Gather接口，R < VL && A >= coreNum * VL
template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5, typename T6>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxGatherRaInt64(__ubuf__ void* valueDst,
                                                                                   __ubuf__ T2* indexDst,
                                                                                   __ubuf__ void* src, uint16_t dimA,
                                                                                   uint16_t dimR)
{
    uint16_t VL = GetVLSize();
    // 向上对齐算上尾块次数
    uint16_t loopA = CeilDivision(dimA, VL);
    uint32_t processA = dimA;
    uint16_t loopR = dimR > 1 ? (dimR - 1) : 0;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T4, AscendC::Reg::RegTraitNumTwo> vregValue0, vregValue1;
        AscendC::Reg::RegTensor<T5, AscendC::Reg::RegTraitNumTwo> vregIndices0, vregIndices1;
        AscendC::Reg::RegTensor<int64_t, AscendC::Reg::RegTraitNumTwo> idx, idxInit;
        AscendC::Reg::MaskReg mask, cmpMask;
        mask = AscendC::Reg::UpdateMask<uint32_t>(processA);

        for (uint16_t i = 0; i < loopA; i++) {
            // Gather实现UB内转置
            uint32_t startIdx = i * VL;
            // 构造起始为 i * VL的1递增数列实现AR转RA
            Arange(idxInit, startIdx);
            Muls(idxInit, idxInit, dimR, mask);
            // Gather的偏移量索引要用uint类型
            MicroAPI::Gather(vregValue0, (__ubuf__ T4*&)src, (AscendC::Reg::RegTensor<uint32_t>&)idxInit, mask);
            // 初始化索引
            Duplicate(vregIndices0, 0);
            for (uint16_t j = 0; j < loopR; j++) {
                Duplicate(vregIndices1, j + 1, mask);
                // 更新下VL个数的索引
                Adds(idx, idxInit, (j + 1), mask);
                MicroAPI::Gather(vregValue1, (__ubuf__ T4*&)src, (AscendC::Reg::RegTensor<uint32_t>&)idx, mask);
                if constexpr (isMin) {
                    Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, mask);
                } else {
                    Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, mask);
                }
                Select(vregValue0, vregValue0, vregValue1, cmpMask);
                Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
            }
            MicroAPI::StoreAlign((__ubuf__ T4*&)valueDst + i * VL, vregValue0, mask);
            MicroAPI::StoreAlign((__ubuf__ T5*&)indexDst + i * VL, vregIndices0, mask);
        }
    }
}

template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5, typename T6, bool needAlign>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxGatherV1Int64(__ubuf__ void* valueDst,
                                                                                   __ubuf__ void* indexDst,
                                                                                   __ubuf__ void* src, uint16_t dimA,
                                                                                   uint16_t dimR, uint16_t dimNextA,
                                                                                   uint32_t startR)
{
    uint16_t VL = GetVLSize();
    uint16_t mainLoop = dimR / VL;
    uint16_t tailSize = dimR - mainLoop * VL;
    uint16_t loopTime = mainLoop > 1 ? mainLoop - 1 : 0;
    uint16_t tailLoop = (tailSize > 0 && mainLoop > 0) ? 1 : 0;
    uint32_t sreg1 = 1;
    uint32_t sreg2 = 2;
    uint32_t sregAdd = 1;
    uint32_t rNextA = dimR * dimNextA;
    if constexpr (needAlign) {
        uint32_t alignNum = BLOCK_SIZE / sizeof(T1);
        rNextA = CeilDivision(rNextA, alignNum) * alignNum;
    }
    uint32_t offsetR = VL * dimNextA;
    uint32_t sregTail = dimR >= VL ? VL : dimR;
    uint32_t seg3 = tailSize;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T6, AscendC::Reg::RegTraitNumTwo> initIndex, aIndex, rIndex, nextAIndex;
        AscendC::Reg::RegTensor<T4, AscendC::Reg::RegTraitNumTwo> value0, value1, valueResult;
        AscendC::Reg::RegTensor<T6, AscendC::Reg::RegTraitNumTwo> indexResult32, level0, level1, indexResult, helpIndex;
        AscendC::Reg::UnalignRegForLoad uValue, uIndex;

        AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::MaskReg allMask32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg newMask = AscendC::Reg::UpdateMask<uint32_t>(seg3);
        AscendC::Reg::MaskReg fullMask = AscendC::Reg::UpdateMask<uint32_t>(sregTail);
        AscendC::Reg::MaskReg cmpMask;

        Duplicate(helpIndex, 1, fullMask);
        Duplicate<T6, AscendC::Reg::MaskMergeMode::MERGING, T6>(helpIndex, VL, oneMask);
        Arange(initIndex, 0);                            // (0, 1, 2, ... , 63)
        Muls(initIndex, initIndex, dimNextA, allMask32); // (0, 1 *dimNextA, 2*dimNextA, ... , 63*dimNextA)
        for (uint16_t a1 = 0; a1 < dimA; a1++) {
            // (0, 1 *dimNextA, 2*dimNextA, ... , 63*dimNextA) + a * rNextA 只修改起始位置
            Adds(aIndex, initIndex, a1 * rNextA, allMask32);
            for (uint16_t a2 = 0; a2 < dimNextA; a2++) {
                Adds(nextAIndex, aIndex, a2, allMask32);
                MicroAPI::Gather(value0, (__ubuf__ T4*&)src, (AscendC::Reg::RegTensor<uint32_t>&)nextAIndex, fullMask);
                Duplicate(level0, 0, fullMask);
                // 载入i+1片数据，初始化i+1片索引，比较 i and i+1
                for (uint16_t r = 0; r < loopTime; r++) {
                    Adds(rIndex, nextAIndex, (r + 1) * offsetR, allMask32);
                    MicroAPI::Gather(value1, (__ubuf__ T4*&)src, (AscendC::Reg::RegTensor<uint32_t>&)rIndex, fullMask);
                    Duplicate(level1, r + 1, fullMask);
                    // NAN
                    if constexpr (isMin) {
                        AscendC::Reg::Compare<T4, CMPMODE::LE>(cmpMask, value0, value1, fullMask);
                    } else {
                        AscendC::Reg::Compare<T4, CMPMODE::GE>(cmpMask, value0, value1, fullMask);
                    }
                    Select(value0, value0, value1, cmpMask);
                    Select(level0, level0, level1, cmpMask);
                }
                // 载入尾块，与主块比较
                for (uint16_t r = 0; r < tailLoop; r++) {
                    Adds(rIndex, nextAIndex, mainLoop * offsetR, allMask32);
                    MicroAPI::Gather(value1, (__ubuf__ T4*&)src, (AscendC::Reg::RegTensor<uint32_t>&)rIndex, newMask);
                    Duplicate(level1, mainLoop, fullMask);
                    // NAN
                    if constexpr (isMin) {
                        AscendC::Reg::Compare<T4, CMPMODE::LT>(cmpMask, value1, value0, newMask);
                    } else {
                        AscendC::Reg::Compare<T4, CMPMODE::GT>(cmpMask, value1, value0, newMask);
                    }
                    Select(value0, value1, value0, cmpMask);
                    Select(level0, level1, level0, cmpMask);
                }
                // 输出最终值 + 相对索引
                if constexpr (isMin) {
                    MicroAPI::Reduce<MicroAPI::ReduceType::MIN>(valueResult, value0, fullMask);
                } else {
                    MicroAPI::Reduce<MicroAPI::ReduceType::MAX>(valueResult, value0, fullMask);
                }
                AscendC::Reg::StoreUnAlign<T4, Reg::PostLiteral::POST_MODE_UPDATE>((__ubuf__ T4*&)valueDst, valueResult,
                                                                                   uValue, 1);
                // 相对索引回溯绝对索引 idx = idx + level_Y[idx] * A
                Duplicate(valueResult, valueResult, fullMask);
                AscendC::Reg::Compare<T4, CMPMODE::EQ>(cmpMask, valueResult, value0, fullMask);
                MicroAPI::Reduce<MicroAPI::ReduceType::MIN>(indexResult, level0, cmpMask);
                Mul(indexResult, helpIndex, indexResult, allMask32);
                AscendC::Reg::DeInterleave(indexResult, indexResult32, indexResult, helpIndex);
                Add(indexResult, indexResult, indexResult32, oneMask);
                Adds(indexResult32, indexResult, startR, oneMask);
                AscendC::Reg::StoreUnAlign<T2, Reg::PostLiteral::POST_MODE_UPDATE>(
                    (__ubuf__ T2*&)indexDst, (AscendC::Reg::RegTensor<T6>&)indexResult32, uIndex, 1);
            }
        }
        AscendC::Reg::StoreUnAlignPost((__ubuf__ T4*&)valueDst, uValue, 0);
        AscendC::Reg::StoreUnAlignPost((__ubuf__ T2*&)indexDst, uIndex, 0);
    }
}
#else
template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5, bool needAlign>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxV1(__ubuf__ void* valueDst,
                                                                        __ubuf__ void* indexDst, __ubuf__ void* src,
                                                                        uint16_t dimA, uint16_t dimR, uint32_t startR)
{
    uint16_t VL = GetVLSize();
    uint16_t mainLoop = dimR / VL;
    uint16_t tailSize = dimR - mainLoop * VL;
    uint16_t loopTime = mainLoop > 1 ? mainLoop - 1 : 0;
    uint16_t tailLoop = (tailSize > 0 && mainLoop > 0) ? 1 : 0;
    uint32_t offset1 = dimR >= VL ? VL : dimR;
    uint32_t sreg0 = tailSize;
    uint32_t sreg1 = 1;
    uint32_t sreg2 = 2;
    uint32_t sreg3 = offset1;
    uint32_t sregAdd = 1;
    uint32_t offsetTail = tailSize;
    if constexpr (needAlign) {
        uint32_t alignNum = BLOCK_SIZE / sizeof(T1);
        offset1 = CeilDivision(offset1, alignNum) * alignNum;
        offsetTail = CeilDivision(offsetTail, alignNum) * alignNum;
    }
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg helpV0Mask, helpV1Mask, b8Mask, fullMask, tailMask, cmpMask, oneMask, twoMask;
        AscendC::Reg::RegTensor<T4> value0, value1, valueResult;
        AscendC::Reg::RegTensor<uint32_t> indexResult32;
        AscendC::Reg::UnalignRegForLoad uValue, uIndex, uSrc;
        b8Mask = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg addMask = AscendC::Reg::UpdateMask<uint32_t>(sregAdd);

        AscendC::Reg::RegTensor<T5> level0, level1, indexResult, helpIndex;
        fullMask = AscendC::Reg::UpdateMask<T5>(sreg3);
        oneMask = AscendC::Reg::UpdateMask<T5>(sreg1);
        twoMask = AscendC::Reg::UpdateMask<T5>(sreg2);
        tailMask = AscendC::Reg::UpdateMask<T5>(sreg0);

        Duplicate(helpIndex, VALUE_ONE, fullMask);
        Duplicate<T5, AscendC::Reg::MaskMergeMode::MERGING, T5>(helpIndex, VL, oneMask);
        AscendC::Reg::LoadUnAlignPre(uSrc, (__ubuf__ T4*&)src);
        for (uint16_t a = 0; a < dimA; a++) {
            AscendC::Reg::LoadUnAlign<T4, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(value0, uSrc, (__ubuf__ T4*&)src,
                                                                                       offset1);
            Duplicate(level0, 0, fullMask);
            // 载入i+1片数据，初始化i+1片索引，比较 i and i+1
            for (uint16_t r = 0; r < loopTime; r++) {
                AscendC::Reg::LoadUnAlign<T4, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(value1, uSrc,
                                                                                           (__ubuf__ T4*&)src, VL);
                Duplicate(level1, r + 1, fullMask);
                // NAN
                if constexpr (isMin) {
                    AscendC::Reg::Compare<T4, CMPMODE::LE>(cmpMask, value0, value1, fullMask);
                } else {
                    AscendC::Reg::Compare<T4, CMPMODE::GE>(cmpMask, value0, value1, fullMask);
                }
                AscendC::Reg::Compare<T4, CMPMODE::NE>(helpV0Mask, value0, value0, fullMask);
                AscendC::Reg::Or(cmpMask, cmpMask, helpV0Mask, b8Mask);
                Select(value0, value0, value1, cmpMask);
                Select(level0, level0, level1, cmpMask);
            }
            // 载入尾块，与主块比较
            for (uint16_t r = 0; r < tailLoop; r++) {
                AscendC::Reg::LoadUnAlign<T4, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                    value1, uSrc, (__ubuf__ T4*&)src, offsetTail);
                Duplicate(level1, mainLoop, fullMask);
                // NAN
                if constexpr (isMin) {
                    AscendC::Reg::Compare<T4, CMPMODE::LT>(cmpMask, value1, value0, tailMask);
                } else {
                    AscendC::Reg::Compare<T4, CMPMODE::GT>(cmpMask, value1, value0, tailMask);
                }
                AscendC::Reg::Compare<T4, CMPMODE::NE>(helpV0Mask, value0, value0, fullMask);
                AscendC::Reg::Compare<T4, CMPMODE::NE>(helpV1Mask, value1, value1, tailMask);
                AscendC::Reg::Not(helpV0Mask, helpV0Mask, b8Mask);
                AscendC::Reg::And(helpV0Mask, helpV0Mask, helpV1Mask, b8Mask);
                AscendC::Reg::Or(cmpMask, cmpMask, helpV0Mask, b8Mask);
                Select(value0, value1, value0, cmpMask);
                Select(level0, level1, level0, cmpMask);
            }
            // 输出最终值 + 相对索引
            if constexpr (isMin) {
                MicroAPI::Reduce<MicroAPI::ReduceType::MIN>(valueResult, value0, fullMask);
            } else {
                MicroAPI::Reduce<MicroAPI::ReduceType::MAX>(valueResult, value0, fullMask);
            }
            AscendC::Reg::StoreUnAlign<T4, Reg::PostLiteral::POST_MODE_UPDATE>((__ubuf__ T4*&)valueDst, valueResult,
                                                                               uValue, 1);
            // 相对索引回溯绝对索引 idx = idx + level_Y[idx] * A
            Duplicate<T4, AscendC::Reg::HighLowPart::LOWEST, AscendC::Reg::MaskMergeMode::ZEROING>(
                valueResult, valueResult, fullMask);
            AscendC::Reg::Compare<T4, CMPMODE::EQ>(cmpMask, valueResult, value0, fullMask);
            // NAN
            AscendC::Reg::Compare<T4, CMPMODE::NE>(helpV1Mask, value0, value0, fullMask);
            AscendC::Reg::Or(cmpMask, cmpMask, helpV1Mask, b8Mask);
            MicroAPI::Reduce<MicroAPI::ReduceType::MIN>(indexResult, level0, cmpMask);
            Mul(indexResult, helpIndex, indexResult, twoMask);
            MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(indexResult32, indexResult, twoMask);
            Adds(indexResult32, indexResult32, startR, addMask);
            AscendC::Reg::StoreUnAlign<T2, Reg::PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ T2*&)indexDst, (AscendC::Reg::RegTensor<int32_t>&)indexResult32, uIndex, 1);
        }
        AscendC::Reg::StoreUnAlignPost((__ubuf__ T4*&)valueDst, uValue, 0);
        AscendC::Reg::StoreUnAlignPost((__ubuf__ T2*&)indexDst, uIndex, 0);
    }
}

template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5, typename T6, bool needAlign>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxGatherV1(__ubuf__ void* valueDst,
                                                                              __ubuf__ void* indexDst,
                                                                              __ubuf__ void* src, uint16_t dimA,
                                                                              uint16_t dimR, uint16_t dimNextA,
                                                                              uint32_t startR)
{
    uint16_t VL = GetVLSize();
    if (dimR < VL) {
        ArgMaxGatherV1ForRLessThanVL64<T4, T5, T6, needAlign>(valueDst, indexDst, src, dimA, dimR, dimNextA, startR);
        return;
    }
    uint16_t mainLoop = dimR / VL;
    uint32_t tailSize = dimR - mainLoop * VL;
    uint16_t loopTime = mainLoop - 1;
    uint16_t tailLoop = tailSize > 0 ? 1 : 0;
    uint32_t rNextA = dimR * dimNextA;
    if constexpr (needAlign) {
        uint32_t alignNum = BLOCK_SIZE / sizeof(T1);
        rNextA = CeilDivision(rNextA, alignNum) * alignNum;
    }
    uint32_t offsetR = VL * dimNextA;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T6> initIndex, aIndex, rIndex, nextAIndex;
        AscendC::Reg::RegTensor<T4> value0, value1, valueResult;
        AscendC::Reg::RegTensor<uint32_t> indexResult32;
        AscendC::Reg::UnalignRegForLoad uValue, uIndex, uSrc;
        AscendC::Reg::RegTensor<T5> level0, level1, indexResult, helpIndex;

        AscendC::Reg::MaskReg helpV0Mask, helpV1Mask, cmpMask;
        AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<T5, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::MaskReg twoMask = AscendC::Reg::CreateMask<T5, AscendC::Reg::MaskPattern::VL2>();
        AscendC::Reg::MaskReg allMask32 = AscendC::Reg::CreateMask<T5, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg newMask = AscendC::Reg::UpdateMask<T5>(tailSize);
        AscendC::Reg::MaskReg addMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::MaskReg b8Mask = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();

        Duplicate(helpIndex, VALUE_ONE, allMask32);
        Duplicate<T5, AscendC::Reg::MaskMergeMode::MERGING, T5>(helpIndex, VL, oneMask);
        Arange(initIndex, 0);                            // (0, 1, 2, ... , 63)
        Muls(initIndex, initIndex, dimNextA, allMask32); // (0, 1 *dimNextA, 2*dimNextA, ... , 63*dimNextA)
        for (uint16_t a1 = 0; a1 < dimA; a1++) {
            // (0, 1 *dimNextA, 2*dimNextA, ... , 63*dimNextA) + a * rNextA 只修改起始位置
            Adds(aIndex, initIndex, a1 * rNextA, allMask32);
            for (uint16_t a2 = 0; a2 < dimNextA; a2++) {
                Adds(nextAIndex, aIndex, a2, allMask32);
                MicroAPI::Gather(value0, (__ubuf__ T4*&)src, (AscendC::Reg::RegTensor<T5>&)nextAIndex, allMask32);
                Duplicate(level0, VALUE_ZERO, allMask32);
                // 载入i+1片数据，初始化i+1片索引，比较 i and i+1
                for (uint16_t r = 0; r < loopTime; r++) {
                    Adds(rIndex, nextAIndex, (r + 1) * offsetR, allMask32);
                    MicroAPI::Gather(value1, (__ubuf__ T4*&)src, (AscendC::Reg::RegTensor<T5>&)rIndex, allMask32);
                    Duplicate(level1, r + 1, allMask32);
                    // NAN
                    if constexpr (isMin) {
                        AscendC::Reg::Compare<T4, CMPMODE::LE>(cmpMask, value0, value1, allMask32);
                    } else {
                        AscendC::Reg::Compare<T4, CMPMODE::GE>(cmpMask, value0, value1, allMask32);
                    }
                    AscendC::Reg::Compare<T4, CMPMODE::NE>(helpV0Mask, value0, value0, allMask32);
                    AscendC::Reg::Or(cmpMask, cmpMask, helpV0Mask, b8Mask);
                    Select(value0, value0, value1, cmpMask);
                    Select(level0, level0, level1, cmpMask);
                }
                // 载入尾块，与主块比较
                for (uint16_t r = 0; r < tailLoop; r++) {
                    Adds(rIndex, nextAIndex, mainLoop * offsetR, allMask32);
                    MicroAPI::Gather(value1, (__ubuf__ T4*&)src, (AscendC::Reg::RegTensor<T5>&)rIndex, newMask);
                    Duplicate(level1, mainLoop, allMask32);
                    // NAN
                    if constexpr (isMin) {
                        AscendC::Reg::Compare<T4, CMPMODE::LT>(cmpMask, value1, value0, newMask);
                    } else {
                        AscendC::Reg::Compare<T4, CMPMODE::GT>(cmpMask, value1, value0, newMask);
                    }
                    AscendC::Reg::Compare<T4, CMPMODE::NE>(helpV0Mask, value0, value0, allMask32);
                    AscendC::Reg::Compare<T4, CMPMODE::NE>(helpV1Mask, value1, value1, newMask);
                    AscendC::Reg::Not(helpV0Mask, helpV0Mask, b8Mask);
                    AscendC::Reg::And(helpV0Mask, helpV0Mask, helpV1Mask, b8Mask);
                    AscendC::Reg::Or(cmpMask, cmpMask, helpV0Mask, b8Mask);
                    Select(value0, value1, value0, cmpMask);
                    Select(level0, level1, level0, cmpMask);
                }
                // 输出最终值 + 相对索引
                if constexpr (isMin) {
                    MicroAPI::Reduce<MicroAPI::ReduceType::MIN>(valueResult, value0, allMask32);
                } else {
                    MicroAPI::Reduce<MicroAPI::ReduceType::MAX>(valueResult, value0, allMask32);
                }
                AscendC::Reg::StoreUnAlign<T4, Reg::PostLiteral::POST_MODE_UPDATE>((__ubuf__ T4*&)valueDst, valueResult,
                                                                                   uValue, 1);
                // 相对索引回溯绝对索引 idx = idx + level_Y[idx] * A
                Duplicate<T4, AscendC::Reg::HighLowPart::LOWEST, AscendC::Reg::MaskMergeMode::ZEROING>(
                    valueResult, valueResult, allMask32);
                AscendC::Reg::Compare<T4, CMPMODE::EQ>(cmpMask, valueResult, value0, allMask32);
                // NAN
                AscendC::Reg::Compare<T4, CMPMODE::NE>(helpV1Mask, value0, value0, allMask32);
                AscendC::Reg::Or(cmpMask, cmpMask, helpV1Mask, b8Mask);
                MicroAPI::Reduce<MicroAPI::ReduceType::MIN>(indexResult, level0, cmpMask);
                Mul(indexResult, helpIndex, indexResult, twoMask);
                MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(indexResult32, indexResult, twoMask);
                Adds(indexResult32, indexResult32, startR, addMask);
                AscendC::Reg::StoreUnAlign<T2, Reg::PostLiteral::POST_MODE_UPDATE>(
                    (__ubuf__ T2*&)indexDst, (AscendC::Reg::RegTensor<int32_t>&)indexResult32, uIndex, 1);
            }
        }
        AscendC::Reg::StoreUnAlignPost((__ubuf__ T4*&)valueDst, uValue, 0);
        AscendC::Reg::StoreUnAlignPost((__ubuf__ T2*&)indexDst, uIndex, 0);
    }
}

template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5, typename T6, bool needAlign>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxGatherV1ForRLessThanVL64(
    __ubuf__ void* valueDst, __ubuf__ void* indexDst, __ubuf__ void* src, uint16_t dimA, uint16_t dimR,
    uint16_t dimNextA, uint32_t startR)
{
    uint16_t VL = GetVLSize();
    uint32_t tailSize = dimR;
    uint32_t rNextA = dimR * dimNextA;
    if constexpr (needAlign) {
        uint32_t alignNum = BLOCK_SIZE / sizeof(T1);
        rNextA = CeilDivision(rNextA, alignNum) * alignNum;
    }
    uint32_t offsetR = VL * dimNextA;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T6> initIndex, aIndex, rIndex, nextAIndex;
        AscendC::Reg::RegTensor<T4> value0, value1, valueResult;
        AscendC::Reg::RegTensor<uint32_t> indexResult32;
        AscendC::Reg::UnalignRegForLoad uValue, uIndex, uSrc;
        AscendC::Reg::RegTensor<T5> level0, level1, indexResult, helpIndex;

        AscendC::Reg::MaskReg helpV1Mask, cmpMask;
        AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<T5, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::MaskReg twoMask = AscendC::Reg::CreateMask<T5, AscendC::Reg::MaskPattern::VL2>();
        AscendC::Reg::MaskReg newMask = AscendC::Reg::UpdateMask<T5>(tailSize);
        AscendC::Reg::MaskReg allMask32 = AscendC::Reg::CreateMask<T5, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg addMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::MaskReg b8Mask = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();

        Duplicate(helpIndex, VALUE_ONE, newMask);
        Duplicate<T5, AscendC::Reg::MaskMergeMode::MERGING, T5>(helpIndex, VL, oneMask);
        Arange(initIndex, 0);                            // (0, 1, 2, ... , 63)
        Muls(initIndex, initIndex, dimNextA, allMask32); // (0, 1 *dimNextA, 2*dimNextA, ... , 63*dimNextA)
        for (uint16_t a1 = 0; a1 < dimA; a1++) {
            // (0, 1 *dimNextA, 2*dimNextA, ... , 63*dimNextA) + a * rNextA 只修改起始位置
            Adds(aIndex, initIndex, a1 * rNextA, allMask32);
            for (uint16_t a2 = 0; a2 < dimNextA; a2++) {
                Adds(nextAIndex, aIndex, a2, allMask32);
                MicroAPI::Gather(value0, (__ubuf__ T4*&)src, (AscendC::Reg::RegTensor<T5>&)nextAIndex, newMask);
                Duplicate(level0, VALUE_ZERO, newMask);
                // 输出最终值 + 相对索引
                if constexpr (isMin) {
                    MicroAPI::Reduce<MicroAPI::ReduceType::MIN>(valueResult, value0, newMask);
                } else {
                    MicroAPI::Reduce<MicroAPI::ReduceType::MAX>(valueResult, value0, newMask);
                }
                AscendC::Reg::StoreUnAlign<T4, Reg::PostLiteral::POST_MODE_UPDATE>((__ubuf__ T4*&)valueDst, valueResult,
                                                                                   uValue, 1);
                // 相对索引回溯绝对索引 idx = idx + level_Y[idx] * A
                Duplicate<T4, AscendC::Reg::HighLowPart::LOWEST, AscendC::Reg::MaskMergeMode::ZEROING>(
                    valueResult, valueResult, newMask);
                AscendC::Reg::Compare<T4, CMPMODE::EQ>(cmpMask, valueResult, value0, newMask);
                // NAN
                AscendC::Reg::Compare<T4, CMPMODE::NE>(helpV1Mask, value0, value0, newMask);
                AscendC::Reg::Or(cmpMask, cmpMask, helpV1Mask, b8Mask);
                MicroAPI::Reduce<MicroAPI::ReduceType::MIN>(indexResult, level0, cmpMask);
                Mul(indexResult, helpIndex, indexResult, twoMask);
                MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(indexResult32, indexResult, twoMask);
                Adds(indexResult32, indexResult32, startR, addMask);
                AscendC::Reg::StoreUnAlign<T2, Reg::PostLiteral::POST_MODE_UPDATE>(
                    (__ubuf__ T2*&)indexDst, (AscendC::Reg::RegTensor<int32_t>&)indexResult32, uIndex, 1);
            }
        }
        AscendC::Reg::StoreUnAlignPost((__ubuf__ T4*&)valueDst, uValue, 0);
        AscendC::Reg::StoreUnAlignPost((__ubuf__ T2*&)indexDst, uIndex, 0);
    }
}

#endif

// AR场景Gather接口，R < VL && A >= cornNum * VL
template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5, typename T6>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxGatherRa(__ubuf__ void* valueDst,
                                                                              __ubuf__ T2* indexDst, __ubuf__ void* src,
                                                                              uint16_t dimA, uint16_t dimR)
{
    uint16_t VL = GetVLSize();
    // 向上对齐算上尾块次数
    uint16_t loopA = CeilDivision(dimA, VL);
    uint32_t processA = dimA;
    uint32_t sregMask = dimA;
    uint16_t loopR = dimR > 1 ? (dimR - 1) : 0;
    uint32_t repeatElmB32 = Ops::Base::GetVRegSize() / sizeof(int32_t);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T4> vregValue0, vregValue1;
        AscendC::Reg::RegTensor<T5> vregIndices0, vregIndices1;
        AscendC::Reg::RegTensor<uint32_t> indicesLower, indicesHigher;
        AscendC::Reg::RegTensor<T6> idx, idxInit;
        AscendC::Reg::MaskReg mask, maskLower, maskHigher, cmpMask, nanMask;

        for (uint16_t i = 0; i < loopA; i++) {
            // Gather实现UB内转置
            mask = AscendC::Reg::UpdateMask<T5>(processA);
            T6 startIdx = i * VL;
            // 构造起始为i * VL的1递增数列实现AR转RA
            Arange(idxInit, startIdx);
            Muls(idxInit, idxInit, dimR, mask);
            // Gather的偏移量索引要用uint类型
            MicroAPI::Gather(vregValue0, (__ubuf__ T4*&)src, (AscendC::Reg::RegTensor<T5>&)idxInit, mask);
            // 初始化索引
            Duplicate(vregIndices0, 0);
            for (uint16_t j = 0; j < loopR; j++) {
                Duplicate(vregIndices1, j + 1, mask);
                // 更新下VL个数的索引
                Adds(idx, idxInit, (j + 1), mask);
                MicroAPI::Gather(vregValue1, (__ubuf__ T4*&)src, (AscendC::Reg::RegTensor<T5>&)idx, mask);
                if constexpr (isMin) {
                    Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, mask);
                } else {
                    Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, mask);
                }
                Compare<T4, CMPMODE::NE>(nanMask, vregValue0, vregValue0, mask);
                AscendC::Reg::Or(cmpMask, cmpMask, nanMask, mask);
                // 获取最大value
                Select(vregValue0, vregValue0, vregValue1, cmpMask);
                // 获取最大value索引
                Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
            }
            AscendC::Reg::AddrReg indicesOffset = AscendC::Reg::CreateAddrReg<uint32_t>(i, VL);
            AscendC::Reg::AddrReg valueOffset = AscendC::Reg::CreateAddrReg<T4>(i, VL);
            MicroAPI::StoreAlign((__ubuf__ T4*&)valueDst, vregValue0, valueOffset, mask);
            if constexpr (IsSameType<T1, half>::value) {
                maskLower = AscendC::Reg::UpdateMask<uint32_t>(sregMask);
                AscendC::Reg::UnPack<uint32_t, T5, AscendC::Reg::HighLowPart::LOWEST>(indicesLower, vregIndices0);
                AscendC::Reg::UnPack<uint32_t, T5, AscendC::Reg::HighLowPart::HIGHEST>(indicesHigher, vregIndices0);
                MicroAPI::StoreAlign(indexDst, (AscendC::Reg::RegTensor<int32_t>&)indicesLower, indicesOffset,
                                     maskLower);
                maskHigher = AscendC::Reg::UpdateMask<uint32_t>(sregMask);
                MicroAPI::StoreAlign(indexDst + repeatElmB32, (AscendC::Reg::RegTensor<int32_t>&)indicesHigher,
                                     indicesOffset, maskHigher);
            } else {
                MicroAPI::StoreAlign(indexDst, (AscendC::Reg::RegTensor<int32_t>&)vregIndices0, indicesOffset, mask);
            }
        }
    }
}

// RA场景VF接口，R < VL && A >= cornNum * VL
template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5, bool isBfloat16Casted>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxRa(__ubuf__ void* valueDst, __ubuf__ T2* indexDst,
                                                                        __ubuf__ void* src, uint16_t dimA,
                                                                        uint16_t alignA, uint16_t dimR, uint32_t startR)
{
    uint16_t VL = GetVLSize();
    if constexpr (IsSameType<T1, bfloat16_t>::value && !isBfloat16Casted) {
        VL = Ops::Base::GetVRegSize() / sizeof(T1);
    }
    // 向上对齐算上尾块次数
    uint16_t loopA = CeilDivision(dimA, VL);
    uint32_t processA = dimA;
    uint32_t sregMask = dimA;
    uint16_t loopR = dimR > 1 ? (dimR - 1) : 0;
    uint16_t offsetA = dimA >= VL ? VL : dimA;
    // B16偏移变量
    uint32_t repeatElmB32 = Ops::Base::GetVRegSize() / sizeof(int32_t);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T4> vregValue0, vregValue1;
        // 指定传入uint类型作为索引
        AscendC::Reg::RegTensor<T5> vregIndices0, vregIndices1;
        AscendC::Reg::RegTensor<uint32_t> indicesLower, indicesHigher;
        AscendC::Reg::UnalignRegForLoad uSrc, tSrc;
        AscendC::Reg::MaskReg mask, maskLower, maskHigher, cmpMask, nanMask;
        for (uint16_t i = 0; i < loopA; i++) {
            auto tmpPtr = (__ubuf__ T4*&)src + alignA;
            mask = AscendC::Reg::UpdateMask<T5>(processA);
            AscendC::Reg::LoadUnAlignPre(uSrc, (__ubuf__ T4*&)src);
            AscendC::Reg::LoadUnAlign<T4, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vregValue0, uSrc,
                                                                                       (__ubuf__ T4*&)src, offsetA);
            // 初始化索引
            Duplicate(vregIndices0, 0);
            for (uint16_t j = 0; j < loopR; j++) { // 从第二行的R开始计算比较
                Duplicate(vregIndices1, j + 1, mask);
                AscendC::Reg::LoadUnAlignPre(tSrc, (__ubuf__ T4*&)tmpPtr);
                AscendC::Reg::LoadUnAlign<T4, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vregValue1, tSrc,
                                                                                           (__ubuf__ T4*&)tmpPtr, VL);
                tmpPtr += (alignA - VL);
                if constexpr (isMin) {
                    Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, mask);
                } else {
                    Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, mask);
                }
                Compare<T4, CMPMODE::NE>(nanMask, vregValue0, vregValue0, mask);
                AscendC::Reg::Or(cmpMask, cmpMask, nanMask, mask);
                // 获取最大value
                Select(vregValue0, vregValue0, vregValue1, cmpMask);
                // 获取最大value索引
                Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
            }
            // CreateAddrReg需要区分数据类型使用
            AscendC::Reg::AddrReg indicesOffset = AscendC::Reg::CreateAddrReg<uint32_t>(i, VL);
            AscendC::Reg::AddrReg valueOffset = AscendC::Reg::CreateAddrReg<T4>(i, VL);
            MicroAPI::StoreAlign((__ubuf__ T4*&)valueDst, vregValue0, valueOffset, mask);
            if constexpr (IsSameType<T1, half>::value || (!isBfloat16Casted && IsSameType<T1, bfloat16_t>::value)) {
                maskLower = AscendC::Reg::UpdateMask<uint32_t>(sregMask);
                AscendC::Reg::UnPack<uint32_t, T5, AscendC::Reg::HighLowPart::LOWEST>(indicesLower, vregIndices0);
                AscendC::Reg::UnPack<uint32_t, T5, AscendC::Reg::HighLowPart::HIGHEST>(indicesHigher, vregIndices0);
                Adds(indicesLower, indicesLower, startR, maskLower);
                MicroAPI::StoreAlign(indexDst, (AscendC::Reg::RegTensor<int32_t>&)indicesLower, indicesOffset,
                                     maskLower);
                maskHigher = AscendC::Reg::UpdateMask<uint32_t>(sregMask);
                Adds(indicesHigher, indicesHigher, startR, maskHigher);
                MicroAPI::StoreAlign(indexDst + repeatElmB32, (AscendC::Reg::RegTensor<int32_t>&)indicesHigher,
                                     indicesOffset, maskHigher);
            } else {
                Adds(vregIndices0, vregIndices0, startR, mask);
                MicroAPI::StoreAlign(indexDst, (AscendC::Reg::RegTensor<int32_t>&)vregIndices0, indicesOffset, mask);
            }
        }
    }
}

template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5> // T4:Value, T5:与value位宽一致的uint类型可以是uint16 uint32 uint64三种
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxAra(__ubuf__ T4* valueDst, __ubuf__ T2* indexDst,
                                                                         __ubuf__ T4* src, uint16_t dimA0,
                                                                         uint16_t dimR, uint16_t dimA1, uint32_t startR)
{
    uint16_t VL = Ops::Base::GetVRegSize() / sizeof(T4);
    uint16_t loopsA0 = dimA0;
    uint16_t loopsA1 = CeilDivision(dimA1, VL);
    uint16_t loopsR = dimR;
    uint32_t repeatElmB32 = Ops::Base::GetVRegSize() / sizeof(int32_t);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T4> vregValue0, vregValue1;
        AscendC::Reg::RegTensor<T5> vregIndices0, vregIndices1;
        AscendC::Reg::RegTensor<uint32_t> indicesLower, indicesHigher;
        AscendC::Reg::MaskReg maskReg, cmpMask, nanMask, maskLower, maskHigher;

        for (uint16_t loopA0 = 0; loopA0 < loopsA0; loopA0++) {
            uint32_t count = dimA1;
            uint32_t sregMask = dimA1;
            for (uint16_t loopA1 = 0; loopA1 < loopsA1; loopA1++) {
                maskReg = AscendC::Reg::UpdateMask<T4>(count);
                AscendC::Reg::LoadAlign(vregValue0, src + loopA0 * dimR * dimA1 + loopA1 * VL);
                Duplicate(vregIndices0, 0);
                for (uint16_t loopR = 1; loopR < loopsR; loopR++) {
                    Duplicate(vregIndices1, loopR, maskReg);
                    AscendC::Reg::LoadAlign(vregValue1, src + loopA0 * dimR * dimA1 + loopR * dimA1 + loopA1 * VL);
                    if constexpr (isMin) {
                        Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, maskReg);
                    } else {
                        Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, maskReg);
                    }
                    Compare<T4, CMPMODE::NE>(nanMask, vregValue0, vregValue0, maskReg); // 处理nan
                    AscendC::Reg::Or(cmpMask, cmpMask, nanMask, maskReg);
                    // 获取最大value
                    Select(vregValue0, vregValue0, vregValue1, cmpMask);
                    // 获取最大value索引
                    Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
                }
                AscendC::Reg::StoreAlign(valueDst + loopA0 * dimA1 + loopA1 * VL, vregValue0, maskReg);

                if constexpr (IsSameType<T5, uint16_t>::value) {
                    maskLower = AscendC::Reg::UpdateMask<uint32_t>(sregMask);
                    AscendC::Reg::UnPack<uint32_t, T5, AscendC::Reg::HighLowPart::LOWEST>(indicesLower, vregIndices0);
                    AscendC::Reg::UnPack<uint32_t, T5, AscendC::Reg::HighLowPart::HIGHEST>(indicesHigher, vregIndices0);
                    Adds(indicesLower, indicesLower, startR, maskLower);
                    MicroAPI::StoreAlign(indexDst + loopA0 * dimA1 + loopA1 * VL,
                                         (AscendC::Reg::RegTensor<T2>&)indicesLower, maskLower);
                    maskHigher = AscendC::Reg::UpdateMask<uint32_t>(sregMask);
                    Adds(indicesHigher, indicesHigher, startR, maskHigher);
                    MicroAPI::StoreAlign(indexDst + loopA0 * dimA1 + loopA1 * VL + repeatElmB32,
                                         (AscendC::Reg::RegTensor<T2>&)indicesHigher, maskHigher);
                } else {
                    Adds(vregIndices0, vregIndices0, startR, maskReg);
                    AscendC::Reg::StoreAlign(indexDst + loopA0 * dimA1 + loopA1 * VL,
                                             (AscendC::Reg::RegTensor<T2>&)vregIndices0, maskReg);
                }
            }
        }
    }
}

template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxAraUnAlign(__ubuf__ T4* valueDst,
                                                                                __ubuf__ T2* indexDst, __ubuf__ T4* src,
                                                                                uint16_t dimA0, uint16_t dimR,
                                                                                uint16_t dimA1, uint16_t dimA1Src,
                                                                                uint16_t dimA1Dst, uint32_t startR)
{
    uint16_t VL = Ops::Base::GetVRegSize() / sizeof(T4);
    if (dimR >= 16 && dimA1Src * 2 <= VL) {
        ArgMaxAraUnAlignCompact<T4, T5>(valueDst, indexDst, src, dimA0, dimR, dimA1, dimA1Src, dimA1Dst, startR);
        return;
    }
    uint16_t loopsA0 = dimA0;
    uint16_t loopsA1 = CeilDivision(dimA1, VL);
    uint16_t loopsR = dimR;
    uint32_t repeatElmB32 = Ops::Base::GetVRegSize() / sizeof(int32_t);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T4> vregValue0, vregValue1;
        AscendC::MicroAPI::RegTensor<T5> vregIndices0, vregIndices1;
        AscendC::MicroAPI::RegTensor<uint32_t> indicesLower, indicesHigher;
        AscendC::MicroAPI::UnalignRegForLoad uSrc, tSrc;
        AscendC::MicroAPI::MaskReg maskReg, cmpMask, nanMask, maskLower, maskHigher;

        for (uint16_t loopA0 = 0; loopA0 < loopsA0; loopA0++) {
            uint32_t count = dimA1;
            uint32_t sregMask = dimA1;
            __ubuf__ T4* srcBase = src + loopA0 * dimR * dimA1Src;
            for (uint16_t loopA1 = 0; loopA1 < loopsA1; loopA1++) {
                uint32_t storeCount = (count >= VL) ? VL : count;
                maskReg = AscendC::MicroAPI::UpdateMask<T4>(count);
                AscendC::MicroAPI::LoadUnAlignPre(uSrc, srcBase);
                AscendC::MicroAPI::LoadUnAlign<T4, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    vregValue0, uSrc, srcBase, storeCount);
                Duplicate(vregIndices0, 0);
                __ubuf__ T4* rowPtr = srcBase;
                for (uint16_t loopR = 1; loopR < loopsR; loopR++) {
                    Duplicate(vregIndices1, loopR, maskReg);
                    rowPtr += (dimA1Src - storeCount);
                    AscendC::MicroAPI::LoadUnAlignPre(tSrc, rowPtr);
                    AscendC::MicroAPI::LoadUnAlign<T4, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        vregValue1, tSrc, rowPtr, storeCount);
                    if constexpr (isMin) {
                        Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, maskReg);
                    } else {
                        Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, maskReg);
                    }
                    Compare<T4, CMPMODE::NE>(nanMask, vregValue0, vregValue0, maskReg);
                    AscendC::MicroAPI::Or(cmpMask, cmpMask, nanMask, maskReg);
                    Select(vregValue0, vregValue0, vregValue1, cmpMask);
                    Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
                }

                AscendC::MicroAPI::StoreAlign(valueDst + loopA0 * dimA1Dst + loopA1 * VL, vregValue0, maskReg);

                if constexpr (IsSameType<T5, uint16_t>::value) {
                    maskLower = AscendC::MicroAPI::UpdateMask<uint32_t>(sregMask);
                    AscendC::MicroAPI::UnPack<uint32_t, T5, AscendC::MicroAPI::HighLowPart::LOWEST>(indicesLower,
                                                                                                    vregIndices0);
                    AscendC::MicroAPI::UnPack<uint32_t, T5, AscendC::MicroAPI::HighLowPart::HIGHEST>(indicesHigher,
                                                                                                     vregIndices0);
                    Adds(indicesLower, indicesLower, startR, maskLower);
                    maskHigher = AscendC::MicroAPI::UpdateMask<uint32_t>(sregMask);
                    Adds(indicesHigher, indicesHigher, startR, maskHigher);
                    MicroAPI::StoreAlign(indexDst + loopA0 * dimA1Dst + loopA1 * VL,
                                         (AscendC::MicroAPI::RegTensor<T2>&)indicesLower, maskLower);
                    MicroAPI::StoreAlign(indexDst + loopA0 * dimA1Dst + loopA1 * VL + repeatElmB32,
                                         (AscendC::MicroAPI::RegTensor<T2>&)indicesHigher, maskHigher);
                } else {
                    Adds(vregIndices0, vregIndices0, startR, maskReg);
                    AscendC::MicroAPI::StoreAlign(indexDst + loopA0 * dimA1Dst + loopA1 * VL,
                                                  (AscendC::MicroAPI::RegTensor<T2>&)vregIndices0, maskReg);
                }
                srcBase += (VL - storeCount);
            }
        }
    }
}

template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxAraInt64(__ubuf__ T4* valueDst,
                                                                              __ubuf__ T2* indexDst, __ubuf__ T4* src,
                                                                              uint16_t dimA0, uint16_t dimR,
                                                                              uint16_t dimA1, uint32_t startR)
{
    uint16_t VL = GetVLSize();
    // 向上对齐算上尾块次数
    uint16_t loopA1 = CeilDivision(dimA1, VL);
    uint32_t sregMask = dimA1;
    uint16_t loopR = dimR;

    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<T4, AscendC::Reg::RegTraitNumTwo> vregValue0, vregValue1;
        AscendC::Reg::RegTensor<T5, AscendC::Reg::RegTraitNumTwo> vregIndices0, vregIndices1;
        AscendC::Reg::RegTensor<T5, AscendC::Reg::RegTraitNumTwo> vregIndicesAdd1, vregIndicesAddStartR;
        AscendC::Reg::UnalignRegForLoad uSrc;
        AscendC::Reg::MaskReg mask, cmpMask;
        for (uint16_t loopA0 = 0; loopA0 < dimA0; loopA0++) {
            uint32_t processA = dimA1;
            AscendC::Reg::Duplicate(vregIndicesAdd1, 1, fullMask);
            AscendC::Reg::Duplicate(vregIndicesAddStartR, startR, fullMask);
            for (uint16_t i = 0; i < loopA1; i++) {
                mask = AscendC::Reg::UpdateMask<uint32_t>(processA);
                AscendC::Reg::LoadAlign(vregValue0, src + loopA0 * dimR * dimA1 + i * VL);
                AscendC::Reg::Duplicate(vregIndices0, 0, fullMask);
                AscendC::Reg::Duplicate(vregIndices1, 0, fullMask);

                // 从第二行的R开始计算比较
                for (uint16_t j = 1; j < loopR; j++) {
                    Add(vregIndices1, vregIndices1, vregIndicesAdd1, mask);
                    AscendC::Reg::LoadAlign(vregValue1, src + loopA0 * dimR * dimA1 + j * dimA1 + i * VL);
                    if constexpr (isMin) {
                        AscendC::Reg::Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, mask);
                    } else {
                        AscendC::Reg::Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, mask);
                    }
                    AscendC::Reg::Select(vregValue0, vregValue0, vregValue1, cmpMask);
                    AscendC::Reg::Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
                }
                MicroAPI::StoreAlign((__ubuf__ T4*)valueDst + loopA0 * dimA1 + i * VL, vregValue0, mask);
                Add(vregIndices0, vregIndices0, vregIndicesAddStartR, mask);
                MicroAPI::StoreAlign((__ubuf__ T5*)indexDst + loopA0 * dimA1 + i * VL, vregIndices0, mask);
            }
        }
    }
}

template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxAraUnAlignInt64(
    __ubuf__ T4* valueDst, __ubuf__ T2* indexDst, __ubuf__ T4* src, uint16_t dimA0, uint16_t dimR, uint16_t dimA1,
    uint16_t dimA1Src, uint16_t dimA1Dst, uint32_t startR)
{
    uint16_t VL = GetVLSize();
    uint16_t loopA1 = CeilDivision(dimA1, VL);
    uint32_t sregMask = dimA1;
    uint16_t loopR = dimR;
    uint16_t rowStride = dimA1Src;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg
            fullMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<T4, AscendC::MicroAPI::RegTraitNumTwo> vregValue0, vregValue1;
        AscendC::MicroAPI::RegTensor<T5, AscendC::MicroAPI::RegTraitNumTwo> vregIndices0, vregIndices1;
        AscendC::MicroAPI::RegTensor<T5, AscendC::MicroAPI::RegTraitNumTwo> vregIndicesAdd1, vregIndicesAddStartR;
        AscendC::MicroAPI::UnalignRegForLoad uSrc, tSrc;
        AscendC::MicroAPI::MaskReg mask, cmpMask;
        for (uint16_t loopA0 = 0; loopA0 < dimA0; loopA0++) {
            uint32_t processA = dimA1;
            AscendC::MicroAPI::Duplicate(vregIndicesAdd1, 1, fullMask);
            AscendC::MicroAPI::Duplicate(vregIndicesAddStartR, startR, fullMask);
            __ubuf__ T4* srcBase = src + loopA0 * dimR * rowStride;
            for (uint16_t i = 0; i < loopA1; i++) {
                uint32_t storeCount = (processA >= VL) ? VL : processA;
                mask = AscendC::MicroAPI::UpdateMask<uint32_t>(processA);
                AscendC::MicroAPI::LoadUnAlignPre(uSrc, srcBase);
                AscendC::MicroAPI::LoadUnAlign<T4, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    vregValue0, uSrc, srcBase, storeCount);
                AscendC::MicroAPI::Duplicate(vregIndices0, 0, fullMask);
                AscendC::MicroAPI::Duplicate(vregIndices1, 0, fullMask);
                __ubuf__ T4* rowPtr = srcBase;

                for (uint16_t j = 1; j < loopR; j++) {
                    Add(vregIndices1, vregIndices1, vregIndicesAdd1, mask);
                    rowPtr += (rowStride - storeCount);
                    AscendC::MicroAPI::LoadUnAlignPre(tSrc, rowPtr);
                    AscendC::MicroAPI::LoadUnAlign<T4, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        vregValue1, tSrc, rowPtr, storeCount);
                    if constexpr (isMin) {
                        AscendC::MicroAPI::Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, mask);
                    } else {
                        AscendC::MicroAPI::Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, mask);
                    }
                    AscendC::MicroAPI::Select(vregValue0, vregValue0, vregValue1, cmpMask);
                    AscendC::MicroAPI::Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
                }
                Add(vregIndices0, vregIndices0, vregIndicesAddStartR, mask);
                MicroAPI::StoreAlign((__ubuf__ T4*)valueDst + loopA0 * dimA1Dst + i * VL, vregValue0, mask);
                MicroAPI::StoreAlign((__ubuf__ T5*)indexDst + loopA0 * dimA1Dst + i * VL, vregIndices0, mask);
                srcBase += (VL - storeCount);
            }
        }
    }
}

template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, bool isBfloat16Casted>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::UpdateResult(__ubuf__ void* srcIndice,
                                                                            __ubuf__ void* srcValue,
                                                                            __ubuf__ void* dstIndice,
                                                                            __ubuf__ void* dstValue, uint64_t num)
{
    auto srcIndicePtr = (__ubuf__ T2*)srcIndice;
    auto srcValuePtr = (__ubuf__ T4*)srcValue;
    auto dstIndicePtr = (__ubuf__ T2*)dstIndice;
    auto dstValuePtr = (__ubuf__ T4*)dstValue;
    uint32_t sreg = static_cast<uint32_t>(num);
    uint32_t sreg32 = static_cast<uint32_t>(num);
    uint32_t vl = GetVLSize();
    if constexpr (IsSameType<T1, bfloat16_t>::value && !isBfloat16Casted) {
        vl = Ops::Base::GetVRegSize() / sizeof(T1);
    }
    uint16_t repeatTimes = CeilDivision(num, vl);
    uint32_t repeatElmB32 = Ops::Base::GetVRegSize() / sizeof(int32_t);
    if constexpr (!IsSameType<T1, int64_t>::value) {
        __VEC_SCOPE__
        {
            AscendC::Reg::UnalignRegForLoad ureg0;
            AscendC::Reg::RegTensor<T4> value0, value1;
            AscendC::Reg::RegTensor<T2> level0, level1, level2, level3;
            AscendC::Reg::MaskReg preg, cmpMask, helpV0Mask, b8Mask, pregLower, pregHigher, preg32;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = AscendC::Reg::UpdateMask<T4>(sreg);
                AscendC::Reg::AddrReg valueOffset = AscendC::Reg::CreateAddrReg<T4>(i, vl);
                AscendC::Reg::AddrReg indicesOffset = AscendC::Reg::CreateAddrReg<uint32_t>(i, vl);
                MicroAPI::LoadAlign(value0, dstValuePtr, valueOffset);
                MicroAPI::LoadAlign(value1, srcValuePtr, valueOffset);
                // compare and select value
                if constexpr (isMin) {
                    AscendC::Reg::Compare<T4, CMPMODE::LE>(cmpMask, value0, value1, preg);
                } else {
                    AscendC::Reg::Compare<T4, CMPMODE::GE>(cmpMask, value0, value1, preg);
                }
                AscendC::Reg::Compare<T4, CMPMODE::NE>(helpV0Mask, value0, value0, preg);
                AscendC::Reg::Or(cmpMask, cmpMask, helpV0Mask, b8Mask);
                Select(value0, value0, value1, cmpMask);
                MicroAPI::StoreAlign(dstValuePtr, value0, valueOffset, preg);
                if constexpr (IsSameType<T1, half>::value || (!isBfloat16Casted && IsSameType<T1, bfloat16_t>::value)) {
                    MicroAPI::LoadAlign(level0, dstIndicePtr, indicesOffset);
                    MicroAPI::LoadAlign(level1, srcIndicePtr, indicesOffset);
                    MicroAPI::LoadAlign(level2, dstIndicePtr + repeatElmB32, indicesOffset);
                    MicroAPI::LoadAlign(level3, srcIndicePtr + repeatElmB32, indicesOffset);
                    AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::LOWEST>(pregLower, cmpMask);
                    AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::HIGHEST>(pregHigher, cmpMask);
                    Select(level0, level0, level1, pregLower);
                    Select(level2, level2, level3, pregHigher);
                    preg32 = AscendC::Reg::UpdateMask<int32_t>(sreg32);
                    MicroAPI::StoreAlign(dstIndicePtr, level0, indicesOffset, preg32);
                    preg32 = AscendC::Reg::UpdateMask<int32_t>(sreg32);
                    MicroAPI::StoreAlign(dstIndicePtr + repeatElmB32, level2, indicesOffset, preg32);
                } else {
                    MicroAPI::LoadAlign(level0, dstIndicePtr, indicesOffset);
                    MicroAPI::LoadAlign(level1, srcIndicePtr, indicesOffset);
                    Select(level0, level0, level1, cmpMask);
                    MicroAPI::StoreAlign(dstIndicePtr, level0, indicesOffset, preg);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<T4, AscendC::Reg::RegTraitNumTwo> value0, value1;
            AscendC::Reg::RegTensor<T2, AscendC::Reg::RegTraitNumTwo> level0, level1;
            AscendC::Reg::MaskReg preg, cmpMask, helpV0Mask;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = AscendC::Reg::UpdateMask<uint32_t>(sreg);
                MicroAPI::LoadAlign(value0, (__ubuf__ T4*)dstValuePtr + i * vl);
                MicroAPI::LoadAlign(value1, (__ubuf__ T4*)srcValuePtr + i * vl);
                // compare and select value
                if constexpr (isMin) {
                    AscendC::Reg::Compare<T4, CMPMODE::LE>(cmpMask, value0, value1, preg);
                } else {
                    AscendC::Reg::Compare<T4, CMPMODE::GE>(cmpMask, value0, value1, preg);
                }
                Select(value0, value0, value1, cmpMask);
                MicroAPI::StoreAlign((__ubuf__ T4*)dstValuePtr + i * vl, value0, preg);

                MicroAPI::LoadAlign(level0, (__ubuf__ T2*)dstIndicePtr + i * vl);
                MicroAPI::LoadAlign(level1, (__ubuf__ T2*)srcIndicePtr + i * vl);
                Select(level0, level0, level1, cmpMask);
                MicroAPI::StoreAlign((__ubuf__ T2*)dstIndicePtr + i * vl, level0, preg);
            }
        }
    }
}

template <typename T1, typename T2, typename T3, bool isMin>
template <typename T4, typename T5>
__aicore__ inline void ArgMaxWithValueBase<T1, T2, T3, isMin>::ArgMaxAraUnAlignCompact(
    __ubuf__ T4* valueDst, __ubuf__ T2* indexDst, __ubuf__ T4* src, uint16_t dimA0, uint16_t dimR, uint16_t dimA1,
    uint16_t dimA1Src, uint16_t dimA1Dst, uint32_t startR)
{
    uint16_t compactA1Src = COMPACT_CHAIN_NUM * dimA1Src;
    uint16_t compactR = dimR / COMPACT_CHAIN_NUM;
    uint16_t tailR = dimR % COMPACT_CHAIN_NUM;
    uint32_t repeatElmB32 = Ops::Base::GetVRegSize() / sizeof(int32_t);
    uint32_t u32DimA1 = dimA1;
    uint32_t sregMask = compactA1Src;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T4> vregVal0, vregVal1, vregC1Val;
        AscendC::MicroAPI::RegTensor<T5> vregIdx0, vregIdx1, vregC1Idx;
        AscendC::MicroAPI::RegTensor<uint32_t> indicesLower, indicesHigher;
        AscendC::MicroAPI::UnalignRegForLoad uSrc, tSrc;
        AscendC::MicroAPI::MaskReg maskC, cmpMask, nanMask;
        AscendC::MicroAPI::MaskReg maskM = AscendC::MicroAPI::UpdateMask<T4>(u32DimA1);
        AscendC::MicroAPI::MaskReg maskLower, maskHigher, mergeMask;
        AscendC::MicroAPI::MaskReg
            b8Mask = AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        maskC = AscendC::MicroAPI::UpdateMask<T4>(sregMask);

        for (uint16_t loopA0 = 0; loopA0 < dimA0; loopA0++) {
            __ubuf__ T4* srcBase = src + loopA0 * dimR * dimA1Src;
            uint32_t idxsregMask = dimA1;
            AscendC::MicroAPI::LoadUnAlignPre(uSrc, srcBase);
            AscendC::MicroAPI::LoadUnAlign<T4, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregVal0, uSrc,
                                                                                                 srcBase, compactA1Src);
            Duplicate(vregIdx0, 0);

            __ubuf__ T4* rowPtr = srcBase;
            for (uint16_t j = 1; j < compactR; j++) {
                // rowPtr += compactA1Src;
                AscendC::MicroAPI::LoadUnAlignPre(tSrc, rowPtr);
                AscendC::MicroAPI::LoadUnAlign<T4, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    vregVal1, tSrc, rowPtr, compactA1Src);
                Duplicate(vregIdx1, COMPACT_CHAIN_NUM * j);

                if constexpr (isMin) {
                    AscendC::MicroAPI::Compare<T4, CMPMODE::LE>(cmpMask, vregVal0, vregVal1, maskC);
                } else {
                    AscendC::MicroAPI::Compare<T4, CMPMODE::GE>(cmpMask, vregVal0, vregVal1, maskC);
                }
                AscendC::MicroAPI::Compare<T4, CMPMODE::NE>(nanMask, vregVal0, vregVal0, maskC);
                AscendC::MicroAPI::Or(cmpMask, cmpMask, nanMask, maskC);
                Select(vregVal0, vregVal0, vregVal1, cmpMask);
                Select(vregIdx0, vregIdx0, vregIdx1, cmpMask);
            }

            // merge chain0 vs chain1 via Reg::Gather from register
            // vregVal0 layout: [chain0(dimA1Src) | chain1(dimA1Src)], same for vregIdx0
            if constexpr (sizeof(T4) == 2) {
                AscendC::MicroAPI::RegTensor<int16_t> gatherIdxReg;
                Arange(gatherIdxReg, (int16_t)dimA1Src);
                AscendC::Reg::Gather(vregC1Val, vregVal0, (AscendC::Reg::RegTensor<uint16_t>&)gatherIdxReg);
                AscendC::Reg::Gather(vregC1Idx, vregIdx0, (AscendC::Reg::RegTensor<uint16_t>&)gatherIdxReg);
            } else {
                AscendC::MicroAPI::RegTensor<int32_t> gatherIdxReg;
                Arange(gatherIdxReg, (int32_t)dimA1Src);
                AscendC::Reg::Gather(vregC1Val, vregVal0, (AscendC::Reg::RegTensor<uint32_t>&)gatherIdxReg);
                AscendC::Reg::Gather(vregC1Idx, vregIdx0, (AscendC::Reg::RegTensor<uint32_t>&)gatherIdxReg);
            }
            Adds(vregC1Idx, vregC1Idx, 1, maskM);

            // merge chain0 vs chain1: strict > tie(idx_le) > NaN(idx_le)
            if constexpr (isMin) {
                AscendC::MicroAPI::Compare<T4, CMPMODE::LT>(mergeMask, vregVal0, vregC1Val, maskM);
            } else {
                AscendC::MicroAPI::Compare<T4, CMPMODE::GT>(mergeMask, vregVal0, vregC1Val, maskM);
            }

            AscendC::MicroAPI::Compare<T4, CMPMODE::EQ>(cmpMask, vregVal0, vregC1Val, maskM);
            AscendC::MicroAPI::Compare<T5, CMPMODE::LE>(nanMask, vregIdx0, vregC1Idx, maskM);
            AscendC::MicroAPI::And(cmpMask, cmpMask, nanMask, maskM);
            AscendC::MicroAPI::Or(mergeMask, mergeMask, cmpMask, maskM);

            AscendC::MicroAPI::Compare<T4, CMPMODE::NE>(cmpMask, vregVal0, vregVal0, maskM);
            AscendC::MicroAPI::Compare<T4, CMPMODE::NE>(maskLower, vregC1Val, vregC1Val, maskM);
            AscendC::MicroAPI::And(maskHigher, cmpMask, maskLower, maskM);
            AscendC::MicroAPI::Not(maskLower, nanMask, maskM);
            AscendC::MicroAPI::And(maskLower, maskHigher, maskLower, maskM);
            AscendC::MicroAPI::Not(maskLower, maskLower, maskM);
            AscendC::MicroAPI::And(cmpMask, cmpMask, maskLower, maskM);
            AscendC::MicroAPI::Or(mergeMask, mergeMask, cmpMask, maskM);

            Select(vregVal0, vregVal0, vregC1Val, mergeMask);
            Select(vregIdx0, vregIdx0, vregC1Idx, mergeMask);

            if (tailR) {
                __ubuf__ T4* tailSrc = src + loopA0 * dimR * dimA1Src + compactR * compactA1Src;
                AscendC::MicroAPI::LoadUnAlignPre(uSrc, tailSrc);
                AscendC::MicroAPI::LoadUnAlign(vregVal1, uSrc, tailSrc, dimA1);
                Duplicate(vregIdx1, COMPACT_CHAIN_NUM * compactR, maskM);

                if constexpr (isMin) {
                    AscendC::MicroAPI::Compare<T4, CMPMODE::LE>(cmpMask, vregVal0, vregVal1, maskM);
                } else {
                    AscendC::MicroAPI::Compare<T4, CMPMODE::GE>(cmpMask, vregVal0, vregVal1, maskM);
                }
                AscendC::MicroAPI::Compare<T4, CMPMODE::NE>(nanMask, vregVal0, vregVal0, maskM);
                AscendC::MicroAPI::Or(cmpMask, cmpMask, nanMask, maskM);
                Select(vregVal0, vregVal0, vregVal1, cmpMask);
                Select(vregIdx0, vregIdx0, vregIdx1, cmpMask);
            }

            AscendC::MicroAPI::StoreAlign(valueDst + loopA0 * dimA1Dst, vregVal0, maskM);

            if constexpr (IsSameType<T5, uint16_t>::value) {
                maskLower = AscendC::MicroAPI::UpdateMask<uint32_t>(idxsregMask);
                AscendC::MicroAPI::UnPack<uint32_t, T5, AscendC::MicroAPI::HighLowPart::LOWEST>(indicesLower, vregIdx0);
                AscendC::MicroAPI::UnPack<uint32_t, T5, AscendC::MicroAPI::HighLowPart::HIGHEST>(indicesHigher,
                                                                                                 vregIdx0);
                Adds(indicesLower, indicesLower, startR, maskLower);
                maskHigher = AscendC::MicroAPI::UpdateMask<uint32_t>(idxsregMask);
                Adds(indicesHigher, indicesHigher, startR, maskHigher);
                MicroAPI::StoreAlign(indexDst + loopA0 * dimA1Dst, (AscendC::MicroAPI::RegTensor<T2>&)indicesLower,
                                     maskLower);
                MicroAPI::StoreAlign(indexDst + loopA0 * dimA1Dst + repeatElmB32,
                                     (AscendC::MicroAPI::RegTensor<T2>&)indicesHigher, maskHigher);
            } else {
                Adds(vregIdx0, vregIdx0, startR, maskM);
                AscendC::MicroAPI::StoreAlign(indexDst + loopA0 * dimA1Dst, (AscendC::MicroAPI::RegTensor<T2>&)vregIdx0,
                                              maskM);
            }
        }
    }
}

} // namespace ArgMaxWithValue

#endif
// ARG_MAX_WITH_VALUE_BASE_H
