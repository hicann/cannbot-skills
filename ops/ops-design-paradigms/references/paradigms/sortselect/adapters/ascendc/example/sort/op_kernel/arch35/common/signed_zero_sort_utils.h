/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIGNED_ZERO_SORT_UTILS_H
#define SIGNED_ZERO_SORT_UTILS_H

#include <type_traits>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "op_kernel/platform_util.h"

namespace SignedZeroSortCommon {
using namespace AscendC;

// 最高位暂存负零标记；调用方的局部排序域必须小于 2^31。
constexpr uint32_t SIGNED_ZERO_INDEX_FLAG = 0x80000000U;

template <typename T>
constexpr bool IS_FLOATING_POINT_V =
    std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float>;

template <typename T>
using FloatBitsT = std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t, uint32_t>;

// 先检测是否含负零；没有负零时走普通 Sort 路径，避免逐元素生成和恢复符号标记。
template <typename T>
__simd_vf__ inline void HasNegativeZeroVecVF(__ubuf__ FloatBitsT<T>* valueAddr, __ubuf__ uint32_t* scalarAddr,
                                             uint32_t totalElems, uint16_t repeatTimes)
{
    using BitsT = FloatBitsT<T>;
    constexpr BitsT signBit = static_cast<BitsT>(BitsT{1} << (sizeof(T) * 8U - 1U));
    constexpr uint32_t vlSize = Ops::Base::GetVRegSize() / sizeof(BitsT);
    Reg::RegTensor<BitsT> valueReg;
    Reg::MaskReg foundMask = Reg::CreateMask<BitsT, Reg::MaskPattern::ALLF>();
    Reg::MaskReg fullMask = Reg::CreateMask<BitsT, Reg::MaskPattern::ALL>();
    for (uint16_t i = 0U; i < repeatTimes; ++i) {
        uint32_t curCount = totalElems - static_cast<uint32_t>(i) * vlSize;
        const uint32_t exceedsVector = static_cast<uint32_t>(curCount > vlSize);
        curCount = exceedsVector * vlSize + (1U - exceedsVector) * curCount;
        Reg::MaskReg valueMask = Reg::UpdateMask<BitsT>(curCount);
        Reg::LoadAlign(valueReg, valueAddr + static_cast<uint32_t>(i) * vlSize);
        Reg::MaskReg negativeZeroMask;
        Reg::Compares<BitsT, CMPMODE::EQ>(negativeZeroMask, valueReg, signBit, valueMask);
        Reg::Or(foundMask, foundMask, negativeZeroMask, fullMask);
    }
    if constexpr (sizeof(BitsT) == sizeof(uint16_t)) {
        Reg::RegTensor<uint16_t> oneReg;
        Reg::RegTensor<uint16_t> resultReg;
        Reg::RegTensor<uint32_t> resultU32Reg;
        Reg::MaskReg oneMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::VL1>();
        Reg::Duplicate(oneReg, static_cast<uint16_t>(1U));
        Reg::Reduce<Reg::ReduceType::MAX>(resultReg, oneReg, foundMask);
        Reg::UnPack<uint32_t, uint16_t, Reg::HighLowPart::LOWEST>(resultU32Reg, resultReg);
        Reg::StoreAlign(scalarAddr, resultU32Reg, oneMask);
    } else {
        Reg::RegTensor<uint32_t> oneReg;
        Reg::RegTensor<uint32_t> resultReg;
        Reg::MaskReg oneMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::VL1>();
        Reg::Duplicate(oneReg, 1U);
        Reg::Reduce<Reg::ReduceType::MAX>(resultReg, oneReg, foundMask);
        Reg::StoreAlign(scalarAddr, resultReg, oneMask);
    }
}

template <typename T>
__aicore__ inline bool HasNegativeZeroVec(LocalTensor<T>& values, LocalTensor<uint32_t>& scalar, uint32_t totalElems)
{
    using BitsT = FloatBitsT<T>;

    constexpr uint32_t vlSize = Ops::Base::GetVRegSize() / sizeof(BitsT);
    uint16_t repeatTimes = static_cast<uint16_t>(Ops::Base::CeilDiv(totalElems, vlSize));
    __ubuf__ BitsT* valueAddr = reinterpret_cast<__ubuf__ BitsT*>(values.GetPhyAddr());
    __ubuf__ uint32_t* scalarAddr = reinterpret_cast<__ubuf__ uint32_t*>(scalar.GetPhyAddr());
    asc_vf_call<HasNegativeZeroVecVF<T>>(dim3{1, 1, 1}, valueAddr, scalarAddr, totalElems, repeatTimes);
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event);
    WaitFlag<HardEvent::V_S>(event);
    return scalar.GetValue(0) != 0U;
}

// 排序前将负零归一为正零，原始符号存入源索引载荷。
// 这些辅助函数使用 SIMD 寄存器计算；改为 SIMT 时须重新核对共享 scratch 预算及性能。
template <typename T>
__simd_vf__ inline void PrepareSignedZeroKeysVecVF(__ubuf__ FloatBitsT<T>* valueAddr,
                                                   __ubuf__ uint32_t* sourceOrderAddr, uint32_t totalElems,
                                                   uint16_t repeatTimes)
{
    using BitsT = FloatBitsT<T>;
    constexpr BitsT signBit = static_cast<BitsT>(BitsT{1} << (sizeof(T) * 8U - 1U));
    constexpr uint32_t vlSize = Ops::Base::GetVRegSize() / sizeof(uint32_t);
    Reg::RegTensor<BitsT> valueReg;
    Reg::RegTensor<BitsT> zeroReg;
    Reg::RegTensor<BitsT> normalizedReg;
    Reg::RegTensor<int32_t> baseIndexReg;
    Reg::RegTensor<int32_t> indexReg;
    Reg::RegTensor<uint32_t> indexFlagReg;
    Reg::RegTensor<uint32_t> flaggedIndexReg;
    Reg::RegTensor<uint32_t> sourceOrderReg;
    Reg::MaskReg fullValueMask = Reg::CreateMask<BitsT, Reg::MaskPattern::ALL>();
    Reg::MaskReg fullIndexMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Duplicate(zeroReg, static_cast<BitsT>(0U), fullValueMask);
    Reg::Duplicate(indexFlagReg, SIGNED_ZERO_INDEX_FLAG, fullIndexMask);
    Reg::Arange(baseIndexReg, 0);
    for (uint16_t i = 0U; i < repeatTimes; ++i) {
        uint32_t curCount = totalElems - static_cast<uint32_t>(i) * vlSize;
        const uint32_t exceedsVector = static_cast<uint32_t>(curCount > vlSize);
        curCount = exceedsVector * vlSize + (1U - exceedsVector) * curCount;
        uint32_t valueCount = curCount;
        uint32_t indexCount = curCount;
        Reg::MaskReg valueMask = Reg::UpdateMask<BitsT>(valueCount);
        Reg::MaskReg indexMask = Reg::UpdateMask<uint32_t>(indexCount);
        Reg::LoadAlign(valueReg, valueAddr + static_cast<uint32_t>(i) * vlSize);
        Reg::MaskReg negativeZeroMask;
        Reg::Compares<BitsT, CMPMODE::EQ>(negativeZeroMask, valueReg, signBit, valueMask);
        Reg::Select(normalizedReg, zeroReg, valueReg, negativeZeroMask);
        Reg::StoreAlign(valueAddr + static_cast<uint32_t>(i) * vlSize, normalizedReg, valueMask);

        Reg::Adds(indexReg, baseIndexReg, static_cast<int32_t>(i * vlSize), indexMask);
        Reg::Or(flaggedIndexReg, (Reg::RegTensor<uint32_t>&)indexReg, indexFlagReg, indexMask);
        Reg::MaskReg negativeZeroIndexMask;
        if constexpr (sizeof(BitsT) == sizeof(uint16_t)) {
            // B16 谓词的第 i 个 lane 使用 bit 2*i，B32 使用 bit 4*i；用于索引前先扩展低半部。
            Reg::UnPack<Reg::HighLowPart::LOWEST>(negativeZeroIndexMask, negativeZeroMask);
        } else {
            negativeZeroIndexMask = negativeZeroMask;
        }
        Reg::Select(sourceOrderReg, flaggedIndexReg, (Reg::RegTensor<uint32_t>&)indexReg, negativeZeroIndexMask);
        Reg::StoreAlign(sourceOrderAddr + static_cast<uint32_t>(i) * vlSize, sourceOrderReg, indexMask);
    }
}

template <typename T>
__aicore__ inline void PrepareSignedZeroKeysVec(LocalTensor<T>& values, LocalTensor<uint32_t>& sourceOrder,
                                                uint32_t totalElems)
{
    using BitsT = FloatBitsT<T>;

    // 每轮按 B32 索引的向量容量处理（arch35 为 64 元素），使值与索引共用有效谓词。
    constexpr uint32_t vlSize = Ops::Base::GetVRegSize() / sizeof(uint32_t);
    uint16_t repeatTimes = static_cast<uint16_t>(Ops::Base::CeilDiv(totalElems, vlSize));
    __ubuf__ BitsT* valueAddr = (__ubuf__ BitsT*)values.GetPhyAddr();
    __ubuf__ uint32_t* sourceOrderAddr = (__ubuf__ uint32_t*)sourceOrder.GetPhyAddr();
    asc_vf_call<PrepareSignedZeroKeysVecVF<T>>(dim3{1, 1, 1}, valueAddr, sourceOrderAddr, totalElems, repeatTimes);
}

// 两阶段排序已有源位置，只需在暂未使用的 B32 输出缓冲中保存负零标记。
// 隐式索引 Sort 自行生成位置，避免额外构造并排序一整份自定义源索引。
template <typename T>
__simd_vf__ inline void PrepareSignedZeroKeysAndFlagsVecVF(__ubuf__ FloatBitsT<T>* valueAddr,
                                                           __ubuf__ FloatBitsT<T>* signFlagAddr, uint32_t totalElems,
                                                           uint16_t repeatTimes)
{
    using BitsT = FloatBitsT<T>;
    constexpr BitsT signBit = static_cast<BitsT>(BitsT{1} << (sizeof(T) * 8U - 1U));
    constexpr uint32_t vlSize = Ops::Base::GetVRegSize() / sizeof(BitsT);
    Reg::RegTensor<BitsT> valueReg;
    Reg::RegTensor<BitsT> zeroReg;
    Reg::RegTensor<BitsT> normalizedReg;
    Reg::RegTensor<BitsT> zeroFlagReg;
    Reg::RegTensor<BitsT> negativeZeroFlagReg;
    Reg::RegTensor<BitsT> signFlagReg;
    Reg::MaskReg fullValueMask = Reg::CreateMask<BitsT, Reg::MaskPattern::ALL>();
    Reg::Duplicate(zeroReg, static_cast<BitsT>(0U), fullValueMask);
    Reg::Duplicate(zeroFlagReg, static_cast<BitsT>(0U), fullValueMask);
    Reg::Duplicate(negativeZeroFlagReg, signBit, fullValueMask);
    for (uint16_t i = 0U; i < repeatTimes; ++i) {
        uint32_t curCount = totalElems - static_cast<uint32_t>(i) * vlSize;
        const uint32_t exceedsVector = static_cast<uint32_t>(curCount > vlSize);
        curCount = exceedsVector * vlSize + (1U - exceedsVector) * curCount;
        Reg::MaskReg valueMask = Reg::UpdateMask<BitsT>(curCount);
        Reg::LoadAlign(valueReg, valueAddr + static_cast<uint32_t>(i) * vlSize);
        Reg::MaskReg negativeZeroMask;
        Reg::Compares<BitsT, CMPMODE::EQ>(negativeZeroMask, valueReg, signBit, valueMask);
        Reg::Select(normalizedReg, zeroReg, valueReg, negativeZeroMask);
        // 逆排名散射仅需非零标记，按值宽度存放即可复用 Compare 谓词；底层按 B32 分配，容量充足。
        Reg::Select(signFlagReg, negativeZeroFlagReg, zeroFlagReg, negativeZeroMask);
        Reg::StoreAlign(valueAddr + static_cast<uint32_t>(i) * vlSize, normalizedReg, valueMask);
        Reg::StoreAlign(signFlagAddr + static_cast<uint32_t>(i) * vlSize, signFlagReg, valueMask);
    }
}

template <typename T>
__aicore__ inline void PrepareSignedZeroKeysAndFlagsVec(LocalTensor<T>& values, LocalTensor<uint32_t>& signFlags,
                                                        uint32_t totalElems)
{
    using BitsT = FloatBitsT<T>;

    constexpr uint32_t vlSize = Ops::Base::GetVRegSize() / sizeof(BitsT);
    uint16_t repeatTimes = static_cast<uint16_t>(Ops::Base::CeilDiv(totalElems, vlSize));
    __ubuf__ BitsT* valueAddr = (__ubuf__ BitsT*)values.GetPhyAddr();
    __ubuf__ BitsT* signFlagAddr = reinterpret_cast<__ubuf__ BitsT*>(signFlags.GetPhyAddr());
    asc_vf_call<PrepareSignedZeroKeysAndFlagsVecVF<T>>(dim3{1, 1, 1}, valueAddr, signFlagAddr, totalElems, repeatTimes);
}

// 通过排序后索引收集原始符号标记，仅将原本为负零的元素恢复为负零。
template <typename T>
__simd_vf__ inline void RestoreSignedZeroValuesByIndexVecVF(__ubuf__ FloatBitsT<T>* valueAddr,
                                                            __ubuf__ uint32_t* indexAddr,
                                                            __ubuf__ uint32_t* sourceOrderAddr, uint32_t totalElems,
                                                            uint16_t repeatTimes)
{
    using BitsT = FloatBitsT<T>;
    constexpr BitsT signBit = static_cast<BitsT>(BitsT{1} << (sizeof(T) * 8U - 1U));
    constexpr uint32_t vlSize = Ops::Base::GetVRegSize() / sizeof(uint32_t);
    Reg::RegTensor<BitsT> valueReg;
    Reg::RegTensor<BitsT> negativeZeroReg;
    Reg::RegTensor<BitsT> restoredValueReg;
    Reg::RegTensor<uint32_t> indexReg;
    Reg::RegTensor<uint32_t> sourceOrderReg;
    Reg::RegTensor<uint32_t> flagReg;
    Reg::RegTensor<uint32_t> indexFlagReg;
    Reg::MaskReg fullValueMask = Reg::CreateMask<BitsT, Reg::MaskPattern::ALL>();
    Reg::MaskReg fullIndexMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Duplicate(negativeZeroReg, signBit, fullValueMask);
    Reg::Duplicate(indexFlagReg, SIGNED_ZERO_INDEX_FLAG, fullIndexMask);
    for (uint16_t i = 0U; i < repeatTimes; ++i) {
        uint32_t curCount = totalElems - static_cast<uint32_t>(i) * vlSize;
        const uint32_t exceedsVector = static_cast<uint32_t>(curCount > vlSize);
        curCount = exceedsVector * vlSize + (1U - exceedsVector) * curCount;
        // UpdateMask 会消耗计数器；值与索引必须使用独立计数，即使两者均为 B32。
        uint32_t valueCount = curCount;
        uint32_t indexCount = curCount;
        Reg::MaskReg valueMask = Reg::UpdateMask<BitsT>(valueCount);
        Reg::MaskReg indexMask = Reg::UpdateMask<uint32_t>(indexCount);
        Reg::LoadAlign(valueReg, valueAddr + static_cast<uint32_t>(i) * vlSize);
        Reg::LoadAlign(indexReg, indexAddr + static_cast<uint32_t>(i) * vlSize);
        Reg::Gather(sourceOrderReg, sourceOrderAddr, indexReg, indexMask);
        Reg::And(flagReg, sourceOrderReg, indexFlagReg, indexMask);
        Reg::MaskReg negativeZeroIndexMask;
        Reg::Compares<uint32_t, CMPMODE::NE>(negativeZeroIndexMask, flagReg, 0U, indexMask);
        Reg::MaskReg negativeZeroValueMask;
        if constexpr (sizeof(BitsT) == sizeof(uint16_t)) {
            Reg::Pack<Reg::HighLowPart::LOWEST>(negativeZeroValueMask, negativeZeroIndexMask);
        } else {
            negativeZeroValueMask = negativeZeroIndexMask;
        }
        Reg::Select(restoredValueReg, negativeZeroReg, valueReg, negativeZeroValueMask);
        Reg::StoreAlign(valueAddr + static_cast<uint32_t>(i) * vlSize, restoredValueReg, valueMask);
    }
}

template <typename T>
__aicore__ inline void RestoreSignedZeroValuesByIndexVec(LocalTensor<T>& values, LocalTensor<uint32_t>& indices,
                                                         LocalTensor<uint32_t>& sourceOrder, uint32_t totalElems)
{
    using BitsT = FloatBitsT<T>;

    constexpr uint32_t vlSize = Ops::Base::GetVRegSize() / sizeof(uint32_t);
    uint16_t repeatTimes = static_cast<uint16_t>(Ops::Base::CeilDiv(totalElems, vlSize));
    __ubuf__ BitsT* valueAddr = reinterpret_cast<__ubuf__ BitsT*>(values.GetPhyAddr());
    __ubuf__ uint32_t* indexAddr = reinterpret_cast<__ubuf__ uint32_t*>(indices.GetPhyAddr());
    __ubuf__ uint32_t* sourceOrderAddr = reinterpret_cast<__ubuf__ uint32_t*>(sourceOrder.GetPhyAddr());
    asc_vf_call<RestoreSignedZeroValuesByIndexVecVF<T>>(dim3{1, 1, 1}, valueAddr, indexAddr, sourceOrderAddr,
                                                        totalElems, repeatTimes);
}

} // namespace SignedZeroSortCommon

#endif // SIGNED_ZERO_SORT_UTILS_H
