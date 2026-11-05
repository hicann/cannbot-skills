/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RADIX_SORT_SIMD_UTILS_H
#define RADIX_SORT_SIMD_UTILS_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "util_type_simd.h"
#include "radix_sort_constants.h"

namespace RadixSortCommon {

using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

// 基数预处理将有符号数或浮点数映射为无符号保序键。
// 正负零必须映射到同一个键；每轮均转换输入，因此归一化放在 TwiddleInFp16/TwiddleInFp32 中。

template <typename UT, uint64_t isDescend>
__simd_callee__ inline void Twiddle(uint16_t repeatTime, uint32_t vfLen, uint32_t inputNum, RegTensor<UT>& xorReg,
                                    __ubuf__ UT* xValuePtr, __ubuf__ UT* uXValuePtr)
{
    Reg::MaskReg xorMask;
    Reg::RegTensor<UT> inputReg;
    Reg::RegTensor<UT> vnotReg;
    Reg::RegTensor<UT> xorResult;
    for (uint16_t i = 0; i < repeatTime; i++) {
        xorMask = Reg::UpdateMask<UT>(inputNum);
        Reg::LoadAlign<UT, Reg::PostLiteral::POST_MODE_UPDATE>(inputReg, xValuePtr, vfLen);
        Reg::Xor(xorResult, inputReg, xorReg, xorMask);
        if constexpr (isDescend == 0) {
            Reg::StoreAlign<UT, Reg::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, xorResult, vfLen, xorMask);
        } else {
            Reg::Not(vnotReg, xorResult, xorMask);
            Reg::StoreAlign<UT, Reg::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, vnotReg, vfLen, xorMask);
        }
    }
}

template <typename UT, uint64_t isDescend>
__simd_vf__ inline void TwiddleInB32VF(__ubuf__ UT* xValuePtr, __ubuf__ UT* uXValuePtr, uint16_t repeatTime,
                                       uint32_t inputNum)
{
    Reg::RegTensor<UT> xorValue;
    Reg::MaskReg xorMask;
    Reg::MaskReg maskB32 = Reg::CreateMask<UT>();
    Reg::Duplicate(xorValue, XOR_OP_VALUE, maskB32);
    Twiddle<UT, isDescend>(repeatTime, VF_LEN_B32, inputNum, xorValue, xValuePtr, uXValuePtr);
}

template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInB32(LocalTensor<T1> inputX, LocalTensor<UT> uInputX, uint32_t numTileData)
{
    __ubuf__ UT* xValuePtr = (__ubuf__ UT*)inputX.GetPhyAddr();
    __ubuf__ UT* uXValuePtr = (__ubuf__ UT*)uInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B32);
    uint32_t inputNum = numTileData;
    asc_vf_call<TwiddleInB32VF<UT, isDescend>>(dim3{1, 1, 1}, xValuePtr, uXValuePtr, repeatTime, inputNum);
}

template <typename UT, uint64_t isDescend>
__simd_vf__ inline void TwiddleInB16VF(__ubuf__ UT* xValuePtr, __ubuf__ UT* uXValuePtr, uint16_t repeatTime,
                                       uint32_t inputNum)
{
    Reg::RegTensor<UT> xorReg;
    Reg::MaskReg maskB16 = Reg::CreateMask<UT>();
    Reg::Duplicate(xorReg, XOR_OP_VALUE_B16, maskB16);
    Twiddle<UT, isDescend>(repeatTime, VF_LEN_B16, inputNum, xorReg, xValuePtr, uXValuePtr);
}

template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInB16(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __ubuf__ UT* xValuePtr = (__ubuf__ UT*)inputX.GetPhyAddr();
    __ubuf__ UT* uXValuePtr = (__ubuf__ UT*)uintInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B16);
    uint32_t inputNum = numTileData;
    asc_vf_call<TwiddleInB16VF<UT, isDescend>>(dim3{1, 1, 1}, xValuePtr, uXValuePtr, repeatTime, inputNum);
}

template <typename UT, uint64_t isDescend>
__simd_vf__ inline void TwiddleInB8VF(__ubuf__ UT* xValuePtr, __ubuf__ UT* uXValuePtr, uint16_t repeatTime,
                                      uint32_t inputNum)
{
    Reg::RegTensor<UT> xorReg;
    Reg::MaskReg maskB8 = Reg::CreateMask<UT>();
    Reg::Duplicate(xorReg, XOR_OP_VALUE_B8, maskB8);
    Twiddle<UT, isDescend>(repeatTime, VF_LEN_B8, inputNum, xorReg, xValuePtr, uXValuePtr);
}

template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInB8(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __ubuf__ UT* xValuePtr = (__ubuf__ UT*)inputX.GetPhyAddr();
    __ubuf__ UT* uXValuePtr = (__ubuf__ UT*)uintInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B8);
    uint32_t inputNum = numTileData;
    asc_vf_call<TwiddleInB8VF<UT, isDescend>>(dim3{1, 1, 1}, xValuePtr, uXValuePtr, repeatTime, inputNum);
}

template <typename UT, uint64_t isDescend>
__simd_vf__ inline void TwiddleInB64VF(__ubuf__ UT* xValuePtr, __ubuf__ UT* uXValuePtr, uint16_t repeatTime,
                                       uint32_t inputNum)
{
    Reg::RegTensor<UT> xorReg;
    Reg::MaskReg predicateDefaultB64 = Reg::CreateMask<UT>();
    Reg::Duplicate(xorReg, XOR_OP_VALUE_B64, predicateDefaultB64);
    Twiddle<UT, isDescend>(repeatTime, VF_LEN_B64, inputNum, xorReg, xValuePtr, uXValuePtr);
}

template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInB64(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __ubuf__ UT* xValuePtr = (__ubuf__ UT*)inputX.GetPhyAddr();
    __ubuf__ UT* uXValuePtr = (__ubuf__ UT*)uintInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B64);
    uint32_t inputNum = numTileData;
    asc_vf_call<TwiddleInB64VF<UT, isDescend>>(dim3{1, 1, 1}, xValuePtr, uXValuePtr, repeatTime, inputNum);
}

template <typename T1, uint64_t isDescend, bool CanonicalizeNan>
__simd_vf__ inline void TwiddleInFp16ImplVF(__ubuf__ uint16_t* xPtr, __ubuf__ uint16_t* outPtr, uint16_t repeatTime,
                                            uint32_t inputNum)
{
    Reg::RegTensor<uint16_t> inputReg;
    Reg::RegTensor<uint16_t> vnotReg;
    Reg::RegTensor<uint16_t> xorMaskReg;
    Reg::RegTensor<uint16_t> vandMask;
    Reg::RegTensor<uint16_t> twiddledZeroReg;
    Reg::RegTensor<uint16_t> canonicalNanReg;
    Reg::RegTensor<uint16_t> absMaskReg;
    Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t>(), xorMask;
    Reg::Duplicate(xorMaskReg, LOWEST_KEY_VALUE_B16, maskB16);
    Reg::Duplicate(vandMask, XOR_OP_VALUE_B16, maskB16);
    Reg::Duplicate(twiddledZeroReg, TWIDDLED_ZERO_BITS_FP16, maskB16);
    // 仅 NaN 规范化模式使用：预计算统一 NaN 键与绝对值掩码。
    if constexpr (CanonicalizeNan) {
        Reg::Duplicate(canonicalNanReg, CANONICAL_NAN_KEY_B16, maskB16);
        Reg::Duplicate(absMaskReg, FLOAT_ABS_MASK_B16, maskB16);
    }

    for (uint16_t i = 0; i < repeatTime; i++) {
        xorMask = Reg::UpdateMask<uint16_t>(inputNum);
        // 加载输入。
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(inputReg, xPtr, VF_LEN_B16);
        // 提取符号位。
        Reg::RegTensor<uint16_t> andValueOne;
        Reg::And(andValueOne, inputReg, vandMask, maskB16);
        // 生成符号掩码。
        Reg::MaskReg cmpValueOne;
        Reg::Compares<uint16_t, CMPMODE::NE>(cmpValueOne, andValueOne, ZERO_VALUE_FLAG_B16, maskB16);
        // 选择转换掩码。
        Reg::RegTensor<uint16_t> finalMaskOne;
        Reg::Select(finalMaskOne, xorMaskReg, vandMask, cmpValueOne);
        // 异或得到保序键。
        Reg::RegTensor<uint16_t> xorVectorOne;
        Reg::Xor(xorVectorOne, inputReg, finalMaskOne, maskB16);

        // 目前每个round都会做twiddleIn，因此直接在twiddleIn方法将负0转换为正0
        // 如果后期需要改为单次twiddleIn，则需要在提取位数时将负0转换为正0
        // 生成负零掩码。
        Reg::MaskReg minusZeroMask;
        Reg::Compares<uint16_t, CMPMODE::EQ>(minusZeroMask, xorVectorOne, TWIDDLED_MINUS_ZERO_BITS_FP16, maskB16);
        // 将负零键统一为正零键。
        Reg::RegTensor<uint16_t> resultReg;
        Reg::Select(resultReg, twiddledZeroReg, xorVectorOne, minusZeroMask);
        // 仅 NaN 规范化模式使用：通过 |bits| > inf 检出任意符号的 NaN，统一映射至最大键。
        // 仅修改排序键，原始值保持不变。
        if constexpr (CanonicalizeNan) {
            constexpr uint16_t INF_BITS = static_cast<uint16_t>(IsSameType<half, T1>::value) * FP16_INFINITY_BITS +
                                          static_cast<uint16_t>(!IsSameType<half, T1>::value) * BF16_INFINITY_BITS;
            Reg::RegTensor<uint16_t> absBits;
            Reg::MaskReg nanMask;
            Reg::And(absBits, inputReg, absMaskReg, xorMask);
            Reg::Compares<uint16_t, CMPMODE::GT>(nanMask, absBits, INF_BITS, xorMask);
            Reg::Select(resultReg, canonicalNanReg, resultReg, nanMask);
        }

        if constexpr (isDescend == 0) {
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(outPtr, resultReg, VF_LEN_B16, xorMask);
        } else {
            Reg::Not(vnotReg, resultReg, xorMask);
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(outPtr, vnotReg, VF_LEN_B16, xorMask);
        }
    }
}

template <typename T1, typename UT, uint64_t isDescend, bool CanonicalizeNan>
__aicore__ inline void TwiddleInFp16Impl(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    auto* xPtr = (__ubuf__ uint16_t*)inputX.GetPhyAddr();
    auto* outPtr = (__ubuf__ uint16_t*)uintInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B16);
    uint32_t inputNum = numTileData;
    asc_vf_call<TwiddleInFp16ImplVF<T1, isDescend, CanonicalizeNan>>(dim3{1, 1, 1}, xPtr, outPtr, repeatTime, inputNum);
}

template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInFp16(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    TwiddleInFp16Impl<T1, UT, isDescend, false>(inputX, uintInputX, numTileData);
}

template <typename T1, uint64_t isDescend, bool CanonicalizeNan>
__simd_vf__ inline void TwiddleInFp32ImplVF(__ubuf__ uint32_t* xValuePtr, __ubuf__ uint32_t* uXValuePtr,
                                            uint16_t repeatTime, uint32_t inputNum)
{
    Reg::RegTensor<uint32_t> inputReg;
    Reg::RegTensor<uint32_t> vnotReg;
    Reg::RegTensor<uint32_t> xorMaskReg;
    Reg::RegTensor<uint32_t> vandMask;
    Reg::RegTensor<uint32_t> twiddledZeroReg;
    Reg::RegTensor<uint32_t> canonicalNanReg;
    Reg::RegTensor<uint32_t> absMaskReg;
    Reg::MaskReg maskB32 = Reg::CreateMask<uint32_t>(), xorMask;
    Reg::Duplicate(xorMaskReg, LOWEST_KEY_VALUE_B32, maskB32);
    Reg::Duplicate(vandMask, XOR_OP_VALUE, maskB32);
    Reg::Duplicate(twiddledZeroReg, TWIDDLED_ZERO_BITS_FP32, maskB32);
    if constexpr (CanonicalizeNan) {
        Reg::Duplicate(canonicalNanReg, CANONICAL_NAN_KEY_B32, maskB32);
        Reg::Duplicate(absMaskReg, FLOAT_ABS_MASK_B32, maskB32);
    }
    for (uint16_t i = 0; i < repeatTime; i++) {
        xorMask = Reg::UpdateMask<uint32_t>(inputNum);
        // 加载输入。
        Reg::LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(inputReg, xValuePtr, VF_LEN_B32);
        // 提取符号位。
        Reg::RegTensor<uint32_t> andValueOne;
        Reg::And(andValueOne, inputReg, vandMask, maskB32);
        // 生成符号掩码。
        Reg::MaskReg cmpValueOne;
        Reg::Compares<uint32_t, CMPMODE::NE>(cmpValueOne, andValueOne, ZERO_VALUE_FLAG_B32, xorMask);
        // 选择转换掩码。
        Reg::RegTensor<uint32_t> finalMaskOne;
        Reg::Select(finalMaskOne, xorMaskReg, vandMask, cmpValueOne);
        // 异或得到保序键。
        Reg::RegTensor<uint32_t> xorVectorZero;
        Reg::Xor(xorVectorZero, inputReg, finalMaskOne, maskB32);

        // 目前每个round都会做twiddleIn，因此直接在twiddleIn方法将负0转换为正0
        // 如果后期需要改为单次twiddleIn，则需要在提取位数时将负0转换为正0
        // 生成负零掩码。
        Reg::MaskReg minusZeroMask;
        Reg::Compares<uint32_t, CMPMODE::EQ>(minusZeroMask, xorVectorZero, TWIDDLED_MINUS_ZERO_BITS_FP32, maskB32);
        // 将负零键统一为正零键。
        Reg::RegTensor<uint32_t> resultReg;
        Reg::Select(resultReg, twiddledZeroReg, xorVectorZero, minusZeroMask);
        if constexpr (CanonicalizeNan) {
            Reg::RegTensor<uint32_t> absBits;
            Reg::MaskReg nanMask;
            Reg::And(absBits, inputReg, absMaskReg, xorMask);
            Reg::Compares<uint32_t, CMPMODE::GT>(nanMask, absBits, FP32_INFINITY_BITS, xorMask);
            Reg::Select(resultReg, canonicalNanReg, resultReg, nanMask);
        }

        if constexpr (isDescend == 0) {
            Reg::StoreAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, resultReg, VF_LEN_B32, xorMask);
        } else {
            Reg::Not(vnotReg, resultReg, maskB32);
            Reg::StoreAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, vnotReg, VF_LEN_B32, xorMask);
        }
    }
}

template <typename T1, typename UT, uint64_t isDescend, bool CanonicalizeNan>
__aicore__ inline void TwiddleInFp32Impl(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __ubuf__ uint32_t* xValuePtr = (__ubuf__ uint32_t*)inputX.GetPhyAddr();
    __ubuf__ uint32_t* uXValuePtr = (__ubuf__ uint32_t*)uintInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B32);
    uint32_t inputNum = numTileData;
    asc_vf_call<TwiddleInFp32ImplVF<T1, isDescend, CanonicalizeNan>>(dim3{1, 1, 1}, xValuePtr, uXValuePtr, repeatTime,
                                                                     inputNum);
}

template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInFp32(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    TwiddleInFp32Impl<T1, UT, isDescend, false>(inputX, uintInputX, numTileData);
}

template <typename UT>
__simd_vf__ inline void ReverseInputDataVF(__ubuf__ UT* inputXValuePtr, __ubuf__ UT* reverseInputXPtr, uint32_t vfLen,
                                           uint16_t repeatTime, uint32_t inputElementNum)
{
    Reg::RegTensor<UT> inputVectorOne;
    Reg::RegTensor<UT> vnotVectorZero;
    Reg::MaskReg predicateDefaultB8 = Reg::CreateMask<UT>();
    for (uint16_t i = 0; i < repeatTime; i++) {
        Reg::MaskReg vnotMask = Reg::UpdateMask<UT>(inputElementNum);
        Reg::LoadAlign<UT, Reg::PostLiteral::POST_MODE_UPDATE>(inputVectorOne, inputXValuePtr, vfLen);
        Reg::Not(vnotVectorZero, inputVectorOne, predicateDefaultB8);
        Reg::StoreAlign<UT, Reg::PostLiteral::POST_MODE_UPDATE>(reverseInputXPtr, vnotVectorZero, vfLen, vnotMask);
    }
}

template <typename T1, typename UT>
__aicore__ inline void ReverseInputData(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __ubuf__ UT* inputXValuePtr = (__ubuf__ UT*)inputX.GetPhyAddr();
    __ubuf__ UT* reverseInputXPtr = (__ubuf__ UT*)uintInputX.GetPhyAddr();
    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(UT);
    uint16_t repeatTime = CeilDivision(numTileData, vfLen);
    uint32_t inputElementNum = numTileData;
    asc_vf_call<ReverseInputDataVF<UT>>(dim3{1, 1, 1}, inputXValuePtr, reverseInputXPtr, vfLen, repeatTime,
                                        inputElementNum);
}

} // namespace RadixSortCommon

#endif
