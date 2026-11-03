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
 * \file sort_lib_vf.h
 * \brief SIMD VF micro-operations for radix sort (histogram, scatter, lookback, twiddle).
 *
 * \internal  Do not include directly — used internally by sort_radix_more_core.h.
 */

#ifndef SORT_LIB_VF_H
#define SORT_LIB_VF_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "sort_lib_util.h"
#include "sort_lib_constants.h"

namespace SortLib::detail {

// ── Twiddle ──

template <typename KeyT, bool isDescend>
__simd_callee__ __aicore__ inline void Twiddle(uint16_t repeatTime, uint32_t vfLen, uint32_t inputNum,
                                               AscendC::Reg::RegTensor<KeyT>& xorReg, __ubuf__ KeyT* xValuePtr,
                                               __ubuf__ KeyT* uXValuePtr)
{
    AscendC::Reg::MaskReg xorMask;
    AscendC::Reg::RegTensor<KeyT> inputReg;
    AscendC::Reg::RegTensor<KeyT> vnotReg;
    AscendC::Reg::RegTensor<KeyT> xorResult;
    for (uint16_t i = 0; i < repeatTime; i++) {
        xorMask = AscendC::Reg::UpdateMask<KeyT>(inputNum);
        AscendC::Reg::LoadAlign<KeyT, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(inputReg, xValuePtr, vfLen);
        AscendC::Reg::Xor(xorResult, inputReg, xorReg, xorMask);
        if constexpr (!isDescend) {
            AscendC::Reg::StoreAlign<KeyT, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, xorResult, vfLen,
                                                                                        xorMask);
        } else {
            AscendC::Reg::Not(vnotReg, xorResult, xorMask);
            AscendC::Reg::StoreAlign<KeyT, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, vnotReg, vfLen,
                                                                                        xorMask);
        }
    }
}

template <typename KeyT, bool isDescend, uint32_t VF_LEN>
__simd_vf__ __aicore__ inline void TwiddleSignedVF(__ubuf__ KeyT* src, __ubuf__ KeyT* dst, uint16_t rpt, uint32_t n)
{
    constexpr KeyT xorVal = static_cast<KeyT>(static_cast<KeyT>(1) << (sizeof(KeyT) * 8 - 1));
    AscendC::Reg::RegTensor<KeyT> xorValue;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<KeyT>();
    AscendC::Reg::Duplicate(xorValue, xorVal, mask);
    Twiddle<KeyT, isDescend>(rpt, VF_LEN, n, xorValue, src, dst);
}

template <typename ValT, typename KeyT, bool isDescend>
__aicore__ inline void TwiddleInSignedInt(AscendC::LocalTensor<ValT> inputX, AscendC::LocalTensor<KeyT> uInputX,
                                          uint32_t numTileData)
{
    constexpr uint32_t vfLen = 256 / sizeof(KeyT);
    uint16_t repeatTime = AscendC::CeilDivision(numTileData, vfLen);
    asc_vf_call<TwiddleSignedVF<KeyT, isDescend, vfLen>>((__ubuf__ KeyT*)inputX.GetPhyAddr(),
                                                         (__ubuf__ KeyT*)uInputX.GetPhyAddr(), repeatTime, numTileData);
}

template <typename T, bool isDescend, uint32_t VF_LEN>
__simd_vf__ __aicore__ inline void TwiddleFpVF(__ubuf__ T* src, __ubuf__ T* dst, uint16_t rpt, uint32_t n, T lowestKey,
                                               T xorVal, T minusZero)
{
    AscendC::Reg::RegTensor<T> inputReg, vnotReg;
    AscendC::Reg::RegTensor<T> xorMaskReg, vandMask;
    AscendC::Reg::RegTensor<T> twiddledZeroReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<T>();
    AscendC::Reg::MaskReg xorMask;
    AscendC::Reg::Duplicate(xorMaskReg, lowestKey, mask);
    AscendC::Reg::Duplicate(vandMask, xorVal, mask);
    AscendC::Reg::Duplicate(twiddledZeroReg, minusZero, mask);

    for (uint16_t i = 0; i < rpt; i++) {
        xorMask = AscendC::Reg::UpdateMask<T>(n);
        AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(inputReg, src, VF_LEN);
        AscendC::Reg::RegTensor<T> andValueOne;
        AscendC::Reg::And(andValueOne, inputReg, vandMask, mask);
        AscendC::Reg::MaskReg cmpValueOne;
        AscendC::Reg::Compares<T, AscendC::CMPMODE::NE>(cmpValueOne, andValueOne, 0, mask);
        AscendC::Reg::RegTensor<T> finalMaskOne;
        AscendC::Reg::Select(finalMaskOne, xorMaskReg, vandMask, cmpValueOne);
        AscendC::Reg::RegTensor<T> xorVector;
        AscendC::Reg::Xor(xorVector, inputReg, finalMaskOne, mask);
        AscendC::Reg::MaskReg minusZeroMask;
        AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(minusZeroMask, xorVector, minusZero, mask);
        AscendC::Reg::RegTensor<T> resultReg;
        AscendC::Reg::Select(resultReg, twiddledZeroReg, xorVector, minusZeroMask);

        if constexpr (!isDescend) {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(dst, resultReg, VF_LEN, xorMask);
        } else {
            AscendC::Reg::Not(vnotReg, resultReg, xorMask);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(dst, vnotReg, VF_LEN, xorMask);
        }
    }
}

template <typename ValT, typename KeyT, bool isDescend, typename T>
__aicore__ inline void TwiddleInFp(AscendC::LocalTensor<ValT> inputX, AscendC::LocalTensor<KeyT> uintInputX,
                                   uint32_t numTileData, T lowestKey, T xorValue, T minusZero)
{
    constexpr uint32_t vfLen = 256 / sizeof(T);
    uint16_t repeatTime = AscendC::CeilDivision(numTileData, vfLen);
    asc_vf_call<TwiddleFpVF<T, isDescend, vfLen>>((__ubuf__ T*)inputX.GetPhyAddr(),
                                                  (__ubuf__ T*)uintInputX.GetPhyAddr(), repeatTime, numTileData,
                                                  lowestKey, xorValue, minusZero);
}

template <typename KeyT, uint32_t VF_LEN>
__simd_vf__ __aicore__ inline void ReverseInputVF(__ubuf__ KeyT* src, __ubuf__ KeyT* dst, uint16_t rpt, uint32_t n)
{
    AscendC::Reg::RegTensor<KeyT> inputVectorOne;
    AscendC::Reg::RegTensor<KeyT> vnotVectorZero;
    AscendC::Reg::MaskReg predicateDefault = AscendC::Reg::CreateMask<KeyT>();
    for (uint16_t i = 0; i < rpt; i++) {
        AscendC::Reg::MaskReg vnotMask = AscendC::Reg::UpdateMask<KeyT>(n);
        AscendC::Reg::LoadAlign<KeyT, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(inputVectorOne, src, VF_LEN);
        AscendC::Reg::Not(vnotVectorZero, inputVectorOne, predicateDefault);
        AscendC::Reg::StoreAlign<KeyT, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(dst, vnotVectorZero, VF_LEN,
                                                                                    vnotMask);
    }
}

template <typename ValT, typename KeyT>
__aicore__ inline void ReverseInputData(AscendC::LocalTensor<ValT> inputX, AscendC::LocalTensor<KeyT> uintInputX,
                                        uint32_t numTileData)
{
    constexpr uint32_t vfLen = 256 / sizeof(KeyT);
    uint16_t repeatTime = AscendC::CeilDivision(numTileData, vfLen);
    asc_vf_call<ReverseInputVF<KeyT, vfLen>>((__ubuf__ KeyT*)inputX.GetPhyAddr(),
                                             (__ubuf__ KeyT*)uintInputX.GetPhyAddr(), repeatTime, numTileData);
}

// ── Histogram ──

template <typename CountT>
__simd_callee__ __aicore__ inline void ComputeSumChist(
    AscendC::Reg::RegTensor<uint16_t>& chist0, AscendC::Reg::RegTensor<uint16_t>& chist1,
    AscendC::Reg::RegTensor<uint16_t>& hist0, AscendC::Reg::RegTensor<uint16_t>& hist1, AscendC::Reg::MaskReg& maskB16,
    AscendC::Reg::MaskReg& maskB32, __ubuf__ CountT* blockExclusiveUbRPtr, __ubuf__ CountT* blockExclusiveUbWPtr,
    __ubuf__ uint16_t* histUbPtr, __ubuf__ uint16_t* histCumsumUbPtr)
{
    // chist is inclusive per-tile cumulative histogram. Subtract the current bin count to get the per-bin exclusive
    // offset used by the later scatter phase.
    AscendC::Reg::RegTensor<uint16_t> exclusiveSumZero, exclusiveSumOne, zeroReg;
    AscendC::Reg::Sub(exclusiveSumZero, chist0, hist0, maskB16);
    AscendC::Reg::Sub(exclusiveSumOne, chist1, hist1, maskB16);

    // Persist each tile's histogram and exclusive cumsum. ComputeOnePass reloads these after global bin offsets have
    // been accumulated across tiles/cores.
    AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(histUbPtr, hist0, VF_LEN_B16,
                                                                                    maskB16);
    AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(histUbPtr, hist1, VF_LEN_B16,
                                                                                    maskB16);
    AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(histCumsumUbPtr, exclusiveSumZero,
                                                                                    VF_LEN_B16, maskB16);
    AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(histCumsumUbPtr, exclusiveSumOne,
                                                                                    VF_LEN_B16, maskB16);

    AscendC::Reg::Duplicate(zeroReg, 0, maskB16);
    AscendC::Reg::RegTensor<uint32_t> sum0, sum1, sum2, sum3;
    AscendC::Reg::Interleave((AscendC::Reg::RegTensor<uint16_t>&)sum0, (AscendC::Reg::RegTensor<uint16_t>&)sum1,
                             exclusiveSumZero, zeroReg);
    AscendC::Reg::Interleave((AscendC::Reg::RegTensor<uint16_t>&)sum2, (AscendC::Reg::RegTensor<uint16_t>&)sum3,
                             exclusiveSumOne, zeroReg);
    if constexpr (sizeof(CountT) == sizeof(uint32_t)) {
        AscendC::Reg::RegTensor<uint32_t> sumIn0, sumIn1, sumIn2, sumIn3;
        AscendC::Reg::LoadAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(sumIn0, blockExclusiveUbRPtr,
                                                                                       VF_LEN_B32);
        AscendC::Reg::LoadAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(sumIn1, blockExclusiveUbRPtr,
                                                                                       VF_LEN_B32);
        AscendC::Reg::LoadAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(sumIn2, blockExclusiveUbRPtr,
                                                                                       VF_LEN_B32);
        AscendC::Reg::LoadAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(sumIn3, blockExclusiveUbRPtr,
                                                                                       VF_LEN_B32);
        // Accumulate the current tile's exclusive cumsum into this core's block-level bin totals.
        AscendC::Reg::Add(sumIn0, sumIn0, sum0, maskB32);
        AscendC::Reg::Add(sumIn1, sumIn1, sum1, maskB32);
        AscendC::Reg::Add(sumIn2, sumIn2, sum2, maskB32);
        AscendC::Reg::Add(sumIn3, sumIn3, sum3, maskB32);
        AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, sumIn0,
                                                                                        VF_LEN_B32, maskB32);
        AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, sumIn1,
                                                                                        VF_LEN_B32, maskB32);
        AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, sumIn2,
                                                                                        VF_LEN_B32, maskB32);
        AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, sumIn3,
                                                                                        VF_LEN_B32, maskB32);
    } else {
        AscendC::Reg::MaskReg maskB64 = AscendC::Reg::CreateMask<int64_t>();
        AscendC::Reg::RegTensor<int64_t> sum0Int64, sum1Int64, sum2Int64, sum3Int64;
        AscendC::Reg::RegTensor<int64_t> sum4Int64, sum5Int64, sum6Int64, sum7Int64;
        AscendC::Reg::Interleave((AscendC::Reg::RegTensor<uint32_t>&)sum0Int64,
                                 (AscendC::Reg::RegTensor<uint32_t>&)sum1Int64, sum0,
                                 (AscendC::Reg::RegTensor<uint32_t>&)zeroReg);
        AscendC::Reg::Interleave((AscendC::Reg::RegTensor<uint32_t>&)sum2Int64,
                                 (AscendC::Reg::RegTensor<uint32_t>&)sum3Int64, sum1,
                                 (AscendC::Reg::RegTensor<uint32_t>&)zeroReg);
        AscendC::Reg::Interleave((AscendC::Reg::RegTensor<uint32_t>&)sum4Int64,
                                 (AscendC::Reg::RegTensor<uint32_t>&)sum5Int64, sum2,
                                 (AscendC::Reg::RegTensor<uint32_t>&)zeroReg);
        AscendC::Reg::Interleave((AscendC::Reg::RegTensor<uint32_t>&)sum6Int64,
                                 (AscendC::Reg::RegTensor<uint32_t>&)sum7Int64, sum3,
                                 (AscendC::Reg::RegTensor<uint32_t>&)zeroReg);
        AscendC::Reg::RegTensor<int64_t> int64In0, int64In1, int64In2, int64In3;
        AscendC::Reg::RegTensor<int64_t> int64In4, int64In5, int64In6, int64In7;
        AscendC::Reg::LoadAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(int64In0, blockExclusiveUbRPtr,
                                                                                      VF_LEN_B64);
        AscendC::Reg::LoadAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(int64In1, blockExclusiveUbRPtr,
                                                                                      VF_LEN_B64);
        AscendC::Reg::LoadAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(int64In2, blockExclusiveUbRPtr,
                                                                                      VF_LEN_B64);
        AscendC::Reg::LoadAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(int64In3, blockExclusiveUbRPtr,
                                                                                      VF_LEN_B64);
        AscendC::Reg::LoadAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(int64In4, blockExclusiveUbRPtr,
                                                                                      VF_LEN_B64);
        AscendC::Reg::LoadAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(int64In5, blockExclusiveUbRPtr,
                                                                                      VF_LEN_B64);
        AscendC::Reg::LoadAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(int64In6, blockExclusiveUbRPtr,
                                                                                      VF_LEN_B64);
        AscendC::Reg::LoadAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(int64In7, blockExclusiveUbRPtr,
                                                                                      VF_LEN_B64);
        // Accumulate the current tile's exclusive cumsum into this core's block-level bin totals.
        AscendC::Reg::Add(int64In0, int64In0, sum0Int64, maskB64);
        AscendC::Reg::Add(int64In1, int64In1, sum1Int64, maskB64);
        AscendC::Reg::Add(int64In2, int64In2, sum2Int64, maskB64);
        AscendC::Reg::Add(int64In3, int64In3, sum3Int64, maskB64);
        AscendC::Reg::Add(int64In4, int64In4, sum4Int64, maskB64);
        AscendC::Reg::Add(int64In5, int64In5, sum5Int64, maskB64);
        AscendC::Reg::Add(int64In6, int64In6, sum6Int64, maskB64);
        AscendC::Reg::Add(int64In7, int64In7, sum7Int64, maskB64);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, int64In0,
                                                                                       VF_LEN_B64, maskB64);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, int64In1,
                                                                                       VF_LEN_B64, maskB64);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, int64In2,
                                                                                       VF_LEN_B64, maskB64);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, int64In3,
                                                                                       VF_LEN_B64, maskB64);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, int64In4,
                                                                                       VF_LEN_B64, maskB64);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, int64In5,
                                                                                       VF_LEN_B64, maskB64);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, int64In6,
                                                                                       VF_LEN_B64, maskB64);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, int64In7,
                                                                                       VF_LEN_B64, maskB64);
    }
}

template <typename KeyT, typename CountT>
__simd_vf__ __aicore__ inline void HistogramB32VF(__ubuf__ KeyT* inP, __ubuf__ CountT* ebW, __ubuf__ uint16_t* hp,
                                                  __ubuf__ uint16_t* cp, __ubuf__ uint8_t* b8p, uint32_t sz,
                                                  uint16_t repeatTime, uint32_t round)
{
    uint32_t bitOffset = round * SHIFT_BIT_NUM;
    __ubuf__ CountT* blockExclusiveUbWPtr = ebW;
    __ubuf__ CountT* blockExclusiveUbRPtr = ebW;
    uint32_t inputElementNum = sz;
    AscendC::Reg::RegTensor<uint32_t> in0, in1, in2, in3;
    AscendC::Reg::RegTensor<uint16_t> hist0, hist1, chist0, chist1;
    AscendC::Reg::MaskReg histMask;
    AscendC::Reg::MaskReg maskB32 = AscendC::Reg::CreateMask<uint32_t>();
    AscendC::Reg::MaskReg maskB16 = AscendC::Reg::CreateMask<uint16_t>();
    AscendC::Reg::Duplicate(hist0, 0, maskB16);
    AscendC::Reg::Duplicate(hist1, 0, maskB16);
    AscendC::Reg::Duplicate(chist0, 0, maskB16);
    AscendC::Reg::Duplicate(chist1, 0, maskB16);
    for (uint16_t i = 0; i < repeatTime; i++) {
        histMask = AscendC::Reg::UpdateMask<uint8_t>(inputElementNum);
        AscendC::Reg::LoadAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(in0, inP, VF_LEN_B32);
        AscendC::Reg::LoadAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(in1, inP, VF_LEN_B32);
        AscendC::Reg::LoadAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(in2, inP, VF_LEN_B32);
        AscendC::Reg::LoadAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(in3, inP, VF_LEN_B32);
        AscendC::Reg::RegTensor<uint32_t> shift0, shift1, shift2, shift3;
        AscendC::Reg::ShiftRights<uint32_t, int16_t>(shift0, in0, bitOffset, maskB32);
        AscendC::Reg::ShiftRights<uint32_t, int16_t>(shift1, in1, bitOffset, maskB32);
        AscendC::Reg::ShiftRights<uint32_t, int16_t>(shift2, in2, bitOffset, maskB32);
        AscendC::Reg::ShiftRights<uint32_t, int16_t>(shift3, in3, bitOffset, maskB32);
        AscendC::Reg::RegTensor<uint16_t> deInter0, deInter1, deInter2, deInter3;
        AscendC::Reg::DeInterleave(deInter0, deInter1, (AscendC::Reg::RegTensor<uint16_t>&)shift0,
                                   (AscendC::Reg::RegTensor<uint16_t>&)shift1);
        AscendC::Reg::DeInterleave(deInter2, deInter3, (AscendC::Reg::RegTensor<uint16_t>&)shift2,
                                   (AscendC::Reg::RegTensor<uint16_t>&)shift3);
        AscendC::Reg::RegTensor<uint8_t> deInter0B8, deInter1B8;
        AscendC::Reg::DeInterleave(deInter0B8, deInter1B8, (AscendC::Reg::RegTensor<uint8_t>&)deInter0,
                                   (AscendC::Reg::RegTensor<uint8_t>&)deInter2);
        AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(b8p, deInter0B8, VF_LEN_B8,
                                                                                       histMask);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN0,
                                 AscendC::Reg::HistogramsType::FREQUENCY>(hist0, deInter0B8, histMask);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN1,
                                 AscendC::Reg::HistogramsType::FREQUENCY>(hist1, deInter0B8, histMask);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN0,
                                 AscendC::Reg::HistogramsType::ACCUMULATE>(chist0, deInter0B8, histMask);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN1,
                                 AscendC::Reg::HistogramsType::ACCUMULATE>(chist1, deInter0B8, histMask);
    }
    ComputeSumChist<CountT>(chist0, chist1, hist0, hist1, maskB16, maskB32, blockExclusiveUbRPtr, blockExclusiveUbWPtr,
                            hp, cp);
}

template <typename KeyT, typename CountT>
__simd_vf__ __aicore__ inline void HistogramB64VF(__ubuf__ KeyT* inP, __ubuf__ CountT* ebW, __ubuf__ uint16_t* hp,
                                                  __ubuf__ uint16_t* cp, __ubuf__ uint8_t* b8p, uint32_t sz,
                                                  uint16_t repeatTime, uint32_t round)
{
    uint32_t bitOffset = round * SHIFT_BIT_NUM;
    __ubuf__ CountT* blockExclusiveUbWPtr = ebW;
    __ubuf__ CountT* blockExclusiveUbRPtr = ebW;
    uint32_t inputElementNum = sz;
    AscendC::Reg::RegTensor<uint64_t> in0, in1, in2, in3, in4, in5, in6, in7;
    AscendC::Reg::RegTensor<uint16_t> hist0, hist1, chist0, chist1;
    AscendC::Reg::MaskReg histMask;
    AscendC::Reg::MaskReg maskB64 = AscendC::Reg::CreateMask<uint64_t>();
    AscendC::Reg::MaskReg maskB32 = AscendC::Reg::CreateMask<uint32_t>();
    AscendC::Reg::MaskReg maskB16 = AscendC::Reg::CreateMask<uint16_t>();
    AscendC::Reg::Duplicate(hist0, 0, maskB16);
    AscendC::Reg::Duplicate(hist1, 0, maskB16);
    AscendC::Reg::Duplicate(chist0, 0, maskB16);
    AscendC::Reg::Duplicate(chist1, 0, maskB16);
    for (uint16_t i = 0; i < repeatTime; i++) {
        histMask = AscendC::Reg::UpdateMask<uint8_t>(inputElementNum);
        AscendC::Reg::LoadAlign<uint64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(in0, inP, VF_LEN_B64);
        AscendC::Reg::LoadAlign<uint64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(in1, inP, VF_LEN_B64);
        AscendC::Reg::LoadAlign<uint64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(in2, inP, VF_LEN_B64);
        AscendC::Reg::LoadAlign<uint64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(in3, inP, VF_LEN_B64);
        AscendC::Reg::LoadAlign<uint64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(in4, inP, VF_LEN_B64);
        AscendC::Reg::LoadAlign<uint64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(in5, inP, VF_LEN_B64);
        AscendC::Reg::LoadAlign<uint64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(in6, inP, VF_LEN_B64);
        AscendC::Reg::LoadAlign<uint64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(in7, inP, VF_LEN_B64);
        AscendC::Reg::RegTensor<uint64_t> shift0, shift1, shift2, shift3, shift4, shift5, shift6, shift7;
        AscendC::Reg::ShiftRights<uint64_t, int16_t>(shift0, in0, bitOffset, maskB64);
        AscendC::Reg::ShiftRights<uint64_t, int16_t>(shift1, in1, bitOffset, maskB64);
        AscendC::Reg::ShiftRights<uint64_t, int16_t>(shift2, in2, bitOffset, maskB64);
        AscendC::Reg::ShiftRights<uint64_t, int16_t>(shift3, in3, bitOffset, maskB64);
        AscendC::Reg::ShiftRights<uint64_t, int16_t>(shift4, in4, bitOffset, maskB64);
        AscendC::Reg::ShiftRights<uint64_t, int16_t>(shift5, in5, bitOffset, maskB64);
        AscendC::Reg::ShiftRights<uint64_t, int16_t>(shift6, in6, bitOffset, maskB64);
        AscendC::Reg::ShiftRights<uint64_t, int16_t>(shift7, in7, bitOffset, maskB64);
        AscendC::Reg::RegTensor<uint32_t> deInter0, deInter1, deInter2, deInter3, deInter4, deInter5, deInter6,
            deInter7;
        AscendC::Reg::DeInterleave(deInter0, deInter1, (AscendC::Reg::RegTensor<uint32_t>&)shift0,
                                   (AscendC::Reg::RegTensor<uint32_t>&)shift1);
        AscendC::Reg::DeInterleave(deInter2, deInter3, (AscendC::Reg::RegTensor<uint32_t>&)shift2,
                                   (AscendC::Reg::RegTensor<uint32_t>&)shift3);
        AscendC::Reg::DeInterleave(deInter4, deInter5, (AscendC::Reg::RegTensor<uint32_t>&)shift4,
                                   (AscendC::Reg::RegTensor<uint32_t>&)shift5);
        AscendC::Reg::DeInterleave(deInter6, deInter7, (AscendC::Reg::RegTensor<uint32_t>&)shift6,
                                   (AscendC::Reg::RegTensor<uint32_t>&)shift7);
        AscendC::Reg::RegTensor<uint16_t> deInter0B16, deInter1B16, deInter2B16, deInter3B16;
        AscendC::Reg::DeInterleave(deInter0B16, deInter1B16, (AscendC::Reg::RegTensor<uint16_t>&)deInter0,
                                   (AscendC::Reg::RegTensor<uint16_t>&)deInter2);
        AscendC::Reg::DeInterleave(deInter2B16, deInter3B16, (AscendC::Reg::RegTensor<uint16_t>&)deInter4,
                                   (AscendC::Reg::RegTensor<uint16_t>&)deInter6);
        AscendC::Reg::RegTensor<uint8_t> deInter0B8, deInter1B8;
        AscendC::Reg::DeInterleave(deInter0B8, deInter1B8, (AscendC::Reg::RegTensor<uint8_t>&)deInter0B16,
                                   (AscendC::Reg::RegTensor<uint8_t>&)deInter2B16);
        AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(b8p, deInter0B8, VF_LEN_B8,
                                                                                       histMask);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN0,
                                 AscendC::Reg::HistogramsType::FREQUENCY>(hist0, deInter0B8, histMask);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN1,
                                 AscendC::Reg::HistogramsType::FREQUENCY>(hist1, deInter0B8, histMask);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN0,
                                 AscendC::Reg::HistogramsType::ACCUMULATE>(chist0, deInter0B8, histMask);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN1,
                                 AscendC::Reg::HistogramsType::ACCUMULATE>(chist1, deInter0B8, histMask);
    }
    ComputeSumChist<CountT>(chist0, chist1, hist0, hist1, maskB16, maskB32, blockExclusiveUbRPtr, blockExclusiveUbWPtr,
                            hp, cp);
}

template <typename KeyT, typename CountT>
__simd_vf__ __aicore__ inline void HistogramB16VF(__ubuf__ KeyT* inP, __ubuf__ CountT* ebW, __ubuf__ uint16_t* hp,
                                                  __ubuf__ uint16_t* cp, __ubuf__ uint8_t* b8p, uint32_t sz,
                                                  uint16_t repeatTime, uint32_t round)
{
    uint32_t bitOffset = round * SHIFT_BIT_NUM;
    __ubuf__ CountT* blockExclusiveUbWPtr = ebW;
    __ubuf__ CountT* blockExclusiveUbRPtr = ebW;
    uint32_t inputElementNum = sz;
    AscendC::Reg::MaskReg histMask;
    AscendC::Reg::RegTensor<uint16_t> in0, in1;
    AscendC::Reg::RegTensor<uint16_t> shift0, shift1;
    AscendC::Reg::RegTensor<uint16_t> hist0, hist1, chist0, chist1;
    AscendC::Reg::MaskReg maskB32 = AscendC::Reg::CreateMask<uint32_t>();
    AscendC::Reg::MaskReg maskB16 = AscendC::Reg::CreateMask<uint16_t>();
    AscendC::Reg::Duplicate(hist0, 0, maskB16);
    AscendC::Reg::Duplicate(hist1, 0, maskB16);
    AscendC::Reg::Duplicate(chist0, 0, maskB16);
    AscendC::Reg::Duplicate(chist1, 0, maskB16);
    for (uint16_t i = 0; i < repeatTime; i++) {
        histMask = AscendC::Reg::UpdateMask<uint8_t>(inputElementNum);
        AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(in0, inP, VF_LEN_B16);
        AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(in1, inP, VF_LEN_B16);
        AscendC::Reg::ShiftRights<uint16_t, int16_t>(shift0, in0, bitOffset, maskB16);
        AscendC::Reg::ShiftRights<uint16_t, int16_t>(shift1, in1, bitOffset, maskB16);
        AscendC::Reg::RegTensor<uint8_t> deInter0B8, deInter1B8;
        AscendC::Reg::DeInterleave(deInter0B8, deInter1B8, (AscendC::Reg::RegTensor<uint8_t>&)shift0,
                                   (AscendC::Reg::RegTensor<uint8_t>&)shift1);
        AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(b8p, deInter0B8, VF_LEN_B8,
                                                                                       histMask);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN0,
                                 AscendC::Reg::HistogramsType::FREQUENCY>(hist0, deInter0B8, histMask);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN1,
                                 AscendC::Reg::HistogramsType::FREQUENCY>(hist1, deInter0B8, histMask);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN0,
                                 AscendC::Reg::HistogramsType::ACCUMULATE>(chist0, deInter0B8, histMask);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN1,
                                 AscendC::Reg::HistogramsType::ACCUMULATE>(chist1, deInter0B8, histMask);
    }
    ComputeSumChist<CountT>(chist0, chist1, hist0, hist1, maskB16, maskB32, blockExclusiveUbRPtr, blockExclusiveUbWPtr,
                            hp, cp);
}

template <typename KeyT, typename CountT>
__simd_vf__ __aicore__ inline void HistogramB8VF(__ubuf__ KeyT* inP, __ubuf__ CountT* ebW, __ubuf__ uint16_t* hp,
                                                 __ubuf__ uint16_t* cp, __ubuf__ uint8_t* b8p, uint32_t sz,
                                                 uint16_t repeatTime, uint32_t round)
{
    (void)round;
    __ubuf__ CountT* blockExclusiveUbWPtr = ebW;
    __ubuf__ CountT* blockExclusiveUbRPtr = ebW;
    uint32_t inputElementNum = sz;
    AscendC::Reg::MaskReg histMask;
    AscendC::Reg::RegTensor<uint8_t> in0;
    AscendC::Reg::RegTensor<uint16_t> hist0, hist1, chist0, chist1;
    AscendC::Reg::MaskReg maskB32 = AscendC::Reg::CreateMask<uint32_t>();
    AscendC::Reg::MaskReg maskB16 = AscendC::Reg::CreateMask<uint16_t>();
    AscendC::Reg::Duplicate(hist0, 0, maskB16);
    AscendC::Reg::Duplicate(hist1, 0, maskB16);
    AscendC::Reg::Duplicate(chist0, 0, maskB16);
    AscendC::Reg::Duplicate(chist1, 0, maskB16);
    for (uint16_t i = 0; i < repeatTime; i++) {
        histMask = AscendC::Reg::UpdateMask<uint8_t>(inputElementNum);
        AscendC::Reg::LoadAlign<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(in0, inP, VF_LEN_B8);
        AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(b8p, in0, VF_LEN_B8, histMask);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN0,
                                 AscendC::Reg::HistogramsType::FREQUENCY>(hist0, in0, histMask);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN1,
                                 AscendC::Reg::HistogramsType::FREQUENCY>(hist1, in0, histMask);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN0,
                                 AscendC::Reg::HistogramsType::ACCUMULATE>(chist0, in0, histMask);
        AscendC::Reg::Histograms<uint8_t, uint16_t, AscendC::Reg::HistogramsBinType::BIN1,
                                 AscendC::Reg::HistogramsType::ACCUMULATE>(chist1, in0, histMask);
    }
    ComputeSumChist<CountT>(chist0, chist1, hist0, hist1, maskB16, maskB32, blockExclusiveUbRPtr, blockExclusiveUbWPtr,
                            hp, cp);
}

// ── Scatter / Lookback / Prefix ──

template <typename CountT>
__simd_vf__ __aicore__ inline void EncodeHistVF(__ubuf__ uint16_t* blockHistPtr, __ubuf__ CountT* blockHistWithFlagPtr)
{
    AscendC::Reg::MaskReg predicateDefaultB16 = AscendC::Reg::CreateMask<uint16_t>();
    AscendC::Reg::RegTensor<uint16_t> blockHistZero, blockHistOne;
    AscendC::Reg::RegTensor<uint16_t> lookaheadOutZero, lookaheadOutOne, lookaheadOutTwo, lookaheadOutThree;
    AscendC::Reg::RegTensor<uint16_t> zeroVector;
    AscendC::Reg::Duplicate(zeroVector, 0, predicateDefaultB16);
    AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(blockHistZero, blockHistPtr,
                                                                                   VF_LEN_B16);
    AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(blockHistOne, blockHistPtr,
                                                                                   VF_LEN_B16);
    AscendC::Reg::Interleave(lookaheadOutZero, lookaheadOutOne, blockHistZero, zeroVector);
    AscendC::Reg::Interleave(lookaheadOutTwo, lookaheadOutThree, blockHistOne, zeroVector);
    if constexpr (AscendC::IsSameType<CountT, uint32_t>::value) {
        AscendC::Reg::RegTensor<uint32_t> aggregateReadyMask;
        AscendC::Reg::MaskReg predicateDefault = AscendC::Reg::CreateMask<uint32_t>();
        AscendC::Reg::Duplicate(aggregateReadyMask, AGGREGATE_READY_MASK, predicateDefault);
        AscendC::Reg::RegTensor<uint32_t> lookaheadOutZeroMask, lookaheadOutOneMask, lookaheadOutTwoMask,
            lookaheadOutThreeMask;
        AscendC::Reg::Or(lookaheadOutZeroMask, (AscendC::Reg::RegTensor<uint32_t>&)lookaheadOutZero, aggregateReadyMask,
                         predicateDefault);
        AscendC::Reg::Or(lookaheadOutOneMask, (AscendC::Reg::RegTensor<uint32_t>&)lookaheadOutOne, aggregateReadyMask,
                         predicateDefault);
        AscendC::Reg::Or(lookaheadOutTwoMask, (AscendC::Reg::RegTensor<uint32_t>&)lookaheadOutTwo, aggregateReadyMask,
                         predicateDefault);
        AscendC::Reg::Or(lookaheadOutThreeMask, (AscendC::Reg::RegTensor<uint32_t>&)lookaheadOutThree,
                         aggregateReadyMask, predicateDefault);
        AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            blockHistWithFlagPtr, lookaheadOutZeroMask, VF_LEN_B32, predicateDefault);
        AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            blockHistWithFlagPtr, lookaheadOutOneMask, VF_LEN_B32, predicateDefault);
        AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            blockHistWithFlagPtr, lookaheadOutTwoMask, VF_LEN_B32, predicateDefault);
        AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            blockHistWithFlagPtr, lookaheadOutThreeMask, VF_LEN_B32, predicateDefault);
    } else {
        AscendC::Reg::RegTensor<int64_t> aggregateReadyMask;
        AscendC::Reg::MaskReg predicateDefault = AscendC::Reg::CreateMask<int64_t>();
        AscendC::Reg::Duplicate(aggregateReadyMask, AGGREGATE_READY_MASK_B64, predicateDefault);
        AscendC::Reg::RegTensor<uint16_t> lookaheadOutZeroB64A, lookaheadOutZeroB64B;
        AscendC::Reg::RegTensor<uint16_t> lookaheadOutOneB64A, lookaheadOutOneB64B;
        AscendC::Reg::RegTensor<uint16_t> lookaheadOutTwoB64A, lookaheadOutTwoB64B;
        AscendC::Reg::RegTensor<uint16_t> lookaheadOutThreeB64A, lookaheadOutThreeB64B;
        AscendC::Reg::Interleave(lookaheadOutZeroB64A, lookaheadOutZeroB64B, lookaheadOutZero, zeroVector);
        AscendC::Reg::Interleave(lookaheadOutOneB64A, lookaheadOutOneB64B, lookaheadOutOne, zeroVector);
        AscendC::Reg::Interleave(lookaheadOutTwoB64A, lookaheadOutTwoB64B, lookaheadOutTwo, zeroVector);
        AscendC::Reg::Interleave(lookaheadOutThreeB64A, lookaheadOutThreeB64B, lookaheadOutThree, zeroVector);
        AscendC::Reg::RegTensor<int64_t> lookaheadOutZeroMaskB64A, lookaheadOutZeroMaskB64B;
        AscendC::Reg::RegTensor<int64_t> lookaheadOutOneMaskB64A, lookaheadOutOneMaskB64B;
        AscendC::Reg::RegTensor<int64_t> lookaheadOutTwoMaskB64A, lookaheadOutTwoMaskB64B;
        AscendC::Reg::RegTensor<int64_t> lookaheadOutThreeMaskB64A, lookaheadOutThreeMaskB64B;
        AscendC::Reg::Or(lookaheadOutZeroMaskB64A, (AscendC::Reg::RegTensor<int64_t>&)lookaheadOutZeroB64A,
                         aggregateReadyMask, predicateDefault);
        AscendC::Reg::Or(lookaheadOutZeroMaskB64B, (AscendC::Reg::RegTensor<int64_t>&)lookaheadOutZeroB64B,
                         aggregateReadyMask, predicateDefault);
        AscendC::Reg::Or(lookaheadOutOneMaskB64A, (AscendC::Reg::RegTensor<int64_t>&)lookaheadOutOneB64A,
                         aggregateReadyMask, predicateDefault);
        AscendC::Reg::Or(lookaheadOutOneMaskB64B, (AscendC::Reg::RegTensor<int64_t>&)lookaheadOutOneB64B,
                         aggregateReadyMask, predicateDefault);
        AscendC::Reg::Or(lookaheadOutTwoMaskB64A, (AscendC::Reg::RegTensor<int64_t>&)lookaheadOutTwoB64A,
                         aggregateReadyMask, predicateDefault);
        AscendC::Reg::Or(lookaheadOutTwoMaskB64B, (AscendC::Reg::RegTensor<int64_t>&)lookaheadOutTwoB64B,
                         aggregateReadyMask, predicateDefault);
        AscendC::Reg::Or(lookaheadOutThreeMaskB64A, (AscendC::Reg::RegTensor<int64_t>&)lookaheadOutThreeB64A,
                         aggregateReadyMask, predicateDefault);
        AscendC::Reg::Or(lookaheadOutThreeMaskB64B, (AscendC::Reg::RegTensor<int64_t>&)lookaheadOutThreeB64B,
                         aggregateReadyMask, predicateDefault);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            blockHistWithFlagPtr, lookaheadOutZeroMaskB64A, VF_LEN_B64, predicateDefault);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            blockHistWithFlagPtr, lookaheadOutZeroMaskB64B, VF_LEN_B64, predicateDefault);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            blockHistWithFlagPtr, lookaheadOutOneMaskB64A, VF_LEN_B64, predicateDefault);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            blockHistWithFlagPtr, lookaheadOutOneMaskB64B, VF_LEN_B64, predicateDefault);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            blockHistWithFlagPtr, lookaheadOutTwoMaskB64A, VF_LEN_B64, predicateDefault);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            blockHistWithFlagPtr, lookaheadOutTwoMaskB64B, VF_LEN_B64, predicateDefault);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            blockHistWithFlagPtr, lookaheadOutThreeMaskB64A, VF_LEN_B64, predicateDefault);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            blockHistWithFlagPtr, lookaheadOutThreeMaskB64B, VF_LEN_B64, predicateDefault);
    }
}

template <typename CountT>
__simd_vf__ __aicore__ inline void LookbackCheckStateVF(__ubuf__ CountT* tilePrevHistValuePtr,
                                                        __ubuf__ uint32_t* ubFlagTensorPtr, uint16_t repeatTime)
{
    AscendC::Reg::MaskReg pRegSelect;
    AscendC::Reg::MaskReg maskB32 = AscendC::Reg::CreateMask<uint32_t>();
    AscendC::Reg::RegTensor<uint32_t> notInitCount, aggReadyCount, prefixReadyCount;
    AscendC::Reg::Duplicate(notInitCount, 0, maskB32);
    AscendC::Reg::Duplicate(aggReadyCount, 0, maskB32);
    AscendC::Reg::Duplicate(prefixReadyCount, 0, maskB32);
    AscendC::Reg::RegTensor<uint32_t> onesVector, zerosVector;
    AscendC::Reg::Duplicate(onesVector, 1, maskB32);
    AscendC::Reg::Duplicate(zerosVector, 0, maskB32);
    for (uint16_t i = 0; i < repeatTime; i++) {
        AscendC::Reg::RegTensor<CountT> prevTileHistValue;
        AscendC::Reg::RegTensor<uint32_t> stateBitValue;
        if constexpr (AscendC::IsSameType<CountT, uint32_t>::value) {
            AscendC::Reg::LoadAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                prevTileHistValue, tilePrevHistValuePtr, VF_LEN_B32);
            AscendC::Reg::ShiftRights<uint32_t, int16_t>(stateBitValue, prevTileHistValue, STATE_BIT_SHF_VALUE,
                                                         maskB32);
            pRegSelect = maskB32;
        } else {
            AscendC::Reg::LoadAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                prevTileHistValue, tilePrevHistValuePtr, VF_LEN_B64);
            AscendC::Reg::MaskReg maskB64 = AscendC::Reg::CreateMask<int64_t>();
            AscendC::Reg::RegTensor<uint64_t> stateTmp;
            AscendC::Reg::ShiftRights<uint64_t, int16_t>(
                stateTmp, (AscendC::Reg::RegTensor<uint64_t>&)prevTileHistValue, STATE_BIT_SHF_VALUE_B64, maskB64);
            AscendC::Reg::Pack(stateBitValue, stateTmp);
            pRegSelect = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::H>();
        }
        AscendC::Reg::MaskReg maskNotInit;
        AscendC::Reg::RegTensor<uint32_t> maskNotInitCount;
        AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::EQ>(maskNotInit, stateBitValue, NOT_INIT_MODE, pRegSelect);
        AscendC::Reg::Select(maskNotInitCount, onesVector, zerosVector, maskNotInit);
        AscendC::Reg::Add(notInitCount, notInitCount, maskNotInitCount, maskNotInit);
        AscendC::Reg::MaskReg maskAggReady;
        AscendC::Reg::RegTensor<uint32_t> maskAggReadyCount;
        AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::EQ>(maskAggReady, stateBitValue, AGG_READY_MODE, pRegSelect);
        AscendC::Reg::Select(maskAggReadyCount, onesVector, zerosVector, maskAggReady);
        AscendC::Reg::Add(aggReadyCount, aggReadyCount, maskAggReadyCount, maskAggReady);
        AscendC::Reg::MaskReg maskPrefixReady;
        AscendC::Reg::RegTensor<uint32_t> maskPrefixReadyCount;
        AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::EQ>(maskPrefixReady, stateBitValue, PREFIX_READY_MODE,
                                                               pRegSelect);
        AscendC::Reg::Select(maskPrefixReadyCount, onesVector, zerosVector, maskPrefixReady);
        AscendC::Reg::Add(prefixReadyCount, prefixReadyCount, maskPrefixReadyCount, maskPrefixReady);
    }
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(notInitCount, notInitCount, maskB32);
    AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                             AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(ubFlagTensorPtr, notInitCount,
                                                                              HIST_MASK_OUT_LEN, maskB32);
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(aggReadyCount, aggReadyCount, maskB32);
    AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                             AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(ubFlagTensorPtr, aggReadyCount,
                                                                              HIST_MASK_OUT_LEN, maskB32);
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(prefixReadyCount, prefixReadyCount, maskB32);
    AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                             AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(ubFlagTensorPtr, prefixReadyCount,
                                                                              HIST_MASK_OUT_LEN, maskB32);
}

template <typename CountT>
__simd_vf__ __aicore__ inline void LookbackAccumVF(__ubuf__ CountT* nowTileHistBufferPtr,
                                                   __ubuf__ CountT* nowTileHistBufferPtrCopy,
                                                   __ubuf__ CountT* tilePrevHistValuePtrCopy, uint16_t repeatTime)
{
    AscendC::Reg::MaskReg predicateDefault = AscendC::Reg::CreateMask<uint16_t>();
    AscendC::Reg::RegTensor<CountT> histMask;
    if constexpr (AscendC::IsSameType<CountT, uint32_t>::value) {
        AscendC::Reg::Duplicate(histMask, VALUE_MASK, predicateDefault);
        for (uint16_t i = 0; i < repeatTime; i++) {
            AscendC::Reg::RegTensor<uint32_t> nowTileHistVal, prevTileHistVal;
            AscendC::Reg::LoadAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                nowTileHistVal, nowTileHistBufferPtr, VF_LEN_B32);
            AscendC::Reg::LoadAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                prevTileHistVal, tilePrevHistValuePtrCopy, VF_LEN_B32);
            AscendC::Reg::And(nowTileHistVal, nowTileHistVal, histMask, predicateDefault);
            AscendC::Reg::And(prevTileHistVal, prevTileHistVal, histMask, predicateDefault);
            AscendC::Reg::Add(nowTileHistVal, nowTileHistVal, prevTileHistVal, predicateDefault);
            AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                nowTileHistBufferPtrCopy, nowTileHistVal, VF_LEN_B32, predicateDefault);
        }
    } else {
        AscendC::Reg::Duplicate(histMask, VALUE_MASK_B64, predicateDefault);
        for (uint16_t i = 0; i < repeatTime; i++) {
            AscendC::Reg::RegTensor<int64_t> nowTileHistVal, prevTileHistVal;
            AscendC::Reg::LoadAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                nowTileHistVal, nowTileHistBufferPtr, VF_LEN_B64);
            AscendC::Reg::LoadAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                prevTileHistVal, tilePrevHistValuePtrCopy, VF_LEN_B64);
            AscendC::Reg::And(nowTileHistVal, nowTileHistVal, histMask, predicateDefault);
            AscendC::Reg::And(prevTileHistVal, prevTileHistVal, histMask, predicateDefault);
            AscendC::Reg::Add(nowTileHistVal, nowTileHistVal, prevTileHistVal, predicateDefault);
            AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                nowTileHistBufferPtrCopy, nowTileHistVal, VF_LEN_B64, predicateDefault);
        }
    }
}

template <typename CountT>
__simd_vf__ __aicore__ inline void SetPrefixReadyVF(__ubuf__ CountT* histCumsumPtr, __ubuf__ CountT* histCumsumPtrCopy,
                                                    uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<CountT> prefixReadyMask, prefixRemainMask;
    AscendC::Reg::MaskReg predicateDefault = AscendC::Reg::CreateMask<CountT>();
    if constexpr (AscendC::IsSameType<CountT, uint32_t>::value) {
        AscendC::Reg::Duplicate(prefixReadyMask, PREFIX_READY_MASK, predicateDefault);
        AscendC::Reg::Duplicate(prefixRemainMask, VALUE_MASK, predicateDefault);
        for (uint16_t repate = 0; repate < repeatTime; repate++) {
            AscendC::Reg::RegTensor<uint32_t> keyCumsumValue;
            AscendC::Reg::LoadAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(keyCumsumValue,
                                                                                           histCumsumPtr, VF_LEN_B32);
            AscendC::Reg::And(keyCumsumValue, keyCumsumValue, prefixRemainMask, predicateDefault);
            AscendC::Reg::Or(keyCumsumValue, keyCumsumValue, prefixReadyMask, predicateDefault);
            AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                histCumsumPtrCopy, keyCumsumValue, VF_LEN_B32, predicateDefault);
        }
    } else {
        AscendC::Reg::Duplicate(prefixReadyMask, PREFIX_READY_MASK_B64, predicateDefault);
        AscendC::Reg::Duplicate(prefixRemainMask, VALUE_MASK_B64, predicateDefault);
        for (uint16_t repate = 0; repate < repeatTime; repate++) {
            AscendC::Reg::RegTensor<int64_t> keyCumsumValue;
            AscendC::Reg::LoadAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(keyCumsumValue, histCumsumPtr,
                                                                                          VF_LEN_B64);
            AscendC::Reg::And(keyCumsumValue, keyCumsumValue, prefixRemainMask, predicateDefault);
            AscendC::Reg::Or(keyCumsumValue, keyCumsumValue, prefixReadyMask, predicateDefault);
            AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                histCumsumPtrCopy, keyCumsumValue, VF_LEN_B64, predicateDefault);
        }
    }
}

// ── Histogram wrappers ──

template <typename KeyT, typename CountT>
__aicore__ inline void GetGlobalExclusiveSumB32(AscendC::LocalTensor<KeyT>& inputX,
                                                AscendC::LocalTensor<CountT>& blockExclusiveUb,
                                                AscendC::LocalTensor<uint16_t>& histUb,
                                                AscendC::LocalTensor<uint16_t>& histCumsumUb,
                                                AscendC::LocalTensor<uint8_t>& inputB8Ub, uint32_t currTileSize,
                                                uint32_t round)
{
    uint16_t repeatTime = AscendC::CeilDivision(currTileSize, VF_LEN_B8);
    asc_vf_call<HistogramB32VF<KeyT, CountT>>(
        (__ubuf__ KeyT*)inputX.GetPhyAddr(), (__ubuf__ CountT*)blockExclusiveUb.GetPhyAddr(),
        (__ubuf__ uint16_t*)histUb.GetPhyAddr(), (__ubuf__ uint16_t*)histCumsumUb.GetPhyAddr(),
        (__ubuf__ uint8_t*)inputB8Ub.GetPhyAddr(), currTileSize, repeatTime, round);
}

template <typename KeyT, typename CountT>
__aicore__ inline void GetGlobalExclusiveSumB64(AscendC::LocalTensor<KeyT>& inputX,
                                                AscendC::LocalTensor<CountT>& blockExclusiveUb,
                                                AscendC::LocalTensor<uint16_t>& histUb,
                                                AscendC::LocalTensor<uint16_t>& histCumsumUb,
                                                AscendC::LocalTensor<uint8_t>& inputB8Ub, uint32_t currTileSize,
                                                uint32_t round)
{
    uint16_t repeatTime = AscendC::CeilDivision(currTileSize, VF_LEN_B8);
    asc_vf_call<HistogramB64VF<KeyT, CountT>>(
        (__ubuf__ KeyT*)inputX.GetPhyAddr(), (__ubuf__ CountT*)blockExclusiveUb.GetPhyAddr(),
        (__ubuf__ uint16_t*)histUb.GetPhyAddr(), (__ubuf__ uint16_t*)histCumsumUb.GetPhyAddr(),
        (__ubuf__ uint8_t*)inputB8Ub.GetPhyAddr(), currTileSize, repeatTime, round);
}

template <typename KeyT, typename CountT>
__aicore__ inline void GetGlobalExclusiveSumB16(AscendC::LocalTensor<KeyT>& inputX,
                                                AscendC::LocalTensor<CountT>& blockExclusiveUb,
                                                AscendC::LocalTensor<uint16_t>& histUb,
                                                AscendC::LocalTensor<uint16_t>& histCumsumUb,
                                                AscendC::LocalTensor<uint8_t>& inputB8Ub, uint32_t currTileSize,
                                                uint32_t round)
{
    uint16_t repeatTime = AscendC::CeilDivision(currTileSize, VF_LEN_B8);
    asc_vf_call<HistogramB16VF<KeyT, CountT>>(
        (__ubuf__ KeyT*)inputX.GetPhyAddr(), (__ubuf__ CountT*)blockExclusiveUb.GetPhyAddr(),
        (__ubuf__ uint16_t*)histUb.GetPhyAddr(), (__ubuf__ uint16_t*)histCumsumUb.GetPhyAddr(),
        (__ubuf__ uint8_t*)inputB8Ub.GetPhyAddr(), currTileSize, repeatTime, round);
}

template <typename KeyT, typename CountT>
__aicore__ inline void GetGlobalExclusiveSumB8(AscendC::LocalTensor<KeyT>& inputX,
                                               AscendC::LocalTensor<CountT>& blockExclusiveUb,
                                               AscendC::LocalTensor<uint16_t>& histUb,
                                               AscendC::LocalTensor<uint16_t>& histCumsumUb,
                                               AscendC::LocalTensor<uint8_t>& inputB8Ub, uint32_t currTileSize,
                                               uint32_t round)
{
    uint16_t repeatTime = AscendC::CeilDivision(currTileSize, VF_LEN_B8);
    asc_vf_call<HistogramB8VF<KeyT, CountT>>(
        (__ubuf__ KeyT*)inputX.GetPhyAddr(), (__ubuf__ CountT*)blockExclusiveUb.GetPhyAddr(),
        (__ubuf__ uint16_t*)histUb.GetPhyAddr(), (__ubuf__ uint16_t*)histCumsumUb.GetPhyAddr(),
        (__ubuf__ uint8_t*)inputB8Ub.GetPhyAddr(), currTileSize, repeatTime, round);
}

// ── SIMT global ops ──
template <typename CountT>
__simt_vf__ LAUNCH_BOUND(RADIX_SORT_NUM) __aicore__
    void SimtGlobalOffset(uint32_t exclusiveBinOffset, __gm__ CountT* exclusiveBinsGm,
                          __ubuf__ CountT* blockExclusiveBuffer)
{
    for (int32_t i = threadIdx.x; i < RADIX_SORT_NUM; i += RADIX_SORT_NUM) {
        CountT srcData = blockExclusiveBuffer[i];
        asc_atomic_add(exclusiveBinsGm + exclusiveBinOffset + i, srcData);
    }
}

template <typename ValT, typename CountT, typename IdxT, bool propagateIndex>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_NUM) __aicore__
    void CopyOutGm(CountT tileDataStart, uint32_t cureTileSize, uint64_t outputXUnsortedAxisOffset,
                   uint64_t exclRoundOffset, __ubuf__ uint16_t* blockExclusiveSumAddr,
                   __gm__ volatile CountT* exclusiveBinsGmAddr, __ubuf__ CountT* blockDataInGlobalPosAddr,
                   __ubuf__ uint32_t* sortedIndexLocalAddr, __ubuf__ CountT* xInputIndexLocalAddr,
                   __ubuf__ uint8_t* sortedValueLocalAddr, __ubuf__ ValT* xInputValueLocalAddr,
                   __ubuf__ CountT* blockHistFlagAddr, __ubuf__ uint16_t* blockHistAddr,
                   __gm__ volatile IdxT* indexDoubleBufferGmAddr, __gm__ volatile ValT* inputXDoubleBufferAddr)
{
    for (int i = threadIdx.x; i < RADIX_SORT_NUM; i += THREAD_DIM_NUM) {
        CountT blockHistCumsumVal = blockHistFlagAddr[i];
        if constexpr (AscendC::IsSameType<CountT, uint32_t>::value) {
            blockHistCumsumVal = blockHistCumsumVal & VALUE_MASK;
        } else {
            blockHistCumsumVal = blockHistCumsumVal & VALUE_MASK_B64;
        }
        uint32_t blockExclusiveSumVal = blockExclusiveSumAddr[i];
        uint32_t blockHistVal = blockHistAddr[i];
        CountT globalKeyOffsetVal = exclusiveBinsGmAddr[exclRoundOffset + i];
        CountT finalpos = globalKeyOffsetVal + blockHistCumsumVal - blockHistVal - blockExclusiveSumVal;
        blockDataInGlobalPosAddr[i] = finalpos;
    }
    asc_syncthreads();
    for (int i = threadIdx.x; i < cureTileSize; i += THREAD_DIM_NUM) {
        CountT localDataIndex = static_cast<CountT>(sortedIndexLocalAddr[i]);
        CountT dataInitIndex = 0;
        if constexpr (propagateIndex) {
            dataInitIndex = xInputIndexLocalAddr[localDataIndex];
        } else {
            dataInitIndex = tileDataStart + localDataIndex;
        }
        CountT dataFinalGlobalPos = blockDataInGlobalPosAddr[sortedValueLocalAddr[i]] + i;
        inputXDoubleBufferAddr[dataFinalGlobalPos + outputXUnsortedAxisOffset] = xInputValueLocalAddr[localDataIndex];
        indexDoubleBufferGmAddr[dataFinalGlobalPos + outputXUnsortedAxisOffset] = static_cast<IdxT>(dataInitIndex);
    }
}

} // namespace SortLib::detail

#endif // SORT_LIB_VF_H
