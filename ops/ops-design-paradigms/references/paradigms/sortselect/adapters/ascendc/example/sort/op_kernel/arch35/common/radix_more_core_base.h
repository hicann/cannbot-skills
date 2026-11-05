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
 * \file radix_more_core_base.h
 * \brief 多核基数排序公共实现：直方图、散射、拷贝函数及 CRTP 基类。
 *        Twiddle 预处理见 radix_sort_simd_utils.h；公共常量见 radix_sort_constants.h。
 */

#ifndef RADIX_MORE_CORE_BASE_H
#define RADIX_MORE_CORE_BASE_H

#include <type_traits>
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "util_type_simd.h"
#include "radix_sort_constants.h"
#include "radix_sort_simd_utils.h"

namespace RadixSortCommon {

using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

template <typename T1>
__aicore__ inline void CopyDataIn(GlobalTensor<T1> inputX, LocalTensor<T1> xLocal, uint64_t tileOffset,
                                  uint32_t currTileSize)
{
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = currTileSize * sizeof(T1);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xLocal, inputX[tileOffset], dataCopyParam, padParams);
}

template <typename CounterT>
__simd_callee__ inline void AccumulateBucketPrefixes(RegTensor<uint16_t>& chist0, RegTensor<uint16_t>& chist1,
                                                     RegTensor<uint16_t>& hist0, RegTensor<uint16_t>& hist1,
                                                     MaskReg& maskB16, __ubuf__ CounterT* blockExclusiveUbRPtr,
                                                     __ubuf__ CounterT* blockExclusiveUbWPtr,
                                                     __ubuf__ uint16_t* histUbPtr,
                                                     __ubuf__ uint16_t* bucketOffsetsUbPtr)
{
    // chist 包含当前桶；减去本桶计数，得到散射使用的块内独占前缀。
    Reg::RegTensor<uint16_t> exclusiveSumZero;
    Reg::RegTensor<uint16_t> exclusiveSumOne;
    Reg::RegTensor<uint16_t> zeroReg;
    Reg::Sub(exclusiveSumZero, chist0, hist0, maskB16);
    Reg::Sub(exclusiveSumOne, chist1, hist1, maskB16);

    // 保存块内计数和独占前缀；跨核偏移累积完成后由 ComputeOnePass 重新读取。
    Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(histUbPtr, hist0, VF_LEN_B16, maskB16);
    Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(histUbPtr, hist1, VF_LEN_B16, maskB16);
    Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(bucketOffsetsUbPtr, exclusiveSumZero, VF_LEN_B16,
                                                                  maskB16);
    Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(bucketOffsetsUbPtr, exclusiveSumOne, VF_LEN_B16,
                                                                  maskB16);

    Reg::Duplicate(zeroReg, 0, maskB16);
    Reg::RegTensor<uint32_t> sum0;
    Reg::RegTensor<uint32_t> sum1;
    Reg::RegTensor<uint32_t> sum2;
    Reg::RegTensor<uint32_t> sum3;
    Reg::Interleave((Reg::RegTensor<uint16_t>&)sum0, (Reg::RegTensor<uint16_t>&)sum1, exclusiveSumZero, zeroReg);
    Reg::Interleave((Reg::RegTensor<uint16_t>&)sum2, (Reg::RegTensor<uint16_t>&)sum3, exclusiveSumOne, zeroReg);
    if constexpr (sizeof(CounterT) == sizeof(uint32_t)) {
        // 32 位计数可直接累加，无需再次扩展寄存器。
        auto mask = Reg::CreateMask<uint32_t>();
        RegTensor<uint32_t> current0;
        RegTensor<uint32_t> current1;
        RegTensor<uint32_t> current2;
        RegTensor<uint32_t> current3;
        Reg::LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(current0, blockExclusiveUbRPtr, VF_LEN_B32);
        Reg::LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(current1, blockExclusiveUbRPtr, VF_LEN_B32);
        Reg::LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(current2, blockExclusiveUbRPtr, VF_LEN_B32);
        Reg::LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(current3, blockExclusiveUbRPtr, VF_LEN_B32);
        Reg::Add(current0, current0, sum0, mask);
        Reg::Add(current1, current1, sum1, mask);
        Reg::Add(current2, current2, sum2, mask);
        Reg::Add(current3, current3, sum3, mask);
        Reg::StoreAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, current0, VF_LEN_B32, mask);
        Reg::StoreAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, current1, VF_LEN_B32, mask);
        Reg::StoreAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, current2, VF_LEN_B32, mask);
        Reg::StoreAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, current3, VF_LEN_B32, mask);
    } else {
        Reg::MaskReg maskB64 = Reg::CreateMask<int64_t>();
        Reg::RegTensor<int64_t> sum0Int64;
        Reg::RegTensor<int64_t> sum1Int64;
        Reg::RegTensor<int64_t> sum2Int64;
        Reg::RegTensor<int64_t> sum3Int64;
        Reg::RegTensor<int64_t> sum4Int64;
        Reg::RegTensor<int64_t> sum5Int64;
        Reg::RegTensor<int64_t> sum6Int64;
        Reg::RegTensor<int64_t> sum7Int64;
        Reg::Interleave((Reg::RegTensor<uint32_t>&)sum0Int64, (Reg::RegTensor<uint32_t>&)sum1Int64, sum0,
                        (Reg::RegTensor<uint32_t>&)zeroReg);
        Reg::Interleave((Reg::RegTensor<uint32_t>&)sum2Int64, (Reg::RegTensor<uint32_t>&)sum3Int64, sum1,
                        (Reg::RegTensor<uint32_t>&)zeroReg);
        Reg::Interleave((Reg::RegTensor<uint32_t>&)sum4Int64, (Reg::RegTensor<uint32_t>&)sum5Int64, sum2,
                        (Reg::RegTensor<uint32_t>&)zeroReg);
        Reg::Interleave((Reg::RegTensor<uint32_t>&)sum6Int64, (Reg::RegTensor<uint32_t>&)sum7Int64, sum3,
                        (Reg::RegTensor<uint32_t>&)zeroReg);
        Reg::RegTensor<int64_t> int64In0;
        Reg::RegTensor<int64_t> int64In1;
        Reg::RegTensor<int64_t> int64In2;
        Reg::RegTensor<int64_t> int64In3;
        Reg::RegTensor<int64_t> int64In4;
        Reg::RegTensor<int64_t> int64In5;
        Reg::RegTensor<int64_t> int64In6;
        Reg::RegTensor<int64_t> int64In7;
        Reg::LoadAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(int64In0, blockExclusiveUbRPtr, VF_LEN_B64);
        Reg::LoadAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(int64In1, blockExclusiveUbRPtr, VF_LEN_B64);
        Reg::LoadAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(int64In2, blockExclusiveUbRPtr, VF_LEN_B64);
        Reg::LoadAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(int64In3, blockExclusiveUbRPtr, VF_LEN_B64);
        Reg::LoadAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(int64In4, blockExclusiveUbRPtr, VF_LEN_B64);
        Reg::LoadAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(int64In5, blockExclusiveUbRPtr, VF_LEN_B64);
        Reg::LoadAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(int64In6, blockExclusiveUbRPtr, VF_LEN_B64);
        Reg::LoadAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(int64In7, blockExclusiveUbRPtr, VF_LEN_B64);
        // 将当前块的桶独占前缀累加到本核桶偏移总量。
        Reg::Add(int64In0, int64In0, sum0Int64, maskB64);
        Reg::Add(int64In1, int64In1, sum1Int64, maskB64);
        Reg::Add(int64In2, int64In2, sum2Int64, maskB64);
        Reg::Add(int64In3, int64In3, sum3Int64, maskB64);
        Reg::Add(int64In4, int64In4, sum4Int64, maskB64);
        Reg::Add(int64In5, int64In5, sum5Int64, maskB64);
        Reg::Add(int64In6, int64In6, sum6Int64, maskB64);
        Reg::Add(int64In7, int64In7, sum7Int64, maskB64);
        Reg::StoreAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, int64In0, VF_LEN_B64,
                                                                     maskB64);
        Reg::StoreAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, int64In1, VF_LEN_B64,
                                                                     maskB64);
        Reg::StoreAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, int64In2, VF_LEN_B64,
                                                                     maskB64);
        Reg::StoreAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, int64In3, VF_LEN_B64,
                                                                     maskB64);
        Reg::StoreAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, int64In4, VF_LEN_B64,
                                                                     maskB64);
        Reg::StoreAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, int64In5, VF_LEN_B64,
                                                                     maskB64);
        Reg::StoreAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, int64In6, VF_LEN_B64,
                                                                     maskB64);
        Reg::StoreAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockExclusiveUbWPtr, int64In7, VF_LEN_B64,
                                                                     maskB64);
    }
}

template <typename CounterT>
__simd_vf__ inline void GetGlobalExclusiveSumB32VF(__ubuf__ uint32_t* inXPtr, __ubuf__ CounterT* blockExclusiveUbWPtr,
                                                   __ubuf__ CounterT* blockExclusiveUbRPtr,
                                                   __ubuf__ uint16_t* histUbPtr, __ubuf__ uint16_t* bucketOffsetsUbPtr,
                                                   __ubuf__ uint8_t* inputB8UbPtr, uint32_t bitOffset,
                                                   uint16_t repeatTime, uint32_t inputElementNum)
{
    Reg::RegTensor<uint32_t> in0;
    Reg::RegTensor<uint32_t> in1;
    Reg::RegTensor<uint32_t> in2;
    Reg::RegTensor<uint32_t> in3;
    Reg::RegTensor<uint16_t> hist0;
    Reg::RegTensor<uint16_t> hist1;
    Reg::RegTensor<uint16_t> chist0;
    Reg::RegTensor<uint16_t> chist1;
    Reg::MaskReg histMask;
    Reg::MaskReg maskB32 = Reg::CreateMask<uint32_t>();
    Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t>();
    Reg::Duplicate(hist0, 0, maskB16);
    Reg::Duplicate(hist1, 0, maskB16);
    Reg::Duplicate(chist0, 0, maskB16);
    Reg::Duplicate(chist1, 0, maskB16);
    for (uint16_t i = 0; i < repeatTime; i++) {
        histMask = Reg::UpdateMask<uint8_t>(inputElementNum);
        // 加载四个 B32 向量，提取当前字节并压缩为 B8，供计数和散射使用。
        Reg::LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(in0, inXPtr, VF_LEN_B32);
        Reg::LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(in1, inXPtr, VF_LEN_B32);
        Reg::LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(in2, inXPtr, VF_LEN_B32);
        Reg::LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(in3, inXPtr, VF_LEN_B32);
        Reg::RegTensor<uint32_t> shift0;
        Reg::RegTensor<uint32_t> shift1;
        Reg::RegTensor<uint32_t> shift2;
        Reg::RegTensor<uint32_t> shift3;
        Reg::ShiftRights<uint32_t, int16_t>(shift0, in0, bitOffset, maskB32);
        Reg::ShiftRights<uint32_t, int16_t>(shift1, in1, bitOffset, maskB32);
        Reg::ShiftRights<uint32_t, int16_t>(shift2, in2, bitOffset, maskB32);
        Reg::ShiftRights<uint32_t, int16_t>(shift3, in3, bitOffset, maskB32);
        Reg::RegTensor<uint16_t> deInter0;
        Reg::RegTensor<uint16_t> deInter1;
        Reg::RegTensor<uint16_t> deInter2;
        Reg::RegTensor<uint16_t> deInter3;
        Reg::DeInterleave(deInter0, deInter1, (Reg::RegTensor<uint16_t>&)shift0, (Reg::RegTensor<uint16_t>&)shift1);
        Reg::DeInterleave(deInter2, deInter3, (Reg::RegTensor<uint16_t>&)shift2, (Reg::RegTensor<uint16_t>&)shift3);

        Reg::RegTensor<uint8_t> deInter0B8;

        Reg::RegTensor<uint8_t> deInter1B8;
        Reg::DeInterleave(deInter0B8, deInter1B8, (Reg::RegTensor<uint8_t>&)deInter0,
                          (Reg::RegTensor<uint8_t>&)deInter2);
        Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(inputB8UbPtr, deInter0B8, VF_LEN_B8, histMask);
        // 同时生成当前块的桶频数和包含当前桶的累积计数。
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::FREQUENCY>(
            hist0, deInter0B8, histMask);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::FREQUENCY>(
            hist1, deInter0B8, histMask);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(
            chist0, deInter0B8, histMask);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(
            chist1, deInter0B8, histMask);
    }
    AccumulateBucketPrefixes(chist0, chist1, hist0, hist1, maskB16, blockExclusiveUbRPtr, blockExclusiveUbWPtr,
                             histUbPtr, bucketOffsetsUbPtr);
}

template <typename UT, typename CounterT>
__aicore__ inline void GetGlobalExclusiveSumB32(LocalTensor<UT>& inputX, LocalTensor<CounterT>& blockExclusiveUb,
                                                LocalTensor<uint16_t>& histUb, LocalTensor<uint16_t>& bucketOffsetsUb,
                                                LocalTensor<uint8_t>& inputB8Ub, uint32_t currTileSize, uint32_t round)
{
    uint32_t bitOffset = round * SHIFT_BIT_NUM;
    uint16_t repeatTime = CeilDivision(currTileSize, VF_LEN_B8);
    __ubuf__ uint32_t* inXPtr = (__ubuf__ uint32_t*)inputX.GetPhyAddr();
    __ubuf__ CounterT* blockExclusiveUbWPtr = (__ubuf__ CounterT*)blockExclusiveUb.GetPhyAddr();
    __ubuf__ CounterT* blockExclusiveUbRPtr = blockExclusiveUbWPtr;
    __ubuf__ uint16_t* histUbPtr = (__ubuf__ uint16_t*)histUb.GetPhyAddr();
    __ubuf__ uint16_t* bucketOffsetsUbPtr = (__ubuf__ uint16_t*)bucketOffsetsUb.GetPhyAddr();
    __ubuf__ uint8_t* inputB8UbPtr = (__ubuf__ uint8_t*)inputB8Ub.GetPhyAddr();
    uint32_t inputElementNum = currTileSize;
    asc_vf_call<GetGlobalExclusiveSumB32VF<CounterT>>(dim3{1, 1, 1}, inXPtr, blockExclusiveUbWPtr, blockExclusiveUbRPtr,
                                                      histUbPtr, bucketOffsetsUbPtr, inputB8UbPtr, bitOffset,
                                                      repeatTime, inputElementNum);
}

template <typename CounterT>
__simd_vf__ inline void GetGlobalExclusiveSumB64VF(__ubuf__ uint64_t* inXPtr, __ubuf__ CounterT* blockExclusiveUbWPtr,
                                                   __ubuf__ CounterT* blockExclusiveUbRPtr,
                                                   __ubuf__ uint16_t* histUbPtr, __ubuf__ uint16_t* bucketOffsetsUbPtr,
                                                   __ubuf__ uint8_t* inputB8UbPtr, uint32_t bitOffset,
                                                   uint16_t repeatTime, uint32_t inputElementNum)
{
    Reg::RegTensor<uint64_t> in0;
    Reg::RegTensor<uint64_t> in1;
    Reg::RegTensor<uint64_t> in2;
    Reg::RegTensor<uint64_t> in3;
    Reg::RegTensor<uint64_t> in4;
    Reg::RegTensor<uint64_t> in5;
    Reg::RegTensor<uint64_t> in6;
    Reg::RegTensor<uint64_t> in7;
    Reg::RegTensor<uint16_t> hist0;
    Reg::RegTensor<uint16_t> hist1;
    Reg::RegTensor<uint16_t> chist0;
    Reg::RegTensor<uint16_t> chist1;
    Reg::MaskReg histMask;
    Reg::MaskReg maskB64 = Reg::CreateMask<uint64_t>();
    Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t>();
    Reg::Duplicate(hist0, 0, maskB16);
    Reg::Duplicate(hist1, 0, maskB16);
    Reg::Duplicate(chist0, 0, maskB16);
    Reg::Duplicate(chist1, 0, maskB16);
    for (uint16_t i = 0; i < repeatTime; i++) {
        histMask = Reg::UpdateMask<uint8_t>(inputElementNum);
        // 加载八个 B64 向量，提取当前字节并压缩为 B8，供计数和散射使用。
        Reg::LoadAlign<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(in0, inXPtr, VF_LEN_B64);
        Reg::LoadAlign<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(in1, inXPtr, VF_LEN_B64);
        Reg::LoadAlign<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(in2, inXPtr, VF_LEN_B64);
        Reg::LoadAlign<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(in3, inXPtr, VF_LEN_B64);
        Reg::LoadAlign<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(in4, inXPtr, VF_LEN_B64);
        Reg::LoadAlign<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(in5, inXPtr, VF_LEN_B64);
        Reg::LoadAlign<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(in6, inXPtr, VF_LEN_B64);
        Reg::LoadAlign<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(in7, inXPtr, VF_LEN_B64);
        Reg::RegTensor<uint64_t> shift0;
        Reg::RegTensor<uint64_t> shift1;
        Reg::RegTensor<uint64_t> shift2;
        Reg::RegTensor<uint64_t> shift3;
        Reg::RegTensor<uint64_t> shift4;
        Reg::RegTensor<uint64_t> shift5;
        Reg::RegTensor<uint64_t> shift6;
        Reg::RegTensor<uint64_t> shift7;
        Reg::ShiftRights<uint64_t, int16_t>(shift0, in0, bitOffset, maskB64);
        Reg::ShiftRights<uint64_t, int16_t>(shift1, in1, bitOffset, maskB64);
        Reg::ShiftRights<uint64_t, int16_t>(shift2, in2, bitOffset, maskB64);
        Reg::ShiftRights<uint64_t, int16_t>(shift3, in3, bitOffset, maskB64);
        Reg::ShiftRights<uint64_t, int16_t>(shift4, in4, bitOffset, maskB64);
        Reg::ShiftRights<uint64_t, int16_t>(shift5, in5, bitOffset, maskB64);
        Reg::ShiftRights<uint64_t, int16_t>(shift6, in6, bitOffset, maskB64);
        Reg::ShiftRights<uint64_t, int16_t>(shift7, in7, bitOffset, maskB64);
        Reg::RegTensor<uint32_t> deInter0;
        Reg::RegTensor<uint32_t> deInter1;
        Reg::RegTensor<uint32_t> deInter2;
        Reg::RegTensor<uint32_t> deInter3;
        Reg::RegTensor<uint32_t> deInter4;
        Reg::RegTensor<uint32_t> deInter5;
        Reg::RegTensor<uint32_t> deInter6;
        Reg::RegTensor<uint32_t> deInter7;
        Reg::DeInterleave(deInter0, deInter1, (Reg::RegTensor<uint32_t>&)shift0, (Reg::RegTensor<uint32_t>&)shift1);
        Reg::DeInterleave(deInter2, deInter3, (Reg::RegTensor<uint32_t>&)shift2, (Reg::RegTensor<uint32_t>&)shift3);
        Reg::DeInterleave(deInter4, deInter5, (Reg::RegTensor<uint32_t>&)shift4, (Reg::RegTensor<uint32_t>&)shift5);
        Reg::DeInterleave(deInter6, deInter7, (Reg::RegTensor<uint32_t>&)shift6, (Reg::RegTensor<uint32_t>&)shift7);
        Reg::RegTensor<uint16_t> deInter0B16;
        Reg::RegTensor<uint16_t> deInter1B16;
        Reg::RegTensor<uint16_t> deInter2B16;
        Reg::RegTensor<uint16_t> deInter3B16;
        Reg::DeInterleave(deInter0B16, deInter1B16, (Reg::RegTensor<uint16_t>&)deInter0,
                          (Reg::RegTensor<uint16_t>&)deInter2);
        Reg::DeInterleave(deInter2B16, deInter3B16, (Reg::RegTensor<uint16_t>&)deInter4,
                          (Reg::RegTensor<uint16_t>&)deInter6);
        Reg::RegTensor<uint8_t> deInter0B8;
        Reg::RegTensor<uint8_t> deInter1B8;
        Reg::DeInterleave(deInter0B8, deInter1B8, (Reg::RegTensor<uint8_t>&)deInter0B16,
                          (Reg::RegTensor<uint8_t>&)deInter2B16);
        Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(inputB8UbPtr, deInter0B8, VF_LEN_B8, histMask);
        // 同时生成当前块的桶频数和包含当前桶的累积计数。
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::FREQUENCY>(
            hist0, deInter0B8, histMask);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::FREQUENCY>(
            hist1, deInter0B8, histMask);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(
            chist0, deInter0B8, histMask);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(
            chist1, deInter0B8, histMask);
    }
    AccumulateBucketPrefixes(chist0, chist1, hist0, hist1, maskB16, blockExclusiveUbRPtr, blockExclusiveUbWPtr,
                             histUbPtr, bucketOffsetsUbPtr);
}

template <typename UT, typename CounterT>
__aicore__ inline void GetGlobalExclusiveSumB64(LocalTensor<UT>& inputX, LocalTensor<CounterT>& blockExclusiveUb,
                                                LocalTensor<uint16_t>& histUb, LocalTensor<uint16_t>& bucketOffsetsUb,
                                                LocalTensor<uint8_t>& inputB8Ub, uint32_t currTileSize, uint32_t round)
{
    uint32_t bitOffset = round * SHIFT_BIT_NUM;
    uint16_t repeatTime = CeilDivision(currTileSize, VF_LEN_B8);
    __ubuf__ uint64_t* inXPtr = (__ubuf__ uint64_t*)inputX.GetPhyAddr();
    __ubuf__ CounterT* blockExclusiveUbWPtr = (__ubuf__ CounterT*)blockExclusiveUb.GetPhyAddr();
    __ubuf__ CounterT* blockExclusiveUbRPtr = blockExclusiveUbWPtr;
    __ubuf__ uint16_t* histUbPtr = (__ubuf__ uint16_t*)histUb.GetPhyAddr();
    __ubuf__ uint8_t* inputB8UbPtr = (__ubuf__ uint8_t*)inputB8Ub.GetPhyAddr();
    __ubuf__ uint16_t* bucketOffsetsUbPtr = (__ubuf__ uint16_t*)bucketOffsetsUb.GetPhyAddr();
    uint32_t inputElementNum = currTileSize;
    asc_vf_call<GetGlobalExclusiveSumB64VF<CounterT>>(dim3{1, 1, 1}, inXPtr, blockExclusiveUbWPtr, blockExclusiveUbRPtr,
                                                      histUbPtr, bucketOffsetsUbPtr, inputB8UbPtr, bitOffset,
                                                      repeatTime, inputElementNum);
}

template <typename CounterT>
__simd_vf__ inline void GetGlobalExclusiveSumB16VF(__ubuf__ uint16_t* inputXValuePtr,
                                                   __ubuf__ CounterT* blockExclusiveUbWPtr,
                                                   __ubuf__ CounterT* blockExclusiveUbRPtr,
                                                   __ubuf__ uint16_t* histUbPtr, __ubuf__ uint16_t* bucketOffsetsUbPtr,
                                                   __ubuf__ uint8_t* inputB8UbPtr, uint32_t bitOffset,
                                                   uint16_t repeatTime, uint32_t inputElementNum)
{
    Reg::MaskReg histMask;
    Reg::RegTensor<uint16_t> in0;
    Reg::RegTensor<uint16_t> in1;
    Reg::RegTensor<uint16_t> shift0;
    Reg::RegTensor<uint16_t> shift1;
    Reg::RegTensor<uint16_t> hist0;
    Reg::RegTensor<uint16_t> hist1;
    Reg::RegTensor<uint16_t> chist0;
    Reg::RegTensor<uint16_t> chist1;
    Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t>();
    Reg::Duplicate(hist0, 0, maskB16);
    Reg::Duplicate(hist1, 0, maskB16);
    Reg::Duplicate(chist0, 0, maskB16);
    Reg::Duplicate(chist1, 0, maskB16);
    for (uint16_t i = 0; i < repeatTime; i++) {
        histMask = Reg::UpdateMask<uint8_t>(inputElementNum);
        // 加载两个 B16 向量，提取当前字节并压缩为 B8，供计数和散射使用。
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(in0, inputXValuePtr, VF_LEN_B16);
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(in1, inputXValuePtr, VF_LEN_B16);
        Reg::ShiftRights<uint16_t, int16_t>(shift0, in0, bitOffset, maskB16);
        Reg::ShiftRights<uint16_t, int16_t>(shift1, in1, bitOffset, maskB16);
        Reg::RegTensor<uint8_t> deInter0B8;
        Reg::RegTensor<uint8_t> deInter1B8;
        Reg::DeInterleave(deInter0B8, deInter1B8, (Reg::RegTensor<uint8_t>&)shift0, (Reg::RegTensor<uint8_t>&)shift1);
        Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(inputB8UbPtr, deInter0B8, VF_LEN_B8, histMask);
        // 同时生成当前块的桶频数和包含当前桶的累积计数。
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::FREQUENCY>(
            hist0, deInter0B8, histMask);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::FREQUENCY>(
            hist1, deInter0B8, histMask);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(
            chist0, deInter0B8, histMask);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(
            chist1, deInter0B8, histMask);
    }
    AccumulateBucketPrefixes(chist0, chist1, hist0, hist1, maskB16, blockExclusiveUbRPtr, blockExclusiveUbWPtr,
                             histUbPtr, bucketOffsetsUbPtr);
}

template <typename UT, typename CounterT>
__aicore__ inline void GetGlobalExclusiveSumB16(LocalTensor<UT>& inputX, LocalTensor<CounterT>& blockExclusiveUb,
                                                LocalTensor<uint16_t>& histUb, LocalTensor<uint16_t>& bucketOffsetsUb,
                                                LocalTensor<uint8_t>& inputB8Ub, uint32_t currTileSize, uint32_t round)
{
    uint32_t bitOffset = round * SHIFT_BIT_NUM;
    uint16_t repeatTime = CeilDivision(currTileSize, VF_LEN_B8);
    __ubuf__ uint16_t* inputXValuePtr = (__ubuf__ uint16_t*)inputX.GetPhyAddr();
    __ubuf__ CounterT* blockExclusiveUbWPtr = (__ubuf__ CounterT*)blockExclusiveUb.GetPhyAddr();
    __ubuf__ CounterT* blockExclusiveUbRPtr = blockExclusiveUbWPtr;
    __ubuf__ uint16_t* histUbPtr = (__ubuf__ uint16_t*)histUb.GetPhyAddr();
    __ubuf__ uint16_t* bucketOffsetsUbPtr = (__ubuf__ uint16_t*)bucketOffsetsUb.GetPhyAddr();
    __ubuf__ uint8_t* inputB8UbPtr = (__ubuf__ uint8_t*)inputB8Ub.GetPhyAddr();
    uint32_t inputElementNum = currTileSize;
    asc_vf_call<GetGlobalExclusiveSumB16VF<CounterT>>(dim3{1, 1, 1}, inputXValuePtr, blockExclusiveUbWPtr,
                                                      blockExclusiveUbRPtr, histUbPtr, bucketOffsetsUbPtr, inputB8UbPtr,
                                                      bitOffset, repeatTime, inputElementNum);
}

template <typename CounterT>
__simd_vf__ inline void GetGlobalExclusiveSumB8VF(__ubuf__ uint8_t* inXPtr, __ubuf__ CounterT* blockExclusiveUbWPtr,
                                                  __ubuf__ CounterT* blockExclusiveUbRPtr, __ubuf__ uint16_t* histUbPtr,
                                                  __ubuf__ uint16_t* bucketOffsetsUbPtr, __ubuf__ uint8_t* inputB8UbPtr,
                                                  uint32_t bitOffset, uint16_t repeatTime, uint32_t inputElementNum)
{
    Reg::MaskReg histMask;
    Reg::RegTensor<uint8_t> in0;
    Reg::RegTensor<uint16_t> hist0;
    Reg::RegTensor<uint16_t> hist1;
    Reg::RegTensor<uint16_t> chist0;
    Reg::RegTensor<uint16_t> chist1;
    Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t>();
    Reg::Duplicate(hist0, 0, maskB16);
    Reg::Duplicate(hist1, 0, maskB16);
    Reg::Duplicate(chist0, 0, maskB16);
    Reg::Duplicate(chist1, 0, maskB16);
    for (uint16_t i = 0; i < repeatTime; i++) {
        histMask = Reg::UpdateMask<uint8_t>(inputElementNum);
        // B8 输入即本轮字节，直接保存并计算桶频数和累积计数。
        Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(in0, inXPtr, VF_LEN_B8);
        Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(inputB8UbPtr, in0, VF_LEN_B8, histMask);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::FREQUENCY>(hist0, in0,
                                                                                                         histMask);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::FREQUENCY>(hist1, in0,
                                                                                                         histMask);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(chist0, in0,
                                                                                                          histMask);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(chist1, in0,
                                                                                                          histMask);
    }
    AccumulateBucketPrefixes(chist0, chist1, hist0, hist1, maskB16, blockExclusiveUbRPtr, blockExclusiveUbWPtr,
                             histUbPtr, bucketOffsetsUbPtr);
}

template <typename UT, typename CounterT>
__aicore__ inline void GetGlobalExclusiveSumB8(LocalTensor<UT>& inputX, LocalTensor<CounterT>& blockExclusiveUb,
                                               LocalTensor<uint16_t>& histUb, LocalTensor<uint16_t>& bucketOffsetsUb,
                                               LocalTensor<uint8_t>& inputB8Ub, uint32_t currTileSize, uint32_t round)
{
    uint32_t bitOffset = round * SHIFT_BIT_NUM;
    uint16_t repeatTime = CeilDivision(currTileSize, VF_LEN_B8);
    __ubuf__ uint8_t* inXPtr = (__ubuf__ uint8_t*)inputX.GetPhyAddr();
    __ubuf__ CounterT* blockExclusiveUbWPtr = (__ubuf__ CounterT*)blockExclusiveUb.GetPhyAddr();
    __ubuf__ CounterT* blockExclusiveUbRPtr = blockExclusiveUbWPtr;
    __ubuf__ uint16_t* histUbPtr = (__ubuf__ uint16_t*)histUb.GetPhyAddr();
    __ubuf__ uint16_t* bucketOffsetsUbPtr = (__ubuf__ uint16_t*)bucketOffsetsUb.GetPhyAddr();
    __ubuf__ uint8_t* inputB8UbPtr = (__ubuf__ uint8_t*)inputB8Ub.GetPhyAddr();
    uint32_t inputElementNum = currTileSize;
    asc_vf_call<GetGlobalExclusiveSumB8VF<CounterT>>(dim3{1, 1, 1}, inXPtr, blockExclusiveUbWPtr, blockExclusiveUbRPtr,
                                                     histUbPtr, bucketOffsetsUbPtr, inputB8UbPtr, bitOffset, repeatTime,
                                                     inputElementNum);
}

inline __simt_vf__ LAUNCH_BOUND(RADIX_SORT_NUM) __aicore__
    void SimtGlobalOffset(uint64_t exclusiveBinOffset, __gm__ int64_t* exclusiveBinsGm,
                          __ubuf__ int64_t* blockExclusiveBuffer)
{
    for (int32_t i = threadIdx.x; i < RADIX_SORT_NUM; i += RADIX_SORT_NUM) {
        int32_t offset = i;
        int64_t srcData = blockExclusiveBuffer[offset];
        asc_atomic_add(exclusiveBinsGm + exclusiveBinOffset + offset, srcData);
    }
}

template <typename T>
__aicore__ inline void CopyGlobalDataIn(GlobalTensor<T> inputX, LocalTensor<T>& xLocal, uint32_t tileOffset,
                                        uint32_t currTileSize)
{
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = currTileSize * sizeof(T);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xLocal, inputX[tileOffset], dataCopyParam, padParams);
}

template <typename CounterT>
__aicore__ inline void CopyIndexDataIn(GlobalTensor<uint32_t> inputIndex, LocalTensor<uint32_t>& xLocal,
                                       uint64_t tileOffset, uint32_t currTileSize)
{
    DataCopyPadExtParams<uint32_t> padParams{false, 0, 0, 0};
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = currTileSize * sizeof(CounterT);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xLocal, inputIndex[tileOffset], dataCopyParam, padParams);
}

template <typename T1, typename IdxT, typename CounterT, typename IndexT, int32_t round>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_NUM) __aicore__
    void CopyOutGm(__ubuf__ const uint64_t* scatterArgs, uint32_t currentTileSize,
                   __ubuf__ uint16_t* blockExclusiveSumAddr, __gm__ volatile CounterT* exclusiveBinsGmAddr,
                   __ubuf__ CounterT* blockDataInGlobalPosAddr, __ubuf__ uint32_t* sortedIndexLocalAddr,
                   __ubuf__ IndexT* xInputIndexLocalAddr, __ubuf__ uint8_t* sortedDigitLocalAddr,
                   __ubuf__ T1* xInputValueLocalAddr, __ubuf__ CounterT* blockHistFlagAddr,
                   __ubuf__ uint16_t* blockHistAddr, __gm__ volatile IdxT* indexDoubleBufferGmAddr,
                   __gm__ volatile T1* inputXDoubleBufferAddr)
{
    const int64_t tileDataStart = static_cast<int64_t>(scatterArgs[SCATTER_TILE_START]);
    const uint64_t outputXUnsortedAxisOffset = scatterArgs[SCATTER_OUTPUT_OFFSET];
    const uint64_t unSortIdOffset = scatterArgs[SCATTER_BIN_OFFSET];
    for (int i = threadIdx.x; i < RADIX_SORT_NUM; i += THREAD_DIM_NUM) {
        // 当前桶的目标起点 = 全局较小桶元素数 + 截至本块的同桶前缀 - 本块同桶数。
        // i 是块内完整排序位置，包含较小桶元素；先减去块内独占前缀，
        // 后续加 i 即得到本桶内的稳定位置。CounterT 随轴长选择，32 位减法的暂时下溢由随后加 i 抵消。
        CounterT blockHistCumsumVal = blockHistFlagAddr[i];
        blockHistCumsumVal = blockHistCumsumVal & RadixCounterTraits<CounterT>::VALUE_MASK;
        uint32_t blockExclusiveSumVal = blockExclusiveSumAddr[i];
        uint32_t blockHistVal = blockHistAddr[i];
        CounterT globalKeyOffsetVal = exclusiveBinsGmAddr[unSortIdOffset + i];
        CounterT finalpos = globalKeyOffsetVal + blockHistCumsumVal - blockHistVal - blockExclusiveSumVal;
        blockDataInGlobalPosAddr[i] = finalpos;
    }
    asc_syncthreads();
    for (int i = threadIdx.x; i < currentTileSize; i += THREAD_DIM_NUM) {
        int64_t localDataIndex = static_cast<int64_t>(sortedIndexLocalAddr[i]);
        int64_t dataInitIndex = 0;
        // 首轮用块起点加局部索引；后续轮次从双缓冲取回原轴索引。
        if constexpr (round != 0) {
            dataInitIndex = xInputIndexLocalAddr[localDataIndex];
        } else {
            dataInitIndex = tileDataStart + localDataIndex;
        }
        CounterT dataFinalGlobalPos = blockDataInGlobalPosAddr[sortedDigitLocalAddr[i]] + i;
        inputXDoubleBufferAddr[dataFinalGlobalPos + outputXUnsortedAxisOffset] = xInputValueLocalAddr[localDataIndex];
        indexDoubleBufferGmAddr[dataFinalGlobalPos + outputXUnsortedAxisOffset] = static_cast<IdxT>(dataInitIndex);
    }
}

template <typename CounterT>
__simd_vf__ inline void MarkAggregateReadyVF(__ubuf__ uint16_t* blockHistPtr, __ubuf__ CounterT* blockHistWithFlagPtr)
{
    Reg::MaskReg predicateDefaultB16 = Reg::CreateMask<uint16_t>();
    Reg::RegTensor<uint16_t> blockHistZero;
    Reg::RegTensor<uint16_t> blockHistOne;
    Reg::RegTensor<uint16_t> expandedHistZero;
    Reg::RegTensor<uint16_t> expandedHistOne;
    Reg::RegTensor<uint16_t> expandedHistTwo;
    Reg::RegTensor<uint16_t> expandedHistThree;
    Reg::RegTensor<uint16_t> zeroVector;
    Reg::Duplicate(zeroVector, 0, predicateDefaultB16);
    Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockHistZero, blockHistPtr, VF_LEN_B16);
    Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockHistOne, blockHistPtr, VF_LEN_B16);
    // 将 uint16 桶计数扩展为 CounterT，附加局部计数就绪标志后发布到 GM。
    Reg::Interleave(expandedHistZero, expandedHistOne, blockHistZero, zeroVector);
    Reg::Interleave(expandedHistTwo, expandedHistThree, blockHistOne, zeroVector);
    if constexpr (sizeof(CounterT) == sizeof(uint32_t)) {
        auto mask = Reg::CreateMask<uint32_t>();
        RegTensor<uint32_t> readyMask;
        Reg::Duplicate(readyMask, RadixCounterTraits<CounterT>::AGGREGATE_MASK, mask);
        RegTensor<uint32_t> readyZero;
        Reg::Or(readyZero, (RegTensor<uint32_t>&)expandedHistZero, readyMask, mask);
        RegTensor<uint32_t> readyOne;
        Reg::Or(readyOne, (RegTensor<uint32_t>&)expandedHistOne, readyMask, mask);
        RegTensor<uint32_t> readyTwo;
        Reg::Or(readyTwo, (RegTensor<uint32_t>&)expandedHistTwo, readyMask, mask);
        RegTensor<uint32_t> readyThree;
        Reg::Or(readyThree, (RegTensor<uint32_t>&)expandedHistThree, readyMask, mask);
        Reg::StoreAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr, readyZero, VF_LEN_B32,
                                                                      mask);
        Reg::StoreAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr, readyOne, VF_LEN_B32, mask);
        Reg::StoreAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr, readyTwo, VF_LEN_B32, mask);
        Reg::StoreAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr, readyThree, VF_LEN_B32,
                                                                      mask);
    } else {
        // 再扩为 64 位槽位；随后按 int64 位模式附加高两位状态，低位计数不变。
        Reg::RegTensor<int64_t> aggregateReadyMask;
        Reg::MaskReg predicateDefault = Reg::CreateMask<int64_t>();
        Reg::Duplicate(aggregateReadyMask, RadixCounterTraits<CounterT>::AGGREGATE_MASK, predicateDefault);
        Reg::RegTensor<uint16_t> expandedHistZeroB64A;
        Reg::RegTensor<uint16_t> expandedHistZeroB64B;
        Reg::RegTensor<uint16_t> expandedHistOneB64A;
        Reg::RegTensor<uint16_t> expandedHistOneB64B;
        Reg::RegTensor<uint16_t> expandedHistTwoB64A;
        Reg::RegTensor<uint16_t> expandedHistTwoB64B;
        Reg::RegTensor<uint16_t> expandedHistThreeB64A;
        Reg::RegTensor<uint16_t> expandedHistThreeB64B;
        Reg::Interleave(expandedHistZeroB64A, expandedHistZeroB64B, expandedHistZero, zeroVector);
        Reg::Interleave(expandedHistOneB64A, expandedHistOneB64B, expandedHistOne, zeroVector);
        Reg::Interleave(expandedHistTwoB64A, expandedHistTwoB64B, expandedHistTwo, zeroVector);
        Reg::Interleave(expandedHistThreeB64A, expandedHistThreeB64B, expandedHistThree, zeroVector);
        Reg::RegTensor<int64_t> expandedHistZeroMaskB64A;
        Reg::RegTensor<int64_t> expandedHistZeroMaskB64B;
        Reg::RegTensor<int64_t> expandedHistOneMaskB64A;
        Reg::RegTensor<int64_t> expandedHistOneMaskB64B;
        Reg::RegTensor<int64_t> expandedHistTwoMaskB64A;
        Reg::RegTensor<int64_t> expandedHistTwoMaskB64B;
        Reg::RegTensor<int64_t> expandedHistThreeMaskB64A;
        Reg::RegTensor<int64_t> expandedHistThreeMaskB64B;
        Reg::Or(expandedHistZeroMaskB64A, (Reg::RegTensor<int64_t>&)expandedHistZeroB64A, aggregateReadyMask,
                predicateDefault);
        Reg::Or(expandedHistZeroMaskB64B, (Reg::RegTensor<int64_t>&)expandedHistZeroB64B, aggregateReadyMask,
                predicateDefault);
        Reg::Or(expandedHistOneMaskB64A, (Reg::RegTensor<int64_t>&)expandedHistOneB64A, aggregateReadyMask,
                predicateDefault);
        Reg::Or(expandedHistOneMaskB64B, (Reg::RegTensor<int64_t>&)expandedHistOneB64B, aggregateReadyMask,
                predicateDefault);
        Reg::Or(expandedHistTwoMaskB64A, (Reg::RegTensor<int64_t>&)expandedHistTwoB64A, aggregateReadyMask,
                predicateDefault);
        Reg::Or(expandedHistTwoMaskB64B, (Reg::RegTensor<int64_t>&)expandedHistTwoB64B, aggregateReadyMask,
                predicateDefault);
        Reg::Or(expandedHistThreeMaskB64A, (Reg::RegTensor<int64_t>&)expandedHistThreeB64A, aggregateReadyMask,
                predicateDefault);
        Reg::Or(expandedHistThreeMaskB64B, (Reg::RegTensor<int64_t>&)expandedHistThreeB64B, aggregateReadyMask,
                predicateDefault);
        Reg::StoreAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr, expandedHistZeroMaskB64A,
                                                                     VF_LEN_B64, predicateDefault);
        Reg::StoreAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr, expandedHistZeroMaskB64B,
                                                                     VF_LEN_B64, predicateDefault);
        Reg::StoreAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr, expandedHistOneMaskB64A,
                                                                     VF_LEN_B64, predicateDefault);
        Reg::StoreAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr, expandedHistOneMaskB64B,
                                                                     VF_LEN_B64, predicateDefault);
        Reg::StoreAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr, expandedHistTwoMaskB64A,
                                                                     VF_LEN_B64, predicateDefault);
        Reg::StoreAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr, expandedHistTwoMaskB64B,
                                                                     VF_LEN_B64, predicateDefault);
        Reg::StoreAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr, expandedHistThreeMaskB64A,
                                                                     VF_LEN_B64, predicateDefault);
        Reg::StoreAlign<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr, expandedHistThreeMaskB64B,
                                                                     VF_LEN_B64, predicateDefault);
    }
}

template <typename CounterT>
__simd_vf__ inline void CountLookbackStatesVF(__ubuf__ CounterT* tilePrevHistValuePtr,
                                              __ubuf__ uint32_t* ubFlagTensorPtr, uint16_t repeatTime)
{
    Reg::MaskReg pRegSelect;
    Reg::MaskReg maskB32 = Reg::CreateMask<uint32_t>();
    Reg::RegTensor<uint32_t> notInitCount;
    Reg::RegTensor<uint32_t> aggReadyCount;
    Reg::RegTensor<uint32_t> prefixReadyCount;
    Reg::Duplicate(notInitCount, 0, maskB32);
    Reg::Duplicate(aggReadyCount, 0, maskB32);
    Reg::Duplicate(prefixReadyCount, 0, maskB32);
    Reg::RegTensor<uint32_t> onesVector;
    Reg::RegTensor<uint32_t> zerosVector;
    Reg::Duplicate(onesVector, 1, maskB32);
    Reg::Duplicate(zerosVector, 0, maskB32);
    for (uint16_t i = 0; i < repeatTime; i++) {
        Reg::RegTensor<CounterT> prevTileHistValue;
        Reg::RegTensor<uint32_t> stateBitValue;
        Reg::LoadAlign<CounterT, Reg::PostLiteral::POST_MODE_UPDATE>(prevTileHistValue, tilePrevHistValuePtr,
                                                                     RadixCounterTraits<CounterT>::LANES);
        // 32 位状态直接右移；64 位状态还需压缩为 B32 供统一计数。
        if constexpr (sizeof(CounterT) == sizeof(uint32_t)) {
            Reg::ShiftRights<uint32_t, int16_t>(stateBitValue, prevTileHistValue,
                                                RadixCounterTraits<CounterT>::STATE_SHIFT, maskB32);
            pRegSelect = maskB32;
        } else {
            Reg::MaskReg maskB64 = Reg::CreateMask<CounterT>();
            Reg::RegTensor<uint64_t> stateTmp;
            // 最高两位存放同步状态，其余位存放桶计数。
            Reg::ShiftRights<uint64_t, int16_t>(stateTmp, (Reg::RegTensor<uint64_t>&)prevTileHistValue,
                                                RadixCounterTraits<CounterT>::STATE_SHIFT, maskB64);
            Reg::Pack(stateBitValue, stateTmp);
            pRegSelect = Reg::CreateMask<uint32_t, Reg::MaskPattern::H>();
        }
        Reg::MaskReg maskNotInit;
        Reg::RegTensor<uint32_t> maskNotInitCount;
        Reg::Compares<uint32_t, CMPMODE::EQ>(maskNotInit, stateBitValue, NOT_INIT_MODE, pRegSelect);
        Reg::Select(maskNotInitCount, onesVector, zerosVector, maskNotInit);
        Reg::Add(notInitCount, notInitCount, maskNotInitCount, maskNotInit);
        Reg::MaskReg maskAggReady;
        Reg::RegTensor<uint32_t> maskAggReadyCount;
        Reg::Compares<uint32_t, CMPMODE::EQ>(maskAggReady, stateBitValue, AGG_READY_MODE, pRegSelect);
        Reg::Select(maskAggReadyCount, onesVector, zerosVector, maskAggReady);
        Reg::Add(aggReadyCount, aggReadyCount, maskAggReadyCount, maskAggReady);
        Reg::MaskReg maskPrefixReady;
        Reg::RegTensor<uint32_t> maskPrefixReadyCount;
        Reg::Compares<uint32_t, CMPMODE::EQ>(maskPrefixReady, stateBitValue, PREFIX_READY_MODE, pRegSelect);
        Reg::Select(maskPrefixReadyCount, onesVector, zerosVector, maskPrefixReady);
        Reg::Add(prefixReadyCount, prefixReadyCount, maskPrefixReadyCount, maskPrefixReady);
    }
    Reg::Reduce<Reg::ReduceType::SUM>(notInitCount, notInitCount, maskB32);
    Reg::StoreAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
        ubFlagTensorPtr, notInitCount, HIST_MASK_OUT_LEN, maskB32);
    Reg::Reduce<Reg::ReduceType::SUM>(aggReadyCount, aggReadyCount, maskB32);
    Reg::StoreAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
        ubFlagTensorPtr, aggReadyCount, HIST_MASK_OUT_LEN, maskB32);
    Reg::Reduce<Reg::ReduceType::SUM>(prefixReadyCount, prefixReadyCount, maskB32);
    Reg::StoreAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
        ubFlagTensorPtr, prefixReadyCount, HIST_MASK_OUT_LEN, maskB32);
}

template <typename CounterT>
__simd_vf__ inline void AccumulateLookbackVF(__ubuf__ CounterT* nowTileHistBufferPtr,
                                             __ubuf__ CounterT* tilePrevHistValuePtrCopy,
                                             __ubuf__ CounterT* nowTileHistBufferPtrCopy, uint16_t repeatTime)
{
    Reg::MaskReg predicateDefault = Reg::CreateMask<CounterT>();
    Reg::RegTensor<CounterT> histMask;
    Reg::Duplicate(histMask, RadixCounterTraits<CounterT>::VALUE_MASK, predicateDefault);
    for (uint16_t i = 0; i < repeatTime; i++) {
        Reg::RegTensor<CounterT> nowTileHistVal;
        Reg::RegTensor<CounterT> prevTileHistVal;
        Reg::LoadAlign<CounterT, Reg::PostLiteral::POST_MODE_UPDATE>(nowTileHistVal, nowTileHistBufferPtr,
                                                                     RadixCounterTraits<CounterT>::LANES);
        Reg::LoadAlign<CounterT, Reg::PostLiteral::POST_MODE_UPDATE>(prevTileHistVal, tilePrevHistValuePtrCopy,
                                                                     RadixCounterTraits<CounterT>::LANES);
        // 累加前一块的桶计数前，先去除状态位。
        Reg::And(nowTileHistVal, nowTileHistVal, histMask, predicateDefault);
        Reg::And(prevTileHistVal, prevTileHistVal, histMask, predicateDefault);
        Reg::Add(nowTileHistVal, nowTileHistVal, prevTileHistVal, predicateDefault);
        Reg::StoreAlign<CounterT, Reg::PostLiteral::POST_MODE_UPDATE>(
            nowTileHistBufferPtrCopy, nowTileHistVal, RadixCounterTraits<CounterT>::LANES, predicateDefault);
    }
}

template <typename CounterT>
__simd_vf__ inline void MarkPrefixReadyVF(__ubuf__ CounterT* histCumsumPtr, __ubuf__ CounterT* histCumsumPtrCopy,
                                          uint16_t repeatTime)
{
    Reg::RegTensor<CounterT> prefixReadyMask;
    Reg::RegTensor<CounterT> prefixRemainMask;
    Reg::MaskReg predicateDefault = Reg::CreateMask<CounterT>();
    Reg::Duplicate(prefixReadyMask, RadixCounterTraits<CounterT>::PREFIX_MASK, predicateDefault);
    Reg::Duplicate(prefixRemainMask, RadixCounterTraits<CounterT>::VALUE_MASK, predicateDefault);
    for (uint16_t repeat = 0; repeat < repeatTime; repeat++) {
        Reg::RegTensor<CounterT> keyCumsumValue;
        Reg::LoadAlign<CounterT, Reg::PostLiteral::POST_MODE_UPDATE>(keyCumsumValue, histCumsumPtr,
                                                                     RadixCounterTraits<CounterT>::LANES);
        // 保留累积计数，发布前缀就绪标志。
        Reg::And(keyCumsumValue, keyCumsumValue, prefixRemainMask, predicateDefault);
        Reg::Or(keyCumsumValue, keyCumsumValue, prefixReadyMask, predicateDefault);
        Reg::StoreAlign<CounterT, Reg::PostLiteral::POST_MODE_UPDATE>(
            histCumsumPtrCopy, keyCumsumValue, RadixCounterTraits<CounterT>::LANES, predicateDefault);
    }
}

// 多核 Radix 公共流水：键预处理 → 桶计数 → 跨块前缀 → 稳定散射。
// CounterT 决定计数和前缀宽度；中间索引取 CounterT/T2 较窄者，末轮输出 T2。
// Derived 提供初始化和散射，T1 为值类型，UT 为无符号键，isDescend 为排序方向。
template <typename Derived, typename T1, typename T2, typename UT, typename CounterT, uint64_t isDescend>
class RadixMoreCoreBase {
public:
    // 中间索引不宽于输出槽位；64 位计数搭配 int32 输出时仍按 uint32 搬运索引。
    using IndexT = std::conditional_t<sizeof(T2) == sizeof(uint32_t), uint32_t, CounterT>;
    static constexpr uint32_t INDEX_WORDS = sizeof(IndexT) / sizeof(uint32_t);
    GlobalTensor<T1> inputXGm_;
    GlobalTensor<T1> outValueGm_;
    GlobalTensor<uint32_t> outIdxGm_;
    // 值与原轴索引成对乒乓交换；索引的 uint32 视图仅作为搬运载体。
    GlobalTensor<T1> outValueDbWK_;
    GlobalTensor<uint32_t> outIdxDbWK_;
    // 保存本轮字节及每块的桶统计，散射阶段直接复用，避免再次构键和计数。
    GlobalTensor<uint8_t> xB8GmWk_;
    GlobalTensor<uint16_t> histTileGmWk_;
    GlobalTensor<uint16_t> tileBucketOffsetsGmWk_;
    // globalHist 保存各字节轮次、各块的带状态前缀；exclusiveBins 保存全局较小桶元素数。
    // 两者均使用 CounterT，但只有 globalHist 的最高两位用于 lookback 状态。
    GlobalTensor<uint32_t> globalHistGmWk_;
    GlobalTensor<CounterT> globalHistGmWkTmp_;
    GlobalTensor<uint32_t> exclusiveBinsGmWk_;
    GlobalTensor<CounterT> exclusiveBinsGmWkTmp_;
    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECIN, 1> inQueueIndex_;
    TQue<QuePosition::VECIN, 1> inQueueGlobalHist_;
    TQue<QuePosition::VECIN, 1> blockExclusiveInQue_;
    TQue<QuePosition::VECIN, 1> blockHistInQue_;
    // 清零、局部 Sort 和散射参数按阶段复用此区；不能跨消费者提前覆盖。
    TBuf<TPosition::VECCALC> tmpUb_;
    TQue<QuePosition::VECOUT, 1> blockUbFlagQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputB8Que_;
    TQue<QuePosition::VECOUT, 1> outIdxQueue_;
    TQue<QuePosition::VECOUT, 1> outValueQueue_;
    TQue<QuePosition::VECOUT, 1> blockHistFlagUbQue_;
    DoubleBufferSimd<T1> inputXDbGm_;
    DoubleBufferSimd<uint32_t> idxDbGm_;
    static constexpr SortConfig byteSortConfig{SortType::RADIX_SORT, false};
    int64_t totalDataNum_ = 0;
    uint32_t numTileData_ = 0;
    int64_t unsortedDimNum_ = 0;
    uint32_t unsortedDimParallel_ = 0;
    uint32_t lastDimTileNum_ = 0;
    uint32_t sortLoopTimes_ = 0;
    uint32_t lastDimRealCore_ = 0;
    uint32_t tmpUbSize_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t realCoreNum_ = 0;
    uint32_t clearCore1_ = 0;
    uint32_t clearCore0_ = 0;
    uint32_t clearSize_ = 0;
    uint32_t clearCount_ = 0;
    uint32_t clearCoreSize0_ = 0;
    uint32_t clearCoreSize1_ = 0;
    // GM/UB 搬运使用 uint32 视图；每个计数或中间索引占 COUNTER_WORDS 个载体元素。
    static constexpr uint32_t COUNTER_WORDS = sizeof(CounterT) / sizeof(uint32_t);
    DataCopyExtParams copyParams{1, 1, 0, 0, 0};
    uint64_t oneBlock_ = Ops::Base::GetUbBlockSize();

    // bins 清零嵌套在 histogram 核集合内，Host 必须保持 clearCore0_ <= clearCore1_。
    __aicore__ inline void ClearWorkspace()
    {
        if (this->blockIdx_ < this->clearCore1_) {
            AscendC::LocalTensor<uint32_t> tmpUb = this->tmpUb_.template Get<uint32_t>();
            Duplicate(tmpUb, static_cast<uint32_t>(0), this->clearSize_ * this->COUNTER_WORDS);
            event_t eventIdMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdMte3);
            if (this->blockIdx_ < this->clearCore0_) {
                this->copyParams.blockLen = this->clearCoreSize0_ * sizeof(CounterT);
                uint64_t offset = static_cast<uint64_t>(this->blockIdx_) * this->clearCoreSize0_ * this->COUNTER_WORDS;
                DataCopyPad(this->exclusiveBinsGmWk_[offset], tmpUb, this->copyParams);
            }
            for (uint32_t i = 0; i < this->clearCount_; i++) {
                uint64_t histGmOffset = (static_cast<uint64_t>(this->blockIdx_) * this->clearCoreSize1_ +
                                         static_cast<uint64_t>(i) * this->clearSize_) *
                                        this->COUNTER_WORDS;
                this->copyParams.blockLen = this->clearSize_ * sizeof(CounterT);
                DataCopyPad(this->globalHistGmWk_[histGmOffset], tmpUb, this->copyParams);
            }
        }
    }

    __aicore__ inline LocalTensor<UT> PreProcess(LocalTensor<T1> inputX, uint32_t numTileData, uint32_t round,
                                                 uint32_t tileId)
    {
        (void)tileId;
        LocalTensor<UT> inputXCopy = inputX.template ReinterpretCast<UT>();
        if constexpr (IsSameType<int32_t, T1>::value) {
            TwiddleInB32<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
        } else if constexpr (IsSameType<half, T1>::value || IsSameType<bfloat16_t, T1>::value) {
            if (static_cast<Derived*>(this)->ShouldCanonicalizeMedianNan(round)) {
                TwiddleInFp16Impl<T1, UT, isDescend, true>(inputX, inputXCopy, numTileData);
            } else {
                TwiddleInFp16<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
            }
        } else if constexpr (IsSameType<float, T1>::value) {
            if (static_cast<Derived*>(this)->ShouldCanonicalizeMedianNan(round)) {
                TwiddleInFp32Impl<T1, UT, isDescend, true>(inputX, inputXCopy, numTileData);
            } else {
                TwiddleInFp32<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
            }
        } else if constexpr (IsSameType<int16_t, T1>::value) {
            TwiddleInB16<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
        } else if constexpr (IsSameType<int8_t, T1>::value) {
            TwiddleInB8<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
        } else if constexpr (IsSameType<int64_t, T1>::value) {
            TwiddleInB64<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
        } else {
            if (isDescend == 1) {
                ReverseInputData<T1, UT>(inputX, inputXCopy, numTileData);
            }
        }
        return inputXCopy;
    }

    __aicore__ inline void PreGlobalExclusiveSum(LocalTensor<UT>& inputXCopy, LocalTensor<CounterT>& blockExclusiveUb,
                                                 LocalTensor<uint16_t>& histUb, LocalTensor<uint16_t>& bucketOffsetsUb,
                                                 LocalTensor<uint8_t>& inputB8Ub, uint32_t currTileSize, uint32_t round,
                                                 uint32_t tileId)
    {
        // 提取当前字节、生成块内计数，并累加本核各桶的独占前缀。
        if constexpr (sizeof(T1) == sizeof(int32_t)) {
            GetGlobalExclusiveSumB32<UT>(inputXCopy, blockExclusiveUb, histUb, bucketOffsetsUb, inputB8Ub, currTileSize,
                                         round);
        } else if constexpr (sizeof(T1) == sizeof(int16_t)) {
            GetGlobalExclusiveSumB16<UT>(inputXCopy, blockExclusiveUb, histUb, bucketOffsetsUb, inputB8Ub, currTileSize,
                                         round);
        } else if constexpr (sizeof(T1) == sizeof(int8_t)) {
            GetGlobalExclusiveSumB8<UT>(inputXCopy, blockExclusiveUb, histUb, bucketOffsetsUb, inputB8Ub, currTileSize,
                                        round);
        } else if constexpr (sizeof(T1) == sizeof(int64_t)) {
            GetGlobalExclusiveSumB64<UT>(inputXCopy, blockExclusiveUb, histUb, bucketOffsetsUb, inputB8Ub, currTileSize,
                                         round);
        }
    }

    __aicore__ inline void GetGlobalExclusiveSum(uint32_t round, uint32_t sortLoopRound, GlobalTensor<T1> inputX)
    {
        uint32_t startTileId = this->blockIdx_ % this->lastDimRealCore_;
        uint32_t unSortId = this->blockIdx_ / this->lastDimRealCore_;
        // unSortId 是当前批的行槽位；unsortedDimIndex 才是输入中的全局行号。
        // 工作区按槽位寻址，避免其容量随总行数 M 增长。
        uint64_t unsortedDimIndex = unSortId + static_cast<uint64_t>(sortLoopRound) * this->unsortedDimParallel_;
        int64_t xUnsortOffset = static_cast<int64_t>(unSortId) * this->totalDataNum_;
        if (unsortedDimIndex < static_cast<uint64_t>(this->unsortedDimNum_)) {
            LocalTensor<uint32_t> blockExclusiveUb = this->blockUbFlagQue_.template AllocTensor<uint32_t>();

            uint64_t exclusiveBinOffset =
                static_cast<uint64_t>(unSortId) * RADIX_SORT_NUM * sizeof(T1) + round * RADIX_SORT_NUM;
            // 以 uint32 视图清零全部桶；载体字数随 CounterT 变化。
            Duplicate(blockExclusiveUb, static_cast<uint32_t>(0), RADIX_SORT_NUM * COUNTER_WORDS);
            for (uint32_t tileId = startTileId; tileId < this->lastDimTileNum_; tileId += this->lastDimRealCore_) {
                // 长轴的 tileOffset 可能超过 int32，地址计算保持 uint64。
                uint64_t tileOffset = static_cast<uint64_t>(tileId) * this->numTileData_;
                uint64_t remainTileDataNum = this->totalDataNum_ - tileOffset;
                if (this->totalDataNum_ < tileOffset) {
                    break;
                }
                uint32_t currTileSize = static_cast<uint32_t>(
                    remainTileDataNum < static_cast<uint64_t>(this->numTileData_) ? remainTileDataNum :
                                                                                    this->numTileData_);
                LocalTensor<T1> xLocal = this->inQueueX_.template AllocTensor<T1>();

                // 首轮读取原始输入，后续轮次读取上一轮的双缓冲输出。
                if (round == 0) {
                    CopyDataIn<T1>(inputX[xUnsortOffset], xLocal, tileOffset, currTileSize);
                } else {
                    CopyDataIn<T1>(this->inputXDbGm_.Current()[xUnsortOffset], xLocal, tileOffset, currTileSize);
                }
                this->inQueueX_.EnQue(xLocal);
                xLocal = this->inQueueX_.template DeQue<T1>();
                // 提取字节前，将有符号数或浮点数转换为保序的无符号键。
                LocalTensor<UT> xUbCopy = PreProcess(xLocal, currTileSize, round, tileId);
                LocalTensor<uint8_t> inputB8Ub = this->inputB8Que_.template AllocTensor<uint8_t>();
                LocalTensor<uint16_t> histUb = this->outIdxQueue_.template AllocTensor<uint16_t>();
                LocalTensor<uint16_t> bucketOffsetsUb = this->outValueQueue_.template AllocTensor<uint16_t>();
                static_cast<Derived*>(this)->OnRadixInputLoaded(xLocal, currTileSize, round, tileId, bucketOffsetsUb);
                LocalTensor<CounterT> blockExclusiveUbTmp = blockExclusiveUb.template ReinterpretCast<CounterT>();
                PreGlobalExclusiveSum(xUbCopy, blockExclusiveUbTmp, histUb, bucketOffsetsUb, inputB8Ub, currTileSize,
                                      round, tileId);
                this->inQueueX_.FreeTensor(xLocal);
                this->inputB8Que_.template EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(inputB8Ub);
                inputB8Ub = this->inputB8Que_.template DeQue<QuePosition::VECCALC, QuePosition::VECOUT, uint8_t>();
                // 保存已提取字节，散射时直接复用。
                this->copyParams.blockCount = 1;
                this->copyParams.blockLen = currTileSize * sizeof(uint8_t);
                this->copyParams.srcStride = 0;
                this->copyParams.dstStride = 0;
                DataCopyPad(this->xB8GmWk_[xUnsortOffset + tileOffset], inputB8Ub, this->copyParams);
                this->inputB8Que_.FreeTensor(inputB8Ub);

                this->outIdxQueue_.EnQue(histUb);
                histUb = this->outIdxQueue_.template DeQue<uint16_t>();
                // 保存各块的 256 桶计数，供回溯计算跨块前缀。
                this->copyParams.blockCount = 1;
                this->copyParams.blockLen = RADIX_SORT_NUM * sizeof(uint16_t);
                this->copyParams.srcStride = 0;
                this->copyParams.dstStride = 0;
                DataCopyPad(
                    this->histTileGmWk_[static_cast<uint64_t>(unSortId) * RADIX_SORT_NUM * this->lastDimTileNum_ +
                                        static_cast<uint64_t>(tileId) * RADIX_SORT_NUM],
                    histUb, this->copyParams);
                this->outIdxQueue_.FreeTensor(histUb);

                this->outValueQueue_.EnQue(bucketOffsetsUb);
                bucketOffsetsUb = this->outValueQueue_.template DeQue<uint16_t>();
                // 保存块内独占前缀，供最终散射地址计算使用。
                DataCopyPad(this->tileBucketOffsetsGmWk_[static_cast<uint64_t>(unSortId) * RADIX_SORT_NUM *
                                                             this->lastDimTileNum_ +
                                                         static_cast<uint64_t>(tileId) * RADIX_SORT_NUM],
                            bucketOffsetsUb, this->copyParams);
                this->outValueQueue_.FreeTensor(bucketOffsetsUb);
            }
            this->blockUbFlagQue_.EnQue(blockExclusiveUb);
            blockExclusiveUb = this->blockUbFlagQue_.template DeQue<uint32_t>();
            if constexpr (sizeof(CounterT) == sizeof(uint32_t)) {
                // 32 位非负计数不超过 2^30-1，可直接使用 DMA 原子加。
                this->copyParams = {1, RADIX_SORT_NUM * sizeof(uint32_t), 0, 0, 0};
                SetAtomicAdd<int32_t>();
                DataCopyPad(this->exclusiveBinsGmWk_[exclusiveBinOffset], blockExclusiveUb, this->copyParams);
                SetAtomicNone();
            } else {
                // 使用 64 位原子加累积各核偏移。
                asc_vf_call<SimtGlobalOffset>(dim3(RADIX_SORT_NUM), exclusiveBinOffset,
                                              (__gm__ int64_t*)(this->exclusiveBinsGmWk_.GetPhyAddr()),
                                              (__ubuf__ CounterT*)(blockExclusiveUb.GetPhyAddr()));
            }
            static_cast<Derived*>(this)->OnRadixRoundComplete(round, sortLoopRound, blockExclusiveUb);
            this->blockUbFlagQue_.FreeTensor(blockExclusiveUb);
        }
    }

    __aicore__ inline void ScatterBlockHist2Global(LocalTensor<uint16_t> blockHist,
                                                   LocalTensor<CounterT> blockHistWithFlag,
                                                   GlobalTensor<CounterT> allblockHistToGm, uint32_t tileId,
                                                   uint32_t round)
    {
        uint32_t unSortId = this->blockIdx_ / this->lastDimRealCore_;
        int64_t unSortIdOffset = static_cast<int64_t>(unSortId) * this->lastDimTileNum_ * RADIX_SORT_NUM * sizeof(T1) +
                                 static_cast<int64_t>(round) * RADIX_SORT_NUM * this->lastDimTileNum_;
        __ubuf__ uint16_t* blockHistPtr = (__ubuf__ uint16_t*)blockHist.GetPhyAddr();
        __ubuf__ CounterT* blockHistWithFlagPtr = (__ubuf__ CounterT*)blockHistWithFlag.GetPhyAddr();
        // 每个桶值最高两位编码软同步状态：
        //   0：未初始化，继续轮询。
        //   1：局部计数就绪，可以累加当前块的计数。
        //   2：完整前缀就绪，累加后停止回溯。
        //   3：保留。
        asc_vf_call<MarkAggregateReadyVF<CounterT>>(dim3{1, 1, 1}, blockHistPtr, blockHistWithFlagPtr);
        this->blockHistFlagUbQue_.EnQue(blockHistWithFlag);
        blockHistWithFlag = this->blockHistFlagUbQue_.template DeQue<CounterT>();
        if (tileId < (this->lastDimTileNum_ - 1)) {
            DataCopyExtParams dataCopyParam;
            dataCopyParam.blockCount = 1;
            dataCopyParam.blockLen = RADIX_SORT_NUM * sizeof(CounterT);
            dataCopyParam.srcStride = 0;
            dataCopyParam.dstStride = 0;
            // 末块没有后继块需要回溯它，因此只由非末块发布计数状态。
            DataCopyPad(allblockHistToGm[unSortIdOffset + RADIX_SORT_NUM * tileId], blockHistWithFlag, dataCopyParam);
        }
        this->blockHistFlagUbQue_.FreeTensor(blockHistWithFlag);
    }

    __aicore__ inline void LookbackGlobal(LocalTensor<CounterT> nowTileHistBuffer,
                                          GlobalTensor<CounterT> allTileHistBuffer, LocalTensor<uint32_t> ubFlagTensor,
                                          uint32_t tileId, uint32_t round)
    {
        uint32_t unSortId = this->blockIdx_ / this->lastDimRealCore_;
        int64_t unSortIdOffset = static_cast<int64_t>(unSortId) * this->lastDimTileNum_ * RADIX_SORT_NUM * sizeof(T1) +
                                 static_cast<int64_t>(round) * RADIX_SORT_NUM * this->lastDimTileNum_;
        __ubuf__ CounterT* nowTileHistBufferPtr = (__ubuf__ CounterT*)nowTileHistBuffer.GetPhyAddr();

        constexpr uint16_t repeatTime = RADIX_SORT_NUM / RadixCounterTraits<CounterT>::LANES;

        // 向前回溯：遇到局部计数就绪的块，累加计数并继续。
        // 前缀就绪的块已包含此前所有块，累加后即可停止。
        for (int i = tileId - 1; i >= 0; --i) {
            int mode = -1;
            uint32_t histTileOffset = RADIX_SORT_NUM * i;
            __ubuf__ uint32_t* ubFlagTensorPtr = (__ubuf__ uint32_t*)ubFlagTensor.GetPhyAddr();
            while (true) {
                // 轮询直到前一块的全部 256 桶均已发布计数或前缀就绪状态。
                LocalTensor<CounterT> xLocal = this->inQueueGlobalHist_.template AllocTensor<CounterT>();
                CopyGlobalDataIn<CounterT>(allTileHistBuffer[unSortIdOffset], xLocal, histTileOffset, RADIX_SORT_NUM);
                this->inQueueGlobalHist_.EnQue(xLocal);
                LocalTensor<CounterT> tilePrevHistValue = this->inQueueGlobalHist_.template DeQue<CounterT>();
                __ubuf__ CounterT* tilePrevHistValuePtr = (__ubuf__ CounterT*)tilePrevHistValue.GetPhyAddr();
                asc_vf_call<CountLookbackStatesVF<CounterT>>(dim3{1, 1, 1}, tilePrevHistValuePtr, ubFlagTensorPtr,
                                                             repeatTime);
                event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventId);
                WaitFlag<HardEvent::V_S>(eventId);
                uint32_t aggReadyCountScalar = ubFlagTensorPtr[AGG_READY_COUNT_INDEX];
                uint32_t prefixReadyCountScalar = ubFlagTensorPtr[PREFIX_READY_COUNT_INDEX];
                if (aggReadyCountScalar == RADIX_SORT_NUM) {
                    mode = AGGREGATE_READY_FLAG;
                } else if (prefixReadyCountScalar == RADIX_SORT_NUM) {
                    mode = PREFIX_READY_FLAG;
                } else {
                    this->inQueueGlobalHist_.FreeTensor(tilePrevHistValue);
                    continue;
                }
                // 在本次快照仍存活时累加；同一前缀缓冲的读写指针分别递增，每段先读后写。
                __ubuf__ CounterT* nowTileHistBufferPtrCopy = nowTileHistBufferPtr;
                event_t eventIdWaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
                SetFlag<HardEvent::S_V>(eventIdWaitV);
                WaitFlag<HardEvent::S_V>(eventIdWaitV);
                asc_vf_call<AccumulateLookbackVF<CounterT>>(dim3{1, 1, 1}, nowTileHistBufferPtr, tilePrevHistValuePtr,
                                                            nowTileHistBufferPtrCopy, repeatTime);
                event_t accumulateDone = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(accumulateDone);
                WaitFlag<HardEvent::V_S>(accumulateDone);
                // 累加 VF 已完成读取，此时才允许下一次轮询复用队列缓冲。
                this->inQueueGlobalHist_.FreeTensor(tilePrevHistValue);
                break;
            }
            if (mode == PREFIX_READY_FLAG) {
                break;
            }
        }
    }

    __aicore__ inline void PublishPrefixReady(LocalTensor<CounterT>& blockHistWithFlag,
                                              GlobalTensor<CounterT> blockHistToGm, uint32_t tileId, uint32_t round)
    {
        uint32_t unSortId = this->blockIdx_ / this->lastDimRealCore_;
        int64_t unSortIdOffset = static_cast<int64_t>(unSortId) * this->lastDimTileNum_ * RADIX_SORT_NUM * sizeof(T1) +
                                 static_cast<int64_t>(round) * RADIX_SORT_NUM * this->lastDimTileNum_;
        __ubuf__ CounterT* histCumsumPtr = (__ubuf__ CounterT*)blockHistWithFlag.GetPhyAddr();
        // 原地附加状态位：每段读入寄存器后才覆盖同一段。
        __ubuf__ CounterT* histCumsumPtrCopy = histCumsumPtr;

        constexpr uint16_t repeatTime = RADIX_SORT_NUM / RadixCounterTraits<CounterT>::LANES;
        asc_vf_call<MarkPrefixReadyVF<CounterT>>(dim3{1, 1, 1}, histCumsumPtr, histCumsumPtrCopy, repeatTime);
        this->blockHistFlagUbQue_.EnQue(blockHistWithFlag);
        blockHistWithFlag = this->blockHistFlagUbQue_.template DeQue<CounterT>();
        DataCopyExtParams dataCopyParam;
        dataCopyParam.blockCount = 1;
        dataCopyParam.blockLen = RADIX_SORT_NUM * sizeof(CounterT);
        dataCopyParam.srcStride = 0;
        dataCopyParam.dstStride = 0;
        DataCopyPad(blockHistToGm[unSortIdOffset + RADIX_SORT_NUM * tileId], blockHistWithFlag, dataCopyParam);
    }

    __aicore__ inline void DataCopyWorkSpaceToUb(LocalTensor<uint8_t> inputX8Ub, LocalTensor<uint16_t> blockExclusiveUb,
                                                 LocalTensor<uint16_t> blockHistUb, uint32_t unSortId, uint32_t tileId,
                                                 uint32_t curSize)
    {
        uint64_t histTileBase = static_cast<uint64_t>(unSortId) * RADIX_SORT_NUM * this->lastDimTileNum_ +
                                static_cast<uint64_t>(tileId) * RADIX_SORT_NUM;
        DataCopyPadExtParams<uint16_t> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParam;
        dataCopyParam.blockCount = 1;
        dataCopyParam.blockLen = RADIX_SORT_NUM * sizeof(uint16_t);
        dataCopyParam.srcStride = 0;
        dataCopyParam.dstStride = 0;
        DataCopyPad(blockExclusiveUb, this->tileBucketOffsetsGmWk_[histTileBase], dataCopyParam, padParams);
        DataCopyPad(blockHistUb, this->histTileGmWk_[histTileBase], dataCopyParam, padParams);

        DataCopyPadExtParams<uint8_t> padParamsB8{false, 0, 0, 0};
        DataCopyExtParams dataCopyParam1;
        dataCopyParam1.blockCount = 1;
        dataCopyParam1.blockLen = curSize * sizeof(uint8_t);
        dataCopyParam1.srcStride = 0;
        dataCopyParam1.dstStride = 0;
        DataCopyPad(inputX8Ub,
                    this->xB8GmWk_[static_cast<uint64_t>(unSortId) * this->totalDataNum_ +
                                   static_cast<uint64_t>(tileId) * this->numTileData_],
                    dataCopyParam1, padParamsB8);
    }

    __aicore__ inline void ComputeOnePass(uint32_t round, uint32_t sortLoopRound, GlobalTensor<T1> inputXGm)
    {
        uint32_t startId = this->blockIdx_ % this->lastDimRealCore_;
        uint32_t unSortId = this->blockIdx_ / this->lastDimRealCore_;
        uint64_t unsortedDimIndex = unSortId + static_cast<uint64_t>(sortLoopRound) * this->unsortedDimParallel_;
        int64_t xUnsortOffset = static_cast<int64_t>(unSortId) * this->totalDataNum_;
        if (unsortedDimIndex < static_cast<uint64_t>(this->unsortedDimNum_)) {
            for (uint32_t tileId = startId; tileId < this->lastDimTileNum_; tileId += this->lastDimRealCore_) {
                uint64_t tileOffset = static_cast<uint64_t>(tileId) * this->numTileData_;
                uint64_t tileDataStart = static_cast<uint64_t>(tileId) * this->numTileData_;
                uint64_t remainTileDataNum = this->totalDataNum_ - tileDataStart;
                if (this->totalDataNum_ < tileDataStart) {
                    break;
                }
                uint32_t currTileSize = static_cast<uint32_t>(
                    remainTileDataNum < static_cast<uint64_t>(this->numTileData_) ? remainTileDataNum :
                                                                                    this->numTileData_);
                // 1. 加载本块值，以及计数阶段预存的当前字节、桶计数和块内桶起点。
                LocalTensor<T1> xLocal = this->inQueueX_.template AllocTensor<T1>();
                if (round == 0) {
                    CopyDataIn<T1>(inputXGm[xUnsortOffset], xLocal, tileOffset, currTileSize);
                } else {
                    CopyDataIn<T1>(this->inputXDbGm_.Current()[xUnsortOffset], xLocal, tileOffset, currTileSize);
                }
                this->inQueueX_.EnQue(xLocal);
                xLocal = this->inQueueX_.template DeQue<T1>();
                LocalTensor<uint8_t> inputX8Ub = this->inputB8Que_.template AllocTensor<uint8_t>();
                LocalTensor<uint16_t> blockExclusiveUb = this->blockExclusiveInQue_.template AllocTensor<uint16_t>();
                LocalTensor<uint16_t> blockHistUb = this->blockHistInQue_.template AllocTensor<uint16_t>();
                DataCopyWorkSpaceToUb(inputX8Ub, blockExclusiveUb, blockHistUb, unSortId, tileId, currTileSize);
                this->blockHistInQue_.EnQue(blockHistUb);
                this->blockExclusiveInQue_.EnQue(blockExclusiveUb);
                this->inputB8Que_.template EnQue<QuePosition::VECIN, QuePosition::VECCALC>(inputX8Ub);
                inputX8Ub = this->inputB8Que_.template DeQue<QuePosition::VECIN, QuePosition::VECCALC, uint8_t>();
                blockHistUb = this->blockHistInQue_.template DeQue<uint16_t>();

                // 2. 发布本块计数与就绪状态，允许后继块在本块排序期间推进 lookback。
                LocalTensor<CounterT> blockHistFlagUb = this->blockHistFlagUbQue_.template AllocTensor<CounterT>();
                ScatterBlockHist2Global(blockHistUb, blockHistFlagUb, this->globalHistGmWkTmp_, tileId, round);

                // 3. 按当前字节稳定排序本块，得到有序字节及其原块内位置。
                LocalTensor<uint8_t> shareTmpBuffer = this->tmpUb_.template Get<uint8_t>();
                AscendC::LocalTensor<uint32_t> sortedValueIndexLocal =
                    this->outIdxQueue_.template AllocTensor<uint32_t>();
                AscendC::LocalTensor<uint8_t> sortedDigitLocal = this->outValueQueue_.template AllocTensor<uint8_t>();
                AscendC::Sort<uint8_t, false, byteSortConfig>(sortedDigitLocal, sortedValueIndexLocal, inputX8Ub,
                                                              shareTmpBuffer, static_cast<uint32_t>(currTileSize));
                this->outValueQueue_.template EnQue<uint8_t>(sortedDigitLocal);
                this->outIdxQueue_.template EnQue<uint32_t>(sortedValueIndexLocal);
                this->inputB8Que_.FreeTensor(inputX8Ub);
                // 4. 复用计数缓冲累加前驱块，得到包含本块的同桶前缀；首块没有前驱。
                AscendC::LocalTensor<uint32_t> ubFlagTensor = this->blockUbFlagQue_.template AllocTensor<uint32_t>();
                // InitBuffer 只分配一个物理 buffer，此次重新申请保留已发布的本块计数。
                LocalTensor<CounterT> blockHistFlagUb1 = this->blockHistFlagUbQue_.template AllocTensor<CounterT>();
                if (tileId > 0) {
                    LookbackGlobal(blockHistFlagUb1, this->globalHistGmWkTmp_, ubFlagTensor, tileId, round);
                }
                this->blockUbFlagQue_.FreeTensor(ubFlagTensor);
                // 5. 发布完整前缀，供后继块提前结束回溯；末块没有后继，无需发布。
                if (tileId < (this->lastDimTileNum_ - 1)) {
                    PublishPrefixReady(blockHistFlagUb1, this->globalHistGmWkTmp_, tileId, round);
                }
                this->blockHistFlagUbQue_.FreeTensor(blockHistFlagUb1);
                // 6. 首轮生成原轴索引，后续轮次读取中间索引，再按全局桶位置稳定散射。
                LocalTensor<uint32_t> xIndexLocal;
                if (round != 0) {
                    xIndexLocal = this->inQueueIndex_.template AllocTensor<uint32_t>();
                    CopyIndexDataIn<IndexT>(this->idxDbGm_.Current()[xUnsortOffset * INDEX_WORDS], xIndexLocal,
                                            tileOffset * INDEX_WORDS, currTileSize);
                    this->inQueueIndex_.EnQue(xIndexLocal);
                    xIndexLocal = this->inQueueIndex_.template DeQue<uint32_t>();
                }
                sortedValueIndexLocal = this->outIdxQueue_.template DeQue<uint32_t>();
                sortedDigitLocal = this->outValueQueue_.template DeQue<uint8_t>();
                AscendC::LocalTensor<CounterT> blockDataInGlobalPos =
                    this->blockUbFlagQue_.template AllocTensor<CounterT>();
                blockExclusiveUb = this->blockExclusiveInQue_.template DeQue<uint16_t>();
                // 同一物理 buffer 改用 uint32 搬运视图，散射仍按 CounterT 前缀解释。
                LocalTensor<uint32_t> blockHistFlagUb2 = this->blockHistFlagUbQue_.template AllocTensor<uint32_t>();
                static_cast<Derived*>(this)->ScatterKeysGlobal(xLocal, sortedValueIndexLocal, xIndexLocal,
                                                               sortedDigitLocal, blockExclusiveUb, blockDataInGlobalPos,
                                                               blockHistFlagUb2, blockHistUb, round, tileDataStart,
                                                               currTileSize, sortLoopRound);
                if (round != 0) {
                    this->inQueueIndex_.FreeTensor(xIndexLocal);
                }
                this->blockHistFlagUbQue_.FreeTensor(blockHistFlagUb2);
                this->inQueueX_.FreeTensor(xLocal);
                this->blockHistInQue_.FreeTensor(blockHistUb);
                this->blockUbFlagQue_.FreeTensor(blockDataInGlobalPos);
                this->blockExclusiveInQue_.FreeTensor(blockExclusiveUb);
                this->outIdxQueue_.FreeTensor(sortedValueIndexLocal);
                this->outValueQueue_.FreeTensor(sortedDigitLocal);
            }
            this->idxDbGm_.selector_ ^= 1;
            this->inputXDbGm_.selector_ ^= 1;
        }
    }

    __aicore__ inline void ProcessRadix(GlobalTensor<T1> inputXGm, int64_t gmOffset, uint32_t sortLoopRound)
    {
        this->ClearWorkspace();
        SyncAll();

        // 一字节类型只有一轮，其余类型均为偶数轮；按奇偶设置初始方向，
        // 使末轮直接写最终输出，省去一次整轴拷贝。T2 仅影响输出索引地址的步长。
        if constexpr (sizeof(T2) == sizeof(uint32_t)) {
            if constexpr (sizeof(T1) == sizeof(int8_t)) {
                this->inputXDbGm_.SetDoubleBuffer(this->outValueDbWK_, this->outValueGm_[gmOffset]);
                this->idxDbGm_.SetDoubleBuffer(this->outIdxDbWK_, this->outIdxGm_[gmOffset]);
            } else {
                this->inputXDbGm_.SetDoubleBuffer(this->outValueGm_[gmOffset], this->outValueDbWK_);
                this->idxDbGm_.SetDoubleBuffer(this->outIdxGm_[gmOffset], this->outIdxDbWK_);
            }
        } else {
            if constexpr (sizeof(T1) == sizeof(int8_t)) {
                this->inputXDbGm_.SetDoubleBuffer(this->outValueDbWK_, this->outValueGm_[gmOffset]);
                this->idxDbGm_.SetDoubleBuffer(this->outIdxDbWK_, this->outIdxGm_[gmOffset * INT64_INDEX_SCALE]);
            } else {
                this->inputXDbGm_.SetDoubleBuffer(this->outValueGm_[gmOffset], this->outValueDbWK_);
                this->idxDbGm_.SetDoubleBuffer(this->outIdxGm_[gmOffset * INT64_INDEX_SCALE], this->outIdxDbWK_);
            }
        }
        // 所有启动核都执行以下屏障，包括尾批中没有有效行的核。
        // 第一个屏障保护跨核桶偏移，第二个保证本轮散射完成后再读下一字节。
        for (uint32_t round = 0; round < static_cast<uint32_t>(sizeof(T1)); round++) {
            this->GetGlobalExclusiveSum(round, sortLoopRound, inputXGm);
            SyncAll();
            this->ComputeOnePass(round, sortLoopRound, inputXGm);
            SyncAll();
        }
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx_ >= this->realCoreNum_) {
            return;
        }
        // 每批覆盖同一份 P_act 行工作区；ProcessRadix 先清零，末轮同步后才能进入下一批。
        for (uint32_t i = 0; i < this->sortLoopTimes_; i++) {
            int64_t loopOffset = static_cast<int64_t>(i) * this->unsortedDimParallel_ * this->totalDataNum_;
            this->ProcessRadix(this->inputXGm_[loopOffset], loopOffset, i);
        }
    }
};

} // namespace RadixSortCommon

#endif // RADIX_MORE_CORE_BASE_H
