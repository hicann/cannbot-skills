/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_MERGE_SORT_H
#define SORT_MERGE_SORT_H

#include <cmath>
#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "common/merge_sort_constants.h"
#include "common/ping_pong_merge_sort.h"

namespace Sort {
using namespace AscendC;

using MergeSortConstants::DEALING_SORT_NUM_ONCE;
using MergeSortConstants::MERGE_LIST_MAX_NUM;
using MergeSortConstants::SIGN_FLIP_ADDEND_FP32;
using MergeSortConstants::SIGN_FLIP_ADDEND_FP16;

constexpr uint32_t SORT_QUEUE_UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();
constexpr uint32_t SORT_PROPOSAL_BYTES = 8;

__simd_vf__ inline void InitMergeIndicesVF(__ubuf__ int32_t* indexPtr, uint32_t vectorLength, uint16_t repeatTimes,
                                           uint32_t remaining)
{
    Reg::RegTensor<int32_t> arange;
    Reg::RegTensor<int32_t> index;
    Reg::Arange(arange, 0);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        Reg::MaskReg mask = Reg::UpdateMask<uint32_t>(remaining);
        Reg::Adds(index, arange, i * vectorLength, mask);
        Reg::StoreAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(indexPtr, index, vectorLength, mask);
    }
}

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t IsDescend>
class MergeSort {
public:
    __aicore__ inline MergeSort() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex, GM_ADDR workspace,
                                const SortRegBaseTilingData* __restrict tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData();
    __aicore__ inline uint32_t GetRowCount(int64_t rowStart) const;
    __aicore__ inline void ProcessCurrentBlock(LocalTensor<T1> xLocal, uint32_t rowCount);
    __aicore__ inline void CopyOutCurrentBlock(uint64_t gmOffset, uint32_t rowCount);
    __aicore__ inline void ProcessSingleBlock(uint64_t gmOffset, int64_t rowStart);
    __aicore__ inline void CopyIn(GlobalTensor<T1> inputX, uint64_t tileOffset, uint32_t rowCount);
    __aicore__ inline void CopyOut(uint64_t gmOffset, uint64_t tileOffset, uint32_t rowCount);
    __aicore__ inline void SortRows(LocalTensor<T1> xLocal, LocalTensor<T1> valueLocal,
                                    LocalTensor<uint32_t> indexLocal, uint32_t rowCount);
    __aicore__ inline void SortBatchedRows(LocalTensor<T1> xLocal, LocalTensor<T1> valueLocal,
                                           LocalTensor<uint32_t> indexLocal, uint32_t rowCount);
    __aicore__ inline void SortRowsBf16(LocalTensor<bfloat16_t> xLocal, LocalTensor<T1> valueLocal,
                                        LocalTensor<uint32_t> indexLocal, uint32_t rowCount);
    __aicore__ inline void SortOneRow(LocalTensor<CONVERT_TYPE> dstValue, LocalTensor<uint32_t> dstIndex,
                                      LocalTensor<CONVERT_TYPE> srcValue, uint32_t repeatTimes);
    __aicore__ inline void FlipSignBit(LocalTensor<CONVERT_TYPE> tensor, uint32_t count);

    GlobalTensor<T1> inputGm_;
    GlobalTensor<T1> outputValueGm_;
    GlobalTensor<T2> outputIndexGm_;

    TPipe* pipe_ = nullptr;
    const SortRegBaseTilingData* tilingData_ = nullptr;
    static constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
    // InitBuffer 选择单缓冲或双缓冲，队列事件保护跨批次复用。
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outputIndexQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outputValueQueue_;
    TBuf<TPosition::VECCALC> indexBuffer_;
    TBuf<TPosition::VECCALC> proposalPingBuffer_;
    TBuf<TPosition::VECCALC> proposalPongBuffer_;
    TBuf<TPosition::VECCALC> castInputBuffer_;

    LocalTensor<uint32_t> baseIndex_;
    uint32_t blockIdx_ = 0;
    uint32_t rowsPerBatch_ = 0;
    uint32_t axisLength_ = 0;
    int64_t totalRows_ = 0;
    int64_t coreRowStart_ = 0;
    int64_t rowStride_ = 0;
    uint64_t coreGmOffset_ = 0;
    uint64_t gmStride_ = 0;
    uint32_t loopCount_ = 0;
    uint32_t alignedAxis_ = 0;
    bool batchMerge_ = false;
};

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t IsDescend>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, IsDescend>::Init(
    GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex, GM_ADDR workspace, const SortRegBaseTilingData* __restrict tilingData,
    TPipe* pipe)
{
    (void)workspace;
    blockIdx_ = GetBlockIdx();
    pipe_ = pipe;
    tilingData_ = tilingData;
    ParseTilingData();

    inputGm_.SetGlobalBuffer((__gm__ T1*)x);
    outputValueGm_.SetGlobalBuffer((__gm__ T1*)value);
    outputIndexGm_.SetGlobalBuffer((__gm__ T2*)sortIndex);

    uint32_t bufferNum = tilingData_->keyParams4;
    pipe_->InitBuffer(inputQueue_, bufferNum, tilingData_->keyParams1);
    pipe_->InitBuffer(outputValueQueue_, bufferNum, tilingData_->keyParams1);
    pipe_->InitBuffer(outputIndexQueue_, bufferNum, tilingData_->keyParams2);

    // 每个提案占 8 字节，各层归并交替使用缓冲，避免输入输出重叠。
    uint32_t indexCount = alignedAxis_;
    if constexpr (IsSameType<half, T1>::value && IsSameType<int64_t, T2>::value) {
        if (batchMerge_) {
            indexCount *= rowsPerBatch_;
        }
    }
    uint32_t proposalBytes = indexCount * SORT_PROPOSAL_BYTES;
    pipe_->InitBuffer(indexBuffer_, indexCount * sizeof(uint32_t));
    pipe_->InitBuffer(proposalPingBuffer_, proposalBytes);
    pipe_->InitBuffer(proposalPongBuffer_, proposalBytes);
    if constexpr (IsSameType<bfloat16_t, T1>::value) {
        pipe_->InitBuffer(castInputBuffer_, alignedAxis_ * sizeof(CONVERT_TYPE) * rowsPerBatch_);
    }

    baseIndex_ = indexBuffer_.AllocTensor<uint32_t>();
    __ubuf__ int32_t* indexPtr = (__ubuf__ int32_t*)baseIndex_.GetPhyAddr();
    uint32_t vectorLength = Ops::Base::GetVRegSize() / sizeof(int32_t);
    uint16_t repeatTimes = CeilDivision(indexCount, vectorLength);
    uint32_t remaining = indexCount;
    asc_vf_call<InitMergeIndicesVF>(dim3{1, 1, 1}, indexPtr, vectorLength, repeatTimes, remaining);
}

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t IsDescend>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, IsDescend>::ParseTilingData()
{
    rowsPerBatch_ = tilingData_->keyParams0;
    axisLength_ = tilingData_->numTileDataSize;
    totalRows_ = tilingData_->unsortedDimNum;
    loopCount_ = tilingData_->sortLoopTimes;
    alignedAxis_ = tilingData_->keyParams3;
    batchMerge_ = tilingData_->keyParams5 != 0U;
    // Tiling 计数可用 uint32，但行号及元素偏移的乘积须在运算前提升到 64 位。
    coreRowStart_ = static_cast<int64_t>(blockIdx_) * rowsPerBatch_;
    rowStride_ = static_cast<int64_t>(tilingData_->unsortedDimParallel) * rowsPerBatch_;
    coreGmOffset_ = static_cast<uint64_t>(coreRowStart_) * axisLength_;
    gmStride_ = static_cast<uint64_t>(rowStride_) * axisLength_;
}

// Sort32/MrgSort 生成降序提案。升序时先把浮点值 x 变为 -x，降序排列 -x 后再恢复符号，
// 即得到原值升序；降序则直接使用原值。此处对位模式加符号位常量，模定宽加法等价于 XOR，
// 只改变符号位、保留其余位，两次应用即可恢复；它不是 Radix 的无符号保序键变换。
template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t IsDescend>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, IsDescend>::FlipSignBit(LocalTensor<CONVERT_TYPE> tensor,
                                                                               uint32_t count)
{
    if constexpr (IsSameType<float, CONVERT_TYPE>::value) {
        LocalTensor<int32_t> castTensor = tensor.template ReinterpretCast<int32_t>();
        Adds(castTensor, castTensor, SIGN_FLIP_ADDEND_FP32, count);
    } else if constexpr (IsSameType<half, CONVERT_TYPE>::value) {
        LocalTensor<int16_t> castTensor = tensor.template ReinterpretCast<int16_t>();
        Adds(castTensor, castTensor, SIGN_FLIP_ADDEND_FP16, count);
    }
}

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t IsDescend>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, IsDescend>::SortOneRow(LocalTensor<CONVERT_TYPE> dstValue,
                                                                              LocalTensor<uint32_t> dstIndex,
                                                                              LocalTensor<CONVERT_TYPE> srcValue,
                                                                              uint32_t repeatTimes)
{
    LocalTensor<CONVERT_TYPE> ping = proposalPingBuffer_.Get<CONVERT_TYPE>();
    LocalTensor<CONVERT_TYPE> pong = proposalPongBuffer_.Get<CONVERT_TYPE>();
    bool sourceIsPing = PingPongMergeSortCommon::SortToProposal(ping, pong, srcValue, baseIndex_, repeatTimes);
    if (sourceIsPing) {
        Extract(dstValue, dstIndex, ping, repeatTimes);
    } else {
        Extract(dstValue, dstIndex, pong, repeatTimes);
    }
}

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t IsDescend>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, IsDescend>::SortBatchedRows(LocalTensor<T1> xLocal,
                                                                                   LocalTensor<T1> valueLocal,
                                                                                   LocalTensor<uint32_t> indexLocal,
                                                                                   uint32_t rowCount)
{
    uint32_t repeats = alignedAxis_ / DEALING_SORT_NUM_ONCE;
    if constexpr (IsDescend == 0) {
        FlipSignBit(xLocal, alignedAxis_ * rowCount);
    }
    LocalTensor<half> ping = proposalPingBuffer_.Get<half>();
    LocalTensor<half> pong = proposalPongBuffer_.Get<half>();
    // Sort32 和 Extract 处理连续行；MergeStage 仅在行内归并。
    // 每层先处理全部行，再进入下一层归并。
    Sort32(ping, xLocal, baseIndex_, repeats * rowCount);
    uint32_t runLength = DEALING_SORT_NUM_ONCE;
    uint32_t runCount = repeats;
    bool inPing = true;
    while (runCount > 1U) {
        for (uint32_t row = 0; row < rowCount; ++row) {
            uint32_t offset = GetSortOffset<half>(row * alignedAxis_);
            if (inPing) {
                PingPongMergeSortCommon::MergeStage(pong[offset], ping[offset], alignedAxis_, runLength, runCount);
            } else {
                PingPongMergeSortCommon::MergeStage(ping[offset], pong[offset], alignedAxis_, runLength, runCount);
            }
        }
        inPing = !inPing;
        runLength *= MERGE_LIST_MAX_NUM;
        runCount = (runCount + MERGE_LIST_MAX_NUM - 1U) / MERGE_LIST_MAX_NUM;
    }
    Extract(valueLocal, indexLocal, inPing ? ping : pong, repeats * rowCount);
    // 合批 Sort32 产生批内索引，输出时转换为原行内索引。
    for (uint32_t row = 1; row < rowCount; ++row) {
        uint32_t offset = row * alignedAxis_;
        LocalTensor<int32_t> indices = indexLocal[offset].template ReinterpretCast<int32_t>();
        Adds(indices, indices, -static_cast<int32_t>(offset), alignedAxis_);
    }
    if constexpr (IsDescend == 0) {
        FlipSignBit(valueLocal, alignedAxis_ * rowCount);
    }
}

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t IsDescend>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, IsDescend>::SortRows(LocalTensor<T1> xLocal,
                                                                            LocalTensor<T1> valueLocal,
                                                                            LocalTensor<uint32_t> indexLocal,
                                                                            uint32_t rowCount)
{
    if constexpr (IsSameType<half, T1>::value && IsSameType<int64_t, T2>::value) {
        if (batchMerge_ && rowCount >= SORT_BATCH_MERGE_MIN_ROWS) {
            SortBatchedRows(xLocal, valueLocal, indexLocal, rowCount);
            return;
        }
    }
    uint32_t repeatTimes = alignedAxis_ / DEALING_SORT_NUM_ONCE;
    if constexpr (IsDescend == 0) {
        FlipSignBit(xLocal, alignedAxis_ * rowCount);
    }
    for (uint32_t row = 0, rowOffset = 0; row < rowCount; ++row, rowOffset += alignedAxis_) {
        SortOneRow(valueLocal[rowOffset], indexLocal[rowOffset], xLocal[rowOffset], repeatTimes);
    }
    if constexpr (IsDescend == 0) {
        FlipSignBit(valueLocal, alignedAxis_ * rowCount);
    }
}

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t IsDescend>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, IsDescend>::SortRowsBf16(LocalTensor<bfloat16_t> xLocal,
                                                                                LocalTensor<T1> valueLocal,
                                                                                LocalTensor<uint32_t> indexLocal,
                                                                                uint32_t rowCount)
{
    LocalTensor<CONVERT_TYPE> castInput = castInputBuffer_.Get<CONVERT_TYPE>();
    Cast(castInput, xLocal, RoundMode::CAST_NONE, alignedAxis_ * rowCount);
    if constexpr (IsDescend == 0) {
        FlipSignBit(castInput, alignedAxis_ * rowCount);
    }
    uint32_t repeatTimes = alignedAxis_ / DEALING_SORT_NUM_ONCE;
    for (uint32_t row = 0, rowOffset = 0; row < rowCount; ++row, rowOffset += alignedAxis_) {
        // Extract 前 SortToProposal 已读完本行，可复用转换输入缓冲存放结果。
        SortOneRow(castInput[rowOffset], indexLocal[rowOffset], castInput[rowOffset], repeatTimes);
    }
    if constexpr (IsDescend == 0) {
        FlipSignBit(castInput, alignedAxis_ * rowCount);
    }
    Cast(valueLocal, castInput, RoundMode::CAST_RINT, alignedAxis_ * rowCount);
}

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t IsDescend>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, IsDescend>::CopyIn(GlobalTensor<T1> inputX, uint64_t tileOffset,
                                                                          uint32_t rowCount)
{
    LocalTensor<T1> xLocal = inputQueue_.AllocTensor<T1>();
    T1 paddingValue = static_cast<T1>(NAN);
    if constexpr (IsDescend == 1) {
        paddingValue = static_cast<T1>(-INFINITY);
    }

    uint32_t alignedCopyElements = RoundUpAlign(axisLength_ * sizeof(T1)) / sizeof(T1);
    uint32_t dmaPaddingElements = alignedCopyElements - axisLength_;
    uint32_t vectorPaddingElements = alignedAxis_ - alignedCopyElements;
    uint32_t dstStride = (vectorPaddingElements * sizeof(T1)) / SORT_QUEUE_UB_BLOCK_BYTES;

    // DataCopyPad 仅补一个 32 字节块以内的尾部；MTE2 补字节对齐尾部，Vector 补其余 Sort32 尾部。
    // 两段写入互不重叠，因此 CopyIn 不需额外 V_MTE2 事件。
    DataCopyPadExtParams<T1> padParams{true, 0, static_cast<uint8_t>(dmaPaddingElements), paddingValue};
    DataCopyExtParams copyParams{static_cast<uint16_t>(rowCount), static_cast<uint32_t>(axisLength_ * sizeof(T1)), 0,
                                 static_cast<int64_t>(dstStride), 0};
    DataCopyPad(xLocal, inputX[tileOffset], copyParams, padParams);

    constexpr uint32_t maxDuplicateRepeatStride = 255U;
    if (vectorPaddingElements > 0U) {
        uint32_t rowStride = (alignedAxis_ * sizeof(T1)) / SORT_QUEUE_UB_BLOCK_BYTES;
        if (rowCount == 1U || rowStride <= maxDuplicateRepeatStride) {
            uint8_t repeatStride = rowCount == 1U ? 0U : static_cast<uint8_t>(rowStride);
            Duplicate(xLocal[alignedCopyElements], paddingValue, static_cast<uint64_t>(vectorPaddingElements),
                      static_cast<uint8_t>(rowCount), 1U, repeatStride);
        } else {
            for (uint32_t row = 0U; row < rowCount; ++row) {
                Duplicate(xLocal[row * alignedAxis_ + alignedCopyElements], paddingValue,
                          static_cast<int32_t>(vectorPaddingElements));
            }
        }
    }
    inputQueue_.EnQue(xLocal);
}

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t IsDescend>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, IsDescend>::CopyOut(uint64_t gmOffset, uint64_t tileOffset,
                                                                           uint32_t rowCount)
{
    uint32_t alignedValueElements = RoundUpAlign(axisLength_ * sizeof(T1)) / sizeof(T1);
    uint32_t valueStride = ((alignedAxis_ - alignedValueElements) * sizeof(T1)) / SORT_QUEUE_UB_BLOCK_BYTES;
    uint32_t alignedIndexElements = RoundUpAlign(axisLength_ * sizeof(T2)) / sizeof(T2);
    uint32_t indexStride = ((alignedAxis_ - alignedIndexElements) * sizeof(T2)) / SORT_QUEUE_UB_BLOCK_BYTES;

    LocalTensor<T1> valueLocal = outputValueQueue_.DeQue<T1>();
    LocalTensor<T2> indexLocal = outputIndexQueue_.DeQue<T2>();
    DataCopyExtParams valueParams{static_cast<uint16_t>(rowCount), static_cast<uint32_t>(axisLength_ * sizeof(T1)),
                                  valueStride, 0, 0};
    DataCopyPad(outputValueGm_[gmOffset + tileOffset], valueLocal, valueParams);
    DataCopyExtParams indexParams{static_cast<uint16_t>(rowCount), static_cast<uint32_t>(axisLength_ * sizeof(T2)),
                                  indexStride, 0, 0};
    DataCopyPad(outputIndexGm_[gmOffset + tileOffset], indexLocal, indexParams);
    outputIndexQueue_.FreeTensor(indexLocal);
    outputValueQueue_.FreeTensor(valueLocal);
}

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t IsDescend>
__aicore__ inline uint32_t MergeSort<T1, T2, CONVERT_TYPE, IsDescend>::GetRowCount(int64_t rowStart) const
{
    if (rowStart >= totalRows_) {
        return 0U;
    }
    uint32_t rowCount = rowsPerBatch_;
    int64_t remainingRows = totalRows_ - rowStart;
    if (remainingRows < static_cast<int64_t>(rowsPerBatch_)) {
        rowCount = static_cast<uint32_t>(remainingRows);
    }
    return rowCount;
}

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t IsDescend>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, IsDescend>::ProcessCurrentBlock(LocalTensor<T1> xLocal,
                                                                                       uint32_t rowCount)
{
    LocalTensor<T1> valueLocal = outputValueQueue_.AllocTensor<T1>();

    LocalTensor<int64_t> indexInt64Local;
    LocalTensor<uint32_t> indexUint32Local;
    if constexpr (IsSameType<int64_t, T2>::value) {
        indexInt64Local = outputIndexQueue_.AllocTensor<int64_t>();
        uint32_t rowElements = alignedAxis_ * rowsPerBatch_;
        indexUint32Local = indexInt64Local.template ReinterpretCast<uint32_t>()[rowElements];
    } else {
        indexUint32Local = outputIndexQueue_.AllocTensor<uint32_t>();
    }

    if constexpr (IsSameType<bfloat16_t, T1>::value) {
        SortRowsBf16(xLocal, valueLocal, indexUint32Local, rowCount);
    } else {
        SortRows(xLocal, valueLocal, indexUint32Local, rowCount);
    }

    if constexpr (IsSameType<int64_t, T2>::value) {
        LocalTensor<int32_t> indexInt32Local = indexUint32Local.template ReinterpretCast<int32_t>();
        Cast(indexInt64Local, indexInt32Local, RoundMode::CAST_NONE, rowCount * alignedAxis_);
        outputIndexQueue_.EnQue<int64_t>(indexInt64Local);
    } else {
        outputIndexQueue_.EnQue<uint32_t>(indexUint32Local);
    }
    outputValueQueue_.EnQue<T1>(valueLocal);
    inputQueue_.FreeTensor(xLocal);
}

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t IsDescend>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, IsDescend>::CopyOutCurrentBlock(uint64_t gmOffset,
                                                                                       uint32_t rowCount)
{
    CopyOut(gmOffset, coreGmOffset_, rowCount);
}

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t IsDescend>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, IsDescend>::ProcessSingleBlock(uint64_t gmOffset,
                                                                                      int64_t rowStart)
{
    uint32_t rowCount = GetRowCount(rowStart);
    if (rowCount == 0U) {
        return;
    }

    CopyIn(inputGm_, gmOffset + coreGmOffset_, rowCount);
    LocalTensor<T1> xLocal = inputQueue_.DeQue<T1>();
    ProcessCurrentBlock(xLocal, rowCount);
    CopyOutCurrentBlock(gmOffset, rowCount);
}

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t IsDescend>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, IsDescend>::Process()
{
    if (blockIdx_ >= GetBlockNum()) {
        return;
    }

    int64_t rowStart = coreRowStart_;
    uint64_t gmOffset = 0;
    for (uint32_t loop = 0; loop < loopCount_; ++loop) {
        ProcessSingleBlock(gmOffset, rowStart);
        rowStart += rowStride_;
        gmOffset += gmStride_;
    }
}

} // namespace Sort

#endif // SORT_MERGE_SORT_H
