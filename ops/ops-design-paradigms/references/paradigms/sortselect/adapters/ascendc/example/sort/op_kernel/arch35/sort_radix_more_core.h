/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RADIX_SORT_MORE_CORE_H
#define RADIX_SORT_MORE_CORE_H

#include <type_traits>
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "common/util_type_simd.h"
#include "sort_tiling_data.h"
#include "simt_api/asc_simt.h"
#include "common/radix_more_core_base.h"

namespace Sort {
using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;
using namespace RadixSortCommon;
using RadixSortCommon::THREAD_DIM_NUM;

// Sort 完成后复用其 scratch；三个 uint64 偏移共占 24 字节。
__aicore__ inline LocalTensor<uint64_t> PrepareScatterArgs(TBuf<TPosition::VECCALC>& scratch, int64_t tileDataStart,
                                                           uint64_t outputOffset, uint64_t binOffset)
{
    auto args = scratch.Get<uint64_t>();
    event_t vectorDone = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(vectorDone);
    WaitFlag<HardEvent::V_S>(vectorDone);
    args.SetValue(SCATTER_TILE_START, static_cast<uint64_t>(tileDataStart));
    args.SetValue(SCATTER_OUTPUT_OFFSET, outputOffset);
    args.SetValue(SCATTER_BIN_OFFSET, binOffset);
    event_t scalarDone = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(scalarDone);
    WaitFlag<HardEvent::S_V>(scalarDone);
    return args;
}

template <typename T1, typename T2, typename UT, typename CounterT, uint64_t isDescend>
class SortRadixMoreCore : public RadixSortCommon::RadixMoreCoreBase<SortRadixMoreCore<T1, T2, UT, CounterT, isDescend>,
                                                                    T1, T2, UT, CounterT, isDescend> {
    using Base = RadixSortCommon::RadixMoreCoreBase<SortRadixMoreCore<T1, T2, UT, CounterT, isDescend>, T1, T2, UT,
                                                    CounterT, isDescend>;
    using IndexT = typename Base::IndexT;
    friend Base;

public:
    __aicore__ inline SortRadixMoreCore(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex, GM_ADDR workspace,
                                const SortRegBaseTilingData* __restrict tilingData, TPipe* pipe);

protected:
    __aicore__ inline bool ShouldCanonicalizeMedianNan(uint32_t) const
    {
        return false;
    }
    __aicore__ inline void OnRadixInputLoaded(LocalTensor<T1>, uint32_t, uint32_t, uint32_t, LocalTensor<uint16_t>) {}
    __aicore__ inline void OnRadixRoundComplete(uint32_t, uint32_t, LocalTensor<uint32_t>) {}
    __aicore__ inline void ParseTilingData();
    __aicore__ inline void ScatterKeysGlobal(LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal,
                                             LocalTensor<uint32_t> xInputIndexLocal,
                                             LocalTensor<uint8_t> sortedDigitLocal,
                                             LocalTensor<uint16_t> blockExclusiveSum,
                                             LocalTensor<CounterT> blockDataInGlobalPos,
                                             LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist,
                                             uint32_t round, int64_t tileDataStart, uint32_t currentTileSize,
                                             uint32_t sortLoopRound);
    template <typename OutputIndex, bool FirstPass>
    __aicore__ inline void ScatterOutput(LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal,
                                         LocalTensor<uint32_t> xInputIndexLocal, LocalTensor<uint8_t> sortedDigitLocal,
                                         LocalTensor<uint16_t> blockExclusiveSum,
                                         LocalTensor<CounterT> blockDataInGlobalPos,
                                         LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist,
                                         uint32_t round, int64_t tileDataStart, uint32_t currentTileSize);

    const SortRegBaseTilingData* tilingData_;
};

template <typename T1, typename T2, typename UT, typename CounterT, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, CounterT, isDescend>::Init(
    GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex, GM_ADDR workspace, const SortRegBaseTilingData* __restrict tilingData,
    TPipe* pipe)
{
    this->blockIdx_ = GetBlockIdx();
    this->pipe_ = pipe;
    tilingData_ = tilingData;
    ParseTilingData();
    this->realCoreNum_ = GetBlockNum();

    this->inputXGm_.SetGlobalBuffer((__gm__ T1*)x);
    this->outValueGm_.SetGlobalBuffer((__gm__ T1*)value);
    this->outIdxGm_.SetGlobalBuffer((__gm__ uint32_t*)sortIndex);
    // 以下分配顺序须与 Host 的工作区预算一致；前两段按清零分工后的容量分配。
    // 计数偏移先以 uint32 载体元素计算，再转换为字节偏移，不能混用单位。
    uint64_t wkOffset = this->clearCoreSize0_ * this->clearCore0_;
    uint64_t oneBlockNumB32 = this->oneBlock_ / sizeof(int32_t);
    wkOffset = wkOffset * this->COUNTER_WORDS;
    wkOffset = Ops::Base::CeilAlign(wkOffset, oneBlockNumB32);
    this->exclusiveBinsGmWk_.SetGlobalBuffer((__gm__ uint32_t*)workspace, wkOffset);
    wkOffset = wkOffset * sizeof(uint32_t);

    uint64_t histOffset = this->clearCount_ * this->clearSize_ * this->clearCore1_;
    histOffset = histOffset * this->COUNTER_WORDS;
    histOffset = Ops::Base::CeilAlign(histOffset, oneBlockNumB32);
    this->globalHistGmWk_.SetGlobalBuffer((__gm__ uint32_t*)(workspace + wkOffset), histOffset);
    wkOffset = wkOffset + histOffset * sizeof(uint32_t);

    uint64_t dbOffset = this->totalDataNum_ * this->unsortedDimParallel_;
    dbOffset = dbOffset * this->COUNTER_WORDS;
    dbOffset = Ops::Base::CeilAlign(dbOffset, oneBlockNumB32);
    this->outIdxDbWK_.SetGlobalBuffer((__gm__ uint32_t*)(workspace + wkOffset), dbOffset);
    wkOffset = wkOffset + dbOffset * sizeof(uint32_t);

    // 块内桶计数和独占前缀只覆盖单个 tile，使用 uint16；跨块累积使用 CounterT。
    uint64_t histTileOffset = this->lastDimTileNum_ * RADIX_SORT_NUM * this->unsortedDimParallel_;
    this->histTileGmWk_.SetGlobalBuffer((__gm__ uint16_t*)(workspace + wkOffset), histTileOffset);
    wkOffset = wkOffset + histTileOffset * sizeof(uint16_t);
    this->tileBucketOffsetsGmWk_.SetGlobalBuffer((__gm__ uint16_t*)(workspace + wkOffset), histTileOffset);
    wkOffset = wkOffset + histTileOffset * sizeof(uint16_t);

    uint64_t xB8Offset = static_cast<uint64_t>(this->lastDimTileNum_) * this->numTileData_ * this->unsortedDimParallel_;
    xB8Offset = Ops::Base::CeilAlign(xB8Offset, this->oneBlock_);
    this->xB8GmWk_.SetGlobalBuffer((__gm__ uint8_t*)(workspace + wkOffset), xB8Offset);
    wkOffset = wkOffset + xB8Offset * sizeof(uint8_t);

    dbOffset = this->totalDataNum_ * this->unsortedDimParallel_;
    dbOffset = Ops::Base::CeilAlign(dbOffset * sizeof(T1), this->oneBlock_) / sizeof(T1);
    this->outValueDbWK_.SetGlobalBuffer((__gm__ T1*)(workspace + wkOffset), dbOffset);

    // 与 Host ComputeRadixTileData 的 tileFactor/ubExtra 逐项对应；tmpUb_ 单独计入。
    this->pipe_->InitBuffer(this->inQueueX_, 1, this->numTileData_ * sizeof(T1));
    this->pipe_->InitBuffer(this->inQueueIndex_, 1, this->numTileData_ * sizeof(CounterT));
    this->pipe_->InitBuffer(this->inQueueGlobalHist_, 1, RADIX_SORT_NUM * sizeof(CounterT));
    this->pipe_->InitBuffer(this->outValueQueue_, 1, this->numTileData_);
    this->pipe_->InitBuffer(this->blockExclusiveInQue_, 1, RADIX_SORT_NUM * sizeof(uint16_t));
    this->pipe_->InitBuffer(this->blockHistInQue_, 1, RADIX_SORT_NUM * sizeof(uint16_t));
    this->pipe_->InitBuffer(this->blockUbFlagQue_, 1, RADIX_SORT_NUM * sizeof(CounterT));
    this->pipe_->InitBuffer(this->inputB8Que_, 1, this->numTileData_);
    this->pipe_->InitBuffer(this->outIdxQueue_, 1, this->numTileData_ * sizeof(uint32_t));
    this->pipe_->InitBuffer(this->tmpUb_, this->tmpUbSize_);
    this->pipe_->InitBuffer(this->blockHistFlagUbQue_, 1, RADIX_SORT_NUM * sizeof(CounterT));

    this->globalHistGmWkTmp_ = this->globalHistGmWk_.template ReinterpretCast<CounterT>();
    this->exclusiveBinsGmWkTmp_ = this->exclusiveBinsGmWk_.template ReinterpretCast<CounterT>();
}

template <typename T1, typename T2, typename UT, typename CounterT, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, CounterT, isDescend>::ParseTilingData()
{
    this->totalDataNum_ = tilingData_->lastAxisNum;                // 每行排序轴长 N
    this->numTileData_ = tilingData_->numTileDataSize;             // 每个 UB 块的元素数
    this->unsortedDimNum_ = tilingData_->unsortedDimNum;           // 待排序行数 M
    this->unsortedDimParallel_ = tilingData_->unsortedDimParallel; // 同时驻留工作区的行数 P_act
    this->lastDimTileNum_ = tilingData_->lastDimTileNum;           // 每行分块数
    this->sortLoopTimes_ = tilingData_->sortLoopTimes;             // 行批次数
    this->lastDimRealCore_ = tilingData_->lastDimNeedCore;         // 每行协作核数
    this->tmpUbSize_ = tilingData_->tmpUbSize;                     // Sort 临时 UB 字节数

    // 后缀 0 对应 exclusiveBins，后缀 1 对应 globalHist；后缀不等于 keyParams 下标。
    this->clearCore1_ = tilingData_->keyParams0; // 用于清零的globalHistGmWk_的核
    this->clearCore0_ = tilingData_->keyParams1; // 用于清零exclusiveBinsGmWk_的核
    this->clearSize_ = tilingData_->keyParams2;  // 每次清零的 CounterT 元素数，按 globalHist 所需 UB 计算
    this->clearCount_ = tilingData_->keyParams3; // 清零globalHistGmWk_ ub循环次数
    this->clearCoreSize0_ = tilingData_->keyParams4; // 清零exclusiveBinsGmWk_,每个核处理多少个数
    this->clearCoreSize1_ = tilingData_->keyParams5; // 清零globalHistGmWk_，每个核处理多少
}

template <typename T1, typename T2, typename UT, typename CounterT, uint64_t isDescend>
template <typename OutputIndex, bool FirstPass>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, CounterT, isDescend>::ScatterOutput(
    LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal, LocalTensor<uint32_t> xInputIndexLocal,
    LocalTensor<uint8_t> sortedDigitLocal, LocalTensor<uint16_t> blockExclusiveSum,
    LocalTensor<CounterT> blockDataInGlobalPos, LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist,
    uint32_t round, int64_t tileDataStart, uint32_t currentTileSize)
{
    uint32_t unSortId = this->blockIdx_ / this->lastDimRealCore_;
    uint64_t outputXUnsortedAxisOffset = static_cast<uint64_t>(unSortId) * this->totalDataNum_;
    uint64_t unSortIdOffset = static_cast<uint64_t>(unSortId) * RADIX_SORT_NUM * sizeof(T1) + round * RADIX_SORT_NUM;
    auto scatterArgs = PrepareScatterArgs(this->tmpUb_, tileDataStart, outputXUnsortedAxisOffset, unSortIdOffset);
    asc_vf_call<CopyOutGm<T1, OutputIndex, CounterT, IndexT, FirstPass ? 0 : 1>>(
        dim3(THREAD_DIM_NUM), (__ubuf__ const uint64_t*)scatterArgs.GetPhyAddr(), currentTileSize,
        (__ubuf__ uint16_t*)(blockExclusiveSum.GetPhyAddr()), (__gm__ CounterT*)(this->exclusiveBinsGmWk_.GetPhyAddr()),
        (__ubuf__ CounterT*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
        (__ubuf__ IndexT*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(sortedDigitLocal.GetPhyAddr()),
        (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ CounterT*)(blockHistFlag.GetPhyAddr()),
        (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()), (__gm__ OutputIndex*)(this->idxDbGm_.Alternate().GetPhyAddr()),
        (__gm__ T1*)(this->inputXDbGm_.Alternate().GetPhyAddr()));
    PipeBarrier<PIPE_V>(); // 散射完成后才能复用 Sort 的 scratch。
}
template <typename T1, typename T2, typename UT, typename CounterT, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, CounterT, isDescend>::ScatterKeysGlobal(
    LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal, LocalTensor<uint32_t> xInputIndexLocal,
    LocalTensor<uint8_t> sortedDigitLocal, LocalTensor<uint16_t> blockExclusiveSum,
    LocalTensor<CounterT> blockDataInGlobalPos, LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist,
    uint32_t round, int64_t tileDataStart, uint32_t currentTileSize, uint32_t sortLoopRound)
{
    (void)sortLoopRound;
    // 中间轮次使用 IndexT 紧凑索引；末轮才转换成用户要求的 T2。
    // 8 位输入只有一轮，直接输出 T2；32 位有符号输出沿用 uint32 位载体。
    using OutputIndex = typename std::conditional<sizeof(T2) == sizeof(int32_t), uint32_t, T2>::type;
    if (round + 1 < sizeof(T1)) {
        if (round == 0) {
            ScatterOutput<IndexT, true>(xInputValueLocal, sortedIndexLocal, xInputIndexLocal, sortedDigitLocal,
                                        blockExclusiveSum, blockDataInGlobalPos, blockHistFlag, blockHist, round,
                                        tileDataStart, currentTileSize);
        } else {
            ScatterOutput<IndexT, false>(xInputValueLocal, sortedIndexLocal, xInputIndexLocal, sortedDigitLocal,
                                         blockExclusiveSum, blockDataInGlobalPos, blockHistFlag, blockHist, round,
                                         tileDataStart, currentTileSize);
        }
    } else {
        if (round == 0) {
            ScatterOutput<OutputIndex, true>(xInputValueLocal, sortedIndexLocal, xInputIndexLocal, sortedDigitLocal,
                                             blockExclusiveSum, blockDataInGlobalPos, blockHistFlag, blockHist, round,
                                             tileDataStart, currentTileSize);
        } else {
            ScatterOutput<OutputIndex, false>(xInputValueLocal, sortedIndexLocal, xInputIndexLocal, sortedDigitLocal,
                                              blockExclusiveSum, blockDataInGlobalPos, blockHistFlag, blockHist, round,
                                              tileDataStart, currentTileSize);
        }
    }
}
} // namespace Sort
#endif // RADIX_SORT_MORE_CORE_H
