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
 * \file sort_radix_more_core.h
 * \brief Flat multi-core radix sort class (SortRadixMoreCore).
 *        SIMD micro-operations are in sort_lib_vf.h; constants in sort_lib_constants.h.
 *
 * \internal  Do not include directly — use sort_lib.h instead.
 */

#ifndef SORT_RADIX_MORE_CORE_H
#define SORT_RADIX_MORE_CORE_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "sort_lib_util.h"
#include "sort_lib_constants.h"
#include "sort_lib_params.h"
#include "sort_lib_vf.h"

namespace SortLib::detail {

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
class SortRadixMoreCore {
public:
    __aicore__ inline SortRadixMoreCore(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex, GM_ADDR workspace,
                                const SortLib::SortParams& params, AscendC::TPipe* pipe);
    __aicore__ inline void Process();

private:
    using KeyT = typename UnsignedKeyType<ValT>::type;
    static constexpr AscendC::SortConfig sortConfigMulti{AscendC::SortType::RADIX_SORT, false};
    static constexpr uint64_t UB_ALIGN_BYTES = 32;

    // === GM input / output ===
    AscendC::GlobalTensor<ValT> inputXGm_;
    AscendC::GlobalTensor<ValT> outValueGm_;
    AscendC::GlobalTensor<uint32_t> outIdxGm_;

    // === GM workspace buffers ===
    AscendC::GlobalTensor<ValT> outValueDbWK_;
    AscendC::GlobalTensor<uint32_t> outIdxDbWK_;
    AscendC::GlobalTensor<uint8_t> xB8GmWk_;
    AscendC::GlobalTensor<uint16_t> histTileGmWk_;
    AscendC::GlobalTensor<uint16_t> histCumsumTileGmWk_;
    AscendC::GlobalTensor<CountT> globalHistGmWk_;
    AscendC::GlobalTensor<uint32_t> exclusiveBinsGmWkAsU32_;
    AscendC::GlobalTensor<CountT> exclusiveBinsGmWkAsCount_;

    // === double buffers ===
    DoubleBufferSimd<ValT> inputXDbGm_;
    DoubleBufferSimd<uint32_t> idxDbGm_;

    // === UB queues ===
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueIndex_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueGlobalHist_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> blockExclusiveInQue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> blockHistInQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpUb_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> blockUbFlagQue_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 1> inputB8Que_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outIdxQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outValueQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> blockHistFlagUbQue_;

    // === runtime state ===
    AscendC::TPipe* pipe_;
    uint32_t blockIdx_ = 0;
    uint32_t realCoreNum_ = 0;
    SortLib::SortParams params_;

    // === tiling parameters ===
    int64_t totalDataNum_ = 0;
    uint32_t numTileData_ = 0;
    uint32_t tileCount_ = 0;
    uint32_t activeCores_ = 0;
    uint32_t tmpUbSize_ = 0;

    // === internal methods ===
    __aicore__ inline void ParserTilingData();
    __aicore__ inline void ClearWorkSpace();
    __aicore__ inline void SetupDoubleBuffer();
    __aicore__ inline AscendC::LocalTensor<KeyT> PreProcess(AscendC::LocalTensor<ValT> inputX, uint32_t numTileData);
    __aicore__ inline void PreGlobalExclusiveSum(AscendC::LocalTensor<KeyT>& inputXCopy,
                                                 AscendC::LocalTensor<CountT>& blockExclusiveUb,
                                                 AscendC::LocalTensor<uint16_t>& histUb,
                                                 AscendC::LocalTensor<uint16_t>& histCumsumUb,
                                                 AscendC::LocalTensor<uint8_t>& inputB8Ub, uint32_t currTileSize,
                                                 uint32_t round);
    __aicore__ inline void GetGlobalExclusiveSum(uint32_t round, AscendC::GlobalTensor<ValT> inputX);
    __aicore__ inline void ScatterBlockHist2Global(AscendC::LocalTensor<uint16_t> blockHist,
                                                   AscendC::LocalTensor<CountT> blockHistWithFlag,
                                                   AscendC::GlobalTensor<CountT> allblockHistToGm, uint32_t tileId,
                                                   uint32_t round);
    __aicore__ inline void LookbackGlobal(AscendC::LocalTensor<CountT> nowTileHistBuffer,
                                          AscendC::GlobalTensor<CountT> allTileHistBuffer,
                                          AscendC::LocalTensor<uint32_t> ubFlagTensor, uint32_t tileId, uint32_t round);
    __aicore__ inline void SetPrefixReadyMask(AscendC::LocalTensor<CountT>& blockHistWithFlag,
                                              AscendC::GlobalTensor<CountT> blockHistToGm, uint32_t tileId,
                                              uint32_t round);
    __aicore__ inline void LoadTilePreComputeData(AscendC::LocalTensor<uint8_t> inputX8Ub,
                                                  AscendC::LocalTensor<uint16_t> blockExclusiveUb,
                                                  AscendC::LocalTensor<uint16_t> blockHistUb, uint32_t tileId,
                                                  uint32_t curSize);
    __aicore__ inline void CopyIndexDataIn(AscendC::GlobalTensor<uint32_t> inputIndex,
                                           AscendC::LocalTensor<uint32_t>& xLocal, uint64_t tileOffset,
                                           uint32_t currTileSize);
    __aicore__ inline void ComputeOnePass(uint32_t round, AscendC::GlobalTensor<ValT> inputXGm);
    __aicore__ inline void ProcessRadix(AscendC::GlobalTensor<ValT> inputXGm);
    __aicore__ inline void ScatterSortedTile(
        AscendC::LocalTensor<ValT> xInputValueLocal, AscendC::LocalTensor<uint32_t> sortedIndexLocal,
        AscendC::LocalTensor<uint32_t> xInputIndexLocal, AscendC::LocalTensor<uint8_t> sortedValueLocal,
        AscendC::LocalTensor<uint16_t> blockExclusiveSum, AscendC::LocalTensor<CountT> blockDataInGlobalPos,
        AscendC::LocalTensor<CountT> blockHistFlag, AscendC::LocalTensor<uint16_t> blockHist, uint32_t round,
        CountT tileDataStart, uint32_t cureTileSize);
    __aicore__ inline void ScatterPairUpcast(
        AscendC::LocalTensor<ValT> xInputValueLocal, AscendC::LocalTensor<uint32_t> sortedIndexLocal,
        AscendC::LocalTensor<uint32_t> xInputIndexLocal, AscendC::LocalTensor<uint8_t> sortedValueLocal,
        AscendC::LocalTensor<uint16_t> blockExclusiveSum, AscendC::LocalTensor<uint32_t> blockDataInGlobalPos,
        AscendC::LocalTensor<CountT> blockHistFlag, AscendC::LocalTensor<uint16_t> blockHist, uint32_t round,
        CountT tileDataStart, uint32_t cureTileSize);
    __aicore__ inline void ScatterPair(
        AscendC::LocalTensor<ValT> xInputValueLocal, AscendC::LocalTensor<uint32_t> sortedIndexLocal,
        AscendC::LocalTensor<uint32_t> xInputIndexLocal, AscendC::LocalTensor<uint8_t> sortedValueLocal,
        AscendC::LocalTensor<uint16_t> blockExclusiveSum, AscendC::LocalTensor<CountT> blockDataInGlobalPos,
        AscendC::LocalTensor<CountT> blockHistFlag, AscendC::LocalTensor<uint16_t> blockHist, uint32_t round,
        CountT tileDataStart, uint32_t cureTileSize);
};

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::ClearWorkSpace()

{
    FillZeros(exclusiveBinsGmWkAsCount_, static_cast<uint64_t>(RADIX_SORT_NUM) * sizeof(ValT), realCoreNum_);

    FillZeros(globalHistGmWk_, static_cast<uint64_t>(tileCount_) * RADIX_SORT_NUM * sizeof(ValT), realCoreNum_);
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline auto SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::PreProcess(
    AscendC::LocalTensor<ValT> inputX, uint32_t numTileData) -> AscendC::LocalTensor<KeyT>

{
    AscendC::LocalTensor<KeyT> inputXCopy = inputX.template ReinterpretCast<KeyT>();
    if constexpr (AscendC::IsSameType<int8_t, ValT>::value || AscendC::IsSameType<int16_t, ValT>::value ||
                  AscendC::IsSameType<int32_t, ValT>::value || AscendC::IsSameType<int64_t, ValT>::value) {
        TwiddleInSignedInt<ValT, KeyT, isDescend>(inputX, inputXCopy, numTileData);
    } else if constexpr (AscendC::IsSameType<half, ValT>::value || AscendC::IsSameType<bfloat16_t, ValT>::value) {
        TwiddleInFp<ValT, KeyT, isDescend, uint16_t>(inputX, inputXCopy, numTileData, LOWEST_KEY_VALUE_B16,
                                                     XOR_OP_VALUE_B16, TWIDDLED_MINUS_ZERO_BITS_FP16);
    } else if constexpr (AscendC::IsSameType<float, ValT>::value) {
        TwiddleInFp<ValT, KeyT, isDescend, uint32_t>(inputX, inputXCopy, numTileData, LOWEST_KEY_VALUE_B32,
                                                     XOR_OP_VALUE, TWIDDLED_MINUS_ZERO_BITS_FP32);
    } else {
        if (isDescend) {
            ReverseInputData<ValT, KeyT>(inputX, inputXCopy, numTileData);
        }
    }
    return inputXCopy;
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::PreGlobalExclusiveSum(

    AscendC::LocalTensor<KeyT>& inputXCopy, AscendC::LocalTensor<CountT>& blockExclusiveUb,
    AscendC::LocalTensor<uint16_t>& histUb, AscendC::LocalTensor<uint16_t>& histCumsumUb,
    AscendC::LocalTensor<uint8_t>& inputB8Ub, uint32_t currTileSize, uint32_t round)
{
    // Extract the current radix byte, build the tile histogram, and update this core's per-bin exclusive totals.
    if constexpr (sizeof(ValT) == sizeof(int32_t)) {
        GetGlobalExclusiveSumB32<KeyT, CountT>(inputXCopy, blockExclusiveUb, histUb, histCumsumUb, inputB8Ub,
                                               currTileSize, round);
    } else if constexpr (sizeof(ValT) == sizeof(int16_t)) {
        GetGlobalExclusiveSumB16<KeyT, CountT>(inputXCopy, blockExclusiveUb, histUb, histCumsumUb, inputB8Ub,
                                               currTileSize, round);
    } else if constexpr (sizeof(ValT) == sizeof(int8_t)) {
        GetGlobalExclusiveSumB8<KeyT, CountT>(inputXCopy, blockExclusiveUb, histUb, histCumsumUb, inputB8Ub,
                                              currTileSize, round);
    } else if constexpr (sizeof(ValT) == sizeof(int64_t)) {
        GetGlobalExclusiveSumB64<KeyT, CountT>(inputXCopy, blockExclusiveUb, histUb, histCumsumUb, inputB8Ub,
                                               currTileSize, round);
    }
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::GetGlobalExclusiveSum(
    uint32_t round, AscendC::GlobalTensor<ValT> inputX)

{
    if (blockIdx_ >= activeCores_) {
        return;
    }
    uint32_t startTileId = blockIdx_ % activeCores_;

    AscendC::LocalTensor<uint32_t> blockExclusiveUb = blockUbFlagQue_.template AllocTensor<uint32_t>();

    uint64_t exclusiveBinOffset = round * RADIX_SORT_NUM;
    if constexpr (sizeof(CountT) == sizeof(uint32_t)) {
        AscendC::Duplicate(blockExclusiveUb, static_cast<uint32_t>(0), RADIX_SORT_NUM);
    } else {
        AscendC::Duplicate(blockExclusiveUb, static_cast<uint32_t>(0), RADIX_SORT_NUM * 2);
    }
    for (uint32_t tileId = startTileId; tileId < tileCount_; tileId += activeCores_) {
        // tileOffset may exceed int32 range for large rows, so keep the address arithmetic in uint64.
        uint64_t tileOffset = static_cast<uint64_t>(tileId) * numTileData_;
        uint64_t remainTileDataNum = totalDataNum_ - tileOffset;
        if (totalDataNum_ < tileOffset) {
            break;
        }
        uint32_t currTileSize = static_cast<uint32_t>(
            remainTileDataNum < static_cast<uint64_t>(numTileData_) ? remainTileDataNum : numTileData_);
        AscendC::LocalTensor<ValT> xLocal = inQueueX_.template AllocTensor<ValT>();

        // Round 0 reads original input. Later rounds read the double-buffered output from the previous pass.
        if (round == 0) {
            CopyGmToUb(inputX, xLocal, tileOffset, currTileSize);
        } else {
            CopyGmToUb(inputXDbGm_.Current(), xLocal, tileOffset, currTileSize);
        }
        inQueueX_.EnQue(xLocal);
        xLocal = inQueueX_.template DeQue<ValT>();
        // Convert signed/floating values to unsigned radix keys before byte extraction.
        AscendC::LocalTensor<KeyT> xUbCopy = PreProcess(xLocal, currTileSize);
        AscendC::LocalTensor<uint8_t> inputB8Ub = inputB8Que_.template AllocTensor<uint8_t>();
        AscendC::LocalTensor<uint16_t> histUb = outIdxQueue_.template AllocTensor<uint16_t>();
        AscendC::LocalTensor<uint16_t> histCumsumUb = outValueQueue_.template AllocTensor<uint16_t>();
        AscendC::LocalTensor<CountT> blockExclusiveUbTmp = blockExclusiveUb.template ReinterpretCast<CountT>();
        PreGlobalExclusiveSum(xUbCopy, blockExclusiveUbTmp, histUb, histCumsumUb, inputB8Ub, currTileSize, round);
        inQueueX_.FreeTensor(xLocal);
        inputB8Que_.template EnQue<AscendC::QuePosition::VECCALC, AscendC::QuePosition::VECOUT>(inputB8Ub);
        inputB8Ub = inputB8Que_.template DeQue<AscendC::QuePosition::VECCALC, AscendC::QuePosition::VECOUT, uint8_t>();
        // Save the extracted radix byte for the scatter phase to avoid recomputing it.
        CopyUbToGm(xB8GmWk_, inputB8Ub, tileOffset, currTileSize);
        inputB8Que_.FreeTensor(inputB8Ub);

        outIdxQueue_.EnQue(histUb);
        histUb = outIdxQueue_.template DeQue<uint16_t>();
        // Save each tile's 256-bin histogram. Lookback uses these counts for inter-tile prefix offsets.
        CopyUbToGm(histTileGmWk_, histUb, static_cast<uint64_t>(tileId) * RADIX_SORT_NUM, RADIX_SORT_NUM);
        outIdxQueue_.FreeTensor(histUb);

        outValueQueue_.EnQue(histCumsumUb);
        histCumsumUb = outValueQueue_.template DeQue<uint16_t>();
        // Save each tile's intra-tile exclusive cumsum for the final scatter address calculation.
        CopyUbToGm(histCumsumTileGmWk_, histCumsumUb, static_cast<uint64_t>(tileId) * RADIX_SORT_NUM, RADIX_SORT_NUM);
        outValueQueue_.FreeTensor(histCumsumUb);
    }
    blockUbFlagQue_.EnQue(blockExclusiveUb);
    blockExclusiveUb = blockUbFlagQue_.template DeQue<uint32_t>();
    if constexpr (sizeof(CountT) == sizeof(uint32_t)) {
        AscendC::SetAtomicAdd<int32_t>();
        CopyUbToGm(exclusiveBinsGmWkAsU32_, blockExclusiveUb, exclusiveBinOffset, RADIX_SORT_NUM);
        AscendC::SetAtomicNone();
    } else {
        // int64 counters can exceed the int32 atomic path, so use the SIMT int64 atomic-add helper.
        asc_vf_call<SimtGlobalOffset<CountT>>(dim3(RADIX_SORT_NUM), exclusiveBinOffset,
                                              (__gm__ CountT*)(exclusiveBinsGmWkAsCount_.GetPhyAddr()),
                                              (__ubuf__ CountT*)(blockExclusiveUb.GetPhyAddr()));
    }
    blockUbFlagQue_.FreeTensor(blockExclusiveUb);
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::ScatterBlockHist2Global(

    AscendC::LocalTensor<uint16_t> blockHist, AscendC::LocalTensor<CountT> blockHistWithFlag,
    AscendC::GlobalTensor<CountT> allblockHistToGm, uint32_t tileId, uint32_t round)
{
    int64_t roundOffset = static_cast<int64_t>(round) * RADIX_SORT_NUM * tileCount_;
    __ubuf__ uint16_t* blockHistPtr = (__ubuf__ uint16_t*)blockHist.GetPhyAddr();
    __ubuf__ CountT* blockHistWithFlagPtr = (__ubuf__ CountT*)blockHistWithFlag.GetPhyAddr();
    asc_vf_call<EncodeHistVF<CountT>>((__ubuf__ uint16_t*)blockHistPtr, (__ubuf__ CountT*)blockHistWithFlagPtr);
    SyncEvent<AscendC::HardEvent::V_MTE3>();
    if (tileId < (tileCount_ - 1)) {
        CopyUbToGm(allblockHistToGm, blockHistWithFlag, (uint64_t)roundOffset + RADIX_SORT_NUM * tileId,
                   RADIX_SORT_NUM);
    }
    SyncEvent<AscendC::HardEvent::MTE3_V>();
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::LookbackGlobal(

    AscendC::LocalTensor<CountT> nowTileHistBuffer, AscendC::GlobalTensor<CountT> allTileHistBuffer,
    AscendC::LocalTensor<uint32_t> ubFlagTensor, uint32_t tileId, uint32_t round)
{
    int64_t roundOffset = static_cast<int64_t>(round) * RADIX_SORT_NUM * tileCount_;
    __ubuf__ CountT* nowTileHistBufferPtr = (__ubuf__ CountT*)nowTileHistBuffer.GetPhyAddr();

    uint16_t repeatTime;
    if constexpr (AscendC::IsSameType<CountT, uint32_t>::value) {
        repeatTime = RADIX_SORT_NUM / VF_LEN_B32;
    } else {
        repeatTime = RADIX_SORT_NUM / VF_LEN_B64;
    }

    // Look back over prior tiles. Aggregate-ready tiles contribute their local histograms and keep scanning.
    // Prefix-ready tiles already include all earlier tiles, so add them and stop.
    for (int i = tileId - 1; i >= 0; --i) {
        int mode = -1;
        uint32_t histTileOffset = RADIX_SORT_NUM * i;
        __ubuf__ uint32_t* ubFlagTensorPtr = (__ubuf__ uint32_t*)ubFlagTensor.GetPhyAddr();
        __ubuf__ CountT* tilePrevHistValuePtrCopy = nullptr;
        while (true) {
            // Poll until the previous tile publishes a full 256-bin aggregate-ready or prefix-ready state.
            AscendC::LocalTensor<CountT> xLocal = inQueueGlobalHist_.template AllocTensor<CountT>();
            CopyGmToUb(allTileHistBuffer, xLocal, (uint64_t)roundOffset + histTileOffset, RADIX_SORT_NUM);
            inQueueGlobalHist_.EnQue(xLocal);
            AscendC::LocalTensor<CountT> tilePrevHistValue = inQueueGlobalHist_.template DeQue<CountT>();
            __ubuf__ CountT* tilePrevHistValuePtr = (__ubuf__ CountT*)tilePrevHistValue.GetPhyAddr();
            tilePrevHistValuePtrCopy = tilePrevHistValuePtr;
            asc_vf_call<LookbackCheckStateVF<CountT>>((__ubuf__ CountT*)tilePrevHistValuePtr,
                                                      (__ubuf__ uint32_t*)ubFlagTensorPtr, repeatTime);
            SyncEvent<AscendC::HardEvent::V_S>();
            uint32_t notInitCountScalar = ubFlagTensorPtr[NOT_INIT_COUNT_INDEX];
            uint32_t aggReadyCountScalar = ubFlagTensorPtr[AGG_READY_COUNT_INDEX];
            uint32_t PrefixReadyCountScalar = ubFlagTensorPtr[PREFIX_READY_COUNT_INDEX];
            if (aggReadyCountScalar == RADIX_SORT_NUM) {
                mode = AGGREGATE_READY_FLAG;
                inQueueGlobalHist_.FreeTensor(tilePrevHistValue);
                break;
            }
            if (PrefixReadyCountScalar == RADIX_SORT_NUM) {
                mode = PREFIX_READY_FLAG;
                inQueueGlobalHist_.FreeTensor(tilePrevHistValue);
                break;
            }
            inQueueGlobalHist_.FreeTensor(tilePrevHistValue);
        }
        __ubuf__ CountT* nowTileHistBufferPtrCopy = nowTileHistBufferPtr;
        SyncEvent<AscendC::HardEvent::S_V>();
        asc_vf_call<LookbackAccumVF<CountT>>((__ubuf__ CountT*)nowTileHistBufferPtr,
                                             (__ubuf__ CountT*)nowTileHistBufferPtrCopy,
                                             (__ubuf__ CountT*)tilePrevHistValuePtrCopy, repeatTime);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventId);
        if (mode == PREFIX_READY_FLAG) {
            break;
        }
    }
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::SetPrefixReadyMask(

    AscendC::LocalTensor<CountT>& blockHistWithFlag, AscendC::GlobalTensor<CountT> blockHistToGm, uint32_t tileId,
    uint32_t round)
{
    int64_t roundOffset = static_cast<int64_t>(round) * RADIX_SORT_NUM * tileCount_;
    __ubuf__ CountT* histCumsumPtr = (__ubuf__ CountT*)blockHistWithFlag.GetPhyAddr();
    __ubuf__ CountT* histCumsumPtrCopy = histCumsumPtr;

    uint16_t repeatTime;
    if constexpr (AscendC::IsSameType<CountT, uint32_t>::value) {
        repeatTime = RADIX_SORT_NUM / VF_LEN_B32;
    } else {
        repeatTime = RADIX_SORT_NUM / VF_LEN_B64;
    }
    asc_vf_call<SetPrefixReadyVF<CountT>>((__ubuf__ CountT*)histCumsumPtr, (__ubuf__ CountT*)histCumsumPtrCopy,
                                          repeatTime);
    SyncEvent<AscendC::HardEvent::V_MTE3>();
    CopyUbToGm(blockHistToGm, blockHistWithFlag, (uint64_t)roundOffset + RADIX_SORT_NUM * tileId, RADIX_SORT_NUM);
    SyncEvent<AscendC::HardEvent::MTE3_V>();
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::LoadTilePreComputeData(

    AscendC::LocalTensor<uint8_t> inputX8Ub, AscendC::LocalTensor<uint16_t> blockExclusiveUb,
    AscendC::LocalTensor<uint16_t> blockHistUb, uint32_t tileId, uint32_t curSize)
{
    CopyGmToUb(histCumsumTileGmWk_, blockExclusiveUb, static_cast<uint64_t>(tileId) * RADIX_SORT_NUM, RADIX_SORT_NUM);
    CopyGmToUb(histTileGmWk_, blockHistUb, static_cast<uint64_t>(tileId) * RADIX_SORT_NUM, RADIX_SORT_NUM);
    CopyGmToUb(xB8GmWk_, inputX8Ub, static_cast<uint64_t>(tileId) * numTileData_, curSize);
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::CopyIndexDataIn(
    AscendC::GlobalTensor<uint32_t> inputIndex, AscendC::LocalTensor<uint32_t>& xLocal, uint64_t tileOffset,
    uint32_t currTileSize)
{
    CopyGmToUb(inputIndex, xLocal, tileOffset, currTileSize * sizeof(CountT) / sizeof(uint32_t));
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::ComputeOnePass(
    uint32_t round, AscendC::GlobalTensor<ValT> inputXGm)

{
    if (blockIdx_ >= activeCores_) {
        return;
    }
    uint32_t startId = blockIdx_ % activeCores_;

    for (uint32_t tileId = startId; tileId < tileCount_; tileId += activeCores_) {
        uint64_t tileOffset = static_cast<uint64_t>(tileId) * numTileData_;
        uint64_t tileDataStart = static_cast<uint64_t>(tileId) * numTileData_;
        uint64_t remainTileDataNum = totalDataNum_ - tileDataStart;
        if (totalDataNum_ < tileDataStart) {
            break;
        }
        uint32_t currTileSize = static_cast<uint32_t>(
            remainTileDataNum < static_cast<uint64_t>(numTileData_) ? remainTileDataNum : numTileData_);
        AscendC::LocalTensor<ValT> xLocal = inQueueX_.template AllocTensor<ValT>();
        if (round == 0) {
            CopyGmToUb(inputXGm, xLocal, tileOffset, currTileSize);
        } else {
            CopyGmToUb(inputXDbGm_.Current(), xLocal, tileOffset, currTileSize);
        }
        inQueueX_.EnQue(xLocal);
        xLocal = inQueueX_.template DeQue<ValT>();
        AscendC::LocalTensor<uint8_t> inputX8Ub = inputB8Que_.template AllocTensor<uint8_t>();
        AscendC::LocalTensor<uint16_t> blockExclusiveUb = blockExclusiveInQue_.template AllocTensor<uint16_t>();
        AscendC::LocalTensor<uint16_t> blockHistUb = blockHistInQue_.template AllocTensor<uint16_t>();
        LoadTilePreComputeData(inputX8Ub, blockExclusiveUb, blockHistUb, tileId, currTileSize);
        blockHistInQue_.EnQue(blockHistUb);
        blockExclusiveInQue_.EnQue(blockExclusiveUb);
        inputB8Que_.template EnQue<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECCALC>(inputX8Ub);
        inputX8Ub = inputB8Que_.template DeQue<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECCALC, uint8_t>();
        blockHistUb = blockHistInQue_.template DeQue<uint16_t>();

        AscendC::LocalTensor<CountT> blockHistFlagUb = blockHistFlagUbQue_.template AllocTensor<CountT>();
        ScatterBlockHist2Global(blockHistUb, blockHistFlagUb, globalHistGmWk_, tileId, round);

        AscendC::LocalTensor<uint8_t> shareTmpBuffer = tmpUb_.template Get<uint8_t>();
        AscendC::LocalTensor<uint32_t> sortedValueIndexLocal = outIdxQueue_.template AllocTensor<uint32_t>();
        AscendC::LocalTensor<uint8_t> sortedValueLocal = outValueQueue_.template AllocTensor<uint8_t>();
        AscendC::Sort<uint8_t, false, sortConfigMulti>(sortedValueLocal, sortedValueIndexLocal, inputX8Ub,
                                                       shareTmpBuffer, static_cast<uint32_t>(currTileSize));
        outValueQueue_.template EnQue<uint8_t>(sortedValueLocal);
        outIdxQueue_.template EnQue<uint32_t>(sortedValueIndexLocal);
        inputB8Que_.FreeTensor(inputX8Ub);
        AscendC::LocalTensor<uint32_t> ubFlagTensor = blockUbFlagQue_.template AllocTensor<uint32_t>();
        if (tileId > 0) {
            LookbackGlobal(blockHistFlagUb, globalHistGmWk_, ubFlagTensor, tileId, round);
        }
        blockUbFlagQue_.FreeTensor(ubFlagTensor);
        if (tileId < (tileCount_ - 1)) {
            SetPrefixReadyMask(blockHistFlagUb, globalHistGmWk_, tileId, round);
        }
        AscendC::LocalTensor<uint32_t> xIndexLocal;
        if (round != 0) {
            xIndexLocal = inQueueIndex_.template AllocTensor<uint32_t>();
            if constexpr (sizeof(CountT) > sizeof(IdxT)) {
                CopyIndexDataIn(idxDbGm_.Current(), xIndexLocal, tileOffset * sizeof(IdxT) / sizeof(uint32_t),
                                currTileSize);
            } else {
                CopyIndexDataIn(idxDbGm_.Current(), xIndexLocal, tileOffset * sizeof(CountT) / sizeof(uint32_t),
                                currTileSize);
            }
            inQueueIndex_.EnQue(xIndexLocal);
            xIndexLocal = inQueueIndex_.template DeQue<uint32_t>();
        }
        sortedValueIndexLocal = outIdxQueue_.template DeQue<uint32_t>();
        sortedValueLocal = outValueQueue_.template DeQue<uint8_t>();
        AscendC::LocalTensor<CountT> blockDataInGlobalPos = blockUbFlagQue_.template AllocTensor<CountT>();
        blockExclusiveUb = blockExclusiveInQue_.template DeQue<uint16_t>();
        ScatterSortedTile(xLocal, sortedValueIndexLocal, xIndexLocal, sortedValueLocal, blockExclusiveUb,
                          blockDataInGlobalPos, blockHistFlagUb, blockHistUb, round, tileDataStart, currTileSize);
        if (round != 0) {
            inQueueIndex_.FreeTensor(xIndexLocal);
        }
        blockHistFlagUbQue_.FreeTensor(blockHistFlagUb);
        inQueueX_.FreeTensor(xLocal);
        blockHistInQue_.FreeTensor(blockHistUb);
        blockUbFlagQue_.FreeTensor(blockDataInGlobalPos);
        blockExclusiveInQue_.FreeTensor(blockExclusiveUb);
        outIdxQueue_.FreeTensor(sortedValueIndexLocal);
        outValueQueue_.FreeTensor(sortedValueLocal);
    }
    idxDbGm_.selector_ ^= 1;
    inputXDbGm_.selector_ ^= 1;
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::SetupDoubleBuffer()
{
    if constexpr (sizeof(ValT) == sizeof(int8_t)) {
        inputXDbGm_.SetDoubleBuffer(outValueDbWK_, outValueGm_[0]);
        idxDbGm_.SetDoubleBuffer(outIdxDbWK_, outIdxGm_[0]);
    } else {
        inputXDbGm_.SetDoubleBuffer(outValueGm_[0], outValueDbWK_);
        idxDbGm_.SetDoubleBuffer(outIdxGm_[0], outIdxDbWK_);
    }
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::ProcessRadix(
    AscendC::GlobalTensor<ValT> inputXGm)

{
    AscendC::SyncAll();
    SetupDoubleBuffer();
    for (uint32_t round = 0; round < static_cast<uint32_t>(sizeof(ValT)); round++) {
        GetGlobalExclusiveSum(round, inputXGm);
        AscendC::SyncAll();
        ComputeOnePass(round, inputXGm);
        AscendC::SyncAll();
    }
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::Process()

{
    ProcessRadix(inputXGm_[0]);
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::Init(GM_ADDR x, GM_ADDR value,
                                                                              GM_ADDR sortIndex, GM_ADDR workspace,
                                                                              const SortLib::SortParams& params,
                                                                              AscendC::TPipe* pipe)
{
    blockIdx_ = AscendC::GetBlockIdx();
    pipe_ = pipe;
    params_ = params;
    ParserTilingData();
    realCoreNum_ = AscendC::GetBlockNum();

    inputXGm_.SetGlobalBuffer((__gm__ ValT*)x);
    outValueGm_.SetGlobalBuffer((__gm__ ValT*)value);
    outIdxGm_.SetGlobalBuffer((__gm__ uint32_t*)sortIndex);

    // ⚠️ Workspace 布局契约：本六段 [0]~[5] 的 offset 计算，
    //    必须与 op_host/arch35/sort_lib_tiling.h 中 SortTilingCompute 的六段总大小计算逐段一致。
    //    变量对应：RADIX_SORT_NUM==BIN_NUM；sizeof(CountT)==counterSize；
    //    UB_ALIGN_BYTES==blockUbSize；totalDataNum_==totalElements；
    //    [3] 段两次 wkOffset += tileHistBytes 对应 host 侧 [3] 段的 ×2。
    // [0] exclusiveBinsGmWk_ — 256 × radixRounds × sizeof(CountT) 字节, 对齐 UB_ALIGN_BYTES
    constexpr uint32_t radixRounds = sizeof(ValT); // radix sort passes
    uint64_t exclBinBytes = static_cast<uint64_t>(RADIX_SORT_NUM) * radixRounds * sizeof(CountT);
    exclBinBytes = RoundUpAlign(exclBinBytes, UB_ALIGN_BYTES);
    exclusiveBinsGmWkAsU32_.SetGlobalBuffer((__gm__ uint32_t*)workspace);
    uint64_t wkOffset = exclBinBytes;

    // [1] globalHistGmWk_ — radixRounds 轮 × 256 bin × tileCount × sizeof(CountT) 字节
    uint64_t globalHistBytes = static_cast<uint64_t>(tileCount_) * RADIX_SORT_NUM * radixRounds * sizeof(CountT);
    globalHistBytes = RoundUpAlign(globalHistBytes, UB_ALIGN_BYTES);
    globalHistGmWk_.SetGlobalBuffer((__gm__ CountT*)(workspace + wkOffset));
    wkOffset += globalHistBytes;

    // [2] outIdxDbWK_ — totalElements × sizeof(CountT) 字节
    uint64_t idxDbBytes = static_cast<uint64_t>(totalDataNum_) * sizeof(CountT);
    idxDbBytes = RoundUpAlign(idxDbBytes, UB_ALIGN_BYTES);
    outIdxDbWK_.SetGlobalBuffer((__gm__ uint32_t*)(workspace + wkOffset));
    wkOffset += idxDbBytes;

    // [3] histTileGmWk_ + histCumsumTileGmWk_ — 各 tileCount × 256 × sizeof(uint16_t)
    uint64_t tileHistBytes = static_cast<uint64_t>(tileCount_) * RADIX_SORT_NUM * sizeof(uint16_t);
    tileHistBytes = RoundUpAlign(tileHistBytes, UB_ALIGN_BYTES);
    histTileGmWk_.SetGlobalBuffer((__gm__ uint16_t*)(workspace + wkOffset));
    wkOffset += tileHistBytes;
    histCumsumTileGmWk_.SetGlobalBuffer((__gm__ uint16_t*)(workspace + wkOffset));
    wkOffset += tileHistBytes;

    // [4] xB8GmWk_ — tileCount × numTileData 字节, 对齐 UB_ALIGN_BYTES
    uint64_t xB8Bytes = static_cast<uint64_t>(tileCount_) * numTileData_;
    xB8Bytes = RoundUpAlign(xB8Bytes, UB_ALIGN_BYTES);
    xB8GmWk_.SetGlobalBuffer((__gm__ uint8_t*)(workspace + wkOffset));
    wkOffset += xB8Bytes;

    // [5] outValueDbWK_ — ValT 结尾段
    outValueDbWK_.SetGlobalBuffer((__gm__ ValT*)(workspace + wkOffset));

    pipe_->InitBuffer(inQueueX_, 1, numTileData_ * sizeof(ValT));
    pipe_->InitBuffer(inQueueIndex_, 1, numTileData_ * sizeof(CountT));
    pipe_->InitBuffer(inQueueGlobalHist_, 1, RADIX_SORT_NUM * sizeof(CountT));
    pipe_->InitBuffer(outValueQueue_, 1, numTileData_);
    pipe_->InitBuffer(blockExclusiveInQue_, 1, RADIX_SORT_NUM * sizeof(uint16_t));
    pipe_->InitBuffer(blockHistInQue_, 1, RADIX_SORT_NUM * sizeof(uint16_t));
    pipe_->InitBuffer(blockUbFlagQue_, 1, RADIX_SORT_NUM * sizeof(CountT));
    pipe_->InitBuffer(inputB8Que_, 1, numTileData_);
    pipe_->InitBuffer(outIdxQueue_, 1, numTileData_ * sizeof(uint32_t));
    pipe_->InitBuffer(tmpUb_, tmpUbSize_);
    pipe_->InitBuffer(blockHistFlagUbQue_, 1, RADIX_SORT_NUM * sizeof(CountT));

    exclusiveBinsGmWkAsCount_ = exclusiveBinsGmWkAsU32_.template ReinterpretCast<CountT>();

    ClearWorkSpace(); // MUST be before InitBuffer
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::ParserTilingData()
{
    totalDataNum_ = params_.totalElements; // total number of elements to sort
    numTileData_ = params_.numTileData;    // max elements per tile (UB capacity limited)
    tileCount_ = params_.tileCount;        // total tiles: ceil(totalElements / numTileData)
    activeCores_ = params_.activeCores;    // cores actually used: min(coreCount, tileCount)
    tmpUbSize_ = params_.tmpUbSize;        // UB size reserved for AscendC::Sort<uint8_t>
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::ScatterPairUpcast(
    AscendC::LocalTensor<ValT> xInputValueLocal, AscendC::LocalTensor<uint32_t> sortedIndexLocal,
    AscendC::LocalTensor<uint32_t> xInputIndexLocal, AscendC::LocalTensor<uint8_t> sortedValueLocal,
    AscendC::LocalTensor<uint16_t> blockExclusiveSum, AscendC::LocalTensor<uint32_t> blockDataInGlobalPos,
    AscendC::LocalTensor<CountT> blockHistFlag, AscendC::LocalTensor<uint16_t> blockHist, uint32_t round,
    CountT tileDataStart, uint32_t cureTileSize)
{
    uint64_t exclRoundOffset = round * RADIX_SORT_NUM;
    if (round == 0) {
        asc_vf_call<CopyOutGm<ValT, CountT, uint32_t, false>>(
            dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize, 0, exclRoundOffset,
            (__ubuf__ uint16_t*)(blockExclusiveSum.GetPhyAddr()),
            (__gm__ CountT*)(exclusiveBinsGmWkAsCount_.GetPhyAddr()),
            (__ubuf__ CountT*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
            (__ubuf__ uint32_t*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(sortedValueLocal.GetPhyAddr()),
            (__ubuf__ ValT*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ CountT*)(blockHistFlag.GetPhyAddr()),
            (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()), (__gm__ uint32_t*)(idxDbGm_.Alternate().GetPhyAddr()),
            (__gm__ ValT*)(inputXDbGm_.Alternate().GetPhyAddr()));
    } else if (round < static_cast<uint32_t>(sizeof(ValT) - 1)) {
        asc_vf_call<CopyOutGm<ValT, CountT, uint32_t, true>>(
            dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize, 0, exclRoundOffset,
            (__ubuf__ uint16_t*)(blockExclusiveSum.GetPhyAddr()),
            (__gm__ CountT*)(exclusiveBinsGmWkAsCount_.GetPhyAddr()),
            (__ubuf__ CountT*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
            (__ubuf__ uint32_t*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(sortedValueLocal.GetPhyAddr()),
            (__ubuf__ ValT*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ CountT*)(blockHistFlag.GetPhyAddr()),
            (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()), (__gm__ uint32_t*)(idxDbGm_.Alternate().GetPhyAddr()),
            (__gm__ ValT*)(inputXDbGm_.Alternate().GetPhyAddr()));
    } else {
        AscendC::GlobalTensor<IdxT> outIdxT2 = (idxDbGm_.Alternate()).template ReinterpretCast<IdxT>();
        asc_vf_call<CopyOutGm<ValT, CountT, IdxT, true>>(
            dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize, 0, exclRoundOffset,
            (__ubuf__ uint16_t*)(blockExclusiveSum.GetPhyAddr()),
            (__gm__ CountT*)(exclusiveBinsGmWkAsCount_.GetPhyAddr()),
            (__ubuf__ CountT*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
            (__ubuf__ uint32_t*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(sortedValueLocal.GetPhyAddr()),
            (__ubuf__ ValT*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ CountT*)(blockHistFlag.GetPhyAddr()),
            (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()), (__gm__ IdxT*)(outIdxT2.GetPhyAddr()),
            (__gm__ ValT*)(inputXDbGm_.Alternate().GetPhyAddr()));
    }
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::ScatterPair(
    AscendC::LocalTensor<ValT> xInputValueLocal, AscendC::LocalTensor<uint32_t> sortedIndexLocal,
    AscendC::LocalTensor<uint32_t> xInputIndexLocal, AscendC::LocalTensor<uint8_t> sortedValueLocal,
    AscendC::LocalTensor<uint16_t> blockExclusiveSum, AscendC::LocalTensor<CountT> blockDataInGlobalPos,
    AscendC::LocalTensor<CountT> blockHistFlag, AscendC::LocalTensor<uint16_t> blockHist, uint32_t round,
    CountT tileDataStart, uint32_t cureTileSize)
{
    uint64_t exclRoundOffset = round * RADIX_SORT_NUM;

    if (round == 0) {
        asc_vf_call<CopyOutGm<ValT, CountT, IdxT, false>>(
            dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize, 0, exclRoundOffset,
            (__ubuf__ uint16_t*)(blockExclusiveSum.GetPhyAddr()),
            (__gm__ CountT*)(exclusiveBinsGmWkAsCount_.GetPhyAddr()),
            (__ubuf__ CountT*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
            (__ubuf__ CountT*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(sortedValueLocal.GetPhyAddr()),
            (__ubuf__ ValT*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ CountT*)(blockHistFlag.GetPhyAddr()),
            (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()), (__gm__ IdxT*)(idxDbGm_.Alternate().GetPhyAddr()),
            (__gm__ ValT*)(inputXDbGm_.Alternate().GetPhyAddr()));
    } else {
        asc_vf_call<CopyOutGm<ValT, CountT, IdxT, true>>(
            dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize, 0, exclRoundOffset,
            (__ubuf__ uint16_t*)(blockExclusiveSum.GetPhyAddr()),
            (__gm__ CountT*)(exclusiveBinsGmWkAsCount_.GetPhyAddr()),
            (__ubuf__ CountT*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
            (__ubuf__ CountT*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(sortedValueLocal.GetPhyAddr()),
            (__ubuf__ ValT*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ CountT*)(blockHistFlag.GetPhyAddr()),
            (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()), (__gm__ IdxT*)(idxDbGm_.Alternate().GetPhyAddr()),
            (__gm__ ValT*)(inputXDbGm_.Alternate().GetPhyAddr()));
    }
}

template <typename ValT, typename IdxT, typename CountT, bool isDescend>
__aicore__ inline void SortRadixMoreCore<ValT, IdxT, CountT, isDescend>::ScatterSortedTile(
    AscendC::LocalTensor<ValT> xInputValueLocal, AscendC::LocalTensor<uint32_t> sortedIndexLocal,
    AscendC::LocalTensor<uint32_t> xInputIndexLocal, AscendC::LocalTensor<uint8_t> sortedValueLocal,
    AscendC::LocalTensor<uint16_t> blockExclusiveSum, AscendC::LocalTensor<CountT> blockDataInGlobalPos,
    AscendC::LocalTensor<CountT> blockHistFlag, AscendC::LocalTensor<uint16_t> blockHist, uint32_t round,
    CountT tileDataStart, uint32_t cureTileSize)
{
    if constexpr (sizeof(ValT) == sizeof(int8_t)) {
        // int8 has only one radix pass, always scatter as output dtype

        uint64_t exclRoundOffset = round * RADIX_SORT_NUM;
        AscendC::GlobalTensor<IdxT> outIdxT2 = (idxDbGm_.Alternate()).template ReinterpretCast<IdxT>();
        asc_vf_call<CopyOutGm<ValT, CountT, IdxT, false>>(
            dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize, 0, exclRoundOffset,
            (__ubuf__ uint16_t*)(blockExclusiveSum.GetPhyAddr()),
            (__gm__ CountT*)(exclusiveBinsGmWkAsCount_.GetPhyAddr()),
            (__ubuf__ CountT*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
            (__ubuf__ CountT*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(sortedValueLocal.GetPhyAddr()),
            (__ubuf__ ValT*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ CountT*)(blockHistFlag.GetPhyAddr()),
            (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()), (__gm__ IdxT*)(outIdxT2.GetPhyAddr()),
            (__gm__ ValT*)(inputXDbGm_.Alternate().GetPhyAddr()));
    } else if constexpr (AscendC::IsSameType<CountT, uint32_t>::value && sizeof(IdxT) == sizeof(int64_t)) {
        ScatterPairUpcast(xInputValueLocal, sortedIndexLocal, xInputIndexLocal, sortedValueLocal, blockExclusiveSum,
                          blockDataInGlobalPos, blockHistFlag, blockHist, round, tileDataStart, cureTileSize);
    } else {
        ScatterPair(xInputValueLocal, sortedIndexLocal, xInputIndexLocal, sortedValueLocal, blockExclusiveSum,
                    blockDataInGlobalPos, blockHistFlag, blockHist, round, tileDataStart, cureTileSize);
    }
}

} // namespace SortLib::detail

#endif // SORT_RADIX_MORE_CORE_H
