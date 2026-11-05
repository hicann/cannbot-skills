/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SMALL_AXIS_TWO_STAGE_BASE_H
#define SMALL_AXIS_TWO_STAGE_BASE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "signed_zero_sort_utils.h"
#include "util_type_simd.h"

namespace SmallAxisCommon {
using namespace AscendC;

constexpr uint32_t TWO_STAGE_THREAD_NUM = 1024;
using SignedZeroSortCommon::FloatBitsT;
using SignedZeroSortCommon::IS_FLOATING_POINT_V;

// 通过逆排名恢复行内顺序。以两行三列升序为例：
// 输入 [[3,1,5],[4,2,6]]，全批排序后 stage1Order=[1,4,0,3,2,5]。
// 逆映射 rankInverse=[2,0,4,3,1,5]，再统计同一行中排名更小的元素数。
// 结果 values=[1,3,5,2,4,6]，indices=[1,0,2,1,0,2]。
template <typename T, typename OutIdxT>
__simt_vf__ LAUNCH_BOUND(TWO_STAGE_THREAD_NUM) __aicore__
    void RankInverseScatter(uint32_t totalElems, uint32_t segmentLen, __ubuf__ T* stage1ValuePtr,
                            __ubuf__ uint32_t* stage1OrderPtr, __ubuf__ uint32_t* signFlagPtr,
                            __ubuf__ uint32_t* rankInversePtr, __ubuf__ T* finalValuePtr, __ubuf__ OutIdxT* finalIdxPtr)
{
    // 建立逆映射 rankInverse[stage1Order[rank]] = rank。
    // 例如 stage1Order=[1,4,0,3,2,5] 对应 rankInverse=[2,0,4,3,1,5]。
    for (uint32_t rank = static_cast<uint32_t>(threadIdx.x); rank < totalElems; rank += TWO_STAGE_THREAD_NUM) {
        uint32_t flatIdx = stage1OrderPtr[rank];
        if constexpr (IS_FLOATING_POINT_V<T>) {
            using BitsT = FloatBitsT<T>;
            __ubuf__ BitsT* signFlagBits = reinterpret_cast<__ubuf__ BitsT*>(signFlagPtr);
            if (signFlagBits[flatIdx] != static_cast<BitsT>(0U)) {
                constexpr BitsT signBit = static_cast<BitsT>(BitsT{1} << (sizeof(T) * 8U - 1U));
                __ubuf__ BitsT* stage1ValueBits = reinterpret_cast<__ubuf__ BitsT*>(stage1ValuePtr);
                stage1ValueBits[rank] = signBit;
            }
        }
        rankInversePtr[flatIdx] = rank;
    }
    asc_syncthreads();

    // 统计同一行中排名更小的元素数，得到最终行内位置。
    for (uint32_t flat = static_cast<uint32_t>(threadIdx.x); flat < totalElems; flat += TWO_STAGE_THREAD_NUM) {
        uint32_t flatIdx = flat;
        uint32_t segId = flatIdx / segmentLen;
        uint32_t laneId = flatIdx - segId * segmentLen;

        uint32_t rank = rankInversePtr[flatIdx];
        uint32_t segBase = segId * segmentLen;
        // 计数循环展开八路，用独立累加器减少循环开销并隐藏访存延迟。
        uint32_t pos0 = 0;
        uint32_t pos1 = 0;
        uint32_t pos2 = 0;
        uint32_t pos3 = 0;
        uint32_t pos4 = 0;
        uint32_t pos5 = 0;
        uint32_t pos6 = 0;
        uint32_t pos7 = 0;
        uint32_t lane = 0;
        for (; lane + 7U < segmentLen; lane += 8U) {
            uint32_t scanBase = segBase + lane;
            pos0 += (rankInversePtr[scanBase] < rank) ? 1U : 0U;
            pos1 += (rankInversePtr[scanBase + 1U] < rank) ? 1U : 0U;
            pos2 += (rankInversePtr[scanBase + 2U] < rank) ? 1U : 0U;
            pos3 += (rankInversePtr[scanBase + 3U] < rank) ? 1U : 0U;
            pos4 += (rankInversePtr[scanBase + 4U] < rank) ? 1U : 0U;
            pos5 += (rankInversePtr[scanBase + 5U] < rank) ? 1U : 0U;
            pos6 += (rankInversePtr[scanBase + 6U] < rank) ? 1U : 0U;
            pos7 += (rankInversePtr[scanBase + 7U] < rank) ? 1U : 0U;
        }
        uint32_t pos = pos0 + pos1 + pos2 + pos3 + pos4 + pos5 + pos6 + pos7;
        for (; lane < segmentLen; ++lane) {
            pos += (rankInversePtr[segBase + lane] < rank) ? 1U : 0U;
        }

        uint32_t outOffset = segBase + pos;
        finalValuePtr[outOffset] = stage1ValuePtr[rank];
        finalIdxPtr[outOffset] = static_cast<OutIdxT>(laneId);
    }
}

// 构造第二阶段紧凑键 key = segId × totalElems + rank。
// 两行三列示例：stage1Order=[1,4,0,3,2,5] → keys=[0,7,2,9,4,11]，
// 同时将 stage1Order 改写为行内索引 [1,1,0,0,2,2]。
// 打包第二阶段键，同时将展平索引改写为行内索引。
inline __simt_vf__ LAUNCH_BOUND(TWO_STAGE_THREAD_NUM) __aicore__
    void BuildStage2KeysSimt(uint32_t totalElems, uint32_t segmentLen, __ubuf__ uint32_t* stage1OrderPtr,
                             __ubuf__ uint16_t* stage2KeyPtr)
{
    for (uint32_t rank = static_cast<uint32_t>(threadIdx.x); rank < totalElems; rank += TWO_STAGE_THREAD_NUM) {
        uint32_t flatIdx = stage1OrderPtr[rank];
        uint32_t segId = flatIdx / segmentLen;
        uint32_t laneId = flatIdx - segId * segmentLen;
        stage2KeyPtr[rank] = static_cast<uint16_t>(segId * totalElems + rank);
        stage1OrderPtr[rank] = laneId;
    }
}

// 两阶段排序公共流水负责分批、UB 分配和连续行加载；派生类提供批次映射与写回。
// T 为值类型，FinalIdxT 为最终 UB 索引类型，IsDescend 控制升降序。
template <typename Derived, typename T, typename FinalIdxT, bool IsDescend>
class SmallAxisTwoStageBase {
public:
    __aicore__ inline void Process()
    {
        Derived* op = static_cast<Derived*>(this);
        if (op->IsProcessInvalid()) {
            return;
        }
        for (uint32_t batchId = blockIdx_; batchId < batchNum_; batchId += blockDim_) {
            uint32_t validSegs = op->ComputeValidSegs(batchId);
            if (validSegs == 0U) {
                continue;
            }
            op->ProcessBatch(batchId, validSegs);
        }
    }

    __aicore__ inline void InitSortBuffers(TPipe* pipe, uint32_t maxFlatElems, uint32_t tmpUbSize, bool useRankInverse,
                                           uint32_t finalIdxElemBytes)
    {
        pipe_ = pipe;
        maxFlatElems_ = maxFlatElems;
        useRankInverse_ = useRankInverse;

        pipe_->InitBuffer(inputBuf_, RoundUpAlign(maxFlatElems_ * sizeof(T)));
        pipe_->InitBuffer(stage1ValueBuf_, RoundUpAlign(maxFlatElems_ * sizeof(T)));
        pipe_->InitBuffer(stage1OrderBuf_, RoundUpAlign(maxFlatElems_ * sizeof(uint32_t)));
        pipe_->InitBuffer(finalIdxBuf_, RoundUpAlign(maxFlatElems_ * finalIdxElemBytes));
        pipe_->InitBuffer(rankInverseBuf_, RoundUpAlign(maxFlatElems_ * sizeof(uint32_t)));
        if (!useRankInverse_) {
            pipe_->InitBuffer(stage2OrderBuf_, RoundUpAlign(maxFlatElems_ * sizeof(uint32_t)));
        }
        pipe_->InitBuffer(tmpBuf_, tmpUbSize);

        inputValues_ = inputBuf_.template Get<T>();
        stage1Values_ = stage1ValueBuf_.template Get<T>();
        stage1Order_ = stage1OrderBuf_.template Get<uint32_t>();
        // Stage1Sort 后不再读取本批 inputValues_，因此 finalValues_ 可复用该缓冲。
        finalValues_ = inputValues_;
        finalIdx_ = finalIdxBuf_.template Get<FinalIdxT>();
        // finalIdx_ 在逆排名散射前尚未生效，可用其 B32 或更宽存储暂存源元素的负零标记。
        stage1SignFlags_ = finalIdxBuf_.template Get<uint32_t>();
        if (useRankInverse_) {
            rankInverse_ = rankInverseBuf_.template Get<uint32_t>();
        } else {
            // 最终索引生成前，finalIdxBuf_ 先复用为 uint16 的 stage2KeysIn_。
            // 当前派生类传 sizeof(OutIdxT)（4/8 字节），Host 精确预算使用相同索引宽度。
            stage2KeysIn_ = finalIdxBuf_.template Get<uint16_t>();
            // stage2KeysOut_ 暂存 uint16 排序键；键不再使用后，BuildOutputs 将其底层
            // rankInverseBuf_ 复用为 uint32 Gather 字节偏移。
            stage2KeysOut_ = rankInverseBuf_.template Get<uint16_t>();
            stage2Order_ = stage2OrderBuf_.template Get<uint32_t>();
        }
        tmp_ = tmpBuf_.template Get<uint8_t>();
    }

    __aicore__ inline void LoadContiguousBatch(GlobalTensor<T>& inputGm, int64_t inputStart, uint32_t totalElems)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParam{1, static_cast<uint32_t>(totalElems * sizeof(T)), 0, 0, 0};
        DataCopyPad(inputValues_, inputGm[inputStart], copyParam, padParams);
        event_t eventIdMte2ToV = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }

    __aicore__ inline void RunTwoStageSort(uint32_t totalElems)
    {
        Stage1Sort(totalElems);
        if (useRankInverse_) {
            ScatterOutputs(totalElems);
        } else {
            BuildStage2Keys(totalElems);
            Stage2Sort(totalElems);
            BuildOutputs(totalElems);
        }
    }

protected:
    TPipe* pipe_ = nullptr;

    TBuf<TPosition::VECCALC> inputBuf_;
    TBuf<TPosition::VECCALC> stage1ValueBuf_;
    TBuf<TPosition::VECCALC> stage1OrderBuf_;
    TBuf<TPosition::VECCALC> finalIdxBuf_;
    TBuf<TPosition::VECCALC> rankInverseBuf_;
    TBuf<TPosition::VECCALC> stage2OrderBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;

    LocalTensor<T> inputValues_;
    LocalTensor<T> stage1Values_;
    LocalTensor<uint32_t> stage1Order_;
    LocalTensor<uint32_t> rankInverse_;
    LocalTensor<uint16_t> stage2KeysIn_;
    LocalTensor<uint16_t> stage2KeysOut_;
    LocalTensor<uint32_t> stage2Order_;
    LocalTensor<T> finalValues_;
    LocalTensor<FinalIdxT> finalIdx_;
    LocalTensor<uint32_t> stage1SignFlags_;
    LocalTensor<uint8_t> tmp_;

    uint32_t blockIdx_ = 0;
    uint32_t blockDim_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t batchNum_ = 0;
    uint32_t segmentLen_ = 0;
    uint32_t maxFlatElems_ = 0;
    bool useRankInverse_ = false;

private:
    // 先将本批全部行展平排序，得到值及原始展平索引；再恢复各行内部顺序。
    // 普通分支排序 uint16 紧凑键（segId × totalElems + rank）；
    // 逆排名分支直接计算同一行中更小排名的数量，省去第二次排序。
    static constexpr SortConfig kStage1SortConfig{SortType::RADIX_SORT, IsDescend};
    static constexpr SortConfig kStage2SortConfig{SortType::RADIX_SORT, false};

    __aicore__ inline void Stage1Sort(uint32_t totalElems)
    {
        if constexpr (IS_FLOATING_POINT_V<T>) {
            if (useRankInverse_) {
                SignedZeroSortCommon::PrepareSignedZeroKeysAndFlagsVec(inputValues_, stage1SignFlags_, totalElems);
                AscendC::Sort<T, false, kStage1SortConfig>(stage1Values_, stage1Order_, inputValues_, tmp_, totalElems);
                return;
            }
        }
        AscendC::Sort<T, false, kStage1SortConfig>(stage1Values_, stage1Order_, inputValues_, tmp_, totalElems);
    }

    __aicore__ inline void ScatterOutputs(uint32_t totalElems)
    {
        asc_vf_call<RankInverseScatter<T, FinalIdxT>>(
            dim3(TWO_STAGE_THREAD_NUM), totalElems, segmentLen_, (__ubuf__ T*)stage1Values_.GetPhyAddr(),
            (__ubuf__ uint32_t*)stage1Order_.GetPhyAddr(), (__ubuf__ uint32_t*)stage1SignFlags_.GetPhyAddr(),
            (__ubuf__ uint32_t*)rankInverse_.GetPhyAddr(), (__ubuf__ T*)finalValues_.GetPhyAddr(),
            (__ubuf__ FinalIdxT*)finalIdx_.GetPhyAddr());
    }

    __aicore__ inline void BuildStage2Keys(uint32_t totalElems)
    {
        asc_vf_call<BuildStage2KeysSimt>(dim3(TWO_STAGE_THREAD_NUM), totalElems, segmentLen_,
                                         (__ubuf__ uint32_t*)stage1Order_.GetPhyAddr(),
                                         (__ubuf__ uint16_t*)stage2KeysIn_.GetPhyAddr());
    }

    __aicore__ inline void Stage2Sort(uint32_t totalElems)
    {
        AscendC::Sort<uint16_t, false, kStage2SortConfig>(stage2KeysOut_, stage2Order_, stage2KeysIn_, tmp_,
                                                          totalElems);
    }

    __aicore__ inline void BuildOutputs(uint32_t totalElems)
    {
        // 两行三列示例：第二阶段将 [0,7,2,9,4,11] 排成 [0,2,4,7,9,11]，
        // 得到 stage2Order=[0,2,4,1,3,5]，据此收集值和行内索引。
        // 结果 values=[1,3,5,2,4,6]，indices=[1,0,2,1,0,2]。
        // 下面 Muls 会将 rankInverseBuf_ 覆盖为 uint32 字节偏移，随后供 Gather 读取。
        LocalTensor<uint32_t> gatherOffsets = stage2KeysOut_.template ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> gatherOffsetsInt = gatherOffsets.template ReinterpretCast<int32_t>();

        Muls(gatherOffsetsInt, stage2Order_.template ReinterpretCast<int32_t>(), static_cast<int32_t>(sizeof(T)),
             totalElems);
        Gather(finalValues_, stage1Values_, gatherOffsets, 0, totalElems);

        Muls(gatherOffsetsInt, stage2Order_.template ReinterpretCast<int32_t>(), static_cast<int32_t>(sizeof(uint32_t)),
             totalElems);
        if constexpr (IsSameType<FinalIdxT, int64_t>::value) {
            LocalTensor<int32_t> gatheredIdxInt32 = stage2Order_.template ReinterpretCast<int32_t>();
            Gather(gatheredIdxInt32, stage1Order_.template ReinterpretCast<int32_t>(), gatherOffsets, 0, totalElems);
            Cast(finalIdx_, gatheredIdxInt32, RoundMode::CAST_NONE, Ops::Base::CeilAlign(totalElems, 4u));
        } else {
            Gather(finalIdx_.template ReinterpretCast<int32_t>(), stage1Order_.template ReinterpretCast<int32_t>(),
                   gatherOffsets, 0, totalElems);
        }
    }
};

} // namespace SmallAxisCommon

#endif
