/*
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/**
 * AdamApplyOneAssign Kernel — AdamKernel<T, RANK>
 */
#ifndef ADAM_APPLY_ONE_ASSIGN_KERNEL_H_
#define ADAM_APPLY_ONE_ASSIGN_KERNEL_H_
#include "kernel_operator.h"
#include "adam_apply_one_assign_tiling_struct.h"
#include "adam_apply_one_assign_struct.h"

// ============================================================
// Kernel 侧辅助函数 (int64_t * 版本, 无 std::vector)
// ============================================================

__aicore__ inline void GetCoreRange(int64_t coreId, int64_t mainTiles, int64_t mainCoreNum, int64_t& start,
                                    int64_t& end)
{
    if (coreId < mainCoreNum) {
        start = coreId * mainTiles;
        end = start + mainTiles;
    } else {
        start = mainCoreNum * mainTiles + (coreId - mainCoreNum) * (mainTiles - 1);
        end = start + mainTiles - 1;
    }
}

__aicore__ inline int64_t GetUBSplitRange(int64_t ubBlockIdx, int64_t ubOuter, int64_t ubFactor, int64_t ubTail)
{
    return (ubBlockIdx == ubOuter - 1) ? ubTail : ubFactor;
}

__aicore__ inline bool FlatToEffectiveCoord(int64_t flat, const int64_t* maxBroShape, int64_t rank, int64_t ubSplitIdx,
                                            int64_t ubFactor, int64_t ubOuter, int64_t* effCoord)
{
    for (int64_t d = 0; d < rank; d++) {
        effCoord[d] = 0;
    }
    int64_t ubBlockIdx = flat % ubOuter;
    int64_t outer = flat / ubOuter;
    for (int64_t d = ubSplitIdx - 1; d >= 0; d--) {
        effCoord[d] = outer % maxBroShape[d];
        outer /= maxBroShape[d];
    }
    effCoord[ubSplitIdx] = ubBlockIdx * ubFactor;
    return true;
}

// CalcOffset / CalcTransferCount: 输入与输出规则相同，共用一份实现
__aicore__ inline int64_t CalcOffset(const int64_t* effCoord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++) {
        offset += effCoord[d] * strides[d];
    }
    return offset; // 元素个数，gmIn_[]/gmOut_[] 的 index
}

__aicore__ inline int64_t CalcTransferCount(const int64_t* normalShape, int64_t rank, int64_t ubSplitIdx,
                                            int64_t ubBlockLength)
{
    int64_t splitElems = (normalShape[ubSplitIdx] == 1) ? 1 : ubBlockLength;
    int64_t innerElems = 1;
    for (int64_t d = ubSplitIdx + 1; d < rank; d++) {
        innerElems *= normalShape[d];
    }
    return splitElems * innerElems;
}

template <typename T>
__simd_vf__ inline void MulAddVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count,
                                 uint32_t oneRepeatSize, uint16_t repeatTimes);

template <typename T, int64_t RANK>
class AdamKernel {
    static constexpr int64_t MAX_RANK = 8; // 最大支持的 rank（坐标缓冲大小）
    static constexpr int64_t MAX_NDDMA_DIMS = 5;
    static constexpr int64_t ND = (RANK <= MAX_NDDMA_DIMS) ? RANK : MAX_NDDMA_DIMS;
    static constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(T);

    AscendC::TPipe pipe_;
    const AdamTilingData<RANK>* td_;
    AscendC::GlobalTensor<T> gmIn_[MAX_INPUT_SLOTS];
    AscendC::GlobalTensor<T> gmOut_[MAX_OUTPUT_SLOTS];
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[PHYS_NODES];
    AscendC::MultiCopyParams<T, ND> nddmaParams_[MAX_INPUT_SLOTS];
    int64_t nddmaOuterIters_[MAX_INPUT_SLOTS];
    int64_t nddmaDims_;

public:
    __aicore__ inline void Init(GM_ADDR inputs[MAX_INPUT_SLOTS], GM_ADDR outputs[MAX_OUTPUT_SLOTS],
                                const AdamTilingData<RANK>* td)
    {
        td_ = td;
        for (int32_t i = 0; i < MAX_INPUT_SLOTS; i++) {
            gmIn_[i].SetGlobalBuffer((__gm__ T*)inputs[i]);
        }
        for (int32_t i = 0; i < MAX_OUTPUT_SLOTS; i++) {
            gmOut_[i].SetGlobalBuffer((__gm__ T*)outputs[i]);
        }
        for (int32_t i = 0; i < PHYS_NODES; i++) {
            pipe_.InitBuffer(buf_[i], td_->perBufBytes);
        }

        const int64_t* dstShape = td_->maxBroShape;
        int64_t k = td_->split.ubSplitIdx;
        nddmaDims_ = (RANK - k <= ND) ? (RANK - k) : ND;
        for (int32_t inp = 0; inp < MAX_INPUT_SLOTS; inp++) {
            int64_t inner = 1;
            int64_t nd = 0;
            for (int64_t d = RANK - 1; d >= k && nd < ND; d--) {
                nddmaParams_[inp].loopInfo.loopSize[nd] = (d == k) ? 0 : dstShape[d];
                nddmaParams_[inp].loopInfo.loopSrcStride[nd] = td_->inputStrides[inp][d];
                nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
                nddmaParams_[inp].loopInfo.loopLpSize[nd] = 0;
                nddmaParams_[inp].loopInfo.loopRpSize[nd] = 0;
                inner *= (d == k) ? td_->split.ubFactor : dstShape[d];
                nd++;
            }
            for (; nd < ND; nd++) {
                nddmaParams_[inp].loopInfo.loopSize[nd] = 1;
                nddmaParams_[inp].loopInfo.loopSrcStride[nd] = 0;
                nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
                nddmaParams_[inp].loopInfo.loopLpSize[nd] = 0;
                nddmaParams_[inp].loopInfo.loopRpSize[nd] = 0;
            }
            // outer loop 只覆盖 flat 层和 NDDMA 之间的 gap: d = k .. RANK-nddmaDims-1
            nddmaOuterIters_[inp] = 1;
            for (int64_t d = k; d < RANK - nddmaDims_; d++) {
                nddmaOuterIters_[inp] *= (d == k) ? td_->split.ubFactor : dstShape[d];
            }
        }
    }

    __aicore__ inline void Process()
    {
        event_t evMTE2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        event_t evVtoMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        event_t evVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        event_t evMTE3toMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

        int64_t start = 0;
        int64_t end = 0;
        GetCoreRange(AscendC::GetBlockIdx(), td_->multicore.mainTiles, td_->multicore.mainCoreNum, start, end);

        // 输入槽位索引 (gmIn_[]) — 命名对齐 ops-nn def: input0~input4 / mul0_x~mul3_x / add2_y
        constexpr int32_t inInput0 = 0; // input0 (dataGrad) — square(S1) 与 mul1(S8) 的源
        constexpr int32_t inInput1 = 1; // input1 (dataV)    — mul2(S3) 操作数
        constexpr int32_t inInput2 = 2; // input2 (dataM)    — mul0(S7) 操作数
        constexpr int32_t inInput3 = 3; // input3 (dataVar)  — sub(S12) 操作数
        constexpr int32_t inInput4 = 4; // input4            — mul4(S11) 乘子
        constexpr int32_t inMul0X = 5;  // mul0_x (dataInputMul)  — mul0(S7) 乘子
        constexpr int32_t inMul1X = 6;  // mul1_x (dataInputMul1) — mul1(S8) 乘子
        constexpr int32_t inMul2X = 7;  // mul2_x (dataInputMul2) — mul2(S3) 乘子
        constexpr int32_t inMul3X = 8;  // mul3_x (dataInputMul3) — mul3(S2) 乘子
        constexpr int32_t inAdd2Y = 9;  // add2_y (dataInputAdd2) — add2(S6) 操作数

        // UB 槽位索引 (buf_[]) — 槽位多角色复用, 生命周期见设计文档 UB 生命周期表
        constexpr int32_t ubLoadData = 0; // data 类输入(grad/m/v/input4/var)装载槽, 复用为 sqrtResult
        constexpr int32_t ubLoadMulX = 1; // 乘子(mul0_x/mul1_x)装载槽, 复用为 add2Result/mul4Result
        constexpr int32_t ubMid = 2;      // 中间量槽: squareResult → truedivResult → output2
        constexpr int32_t ubOutM = 3;     // output1(m 路)累加槽: mul1Result 经 MulAddDst 就地累加
        constexpr int32_t ubOutV = 4;     // output0(v 路)累加槽: mul3Result 经 MulAddDst 就地累加

        // 输出槽位索引 (gmOut_[]) — 命名对齐 def: Output("out0"/"out1"/"out2")
        constexpr int32_t out0 = 0; // out0 (output0) — v 路更新: v·mul2_x + grad²·mul3_x
        constexpr int32_t out1 = 1; // out1 (output1) — m 路更新: m·mul0_x + grad·mul1_x
        constexpr int32_t out2 = 2; // out2 (output2) — var 更新: var − truediv·input4

        int64_t innerCount = 1;
        for (int64_t d = td_->split.ubSplitIdx + 1; d < RANK; d++) {
            innerCount *= td_->maxBroShape[d];
        }

        int64_t coord[MAX_RANK] = {};
        for (int64_t flat = start; flat < end; flat++) {
            int64_t ubBlockLength =
                GetUBSplitRange(flat % td_->split.ubOuter, td_->split.ubOuter, td_->split.ubFactor, td_->split.ubTail);
            int64_t count = ubBlockLength * innerCount;
            FlatToEffectiveCoord(flat, td_->maxBroShape, RANK, td_->split.ubSplitIdx, td_->split.ubFactor,
                                 td_->split.ubOuter, coord);

            if (flat != start) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
            }

            CopyInBrc(coord, inInput0, ubLoadData, ubBlockLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::Mul(buf_[ubMid].Get<T>(), buf_[ubLoadData].Get<T>(), buf_[ubLoadData].Get<T>(), count);

            CopyInBrc(coord, inMul1X, ubLoadMulX, ubBlockLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::Mul(buf_[ubOutM].Get<T>(), buf_[ubLoadData].Get<T>(), buf_[ubLoadMulX].Get<T>(), count);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);

            CopyInBrc(coord, inInput2, ubLoadData, ubBlockLength);
            CopyInBrc(coord, inMul0X, ubLoadMulX, ubBlockLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            uint16_t repeatTimesS3 = AscendC::CeilDivision(count, VL);
            asc_vf_call<MulAddVF<T>>((__ubuf__ T*)buf_[ubOutM].Get<T>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[ubLoadMulX].Get<T>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[ubLoadData].Get<T>().GetPhyAddr(), count, VL, repeatTimesS3);

            CopyInBrc(coord, inMul3X, ubOutV, ubBlockLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::Mul(buf_[ubOutV].Get<T>(), buf_[ubMid].Get<T>(), buf_[ubOutV].Get<T>(), count);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);

            CopyInBrc(coord, inInput1, ubLoadData, ubBlockLength);
            CopyInBrc(coord, inMul2X, ubMid, ubBlockLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            uint16_t repeatTimesS5 = AscendC::CeilDivision(count, VL);
            asc_vf_call<MulAddVF<T>>((__ubuf__ T*)buf_[ubOutV].Get<T>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[ubLoadData].Get<T>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[ubMid].Get<T>().GetPhyAddr(), count, VL, repeatTimesS5);

            AscendC::Sqrt(buf_[ubLoadData].Get<T>(), buf_[ubOutV].Get<T>(), count);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);

            CopyInBrc(coord, inAdd2Y, ubMid, ubBlockLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::Add(buf_[ubLoadMulX].Get<T>(), buf_[ubLoadData].Get<T>(), buf_[ubMid].Get<T>(), count);

            AscendC::Muls(buf_[ubOutM].Get<T>(), buf_[ubOutM].Get<T>(), T(1.0), count);
            AscendC::Div(buf_[ubMid].Get<T>(), buf_[ubOutM].Get<T>(), buf_[ubLoadMulX].Get<T>(), count);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);

            CopyInBrc(coord, inInput4, ubLoadData, ubBlockLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::Mul(buf_[ubLoadMulX].Get<T>(), buf_[ubMid].Get<T>(), buf_[ubLoadData].Get<T>(), count);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);

            CopyInBrc(coord, inInput3, ubLoadData, ubBlockLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::Sub(buf_[ubMid].Get<T>(), buf_[ubLoadData].Get<T>(), buf_[ubLoadMulX].Get<T>(), count);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);

            CopyOutOne(coord, out0, ubOutV, ubBlockLength);
            CopyOutOne(coord, out1, ubOutM, ubBlockLength);
            CopyOutOne(coord, out2, ubMid, ubBlockLength);

            if (flat != end - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
            }
        }
    }

private:
    __aicore__ inline void CopyInBrc(const int64_t* coord, int32_t inputIdx, int32_t slot, int64_t ubBlockLength)
    {
        int64_t k = td_->split.ubSplitIdx;
        int64_t off = CalcOffset(coord, td_->inputStrides[inputIdx], RANK);
        const int64_t* dstShape = td_->maxBroShape;

        auto params = nddmaParams_[inputIdx];
        int64_t kNd = RANK - 1 - k;
        int64_t inner = 1;
        for (int64_t nd = 0; nd < ND; nd++) {
            if (nd == kNd) {
                params.loopInfo.loopSize[nd] = ubBlockLength;
            }
            params.loopInfo.loopDstStride[nd] = inner;
            inner *= params.loopInfo.loopSize[nd];
        }

        static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad,
                                                     AscendC::NdDmaConfig::unsetPad, false};

        if constexpr (RANK <= MAX_NDDMA_DIMS) {
            AscendC::DataCopy<T, ND, cfg>(buf_[slot].Get<T>(), gmIn_[inputIdx][off], params);
        } else {
            AscendC::LocalTensor<T> buf = buf_[slot].Get<T>();
            int64_t elemBase = off;
            for (int64_t oi = 0; oi < nddmaOuterIters_[inputIdx]; oi++) {
                int64_t elemAdj = 0;
                int64_t tmp = oi;
                for (int64_t d = RANK - nddmaDims_ - 1; d >= k; d--) {
                    int64_t sz = (d == k) ? ubBlockLength : dstShape[d];
                    elemAdj += (tmp % sz) * td_->inputStrides[inputIdx][d];
                    tmp /= sz;
                }
                AscendC::DataCopy<T, ND, cfg>(buf[oi * inner], gmIn_[inputIdx][elemBase + elemAdj], params);
            }
        }
    }

    __aicore__ inline void CopyOutOne(const int64_t* coord, int32_t outputIdx, int32_t slot, int64_t ubBlockLength)
    {
        int64_t off = CalcOffset(coord, td_->outputStrides[outputIdx], RANK);
        int64_t cnt = CalcTransferCount(td_->outputShapes[outputIdx], RANK, td_->split.ubSplitIdx, ubBlockLength);
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = cnt * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        AscendC::DataCopyPad(gmOut_[outputIdx][off], buf_[slot].Get<T>(), extParams);
    }
};

// VF 函数: MulAddDst 包装
template <typename T>
__simd_vf__ inline void MulAddVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count,
                                 uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg0, srcReg1, dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr, aReg);
        AscendC::Reg::LoadAlign(srcReg1, src1Addr, aReg);
        AscendC::Reg::LoadAlign(dstReg, dstAddr, aReg);
        AscendC::Reg::MulAddDst(dstReg, srcReg0, srcReg1, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}
#endif // ADAM_APPLY_ONE_ASSIGN_KERNEL_H_
