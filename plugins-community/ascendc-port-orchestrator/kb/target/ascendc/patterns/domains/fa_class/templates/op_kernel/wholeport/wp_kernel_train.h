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
 * \file flash_attention_score_kernel_train.h  (whole-port: wp_kernel_train.h)
 * \brief arch35 FlashAttentionScoreKernelTrain<Cube,Vec> orchestrator body-copied
 *        VERBATIM into the whole-port (the 4-deep AIC/AIV Process pipeline:
 *        IterateBmm1->ProcessVec1->IterateBmm2->ProcessVec2 + 3 drain loops;
 *        GetAttentionOffset native layout offsets; GetS2LoopRange sparse geometry).
 *        RED LINE: NO #include "arch35/", NO aclnn/aclop. include-adapt: base +
 *        dropmask -> wp_*. Guard renamed.
 */
#ifndef WP_FLASH_ATTENTION_SCORE_KERNEL_TRAIN_H_
#define WP_FLASH_ATTENTION_SCORE_KERNEL_TRAIN_H_
#include "wp_kernel_base.h"
#include "wp_dropmask.h"
namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreKernelTrain
    : public FlashAttentionNoQuantKernelBase<FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>, CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    using BaseClass = FlashAttentionNoQuantKernelBase<FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>, CubeBlockType, VecBlockType>;
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr<isInfer> &runParam,
        RunInfo<isInfer> &runInfo);
    __aicore__ inline void Process();
private:
    __aicore__ inline void GetAttentionOffset(RunParamStr<isInfer> &runParam);
    __aicore__ inline void CalS1OuterSize(const int64_t &multiCoreInnerOffset, RunParamStr<isInfer> &runParam);
    __aicore__ inline void GetS2LoopRange(RunParamStr<isInfer> &runParam);
    __aicore__ inline int64_t CalcRealTimes(int64_t relativePos, int64_t length);
    __aicore__ inline int64_t CalcRealCoreIdx(int64_t relativePos, int64_t times, int64_t offsetCoreIdx,
        bool isPartialCalc);
    __aicore__ inline int64_t CalcRealCoreIdxVarlen(int64_t calcLoops, int64_t calcLoopsRemain,
                                                    int64_t cycleCoreNums);
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::InitUniqueConstInfo()
{
    this->constInfo.gS1o = this->constInfo.gSize * this->constInfo.s1OuterSize;
    this->constInfo.n2GS1o = this->constInfo.n2Size * this->constInfo.gS1o;
    this->constInfo.scaleValue = this->sharedParams.scaleValue;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr<isInfer> &runParam, RunInfo<isInfer> &runInfo)
{
    // 训练的TND场景的Mask以及Pse都是不带padding的
    runInfo.b1SSOffset = runParam.b1SSOffset;
    // 训练的非TND场景的mask的sequence length一定等于qk的sequence length
    runInfo.b1SSAttenMaskOffset = runParam.b1SSOffset;
    if constexpr (hasDrop) {
        runInfo.b1SSOffsetAlign = runParam.b1SSOffsetAlign16;
    }

    // 训练的preTokens和NextTokens已经被转换好
    if constexpr (hasAtten) {
        if ASCEND_IS_AIV {
            runInfo.preTokensPerBatch = this->attenMaskInfo.preTokens;
            runInfo.nextTokensPerBatch = this->attenMaskInfo.nextTokens;
        }
    }

    runInfo.vecCoreOffset = this->constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::CalcRealCoreIdx(
    int64_t relativePos, int64_t times, int64_t offsetCoreIdx, bool isPartialCalc)
{
    int64_t n1oIdx = 0;
    int64_t s1oIdx = 0;
    int64_t realCoreIdx = 0;
    int64_t partialCalcLength = this->sharedParams.firstFullLoadS1OuterIdx + 1;

    // 新分核方式中，将顺序-部分计算，倒序-部分计算，全量计算 三种计算区域进行独立计算
    // coreIdx 表示第 times 次计算的S1方向基本块在对应计算区域中的位置
    int64_t coreIdx = times * this->sharedParams.coreNum + relativePos;
    if (isPartialCalc && this->sharedParams.firstFullLoadS1OuterIdx != -1) {
        // 非sparse场景，firstFullLoadS1OuterIdx才为-1
        n1oIdx = coreIdx / partialCalcLength;
        s1oIdx = coreIdx % partialCalcLength;
    } else {
        n1oIdx = coreIdx / (this->constInfo.s1OuterSize - partialCalcLength);
        s1oIdx = coreIdx % (this->constInfo.s1OuterSize - partialCalcLength);
    }

    // realCoreIdx 将对应计算区域中S1方向基本块的索引，转换为实际的索引值
    realCoreIdx = n1oIdx * this->constInfo.s1OuterSize + s1oIdx + offsetCoreIdx;

    return realCoreIdx;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::CalcRealCoreIdxVarlen(
    int64_t calcLoops, int64_t calcLoopsRemain, int64_t cycleCoreNums)
{
    int64_t realCoreIdx = 0;
    if (calcLoopsRemain == 0) {
        realCoreIdx = calcLoops * cycleCoreNums + this->aicIdx;
    } else {
        realCoreIdx = calcLoops * cycleCoreNums + (cycleCoreNums - this->aicIdx - 1);
    }

    return realCoreIdx;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::CalcRealTimes(
    int64_t relativePos, int64_t length)
{
    int64_t realTimes = length / this->sharedParams.coreNum;
    int64_t remainCycles = length % this->sharedParams.coreNum;

    return (relativePos < remainCycles) ? realTimes + 1 : realTimes;
}

/*
 * 新分核方式（提高L2上数据的复用）：
 * 一、顺序分核：将每个S1的基本块，依次分发给各个核计算；
 *      s1基本块1 -- core1
 *      s1基本块2 -- core2
 *      s1基本块3 -- core3
 *      s1基本块4 -- core1
 *      ...
 * 二、对称分核：为了让Sparse场景更好的负载均衡，上半部分N采用顺序分核，下半部分N与上半部分N对称分核
 *      <上半部分N>
 *      s1基本块1 -- core1
 *      s1基本块2 -- core2
 *      s1基本块3 -- core3
 *      <下半部分N>
 *      s1基本块4 -- core3
 *      s1基本块5 -- core2
 *      s1基本块6 -- core1
 *
 * 三、正倒序循环分核：将每个S1的基本块，先正序依次分发给各个核计算，再倒序依次分发给各个核计算，以此循环
 *      s1基本块1 -- core1
 *      s1基本块2 -- core2
 *      s1基本块3 -- core3
 *      s1基本块4 -- core3
 *      s1基本块5 -- core2
 *      s1基本块6 -- core1
 *      s1基本块7 -- core1
 *      s1基本块8 -- core2
 *      ...
 */
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::Process()
{
    int64_t multiCoreInnerOffset = 0;
    int64_t multiCoreInnerLimit = 0;

    // 新分核模式
    // 1、非TND场景 S2全量计算的部分，采用顺序分核；
    // 2、非TND场景 S2部分计算的部分，采用对称分核：将N分成一半，上半部分顺序分核，下半部分与上半部分对称分核；
    // 3、TND场景 正倒序循环分核
    int64_t halfN = 0;
    int64_t partialCalcForwardNum = 0;              // 当前核 在顺序部分计算中分配的S1方向上基本块个数；
    int64_t partialCalcReverseNum = 0;              // 当前核 在倒序部分计算中分配的S1方向上基本块个数；
    int64_t partialCalcNum = 0;                     // 当前核 在部分计算中分配的S1方向上基本块个数；
    int64_t fullCalcForwardNum = 0;                 // 当前核 在全量计算中分配的S1方向上基本块个数；
    int64_t halfNCoreIdx = 0;                       // 下半部分第一个S1方向基本块对应的核索引；
    int64_t partialCalcLength = this->sharedParams.firstFullLoadS1OuterIdx + 1;     // 部分计算在单个S1上的长度；
    int64_t relativePosReverse = 0;                 // 当前核 与第一个S1方向基本块对应的核索引 相差的个数
    int64_t varlenCalcLoops = 0;                    // TND场景 需要进行计算的循环次数(正序+倒序为一次循环)
    int64_t varlenCalcLoopsRemain = 0;
    int64_t varlenCalcTimes = 0;                    // TND场景 需要计算的S1方向上基本块总数
    int64_t varlenCycleCoreNums = this->sharedParams.coreNum * 2; // TND场景 一次循环正序+倒序为两倍核数

    if (this->sharedParams.splitCoreMode == 1) {
        if (layout == LayOutTypeEnum::LAYOUT_TND) {
            varlenCalcLoops = this->sharedParams.totalSize / varlenCycleCoreNums;
            varlenCalcLoopsRemain = this->sharedParams.totalSize % varlenCycleCoreNums;
            varlenCalcTimes = varlenCalcLoops * 2;
            if (varlenCalcLoopsRemain >= this->aicIdx + 1) {
                varlenCalcTimes++;
                if (varlenCalcLoopsRemain > this->sharedParams.coreNum &&
                    (this->aicIdx + 1) > varlenCycleCoreNums - varlenCalcLoopsRemain) {
                    varlenCalcTimes++;
                }
            }
            multiCoreInnerOffset = 0;
            multiCoreInnerLimit = varlenCalcTimes;
        } else {
            const int64_t totalN = this->constInfo.n2G * this->sharedParams.bSize;
            halfN = CeilDiv(totalN, 2);
            int64_t partialCalcForwardLength = halfN * partialCalcLength;
            int64_t partialCalcReverseLength = (totalN - halfN) * partialCalcLength;
            int64_t fullCalcForwardLength = 0;
            if (this->sharedParams.firstFullLoadS1OuterIdx == -1) {
                fullCalcForwardLength = totalN * this->constInfo.s1OuterSize;
            } else {
                fullCalcForwardLength = totalN * (this->constInfo.s1OuterSize - partialCalcLength);
            }
            halfNCoreIdx = (partialCalcForwardLength - 1) % this->sharedParams.coreNum;
            relativePosReverse = (halfNCoreIdx - this->aicIdx + this->sharedParams.coreNum) % this->sharedParams.coreNum;
            partialCalcForwardNum = CalcRealTimes(this->aicIdx, partialCalcForwardLength);
            partialCalcReverseNum = CalcRealTimes(relativePosReverse, partialCalcReverseLength);
            fullCalcForwardNum = CalcRealTimes(this->aicIdx, fullCalcForwardLength);
            partialCalcNum = partialCalcForwardNum + partialCalcReverseNum;
            // 表示当前核需要计算的次数
            multiCoreInnerOffset = 0;
            multiCoreInnerLimit = partialCalcForwardNum + partialCalcReverseNum + fullCalcForwardNum;
        }
    } else {
        // 表示核内切分起点
        multiCoreInnerOffset = this->sharedParams.multiCoreInnerOffset;
        multiCoreInnerLimit = this->sharedParams.multiCoreInnerLimit;
    }

    if (unlikely(this->aicIdx >= this->sharedParams.coreNum)) {
        return;
    }

    // 初始化AxisIdx
    RunParamStr<isInfer> runParam;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        if (this->sharedParams.splitCoreMode == 1) {
            CalS1OuterSize(this->aicIdx, runParam);
        } else {
            CalS1OuterSize(multiCoreInnerOffset, runParam);
        }
    }
    RunInfo<isInfer> runInfo[4];
    int64_t taskId = 0;
    bool notThirdLast = true;
    bool notSecondLast = true;
    bool notLast = true;
    int64_t thirdLast = multiCoreInnerLimit;
    int64_t secondLast = multiCoreInnerLimit + 1;
    int64_t last = multiCoreInnerLimit + 2;
    multiCoreInnerLimit += 3;
    int64_t realCoreInnerIdx = 0;
    int64_t realMultiCoreInnerIdxInS2 = -1;
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
         multiCoreInnerIdx++) {
        if (multiCoreInnerIdx == secondLast) {
            notSecondLast = false;
        } else if (multiCoreInnerIdx == last) {
            notLast = false;
        } else if (multiCoreInnerIdx == thirdLast) {
            notThirdLast = false;
        } else {
            // 非最后三次伪循环，需要将当前核处理的次数转化为S1方向上基本块的索引值
            if (this->sharedParams.splitCoreMode == 1) {
                if (layout == LayOutTypeEnum::LAYOUT_TND) {
                    int64_t curCalcLoops = multiCoreInnerIdx >> 1;
                    int64_t curCalcLoopsRemain = multiCoreInnerIdx & 1;
                    realCoreInnerIdx = CalcRealCoreIdxVarlen(curCalcLoops, curCalcLoopsRemain, varlenCycleCoreNums);
                } else {
                    if (multiCoreInnerIdx >= 0 && multiCoreInnerIdx < partialCalcForwardNum) {
                        realCoreInnerIdx = CalcRealCoreIdx(this->aicIdx, multiCoreInnerIdx, 0, true);
                    } else if (multiCoreInnerIdx >= partialCalcForwardNum && multiCoreInnerIdx < partialCalcNum) {
                        realCoreInnerIdx = CalcRealCoreIdx(relativePosReverse, multiCoreInnerIdx - partialCalcForwardNum,
                                                        halfN * this->constInfo.s1OuterSize, true);
                    } else {
                        realCoreInnerIdx = CalcRealCoreIdx(this->aicIdx, multiCoreInnerIdx - partialCalcNum,
                                                        partialCalcLength, false);
                    }
                }
            } else {
                realCoreInnerIdx = multiCoreInnerIdx;
            }
        }

        int64_t s2LoopLimit = 0;
        bool notLastThreeLoop = notThirdLast && notSecondLast && notLast;
        bool notLastTwoLoop = notSecondLast && notLast;
        if (notLastThreeLoop) {
            this->ComputeAxisIdx(realCoreInnerIdx, runParam);
            this->GetAttentionOffset(runParam);
            // s2轴循环计数, 支持sparse和非sparse场景
            this->GetS2LoopRange(runParam);
            s2LoopLimit = CeilDiv(runParam.s2LineEndIdx - runParam.s2LineStartIdx, this->s2BaseSize) - 1;
        }
        if ASCEND_IS_AIV {
            if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION ||
                        IsSameType<INPUT_T, float>::value) {
                if (this->sharedParams.implMode ==
                    static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                    this->constInfo.softMaxCheckRes = true;
                }
            }
        }
        if (s2LoopLimit != -1) {
            realMultiCoreInnerIdxInS2++;
        }
        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
            if (notLastThreeLoop) {
                RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];
                this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, realCoreInnerIdx);
                if (this->sharedParams.splitCoreMode == 1) {
                    runInfo1.multiCoreIdxMod3 = realMultiCoreInnerIdxInS2 % 3;
                }
                if ASCEND_IS_AIC {
                    this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo1, this->constInfo);
                }
            }
            if (taskId > 0 && notLastTwoLoop) {
                if ASCEND_IS_AIV {
                    auto &runInfo3 = runInfo[(taskId + 3) & 3];
                    this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo3,
                        this->constInfo);
                }
            }
            if (taskId > 1 && notLast) {
                RunInfo<isInfer> &runInfo2 = runInfo[(taskId + 2) & 3];
                if ASCEND_IS_AIC {
                    if constexpr (BaseClass::bmm2Write2Ub) {
                        this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo2,
                            this->constInfo);
                    } else {
                        this->cubeBlock.IterateBmm2(this->bmm2ResGmBuffers.Get(), this->l1PBuffers, runInfo2,
                            this->constInfo);
                    }
                }
            }
            if (taskId > 2) {
                if ASCEND_IS_AIV {
                    RunInfo<isInfer> &runInfo3 = runInfo[(taskId + 1) & 3];
                    if constexpr (BaseClass::bmm2Write2Ub) {
                        this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo3, this->constInfo);
                    } else {
                        this->vecBlock.ProcessVec2(this->bmm2ResGmBuffers.Get(), runInfo3, this->constInfo);
                    }
                }
            }
            taskId++;
        }
    }
}

// =========================================== private functions ===========================================
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::GetAttentionOffset(
    RunParamStr<isInfer> &runParam)
{
    if ASCEND_IS_AIV {
        int64_t bOffsetOut = 0;
        int64_t s1OffsetOut = 0;
        int64_t n2OffsetOut = 0;
        int64_t gOffsetOut = 0;
        int64_t subBlockS1Offset = 0;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            // (BS)ND
            bOffsetOut = this->s1SizeAcc * this->constInfo.n2GDv;
            s1OffsetOut = runParam.s1oIdx * this->constInfo.s1BaseN2GDv;
            n2OffsetOut = runParam.n2oIdx * this->constInfo.gDv;
            gOffsetOut = runParam.goIdx * this->constInfo.dSizeV;
            subBlockS1Offset = this->constInfo.subBlockIdx * runParam.firstHalfS1RealSize * this->constInfo.n2GDv;
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            // BSH/BSNGD
            bOffsetOut = runParam.boIdx * this->constInfo.n2GS1Dv;
            s1OffsetOut = runParam.s1oIdx * this->constInfo.s1BaseN2GDv;
            n2OffsetOut = runParam.n2oIdx * this->constInfo.gDv;
            gOffsetOut = runParam.goIdx * this->constInfo.dSizeV;
            subBlockS1Offset = this->constInfo.subBlockIdx * runParam.firstHalfS1RealSize * this->constInfo.n2GDv;
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_SBH) {
            // SBH/SBNGD
            s1OffsetOut = runParam.s1oIdx * this->constInfo.s1BaseBN2GDv;
            bOffsetOut = runParam.boIdx * this->constInfo.n2GDv;
            n2OffsetOut = runParam.n2oIdx * this->constInfo.gDv;
            gOffsetOut = runParam.goIdx * this->constInfo.dSizeV;
            subBlockS1Offset = this->constInfo.subBlockIdx * runParam.firstHalfS1RealSize * this->constInfo.bN2GDv;
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            // bnsd
            bOffsetOut = runParam.boIdx * this->constInfo.n2GS1Dv;
            n2OffsetOut = runParam.n2oIdx * this->constInfo.gS1Dv;
            gOffsetOut = runParam.goIdx * this->constInfo.s1Dv;
            s1OffsetOut = runParam.s1oIdx * this->constInfo.s1BaseDv;
            subBlockS1Offset = this->constInfo.subBlockIdx * runParam.firstHalfS1RealSize * this->constInfo.dSizeV;
        }
        runParam.attentionOutOffset = bOffsetOut + n2OffsetOut + gOffsetOut + s1OffsetOut + subBlockS1Offset;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::CalS1OuterSize(
    const int64_t &multiCoreInnerOffset, RunParamStr<isInfer> &runParam)
{
    int64_t actualS1Outersize = 0;
    runParam.boIdx = 0;
    this->s1OuterSizeAcc = 0;
    runParam.b1SSOffset = 0;
    runParam.b1SSOffsetAlign16 = 0;
    this->s1SizeAcc = 0;
    this->s2SizeAcc = 0;

    int64_t actualS1Len;
    int64_t actualS2Len;
    for (int64_t i = 0; i < this->sharedParams.bSize; ++i) {
        this->GetSeqQlenKvlenByBoidx(i, actualS1Len, actualS2Len);
        actualS1Outersize += (CeilDiv(actualS1Len, this->s1BaseSize) * this->constInfo.n2G);
        if (multiCoreInnerOffset >= actualS1Outersize) {
            this->s1OuterSizeAcc = actualS1Outersize;
            this->s1SizeAcc += actualS1Len;
            this->s2SizeAcc += actualS2Len;
            runParam.b1SSOffset += actualS1Len * actualS2Len;
            runParam.b1SSOffsetAlign16 += actualS1Len * Align(actualS2Len);
            runParam.boIdx++;
            continue;
        }
        break;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::GetS2LoopRange(
    RunParamStr<isInfer> &runParam)
{
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        int64_t actualS1Len;
        int64_t actualS2Len;
        this->GetSeqQlenKvlenByBoidx(runParam.boIdx, actualS1Len, actualS2Len);
        if constexpr (hasAtten) {
            if (this->sharedParams.sparseType == static_cast<uint8_t>(SparseModeEnum::CAUSAL)) {
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize, actualS2Len);
            } else if (this->sharedParams.sparseType == static_cast<uint8_t>(SparseModeEnum::RIGHT_DOWN_CAUSAL)) {
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx =
                    Min((runParam.s1oIdx + 1) * this->s1BaseSize + actualS2Len - actualS1Len, actualS2Len);
            } else if (this->sharedParams.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::BAND)) {
                runParam.s2LineStartIdx = Max(
                    runParam.s1oIdx * this->s1BaseSize - this->sharedParams.s1SparseValidSize, 0);
                runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize +
                                                this->sharedParams.s2SparseValidSize,
                                            actualS2Len);
                // s1baseSize行都无效时，需要将startIdx设置为0，,endIdx设置为S2realSize
                if (runParam.s2LineEndIdx - runParam.s2LineStartIdx <= 0) {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx = actualS2Len;
                }
            } else if (this->sharedParams.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::BAND_COMPRESS)) {
                runParam.s2LineStartIdx =
                    Max(runParam.s1oIdx * this->s1BaseSize - actualS1Len +
                            Max(actualS2Len - this->sharedParams.preTokens, 0),
                        0);
                runParam.s2LineEndIdx =
                    Min((runParam.s1oIdx + 1) * this->s1BaseSize + actualS2Len -
                            Max(actualS1Len - this->sharedParams.nextTokens, 0),
                        actualS2Len);
                // s1baseSize行都无效时，需要将startIdx设置为0，,endIdx设置为S2realSize
                if (runParam.s2LineEndIdx - runParam.s2LineStartIdx <= 0) {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx = actualS2Len;
                }
            } else if (this->sharedParams.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::RIGHT_DOWN_CAUSAL_BAND)) {
                if (runParam.boIdx == this->sharedParams.bandIndex) {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize + actualS2Len +
                                                    this->sharedParams.nextTokens - actualS1Len,
                                                actualS2Len);
                } else {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx =
                        Min((runParam.s1oIdx + 1) * this->s1BaseSize + actualS2Len - actualS1Len, actualS2Len);
                }
            } else if (this->sharedParams.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::BAND_LEFT_UP_CAUSAL)) {
                if (runParam.boIdx == this->sharedParams.bandIndex) {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx =
                        Min((runParam.s1oIdx + 1) * this->s1BaseSize + actualS2Len -
                                Max(actualS1Len - this->sharedParams.nextTokens, 0),
                            actualS2Len);
                } else {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize, actualS2Len);
                }
            } else if (this->sharedParams.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx = Max((runParam.s1oIdx + 1) * this->s1BaseSize - actualS1Len + actualS2Len,
                                            ((__gm__ int64_t *)(this->attenMaskInfo.prefixNAddr))[runParam.boIdx]);
                runParam.s2LineEndIdx = Min(runParam.s2LineEndIdx, actualS2Len);
                if (runParam.s2LineEndIdx - runParam.s2LineStartIdx <= 0) {
                    runParam.s2LineEndIdx = actualS2Len;
                }
            } else {
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx = actualS2Len;
            }
        } else {
            runParam.s2LineStartIdx = 0;
            runParam.s2LineEndIdx = actualS2Len;
        }

        if constexpr (hasDrop) {
            // 外部传入的drop_mask未进行bit转bool, 需要进行对齐
            runParam.s2LineStartIdx = runParam.s2LineStartIdx >> 4 << 4;
            runParam.s2LineEndIdx = Min(CeilDiv(runParam.s2LineEndIdx, 16) << 4, actualS2Len);
        }
    } else {
        if constexpr (hasAtten) {
            // 计算S2的循环范围相关参数: 后续可 使用static_cast<uint32_t>优化scale性能
            if (this->sharedParams.sparseType ==
                static_cast<uint8_t>(SparseModeEnum::CAUSAL)) { // 下三角
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize, this->constInfo.s2Size);
            } else if (this->sharedParams.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::BAND)) {
                // 对角线往外扩散场景, s1和s2可能不同
                runParam.s2LineStartIdx = Max(
                    runParam.s1oIdx * this->s1BaseSize - this->sharedParams.s1SparseValidSize, 0);
                runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize +
                                                this->sharedParams.s2SparseValidSize,
                                            this->constInfo.s2Size);
            } else if (this->sharedParams.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx =
                    Max(this->s1BaseSize * (runParam.s1oIdx + 1) - this->constInfo.s1Size + this->constInfo.s2Size,
                        ((__gm__ int64_t *)(this->attenMaskInfo.prefixNAddr))[runParam.boIdx]);
                if (runParam.s2LineEndIdx <= runParam.s2LineStartIdx) {
                    // 无效行场景至少要算一个基本块
                    runParam.s2LineEndIdx = this->s2BaseSize;
                } else {
                    runParam.s2LineEndIdx = CeilDiv(runParam.s2LineEndIdx, this->s2BaseSize) * this->s2BaseSize;
                }
                runParam.s2LineEndIdx = Min(runParam.s2LineEndIdx, this->constInfo.s2Size);
            } else { // 其它场景, 如无attention mask
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx = this->constInfo.s2Size;
            }
        } else {
            runParam.s2LineStartIdx = 0;
            runParam.s2LineEndIdx = this->constInfo.s2Size;
        }
    }

    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
        if (this->sharedParams.implMode ==
            static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
            if (this->sharedParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
                // s1baseSize行都无效时, 将startIdx设置为0, endIdx设置为S2realSize
                if (runParam.s2LineEndIdx - runParam.s2LineStartIdx <= 0) {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx = Min(this->constInfo.s2Size, 128L);
                }
            }
        }
    }
    if constexpr (BaseClass::isFp8) {
        runParam.s2LineStartIdx = runParam.s2LineStartIdx >> 7 << 7;
    }

    return;
}
}
#endif // WP_FLASH_ATTENTION_SCORE_KERNEL_TRAIN_H_
