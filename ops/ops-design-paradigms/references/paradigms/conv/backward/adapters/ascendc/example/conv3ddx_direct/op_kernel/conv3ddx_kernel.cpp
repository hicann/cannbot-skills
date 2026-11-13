/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3ddx_kernel.cpp
 * \brief conv3ddx 直调 kernel（`<<<>>>` launch，AIC-only `__cube__`）
 *
 * 实现真值：generic/input/compute-flow.md §5～§8（指令语义/窗口/任务/DAG）+
 * adapters/ascendc/{instruction-mapping,kernel-template,sync-implementation}.md（API 参数与同步协议）。
 * 指令参数与 CANN 包内置 conv3d_backprop_input_v2（arch35）实现逐项对齐（GM→L1 的 Nd2Nz/Dn2Nz、
 * Load3D/Load2D、Mmad、Fixpipe 参数值同构；L1 队列深度与 TQue 形态同构）。
 *
 * 编译维度：dtype（half/bfloat16_t）× loadB2Condition（1/2/3）共 6 个 launch 入口；
 * dedy/dedx/weight 的 NCDHW/NDHWC 双格式为运行时分支（payload fmtIsNdhwc）。
 *
 * 初始化协议（preprocessing §5.2）：主计算完整写回变体——
 * needCompute 块经 Fixpipe 全量写回有效 M/N；无有效乘积块（needCompute=false）由
 * StoreZeroBlock 显式写零覆盖本块有效区域；主计算入口按 AIC 通路编译（无融合等待）。
 */

#include "acl/acl.h"
#include "kernel_operator.h"

#include "conv3ddx_launch.h"
#include "conv3ddx_tiling_data.h"

using namespace AscendC;

namespace {

/* ---------------- 标量小工具 ---------------- */

__aicore__ inline uint32_t CeilDivL(uint64_t a, uint64_t b)
{
    return b == 0 ? 0 : static_cast<uint32_t>((a + b - 1) / b);
}

__aicore__ inline uint32_t AlignUp16L(uint32_t a)
{
    return (a + 15u) >> 4 << 4;
}

__aicore__ inline uint32_t CeilDiv16L(uint32_t a)
{
    return (a + 15u) >> 4;
}

__aicore__ inline uint32_t AlignUpTo(uint32_t a, uint32_t b)
{
    return CeilDivL(a, b) * b;
}

constexpr uint32_t BLOCK_SIZE_ELEM = 256u; /* 512B 分形（fp16/bf16 256 元素） */

/* ---------------- tiling 装载（GM → 本地 POD） ---------------- */

__aicore__ inline void ReadTilingFromGM(GM_ADDR tilingGm, Conv3DdxTilingData& td)
{
    /* payload 位于路由头（7×uint32=28B）之后 */
    __gm__ const uint8_t* p = reinterpret_cast<__gm__ const uint8_t*>(tilingGm) + 28u;
    uint8_t* dst = reinterpret_cast<uint8_t*>(&td);
    for (uint32_t i = 0; i < sizeof(td); ++i) {
        dst[i] = p[i];
    }
}

/* ---------------- Kernel 实现 ---------------- */

template <typename T, uint32_t LOADB2>
class Conv3ddxKernel {
public:
    __aicore__ inline Conv3ddxKernel() {}
    __aicore__ inline ~Conv3ddxKernel() {}

    __aicore__ inline void Init(GM_ADDR filter, GM_ADDR dedy, GM_ADDR dedx, const Conv3DdxTilingData& td)
    {
        td_ = td;
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(filter));
        dyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dedy));
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dedx));
        xRaw_ = reinterpret_cast<__gm__ T*>(dedx);
        ndhwc_ = (td.fmtIsNdhwc != 0);

        coreId_ = GetBlockIdx();
        coreNum_ = td.coreNum;

        hoWo_ = td.ho * td.wo;
        inputSlotCount_ = (td.al0Pbuffer >= 2 && td.bl0Pbuffer >= 2) ? 2u : 1u;
        l0aHalfElem_ = (td.al0Pbuffer >= 2) ? (CONV3DDX_L0A_BYTES / 2 / sizeof(T)) : 0;
        l0bHalfElem_ = (td.bl0Pbuffer >= 2) ? (CONV3DDX_L0B_BYTES / 2 / sizeof(T)) : 0;
        cSlotCount_ = (td.cl0Pbuffer >= 2) ? 2u : 1u;

        /* Buffer 分配（buffer-and-types §3；TQue 形态与内置一致：A1/B1 深度 2、CO1 深度 1） */
        pipe_.InitBuffer(queAl1_, td.al1Pbuffer, td.aSlotBytes);
        pipe_.InitBuffer(queBl1_, td.bl1Pbuffer, td.bSlotBytes);
        pipe_.InitBuffer(a2Buf_, CONV3DDX_L0A_BYTES);
        pipe_.InitBuffer(b2Buf_, CONV3DDX_L0B_BYTES);
        pipe_.InitBuffer(queCl0Ping_, 1, CONV3DDX_L0C_BYTES / cSlotCount_);
        if (cSlotCount_ == 2) {
            pipe_.InitBuffer(queCl0Pong_, 1, CONV3DDX_L0C_BYTES / cSlotCount_);
        }

        a2Base_ = a2Buf_.Get<T>();
        b2Base_ = b2Buf_.Get<T>();

        /* Load3D 参数初值（instruction-mapping §4.1；前放大后单位步长） */
        load3d_.strideW = 1;
        load3d_.strideH = 1;
        load3d_.filterW = static_cast<uint8_t>(td.wk);
        load3d_.filterH = static_cast<uint8_t>(td.hk);
        load3d_.filterSizeW = td.wk > 255u;
        load3d_.filterSizeH = td.hk > 255u;
        load3d_.dilationFilterW = static_cast<uint8_t>(td.dilationW);
        load3d_.dilationFilterH = static_cast<uint8_t>(td.dilationH);
        load3d_.enTranspose = false;
        load3d_.enSmallK = false;
        load3d_.fMatrixCtrl = false;
        load3d_.padValue = static_cast<T>(0);

        /* Load2D 参数初值（instruction-mapping §5.1） */
        load2dv2_.mStartPosition = 0;
        load2dv2_.kStartPosition = 0;
        load2dv2_.mStep = 1;
        load2dv2_.kStep = 1;
        load2dv2_.dstStride = 1;
        load2dv2_.ifTranspose = (LOADB2 != CONV3DDX_TPL_REVERSE_ONLY);
        load2dv2_.sid = 0;

        /* L1 持有状态复位 */
        a1Valid_ = false;
        b1Valid_ = false;
        a1SegFirst_ = 0xFFFFFFFFu;
        a1Dout_ = 0xFFFFFFFFu;
        b1SegFirst_ = 0xFFFFFFFFu;
        b1IdKd_ = 0xFFFFFFFFu;
        b1IdCin_ = 0xFFFFFFFFu;
        b1IdGroup_ = 0xFFFFFFFFu;
        l0Slot_ = 0;
        cSlot_ = 0;
    }

    __aicore__ inline void Process()
    {
        const Conv3DdxTilingData& td = td_;
        const uint64_t mCount = CeilDivL(static_cast<uint64_t>(td.hi) * td.wi, td.singleCoreM);
        const uint64_t nCount = CeilDivL(td.cinG, td.singleCoreCin);
        const uint64_t dCount = CeilDivL(td.di, td.singleCoreDin);
        /* 任务轴序（compute-flow §6.1）：batch/group 在外层，内层从慢到快按条件三选一 */
        uint64_t cnt0 = 0;
        uint64_t cnt1 = 0;
        uint64_t cnt2 = 0;
        uint32_t ax0 = 0;
        uint32_t ax1 = 0;
        uint32_t ax2 = 0; /* 轴编码：0=空间 M、1=深度 D、2=通道 N */
        if (td.dk > 1) {
            ax0 = 0;
            ax1 = 1;
            ax2 = 2;
            cnt0 = mCount;
            cnt1 = dCount;
            cnt2 = nCount;
        } else if ((td.hk > 1) || (mCount > nCount)) {
            ax0 = 1;
            ax1 = 0;
            ax2 = 2;
            cnt0 = dCount;
            cnt1 = mCount;
            cnt2 = nCount;
        } else {
            ax0 = 1;
            ax1 = 2;
            ax2 = 0;
            cnt0 = dCount;
            cnt1 = nCount;
            cnt2 = mCount;
        }
        const uint64_t bgCount = static_cast<uint64_t>(td.batch) * td.group;
        const uint64_t totalTasks = bgCount * cnt0 * cnt1 * cnt2;

        for (uint64_t taskId = coreId_; taskId < totalTasks; taskId += coreNum_) {
            uint64_t t = taskId;
            const uint64_t idx2 = t % cnt2;
            t /= cnt2;
            const uint64_t idx1 = t % cnt1;
            t /= cnt1;
            const uint64_t idx0 = t % cnt0;
            t /= cnt0;
            const uint64_t bg = t % bgCount;
            const uint64_t batchIdx = bg / td.group;
            const uint64_t groupIdx = bg % td.group;

            uint64_t mIdx = 0;
            uint64_t dIdx = 0;
            uint64_t nIdx = 0;
            if (ax0 == 0) {
                mIdx = idx0;
            } else if (ax0 == 1) {
                dIdx = idx0;
            } else {
                nIdx = idx0;
            }
            if (ax1 == 0) {
                mIdx = idx1;
            } else if (ax1 == 1) {
                dIdx = idx1;
            } else {
                nIdx = idx1;
            }
            if (ax2 == 0) {
                mIdx = idx2;
            } else if (ax2 == 1) {
                dIdx = idx2;
            } else {
                nIdx = idx2;
            }

            const uint64_t spaceStart = mIdx * td.singleCoreM;
            const uint64_t mTotal = static_cast<uint64_t>(td.hi) * td.wi;
            const uint64_t spaceUse = (spaceStart + td.singleCoreM <= mTotal) ? td.singleCoreM : (mTotal - spaceStart);
            const uint64_t channelStart = nIdx * td.singleCoreCin;
            const uint64_t channelUse =
                (channelStart + td.singleCoreCin <= td.cinG) ? td.singleCoreCin : (td.cinG - channelStart);
            const uint64_t depthStart = dIdx * td.singleCoreDin;
            const uint64_t depthUse =
                (depthStart + td.singleCoreDin <= td.di) ? td.singleCoreDin : (td.di - depthStart);
            batchIdx_ = batchIdx;
            groupIdx_ = groupIdx;

            for (uint64_t d = 0; d < depthUse; ++d) {
                curDinIdx_ = static_cast<uint32_t>(depthStart + d);
                for (uint64_t n = 0; n < channelUse; n += td.baseN) {
                    cinStart_ = static_cast<uint32_t>(channelStart + n);
                    baseUseN_ = (n + td.baseN <= channelUse) ? td.baseN : static_cast<uint32_t>(channelUse - n);
                    nCompute_ = AlignUp16L(baseUseN_);
                    blockBaseN_ = CeilDiv16L(baseUseN_);
                    if (blockBaseN_ == 0) {
                        blockBaseN_ = 1;
                    }
                    for (uint64_t m = 0; m < spaceUse; m += td.baseM) {
                        mStart_ = static_cast<uint32_t>(spaceStart + m);
                        baseUseM_ = (m + td.baseM <= spaceUse) ? td.baseM : static_cast<uint32_t>(spaceUse - m);
                        mCompute_ = AlignUp16L(baseUseM_);
                        UpdateL1ComputeInfo();
                        if (!needCompute_) {
                            StoreZeroBlock();
                            continue;
                        }
                        ComputeOneBlock();
                    }
                }
            }
        }
        FreeB1Tensor();
        FreeA1Tensor();
    }

    __aicore__ inline void End()
    {
        /* 全部队列成对平衡（Alloc/EnQue/DeQue/FreeTensor），TPipe 析构完成事件收尾 */
    }

private:
    /* ---------------- 单输出块主计算（kernel-template §2.2 骨架 + sync §1 事件对） ---------------- */

    __aicore__ inline void ComputeOneBlock()
    {
        const Conv3DdxTilingData& td = td_;
        auto& q = (cSlot_ == 0) ? queCl0Ping_ : queCl0Pong_;
        LocalTensor<float> useC1 = q.template AllocTensor<float>();
        bool firstContribution = true;
        for (uint32_t kd = 0; kd < td.dk; ++kd) {
            const int64_t dTmp =
                static_cast<int64_t>(curDinIdx_) + td.padFront - static_cast<int64_t>(kd) * td.dilationD;
            if (dTmp < 0 || (td.strideD > 1 && dTmp % td.strideD != 0) ||
                dTmp >= static_cast<int64_t>(td.dout) * td.strideD) {
                continue;
            }
            const uint32_t curDoutIdx = static_cast<uint32_t>(td.strideD > 1 ? dTmp / td.strideD : dTmp);
            for (uint64_t kIdx = 0; kIdx < td.kIter; ++kIdx) {
                const uint64_t kStart = kIdx * td.baseK;
                baseUseK_ = (kIdx + 1 == td.kIter) ? td.tailK : td.baseK;
                EnsureA1(curDoutIdx, static_cast<uint32_t>(kIdx));
                EnsureB1(kd, static_cast<uint32_t>(kIdx));
                /* L0 输入槽交接（sync-implementation §1 稳态事件对） */
                WaitFlag<HardEvent::M_MTE1>(l0Slot_);
                IssueLoadToA2(kStart);
                IssueLoadToB2(kStart);
                SetFlag<HardEvent::MTE1_M>(l0Slot_);
                WaitFlag<HardEvent::MTE1_M>(l0Slot_);
                IssueMmad(useC1, firstContribution);
                firstContribution = false;
                SetFlag<HardEvent::M_MTE1>(l0Slot_);
                l0Slot_ = (l0Slot_ + 1) % inputSlotCount_;
                /* 段末释放（最后消费者 MTE1 读取已提交后由队列发布归还） */
                ReleaseA1IfSegmentEnd(curDoutIdx, static_cast<uint32_t>(kIdx));
                ReleaseB1IfSegmentEnd(static_cast<uint32_t>(kIdx));
            }
        }
        if (!firstContribution) {
            q.EnQue(useC1);
            LocalTensor<float> c1v = q.template DeQue<float>();
            LoadL0c2Gm(c1v);
            q.FreeTensor(c1v);
        } else {
            q.FreeTensor(useC1);
        }
        cSlot_ = (cSlot_ + 1) % cSlotCount_;
    }

    /* ---------------- 行带与深度采样（compute-flow §5.2.1） ---------------- */

    __aicore__ inline void UpdateL1ComputeInfo()
    {
        const Conv3DdxTilingData& td = td_;
        bool dValid = false;
        for (uint32_t kd = 0; kd < td.dk; ++kd) {
            const int64_t dTmp =
                static_cast<int64_t>(curDinIdx_) + td.padFront - static_cast<int64_t>(kd) * td.dilationD;
            if (dTmp >= 0 && (td.strideD == 1 || dTmp % td.strideD == 0) &&
                dTmp < static_cast<int64_t>(td.dout) * td.strideD) {
                dValid = true;
                break;
            }
        }
        const uint32_t curHiIdx = mStart_ / td.wi;
        const uint32_t curWiIdx = mStart_ % td.wi;
        curHoIdx_ = static_cast<int32_t>(curHiIdx) - td.backpropPadUp;
        const uint32_t hiCal = CeilDivL(static_cast<uint64_t>(baseUseM_) + curWiIdx, td.wi);
        const int64_t endHoIdx = static_cast<int64_t>(curHoIdx_) + hiCal + (td.hk - 1) * td.dilationH;
        if (dValid && curHoIdx_ < static_cast<int32_t>(td.hoExpand) && endHoIdx > 0) {
            const uint32_t realStart = curHoIdx_ < 0 ? 0u : static_cast<uint32_t>(curHoIdx_);
            const uint32_t realEnd =
                static_cast<uint32_t>(endHoIdx) > td.hoExpand ? td.hoExpand : static_cast<uint32_t>(endHoIdx);
            curHoSize_ = realEnd - realStart;
            needCompute_ = true;
        } else {
            curHoSize_ = 0;
            needCompute_ = false;
        }
        /* 新输出块：A1 身份失效（B1 由 enableFullLoad 决定是否跨块保留） */
        a1SegFirst_ = 0xFFFFFFFFu;
        a1Dout_ = 0xFFFFFFFFu;
    }

    /* ---------------- L1A 段装载（instruction-mapping §2 + 内置真值参数表） ---------------- */

    __aicore__ inline void EnsureA1(uint32_t curDoutIdx, uint32_t kIdx)
    {
        const Conv3DdxTilingData& td = td_;
        const uint32_t segFirst = (kIdx / td.stepKa) * td.stepKa;
        if (segFirst == a1SegFirst_ && curDoutIdx == a1Dout_ && a1Valid_) {
            return;
        }
        FreeA1Tensor();
        a1SegFirst_ = segFirst;
        a1Dout_ = curDoutIdx;
        const uint64_t segStartK = static_cast<uint64_t>(segFirst) * td.baseK;
        const uint32_t segIters = (td.kIter - segFirst) < td.stepKa ? (td.kIter - segFirst) : td.stepKa;
        const uint32_t segUseK =
            static_cast<uint32_t>(static_cast<uint64_t>(segIters) * td.baseK < td.kValue - segStartK ?
                                      static_cast<uint64_t>(segIters) * td.baseK :
                                      td.kValue - segStartK);
        aSegStartK_ = static_cast<uint32_t>(segStartK);

        /* 通道范围（父段 Cout 覆盖；窗口配对保证段起点为分形对齐） */
        const uint32_t coutIdx = static_cast<uint32_t>(segStartK / td.hkWk);
        uint32_t coutSize = segUseK / td.hkWk;
        const uint32_t coutRemain = td.coutG - coutIdx;
        coutSize = coutSize < coutRemain ? coutSize : coutRemain;
        const uint32_t coutAligned = AlignUp16L(coutSize);
        const uint32_t coutGAligned = AlignUp16L(td.coutG);
        channelSize_ = coutAligned < coutGAligned ? coutAligned : coutGAligned;
        if (channelSize_ == 0) {
            channelSize_ = CONV3DDX_C0;
        }

        /* W 裁剪（负反向边界先裁剪真实 dedy，compute-flow §5.2.1） */
        const int32_t bpLeft = td.backpropPadLeft;
        const int32_t bpRight = td.backpropPadRight;
        const uint32_t woExpandRaw = td.woExpand;
        uint32_t woStart = 0;
        uint32_t woEnd = CeilDivL(woExpandRaw, td.strideW);
        bool resetWoExpand = false;
        if (bpLeft < 0) {
            resetWoExpand = true;
            woStart = CeilDivL(static_cast<uint32_t>(-bpLeft), td.strideW);
        }
        if (bpRight < 0) {
            resetWoExpand = true;
            /* 范式 compute-flow §5.2.1：woEnd=ceilDiv(expandedW+backpropPadRight,strideW)——带符号相加，
             * 负右边界收窄有效列（若对 bpRight 取绝对值会虚增 realWo，致 GM 读越界与 L1A 行宽错位）。
             * 合法规格恒有 expandedW+bpRight=wi+padLeft≥1（正向输出几何恒等式），int64 相加后转无符号安全。 */
            woEnd = CeilDivL(static_cast<uint64_t>(static_cast<int64_t>(woExpandRaw) + bpRight), td.strideW);
        }
        uint32_t woExpandLocal = woExpandRaw;
        uint32_t realWo = td.wo;
        if (resetWoExpand) {
            woExpandLocal = (woEnd <= woStart) ? 1u : (woEnd - woStart - 1) * td.strideW + 1u;
            realWo = woEnd > woStart ? (woEnd - woStart) : 0u;
        }
        woExpandLocal_ = woExpandLocal;

        /* stride 相位：首个真实 dedy 行在父段内的行偏移与其绝对行号 */
        const uint32_t realStart = curHoIdx_ < 0 ? 0u : static_cast<uint32_t>(curHoIdx_);
        const uint32_t hSkip =
            (td.strideH > 1 && realStart % td.strideH != 0) ? (AlignUpTo(realStart, td.strideH) - realStart) : 0u;
        const uint32_t hLoop = (curHoSize_ > hSkip) ? CeilDivL(curHoSize_ - hSkip, td.strideH) : 0u;
        const uint32_t firstRealRow = CeilDivL(realStart, td.strideH);

        const uint64_t a1DstOffset = static_cast<uint64_t>(hSkip) * woExpandLocal * CONV3DDX_C0;

        /* FMatrix（padList 顺序左、右、上、下=255 哨兵；instruction-mapping §4.2） */
        load3d_.l1H = static_cast<uint16_t>(curHoSize_);
        load3d_.l1W = static_cast<uint16_t>(woExpandLocal);
        load3d_.padList[0] = static_cast<uint8_t>(
            bpLeft < 0 ?
                static_cast<int32_t>(CeilDivL(static_cast<uint32_t>(-bpLeft), td.strideW) * td.strideW) + bpLeft :
                bpLeft);
        load3d_.padList[1] = static_cast<uint8_t>(bpRight < 0 ? static_cast<int32_t>(woExpandRaw) + bpRight -
                                                                    static_cast<int32_t>((woEnd - 1) * td.strideW + 1) :
                                                                bpRight);
        load3d_.padList[2] = static_cast<uint8_t>(curHoIdx_ < 0 ? -curHoIdx_ : 0);
        load3d_.padList[3] = 255u;
        load3d_.channelSize = static_cast<uint16_t>(channelSize_);

        LocalTensor<T> a1Alloc = queAl1_.template AllocTensor<T>();
        FillZero(a1Alloc, td.aSlotBytes);
        const uint64_t coGlobal = groupIdx_ * td.coutG + coutIdx;
        const uint64_t dyBatchOff = ndhwc_ ? batchIdx_ * static_cast<uint64_t>(td.dout) * hoWo_ * td.cout :
                                             batchIdx_ * static_cast<uint64_t>(td.cout) * td.doHoWo;
        uint64_t srcBase = dyBatchOff;
        if (!ndhwc_) {
            srcBase += coGlobal * td.doHoWo + static_cast<uint64_t>(curDoutIdx) * hoWo_ +
                       static_cast<uint64_t>(firstRealRow) * td.wo + woStart;
        } else {
            srcBase += static_cast<uint64_t>(curDoutIdx) * hoWo_ * td.cout +
                       static_cast<uint64_t>(firstRealRow) * td.wo * td.cout +
                       static_cast<uint64_t>(woStart) * td.cout + coGlobal;
        }
        if (hLoop > 0) {
            if (td.strideW * td.strideH > 1 || realWo != td.wo || resetWoExpand) {
                /* 逐行搬入：目的步幅表达 W/H 前放大（内置 CalcLoadToA1Dn2Nz/Nd2Nz 参数表） */
                if (!ndhwc_) {
                    Dn2NzParams p;
                    p.dnNum = static_cast<uint16_t>(hLoop);
                    p.nValue = static_cast<uint16_t>(realWo);
                    p.dValue = coutSize;
                    p.srcDnMatrixStride = td.wo;
                    p.srcDValue = td.doHoWo;
                    p.dstNzC0Stride = static_cast<uint16_t>(curHoSize_ * woExpandLocal);
                    p.dstNzNStride = static_cast<uint16_t>(td.strideW);
                    p.dstNzMatrixStride = static_cast<uint32_t>(td.strideH) * woExpandLocal * CONV3DDX_C0;
                    DataCopy(a1Alloc[a1DstOffset], dyGm_[srcBase], p);
                } else {
                    Nd2NzParams p;
                    p.ndNum = static_cast<uint16_t>(hLoop);
                    p.nValue = static_cast<uint16_t>(realWo);
                    p.dValue = coutSize;
                    p.srcNdMatrixStride = static_cast<uint64_t>(td.wo) * td.cout;
                    p.srcDValue = td.cout;
                    p.dstNzC0Stride = static_cast<uint16_t>(curHoSize_ * woExpandLocal);
                    p.dstNzNStride = static_cast<uint16_t>(td.strideW);
                    p.dstNzMatrixStride = static_cast<uint32_t>(td.strideH) * woExpandLocal * CONV3DDX_C0;
                    DataCopy(a1Alloc[a1DstOffset], dyGm_[srcBase], p);
                }
            } else {
                /* 无插零/裁剪：连续行合成单矩阵 */
                if (!ndhwc_) {
                    Dn2NzParams p;
                    p.dnNum = 1;
                    p.nValue = static_cast<uint16_t>(hLoop * td.wo);
                    p.dValue = coutSize;
                    p.srcDnMatrixStride = 0;
                    p.srcDValue = td.doHoWo;
                    p.dstNzC0Stride = static_cast<uint16_t>(curHoSize_ * woExpandLocal);
                    p.dstNzNStride = 1;
                    p.dstNzMatrixStride = 0;
                    DataCopy(a1Alloc[a1DstOffset], dyGm_[srcBase], p);
                } else {
                    Nd2NzParams p;
                    p.ndNum = 1;
                    p.nValue = static_cast<uint16_t>(hLoop * td.wo);
                    p.dValue = coutSize;
                    p.srcNdMatrixStride = 0;
                    p.srcDValue = td.cout;
                    p.dstNzC0Stride = static_cast<uint16_t>(curHoSize_ * woExpandLocal);
                    p.dstNzNStride = 1;
                    p.dstNzMatrixStride = 0;
                    DataCopy(a1Alloc[a1DstOffset], dyGm_[srcBase], p);
                }
            }
        }
        queAl1_.EnQue(a1Alloc);
        a1v_ = queAl1_.template DeQue<T>();
        a1Valid_ = true;
    }

    __aicore__ inline void ReleaseA1IfSegmentEnd(uint32_t curDoutIdx, uint32_t kIdx)
    {
        const Conv3DdxTilingData& td = td_;
        const uint32_t segFirst = (kIdx / td.stepKa) * td.stepKa;
        const uint32_t segIters = (td.kIter - segFirst) < td.stepKa ? (td.kIter - segFirst) : td.stepKa;
        if (kIdx + 1 == segFirst + segIters && a1Valid_) {
            (void)curDoutIdx;
            queAl1_.FreeTensor(a1v_);
            a1Valid_ = false;
        }
    }

    __aicore__ inline void FreeA1Tensor()
    {
        if (a1Valid_) {
            queAl1_.FreeTensor(a1v_);
            a1Valid_ = false;
        }
    }

    /* ---------------- L1B 段装载（instruction-mapping §3 + 内置真值参数表） ---------------- */

    __aicore__ inline void EnsureB1(uint32_t kd, uint32_t kIdx)
    {
        const Conv3DdxTilingData& td = td_;
        const uint32_t segFirst = (kIdx / td.stepKb) * td.stepKb;
        if (segFirst == b1SegFirst_ && b1Valid_ && b1IdKd_ == kd && b1IdCin_ == cinStart_ && b1IdGroup_ == groupIdx_) {
            return;
        }
        FreeB1Tensor();
        b1SegFirst_ = segFirst;
        b1IdKd_ = kd;
        b1IdCin_ = cinStart_;
        b1IdGroup_ = groupIdx_;
        const uint64_t segStartK = static_cast<uint64_t>(segFirst) * td.baseK;
        const uint32_t segIters = (td.kIter - segFirst) < td.stepKb ? (td.kIter - segFirst) : td.stepKb;
        const uint32_t segUseK =
            static_cast<uint32_t>(static_cast<uint64_t>(segIters) * td.baseK < td.kValue - segStartK ?
                                      static_cast<uint64_t>(segIters) * td.baseK :
                                      td.kValue - segStartK);
        bSegStartK_ = static_cast<uint32_t>(segStartK);

        const uint32_t coutIdx = static_cast<uint32_t>(segStartK / td.hkWk);
        uint32_t coutSize = segUseK / td.hkWk;
        const uint32_t coutRemain = td.coutG - coutIdx;
        coutSize = coutSize < coutRemain ? coutSize : coutRemain;
        curCout1Cout0Size_ = AlignUp16L(coutSize);
        curLoadKbl1_ = curCout1Cout0Size_ * td.hkWk;

        const uint32_t cinRemain = td.cinG - cinStart_;
        const uint32_t cinSize = td.baseN < cinRemain ? td.baseN : cinRemain;
        const uint32_t cinAligned = AlignUp16L(cinSize);

        const uint64_t coGlobal = groupIdx_ * td.coutG + coutIdx;
        uint64_t srcBase = 0;
        if (LOADB2 == CONV3DDX_TPL_TRANSPOSE_ONLY) {
            /* 核体积 1：两种 weight 格式存储同为 (cout, cinG)（内置 LoadToB1Nd2NzForDkHkWkIsOne） */
            srcBase = coGlobal * td.cinG + cinStart_;
        } else if (!ndhwc_) {
            srcBase = coGlobal * td.cinG * td.dkHkWk + static_cast<uint64_t>(cinStart_) * td.dkHkWk +
                      static_cast<uint64_t>(kd) * td.hkWk;
        } else {
            srcBase = coGlobal * td.cinG * td.dkHkWk + static_cast<uint64_t>(kd) * td.hkWk * td.cinG + cinStart_;
        }

        LocalTensor<T> b1Alloc = queBl1_.template AllocTensor<T>();
        FillZero(b1Alloc, td.bSlotBytes);
        if (coutSize > 0 && cinSize > 0) {
            if (LOADB2 == CONV3DDX_TPL_TRANSPOSE_ONLY) {
                Nd2NzParams p;
                p.ndNum = 1;
                p.nValue = static_cast<uint16_t>(coutSize);
                p.dValue = cinSize;
                p.srcNdMatrixStride = 0;
                p.srcDValue = td.cinG;
                p.dstNzC0Stride = static_cast<uint16_t>(curCout1Cout0Size_);
                p.dstNzNStride = 1;
                p.dstNzMatrixStride = 0;
                DataCopy(b1Alloc, wGm_[srcBase], p);
            } else if (LOADB2 == CONV3DDX_TPL_TRANSPOSE_AND_REVERSE) {
                if (!ndhwc_) {
                    Dn2NzParams p;
                    p.dnNum = static_cast<uint16_t>(coutSize);
                    p.nValue = static_cast<uint16_t>(td.hkWk);
                    p.dValue = cinSize;
                    p.srcDnMatrixStride = static_cast<uint64_t>(td.cinG) * td.dkHkWk;
                    p.srcDValue = td.dkHkWk;
                    p.dstNzC0Stride = static_cast<uint16_t>(td.hkWk * curCout1Cout0Size_);
                    p.dstNzNStride = static_cast<uint16_t>(curCout1Cout0Size_);
                    p.dstNzMatrixStride = CONV3DDX_C0;
                    DataCopy(b1Alloc, wGm_[srcBase], p);
                } else {
                    Nd2NzParams p;
                    p.ndNum = static_cast<uint16_t>(coutSize);
                    p.nValue = static_cast<uint16_t>(td.hkWk);
                    p.dValue = cinSize;
                    p.srcNdMatrixStride = static_cast<uint64_t>(td.cinG) * td.dkHkWk;
                    p.srcDValue = td.cinG;
                    p.dstNzC0Stride = static_cast<uint16_t>(td.hkWk * curCout1Cout0Size_);
                    p.dstNzNStride = static_cast<uint16_t>(curCout1Cout0Size_);
                    p.dstNzMatrixStride = CONV3DDX_C0;
                    DataCopy(b1Alloc, wGm_[srcBase], p);
                }
            } else { /* LOADB2 == CONV3DDX_TPL_REVERSE_ONLY（仅逆序，内置 Dn2NzTranspose/Nd2NzTranspose） */
                if (!ndhwc_) {
                    Dn2NzParams p;
                    p.dnNum = static_cast<uint16_t>(cinSize);
                    p.nValue = static_cast<uint16_t>(td.hkWk);
                    p.dValue = coutSize;
                    p.srcDnMatrixStride = td.dkHkWk;
                    p.srcDValue = static_cast<uint64_t>(td.cinG) * td.dkHkWk;
                    p.dstNzC0Stride = static_cast<uint16_t>(td.hkWk * cinAligned);
                    p.dstNzNStride = static_cast<uint16_t>(cinAligned);
                    p.dstNzMatrixStride = CONV3DDX_C0;
                    DataCopy(b1Alloc, wGm_[srcBase], p);
                } else {
                    Dn2NzParams p;
                    p.dnNum = static_cast<uint16_t>(td.hkWk);
                    p.dValue = coutSize;
                    p.nValue = static_cast<uint16_t>(cinSize);
                    p.srcDnMatrixStride = td.cinG;
                    p.srcDValue = static_cast<uint64_t>(td.cinG) * td.dkHkWk;
                    p.dstNzC0Stride = static_cast<uint16_t>(td.hkWk * cinAligned);
                    p.dstNzNStride = 1;
                    p.dstNzMatrixStride = cinAligned * CONV3DDX_C0;
                    DataCopy(b1Alloc, wGm_[srcBase], p);
                }
            }
        }
        queBl1_.EnQue(b1Alloc);
        b1v_ = queBl1_.template DeQue<T>();
        b1Valid_ = true;
    }

    __aicore__ inline void ReleaseB1IfSegmentEnd(uint32_t kIdx)
    {
        const Conv3DdxTilingData& td = td_;
        if (td.enableFullLoad != 0) {
            return; /* 驻留：跨输出块/任务复用，身份变化或结束时释放（compute-flow §8.4） */
        }
        const uint32_t segFirst = (kIdx / td.stepKb) * td.stepKb;
        const uint32_t segIters = (td.kIter - segFirst) < td.stepKb ? (td.kIter - segFirst) : td.stepKb;
        if (kIdx + 1 == segFirst + segIters && b1Valid_) {
            queBl1_.FreeTensor(b1v_);
            b1Valid_ = false;
        }
    }

    __aicore__ inline void FreeB1Tensor()
    {
        if (b1Valid_) {
            queBl1_.FreeTensor(b1v_);
            b1Valid_ = false;
        }
    }

    /* ---------------- L1→L0 装载指令（instruction-mapping §4/§5） ---------------- */

    __aicore__ inline void IssueLoadToA2(uint64_t kStart)
    {
        const Conv3DdxTilingData& td = td_;
        const uint32_t kLocalA = static_cast<uint32_t>(kStart - aSegStartK_);
        load3d_.kStartPt = static_cast<uint16_t>(kLocalA);
        load3d_.kExtension = static_cast<uint16_t>(baseUseK_);
        load3d_.mExtension = static_cast<uint16_t>(baseUseM_);
        load3d_.mStartPt = static_cast<uint16_t>(mStart_ % td.wi);
        LocalTensor<T> a2 = a2Base_;
        if (l0Slot_ == 1) {
            a2 = a2[l0aHalfElem_];
        }
#if defined(ASC_DEVKIT_VERSION_NUM) && ASC_DEVKIT_VERSION_NUM >= 90000000
        LoadDataRepeatParamWithStride rp = {0, 1, 0, static_cast<uint16_t>(CeilDiv16L(baseUseM_))};
        SetLoadDataRepeatWithStride(rp);
        LoadDataWithStride(a2, a1v_, load3d_);
#else
        LoadDataRepeatParam rp = {0, 1, 0};
        SetLoadDataRepeat(rp);
        LoadData(a2, a1v_, load3d_);
#endif
    }

    __aicore__ inline void IssueLoadToB2(uint64_t kStart)
    {
        const Conv3DdxTilingData& td = td_;
        LocalTensor<T> b2 = b2Base_;
        if (l0Slot_ == 1) {
            b2 = b2[l0bHalfElem_];
        }
        const uint32_t kLocalB = static_cast<uint32_t>(kStart - bSegStartK_);
        const uint32_t bKStart = kLocalB / CONV3DDX_C0;
        const uint32_t bKEnd = bKStart + CeilDiv16L(baseUseK_);
        const uint32_t curKRepeat = CeilDiv16L(baseUseK_);
        if (LOADB2 == CONV3DDX_TPL_TRANSPOSE_AND_REVERSE) {
            /* 转置加逆序：每个 K 分形一次 LoadData，核空间逆序由源偏移表达 */
            load2dv2_.mStartPosition = 0;
            load2dv2_.kStartPosition = 0;
            load2dv2_.mStep = 1;
            load2dv2_.kStep = static_cast<uint16_t>(blockBaseN_);
            load2dv2_.srcStride = static_cast<int32_t>(curLoadKbl1_ / CONV3DDX_C0);
            load2dv2_.dstStride = 1;
            uint32_t dstOffset = 0;
            for (uint32_t i = 0; i < curKRepeat; ++i) {
                const uint32_t r = bKStart + i;
                const uint32_t idxC1out = r / td.hkWk;
                const uint32_t srcOffset =
                    idxC1out * BLOCK_SIZE_ELEM + (td.hkWk - 1 - (r % td.hkWk)) * curCout1Cout0Size_ * CONV3DDX_C0;
                LoadData(b2[dstOffset], b1v_[srcOffset], load2dv2_);
                dstOffset += blockBaseN_ * BLOCK_SIZE_ELEM;
            }
        } else if (LOADB2 == CONV3DDX_TPL_TRANSPOSE_ONLY) {
            /* 仅转置：K 分形 = cout1 分形，源偏移按分形正序递增（核体积 1，空间逆序为恒等） */
            load2dv2_.mStartPosition = 0;
            load2dv2_.kStartPosition = 0;
            load2dv2_.mStep = 1;
            load2dv2_.kStep = static_cast<uint16_t>(blockBaseN_);
            load2dv2_.srcStride = static_cast<int32_t>(curLoadKbl1_ / CONV3DDX_C0);
            load2dv2_.dstStride = 1;
            uint32_t dstOffset = 0;
            for (uint32_t i = 0; i < curKRepeat; ++i) {
                const uint32_t idxC1out = bKStart + i;
                const uint32_t srcOffset = idxC1out * BLOCK_SIZE_ELEM;
                LoadData(b2[dstOffset], b1v_[srcOffset], load2dv2_);
                dstOffset += blockBaseN_ * BLOCK_SIZE_ELEM;
            }
        } else {
            /* 仅逆序：按 Cout1 窗口分段提交，负 srcStride 沿 K 逆序读取（内置 ReverseOnlyCommon） */
            load2dv2_.mStep = static_cast<uint16_t>(blockBaseN_);
            load2dv2_.srcStride = -static_cast<int32_t>(blockBaseN_);
            load2dv2_.dstStride = static_cast<uint16_t>(blockBaseN_);
            uint32_t dstOffset = 0;
            const uint32_t wBegin = bKStart / td.hkWk;
            const uint32_t wEnd = CeilDivL(bKEnd, td.hkWk);
            for (uint32_t co1w = wBegin; co1w < wEnd; ++co1w) {
                const uint32_t windowBegin = co1w * td.hkWk;
                const uint32_t windowEnd = windowBegin + td.hkWk;
                const uint32_t first = bKStart > windowBegin ? bKStart : windowBegin;
                const uint32_t last = bKEnd < windowEnd ? bKEnd : windowEnd;
                load2dv2_.kStartPosition = windowBegin;
                load2dv2_.mStartPosition = (windowEnd - 1 - first) * blockBaseN_;
                load2dv2_.kStep = static_cast<uint16_t>(last - first);
                LoadData(b2[dstOffset], b1v_[0], load2dv2_);
                dstOffset += (last - first) * blockBaseN_ * BLOCK_SIZE_ELEM;
            }
        }
    }

    /* ---------------- MMAD（instruction-mapping §6） ---------------- */

    __aicore__ inline void IssueMmad(LocalTensor<float>& useC1, bool firstContribution)
    {
        LocalTensor<T> a2 = a2Base_;
        LocalTensor<T> b2 = b2Base_;
        if (l0Slot_ == 1) {
            a2 = a2[l0aHalfElem_];
            b2 = b2[l0bHalfElem_];
        }
        MmadParams mp;
        mp.m = static_cast<uint16_t>(mCompute_); /* 任务末 M 块向 16 对齐（language §8.3） */
        mp.n = static_cast<uint16_t>(baseUseN_);
        mp.k = static_cast<uint16_t>(baseUseK_);
        mp.unitFlag = 0;
        mp.kDirectionAlign = false;
        mp.cmatrixSource = false;
        mp.cmatrixInitVal = firstContribution;
        Mmad(useC1, a2, b2, mp);
    }

    /* ---------------- 写回（instruction-mapping §7 + 内置真值参数） ---------------- */

    __aicore__ inline void LoadL0c2Gm(LocalTensor<float>& c1v)
    {
        const Conv3DdxTilingData& td = td_;
        const uint64_t ciGlobal = groupIdx_ * td.cinG + cinStart_;
        const uint64_t xBatchOff = ndhwc_ ? batchIdx_ * static_cast<uint64_t>(td.di) * td.hiWi * td.cin :
                                            batchIdx_ * static_cast<uint64_t>(td.cin) * td.diHiWi;
        uint64_t dstOffset = xBatchOff;
        if (!ndhwc_) {
            dstOffset += ciGlobal * td.diHiWi + static_cast<uint64_t>(curDinIdx_) * td.hiWi + mStart_;
            FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> fp;
            fp.mSize = static_cast<uint16_t>(baseUseM_);
            fp.nSize = static_cast<uint16_t>(baseUseN_);
            fp.srcStride = static_cast<uint16_t>(mCompute_);
            fp.dstStride = td.diHiWi;
            fp.params.dnNum = 1;
            fp.params.srcNzMatrixStride = 0;
            fp.params.dstDnMatrixStride = 0;
            fp.params.srcNzC0Stride = 1;
            fp.quantPre = kIsBf16 ? QuantMode_t::F322BF16 : QuantMode_t::F322F16;
            fp.reluEn = false;
            fp.unitFlag = 0;
            Fixpipe<T, float, CFG_COLUMN_MAJOR>(xGm_[dstOffset], c1v, fp);
        } else {
            dstOffset += (static_cast<uint64_t>(curDinIdx_) * td.hiWi + mStart_) * td.cin + ciGlobal;
            FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fp;
            fp.mSize = static_cast<uint16_t>(baseUseM_);
            fp.nSize = static_cast<uint16_t>(baseUseN_);
            fp.srcStride = static_cast<uint16_t>(mCompute_);
            fp.dstStride = td.cin;
            fp.params.ndNum = 1;
            fp.quantPre = kIsBf16 ? QuantMode_t::F322BF16 : QuantMode_t::F322F16;
            fp.reluEn = false;
            fp.unitFlag = 0;
            Fixpipe<T, float, CFG_ROW_MAJOR>(xGm_[dstOffset], c1v, fp);
        }
    }

    /* ---------------- 无贡献块显式写零（初始化协议：主计算完整写回变体） ---------------- */

    __aicore__ inline void StoreZeroBlock()
    {
        const Conv3DdxTilingData& td = td_;
        const uint64_t ciGlobal = groupIdx_ * td.cinG + cinStart_;
        const uint64_t xBatchOff = ndhwc_ ? batchIdx_ * static_cast<uint64_t>(td.di) * td.hiWi * td.cin :
                                            batchIdx_ * static_cast<uint64_t>(td.cin) * td.diHiWi;
        for (uint32_t n = 0; n < baseUseN_; ++n) {
            const uint64_t ci = ciGlobal + n;
            for (uint32_t m = 0; m < baseUseM_; ++m) {
                uint64_t off = xBatchOff;
                if (!ndhwc_) {
                    off += ci * td.diHiWi + static_cast<uint64_t>(curDinIdx_) * td.hiWi + mStart_ + m;
                } else {
                    off += (static_cast<uint64_t>(curDinIdx_) * td.hiWi + mStart_ + m) * td.cin + ci;
                }
                xRaw_[off] = static_cast<T>(0);
            }
        }
    }

    /* ---------------- 零填充（instruction-mapping §2.2 Fill 协议） ---------------- */

    __aicore__ inline void FillZero(const LocalTensor<T>& buf, uint32_t slotBytes)
    {
        InitConstValueParams<T> z;
        z.repeatTimes = 1;
        z.blockNum = static_cast<uint16_t>(slotBytes / 32u);
        z.dstGap = 0;
        z.initValue = static_cast<T>(0);
        InitConstValue(buf, z);
        PipeBarrier<PIPE_MTE2>();
    }

    /* ---------------- 状态 ---------------- */
    static constexpr bool kIsBf16 = IsSameType<T, bfloat16_t>::value;

    Conv3DdxTilingData td_;
    GlobalTensor<T> wGm_;
    GlobalTensor<T> dyGm_;
    GlobalTensor<T> xGm_;
    __gm__ T* xRaw_ = nullptr;
    bool ndhwc_ = false;

    TPipe pipe_;
    TQue<TPosition::A1, 2> queAl1_;
    TQue<TPosition::B1, 2> queBl1_;
    TBuf<TPosition::A2> a2Buf_;
    TBuf<TPosition::B2> b2Buf_;
    TQue<TPosition::CO1, 1> queCl0Ping_;
    TQue<TPosition::CO1, 1> queCl0Pong_;

    LocalTensor<T> a2Base_;
    LocalTensor<T> b2Base_;
    LocalTensor<T> a1v_;
    LocalTensor<T> b1v_;

    LoadData3DParamsV2<T> load3d_;
    LoadData2DParamsV2 load2dv2_;

    uint32_t coreId_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t hoWo_ = 0;
    uint32_t inputSlotCount_ = 1;
    uint32_t l0aHalfElem_ = 0;
    uint32_t l0bHalfElem_ = 0;
    uint32_t cSlotCount_ = 1;

    uint32_t l0Slot_ = 0;
    uint32_t cSlot_ = 0;
    uint64_t batchIdx_ = 0;
    uint64_t groupIdx_ = 0;

    uint32_t curDinIdx_ = 0;
    uint32_t mStart_ = 0;
    uint32_t cinStart_ = 0;
    uint32_t baseUseM_ = 0;
    uint32_t baseUseN_ = 0;
    uint32_t baseUseK_ = 0;
    uint32_t mCompute_ = 0;
    uint32_t nCompute_ = 0;
    uint32_t blockBaseN_ = 1;
    bool needCompute_ = false;
    int32_t curHoIdx_ = 0;
    uint32_t curHoSize_ = 0;
    uint32_t woExpandLocal_ = 0;

    uint32_t a1SegFirst_ = 0xFFFFFFFFu;
    uint32_t a1Dout_ = 0xFFFFFFFFu;
    uint32_t aSegStartK_ = 0;
    bool a1Valid_ = false;
    uint32_t channelSize_ = 0;

    uint32_t b1SegFirst_ = 0xFFFFFFFFu;
    uint32_t bSegStartK_ = 0;
    uint32_t b1IdKd_ = 0xFFFFFFFFu;
    uint32_t b1IdCin_ = 0xFFFFFFFFu;
    uint32_t b1IdGroup_ = 0xFFFFFFFFu;
    bool b1Valid_ = false;
    uint32_t curCout1Cout0Size_ = 0;
    uint32_t curLoadKbl1_ = 0;
};

} /* namespace */

/* ---------------- Kernel 入口与 launch 封装 ---------------- */

template <typename T, uint32_t LOADB2>
__global__ __aicore__ __cube__ void conv3ddx_entry(GM_ADDR inputSize, GM_ADDR filter, GM_ADDR dedy, GM_ADDR dedx,
                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    (void)inputSize;
    (void)workspace;
    if ASCEND_IS_AIC {
        Conv3DdxTilingData td;
        ReadTilingFromGM(tiling, td);
        if (GetBlockIdx() >= td.coreNum) {
            return;
        }
        Conv3ddxKernel<T, LOADB2> op;
        op.Init(filter, dedy, dedx, td);
        op.Process();
        op.End();
    }
}

extern "C" void LaunchConv3DdxFp16B1(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy,
                                     void* dedx, void* workspace, void* tiling)
{
    conv3ddx_entry<half, CONV3DDX_TPL_TRANSPOSE_AND_REVERSE><<<blocks, nullptr, (aclrtStream)stream>>>(
        (GM_ADDR)inputSize, (GM_ADDR)filter, (GM_ADDR)dedy, (GM_ADDR)dedx, (GM_ADDR)workspace, (GM_ADDR)tiling);
}

extern "C" void LaunchConv3DdxFp16B2(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy,
                                     void* dedx, void* workspace, void* tiling)
{
    conv3ddx_entry<half, CONV3DDX_TPL_REVERSE_ONLY><<<blocks, nullptr, (aclrtStream)stream>>>(
        (GM_ADDR)inputSize, (GM_ADDR)filter, (GM_ADDR)dedy, (GM_ADDR)dedx, (GM_ADDR)workspace, (GM_ADDR)tiling);
}

extern "C" void LaunchConv3DdxFp16B3(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy,
                                     void* dedx, void* workspace, void* tiling)
{
    conv3ddx_entry<half, CONV3DDX_TPL_TRANSPOSE_ONLY><<<blocks, nullptr, (aclrtStream)stream>>>(
        (GM_ADDR)inputSize, (GM_ADDR)filter, (GM_ADDR)dedy, (GM_ADDR)dedx, (GM_ADDR)workspace, (GM_ADDR)tiling);
}

extern "C" void LaunchConv3DdxBf16B1(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy,
                                     void* dedx, void* workspace, void* tiling)
{
    conv3ddx_entry<bfloat16_t, CONV3DDX_TPL_TRANSPOSE_AND_REVERSE><<<blocks, nullptr, (aclrtStream)stream>>>(
        (GM_ADDR)inputSize, (GM_ADDR)filter, (GM_ADDR)dedy, (GM_ADDR)dedx, (GM_ADDR)workspace, (GM_ADDR)tiling);
}

extern "C" void LaunchConv3DdxBf16B2(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy,
                                     void* dedx, void* workspace, void* tiling)
{
    conv3ddx_entry<bfloat16_t, CONV3DDX_TPL_REVERSE_ONLY><<<blocks, nullptr, (aclrtStream)stream>>>(
        (GM_ADDR)inputSize, (GM_ADDR)filter, (GM_ADDR)dedy, (GM_ADDR)dedx, (GM_ADDR)workspace, (GM_ADDR)tiling);
}

extern "C" void LaunchConv3DdxBf16B3(uint32_t blocks, void* stream, void* inputSize, void* filter, void* dedy,
                                     void* dedx, void* workspace, void* tiling)
{
    conv3ddx_entry<bfloat16_t, CONV3DDX_TPL_TRANSPOSE_ONLY><<<blocks, nullptr, (aclrtStream)stream>>>(
        (GM_ADDR)inputSize, (GM_ADDR)filter, (GM_ADDR)dedy, (GM_ADDR)dedx, (GM_ADDR)workspace, (GM_ADDR)tiling);
}
