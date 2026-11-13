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
 * \file conv3ddw_kernel.h
 * \brief 普通整窗 MN_StreamK：A=dedy，B=x 正向窗口，C=FP32 dedw。
 *
 * 一个任务固定 co 块、ci 块、kd，归约全部 batch/dout/ho/wo。
 * L1A/B 保留父段，L0A/B 双槽轮转，L0C 在精度段结束后原子累加写回。
 * 初始化协议：所有 AIV 清零实际输出 -> MODE0 汇合 -> MODE2 通知 AIC。
 */

#ifndef CONV3DDW_KERNEL_H
#define CONV3DDW_KERNEL_H

#include "kernel_operator.h"
#include "utils/init_global_memory.h"
#include "conv3ddw_launch.h"

namespace Conv3Ddw {
using namespace AscendC;

__aicore__ inline uint32_t CeilDiv(uint64_t x, uint64_t y)
{
    return (x + y - 1) / y;
}
__aicore__ inline uint32_t Align16(uint32_t x)
{
    return (x + 15u) / 16u * 16u;
}
__aicore__ inline uint32_t Min(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}

__aicore__ inline void ReadTiling(GM_ADDR gm, Conv3DdwTilingData& td)
{
    auto src = reinterpret_cast<__gm__ const uint8_t*>(gm);
    auto dst = reinterpret_cast<uint8_t*>(&td);
    for (uint32_t i = 0; i < sizeof(td); ++i)
        dst[i] = src[i];
}

__aicore__ inline void InitializeOutput(GM_ADDR dedw, const Conv3DdwTilingData& td)
{
    constexpr uint16_t initFlag = 9;
    if ASCEND_IS_AIV {
        TPipe pipe;
        GlobalTensor<float> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dedw));
        const uint64_t size = uint64_t(td.cout) * td.cin1G * td.dk * td.hk * td.wk;
        const uint64_t perCore = uint64_t(CeilDiv(CeilDiv(size, GetBlockNum()), 32)) * 32;
        const uint64_t start = uint64_t(GetBlockIdx()) * perCore;
        if (start < size) {
            const uint64_t length = size - start < perCore ? size - start : perCore;
            auto region = output[start];
            Fill<float>(region, length, 0.0f);
        }
        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_AIV_FLAG);
        CrossCoreWaitFlag(SYNC_AIV_FLAG);
        CrossCoreSetFlag<2, PIPE_MTE3>(initFlag);
        pipe.Destroy();
    }
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<2, PIPE_FIX>(initFlag);
    }
}

template <typename T>
class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dedy, GM_ADDR dedw, const Conv3DdwTilingData& td)
    {
        td_ = td;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        dedy_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dedy));
        dedw_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dedw));
        pipe_.InitBuffer(a1Queue_, td.al1Pbuffer, uint64_t(td.al1Bound) * sizeof(T));
        pipe_.InitBuffer(b1Queue_, td.bl1Pbuffer, uint64_t(td.bl1Bound) * sizeof(T));
        a2Slot_ = TOTAL_L0A_SIZE / td.al0Pbuffer / sizeof(T);
        b2Slot_ = TOTAL_L0B_SIZE / td.bl0Pbuffer / sizeof(T);
        pipe_.InitBuffer(a2Buffer_, uint64_t(a2Slot_) * sizeof(T) * td.al0Pbuffer);
        pipe_.InitBuffer(b2Buffer_, uint64_t(b2Slot_) * sizeof(T) * td.bl0Pbuffer);
        pipe_.InitBuffer(c0Ping_, 1, uint64_t(td.baseM) * td.baseN * sizeof(float));
        if (td.cl0Pbuffer == 2)
            pipe_.InitBuffer(c0Pong_, 1, uint64_t(td.baseM) * td.baseN * sizeof(float));
        a2_ = a2Buffer_.template Get<T>();
        b2_ = b2Buffer_.template Get<T>();
        // 当前 SDK 的 TPipe 初始化固定 M_MTE1 事件，生成类只消费和归还轮内事件。
    }

    __aicore__ inline void Process()
    {
        const uint32_t mCount = CeilDiv(td_.cout1G, td_.singleCoreM);
        const uint32_t nCount = CeilDiv(td_.cin1G, td_.singleCoreCin);
        const uint64_t total = uint64_t(mCount) * nCount * td_.dk * td_.realGroup;
        for (uint64_t task = GetBlockIdx(); task < total; task += td_.usedCoreNum) {
            uint64_t index = task;
            groupIdx_ = index % td_.realGroup;
            index /= td_.realGroup;
            kdIdx_ = index % td_.dk;
            index /= td_.dk;
            ciStart_ = (index % nCount) * td_.singleCoreCin;
            index /= nCount;
            coStart_ = index * td_.singleCoreM;
            mUse_ = Min(td_.singleCoreM, td_.cout1G - coStart_);
            ciUse_ = Min(td_.singleCoreCin, td_.cin1G - ciStart_);
            nUse_ = Align16(ciUse_) * td_.hk * td_.wk;
            mCompute_ = mUse_ == 1 ? 16 : mUse_;
            ProcessTask();
        }
    }

    __aicore__ inline void End()
    {
        ReleaseParents();
        // 固定事件的末次等待与释放由 TPipe::Destroy 完成。
        pipe_.Destroy();
    }

private:
    Conv3DdwTilingData td_{};
    TPipe pipe_;
    TQue<TPosition::A1, 2> a1Queue_;
    TQue<TPosition::B1, 2> b1Queue_;
    TQue<TPosition::CO1, 1> c0Ping_, c0Pong_;
    TBuf<TPosition::A2> a2Buffer_;
    TBuf<TPosition::B2> b2Buffer_;
    GlobalTensor<T> x_, dedy_;
    GlobalTensor<float> dedw_;
    LocalTensor<T> a1_, b1_, a2_, b2_;
    uint32_t a2Slot_ = 0, b2Slot_ = 0, l0Slot_ = 0, cSlot_ = 0;
    uint32_t groupIdx_ = 0, kdIdx_ = 0, coStart_ = 0, ciStart_ = 0;
    uint32_t mUse_ = 0, mCompute_ = 0, ciUse_ = 0, nUse_ = 0;
    uint32_t aParentStart_ = 0, bParentStart_ = 0, aParentSpan_ = 0;
    uint32_t bParentH_ = 0, bParentPadUp_ = 0;
    bool aValid_ = false, bValid_ = false, bHasData_ = false;

    __aicore__ inline uint64_t FeatureOffset(uint32_t batch, uint32_t c, uint32_t d, uint32_t h, uint32_t w,
                                             bool dy) const
    {
        const uint64_t channels = dy ? td_.cout : td_.cin;
        const uint64_t depth = dy ? td_.dout : td_.di;
        const uint64_t height = dy ? td_.ho : td_.hi;
        const uint64_t width = dy ? td_.wo : td_.wi;
        if (td_.xFormat == CONV3DDW_NDHWC)
            return ((((uint64_t(batch) * depth + d) * height + h) * width + w) * channels + c);
        return ((((uint64_t(batch) * channels + c) * depth + d) * height + h) * width + w);
    }

    __aicore__ inline uint64_t OutputOffset() const
    {
        const uint64_t co = uint64_t(groupIdx_) * td_.cout1G + coStart_;
        if (td_.yFormat == CONV3DDW_NDHWC)
            return ((co * td_.dk + kdIdx_) * td_.hk * td_.wk) * td_.cin1G + ciStart_;
        if (td_.yFormat == CONV3DDW_DHWCN)
            return uint64_t(kdIdx_) * td_.hk * td_.wk * td_.cin1G * td_.cout + uint64_t(ciStart_) * td_.cout + co;
        return (co * td_.cin1G + ciStart_) * td_.dk * td_.hk * td_.wk + uint64_t(kdIdx_) * td_.hk * td_.wk;
    }

    __aicore__ inline void ReleaseParents()
    {
        if (aValid_) {
            a1Queue_.FreeTensor(a1_);
            aValid_ = false;
        }
        if (bValid_) {
            b1Queue_.FreeTensor(b1_);
            bValid_ = false;
        }
    }

    /* MTE2：dedy 父段排为 (M1,K1,K0,M0)，空间和通道尾部由搬运补齐。 */
    __aicore__ inline void LoadToA1(uint32_t batch, uint32_t od, uint32_t hoStart, uint32_t start, uint32_t length)
    {
        if (aValid_)
            a1Queue_.FreeTensor(a1_);
        a1_ = a1Queue_.template AllocTensor<T>();
        const uint32_t co = groupIdx_ * td_.cout1G + coStart_;
        const uint64_t src = FeatureOffset(batch, co, od, hoStart, 0, true) +
                             uint64_t(start) * (td_.xFormat == CONV3DDW_NDHWC ? td_.cout : 1);
        aParentSpan_ = Align16(length);
        aParentStart_ = start;
        if (td_.xFormat == CONV3DDW_NDHWC) {
            Nd2NzParams p{};
            p.ndNum = 1;
            p.nValue = length;
            p.dValue = mUse_;
            p.srcDValue = td_.cout;
            p.dstNzC0Stride = aParentSpan_;
            p.dstNzNStride = 1;
            p.dstNzMatrixStride = 1;
            DataCopy(a1_, dedy_[src], p);
        } else {
            Dn2NzParams p{};
            p.dnNum = 1;
            p.nValue = length;
            p.dValue = mUse_;
            p.srcDValue = uint64_t(td_.dout) * td_.ho * td_.wo;
            p.dstNzC0Stride = aParentSpan_;
            p.dstNzNStride = 1;
            p.dstNzMatrixStride = 1;
            DataCopy(a1_, dedy_[src], p);
        }
        a1Queue_.EnQue(a1_);
        a1_ = a1Queue_.template DeQue<T>();
        aValid_ = true;
    }

    __aicore__ inline uint32_t RowsToCopy(uint32_t length) const
    {
        if (length > td_.wo) {
            uint32_t rows = CeilDiv(length, td_.wo);
            if (length % td_.wo != 0 && (uint64_t(length) * 2) % td_.wo != 0)
                ++rows;
            return rows;
        }
        return td_.wo % length == 0 ? 1 : 2;
    }

    /* MTE2：裁到真实 x 行带，保留正向 pad；深度位置由外层精度段决定。 */
    __aicore__ inline void LoadToB1(uint32_t batch, uint32_t id, uint32_t hoStart, uint32_t start, uint32_t length)
    {
        if (bValid_) {
            b1Queue_.FreeTensor(b1_);
            bValid_ = false;
        }
        bParentStart_ = start;
        const int64_t top = int64_t(hoStart + start / td_.wo) * td_.strideH - td_.padUp;
        const int64_t rows = int64_t(RowsToCopy(length) - 1) * td_.strideH + int64_t(td_.hk - 1) * td_.dilationH + 1;
        const int64_t begin = top < 0 ? 0 : top;
        const int64_t end = top + rows > td_.hi ? td_.hi : top + rows;
        bHasData_ = begin < end;
        if (!bHasData_)
            return;
        bParentH_ = end - begin;
        bParentPadUp_ = top < 0 ? -top : 0;
        b1_ = b1Queue_.template AllocTensor<T>();
        const uint32_t ci = groupIdx_ * td_.cin1G + ciStart_;
        const uint64_t src = FeatureOffset(batch, ci, id, begin, 0, false);
        const uint32_t span = bParentH_ * td_.wi;
        if (td_.xFormat == CONV3DDW_NDHWC) {
            Nd2NzParams p{};
            p.ndNum = 1;
            p.nValue = span;
            p.dValue = ciUse_;
            p.srcDValue = td_.cin;
            p.dstNzC0Stride = span;
            p.dstNzNStride = 1;
            p.dstNzMatrixStride = uint64_t(Align16(ciUse_)) * span;
            DataCopy(b1_, x_[src], p);
        } else {
            Dn2NzParams p{};
            p.dnNum = 1;
            p.nValue = span;
            p.dValue = ciUse_;
            p.srcDValue = uint64_t(td_.di) * td_.hi * td_.wi;
            p.dstNzC0Stride = span;
            p.dstNzNStride = 1;
            p.dstNzMatrixStride = uint64_t(Align16(ciUse_)) * span;
            DataCopy(b1_, x_[src], p);
        }
        b1Queue_.EnQue(b1_);
        b1_ = b1Queue_.template DeQue<T>();
        bValid_ = true;
    }

    /* MTE1：A 路转置分形；B 路以正向 stride/dilation 提取窗口，K 顺序配对。 */
    __aicore__ inline void CopyInMte1(uint32_t kStart, uint32_t kUse)
    {
        LoadData3DParamsV2<T> b{};
        b.filterW = td_.wk;
        b.filterH = td_.hk;
        b.strideW = td_.strideW;
        b.strideH = td_.strideH;
        b.dilationFilterW = td_.dilationW;
        b.dilationFilterH = td_.dilationH;
        b.mStartPt = bParentStart_ % td_.wo + kStart - bParentStart_;
        b.mExtension = kUse;
        b.kStartPt = 0;
        b.kExtension = nUse_;
        b.channelSize = Align16(ciUse_);
        b.l1H = bParentH_;
        b.l1W = td_.wi;
        b.padList[0] = td_.padLeft;
        b.padList[1] = td_.padRight;
        b.padList[2] = bParentPadUp_;
        b.padList[3] = 255;
        b.enTranspose = false;
        b.enSmallK = false;
        b.fMatrixCtrl = false;
        b.padValue = static_cast<T>(0);
        auto bDest = b2_[l0Slot_ * b2Slot_];
        LoadDataRepeatParamWithStride repeat{0, 1, 0, static_cast<uint16_t>(CeilDiv(nUse_, 16))};
        SetLoadDataRepeatWithStride(repeat);
        LoadDataWithStride(bDest, b1_, b);
        LoadData2DParamsV2 a{};
        a.mStep = CeilDiv(kUse, 16);
        a.kStep = CeilDiv(mCompute_, 16);
        a.srcStride = aParentSpan_ / 16;
        a.dstStride = a.kStep;
        a.ifTranspose = true;
        const uint64_t aOffset = uint64_t(kStart - aParentStart_) * 16;
        auto aDest = a2_[l0Slot_ * a2Slot_];
        LoadData(aDest, a1_[aOffset], a);
    }

    __aicore__ inline void ComputeDepth(LocalTensor<float>& c, uint32_t batch, uint32_t od, uint32_t id,
                                        uint32_t hoStart, uint32_t hoUse, bool& first)
    {
        const uint32_t kLength = hoUse * td_.wo;
        const uint32_t aCapacity = td_.stepKa * td_.baseK;
        const uint32_t bCapacity = td_.stepKb * td_.baseK;
        ReleaseParents();
        for (uint32_t kStart = 0; kStart < kLength; kStart += td_.baseK) {
            if (kStart % aCapacity == 0)
                LoadToA1(batch, od, hoStart, kStart, Min(aCapacity, kLength - kStart));
            if (kStart % bCapacity == 0)
                LoadToB1(batch, id, hoStart, kStart, Min(bCapacity, kLength - kStart));
            if (!bHasData_)
                continue;
            const uint32_t kUse = Min(td_.baseK, kLength - kStart);
            WaitFlag<HardEvent::M_MTE1>(l0Slot_);
            CopyInMte1(kStart, kUse);
            SetFlag<HardEvent::MTE1_M>(l0Slot_);
            WaitFlag<HardEvent::MTE1_M>(l0Slot_);
            MmadParams p{};
            p.m = mCompute_;
            p.n = nUse_;
            p.k = kUse;
            p.cmatrixInitVal = first;
            p.cmatrixSource = 0;
            p.unitFlag = 0;
            p.kDirectionAlign = false;
            Mmad(c, a2_[l0Slot_ * a2Slot_], b2_[l0Slot_ * b2Slot_], p);
            first = false;
            if (uint64_t(mCompute_) * nUse_ < 2560)
                PipeBarrier<PIPE_M>();
            SetFlag<HardEvent::M_MTE1>(l0Slot_);
            l0Slot_ ^= 1;
        }
        ReleaseParents();
    }

    /* Fixpipe：源为 (N1,M,N0)，恢复三种 dedw 布局；整窗与 pointwise 分别构造。 */
    __aicore__ inline void CopyOut(LocalTensor<float>& c)
    {
        const uint32_t hw = td_.hk * td_.wk;
        const uint32_t mAligned = Align16(mCompute_);
        const uint64_t destination = OutputOffset();
        SetAtomicAdd<float>();
        if (td_.yFormat == CONV3DDW_NDHWC || (td_.yFormat == CONV3DDW_NCDHW && td_.dk * hw == 1)) {
            FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> p{};
            p.mSize = mUse_;
            p.nSize = ciUse_;
            p.srcStride = hw * mAligned;
            p.dstStride = td_.cin1G * td_.dk * hw;
            p.params.ndNum = hw;
            p.params.srcNdStride = mAligned;
            p.params.dstNdStride = td_.cin1G;
            if (td_.dk * hw == 1)
                p.dstStride = td_.cin1G;
            p.quantPre = QuantMode_t::NoQuant;
            p.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(dedw_[destination], c, p);
        } else {
            FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> p{};
            if (td_.yFormat == CONV3DDW_NCDHW) {
                p.mSize = hw;
                p.nSize = ciUse_;
                p.srcStride = hw * mAligned;
                p.dstStride = td_.dk * hw;
                p.params.dnNum = mUse_;
                p.params.srcNzMatrixStride = 1;
                p.params.srcNzC0Stride = mAligned;
                p.params.dstDnMatrixStride = uint64_t(td_.cin1G) * td_.dk * hw;
            } else {
                p.mSize = mUse_;
                p.nSize = ciUse_;
                p.srcStride = hw * mAligned;
                p.dstStride = td_.cout;
                p.params.dnNum = hw;
                p.params.srcNzMatrixStride = mAligned;
                p.params.srcNzC0Stride = 1;
                p.params.dstDnMatrixStride = uint64_t(td_.cin1G) * td_.cout;
            }
            p.quantPre = QuantMode_t::NoQuant;
            p.unitFlag = 0;
            Fixpipe<float, float, CFG_COLUMN_MAJOR>(dedw_[destination], c, p);
        }
        SetAtomicNone();
    }

    __aicore__ inline void ProcessTask()
    {
        const uint64_t ndLength = uint64_t(td_.batch) * td_.dout;
        const uint32_t ndh = CeilDiv(65536, td_.wo);
        const uint32_t segmentH = Min(ndh, td_.ho);
        const uint32_t segmentND = Min(CeilDiv(ndh, segmentH), ndLength);
        for (uint64_t ndStart = 0; ndStart < ndLength; ndStart += segmentND) {
            const uint32_t ndUse = Min(segmentND, ndLength - ndStart);
            for (uint32_t hoStart = 0; hoStart < td_.ho; hoStart += segmentH) {
                const uint32_t hoUse = Min(segmentH, td_.ho - hoStart);
                auto& queue = cSlot_ == 0 ? c0Ping_ : c0Pong_;
                auto c = queue.template AllocTensor<float>();
                bool first = true;
                for (uint32_t nd = 0; nd < ndUse; ++nd) {
                    const uint64_t position = ndStart + nd;
                    const uint32_t batch = position / td_.dout;
                    const uint32_t od = position % td_.dout;
                    const int64_t id = int64_t(od) * td_.strideD + int64_t(kdIdx_) * td_.dilationD - td_.padFront;
                    if (id < 0 || id >= td_.di)
                        continue;
                    ComputeDepth(c, batch, od, id, hoStart, hoUse, first);
                }
                if (!first) {
                    queue.EnQue(c);
                    c = queue.template DeQue<float>();
                    CopyOut(c);
                }
                queue.FreeTensor(c);
                if (td_.cl0Pbuffer == 2)
                    cSlot_ ^= 1;
            }
        }
    }
};
} // namespace Conv3Ddw
#endif
