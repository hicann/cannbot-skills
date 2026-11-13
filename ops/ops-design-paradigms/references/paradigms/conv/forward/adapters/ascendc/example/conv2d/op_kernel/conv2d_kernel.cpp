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
 * \file conv2d_kernel.cpp
 * \brief Conv2D Cube kernel 泛化实现
 *
 * 泛化能力：1D/2D/3D、任意 stride/dilation/非对称 pad、BL1 驻留/NONE 流式、K 分块。
 *
 * 数据流主线（compute-flow.md §2）：
 *   GM → L1 → L0 → MMAD → L0C → FIXPIPE → GM
 *
 * 迭代骨架（compute-flow.md §6.4 泛化版，含 D 轴外层循环 + kd 切片归约）：
 *   for batchIter → for dOutIter → for mBlkIter → for nBlkIter(=1 if bRes)
 *     → for kdIter → for kChunkIter → for kL0Iter
 *   A/B 侧按 (kd, kC) 逐片装载（compute-flow.md §2.2 切片基址/通道步进/通道块偏移公式）。
 *
 * 拉起方式不在本文件范围：`<<<>>>` 内核直调与 aclnn 注册两段式是拉起本算子的两种接入形态，
 * 由上层工作流技能负责（ops/ascendc-direct-invoke-template、ops/ascendc-registry-invoke-template）。
 */

#include "kernel_operator.h"
#include "conv2d_tiling_data.h"

using namespace AscendC;

namespace {

__aicore__ inline uint32_t AlignB(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b * b;
}

__aicore__ inline uint32_t CeilDivLocal(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

/// 膨胀后核尺寸：dk = (K-1)·d + 1
__aicore__ inline uint32_t CalcDkAicore(uint32_t k, uint32_t d)
{
    return (k - 1) * d + 1;
}

/// DN2NZ 搬运（fmap：GM → L1），对应 compute-flow.md §4.2 + §2.1
__aicore__ inline void LoadFmapL1(const Conv2DTilingData& td, const LocalTensor<half>& dst,
                                  const GlobalTensor<half>& src, uint32_t hiLoad, uint32_t gmOff, uint32_t cinThisChunk)
{
    Dn2NzParams p;
    p.dnNum = 1;
    p.nValue = static_cast<uint16_t>(hiLoad * td.spec.wi);
    p.dValue = static_cast<uint32_t>(cinThisChunk);
    p.srcDnMatrixStride = 0;
    p.srcDValue = static_cast<uint32_t>(td.spec.di * td.spec.hi * td.spec.wi); // 3D 含 Di
    p.dstNzC0Stride = static_cast<uint16_t>(hiLoad * td.spec.wi);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 0;
    DataCopy(dst, src[gmOff], p);
}

/// DN2NZ 搬运（weight：GM → L1），对应 compute-flow.md §4.2 + §2.2
/// 按 (kd, kC) 逐片装载：切片基址/通道块偏移由调用方按 §2.2 公式计算
/// 3D 陷阱：kd 片跨通道步进 = Kd·Kh·Kw（不是 Kh·Kw）
/// 目标布局步进按 nBlk（本块 N 交错宽度），不能用全量 Co 步长（§4.2 目标布局推导）
__aicore__ inline void LoadWeightL1(const Conv2DTilingData& td, const LocalTensor<half>& dst,
                                    const GlobalTensor<half>& src, uint32_t coThis, uint32_t cinThis, uint32_t gmOff)
{
    Dn2NzParams p;
    p.dnNum = static_cast<uint16_t>(coThis);
    p.nValue = static_cast<uint16_t>(td.spec.kh * td.spec.kw);
    p.dValue = static_cast<uint32_t>(cinThis); // 通道块真值（尾段 C0 补零）
    p.srcDnMatrixStride = static_cast<uint32_t>(td.spec.ci * td.spec.kd * td.spec.kh * td.spec.kw);
    p.srcDValue = static_cast<uint32_t>(td.spec.kd * td.spec.kh * td.spec.kw); // ⚠️ 3D 乘 Kd
    p.dstNzC0Stride = static_cast<uint16_t>(td.spec.kh * td.spec.kw * td.nBlk);
    p.dstNzNStride = static_cast<uint16_t>(td.nBlk);
    p.dstNzMatrixStride = FRACTAL_K;
    DataCopy(dst, src[gmOff], p);
}

/// 单步 K 归约（MmadStep），对应 sync-implementation.md §2 事件对六步
__aicore__ inline void MmadStep(const Conv2DTilingData& td, LocalTensor<half>& al0, LocalTensor<half>& bl0,
                                LocalTensor<float>& cl0, LocalTensor<half>& al1, LocalTensor<half>& bl1,
                                LocalTensor<float>& btv, uint32_t posK, uint32_t curK, uint32_t curChannel,
                                uint32_t curM, uint32_t posM, uint32_t curN, uint32_t mAlign, uint32_t nAlign,
                                bool isFirst, bool isLast)
{
    constexpr event_t EV_L0 = static_cast<event_t>(0);
    const uint32_t nStep2 = CeilDivLocal(curN, FRACTAL_N);

    WaitFlag<HardEvent::M_MTE1>(EV_L0);

    // LOAD3D（compute-flow.md §4.3 窗口几何 + 块定位）
    Load3DBitModeParam a3;
    a3.SetStrideW(static_cast<uint8_t>(td.spec.sw));
    a3.SetStrideH(static_cast<uint8_t>(td.spec.sh));
    a3.SetFilterW(static_cast<uint8_t>(td.spec.kw));
    a3.SetFilterH(static_cast<uint8_t>(td.spec.kh));
    a3.SetDilationFilterW(static_cast<uint8_t>(td.spec.dw));
    a3.SetDilationFilterH(static_cast<uint8_t>(td.spec.dh));
    a3.SetChannelSize(static_cast<uint16_t>(curChannel));
    a3.SetKExtension(static_cast<uint16_t>(curK));
    a3.SetMExtension(static_cast<uint16_t>(curM));
    a3.SetKStartPt(static_cast<uint16_t>(posK));
    a3.SetMStartPt(static_cast<uint16_t>(posM));
    LoadDataRepeatParamWithStride rp = {0, 1, 0, static_cast<uint16_t>(mAlign / FRACTAL_M)};
    SetLoadDataRepeatWithStride(rp);
    LoadData<TPosition::A2, TPosition::A1, half>(al0, al1, a3);

    // LOAD2D（compute-flow.md §4.4，全按分形数）
    Load2DBitModeParam b2;
    b2.SetMStartPosition(0);
    b2.SetMStep(static_cast<uint16_t>(nStep2));
    b2.SetKStartPosition(posK / FRACTAL_K);
    b2.SetKStep(static_cast<uint16_t>(curK / FRACTAL_K));
    b2.SetSrcStride(static_cast<int32_t>(td.nBlk / FRACTAL_N));
    b2.SetDstStride(static_cast<uint16_t>(nStep2));
    b2.SetIfTranspose(false);
    LoadData<TPosition::B2, TPosition::B1, half>(bl0, bl1, b2);

    SetFlag<HardEvent::MTE1_M>(EV_L0);
    WaitFlag<HardEvent::MTE1_M>(EV_L0);

    // MMAD（compute-flow.md §4.5）
    MmadParams mp;
    mp.m = static_cast<uint16_t>(mAlign);
    mp.n = static_cast<uint16_t>(nAlign);
    mp.k = static_cast<uint16_t>(curK);
    mp.cmatrixInitVal = isFirst && btv.IsEmpty();
    mp.cmatrixSource = isFirst && !btv.IsEmpty();
    mp.unitFlag = isLast ? 3 : 2;
    Mmad<float, half, half>(cl0, al0, bl0, mp);

    SetFlag<HardEvent::M_MTE1>(EV_L0);
}

/// FIXPIPE 搬出（compute-flow.md §4.6）
__aicore__ inline void FixpipeOut(const Conv2DTilingData& td, const GlobalTensor<half>& yGm,
                                  const LocalTensor<float>& cl0, uint32_t curM, uint32_t curN, uint32_t mAlign,
                                  uint32_t nAlign, uint32_t yOff)
{
    FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> fp;
    fp.nSize = static_cast<uint16_t>(curN);
    fp.mSize = static_cast<uint16_t>(curM);
    fp.srcStride = static_cast<uint16_t>(mAlign);
    fp.dstStride = td.dout * td.ho * td.wo;
    fp.params.dnNum = 1;
    // 源侧相邻 dn 矩阵步进：对齐 L0C 跨度（mAlign × nAlign）÷ N0（§4.6 N0 元素组口径）
    fp.params.srcNzMatrixStride = static_cast<uint16_t>(mAlign * nAlign / FRACTAL_N);
    fp.params.dstDnMatrixStride = td.ho * td.wo;
    fp.params.srcNzC0Stride = 1;
    fp.quantPre = QuantMode_t::F322F16;
    fp.deqScalar = 0x3F800000;
    fp.unitFlag = 3;
    Fixpipe<half, float, CFG_COLUMN_MAJOR>(yGm[yOff], cl0, fp);
}

/// Kernel 主体：泛化迭代骨架（batch × dout × mBlk × nBlk × kd × kChunk × kL0）
__aicore__ inline void Conv2dImpl(const Conv2DTilingData& td, const GlobalTensor<half>& xGm,
                                  const GlobalTensor<half>& wGm, const GlobalTensor<half>& bGm,
                                  const GlobalTensor<half>& yGm, uint32_t blockIdx)
{
    // ── 多核反解（compute-flow.md §5.2：最内维取模）──
    const uint32_t mIdx = (td.mDim > 1) ? (blockIdx % td.mDim) : 0;
    const uint32_t doIdx = (td.doDim > 1) ? ((blockIdx / td.mDim) % td.doDim) : 0;
    const uint32_t bIdx = (td.batchDim > 1) ? (blockIdx / (td.mDim * td.doDim)) : 0;

    // ⚠️ 空尾核判界（compute-flow.md §5.3）：CeilDiv 切分存在 mStart ≥ M 的空尾核，
    //    必须先判界再算余量——无符号减法下溢会绕过 ==0 检查导致越界迭代
    const uint32_t mStart = mIdx * td.singleCoreM;
    if (mStart >= td.m) {
        return;
    }
    uint32_t curMTotal = (mStart + td.singleCoreM <= td.m) ? td.singleCoreM : (td.m - mStart);

    const uint32_t dStart = doIdx * td.singleCoreDo;
    const uint32_t batchStart = bIdx * td.singleCoreBatch;

    const uint32_t dkH = CalcDkAicore(td.spec.kh, td.spec.dh);
    const uint32_t khkw = td.spec.kh * td.spec.kw;
    const uint32_t srcDv = td.spec.di * td.spec.hi * td.spec.wi; // 单通道 D/H/W 平面跨度
    const uint32_t batchDv = td.spec.ci * srcDv;                 // 跨 batch 步进 = Ci·Di·Hi·Wi
    const uint32_t kChunkLoop = CeilDivLocal(td.k, td.kChunk);

    // ── TPipe 与缓冲声明 ──
    TPipe pipe;
    TQue<TPosition::A1, 1> queAl1;
    pipe.InitBuffer(queAl1, 1, td.al1Size);
    TQue<TPosition::B1, 1> queBl1;
    pipe.InitBuffer(queBl1, 1, td.bl1Size);
    TBuf<TPosition::A2> al0Buf;
    pipe.InitBuffer(al0Buf, L0A_SIZE_BYTE);
    TBuf<TPosition::B2> bl0Buf;
    pipe.InitBuffer(bl0Buf, L0B_SIZE_BYTE);
    TQue<TPosition::CO1, 2> queCl0;
    pipe.InitBuffer(queCl0, 1, L0C_SIZE_BYTE);
    TQue<TPosition::A1, 1> queBias;
    TQue<TPosition::C2, 1> queBt;
    pipe.InitBuffer(queBias, 1, td.coAlign * sizeof(half));
    pipe.InitBuffer(queBt, 1, AlignB(td.nBlk * sizeof(float), 4096));

    auto al0 = al0Buf.Get<half>();
    auto bl0 = bl0Buf.Get<half>();

    // ── bias 全量装载（bias-bt-path.md）──
    auto biasL1 = queBias.AllocTensor<half>();
    DataCopy(biasL1, bGm, DataCopyParams(1, td.coAlign * 2 / 32, 0, 0));
    queBias.EnQue(biasL1);
    auto biasL1v = queBias.DeQue<half>();

    // ── B 侧驻留判定（bResident=1 时全量装载一次，跨全部 M 块；仅 2D/1D，3D 逐 kd 片装载）──
    LocalTensor<half> bl1Res;
    if (td.bResident == 1) {
        auto bl1Full = queBl1.AllocTensor<half>();
        LoadWeightL1(td, bl1Full, wGm, td.spec.co, td.ciAlign, 0);
        queBl1.EnQue(bl1Full);
        bl1Res = queBl1.DeQue<half>();
    }

    // ── 外层迭代：batch → dout → M 块 → N 块 → kd 切片 → K 分块 → K 步进 ──
    const uint32_t batchEnd =
        (batchStart + td.singleCoreBatch < td.spec.n) ? (batchStart + td.singleCoreBatch) : td.spec.n;
    for (uint32_t b = batchStart; b < batchEnd; ++b) {
        const uint32_t dEnd = (dStart + td.singleCoreDo < td.dout) ? (dStart + td.singleCoreDo) : td.dout;
        for (uint32_t dOut = dStart; dOut < dEnd; ++dOut) {
            for (uint32_t mOff = 0; mOff < curMTotal; mOff += td.mBlk) {
                const uint32_t tileM = (mOff + td.mBlk <= curMTotal) ? td.mBlk : (curMTotal - mOff);
                if (tileM == 0) {
                    continue;
                }
                const uint32_t mAlign = AlignB(tileM, FRACTAL_M);
                const uint32_t mStartTile = mStart + mOff;

                // 行推理（compute-flow.md §6.3）
                const uint32_t hoStart = mStartTile / td.wo;
                const uint32_t hoEnd = CeilDivLocal(mStartTile + tileM, td.wo);
                uint32_t hiLoad = (hoEnd - hoStart - 1) * td.spec.sh + dkH;

                int32_t hiIdx = static_cast<int32_t>(hoStart * td.spec.sh) - static_cast<int32_t>(td.spec.padT);
                uint32_t padTopL1 = 0;
                if (hiIdx < 0) {
                    padTopL1 = static_cast<uint32_t>(-hiIdx);
                    hiIdx = 0;
                    hiLoad -= padTopL1;
                }
                // ⚠️ GM 下边界钳制（compute-flow.md §6.3）：装载行数不得超过 [hiIdx, Hi)。
                //    越界行属底部 pad，由 FMatrix padBottom=255 虚拟填零；
                //    不钳制会从 GM 越界读进 AL1 且把越界数据当作有效图像行参与计算
                const uint32_t hiAvail = (static_cast<uint32_t>(hiIdx) < td.spec.hi) ? (td.spec.hi - hiIdx) : 0;
                hiLoad = (hiLoad < hiAvail) ? hiLoad : hiAvail;

                // N 块迭代（nBlk 恒 = coAlign 时退化为单次）
                const uint32_t nLoop = CeilDivLocal(td.coAlign, td.nBlk);
                for (uint32_t nI = 0; nI < nLoop; ++nI) {
                    const uint32_t coStart = nI * td.nBlk;
                    uint32_t curN = td.spec.co - coStart;
                    curN = (curN > td.nBlk) ? td.nBlk : curN;
                    if (curN == 0) {
                        continue;
                    }
                    const uint32_t nAlign = AlignB(curN, FRACTAL_N);

                    // BT 装载（bias-bt-path.md 步骤 4）
                    auto bt = queBt.AllocTensor<float>();
                    DataCopy(bt, biasL1v[coStart], DataCopyParams(1, AlignB(curN, FRACTAL_N) * 2 / 32, 0, 0));
                    queBt.EnQue(bt);
                    auto btv = queBt.DeQue<float>();

                    // K 归约（compute-flow.md §6.4/§6.6）：kd 切片 × K 分块 × L0 步进累加进同一份 L0C
                    auto cl0 = queCl0.AllocTensor<float>();
                    bool firstMmad = true;
                    for (uint32_t kdI = 0; kdI < td.spec.kd; ++kdI) {
                        // D 向输入平面（3D：每个 kd 取一个输入平面；平面整体越界走全零装载）
                        const int32_t diIdx = static_cast<int32_t>(dOut * td.spec.sd + kdI * td.spec.dd) -
                                              static_cast<int32_t>(td.spec.padDH);
                        const bool dPad = (diIdx < 0) || (diIdx >= static_cast<int32_t>(td.spec.di));

                        for (uint32_t kC = 0; kC < kChunkLoop; ++kC) {
                            uint32_t kRem = td.k - kC * td.kChunk;
                            kRem = (kRem > td.kChunk) ? td.kChunk : kRem;
                            const uint32_t cinThis = kRem / khkw; // 本片通道数（恒为 C0 整数倍）
                            const uint32_t kTailLoop = CeilDivLocal(kRem, td.kL0);

                            // A 侧装载（按 (kd, kC) 逐片；D 向 pad 或 H 向全 pad 时全零装载）
                            auto al1 = queAl1.AllocTensor<half>();
                            if (dPad || hiLoad == 0 || td.spec.wi == 0) {
                                InitConstValueParams<half> ivp;
                                ivp.repeatTimes = 1;
                                ivp.blockNum = static_cast<uint16_t>(td.al1Size / 32);
                                ivp.dstGap = 0;
                                ivp.initValue = static_cast<half>(0);
                                InitConstValue(al1, ivp);
                                uint8_t padMax[4] = {255, 255, 255, 255};
                                Load3DSetFMatrixCal(1, 1, padMax);
                            } else {
                                // 含 batch 偏移：x 为 [N][Ci][Di][Hi][Wi]，跨 batch 步进 = Ci·Di·Hi·Wi
                                const uint32_t gmXOff = b * batchDv + static_cast<uint32_t>(diIdx) * srcDv +
                                                        static_cast<uint32_t>(hiIdx) * td.spec.wi;
                                LoadFmapL1(td, al1, xGm, hiLoad, gmXOff, cinThis);
                                uint8_t padList[4] = {static_cast<uint8_t>(td.spec.padL),
                                                      static_cast<uint8_t>(td.spec.padR),
                                                      static_cast<uint8_t>(padTopL1), static_cast<uint8_t>(255)};
                                Load3DSetFMatrixCal(static_cast<uint16_t>(hiLoad), static_cast<uint16_t>(td.spec.wi),
                                                    padList);
                            }
                            queAl1.EnQue(al1);
                            auto al1v = queAl1.DeQue<half>();

                            // B 侧流式装载（bResident=1 时用驻留份；3D 不驻留，按 (kd, kC) 逐片）
                            // 切片基址 = coStart·(Ci·Kd·KhKw) + kd·KhKw + kC·cinThis·(Kd·KhKw)（§2.2）
                            LocalTensor<half> bl1v;
                            if (td.bResident == 0) {
                                auto bl1 = queBl1.AllocTensor<half>();
                                const uint32_t wOff = coStart * td.spec.ci * td.spec.kd * khkw + kdI * khkw +
                                                      kC * cinThis * td.spec.kd * khkw;
                                LoadWeightL1(td, bl1, wGm, curN, cinThis, wOff);
                                queBl1.EnQue(bl1);
                                bl1v = queBl1.DeQue<half>();
                            } else {
                                bl1v = bl1Res;
                            }

                            for (uint32_t kI = 0; kI < kTailLoop; ++kI) {
                                const uint32_t posK = kI * td.kL0; // AL1/BL1 重装后归零（§4.3）
                                uint32_t curK = (kRem > posK) ? (kRem - posK) : 0;
                                curK = (curK > td.kL0) ? td.kL0 : curK;
                                if (curK == 0) {
                                    continue;
                                }
                                const bool isLast =
                                    (kdI == td.spec.kd - 1) && (kC == kChunkLoop - 1) && (kI == kTailLoop - 1);

                                MmadStep(td, al0, bl0, cl0, al1v, bl1v, btv, posK, curK, cinThis, tileM,
                                         mStartTile % td.wo, curN, mAlign, nAlign, firstMmad, isLast);
                                firstMmad = false;
                            }
                            queAl1.FreeTensor(al1v);
                            if (td.bResident == 0) {
                                queBl1.FreeTensor(bl1v);
                            }
                        }
                    }
                    queBt.FreeTensor(btv);

                    // FIXPIPE 搬出
                    queCl0.EnQue(cl0);
                    auto cl0v = queCl0.DeQue<float>();
                    const uint32_t yOff = ((b * td.spec.co + coStart) * td.dout + dOut) * td.ho * td.wo + mStartTile;
                    FixpipeOut(td, yGm, cl0v, tileM, curN, mAlign, nAlign, yOff);
                    queCl0.FreeTensor(cl0v);
                }
            }
        }
    }

    // 释放驻留资源
    if (td.bResident == 1) {
        queBl1.FreeTensor(bl1Res);
    }
    queBias.FreeTensor(biasL1v);
}

} // anonymous namespace

__global__ __aicore__ void conv2d(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, const Conv2DTilingData td)
{
    if (GetBlockIdx() >= td.blockDim) {
        return;
    }
    if ASCEND_IS_AIC {
        GlobalTensor<half> xGm, wGm, bGm, yGm;
        xGm.SetGlobalBuffer((__gm__ half*)x);
        wGm.SetGlobalBuffer((__gm__ half*)weight);
        bGm.SetGlobalBuffer((__gm__ half*)bias);
        yGm.SetGlobalBuffer((__gm__ half*)y);
        Conv2dImpl(td, xGm, wGm, bGm, yGm, GetBlockIdx());
    }
}
