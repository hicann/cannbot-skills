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
 * \file regbase_matmul.h
 * \brief task#34 close-port of the CANN arch35 regbase matmul micro-op stack.
 *
 * This is a TRIMMED, self-contained close-port of
 *   ~/workspace/cann/ops-transformer/attention/common/op_kernel/matmul.h
 * keeping ONLY the fp16/bf16 path primitives the FA-A5 BSH skeleton needs:
 *   - MMParam / MakeMMParam / ABLayout
 *   - C310 ABLayout-templated LoadDataToL0A<T,AL> / LoadDataToL0B<T,BL>
 *     (matmul.h L569-654, the regbase loads that MatmulFull/K/N call)
 *   - MatmulFull / MatmulK / MatmulN / MatmulBase
 *
 * The mx-fp8 / fp4 / int8 sub-paths (LoadDataToL0AMx/BMx and their
 * `if constexpr (IsSameType<..., mx_fp8_e4m3_t>)` guards) are intentionally
 * DROPPED: `mx_fp8_e4m3_t` is not declared on the A5 CANN (probed 2026-06-01),
 * and the fp16 path never instantiates them. Dropping them is a legitimate
 * close-port adaptation, not an algorithm change.
 *
 * Compiles on A5 (Ascend950PR_9579, dav-c310, __CCE_AICORE__==310): the regbase
 * load path is guarded by `#if ((__CCE_AICORE__==310)||(__DAV_310R6__)||(__NPU_ARCH__==5102))`.
 */
#ifndef REGBASE_MATMUL_H
#define REGBASE_MATMUL_H

#include "regbase_buffers_policy.h"
using namespace AscendC;

namespace fa_base_matmul {

constexpr uint32_t UNITFLAG_DISABLE = 0;
constexpr uint32_t UNITFLAG_ENABLE = 2;
constexpr uint32_t UNITFLAG_EN_OUTER_LAST = 3;
static constexpr uint32_t FP16_ONE_FRACTAL_ELEMENT = 16;
static constexpr uint32_t ONE_FRACTAL_H_ELEMENT = 16;
static constexpr uint32_t ONE_FRACTAL_W_BYTE = 32;
static constexpr uint32_t LOAD3D_L1W_SIZE = 16;
static constexpr uint8_t LOAD3D_STRIDE_W = 1;
static constexpr uint8_t LOAD3D_STRIDE_H = 1;
static constexpr uint8_t LOAD3D_FILTER_W = 1;
static constexpr uint8_t LOAD3D_FILTER_H = 1;
static constexpr uint8_t LOAD3D_DILA_FILTER_W = 1;
static constexpr uint8_t LOAD3D_DILA_FILTER_H = 1;
static constexpr uint32_t K_STEP_ALIGN_BASE = 2;

// VERBATIM close-port of CANN arch35 matmul.h:88-99 (GetBlockNum, C310/__NPU_ARCH__==5102
// path) — the K-direction fractal-block count used by LoadDataToL0A/B kStep.
template <typename T>
__aicore__ inline uint32_t GetBlockNum(uint32_t size) {
    if constexpr (IsSameType<T, float>::value) {
        return ((size + 7) >> 3 << 3) >> 3;
    } else {
        return ((size + 15) >> 4 << 4) >> 4;
    }
}

// CeilAlign(x, base) — round x up to a multiple of base. Used only by the fp32
// (IsSameType<T,float>) transpose sub-path; fp16/bf16 never hits it but it must compile.
__aicore__ inline uint32_t CeilAlign(uint32_t x, uint32_t base) {
    if (base == 0) { return x; }
    return (x + base - 1) / base * base;
}

struct MMParam {
    uint32_t singleM;
    uint32_t singleN;
    uint32_t singleK;
    bool isLeftTranspose;
    bool isRightTranspose;
    bool cmatrixInitVal = true;
    bool isOutKFisrt = true;
    uint32_t unitFlag = 0;
    uint32_t realM = 0;
};

__aicore__ inline MMParam MakeMMParam(uint32_t singleM, uint32_t singleN, uint32_t singleK, bool isLeftTranspose,
                                      bool isRightTranspose, bool cmatrixInitVal = true, bool isOutKFisrt = true,
                                      uint32_t unitFlag = 0, uint32_t realM = 0)
{
    return {.singleM = singleM,
            .singleN = singleN,
            .singleK = singleK,
            .isLeftTranspose = isLeftTranspose,
            .isRightTranspose = isRightTranspose,
            .cmatrixInitVal = cmatrixInitVal,
            .isOutKFisrt = isOutKFisrt,
            .unitFlag = unitFlag,
            .realM = realM};
}

enum class ABLayout {
    MK = 0,
    KM = 1,
    KN = 2,
    NK = 3,
};

// ---- C310 regbase L1->L0 loads (matmul.h L568-654) ----
static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding

// L1->L0A
template <typename T, ABLayout AL>
__aicore__ inline void LoadDataToL0A(LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor,
                                     const MMParam& mmParam, uint64_t L1Aoffset, uint32_t kSplitSize,
                                     uint32_t mSplitSize)
{
    if constexpr (AL == ABLayout::MK) {
        // VERBATIM close-port of CANN arch35 LoadDataToL0A (fp16/bf16 non-mx path),
        // matmul.h:102-152 (~/workspace/cann/ops-transformer/attention/common/op_kernel/matmul.h).
        // task#34 D=128 FIX: the prior trimmed form set dstStride = kStep, but CANN
        // sets dstStride = mStep (line 134, non-transpose). For D<=64 mStep==kStep so
        // the bug was invisible; at D=128 (kStep=8, mStep=4) the L0A NZ fragment stride
        // was doubled → Q.K^T scrambled → softmax max/sum drift. (probe 2026-06-01:
        // raw S mean_abs_diff 0.022, emitted sm_max ~0.3 vs true ~0.05.)
        LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.ifTranspose = mmParam.isLeftTranspose;
        if (loadData2DParamsA.ifTranspose) {
            loadData2DParamsA.mStep = ((kSplitSize + 15) >> 4 << 4) >> 4;
            loadData2DParamsA.kStep = GetBlockNum<T>(mSplitSize);
        } else {
            loadData2DParamsA.mStep = ((mSplitSize + 15) >> 4 << 4) >> 4;
            loadData2DParamsA.kStep = GetBlockNum<T>(kSplitSize);
        }
        if constexpr (IsSameType<T, float>::value) {
            if (loadData2DParamsA.ifTranspose) {
                loadData2DParamsA.kStep = CeilAlign(loadData2DParamsA.kStep, K_STEP_ALIGN_BASE);
            }
        }
        loadData2DParamsA.srcStride =
            loadData2DParamsA.ifTranspose ? ((mmParam.singleK + 15) >> 4 << 4) >> 4 : loadData2DParamsA.mStep;
        if (mmParam.realM != 0) {
            loadData2DParamsA.mStep = ((mmParam.realM + 15) >> 4 << 4) >> 4;
        }
        loadData2DParamsA.dstStride =
            loadData2DParamsA.ifTranspose ? (mSplitSize + 15) >> 4 : loadData2DParamsA.mStep;
        LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParamsA);
    } else if constexpr (AL == ABLayout::KM) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes = (kSplitSize / ONE_FRACTAL_H_ELEMENT) *
                                       (mmParam.singleM / (ONE_FRACTAL_W_BYTE / sizeof(T)));
        loadData2DParams.srcStride = 1;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = true;
        LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParams);
    }
}

// L1->L0B
template <typename T, ABLayout BL>
__aicore__ inline void LoadDataToL0B(LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor,
                                     const MMParam& mmParam, uint64_t L1Boffset, uint32_t kSplitSize,
                                     uint32_t nSplitSize, int nLoops = 1)
{
    if constexpr (BL == ABLayout::KN) {
        // VERBATIM close-port of CANN arch35 LoadDataToL0B (fp16/bf16 non-mx path),
        // matmul.h:220-251 (~/workspace/cann/ops-transformer/attention/common/op_kernel/matmul.h).
        // task#34 D=128 FIX: the prior trimmed form set dstStride = kStep and
        // srcStride = mStep unconditionally; CANN computes them per the ifTranspose
        // branch (line 226 ifTranspose = !isRightTranspose; line 249 srcStride; line
        // 251 dstStride = mStep for non-transpose). For D<=64 these coincided; at
        // D=128 the K-load fragment stride was wrong → S/O drift.
        LoadData2DParamsV2 loadData2DParamsB;
        loadData2DParamsB.mStartPosition = 0;
        loadData2DParamsB.kStartPosition = 0;
        loadData2DParamsB.ifTranspose = !mmParam.isRightTranspose;
        if (loadData2DParamsB.ifTranspose) {
            loadData2DParamsB.mStep = ((kSplitSize + 15) >> 4 << 4) >> 4;
            loadData2DParamsB.kStep = GetBlockNum<T>(nSplitSize);
        } else {
            loadData2DParamsB.mStep = ((nSplitSize + 15) >> 4 << 4) >> 4;
            loadData2DParamsB.kStep = GetBlockNum<T>(kSplitSize);
        }
        if constexpr (IsSameType<T, float>::value) {
            if (loadData2DParamsB.ifTranspose) {
                loadData2DParamsB.kStep = CeilAlign(loadData2DParamsB.kStep, K_STEP_ALIGN_BASE);
            }
        }
        loadData2DParamsB.srcStride = loadData2DParamsB.ifTranspose
            ? (((mmParam.singleK + 15) >> 4 << 4) >> 4)
            : (((mmParam.singleN + 15) >> 4 << 4) >> 4);
        loadData2DParamsB.dstStride =
            loadData2DParamsB.ifTranspose ? (nSplitSize + 15) >> 4 : loadData2DParamsB.mStep;
        LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParamsB);
    } else if constexpr (BL == ABLayout::NK) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes = (nSplitSize + (ONE_FRACTAL_H_ELEMENT - 1)) / ONE_FRACTAL_H_ELEMENT *
                                       (kSplitSize / (ONE_FRACTAL_W_BYTE / sizeof(T)));
        loadData2DParams.srcStride = 1;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = false;
        LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParams);
    }
}

// ---- 全载 (single-K, single-N): MatmulFull (matmul.h L657-716) ----
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType>
__aicore__ inline void MatmulFull(const LocalTensor<A> &aL1Tensor,
                                  const LocalTensor<B> &bL1Tensor,
                                  L0AType &aL0BuffsDb,
                                  L0BType &bL0BuffsDb,
                                  const LocalTensor<C> &cL0Tensor,
                                  struct MMParam &param)
{
    Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
    l0aBuffer.template Wait<HardEvent::M_MTE1>();
    LocalTensor<A> L0ATensor = l0aBuffer.template GetTensor<A>();
    LoadDataToL0A<A, AL>(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM);
    l0aBuffer.template Set<HardEvent::MTE1_M>();

    Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
    l0bBuffer.template Wait<HardEvent::M_MTE1>();
    LocalTensor<B> L0BTensor = l0bBuffer.template GetTensor<B>();
    LoadDataToL0B<B, BL>(L0BTensor, bL1Tensor, param, 0, param.singleK, param.singleN);
    l0bBuffer.template Set<HardEvent::MTE1_M>();

    l0aBuffer.template Wait<HardEvent::MTE1_M>();
    l0bBuffer.template Wait<HardEvent::MTE1_M>();

    MmadParams mmadParams;
    mmadParams.m = param.singleM;
    if (param.realM != 0) {
        mmadParams.m = param.realM;
    }
    mmadParams.n = param.singleN;
    mmadParams.k = param.singleK;
    mmadParams.cmatrixInitVal = param.isOutKFisrt;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = param.unitFlag;
    if (mmadParams.m == 1) {
        mmadParams.m = 16;
    }
    Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

    l0aBuffer.template Set<HardEvent::M_MTE1>();
    l0bBuffer.template Set<HardEvent::M_MTE1>();
}

// ---- 切K (matmul.h L719-799) ----
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType>
__aicore__ inline void MatmulK(const LocalTensor<A> &aL1Tensor,
                               const LocalTensor<B> &bL1Tensor,
                               L0AType &aL0BuffsDb,
                               L0BType &bL0BuffsDb,
                               const LocalTensor<C> &cL0Tensor,
                               const MMParam &param)
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t tailSize = param.singleK % baseK;
    uint32_t tailK = tailSize ? tailSize : baseK;
    uint64_t L1Aoffset = param.isLeftTranspose ? baseK << 4 : ((param.singleM + 15) >> 4 << 4) * baseK;
    uint64_t L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 4;

    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;
        Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.template Wait<HardEvent::M_MTE1>();
        LocalTensor<A> L0ATensor = l0aBuffer.template GetTensor<A>();
        LoadDataToL0A<A, AL>(L0ATensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM);

        Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.template Wait<HardEvent::M_MTE1>();
        LocalTensor<B> L0BTensor = l0bBuffer.template GetTensor<B>();
        uint64_t loopNum = param.isRightTranspose ? 1 : kLoops;
        LoadDataToL0B<B, BL>(L0BTensor, bL1Tensor, param, k * L1Boffset, tileK, param.singleN, loopNum);
        l0bBuffer.template Set<HardEvent::MTE1_M>();
        l0bBuffer.template Wait<HardEvent::MTE1_M>();

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = param.singleN;
        mmadParams.k = tileK;
        if (mmadParams.m == 1) {
            mmadParams.m = 16;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt && (k == 0);
        mmadParams.cmatrixSource = false;
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1) ?
                                  UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE;
        }
        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

        l0aBuffer.template Set<HardEvent::M_MTE1>();
        l0bBuffer.template Set<HardEvent::M_MTE1>();
    }
}

// ---- 切N (matmul.h L874-949) ----
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType>
__aicore__ inline void MatmulN(const LocalTensor<A> &aL1Tensor,
                               const LocalTensor<B> &bL1Tensor,
                               L0AType &aL0BuffsDb,
                               L0BType &bL0BuffsDb,
                               const LocalTensor<C> &cL0Tensor,
                               const MMParam &param)
{
    uint32_t nLoops = (param.singleN + baseN - 1) / baseN;
    uint32_t tailSize = param.singleN % baseN;
    uint32_t tailN = tailSize ? tailSize : baseN;
    uint64_t L1Boffset = param.isRightTranspose ? (baseN << 4) : ((param.singleK + 15) >> 4 << 4) * baseN;
    uint64_t L0Coffset = ((param.singleM + 15) >> 4 << 4) * baseN;
    if (param.realM != 0) {
        L0Coffset = ((param.realM + 15) >> 4 << 4) * baseN;
    }

    Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
    l0aBuffer.template Wait<HardEvent::M_MTE1>();
    LocalTensor<A> L0ATensor = l0aBuffer.template GetTensor<A>();
    LoadDataToL0A<A, AL>(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM);
    for (uint32_t n = 0; n < nLoops; n++) {
        uint32_t tileN = (n == (nLoops - 1)) ? tailN : baseN;

        Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.template Wait<HardEvent::M_MTE1>();
        LocalTensor<B> L0BTensor = l0bBuffer.template GetTensor<B>();
        uint64_t loopNum = param.isRightTranspose ? nLoops : 1;
        LoadDataToL0B<B, BL>(L0BTensor, bL1Tensor, param, n * L1Boffset, param.singleK, tileN, loopNum);
        l0bBuffer.template Set<HardEvent::MTE1_M>();
        l0bBuffer.template Wait<HardEvent::MTE1_M>();

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = tileN;
        mmadParams.k = param.singleK;
        if (mmadParams.m == 1) {
            mmadParams.m = FP16_ONE_FRACTAL_ELEMENT;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = param.unitFlag;
        Mmad(cL0Tensor[n * L0Coffset], L0ATensor, L0BTensor, mmadParams);

        l0bBuffer.template Set<HardEvent::M_MTE1>();
    }
    l0aBuffer.template Set<HardEvent::M_MTE1>();
}

// ---- MatmulBase dispatcher (matmul.h L1003-1018) ----
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType>
__aicore__ inline void MatmulBase(const LocalTensor<A> &aL1Tensor,
                                  const LocalTensor<B> &bL1Tensor,
                                  L0AType &aL0BuffsDb,
                                  L0BType &bL0BuffsDb,
                                  const LocalTensor<C> &cL0Tensor,
                                  struct MMParam &param)
{
    if ((param.singleK + baseK - 1) / baseK > 1) {
        MatmulK<A, B, C, baseM, baseN, baseK, AL, BL>(aL1Tensor, bL1Tensor, aL0BuffsDb, bL0BuffsDb, cL0Tensor, param);
    } else if ((param.singleN + baseN - 1) / baseN > 1) {
        MatmulN<A, B, C, baseM, baseN, baseK, AL, BL>(aL1Tensor, bL1Tensor, aL0BuffsDb, bL0BuffsDb, cL0Tensor, param);
    } else {
        MatmulFull<A, B, C, baseM, baseN, baseK, AL, BL>(aL1Tensor, bL1Tensor, aL0BuffsDb, bL0BuffsDb, cL0Tensor, param);
    }
}

} // namespace fa_base_matmul

#endif // REGBASE_MATMUL_H
