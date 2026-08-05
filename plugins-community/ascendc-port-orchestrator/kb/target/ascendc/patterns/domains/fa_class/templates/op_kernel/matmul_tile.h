/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_TILE_H
#define MATMUL_TILE_H

#include "kernel_operator.h"

// Manual matmul micro-ops for arch35 / A5 (Ascend950PR, CANN 9.x).
// FixpipeNzL0cToNdGm emits ND output (ndNum=1, srcNdStride=0, dstNdStride=0 — td-8 #11).
//
// arch35 fixpipe NOTE (2026-06-16): the L0C->GM fixpipe MUST use FixpipeParamsC310 (NZ2ND)
// with an explicit quantPre cast mode (F322BF16 for float->bf16, F322F16 for float->half).
// The prior arch22 form (FixpipeParamsV220, no cast) is ILLEGAL on arch35 — float->lower-prec
// without quantPre raises device error 169 subErrType 0x4. The C310 form below is transplanted
// from the regbase-GDN worker's gdn_regbase_mm.h::RbL0cToGm, which COMPILED + RAN on 950PR /
// CANN-9.x (2026-06-16). PENDING: compile + numeric re-verify of THIS template file on a clean
// 9.x env before treating it as production-verified.
//
// L0C accum is always float; output is cast to Tout (bf16/half) by the fixpipe quantPre.

template<typename T>
__aicore__ inline void LoadNdGmToNzL1(const AscendC::LocalTensor<T> &dst,
                                      const AscendC::GlobalTensor<T> &src,
                                      uint32_t m, uint32_t n, uint32_t ld)
{
    AscendC::Nd2NzParams params;
    params.ndNum = 1;
    params.nValue = m;
    params.dValue = n;
    params.srcNdMatrixStride = 0;
    params.srcDValue = ld;
    params.dstNzC0Stride = m;
    params.dstNzNStride = 1;
    params.dstNzMatrixStride = 0;
    AscendC::DataCopy(dst, src, params);
}

template<typename T>
__aicore__ inline void LoadNdGmToNzL1(const AscendC::LocalTensor<T> &dst,
                                      const AscendC::GlobalTensor<T> &src,
                                      uint32_t m, uint32_t n, uint32_t ld,
                                      uint32_t dstNzC0Stride)
{
    AscendC::Nd2NzParams params;
    params.ndNum = 1;
    params.nValue = m;
    params.dValue = n;
    params.srcNdMatrixStride = 0;
    params.srcDValue = ld;
    params.dstNzC0Stride = dstNzC0Stride;
    params.dstNzNStride = 1;
    params.dstNzMatrixStride = 0;
    AscendC::DataCopy(dst, src, params);
}

template<typename T>
__aicore__ inline void LoadNzL1ToZzL0A(const AscendC::LocalTensor<T> &dst,
                                       const AscendC::LocalTensor<T> &src,
                                       uint32_t m, uint32_t k, uint32_t colC0Stride)
{
    AscendC::LoadData3DParamsV2<T> params;
    params.l1H = 1;
    params.l1W = colC0Stride;
    params.channelSize = k;
    params.kExtension = k;
    params.mExtension = m;
    params.strideH = 1;
    params.strideW = 1;
    params.filterH = 1;
    params.filterW = 1;
    params.dilationFilterH = 1;
    params.dilationFilterW = 1;
    AscendC::LoadData(dst, src, params);
}

template<typename T>
__aicore__ inline void LoadNzL1ToZnL0B(const AscendC::LocalTensor<T> &dst,
                                       const AscendC::LocalTensor<T> &src,
                                       uint32_t k, uint32_t n, uint32_t colC0Stride)
{
    AscendC::LoadData3DParamsV2<T> params;
    params.l1H = 1;
    params.l1W = colC0Stride;
    params.channelSize = n;
    params.kExtension = n;
    params.mExtension = k;
    params.strideH = 1;
    params.strideW = 1;
    params.filterH = 1;
    params.filterW = 1;
    params.dilationFilterH = 1;
    params.dilationFilterW = 1;
    AscendC::LoadData(dst, src, params);
}

// L0C(NZ, float accum) -> GM(ND, row stride = dstStride), casting float accum -> Tout.
// arch35 form: FixpipeParamsC310<ROW_MAJOR> (NZ2ND) + templated Fixpipe<Tout,Tacc,CFG>.
// quantPre selects the float->Tout cast (F322BF16 / F322F16); REQUIRED on dav_c310 — the
// default NoQuant is an illegal FIXP config for float->lower-precision (device error 169 0x4).
// MT_FIXP_ROW_MAJOR_GM = {ROW_MAJOR (enable NZ2ND, ND output), false (dest is GM not UB)}.
template<typename Tout, typename Tacc = float>
__aicore__ inline void FixpipeNzL0cToNdGmStride(const AscendC::GlobalTensor<Tout> &dst,
                                                 const AscendC::LocalTensor<Tacc> &src,
                                                 uint32_t m, uint32_t n, uint32_t dstStride)
{
    constexpr AscendC::FixpipeConfig MT_FIXP_ROW_MAJOR_GM = {AscendC::CO2Layout::ROW_MAJOR, false};
    AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
    params.nSize = (n + 7) >> 3 << 3;                       // N 8-elem (32B) aligned
    params.mSize = (m + 1) >> 1 << 1;                       // even M
    params.srcStride = ((params.mSize + 15) >> 4) << 4;     // L0C NZ fragment stride
    params.dstStride = dstStride;                           // ND row stride in dst GM
    params.dualDstCtl = 0;                                  // single GM destination
    if constexpr (AscendC::IsSameType<Tout, bfloat16_t>::value) {
        params.quantPre = AscendC::QuantMode_t::F322BF16;   // float accum -> bf16
    } else if constexpr (AscendC::IsSameType<Tout, half>::value) {
        params.quantPre = AscendC::QuantMode_t::F322F16;    // float accum -> half
    }
    params.params.ndNum = 1;
    params.params.srcNdStride = 0;
    params.params.dstNdStride = 0;
    AscendC::Fixpipe<Tout, Tacc, MT_FIXP_ROW_MAJOR_GM>(dst, src, params);
}

// Tight-ND variant: dst row stride == n.
template<typename Tout, typename Tacc = float>
__aicore__ inline void FixpipeNzL0cToNdGm(const AscendC::GlobalTensor<Tout> &dst,
                                           const AscendC::LocalTensor<Tacc> &src,
                                           uint32_t m, uint32_t n)
{
    FixpipeNzL0cToNdGmStride<Tout, Tacc>(dst, src, m, n, n);
}

#endif // MATMUL_TILE_H
