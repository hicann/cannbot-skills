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
 * \file conv3dtranspose_tiling.cpp
 * \brief conv3dtranspose 直调 Host 求解器（Conv3DtSolveTiling 强符号实现）
 *
 * 实现真值：卷积反向范式 generic/input/tiling/tiling.md（切分算法与公共公式）+
 * generic/input/tiling/templates/{standard,small-shape,l1-resident}/tiling.md（三模板求解流程）+
 * adapters/ascendc/tilingkey-templates.md §2（loadB2Condition 判定与 key 编码）+
 * generic/preprocessing.md §4.2/§4.3/§5.2/§5.3（转置形状模式、派生量、initOutputFlag、bias 前置）+
 * adapters/ascendc/buffer-and-types.md §1（直调平台查询，含 BT 容量）。
 *
 * 与 conv3ddx 求解器的转置差异（同底本核核心的适配点）：
 *   1. 转置几何——b⁺=Keff−1−p⁺+q+rho（rho=Ly−(base+q)，显式 input_size 模式校验 0≤rho<stride；
 *      属性推导模式为 input_size 全零哨兵，本 ABI 不携带 x batch/filter CinG，无法推导 N/Cin，返回
 *      专项错误码；E1 越界 rho 由显式模式校验拒绝）；
 *   2. q 约束——q≥0 且 q<stride or q<dilation（E9，逐轴）；
 *   3. initOutputFlag——preprocessing §5.2 转置条件（rhoD>padBack / rhoH>padDown / strideH>hk /
 *      qD>0 / qH>0 / dilationD>1 / Tiling strideD>dk 补充条件）；
 *   4. bias——TilingData 增 hasBiasFlag/isBiasFullLoad/biasAllocBytes；UpdateIsBiasFullLoad 在最终
 *      baseN/singleCoreCin 形成后按 BT 容量判定；L1 候选预算计入 biasBudget=alignUp(baseN*4,64)，
 *      实际分配按 singleCoreCin（全量）/baseN（分块）复核（tiling.md §5.1）；bias 长度=Cin 校验（E10）。
 *
 * 平台资源（buffer-and-types §1.2 直调条目）：aclrtGetSocName + fe::PlatformInfoManager →
 * fe::PlatFormInfos → platform_ascendc::PlatformAscendC 查询 AIC 核数、L1/L0A/L0B/L0C 与 BT 容量
 * （本机实测 BT=4096B，探针证据见 $WORK_DIR/1-实现记录.md）。libplatform/libtiling_api 为旧 C++
 * ABI 库，本编译单元以 _GLIBCXX_USE_CXX11_ABI=0 编译；对外接口为纯 C POD，不受影响。
 *
 * L2 拦截层语义（0-测试工程说明 §9）：Conv3DtSpec 属性为 uint32，符号信息在 ABI 边界丢失
 * （E3 sw=0 → 0；E4 padWf=-1 → 0xFFFFFFFF）。本求解器按无符号范围校验拒绝：stride/dilation ≥ 1
 * 且 ≤ 指令上限；pads ≤ 255；q 约束按无符号比较；几何一致性按转置形状关系复核。
 */

#ifndef _GLIBCXX_USE_CXX11_ABI
#define _GLIBCXX_USE_CXX11_ABI 0
#endif

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "acl/acl.h"
#include "platform/platform_info.h"
#include "utils/tiling/platform/platform_ascendc.h"

#include "conv3dtranspose_launch.h"
#include "conv3dtranspose_tiling_data.h"

namespace {

/* ---------------- 小工具 ---------------- */

inline uint32_t CeilDivU(uint64_t a, uint64_t b)
{
    return b == 0 ? 0 : static_cast<uint32_t>((a + b - 1) / b);
}

inline uint32_t AlignUpU(uint64_t a, uint64_t b)
{
    return CeilDivU(a, b) * static_cast<uint32_t>(b);
}

inline uint32_t AlignDownU(uint64_t a, uint64_t b)
{
    return b == 0 ? 0 : static_cast<uint32_t>(a / b) * static_cast<uint32_t>(b);
}

inline uint64_t GcdU(uint64_t a, uint64_t b)
{
    while (b != 0) {
        const uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/* ---------------- 平台资源查询（buffer-and-types §1.2，含 BT 容量） ---------------- */

struct PlatformCaps {
    uint32_t maxCores = 0;
    uint64_t l1Bytes = 0;
    uint64_t l0aBytes = 0;
    uint64_t l0bBytes = 0;
    uint64_t l0cBytes = 0;
    uint64_t btBytes = 0; /* bias table 容量（bias-bt-path §1；直调经 PlatformAscendC 查询） */
};

bool QueryPlatformCaps(PlatformCaps& caps)
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        return false;
    }
    auto& mgr = fe::PlatformInfoManager::Instance();
    if (mgr.InitializePlatformInfo(socName) != 0) {
        (void)mgr.Finalize();
        return false;
    }
    fe::PlatFormInfos infos;
    fe::OptionalInfos optInfos;
    const bool ok = mgr.GetPlatformInfos(socName, infos, optInfos) == 0;
    (void)mgr.Finalize();
    if (!ok) {
        return false;
    }
    platform_ascendc::PlatformAscendC plat(&infos);
    caps.maxCores = plat.GetCoreNumAic();
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::L1, caps.l1Bytes);
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, caps.l0aBytes);
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, caps.l0bBytes);
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, caps.l0cBytes);
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::BT, caps.btBytes);
    return caps.maxCores > 0 && caps.l1Bytes > 0 && caps.l0aBytes > 0 && caps.l0bBytes > 0 && caps.l0cBytes > 0;
}

/* ---------------- 求解上下文 ---------------- */

struct SolveCtx {
    /* 规格归一结果 */
    uint32_t batch = 0, cin = 0, cout = 0, cinG = 0, coutG = 0, group = 0;
    uint32_t di = 0, hi = 0, wi = 0, dout = 0, ho = 0, wo = 0;
    uint32_t dk = 0, hk = 0, wk = 0;
    uint32_t strideD = 0, strideH = 0, strideW = 0;
    uint32_t dilationD = 0, dilationH = 0, dilationW = 0;
    uint32_t padFront = 0, padBack = 0, padUp = 0, padDown = 0, padLeft = 0, padRight = 0;
    int32_t backpropPadUp = 0, backpropPadDown = 0, backpropPadLeft = 0, backpropPadRight = 0;
    uint32_t rhoD = 0, rhoH = 0, rhoW = 0;                   /* 转置余量 Ly-(base+q)，显式模式 ∈[0,stride) */
    uint32_t qD = 0, qH = 0, qW = 0;                         /* output_padding 空间分量 */
    uint32_t khDilation = 0, kwDilation = 0, kdDilation = 0; /* KeffH/KeffW/KeffD */
    uint32_t hkWk = 0, dkHkWk = 0, windowSpan = 0;
    uint32_t mValue = 0, nValue = 0, kValue = 0;
    uint32_t hoExpand = 0, woExpand = 0;
    uint32_t initOutputFlag = 0;
    uint32_t filterNdhwc = 0; /* weight 物理格式（与 data_format 配对） */
    uint32_t hasBias = 0;     /* bias 存在性（与 launch bias 指针一致） */

    /* 平台 */
    PlatformCaps caps;

    /* 求解结果 */
    uint32_t baseM = 0, baseN = 0, baseK = 0;
    uint32_t stepKa = 0, stepKb = 0;
    uint32_t al1Pbuffer = 0, bl1Pbuffer = 0, al0Pbuffer = 0, bl0Pbuffer = 0, cl0Pbuffer = 0;
    uint32_t singleCoreM = 0, singleCoreCin = 0, singleCoreDin = 0, singleCoreCout = 0;
    uint32_t enableFullLoad = 0;
    uint32_t isBiasFullLoad = 0;
    uint32_t biasAllocBytes = 0;
    uint32_t loadB2 = 0;
    uint32_t templateId = 0xFFFFFFFFu;
};

/* ---------------- 校验与归一（preprocessing §2.1/§4.2/§4.3/§5.2/§5.3） ---------------- */

int ValidateAndNormalize(const Conv3DtSpec& spec, SolveCtx& c)
{
    /* dtype 支持与三路一致（E2a/E2b；bias 恒 FP32 不编码） */
    if (spec.filterDtype != spec.dedyDtype) {
        return -101; /* 三路 dtype 不一致 */
    }
    if (spec.filterDtype != CONV3DT_DTYPE_FP16 && spec.filterDtype != CONV3DT_DTYPE_BF16) {
        return -102; /* 不支持 dtype（如 float32），无对应编译实例 */
    }
    /* format 支持（E8） */
    if (spec.fmt != CONV3DT_FMT_NCDHW && spec.fmt != CONV3DT_FMT_NDHWC) {
        return -103;
    }
    /* bias 存在性与长度（E10；preprocessing §5.3：非空 bias 为一维 FP32 长度 Cin） */
    if (spec.hasBias > 1u) {
        return -114;
    }
    c.hasBias = spec.hasBias;
    /* 正值与上限（E3/E4：uint32 ABI 下负值编码为大数，按无符号范围拒绝） */
    const uint32_t pos[] = {spec.n, spec.cin, spec.cout, spec.dout, spec.ho, spec.wo, spec.kd, spec.kh, spec.kw};
    for (uint32_t v : pos) {
        if (v == 0 || v > 0x7FFFFFFFu) {
            return -104;
        }
    }
    if (spec.groups == 0 || spec.groups > 0x7FFFFFFFu) {
        return -105;
    }
    const uint32_t strides[] = {spec.sd, spec.sh, spec.sw};
    for (uint32_t v : strides) {
        if (v < 1 || v > CONV3DT_MAX_STRIDE) {
            return -106; /* strides 含 0 / 负值编码 / 超 Load3D 指令上限 */
        }
    }
    const uint32_t dilations[] = {spec.dd, spec.dh, spec.dw};
    for (uint32_t v : dilations) {
        if (v < 1 || v > CONV3DT_MAX_DILATION) {
            return -107;
        }
    }
    const uint32_t pads[] = {spec.padDf, spec.padDb, spec.padHf, spec.padHb, spec.padWf, spec.padWb};
    for (uint32_t v : pads) {
        if (v > CONV3DT_MAX_PAD) {
            return -108; /* pads 负值编码 / 超 8 位搬运字段上限 */
        }
    }
    /* 分组整除（E5） */
    if (spec.cin % spec.groups != 0 || spec.cout % spec.groups != 0) {
        return -109;
    }
    /* output_padding q 约束（E9，preprocessing §4.2：q≥0 且 q<stride or q<dilation；
     * uint32 ABI 下负 q 编码为大数，落入同一拒绝分支） */
    const uint32_t qs[] = {spec.qd, spec.qh, spec.qw};
    const uint32_t qStrides[] = {spec.sd, spec.sh, spec.sw};
    const uint32_t qDilations[] = {spec.dd, spec.dh, spec.dw};
    for (uint32_t i = 0; i < 3; ++i) {
        if (qs[i] >= qStrides[i] && qs[i] >= qDilations[i]) {
            return -113; /* q 违反 q<stride or q<dilation */
        }
    }
    /* input_size 形状模式（preprocessing §4.2）：全零哨兵=属性推导，五维均正=显式模式 */
    const bool allZero = (spec.n == 0 && spec.cin == 0 && spec.di == 0 && spec.hi == 0 && spec.wi == 0);
    const bool allPositive = (spec.n > 0 && spec.cin > 0 && spec.di > 0 && spec.hi > 0 && spec.wi > 0);
    if (!allZero && !allPositive) {
        return -115; /* 部分为零的形状：具体字段错误 */
    }

    c.batch = spec.n;
    c.cin = spec.cin;
    c.cout = spec.cout;
    c.group = spec.groups;
    c.cinG = spec.cin / spec.groups;
    c.coutG = spec.cout / spec.groups;
    c.dout = spec.dout;
    c.ho = spec.ho;
    c.wo = spec.wo;
    c.dk = spec.kd;
    c.hk = spec.kh;
    c.wk = spec.kw;
    c.strideD = spec.sd;
    c.strideH = spec.sh;
    c.strideW = spec.sw;
    c.dilationD = spec.dd;
    c.dilationH = spec.dh;
    c.dilationW = spec.dw;
    c.padFront = spec.padDf;
    c.padBack = spec.padDb;
    c.padUp = spec.padHf;
    c.padDown = spec.padHb;
    c.padLeft = spec.padWf;
    c.padRight = spec.padWb;
    c.qD = spec.qd;
    c.qH = spec.qh;
    c.qW = spec.qw;
    c.filterNdhwc = (spec.fmt == CONV3DT_FMT_NDHWC) ? 1u : 0u;

    c.khDilation = (c.hk - 1) * c.dilationH + 1;
    c.kwDilation = (c.wk - 1) * c.dilationW + 1;
    c.kdDilation = (c.dk - 1) * c.dilationD + 1;
    c.hkWk = c.hk * c.wk;
    c.dkHkWk = c.dk * c.hkWk;
    c.windowSpan = c.hkWk * CONV3DT_C0;

    /* bias 长度=Cin（E10；hasBias=1 时 biasElems 必须等于 cin） */
    if (c.hasBias != 0 && spec.biasElems != c.cin) {
        return -114;
    }

    /* 转置形状关系逐轴校验（preprocessing §4.2/§4.3，显式 input_size 模式）：
     * base=(Lx-1)*stride-pF-pB+Keff；rho=Ly-(base+q)，要求 0<=rho<stride（E1 越界在此拒绝）。
     * 属性推导模式（全零哨兵）：di/hi/wi=base+q；但本 ABI（Conv3DtSpec）不携带 x batch 与
     * filter CinG，N/Cin 无法推导，返回专项错误（测试框架恒走显式模式，见 launch.h 契约）。 */
    const int64_t baseD = static_cast<int64_t>(c.dout - 1) * c.strideD - c.padFront - c.padBack + c.kdDilation;
    const int64_t baseH = static_cast<int64_t>(c.ho - 1) * c.strideH - c.padUp - c.padDown + c.khDilation;
    const int64_t baseW = static_cast<int64_t>(c.wo - 1) * c.strideW - c.padLeft - c.padRight + c.kwDilation;
    if (allZero) {
        c.di = static_cast<uint32_t>(baseD + c.qD);
        c.hi = static_cast<uint32_t>(baseH + c.qH);
        c.wi = static_cast<uint32_t>(baseW + c.qW);
        return -116; /* 属性推导模式：N/Cin 不可推导（本 ABI 无 x batch/filter CinG 字段） */
    }
    const int64_t rhoD = static_cast<int64_t>(spec.di) - (baseD + c.qD);
    const int64_t rhoH = static_cast<int64_t>(spec.hi) - (baseH + c.qH);
    const int64_t rhoW = static_cast<int64_t>(spec.wi) - (baseW + c.qW);
    if (rhoD < 0 || rhoD >= static_cast<int64_t>(c.strideD)) {
        return -112; /* D 轴 rho 越界 */
    }
    if (rhoH < 0 || rhoH >= static_cast<int64_t>(c.strideH)) {
        return -110; /* H 轴 rho 越界 */
    }
    if (rhoW < 0 || rhoW >= static_cast<int64_t>(c.strideW)) {
        return -111; /* W 轴 rho 越界 */
    }
    c.di = spec.di;
    c.hi = spec.hi;
    c.wi = spec.wi;
    c.rhoD = static_cast<uint32_t>(rhoD);
    c.rhoH = static_cast<uint32_t>(rhoH);
    c.rhoW = static_cast<uint32_t>(rhoW);

    /* 派生量（preprocessing §4.3）：反向边界 b⁻=Keff-1-p⁻、b⁺=Keff-1-p⁺+q+rho（转置场景） */
    c.backpropPadUp = static_cast<int32_t>(c.khDilation - 1 - c.padUp);
    c.backpropPadDown = static_cast<int32_t>(c.khDilation - 1 - c.padDown + c.qH + c.rhoH);
    c.backpropPadLeft = static_cast<int32_t>(c.kwDilation - 1 - c.padLeft);
    c.backpropPadRight = static_cast<int32_t>(c.kwDilation - 1 - c.padRight + c.qW + c.rhoW);
    c.hoExpand = (c.ho - 1) * c.strideH + 1;
    c.woExpand = (c.wo - 1) * c.strideW + 1;
    c.mValue = AlignUpU(static_cast<uint64_t>(c.hi) * c.wi, CONV3DT_FRACTAL);
    c.nValue = AlignUpU(c.cinG, CONV3DT_FRACTAL);
    c.kValue = AlignUpU(c.coutG, CONV3DT_C0) * c.hkWk;

    /* 输出初始化标志（preprocessing §5.2 转置条件：rhoD>padBack / rhoH>padDown / strideH>hk /
     * qD>0 / qH>0 / dilationD>1 / 常规核窗口 Tiling 的 strideD>dk 补充条件；qW 不参与判定） */
    c.initOutputFlag = 0;
    if (c.rhoD > c.padBack || c.rhoH > c.padDown || c.strideH > c.hk || c.qD > 0 || c.qH > 0 || c.dilationD > 1 ||
        c.strideD > c.dk) {
        c.initOutputFlag = 1;
    }
    return CONV3DT_OK;
}

/* ---------------- 指令范围预检（tiling.md §4 maxM 收紧依据） ---------------- */

bool BackpropPadWithinField(const SolveCtx& c)
{
    /* 覆盖量复算（b⁺ 含 q+rho 已并入 backpropPadRight/Down，可由 wi/hi 与 bpLeft/bpUp 恒等推出） */
    const int64_t bpPadRight = static_cast<int64_t>(c.wi) - (static_cast<int64_t>(c.wo - 1) * c.strideW + 1) +
                               static_cast<int64_t>((c.wk - 1) * c.dilationW) - c.backpropPadLeft;
    const int64_t bpPadDown = static_cast<int64_t>(c.hi) - (static_cast<int64_t>(c.ho - 1) * c.strideH + 1) +
                              static_cast<int64_t>((c.hk - 1) * c.dilationH) - c.backpropPadUp;
    return c.backpropPadLeft <= static_cast<int64_t>(CONV3DT_MAX_PAD) &&
           c.backpropPadUp <= static_cast<int64_t>(CONV3DT_MAX_PAD) &&
           bpPadRight <= static_cast<int64_t>(CONV3DT_MAX_PAD) && bpPadDown <= static_cast<int64_t>(CONV3DT_MAX_PAD);
}

/* ---------------- L0 基本块（tiling.md §4 AdjustBaseMNK） ---------------- */

void AdjustBaseMNK(SolveCtx& c, uint32_t incomingK)
{
    const uint64_t l0a = c.caps.l0aBytes;
    const uint64_t l0b = c.caps.l0bBytes;
    const uint64_t l0c = c.caps.l0cBytes;

    uint32_t maxM = (c.baseN <= CONV3DT_FRACTAL) ? 512u : 1024u;
    if (!BackpropPadWithinField(c)) {
        maxM = 512u; /* 反向边界复算超 8 位字段上限时收紧 */
    }

    const uint32_t oldM = c.baseM;
    const uint32_t oldN = c.baseN;
    c.baseM = c.baseM < c.mValue ? c.baseM : c.mValue;
    const uint32_t capM = static_cast<uint32_t>(l0a / (CONV3DT_C0 * CONV3DT_ELEM_BYTES * c.al0Pbuffer));
    c.baseM = c.baseM < capM ? c.baseM : capM;
    c.baseN = c.baseN < c.nValue ? c.baseN : c.nValue;

    if (c.baseN < oldN) {
        const uint32_t mByL0C =
            static_cast<uint32_t>(l0c / (CONV3DT_ACC_BYTES * c.cl0Pbuffer * c.baseN) / CONV3DT_FRACTAL) *
            CONV3DT_FRACTAL;
        const uint32_t cand = mByL0C == 0 ? CONV3DT_FRACTAL : mByL0C;
        const uint32_t m1 = maxM < cand ? maxM : cand;
        c.baseM = m1 < c.mValue ? m1 : c.mValue;
    }
    if (c.baseM < oldM) {
        const uint32_t nByL0C =
            static_cast<uint32_t>(l0c / (CONV3DT_ACC_BYTES * c.cl0Pbuffer * c.baseM) / CONV3DT_FRACTAL) *
            CONV3DT_FRACTAL;
        const uint32_t cand = nByL0C == 0 ? CONV3DT_FRACTAL : nByL0C;
        const uint32_t n1 = 1024u < cand ? 1024u : cand;
        c.baseN = n1 < c.nValue ? n1 : c.nValue;
    }

    /* K 窗口配对（tiling.md §4.1）：从 maxK 按 C0 递减，仅检查大于 C0 的候选 */
    const uint64_t denom = c.al0Pbuffer * CONV3DT_ELEM_BYTES * (c.baseM > c.baseN ? c.baseM : c.baseN);
    uint32_t maxK = static_cast<uint32_t>(l0a / denom);
    maxK = AlignDownU(maxK, CONV3DT_C0);
    maxK = maxK < CONV3DT_C0 ? CONV3DT_C0 : maxK;
    maxK = maxK < c.kValue ? maxK : c.kValue;
    uint32_t chosen = 0;
    for (uint32_t cand = maxK; cand > CONV3DT_C0; cand -= CONV3DT_C0) {
        const bool condW = (c.wk > 1) && (cand % (c.wk * CONV3DT_C0) == 0);
        const bool condH = (c.hk > 1) && (cand % (c.hk * CONV3DT_C0) == 0);
        const bool condFull = (cand % c.windowSpan == 0) || (c.windowSpan % cand == 0);
        if (condW || condH || condFull) {
            chosen = cand;
            break;
        }
    }
    c.baseK = (chosen != 0) ? chosen : (incomingK < maxK ? incomingK : maxK);

    /* 选定 K 后更新部分和份数并核对三路 L0 容量（不足时按 M→N→K 收缩一个粒度） */
    c.cl0Pbuffer = (static_cast<uint64_t>(c.baseM) * c.baseN * CONV3DT_ACC_BYTES * 2 <= l0c) ? 2u : 1u;
    for (int guard = 0; guard < 400; ++guard) {
        const uint64_t a = static_cast<uint64_t>(c.baseM) * c.baseK * CONV3DT_ELEM_BYTES * c.al0Pbuffer;
        const uint64_t b = static_cast<uint64_t>(c.baseN) * c.baseK * CONV3DT_ELEM_BYTES * c.bl0Pbuffer;
        const uint64_t cc = static_cast<uint64_t>(c.baseM) * c.baseN * CONV3DT_ACC_BYTES * c.cl0Pbuffer;
        if (a <= l0a && b <= l0b && cc <= l0c) {
            break;
        }
        if (c.baseM > CONV3DT_FRACTAL && (a > l0a || cc > l0c)) {
            c.baseM -= CONV3DT_FRACTAL;
        } else if (c.baseN > CONV3DT_FRACTAL && (b > l0b || cc > l0c)) {
            c.baseN -= CONV3DT_FRACTAL;
        } else if (c.baseK > CONV3DT_C0) {
            c.baseK -= CONV3DT_C0;
        } else {
            break;
        }
        c.cl0Pbuffer = (static_cast<uint64_t>(c.baseM) * c.baseN * CONV3DT_ACC_BYTES * 2 <= l0c) ? 2u : 1u;
    }
}

/* ---------------- L1 驻留容量（tiling.md §5 + §5.1 可选 bias 预算） ---------------- */

uint32_t CoveredRows(uint32_t spaceLength, uint32_t wi)
{
    if (spaceLength % wi == 0 || wi % spaceLength == 0) {
        return CeilDivU(spaceLength, wi);
    }
    return spaceLength > wi ? (spaceLength / wi + 2) : 2;
}

uint32_t ResidentRows(const SolveCtx& c, uint32_t spaceLength)
{
    const uint32_t rows = CoveredRows(spaceLength, c.wi);
    const uint32_t span = rows + c.khDilation - 1;
    return span < c.hoExpand ? span : c.hoExpand;
}

uint32_t CopyLines(uint32_t residentK, uint32_t windowSpan)
{
    if (residentK % windowSpan == 0 || windowSpan % residentK == 0) {
        return CeilDivU(residentK, windowSpan);
    }
    return residentK > windowSpan ? (residentK / windowSpan + 2) : 2;
}

/* bias 候选预算（tiling.md §5.1：候选检查统一计入，按当前 baseN 重算） */
inline uint64_t BiasBudget(const SolveCtx& c)
{
    return c.hasBias != 0 ? AlignUpU(static_cast<uint64_t>(c.baseN) * CONV3DT_BIAS_BYTES, 64u) : 0u;
}

/* 实际槽字节数（Kernel InitBuffer 分配量；宽度预算取 Wo*s_w ≥ woExpand） */
void CalcSlotBytes(const SolveCtx& c, uint32_t& aSlot, uint32_t& bSlot)
{
    const uint32_t rows = ResidentRows(c, c.baseM);
    const uint32_t widthBudget = c.wo * c.strideW;
    uint32_t aChannels = CeilDivU(static_cast<uint64_t>(c.stepKa) * c.baseK, c.hkWk);
    const uint32_t aChCap = AlignUpU(c.coutG, CONV3DT_C0);
    aChannels = aChannels < aChCap ? aChannels : aChCap;
    if (aChannels == 0) {
        aChannels = 1;
    }
    aSlot = AlignUpU(static_cast<uint64_t>(rows) * widthBudget * aChannels * CONV3DT_ELEM_BYTES, 32u);
    bSlot = AlignUpU(static_cast<uint64_t>(c.baseN) * c.stepKb * c.baseK * CONV3DT_ELEM_BYTES, 32u);
}

bool IsL1ParamsValid(SolveCtx& c)
{
    if (c.stepKa == 0 || c.stepKb == 0 || c.baseK == 0) {
        return false;
    }
    const uint64_t lenA = static_cast<uint64_t>(c.stepKa) * c.baseK;
    const uint64_t lenB = static_cast<uint64_t>(c.stepKb) * c.baseK;
    /* 每路驻留长度覆盖完整 kValue 或为 windowSpan 的整数倍 */
    if (lenA < c.kValue && lenA % c.windowSpan != 0) {
        return false;
    }
    if (lenB < c.kValue && lenB % c.windowSpan != 0) {
        return false;
    }
    const uint32_t rows = ResidentRows(c, c.baseM);
    const uint32_t widthBudget = c.wo * c.strideW;
    uint32_t aCh = CeilDivU(lenA, c.hkWk);
    if (aCh == 0) {
        aCh = 1;
    }
    const uint64_t dyBudget = static_cast<uint64_t>(rows) * widthBudget * aCh * CONV3DT_ELEM_BYTES * c.al1Pbuffer;
    const uint32_t lines = CopyLines(static_cast<uint32_t>(lenB < 0xFFFFFFFFull ? lenB : 0xFFFFFFFFull), c.windowSpan);
    const uint64_t wBudget = static_cast<uint64_t>(c.baseN) * lines * c.windowSpan * CONV3DT_ELEM_BYTES * c.bl1Pbuffer;
    const uint64_t biasBudget = BiasBudget(c);
    if (dyBudget + wBudget + biasBudget >= c.caps.l1Bytes) {
        return false;
    }
    uint32_t aSlot = 0;
    uint32_t bSlot = 0;
    CalcSlotBytes(c, aSlot, bSlot);
    const uint64_t total =
        static_cast<uint64_t>(aSlot) * c.al1Pbuffer + static_cast<uint64_t>(bSlot) * c.bl1Pbuffer + biasBudget;
    return total <= c.caps.l1Bytes;
}

/* ---------------- CalStepK（tiling.md §6） ---------------- */

bool LadderMatchStep(SolveCtx& c)
{
    const uint32_t coutBlocks = CeilDivU(c.coutG, CONV3DT_C0);
    const uint32_t initialK1 = CeilDivU(coutBlocks, c.al1Pbuffer) * c.windowSpan;
    const uint32_t initialK2 =
        CeilDivU(c.caps.l1Bytes / CONV3DT_ELEM_BYTES, static_cast<uint64_t>(c.baseN) * c.al1Pbuffer);
    uint32_t initialK = initialK1 < initialK2 ? initialK1 : initialK2;
    if (initialK == 0) {
        initialK = c.windowSpan;
    }
    uint32_t step = CeilDivU(initialK, c.baseK);
    if (step == 0) {
        step = 1;
    }
    for (; step >= 1; --step) {
        c.stepKa = step;
        c.stepKb = step;
        if (IsL1ParamsValid(c)) {
            return true;
        }
    }
    c.stepKa = 0;
    c.stepKb = 0;
    return false;
}

bool EqualL1MatchStep(SolveCtx& c)
{
    const uint32_t coutBlocks = CeilDivU(c.coutG, CONV3DT_C0);
    const uint32_t rows = ResidentRows(c, c.baseM);
    const uint64_t dyBlockBytes = static_cast<uint64_t>(CONV3DT_ELEM_BYTES) * rows * (c.wo * c.strideW) * CONV3DT_C0;
    const uint64_t wBlockBytes = static_cast<uint64_t>(c.windowSpan) * c.baseN * CONV3DT_ELEM_BYTES;

    uint32_t dyBlocks = static_cast<uint32_t>((c.caps.l1Bytes / 2 / c.al1Pbuffer) / dyBlockBytes);
    dyBlocks = dyBlocks == 0 ? 1 : dyBlocks;
    dyBlocks = dyBlocks < coutBlocks ? dyBlocks : coutBlocks;
    if (static_cast<uint64_t>(dyBlocks) * c.al1Pbuffer >= coutBlocks) {
        dyBlocks = CeilDivU(coutBlocks, c.al1Pbuffer);
    }
    uint32_t wBlocks = static_cast<uint32_t>((c.caps.l1Bytes / 2 / c.bl1Pbuffer) / wBlockBytes);
    wBlocks = wBlocks == 0 ? 1 : wBlocks;
    wBlocks = wBlocks < coutBlocks ? wBlocks : coutBlocks;
    if (static_cast<uint64_t>(wBlocks) * c.bl1Pbuffer >= coutBlocks) {
        wBlocks = CeilDivU(coutBlocks, c.bl1Pbuffer);
    }

    /* 保留较大者，将较小者减为不超过原值且能整除较大者的最大整数 */
    uint32_t hi = dyBlocks >= wBlocks ? dyBlocks : wBlocks;
    uint32_t lo = dyBlocks >= wBlocks ? wBlocks : dyBlocks;
    if (hi == 0) {
        hi = 1;
    }
    while (lo > 1 && hi % lo != 0) {
        --lo;
    }
    if (lo == 0) {
        lo = 1;
    }
    const uint32_t dyBlocks2 = dyBlocks >= wBlocks ? hi : lo;
    const uint32_t wBlocks2 = dyBlocks >= wBlocks ? lo : hi;

    uint64_t ka = CeilDivU(static_cast<uint64_t>(dyBlocks2) * c.windowSpan, c.baseK);
    const uint64_t kaCap = 65535u / c.baseK;
    if (ka < 1) {
        ka = 1;
    }
    if (ka > kaCap) {
        ka = kaCap > 0 ? kaCap : 1;
    }
    uint32_t kb = CeilDivU(static_cast<uint64_t>(wBlocks2) * c.windowSpan, c.baseK);
    if (kb < 1) {
        kb = 1;
    }
    /* 较大 step 向下对齐到较小 step 的整数倍 */
    if (ka > kb) {
        ka = (ka / kb) * kb;
        if (ka < 1) {
            ka = kb;
        }
    } else if (kb > ka) {
        kb = (kb / ka) * ka;
        if (kb < 1) {
            kb = ka;
        }
    }
    c.stepKa = static_cast<uint32_t>(ka);
    c.stepKb = kb;
    if (IsL1ParamsValid(c)) {
        return true;
    }
    c.stepKa = 0;
    c.stepKb = 0;
    return false;
}

/* 首次求解：先阶梯候选，不合法再试等分（容量回调沿用同一顺序） */
bool CalStepK(SolveCtx& c)
{
    if (LadderMatchStep(c)) {
        return true;
    }
    return EqualL1MatchStep(c);
}

/* ---------------- LegalProtection（tiling.md §7 容量回调） ---------------- */

bool LegalProtection(SolveCtx& c)
{
    const uint32_t savedBaseK = c.baseK;
    const uint32_t savedBaseM = c.baseM;
    const uint32_t savedBaseN = c.baseN;
    const uint32_t savedAl1 = c.al1Pbuffer;
    const uint32_t savedBl1 = c.bl1Pbuffer;

    /* 1. 减小 K：从当前 baseK-C0 起按 C0 递减至 C0，逐个重选驻留 */
    for (uint32_t k = (c.baseK > CONV3DT_C0) ? c.baseK - CONV3DT_C0 : CONV3DT_C0;; k -= CONV3DT_C0) {
        c.baseK = k;
        if (CalStepK(c)) {
            return true;
        }
        if (k == CONV3DT_C0) {
            break;
        }
    }
    c.baseK = CONV3DT_C0;
    /* 2. 减小 M/N：交替收缩（各自至少保留一个分形），逐候选重选驻留 */
    for (int guard = 0; guard < 256; ++guard) {
        const bool canShrinkM = c.baseM > CONV3DT_FRACTAL && (c.baseM > c.wi || c.wi <= CONV3DT_FRACTAL);
        const bool canShrinkN = c.baseN > CONV3DT_FRACTAL;
        if (canShrinkM && (c.baseM > c.baseN || !canShrinkN)) {
            c.baseM -= CONV3DT_FRACTAL;
        } else if (canShrinkN) {
            c.baseN -= CONV3DT_FRACTAL;
        } else {
            break;
        }
        if (CalStepK(c)) {
            return true;
        }
    }
    /* 3. 恢复较大 K：重算窗口配对 maxK，从 maxK-C0 向下重新搜索 */
    c.baseM = savedBaseM;
    c.baseN = savedBaseN;
    AdjustBaseMNK(c, savedBaseK);
    if (CalStepK(c)) {
        return true;
    }
    /* 4. 调整份数：(2,2) → (2,1) → (1,2) → (1,1)，每次重新执行容量检查 */
    c.baseM = savedBaseM;
    c.baseN = savedBaseN;
    c.baseK = savedBaseK;
    struct PBufPair {
        uint32_t a;
        uint32_t b;
    };
    const PBufPair combos[] = {{2, 1}, {1, 2}, {1, 1}};
    for (const auto& pb : combos) {
        c.al1Pbuffer = pb.a;
        c.bl1Pbuffer = pb.b;
        if (CalStepK(c)) {
            return true;
        }
    }
    c.al1Pbuffer = savedAl1;
    c.bl1Pbuffer = savedBl1;
    return false;
}

/* ---------------- SetSingleCoreInfo（tiling.md §8 公共任务恢复） ---------------- */

void SetSingleCoreInfo(SolveCtx& c)
{
    const uint32_t maxCores = c.caps.maxCores;
    uint32_t spaceParts = CeilDivU(static_cast<uint64_t>(c.hi) * c.wi, c.baseM);
    uint32_t channelParts = CeilDivU(c.nValue, c.baseN);
    if (spaceParts == 0) {
        spaceParts = 1;
    }
    if (channelParts == 0) {
        channelParts = 1;
    }
    const uint64_t initialTasks = static_cast<uint64_t>(c.batch) * c.di * c.group * spaceParts * channelParts;

    const bool kernelVolOne = c.dkHkWk == 1;
    const uint32_t nPreferred = kernelVolOne ? 256u : 128u;
    const uint32_t mPreferred = kernelVolOne ? 256u : 512u;

    /* 通道划分：任务数不超核数且 channelParts 与核数互有整除关系，或 baseN 超阈值时均分通道 */
    const bool dividesEither = (maxCores % channelParts == 0) || (channelParts % maxCores == 0);
    if ((initialTasks <= maxCores && dividesEither) || c.baseN > nPreferred) {
        uint32_t newBaseN = AlignUpU(c.nValue / channelParts, CONV3DT_FRACTAL);
        if (newBaseN == 0) {
            newBaseN = CONV3DT_FRACTAL;
        }
        if (newBaseN > c.nValue) {
            newBaseN = c.nValue;
        }
        c.baseN = newBaseN;
        channelParts = CeilDivU(c.nValue, c.baseN);
    }
    c.singleCoreCin = c.baseN;
    c.singleCoreCout = c.coutG;
    c.singleCoreDin = 1;

    /* 空间划分 */
    c.singleCoreM = c.baseM;
    const uint64_t mTotal = static_cast<uint64_t>(c.hi) * c.wi;
    if (initialTasks <= maxCores) {
        uint32_t whole = AlignUpU(mTotal / spaceParts, c.wi);
        if (whole == 0) {
            whole = c.baseM;
        }
        c.singleCoreM = whole;
    } else if (c.baseM >= mPreferred) {
        const uint32_t wholeRowM = (c.baseM / c.wi > 0 ? c.baseM / c.wi : 1) * c.wi;
        if (static_cast<uint64_t>(wholeRowM) * spaceParts >= c.mValue) {
            c.singleCoreM = wholeRowM;
        }
    }
    if (c.singleCoreM == 0) {
        c.singleCoreM = c.baseM;
    }
    if (c.singleCoreM < c.baseM) {
        c.baseM = AlignUpU(c.singleCoreM, CONV3DT_FRACTAL);
        if (c.baseM == 0) {
            c.baseM = CONV3DT_FRACTAL;
        }
    }
}

/* ---------------- loadB2 判定（tilingkey-templates §2） ---------------- */

void SetTilingCondition(SolveCtx& c)
{
    const uint64_t kernelHW = static_cast<uint64_t>(c.hk) * c.wk;
    const uint64_t kernelDHW = static_cast<uint64_t>(c.dk) * kernelHW;
    if (kernelDHW == 1) {
        c.loadB2 = CONV3DT_TPL_TRANSPOSE_ONLY;
    } else if (c.filterNdhwc && c.baseN >= kernelHW) {
        c.loadB2 = CONV3DT_TPL_REVERSE_ONLY;
    } else if (!c.filterNdhwc && CONV3DT_ELEM_BYTES * kernelDHW <= 64 / c.cin) {
        /* 64B 条件用等价除法避免乘积溢出（kernelDHW*2 ≤ 64/cin） */
        c.loadB2 = CONV3DT_TPL_REVERSE_ONLY;
    } else {
        c.loadB2 = CONV3DT_TPL_TRANSPOSE_AND_REVERSE;
    }
}

/* ---------------- bias 实际分配复核（tiling.md §5.1 / bias-bt-path §1） ---------------- */

bool UpdateIsBiasFullLoad(SolveCtx& c)
{
    if (c.hasBias == 0) {
        c.isBiasFullLoad = 0;
        c.biasAllocBytes = 0;
        return true;
    }
    /* 全量条件：任务通道段（singleCoreCin）实际分配可容纳于 BT；否则按当前 N 子块（baseN）分块 */
    const uint64_t fullAlloc = AlignUpU(static_cast<uint64_t>(c.singleCoreCin) * CONV3DT_BIAS_BYTES, 64u);
    c.isBiasFullLoad = (fullAlloc <= c.caps.btBytes) ? 1u : 0u;
    const uint64_t alloc =
        c.isBiasFullLoad != 0 ? fullAlloc : AlignUpU(static_cast<uint64_t>(c.baseN) * CONV3DT_BIAS_BYTES, 64u);
    c.biasAllocBytes = static_cast<uint32_t>(alloc);
    if (alloc > c.caps.btBytes) {
        return false; /* 分块 BT 通路能力不足（baseN*4 > BTBytes），不发布未通过的字段 */
    }
    /* L1 实际范围复核（buffer-and-types §3.1：bias 槽加入总预算） */
    uint32_t aSlot = 0;
    uint32_t bSlot = 0;
    CalcSlotBytes(c, aSlot, bSlot);
    const uint64_t total =
        static_cast<uint64_t>(aSlot) * c.al1Pbuffer + static_cast<uint64_t>(bSlot) * c.bl1Pbuffer + alloc;
    if (total > c.caps.l1Bytes) {
        return false;
    }
    return true;
}

/* ---------------- 发布（tilingkey-templates §3/§4） ---------------- */

void SetTilingData(const SolveCtx& c, Conv3DtTiling* tiling)
{
    Conv3DtTilingData td;
    (void)memset(&td, 0, sizeof(td));
    td.batch = c.batch;
    td.cin = c.cin;
    td.cout = c.cout;
    td.cinG = c.cinG;
    td.coutG = c.coutG;
    td.group = c.group;
    td.di = c.di;
    td.hi = c.hi;
    td.wi = c.wi;
    td.dout = c.dout;
    td.ho = c.ho;
    td.wo = c.wo;
    td.dk = c.dk;
    td.hk = c.hk;
    td.wk = c.wk;
    td.strideD = c.strideD;
    td.strideH = c.strideH;
    td.strideW = c.strideW;
    td.dilationD = c.dilationD;
    td.dilationH = c.dilationH;
    td.dilationW = c.dilationW;
    td.padFront = c.padFront;
    td.padBack = c.padBack;
    td.padUp = c.padUp;
    td.padDown = c.padDown;
    td.padLeft = c.padLeft;
    td.padRight = c.padRight;
    td.backpropPadUp = c.backpropPadUp;
    td.backpropPadDown = c.backpropPadDown;
    td.backpropPadLeft = c.backpropPadLeft;
    td.backpropPadRight = c.backpropPadRight;
    td.c0 = CONV3DT_C0;
    td.mValue = c.mValue;
    td.nValue = c.nValue;
    td.kValue = c.kValue;
    td.kIter = CeilDivU(c.kValue, c.baseK);
    td.tailK = c.kValue - (td.kIter - 1) * c.baseK;
    td.hkWk = c.hkWk;
    td.dkHkWk = c.dkHkWk;
    td.khDilation = c.khDilation;
    td.kwDilation = c.kwDilation;
    td.hoExpand = c.hoExpand;
    td.woExpand = c.woExpand;
    td.doHoWo = c.dout * c.ho * c.wo;
    td.hiWi = c.hi * c.wi;
    td.diHiWi = c.di * c.hi * c.wi;
    td.baseM = c.baseM;
    td.baseN = c.baseN;
    td.baseK = c.baseK;
    td.singleCoreM = c.singleCoreM;
    td.singleCoreCin = c.singleCoreCin;
    td.singleCoreDin = c.singleCoreDin;
    td.singleCoreCout = c.singleCoreCout;
    td.singleCoreBatch = 1;
    td.singleCoreGroup = 1;
    td.singleIterateDk = c.dk;
    td.stepKa = c.stepKa;
    td.stepKb = c.stepKb;
    CalcSlotBytes(c, td.aSlotBytes, td.bSlotBytes);
    td.hasBiasFlag = c.hasBias;
    td.isBiasFullLoad = c.isBiasFullLoad;
    td.biasAllocBytes = c.biasAllocBytes;
    td.al1Pbuffer = c.al1Pbuffer;
    td.bl1Pbuffer = c.bl1Pbuffer;
    td.al0Pbuffer = c.al0Pbuffer;
    td.bl0Pbuffer = c.bl0Pbuffer;
    td.cl0Pbuffer = c.cl0Pbuffer;
    td.initOutputFlag = c.initOutputFlag;
    td.enableFullLoad = c.enableFullLoad;
    td.iterateOrder = 1;
    td.fmtIsNdhwc = c.filterNdhwc;

    const uint64_t mCount = CeilDivU(static_cast<uint64_t>(c.hi) * c.wi, c.singleCoreM);
    const uint64_t nCount = CeilDivU(c.cinG, c.singleCoreCin);
    const uint64_t dCount = CeilDivU(c.di, c.singleCoreDin);
    td.totalTasks = static_cast<uint64_t>(c.batch) * c.group * mCount * nCount * dCount;
    td.coreNum = td.totalTasks < c.caps.maxCores ? static_cast<uint32_t>(td.totalTasks) : c.caps.maxCores;
    if (td.coreNum == 0) {
        td.coreNum = 1;
    }

    tiling->blockDim = td.coreNum;
    tiling->tilingKey = CONV3DT_TILING_KEY(c.loadB2);
    tiling->templateId = c.templateId;
    tiling->loadB2 = c.loadB2;
    tiling->workspaceBytes = 0;
    tiling->initOutputFlag = c.initOutputFlag;
    tiling->hasBiasFlag = c.hasBias;
    tiling->reserved[0] = 0;
    (void)memcpy(tiling->payload, &td, sizeof(td));
}

/* ---------------- 模板准入（tiling.md §2 优先序） ---------------- */

bool SmallShapeIsCapable(const SolveCtx& c)
{
    /* estimatedTasks=ceilDiv(mValue*nValue*accBytes,L0CBytes)*N*Di*G；K 上限为入口条件 */
    if (c.kValue > 32768u) {
        return false;
    }
    const uint64_t perL0c = CeilDivU(static_cast<uint64_t>(c.mValue) * c.nValue * CONV3DT_ACC_BYTES, c.caps.l0cBytes);
    const uint64_t estimatedTasks = perL0c * c.batch * c.di * c.group;
    if (estimatedTasks >= static_cast<uint64_t>(c.caps.maxCores) * 5 / 2) {
        return false;
    }
    return estimatedTasks % c.caps.maxCores != 0;
}

bool L1ResidentIsCapable(const SolveCtx& c)
{
    /* 几何条件：Dk=1、mValue>nValue、Wi<=512 */
    if (c.dk != 1 || c.mValue <= c.nValue || c.wi > 512) {
        return false;
    }
    /* 收益条件：mValue*nValue < (mValue+nValue)*512 或 kValue<=128 */
    const bool benefit =
        (static_cast<uint64_t>(c.mValue) * c.nValue < (static_cast<uint64_t>(c.mValue) + c.nValue) * 512) ||
        c.kValue <= 128u;
    if (!benefit) {
        return false;
    }
    /* 容量初筛：kValue*min(nPreferred,nValue)*weightBytes*2 <= L1Bytes（为优选通道块预留） */
    const uint32_t nPreferred = (c.dkHkWk == 1) ? 256u : 128u;
    const uint32_t nMin = nPreferred < c.nValue ? nPreferred : c.nValue;
    return static_cast<uint64_t>(c.kValue) * nMin * CONV3DT_ELEM_BYTES * 2 <= c.caps.l1Bytes;
}

/* ---------------- 三模板求解 ---------------- */

void InitBaseMNKCommon(SolveCtx& c)
{
    /* 核体积 1 起点 (256,256,64)，其余 (512,128,32)；两路输入双 Buffer，部分和单 Buffer */
    if (c.dkHkWk == 1) {
        c.baseM = 256;
        c.baseN = 256;
        c.baseK = 64;
    } else {
        c.baseM = 512;
        c.baseN = 128;
        c.baseK = 32;
    }
    c.al0Pbuffer = 2;
    c.bl0Pbuffer = 2;
    c.cl0Pbuffer = 1;
    c.al1Pbuffer = 2;
    c.bl1Pbuffer = 2;
    c.enableFullLoad = 0;
}

bool SolveStandard(SolveCtx& c)
{
    /* standard §4：InitBaseMNK → InitL1Params → SetSingleCoreInfo → CalStepK → LegalProtection */
    InitBaseMNKCommon(c);
    const uint32_t incomingK = c.baseK;
    AdjustBaseMNK(c, incomingK);
    SetSingleCoreInfo(c);
    if (!CalStepK(c) || !IsL1ParamsValid(c)) {
        if (!LegalProtection(c)) {
            return false;
        }
        SetSingleCoreInfo(c);
        if (!IsL1ParamsValid(c)) {
            return false;
        }
    }
    SetTilingCondition(c);
    c.templateId = CONV3DT_TPL_INNER_PRODUCT;
    return true;
}

bool SolveSmallShape(SolveCtx& c)
{
    /* small-shape §4：InitBaseMNK → InitL1Params → AdjustSingleCoreAndL0Info → CalStepK → 保护 */
    InitBaseMNKCommon(c);
    uint32_t incomingK = c.baseK;
    AdjustBaseMNK(c, incomingK);

    /* 均衡搜索（small-shape §2）：分段数范围含两端，(rounds, tasks, loadBytes) 评价量 */
    const uint32_t maxM = (c.nValue <= CONV3DT_FRACTAL) ? 512u : 1024u;
    const uint32_t maxN = (c.mValue <= CONV3DT_FRACTAL) ? 512u : 1024u;
    const uint64_t mTotal = static_cast<uint64_t>(c.hi) * c.wi;
    const uint32_t spacePartsMin = CeilDivU(mTotal, maxM) < 1 ? 1 : CeilDivU(mTotal, maxM);
    const uint32_t spacePartsMax = CeilDivU(mTotal, 64u) < 1 ? 1 : CeilDivU(mTotal, 64u);
    const uint32_t channelPartsMin = CeilDivU(c.nValue, maxN) < 1 ? 1 : CeilDivU(c.nValue, maxN);
    const uint32_t channelPartsMax = CeilDivU(c.nValue, 64u) < 1 ? 1 : CeilDivU(c.nValue, 64u);

    uint32_t bestM = c.baseM;
    uint32_t bestN = c.baseN;
    uint32_t bestTaskM = c.baseM;
    uint64_t bestTasks = 0;
    uint64_t bestRounds = 0xFFFFFFFFFFFFFFFFull;
    uint64_t bestLoad = 0xFFFFFFFFFFFFFFFFull;
    bool hasBest = false;
    for (uint32_t sp = spacePartsMin;; ++sp) {
        const uint32_t calcM0 = AlignUpU(CeilDivU(mTotal, sp), CONV3DT_FRACTAL);
        if (calcM0 == 0) {
            if (sp >= spacePartsMax) {
                break;
            }
            continue;
        }
        for (uint32_t cp = channelPartsMin;; ++cp) {
            const uint32_t calcN0 = AlignUpU(CeilDivU(c.nValue, cp), CONV3DT_FRACTAL);
            if (calcN0 == 0) {
                if (cp >= channelPartsMax) {
                    break;
                }
                continue;
            }
            uint32_t taskM = calcM0;
            const uint32_t wholeRowM = (calcM0 / c.wi > 0 ? calcM0 / c.wi : 1) * c.wi;
            if (CeilDivU(mTotal, wholeRowM) == CeilDivU(mTotal, calcM0)) {
                taskM = wholeRowM; /* 整行候选与 calcM 同段数时取整行 */
            }
            const uint32_t calcM1 = taskM < calcM0 ? AlignUpU(taskM, CONV3DT_FRACTAL) : calcM0;
            /* 单份部分和容量约束：calcM*calcN*accBytes ≤ L0C */
            if (static_cast<uint64_t>(calcM1) * calcN0 * CONV3DT_ACC_BYTES > c.caps.l0cBytes) {
                if (cp >= channelPartsMax) {
                    break;
                }
                continue;
            }
            const uint64_t tasks =
                static_cast<uint64_t>(c.batch) * c.di * c.group * CeilDivU(mTotal, taskM) * CeilDivU(c.nValue, calcN0);
            const uint64_t rounds = CeilDivU(tasks, c.caps.maxCores);
            const uint32_t rows = ResidentRows(c, taskM);
            const uint64_t loadBytes = static_cast<uint64_t>(CONV3DT_ELEM_BYTES) * calcN0 * c.hkWk * c.cout +
                                       static_cast<uint64_t>(CONV3DT_ELEM_BYTES) * rows * (c.wo * c.strideW) * c.cout;
            if (!hasBest || rounds < bestRounds ||
                (rounds == bestRounds && tasks >= bestTasks && loadBytes <= bestLoad)) {
                hasBest = true;
                bestRounds = rounds;
                bestTasks = tasks;
                bestLoad = loadBytes;
                bestM = calcM1;
                bestN = calcN0;
                bestTaskM = taskM;
            }
            if (cp >= channelPartsMax) {
                break;
            }
        }
        if (sp >= spacePartsMax) {
            break;
        }
    }
    if (hasBest) {
        c.baseM = bestM;
        c.baseN = bestN;
    }
    /* 搜索完成后按公共 AdjustBaseMNK 收敛；任务通道/深度字段按文档取 baseN/coutG/1 */
    AdjustBaseMNK(c, incomingK);
    incomingK = c.baseK;
    if (!CalStepK(c) || !IsL1ParamsValid(c)) {
        if (!LegalProtection(c)) {
            return false;
        }
        SetSingleCoreInfo(c);
        if (!IsL1ParamsValid(c)) {
            return false;
        }
    } else {
        c.singleCoreCin = c.baseN;
        c.singleCoreCout = c.coutG;
        c.singleCoreDin = 1;
        c.singleCoreM = hasBest ? bestTaskM : c.baseM;
        if (c.singleCoreM < c.baseM) {
            c.baseM = AlignUpU(c.singleCoreM, CONV3DT_FRACTAL);
            if (c.baseM == 0) {
                c.baseM = CONV3DT_FRACTAL;
            }
        }
    }
    SetTilingCondition(c);
    c.templateId = CONV3DT_TPL_SMALL_SHAPE;
    return true;
}

bool LadderMatchStepKWithFullLoad(SolveCtx& c)
{
    /* l1-resident §4：stepKb=ceilDiv(kValue,baseK)；stepKa 从 stepKb 递减，首个
     * 同时满足 L1 合法与 stepKb%stepKa==0 的值 */
    c.stepKb = CeilDivU(c.kValue, c.baseK);
    if (c.stepKb == 0) {
        c.stepKb = 1;
    }
    for (uint32_t ka = c.stepKb; ka >= 1; --ka) {
        if (c.stepKb % ka != 0) {
            continue;
        }
        c.stepKa = ka;
        if (IsL1ParamsValid(c)) {
            return true;
        }
    }
    c.stepKa = 0;
    c.stepKb = 0;
    return false;
}

bool SolveFullLoad(SolveCtx& c)
{
    /* l1-resident §2：K 越长 N 起点越小；M 受部分和与操作数容量限制；K 由 L0B 容量反推 */
    if (c.kValue > 1024u) {
        c.baseN = 64;
    } else if (c.kValue > 512u) {
        c.baseN = 128;
    } else if (c.kValue > 256u) {
        c.baseN = 256;
    } else {
        c.baseN = 512;
    }
    c.al0Pbuffer = 2;
    c.bl0Pbuffer = 2;
    c.cl0Pbuffer = 1;
    if (c.kValue <= 128u && (c.nValue >= 2 * c.kValue || c.nValue >= 128u)) {
        c.cl0Pbuffer = 2;
    }
    c.al1Pbuffer = 2;
    c.bl1Pbuffer = 1;
    c.enableFullLoad = 1;

    c.baseM =
        static_cast<uint32_t>(c.caps.l0cBytes / (static_cast<uint64_t>(c.baseN) * CONV3DT_ACC_BYTES * c.cl0Pbuffer));
    const uint32_t mCap = static_cast<uint32_t>(c.caps.l0aBytes / (CONV3DT_C0 * CONV3DT_ELEM_BYTES * c.al0Pbuffer));
    c.baseM = c.baseM < mCap ? c.baseM : mCap;
    c.baseK =
        static_cast<uint32_t>(c.caps.l0bBytes / (static_cast<uint64_t>(c.baseN) * CONV3DT_ELEM_BYTES * c.bl0Pbuffer));
    c.baseM = c.baseM < c.mValue ? c.baseM : c.mValue;

    /* §3 任务划分：先按通道段数均分 baseN，再整行任务搜索/公因子划分 */
    uint32_t channelParts = CeilDivU(c.nValue, c.baseN);
    if (channelParts == 0) {
        channelParts = 1;
    }
    c.baseN = AlignUpU(c.nValue / channelParts, CONV3DT_FRACTAL);
    if (c.baseN == 0) {
        c.baseN = CONV3DT_FRACTAL;
    }
    if (c.baseN > c.nValue) {
        c.baseN = c.nValue;
    }
    c.singleCoreCin = c.baseN;
    c.singleCoreCout = c.coutG;
    c.singleCoreDin = 1;
    c.singleCoreM = c.baseM;

    const uint64_t mTotal = static_cast<uint64_t>(c.hi) * c.wi;
    const uint64_t hiFactor = static_cast<uint64_t>(c.batch) * c.di * c.group * channelParts * c.hi;
    if (hiFactor != 0 && hiFactor % c.caps.maxCores == 0) {
        /* 整行任务数能被核数整除：按公因子直接划分 */
        const uint64_t gcdv = GcdU(static_cast<uint64_t>(c.batch) * c.group * channelParts, c.caps.maxCores);
        const uint32_t remainingFactor = static_cast<uint32_t>(c.caps.maxCores / (gcdv == 0 ? 1 : gcdv));
        const uint32_t depthFactor = static_cast<uint32_t>(GcdU(c.di, remainingFactor));
        c.singleCoreDin = depthFactor == 0 ? 1 : static_cast<uint32_t>(c.di / depthFactor);
        const uint64_t taskM = (static_cast<uint64_t>(c.hi) * depthFactor / remainingFactor) * c.wi;
        c.singleCoreM = static_cast<uint32_t>(taskM == 0 ? c.baseM : taskM);
    } else {
        c.singleCoreDin = 1;
        /* 递增分段数搜索整行候选：槽位占用 ≥ 80%，代价严格下降才替换 */
        const uint32_t partsMax = CeilDivU(mTotal, c.baseM);
        uint64_t bestCost = 0xFFFFFFFFFFFFFFFFull;
        for (uint32_t parts = 1; parts <= partsMax; ++parts) {
            const uint64_t taskM = static_cast<uint64_t>(CeilDivU(c.hi, parts)) * c.wi;
            const uint64_t tasks = static_cast<uint64_t>(c.batch) * c.di * c.group * channelParts *
                                   CeilDivU(mTotal, static_cast<uint32_t>(taskM));
            const uint64_t slots = AlignUpU(tasks, c.caps.maxCores);
            if (tasks * 5 < slots * 4) {
                continue;
            }
            const uint64_t cost =
                CeilDivU(tasks, c.caps.maxCores) * (1 + CeilDivU(static_cast<uint32_t>(taskM), c.baseM));
            if (cost < bestCost) {
                bestCost = cost;
                c.singleCoreM = static_cast<uint32_t>(taskM);
            }
        }
        if (c.singleCoreM == 0) {
            c.singleCoreM = c.baseM;
        }
    }
    if (c.singleCoreM < c.baseM) {
        c.baseM = AlignUpU(c.singleCoreM, CONV3DT_FRACTAL);
        if (c.baseM == 0) {
            c.baseM = CONV3DT_FRACTAL;
        }
    }
    AdjustBaseMNK(c, c.baseK);
    /* 完整驻留前提（compute-flow §8.4）：L1B 覆盖任务全部 N——baseN 低于任务通道段时收窄任务 */
    if (c.baseN < c.singleCoreCin) {
        c.singleCoreCin = c.baseN;
    }

    if (!LadderMatchStepKWithFullLoad(c)) {
        if (!LegalProtection(c)) {
            return false;
        }
        /* 回调后仍须满足完整 K 驻留条件：stepKb*baseK>=kValue 且 stepKa 整除 stepKb */
        if (static_cast<uint64_t>(c.stepKb) * c.baseK < c.kValue || c.stepKb % c.stepKa != 0) {
            return false;
        }
        if (c.baseN < c.singleCoreCin) {
            c.singleCoreCin = c.baseN;
        }
    }
    SetTilingCondition(c);
    c.templateId = CONV3DT_TPL_FULL_LOAD;
    return true;
}

} /* namespace */

/* ---------------- 对外入口（conv3dtranspose_launch.h ABI） ---------------- */

extern "C" int Conv3DtSolveTiling(const struct Conv3DtSpec* spec, struct Conv3DtTiling* tiling)
{
    if (spec == nullptr || tiling == nullptr) {
        return -1;
    }
    (void)memset(tiling, 0, sizeof(*tiling));

    SolveCtx c;
    const int norm = ValidateAndNormalize(*spec, c);
    if (norm != CONV3DT_OK) {
        return norm;
    }
    if (!QueryPlatformCaps(c.caps)) {
        return -200; /* 平台资源查询失败或必要资源为零（buffer-and-types §1.2） */
    }
    if (c.caps.maxCores > CONV3DT_MAX_CORES) {
        c.caps.maxCores = CONV3DT_MAX_CORES;
    }

    /* 模板选择（tiling.md §2 优先序：small-shape → l1-resident → standard） */
    bool solved = false;
    if (SmallShapeIsCapable(c)) {
        solved = SolveSmallShape(c);
    }
    if (!solved && L1ResidentIsCapable(c)) {
        solved = SolveFullLoad(c);
    }
    if (!solved) {
        solved = SolveStandard(c);
    }
    if (!solved) {
        return -201; /* 资源不足：容量回调后仍无合法计划 */
    }

    /* bias 实际分配复核（tiling.md §5.1：最终 baseN/singleCoreCin 形成后重做；
     * 全量条件与实际容量不一致时降级分块并复核，仍不过则返回资源原因） */
    if (!UpdateIsBiasFullLoad(c)) {
        return -202; /* bias L1/BT 实际分配超限 */
    }

    SetTilingData(c, tiling);
    return CONV3DT_OK;
}
