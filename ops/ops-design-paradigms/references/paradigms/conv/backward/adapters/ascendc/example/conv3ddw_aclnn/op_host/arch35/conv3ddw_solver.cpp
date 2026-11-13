/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define CONV3DDW_HOST_ONLY

/*!
 * \file conv3ddw_tiling.cpp
 * \brief MN_StreamK Host 求解：普通整窗、不切宽、原组、无跨核 K 归约。
 *
 * 规则对应 filter/tiling：InitBaseBlock -> SetStepK4SplitMN -> 小任务调整 -> 最终检查。
 * 平台查询和纯求解分开，直调与注册共用纯求解；不依赖参考算子仓的头文件或库。
 */

#include "conv3ddw_launch.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

#ifndef CONV3DDW_HOST_ONLY
#include "acl/acl.h"
#include "platform/platform_info.h"
#endif

namespace {

uint64_t CeilDiv(uint64_t x, uint64_t y)
{
    return x / y + (x % y != 0);
}
uint64_t AlignUp(uint64_t x, uint64_t a)
{
    return CeilDiv(x, a) * a;
}
uint64_t AlignDown(uint64_t x, uint64_t a)
{
    return x / a * a;
}

int Reject(char* error, uint32_t size, const char* message)
{
    if (error != nullptr && size != 0)
        std::snprintf(error, size, "%s", message);
    return -1;
}

bool CheckedProduct(const uint32_t* values, size_t count, uint64_t limit)
{
    uint64_t product = 1;
    for (size_t i = 0; i < count; ++i) {
        if (values[i] == 0 || product > limit / values[i])
            return false;
        product *= values[i];
    }
    return true;
}

bool CheckAxis(uint32_t input, uint32_t output, uint32_t kernel, uint32_t stride, uint32_t dilation, uint32_t front,
               uint32_t back)
{
    if (input == 0 || output == 0 || kernel == 0 || stride == 0 || dilation == 0)
        return false;
    const uint64_t effective = uint64_t(kernel - 1) * dilation + 1;
    const uint64_t padded = uint64_t(input) + front + back;
    return padded >= effective && (padded - effective) / stride + 1 == output;
}

struct Solver {
    Conv3DdwTilingData td{};
    const Conv3DdwPlatformCaps& cap;
    uint64_t mValue, nValue, kValue, windowSpan;

    explicit Solver(const Conv3DdwPlatformCaps& c)
        : cap(c)
    {}

    uint32_t GetBaseK(uint32_t m, uint32_t n) const
    {
        uint64_t k =
            std::min(AlignDown(cap.l0aBytes / (uint64_t(m) * 4), 16), AlignDown(cap.l0bBytes / (uint64_t(n) * 4), 16));
        if (AlignUp(kValue, 16) < k)
            k = AlignUp(kValue, 16);
        else if (td.wo < k && td.wo % 16 == 0)
            k = AlignDown(k, td.wo);
        return static_cast<uint32_t>(k);
    }

    bool UpdateBounds()
    {
        const uint64_t aBound = uint64_t(td.stepKa) * td.baseK * td.baseM;
        const uint64_t k = uint64_t(td.stepKb) * td.baseK;
        uint64_t rows;
        if (k % td.wo == 0 || td.wo % k == 0)
            rows = CeilDiv(k, td.wo);
        else
            rows = k > td.wo ? k / td.wo + 2 : 2;
        const uint64_t hi = (rows - 1) * td.strideH + uint64_t(td.hk - 1) * td.dilationH + 1;
        const uint64_t nBlocks = CeilDiv(td.baseN, 16);
        const uint64_t hw = uint64_t(td.hk) * td.wk;
        uint64_t copies = CeilDiv(nBlocks, hw);
        if (hw > nBlocks && hw % nBlocks != 0)
            ++copies;
        else if (2 * nBlocks % hw != 0)
            ++copies;
        const uint64_t channels = std::max(uint64_t(td.baseN) / windowSpan, uint64_t(1)) * 16;
        const uint64_t bBound = hi * td.wi * std::min(channels, copies * 16);
        if (aBound > UINT32_MAX || bBound > UINT32_MAX)
            return false;
        td.al1Bound = static_cast<uint32_t>(aBound);
        td.bl1Bound = static_cast<uint32_t>(bBound);
        return (aBound * td.al1Pbuffer + bBound * td.bl1Pbuffer) * 2 <= cap.l1Bytes;
    }

    bool SetStepK4SplitMN()
    {
        td.al1Pbuffer = td.bl1Pbuffer = 2;
        bool offDoubleBuffer = false;
        for (int depth = 16; depth >= 1;) {
            if (depth < 2)
                td.al1Pbuffer = td.bl1Pbuffer = 1;
            const uint64_t maxIter = CeilDiv(kValue, td.baseK);
            td.stepKa =
                static_cast<uint32_t>(std::max(std::min(uint64_t(depth) / td.al1Pbuffer, maxIter), uint64_t(1)));
            td.stepKb =
                static_cast<uint32_t>(std::max(std::min(uint64_t(depth) / td.bl1Pbuffer, maxIter), uint64_t(1)));
            if (td.stepKa > td.stepKb)
                td.stepKa = td.stepKa / td.stepKb * td.stepKb;
            if (td.stepKb > td.stepKa)
                td.stepKb = td.stepKb / td.stepKa * td.stepKa;
            if (UpdateBounds()) {
                td.singleCoreM = td.baseM;
                td.singleCoreN =
                    static_cast<uint32_t>(std::max(uint64_t(td.baseN) / windowSpan, uint64_t(1)) * windowSpan);
                return true;
            }
            depth -= depth > 2 ? 2 : 1;
            if (depth <= 2 && !offDoubleBuffer) {
                offDoubleBuffer = true;
                td.al1Pbuffer = td.bl1Pbuffer = 1;
                depth = 16;
            }
        }
        return false;
    }

    bool UpdateL0()
    {
        const uint64_t cBytes = uint64_t(td.baseM) * td.baseN * sizeof(float);
        td.cl0Pbuffer = cBytes * 2 <= cap.l0cBytes ? 2 : 1;
        return td.baseK > 0 && cBytes <= cap.l0cBytes;
    }

    uint64_t TaskCount() const
    {
        return CeilDiv(mValue, td.singleCoreM) * CeilDiv(nValue, td.singleCoreN) * td.dk * td.realGroup;
    }

    bool AdjustSmallCaseBaseBlock()
    {
        uint64_t total = TaskCount();
        if (total >= cap.aicNum)
            return true;
        const uint64_t splitCores = cap.aicNum / total;
        if (cap.aicNum % total == 0 && (td.singleCoreBatchDout >= splitCores || td.ho >= splitCores))
            return true;
        const uint64_t nCount = CeilDiv(nValue, td.singleCoreN);
        const uint64_t maxMCount = std::min(CeilDiv(mValue, 16), uint64_t(cap.aicNum)) / td.dk / nCount;
        for (uint64_t count = maxMCount; count > 0; --count) {
            const uint64_t m = AlignUp(CeilDiv(mValue, count), 16);
            total = CeilDiv(mValue, m) * nCount * td.dk;
            if (total > cap.aicNum || cap.aicNum % total != 0)
                continue;
            const uint32_t k = GetBaseK(m, td.baseN);
            const uint64_t streamKDim = cap.aicNum / total;
            if (td.singleCoreBatchDout < streamKDim && k < streamKDim * td.wo)
                continue;
            if (m * td.baseN * 4 > cap.l0cBytes)
                break;
            const auto saved = td;
            td.baseM = static_cast<uint32_t>(m);
            td.baseK = k;
            if (UpdateL0() && SetStepK4SplitMN())
                break;
            td = saved;
        }
        const uint64_t mCount = CeilDiv(mValue, td.singleCoreM);
        total = TaskCount();
        if (total <= cap.aicNum / 2 || total >= cap.aicNum || td.baseN < td.baseM)
            return true;
        const uint64_t maxNCount = std::min(CeilDiv(nValue, windowSpan), uint64_t(cap.aicNum) / td.dk / mCount);
        const uint64_t oldNCount = CeilDiv(nValue, td.singleCoreN);
        for (uint64_t count = maxNCount; count > oldNCount; --count) {
            const uint64_t n = AlignUp(CeilDiv(nValue, count), windowSpan);
            total = CeilDiv(nValue, n) * mCount * td.dk;
            if (total > cap.aicNum || n * td.baseM * 4 > cap.l0cBytes)
                continue;
            const auto saved = td;
            td.baseN = static_cast<uint32_t>(n);
            td.baseK = GetBaseK(td.baseM, td.baseN);
            if (UpdateL0() && SetStepK4SplitMN())
                break;
            td = saved;
        }
        return true;
    }

    bool Solve()
    {
        mValue = AlignUp(td.cout1G, 16);
        nValue = AlignUp(td.cin1G, 16) * td.hk * td.wk;
        kValue = uint64_t(td.ho) * td.wo;
        windowSpan = uint64_t(td.hk) * td.wk * 16;
        if (nValue > UINT32_MAX || kValue > UINT32_MAX)
            return false;
        td.baseM = static_cast<uint32_t>(AlignUp(mValue > 256 ? mValue / CeilDiv(mValue, 256) : mValue, 16));
        td.baseN = static_cast<uint32_t>(AlignUp(nValue > 256 ? nValue / CeilDiv(nValue, 256) : nValue, 16));
        const uint32_t nAlign = td.hk * td.wk == 9 ? 144 : 16;
        if (td.baseN > nAlign)
            td.baseN = td.baseN / nAlign * nAlign;
        td.baseN = static_cast<uint32_t>(
            std::min(uint64_t(td.baseN), std::max(uint64_t(td.baseN) / windowSpan, uint64_t(1)) * windowSpan));
        td.baseK = GetBaseK(td.baseM, td.baseN);
        td.al0Pbuffer = td.bl0Pbuffer = 2;
        if (!UpdateL0() || !SetStepK4SplitMN() || !AdjustSmallCaseBaseBlock())
            return false;

        /* 当前示例仅实例化整窗、无跨核 K 归约。保留自然选择判断，不强制改写字段。 */
        if (td.baseN % windowSpan != 0)
            return false;
        const uint64_t remainder = TaskCount() % cap.aicNum;
        const bool formatDisablesStreamK = td.yFormat != CONV3DDW_NCDHW && (td.dk != 1 || td.di != 1 || td.dout != 1);
        if (remainder != 0 && remainder <= cap.aicNum / 2 && !formatDisablesStreamK)
            return false;
        td.streamkType = 0;
        td.singleCoreK = static_cast<uint32_t>(kValue);
        td.singleCoreCin = td.singleCoreN / (td.hk * td.wk);
        td.singleCoreCout = td.singleCoreM;
        td.singleCoreHo = td.ho;
        td.iterateOrder = mValue > nValue;
        td.usedCoreNum = cap.aicNum;
        return UpdateBounds();
    }
};

} // namespace

extern "C" int Conv3DdwSolveTilingWithCaps(const Conv3DdwSpec* s, const Conv3DdwPlatformCaps* cap,
                                           Conv3DdwTilingData* out, char* error, uint32_t errorSize)
{
    if (out != nullptr)
        std::memset(out, 0, sizeof(*out));
    if (s == nullptr || cap == nullptr || out == nullptr)
        return Reject(error, errorSize, "null spec/capacity/output");
    if (s->dtype > CONV3DDW_BF16 || s->xFormat > CONV3DDW_NDHWC || s->yFormat > CONV3DDW_DHWCN)
        return Reject(error, errorSize, "unsupported dtype or layout");
    if (s->groups != 1)
        return Reject(error, errorSize, "this example uses original group=1");
    if (cap->aicNum == 0 || uint64_t(cap->aivNum) < uint64_t(cap->aicNum) * 2 || cap->l0aBytes == 0 ||
        cap->l0bBytes == 0 || cap->l0cBytes == 0 || cap->l1Bytes == 0)
        return Reject(error, errorSize, "invalid resources for MIX AIC:AIV=1:2");
    if (!CheckAxis(s->di, s->dout, s->dk, s->sd, s->dd, s->padFront, s->padBack) ||
        !CheckAxis(s->hi, s->ho, s->hk, s->sh, s->dh, s->padUp, s->padDown) ||
        !CheckAxis(s->wi, s->wo, s->wk, s->sw, s->dw, s->padLeft, s->padRight))
        return Reject(error, errorSize, "inconsistent forward convolution geometry");
    const uint32_t xDims[] = {s->batch, s->cin, s->di, s->hi, s->wi};
    const uint32_t dyDims[] = {s->batch, s->cout, s->dout, s->ho, s->wo};
    const uint32_t dwDims[] = {s->cout, s->cin, s->dk, s->hk, s->wk};
    if (!CheckedProduct(xDims, 5, INT64_MAX / 2) || !CheckedProduct(dyDims, 5, INT64_MAX / 2) ||
        !CheckedProduct(dwDims, 5, INT64_MAX / 4))
        return Reject(error, errorSize, "tensor size overflow or zero dimension");
    if (s->hk > 255 || s->wk > 255 || s->dh > 255 || s->dw > 255 || s->sh > 63 || s->sw > 63 ||
        std::max({s->padUp, s->padDown, s->padLeft, s->padRight}) > 255 || uint64_t(s->hk) * s->wk * 16 > 65536 ||
        s->wo > 512 || s->hi > 32767 || s->wi > 32767)
        return Reject(error, errorSize, "plan requires a split-window or width-split implementation");
    Solver solver(*cap);
    auto& t = solver.td;
    t.batch = s->batch;
    t.cin = s->cin;
    t.cout = s->cout;
    t.cin1G = s->cin;
    t.cout1G = s->cout;
    t.di = s->di;
    t.hi = s->hi;
    t.wi = s->wi;
    t.dout = s->dout;
    t.ho = s->ho;
    t.wo = s->wo;
    t.dk = s->dk;
    t.hk = s->hk;
    t.wk = s->wk;
    t.realGroup = s->groups;
    t.group = s->groups;
    t.strideD = s->sd;
    t.strideH = s->sh;
    t.strideW = s->sw;
    t.dilationD = s->dd;
    t.dilationH = s->dh;
    t.dilationW = s->dw;
    t.padFront = s->padFront;
    t.padBack = s->padBack;
    t.padUp = s->padUp;
    t.padDown = s->padDown;
    t.padLeft = s->padLeft;
    t.padRight = s->padRight;
    t.xFormat = s->xFormat;
    t.yFormat = s->yFormat;
    t.dtype = s->dtype;
    t.splitWo = s->wo;
    t.singleCoreBatchDout = uint64_t(s->batch) * s->dout;
    if (!solver.Solve())
        return Reject(error, errorSize, "no supported ordinary MN_StreamK plan for this shape/resources");
    *out = t;
    if (error != nullptr && errorSize != 0)
        error[0] = '\0';
    return 0;
}

#ifndef CONV3DDW_HOST_ONLY
extern "C" int Conv3DdwQueryPlatform(Conv3DdwPlatformCaps* caps)
{
    if (caps == nullptr)
        return -1;
    *caps = {};
    const char* socName = aclrtGetSocName();
    if (socName == nullptr)
        return -1;
    auto& manager = fe::PlatformInfoManager::Instance();
    if (manager.InitializePlatformInfo(socName) != 0)
        return -1;
    fe::PlatformInfo info;
    fe::OptionalInfo optional;
    const int result = manager.GetPlatformInfo(socName, info, optional);
    (void)manager.Finalize();
    if (result != 0)
        return -1;
    caps->aicNum = info.soc_info.ai_core_cnt;
    caps->aivNum = info.soc_info.vector_core_cnt;
    caps->l0aBytes = info.ai_core_spec.l0_a_size;
    caps->l0bBytes = info.ai_core_spec.l0_b_size;
    caps->l0cBytes = info.ai_core_spec.l0_c_size;
    caps->l1Bytes = info.ai_core_spec.l1_size;
    return 0;
}

extern "C" int Conv3DdwSolveTiling(const Conv3DdwSpec* spec, Conv3DdwTilingData* tiling, char* error,
                                   uint32_t errorSize)
{
    Conv3DdwPlatformCaps caps{};
    if (Conv3DdwQueryPlatform(&caps) != 0)
        return Reject(error, errorSize, "platformInfo query failed");
    return Conv3DdwSolveTilingWithCaps(spec, &caps, tiling, error, errorSize);
}
#endif
