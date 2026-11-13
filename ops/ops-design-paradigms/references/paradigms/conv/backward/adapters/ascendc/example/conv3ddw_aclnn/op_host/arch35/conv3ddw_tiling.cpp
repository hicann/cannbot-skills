/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include <climits>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/conv3ddw_launch.h"
#include "../../op_kernel/arch35/conv3ddw_struct.h"

namespace {
bool Positive(uint64_t value)
{
    return value > 0 && value <= UINT32_MAX;
}

ge::graphStatus Tiling(gert::TilingContext* ctx)
{
    const auto* x = ctx->GetInputTensor(0);
    const auto* size = ctx->GetInputTensor(1);
    const auto* dy = ctx->GetInputTensor(2);
    const auto* output = ctx->GetOutputDesc(0);
    const auto* shape = ctx->GetOutputShape(0);
    const auto* attrs = ctx->GetAttrs();
    if (x == nullptr || size == nullptr || dy == nullptr || output == nullptr || shape == nullptr || attrs == nullptr ||
        ctx->GetPlatformInfo() == nullptr)
        return ge::GRAPH_FAILED;
    if (x->GetDataType() != dy->GetDataType() ||
        (x->GetDataType() != ge::DT_FLOAT16 && x->GetDataType() != ge::DT_BF16) ||
        size->GetDataType() != ge::DT_INT32 || output->GetDataType() != ge::DT_FLOAT || size->GetShapeSize() != 5 ||
        size->GetStorageFormat() != ge::FORMAT_ND || x->GetStorageFormat() != dy->GetStorageFormat())
        return ge::GRAPH_FAILED;
    auto fmt = x->GetStorageFormat();
    auto ofmt = output->GetStorageFormat();
    if ((fmt != ge::FORMAT_NCDHW && fmt != ge::FORMAT_NDHWC) ||
        (ofmt != ge::FORMAT_NCDHW && ofmt != ge::FORMAT_NDHWC && ofmt != ge::FORMAT_DHWCN))
        return ge::GRAPH_FAILED;
    const auto& xs = x->GetOriginShape();
    const auto& ys = dy->GetOriginShape();
    const auto& os = shape->GetOriginShape();
    if (xs.GetDimNum() != 5 || ys.GetDimNum() != 5 || os.GetDimNum() != 5)
        return ge::GRAPH_FAILED;
    const auto* values = size->GetData<int32_t>();
    if (values == nullptr)
        return ge::GRAPH_FAILED;
    for (size_t i = 0; i < 5; ++i) {
        if (!Positive(xs.GetDim(i)) || !Positive(ys.GetDim(i)) || !Positive(os.GetDim(i)) || values[i] != os.GetDim(i))
            return ge::GRAPH_FAILED;
    }
    const auto* strides = attrs->GetListInt(0);
    const auto* pads = attrs->GetListInt(1);
    const auto* dilations = attrs->GetListInt(2);
    const auto* groups = attrs->GetInt(3);
    const auto* hf32 = attrs->GetBool(5);
    const auto* format = attrs->GetStr(4);
    const auto* outputPadding = attrs->GetListInt(6);
    if (!strides || !pads || !dilations || !groups || !hf32 || !format || !outputPadding || strides->GetSize() != 5 ||
        pads->GetSize() != 6 || dilations->GetSize() != 5 || outputPadding->GetSize() != 5 || *hf32 || *groups != 1 ||
        std::strcmp(format, fmt == ge::FORMAT_NDHWC ? "NDHWC" : "NCDHW") != 0)
        return ge::GRAPH_FAILED;
    for (size_t i = 0; i < 5; ++i)
        if (!Positive(strides->GetData()[i]) || !Positive(dilations->GetData()[i]) || outputPadding->GetData()[i] != 0)
            return ge::GRAPH_FAILED;
    for (size_t i = 0; i < 6; ++i)
        if (pads->GetData()[i] < 0 || pads->GetData()[i] > UINT32_MAX)
            return ge::GRAPH_FAILED;
    const int c = fmt == ge::FORMAT_NDHWC ? 4 : 1;
    const int d = fmt == ge::FORMAT_NDHWC ? 1 : 2;
    if (strides->GetData()[0] != 1 || strides->GetData()[c] != 1 || dilations->GetData()[0] != 1 ||
        dilations->GetData()[c] != 1 || xs.GetDim(0) != ys.GetDim(0))
        return ge::GRAPH_FAILED;
    Conv3DdwSpec s{};
    s.batch = xs.GetDim(0);
    s.cin = xs.GetDim(c);
    s.cout = ys.GetDim(c);
    s.di = xs.GetDim(d);
    s.hi = xs.GetDim(d + 1);
    s.wi = xs.GetDim(d + 2);
    s.dout = ys.GetDim(d);
    s.ho = ys.GetDim(d + 1);
    s.wo = ys.GetDim(d + 2);
    s.sd = strides->GetData()[d];
    s.sh = strides->GetData()[d + 1];
    s.sw = strides->GetData()[d + 2];
    s.dd = dilations->GetData()[d];
    s.dh = dilations->GetData()[d + 1];
    s.dw = dilations->GetData()[d + 2];
    s.padFront = pads->GetData()[0];
    s.padBack = pads->GetData()[1];
    s.padUp = pads->GetData()[2];
    s.padDown = pads->GetData()[3];
    s.padLeft = pads->GetData()[4];
    s.padRight = pads->GetData()[5];
    s.groups = *groups;
    int kd = ofmt == ge::FORMAT_NCDHW ? 2 : ofmt == ge::FORMAT_NDHWC ? 1 : 0;
    int ci = ofmt == ge::FORMAT_NCDHW ? 1 : ofmt == ge::FORMAT_NDHWC ? 4 : 3;
    int co = ofmt == ge::FORMAT_DHWCN ? 4 : 0;
    if (os.GetDim(co) != s.cout || os.GetDim(ci) != s.cin)
        return ge::GRAPH_FAILED;
    s.dk = os.GetDim(kd);
    s.hk = os.GetDim(kd + 1);
    s.wk = os.GetDim(kd + 2);
    s.dtype = x->GetDataType() == ge::DT_BF16 ? CONV3DDW_BF16 : CONV3DDW_FP16;
    s.xFormat = fmt == ge::FORMAT_NDHWC ? CONV3DDW_NDHWC : CONV3DDW_NCDHW;
    s.yFormat = ofmt == ge::FORMAT_NDHWC ? CONV3DDW_NDHWC : ofmt == ge::FORMAT_DHWCN ? CONV3DDW_DHWCN : CONV3DDW_NCDHW;
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    Conv3DdwPlatformCaps caps{};
    caps.aicNum = platform.GetCoreNumAic();
    caps.aivNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, caps.l0aBytes);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, caps.l0bBytes);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, caps.l0cBytes);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, caps.l1Bytes);
    Conv3DdwTilingData t{};
    char error[256]{};
    if (Conv3DdwSolveTilingWithCaps(&s, &caps, &t, error, sizeof(error)) != 0)
        return ge::GRAPH_FAILED;
    auto* raw = ctx->GetRawTilingData();
    if (raw == nullptr || raw->GetCapacity() < sizeof(t))
        return ge::GRAPH_FAILED;
    std::memcpy(raw->GetData(), &t, sizeof(t));
    raw->SetDataSize(sizeof(t));
    ctx->SetBlockDim(t.usedCoreNum);
    ctx->SetScheduleMode(1);
    ctx->SetTilingKey(GET_TPL_TILING_KEY(2));
    auto* workspace = ctx->GetWorkspaceSizes(1);
    if (workspace == nullptr)
        return ge::GRAPH_FAILED;
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(Conv3ddw).Tiling(Tiling);
