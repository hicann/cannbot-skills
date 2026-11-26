/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* DRAFT — 范式参考模板，未编译、未上板 */

/*!
 * \file max_pool_with_argmax_tiling.cpp
 * \brief 正向 host tiling：窗口参数归一化 + 输出推导 + 两步法核间切分
 */

#include <string>
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "../op_kernel/arch35/max_pool_with_argmax_tiling_data.h"
#include "../op_kernel/arch35/max_pool_with_argmax_tiling_key.h"

namespace optiling {

using namespace NsMaxPoolWithArgmax;

static constexpr int64_t PER_CORE_MIN = 1024; // splitting.md §3：下限对齐编组粒度
static constexpr uint32_t DCACHE_SIZE = 128 * 1024;
static constexpr uint32_t STATIC_UB_ESTIMATE = 0; // 正向直除形态，无 UB 数据面

enum AttrPos {
    IDX_KSIZE = 0,
    IDX_STRIDES,
    IDX_PADS,
    IDX_DILATIONS,
    IDX_FORMAT
};

struct MaxPoolWithArgmaxCompileInfo {};

static ge::graphStatus MaxPoolWithArgmaxTilingFunc(gert::TilingContext* context)
{
    // 1. 平台信息（kernel-entrance.md §2：平台参数从接口获取，禁止写死）
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    // 2. 输入 shape 与属性（属性归一化在 host：kernel 只见数值，kernel-entrance.md §2）
    auto xInput = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInput);
    auto xShape = xInput->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() != 4, OP_LOGE(context, "x must be 4D"), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const gert::TypedContinuousVector<int64_t>* ksize = attrs->GetListInt(IDX_KSIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    const gert::TypedContinuousVector<int64_t>* strides = attrs->GetListInt(IDX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    const gert::TypedContinuousVector<int64_t>* pads = attrs->GetListInt(IDX_PADS);
    OP_CHECK_NULL_WITH_CONTEXT(context, pads);
    const gert::TypedContinuousVector<int64_t>* dilations = attrs->GetListInt(IDX_DILATIONS);
    OP_CHECK_NULL_WITH_CONTEXT(context, dilations);
    const char* formatStr = attrs->GetAttrPointer<char>(IDX_FORMAT);
    std::string dataFormat = (formatStr != nullptr) ? std::string(formatStr) : std::string("NCHW");
    OP_CHECK_IF(dataFormat != "NCHW" && dataFormat != "NHWC", OP_LOGE(context, "data_format must be NCHW or NHWC"),
                return ge::GRAPH_FAILED);
    bool isNhwc = (dataFormat == "NHWC");

    int32_t n = static_cast<int32_t>(xShape.GetDim(0));
    int32_t c = isNhwc ? static_cast<int32_t>(xShape.GetDim(3)) : static_cast<int32_t>(xShape.GetDim(1));
    int32_t inH = isNhwc ? static_cast<int32_t>(xShape.GetDim(1)) : static_cast<int32_t>(xShape.GetDim(2));
    int32_t inW = isNhwc ? static_cast<int32_t>(xShape.GetDim(2)) : static_cast<int32_t>(xShape.GetDim(3));

    int32_t kH = static_cast<int32_t>(*(ksize->GetData()));
    int32_t kW = static_cast<int32_t>(*(ksize->GetData() + 1));
    int32_t sH = static_cast<int32_t>(*(strides->GetData()));
    int32_t sW = static_cast<int32_t>(*(strides->GetData() + 1));
    int32_t padH = static_cast<int32_t>(*(pads->GetData()));
    int32_t padW = static_cast<int32_t>(*(pads->GetData() + 1));
    int32_t dilH = static_cast<int32_t>(*(dilations->GetData()));
    int32_t dilW = static_cast<int32_t>(*(dilations->GetData() + 1));
    OP_CHECK_IF(kH <= 0 || kW <= 0 || sH <= 0 || sW <= 0 || dilH <= 0 || dilW <= 0 || padH < 0 || padW < 0,
                OP_LOGE(context, "invalid window params"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(padH > ((kH - 1) * dilH + 1) / 2 || padW > ((kW - 1) * dilW + 1) / 2,
                OP_LOGE(context, "pads must not exceed half of the effective span"), return ge::GRAPH_FAILED);

    // 3. 窗口数学（window-math.md §1）：out = floor((in + 2p − span)/s) + 1（floor 模式）
    int64_t hspan = static_cast<int64_t>(kH - 1) * dilH + 1;
    int64_t wspan = static_cast<int64_t>(kW - 1) * dilW + 1;
    int64_t outH = (inH + 2LL * padH - hspan) / sH + 1;
    int64_t outW = (inW + 2LL * padW - wspan) / sW + 1;
    OP_CHECK_IF(outH < 1 || outW < 1, OP_LOGE(context, "output dims must be >= 1"), return ge::GRAPH_FAILED);

    // 4. 两步法核间切分（splitting.md §3）
    uint64_t count = static_cast<uint64_t>(n) * c * outH * outW;
    uint64_t perCore = (count + coreNum - 1) / coreNum;
    if (perCore < static_cast<uint64_t>(PER_CORE_MIN)) {
        perCore = static_cast<uint64_t>(PER_CORE_MIN);
    }
    int32_t needCoreNum = static_cast<int32_t>((count + perCore - 1) / perCore);
    if (needCoreNum < 1) {
        needCoreNum = 1; // 空输出仍按 1 核发起（splitting.md §3）
    }

    // 5. TilingData 写入（标量包，不含线程数——kernel-entrance.md §2）
    MaxPoolWithArgmaxTilingData* tiling = context->GetTilingData<MaxPoolWithArgmaxTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(MaxPoolWithArgmaxTilingData), 0, sizeof(MaxPoolWithArgmaxTilingData)) != EOK,
                OP_LOGE(context, "Failed to set tiling data"), return ge::GRAPH_FAILED);
    tiling->count = static_cast<int64_t>(count);
    tiling->channels = c;
    tiling->inH = inH;
    tiling->inW = inW;
    tiling->outH = static_cast<int32_t>(outH);
    tiling->outW = static_cast<int32_t>(outW);
    tiling->kH = kH;
    tiling->kW = kW;
    tiling->sH = sH;
    tiling->sW = sW;
    tiling->padH = padH;
    tiling->padW = padW;
    tiling->dilH = dilH;
    tiling->dilW = dilW;
    tiling->needCoreNum = needCoreNum;

    context->SetBlockDim(needCoreNum);

    // 6. workspace 与 UB 预算（直除形态无 UB 数据面，仅预留 SIMT DataCache）
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = static_cast<size_t>(sysWorkspaceSize);
    OP_CHECK_IF((ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE), OP_LOGE(context, "ubSize %lu too small", ubSize),
                return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context, "SetLocalMemorySize failed"), return ge::GRAPH_FAILED);

    // 7. TilingKey：TPL 字段取值绑定（kernel-entrance.md §3：只编码场景模式）
    uint64_t format = isNhwc ? static_cast<uint64_t>(TPL_NHWC_FORMAT) : static_cast<uint64_t>(TPL_NCHW_FORMAT);
    uint64_t isInt32Idx = TPL_INT32; // mask 索引 dtype 档：int32（大 shape 升 int64 时改此处）
    context->SetTilingKey(GET_TPL_TILING_KEY(format, isInt32Idx));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMaxPoolWithArgmax([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaxPoolWithArgmax)
    .Tiling(MaxPoolWithArgmaxTilingFunc)
    .TilingParse<MaxPoolWithArgmaxCompileInfo>(TilingParseForMaxPoolWithArgmax);

} // namespace optiling
