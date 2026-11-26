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
 * \file max_pool_grad_with_argmax_tiling.cpp
 * \brief 反向 host tiling：两步法核间切分（inSize 主坐标）
 */

#include <string>
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "../op_kernel/arch35/max_pool_grad_with_argmax_tiling_data.h"
#include "../op_kernel/arch35/max_pool_grad_with_argmax_tiling_key.h"

namespace optiling {

using namespace NsMaxPoolGradWithArgmax;

static constexpr int64_t PER_CORE_MIN = 1024;
static constexpr uint32_t DCACHE_SIZE = 128 * 1024;
static constexpr uint32_t STATIC_UB_ESTIMATE = 32; // 快除参数表 8 项（INDEX_SIZE_T 双档最大 64B）

enum InputPos {
    IDX_X = 0,
    IDX_GRAD,
    IDX_ARGMAX
};
enum AttrPos {
    ATTR_KSIZE = 0,
    ATTR_STRIDES,
    ATTR_PADS,
    ATTR_DILATIONS
};

struct MaxPoolGradWithArgmaxCompileInfo {};

static ge::graphStatus MaxPoolGradWithArgmaxTilingFunc(gert::TilingContext* context)
{
    // 1. 平台信息
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    // 2. 输入 shape：x 为原输入（NCHW，y 形状源，值不消费）；grad/argmax 为正向输出同形
    auto xInput = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInput);
    auto xShape = xInput->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() != 4, OP_LOGE(context, "x must be 4D NCHW"), return ge::GRAPH_FAILED);
    auto gradInput = context->GetInputShape(IDX_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradInput);
    auto gradShape = gradInput->GetStorageShape();
    OP_CHECK_IF(gradShape.GetDimNum() != 4, OP_LOGE(context, "grad must be 4D NCHW"), return ge::GRAPH_FAILED);

    int32_t n = static_cast<int32_t>(xShape.GetDim(0));
    int32_t c = static_cast<int32_t>(xShape.GetDim(1));
    int32_t inH = static_cast<int32_t>(xShape.GetDim(2));
    int32_t inW = static_cast<int32_t>(xShape.GetDim(3));
    int32_t outH = static_cast<int32_t>(gradShape.GetDim(2));
    int32_t outW = static_cast<int32_t>(gradShape.GetDim(3));

    // 3. 窗口参数（grad-output 维与窗口公式一致性由 infershape 侧保证）
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const gert::TypedContinuousVector<int64_t>* ksize = attrs->GetListInt(ATTR_KSIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    const gert::TypedContinuousVector<int64_t>* strides = attrs->GetListInt(ATTR_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    const gert::TypedContinuousVector<int64_t>* pads = attrs->GetListInt(ATTR_PADS);
    OP_CHECK_NULL_WITH_CONTEXT(context, pads);
    const gert::TypedContinuousVector<int64_t>* dilations = attrs->GetListInt(ATTR_DILATIONS);
    OP_CHECK_NULL_WITH_CONTEXT(context, dilations);
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

    // 4. 两步法核间切分（splitting.md §3：反向以 inSize 为主坐标）
    uint64_t count = static_cast<uint64_t>(n) * c * inH * inW;
    uint64_t perCore = (count + coreNum - 1) / coreNum;
    if (perCore < static_cast<uint64_t>(PER_CORE_MIN)) {
        perCore = static_cast<uint64_t>(PER_CORE_MIN);
    }
    int32_t needCoreNum = static_cast<int32_t>((count + perCore - 1) / perCore);
    if (needCoreNum < 1) {
        needCoreNum = 1;
    }

    // 5. TilingData 写入
    MaxPoolGradWithArgmaxTilingData* tiling = context->GetTilingData<MaxPoolGradWithArgmaxTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(MaxPoolGradWithArgmaxTilingData), 0, sizeof(MaxPoolGradWithArgmaxTilingData)) != EOK,
        OP_LOGE(context, "Failed to set tiling data"), return ge::GRAPH_FAILED);
    tiling->count = static_cast<int64_t>(count);
    tiling->inH = inH;
    tiling->inW = inW;
    tiling->outH = outH;
    tiling->outW = outW;
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

    // 6. workspace 与 UB 预算（快除参数表 8 项 + SIMT DataCache 预留）
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = static_cast<size_t>(sysWorkspaceSize);
    OP_CHECK_IF((ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE), OP_LOGE(context, "ubSize %lu too small", ubSize),
                return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context, "SetLocalMemorySize failed"), return ge::GRAPH_FAILED);

    // 7. TilingKey：mask 索引 dtype 档（正反向须同档同口径，grad-routing.md §4 契约）
    uint64_t isInt32Idx = TPL_INT32;
    context->SetTilingKey(GET_TPL_TILING_KEY(isInt32Idx));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMaxPoolGradWithArgmax([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaxPoolGradWithArgmax)
    .Tiling(MaxPoolGradWithArgmaxTilingFunc)
    .TilingParse<MaxPoolGradWithArgmaxCompileInfo>(TilingParseForMaxPoolGradWithArgmax);

} // namespace optiling
