/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sort_tiling.h"
#include "sort_tiling_arch35.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"

#include <string>

using namespace ge;

namespace optiling {
static ge::graphStatus Tiling4Sort(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    OP_LOGD(context->GetNodeName(), "AscendC Sort tiling");
    OP_CHECK_IF(SortTilingSimt(context, ascendcPlatform.GetCoreNumAiv()) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "SortTilingSimt", "GRAPH_FAILED",
                                                      "The value of SortTilingSimt must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Sort(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "AscendC Tiling preparation");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Sort).Tiling(Tiling4Sort).TilingParse<SortCompileInfo>(TilingPrepare4Sort);
} // namespace optiling
