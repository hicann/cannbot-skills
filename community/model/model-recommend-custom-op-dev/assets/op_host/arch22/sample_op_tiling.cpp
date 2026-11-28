/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/sample_op_tiling_data.h"
#include "../../op_kernel/arch22/sample_op_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 按行分核：rowsPerCore = ceil(totalRows / blockDim)
static ge::graphStatus SampleOpTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    auto xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape storageShape = xShape->GetStorageShape();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();

    const uint64_t totalLength = storageShape.GetShapeSize();
    const uint64_t totalRows = (storageShape.GetDimNum() > 0) ? storageShape.GetDim(0) : 1;
    const uint64_t cols = (totalRows == 0) ? 0 : totalLength / totalRows;

    SampleOpTilingData* tiling = context->GetTilingData<SampleOpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    uint64_t blockDim = 1;
    uint64_t rowsPerCore = totalRows;
    if (totalRows > 0) {
        blockDim = std::min<uint64_t>(static_cast<uint64_t>(coreNum), totalRows);
        blockDim = std::max<uint64_t>(blockDim, 1);
        rowsPerCore = CeilDiv(totalRows, blockDim);
        blockDim = CeilDiv(totalRows, rowsPerCore);
        blockDim = std::min<uint64_t>(blockDim, static_cast<uint64_t>(coreNum));
    }

    // TODO(<op>): cols 超过 UB 容量时改为行内分块（tile 列）
    const uint64_t typeSize = (dtype == ge::DT_FLOAT) ? 4 : 2;
    const uint64_t bytesPerRow = cols * typeSize;
    OP_CHECK_IF(bytesPerRow * 2 > ubSize,
        OP_LOGE(context, "cols too large for UB: bytesPerRow=%lu, ubSize=%lu", bytesPerRow, ubSize),
        return ge::GRAPH_FAILED);

    tiling->totalRows = totalRows;
    tiling->cols = cols;
    tiling->rowsPerCore = rowsPerCore;
    tiling->blockDim = blockDim;
    tiling->schMode = TILING_KEY_MODE_0;

    context->SetBlockDim(blockDim);
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtype), tiling->schMode);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSampleOp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SampleOpCompileInfo {};

IMPL_OP_OPTILING(SampleOp)
    .Tiling(SampleOpTilingFunc)
    .TilingParse<SampleOpCompileInfo>(TilingParseForSampleOp);

} // namespace optiling
