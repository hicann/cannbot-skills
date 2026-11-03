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
 * \file tensor_scatter_nd_update_tiling.cpp
 * \brief Tiling entry for tensor_scatter_nd_update（v2.0 多模板注册重构，MDE §11.2.3）
 *
 * 入口仅做 registry 分派（priority 升序：NondetTiling=0 → DetTiling=10 兜底）；
 * 公共校验/解析在 BaseTiling（tensor_scatter_nd_update_tiling_base.cpp），
 * 确定路径逻辑在 DetTiling，非确定路径逻辑在 NondetTiling。
 * v1.x 的空 TensorScatterNdUpdateCompileInfo 结构增强为填充 coreNum/ubSize
 * （供 GetPlatformInfo 在 platformInfoPtr 为空时兜底，对齐 bucketize_v2 / 两基例 v2.0 模式）。
 */

#include "op_host/tiling_templates_registry.h"
#include "tensor_scatter_nd_update_base_tiling.h"

namespace optiling {

ge::graphStatus Tiling4TensorScatterNdUpdate(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling for tensor_scatter_nd_update is running.");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4TensorScatterNdUpdate(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<TensorScatterNdUpdateCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TensorScatterNdUpdate)
    .Tiling(Tiling4TensorScatterNdUpdate)
    .TilingParse<TensorScatterNdUpdateCompileInfo>(TilingPrepare4TensorScatterNdUpdate);

} // namespace optiling
