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
 * \file tensor_scatter_nd_update_det_tiling.h
 * \brief 确定性三段式模板声明（v2.0，MDE §11.5.2）
 *
 * TensorScatterNdUpdateDetTiling：v1.x TensorScatterNdUpdateTilingFunc 主体迁移
 *   （通用字段填充 / keyDtype 判定 / SortLib tiling / workspace 4 段布局 /
 *    blockDim + ScheduleMode），priority=10 兜底，IsCapable 恒 true。
 */

#ifndef TENSOR_SCATTER_ND_UPDATE_DET_TILING_H_
#define TENSOR_SCATTER_ND_UPDATE_DET_TILING_H_

#include "tensor_scatter_nd_update_base_tiling.h"

namespace optiling {

class TensorScatterNdUpdateDetTiling : public TensorScatterNdUpdateBaseTiling {
public:
    explicit TensorScatterNdUpdateDetTiling(gert::TilingContext* context)
        : TensorScatterNdUpdateBaseTiling(context)
    {}
    ~TensorScatterNdUpdateDetTiling() override {}

private:
    bool IsCapable() override
    {
        return true;
    } // 兜底：恒 true
    ge::graphStatus DoOpTiling() override;       // v1.x 步骤 4~6（keyDtype/SortLib/ws 布局）
    uint64_t GetTilingKey() const override;      // GET_TPL_TILING_KEY(countMode_, keyDtype_, DET)
    ge::graphStatus GetWorkspaceSize() override; // v1.x 步骤 7（userWs + sysWs）
    ge::graphStatus PostTiling() override;       // v1.x 步骤 8（blockDim + ScheduleMode(1) + SetLocalMemorySize）
    void DumpTilingInfo() override;              // v1.x DumpTilingData + TilingResult 日志

    uint32_t countMode_ = 0; // TSNU_COUNT_U32 / I64（SortLib IsInt32Safe(N)）
    uint32_t keyDtype_ = 0;  // TSNU_KEY_I16 / I32 / I64（按 totalVarElems_）
    int64_t userWsSize_ = 0;
};

} // namespace optiling
#endif
