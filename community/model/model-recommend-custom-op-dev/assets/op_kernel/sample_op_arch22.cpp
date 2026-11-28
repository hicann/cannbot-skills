/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch22/sample_op.h"
#include "arch22/sample_op_tiling_key.h"

// 模板参数与 sample_op_tiling_key.h 中 ASCENDC_TPL_* 定义对应
template <typename D_T, int SCH_MODE>
__global__ __aicore__ void sample_op(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SampleOpTilingData);
    GET_TILING_DATA_WITH_STRUCT(SampleOpTilingData, tilingData, tiling);

    NsSampleOp::SampleOp<D_T, SCH_MODE> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
