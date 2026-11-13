/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "arch35/conv3ddw_struct.h"
#include "arch35/conv3ddw_kernel.h"

template <uint32_t templateId>
__global__ __aicore__ void conv3ddw(GM_ADDR x, GM_ADDR filter_size, GM_ADDR out_backprop, GM_ADDR y, GM_ADDR workspace,
                                    GM_ADDR tiling)
{
    static_assert(templateId == 2);
    REGISTER_TILING_DEFAULT(Conv3DdwTilingData);
    GET_TILING_DATA_WITH_STRUCT(Conv3DdwTilingData, td, tiling);
    Conv3Ddw::InitializeOutput(y, td);
    if ASCEND_IS_AIC {
        if (AscendC::GetBlockIdx() >= td.usedCoreNum)
            return;
        Conv3Ddw::Kernel<DTYPE_X> op;
        op.Init(x, out_backprop, y, td);
        op.Process();
        op.End();
    }
}
