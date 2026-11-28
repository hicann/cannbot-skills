/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SAMPLE_OP_TILING_KEY_H__
#define __SAMPLE_OP_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define TILING_KEY_MODE_0  0  // 单核
#define TILING_KEY_MODE_2  2  // 多核

// TODO(<op>): 按 dtype x 切分模式声明模板参数，与 kernel 模板实参一一对应
ASCENDC_TPL_ARGS_DECL(SampleOp,
    ASCENDC_TPL_DATATYPE_DECL(D_T, C_DT_FLOAT16, C_DT_BF16, C_DT_FLOAT),
    ASCENDC_TPL_UINT_DECL(SCH_MODE, 8, ASCENDC_TPL_UI_LIST,
        TILING_KEY_MODE_0, TILING_KEY_MODE_2)
);

// TODO(<op>): 枚举全部组合
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, TILING_KEY_MODE_0)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, TILING_KEY_MODE_2)
    )
);

#endif // __SAMPLE_OP_TILING_KEY_H__
