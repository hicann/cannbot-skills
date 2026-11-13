/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CONV3DDW_STRUCT_H
#define CONV3DDW_STRUCT_H
#include "ascendc/host_api/tiling/template_argument.h"

/* MN_StreamK 的枚举值为 2，位于列表下标 1；实际 key 由宏计算。 */
ASCENDC_TPL_ARGS_DECL(Conv3ddw, ASCENDC_TPL_UINT_DECL(templateId, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 1, 2));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_UINT_SEL(templateId, ASCENDC_TPL_UI_LIST, 2)));
#endif
