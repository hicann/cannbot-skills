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
 * \file max_pool_grad_with_argmax_tiling_key.h
 * \brief TilingKey：ASCENDC_TPL_ARGS_DECL 模板编程（范式规则 §5.5.4 强制形态）
 *
 * 反向 SIMT 仅 NCHW 布局（实测 SIMT 形态如此，NHWC 走非 SIMT 路线），
 * 唯一 TPL 维度：mask 索引 dtype 档。
 */

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_TILING_KEY_H_
#define MAX_POOL_GRAD_WITH_ARGMAX_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

namespace NsMaxPoolGradWithArgmax {
#define TPL_INT64 0
#define TPL_INT32 1

ASCENDC_TPL_ARGS_DECL(MaxPoolGradWithArgmax,
                      ASCENDC_TPL_UINT_DECL(isInt32Idx, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_INT64, TPL_INT32));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(isInt32Idx, ASCENDC_TPL_UI_LIST, TPL_INT64, TPL_INT32),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(MaxPoolGradWithArgmaxTilingData)));

} // namespace NsMaxPoolGradWithArgmax

#endif // MAX_POOL_GRAD_WITH_ARGMAX_TILING_KEY_H_
