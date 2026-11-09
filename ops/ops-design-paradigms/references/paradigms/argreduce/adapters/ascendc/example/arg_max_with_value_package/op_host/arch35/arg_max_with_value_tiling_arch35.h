/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file arg_max_with_value_tiling_arch35.h
 * \brief arg_max_with_value_tiling_arch35
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_ARG_MAX_WITH_VALUE_TILING_ARCH35_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_ARG_MAX_WITH_VALUE_TILING_ARCH35_H

#include "register/tilingdata_base.h"
#include "arg_common_base_tiling_arch35.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(ArgMaxWithValue, ArgMaxWithValueTilingData)
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_ARG_MAX_WITH_VALUE_TILING_ARCH35_H