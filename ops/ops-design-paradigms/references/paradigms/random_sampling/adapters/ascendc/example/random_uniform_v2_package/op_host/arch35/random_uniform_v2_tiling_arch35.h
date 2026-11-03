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
 * \file random_uniform_v2_tiling_arch35.h
 * \brief
 */
#ifndef RANDOM_UNIFORM_V2_TILING_ARCH35_H
#define RANDOM_UNIFORM_V2_TILING_ARCH35_H

#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "util/math_util.h"
#include "../../../random_common/op_host/arch35/random_tiling_arch35.h"

namespace optiling {
class RandomUniformV2TilingArch35 : public RandomTilingArch35 {
public:
    explicit RandomUniformV2TilingArch35(gert::TilingContext* context);
    ~RandomUniformV2TilingArch35() override = default;

protected:
private:
    static OpTilingConfig BuildOpConfig();
};

} // namespace optiling
#endif // RANDOM_UNIFORM_V2_TILING_ARCH35_H
