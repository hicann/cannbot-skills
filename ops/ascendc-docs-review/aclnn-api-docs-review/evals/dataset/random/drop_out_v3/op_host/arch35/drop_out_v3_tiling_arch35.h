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
 * \file drop_out_v3_tiling_arch35.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DROP_OUT_V3_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DROP_OUT_V3_H_

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "../../../random_common/op_host/arch35/random_tiling_arch35.h"

namespace optiling {

class DropOutV3Tiling : public RandomTilingArch35 {
public:
    explicit DropOutV3Tiling(gert::TilingContext* context)
        : RandomTilingArch35(context, BuildOpConfig()) {}

protected:
    ge::graphStatus UniqueProcess() override;

private:
    static OpTilingConfig BuildOpConfig();
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_DROP_OUT_V3_H_
