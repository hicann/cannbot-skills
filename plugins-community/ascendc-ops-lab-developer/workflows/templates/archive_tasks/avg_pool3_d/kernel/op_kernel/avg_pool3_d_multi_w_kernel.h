/*
 * Copyright (c) 2026 Huawei Technologies, Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "avg_pool3_d_kernel_base.h"

class AvgPool3DMultiWKernel : public AvgPool3DKernelBase<AvgPool3DMultiWKernel> {
protected:
    friend class AvgPool3DKernelBase<AvgPool3DMultiWKernel>;
};
