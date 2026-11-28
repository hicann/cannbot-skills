/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SAMPLE_OP_TILING_DATA_H__
#define __SAMPLE_OP_TILING_DATA_H__

#include <cstdint>

struct SampleOpTilingData {
    uint64_t totalRows = 0;
    uint64_t cols = 0;
    uint64_t rowsPerCore = 0;
    uint64_t blockDim = 0;
    uint32_t schMode = 0;
};

#endif // __SAMPLE_OP_TILING_DATA_H__
