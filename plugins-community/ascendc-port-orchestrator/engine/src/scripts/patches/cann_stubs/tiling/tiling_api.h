// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// DEBT-110 Scope B: CANN host-side stub for port_a3_to_a5 build.
//
// Replaces CANN's `tiling/tiling_api.h` with a minimal shim that exposes
// the optiling::TCubeTiling / optiling::SoftMaxTiling tag types referenced
// by op_host/<op>_tiling.h files when they call
// `TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTilingR);`.
//
// TCubeTiling fields are populated to match the public CANN matmul tiling
// contract (kernel_operator.h public surface) so that any kernel code that
// reads matmulTilingR.M / .N / .K / etc. still parses. Values are zero by
// default — port_a3_to_a5 host-side build is exercised for compilation, not
// runtime correctness.

#ifndef A5OPS_CANN_STUB_TILING_API_H_
#define A5OPS_CANN_STUB_TILING_API_H_

#include "register/tilingdata_base.h"
#include "tiling_context.h"
#include "platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TCubeTiling)
    TILING_DATA_FIELD_DEF(int, usedCoreNum);
    TILING_DATA_FIELD_DEF(int, M);
    TILING_DATA_FIELD_DEF(int, K);
    TILING_DATA_FIELD_DEF(int, N);
    TILING_DATA_FIELD_DEF(int, singleCoreM);
    TILING_DATA_FIELD_DEF(int, singleCoreN);
    TILING_DATA_FIELD_DEF(int, singleCoreK);
    TILING_DATA_FIELD_DEF(int, baseM);
    TILING_DATA_FIELD_DEF(int, baseN);
    TILING_DATA_FIELD_DEF(int, baseK);
    TILING_DATA_FIELD_DEF(int, depthA1);
    TILING_DATA_FIELD_DEF(int, depthB1);
    TILING_DATA_FIELD_DEF(int, stepM);
    TILING_DATA_FIELD_DEF(int, stepN);
    TILING_DATA_FIELD_DEF(int, A1Length);
    TILING_DATA_FIELD_DEF(int, B1Length);
    TILING_DATA_FIELD_DEF(int, CO1Length);
    TILING_DATA_FIELD_DEF(int, isBias);
    TILING_DATA_FIELD_DEF(int, transLength);
    TILING_DATA_FIELD_DEF(int, iterateOrder);
    TILING_DATA_FIELD_DEF(int, shareMode);
    TILING_DATA_FIELD_DEF(int, shareL1Size);
    TILING_DATA_FIELD_DEF(int, shareL0CSize);
    TILING_DATA_FIELD_DEF(int, shareUbSize);
    TILING_DATA_FIELD_DEF(int, batchM);
    TILING_DATA_FIELD_DEF(int, batchN);
    TILING_DATA_FIELD_DEF(int, singleBatchM);
    TILING_DATA_FIELD_DEF(int, singleBatchN);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(SoftMaxTiling)
END_TILING_DATA_DEF;

}  // namespace optiling

#endif  // A5OPS_CANN_STUB_TILING_API_H_
