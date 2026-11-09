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
 * \file arg_common_base_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ARG_COMMON_BASE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ARG_COMMON_BASE_H_
#include <cstdint>
#include "register/op_impl_registry.h"

namespace optiling {
using namespace ge;

struct ArgOpsCompileInfo {
    int64_t ub_ele{0};
    int64_t core_num{0};
    int64_t max_segment_len{0};
    bool is_vccmp_support{false};
    bool with_value{false};
    bool is_data_move_pad_support{false};
    bool is_vsel_support{false};
    int64_t block_size{32};
    int64_t segment_len{0};
    int64_t first_dim_segment{0};
    int64_t ubSize{0};
    int64_t coreNum{0};
    uint64_t vRegSize = 0;
};

struct ArgOpsTilingParams {
    int64_t tiling_mode;
    int64_t first_dim_size;
    int64_t axis_size;
    int64_t last_dim_size;
    int64_t act_core_num;
    int64_t one_core_ele;
    int64_t last_core_ele;
    // for arg last dim
    int64_t align_num;
    int64_t axis_size_one_time;
    int64_t loop_times;
    int64_t tail_size;
    // for arg last dim and not last dim
    int64_t one_core_segment_loop;
    int64_t one_core_segment_tail;
    int64_t one_core_segment_tail_data;
    int64_t one_core_offset;
    int64_t last_core_segment_loop;
    int64_t last_core_segment_tail;
    int64_t last_core_segment_tail_data;
    int64_t last_core_offset;
    int64_t tiling_core_num;
    int64_t n_inner;
};

ge::graphStatus ArgOpsTiling(gert::TilingContext* context);

ge::graphStatus TilingPrepareForArgOps(gert::TilingParseContext* context);

ge::graphStatus TilingPrepareForArgWithValueOps(gert::TilingParseContext* context);
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_ARG_COMMON_BASE_H_
