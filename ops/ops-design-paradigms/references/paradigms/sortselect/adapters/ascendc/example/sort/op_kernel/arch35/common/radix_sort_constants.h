/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RADIX_SORT_CONSTANTS_H
#define RADIX_SORT_CONSTANTS_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace RadixSortCommon {

using namespace AscendC;

const int32_t VF_LEN_B32 = Ops::Base::GetVRegSize() / sizeof(int32_t);
const int32_t VF_LEN_B16 = Ops::Base::GetVRegSize() / sizeof(int16_t);
const int32_t VF_LEN_B8 = Ops::Base::GetVRegSize() / sizeof(int8_t);
const int32_t VF_LEN_B64 = Ops::Base::GetVRegSize() / sizeof(int64_t);

// 基数排序正负0优化
const uint16_t TWIDDLED_MINUS_ZERO_BITS_FP16 = 0x7fff;
const uint16_t TWIDDLED_ZERO_BITS_FP16 = 0x8000;
const uint32_t TWIDDLED_MINUS_ZERO_BITS_FP32 = 0x7fffffff;
const uint32_t TWIDDLED_ZERO_BITS_FP32 = 0x80000000;
const uint16_t CANONICAL_NAN_KEY_B16 = 0xFFFFU;
const uint32_t CANONICAL_NAN_KEY_B32 = 0xFFFFFFFFU;
const uint16_t FLOAT_ABS_MASK_B16 = 0x7FFFU;
const uint32_t FLOAT_ABS_MASK_B32 = 0x7FFFFFFFU;
const uint16_t FP16_INFINITY_BITS = 0x7C00U;
const uint16_t BF16_INFINITY_BITS = 0x7F80U;
const uint32_t FP32_INFINITY_BITS = 0x7F800000U;

const uint16_t LOWEST_KEY_VALUE_B16 = uint16_t(-1);
const uint32_t LOWEST_KEY_VALUE_B32 = uint32_t(-1);
const uint16_t XOR_OP_VALUE_B16 = (uint16_t(1) << 15);
const uint32_t ZERO_VALUE_FLAG_B32 = 0;
const uint16_t ZERO_VALUE_FLAG_B16 = 0;
const uint32_t SHIFT_BIT_NUM = 8;
const uint32_t HIST_MASK_OUT_LEN = 8;
const uint64_t XOR_OP_VALUE_B64 = 0x8000000000000000;
const uint32_t RADIX_SORT_NUM = 1U << SHIFT_BIT_NUM;
// Sort scratch 中三个散射参数的槽位；生产者与 SIMT 消费者共用此约定。
constexpr uint32_t SCATTER_TILE_START = 0;
constexpr uint32_t SCATTER_OUTPUT_OFFSET = 1;
constexpr uint32_t SCATTER_BIN_OFFSET = 2;
const uint32_t INT64_INDEX_SCALE = sizeof(int64_t) / sizeof(uint32_t);
const uint8_t XOR_OP_VALUE_B8 = (uint8_t(1) << 7);
const uint32_t XOR_OP_VALUE = 0x80000000;
// 每个直方图或前缀值的最高两位存放软同步状态。
const int32_t AGGREGATE_READY_FLAG = 1;
const int32_t PREFIX_READY_FLAG = 2;
const uint32_t NOT_INIT_MODE = 0;
const uint32_t AGG_READY_MODE = 1;
const uint32_t PREFIX_READY_MODE = 2;
// Reduce 输出按 HIST_MASK_OUT_LEN 个 uint32 的槽距存放三类状态计数。
const uint32_t NOT_INIT_COUNT_INDEX = 0;
const uint32_t AGG_READY_COUNT_INDEX = HIST_MASK_OUT_LEN;
const uint32_t PREFIX_READY_COUNT_INDEX = 2U * HIST_MASK_OUT_LEN;
constexpr uint32_t THREAD_DIM_NUM = 1024;
// 两种计数格式共用同一状态机：高两位为状态，低 30/62 位为非负计数。
// Host 必须保证 N 不超过 VALUE_MASK；最终输出索引的 dtype 与本格式无关。
template <typename CounterT>
struct RadixCounterTraits {
    static constexpr uint16_t LANES = Ops::Base::GetVRegSize() / sizeof(CounterT);
    static constexpr int16_t STATE_SHIFT = sizeof(CounterT) * SHIFT_BIT_NUM - 2;
    static constexpr CounterT VALUE_MASK = static_cast<CounterT>((uint64_t{1} << STATE_SHIFT) - 1);
    static constexpr CounterT AGGREGATE_MASK = static_cast<CounterT>(uint64_t{1} << STATE_SHIFT);
    static constexpr CounterT PREFIX_MASK = static_cast<CounterT>(uint64_t{2} << STATE_SHIFT);
};
} // namespace RadixSortCommon

#endif
