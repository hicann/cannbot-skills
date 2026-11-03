/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SORT_LIB_CONSTANTS_H
#define SORT_LIB_CONSTANTS_H

/*!
 * \file sort_lib_constants.h
 * \brief Internal constants and UnsignedKeyType trait for SortLib.
 *
 * \internal  Do not include directly — used internally by SortLib.
 */

#include <cstdint>
#include <type_traits>

namespace SortLib::detail {

template <typename T>
struct UnsignedKeyType {
    using type = typename std::conditional<
        sizeof(T) == sizeof(int8_t), uint8_t,
        typename std::conditional<
            sizeof(T) == sizeof(int16_t), uint16_t,
            typename std::conditional<sizeof(T) == sizeof(int32_t), uint32_t, uint64_t>::type>::type>::type;
};

constexpr int32_t VF_LEN_B32 = 64;
constexpr int32_t VF_LEN_B16 = 128;
constexpr int32_t VF_LEN_B8 = 256;
constexpr int32_t VF_LEN_B64 = 32;

const uint16_t TWIDDLED_MINUS_ZERO_BITS_FP16 = 0x7fff;
const uint16_t TWIDDLED_ZERO_BITS_FP16 = 0x8000;
const uint32_t TWIDDLED_MINUS_ZERO_BITS_FP32 = 0x7fffffff;
const uint32_t TWIDDLED_ZERO_BITS_FP32 = 0x80000000;
const uint16_t LOWEST_KEY_VALUE_B16 = uint16_t(-1);
const uint32_t LOWEST_KEY_VALUE_B32 = uint32_t(-1);
const uint16_t XOR_OP_VALUE_B16 = (uint16_t(1) << 15);
const uint32_t ZERO_VALUE_FLAG_B32 = 0;
const uint16_t ZERO_VALUE_FLAG_B16 = 0;
const uint32_t SHIFT_BIT_NUM = 8;
const uint32_t HIST_MASK_OUT_LEN = 8;
const uint64_t XOR_OP_VALUE_B64 = 0x8000000000000000;
const uint32_t RADIX_SORT_NUM = 256;
const uint8_t XOR_OP_VALUE_B8 = (uint8_t(1) << 7);
const uint32_t XOR_OP_VALUE = 0x80000000;
const int16_t STATE_BIT_SHF_VALUE = 30;
const int16_t STATE_BIT_SHF_VALUE_B64 = 62;
const int32_t AGGREGATE_READY_FLAG = 1;
const int32_t PREFIX_READY_FLAG = 2;
const uint32_t NOT_INIT_MODE = 0;
const uint32_t AGG_READY_MODE = 1;
const uint32_t PREFIX_READY_MODE = 2;
const uint32_t NOT_INIT_COUNT_INDEX = 0;
const uint32_t AGG_READY_COUNT_INDEX = 8;
const uint32_t PREFIX_READY_COUNT_INDEX = 16;
const uint32_t THREAD_DIM_NUM = 1024;
const uint32_t AGGREGATE_READY_MASK = 0x40000000;
const uint64_t AGGREGATE_READY_MASK_B64 = 0x4000000000000000;
const uint32_t PREFIX_READY_MASK = 0x80000000;
const uint64_t PREFIX_READY_MASK_B64 = 0x8000000000000000;
const uint32_t VALUE_MASK = 0x3fffffff;
const uint64_t VALUE_MASK_B64 = 0x3fffffffffffffff;

} // namespace SortLib::detail
#endif
