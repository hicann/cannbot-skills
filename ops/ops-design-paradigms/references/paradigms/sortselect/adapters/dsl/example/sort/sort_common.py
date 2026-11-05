# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""共享数值约定：保序键、字节提取及 Radix 容量常量。"""

import cannbotdsl as cb


BITS_PER_BYTE = 8
WORD_BYTES = 4
WORD_BASE = 1 << (WORD_BYTES * BITS_PER_BYTE)
FP16_INFINITY_BITS = 0x7C00
BF16_INFINITY_BITS = 0x7F80
FP32_INFINITY_BITS = 0x7F800000
RADIX_BINS = 1 << BITS_PER_BYTE
# 各常量分别约束局部容量、插入批次与执行线程，不因当前数值相同而共用。
LOCAL_RADIX_ALIGNMENT = 256  # 局部排序宽度按元素数补齐，不是 UB 字节对齐。
INSERTION_MAX_BATCH_ROWS = 1024
INSERTION_THREADS = 1024  # 一线程排序一行，批次上限不得超过线程数。
RADIX_SCATTER_THREADS = 1024
DEFAULT_RADIX_TILE = 1024
GLOBAL_RADIX_TILES = (1024, 2048, 4096, 8192)
# 本 arch35 示例的 SIMT UB 保留预算；entry 先从设备总 UB 扣除，模板只用剩余空间。
# 迁移 SDK/设备时须重新核对保留预算，不能把这 32 KiB 与模板临时缓冲重复使用。
SIMT_SCRATCH_BYTES = 32768
# 块内桶计数使用 uint16；按向量宽度对齐后的可用元素上限还会进一步收紧。
LOCAL_RADIX_COUNT_MAX = 0xFFFF


def key_dtype(dtype):
    return (
        cb.dtypes.uint64
        if dtype in (cb.dtypes.int64, cb.dtypes.uint64)
        else cb.dtypes.uint32
    )


def dtype_bytes(dtype):
    if dtype in (cb.dtypes.int8, cb.dtypes.uint8):
        return 1
    if dtype in (
        cb.dtypes.int16,
        cb.dtypes.uint16,
        cb.dtypes.float16,
        cb.dtypes.bfloat16,
    ):
        return 2
    if dtype in (cb.dtypes.int64, cb.dtypes.uint64):
        return 8
    return 4


@cb.jit
def ordered_key(value, dtype: cb.Constexpr, descending: cb.Constexpr[bool]):
    size = dtype_bytes(dtype)
    carrier = (
        cb.dtypes.uint8
        if size == 1
        else cb.dtypes.uint16
        if size == 2
        else cb.dtypes.uint32
        if size == 4
        else cb.dtypes.uint64
    )
    result_type = key_dtype(dtype)
    sign = 1 << (size * BITS_PER_BYTE - 1)
    maximum = (1 << (size * BITS_PER_BYTE)) - 1
    # 按原 dtype 位宽取位模式；扩宽只提供运算载体，不改变有效键空间。
    bits = result_type(cb.simt.bitcast(value, dtype=carrier))
    key = result_type(bits)
    if dtype in (cb.dtypes.float16, cb.dtypes.bfloat16, cb.dtypes.float32):
        infinity = (
            FP16_INFINITY_BITS
            if dtype == cb.dtypes.float16
            else BF16_INFINITY_BITS
            if dtype == cb.dtypes.bfloat16
            else FP32_INFINITY_BITS
        )
        magnitude = bits % result_type(sign)
        # ±0 使用同一个键，稳定排序因此保留两种零的输入次序。
        if magnitude == 0:
            key = result_type(sign)
        # 正负 NaN 及不同 payload 均放到升序末端；这里只归一键，不改原值。
        elif magnitude > infinity:
            key = result_type(maximum)
        # 负浮点的位序与数值序相反：逐位反转；非负数翻转符号位接在其后。
        elif bits >= sign:
            key = result_type(maximum) - bits
        else:
            key = bits ^ result_type(sign)
    elif dtype in (cb.dtypes.int8, cb.dtypes.int16, cb.dtypes.int32, cb.dtypes.int64):
        # 补码有符号整数翻转最高位，将最小负数映到无符号键零。
        key = bits ^ result_type(sign)
    # 无符号整数直接使用原位模式；降序统一反射键空间，NaN 随之移到首端。
    if descending:
        key = result_type(maximum) - key
    return key


@cb.jit
def radix_digit(key, storage: cb.Constexpr, byte: cb.Constexpr[int]):
    # DSL 定宽 int64 整除向零截断（SDK types/_integer.py）。
    # 有符号键为负且低 32 位非零时减一，修正为向下取整，再取无符号高 32 位。
    # tests/test_device.py 覆盖全部八字节及完整排序，包括 0xFFFFFFFF00000000。
    word = cb.dtypes.uint32(key)
    if storage == cb.dtypes.uint64 and byte >= WORD_BYTES:
        signed = cb.dtypes.int64(key)
        high = signed // cb.dtypes.int64(WORD_BASE)
        if signed < 0 and word != 0:
            high = high - 1
        word = cb.dtypes.uint32(high)
    return (
        word // cb.dtypes.uint32(1 << ((byte % WORD_BYTES) * BITS_PER_BYTE))
    ) % cb.dtypes.uint32(RADIX_BINS)
