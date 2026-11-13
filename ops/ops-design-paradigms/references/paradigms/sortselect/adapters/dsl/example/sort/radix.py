# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""块内稳定 Radix：按 toolkit Sort 的阶段组织 DSL 实现。

源码映射与语义差异见 ../../toolkit-radix-port.md。
这里是组合式 DSL 原语，不依赖尚未提供的高阶 Sort API。
"""

import cannbotdsl as cb
from cannbotdsl import reg

from sort_common import BITS_PER_BYTE, RADIX_BINS

# arch35 的向量寄存器宽度；16 位计数将 256 个桶分成两个寄存器保存。
VECTOR_BYTES = 256
HISTOGRAM_HALF_BINS = RADIX_BINS // 2
PARTITION_BITS = 2
PARTITION_GROUPS = 1 << PARTITION_BITS
BYTE_PARTITION_PASSES = BITS_PER_BYTE // PARTITION_BITS
LOCAL_RADIX_THREADS = 256


@cb.jit
def get_exclusive_sum(digits, offsets, length):
    # 对应 toolkit GetExclusiveSum 的统计部分；digit 提取由 caller 完成。
    # DSL frequency 返回本块频数，必须显式累加；不能照搬 C++ 的原地更新签名。
    with cb.vf(mode="simd"):
        mask16 = reg.create_mask(pattern="all", elem_bits=16)
        zero16 = reg.vdups(0, cb.dtypes.uint16)
        frequency0 = zero16
        frequency1 = zero16
        cumulative0 = zero16
        cumulative1 = zero16
        counter = reg.mask_counter(length)
        for chunk in range((length + VECTOR_BYTES - 1) // VECTOR_BYTES):
            valid, counter = reg.update_mask(counter, elem_bits=8)
            source = reg.vload(digits, chunk * VECTOR_BYTES)
            frequency0 = reg.vadd(
                frequency0,
                reg.vhistogram_frequency(source, mask=valid, bin=0),
                mask=mask16,
            )
            frequency1 = reg.vadd(
                frequency1,
                reg.vhistogram_frequency(source, mask=valid, bin=1),
                mask=mask16,
            )
            cumulative0 = reg.vhistogram_accumulate(
                cumulative0, source, mask=valid, bin=0
            )
            cumulative1 = reg.vhistogram_accumulate(
                cumulative1, source, mask=valid, bin=1
            )
        offsets0 = reg.vsub(cumulative0, frequency0, mask=mask16)
        offsets1 = reg.vsub(cumulative1, frequency1, mask=mask16)
        reg.vstore(offsets, 0, offsets0, mask16)
        reg.vstore(offsets, HISTOGRAM_HALF_BINS, offsets1, mask16)


@cb.jit
def eight_bits_sort(digits, byte_order, length):
    # 对应 toolkit EightBitsSort；只输出每个 256-lane 块内的 uint8 排列。
    # digits 保持原序，按 byte_order gather 后做下一轮，省去有序 digit 副本。
    # 每个向量做四次稳定 2-bit 分组，形成按完整字节排序的 lane 排列。
    with cb.vf(mode="simd"):
        for step in cb.range_constexpr(BYTE_PARTITION_PASSES):
            mask8 = reg.create_mask(pattern="all", elem_bits=8)
            cursor = reg.vstore_unalign_begin(byte_order)
            for chunk in range((length + VECTOR_BYTES - 1) // VECTOR_BYTES):
                source = reg.vload(digits, chunk * VECTOR_BYTES)
                index = reg.vreinterpret(
                    reg.varange(0, cb.dtypes.int8), cb.dtypes.uint8
                )
                # 首轮使用原 lane；后续轮次只加载一次前轮排列。0/1 次循环表达该特化加载。
                for _ in range(0, int(step != 0)):
                    index = reg.vload(byte_order, chunk * VECTOR_BYTES)
                values = reg.vgather_reg(source, index)
                digit = reg.vbitwise_and(
                    reg.vshr(values, step * PARTITION_BITS, mask=mask8),
                    reg.vdups(PARTITION_GROUPS - 1, cb.dtypes.uint8),
                    mask=mask8,
                )
                for group in cb.range_constexpr(PARTITION_GROUPS):
                    selected = reg.veqs(digit, group, mask=mask8)
                    compact = reg.vsqueeze_and_storeunalign_init(index, mask=selected)
                    reg.vsqueeze_and_storeunalign(byte_order, 0, compact, cursor)
            reg.vsqueeze_and_storeunalign_finalize(byte_order, 0, cursor)
            reg.vmem_bar("vst_vld")


@cb.jit
def local_sort(
    digits,
    byte_order,
    offsets,
    permutation,
    length,
    pack_digits: cb.Constexpr[bool] = False,
):
    # 对应 toolkit LocalSort；在寄存器中扩展 lane 索引，无需 tmpIndexU16 UB。
    with cb.vf(mode="simd"):
        mask16 = reg.create_mask(pattern="all", elem_bits=16)
        zero16 = reg.vdups(0, cb.dtypes.uint16)
        zero8 = reg.vdups(0, cb.dtypes.uint8)
        # 1. 载入各桶的写游标：整块桶起点 + 已处理向量的同桶数量。
        offsets0 = reg.vload(offsets, 0)
        offsets1 = reg.vload(offsets, HISTOGRAM_HALF_BINS)
        local0 = reg.vreinterpret(reg.varange(0, cb.dtypes.int16), cb.dtypes.uint16)
        local1 = reg.vreinterpret(
            reg.varange(HISTOGRAM_HALF_BINS, cb.dtypes.int16), cb.dtypes.uint16
        )
        counter = reg.mask_counter(length)
        for chunk in range((length + VECTOR_BYTES - 1) // VECTOR_BYTES):
            valid, counter = reg.update_mask(counter, elem_bits=8)
            source = reg.vload(digits, chunk * VECTOR_BYTES)
            index = reg.vload(byte_order, chunk * VECTOR_BYTES)
            keys = reg.vgather_reg(source, index)
            frequency0 = reg.vhistogram_frequency(keys, mask=valid, bin=0)
            frequency1 = reg.vhistogram_frequency(keys, mask=valid, bin=1)
            cumulative0 = reg.vhistogram_accumulate(zero16, keys, mask=valid, bin=0)
            cumulative1 = reg.vhistogram_accumulate(zero16, keys, mask=valid, bin=1)
            prefix0 = reg.vsub(cumulative0, frequency0, mask=mask16)
            prefix1 = reg.vsub(cumulative1, frequency1, mask=mask16)
            # 2. 将 uint16 游标拆为高/低字节泳道，按 8 位桶号 Gather，再拼回 16 位。
            offset_low, offset_high = reg.vdeinterleave(
                reg.vreinterpret_lanes(offsets0, cb.dtypes.uint8),
                reg.vreinterpret_lanes(offsets1, cb.dtypes.uint8),
            )
            bucket_low = reg.vgather_reg(offset_low, keys)
            bucket_high = reg.vgather_reg(offset_high, keys)
            bucket0, bucket1 = reg.vinterleave(bucket_low, bucket_high)
            # 3. Gather 本向量内各桶的独占起点。有效元素所属桶的起点最多为 255，
            # 因而仅取低字节即可；高字节只可能出现在没有有效元素的桶中。
            prefix_low, prefix_high = reg.vdeinterleave(
                reg.vreinterpret_lanes(prefix0, cb.dtypes.uint8),
                reg.vreinterpret_lanes(prefix1, cb.dtypes.uint8),
            )
            local_prefix = reg.vgather_reg(prefix_low, keys)
            subtract0, subtract1 = reg.vinterleave(local_prefix, zero8)
            # 4. 全块排名 = 桶写游标 + 当前有序 lane - 本向量桶起点。
            # 减去桶起点后得到桶内秩，同键仍按原 lane 的先后写入。
            rank0 = reg.vsub(
                reg.vadd(
                    reg.vreinterpret_lanes(bucket0, cb.dtypes.uint16),
                    local0,
                    mask=mask16,
                ),
                reg.vreinterpret_lanes(subtract0, cb.dtypes.uint16),
                mask=mask16,
            )
            rank1 = reg.vsub(
                reg.vadd(
                    reg.vreinterpret_lanes(bucket1, cb.dtypes.uint16),
                    local1,
                    mask=mask16,
                ),
                reg.vreinterpret_lanes(subtract1, cb.dtypes.uint16),
                mask=mask16,
            )
            # 将原 lane 加上向量起点，并扩为 uint32 地址/索引后散射 permutation。
            index0, index1 = reg.vinterleave(index, zero8)
            original0 = reg.vadds(
                reg.vreinterpret_lanes(index0, cb.dtypes.uint16),
                cb.dtypes.uint16(chunk * VECTOR_BYTES),
                mask=mask16,
            )
            original1 = reg.vadds(
                reg.vreinterpret_lanes(index1, cb.dtypes.uint16),
                cb.dtypes.uint16(chunk * VECTOR_BYTES),
                mask=mask16,
            )
            ranks0, ranks1 = reg.vinterleave(rank0, zero16)
            ranks2, ranks3 = reg.vinterleave(rank1, zero16)
            upper0 = zero16
            upper1 = zero16
            if pack_digits:
                # 多核用 uint32 高 16 位携带桶号；局部原位置受 W≤65280 约束。
                # 其他 caller 默认关闭，继续得到纯 originalLocalPosition。
                digit0, digit1 = reg.vinterleave(keys, zero8)
                upper0 = reg.vreinterpret_lanes(digit0, cb.dtypes.uint16)
                upper1 = reg.vreinterpret_lanes(digit1, cb.dtypes.uint16)
            ids0, ids1 = reg.vinterleave(original0, upper0)
            ids2, ids3 = reg.vinterleave(original1, upper1)
            remaining = reg.mask_counter(length - chunk * VECTOR_BYTES)
            valid0, remaining = reg.update_mask(remaining, elem_bits=32)
            valid1, remaining = reg.update_mask(remaining, elem_bits=32)
            valid2, remaining = reg.update_mask(remaining, elem_bits=32)
            valid3, remaining = reg.update_mask(remaining, elem_bits=32)
            reg.vscatter(
                permutation,
                reg.vreinterpret_lanes(ids0, cb.dtypes.uint32),
                reg.vreinterpret_lanes(ranks0, cb.dtypes.uint32),
                mask=valid0,
            )
            reg.vscatter(
                permutation,
                reg.vreinterpret_lanes(ids1, cb.dtypes.uint32),
                reg.vreinterpret_lanes(ranks1, cb.dtypes.uint32),
                mask=valid1,
            )
            reg.vscatter(
                permutation,
                reg.vreinterpret_lanes(ids2, cb.dtypes.uint32),
                reg.vreinterpret_lanes(ranks2, cb.dtypes.uint32),
                mask=valid2,
            )
            reg.vscatter(
                permutation,
                reg.vreinterpret_lanes(ids3, cb.dtypes.uint32),
                reg.vreinterpret_lanes(ranks3, cb.dtypes.uint32),
                mask=valid3,
            )
            # 给后续向量预留本向量的同桶位置，保持跨向量的输入次序。
            offsets0 = reg.vadd(offsets0, frequency0, mask=mask16)
            offsets1 = reg.vadd(offsets1, frequency1, mask=mask16)


@cb.jit
def radix_permutation(digits, permutation, byte_order, offsets, length):
    # 对应 toolkit SortU8ElementsWithRound 的三个阶段。
    # permutation[排序位置] = 原位置；offsets 始终保存整 tile 的独占桶起点。
    # caller 提供 256 对齐宽度，尾 digit=255，且所有缓冲互不重叠。
    get_exclusive_sum(digits, offsets, length)
    eight_bits_sort(digits, byte_order, length)
    local_sort(digits, byte_order, offsets, permutation, length)


@cb.jit
def extract_key_digit(
    keys,
    digits,
    offsets,
    length,
    width: cb.Constexpr[int],
    storage: cb.Constexpr,
    byte: cb.Constexpr[int],
):
    # toolkit ConvertRegToWithShift：先选 32 位字，再移位、DeInterleave 压紧。
    words = cb.reinterpret(
        keys,
        cb.dtypes.uint32,
        cb.make_layout((width * (2 if storage == cb.dtypes.uint64 else 1),)),
    )
    with cb.vf(mode="simd"):
        mask32 = reg.create_mask(pattern="all", elem_bits=32)
        mask8 = reg.create_mask(pattern="all", elem_bits=8)
        padding = reg.vdups(255, cb.dtypes.uint8)
        mask16 = reg.create_mask(pattern="all", elem_bits=16)
        frequency0 = reg.vdups(0, cb.dtypes.uint16)
        frequency1 = reg.vdups(0, cb.dtypes.uint16)
        cumulative0 = reg.vdups(0, cb.dtypes.uint16)
        cumulative1 = reg.vdups(0, cb.dtypes.uint16)
        counter = reg.mask_counter(length)
        for chunk in range((length + VECTOR_BYTES - 1) // VECTOR_BYTES):
            valid, counter = reg.update_mask(counter, elem_bits=8)
            word0 = reg.vdups(0, cb.dtypes.uint32)
            if storage == cb.dtypes.uint64:
                low0, high0 = reg.vdeinterleave(
                    reg.vload(words, chunk * 512 + 0),
                    reg.vload(words, chunk * 512 + 64),
                )
                if byte < 4:
                    word0 = low0
                else:
                    word0 = high0
            else:
                word0 = reg.vload(words, chunk * 256 + 0)
            word0 = reg.vshr(word0, (byte % 4) * 8, mask=mask32)
            word1 = reg.vdups(0, cb.dtypes.uint32)
            if storage == cb.dtypes.uint64:
                low1, high1 = reg.vdeinterleave(
                    reg.vload(words, chunk * 512 + 128),
                    reg.vload(words, chunk * 512 + 192),
                )
                if byte < 4:
                    word1 = low1
                else:
                    word1 = high1
            else:
                word1 = reg.vload(words, chunk * 256 + 64)
            word1 = reg.vshr(word1, (byte % 4) * 8, mask=mask32)
            word2 = reg.vdups(0, cb.dtypes.uint32)
            if storage == cb.dtypes.uint64:
                low2, high2 = reg.vdeinterleave(
                    reg.vload(words, chunk * 512 + 256),
                    reg.vload(words, chunk * 512 + 320),
                )
                if byte < 4:
                    word2 = low2
                else:
                    word2 = high2
            else:
                word2 = reg.vload(words, chunk * 256 + 128)
            word2 = reg.vshr(word2, (byte % 4) * 8, mask=mask32)
            word3 = reg.vdups(0, cb.dtypes.uint32)
            if storage == cb.dtypes.uint64:
                low3, high3 = reg.vdeinterleave(
                    reg.vload(words, chunk * 512 + 384),
                    reg.vload(words, chunk * 512 + 448),
                )
                if byte < 4:
                    word3 = low3
                else:
                    word3 = high3
            else:
                word3 = reg.vload(words, chunk * 256 + 192)
            word3 = reg.vshr(word3, (byte % 4) * 8, mask=mask32)
            half0, unused0 = reg.vdeinterleave(
                reg.vreinterpret_lanes(word0, cb.dtypes.uint16),
                reg.vreinterpret_lanes(word1, cb.dtypes.uint16),
            )
            half1, unused1 = reg.vdeinterleave(
                reg.vreinterpret_lanes(word2, cb.dtypes.uint16),
                reg.vreinterpret_lanes(word3, cb.dtypes.uint16),
            )
            packed, unused2 = reg.vdeinterleave(
                reg.vreinterpret_lanes(half0, cb.dtypes.uint8),
                reg.vreinterpret_lanes(half1, cb.dtypes.uint8),
            )
            packed = reg.vmerge(valid, packed, padding)
            reg.vstore(digits, chunk * VECTOR_BYTES, packed, mask8)
            frequency0 = reg.vadd(
                frequency0,
                reg.vhistogram_frequency(packed, mask=valid, bin=0),
                mask=mask16,
            )
            frequency1 = reg.vadd(
                frequency1,
                reg.vhistogram_frequency(packed, mask=valid, bin=1),
                mask=mask16,
            )
            cumulative0 = reg.vhistogram_accumulate(
                cumulative0, packed, mask=valid, bin=0
            )
            cumulative1 = reg.vhistogram_accumulate(
                cumulative1, packed, mask=valid, bin=1
            )
        reg.vstore(offsets, 0, reg.vsub(cumulative0, frequency0, mask=mask16), mask16)
        reg.vstore(
            offsets,
            HISTOGRAM_HALF_BINS,
            reg.vsub(cumulative1, frequency1, mask=mask16),
            mask16,
        )


@cb.jit
def update_keys_and_ids(
    source,
    source_ids,
    destination,
    destination_ids,
    permutation,
    length,
    width: cb.Constexpr[int],
    storage: cb.Constexpr,
):
    # toolkit UpdateValueAndIndexByLocalIndex / GatherAndStoreB64Elements。
    source_words = cb.reinterpret(
        source,
        cb.dtypes.uint32,
        cb.make_layout((width * (2 if storage == cb.dtypes.uint64 else 1),)),
    )
    destination_words = cb.reinterpret(
        destination,
        cb.dtypes.uint32,
        cb.make_layout((width * (2 if storage == cb.dtypes.uint64 else 1),)),
    )
    with cb.vf(mode="simd"):
        counter = reg.mask_counter(length)
        for chunk in range((length + 63) // 64):
            valid, counter = reg.update_mask(counter, elem_bits=32)
            order = reg.vload(permutation, chunk * 64)
            ids = reg.vgather(source_ids, order, mask=valid)
            reg.vstore(destination_ids, chunk * 64, ids, valid)
            if storage == cb.dtypes.uint64:
                low_index = reg.vshl(order, 1, mask=valid)
                high_index = reg.vadds(low_index, 1, mask=valid)
                low = reg.vgather(source_words, low_index, mask=valid)
                high = reg.vgather(source_words, high_index, mask=valid)
                packed0, packed1 = reg.vinterleave(low, high)
                remaining = reg.mask_counter((length - chunk * 64) * 2)
                valid0, remaining = reg.update_mask(remaining, elem_bits=32)
                valid1, remaining = reg.update_mask(remaining, elem_bits=32)
                reg.vstore(destination_words, chunk * 128, packed0, valid0)
                reg.vstore(destination_words, chunk * 128 + 64, packed1, valid1)
            else:
                keys = reg.vgather(source_words, order, mask=valid)
                reg.vstore(destination_words, chunk * 64, keys, valid)


@cb.jit
def update_ids(source_ids, destination_ids, permutation, length):
    # caller 只消费排序索引时，末轮不再物化无后续读取的键。
    with cb.vf(mode="simd"):
        counter = reg.mask_counter(length)
        for chunk in range((length + 63) // 64):
            valid, counter = reg.update_mask(counter, elem_bits=32)
            order = reg.vload(permutation, chunk * 64)
            reg.vstore(
                destination_ids,
                chunk * 64,
                reg.vgather(source_ids, order, mask=valid),
                valid,
            )


@cb.jit
def sort_local_keys(
    ping,
    pong,
    ping_ids,
    pong_ids,
    digits,
    permutation,
    byte_order,
    offsets,
    length,
    width: cb.Constexpr[int],
    storage: cb.Constexpr,
    passes: cb.Constexpr[int],
    ids_only: cb.Constexpr[bool] = False,
):
    # byte 从 0 开始：偶数 byte 写 pong，奇数 byte 写 ping；总 passes 为偶数时结果在 ping。
    # passes 按有效键字节数确定，可小于载体宽度；ids_only 仅省去末轮键写回。
    for byte in cb.range_constexpr(passes):
        if byte % 2 == 0:
            extract_key_digit(ping, digits, offsets, length, width, storage, byte)
        else:
            extract_key_digit(pong, digits, offsets, length, width, storage, byte)
        eight_bits_sort(digits, byte_order, length)
        local_sort(digits, byte_order, offsets, permutation, length)
        if ids_only and byte == passes - 1:
            if byte % 2 == 0:
                update_ids(ping_ids, pong_ids, permutation, length)
            else:
                update_ids(pong_ids, ping_ids, permutation, length)
        else:
            if byte % 2 == 0:
                update_keys_and_ids(
                    ping,
                    ping_ids,
                    pong,
                    pong_ids,
                    permutation,
                    length,
                    width,
                    storage,
                )
            else:
                update_keys_and_ids(
                    pong,
                    pong_ids,
                    ping,
                    ping_ids,
                    permutation,
                    length,
                    width,
                    storage,
                )


@cb.jit
def load_keys32(
    x,
    keys,
    ids,
    source,
    length,
    width: cb.Constexpr[int],
    dtype: cb.Constexpr,
    descending: cb.Constexpr[bool],
):
    # 与 AscendC LoadContiguousBatch/TwiddleInData 对齐：DMA 连续输入，SIMD 构键。
    raw = cb.reinterpret(keys, dtype, cb.make_layout((width,)))
    count64 = cb.dtypes.int64(length)
    cb.mem_copy(
        cb.strided_slice(raw, (0,), (count64,)),
        cb.strided_slice(x, (source,), (count64,)),
    )
    with cb.vf(mode="simd"):
        counter = reg.mask_counter(length)
        lane = reg.vreinterpret(reg.varange(0, cb.dtypes.int32), cb.dtypes.uint32)
        for chunk in range((length + 63) // 64):
            valid, counter = reg.update_mask(counter, elem_bits=32)
            bits = reg.vload(keys, chunk * 64)
            ordered = bits
            if dtype == cb.dtypes.float32:
                sign = reg.vdups(0x80000000, cb.dtypes.uint32)
                magnitude = reg.vbitwise_and(
                    bits, reg.vdups(0x7FFFFFFF, cb.dtypes.uint32), mask=valid
                )
                positive = reg.vbitwise_xor(bits, sign, mask=valid)
                negative = reg.vnot(bits, mask=valid)
                ordered = reg.vselect(
                    negative,
                    positive,
                    cond_mask=reg.vges(bits, cb.dtypes.uint32(0x80000000), mask=valid),
                )
                ordered = reg.vselect(
                    sign,
                    ordered,
                    cond_mask=reg.veqs(magnitude, cb.dtypes.uint32(0), mask=valid),
                )
                ordered = reg.vselect(
                    reg.vdups(0xFFFFFFFF, cb.dtypes.uint32),
                    ordered,
                    cond_mask=reg.vgts(
                        magnitude, cb.dtypes.uint32(0x7F800000), mask=valid
                    ),
                )
            elif dtype == cb.dtypes.int32:
                ordered = reg.vbitwise_xor(
                    bits, reg.vdups(0x80000000, cb.dtypes.uint32), mask=valid
                )
            if descending:
                ordered = reg.vnot(ordered, mask=valid)
            reg.vstore(keys, chunk * 64, ordered, valid)
            original = reg.vadds(lane, cb.dtypes.uint32(chunk * 64), mask=valid)
            reg.vstore(ids, chunk * 64, original, valid)
