# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""块内稳定 Radix：生成字节排列，再随键传递原位置。"""

import cannbotdsl as cb
from cannbotdsl import reg

from sort_common import BITS_PER_BYTE, RADIX_BINS, radix_digit

# arch35 的向量寄存器宽度；16 位计数将 256 个桶分成两个寄存器保存。
VECTOR_BYTES = 256
HISTOGRAM_HALF_BINS = RADIX_BINS // 2
PARTITION_BITS = 2
PARTITION_GROUPS = 1 << PARTITION_BITS
BYTE_PARTITION_PASSES = BITS_PER_BYTE // PARTITION_BITS
LOCAL_RADIX_THREADS = 256


@cb.jit
def radix_permutation(digits, permutation, scratch, byte_order, offsets, length):
    # 返回「排序位置 → 原位置」的稳定排列，以及各桶的独占起点。
    # 先统计整块桶计数，再在每个向量内按两位分组压缩，最后叠加跨向量桶偏移。
    # 整块统计：bin=0/1 分别对应桶 0..127/128..255；累计值减频数得到独占桶起点。
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
    # 每个向量做四次稳定 2-bit 分组，形成按完整字节排序的 lane 排列。
    for step in cb.range_constexpr(BYTE_PARTITION_PASSES):
        with cb.vf(mode="simd"):
            mask8 = reg.create_mask(pattern="all", elem_bits=8)
            cursor = reg.vstore_unalign_begin(scratch)
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
                    reg.vsqueeze_and_storeunalign(scratch, 0, compact, cursor)
            reg.vsqueeze_and_storeunalign_finalize(scratch, 0, cursor)
            reg.vmem_bar("vst_vld")
            for chunk in range((length + VECTOR_BYTES - 1) // VECTOR_BYTES):
                index = reg.vload(scratch, chunk * VECTOR_BYTES)
                reg.vstore(byte_order, chunk * VECTOR_BYTES, index, mask8)
            reg.vmem_bar("vst_vld")
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
            ids0, ids1 = reg.vinterleave(original0, zero16)
            ids2, ids3 = reg.vinterleave(original1, zero16)
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
def sort_local_keys(
    ping,
    pong,
    ping_ids,
    pong_ids,
    digits,
    permutation,
    scratch,
    byte_order,
    offsets,
    length,
    width: cb.Constexpr[int],
    storage: cb.Constexpr,
    passes: cb.Constexpr[int],
):
    # byte 从 0 开始：偶数 byte 写 pong，奇数 byte 写 ping；总 passes 为偶数时结果在 ping。
    # passes 按原 dtype 字节数确定，可小于键存储宽度；键与原索引始终一起重排。
    for byte in cb.range_constexpr(passes):
        with cb.vf(mode="simt", thread=LOCAL_RADIX_THREADS):
            tid = cb.dtypes.int32(cb.simt.thread_idx()[0])
            for axis in range(tid, width, LOCAL_RADIX_THREADS):
                # 填充放最大桶；稳定分组使其留在真实元素之后，最终 mask 只输出真实元素。
                digit = cb.dtypes.uint8(RADIX_BINS - 1)
                if axis < length:
                    if byte % 2 == 0:
                        digit = cb.dtypes.uint8(radix_digit(ping[axis], storage, byte))
                    else:
                        digit = cb.dtypes.uint8(radix_digit(pong[axis], storage, byte))
                digits[axis] = digit
        radix_permutation(digits, permutation, scratch, byte_order, offsets, length)
        with cb.vf(mode="simt", thread=LOCAL_RADIX_THREADS):
            tid = cb.dtypes.int32(cb.simt.thread_idx()[0])
            for axis in range(tid, cb.dtypes.int32(length), LOCAL_RADIX_THREADS):
                original = cb.dtypes.int32(permutation[axis])
                if byte % 2 == 0:
                    pong[axis] = ping[original]
                    pong_ids[axis] = ping_ids[original]
                else:
                    ping[axis] = pong[original]
                    ping_ids[axis] = pong_ids[original]
