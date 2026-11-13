# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""多核长轴流水：SIMD 统计、局部排列与无等待 lookback 散射。"""

import cannbotdsl as cb


from cannbotdsl import reg
from radix import (
    VECTOR_BYTES,
    get_exclusive_sum,
    eight_bits_sort,
    local_sort,
)
from sort_common import (
    RADIX_BINS,
    RADIX_SCATTER_THREADS,
    dtype_bytes,
    ordered_key,
    key_dtype,
    radix_digit,
)


HISTOGRAM_THREADS = RADIX_BINS  # 每个线程发布一个桶的计数。


@cb.jit
def copy_storage(dst, src):
    # 按视图的有效范围搬运字节，保持原位模式。
    cb.mem_copy(dst.view(cb.dtypes.uint8), src.view(cb.dtypes.uint8))


@cb.kernel
class RadixMoreCore:
    """单次 launch 内完成统计及局部排列、无等待 lookback、稳定散射。"""

    def __init__(
        self, dtype, descending, tile_size, counter_type=cb.dtypes.int64, active_rows=0
    ):
        self.dtype = dtype
        self.descending = descending
        self.tile_size = tile_size
        self.counter_type = counter_type
        self.active_rows = active_rows
        # 跨核只交换原值和原轴索引；保序键仅在提取 digit 的寄存器中存在。
        self.digits = cb.Buffer(
            cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((self.tile_size,))
        )
        self.permutation = cb.Buffer(
            cb.MemLoc.UB, cb.dtypes.uint32, cb.make_layout((self.tile_size,))
        )
        self.byte_order = cb.Buffer(
            cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((self.tile_size,))
        )
        # 当前 tile 的局部桶起点。
        self.offsets = cb.Buffer(
            cb.MemLoc.UB, cb.dtypes.uint16, cb.make_layout((RADIX_BINS,))
        )
        self.raw_tile = cb.Buffer(cb.MemLoc.UB, dtype, cb.make_layout((tile_size,)))
        self.index_tile = cb.Buffer(
            cb.MemLoc.UB, counter_type, cb.make_layout((tile_size,))
        )
        self.scatter_bases = cb.Buffer(
            cb.MemLoc.UB, counter_type, cb.make_layout((RADIX_BINS,))
        )

    def __call__(
        self,
        x: cb.Tensor,
        values: cb.Tensor,
        indices: cb.Tensor,
        ping: cb.Tensor,
        ping_ids: cb.Tensor,
        pong: cb.Tensor,
        pong_ids: cb.Tensor,
        hist: cb.Tensor,
        states: cb.Tensor,
        rows: cb.dtypes.int64,
        length: cb.dtypes.int64,
        inner: cb.dtypes.int64,
        blocks: cb.dtypes.int64,
    ):
        batch_rows = rows
        if self.active_rows > 0:
            batch_rows = cb.dtypes.int64(self.active_rows)
        for row_start in range(0, rows, batch_rows):
            valid_rows = rows - row_start
            if valid_rows > batch_rows:
                valid_rows = batch_rows
            # 首轮直接读取 x；后续轮读取当前原值/索引，不归一或覆盖原始位模式。
            # 屏障位于任务分支之外，空闲核也必须参加；末轮直接写最终输出。
            tiles = (length + self.tile_size - 1) // self.tile_size
            self.clear_states(states, valid_rows, tiles, blocks)
            cb.global_sync_all()
            for byte_index in cb.range_constexpr(dtype_bytes(self.dtype)):
                if byte_index % 2 == 0:
                    self.histogram(
                        x,
                        ping,
                        states,
                        hist,
                        valid_rows,
                        length,
                        inner,
                        blocks,
                        row_start,
                        byte_index,
                        byte_index == 0,
                    )
                else:
                    self.histogram(
                        x,
                        pong,
                        states,
                        hist,
                        valid_rows,
                        length,
                        inner,
                        blocks,
                        row_start,
                        byte_index,
                        False,
                    )
                # lookback 的回退读取与全局桶起点都依赖完整统计。
                cb.global_sync_all()
                if byte_index % 2 == 0:
                    self.scatter(
                        x,
                        values,
                        indices,
                        ping,
                        ping_ids,
                        pong,
                        pong_ids,
                        hist,
                        states,
                        valid_rows,
                        length,
                        inner,
                        blocks,
                        row_start,
                        byte_index,
                        byte_index == dtype_bytes(self.dtype) - 1,
                    )
                else:
                    self.scatter(
                        x,
                        values,
                        indices,
                        pong,
                        pong_ids,
                        ping,
                        ping_ids,
                        hist,
                        states,
                        valid_rows,
                        length,
                        inner,
                        blocks,
                        row_start,
                        byte_index,
                        byte_index == dtype_bytes(self.dtype) - 1,
                    )
                if byte_index + 1 < dtype_bytes(self.dtype):
                    # 下一字节开始读取前，所有散射目标必须写入完成。
                    cb.global_sync_all()
            # 最后 Scatter 仍读取共享 hist/states，全部核完成后才复用。
            if row_start + batch_rows < rows:
                cb.global_sync_all()

    @cb.jit
    def load_value_tile(
        self,
        x: cb.Tensor,
        current_values: cb.Tensor,
        source,
        value_source,
        valid,
        byte_index: cb.Constexpr[int],
        first_pass: cb.Constexpr[bool],
    ):
        if first_pass:
            copy_storage(
                cb.strided_slice(self.raw_tile, (0,), (valid,)),
                cb.strided_slice(x, (source,), (valid,)),
            )
        else:
            copy_storage(
                cb.strided_slice(self.raw_tile, (0,), (valid,)),
                cb.strided_slice(current_values, (value_source,), (valid,)),
            )
        self.extract_value_digit(
            self.raw_tile, valid, byte_index, dtype_bytes(self.dtype)
        )

    @cb.jit
    def load_digit_word(self, storage, offset, byte_index: cb.Constexpr[int]):
        source_bytes = dtype_bytes(self.dtype)
        word = reg.vdups(0, cb.dtypes.uint32)
        if source_bytes == 8:
            words = storage
            low, high = reg.vdeinterleave(
                reg.vload(words, offset * 2), reg.vload(words, offset * 2 + 64)
            )
            if byte_index < 4:
                word = low
            else:
                word = high
        else:
            carrier = (
                cb.dtypes.uint8
                if source_bytes == 1
                else cb.dtypes.uint16
                if source_bytes == 2
                else cb.dtypes.uint32
            )
            word = reg.vreinterpret(reg.vload(storage, offset), carrier)
        bits = 32 if source_bytes == 8 else source_bytes * 8
        mask = reg.create_mask(elem_bits=bits)
        if self.dtype in (cb.dtypes.float16, cb.dtypes.bfloat16, cb.dtypes.float32):
            carrier = cb.dtypes.uint16 if source_bytes == 2 else cb.dtypes.uint32
            sign = 1 << (bits - 1)
            infinity = (
                0x7C00
                if self.dtype == cb.dtypes.float16
                else 0x7F80
                if self.dtype == cb.dtypes.bfloat16
                else 0x7F800000
            )
            magnitude = reg.vbitwise_and(word, reg.vdups(sign - 1, carrier), mask=mask)
            positive = reg.vbitwise_xor(word, reg.vdups(sign, carrier), mask=mask)
            negative = reg.vnot(word, mask=mask)
            ordered = reg.vselect(
                negative, positive, cond_mask=reg.vges(word, sign, mask=mask)
            )
            ordered = reg.vselect(
                reg.vdups(sign, carrier),
                ordered,
                cond_mask=reg.veqs(magnitude, 0, mask=mask),
            )
            word = reg.vselect(
                reg.vdups((1 << bits) - 1, carrier),
                ordered,
                cond_mask=reg.vgts(magnitude, infinity, mask=mask),
            )
        elif self.dtype in (
            cb.dtypes.int8,
            cb.dtypes.int16,
            cb.dtypes.int32,
            cb.dtypes.int64,
        ):
            if source_bytes != 8 or byte_index >= 4:
                carrier = (
                    cb.dtypes.uint8
                    if source_bytes == 1
                    else cb.dtypes.uint16
                    if source_bytes == 2
                    else cb.dtypes.uint32
                )
                word = reg.vbitwise_xor(
                    word, reg.vdups(1 << (bits - 1), carrier), mask=mask
                )
        if self.descending:
            word = reg.vnot(word, mask=mask)
        return reg.vshr(word, (byte_index % (bits // 8)) * 8, mask=mask)

    @cb.jit
    def extract_value_digit(
        self,
        storage,
        valid,
        byte_index: cb.Constexpr[int],
        source_bytes: cb.Constexpr[int],
    ):
        word_storage = storage
        if source_bytes == 8:
            word_storage = cb.reinterpret(
                storage, cb.dtypes.uint32, cb.make_layout((self.tile_size * 2,))
            )
        with cb.vf(mode="simd"):
            mask8 = reg.create_mask(elem_bits=8)
            mask16 = reg.create_mask(elem_bits=16)
            fill = reg.vdups(255, cb.dtypes.uint8)
            frequency0 = reg.vdups(0, cb.dtypes.uint16)
            frequency1 = reg.vdups(0, cb.dtypes.uint16)
            cumulative0 = reg.vdups(0, cb.dtypes.uint16)
            cumulative1 = reg.vdups(0, cb.dtypes.uint16)
            counter = reg.mask_counter(valid)
            for chunk in range((valid + VECTOR_BYTES - 1) // VECTOR_BYTES):
                tail, counter = reg.update_mask(counter, elem_bits=8)
                packed = reg.vdups(0, cb.dtypes.uint8)
                if source_bytes == 1:
                    packed = self.load_digit_word(
                        word_storage, chunk * VECTOR_BYTES, byte_index
                    )
                elif source_bytes == 2:
                    word0 = self.load_digit_word(
                        word_storage, chunk * VECTOR_BYTES, byte_index
                    )
                    word1 = self.load_digit_word(
                        word_storage, chunk * VECTOR_BYTES + 128, byte_index
                    )
                    packed, unused = reg.vdeinterleave(
                        reg.vreinterpret_lanes(word0, cb.dtypes.uint8),
                        reg.vreinterpret_lanes(word1, cb.dtypes.uint8),
                    )
                else:
                    word0 = self.load_digit_word(
                        word_storage, chunk * VECTOR_BYTES, byte_index
                    )
                    word1 = self.load_digit_word(
                        word_storage, chunk * VECTOR_BYTES + 64, byte_index
                    )
                    word2 = self.load_digit_word(
                        word_storage, chunk * VECTOR_BYTES + 128, byte_index
                    )
                    word3 = self.load_digit_word(
                        word_storage, chunk * VECTOR_BYTES + 192, byte_index
                    )
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
                packed = reg.vmerge(tail, packed, fill)
                reg.vstore(self.digits, chunk * VECTOR_BYTES, packed, mask8)
                frequency0 = reg.vadd(
                    frequency0,
                    reg.vhistogram_frequency(packed, mask=tail, bin=0),
                    mask=mask16,
                )
                frequency1 = reg.vadd(
                    frequency1,
                    reg.vhistogram_frequency(packed, mask=tail, bin=1),
                    mask=mask16,
                )
                cumulative0 = reg.vhistogram_accumulate(
                    cumulative0, packed, mask=tail, bin=0
                )
                cumulative1 = reg.vhistogram_accumulate(
                    cumulative1, packed, mask=tail, bin=1
                )
            reg.vstore(
                self.offsets, 0, reg.vsub(cumulative0, frequency0, mask=mask16), mask16
            )
            reg.vstore(
                self.offsets,
                128,
                reg.vsub(cumulative1, frequency1, mask=mask16),
                mask16,
            )

    @cb.jit
    def clear_states(self, states: cb.Tensor, rows, tiles, blocks):
        count = rows * (tiles + 1) * RADIX_BINS * dtype_bytes(self.dtype)
        block_id = cb.dtypes.int64(cb.get_block_idx())
        with cb.vf(mode="simt", thread=RADIX_BINS):
            thread = cb.dtypes.int64(cb.simt.thread_idx()[0])
            for position in range(
                block_id * RADIX_BINS + thread, count, blocks * RADIX_BINS
            ):
                cb.simt.l2cache_store(states.ptr(position), self.counter_type(0))
            cb.simt.threadfence()

    @cb.jit
    def prepare_digits(
        self,
        x: cb.Tensor,
        current_values: cb.Tensor,
        source,
        value_source,
        valid,
        inner,
        byte_index: cb.Constexpr[int],
        first_pass: cb.Constexpr[bool],
    ):
        # Histogram 和 Scatter 共用原值提取与桶统计，连续路径每轮均融合为一个 SIMD VF。
        if inner == 1:
            self.load_value_tile(
                x, current_values, source, value_source, valid, byte_index, first_pass
            )
        else:
            with cb.vf(mode="simt", thread=HISTOGRAM_THREADS):
                thread = cb.dtypes.int64(cb.simt.thread_idx()[0])
                for local in range(thread, self.tile_size, HISTOGRAM_THREADS):
                    digit = cb.dtypes.uint8(RADIX_BINS - 1)
                    if local < valid:
                        key = key_dtype(self.dtype)(0)
                        if first_pass:
                            key = ordered_key(
                                cb.simt.l2cache_load(x.ptr(source + local * inner)),
                                self.dtype,
                                self.descending,
                            )
                        else:
                            key = ordered_key(
                                cb.simt.l2cache_load(
                                    current_values.ptr(value_source + local)
                                ),
                                self.dtype,
                                self.descending,
                            )
                        digit = cb.dtypes.uint8(
                            radix_digit(key, key_dtype(self.dtype), byte_index)
                        )
                    self.digits[local] = digit
                cb.simt.threadfence()
        if inner != 1:
            get_exclusive_sum(self.digits, self.offsets, valid)

    @cb.jit
    def histogram(
        self,
        x: cb.Tensor,
        current_values: cb.Tensor,
        states: cb.Tensor,
        hist: cb.Tensor,
        rows: cb.dtypes.int64,
        length: cb.dtypes.int64,
        inner: cb.dtypes.int64,
        blocks: cb.dtypes.int64,
        row_start: cb.dtypes.int64,
        byte_index: cb.Constexpr[int],
        first_pass: cb.Constexpr[bool],
    ):
        tiles = (length + self.tile_size - 1) // self.tile_size
        for task in range(cb.dtypes.int64(cb.get_block_idx()), rows * tiles, blocks):
            row = task // tiles
            start = task % tiles * self.tile_size
            # 将逻辑行 row 还原为 [O,N,I] 的 (o,i)，轴内相邻元素在 GM 中相隔 inner。
            global_row = row_start + row
            source_base = global_row // inner * length * inner + global_row % inner
            valid = length - start
            if valid > self.tile_size:
                valid = cb.dtypes.int64(self.tile_size)
            self.prepare_digits(
                x,
                current_values,
                source_base + start * inner,
                row * length + start,
                valid,
                inner,
                byte_index,
                first_pass,
            )
            with cb.vf(mode="simt", thread=HISTOGRAM_THREADS):
                thread = cb.dtypes.int64(cb.simt.thread_idx()[0])
                end = valid
                if thread + 1 < RADIX_BINS:
                    end = cb.dtypes.int64(self.offsets[thread + 1])
                cb.simt.l2cache_store(
                    hist.ptr(task * RADIX_BINS + thread),
                    hist.dtype(end - cb.dtypes.int64(self.offsets[thread])),
                )
                base = ((byte_index * rows + row) * (tiles + 1) + tiles) * RADIX_BINS
                cb.simt.atomic_add(
                    states.ptr(base + thread),
                    self.counter_type(self.offsets[thread]),
                )
                cb.simt.threadfence()

    @cb.jit
    def scatter(
        self,
        x: cb.Tensor,
        values: cb.Tensor,
        indices: cb.Tensor,
        current_values: cb.Tensor,
        source_indices: cb.Tensor,
        output_values: cb.Tensor,
        output_indices: cb.Tensor,
        hist: cb.Tensor,
        states: cb.Tensor,
        rows: cb.dtypes.int64,
        length: cb.dtypes.int64,
        inner: cb.dtypes.int64,
        blocks: cb.dtypes.int64,
        row_start: cb.dtypes.int64,
        byte_index: cb.Constexpr[int],
        final_pass: cb.Constexpr[bool],
    ):
        tiles = (length + self.tile_size - 1) // self.tile_size
        for task in range(cb.dtypes.int64(cb.get_block_idx()), rows * tiles, blocks):
            row = task // tiles
            row_base = row * length
            global_row = row_start + row
            output_base = global_row // inner * length * inner + global_row % inner
            start = task % tiles * self.tile_size
            valid = length - start
            if valid > self.tile_size:
                valid = cb.dtypes.int64(self.tile_size)
            self.prepare_digits(
                x,
                current_values,
                output_base + start * inner,
                row_base + start,
                valid,
                inner,
                byte_index,
                byte_index == 0,
            )
            if byte_index != 0 and inner == 1:
                copy_storage(
                    cb.strided_slice(self.index_tile, (0,), (valid,)),
                    cb.strided_slice(source_indices, (row_base + start,), (valid,)),
                )
            eight_bits_sort(self.digits, self.byte_order, valid)
            local_sort(
                self.digits,
                self.byte_order,
                self.offsets,
                self.permutation,
                valid,
                True,
            )
            state_base = (byte_index * rows + row) * (tiles + 1) * RADIX_BINS
            tile_id = task % tiles
            with cb.vf(mode="simt", thread=RADIX_BINS):
                digit = cb.dtypes.int64(cb.simt.thread_idx()[0])
                frequency = cb.simt.l2cache_load(hist.ptr(task * RADIX_BINS + digit))
                inclusive = frequency
                # 已发布状态编码为 -inclusive-1；0 表示未发布。
                # 负编码在 N 达到 int64 上界时也不需要 N+1 的正值。
                # 未发布时读取 Histogram 已完成的频数，不等待前驱核。
                previous = tile_id - 1
                while previous >= 0:
                    cached = cb.simt.l2cache_load(
                        states.ptr(state_base + previous * RADIX_BINS + digit)
                    )
                    if cached != self.counter_type(0):
                        inclusive = inclusive - (cached + self.counter_type(1))
                        previous = cb.dtypes.int64(-1)
                    else:
                        inclusive = inclusive + cb.simt.l2cache_load(
                            hist.ptr((row * tiles + previous) * RADIX_BINS + digit)
                        )
                        previous = previous - 1
                cb.simt.l2cache_store(
                    states.ptr(state_base + tile_id * RADIX_BINS + digit),
                    -inclusive - self.counter_type(1),
                )
                bucket_base = cb.simt.l2cache_load(
                    states.ptr(state_base + tiles * RADIX_BINS + digit)
                )
                self.scatter_bases[digit] = (
                    bucket_base
                    + inclusive
                    - frequency
                    - self.counter_type(self.offsets[digit])
                )
                cb.simt.threadfence()
            with cb.vf(mode="simt", thread=RADIX_SCATTER_THREADS):
                thread = cb.dtypes.int64(cb.simt.thread_idx()[0])
                for position in range(thread, self.tile_size, RADIX_SCATTER_THREADS):
                    if start + position < length:
                        # 低 16 位是 tile 内原位置，高 16 位是桶号，省去 digits 随机读取。
                        packed = self.permutation[position]
                        original = cb.dtypes.int64(packed % cb.dtypes.uint32(65536))
                        # 首字节原轴坐标已知；后续轮原值和原索引成对传递。
                        source_axis = start + original
                        raw = x.dtype(0)
                        if inner == 1:
                            raw = self.raw_tile[original]
                        else:
                            if byte_index == 0:
                                raw = cb.simt.l2cache_load(
                                    x.ptr(output_base + source_axis * inner)
                                )
                            else:
                                raw = cb.simt.l2cache_load(
                                    current_values.ptr(row_base + start + original)
                                )
                        if byte_index != 0:
                            if inner == 1:
                                source_axis = cb.dtypes.int64(self.index_tile[original])
                            else:
                                source_axis = cb.dtypes.int64(
                                    cb.simt.l2cache_load(
                                        source_indices.ptr(row_base + start + original)
                                    )
                                )
                        digit = cb.dtypes.int64(packed // cb.dtypes.uint32(65536))
                        destination = row_base + self.scatter_bases[digit] + position

                        # 所有轮次搬移原值；末轮直接输出，保留零符号和 NaN payload。
                        if final_pass:
                            output_position = (
                                output_base + (destination - row_base) * inner
                            )
                            cb.simt.l2cache_store(values.ptr(output_position), raw)
                            cb.simt.l2cache_store(
                                indices.ptr(output_position),
                                cb.dtypes.int64(source_axis),
                            )
                        else:
                            cb.simt.l2cache_store(output_values.ptr(destination), raw)
                            cb.simt.l2cache_store(
                                output_indices.ptr(destination),
                                output_indices.dtype(source_axis),
                            )
                cb.simt.threadfence()
