# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""多核长轴流水：共享 GM 工作区完成计数、跨块前缀和稳定散射。"""

import cannbotdsl as cb
from radix import VECTOR_BYTES, radix_permutation
from sort_common import (
    RADIX_BINS,
    RADIX_SCATTER_THREADS,
    dtype_bytes,
    ordered_key,
    radix_digit,
)


# SIMT warp 固定为 32 lane；必须满足 2**WARP_SCAN_STEPS == WARP_LANES。
# 五轮依次交换距离 1、2、4、8、16 的前驱值，才覆盖完整 warp 的前缀。
WARP_LANES = 32
WARP_SCAN_STEPS = 5
HISTOGRAM_THREADS = RADIX_BINS  # 每个线程发布一个桶的计数。


@cb.kernel
class RadixMoreCore:
    """单次 launch 内完成稳定的多核 LSD 排序：计数、全局前缀、散射。"""

    def __init__(self, dtype, descending, tile_size):
        self.dtype = dtype
        self.descending = descending
        self.tile_size = tile_size
        # 每核仅保留一个 tile 的局部排列；跨核交换的键和原轴索引放在 GM ping/pong 中。
        self.digits = cb.Buffer(cb.MemLoc.UB, (self.tile_size,), cb.dtypes.uint8)
        self.permutation = cb.Buffer(cb.MemLoc.UB, (self.tile_size,), cb.dtypes.uint32)
        self.scratch = cb.Buffer(
            cb.MemLoc.UB, (self.tile_size + VECTOR_BYTES,), cb.dtypes.uint8
        )
        self.byte_order = cb.Buffer(cb.MemLoc.UB, (self.tile_size,), cb.dtypes.uint8)
        # self.offsets 是块内 UB 桶起点，区别于 __call__ 参数 offsets 中的全局块/桶偏移。
        self.offsets = cb.Buffer(cb.MemLoc.UB, (RADIX_BINS,), cb.dtypes.uint16)
        self.totals = cb.Buffer(
            cb.MemLoc.UB, (RADIX_BINS // WARP_LANES,), cb.dtypes.int64
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
        offsets: cb.Tensor,
        ranks: cb.Tensor,
        rows: cb.dtypes.int64,
        length: cb.dtypes.int64,
        inner: cb.dtypes.int64,
        blocks: cb.dtypes.int64,
    ):
        # 偶数轮 ping → pong，奇数轮 pong → ping；首轮从 x 生成键与原轴索引。
        # 屏障位于任务分支之外，空闲核也必须参加；末轮直接写最终输出。
        for byte_index in cb.range_constexpr(dtype_bytes(self.dtype)):
            if byte_index % 2 == 0:
                self.histogram(
                    x,
                    ping,
                    ping_ids,
                    ranks,
                    hist,
                    rows,
                    length,
                    inner,
                    blocks,
                    byte_index,
                    byte_index == 0,
                )
            else:
                self.histogram(
                    x,
                    pong,
                    pong_ids,
                    ranks,
                    hist,
                    rows,
                    length,
                    inner,
                    blocks,
                    byte_index,
                    False,
                )
            # 全局前缀计算前，所有核的计数和块内排名必须可见。
            cb.global_sync_all()
            self.prefix(hist, offsets, rows, length, blocks)
            # 散射会读取其他核生成的偏移，因此所有核都必须到达屏障。
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
                    ranks,
                    offsets,
                    rows,
                    length,
                    inner,
                    blocks,
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
                    ranks,
                    offsets,
                    rows,
                    length,
                    inner,
                    blocks,
                    byte_index,
                    byte_index == dtype_bytes(self.dtype) - 1,
                )
            if byte_index + 1 < dtype_bytes(self.dtype):
                # 下一字节开始读取前，所有散射目标必须写入完成。
                cb.global_sync_all()

    @cb.jit
    def histogram(
        self,
        x: cb.Tensor,
        keys: cb.Tensor,
        indices: cb.Tensor,
        ranks: cb.Tensor,
        hist: cb.Tensor,
        rows: cb.dtypes.int64,
        length: cb.dtypes.int64,
        inner: cb.dtypes.int64,
        blocks: cb.dtypes.int64,
        byte_index: cb.Constexpr[int],
        first_pass: cb.Constexpr[bool],
    ):
        tiles = (length + self.tile_size - 1) // self.tile_size
        for task in range(cb.get_block_idx(), rows * tiles, blocks):
            row = task // tiles
            start = task % tiles * self.tile_size
            # 将逻辑行 row 还原为 [O,N,I] 的 (o,i)，轴内相邻元素在 GM 中相隔 inner。
            source_base = row // inner * length * inner + row % inner
            valid = length - start
            if valid > self.tile_size:
                valid = cb.dtypes.int64(self.tile_size)
            with cb.vf(mode="simt", thread=HISTOGRAM_THREADS):
                thread = cb.dtypes.int64(cb.simt.thread_idx()[0])
                for local in range(thread, self.tile_size, HISTOGRAM_THREADS):
                    # 尾部填充取最大桶，稳定排列后位于同桶真实元素之后。
                    digit = cb.dtypes.uint8(RADIX_BINS - 1)
                    if local < valid:
                        key = keys.dtype(0)
                        if first_pass:
                            key = ordered_key(
                                cb.simt.l2cache_load(
                                    x.ptr(source_base + (start + local) * inner)
                                ),
                                self.dtype,
                                self.descending,
                            )
                            cb.simt.l2cache_store(
                                keys.ptr(row * length + start + local), key
                            )
                            cb.simt.l2cache_store(
                                indices.ptr(row * length + start + local), start + local
                            )
                        else:
                            key = cb.simt.l2cache_load(
                                keys.ptr(row * length + start + local)
                            )
                        digit = cb.dtypes.uint8(
                            radix_digit(key, keys.dtype, byte_index)
                        )
                    self.digits[local] = digit
                cb.simt.threadfence()
            radix_permutation(
                self.digits,
                self.permutation,
                self.scratch,
                self.byte_order,
                self.offsets,
                valid,
            )
            with cb.vf(mode="simt", thread=HISTOGRAM_THREADS):
                thread = cb.dtypes.int64(cb.simt.thread_idx()[0])
                # ranks 打包「块内原位置 × tile_size + 桶内排名」，散射时除余解码。
                # tile_size 最大 8192，打包值小于 2^26，可放入 uint32。
                for position in range(thread, valid, HISTOGRAM_THREADS):
                    original = cb.dtypes.int64(self.permutation[position])
                    digit = cb.dtypes.int64(self.digits[original])
                    cb.simt.l2cache_store(
                        ranks.ptr(row * length + start + position),
                        cb.dtypes.uint32(original) * cb.dtypes.uint32(self.tile_size)
                        + cb.dtypes.uint32(
                            position - cb.dtypes.int64(self.offsets[digit])
                        ),
                    )
                end = valid
                if thread + 1 < RADIX_BINS:
                    end = cb.dtypes.int64(self.offsets[thread + 1])
                cb.simt.l2cache_store(
                    hist.ptr(task * RADIX_BINS + thread),
                    end - cb.dtypes.int64(self.offsets[thread]),
                )
                cb.simt.threadfence()

    @cb.jit
    def prefix(
        self,
        hist: cb.Tensor,
        offsets: cb.Tensor,
        rows: cb.dtypes.int64,
        length: cb.dtypes.int64,
        blocks: cb.dtypes.int64,
    ):
        block_id = cb.get_block_idx()
        with cb.vf(mode="simt", thread=RADIX_BINS):
            digit = cb.dtypes.int64(cb.simt.thread_idx()[0])
            tiles = (length + self.tile_size - 1) // self.tile_size
            row = cb.dtypes.int64(block_id)
            while row < rows:
                total = cb.dtypes.int64(0)
                for tile in range(tiles):
                    total = total + cb.simt.l2cache_load(
                        hist.ptr((row * tiles + tile) * RADIX_BINS + digit)
                    )
                lane = cb.dtypes.int32(digit % WARP_LANES)
                warp = cb.dtypes.int32(digit // WARP_LANES)
                prefix = total
                for step in cb.range_constexpr(WARP_SCAN_STEPS):
                    previous = cb.simt.shfl_up(prefix, 1 << step)
                    if lane >= 1 << step:
                        prefix = prefix + previous
                if lane == WARP_LANES - 1:
                    self.totals[warp] = prefix
                cb.simt.syncthreads()
                # 先加所有较小桶的元素数，再按 tile 顺序累加本桶计数，保证跨块稳定。
                before = prefix - total
                for bucket in range(warp):
                    before = before + self.totals[bucket]
                for tile in range(tiles):
                    address = (row * tiles + tile) * RADIX_BINS + digit
                    cb.simt.l2cache_store(offsets.ptr(address), before)
                    before = before + cb.simt.l2cache_load(hist.ptr(address))
                cb.simt.syncthreads()
                row = row + blocks
            cb.simt.threadfence()

    @cb.jit
    def scatter(
        self,
        x: cb.Tensor,
        values: cb.Tensor,
        indices: cb.Tensor,
        keys: cb.Tensor,
        source_indices: cb.Tensor,
        output_keys: cb.Tensor,
        output_indices: cb.Tensor,
        ranks: cb.Tensor,
        offsets: cb.Tensor,
        rows: cb.dtypes.int64,
        length: cb.dtypes.int64,
        inner: cb.dtypes.int64,
        blocks: cb.dtypes.int64,
        byte_index: cb.Constexpr[int],
        final_pass: cb.Constexpr[bool],
    ):
        tiles = (length + self.tile_size - 1) // self.tile_size
        for task in range(cb.get_block_idx(), rows * tiles, blocks):
            row = task // tiles
            row_base = row * length
            output_base = row // inner * length * inner + row % inner
            start = task % tiles * self.tile_size
            with cb.vf(mode="simt", thread=RADIX_SCATTER_THREADS):
                thread = cb.dtypes.int64(cb.simt.thread_idx()[0])
                for position in range(thread, self.tile_size, RADIX_SCATTER_THREADS):
                    if start + position < length:
                        encoded = cb.simt.l2cache_load(
                            ranks.ptr(row_base + start + position)
                        )
                        original = cb.dtypes.int64(
                            encoded // cb.dtypes.uint32(self.tile_size)
                        )
                        within = cb.dtypes.int64(
                            encoded % cb.dtypes.uint32(self.tile_size)
                        )
                        key = cb.simt.l2cache_load(
                            keys.ptr(row_base + start + original)
                        )
                        digit = cb.dtypes.int64(
                            radix_digit(key, keys.dtype, byte_index)
                        )
                        destination = (
                            row_base
                            + cb.simt.l2cache_load(
                                offsets.ptr(task * RADIX_BINS + digit)
                            )
                            + within
                        )
                        source_axis = cb.simt.l2cache_load(
                            source_indices.ptr(row_base + start + original)
                        )
                        # 排序只移动保序键和原轴索引；末轮从 x 收集原值，保留零符号和 NaN 位模式。
                        if final_pass:
                            output_position = (
                                output_base + (destination - row_base) * inner
                            )
                            cb.simt.l2cache_store(
                                values.ptr(output_position),
                                cb.simt.l2cache_load(
                                    x.ptr(output_base + source_axis * inner)
                                ),
                            )
                            cb.simt.l2cache_store(
                                indices.ptr(output_position), source_axis
                            )
                        else:
                            cb.simt.l2cache_store(output_keys.ptr(destination), key)
                            cb.simt.l2cache_store(
                                output_indices.ptr(destination), source_axis
                            )
                cb.simt.threadfence()
