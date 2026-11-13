# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""合批 Radix 与行内恢复：逆排名和稳定行分组两条分支共用第一阶段。"""

import cannbotdsl as cb

from radix import load_keys32, sort_local_keys
from sort_common import RADIX_BINS, dtype_bytes, key_dtype, ordered_key

# 构键和恢复的线程数独立于局部 Radix 内部的 SIMD 排列。
TWO_STAGE_THREADS = 1024


@cb.kernel
class SmallAxisTwoStage:
    """整批展平排序后，以逆排名或稳定行号排序恢复各行及原轴索引。"""

    def __init__(
        self,
        dtype,
        row_stride,
        batch_rows,
        descending,
        rank_inverse=True,
    ):
        self.batch_rows = batch_rows
        self.rank_inverse = rank_inverse
        # 公开入口 B≤256；直接调用更宽批次时仍保留两字节行号。
        self.stage2_passes = 1 if batch_rows <= 256 else 2
        self.dtype = dtype
        self.row_stride = row_stride
        self.descending = descending
        self.key_type = key_dtype(dtype)
        self.passes = dtype_bytes(dtype)
        self.ping = cb.Buffer(
            cb.MemLoc.UB, self.key_type, cb.make_layout((row_stride,))
        )
        self.pong = cb.Buffer(
            cb.MemLoc.UB, self.key_type, cb.make_layout((row_stride,))
        )
        self.ping_ids = cb.Buffer(
            cb.MemLoc.UB, cb.dtypes.uint32, cb.make_layout((row_stride,))
        )
        self.pong_ids = cb.Buffer(
            cb.MemLoc.UB, cb.dtypes.uint32, cb.make_layout((row_stride,))
        )
        self.digits = cb.Buffer(
            cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((row_stride,))
        )
        self.permutation = cb.Buffer(
            cb.MemLoc.UB, cb.dtypes.uint32, cb.make_layout((row_stride,))
        )
        self.byte_order = cb.Buffer(
            cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((row_stride,))
        )
        self.offsets = cb.Buffer(
            cb.MemLoc.UB, cb.dtypes.uint16, cb.make_layout((RADIX_BINS,))
        )

    def __call__(
        self,
        x: cb.Tensor,
        values: cb.Tensor,
        indices: cb.Tensor,
        rows: cb.dtypes.int64,
        length: cb.dtypes.int64,
        inner: cb.dtypes.int64,
        blocks: cb.dtypes.int64,
    ):
        block_id = cb.get_block_idx()
        for batch in range(
            block_id, (rows + self.batch_rows - 1) // self.batch_rows, blocks
        ):
            row_start = batch * self.batch_rows
            valid_rows = rows - row_start
            if valid_rows > self.batch_rows:
                valid_rows = cb.dtypes.int64(self.batch_rows)
            total = cb.dtypes.int32(length * valid_rows)
            local_length = cb.dtypes.int32(length)
            row_base = row_start // inner * length * inner + row_start % inner
            # 按原布局收集本批行，UB 内紧凑展平，供第一阶段跨行排序。
            if (
                self.dtype in (cb.dtypes.float32, cb.dtypes.int32, cb.dtypes.uint32)
                and inner == 1
            ):
                load_keys32(
                    x,
                    self.ping,
                    self.ping_ids,
                    row_base,
                    total,
                    self.row_stride,
                    self.dtype,
                    self.descending,
                )
            else:
                with cb.vf(mode="simt", thread=TWO_STAGE_THREADS):
                    tid = cb.dtypes.int32(cb.simt.thread_idx()[0])
                    for flat in range(tid, total, TWO_STAGE_THREADS):
                        row = row_start + flat // local_length
                        axis = flat % local_length
                        source = cb.dtypes.int64(0)
                        if inner == 1:
                            source = row_base + cb.dtypes.int64(flat)
                        elif self.batch_rows == 1:
                            source = row_base + cb.dtypes.int64(flat) * inner
                        else:
                            source = (
                                row // inner * length + axis
                            ) * inner + row % inner
                        self.ping[flat] = ordered_key(
                            x[source], self.dtype, self.descending
                        )
                        self.ping_ids[flat] = cb.dtypes.uint32(flat)
            sort_local_keys(
                self.ping,
                self.pong,
                self.ping_ids,
                self.pong_ids,
                self.digits,
                self.permutation,
                self.byte_order,
                self.offsets,
                total,
                self.row_stride,
                self.key_type,
                self.passes,
                True,
            )
            # 第一轮得到全批唯一排名；恢复行内顺序时比较排名，不再次比较数值。
            if self.rank_inverse:
                with cb.vf(mode="simt", thread=TWO_STAGE_THREADS):
                    tid = cb.dtypes.int32(cb.simt.thread_idx()[0])
                    for rank in range(tid, total, TWO_STAGE_THREADS):
                        original = cb.dtypes.uint32(0)
                        if self.passes % 2 == 0:
                            original = self.ping_ids[rank]
                        else:
                            original = self.pong_ids[rank]
                        self.permutation[original] = cb.dtypes.uint32(rank)
                    cb.simt.syncthreads()
                    for flat in range(tid, total, TWO_STAGE_THREADS):
                        row = row_start + flat // local_length
                        axis = flat % local_length
                        rank = self.permutation[flat]
                        segment_base = flat - axis
                        # 八条独立累加链缩短行内计秩的循环依赖；尾部单独处理。
                        pos0 = cb.dtypes.int32(0)
                        pos1 = cb.dtypes.int32(0)
                        pos2 = cb.dtypes.int32(0)
                        pos3 = cb.dtypes.int32(0)
                        pos4 = cb.dtypes.int32(0)
                        pos5 = cb.dtypes.int32(0)
                        pos6 = cb.dtypes.int32(0)
                        pos7 = cb.dtypes.int32(0)
                        j = cb.dtypes.int32(0)
                        while j + 7 < local_length:
                            if self.permutation[segment_base + j + 0] < rank:
                                pos0 = pos0 + 1
                            if self.permutation[segment_base + j + 1] < rank:
                                pos1 = pos1 + 1
                            if self.permutation[segment_base + j + 2] < rank:
                                pos2 = pos2 + 1
                            if self.permutation[segment_base + j + 3] < rank:
                                pos3 = pos3 + 1
                            if self.permutation[segment_base + j + 4] < rank:
                                pos4 = pos4 + 1
                            if self.permutation[segment_base + j + 5] < rank:
                                pos5 = pos5 + 1
                            if self.permutation[segment_base + j + 6] < rank:
                                pos6 = pos6 + 1
                            if self.permutation[segment_base + j + 7] < rank:
                                pos7 = pos7 + 1
                            j = j + 8
                        position = pos0 + pos1 + pos2 + pos3 + pos4 + pos5 + pos6 + pos7
                        for tail in range(j, local_length):
                            if self.permutation[segment_base + tail] < rank:
                                position = position + 1
                        destination = cb.dtypes.int64(0)
                        source = cb.dtypes.int64(0)
                        if inner == 1:
                            destination = row_base + cb.dtypes.int64(
                                flat // local_length * local_length + position
                            )
                            source = row_base + cb.dtypes.int64(flat)
                        else:
                            destination = (
                                row // inner * length + position
                            ) * inner + row % inner
                            source = (
                                row // inner * length + axis
                            ) * inner + row % inner
                        values[destination] = x[source]
                        indices[destination] = cb.dtypes.int64(axis)
            else:
                with cb.vf(mode="simt", thread=TWO_STAGE_THREADS):
                    tid = cb.dtypes.int32(cb.simt.thread_idx()[0])
                    for rank in range(tid, total, TWO_STAGE_THREADS):
                        original = cb.dtypes.uint32(0)
                        if self.passes % 2 == 0:
                            original = self.ping_ids[rank]
                        else:
                            original = self.pong_ids[rank]
                        # Stage1 已按数值稳定排序；仅按行号稳定分组即可保留行内次序。
                        # 携带原展平索引，供分组后恢复原值。
                        self.ping[rank] = self.key_type(
                            cb.dtypes.int32(original) // local_length
                        )
                        self.ping_ids[rank] = original
                sort_local_keys(
                    self.ping,
                    self.pong,
                    self.ping_ids,
                    self.pong_ids,
                    self.digits,
                    self.permutation,
                    self.byte_order,
                    self.offsets,
                    total,
                    self.row_stride,
                    self.key_type,
                    self.stage2_passes,
                    True,
                )
                with cb.vf(mode="simt", thread=TWO_STAGE_THREADS):
                    tid = cb.dtypes.int32(cb.simt.thread_idx()[0])
                    for flat in range(tid, total, TWO_STAGE_THREADS):
                        original = cb.dtypes.int64(0)
                        if self.stage2_passes % 2 == 0:
                            original = cb.dtypes.int64(self.ping_ids[flat])
                        else:
                            original = cb.dtypes.int64(self.pong_ids[flat])
                        row = row_start + flat // local_length
                        axis = original % local_length
                        destination = cb.dtypes.int64(0)
                        source = cb.dtypes.int64(0)
                        if inner == 1:
                            destination = row_base + cb.dtypes.int64(flat)
                            source = row_base + original
                        else:
                            destination = (
                                row // inner * length + flat % local_length
                            ) * inner + row % inner
                            source = (
                                row // inner * length + axis
                            ) * inner + row % inner
                        values[destination] = x[source]
                        indices[destination] = cb.dtypes.int64(axis)
