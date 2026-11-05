# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""合批 Radix 与行内恢复：逆排名和紧凑键两条分支共用第一阶段。"""

import cannbotdsl as cb

from radix import LOCAL_RADIX_THREADS, VECTOR_BYTES, sort_local_keys
from sort_common import RADIX_BINS, dtype_bytes, key_dtype, ordered_key


@cb.kernel
class SmallAxisTwoStage:
    """整批展平排序后，以逆排名或紧凑键排序恢复各行及原轴索引。"""

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
        self.dtype = dtype
        self.row_stride = row_stride
        self.descending = descending
        self.key_type = key_dtype(dtype)
        self.passes = dtype_bytes(dtype)
        self.ping = cb.Buffer(cb.MemLoc.UB, (row_stride,), self.key_type)
        self.pong = cb.Buffer(cb.MemLoc.UB, (row_stride,), self.key_type)
        self.ping_ids = cb.Buffer(cb.MemLoc.UB, (row_stride,), cb.dtypes.uint32)
        self.pong_ids = cb.Buffer(cb.MemLoc.UB, (row_stride,), cb.dtypes.uint32)
        self.digits = cb.Buffer(cb.MemLoc.UB, (row_stride,), cb.dtypes.uint8)
        self.permutation = cb.Buffer(cb.MemLoc.UB, (row_stride,), cb.dtypes.uint32)
        self.scratch = cb.Buffer(
            cb.MemLoc.UB, (row_stride + VECTOR_BYTES,), cb.dtypes.uint8
        )
        self.byte_order = cb.Buffer(cb.MemLoc.UB, (row_stride,), cb.dtypes.uint8)
        self.stage1_ids = cb.Buffer(cb.MemLoc.UB, (row_stride,), cb.dtypes.uint32)
        self.offsets = cb.Buffer(cb.MemLoc.UB, (RADIX_BINS,), cb.dtypes.uint16)

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
            with cb.vf(mode="simt", thread=LOCAL_RADIX_THREADS):
                tid = cb.dtypes.int32(cb.simt.thread_idx()[0])
                for flat in range(tid, total, LOCAL_RADIX_THREADS):
                    row = row_start + flat // local_length
                    axis = flat % local_length
                    source = cb.dtypes.int64(0)
                    if inner == 1:
                        source = row_base + cb.dtypes.int64(flat)
                    elif self.batch_rows == 1:
                        source = row_base + cb.dtypes.int64(flat) * inner
                    else:
                        source = (row // inner * length + axis) * inner + row % inner
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
                self.scratch,
                self.byte_order,
                self.offsets,
                total,
                self.row_stride,
                self.key_type,
                self.passes,
            )
            # 第一轮得到全批唯一排名；恢复行内顺序时比较排名，不再次比较数值。
            if self.rank_inverse:
                with cb.vf(mode="simt", thread=LOCAL_RADIX_THREADS):
                    tid = cb.dtypes.int32(cb.simt.thread_idx()[0])
                    for rank in range(tid, total, LOCAL_RADIX_THREADS):
                        original = cb.dtypes.uint32(0)
                        if self.passes % 2 == 0:
                            original = self.ping_ids[rank]
                        else:
                            original = self.pong_ids[rank]
                        self.permutation[original] = cb.dtypes.uint32(rank)
                    cb.simt.syncthreads()
                    for flat in range(tid, total, LOCAL_RADIX_THREADS):
                        row = row_start + flat // local_length
                        axis = flat % local_length
                        rank = self.permutation[flat]
                        position = cb.dtypes.int32(0)
                        segment_base = flat - axis
                        for j in range(local_length):
                            if self.permutation[segment_base + j] < rank:
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
                with cb.vf(mode="simt", thread=LOCAL_RADIX_THREADS):
                    tid = cb.dtypes.int32(cb.simt.thread_idx()[0])
                    for rank in range(tid, total, LOCAL_RADIX_THREADS):
                        original = cb.dtypes.uint32(0)
                        if self.passes % 2 == 0:
                            original = self.ping_ids[rank]
                        else:
                            original = self.pong_ids[rank]
                        self.stage1_ids[rank] = original
                        # 行号作为高位、全批排名作为低位：第二次排序先分行，再保持各行数值顺序。
                        # Host 保证 batch²×length−1 ≤ 65535，因此只需排序键的低两个字节。
                        self.ping[rank] = self.key_type(
                            cb.dtypes.int32(original) // local_length * total + rank
                        )
                        self.ping_ids[rank] = cb.dtypes.uint32(rank)
                sort_local_keys(
                    self.ping,
                    self.pong,
                    self.ping_ids,
                    self.pong_ids,
                    self.digits,
                    self.permutation,
                    self.scratch,
                    self.byte_order,
                    self.offsets,
                    total,
                    self.row_stride,
                    self.key_type,
                    2,
                )
                with cb.vf(mode="simt", thread=LOCAL_RADIX_THREADS):
                    tid = cb.dtypes.int32(cb.simt.thread_idx()[0])
                    for flat in range(tid, total, LOCAL_RADIX_THREADS):
                        original = cb.dtypes.int64(self.stage1_ids[self.ping_ids[flat]])
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
