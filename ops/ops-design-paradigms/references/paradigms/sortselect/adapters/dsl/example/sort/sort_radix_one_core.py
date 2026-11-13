# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""单核局部 Radix：在 UB 中排序单行键与原索引，再恢复原值。"""

import cannbotdsl as cb

from radix import (
    load_keys32,
    LOCAL_RADIX_THREADS,
    radix_permutation,
    sort_local_keys,
)
from sort_common import RADIX_BINS, dtype_bytes, key_dtype, ordered_key, radix_digit


@cb.kernel
class RadixOneCore:
    """每核逐行排序；一次仅一行驻留 UB，输出值按原轴索引从 GM 收集。"""

    def __init__(
        self,
        dtype,
        row_stride,
        batch_rows,
        descending,
    ):
        if batch_rows != 1:
            raise ValueError("single-core Radix requires groups == 1")
        self.dtype = dtype
        self.row_stride = row_stride
        self.descending = descending
        self.key_type = key_dtype(dtype)
        self.passes = dtype_bytes(dtype)
        # UB 存放可排序键及原轴索引；原始 values 从 GM 按最终索引收集，不由键反算。
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
        for row in range(block_id, rows, blocks):
            total = cb.dtypes.int32(length)
            row_base = row // inner * length * inner + row % inner
            # 装载一行；inner 表示原布局中相邻轴元素的跨度。
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
                with cb.vf(mode="simt", thread=LOCAL_RADIX_THREADS):
                    tid = cb.dtypes.int32(cb.simt.thread_idx()[0])
                    for axis in range(tid, total, LOCAL_RADIX_THREADS):
                        source = row_base + cb.dtypes.int64(axis) * inner
                        self.ping[axis] = ordered_key(
                            x[source], self.dtype, self.descending
                        )
                        self.ping_ids[axis] = cb.dtypes.uint32(axis)
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
            )
            # 索引始终相对本行；恢复原值可保留零符号和 NaN 的位模式。
            with cb.vf(mode="simt", thread=LOCAL_RADIX_THREADS):
                tid = cb.dtypes.int32(cb.simt.thread_idx()[0])
                for axis in range(tid, total, LOCAL_RADIX_THREADS):
                    original = cb.dtypes.int64(0)
                    if self.passes % 2 == 0:
                        original = cb.dtypes.int64(self.ping_ids[axis])
                    else:
                        original = cb.dtypes.int64(self.pong_ids[axis])
                    destination = row_base + cb.dtypes.int64(axis) * inner
                    source = row_base + original * inner
                    values[destination] = x[source]
                    indices[destination] = original


@cb.kernel
class NarrowRadixOneCore:
    """一/二字节稳定排序：复合排列保留原位置，避免键/索引双缓冲。"""

    def __init__(self, dtype, row_stride, batch_rows, descending):
        if batch_rows != 1 or dtype_bytes(dtype) not in (1, 2):
            raise ValueError("narrow local radix requires one row and 1/2-byte input")
        self.dtype = dtype
        self.row_stride = row_stride
        self.descending = descending
        self.passes = dtype_bytes(dtype)
        self.digits = cb.Buffer(
            cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((row_stride,))
        )
        self.first_order = cb.Buffer(
            cb.MemLoc.UB, cb.dtypes.uint32, cb.make_layout((row_stride,))
        )
        self.byte_order = cb.Buffer(
            cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((row_stride,))
        )
        self.offsets = cb.Buffer(
            cb.MemLoc.UB, cb.dtypes.uint16, cb.make_layout((RADIX_BINS,))
        )
        if self.passes == 2:
            self.original_values = cb.Buffer(
                cb.MemLoc.UB, dtype, cb.make_layout((row_stride,))
            )
            self.second_order = cb.Buffer(
                cb.MemLoc.UB, cb.dtypes.uint32, cb.make_layout((row_stride,))
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
        for row in range(cb.get_block_idx(), rows, blocks):
            total = cb.dtypes.int32(length)
            row_base = row // inner * length * inner + row % inner
            if self.passes == 2:
                if inner == 1:
                    cb.mem_copy(
                        cb.strided_slice(self.original_values, (0,), (length,)),
                        cb.strided_slice(x, (row_base,), (length,)),
                    )
            with cb.vf(mode="simt", thread=LOCAL_RADIX_THREADS):
                tid = cb.dtypes.int32(cb.simt.thread_idx()[0])
                for axis in range(tid, self.row_stride, LOCAL_RADIX_THREADS):
                    digit = cb.dtypes.uint8(RADIX_BINS - 1)
                    if axis < total:
                        raw = self.dtype(0)
                        if self.passes == 2:
                            if inner == 1:
                                raw = self.original_values[axis]
                            else:
                                raw = x[row_base + cb.dtypes.int64(axis) * inner]
                                self.original_values[axis] = raw
                        else:
                            raw = x[row_base + cb.dtypes.int64(axis) * inner]
                        key = ordered_key(raw, self.dtype, self.descending)
                        digit = cb.dtypes.uint8(
                            radix_digit(key, key_dtype(self.dtype), 0)
                        )
                    self.digits[axis] = digit
            radix_permutation(
                self.digits,
                self.first_order,
                self.byte_order,
                self.offsets,
                total,
            )
            if self.passes == 2:
                # 第一遍排列是原位置，第二遍排列是第一遍的位置；最终组合两者。
                with cb.vf(mode="simt", thread=LOCAL_RADIX_THREADS):
                    tid = cb.dtypes.int32(cb.simt.thread_idx()[0])
                    for axis in range(tid, self.row_stride, LOCAL_RADIX_THREADS):
                        digit = cb.dtypes.uint8(RADIX_BINS - 1)
                        if axis < total:
                            original = cb.dtypes.int32(self.first_order[axis])
                            key = ordered_key(
                                self.original_values[original],
                                self.dtype,
                                self.descending,
                            )
                            digit = cb.dtypes.uint8(
                                radix_digit(key, key_dtype(self.dtype), 1)
                            )
                        self.digits[axis] = digit
                radix_permutation(
                    self.digits,
                    self.second_order,
                    self.byte_order,
                    self.offsets,
                    total,
                )
            with cb.vf(mode="simt", thread=LOCAL_RADIX_THREADS):
                tid = cb.dtypes.int32(cb.simt.thread_idx()[0])
                for axis in range(tid, total, LOCAL_RADIX_THREADS):
                    original = cb.dtypes.int64(0)
                    if self.passes == 1:
                        original = cb.dtypes.int64(self.first_order[axis])
                    else:
                        original = cb.dtypes.int64(
                            self.first_order[cb.dtypes.int32(self.second_order[axis])]
                        )
                    destination = row_base + cb.dtypes.int64(axis) * inner
                    if self.passes == 1:
                        raw = cb.dtypes.uint8(self.digits[original])
                        if self.descending:
                            raw = cb.dtypes.uint8(255) - raw
                        if self.dtype == cb.dtypes.int8:
                            raw = raw ^ cb.dtypes.uint8(128)
                        values[destination] = cb.simt.bitcast(raw, dtype=self.dtype)
                    else:
                        values[destination] = self.original_values[original]
                    indices[destination] = original
