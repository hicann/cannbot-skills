# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""单核局部 Radix：在 UB 中排序单行键与原索引，再恢复原值。"""

import cannbotdsl as cb

from radix import LOCAL_RADIX_THREADS, VECTOR_BYTES, sort_local_keys
from sort_common import RADIX_BINS, dtype_bytes, key_dtype, ordered_key


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
        for row in range(block_id, rows, blocks):
            total = cb.dtypes.int32(length)
            row_base = row // inner * length * inner + row % inner
            # 装载一行；inner 表示原布局中相邻轴元素的跨度。
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
                self.scratch,
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
