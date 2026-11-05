# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""短轴插入排序：批量加载独立行，在每行内保持等值元素的输入次序。"""

import cannbotdsl as cb

from sort_common import INSERTION_THREADS, key_dtype, ordered_key


@cb.kernel
class InsertionSort:
    """线程独立处理一行的稳定插入排序，适用于极短轴。"""

    def __init__(self, dtype, row_stride, batch_rows, descending):
        self.dtype = dtype
        self.row_stride = row_stride
        self.batch_rows = batch_rows
        self.descending = descending
        self.values = cb.Buffer(cb.MemLoc.UB, (batch_rows * row_stride,), dtype)
        self.keys = cb.Buffer(
            cb.MemLoc.UB, (batch_rows * row_stride,), key_dtype(dtype)
        )
        self.ids = cb.Buffer(cb.MemLoc.UB, (batch_rows * row_stride,), cb.dtypes.uint32)

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
        with cb.vf(mode="simt", thread=INSERTION_THREADS):
            thread = cb.dtypes.int64(cb.simt.thread_idx()[0])
            batch = cb.dtypes.int64(block_id)
            while batch * self.batch_rows < rows:
                flat = thread
                while flat < self.batch_rows * length:
                    segment = flat // length
                    axis = flat % length
                    # 非末轴时优先沿连续的内维分配线程，UB 仍保持逐行布局。
                    if inner != 1:
                        segment = flat % self.batch_rows
                        axis = flat // self.batch_rows
                    row = batch * self.batch_rows + segment
                    if row < rows:
                        source = (row // inner * length + axis) * inner + row % inner
                        offset = segment * self.row_stride + axis
                        self.values[offset] = x[source]
                        self.keys[offset] = ordered_key(
                            x[source], self.dtype, self.descending
                        )
                        self.ids[offset] = cb.dtypes.uint32(axis)
                    flat = flat + INSERTION_THREADS
                # 所有线程装载完成后，行负责人才能读取整行。
                cb.simt.syncthreads()
                row = batch * self.batch_rows + thread
                if thread < self.batch_rows and row < rows:
                    base = thread * self.row_stride
                    element = cb.dtypes.int64(1)
                    while element < length:
                        value = self.values[base + element]
                        key = self.keys[base + element]
                        index = self.ids[base + element]
                        position = cb.dtypes.int64(element)
                        moving = cb.dtypes.int32(1)
                        while position > 0 and moving != 0:
                            # 仅移动严格更大的键，等值元素不会越过先到的元素，保证稳定性。
                            if self.keys[base + position - 1] > key:
                                self.keys[base + position] = self.keys[
                                    base + position - 1
                                ]
                                self.values[base + position] = self.values[
                                    base + position - 1
                                ]
                                self.ids[base + position] = self.ids[
                                    base + position - 1
                                ]
                                position = position - 1
                            else:
                                moving = cb.dtypes.int32(0)
                        self.keys[base + position] = key
                        self.values[base + position] = value
                        self.ids[base + position] = index
                        element = element + 1
                # 行排序完成后，协作写回线程才能读取结果。
                cb.simt.syncthreads()
                flat = thread
                while flat < self.batch_rows * length:
                    segment = flat // length
                    axis = flat % length
                    if inner != 1:
                        segment = flat % self.batch_rows
                        axis = flat // self.batch_rows
                    row = batch * self.batch_rows + segment
                    if row < rows:
                        destination = (
                            row // inner * length + axis
                        ) * inner + row % inner
                        offset = segment * self.row_stride + axis
                        values[destination] = self.values[offset]
                        indices[destination] = cb.dtypes.int64(self.ids[offset])
                    flat = flat + INSERTION_THREADS
                # 写回读者结束后，下一批才能覆盖共享 UB。
                cb.simt.syncthreads()
                batch = batch + blocks
