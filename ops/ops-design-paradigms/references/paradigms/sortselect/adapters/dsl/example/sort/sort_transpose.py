# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""布局转换：将连续批次 [outer,height,width] 转成 [outer,width,height]。"""

import cannbotdsl as cb

TRANSPOSE_TILE = 32
TRANSPOSE_THREADS = 256


@cb.kernel
class SortTranspose:
    """按二维 tile 搬入 UB 后交换行列；尾 tile 仅访问有效坐标。"""

    def __init__(self, dtype):
        # UB 行距为 T+1，使列访问的行起点错开，减轻潜在的 bank 冲突。
        # 额外一列仅用于 padding，不存输入、不写回 GM；GM 仍按紧凑行距寻址。
        self.tile = cb.Buffer(
            cb.MemLoc.UB, (TRANSPOSE_TILE * (TRANSPOSE_TILE + 1),), dtype
        )

    def __call__(
        self,
        x: cb.Tensor,
        output: cb.Tensor,
        outer: cb.dtypes.int64,
        height: cb.dtypes.int64,
        width: cb.dtypes.int64,
        blocks: cb.dtypes.int64,
    ):
        row_tiles = (height + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE
        col_tiles = (width + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE
        block = cb.dtypes.int64(cb.get_block_idx())
        with cb.vf(mode="simt", thread=TRANSPOSE_THREADS):
            thread = cb.dtypes.int64(cb.simt.thread_idx()[0])
            task = block
            while task < outer * row_tiles * col_tiles:
                batch = task // (row_tiles * col_tiles)
                row_start = task // col_tiles % row_tiles * TRANSPOSE_TILE
                col_start = task % col_tiles * TRANSPOSE_TILE
                for local in range(
                    thread, TRANSPOSE_TILE * TRANSPOSE_TILE, TRANSPOSE_THREADS
                ):
                    row = local // TRANSPOSE_TILE
                    col = local % TRANSPOSE_TILE
                    if row_start + row < height and col_start + col < width:
                        self.tile[row * (TRANSPOSE_TILE + 1) + col] = x[
                            (batch * height + row_start + row) * width + col_start + col
                        ]
                cb.simt.syncthreads()
                for local in range(
                    thread, TRANSPOSE_TILE * TRANSPOSE_TILE, TRANSPOSE_THREADS
                ):
                    col = local // TRANSPOSE_TILE
                    row = local % TRANSPOSE_TILE
                    if row_start + row < height and col_start + col < width:
                        output[
                            (batch * width + col_start + col) * height + row_start + row
                        ] = self.tile[row * (TRANSPOSE_TILE + 1) + col]
                # 下一块覆盖 UB 前，所有线程必须完成读取。
                cb.simt.syncthreads()
                task = task + blocks


@cb.host
def transpose_entry(
    x: cb.Tensor,
    output: cb.Tensor,
    outer: cb.dtypes.int64,
    height: cb.dtypes.int64,
    width: cb.dtypes.int64,
    blocks: cb.dtypes.int64,
):
    SortTranspose(x.dtype)[blocks](x, output, outer, height, width, blocks)
