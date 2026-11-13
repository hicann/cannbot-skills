# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""将连续 [outer,height,width] 转为 [outer,width,height]，优先使用 NDDMA。"""

import cannbotdsl as cb

TRANSPOSE_TILE = 32
TRANSPOSE_LARGE_TILE = 64
TRANSPOSE_THREADS = 1024
TRANSPOSE_MAX_ELEMENTS = 128 * 65


def transpose_buffer_elements(tile_size):
    return max(tile_size * (tile_size + 1), TRANSPOSE_MAX_ELEMENTS)


def select_transpose_tile(height, width):
    return TRANSPOSE_LARGE_TILE if min(height, width) >= 128 else TRANSPOSE_TILE


@cb.kernel
class SortTranspose:
    # words=2 时 GM/UB 的物理类型为 uint32，末维两个字始终一起移动。
    def __init__(self, dtype, tile_size, words):
        self.words = words
        self.tile_size = tile_size
        self.tile = cb.Buffer(
            cb.MemLoc.UB,
            dtype,
            cb.make_layout((transpose_buffer_elements(tile_size) * words,)),
        )

    @cb.jit
    def tiled(
        self,
        x,
        output,
        outer,
        height,
        width,
        blocks,
        tile_rows: cb.Constexpr[int],
        tile_cols: cb.Constexpr[int],
    ):
        # permute 只建立视图；显式 NDDMA 才按该布局从 GM 实际搬入 UB。
        source_view = x.view(outer, height, width, self.words).permute(0, 2, 1, 3)
        destination_view = output.view(outer, width, height, self.words)
        row_tiles = (height + tile_rows - 1) // tile_rows
        col_tiles = (width + tile_cols - 1) // tile_cols
        for task in range(
            cb.dtypes.int64(cb.get_block_idx()), outer * row_tiles * col_tiles, blocks
        ):
            batch = task // (row_tiles * col_tiles)
            row_index = task // col_tiles % row_tiles
            col_index = task % col_tiles
            valid_rows = height - row_index * tile_rows
            if valid_rows > tile_rows:
                valid_rows = cb.dtypes.int64(tile_rows)
            valid_cols = width - col_index * tile_cols
            if valid_cols > tile_cols:
                valid_cols = cb.dtypes.int64(tile_cols)
            tile_view = cb.reinterpret(
                self.tile,
                x.dtype,
                cb.make_layout(
                    (1, valid_cols, valid_rows, self.words),
                    stride=(
                        tile_rows * tile_cols * self.words,
                        tile_rows * self.words,
                        self.words,
                        1,
                    ),
                ),
            )
            source = cb.tile_slice(
                source_view,
                (1, tile_cols, tile_rows, self.words),
                (batch, col_index, row_index, 0),
            )
            destination = cb.tile_slice(
                destination_view,
                (1, tile_cols, tile_rows, self.words),
                (batch, col_index, row_index, 0),
            )
            cb.mem_copy(tile_view, source, cb.copy_options(engine="nddma"))
            cb.mem_copy(destination, tile_view)

    @cb.jit
    def packed_output(
        self, x, output, outer, height, width, blocks, tile_cols: cb.Constexpr[int]
    ):
        source_view = x.view(outer, height, width, self.words).permute(0, 2, 1, 3)
        col_tiles = (width + tile_cols - 1) // tile_cols
        for task in range(
            cb.dtypes.int64(cb.get_block_idx()), outer * col_tiles, blocks
        ):
            batch = task // col_tiles
            col_index = task % col_tiles
            valid_cols = width - col_index * tile_cols
            if valid_cols > tile_cols:
                valid_cols = cb.dtypes.int64(tile_cols)
            # 完整覆盖 height，输出 tile 可在 UB 紧凑排列并连续写回。
            # 若仍保留固定行距，短行会被拆成大量小 DMA burst。
            count = valid_cols * height * self.words
            tile_view = cb.reinterpret(
                self.tile, x.dtype, cb.make_layout((1, valid_cols, height, self.words))
            )
            source = cb.tile_slice(
                source_view, (1, tile_cols, 64, self.words), (batch, col_index, 0, 0)
            )
            cb.mem_copy(tile_view, source, cb.copy_options(engine="nddma"))
            flat_tile = cb.reinterpret(self.tile, x.dtype, cb.make_layout((count,)))
            destination = cb.tile_slice(
                output.view(outer, width * height * self.words),
                cb.make_tiler((1, tile_cols * height * self.words), alignment=(1, 1)),
                (batch, col_index),
            )
            cb.mem_copy(destination, flat_tile.view(1, count))

    @cb.jit
    def simt_small(
        self,
        x: cb.Tensor,
        output: cb.Tensor,
        outer: cb.dtypes.int64,
        height: cb.dtypes.int64,
        width: cb.dtypes.int64,
        blocks: cb.dtypes.int64,
        tile_rows: cb.Constexpr[int],
        tile_cols: cb.Constexpr[int],
    ):
        # 成对 uint32 重新解释为 uint64；仅改变视图，不做数值转换。
        x = x.view(cb.dtypes.uint64)
        output = output.view(cb.dtypes.uint64)
        tile = cb.reinterpret(
            self.tile,
            cb.dtypes.uint64,
            cb.make_layout((transpose_buffer_elements(self.tile_size),)),
        )
        row_tiles = (height + tile_rows - 1) // tile_rows
        col_tiles = (width + tile_cols - 1) // tile_cols
        block = cb.dtypes.int64(cb.get_block_idx())
        # 每块公共的 64 位除法和 GM 基址由 Scalar 计算；VF 仅计算 int32 块内坐标。
        for task in range(block, outer * row_tiles * col_tiles, blocks):
            batch = task // (row_tiles * col_tiles)
            row_start = task // col_tiles % row_tiles * tile_rows
            col_start = task % col_tiles * tile_cols
            valid_rows = cb.dtypes.int32(tile_rows)
            valid_cols = cb.dtypes.int32(tile_cols)
            if height - row_start < tile_rows:
                valid_rows = cb.dtypes.int32(height - row_start)
            if width - col_start < tile_cols:
                valid_cols = cb.dtypes.int32(width - col_start)
            source_base = (batch * height + row_start) * width + col_start
            destination_base = (batch * width + col_start) * height + row_start
            with cb.vf(mode="simt", thread=TRANSPOSE_THREADS):
                thread = cb.dtypes.int32(cb.simt.thread_idx()[0])
                for local in range(thread, tile_rows * tile_cols, TRANSPOSE_THREADS):
                    row = local // tile_cols
                    col = local % tile_cols
                    if row < valid_rows and col < valid_cols:
                        tile[row * (tile_cols + 1) + col] = x[
                            source_base + cb.dtypes.int64(row) * width + col
                        ]
                cb.simt.syncthreads()
                for local in range(thread, tile_rows * tile_cols, TRANSPOSE_THREADS):
                    col = local // tile_rows
                    row = local % tile_rows
                    if row < valid_rows and col < valid_cols:
                        output[
                            destination_base + cb.dtypes.int64(col) * height + row
                        ] = tile[row * (tile_cols + 1) + col]
                # 下一块复用 UB 前，所有线程必须完成读取。
                cb.simt.syncthreads()

    def __call__(
        self,
        x: cb.Tensor,
        output: cb.Tensor,
        outer: cb.dtypes.int64,
        height: cb.dtypes.int64,
        width: cb.dtypes.int64,
        blocks: cb.dtypes.int64,
    ):
        if self.words == 2 and height <= 32 and outer * height * width <= 131072:
            # 小容量 64 位短行的 NDDMA 成对搬运在实测中仍有退化，保留 SIMT。
            if width >= 4096:
                if height <= 8:
                    self.simt_small(x, output, outer, height, width, blocks, 8, 128)
                elif height <= 16:
                    self.simt_small(x, output, outer, height, width, blocks, 16, 128)
                else:
                    self.simt_small(x, output, outer, height, width, blocks, 64, 128)
            else:
                self.simt_small(x, output, outer, height, width, blocks, 32, 32)
        elif height <= 64:
            if height <= 16 and width >= 65536:
                self.packed_output(x, output, outer, height, width, blocks, 512)
            elif width >= 2048:
                self.packed_output(x, output, outer, height, width, blocks, 128)
            else:
                self.packed_output(x, output, outer, height, width, blocks, 32)
        elif height >= 4096 and width <= 64:
            if width <= 16 and height >= 65536:
                self.tiled(x, output, outer, height, width, blocks, 512, 16)
            else:
                self.tiled(x, output, outer, height, width, blocks, 128, 64)
        else:
            self.tiled(
                x, output, outer, height, width, blocks, self.tile_size, self.tile_size
            )


@cb.host
def transpose_entry(
    x: cb.Tensor,
    output: cb.Tensor,
    outer: cb.dtypes.int64,
    height: cb.dtypes.int64,
    width: cb.dtypes.int64,
    blocks: cb.dtypes.int64,
    tile_size: cb.Constexpr[int],
    words: cb.Constexpr[int] = 1,
):
    SortTranspose(x.dtype, tile_size, words)[blocks](
        x, output, outer, height, width, blocks
    )
