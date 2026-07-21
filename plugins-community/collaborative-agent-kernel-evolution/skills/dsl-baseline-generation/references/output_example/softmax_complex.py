# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import tile.language as tl

# bf16 variant: input/output GM data is bfloat16
# UB computation uses float32 for precision (Cast bf16↔f32 handled during lowering)


@ascend_kernel
def softmax_kernel(input_ptr, output_ptr,
                   rows_per_core, tile_length, n_tiles, rows):

    pid = tl.program_id(0)
    row_start_idx = pid * rows_per_core
    row_end_idx = min(row_start_idx + rows_per_core, rows)

    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    row_tile_ub = tl.alloc_ub(tile_length, dtype=tl.float32)   # tile of input row; float32 compute even for bf16 IO
    exp_tile_ub = tl.alloc_ub(tile_length, dtype=tl.float32)   # tile of exp(x - max); float32 compute even for bf16 IO
    shared_ub = tl.alloc_ub(tile_length, dtype=tl.float32)   # reduction workspace; float32 compute even for bf16 IO
    out_ub = tl.alloc_ub(tile_length, dtype=tl.float32)   # reduction workspace; float32 compute even for bf16 IO

    # ------------------------------------------------------------
    # Per-row computation
    # ------------------------------------------------------------
    for row_idx in range(row_start_idx, row_end_idx):

        # ========================================================
        # PASS 1: compute global max of a long row (tiled)
        # ========================================================
        row_max = -1e30

        for tile_id in range(n_tiles):
            col_start = tile_id * tile_length
            offsets = row_idx * (tile_length * n_tiles) \
                + col_start + tl.arange(0, tile_length)

            # ---- Load tile ----
            with tl.copyin():
                tl.load(input_ptr + offsets, row_tile_ub)

            # ---- Compute tile max ----
            with tl.compute():
                tl.reduce_max(shared_ub, row_tile_ub, shared_ub)
                tile_max = tl.extract_scalar(shared_ub, 0)
            row_max = tl.max(row_max, tile_max)

        # ========================================================
        # PASS 2: compute global sum of exp(x - row_max)
        # ========================================================
        row_sum = 0.0

        for tile_id in range(n_tiles):
            col_start = tile_id * tile_length
            offsets = row_idx * (tile_length * n_tiles) \
                + col_start + tl.arange(0, tile_length)

            # ---- Load tile ----
            with tl.copyin():
                tl.load(input_ptr + offsets, row_tile_ub)

            # ---- Compute exp(x - max) for this tile ----
            with tl.compute():
                tl.vsub_scalar(exp_tile_ub, row_tile_ub, row_max)
                tl.vexp(exp_tile_ub, exp_tile_ub)
                tl.reduce_sum(shared_ub, exp_tile_ub, shared_ub)
                tile_sum = tl.extract_scalar(shared_ub, 0)
            row_sum = row_sum + tile_sum

        # ========================================================
        # PASS 3: normalize each tile and store output
        # ========================================================
        for tile_id in range(n_tiles):
            col_start = tile_id * tile_length
            offsets = row_idx * (tile_length * n_tiles) \
                + col_start + tl.arange(0, tile_length)

            # ---- Load tile ----
            with tl.copyin():
                tl.load(input_ptr + offsets, row_tile_ub)

            # ---- exp(x - max) for this tile ----
            with tl.compute():
                tl.vsub_scalar(exp_tile_ub, row_tile_ub, row_max)
                tl.vexp(exp_tile_ub, exp_tile_ub)

                # normalize output_tile = exp / row_sum
                tl.vdiv_scalar(out_ub, exp_tile_ub, row_sum)

            # ---- store tile ----
            with tl.copyout():
                tl.store(output_ptr + offsets, out_ub)


def softmax_host(x: torch.Tensor, output: torch.Tensor):
    # Input x and output are bfloat16 tensors
    rows = x.shape[0]
    cols = x.shape[1]

    # ------------------------------------------------------------
    # Core Partitioning: dynamically query Vector core count
    # ------------------------------------------------------------
    n_cores = tl.num_vec_cores()
    n_used = min(n_cores, rows)
    # Ceiling division so the last core picks up the remainder; kernel caps row_end_idx at rows.
    rows_per_core = (rows + n_used - 1) // n_used

    # ------------------------------------------------------------
    # Tiling Strategy (column tiling)
    # ------------------------------------------------------------
    # if columns too long → tile them
    max_tile_len = 8192          # user-defined UB capacity
    tile_length = max_tile_len
    n_tiles = (cols + tile_length - 1) // tile_length

    softmax_kernel[n_used](
        x, output,
        rows_per_core,
        tile_length,
        n_tiles,
        rows
    )
