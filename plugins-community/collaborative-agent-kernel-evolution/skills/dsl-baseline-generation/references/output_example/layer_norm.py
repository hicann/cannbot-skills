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
def layernorm_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr,
                     base_rows, pivot, tile_length, n_tiles, norm_size, eps):

    pid = tl.program_id(0)
    # Pivot distribution: first 'pivot' cores get one extra row
    my_rows = base_rows + (1 if pid < pivot else 0)
    row_start_idx = pid * base_rows + min(pid, pivot)
    row_end_idx = row_start_idx + my_rows

    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    row_tile_ub = tl.alloc_ub(tile_length, dtype=tl.float32)   # tile of input row; float32 compute even for bf16 IO
    weight_tile_ub = tl.alloc_ub(tile_length, dtype=tl.float32)  # tile of weight; float32 compute even for bf16 IO
    bias_tile_ub = tl.alloc_ub(tile_length, dtype=tl.float32)  # tile of bias; float32 compute even for bf16 IO
    # temporary computation buffer; float32 compute even for bf16 IO
    temp_ub = tl.alloc_ub(tile_length, dtype=tl.float32)
    shared_ub = tl.alloc_ub(tile_length, dtype=tl.float32)   # reduction workspace; float32 compute even for bf16 IO
    out_ub = tl.alloc_ub(tile_length, dtype=tl.float32)   # output buffer; float32 compute even for bf16 IO

    # ------------------------------------------------------------
    # Per-row computation
    # ------------------------------------------------------------
    for row_idx in range(row_start_idx, row_end_idx):

        # ========================================================
        # PASS 1: compute mean of the row (tiled)
        # ========================================================
        row_sum = 0.0

        for tile_id in range(n_tiles):
            col_start = tile_id * tile_length
            offsets = row_idx * norm_size + col_start + tl.arange(0, tile_length)

            # ---- Load tile ----
            with tl.copyin():
                tl.load(input_ptr + offsets, row_tile_ub)

            # ---- Compute tile sum ----
            with tl.compute():
                tl.reduce_sum(shared_ub, row_tile_ub, shared_ub)
                tile_sum = tl.extract_scalar(shared_ub, 0)
            row_sum = row_sum + tile_sum

        # Compute mean
        row_mean = row_sum / norm_size

        # ========================================================
        # PASS 2: compute variance (sum of squared differences)
        # ========================================================
        row_var_sum = 0.0

        for tile_id in range(n_tiles):
            col_start = tile_id * tile_length
            offsets = row_idx * norm_size + col_start + tl.arange(0, tile_length)

            # ---- Load tile ----
            with tl.copyin():
                tl.load(input_ptr + offsets, row_tile_ub)

            # ---- Compute (x - mean)^2 for this tile ----
            with tl.compute():
                tl.vsub_scalar(temp_ub, row_tile_ub, row_mean)
                tl.vmul(temp_ub, temp_ub, temp_ub)
                tl.reduce_sum(shared_ub, temp_ub, shared_ub)
                tile_var = tl.extract_scalar(shared_ub, 0)
            row_var_sum = row_var_sum + tile_var

        # Compute variance and std
        row_var = row_var_sum / norm_size
        row_std = tl.vsqrt(row_var + eps)

        # ========================================================
        # PASS 3: normalize, scale, and shift each tile
        # ========================================================
        for tile_id in range(n_tiles):
            col_start = tile_id * tile_length
            offsets = row_idx * norm_size + col_start + tl.arange(0, tile_length)
            weight_offsets = col_start + tl.arange(0, tile_length)

            # ---- Load input, weight, and bias tiles ----
            with tl.copyin():
                tl.load(input_ptr + offsets, row_tile_ub)
                tl.load(weight_ptr + weight_offsets, weight_tile_ub)
                tl.load(bias_ptr + weight_offsets, bias_tile_ub)

            # ---- Normalize: (x - mean) / std ----
            with tl.compute():
                tl.vsub_scalar(temp_ub, row_tile_ub, row_mean)
                tl.vdiv_scalar(temp_ub, temp_ub, row_std)

                # Scale and shift: out = normalized * weight + bias
                tl.vmul(out_ub, temp_ub, weight_tile_ub)
                tl.vadd(out_ub, out_ub, bias_tile_ub)

            # ---- Store tile ----
            with tl.copyout():
                tl.store(output_ptr + offsets, out_ub)


def layernorm_host(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                   output: torch.Tensor, eps: float = 1e-5):
    # Input x and output are bfloat16 tensors
    # ------------------------------------------------------------
    # Reshape input to 2D: (batch, norm_size)
    # ------------------------------------------------------------
    batch_size = x.shape[0]
    norm_size = weight.numel()  # features * dim1 * dim2
    rows = batch_size

    # ------------------------------------------------------------
    # Core Partitioning: dynamically query Vector core count
    # ------------------------------------------------------------
    n_cores = tl.num_vec_cores()
    n_used = min(n_cores, rows)   # avoid launching empty cores
    base_rows = rows // n_used        # each core processes at least this many rows
    pivot = rows % n_used         # first 'pivot' cores get base_rows + 1

    # ------------------------------------------------------------
    # Tiling Strategy (column tiling for normalized dimensions)
    # ------------------------------------------------------------
    max_tile_len = 4096          # user-defined UB capacity
    tile_length = min(max_tile_len, norm_size)
    n_tiles = (norm_size + tile_length - 1) // tile_length

    layernorm_kernel[n_used](
        x, weight, bias, output,
        base_rows, pivot,
        tile_length,
        n_tiles,
        norm_size,
        eps
    )
