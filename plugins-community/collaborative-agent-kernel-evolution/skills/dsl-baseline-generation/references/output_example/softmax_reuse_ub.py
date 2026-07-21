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


@ascend_kernel
def softmax_kernel(input_ptr, output_ptr, rows, n_used, tile_length):
    pid = tl.program_id(0)

    # Each core processes a pivot-distributed range of rows so that all rows are covered.
    base_rows = rows // n_used
    pivot = rows % n_used

    if pid < pivot:
        # First 'pivot' cores handle one extra row each.
        row_start_idx = pid * (base_rows + 1)
        row_end_idx = row_start_idx + (base_rows + 1)
    else:
        # Remaining cores handle 'base_rows' rows each.
        row_start_idx = pivot * (base_rows + 1) + (pid - pivot) * base_rows
        row_end_idx = row_start_idx + base_rows

    # Allocate UB Buffers
    row_ub = tl.alloc_ub(tile_length, dtype=tl.float32)   # original row — must NOT be polluted
    exp_ub = tl.alloc_ub(tile_length, dtype=tl.float32)   # exp(row - max)
    shared_ub = tl.alloc_ub(tile_length, dtype=tl.float32)   # reduction workspace

    # Computation Logic
    for row_idx in range(row_start_idx, row_end_idx):

        # Compute offsets for this row
        offsets = row_idx * tile_length + tl.arange(0, tile_length)

        # -------------------------
        # Copy row into UB
        # -------------------------
        with tl.copyin():
            tl.load(input_ptr + offsets, row_ub)

        # -------------------------
        # Compute softmax
        # -------------------------
        with tl.compute():
            # --- Pass 1: max reduction ---
            tl.reduce_max(shared_ub, row_ub, shared_ub)
            row_max = tl.extract_scalar(shared_ub, 0)

            # --- Pass 2: exp(row - max) ---
            tl.vsub_scalar(exp_ub, row_ub, row_max)
            tl.vexp(exp_ub, exp_ub)

            # --- Pass 3: sum reduction ---
            tl.reduce_sum(shared_ub, exp_ub, shared_ub)
            row_sum = tl.extract_scalar(shared_ub, 0)

            # --- Pass 4: normalize ---
            tl.vdiv_scalar(exp_ub, exp_ub, row_sum)

        # -------------------------
        # Store result from UB
        # -------------------------
        with tl.copyout():
            tl.store(output_ptr + offsets, exp_ub)


def softmax_host(x: torch.Tensor, output: torch.Tensor):
    rows = x.shape[0]
    cols = x.shape[1]

    # Core Partitioning: dynamically query Vector core count
    n_cores = tl.num_vec_cores()
    n_used = min(n_cores, rows)

    # Tiling Strategy (row fits UB)
    tile_length = cols

    # Launch kernel with total rows and number of used cores; kernel does pivot distribution.
    softmax_kernel[n_used](x, output, rows, n_used, tile_length)
