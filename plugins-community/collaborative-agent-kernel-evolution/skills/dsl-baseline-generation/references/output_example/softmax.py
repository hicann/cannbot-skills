# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import torch
import tile.language as tl

# bf16 variant: input/output GM data is bfloat16
# UB computation uses float32 for precision (Cast bf16↔f32 handled during lowering)


@ascend_kernel
def softmax_kernel(input_ptr, output_ptr, base_rows, pivot, tile_length):
    pid = tl.program_id(0)
    # Pivot distribution: first 'pivot' cores get one extra row
    my_rows = base_rows + (1 if pid < pivot else 0)
    row_start_idx = pid * base_rows + min(pid, pivot)
    row_end_idx = row_start_idx + my_rows

    # Allocate UB Buffers
    row_ub = tl.alloc_ub(tile_length, dtype=tl.float32)  # float32 compute even for bf16 IO
    exp_ub = tl.alloc_ub(tile_length, dtype=tl.float32)  # float32 compute even for bf16 IO
    shared_ub = tl.alloc_ub(tile_length, dtype=tl.float32)  # float32 compute even for bf16 IO
    out_ub = tl.alloc_ub(tile_length, dtype=tl.float32)  # float32 compute even for bf16 IO

    # Computation Logic
    for row_idx in range(row_start_idx, row_end_idx):
        offsets = row_idx * tile_length + tl.arange(0, tile_length)

        with tl.copyin():
            tl.load(input_ptr + offsets, row_ub)

        with tl.compute():
            tl.reduce_max(shared_ub, row_ub, shared_ub)
            row_max = extract_scalar(shared_ub, 0)
            tl.vsub_scalar(exp_ub, row_ub, row_max)
            tl.vexp(exp_ub, exp_ub)
            tl.reduce_sum(shared_ub, exp_ub, shared_ub)
            row_sum = extract_scalar(shared_ub, 0)
            tl.vdiv_scalar(out_ub, exp_ub, row_sum)

        with tl.copyout():
            tl.store(output_ptr + offsets, out_ub)


def softmax_host(x: torch.Tensor, output: torch.Tensor):
    # Input x and output are bfloat16 tensors
    rows = x.shape[0]
    cols = x.shape[1]

    # Core Partitioning: dynamically query Vector core count
    n_cores = tl.num_vec_cores()
    n_used = min(n_cores, rows)   # avoid launching empty cores
    base_rows = rows // n_used        # each core processes at least this many rows
    pivot = rows % n_used         # first 'pivot' cores get base_rows + 1

    # Tiling Strategy: entire row fits into UB
    tile_length = cols

    softmax_kernel[n_used](x, output, base_rows, pivot, tile_length)
