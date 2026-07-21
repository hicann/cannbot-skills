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
def reduce_sum_kernel(
    input_ptr,                # [N]
    partial_gm,               # [n_used]
    output_ptr,               # scalar
    base_elems,
    pivot,
    tile_size,
    inner_loops
):
    pid = tl.program_id(0)
    # Pivot distribution: first 'pivot' cores get one extra element
    my_elems = base_elems + (1 if pid < pivot else 0)
    start = pid * base_elems + min(pid, pivot)

    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    x_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 compute even for bf16 IO
    accum_ub = tl.alloc_ub(tile_size, dtype=tl.float32)   # reused buffer; float32 compute even for bf16 IO
    shared_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # reduction workspace; float32 compute even for bf16 IO

    # ------------------------------------------------------------
    # Phase 1: Per-core partial reduction
    # ------------------------------------------------------------
    partial_sum = 0.0

    for i in range(inner_loops):
        # Compute how many elements remain for this core starting at this tile
        remaining = my_elems - i * tile_size
        if remaining <= 0:
            break
        valid_count = remaining if remaining < tile_size else tile_size

        tile_start = start + i * tile_size
        offsets = tile_start + tl.arange(0, tile_size)
        mask = tl.arange(0, tile_size) < valid_count

        # -------------------------------
        # COPYIN (masked to avoid OOB on last partial tile)
        # -------------------------------
        with tl.copyin():
            tl.load(input_ptr + offsets, x_ub, mask=mask, other=0.0)

        # -------------------------------
        # COMPUTE (reduce tile)
        # -------------------------------
        with tl.compute():
            tl.reduce_sum(accum_ub, x_ub, shared_ub)
            tile_sum = extract_scalar(accum_ub, 0)
            partial_sum = partial_sum + tile_sum

    # ------------------------------------------------------------
    # Write per-core partial sum to GM
    # ------------------------------------------------------------
    with tl.copyout():
        tl.set_scalar(partial_gm, pid, partial_sum)

    # ------------------------------------------------------------
    # Phase 2: Core 0 performs final reduction
    # ------------------------------------------------------------
    if pid == 0:

        # Load partial results from GM into UB
        with tl.copyin():
            tl.load(partial_gm + tl.arange(0, tl.num_programs(0)), accum_ub)

        # Final reduction
        with tl.compute():
            tl.reduce_sum(shared_ub, accum_ub, shared_ub)
            final_sum = extract_scalar(shared_ub, 0)

        # Copy out final sum to output scalar
        with tl.copyout():
            tl.set_scalar(output_ptr, 0, final_sum)


def reduce_sum_host(x: torch.Tensor, output: torch.Tensor):
    # Input x and output are bfloat16 tensors
    total_elems = x.numel()

    # ------------------------------------------------------------
    # Core Partitioning: dynamically query Vector core count
    # ------------------------------------------------------------
    n_cores = tl.num_vec_cores()
    n_used = min(n_cores, total_elems)   # avoid launching empty cores
    base_elems = total_elems // n_used        # each core processes at least this
    pivot = total_elems % n_used         # first 'pivot' cores get base_elems + 1

    # Allocate GM tensor for per-core partial sums (float32 workspace)
    partial_gm = torch.empty(n_used, dtype=torch.float32, device=x.device)

    # ------------------------------------------------------------
    # Tiling Strategy
    # ------------------------------------------------------------
    tile_size = 2048
    max_elems = base_elems + (1 if pivot > 0 else 0)
    inner_loops = (max_elems + tile_size - 1) // tile_size

    # ------------------------------------------------------------
    # Launch kernel
    # ------------------------------------------------------------
    reduce_sum_kernel[n_used](
        x,
        partial_gm,
        output,            # scalar output
        base_elems, pivot,
        tile_size,
        inner_loops
    )
