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
def mse_loss_kernel(
    pred_ptr,                 # [N]
    target_ptr,               # [N]
    workspace,                # [n_used]  per-core partial sums
    output_ptr,               # final scalar
    base_elems,
    pivot,
    tile_size,
    inner_loops,
    total_elems               # for computing mean
):
    pid = tl.program_id(0)
    # Pivot distribution: first 'pivot' cores get one extra element
    my_elems = base_elems + (1 if pid < pivot else 0)
    start = pid * base_elems + min(pid, pivot)

    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    pred_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 compute even for bf16 IO
    target_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 compute even for bf16 IO
    diff_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 compute even for bf16 IO
    sq_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 compute even for bf16 IO
    shared_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 compute even for bf16 IO
    workspace_out_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 compute even for bf16 IO
    workspace_in_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 compute even for bf16 IO
    output_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 compute even for bf16 IO

    # ------------------------------------------------------------
    # Phase 1: per-core partial reduction of squared errors
    # ------------------------------------------------------------
    partial_sum = 0.0

    for i in range(inner_loops):
        tile_start = start + i * tile_size
        offsets = tile_start + tl.arange(0, tile_size)

        # -------------------------------
        # COPYIN
        # -------------------------------
        with tl.copyin():
            tl.load(pred_ptr + offsets, pred_ub)
            tl.load(target_ptr + offsets, target_ub)

        # -------------------------------
        # COMPUTE: squared error between pred and target
        # -------------------------------
        with tl.compute():
            tl.vsub(diff_ub, pred_ub, target_ub)
            tl.vmul(sq_ub, diff_ub, diff_ub)
            tl.reduce_sum(sq_ub, sq_ub, shared_ub)
            tile_sum = extract_scalar(sq_ub, 0)
            partial_sum = partial_sum + tile_sum

    # ------------------------------------------------------------
    # Write per-core partial sum
    # ------------------------------------------------------------
    with tl.copyout():
        tl.set_scalar(workspace_out_ub, 0, partial_sum)
        tl.store(workspace + tl.arange(pid, pid + 1), workspace_out_ub)

    # ------------------------------------------------------------
    # Phase 2: Core 0 performs final reduce + mean
    # ------------------------------------------------------------
    if pid == 0:

        # ---------------------------
        # Load all partial sums
        # ---------------------------
        with tl.copyin():
            tl.load(workspace + tl.arange(0, tl.num_programs(0)), workspace_in_ub)

        # ---------------------------
        # Reduce
        # ---------------------------
        with tl.compute():
            tl.reduce_sum(shared_ub, workspace_in_ub, shared_ub)
            sum_sq = extract_scalar(shared_ub, 0)
            mse = sum_sq / float(total_elems)
            tl.set_scalar(output_ub, 0, mse)

        # ---------------------------
        # Copyout final scalar
        # ---------------------------
        with tl.copyout():
            tl.store(output_ptr + tl.arange(0, 1), output_ub)


def mse_loss_host(pred: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    # Input x and output are bfloat16 tensors
    total_elems = pred.numel()

    # ------------------------------------------------------------
    # Core Partitioning: dynamically query Vector core count
    # ------------------------------------------------------------
    n_cores = tl.num_vec_cores()
    n_used = min(n_cores, total_elems)   # avoid launching empty cores
    base_elems = total_elems // n_used        # each core processes at least this
    pivot = total_elems % n_used         # first 'pivot' cores get base_elems + 1

    # GM buffer for per-core partial results (float32 workspace, size = n_used, not n_cores)
    workspace = torch.empty(n_used, dtype=torch.float32, device=pred.device)

    # ------------------------------------------------------------
    # Tiling Strategy
    # ------------------------------------------------------------
    # Use a tile size of 1 to ensure no partial tiles and avoid potential
    # out-of-bounds accesses when the per-core element count is not
    # divisible by a larger tile size.
    tile_size = 1
    max_elems = base_elems + (1 if pivot > 0 else 0)
    inner_loops = (max_elems + tile_size - 1) // tile_size

    # ------------------------------------------------------------
    # Launch kernel
    # ------------------------------------------------------------
    mse_loss_kernel[n_used](
        pred,
        target,
        workspace,
        output,
        base_elems, pivot,
        tile_size,
        inner_loops,
        total_elems
    )
