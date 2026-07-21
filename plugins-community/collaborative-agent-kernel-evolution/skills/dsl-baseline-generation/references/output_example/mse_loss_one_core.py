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
from vecpipe import ascend_kernel


@ascend_kernel
def mse_loss_kernel(pred_ptr, target_ptr, out_ptr,
                    n_elements, tile_length, n_tiles):

    pid = tl.program_id(0)  # single core

    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    pred_ub = tl.alloc_ub(tile_length, dtype=tl.float32)
    target_ub = tl.alloc_ub(tile_length, dtype=tl.float32)
    diff_ub = tl.alloc_ub(tile_length, dtype=tl.float32)
    sq_ub = tl.alloc_ub(tile_length, dtype=tl.float32)

    acc_ub = tl.alloc_ub(tile_length, dtype=tl.float32)
    shared_ub = tl.alloc_ub(tile_length, dtype=tl.float32)

    out_ub = tl.alloc_ub(1, dtype=tl.float32)

    # ------------------------------------------------------------
    # Initialize accumulator
    # ------------------------------------------------------------
    with tl.compute():
        tl.duplicate(acc_ub, 0.0)

    # ------------------------------------------------------------
    # Tiled reduction over all elements
    # ------------------------------------------------------------
    for tile_id in range(n_tiles):

        tile_start = tile_id * tile_length
        offsets = tile_start + tl.arange(0, tile_length)

        # ---- Load tiles ----
        with tl.copyin():
            tl.load(pred_ptr + offsets, pred_ub)
            tl.load(target_ptr + offsets, target_ub)

        # ---- Compute squared error ----
        with tl.compute():
            tl.vsub(diff_ub, pred_ub, target_ub)
            tl.vmul(sq_ub, diff_ub, diff_ub)
            tl.vadd(acc_ub, acc_ub, sq_ub)

    # ------------------------------------------------------------
    # Final reduction and mean
    # ------------------------------------------------------------
    with tl.compute():
        tl.reduce_sum(shared_ub, acc_ub, shared_ub)
        total_sum = tl.extract_scalar(shared_ub, 0)
        mean_val = total_sum / n_elements

        # write scalar into UB buffer
        tl.duplicate(out_ub, mean_val, count=1)

    # ------------------------------------------------------------
    # Copy-out
    # ------------------------------------------------------------
    with tl.copyout():
        tl.store(out_ptr, out_ub)


def mse_loss_host(pred: torch.Tensor, target: torch.Tensor, output: torch.Tensor):

    # ------------------------------------------------------------
    # Core Partitioning
    # ------------------------------------------------------------
    n_cores = 1

    # ------------------------------------------------------------
    # Tiling Strategy
    # ------------------------------------------------------------
    # UB can comfortably hold a few KB; we tile the 1D vector
    n_elements = pred.numel()

    tile_length = 1024          # matches Triton TILE_SIZE
    n_tiles = (n_elements + tile_length - 1) // tile_length

    # ------------------------------------------------------------
    # Kernel Launch
    # ------------------------------------------------------------
    mse_loss_kernel[n_cores](
        pred,
        target,
        output,
        n_elements,
        tile_length,
        n_tiles
    )
