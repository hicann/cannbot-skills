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
import torch

# bf16 variant: input/output GM data is bfloat16
# UB computation uses float32 for precision (Cast bf16↔f32 handled during lowering)

# ============================================================
# 2. KERNEL FUNCTION
# ============================================================


@ascend_kernel
def cumsum_kernel(
    input_ptr, output_ptr,
    scan_len,
    inner_size,
    tile_inner,
    base_tasks,
    pivot
):
    """
    Kernel computes inclusive prefix sum along scan axis (axis=0).
    """

    pid = tl.program_id(0)

    # ------------------------------------------------------------
    # Pivot distribution: first 'pivot' cores get one extra task
    # ------------------------------------------------------------
    my_tasks = base_tasks + (1 if pid < pivot else 0)
    task_start = pid * base_tasks + min(pid, pivot)
    task_end = task_start + my_tasks

    # ------------------------------------------------------------
    # 2.1 Allocate UB Buffers
    # ------------------------------------------------------------
    x_ub = tl.alloc_ub(tile_inner, dtype=tl.float32)  # float32 compute even for bf16 IO
    acc_ub = tl.alloc_ub(tile_inner, dtype=tl.float32)  # float32 compute even for bf16 IO
    out_ub = tl.alloc_ub(tile_inner, dtype=tl.float32)  # float32 compute even for bf16 IO

    # ============================================================
    # 2.2 Computation Logic
    # ============================================================
    for task_id in range(task_start, task_end):

        # Decode task_id → inner offset
        inner_base = task_id * tile_inner

        # Remaining elements from this inner_base to the end
        remaining = inner_size - inner_base
        if remaining <= 0:
            # No work left for this (overallocated) task
            continue

        # Number of elements this task will actually process
        curr_tile = tile_inner if remaining >= tile_inner else remaining

        # --------------------------------------------------------
        # acc_ub = 0 (initialize prefix accumulator)
        # --------------------------------------------------------
        with tl.compute():
            tl.duplicate(acc_ub, 0.0)

        # --------------------------------------------------------
        # Sequential scan along scan axis
        # --------------------------------------------------------
        for i in range(scan_len):

            base = i * inner_size + inner_base
            offsets = base + tl.arange(0, curr_tile)

            # Load x[i, inner_slice]
            with tl.copyin():
                tl.load(input_ptr + offsets, x_ub[:curr_tile])

            # 将当前片累加到前缀累加器中
            with tl.compute():
                tl.vadd(acc_ub[:curr_tile], acc_ub[:curr_tile], x_ub[:curr_tile])
                tl.vadd_scalar(out_ub[:curr_tile], acc_ub[:curr_tile], 0.0)

            with tl.copyout():
                tl.store(output_ptr + offsets, out_ub[:curr_tile])


def cumsum_host(x: torch.Tensor, output: torch.Tensor):
    """
    Host Function:
    - Core partitioning
    - Tiling strategy
    - Kernel launch
    """
    # Input x and output are bfloat16 tensors

    # Input already permuted: scan axis is axis 0
    scan_len = x.shape[0]
    inner_size = x.numel() // scan_len

    # ============================================================
    # 1.2 Tiling Strategy
    # ============================================================
    max_tile_inner = 1024
    tile_inner = min(max_tile_inner, inner_size)

    # ------------------------------------------------------------
    # Task definition: one task = one tile_inner slice
    # ------------------------------------------------------------
    total_tasks = (inner_size + tile_inner - 1) // tile_inner

    # ============================================================
    # 1.1 Core Partitioning: dynamically query Vector core count
    # ============================================================
    n_cores = tl.num_vec_cores()
    n_used = min(n_cores, total_tasks)   # avoid launching empty cores
    base_tasks = total_tasks // n_used        # each core handles at least this many tasks
    pivot = total_tasks % n_used         # first 'pivot' cores get one extra task

    # ============================================================
    # Kernel Launch
    # ============================================================
    cumsum_kernel[n_used](
        x, output,
        scan_len,
        inner_size,
        tile_inner,
        base_tasks,
        pivot
    )
