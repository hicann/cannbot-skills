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


@ascend_kernel
def avgpool2d_kernel(
    input_ptr, output_ptr,
    batch_size, height, width, channels,
    out_h, out_w,
    kernel_h, kernel_w,
    tile_w,
    total_tasks,
    tasks_per_core
):
    # Kernel
    # Each core:
    #   - decodes its task_id into (b, oh, ow_group)
    #   - loads tile_w * channels elements into UB
    #   - computes windows_per_load output windows
    #   - safely handles tail windows (ow >= out_w)

    pid = tl.program_id(0)

    task_start = pid * tasks_per_core
    task_end = min(task_start + tasks_per_core, total_tasks)

    windows_per_load = tile_w // kernel_w
    kernel_area = kernel_h * kernel_w
    groups_per_row = (out_w + windows_per_load - 1) // windows_per_load

    # ------------------------------------------------
    # UB buffers
    # ------------------------------------------------
    x_ub = tl.alloc_ub(tile_w * channels, dtype=tl.float32)
    sum_ub = tl.alloc_ub(windows_per_load * channels, dtype=tl.float32)
    out_ub = tl.alloc_ub(windows_per_load * channels, dtype=tl.float32)

    for task_id in range(task_start, task_end):

        # --------------------------------------------
        # decode task
        # --------------------------------------------
        b = task_id // (out_h * groups_per_row)
        tmp = task_id % (out_h * groups_per_row)
        oh = tmp // groups_per_row
        ow_group = tmp % groups_per_row

        ow0 = ow_group * windows_per_load
        h0 = oh * kernel_h
        w0 = ow0 * kernel_w

        # --------------------------------------------
        # clear accumulators
        # --------------------------------------------
        with tl.compute():
            tl.duplicate(sum_ub, 0.0)

        # --------------------------------------------
        # pooling
        # --------------------------------------------
        for kh in range(kernel_h):
            ih = h0 + kh

            # compute valid load width
            max_valid_w = width - w0
            if max_valid_w <= 0:
                continue

            load_w = min(tile_w, max_valid_w)

            base = (
                b * height * width * channels +
                ih * width * channels +
                w0 * channels
            )

            with tl.copyin():
                tl.load(
                    input_ptr + base + tl.arange(0, load_w * channels),
                    x_ub
                )

            with tl.compute():
                for w in range(windows_per_load):
                    ow = ow0 + w
                    if ow < out_w:
                        x_offset = w * kernel_w
                        if x_offset + kernel_w <= load_w:
                            sum_ptr = sum_ub + w * channels
                            x_ptr = x_ub + x_offset * channels
                            for t in range(kernel_w):
                                tl.vadd(
                                    sum_ptr,
                                    sum_ptr,
                                    x_ptr + t * channels,
                                    channels
                                )

        # --------------------------------------------
        # divide
        # --------------------------------------------
        with tl.compute():
            tl.vdiv_scalar(out_ub, sum_ub, kernel_area)

        # --------------------------------------------
        # store
        # --------------------------------------------
        with tl.copyout():
            for w in range(windows_per_load):
                ow = ow0 + w
                if ow < out_w:
                    out_base = (
                        b * out_h * out_w * channels +
                        oh * out_w * channels +
                        ow * channels
                    )
                    tl.store(
                        output_ptr + out_base + tl.arange(0, channels),
                        out_ub + w * channels
                    )


def avgpool2d_host(
    x: torch.Tensor,
    output: torch.Tensor,
    kernel_size: int
):
    batch, height, width, channels = x.shape

    kernel_h = kernel_size
    kernel_w = kernel_size

    # ------------------------------------------------------------
    # Output spatial dimensions
    # ------------------------------------------------------------
    # We assume the stride equals the kernel size and there is no padding,
    # so the output is a simple non-overlapping tiling of the input.
    out_h = height // kernel_h
    out_w = width // kernel_w

    # ============================================================
    # UB tiling strategy (width dimension)
    # ============================================================
    # tile_w:
    #   Number of *input* width elements loaded per GM → UB transfer.
    #
    # For kernel_w = 3:
    #   tile_w = 18 means:
    #     - 18 input columns are loaded at once
    #     - 18 / 3 = 6 output windows are computed per load
    #
    # This maximizes:
    #   - memory coalescing
    #   - UB reuse
    #   - arithmetic intensity
    #
    tile_w = 18

    # windows_per_load:
    #   How many output windows we compute per load
    #
    # Example:
    #   tile_w = 18, kernel_w = 3
    #   → windows_per_load = 6
    #
    windows_per_load = tile_w // kernel_w

    n_cores = tl.num_vec_cores()

    # ============================================================
    # Task decomposition
    # ============================================================
    # We define one "task" as:
    #   - one batch index (b)
    #   - one output row (oh)
    #   - one *group* of output columns
    #
    # Each group computes:
    #   windows_per_load output columns
    #
    groups_per_row = (out_w + windows_per_load - 1) // windows_per_load

    # Total number of independent tasks in the whole workload
    #
    # The task space spans batch by out_h by groups_per_row.
    total_tasks = batch * out_h * groups_per_row

    # ============================================================
    # Task-to-core mapping
    # ============================================================
    # We use a simple linear partition:
    #
    #   Each core processes a contiguous range of tasks.
    #
    # Ceiling division ensures:
    #   - all tasks are covered
    #   - some cores may process one extra task
    #
    n_used = min(n_cores, total_tasks)
    tasks_per_core = (total_tasks + n_used - 1) // n_used

    # ============================================================

    #
    avgpool2d_kernel[n_used](
        x, output,
        batch, height, width, channels,
        out_h, out_w,
        kernel_h, kernel_w,
        tile_w,
        total_tasks,
        tasks_per_core
    )
