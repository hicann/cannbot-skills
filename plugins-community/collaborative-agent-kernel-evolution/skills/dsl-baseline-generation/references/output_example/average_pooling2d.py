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


@ascend_kernel
def avgpool2d_kernel(
    input_ptr, output_ptr,
    batch_size, height, width, channels,
    out_h, out_w,
    kernel_h, kernel_w,
    tile_c,
    base_tasks,
    pivot
):

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
    x_ub = tl.alloc_ub(tile_c, dtype=tl.float32)     # input slice; float32 compute even for bf16 IO
    sum_ub = tl.alloc_ub(tile_c, dtype=tl.float32)     # accumulation; float32 compute even for bf16 IO
    out_ub = tl.alloc_ub(tile_c, dtype=tl.float32)     # final output; float32 compute even for bf16 IO

    kernel_area = kernel_h * kernel_w

    # ============================================================
    # 2.2 Computation Logic
    # ============================================================
    for task_id in range(task_start, task_end):

        # Decode task_id → (b, oh, ow)
        b = task_id // (out_h * out_w)
        tmp = task_id % (out_h * out_w)
        oh = tmp // out_w
        ow = tmp % out_w

        # pool window start
        h0 = oh * kernel_h
        w0 = ow * kernel_w

        # Process channels in vector tiles
        for c0 in range(0, channels, tile_c):

            # ----------------------------------------------------
            # 将累加器清零
            # ----------------------------------------------------
            with tl.compute():
                tl.duplicate(sum_ub, 0.0)

            # ----------------------------------------------------
            # Accumulate kernel window
            # sum_ub += all x[b, ih, iw, c_slice]
            # ----------------------------------------------------
            for kh in range(kernel_h):
                ih = h0 + kh
                for kw in range(kernel_w):
                    iw = w0 + kw

                    base = (
                        b * height * width * channels +
                        ih * width * channels +
                        iw * channels +
                        c0
                    )
                    offsets = base + tl.arange(0, tile_c)

                    # Load one tile into x_ub
                    with tl.copyin():
                        tl.load(input_ptr + offsets, x_ub)

                    # Accumulate
                    with tl.compute():
                        tl.vadd(sum_ub, sum_ub, x_ub)

            # ----------------------------------------------------
            # 将累加结果除以窗口面积得到平均值
            # ----------------------------------------------------
            with tl.compute():
                tl.vdiv_scalar(out_ub, sum_ub, kernel_area)

            # ----------------------------------------------------
            # Store result y[b, oh, ow, c_slice]
            # ----------------------------------------------------
            out_base = (
                b * out_h * out_w * channels +
                oh * out_w * channels +
                ow * channels +
                c0
            )
            out_offsets = out_base + tl.arange(0, tile_c)

            with tl.copyout():
                tl.store(output_ptr + out_offsets, out_ub)


def avgpool2d_host(x: torch.Tensor, output: torch.Tensor, kernel_size):
    """
    Host Function:
    - Core partitioning
    - Tiling strategy
    - Launch kernel
    """
    # Input x and output are bfloat16 tensors

    # Input shape: NHWC
    batch, height, width, channels = x.shape
    kernel_h = kernel_size
    kernel_w = kernel_size

    out_h = height // kernel_h
    out_w = width // kernel_w

    total_tasks = batch * out_h * out_w

    # ============================================================
    # 1.1 Core Partitioning: dynamically query Vector core count
    # ============================================================
    n_cores = tl.num_vec_cores()
    n_used = min(n_cores, total_tasks)   # avoid launching empty cores
    base_tasks = total_tasks // n_used        # each core handles at least this many tasks
    pivot = total_tasks % n_used         # first 'pivot' cores get one extra task

    # ============================================================
    # 1.2 Tiling Strategy
    # ============================================================
    # UB capacity constraint: need 3 buffers (x_ub, sum_ub, out_ub)
    # Each tile handles tile_c contiguous channels.
    max_tile_c = 1024
    tile_c = min(max_tile_c, channels)

    # Kernel launch
    avgpool2d_kernel[n_used](
        x, output,
        batch, height, width, channels,
        out_h, out_w,
        kernel_h, kernel_w,
        tile_c,
        base_tasks, pivot
    )
