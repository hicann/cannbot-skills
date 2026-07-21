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


@ascend_kernel
def avgpool2d_kernel(
    input_ptr, output_ptr,
    batch_size, height, width, channels,
    out_h, out_w,
    kernel_h, kernel_w,
    tile_w,
    tasks_per_core,
    total_tasks
):

    pid = tl.program_id(0)

    task_start = pid * tasks_per_core
    task_end = min(task_start + tasks_per_core, total_tasks)

    # UB 缓冲区说明:
    #  x_ub  : 大小为 tw * C, 一次性载入 tw 列(每列 C 个通道)
    #  sum_ub: 大小为 C, 用于累加
    #  out_ub: 大小为 C, 用于输出
    x_ub = tl.alloc_ub(tile_w * channels, dtype=tl.float32)
    sum_ub = tl.alloc_ub(channels, dtype=tl.float32)
    out_ub = tl.alloc_ub(channels, dtype=tl.float32)

    kernel_area = kernel_h * kernel_w

    for task_id in range(task_start, task_end):

        # 将 task_id 解码为输出坐标 (b, oh, ow)
        b = task_id // (out_h * out_w)
        tmp = task_id % (out_h * out_w)
        oh = tmp // out_w
        ow = tmp % out_w

        h0 = oh * kernel_h
        w0 = ow * kernel_w

        # 将累加器清零
        with tl.compute():
            tl.duplicate(sum_ub, 0.0)

        # accumulate kernel window
        for kh in range(kernel_h):
            ih = h0 + kh

            # kw 分块
            for kw0 in range(0, kernel_w, tile_w):

                tw = min(tile_w, kernel_w - kw0)

                # ----------------------------
                # 一次 load tw * channels
                # ----------------------------
                base = (
                    b * height * width * channels +
                    ih * width * channels +
                    (w0 + kw0) * channels
                )
                offsets = base + tl.arange(0, tw * channels)

                with tl.copyin():
                    tl.load(input_ptr + offsets, x_ub)  # count = tw*C 默认

                # ----------------------------
                # 按行（channels）累加 tw 次
                # x_ub 按 [tw][channels] 组织
                # ----------------------------
                with tl.compute():
                    for t in range(tw):
                        tl.vadd(sum_ub,
                                sum_ub,
                                x_ub[t * channels],
                                channels)     # count = channels

        # divide
        with tl.compute():
            tl.vdiv_scalar(out_ub, sum_ub, kernel_area)

        # store
        out_base = (
            b * out_h * out_w * channels +
            oh * out_w * channels +
            ow * channels
        )
        out_offsets = out_base + tl.arange(0, channels)

        with tl.copyout():
            tl.store(output_ptr + out_offsets, out_ub)


def avgpool2d_host(x: torch.Tensor, output: torch.Tensor, kernel_size):

    # Input is NHWC
    batch, height, width, channels = x.shape
    kernel_h = kernel_size
    kernel_w = kernel_size

    # Output size
    out_h = height // kernel_h
    out_w = width // kernel_w

    # ================================================
    # 1. Core Partitioning: dynamically query Vector core count
    # ================================================
    n_cores = tl.num_vec_cores()
    total_tasks = batch * out_h * out_w

    n_used = min(n_cores, total_tasks)
    # Ceiling division so the last core picks up the remainder; kernel caps task_end at total_tasks.
    tasks_per_core = (total_tasks + n_used - 1) // n_used

    # ================================================
    # 2. Tiling Strategy
    # ================================================
    # tile_w：一次处理多少个宽度上的元素
    # 不会影响 sum_ub 的 UB 占用（固定为 channels 大小）
    # x_ub 的大小为 tile_w 乘以 channels
    tile_w = 3

    # ================================================
    # 3. Kernel Launch
    # ================================================
    avgpool2d_kernel[n_used](
        x, output,
        batch, height, width, channels,
        out_h, out_w,
        kernel_h, kernel_w,
        tile_w,
        tasks_per_core,
        total_tasks
    )
