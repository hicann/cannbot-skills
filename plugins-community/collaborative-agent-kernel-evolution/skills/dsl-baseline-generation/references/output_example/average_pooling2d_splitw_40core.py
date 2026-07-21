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
    total_tasks,    # 新：总任务数 = batch * out_h * out_w
    n_cores         # 新：核数（host 端传入 40）
):

    pid = tl.program_id(0)

    # -------------------------
    # 计算每 core 的任务区间（尽可能平均分配）
    # base 个任务给每个核，前 rem 个核多分配 1 个任务
    # -------------------------
    base = total_tasks // n_cores
    rem = total_tasks % n_cores

    # 如果 pid < rem => 每个这类核处理 base+1 个任务，起点为 pid*(base+1)
    # 否则起点为 rem*(base+1) + (pid-rem)*base
    is_small = pid < rem

    # 计算 start/end（整数算术）
    # 注意 tl 本身支持 Python 风格算术，这里按直观写法
    if is_small:
        task_start = pid * (base + 1)
        task_end = task_start + (base + 1)
    else:
        task_start = rem * (base + 1) + (pid - rem) * base
        task_end = task_start + base

    # UB 缓冲区说明:
    #  x_ub  : 大小为 tile_w * channels, 一次性载入 tile_w 列(每列 channels 个通道)
    #  sum_ub: 大小为 channels, 用于累加
    #  out_ub: 大小为 channels, 用于输出
    x_ub = tl.alloc_ub(tile_w * channels, dtype=tl.float32)
    sum_ub = tl.alloc_ub(channels, dtype=tl.float32)
    out_ub = tl.alloc_ub(channels, dtype=tl.float32)

    kernel_area = kernel_h * kernel_w
    inv_area = 1.0 / float(kernel_area)   # 使用浮点倒数避免整数除法问题

    # =================================================
    # 主任务循环（每个 task_id 对应一个输出点 y[b, oh, ow]）
    # =================================================
    for task_id in range(task_start, task_end):

        # decode (b, oh, ow) —— total_tasks == batch * out_h * out_w
        b = task_id // (out_h * out_w)
        tmp = task_id % (out_h * out_w)
        oh = tmp // out_w
        ow = tmp % out_w

        # 输入窗口左上角
        h0 = oh * kernel_h
        w0 = ow * kernel_w

        # 将累加器清零
        with tl.compute():
            tl.duplicate(sum_ub, 0.0)

        # accumulate kernel window
        for kh in range(kernel_h):
            ih = h0 + kh

            # 按 tile_w 分块遍历 kernel_w
            for kw0 in range(0, kernel_w, tile_w):

                tw = min(tile_w, kernel_w - kw0)

                # ----------------------------
                # 一次 load tw * channels（连续区域）
                # ----------------------------
                base = (
                    b * height * width * channels +
                    ih * width * channels +
                    (w0 + kw0) * channels
                )
                offsets = base + tl.arange(0, tw * channels)

                with tl.copyin():
                    # 大多数后端支持不带 count 的 tl.load；若支持 count 可传入
                    try:
                        tl.load(input_ptr + offsets, x_ub, count=tw * channels)
                    except Exception:
                        tl.load(input_ptr + offsets, x_ub)

                # ----------------------------
                # 按行（channels）累加 tw 次
                # x_ub 按 [tw][channels] 组织
                # ----------------------------
                with tl.compute():
                    for t in range(tw):
                        # 正确的指针偏移写法
                        src_ptr = x_ub + t * channels
                        tl.vadd(
                            sum_ub,         # dest = src1 + src2
                            sum_ub,         # src1
                            src_ptr,        # src2
                            channels
                        )

        # 归一化（乘以浮点倒数）
        with tl.compute():
            tl.vmul_scalar(out_ub, sum_ub, inv_area)

        # store 返回输出 y[b, oh, ow, :]
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

    # 允许 total_tasks < n_cores 的情况：某些核可能无任务（task_count=0）
    # 但 kernel 内的分配逻辑仍然成立（base=0, rem=total_tasks）
    # 所以不需要强制要求能整除
    tasks_per_core = None  # 已不再使用，kernel 内自行计算

    # ================================================
    # 2. Tiling Strategy
    # ================================================
    # tile_w：一次处理多少个宽度上的元素
    # 选择时注意 UB 大小： x_ub = tile_w * channels
    tile_w = 3    # 可根据 UB 及性能调优

    # ================================================
    # 3. Kernel Launch（传入 total_tasks 与 n_cores）
    # ================================================
    avgpool2d_kernel[n_cores](
        x, output,
        batch, height, width, channels,
        out_h, out_w,
        kernel_h, kernel_w,
        tile_w,
        total_tasks,
        n_cores
    )
