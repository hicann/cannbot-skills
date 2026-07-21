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
def leaky_relu_kernel(input_ptr, output_ptr,
                      base_elems, pivot, tile_size, inner_loops, negative_slope):

    pid = tl.program_id(0)
    # Pivot distribution: first 'pivot' cores get one extra element
    my_elems = base_elems + (1 if pid < pivot else 0)
    start = pid * base_elems + min(pid, pivot)

    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    x_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 compute even for bf16 IO
    pos_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 compute even for bf16 IO
    neg_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 compute even for bf16 IO
    out_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 compute even for bf16 IO

    # ------------------------------------------------------------
    # Tile loop (inner_loops covers my_elems rounded up to tile_size)
    # ------------------------------------------------------------
    for i in range(inner_loops):
        tile_start = start + i * tile_size
        offsets = tile_start + tl.arange(0, tile_size)
        # Mask to ensure we only touch elements within this core's range
        mask = offsets < (start + my_elems)

        # --------------------------------------------------------
        # COPYIN
        # --------------------------------------------------------
        with tl.copyin():
            tl.load(input_ptr + offsets, x_ub, mask=mask)

        # --------------------------------------------------------
        # COMPUTE
        # --------------------------------------------------------
        with tl.compute():
            # 取正部 max(x, 0)
            tl.vmax(pos_ub, x_ub, 0.0)

            # 取负部 min(x, 0)
            tl.vmin(neg_ub, x_ub, 0.0)

            # 负部乘以斜率系数
            tl.vmul_scalar(neg_ub, neg_ub, negative_slope)

            # 正部与缩放后的负部相加得到输出
            tl.vadd(out_ub, pos_ub, neg_ub)

        # --------------------------------------------------------
        # COPYOUT
        # --------------------------------------------------------
        with tl.copyout():
            tl.store(output_ptr + offsets, out_ub, mask=mask)


def leaky_relu_host(x: torch.Tensor, output: torch.Tensor, negative_slope: float):
    # Input x and output are bfloat16 tensors
    total_elems = x.numel()

    # ------------------------------------------------------------
    # Core Partitioning: dynamically query Vector core count
    # ------------------------------------------------------------
    n_cores = tl.num_vec_cores()
    n_used = min(n_cores, total_elems)   # avoid launching empty cores
    base_elems = total_elems // n_used        # each core processes at least this
    pivot = total_elems % n_used         # first 'pivot' cores get base_elems + 1

    # ------------------------------------------------------------
    # Tiling Strategy
    # ------------------------------------------------------------
    tile_size = 2048
    max_elems = base_elems + (1 if pivot > 0 else 0)
    inner_loops = (max_elems + tile_size - 1) // tile_size

    # ------------------------------------------------------------
    # Launch kernel
    # ------------------------------------------------------------
    leaky_relu_kernel[n_used](
        x, output,
        base_elems, pivot,
        tile_size,
        inner_loops,
        negative_slope
    )
