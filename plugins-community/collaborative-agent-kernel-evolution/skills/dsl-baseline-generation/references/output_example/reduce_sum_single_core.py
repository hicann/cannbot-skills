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

# ============================================================================
# Single-core streaming global reduction to a 0-dim `{}` scalar.
#
# DEFAULT skeleton for ops whose output is a single scalar aggregated over the
# WHOLE tensor: global norm (foreach_norm per element), global MSE, sum-over-all.
#
# One core loops over ALL tiles and accumulates into a single fp32 scalar, then
# writes it out once. There is deliberately:
#   - NO per-core partial buffer in GM,
#   - NO SyncAll(),
#   - NO cross-core final reduce.
# That removes the entire class of cross-core ordering / workspace-stride /
# last-core-OOB bugs that make the multi-core variant (reduce_sum.py) so hard to
# get correct. UB only ever holds one tile, so an arbitrarily large tensor fits;
# this trades throughput for correctness.
#
# ⚠️ SCOPE: single-core-first is ONLY for `{}` scalar-output global reductions.
# Elementwise ops and axis/row reductions (softmax, layernorm — one result per
# row) stay MULTI-CORE. Do not copy this single-core shape onto them.
#
# Promotion path: keep this until it passes the precision gate, THEN switch to
# the multi-core reduce_sum.py for performance (dsl-optimization / evolution).
#
# bf16/fp16 variant: input/output GM may be bf16/fp16; the UB accumulator is
# float32 for precision (Cast handled during lowering).
# ============================================================================


@ascend_kernel
def reduce_sum_single_core_kernel(
    input_ptr,                # [N]
    output_ptr,               # scalar {}
    total_elems,
    tile_size,
    inner_loops
):
    pid = tl.program_id(0)    # launched with 1 core: only pid 0 does work
    if pid != 0:
        return

    x_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 compute even for bf16/fp16 IO
    accum_ub = tl.alloc_ub(tile_size, dtype=tl.float32)
    shared_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # reduction workspace (in-UB, not GM)

    # ------------------------------------------------------------
    # Stream every tile of the whole tensor, accumulating one scalar
    # ------------------------------------------------------------
    partial_sum = 0.0
    for i in range(inner_loops):
        remaining = total_elems - i * tile_size
        if remaining <= 0:
            break
        valid_count = remaining if remaining < tile_size else tile_size

        offsets = i * tile_size + tl.arange(0, tile_size)
        mask = tl.arange(0, tile_size) < valid_count      # mask tail -> no OOB / garbage read

        with tl.copyin():
            tl.load(input_ptr + offsets, x_ub, mask=mask, other=0.0)

        with tl.compute():
            # p-norm hook (foreach_norm): branch on the `scalar` (p) attribute HERE.
            #   p == 1   -> reduce_sum(|x|)
            #   p == 2   -> reduce_sum(x * x)
            #   p == inf -> reduce_max(|x|)   (accumulate with max, not +)
            #   general  -> reduce_sum(|x|^p)
            tl.reduce_sum(accum_ub, x_ub, shared_ub)
            partial_sum = partial_sum + extract_scalar(accum_ub, 0)

    # ------------------------------------------------------------
    # Finalize (p-norm): p == 2 -> sqrt(partial_sum);
    #                    general -> pow(partial_sum, 1/p);
    #                    p == 1 / inf -> write as-is.
    # ------------------------------------------------------------
    with tl.copyout():
        tl.set_scalar(output_ptr, 0, partial_sum)


def reduce_sum_single_core_host(x: torch.Tensor, output: torch.Tensor):
    total_elems = x.numel()

    # Single core on purpose (correctness-first): no partial_gm / workspace.
    tile_size = 2048
    inner_loops = (total_elems + tile_size - 1) // tile_size

    reduce_sum_single_core_kernel[1](
        x,
        output,            # scalar {} output
        total_elems,
        tile_size,
        inner_loops,
    )
