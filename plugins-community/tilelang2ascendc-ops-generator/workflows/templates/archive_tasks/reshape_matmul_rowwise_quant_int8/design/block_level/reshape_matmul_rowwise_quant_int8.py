#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Block-level TileLang design for reshape-view matmul + row-wise int8 quant.

Computation target (equivalent to `current_task/model.py`):
1) X: (M, N), H: (H_K, H_K), where N % H_K == 0.
2) View X as (M * (N / H_K), H_K), run matmul with H, reshape back to (M, N).
3) Per-row dynamic int8 quant on the reshaped output:
   scale_i = max(abs(row_i)) / 127, clamp_min(1e-12)
   y_i = round(row_i / scale_i), clipped to int8 range.

This block-level file keeps only scheduling/pipeline/cross-scope skeleton.
Tile-level compute details are filled in by
`design/tile_level/reshape_matmul_rowwise_quant_int8.py`.
"""

import sys
from pathlib import Path

import tilelang
import tilelang.language as T

_COMMON_DIR = Path(__file__).resolve().parent.parent
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))
from reshape_matmul_rowwise_quant_int8_common import (  
    pass_configs,
    kernel_setup,
)


@tilelang.jit(out_idx=[2], workspace_idx=3, pass_configs=pass_configs)
def reshape_matmul_rowwise_quant_int8(
    m_size,
    n_size,
    h_k,
    dtype="bfloat16",
    accum_dtype="float",
):
    c = kernel_setup(m_size, n_size, h_k)
    block_m, block_n, block_k, k_l1 = c.block_m, c.block_n, c.block_k, c.k_l1
    m_num, n_num, n_tiles_per_h = c.m_num, c.n_num, c.n_tiles_per_h

    @T.prim_func
    def main(
        x_in: T.Tensor((m_size, n_size), dtype),
        h_mat: T.Tensor((h_k, h_k), dtype),
        y_out: T.Tensor((m_size, n_size), "int8"),
        workspace: T.Tensor((m_size, n_size), accum_dtype),
    ):
        # One kernel task handles one output row tile (block_M rows).
        # It computes all N-tiles for this row tile, then quantizes row-wise.
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid

            with T.Scope("C"):
                # TODO(tile-level):
                # - loop by over all output column tiles
                # - map each by to (group_id, col_in_group) for reshape-view matmul
                # - run block matmul and write tile result to workspace
                # - signal Vector scope when all N-tiles are ready
                _ = n_num
                _ = n_tiles_per_h
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                T.wait_cross_flag(0)
                # TODO(tile-level):
                """
                # - pass 1: compute per-row absmax across all N tiles
                # - derive row scale = max(abs(row))/127 with clamp_min(1e-12)
                # - pass 2: divide by row scale, round, cast/saturate to int8
                # - write final quantized result to Y
                """
                pass

    return main
