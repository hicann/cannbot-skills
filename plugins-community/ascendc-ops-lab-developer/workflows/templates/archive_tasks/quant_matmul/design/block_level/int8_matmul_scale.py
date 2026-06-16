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

"""Block-level TileLang design for int8 matmul + per-column scale.

This layer only captures block scheduling, Cube/Vector collaboration, and the
cross-scope handoff. Tile-level compute details are intentionally left as TODOs
and are filled in by `design/tile_level/int8_matmul_scale.py`.
"""

import sys
from pathlib import Path

import tilelang
import tilelang.language as T

_COMMON_DIR = Path(__file__).resolve().parent.parent
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))
from int8_matmul_scale_common import (  
    pass_configs,
    kernel_setup,
)


@tilelang.jit(out_idx=[3], workspace_idx=4, pass_configs=pass_configs)
def int8_matmul_scale(
    m_size,
    n_size,
    k_size,
    dtype="int8",
    accum_dtype="int32",
):
    c = kernel_setup(m_size, n_size, k_size)
    block_m, block_n, block_k, k_l1 = c.block_m, c.block_n, c.block_k, c.k_l1
    m_num, n_num = c.m_num, c.n_num

    @T.prim_func
    def main(
        # Block-level interface for int8 matmul + per-column scale.
        a_mat: T.Tensor((m_size, k_size), dtype),
        b_mat: T.Tensor((k_size, n_size), dtype),
        scale: T.Tensor((n_size,), "float32"),
        c_mat: T.Tensor((m_size, n_size), "float16"),
        workspace: T.Tensor((m_size, n_size), accum_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            with T.Scope("C"):
                # TODO(tile-level):
                # - stage int8 A/B tiles through L1/L0
                # - run block matmul for tile (bx, by) with int32 accumulation
                # - store the accumulator tile into workspace
                # - signal Vector scope when the tile is ready
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                T.wait_cross_flag(0)

                # TODO(tile-level):
                # - load the accumulator tile from workspace
                # - broadcast the per-column scale tile
                # - convert int32 -> float32, apply scale, cast to float16
                # - store the final tile into C
                pass

    return main
