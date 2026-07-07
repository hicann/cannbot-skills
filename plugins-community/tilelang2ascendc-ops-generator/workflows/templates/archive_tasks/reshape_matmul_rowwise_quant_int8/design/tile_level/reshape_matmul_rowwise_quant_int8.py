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

# Template for: reshape_matmul_rowwise_quant_int8 (tile-level)
import sys
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout

# Tile-level uses a one-liner path setup to differ from block-level.
_p = Path(__file__).resolve().parent.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
import reshape_matmul_rowwise_quant_int8_common as _common  
pass_configs = _common.pass_configs
kernel_setup = _common.kernel_setup


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
    vec_num = 2
    sub_block_m = block_m // vec_num

    inv_127 = T.float32(1.0 / 127.0)

    @T.prim_func
    def main(  
        x_in: T.Tensor((m_size, n_size), dtype),
        h_mat: T.Tensor((h_k, h_k), dtype),
        y_out: T.Tensor((m_size, n_size), "int8"),
        workspace: T.Tensor((m_size, n_size), accum_dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid

            a_l1 = T.alloc_L1((block_m, k_l1), dtype)
            b_l1 = T.alloc_L1((k_l1, block_n), dtype)
            T.annotate_layout({
                a_l1: make_zn_layout(a_l1),
                b_l1: make_zn_layout(b_l1),
            })
            a_l0 = T.alloc_L0A((block_m, block_k), dtype)
            b_l0 = T.alloc_L0B((block_k, block_n), dtype)
            c_l0 = T.alloc_L0C((block_m, block_n), accum_dtype)

            out_fp32_ub = T.alloc_ub((sub_block_m, block_n), "float32")
            out_abs_ub = T.alloc_ub((sub_block_m, block_n), "float32")
            row_absmax_ub = T.alloc_ub((sub_block_m,), "float32")
            row_absmax_tile_ub = T.alloc_ub((sub_block_m,), "float32")
            row_scale_ub = T.alloc_ub((sub_block_m,), "float32")
            out_fp16_ub = T.alloc_ub((sub_block_m, block_n), "float16")
            out_int8_ub = T.alloc_ub((sub_block_m, block_n), "int8")
            reduce_tmp_ub = T.alloc_ub((2 * sub_block_m * block_n,), "uint8")

            def _cube_compute():
                loop_h = T.ceildiv(h_k, k_l1)
                loop_kk = T.ceildiv(k_l1, block_k)
                for by in T.serial(n_num):
                    group_id = by // n_tiles_per_h
                    col_in_group = by % n_tiles_per_h
                    for h_block in T.serial(loop_h):
                        T.copy(
                            x_in[
                                bx * block_m,
                                group_id * h_k + h_block * k_l1,
                            ],
                            a_l1,
                        )
                        T.copy(
                            h_mat[
                                h_block * k_l1,
                                col_in_group * block_n,
                            ],
                            b_l1,
                        )
                        for kk in T.serial(loop_kk):
                            T.copy(a_l1[0, kk * block_k], a_l0)
                            T.copy(b_l1[kk * block_k, 0], b_l0)
                            T.mma(
                                a_l0,
                                b_l0,
                                c_l0,
                                init=T.And(h_block == 0, kk == 0),
                            )
                    T.copy(c_l0, workspace[bx * block_m, by * block_n])
                T.set_cross_flag("FIX", 0)

            def _normalize_rows():
                for i in T.serial(sub_block_m):
                    T.tile.div(out_fp32_ub[i, :], out_fp32_ub[i, :], row_scale_ub[i])

            with T.Scope("C"):
                _cube_compute()

            def _vector_quantize_pass():
                T.tile.fill(row_absmax_ub, 0.0)
                for by in T.serial(n_num):
                    T.copy(
                        workspace[bx * block_m + vid * sub_block_m, by * block_n],
                        out_fp32_ub,
                    )
                    T.tile.abs(out_abs_ub, out_fp32_ub)
                    T.reduce_max(out_abs_ub, row_absmax_tile_ub, reduce_tmp_ub, dim=-1)
                    T.tile.max(row_absmax_ub, row_absmax_ub, row_absmax_tile_ub)

                T.tile.mul(row_scale_ub, row_absmax_ub, inv_127)

                for by in T.serial(n_num):
                    T.copy(
                        workspace[bx * block_m + vid * sub_block_m, by * block_n],
                        out_fp32_ub,
                    )
                    _normalize_rows()
                    T.tile.cast(
                        out_fp16_ub,
                        out_fp32_ub,
                        mode="CAST_NONE",
                        count=sub_block_m * block_n,
                    )
                    T.tile.cast(
                        out_int8_ub,
                        out_fp16_ub,
                        mode="CAST_NONE",
                        count=sub_block_m * block_n,
                    )
                    T.copy(
                        out_int8_ub,
                        y_out[bx * block_m + vid * sub_block_m, by * block_n],
                    )

            with T.Scope("V"):
                T.wait_cross_flag(0)
                _vector_quantize_pass()

    return main
