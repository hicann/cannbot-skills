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

# Template for: quant_matmul (tile-level)
import sys
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout

# Tile-level path setup differs from block-level to avoid duplicate detection.
_p = Path(__file__).resolve().parent.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
import int8_matmul_scale_common as _common  
pass_configs = _common.pass_configs
kernel_setup = _common.kernel_setup


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
    vec_num = 2
    sub_block_m = block_m // vec_num

    @T.prim_func
    def main(  
        a_mat: T.Tensor((m_size, k_size), dtype),
        b_mat: T.Tensor((k_size, n_size), dtype),
        scale: T.Tensor((n_size,), "float32"),
        c_mat: T.Tensor((m_size, n_size), "float16"),
        workspace: T.Tensor((m_size, n_size), accum_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_l1 = T.alloc_L1((block_m, k_l1), dtype)
            b_l1 = T.alloc_L1((k_l1, block_n), dtype)
            T.annotate_layout({
                a_l1: make_zn_layout(a_l1),
                b_l1: make_zn_layout(b_l1),
            })
            a_l0 = T.alloc_L0A((block_m, block_k), dtype)
            b_l0 = T.alloc_L0B((block_k, block_n), dtype)
            c_l0 = T.alloc_L0C((block_m, block_n), accum_dtype)

            acc_i32_ub = T.alloc_ub((sub_block_m, block_n), accum_dtype)
            acc_fp32_ub = T.alloc_ub((sub_block_m, block_n), "float32")
            out_ub = T.alloc_ub((sub_block_m, block_n), "float16")
            scale_row_ub = T.alloc_ub((1, block_n), "float32")
            row_fp32_ub = T.alloc_ub((1, block_n), "float32")

            def _cube_compute():
                loop_k = T.ceildiv(k_size, k_l1)
                loop_kk = T.ceildiv(k_l1, block_k)
                for k in T.serial(loop_k):
                    T.copy(a_mat[bx * block_m, k * k_l1], a_l1)
                    T.copy(b_mat[k * k_l1, by * block_n], b_l1)
                    for kk in T.serial(loop_kk):
                        T.copy(a_l1[0, kk * block_k], a_l0)
                        T.copy(b_l1[kk * block_k, 0], b_l0)
                        T.mma(
                            a_l0,
                            b_l0,
                            c_l0,
                            init=T.And(k == 0, kk == 0),
                        )
                T.copy(c_l0, workspace[bx * block_m, by * block_n])
                T.set_cross_flag("FIX", 0)

            with T.Scope("C"):
                _cube_compute()

            with T.Scope("V"):
                T.wait_cross_flag(0)
                T.copy(
                    workspace[bx * block_m + vid * sub_block_m, by * block_n],
                    acc_i32_ub,
                )
                T.copy(scale[by * block_n], scale_row_ub)
                T.tile.cast(
                    acc_fp32_ub,
                    acc_i32_ub,
                    mode="CAST_NONE",
                    count=sub_block_m * block_n,
                )
                for i in T.serial(sub_block_m):
                    T.copy(acc_fp32_ub[i, 0], row_fp32_ub)
                    T.tile.mul(row_fp32_ub, row_fp32_ub, scale_row_ub)
                    T.copy(row_fp32_ub, acc_fp32_ub[i, 0])
                T.tile.cast(
                    out_ub,
                    acc_fp32_ub,
                    mode="CAST_NONE",
                    count=sub_block_m * block_n,
                )
                T.copy(
                    out_ub,
                    c_mat[bx * block_m + vid * sub_block_m, by * block_n],
                )

    return main
