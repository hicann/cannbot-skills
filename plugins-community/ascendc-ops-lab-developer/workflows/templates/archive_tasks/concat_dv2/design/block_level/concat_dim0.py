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

"""Block-level TileLang design for concat_dv2 limited to concat_dim == 0.

This design mirrors the effective semantics of the Ascend C implementation:
each input is treated as a 2D tensor of shape [dim0_i, sameDimSize], and the
output is formed by concatenating rows along dim0. Each block owns a contiguous
row tile of the output and resolves which input tensor supplies each row.
Fine-grained UB copy details are intentionally left to the tile-level file.
"""

import sys
from pathlib import Path

import tilelang
import tilelang.language as T

_COMMON_DIR = Path(__file__).resolve().parent.parent
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))
from concat_dim0_common import (  
    pass_configs,
    select_block_m,
)


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def concat_dim0_1(m0, n_dim, dtype="float32"):
    total_m = m0
    block_m = select_block_m(total_m)
    vec_num = 2
    sub_block_m = block_m // vec_num
    block_num = (total_m + block_m - 1) // block_m

    def _process_row(r, row_base, x0, y_out):
        row_idx = row_base + r
        if row_idx < total_m:
            # TODO(tile-level):
            # - load X0[row_idx, :] into UB
            # - store the row to Y[row_idx, :]
            _ = (x0, y_out, row_idx)

    @T.prim_func
    def main(
        x0: T.Tensor((m0, n_dim), dtype),
        y_out: T.Tensor((total_m, n_dim), dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_m + vid * sub_block_m

            with T.Scope("V"):
                for r in T.serial(sub_block_m):
                    _process_row(r, row_base, x0, y_out)

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def concat_dim0_2(m0, m1, n_dim, dtype="float32"):
    total_m = m0 + m1
    block_m = select_block_m(total_m)
    vec_num = 2
    sub_block_m = block_m // vec_num
    block_num = (total_m + block_m - 1) // block_m

    def _process_row(r, row_base, x0, x1, y_out):
        row_idx = row_base + r
        if row_idx < total_m:
            # TODO(tile-level):
            # - if row_idx < M0, read X0[row_idx, :]
            # - else read X1[row_idx - M0, :]
            # - store the selected row into Y[row_idx, :]
            _ = (x0, x1, y_out, row_idx)

    @T.prim_func
    def main(
        x0: T.Tensor((m0, n_dim), dtype),
        x1: T.Tensor((m1, n_dim), dtype),
        y_out: T.Tensor((total_m, n_dim), dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_m + vid * sub_block_m

            with T.Scope("V"):
                for r in T.serial(sub_block_m):
                    _process_row(r, row_base, x0, x1, y_out)

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def concat_dim0_3(m0, m1, m2, n_dim, dtype="float32"):
    total_m = m0 + m1 + m2
    block_m = select_block_m(total_m)
    vec_num = 2
    sub_block_m = block_m // vec_num
    block_num = (total_m + block_m - 1) // block_m

    def _process_row(r, row_base, inputs, y_out):
        x0, x1, x2 = inputs
        row_idx = row_base + r
        if row_idx < total_m:
            # TODO(tile-level):
            # - if row_idx < M0, read X0[row_idx, :]
            # - elif row_idx < M0 + M1, read X1[row_idx - M0, :]
            # - else read X2[row_idx - M0 - M1, :]
            # - store the selected row into Y[row_idx, :]
            _ = (x0, x1, x2, y_out, row_idx)

    @T.prim_func
    def main(
        # Block-level signature for 3-input concat (dim0).
        x0: T.Tensor((m0, n_dim), dtype),
        x1: T.Tensor((m1, n_dim), dtype),
        x2: T.Tensor((m2, n_dim), dtype),
        y_out: T.Tensor((total_m, n_dim), dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_m + vid * sub_block_m

            with T.Scope("V"):
                for r in T.serial(sub_block_m):
                    _process_row(r, row_base, (x0, x1, x2), y_out)

    return main


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def concat_dim0_4(*args):
    a = [*args, "float32"]
    m0, m1, m2, m3, n_dim, dtype = a[:6]
    total_m = m0 + m1 + m2 + m3
    block_m = select_block_m(total_m)
    vec_num = 2
    sub_block_m = block_m // vec_num
    block_num = (total_m + block_m - 1) // block_m

    def _process_row(r, row_base, inputs, y_out):
        x0, x1, x2, x3 = inputs
        row_idx = row_base + r
        if row_idx < total_m:
            # TODO(tile-level):
            # - if row_idx < M0, read X0[row_idx, :]
            # - elif row_idx < M0 + M1, read X1[row_idx - M0, :]
            # - elif row_idx < M0 + M1 + M2, read X2[row_idx - M0 - M1, :]
            # - else read X3[row_idx - M0 - M1 - M2, :]
            # - store the selected row into Y[row_idx, :]
            _ = (x0, x1, x2, x3, y_out, row_idx)

    @T.prim_func
    def main(
        # Block-level signature for 4-input concat (dim0).
        x0: T.Tensor((m0, n_dim), dtype),
        x1: T.Tensor((m1, n_dim), dtype),
        x2: T.Tensor((m2, n_dim), dtype),
        x3: T.Tensor((m3, n_dim), dtype),
        y_out: T.Tensor((total_m, n_dim), dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_m + vid * sub_block_m

            with T.Scope("V"):
                for r in T.serial(sub_block_m):
                    _process_row(r, row_base, (x0, x1, x2, x3), y_out)

    return main
