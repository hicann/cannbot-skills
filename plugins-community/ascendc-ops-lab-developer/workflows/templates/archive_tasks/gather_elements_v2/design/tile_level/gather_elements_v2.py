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

import sys
from dataclasses import dataclass
from pathlib import Path

import tilelang
import tilelang.language as T

_COMMON_DIR = Path(__file__).resolve().parent.parent
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

from gather_elements_v2_common import (  
    pass_configs,
    _kernel_setup,
    _gather_cols,
    RowProcessCtx,
    _process_rows_for_bx,
)


# NOTE: The two @T.prim_func functions below (last_dim_kernel, indexed_kernel)
# share the same pipeline boilerplate (Kernel setup, UB allocation, gather
# skeleton) except for the RowMap indirection in indexed_kernel.  The common
# helpers (_gather_cols, _process_rows_for_bx) are defined in
# gather_elements_v2_common and called from each prim_func.


@dataclass
class GatherConfig:
    m_size: int
    x_rows: int
    x_g: int
    i_g: int
    x_stride: int
    y_stride: int
    mode: str = "last_dim"
    dtype: str = "float32"


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def gather_elements_v2(cfg):
    m_size, x_rows, x_g, i_g, x_stride, y_stride, mode, dtype = (
        cfg.m_size, cfg.x_rows, cfg.x_g, cfg.i_g, cfg.x_stride, cfg.y_stride, cfg.mode, cfg.dtype
    )
    c = _kernel_setup(m_size)
    block_m, m_num, used_core_num = c.block_m, c.m_num, c.used_core_num
    tasks_per_core, sub_block_m = c.tasks_per_core, c.sub_block_m

    def _gather_scope_body(*args):
        cid, vid, x_ub, index_ub, out_ub, x_in, index_in, y_out, row_map = args
        for local_idx in T.serial(tasks_per_core):
            bx = cid * tasks_per_core + local_idx
            if bx < m_num:
                _process_rows_for_bx(RowProcessCtx(
                    bx, vid, sub_block_m, block_m, m_size,
                    x_in, index_in, y_out, i_g,
                    x_ub, index_ub, out_ub, row_map=row_map,
                ))

    @T.prim_func
    def last_dim_kernel(
        x_in: T.Tensor((m_size, x_stride), dtype),
        index_in: T.Tensor((m_size, y_stride), "int32"),
        y_out: T.Tensor((m_size, y_stride), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((1, x_stride), dtype)
            index_ub = T.alloc_ub((1, y_stride), "int32")
            out_ub = T.alloc_ub((1, y_stride), dtype)

            with T.Scope("V"):
                _gather_scope_body(cid, vid, x_ub, index_ub, out_ub,
                                   x_in, index_in, y_out, None)

    @T.prim_func
    def indexed_kernel(
        x_in: T.Tensor((x_rows, x_stride), dtype),
        index_in: T.Tensor((m_size, y_stride), "int32"),
        row_map: T.Tensor((m_size,), "int32"),
        y_out: T.Tensor((m_size, y_stride), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((1, x_stride), dtype)
            index_ub = T.alloc_ub((1, y_stride), "int32")
            out_ub = T.alloc_ub((1, y_stride), dtype)

            with T.Scope("V"):
                _gather_scope_body(cid, vid, x_ub, index_ub, out_ub,
                                   x_in, index_in, y_out, row_map)

    if mode == "last_dim":
        return last_dim_kernel
    # transpose mode uses the same indexed kernel logic
    return indexed_kernel
