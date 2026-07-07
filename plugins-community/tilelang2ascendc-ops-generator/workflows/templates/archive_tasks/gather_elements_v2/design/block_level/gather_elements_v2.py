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
from pathlib import Path

import tilelang
import tilelang.language as T

_COMMON_DIR = Path(__file__).resolve().parent.parent
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))
from gather_elements_v2_common import (  
    pass_configs,
    _kernel_setup,
)


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def gather_elements_v2(m_size, x_g, i_g, dtype="float32"):
    c = _kernel_setup(m_size)
    block_m, m_num, used_core_num = c.block_m, c.m_num, c.used_core_num
    tasks_per_core, sub_block_m = c.tasks_per_core, c.sub_block_m

    @T.prim_func
    def single_row(
        x_in: T.Tensor((m_size * x_g,), dtype),
        index_in: T.Tensor((m_size * i_g,), "int32"),
        y_out: T.Tensor((m_size * i_g,), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            def _process_rows_for_bx(bx, vid):
                for row in T.serial(sub_block_m):
                    row_idx = bx * block_m + vid * sub_block_m + row
                    if row_idx < m_size:
                        x_base = row_idx * x_g
                        y_base = row_idx * i_g
                        _ = (x_in, index_in, y_out, x_base, y_base)

            def _process_block(bx, vid):
                if bx < m_num:
                    _process_rows_for_bx(bx, vid)

            with T.Scope("V"):
                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    _process_block(bx, vid)

    return single_row
