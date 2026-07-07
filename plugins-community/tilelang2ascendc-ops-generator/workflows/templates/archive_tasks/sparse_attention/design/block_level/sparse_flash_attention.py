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

"""Block-level TileLang design for sparse_flash_attention.

This block-level decomposition keeps the irregular sparse gather inside the
kernel, but places it on the Vector side. Each block first uses sparse indices
to gather K/V rows into dense UB tiles, then fuses:

1. Q @ K_selected^T
2. scaled softmax across the sparse token axis
3. P @ V_selected

The file intentionally leaves tile math as TODO(tile-level) while fixing the
block ownership, workspace contract, and C/V synchronization skeleton.
"""

import sys
from pathlib import Path

import tilelang
import tilelang.language as T

_COMMON_DIR = Path(__file__).resolve().parent.parent
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))
from sparse_flash_attention_common import (  
    pass_configs,
    kernel_setup,
    SparseAttnConfig,
)


@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7, 8, 9], pass_configs=pass_configs)
def sparse_flash_attention_fwd(cfg):
    # Block-level kernel configuration extraction.
    c = kernel_setup(cfg)
    group_size, block_group, block_sparse = c.group_size, c.block_group, c.block_sparse
    dtype, accum_dtype = c.dtype, c.accum_dtype
    total_bkvh, block_num = c.total_bkvh, c.block_num
    q_shape, kv_shape = c.q_shape, c.kv_shape
    sparse_index_shape, output_shape = c.sparse_index_shape, c.output_shape

    @T.prim_func
    def main(*args):
        (q, k, v, sparse_index, output,
         workspace_scores, workspace_probs, workspace_out, workspace_k, workspace_v) = args
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            # Block-level coordinate decomposition.
            global_bkvh = cid // q_seq_len
            q_idx = cid % q_seq_len
            subgroup = block_group // 2
            subgroup_begin = vid * subgroup

            _ = (q, k, v, sparse_index, output,
                 workspace_scores, workspace_probs, workspace_out, workspace_k, workspace_v,
                 global_bkvh, q_idx, subgroup_begin)  

            with T.Scope("C"):
                # TODO(tile-level):
                # - load the dense padded Q tile for (global_bkvh, q_idx)
                # - wait until workspace_k[cid, :, :] is ready from the V side
                # - load the gathered dense K tile from workspace_k into L1
                # - store scores into workspace_scores[cid, :, :]
                # - signal the Vector softmax/gather-V stage
                T.wait_cross_flag(0)
                T.set_cross_flag("FIX", 1)

                # TODO(tile-level):
                # - wait until workspace_probs[cid, :, :] and workspace_v are ready
                # - load the dense V tile from workspace_v into L1
                # - compute workspace_out[cid, :, :] = P @ V_selected
                # - signal the Vector writeback stage
                T.wait_cross_flag(2)
                T.set_cross_flag("FIX", 3)

            with T.Scope("V"):
                # Vector side works on half of the padded group rows per `vid`,
                # covering the range [subgroup_begin, subgroup_begin + subgroup).

                # TODO(tile-level):
                # - use SparseIndex[global_bkvh, q_idx, :] to gather K rows
                #   from GM into a dense UB tile via row-wise T.copy
                # - zero-fill padded rows
                # - write the gathered tile to workspace_k[cid, :, :]
                # - signal the Cube QK stage
                T.set_cross_flag("MTE3", 0)

                # TODO(tile-level):
                # - wait for workspace_scores[cid, :, :] to become ready
                # - apply scale and mask padded sparse columns
                # - compute softmax along the sparse axis
                # - write probabilities into workspace_probs
                # - gather V rows into a dense UB tile and write workspace_v
                # - signal the Cube PV stage
                T.wait_cross_flag(1)
                T.set_cross_flag("MTE3", 2)

                # TODO(tile-level):
                # - wait for workspace_out[cid, :, :] to become ready
                # - cast and write the owned subgroup rows into Output
                T.wait_cross_flag(3)

    return main
