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

# Template for: sparse_attention (tile-level)
import sys
from pathlib import Path

import tilelang
from tilelang import DataType, language as T

# Tile-level path setup differs from block-level.
_p = Path(__file__).resolve().parent.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
import sparse_flash_attention_common as _common  
pass_configs = _common.pass_configs
kernel_setup = _common.kernel_setup
SparseAttnConfig = _common.SparseAttnConfig


@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7, 8, 9], pass_configs=pass_configs)
def sparse_flash_attention_fwd(cfg):
    c = kernel_setup(cfg)
    group_size, block_group, block_sparse = c.group_size, c.block_group, c.block_sparse
    dtype, accum_dtype = c.dtype, c.accum_dtype
    total_bkvh, block_num = c.total_bkvh, c.block_num
    q_shape, kv_shape = c.q_shape, c.kv_shape
    sparse_index_shape, output_shape = c.sparse_index_shape, c.output_shape

    sm_scale = (1.0 / dim) ** 0.5
    subgroup = block_group // 2

    @T.prim_func
    def main(*args):
        (q, k, v, sparse_index, output,
         workspace_scores, workspace_probs, workspace_out, workspace_k, workspace_v) = args
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            global_bkvh = cid // q_seq_len
            q_idx = cid % q_seq_len
            subgroup_begin = vid * subgroup

            q_l1 = T.alloc_L1([block_group, dim], dtype)
            k_l1 = T.alloc_L1([block_sparse, dim], dtype)
            v_l1 = T.alloc_L1([block_sparse, dim], dtype)
            prob_l1 = T.alloc_L1([block_group, block_sparse], dtype)

            scores_l0c = T.alloc_L0C([block_group, block_sparse], accum_dtype)
            out_l0c = T.alloc_L0C([block_group, dim], accum_dtype)

            selected_k_ub = T.alloc_ub([block_sparse, dim], dtype)
            selected_v_ub = T.alloc_ub([block_sparse, dim], dtype)
            scores_ub = T.alloc_ub([subgroup, block_sparse], accum_dtype)
            probs_ub = T.alloc_ub([subgroup, block_sparse], dtype)
            out_ub = T.alloc_ub([subgroup, dim], accum_dtype)
            out_half = T.alloc_ub([subgroup, dim], dtype)
            row_max = T.alloc_ub([subgroup], accum_dtype)
            row_sum = T.alloc_ub([subgroup], accum_dtype)
            reduce_tmp = T.alloc_ub(
                [3 * DataType(accum_dtype).bits // 8 * subgroup * block_sparse],
                "uint8",
            )

            with T.Scope("C"):
                T.wait_cross_flag(0)
                T.copy(q[global_bkvh, q_idx, :, :], q_l1)
                T.copy(workspace_k[cid, :, :], k_l1)
                T.gemm_v0(q_l1, k_l1, scores_l0c, transpose_B=True, init=True)
                T.copy(scores_l0c, workspace_scores[cid, :, :])
                T.set_cross_flag("FIX", 1)

                T.wait_cross_flag(2)
                T.copy(workspace_probs[cid, :, :], prob_l1)
                T.copy(workspace_v[cid, :, :], v_l1)
                T.gemm_v0(prob_l1, v_l1, out_l0c, init=True)
                T.copy(out_l0c, workspace_out[cid, :, :])
                T.set_cross_flag("FIX", 3)

            def _gather_k_to_workspace():
                T.tile.fill(selected_k_ub, 0.0)
                for sparse_i in range(sparse_size):
                    token_idx = sparse_index[global_bkvh, q_idx, sparse_i]
                    T.copy(k[global_bkvh, token_idx, :], selected_k_ub[sparse_i, :])
                T.copy(selected_k_ub, workspace_k[cid, :, :])
                T.set_cross_flag("MTE3", 0)

            def _compute_softmax_and_store():
                T.wait_cross_flag(1)
                T.copy(
                    workspace_scores[
                        cid,
                        subgroup_begin:subgroup_begin + subgroup,
                        :,
                    ],
                    scores_ub,
                )
                T.tile.mul(scores_ub, scores_ub, sm_scale)

                def _apply_sparse_mask():
                    for row_i in range(subgroup):
                        for sparse_i in range(sparse_size, block_sparse):
                            scores_ub[row_i, sparse_i] = -2**30

                _apply_sparse_mask()
                T.reduce_max(scores_ub, row_max, reduce_tmp, dim=-1)
                for row_i in range(subgroup):
                    T.tile.sub(scores_ub[row_i, :], scores_ub[row_i, :], row_max[row_i])
                T.tile.exp(scores_ub, scores_ub)
                T.reduce_sum(scores_ub, row_sum, reduce_tmp, dim=-1)
                for row_i in range(subgroup):
                    T.tile.div(scores_ub[row_i, :], scores_ub[row_i, :], row_sum[row_i])
                T.copy(scores_ub, probs_ub)
                T.copy(
                    probs_ub,
                    workspace_probs[
                        cid,
                        subgroup_begin:subgroup_begin + subgroup,
                        :,
                    ],
                )

            def _gather_v_and_write_output():
                T.tile.fill(selected_v_ub, 0.0)
                for sparse_i in range(sparse_size):
                    token_idx = sparse_index[global_bkvh, q_idx, sparse_i]
                    T.copy(v[global_bkvh, token_idx, :], selected_v_ub[sparse_i, :])
                T.copy(selected_v_ub, workspace_v[cid, :, :])
                T.set_cross_flag("MTE3", 2)

                T.wait_cross_flag(3)
                T.copy(
                    workspace_out[
                        cid,
                        subgroup_begin:subgroup_begin + subgroup,
                        :,
                    ],
                    out_ub,
                )
                T.copy(out_ub, out_half)
                T.copy(
                    out_half,
                    output[
                        global_bkvh,
                        q_idx,
                        subgroup_begin:subgroup_begin + subgroup,
                        :,
                    ],
                )

            with T.Scope("V"):
                _gather_k_to_workspace()
                _compute_softmax_and_store()
                _gather_v_and_write_output()

    return main
