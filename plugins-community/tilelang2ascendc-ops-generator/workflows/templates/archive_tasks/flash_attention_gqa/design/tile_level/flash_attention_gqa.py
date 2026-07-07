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

# Template for: flash_attention_gqa
import sys
from pathlib import Path

import tilelang
from tilelang import DataType, language as T
from flash_attention_gqa_common import pass_configs, _kernel_setup, GQAConfig, \
    ProdWorkspace, _store_prod_workspace

_COMMON_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_COMMON_DIR)) if str(_COMMON_DIR) not in sys.path else None


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6, 7], pass_configs=pass_configs)
def flash_attention_gqa_fwd(cfg):
    c = _kernel_setup(cfg)
    block_q_packed, block_kv_seq, block_bkvh = c.block_q_packed, c.block_kv_seq, c.block_bkvh
    prelaunch, ring_slots, dtype, accum_dtype = c.prelaunch, c.ring_slots, c.dtype, c.accum_dtype
    group_size, packed_q_seq_len = c.group_size, c.packed_q_seq_len
    q_shape, kv_shape, output_shape = c.q_shape, c.kv_shape, c.output_shape
    total_bkvh, q_blocks_per_head = c.total_bkvh, c.q_blocks_per_head
    q_blocks, bkvh_blocks = c.q_blocks, c.bkvh_blocks
    block_num, kv_loops = c.block_num, c.kv_loops

    sm_scale = (1.0 / dim) ** 0.5

    @T.prim_func
    def main(*args):
        q, k, v, output, workspace_1, workspace_2, workspace_3, workspace_meta = args
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bkvh_block_id = cid

            q_l1 = T.alloc_L1([block_q_packed, dim], dtype)
            k_l1 = T.alloc_L1([block_kv_seq, dim], dtype)
            v_l1 = T.alloc_L1([block_kv_seq, dim], dtype)

            acc_s_l1 = T.alloc_L1([block_q_packed, block_kv_seq], dtype)

            acc_s_l0c = T.alloc_L0C([block_q_packed, block_kv_seq], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_q_packed, dim], accum_dtype)

            acc_o = T.alloc_ub([block_q_packed // 2, dim], accum_dtype)
            sumexp = T.alloc_ub([block_q_packed // 2], accum_dtype)
            m_i = T.alloc_ub([block_q_packed // 2], accum_dtype)

            acc_s_ub = T.alloc_ub([block_q_packed // 2, block_kv_seq], accum_dtype)
            m_i_prev = T.alloc_ub([block_q_packed // 2], accum_dtype)
            acc_s_ub_ = T.alloc_ub([block_q_packed // 2, block_kv_seq], accum_dtype)
            tmp_ub = T.alloc_ub(
                [3 * DataType(accum_dtype).bits // 8 * block_q_packed // 2 * block_kv_seq],
                "uint8",
            )
            sumexp_i_ub = T.alloc_ub([block_q_packed // 2], accum_dtype)
            acc_s_half = T.alloc_ub([block_q_packed // 2, block_kv_seq], dtype)
            acc_o_ub = T.alloc_ub([block_q_packed // 2, dim], accum_dtype)
            acc_o_half = T.alloc_ub([block_q_packed // 2, dim], dtype)
            alpha_ub = T.alloc_ub([block_q_packed // 2], accum_dtype)
            sumexp_meta_ub = T.alloc_ub([block_q_packed // 2], accum_dtype)

            def _process_producer_timestep(t, global_bkvh, cid, vid):
                slot_prod = t % ring_slots
                with T.Scope("C"):
                    T.copy(k[global_bkvh, t * block_kv_seq:(t + 1) * block_kv_seq, :], k_l1)
                    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                    T.copy(acc_s_l0c, workspace_1[cid, slot_prod, :, :])
                    T.set_cross_flag("FIX", 0)
                with T.Scope("V"):
                    T.wait_cross_flag(0)
                    T.tile.fill(acc_s_ub, 0.0)
                    T.copy(m_i, m_i_prev)
                    T.copy(
                        workspace_1[
                            cid,
                            slot_prod,
                            vid * block_q_packed // 2:vid * block_q_packed // 2 + block_q_packed // 2,
                            :,
                        ],
                        acc_s_ub_,
                    )
                    T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                    T.reduce_max(acc_s_ub, m_i, tmp_ub, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    T.tile.sub(m_i_prev, m_i_prev, m_i)
                    T.tile.exp(m_i_prev, m_i_prev)
                    for h_i in range(block_q_packed // 2):
                        T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
                    T.tile.exp(acc_s_ub, acc_s_ub)
                    T.reduce_sum(acc_s_ub, sumexp_i_ub, tmp_ub, dim=-1)
                    _store_prod_workspace(ProdWorkspace(
                        acc_s_ub, acc_s_half, workspace_2, workspace_meta,
                        m_i_prev, sumexp_i_ub, cid, vid, block_q_packed, slot_prod))

            def _process_consumer_timestep(t, global_bkvh, cid, vid):
                now_k = t - prelaunch
                slot_cons = now_k % ring_slots
                with T.Scope("C"):
                    T.wait_cross_flag(1)
                    T.copy(workspace_2[cid, slot_cons, :, :], acc_s_l1)
                    T.copy(
                        v[global_bkvh, now_k * block_kv_seq:(now_k + 1) * block_kv_seq, :],
                        v_l1,
                    )
                    T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                    T.copy(acc_o_l0c, workspace_3[cid, slot_cons, :, :])
                    T.set_cross_flag("FIX", 2)
                with T.Scope("V"):
                    T.wait_cross_flag(2)
                    for h_i in range(block_q_packed // 2):
                        alpha_ub[h_i] = workspace_meta[
                            cid,
                            slot_cons,
                            vid * block_q_packed // 2 + h_i,
                            0,
                        ]
                        sumexp_meta_ub[h_i] = workspace_meta[
                            cid,
                            slot_cons,
                            vid * block_q_packed // 2 + h_i,
                            1,
                        ]
                    T.copy(
                        workspace_3[
                            cid,
                            slot_cons,
                            vid * block_q_packed // 2:vid * block_q_packed // 2 + block_q_packed // 2,
                            :,
                        ],
                        acc_o_ub,
                    )
                    for h_i in range(block_q_packed // 2):
                        T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], alpha_ub[h_i])
                    T.tile.add(acc_o, acc_o, acc_o_ub)
                    T.tile.mul(sumexp, sumexp, alpha_ub)
                    T.tile.add(sumexp, sumexp, sumexp_meta_ub)

            def _process_epilogue(global_bkvh, q_block_idx, vid):
                with T.Scope("V"):
                    for h_i in range(block_q_packed // 2):
                        T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])
                    T.copy(acc_o, acc_o_half)
                    q_col_start = q_block_idx * block_q_packed + vid * block_q_packed // 2
                    q_col_end = q_col_start + block_q_packed // 2
                    T.copy(
                        acc_o_half,
                        output[global_bkvh, q_col_start:q_col_end, :],
                    )

            def _process_q_block(bx):
                bkvh_i = bx // q_blocks_per_head
                q_block_idx = bx % q_blocks_per_head
                global_bkvh = bkvh_block_id * block_bkvh + bkvh_i
                if global_bkvh < total_bkvh:
                    with T.Scope("C"):
                        T.copy(
                            q[
                                global_bkvh,
                                q_block_idx * block_q_packed:(q_block_idx + 1) * block_q_packed,
                                :,
                            ],
                            q_l1,
                        )
                    with T.Scope("V"):
                        T.tile.fill(acc_o, 0.0)
                        T.tile.fill(sumexp, 0.0)
                        T.tile.fill(m_i, -2**30)
                    for t in T.serial(kv_loops + prelaunch):
                        if t < kv_loops:
                            _process_producer_timestep(t, global_bkvh, cid, vid)
                        if t >= prelaunch:
                            _process_consumer_timestep(t, global_bkvh, cid, vid)
                    _process_epilogue(global_bkvh, q_block_idx, vid)

            for bx in T.serial(q_blocks):
                _process_q_block(bx)

    return main
