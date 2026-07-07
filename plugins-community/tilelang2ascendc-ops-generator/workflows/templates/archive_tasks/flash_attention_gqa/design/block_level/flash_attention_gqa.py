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

"""Block-level TileLang design for flash_attention_gqa.

This is the block-level decomposition reverse engineered from the existing
tile-level implementation. It keeps the same launch topology, packed-GQA data
layout, ring-buffer protocol, and stage boundaries, while intentionally leaving
the fine-grained math to the tile-level implementation.
"""

import sys
from pathlib import Path

import tilelang
import tilelang.language as T

_COMMON_DIR = Path(__file__).resolve().parent.parent
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))
from flash_attention_gqa_common import (  
    pass_configs,
    _kernel_setup,
    GQAConfig,
)


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

    @T.prim_func
    def main(*args):
        q, k, v, output, workspace_s, workspace_p, workspace_o, workspace_meta = args
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bkvh_block_id = cid

            _ = (q, k, v, output, workspace_s, workspace_p, workspace_o,
                 workspace_meta, bkvh_block_id, vid)  

            def _process_kv_producer(t):
                slot_prod = t % ring_slots
                with T.Scope("C"):
                    T.set_cross_flag("FIX", 0)
                with T.Scope("V"):
                    T.wait_cross_flag(0)
                    T.set_cross_flag("MTE3", 1)

            def _process_kv_consumer(t):
                now_k = t - prelaunch
                slot_cons = now_k % ring_slots
                with T.Scope("C"):
                    T.wait_cross_flag(1)
                    T.set_cross_flag("FIX", 2)
                with T.Scope("V"):
                    T.wait_cross_flag(2)

            def _process_q_block(bx):
                bkvh_i = bx // q_blocks_per_head
                q_block_idx = bx % q_blocks_per_head
                global_bkvh = bkvh_block_id * block_bkvh + bkvh_i
                if global_bkvh < total_bkvh:
                    with T.Scope("C"):
                        pass
                    with T.Scope("V"):
                        pass
                    for t in T.serial(kv_loops + prelaunch):
                        if t < kv_loops:
                            _process_kv_producer(t)
                        if t >= prelaunch:
                            _process_kv_consumer(t)
                    with T.Scope("V"):
                        _ = q_block_idx

            for bx in T.serial(q_blocks):
                _process_q_block(bx)

    return main
