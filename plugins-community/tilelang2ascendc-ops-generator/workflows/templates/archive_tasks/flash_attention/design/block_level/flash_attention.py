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

"""Block-level TileLang design for flash_attention.

This file captures task decomposition, stage boundaries, ring-buffer ownership,
and cross-scope synchronization. Fine-grained math is intentionally left as
TODOs and should be filled in by the tile-level design.
"""

from collections import namedtuple

import tilelang
import tilelang.language as T


# Container for the four workspace tensors to keep T.prim_func signature lean.
FAWorkspaces = namedtuple("FAWorkspaces", ["ws_s", "ws_p", "ws_o", "ws_meta"])

# Pipeline configuration — groups ring-buffer parameters shared across
# producer / consumer helpers (G.FNM.03 compliance).
PipelineConfig = namedtuple("PipelineConfig", ["prelaunch", "ring_slots", "kv_loops"])


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


# ---------------------------------------------------------------------------
#  Loop-body helpers — one call per pipeline iteration inside the t-loop.
#  Keeping the per-iteration logic in helper functions keeps the main()
#  nesting depth ≤ 4 (T.Kernel → for → if → helper).
# ---------------------------------------------------------------------------

def _kv_loop_producer(t, slot_prod, cid, cfg: PipelineConfig):
    """Pipeline producer stage: C1(t) and V1(t) for one kv-step.

    Args:
        t: current kv-step index.
        slot_prod: pre-computed ring-buffer slot for this production step.
        cid: core id.
        cfg: pipeline configuration (prelaunch, ring_slots, kv_loops).
    """
    with T.Scope("C"):
        # TODO(tile-level, C1):
        # - load k tile for kv-step t
        # - compute S = q @ k^T for current q-block
        # - store S into workspace_s[cid, slot_prod, ...]
        # - signal that C1(t) is ready for V1(t)
        T.set_cross_flag("FIX", 0)

    with T.Scope("v"):
        # TODO(tile-level, V1):
        # - wait until C1(t) has produced workspace_s[cid, slot_prod, ...]
        # - apply scaling and online softmax update on current half-tile
        # - store P into workspace_p[cid, slot_prod, ...]
        # - store merge metadata into workspace_meta[cid, slot_prod, ...]
        # - signal that V1(t) is ready for C2(t)
        T.wait_cross_flag(0)
        T.set_cross_flag("MTE3", 1)


def _kv_loop_consumer(now_k, slot_cons, cid, cfg: PipelineConfig):
    """Pipeline consumer stage: C2(now_k) and V2(now_k) for one kv-step.

    Args:
        now_k: the kv-step whose results are now ready for consumption.
        slot_cons: pre-computed ring-buffer slot for consumption.
        cid: core id.
        cfg: pipeline configuration (prelaunch, ring_slots, kv_loops).
    """
    with T.Scope("C"):
        # TODO(tile-level, C2):
        """
        # - wait until V1(now_k) has produced workspace_p[cid, slot_cons, ...]
        # - load v tile for now_k
        # - compute O_tmp = P @ v
        # - store O_tmp into workspace_o[cid, slot_cons, ...]
        # - signal that C2(now_k) is ready for V2(now_k)
        """        
        T.wait_cross_flag(1)
        T.set_cross_flag("FIX", 2)

    with T.Scope("v"):
        # TODO(tile-level, V2):
        # - wait until C2(now_k) has produced workspace_o[cid, slot_cons, ...]
        # - load alpha / local sumexp produced by V1(now_k)
        # - rescale previous accumulator and merge O_tmp
        # - update the running output/sumexp state for this half tile
        T.wait_cross_flag(2)


# ---------------------------------------------------------------------------
#  JIT-compiled entry point
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[3], workspace_idx=[4], pass_configs=pass_configs)
def flash_attention_fwd(
    batch,
    seq_len,
    heads,
    dim,
):
    block_m, block_n = 64, 64
    prelaunch = 2
    ring_slots = prelaunch + 1

    dtype = "float16"
    accum_dtype = "float"

    shape = [batch, heads, seq_len, dim]
    block_num = seq_len // block_m * heads * batch
    kv_loops = T.ceildiv(seq_len, block_n)

    pipeline_cfg = PipelineConfig(prelaunch, ring_slots, kv_loops)

    @T.prim_func
    def main(
        q: T.Tensor(shape, dtype),
        k: T.Tensor(shape, dtype),
        v: T.Tensor(shape, dtype),
        output: T.Tensor(shape, dtype),
        workspaces: FAWorkspaces,
    ):
        workspace_s = workspaces.ws_s
        workspace_p = workspaces.ws_p
        workspace_o = workspaces.ws_o
        workspace_meta = workspaces.ws_meta

        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_m)
            by = cid // (seq_len // block_m) % heads
            bz = cid // (seq_len // block_m) // heads % batch

            # Block-level task partition:
            """
              Each kernel instance handles one (batch, head, q-block) tile:
              q[bz, by, bx * block_m:(bx + 1) * block_m, :]

            Ring pipeline timeline for prelaunch=2:
              t=0: C1(0) / V1(0)
              t=1: C1(1) / V1(1)
              t=2: C1(2) + C2(0) / V1(2) + V2(0)
              t=3: C1(3) + C2(1) / V1(3) + V2(1)
              ...
              tail: C2(...) / V2(...)
            """

            with T.Scope("C"):
                # TODO(tile-level, C0):
                # - preload the current q-block into on-chip Cube-side memory
                # - keep it resident across the whole C1/C2 pipeline
                pass

            with T.Scope("v"):
                # TODO(tile-level, V0):
                # - initialize the running output accumulator, running max, and
                #   running sumexp for this vid-owned half tile before V1/V2
                pass

            for t in T.serial(kv_loops + prelaunch):
                if t < kv_loops:
                    _kv_loop_producer(t, t % ring_slots, cid, pipeline_cfg)

                if t >= prelaunch:
                    now_k = t - prelaunch
                    _kv_loop_consumer(now_k, now_k % ring_slots, cid,
                                      pipeline_cfg)

            with T.Scope("v"):
                # TODO(tile-level, V2-epilogue):
                # - normalize the running accumulator by the final sumexp
                # - cast and write the final output half tile to output
                pass

    return main
