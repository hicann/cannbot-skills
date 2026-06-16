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

from collections import namedtuple
from dataclasses import dataclass

import tilelang
from tilelang import DataType, language as T
from tilelang.intrinsics import make_zn_layout
from tvm.tir import BufferRegion
from tvm.ir import Range


@dataclass
class FlashAttentionConfig:
    """Configuration encapsulating FlashAttention kernel shapes and data type."""
    batch: int
    heads: int
    q_seq_len: int
    kv_seq_len: int
    dim: int
    dtype: str = "float16"


# Container for the four workspace tensors to keep T.prim_func signature lean.
FAWorkspaces = namedtuple("FAWorkspaces", ["ws_s", "ws_p", "ws_o", "ws_meta"])


# ---------------------------------------------------------------------------
#  Parameter-group containers — encapsulate correlated arguments so that
#  tile-level helper signatures stay ≤ 5 (G.FNM.03 compliance).
# ---------------------------------------------------------------------------

@dataclass
class TileLoopContext:
    """Loop iteration context identifying the current tile position."""
    bx: int
    by: int
    bz: int
    task_idx: int
    vid: int
    t: int  # KV-loop iteration (t for producer stages, now_k for consumer stages)


TileShapeInfo = namedtuple("TileShapeInfo", [
    "block_m", "block_n", "dim", "cube_k", "qk_k_loops", "ring_slots",
    "kv_seq_len_padded", "kv_loops", "q_seq_len", "heads",
    "sm_scale", "tail_valid", "prelaunch",
])

C1Bufs = namedtuple("C1Bufs", ["q_l1", "k_l1", "lhs_l0", "rhs_l0", "acc_s_l0c"])
C2Bufs = namedtuple("C2Bufs", ["acc_s_l1", "lhs_l0", "rhs_l0", "acc_o_l0c", "v_l1"])
V1Bufs = namedtuple("V1Bufs", ["acc_s_ub", "acc_s_ub_", "m_i", "m_i_prev",
                                "sumexp_i_ub", "acc_s_half", "mask_col"])
V1Wss = namedtuple("V1Wss", ["ws_s", "ws_p", "ws_meta"])
V2Bufs = namedtuple("V2Bufs", ["acc_o", "sumexp", "acc_o_ub", "alpha_ub", "sumexp_meta_ub"])
OutBufs = namedtuple("OutBufs", ["acc_o", "sumexp", "acc_o_half"])


@dataclass
class TaskContext:
    """Per-task context identifying the tile being processed."""
    bx: int
    by: int
    bz: int
    task_idx: int
    vid: int


TaskBufs = namedtuple("TaskBufs", [
    "c1", "c2", "v1", "v1_ws", "v2", "out",
    "ws1", "ws2", "ws3", "ws_meta",
])

TileBufs = namedtuple("TileBufs", [
    "q_l1", "k_l1", "v_l1", "acc_s_l1",
    "lhs_l0", "rhs_l0", "acc_s_l0c", "acc_o_l0c",
    "acc_o", "sumexp", "m_i",
    "acc_s_ub", "m_i_prev", "acc_s_ub_", "sumexp_i_ub",
    "acc_s_half", "acc_o_ub", "acc_o_half",
    "alpha_ub", "sumexp_meta_ub", "mask_col",
])


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


# ---------------------------------------------------------------------------
#  Tile-level helpers — each performs one well-defined sub-step of the
#  FlashAttention pipeline.  They are called from inside the T.prim_func
#  so that all T.* / T.tile.* calls are traced correctly.
# ---------------------------------------------------------------------------

def _init_acc_state(acc_o, sumexp, m_i):
    """Initialize running accumulator, sumexp, and max to starting values."""
    T.tile.fill(acc_o, 0.0)
    T.tile.fill(sumexp, 0.0)
    T.tile.fill(m_i, -2 ** 30)


def _c1_load_k_compute_s(ctx: TileLoopContext, shape: TileShapeInfo,
                        bufs: C1Bufs, workspace_1):
    """C1: Load K tile from GM and compute S = Q @ K^T, store to workspace."""
    slot_prod = ctx.t % shape.ring_slots
    with T.Scope("C"):
        T.copy(k[ctx.bz, ctx.by, ctx.t * shape.block_n:
                 (ctx.t + 1) * shape.block_n, :], bufs.k_l1)
        for kk in T.serial(shape.qk_k_loops):
            T.copy(bufs.q_l1[0, kk * shape.cube_k], bufs.lhs_l0)
            T.copy(bufs.k_l1[0, kk * shape.cube_k], bufs.rhs_l0, transpose=True)
            T.mma(bufs.lhs_l0, bufs.rhs_l0, bufs.acc_s_l0c, init=(kk == 0))
        T.copy(bufs.acc_s_l0c, workspace_1[ctx.task_idx, slot_prod, :, :])
        T.set_cross_flag("FIX", 0)


def _v1_softmax_update(ctx: TileLoopContext, shape: TileShapeInfo,
                        bufs: V1Bufs, wss: V1Wss):
    """V1: Load S, apply scale/mask/softmax, store P and merge metadata."""
    slot_prod = ctx.t % shape.ring_slots
    h_start = ctx.vid * shape.block_m // 2
    h_slice = slice(h_start, h_start + shape.block_m // 2)

    with T.Scope("v"):
        T.tile.fill(bufs.acc_s_ub, 0.0)
        T.copy(bufs.m_i, bufs.m_i_prev)
        T.wait_cross_flag(0)
        T.copy(wss.ws_s[ctx.task_idx, slot_prod, h_slice, :], bufs.acc_s_ub_)
        T.tile.add(bufs.acc_s_ub, bufs.acc_s_ub, bufs.acc_s_ub_)
        T.tile.mul(bufs.acc_s_ub, bufs.acc_s_ub, shape.sm_scale)

        # mask columns beyond kv_seq_len in the tail KV tile
        if shape.tail_valid != 0 and ctx.t == shape.kv_loops - 1:
            T.tile.fill(bufs.mask_col, T.float32(-2 ** 30))
            T.tile.fill(
                BufferRegion(bufs.mask_col, [Range(0, shape.tail_valid)]),
                T.float32(0.0),
            )
            for h_i in range(shape.block_m // 2):
                T.tile.add(bufs.acc_s_ub[h_i, :], bufs.acc_s_ub[h_i, :],
                           bufs.mask_col)

        T.reduce_max(bufs.acc_s_ub, bufs.m_i, dim=-1)
        T.tile.max(bufs.m_i, bufs.m_i, bufs.m_i_prev)
        T.tile.sub(bufs.m_i_prev, bufs.m_i_prev, bufs.m_i)
        T.tile.exp(bufs.m_i_prev, bufs.m_i_prev)
        for h_i in range(shape.block_m // 2):
            T.tile.sub(bufs.acc_s_ub[h_i, :], bufs.acc_s_ub[h_i, :],
                       bufs.m_i[h_i])
        T.tile.exp(bufs.acc_s_ub, bufs.acc_s_ub)
        T.reduce_sum(bufs.acc_s_ub, bufs.sumexp_i_ub, dim=-1)

        T.copy(bufs.acc_s_ub, bufs.acc_s_half)
        T.copy(bufs.acc_s_half, wss.ws_p[ctx.task_idx, slot_prod, h_slice, :])
        for h_i in range(shape.block_m // 2):
            row = h_start + h_i
            wss.ws_meta[ctx.task_idx, slot_prod, row, 0] = bufs.m_i_prev[h_i]
            wss.ws_meta[ctx.task_idx, slot_prod, row, 1] = bufs.sumexp_i_ub[h_i]
        T.set_cross_flag("MTE3", 1)


def _c2_compute_pv(ctx: TileLoopContext, shape: TileShapeInfo,
                    bufs: C2Bufs, workspace_2, workspace_3):
    slot_cons = ctx.t % shape.ring_slots
    with T.Scope("C"):
        T.wait_cross_flag(1)
        T.copy(workspace_2[ctx.task_idx, slot_cons, :, :], bufs.acc_s_l1)
        T.copy(
            v[ctx.bz, ctx.by,
              ctx.t * shape.block_n: (ctx.t + 1) * shape.block_n, :],
            bufs.v_l1,
        )
        for dd in T.serial(shape.qk_k_loops):
            T.copy(bufs.acc_s_l1, bufs.lhs_l0)
            T.copy(bufs.v_l1[0, dd * shape.cube_k], bufs.rhs_l0)
            T.mma(bufs.lhs_l0, bufs.rhs_l0, bufs.acc_o_l0c, init=True)
            T.copy(
                bufs.acc_o_l0c,
                workspace_3[
                    ctx.task_idx, slot_cons, :,
                    dd * shape.cube_k: (dd + 1) * shape.cube_k,
                ],
            )
        T.set_cross_flag("FIX", 2)


def _v2_merge_output(ctx: TileLoopContext, shape: TileShapeInfo,
                      bufs: V2Bufs, workspace_3, workspace_meta):
    """V2: Rescale previous accumulator and merge new O_tmp."""
    slot_cons = ctx.t % shape.ring_slots
    h_start = ctx.vid * shape.block_m // 2
    h_slice = slice(h_start, h_start + shape.block_m // 2)

    with T.Scope("v"):
        T.wait_cross_flag(2)
        for h_i in range(shape.block_m // 2):
            row = h_start + h_i
            bufs.alpha_ub[h_i] = workspace_meta[
                ctx.task_idx, slot_cons, row, 0]
            bufs.sumexp_meta_ub[h_i] = workspace_meta[
                ctx.task_idx, slot_cons, row, 1]
        T.copy(workspace_3[ctx.task_idx, slot_cons, h_slice, :], bufs.acc_o_ub)
        for h_i in range(shape.block_m // 2):
            T.tile.mul(bufs.acc_o[h_i, :], bufs.acc_o[h_i, :],
                       bufs.alpha_ub[h_i])
        T.tile.add(bufs.acc_o, bufs.acc_o, bufs.acc_o_ub)
        T.tile.mul(bufs.sumexp, bufs.sumexp, bufs.alpha_ub)
        T.tile.add(bufs.sumexp, bufs.sumexp, bufs.sumexp_meta_ub)


def _finalize_output(ctx: TileLoopContext, shape: TileShapeInfo,
                      bufs: OutBufs, output):
    """Normalize accumulator by sumexp and write the final output half-tile."""
    with T.Scope("v"):
        for h_i in range(shape.block_m // 2):
            T.tile.div(bufs.acc_o[h_i, :], bufs.acc_o[h_i, :],
                       bufs.sumexp[h_i])
        T.copy(bufs.acc_o, bufs.acc_o_half)
        row_start = ctx.bx * shape.block_m + ctx.vid * shape.block_m // 2
        if row_start < shape.q_seq_len:
            T.copy(
                bufs.acc_o_half,
                output[ctx.bz, ctx.by,
                       row_start: row_start + shape.block_m // 2, :],
            )


def _allocate_tile_buffers(shape: TileShapeInfo, dtype: str, accum_dtype: str):
    """Allocate and return all on-chip buffers for one core."""
    q_l1 = T.alloc_L1([shape.block_m, shape.dim], dtype)
    k_l1 = T.alloc_L1([shape.block_n, shape.dim], dtype)
    v_l1 = T.alloc_L1([shape.block_n, shape.dim], dtype)
    acc_s_l1 = T.alloc_L1([shape.block_m, shape.block_n], dtype)
    T.annotate_layout({
        q_l1: make_zn_layout(q_l1),
        k_l1: make_zn_layout(k_l1),
        v_l1: make_zn_layout(v_l1),
        acc_s_l1: make_zn_layout(acc_s_l1),
    })

    lhs_l0 = T.alloc_L0A([shape.block_m, shape.cube_k], dtype)
    rhs_l0 = T.alloc_L0B([shape.cube_k, shape.block_n], dtype)
    acc_s_l0c = T.alloc_L0C([shape.block_m, shape.block_n], accum_dtype)
    acc_o_l0c = T.alloc_L0C([shape.block_m, shape.block_n], accum_dtype)

    acc_o = T.alloc_ub([shape.block_m // 2, shape.dim], accum_dtype)
    sumexp = T.alloc_ub([shape.block_m // 2], accum_dtype)
    m_i = T.alloc_ub([shape.block_m // 2], accum_dtype)

    acc_s_ub = T.alloc_ub([shape.block_m // 2, shape.block_n], accum_dtype)
    m_i_prev = T.alloc_ub([shape.block_m // 2], accum_dtype)
    acc_s_ub_ = T.alloc_ub([shape.block_m // 2, shape.block_n], accum_dtype)
    sumexp_i_ub = T.alloc_ub([shape.block_m // 2], accum_dtype)
    acc_s_half = T.alloc_ub([shape.block_m // 2, shape.block_n], dtype)
    acc_o_ub = T.alloc_ub([shape.block_m // 2, shape.dim], accum_dtype)
    acc_o_half = T.alloc_ub([shape.block_m // 2, shape.dim], dtype)
    alpha_ub = T.alloc_ub([shape.block_m // 2], accum_dtype)
    sumexp_meta_ub = T.alloc_ub([shape.block_m // 2], accum_dtype)
    mask_col = T.alloc_ub([shape.block_n], accum_dtype)

    return TileBufs(
        q_l1, k_l1, v_l1, acc_s_l1,
        lhs_l0, rhs_l0, acc_s_l0c, acc_o_l0c,
        acc_o, sumexp, m_i,
        acc_s_ub, m_i_prev, acc_s_ub_, sumexp_i_ub,
        acc_s_half, acc_o_ub, acc_o_half,
        alpha_ub, sumexp_meta_ub, mask_col,
    )


def _process_task(tctx: TaskContext, shape: TileShapeInfo,
                  bufs: TaskBufs, output):
    """Process one (batch, head, q-block) tile through the full FA pipeline.

    Preloads Q into L1, runs the ring-pipelined C/V stages, and writes the
    normalized output half-tile.
    """
    with T.Scope("C"):
        T.copy(q[tctx.bz, tctx.by,
                 tctx.bx * shape.block_m: (tctx.bx + 1) * shape.block_m, :],
               bufs.c1.q_l1)
    with T.Scope("v"):
        _init_acc_state(bufs.v2.acc_o, bufs.v2.sumexp, bufs.v1.m_i)

    for t in T.serial(shape.kv_loops + shape.prelaunch):
        if t < shape.kv_loops:
            ctx = TileLoopContext(tctx.bx, tctx.by, tctx.bz,
                                  tctx.task_idx, tctx.vid, t)
            _c1_load_k_compute_s(ctx, shape, bufs.c1, bufs.ws1)
            _v1_softmax_update(ctx, shape, bufs.v1, bufs.v1_ws)
        if t >= shape.prelaunch:
            now_k = t - shape.prelaunch
            ctx = TileLoopContext(tctx.bx, tctx.by, tctx.bz,
                                  tctx.task_idx, tctx.vid, now_k)
            _c2_compute_pv(ctx, shape, bufs.c2, bufs.ws2, bufs.ws3)
            _v2_merge_output(ctx, shape, bufs.v2, bufs.ws3, bufs.ws_meta)

    ctx = TileLoopContext(tctx.bx, tctx.by, tctx.bz,
                          tctx.task_idx, tctx.vid, 0)
    _finalize_output(ctx, shape, bufs.out, output)


# ---------------------------------------------------------------------------
#  JIT-compiled entry point
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[3], workspace_idx=[4], pass_configs=pass_configs)
def flash_attention_fwd(cfg: FlashAttentionConfig):
    batch = cfg.batch
    heads = cfg.heads
    q_seq_len = cfg.q_seq_len
    kv_seq_len = cfg.kv_seq_len
    dim = cfg.dim
    dtype = cfg.dtype

    block_m, block_n = 64, 64
    cube_k = 64
    num_physical_cores = 20
    prelaunch = 2
    ring_slots = prelaunch + 1

    accum_dtype = "float"

    sm_scale = (1.0 / dim) ** 0.5

    # Pad kv_seq_len to block_n alignment for tensor shapes
    kv_seq_len_padded = ((kv_seq_len + block_n - 1) // block_n) * block_n

    q_shape = [batch, heads, q_seq_len, dim]
    kv_shape = [batch, heads, kv_seq_len_padded, dim]
    out_shape = [batch, heads, q_seq_len, dim]

    q_blocks = T.ceildiv(q_seq_len, block_m)
    block_num = q_blocks * heads * batch
    used_core_num = min(num_physical_cores, block_num)
    tasks_per_core = T.ceildiv(block_num, used_core_num)
    kv_loops = T.ceildiv(kv_seq_len, block_n)
    qk_k_loops = T.ceildiv(dim, cube_k)

    if dim % cube_k != 0:
        raise ValueError(f"dim ({dim}) must be divisible by cube_k ({cube_k})")
    if block_num <= 0:
        raise ValueError(f"block_num ({block_num}) must be > 0")

    tail_valid = kv_seq_len % block_n

    # Bundle shape/tiling parameters into a single container so that
    # tile-level helpers receive at most 5 arguments (G.FNM.03).
    shape_info = TileShapeInfo(
        block_m, block_n, dim, cube_k, qk_k_loops, ring_slots,
        kv_seq_len_padded, kv_loops, q_seq_len, heads,
        sm_scale, tail_valid, prelaunch,
    )

    @T.prim_func
    def main(
        q: T.Tensor(q_shape, dtype),
        k: T.Tensor(kv_shape, dtype),
        v: T.Tensor(kv_shape, dtype),
        output: T.Tensor(out_shape, dtype),
        workspaces: FAWorkspaces,
    ):
        workspace_1 = workspaces.ws_s
        workspace_2 = workspaces.ws_p
        workspace_3 = workspaces.ws_o
        workspace_meta = workspaces.ws_meta

        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            tb = _allocate_tile_buffers(shape_info, dtype, accum_dtype)

            bufs = TaskBufs(
                C1Bufs(tb.q_l1, tb.k_l1, tb.lhs_l0, tb.rhs_l0, tb.acc_s_l0c),
                C2Bufs(tb.acc_s_l1, tb.lhs_l0, tb.rhs_l0, tb.acc_o_l0c,
                       tb.v_l1),
                V1Bufs(tb.acc_s_ub, tb.acc_s_ub_, tb.m_i, tb.m_i_prev,
                       tb.sumexp_i_ub, tb.acc_s_half, tb.mask_col),
                V1Wss(workspace_1, workspace_2, workspace_meta),
                V2Bufs(tb.acc_o, tb.sumexp, tb.acc_o_ub, tb.alpha_ub,
                       tb.sumexp_meta_ub),
                OutBufs(tb.acc_o, tb.sumexp, tb.acc_o_half),
                workspace_1, workspace_2, workspace_3, workspace_meta,
            )

            for local_idx in T.serial(tasks_per_core):
                task_idx = cid * tasks_per_core + local_idx
                if task_idx >= block_num:
                    continue
                bx = task_idx % q_blocks
                by = task_idx // q_blocks % heads
                bz = task_idx // q_blocks // heads % batch
                _process_task(TaskContext(bx, by, bz, task_idx, vid),
                              shape_info, bufs, output)

    return main
