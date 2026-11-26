# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""单 GEMM 流水骨架模板：复制到 `tools/skeleton_gemm.py` 后只改 kernel 体。

结构即 `templates/gemm_pipelined.md` 的契约（与
``experimental/kernel/matmul/matmul_exp_scale_ub/`` 的 GEMM 段一致）：

* L1A/L1B 各 2 份缓冲，按 `l1_phase` 交替；
* L0A/L0B 各 2 份缓冲，按 `l0_phase` 交替；
* `bk > sk` 且 `bk % sk == 0`；
* 每份片上缓冲一块 `tla.mutex`（id 唯一），lock/unlock 按 pipe 配对；
* 不出现 `auto_sync` / `flag` / `pipe_barrier` / `cross_flag`。

允许改动的只有：`Tiling` 的默认值（由 `skeleton_profile.py` 通过 `--tile` / `--map` 覆盖）、
kernel 体内部的 MMAD 组合与 epilogue 结构。缓冲份数、mutex 配对顺序、双缓冲嵌套关系不得改动。

本文件只是 GEMM 基线，不是交付 kernel：融合算子的 epilogue 与跨核交棒要另起 `src/` 实现，
在那里按 develop 规则用 `cross_flag` 做 C→V / AIC↔AIV 交棒，不要把这个骨架改造成融合
kernel 再回来跑骨架测量——否则骨架的 `torch.mm` 参考与 mutex-only 自检都不再成立。

A/B 的 row/col layout 由 host 侧 `layout_tag` 决定（`col` 即转置视图，物理为 K×M / N×K），
kernel 内无需分支；`a.origin_shape`、`b.origin_shape` 始终是逻辑 `[M, K]`、`[K, N]`。
"""

from dataclasses import dataclass

import catlass.tla as tla


@dataclass(frozen=True)
class Tiling:
    bm: tla.Constexpr[int] = 256
    bn: tla.Constexpr[int] = 256
    bk: tla.Constexpr[int] = 128
    sk: tla.Constexpr[int] = 32
    sm: tla.Constexpr[int] = 8
    sn: tla.Constexpr[int] = 8


@tla.kernel
def gemm_pipelined(
    a: tla.Tensor,
    b: tla.Tensor,
    c: tla.Tensor,
    _tiling: tla.Constexpr[Tiling],
) -> None:
    """C[M, N] = A[M, K] @ B[K, N]，单 batch，fp32 L0C 累加。"""
    zero = 0
    one = 1
    bm = _tiling.bm
    bn = _tiling.bn
    bk = _tiling.bk
    sk = _tiling.sk
    sm = _tiling.sm
    sn = _tiling.sn

    dtype = a.ptr.dtype
    m = a.origin_shape[0]
    ktot = a.origin_shape[1]
    n = b.origin_shape[1]
    nm = (m + bm - 1) // bm
    nn = (n + bn - 1) // bn
    nk = (ktot + bk - 1) // bk

    # 每份片上缓冲一块 mutex；id 是硬件资源，不是句柄。
    m_l1a0 = tla.mutex("l1_a0", id=0)
    m_l1b0 = tla.mutex("l1_b0", id=1)
    m_l1a1 = tla.mutex("l1_a1", id=2)
    m_l1b1 = tla.mutex("l1_b1", id=3)
    m_l0a0 = tla.mutex("l0_a0", id=4)
    m_l0b0 = tla.mutex("l0_b0", id=5)
    m_l0a1 = tla.mutex("l0_a1", id=6)
    m_l0b1 = tla.mutex("l0_b1", id=7)
    m_l0c = tla.mutex("l0_c", id=8)

    l1a0_ptr = tla.allocate(bm * bk, dtype, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(bm * bk, dtype, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(bk * bn, dtype, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(bk * bn, dtype, tla.AddressSpace.l1, 512)
    l0a0_ptr = tla.allocate(bm * sk, dtype, tla.AddressSpace.l0a, 512)
    l0a1_ptr = tla.allocate(bm * sk, dtype, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(sk * bn, dtype, tla.AddressSpace.l0b, 512)
    l0b1_ptr = tla.allocate(sk * bn, dtype, tla.AddressSpace.l0b, 512)
    l0c = tla.allocate(bm * bn, tla.Float32, tla.AddressSpace.l0c, 1024)

    gsm = (nm + sm - 1) // sm
    gsn = (nn + sn - 1) // sn
    total = gsm * gsn * sm * sn

    with tla.cube():
        for task in tla.range(tla.arch.block_idx(), total, tla.arch.block_num()):
            per_super = sm * sn
            sb = task // per_super
            inner = task % per_super
            mt = (sb // gsn) * sm + inner // sn
            nt = (sb % gsn) * sn + inner % sn
            if mt < nm and nt < nn:
                og = tla.tile_view(c, tla.make_shape(bm, bn), tla.make_coord(mt, nt))
                l0_c = tla.make_tensor_like(l0c, og)

                ag_head = tla.tile_view(
                    a, tla.make_shape(bm, bk), tla.make_coord(mt, zero)
                )
                bg_head = tla.tile_view(
                    b, tla.make_shape(bk, bn), tla.make_coord(zero, nt)
                )
                al = tla.make_tensor_like(l1a0_ptr, ag_head)
                bl = tla.make_tensor_like(l1b0_ptr, bg_head)
                m_l1a0.lock(pipe=tla.arch.MTE2)
                tla.copy(al, ag_head)
                m_l1a0.unlock(pipe=tla.arch.MTE2)
                m_l1b0.lock(pipe=tla.arch.MTE2)
                tla.copy(bl, bg_head)
                m_l1b0.unlock(pipe=tla.arch.MTE2)
                m_l0c.lock(pipe=tla.arch.CUBE)

                l1_phase = tla.as_numeric(zero)
                for kb in tla.range(zero, nk, one):
                    if l1_phase == zero:
                        m_l1a0.lock(pipe=tla.arch.MTE1)
                        m_l1b0.lock(pipe=tla.arch.MTE1)
                    else:
                        m_l1a1.lock(pipe=tla.arch.MTE1)
                        m_l1b1.lock(pipe=tla.arch.MTE1)

                    if kb + one < nk:
                        ag_n = tla.tile_view(
                            a, tla.make_shape(bm, bk), tla.make_coord(mt, kb + one)
                        )
                        bg_n = tla.tile_view(
                            b, tla.make_shape(bk, bn), tla.make_coord(kb + one, nt)
                        )
                        na_ptr = l1a1_ptr if l1_phase == zero else l1a0_ptr
                        nb_ptr = l1b1_ptr if l1_phase == zero else l1b0_ptr
                        al_n = tla.make_tensor_like(na_ptr, ag_n)
                        bl_n = tla.make_tensor_like(nb_ptr, bg_n)
                        if l1_phase == zero:
                            m_l1a1.lock(pipe=tla.arch.MTE2)
                            m_l1b1.lock(pipe=tla.arch.MTE2)
                        else:
                            m_l1a0.lock(pipe=tla.arch.MTE2)
                            m_l1b0.lock(pipe=tla.arch.MTE2)
                        tla.copy(al_n, ag_n)
                        tla.copy(bl_n, bg_n)
                        if l1_phase == zero:
                            m_l1a1.unlock(pipe=tla.arch.MTE2)
                            m_l1b1.unlock(pipe=tla.arch.MTE2)
                        else:
                            m_l1a0.unlock(pipe=tla.arch.MTE2)
                            m_l1b0.unlock(pipe=tla.arch.MTE2)

                    ca_ptr = l1a0_ptr if l1_phase == zero else l1a1_ptr
                    cb_ptr = l1b0_ptr if l1_phase == zero else l1b1_ptr
                    ag_c = tla.tile_view(
                        a, tla.make_shape(bm, bk), tla.make_coord(mt, kb)
                    )
                    bg_c = tla.tile_view(
                        b, tla.make_shape(bk, bn), tla.make_coord(kb, nt)
                    )
                    al_c = tla.make_tensor_like(ca_ptr, ag_c)
                    bl_c = tla.make_tensor_like(cb_ptr, bg_c)

                    aa = tla.tile_view(
                        al_c, tla.make_shape(bm, sk), tla.make_coord(zero, zero)
                    )
                    bb = tla.tile_view(
                        bl_c, tla.make_shape(sk, bn), tla.make_coord(zero, zero)
                    )
                    a0 = tla.make_tensor_like(l0a0_ptr, aa)
                    b0 = tla.make_tensor_like(l0b0_ptr, bb)
                    m_l0a0.lock(pipe=tla.arch.MTE1)
                    m_l0b0.lock(pipe=tla.arch.MTE1)
                    tla.copy(a0, aa)
                    tla.copy(b0, bb)
                    m_l0a0.unlock(pipe=tla.arch.MTE1)
                    m_l0b0.unlock(pipe=tla.arch.MTE1)

                    l0_phase = tla.as_numeric(zero)
                    for kk in tla.range(zero, bk // sk, one):
                        if l0_phase == zero:
                            m_l0a0.lock(pipe=tla.arch.CUBE)
                            m_l0b0.lock(pipe=tla.arch.CUBE)
                        else:
                            m_l0a1.lock(pipe=tla.arch.CUBE)
                            m_l0b1.lock(pipe=tla.arch.CUBE)

                        if kk + one < bk // sk:
                            aa_n = tla.tile_view(
                                al_c,
                                tla.make_shape(bm, sk),
                                tla.make_coord(zero, kk + one),
                            )
                            bb_n = tla.tile_view(
                                bl_c,
                                tla.make_shape(sk, bn),
                                tla.make_coord(kk + one, zero),
                            )
                            na0_ptr = l0a1_ptr if l0_phase == zero else l0a0_ptr
                            nb0_ptr = l0b1_ptr if l0_phase == zero else l0b0_ptr
                            a0_n = tla.make_tensor_like(na0_ptr, aa_n)
                            b0_n = tla.make_tensor_like(nb0_ptr, bb_n)
                            if l0_phase == zero:
                                m_l0a1.lock(pipe=tla.arch.MTE1)
                                m_l0b1.lock(pipe=tla.arch.MTE1)
                            else:
                                m_l0a0.lock(pipe=tla.arch.MTE1)
                                m_l0b0.lock(pipe=tla.arch.MTE1)
                            tla.copy(a0_n, aa_n)
                            tla.copy(b0_n, bb_n)
                            if l0_phase == zero:
                                m_l0a1.unlock(pipe=tla.arch.MTE1)
                                m_l0b1.unlock(pipe=tla.arch.MTE1)
                            else:
                                m_l0a0.unlock(pipe=tla.arch.MTE1)
                                m_l0b0.unlock(pipe=tla.arch.MTE1)

                        ca0_ptr = l0a0_ptr if l0_phase == zero else l0a1_ptr
                        cb0_ptr = l0b0_ptr if l0_phase == zero else l0b1_ptr
                        aa_c = tla.tile_view(
                            al_c, tla.make_shape(bm, sk), tla.make_coord(zero, kk)
                        )
                        bb_c = tla.tile_view(
                            bl_c, tla.make_shape(sk, bn), tla.make_coord(kk, zero)
                        )
                        a0_c = tla.make_tensor_like(ca0_ptr, aa_c)
                        b0_c = tla.make_tensor_like(cb0_ptr, bb_c)
                        is_last = (kb == nk - 1) and (kk == bk // sk - 1)
                        tla.mmad(
                            l0_c,
                            a0_c,
                            b0_c,
                            init_c=(kb == 0 and kk == 0),
                            unit_flag=0b11 if is_last else 0b10,
                        )
                        if l0_phase == zero:
                            m_l0a0.unlock(pipe=tla.arch.CUBE)
                            m_l0b0.unlock(pipe=tla.arch.CUBE)
                        else:
                            m_l0a1.unlock(pipe=tla.arch.CUBE)
                            m_l0b1.unlock(pipe=tla.arch.CUBE)
                        l0_phase = one - l0_phase

                    if l1_phase == zero:
                        m_l1a0.unlock(pipe=tla.arch.MTE1)
                        m_l1b0.unlock(pipe=tla.arch.MTE1)
                    else:
                        m_l1a1.unlock(pipe=tla.arch.MTE1)
                        m_l1b1.unlock(pipe=tla.arch.MTE1)
                    l1_phase = one - l1_phase

                m_l0c.unlock(pipe=tla.arch.CUBE)
                m_l0c.lock(pipe=tla.arch.FIX)
                tla.copy(og, l0_c, tla.params.CopyL0C2DstParams(unit_flag=0b11))
                m_l0c.unlock(pipe=tla.arch.FIX)
