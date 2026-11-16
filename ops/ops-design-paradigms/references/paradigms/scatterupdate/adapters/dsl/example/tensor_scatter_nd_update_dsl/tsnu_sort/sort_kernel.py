# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""tensor_scatter_nd_update 自包含排序库（tsnu_sort，超发核支持，ASC sort_lib 同构）。

来源：vendor 自 custom_scatter_add_dsl/sort_kernel.py + lib/（radix_tiling/radix_passes/sort_dsl）——
仅改 import 行为包内相对导入，逻辑零改动。

ASC 同构结构（每算子内嵌 sort_lib/）+ 库化超发支持（用户知识）：
- **多核路径**：active_cores 为运行期参数——计算分块按 active_cores 步进
  （b >= active_cores 的超发核 n_my<=0 空跑计算），**屏障无条件全核到达**
  ——宿主算子 blockDim = max(phaseCores, sortCores) > activeCores 时
  排序阶段不死锁（= ASC SortRadixMoreCore 的 blockIdx>=activeCores 守卫
  + SyncAll 无条件结构）
- **单核路径**：全核冗余执行（= ASC SortRadixOneCore 无守卫无屏障——
  各核对同一输入做相同排序，幂等共写； ASC "0 SyncAll、0 GM workspace"）
- perm 恒 int32（= ASC SortInvoke<KeyT, int32_t, CountT, false>）
- 复用项目 lib/ 相位库（radix_passes / sort_dsl / tiling = 基础设施）

派生自 radix_sort_simd_dsl/radix_kernel.py（RadixMCKernel 轮循环），
差异仅：nb = get_block_num() → active_cores（运行期参数）。

UB 布局同源：tiling 预算 per_elem（§19.63 后 4B sorter 内部 + 通道）。
"""

import cannbotdsl
from cannbotdsl import (
    Dim,
    Tensor,
    TensorSpec,
    dtypes,
    tile_slice,
    RingBuffer,
    make_layout,
    copy_options,
)
from cannbotdsl.buffer import Buffer, MemLoc
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import simt
from cannbotdsl.ops import reg
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.sync import global_sync_all
from cannbotdsl.tensor import MemLoc as _ML  # noqa: F401

from .radix_tiling import radix_tiling, DCACHE_SIZE
from .radix_passes import prescan_pass, publish_agg_pass, widen_hist_pass
from .radix_passes import prescan_pass_c8, publish_agg_pass_c8, widen_hist_pass_c8  # noqa: F401
from .sort_dsl import SortDSL, FullSortBuffers, ub_requirement_full


def _mask8_table(ds, signed, desc):
    """b8 域按轮 XOR 掩码（与 radix_kernel 同源）。"""
    if not signed:
        return [0xFF] * ds if desc else [0] * ds
    if not desc:
        return [0] * (ds - 1) + [0x80]
    return [0xFF] * (ds - 1) + [0x7F]


def _barrier():
    """核间屏障——全核无条件到达（超发安全的根基）。"""
    global_sync_all()


@jit
def stub_pass(src, ov, oi, n):
    """V2 sort 桩：等流量线性拷贝（b8→ov，i→oi），计时时序用
    （= radix_kernel.stub_pass 内嵌副本，D2 自含原则）。
    """
    nblk = cannbotdsl.ceil_div(n, 256)
    with vf(mode="simd"):
        reg.full_mask(elem_bits=8)
        reg.full_mask(elem_bits=32)
        c8 = reg.mask_counter(n)
        for i in cannbotdsl.range(nblk):
            bm, c8 = reg.update_mask(c8, elem_bits=8)
            reg.vstore(ov, i * 256, reg.vload(src, i * 256), bm)
        c32 = reg.mask_counter(n)
        for i in cannbotdsl.range(nblk):
            bm2, c32 = reg.update_mask(c32, elem_bits=32)
            reg.vstore(oi, i * 256, reg.varange(i * 256, dtypes.uint32), bm2)


class SortCfg:
    """内嵌排序配置（编译期；active_cores/n 为运行期）。"""

    def __init__(
        self,
        key_dt,
        tile,
        counter8=False,
        spin_limit=2_000_000,
        stub_scatter=False,
        stub_sort=False,
    ):
        # 键域三档（= ASC keyDtype）：int16/int32/int64（+ 无符号族）
        assert key_dt in ("int16", "uint16", "int32", "uint32", "int64", "uint64"), (
            key_dt
        )
        self.ds = getattr(dtypes, key_dt).width // 8
        self.rounds = self.ds
        self.counter8 = bool(counter8)
        self.spin_limit = int(spin_limit)
        self.tile = int(tile)
        self.nu32 = tile * self.ds // 4
        self.val_dt_name = key_dt
        self.mask8 = _mask8_table(self.ds, key_dt.startswith("int"), False)
        self.vdt = getattr(dtypes, key_dt)
        self.cdt = dtypes.int64 if counter8 else dtypes.int32
        self.agg = 2**62 if counter8 else 2**30
        self.idx_dtype = dtypes.int32  # perm 恒 i32（ASC SortInvoke 同构）
        self.stub_scatter = bool(stub_scatter)
        self.stub_sort = bool(stub_sort)
        self.sorter = SortDSL(index_dtype="u32")
        self.a_native = self.ds == 2
        self.a_load_n = tile if self.ds == 2 else self.nu32


@jit
def sort_multicore_phase(
    cfg: SortCfg,
    b,
    n,
    active_cores,
    gm_x,
    out_val,
    out_idx,
    ws_b8,
    ws_hist,
    ws_cumsum,
    ws_excl,
    ws_gh,
    db_v,
    db_i,
):
    """多核排序轮循环（在复合 kernel 体内调用；超发安全）。

    与 radix_kernel.RadixMCKernel.__call__ 同源，唯一差异：
    nb = active_cores（运行期参数）而非 get_block_num()——超发核
    （b >= active_cores）的 n_my <= 0，tile 循环空跑，屏障全核到达。
    """
    tile_size = cfg.tile
    nb = active_cores  # ★ 超发：分块域 = active_cores
    tc = (n + tile_size - 1) // tile_size
    # ── UB 布局（同 RadixMCKernel）──
    if const_expr(cfg.a_native):
        ch = RingBuffer(MemLoc.UB, cfg.vdt, make_layout((tile_size,)), depth=1)
    else:
        ch = RingBuffer(MemLoc.UB, dtypes.uint32, make_layout((cfg.a_load_n,)), depth=1)
    ch_val = RingBuffer(MemLoc.UB, cfg.vdt, make_layout((tile_size,)), depth=1)
    ch_idx = RingBuffer(MemLoc.UB, cfg.idx_dtype, make_layout((tile_size,)), depth=1)
    ch_b8 = RingBuffer(MemLoc.UB, dtypes.uint8, make_layout((tile_size,)), depth=1)
    ch_sv = RingBuffer(MemLoc.UB, dtypes.uint8, make_layout((tile_size,)), depth=1)
    ch_si = RingBuffer(MemLoc.UB, dtypes.uint32, make_layout((tile_size,)), depth=1)
    ch_hist = RingBuffer(MemLoc.UB, dtypes.uint16, make_layout((256,)), depth=1)
    ch_cum = RingBuffer(MemLoc.UB, dtypes.uint16, make_layout((256,)), depth=1)
    ch_gh = RingBuffer(MemLoc.UB, cfg.cdt, make_layout((256,)), depth=1)
    ch_excl = RingBuffer(MemLoc.UB, cfg.cdt, make_layout((256,)), depth=1)
    b8_stage = Buffer(MemLoc.UB, dtypes.uint8, make_layout((tile_size,)))
    hist_ub = Buffer(MemLoc.UB, dtypes.uint16, make_layout((256,)))
    cumsum_ub = Buffer(MemLoc.UB, dtypes.uint16, make_layout((256,)))
    excl_acc = Buffer(MemLoc.UB, cfg.cdt, make_layout((256,)))
    hist32 = Buffer(MemLoc.UB, cfg.cdt, make_layout((256,)))
    cum32 = Buffer(MemLoc.UB, cfg.cdt, make_layout((256,)))
    pos_table = Buffer(MemLoc.UB, cfg.cdt, make_layout((256,)))
    bufs = cfg.sorter.declare_buffers(tile_size)

    ws_gh_flat = ws_gh.view(cfg.rounds * tc * 256)
    ws_excl_flat = ws_excl.view(cfg.rounds * 256)
    ws_hist_flat = ws_hist.view(tc * 256)
    ws_cumsum_flat = ws_cumsum.view(tc * 256)
    # 超发核（b >= active_cores 且 tc <= active_cores）：tc - b <= 0 →
    # 分子钳 0 → n_my = 0（tile 循环空跑，屏障照常到达）。
    # 注：生产 radix_kernel 的 n_my 恒非负（nb=active=launch），负分子
    # 路径仅超发场景出现，故钳位在内嵌版而非原版回移。
    n_my = cannbotdsl.ceil_div(max(tc - b, 0), nb)
    prescan_fn = prescan_pass_c8 if cfg.counter8 else prescan_pass
    widen_fn = widen_hist_pass_c8 if cfg.counter8 else widen_hist_pass
    publish_fn = publish_agg_pass_c8 if cfg.counter8 else publish_agg_pass

    for r in range_constexpr(cfg.rounds):
        src_v = gm_x if r == 0 else (db_v if (r % 2) == 1 else out_val)
        dst_v = db_v if (r % 2) == 0 else out_val
        dst_i = db_i if (r % 2) == 0 else out_idx
        idx_src = db_i if (r == 0 or (r % 2) == 1) else out_idx
        excl_off = r * 256
        gh_row = r * tc
        # ── A 相：预扫描 ──
        with vf(mode="simd"):
            if const_expr(cfg.counter8):
                m64 = reg.full_mask(elem_bits=64)
                z32 = reg.vdups(0, dtypes.int32, mask=reg.full_mask(elem_bits=32))
                _zr, _ = reg.vinterleave(z32, z32)
                z64 = reg.vreinterpret_lanes(_zr, dtypes.int64)
                for _k in range_constexpr(8):
                    reg.vstore(excl_acc, _k * 32, z64, m64)
            else:
                m32 = reg.full_mask(elem_bits=32)
                z = reg.vdups(0, dtypes.int32, mask=m32)
                for _k in range_constexpr(4):
                    reg.vstore(excl_acc, _k * 64, z, m32)
        for i in cannbotdsl.range(n_my):
            tile = b + i * nb
            if const_expr(cfg.counter8):
                base = (tile + dtypes.int64(0)) * tile_size
            else:
                base = tile * tile_size
            _nt = min(n - base, tile_size)
            n_tile = _nt
            if const_expr(cfg.a_native):
                a_src = src_v
            else:
                a_src = src_v.view(dtype=dtypes.uint32)
            mem_copy(ch.next_write(), tile_slice(a_src, (cfg.a_load_n,), (tile,)))
            slot = ch.next_read()
            prescan_fn(
                slot,
                b8_stage,
                hist_ub,
                cumsum_ub,
                excl_acc,
                n_tile,
                cfg.ds,
                r * 8,
                cfg.mask8[r],
            )
            mem_copy(tile_slice(ws_b8, (tile_size,), (tile,)), b8_stage)
            mem_copy(tile_slice(ws_hist_flat, (256,), (tile,)), hist_ub)
            mem_copy(tile_slice(ws_cumsum_flat, (256,), (tile,)), cumsum_ub)
            # excl 只累积不发布（发布移出 tile 循环 = ASC GetGlobalExclusiveSum
            # 尾部——每核每轮一次原子加；循环内发布会致 n_my≥2 重复累加
            # → excl 虚增 → pos 越界 [C1-R 修复，与 radix_kernel 同款]）
        slot2 = ch_excl.next_write()
        if const_expr(cfg.counter8):
            with vf(mode="simd"):
                m64 = reg.full_mask(elem_bits=64)
                for _k in range_constexpr(8):
                    reg.vstore(slot2, _k * 32, reg.vload(excl_acc, _k * 32), m64)
            _s2c = ch_excl.next_read()
            with vf(mode="simt", thread=256):
                _t = simt.thread_idx()[0]
                simt.atomic_add(ws_excl_flat.ptr(excl_off + _t), dtypes.int64(_s2c[_t]))
        else:
            with vf(mode="simd"):
                m32 = reg.full_mask(elem_bits=32)
                for _k in range_constexpr(4):
                    reg.vstore(slot2, _k * 64, reg.vload(excl_acc, _k * 64), m32)
            mem_copy(
                tile_slice(ws_excl_flat, (256,), (r,)),
                ch_excl.next_read(),
                options=copy_options(atomic_add=dtypes.bool(True)),
            )
        _barrier()
        # ── B1 相：发布（全核发布完再屏障）──
        for i in cannbotdsl.range(n_my):
            tile = b + i * nb
            if const_expr(cfg.counter8):
                base = (tile + dtypes.int64(0)) * tile_size
            else:
                base = tile * tile_size
            _nt = min(n - base, tile_size)
            n_tile = _nt
            mem_copy(ch_hist.next_write(), tile_slice(ws_hist_flat, (256,), (tile,)))
            mem_copy(ch_cum.next_write(), tile_slice(ws_cumsum_flat, (256,), (tile,)))
            widen_fn(ch_hist.next_read(), ch_cum.next_read(), hist32, cum32)
            gh_slot = ch_gh.next_write()
            publish_fn(hist32, gh_slot)
            mem_copy(
                tile_slice(ws_gh_flat, (256,), (gh_row + tile,)), ch_gh.next_read()
            )
        _barrier()
        # ── B2 相：排序 + lookback + pos + 散射（发布全核就绪后）──
        for i in cannbotdsl.range(n_my):
            tile = b + i * nb
            if const_expr(cfg.counter8):
                base = (tile + dtypes.int64(0)) * tile_size
            else:
                base = tile * tile_size
            _nt = min(n - base, tile_size)
            n_tile = _nt
            mem_copy(ch_hist.next_write(), tile_slice(ws_hist_flat, (256,), (tile,)))
            mem_copy(ch_cum.next_write(), tile_slice(ws_cumsum_flat, (256,), (tile,)))
            widen_fn(ch_hist.next_read(), ch_cum.next_read(), hist32, cum32)
            mem_copy(ch_b8.next_write(), tile_slice(ws_b8, (tile_size,), (tile,)))
            src = ch_b8.next_read()
            ov = ch_sv.next_write()
            oi = ch_si.next_write()
            if const_expr(cfg.stub_sort):
                stub_pass(src, ov, oi, n_tile)
            else:
                cfg.sorter.sort(bufs, src, ov, oi, n_tile)
            sv_slot = ch_sv.next_read()
            si_slot = ch_si.next_read()
            mem_copy(ch_val.next_write(), tile_slice(src_v, (tile_size,), (tile,)))
            val_slot = ch_val.next_read()
            mem_copy(ch_idx.next_write(), tile_slice(idx_src, (tile_size,), (tile,)))
            idx_slot = ch_idx.next_read()
            with vf(mode="simt", thread=256):
                t = simt.thread_idx()[0]
                acc = hist32[t]
                j = tile - 1
                stop = 0
                while (j >= 0) and (stop == 0):
                    v = simt.l2cache_load(ws_gh_flat.ptr(gh_row * 256 + j * 256 + t))
                    it = 0
                    # 位域三态（= ASC）：bit30|hist=聚合就绪、bit31|value=前缀就绪、
                    # 0=未发布。屏障后 raw 必已发布，仅对 v==0 短自旋；
                    # 超时后按 raw=0 解码，绝不再把 0 误当已就绪。
                    while (v == 0) and (it < cfg.spin_limit):
                        it += 1
                        simt.dcci_single(ws_gh_flat.ptr(gh_row * 256 + j * 256 + t))
                        v = simt.l2cache_load(
                            ws_gh_flat.ptr(gh_row * 256 + j * 256 + t)
                        )
                    if v >= cfg.agg:
                        acc = acc + (v - cfg.agg)
                    elif v < 0:
                        acc = acc + (v + cfg.agg)
                        stop = 1
                    else:
                        acc = acc + 0
                    j = j - 1
                simt.l2cache_store(
                    ws_gh_flat.ptr(gh_row * 256 + tile * 256 + t), acc - cfg.agg
                )
                ex = simt.l2cache_load(ws_excl_flat.ptr(excl_off + t))
                pos_table[t] = ex + acc - hist32[t] - cum32[t]
                simt.syncthreads()
                n_t = n_tile
                e = t
                while e < n_t:
                    if const_expr(cfg.stub_scatter):
                        pass
                    else:
                        sv = sv_slot[e]
                        si = si_slot[e]
                        p = pos_table[sv] + e
                        val = val_slot[si]
                        simt.l2cache_store(dst_v.ptr(p), val)
                        if r == 0:
                            simt.l2cache_store(
                                dst_i.ptr(p),
                                simt.cast(base, dtype=dtypes.int32)
                                + simt.cast(si, dtype=dtypes.int32),
                            )
                        else:
                            pi = idx_slot[si]
                            simt.l2cache_store(dst_i.ptr(p), pi)
                    e = e + 256
                simt.threadfence()
        _barrier()


@jit
def sort_singlecore_phase(cfg: SortCfg, gm_in, out_val, out_idx, n):
    """单核排序（在复合 kernel 体内调用；全核冗余执行 = ASC OneCore）。

    每个启动核独立执行同一排序（相同输入 → 相同输出，幂等共写，
    无内部屏障）——与 ASC SortRadixOneCore 无 blockIdx 守卫同构。
    """
    tile_cap = cfg.tile  # n <= tile（单核域）
    if const_expr(cfg.ds == 8):
        # ds=8：u32 窗口通道（= SC-cap §19.60 view 形态；pass1_full
        # 的 u32 粒度 vload 与 int64 原生通道会错配）
        ch = RingBuffer(MemLoc.UB, dtypes.int32, make_layout((cfg.nu32,)), depth=1)
        mem_copy(
            ch.next_write(),
            tile_slice(gm_in.view(dtype=dtypes.uint32), (cfg.nu32,), (0,)),
        )
    else:
        ch = RingBuffer(MemLoc.UB, cfg.vdt, make_layout((tile_cap,)), depth=1)
        mem_copy(ch.next_write(), tile_slice(gm_in, (tile_cap,), (0,)))
    src = ch.next_read()
    # §19.62/§19.63 布局：sidx 单区 + pristine 别名 + val 单 buffer 原地
    sidx = Buffer(MemLoc.UB, dtypes.uint32, make_layout((tile_cap,)))
    bufs = FullSortBuffers(
        val=Buffer(MemLoc.UB, dtypes.uint8, make_layout((tile_cap,))),
        idx8=[
            Buffer(MemLoc.UB, dtypes.uint8, make_layout((tile_cap,))),
            Buffer(MemLoc.UB, dtypes.uint8, make_layout((tile_cap,))),
        ],
        pristine=sidx.view(dtype=dtypes.uint8),
        excl16=Buffer(MemLoc.UB, dtypes.uint16, make_layout((256,))),
        sidx=sidx,
        next_idx=Buffer(MemLoc.UB, dtypes.uint32, make_layout((tile_cap,))),
        idx_out=Buffer(MemLoc.UB, dtypes.uint32, make_layout((tile_cap,))),
        val_a=Buffer(MemLoc.UB, cfg.vdt, make_layout((tile_cap,))),
        val_b=Buffer(MemLoc.UB, cfg.vdt, make_layout((tile_cap,))),
    )
    nu32_cap = cfg.nu32
    cfg.sorter.sort_full(bufs, src, n, cfg.ds, cfg.mask8, nu32_cap)
    mem_copy(tile_slice(out_val, (tile_cap,), (0,)), bufs.val_b)
    mem_copy(tile_slice(out_idx, (tile_cap,), (0,)), bufs.idx_out)
