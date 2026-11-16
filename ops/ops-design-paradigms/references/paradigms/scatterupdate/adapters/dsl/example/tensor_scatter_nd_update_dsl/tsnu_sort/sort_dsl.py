# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SortDSL —— tile 内稳定排序可复用组件（AscendC SortImpl 逐段同构）。

双模式：
  u8   单轮字节排序（多核 LSD 轮内调用，= SortImpl sizeof(T)==1 路径）
  full 全 dtype 单核整排序（= SortImpl sizeof(T)>1 多字节路径 +
       SortRadixOneCore 编排，§19.6）：twiddle 融合为 b8 域按轮 XOR 掩码，
       轮间 UB ping-pong 值/索引重排，0 屏障 0 GM workspace

算法（sort_impl.h:788-869，sizeof(T)==1）：
  Pass 1  GetExclusiveSum: 全局直方图（reg.vdups(0) 初始化 + 单循环累加 + 计数器链）
  Pass 2  EightBitsSort: 索引单 squeeze + vgather_reg 值置换（pristine 源 gather）
  Pass 3  LocalSort: 跨块合并（diff 表恒等变形，见 radix_sort_simd_design.md §13.14）
  Pass 4  值输出: gather-permute u8 直出

索引输出双模（index_dtype，模板参数式，默认 "u32" = AscendC 同构）：
  "u32"  寄存器宽化 + 直接 u32 散射（= Interleave(u16,zero)+Scatter(u32) 同构）
         10 B/元素（含 pristine，§19.63 val 原地后）
  "u16"  u16 散射直出（接口扩展），8 B/元素（含 pristine）
         ⚠️ u16 粒度 vscatter 为慢路径，仅 UB 受限时选用

容量：无组件级上限（= AscendC::Sort 隐式约束——tile 上限属 tiling 层职责），
tiling 侧由 ub_requirement() 求逆 + get_mem_size("ub") 推导（公式见其文档）；
组件仅要求 n ≤ npad（declare_buffers 的 OIB），任意 N 由调用方 tile 循环处理。

尾块：reg.mask_counter(N) 计数器链跨运行时循环（= AscendC UpdateMask 免分支机制），
GM↔UB 精确 N 拷贝由调用方负责（tile_slice 视图，形状严格匹配）。

用法（kernel 体内，多 tile 循环的规范模式——rms_norm 同款）：
    sorter = SortDSL(index_dtype="u32")
    in_ch = RingBuffer(MemLoc.UB, dtypes.uint8, make_layout((ts,)), depth=2)
    out_val_ch = RingBuffer(MemLoc.UB, dtypes.uint8, make_layout((ts,)), depth=2)
    out_idx_ch = RingBuffer(MemLoc.UB, dtypes.uint32, make_layout((ts,)), depth=2)  # u16 模式用 uint16
    bufs = sorter.declare_buffers(ts)             # 内部态（val/idx8/excl16）
    n_tiles = ceil_div(total, ts)
    for t in cannbotdsl.range(n_tiles):           # 运行时 tile 循环
        mem_copy(in_ch.next_write(), tile_slice(gm_in, (ts,), (t,)))
        src = in_ch.next_read()
        n_tile = min(total - t * ts, ts)          # 运行时（尾 tile）
        ov, oi = out_val_ch.next_write(), out_idx_ch.next_write()
        sorter.sort(bufs, src, ov, oi, n_tile)    # @jit 逐 Pass（内部循环设备化）
        mem_copy(tile_slice(gm_out_val, (ts,), (t,)), out_val_ch.next_read())
        mem_copy(tile_slice(gm_out_idx, (ts,), (t,)), out_idx_ch.next_read())
    # ⚠️ 多 tile 输出必须经 RingBuffer next_write/next_read 配对（同步即配对）；
    #    静态 Buffer 直拷 GM 存在跨迭代 MTE 读 / VEC 写竞态（可复现）。
"""

import os
from collections import namedtuple

from cannbotdsl.ops import reg
import cannbotdsl
from cannbotdsl import dtypes, ceil_div, make_layout, from_tensor
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import MemLoc


VL8 = 256  # u8 VL
VL32 = 64  # u32 VL（full 模式提取链用）
FIXED_OVERHEAD = 516  # excl16(512) + 退化占位(4)——组件内部算术常数，非平台容量

# 仅内部态（VEC 有序复用）；输出槽位由调用方提供（多 tile 须 RingBuffer next_write/next_read
# 配对同步——rms_norm 样例模式；单次调用可用静态 Buffer）。
# pristine：src 的 pass1 单区域拷贝——跨管线源 buffer 每迭代只允许在单个 vf 区域访问
# （asc_lock 连续同 (mutex_id,pipe) 锁对不提供同步——mutex 闪烁释放，下一迭代 MTE2
#  的 pending 获取可在闪烁窗口捕获 → 输入中途覆写，确定性损坏，见 §13.19）
SortBuffers = namedtuple("SortBuffers", "val idx8 pristine excl16")
FullSortBuffers = namedtuple(
    "FullSortBuffers", "val idx8 pristine excl16 sidx next_idx idx_out val_a val_b"
)


class SortDSL:
    """tile 内 u8 稳定排序（值 + 稳定索引）。

    参数：
        index_dtype: "u32"（默认，AscendC 同构）
                     "u16"（接口扩展；u16 粒度散射慢路径）

    容量：无组件级上限（AscendC::Sort 同为隐式约束——由调用方缓冲决定）。
    tile 大小由 tiling 层用 ub_requirement() + get_mem_size("ub") 推导
    （组件不持有平台容量假设）；算子层经 tile 循环处理任意 N。
    """

    def __init__(self, index_dtype: str = "u32"):
        if index_dtype not in ("u16", "u32"):
            raise ValueError(f"index_dtype 须为 'u16' 或 'u32'，得到 {index_dtype!r}")
        self.index_dtype = index_dtype

    # ── kernel 体内接口 ──────────────────────────────────────────

    def declare_buffers(self, npad: int) -> SortBuffers:
        """在 kernel 体内声明内部 UB 缓冲（须在 vf 块外调用）。

        输出槽位由调用方经 sort(..., out_val, out_idx, ...) 传入：
        - 多 tile 循环：RingBuffer(depth≥1) 的 next_write() 槽 + 输出拷贝经 next_read()
          （配对即同步——静态 Buffer 直拷 GM 会产生跨迭代 MTE 读/VEC 写竞态，
          rms_norm 样例的规范模式）
        - 单次调用：静态 Buffer 亦可
        """
        return SortBuffers(
            # §19.63：val 单 buffer 原地（= ASC srcU8；K78 真实代码路径验证）
            val=Buffer(MemLoc.UB, dtypes.uint8, make_layout((npad,))),
            idx8=[
                Buffer(MemLoc.UB, dtypes.uint8, make_layout((npad,))),
                Buffer(MemLoc.UB, dtypes.uint8, make_layout((npad,))),
            ],
            pristine=Buffer(MemLoc.UB, dtypes.uint8, make_layout((npad,))),
            excl16=Buffer(MemLoc.UB, dtypes.uint16, make_layout((256,))),
        )

    def sort(self, bufs: SortBuffers, src_slot, out_val, out_idx, n) -> None:
        """编排入口（普通方法，仅调用——核内控制流必须在 @jit 内，见类文档）。

        ⚠️ 本方法不得含 with vf / 设备循环（普通方法的控制流不经前端区域分析，
        循环会以普通 Python 执行 = trace 期展开进 vf 区域，击穿 6KB VF 溢出栈）。
        src_slot: pristine 原始值；out_val/out_idx: 输出槽（多 tile 须 RingBuffer
        next_write/next_read 配对——静态 Buffer 直拷 GM 有跨迭代竞态，rms_norm 模式）；
        n: Python int（静态折叠）或运行时 Int32（nblk = ceil_div 运行时计算）。
        ⚠️ 不变式：n ≤ npad 且 n ≤ 输出槽长度。
        """
        self.pass1(bufs, src_slot, n)  # src 唯一读取点（单 vf 区域）
        self.pass2_all4(bufs, n)  # §19.42：4×2-bit 合一 vf
        self.pass3(bufs, out_idx, n)
        self.pass4(bufs, out_idx, out_val, n)

    @jit
    def pass1(self, bufs: SortBuffers, src_slot, n: int) -> None:
        """全局直方图 + 值/索引初始化（= AscendC GetExclusiveSum + init 循环）。"""
        nblk = ceil_div(n, VL8)  # 静态 n 折叠为 Python 值；运行时 n 设备侧计算
        with vf(mode="simd"):
            reg.full_mask(elem_bits=8)
            m16 = reg.full_mask(elem_bits=16)
            c8 = reg.mask_counter(n)
            hist0 = reg.vdups(0, dtypes.uint16, mask=m16)
            hist1 = reg.vdups(0, dtypes.uint16, mask=m16)
            cum0 = reg.vdups(0, dtypes.uint16, mask=m16)
            cum1 = reg.vdups(0, dtypes.uint16, mask=m16)
            for b in cannbotdsl.range(nblk):
                bm, c8 = reg.update_mask(c8, elem_bits=8)
                val = reg.vload(src_slot, b * VL8)
                hist0 = reg.vadd(
                    hist0, reg.vhistogram_frequency(val, mask=bm, bin=0), mask=m16
                )
                hist1 = reg.vadd(
                    hist1, reg.vhistogram_frequency(val, mask=bm, bin=1), mask=m16
                )
                cum0 = reg.vhistogram_accumulate(cum0, val, mask=bm, bin=0)
                cum1 = reg.vhistogram_accumulate(cum1, val, mask=bm, bin=1)
                reg.vstore(bufs.val, b * VL8, val, bm)
                reg.vstore(
                    bufs.pristine, b * VL8, val, bm
                )  # 源单区域化：后续 pass 全读 pristine
            excl0 = reg.vsub(cum0, hist0, mask=m16)
            excl1 = reg.vsub(cum1, hist1, mask=m16)
            reg.vstore(bufs.excl16, 0, excl0, m16)
            reg.vstore(bufs.excl16, 128, excl1, m16)
            c8i = reg.mask_counter(n)
            for b in cannbotdsl.range(nblk):
                bm, c8i = reg.update_mask(c8i, elem_bits=8)
                reg.vstore(bufs.idx8[0], b * VL8, reg.varange(0, dtypes.uint8), bm)

    @jit
    def pass2_all4(self, bufs: SortBuffers, n: int) -> None:
        """4×2-bit 合一 vf（= ASC EightBitsSort 单 __simd_vf__ 同构，§19.42）。

        原 pass2_round×4 = 8 个 vf 块（每轮 squeeze/置换两块）——core 以 vf
        为调度单位、启动有固定开销（用户领域知识），合并省 7 次启动/字节轮。
        轮内与轮间 RAW 以 vmem_bar 显式同步（陷阱 C 工具 = ASC
        LocalMemBar<VEC_STORE,VEC_LOAD> 等价；AR 组每轮 begin/finalize
        作用域自洽 = ASC ClearSpr<AR> 的组纪律等价）。
        """
        nblk = ceil_div(n, VL8)
        with vf(mode="simd"):
            reg.full_mask(elem_bits=8)
            for r in range_constexpr(4):
                vi = r % 2
                vo = (r + 1) % 2
                # ── 索引 squeeze（原独立 vf-1）──
                ui = reg.vstore_unalign_begin(bufs.idx8[vo])
                c8j = reg.mask_counter(n)
                for b in cannbotdsl.range(nblk):
                    bm, c8j = reg.update_mask(c8j, elem_bits=8)
                    idx = reg.vload(bufs.idx8[vi], b * VL8)
                    val = reg.vload(bufs.val, b * VL8)
                    digit = reg.vbitwise_and(
                        reg.vshr(val, r * 2, mask=bm),
                        reg.vdups(3, dtypes.uint8, mask=bm),
                        mask=bm,
                    )
                    m0 = reg.veqs(digit, 0, mask=bm)
                    m1 = reg.veqs(digit, 1, mask=bm)
                    m2 = reg.veqs(digit, 2, mask=bm)
                    m3 = reg.veqs(digit, 3, mask=bm)
                    reg.vsqueeze_and_storeunalign(
                        bufs.idx8[vo],
                        0,
                        reg.vsqueeze_and_storeunalign_init(idx, mask=m0),
                        ui,
                    )
                    reg.vsqueeze_and_storeunalign(
                        bufs.idx8[vo],
                        0,
                        reg.vsqueeze_and_storeunalign_init(idx, mask=m1),
                        ui,
                    )
                    reg.vsqueeze_and_storeunalign(
                        bufs.idx8[vo],
                        0,
                        reg.vsqueeze_and_storeunalign_init(idx, mask=m2),
                        ui,
                    )
                    reg.vsqueeze_and_storeunalign(
                        bufs.idx8[vo],
                        0,
                        reg.vsqueeze_and_storeunalign_init(idx, mask=m3),
                        ui,
                    )
                reg.vsqueeze_and_storeunalign_finalize(bufs.idx8[vo], 0, ui)
                reg.vmem_bar("vst_vld")  # squeeze 写 idx8[vo] → 置换读
                # ── 值置换（从 pristine gather；原独立 vf-2）──
                c8p = reg.mask_counter(n)
                for b in cannbotdsl.range(nblk):
                    bm, c8p = reg.update_mask(c8p, elem_bits=8)
                    orig = reg.vload(bufs.pristine, b * VL8)
                    idx = reg.vload(bufs.idx8[vo], b * VL8)
                    perm = reg.vgather_reg(orig, idx)
                    reg.vstore(bufs.val, b * VL8, perm, bm)
                # §19.63：val 原地——相位 A 只读/相位 B 只写（vmem_bar 分隔），
                # 子轮末 bar 保下轮 digit RAW（与 idx8 乒乓 RAW 同构）
                reg.vmem_bar(
                    "vst_vld"
                )  # 轮间 RAW（idx8 vi↔vo；val 置换写→下轮 digit 读）

    @jit
    def pass2_round(self, bufs: SortBuffers, r: int, n: int) -> None:
        """一轮 2-bit 排序：索引单 squeeze + 值置换（= AscendC EightBitsSort 单轮）。"""
        nblk = ceil_div(n, VL8)  # 静态 n 折叠为 Python 值；运行时 n 设备侧计算
        vi = r % 2
        vo = (r + 1) % 2
        with vf(mode="simd"):
            reg.full_mask(elem_bits=8)
            ui = reg.vstore_unalign_begin(bufs.idx8[vo])
            c8j = reg.mask_counter(n)
            for b in cannbotdsl.range(nblk):
                bm, c8j = reg.update_mask(c8j, elem_bits=8)
                idx = reg.vload(bufs.idx8[vi], b * VL8)
                val = reg.vload(bufs.val, b * VL8)
                digit = reg.vbitwise_and(
                    reg.vshr(val, r * 2, mask=bm),
                    reg.vdups(3, dtypes.uint8, mask=bm),
                    mask=bm,
                )
                m0 = reg.veqs(digit, 0, mask=bm)
                m1 = reg.veqs(digit, 1, mask=bm)
                m2 = reg.veqs(digit, 2, mask=bm)
                m3 = reg.veqs(digit, 3, mask=bm)
                reg.vsqueeze_and_storeunalign(
                    bufs.idx8[vo],
                    0,
                    reg.vsqueeze_and_storeunalign_init(idx, mask=m0),
                    ui,
                )
                reg.vsqueeze_and_storeunalign(
                    bufs.idx8[vo],
                    0,
                    reg.vsqueeze_and_storeunalign_init(idx, mask=m1),
                    ui,
                )
                reg.vsqueeze_and_storeunalign(
                    bufs.idx8[vo],
                    0,
                    reg.vsqueeze_and_storeunalign_init(idx, mask=m2),
                    ui,
                )
                reg.vsqueeze_and_storeunalign(
                    bufs.idx8[vo],
                    0,
                    reg.vsqueeze_and_storeunalign_init(idx, mask=m3),
                    ui,
                )
            reg.vsqueeze_and_storeunalign_finalize(bufs.idx8[vo], 0, ui)
        # 值置换（从 pristine gather——索引域为原始位置；掩码存储 = StoreAlign(maskReg)）
        with vf(mode="simd"):
            reg.full_mask(elem_bits=8)
            c8p = reg.mask_counter(n)
            for b in cannbotdsl.range(nblk):
                bm, c8p = reg.update_mask(c8p, elem_bits=8)
                orig = reg.vload(bufs.pristine, b * VL8)
                idx = reg.vload(bufs.idx8[vo], b * VL8)
                perm = reg.vgather_reg(orig, idx)
                reg.vstore(bufs.val, b * VL8, perm, bm)

    @jit
    def pass3(self, bufs: SortBuffers, out_idx, n) -> None:
        """跨块合并（= AscendC LocalSort，diff 表恒等变形）；argsort → out_idx。"""
        nblk = ceil_div(n, VL8)  # 静态 n 折叠为 Python 值；运行时 n 设备侧计算
        cnt_sc = self.index_dtype
        with vf(mode="simd"):
            reg.full_mask(elem_bits=8)
            m16 = reg.full_mask(elem_bits=16)
            zero16 = reg.vdups(0, dtypes.uint16, mask=m16)
            bo0 = reg.vload(bufs.excl16, 0)
            bo1 = reg.vload(bufs.excl16, 128)
            c8r = reg.mask_counter(n)
            csc = reg.mask_counter(n)
            for b in cannbotdsl.range(nblk):
                bm, c8r = reg.update_mask(c8r, elem_bits=8)
                val = reg.vload(bufs.val, b * VL8)
                idx8 = reg.vload(bufs.idx8[0], b * VL8)
                hist0 = reg.vhistogram_frequency(val, mask=bm, bin=0)
                hist1 = reg.vhistogram_frequency(val, mask=bm, bin=1)
                cum0 = reg.vhistogram_accumulate(zero16, val, mask=bm, bin=0)
                cum1 = reg.vhistogram_accumulate(zero16, val, mask=bm, bin=1)
                loff0 = reg.vsub(cum0, hist0, mask=m16)
                loff1 = reg.vsub(cum1, hist1, mask=m16)
                diff0 = reg.vsub(bo0, loff0, mask=m16)
                diff1 = reg.vsub(bo1, loff1, mask=m16)
                d8a = reg.vreinterpret_lanes(diff0, dtypes.uint8)
                d8b = reg.vreinterpret_lanes(diff1, dtypes.uint8)
                lo8, hi8 = reg.vdeinterleave(d8a, d8b)
                g_lo = reg.vgather_reg(lo8, val)
                g_hi = reg.vgather_reg(hi8, val)
                r0, r1 = reg.vinterleave(g_lo, g_hi)
                diffk0 = reg.vreinterpret_lanes(r0, dtypes.uint16)
                diffk1 = reg.vreinterpret_lanes(r1, dtypes.uint16)
                pos0 = reg.varange(0, dtypes.uint16)
                pos1 = reg.vadd(
                    reg.varange(0, dtypes.uint16),
                    reg.vdups(128, dtypes.uint16, mask=m16),
                    mask=m16,
                )
                gpos0 = reg.vadd(diffk0, pos0, mask=m16)
                gpos1 = reg.vadd(diffk1, pos1, mask=m16)
                i16l = reg.vunpack(idx8, dtypes.uint16, part="lower")
                i16h = reg.vunpack(idx8, dtypes.uint16, part="upper")
                off16 = reg.vdups(b * VL8, dtypes.uint16, mask=m16)
                gidx0 = reg.vadd(i16l, off16, mask=m16)
                gidx1 = reg.vadd(i16h, off16, mask=m16)
                if const_expr(cnt_sc == "u16"):
                    mc0, csc = reg.update_mask(csc, elem_bits=16)
                    reg.vscatter(out_idx, gidx0, gpos0, mask=mc0)
                    mc1, csc = reg.update_mask(csc, elem_bits=16)
                    reg.vscatter(out_idx, gidx1, gpos1, mask=mc1)
                else:
                    gps = [
                        reg.vunpack(gpos0, dtypes.uint32, part="lower"),
                        reg.vunpack(gpos0, dtypes.uint32, part="upper"),
                        reg.vunpack(gpos1, dtypes.uint32, part="lower"),
                        reg.vunpack(gpos1, dtypes.uint32, part="upper"),
                    ]
                    gis = [
                        reg.vunpack(gidx0, dtypes.uint32, part="lower"),
                        reg.vunpack(gidx0, dtypes.uint32, part="upper"),
                        reg.vunpack(gidx1, dtypes.uint32, part="lower"),
                        reg.vunpack(gidx1, dtypes.uint32, part="upper"),
                    ]
                    for d in cannbotdsl.range_constexpr(4):
                        md, csc = reg.update_mask(csc, elem_bits=32)
                        reg.vscatter(out_idx, gis[d], gps[d], mask=md)
                bo0 = reg.vadd(bo0, hist0, mask=m16)
                bo1 = reg.vadd(bo1, hist1, mask=m16)

    @jit
    def pass4(self, bufs: SortBuffers, out_idx, out_val, n) -> None:
        """值输出 gather-permute（= AscendC UpdateValueByLocalIndex）；源 = pristine。"""
        nblk = ceil_div(n, VL8)  # 静态 n 折叠为 Python 值；运行时 n 设备侧计算
        cnt_sc = self.index_dtype
        with vf(mode="simd"):
            reg.full_mask(elem_bits=8)
            reg.full_mask(elem_bits=16)
            c8o = reg.mask_counter(n)
            c16o = reg.mask_counter(n)
            for b in cannbotdsl.range(nblk):
                if const_expr(cnt_sc == "u16"):
                    a0 = reg.vload(out_idx, b * VL8)
                    a1 = reg.vload(out_idx, b * VL8 + 128)
                else:
                    q0 = reg.vload(out_idx, b * VL8 + 0)
                    q1 = reg.vload(out_idx, b * VL8 + 64)
                    q2 = reg.vload(out_idx, b * VL8 + 128)
                    q3 = reg.vload(out_idx, b * VL8 + 192)
                    a0, _z0 = reg.vdeinterleave(
                        reg.vreinterpret_lanes(q0, dtypes.uint16),
                        reg.vreinterpret_lanes(q1, dtypes.uint16),
                    )
                    a1, _z1 = reg.vdeinterleave(
                        reg.vreinterpret_lanes(q2, dtypes.uint16),
                        reg.vreinterpret_lanes(q3, dtypes.uint16),
                    )
                mc0, c16o = reg.update_mask(c16o, elem_bits=16)
                g0 = reg.vgather(bufs.pristine, a0, mask=mc0)
                mc1, c16o = reg.update_mask(c16o, elem_bits=16)
                g1 = reg.vgather(bufs.pristine, a1, mask=mc1)
                r0 = reg.vreinterpret_lanes(g0, dtypes.uint8)
                r1 = reg.vreinterpret_lanes(g1, dtypes.uint8)
                vals, _junk = reg.vdeinterleave(r0, r1)
                bm, c8o = reg.update_mask(c8o, elem_bits=8)
                reg.vstore(out_val, b * VL8, vals, bm)

    # ══════════ full 模式（全 dtype 单核整排序，= SortImpl 多字节路径）══════════

    def declare_buffers_full(
        self, npad: int, ds: int, vdtype=None, idx_out=None
    ) -> "FullSortBuffers":
        """full 模式内部缓冲（须在 vf 块外调用；npad 为 256 倍数）。

        = ASC ArrangeCommonTmpBuffer + 追加区同构（§19.6）：
        val=srcU8（§19.63 单 buffer 原地 = ASC srcU8：相位 A digit 读/
        相位 B gather 回写同一块）/ pristine=srcU8Copy（§19.62：
        sidx 区前 1/4 字节 u8 别名视图 = ASC srcValueU8Back ×
        sortedLocalIndex 同构——原独立 +1B/elem 偏差解除，K77 生命周期/
        WAR 验证）/ idx8 乒乓（= tmpIndexU8/16）/ excl16 /
        sidx=sortedLocalIndex / next_idx=sortedIndexNextRound /
        val_a=srcForNextRound（原生 dtype）/ val_b=dst（原生 dtype，终态值）/
        idx_out=dstIndex（u32，终态索引）。
        idx_out：外部视图（§19.61 idx64 半区布局——双倍 buffer 后半 u32
        窗口；None = 常规分配）。
        """
        vdt = (
            vdtype
            if vdtype is not None
            else {2: dtypes.int16, 4: dtypes.int32, 8: dtypes.int64}[ds]
        )
        # §19.62：pristine 别名进 sidx 区（生命周期不相交：pristine 活
        # [pass1_full, pass2 末]，sidx 活 [pass3, pass5f 末]）
        sidx = Buffer(MemLoc.UB, dtypes.uint32, make_layout((npad,)))
        return FullSortBuffers(
            # §19.63：val 单 buffer 原地（= ASC srcU8；K78 验证）
            val=Buffer(MemLoc.UB, dtypes.uint8, make_layout((npad,))),
            idx8=[
                Buffer(MemLoc.UB, dtypes.uint8, make_layout((npad,))),
                Buffer(MemLoc.UB, dtypes.uint8, make_layout((npad,))),
            ],
            pristine=sidx.view(dtype=dtypes.uint8),
            excl16=Buffer(MemLoc.UB, dtypes.uint16, make_layout((256,))),
            sidx=sidx,
            next_idx=Buffer(MemLoc.UB, dtypes.uint32, make_layout((npad,))),
            idx_out=(
                idx_out
                if idx_out is not None
                else Buffer(MemLoc.UB, dtypes.uint32, make_layout((npad,)))
            ),
            val_a=Buffer(MemLoc.UB, vdt, make_layout((npad,))),
            val_b=Buffer(MemLoc.UB, vdt, make_layout((npad,))),
        )

    def sort_full(
        self, bufs: "FullSortBuffers", src_slot, n, ds: int, mask8, nu32_cap=None
    ) -> None:
        """full 模式编排（普通方法，仅调用——@kernel 内 for 皆设备循环教训）。

        = ASC SortImpl<ConvType> 带索引多字节路径逐段同构：
        R0: 提取字节0（源=输入槽直读）→ u8 排序 → R0 索引 → next_idx、
            值重排 → val_a
        R1..ds-1: 提取字节r（源=prev_val）→ u8 排序 → 局部索引 sidx →
                  值&索引双追逐 → curr（奇数轮恒写 val_b/idx_out = 终态输出区）
        mask8: b8 域按轮 XOR 掩码（twiddle 融合，与多核同表；int 表或
        运行期标量表皆可——D1 desc meta 化）。
        终态：bufs.val_b（原生 dtype 排序值）+ bufs.idx_out（u32 索引）。
        """
        self.sort_full_r0(bufs, src_slot, n, ds, mask8, nu32_cap)
        prev_val, curr_val = bufs.val_a, bufs.val_b
        prev_idx, curr_idx = bufs.next_idx, bufs.idx_out
        for r in range(1, ds):
            pv = (
                prev_val
                if ds in (2, 4)
                # §19.38：ds=2 原生直通
                else from_tensor(
                    prev_val,
                    dtype=dtypes.int32,
                    layout=make_layout((nu32_cap or nu32_of(n, ds),)),
                )
            )
            self.pass1_full(pv, bufs, n, ds, r * 8, mask8[r])
            self.pass2_all4(bufs, n)  # §19.42：合一 vf
            self.pass3(bufs, bufs.sidx, n)
            self.pass5f(
                prev_val, prev_idx, bufs.sidx, curr_val, curr_idx, n, ds, nu32_cap
            )
            if r < ds - 1:
                prev_val, curr_val = curr_val, prev_val
                prev_idx, curr_idx = curr_idx, prev_idx

    def debug_pass2_only(self, bufs, src_slot, n, ds, mask8):
        """调试编排：pass1_full + pass2 合一 vf。"""
        self.pass1_full(src_slot, bufs, n, ds, 0, mask8[0])
        self.pass2_all4(bufs, n)

    def debug_r1_head(self, bufs, prev_val, n, ds, mask8, nu32_cap=None):
        """R1 的 pass1_full + pass2 + pass3 → sidx（不含 pass5f，二分用）。"""
        pv = (
            prev_val
            if ds in (2, 4)
            # §19.38：ds=2 原生直通
            else from_tensor(
                prev_val,
                dtype=dtypes.int32,
                layout=make_layout((nu32_cap or nu32_of(n, ds),)),
            )
        )
        self.pass1_full(pv, bufs, n, ds, 8, mask8[1])
        self.pass2_all4(bufs, n)
        self.pass3(bufs, bufs.sidx, n)

    def sort_full_r0(self, bufs, src_slot, n, ds, mask8, nu32_cap=None):
        """R0 段（平铺方法；调试 dump 与 sort_full 共用——@kernel 内 for 皆
        设备循环，R0 的 4 次 pass2 必须在平铺层展开）。

        §19.42 修复：本函数历史上误含 R≥1 循环（自 sort_full 提取时的
        复制残留），叠加 sort_full 自身循环致 R1..ds-1 双重执行（ds=2 跑
        3 轮、ds=4 跑 7 轮、ds=8 跑 15 轮）——正确性被稳定排序幂等性
        掩盖（.asc 24 条 mem_bar = 3×8 实证），1.5-1.9x 冗余功。
        本函数回归纯 R0；R≥1 由 sort_full 负责。
        """
        # R0 直读槽（int32 零转换）；pass4f 槽上 reinterpret（#27 正解：
        # 槽+dtype 变更唯一合法路径，且继承 next_write/next_read 配对同步身份）
        self.pass1_full(src_slot, bufs, n, ds, 0, mask8[0])
        self.pass2_all4(bufs, n)  # §19.42：合一 vf
        self.pass3(bufs, bufs.next_idx, n)
        self.pass4f(src_slot, bufs.next_idx, bufs.val_a, n, ds, nu32_cap)

    @jit
    def pass1_full(self, src, bufs, n, ds: int, shift: int, xormask) -> None:
        """full 模式字节提取 + 直方图（= ASC GetExclusiveSum<ConvType>，含
        ConvertRegToWithShift 三链 + b8 域 XOR 掩码融合）。

        src: R0=输入槽（ds=2 原生 vdt §19.38；ds=4/8 int32）；R≥1 =
        prev_val（ds=2/4 原生直通；ds=8 int32 reinterpret 视图，继承源
        同步身份）；双写 val/pristine = ASC tmpSrc/tmpCopy。
        xormask: b8 域 twiddle 掩码（运行期标量或 int，0 = 恒等；D1）。
        """
        nblk = ceil_div(n, VL8)
        with vf(mode="simd"):
            m16 = reg.full_mask(elem_bits=16)
            m32 = reg.full_mask(elem_bits=32)
            hist0 = reg.vdups(0, dtypes.uint16, mask=m16)
            hist1 = reg.vdups(0, dtypes.uint16, mask=m16)
            cum0 = reg.vdups(0, dtypes.uint16, mask=m16)
            cum1 = reg.vdups(0, dtypes.uint16, mask=m16)
            c8 = reg.mask_counter(n)
            xm8 = reg.vdups(xormask, dtypes.uint8)  # D1：twiddle 运行期化（轮级广播）
            for b in cannbotdsl.range(nblk):
                bm, c8 = reg.update_mask(c8, elem_bits=8)
                if const_expr(ds == 2):
                    # §19.38：原生 u16 lane（= ASC HistogramB16VF 同构；
                    # U1 误判产物 u32 打包域回归原生，§19.31 同款）
                    q0 = reg.vload(src, b * 256 + 0)
                    q1 = reg.vload(src, b * 256 + 128)
                    if const_expr(shift > 0):
                        q0 = reg.vshr(q0, shift, mask=m16)
                        q1 = reg.vshr(q1, shift, mask=m16)
                    b8, _j2 = reg.vdeinterleave(
                        reg.vreinterpret_lanes(q0, dtypes.uint8),
                        reg.vreinterpret_lanes(q1, dtypes.uint8),
                    )
                elif const_expr(ds == 4):
                    q0 = reg.vload(src, b * VL8 + 0 * VL32)
                    q1 = reg.vload(src, b * VL8 + 1 * VL32)
                    q2 = reg.vload(src, b * VL8 + 2 * VL32)
                    q3 = reg.vload(src, b * VL8 + 3 * VL32)
                    if const_expr(shift > 0):
                        q0 = reg.vshr(q0, shift, mask=m32)
                        q1 = reg.vshr(q1, shift, mask=m32)
                        q2 = reg.vshr(q2, shift, mask=m32)
                        q3 = reg.vshr(q3, shift, mask=m32)
                    d01lo, _a = reg.vdeinterleave(
                        reg.vreinterpret_lanes(q0, dtypes.uint16),
                        reg.vreinterpret_lanes(q1, dtypes.uint16),
                    )
                    d23lo, _b = reg.vdeinterleave(
                        reg.vreinterpret_lanes(q2, dtypes.uint16),
                        reg.vreinterpret_lanes(q3, dtypes.uint16),
                    )
                    b8, _j4 = reg.vdeinterleave(
                        reg.vreinterpret_lanes(d01lo, dtypes.uint8),
                        reg.vreinterpret_lanes(d23lo, dtypes.uint8),
                    )
                else:
                    q0 = reg.vload(src, b * 512 + 0 * VL32)
                    q1 = reg.vload(src, b * 512 + 1 * VL32)
                    q2 = reg.vload(src, b * 512 + 2 * VL32)
                    q3 = reg.vload(src, b * 512 + 3 * VL32)
                    q4 = reg.vload(src, b * 512 + 4 * VL32)
                    q5 = reg.vload(src, b * 512 + 5 * VL32)
                    q6 = reg.vload(src, b * 512 + 6 * VL32)
                    q7 = reg.vload(src, b * 512 + 7 * VL32)
                    s2 = shift % 32
                    lo0, hi0 = reg.vdeinterleave(
                        reg.vreinterpret_lanes(q0, dtypes.uint32),
                        reg.vreinterpret_lanes(q1, dtypes.uint32),
                    )
                    lo1, hi1 = reg.vdeinterleave(
                        reg.vreinterpret_lanes(q2, dtypes.uint32),
                        reg.vreinterpret_lanes(q3, dtypes.uint32),
                    )
                    lo2, hi2 = reg.vdeinterleave(
                        reg.vreinterpret_lanes(q4, dtypes.uint32),
                        reg.vreinterpret_lanes(q5, dtypes.uint32),
                    )
                    lo3, hi3 = reg.vdeinterleave(
                        reg.vreinterpret_lanes(q6, dtypes.uint32),
                        reg.vreinterpret_lanes(q7, dtypes.uint32),
                    )
                    if const_expr(shift < 32):
                        w0, w1, w2, w3 = lo0, lo1, lo2, lo3
                    else:
                        w0, w1, w2, w3 = hi0, hi1, hi2, hi3
                    if const_expr(shift % 32 > 0):
                        w0 = reg.vshr(w0, s2, mask=m32)
                        w1 = reg.vshr(w1, s2, mask=m32)
                        w2 = reg.vshr(w2, s2, mask=m32)
                        w3 = reg.vshr(w3, s2, mask=m32)
                    d01lo, _c = reg.vdeinterleave(
                        reg.vreinterpret_lanes(w0, dtypes.uint16),
                        reg.vreinterpret_lanes(w1, dtypes.uint16),
                    )
                    d23lo, _d = reg.vdeinterleave(
                        reg.vreinterpret_lanes(w2, dtypes.uint16),
                        reg.vreinterpret_lanes(w3, dtypes.uint16),
                    )
                    b8, _j8 = reg.vdeinterleave(
                        reg.vreinterpret_lanes(d01lo, dtypes.uint8),
                        reg.vreinterpret_lanes(d23lo, dtypes.uint8),
                    )
                # ── twiddle 融合（b8 域 XOR，§17.6；D1：xm8 运行期，0 = 恒等）──
                b8 = reg.vbitwise_xor(b8, xm8, mask=bm)
                reg.vstore(bufs.val, b * VL8, b8, bm)
                reg.vstore(bufs.pristine, b * VL8, b8, bm)
                hist0 = reg.vadd(
                    hist0, reg.vhistogram_frequency(b8, mask=bm, bin=0), mask=m16
                )
                hist1 = reg.vadd(
                    hist1, reg.vhistogram_frequency(b8, mask=bm, bin=1), mask=m16
                )
                cum0 = reg.vhistogram_accumulate(cum0, b8, mask=bm, bin=0)
                cum1 = reg.vhistogram_accumulate(cum1, b8, mask=bm, bin=1)
            excl0 = reg.vsub(cum0, hist0, mask=m16)
            excl1 = reg.vsub(cum1, hist1, mask=m16)
            reg.vstore(bufs.excl16, 0, excl0, m16)
            reg.vstore(bufs.excl16, 128, excl1, m16)
            c8i = reg.mask_counter(n)
            for b in cannbotdsl.range(nblk):
                bm, c8i = reg.update_mask(c8i, elem_bits=8)
                reg.vstore(bufs.idx8[0], b * VL8, reg.varange(0, dtypes.uint8), bm)

    @jit
    def pass4f(self, src, idx_buf, dst_val, n, ds: int, nu32_cap=None) -> None:
        """R0 值重排（= ASC UpdateValueByLocalIndex<ConvType>）。

        src: R0 输入槽（ds=2 原生 vdt，§19.38）；idx_buf: R0 索引（u32）；
        dst_val: 目标值缓冲（原生 dtype Buffer）。
        ds==2 = u16 版（§19.10 还原 + §19.38 原生槽：reinterpret 退化为
        dtype-only 视图；bm16 单链，ASC 同构；U32MODE g32 备选已删——
        §19.18 A/B 已定案且与原生槽不兼容）。
        """
        if const_expr(ds == 2):
            nrep = ceil_div(n, 128)
            _c = nu32_cap or nu32_of(n, ds)
            src16 = from_tensor(src, dtype=dtypes.uint16, layout=make_layout((2 * _c,)))
            dst16 = from_tensor(
                dst_val, dtype=dtypes.uint16, layout=make_layout((2 * _c,))
            )
            with vf(mode="simd"):
                reg.full_mask(elem_bits=16)
                c16 = reg.mask_counter(n)
                for b in cannbotdsl.range(nrep):
                    bm16, c16 = reg.update_mask(c16, elem_bits=16)
                    q0 = reg.vload(idx_buf, b * 128 + 0)
                    q1 = reg.vload(idx_buf, b * 128 + VL32)
                    i16, _h = reg.vdeinterleave(
                        reg.vreinterpret_lanes(q0, dtypes.uint16),
                        reg.vreinterpret_lanes(q1, dtypes.uint16),
                    )
                    g = reg.vgather(src16, i16, mask=bm16)
                    reg.vstore(dst16, b * 128, g, bm16)
        elif const_expr(ds == 4):
            nrep = ceil_div(n, VL32)
            with vf(mode="simd"):
                m32 = reg.full_mask(elem_bits=32)
                c32 = reg.mask_counter(n)
                for b in cannbotdsl.range(nrep):
                    bm, c32 = reg.update_mask(c32, elem_bits=32)
                    q = reg.vload(idx_buf, b * VL32)
                    g = reg.vgather(src, q, mask=bm)
                    reg.vstore(dst_val, b * VL32, g, bm)
        else:
            # u64：GatherAndStoreB64Elements 同构（u32 域 ×2 gather + interleave + 双 store）
            # ⚠️ 交织数据 mask 倍增（= ASC MaskInterleave）：u64 每元素 2 u32 lane，
            # r0/r1 的存储掩码用 2n 计数器链（每迭代 2 次 update_mask 供 r0/r1）
            nrep = ceil_div(n, VL32)
            _c = nu32_cap or nu32_of(n, ds)
            src32 = from_tensor(src, dtype=dtypes.uint32, layout=make_layout((_c,)))
            dst32 = from_tensor(dst_val, dtype=dtypes.uint32, layout=make_layout((_c,)))
            with vf(mode="simd"):
                m32 = reg.full_mask(elem_bits=32)
                c32 = reg.mask_counter(n)
                c64 = reg.mask_counter(2 * n)
                ones = reg.vdups(1, dtypes.uint32, mask=m32)
                for b in cannbotdsl.range(nrep):
                    bm, c32 = reg.update_mask(c32, elem_bits=32)
                    ml, c64 = reg.update_mask(c64, elem_bits=32)
                    mh, c64 = reg.update_mask(c64, elem_bits=32)
                    q = reg.vload(idx_buf, b * VL32)
                    q2 = reg.vadd(q, q, mask=bm)
                    q2p1 = reg.vadd(q2, ones, mask=bm)
                    lo = reg.vgather(src32, q2, mask=bm)
                    hi = reg.vgather(src32, q2p1, mask=bm)
                    r0, r1 = reg.vinterleave(lo, hi)
                    reg.vstore(dst32, b * 2 * VL32 + 0, r0, ml)
                    reg.vstore(dst32, b * 2 * VL32 + VL32, r1, mh)

    @jit
    def pass5f(
        self, prev_val, prev_idx, sidx, curr_val, curr_idx, n, ds: int, nu32_cap=None
    ) -> None:
        """R≥1 值&索引双追逐（= ASC UpdateValueAndIndexByLocalIndex<ConvType,u32>）。

        值按 ds 分支（同 pass4f，源=prev_val）；索引恒 u32 直 gather。
        ds==2 = u16 修复版（§19.10：唯一 c16 链 + reg.vinterleave(bm16,
        allf16)=pintlv_b16 派生索引双掩码，ASC u16 重载逐句同构）。
        U32MODE 备选已删（§19.20：§19.18 A/B 定案后零收益）。
        """
        if const_expr(ds == 2):
            nrep = ceil_div(n, 128)
            _c = nu32_cap or nu32_of(n, ds)
            pv16 = from_tensor(
                prev_val, dtype=dtypes.uint16, layout=make_layout((2 * _c,))
            )
            cv16 = from_tensor(
                curr_val, dtype=dtypes.uint16, layout=make_layout((2 * _c,))
            )
            with vf(mode="simd"):
                reg.full_mask(elem_bits=16)
                allf16 = reg.create_mask(pattern="allf", elem_bits=16)
                c16 = reg.mask_counter(n)
                for b in cannbotdsl.range(nrep):
                    bm16, c16 = reg.update_mask(c16, elem_bits=16)
                    bm_a, bm_b = reg.vinterleave(bm16, allf16)  # pintlv_b16 派生
                    q0 = reg.vload(sidx, b * 128 + 0)
                    q1 = reg.vload(sidx, b * 128 + VL32)
                    i16, _h = reg.vdeinterleave(
                        reg.vreinterpret_lanes(q0, dtypes.uint16),
                        reg.vreinterpret_lanes(q1, dtypes.uint16),
                    )
                    g = reg.vgather(pv16, i16, mask=bm16)
                    reg.vstore(cv16, b * 128, g, bm16)
                    gi0 = reg.vgather(prev_idx, q0, mask=bm_a)
                    reg.vstore(curr_idx, b * 128 + 0, gi0, bm_a)
                    gi1 = reg.vgather(prev_idx, q1, mask=bm_b)
                    reg.vstore(curr_idx, b * 128 + VL32, gi1, bm_b)
            return
        elif const_expr(ds == 4):
            nrep = ceil_div(n, VL32)
            with vf(mode="simd"):
                m32 = reg.full_mask(elem_bits=32)
                c32 = reg.mask_counter(n)
                for b in cannbotdsl.range(nrep):
                    bm, c32 = reg.update_mask(c32, elem_bits=32)
                    q = reg.vload(sidx, b * VL32)
                    g = reg.vgather(prev_val, q, mask=bm)
                    reg.vstore(curr_val, b * VL32, g, bm)
                    gi = reg.vgather(prev_idx, q, mask=bm)
                    reg.vstore(curr_idx, b * VL32, gi, bm)
        else:
            nrep = ceil_div(n, VL32)
            _c = nu32_cap or nu32_of(n, ds)
            pv32 = from_tensor(prev_val, dtype=dtypes.uint32, layout=make_layout((_c,)))
            cv32 = from_tensor(curr_val, dtype=dtypes.uint32, layout=make_layout((_c,)))
            with vf(mode="simd"):
                m32 = reg.full_mask(elem_bits=32)
                c32 = reg.mask_counter(n)
                c64 = reg.mask_counter(2 * n)
                ones = reg.vdups(1, dtypes.uint32, mask=m32)
                for b in cannbotdsl.range(nrep):
                    bm, c32 = reg.update_mask(c32, elem_bits=32)
                    ml, c64 = reg.update_mask(c64, elem_bits=32)
                    mh, c64 = reg.update_mask(c64, elem_bits=32)
                    q = reg.vload(sidx, b * VL32)
                    q2 = reg.vadd(q, q, mask=bm)
                    q2p1 = reg.vadd(q2, ones, mask=bm)
                    lo = reg.vgather(pv32, q2, mask=bm)
                    hi = reg.vgather(pv32, q2p1, mask=bm)
                    r0, r1 = reg.vinterleave(lo, hi)
                    reg.vstore(cv32, b * 2 * VL32 + 0, r0, ml)
                    reg.vstore(cv32, b * 2 * VL32 + VL32, r1, mh)
                    gi = reg.vgather(prev_idx, q, mask=bm)
                    reg.vstore(curr_idx, b * VL32, gi, bm)

    # ── host 侧接口 ─────────────────────────────────────────────

    @staticmethod
    def ub_requirement(index_dtype: str, npad: int, io_depth: int = 1) -> int:
        """给定垫长元素数，返回典型布局 UB 占用（字节）。组件唯一容量接口，tiling 预算用。

        组件内部（val/idx8 ping-pong 4B/元素 + pristine 1B/元素 + excl16）+
        调用方 I/O 通道（in 1B + out_val 1B + out_idx idx_bytes B，各 ×io_depth）。
        io_depth=1（串行，v1）/ 2（流水，v2）。

        tiling 求逆（tile 上限属 tiling 层职责，组件无 max_count/assert_count）：
            per_elem = 5 + io_depth * (2 + idx_bytes)     # idx_bytes: u16→2, u32→4
            max_np = (get_mem_size("ub") - FIXED_OVERHEAD) // per_elem
            tile = max_np // VL8 * VL8                    # 对齐块粒度
        """
        if index_dtype not in ("u16", "u32"):
            raise ValueError(index_dtype)
        idx_bytes = 2 if index_dtype == "u16" else 4
        internal = (
            4 * npad + FIXED_OVERHEAD
        )  # val 1（§19.63 原地）+ idx8 2 + pristine 1
        io = io_depth * (1 + 1 + idx_bytes) * npad
        return internal + io


def ub_requirement_full(npad: int, ds: int, index_size: int = 4) -> int:
    """full 模式 UB 字节需求（含装载通道与输出区）。

    = ASC TrySingleCore 装载方程同构（§20 修正）：ch ds + val 1（§19.63
    单 buffer 原地）+ idx8×2 2 + sidx 4（pristine §19.62 别名入内，不再
    独立）+ next_idx 4 + idx_out 4 + val_a ds + val_b ds + excl16 512 =
    npad·(3ds+15) + 512。
    ⚠️ §19.6/§19.8 原推导漏计 val[1]（3ds+16）
    ub_storage=223232 = 7680·29+512 证伪；阈值邻域 launch 107000 即此。
    ASC 为 3ds+15 —— **§19.63 全对齐**（审计 B2 闭环）。
    §19.61：index_size=8 时 idx_out 双倍（半区布局，= ASC idxTotal ×2
    for i64）→ +4B/元素（3ds+19）。
    """
    return npad * (3 * ds + 15 + (4 if index_size == 8 else 0)) + 512


def nu32_of(n, ds):
    """full 模式 u32 视图窗口（= npad·ds//4，ceil256 对齐；reinterpret
    静态 shape 用，窗口须 ≤ 每个背衬分配——单核 N 烘焙为 Python 常量）。
    """
    return (n + VL8 - 1) // VL8 * VL8 * ds // 4
