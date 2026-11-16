# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""radix_sort_simd P2/P3 相位库——AscendC sort_lib_vf.h 的 @jit 对应物。

结构约定（设计文档 §15.5 cookbook + §16/§17 限制矩阵）：
- 每相位一个 @jit（核内设备循环/控制流唯一合法形态）
- kernel 级 tile 循环（for 包裹）+ 相位间手写模式 0 三件套屏障
- 跨核 GM 访问：轮询/散射读 = l2cache_load，散射写 = l2cache_store/plain store
- u32 计数段的原子目的以 int32 视图承载（§15.4-#8）

字节提取链（§17.6 统一泛化；§19.31 后 ds=2 为原生 u16 域，其余 u32 打包域）：
  2B  原生 u16（= ASC HistogramB16VF）：q0,q1 → shift(u16) → reinterpret u8
      → deinterleave.even = 每元素字节 r
  4B  双层：q0..q3 → shift → deinterleave(u16) → deinterleave(u8)
  8B  分流+双层：q0..q7 → deinterleave(u32) 分 lo/hi 流 → 4B 链取字节 r%4
twiddle 融合（§17.6）：任意 dtype/方向的键变换 ≡ 提取字节后 XOR 常量掩码
mask8[r]（asc-signed: 末轮 0x80；desc-signed: 非末轮 0xFF/末轮 0x7F；
desc-unsigned: 0xFF；asc-unsigned: 0）。
"""

from cannbotdsl.ops import reg
import cannbotdsl
from cannbotdsl import ceil_div, dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.vf import vf

VL8 = 256  # u8 每 vreg lane 数（1 块 = 256 元素）
VL16 = 128  # u16 每 vreg lane 数（§19.31 ds=2 原生装载）
VL32 = 64  # u32 每 vreg lane 数
AGG_FLAG = 2**30  # 就绪标志（int32 位级等价 ASC AGGREGATE_READY_MASK）


@jit
def publish_agg_pass(hist32, slot):
    """AGG 标志发布：hist32 + 2^30 → int32 槽（调用方 mem_copy 至 ws[1] 行）。

    = ASC ScatterBlockHist2Global 的 EncodeHistVF + CopyUbToGm（MTE3 发布，
    T4-bare 验证的跨核可见通路）。末 tile 行无人回看，守卫省略（无害偏差）。
    """
    with vf(mode="simd"):
        m32 = reg.full_mask(elem_bits=32)
        f = reg.vdups(AGG_FLAG, dtypes.int32, mask=m32)
        reg.vstore(slot, 0, reg.vadd(reg.vload(hist32, 0), f, mask=m32), m32)
        reg.vstore(slot, 64, reg.vadd(reg.vload(hist32, 64), f, mask=m32), m32)
        reg.vstore(slot, 128, reg.vadd(reg.vload(hist32, 128), f, mask=m32), m32)
        reg.vstore(slot, 192, reg.vadd(reg.vload(hist32, 192), f, mask=m32), m32)


@jit
def widen_hist_pass(hist_ub, cumsum_ub, hist32, cumsum32):
    """u16 hist/cumsum（256）宽化为 int32 UB 缓冲（SIMT 消费 + pos 表用）。

    Interleave(x, 0) 宽化 = ASC EncodeHistVF 的前半（无标志位）。
    """
    with vf(mode="simd"):
        m16 = reg.full_mask(elem_bits=16)
        m32 = reg.full_mask(elem_bits=32)
        z = reg.vdups(0, dtypes.uint16, mask=m16)
        h0 = reg.vload(hist_ub, 0)
        h1 = reg.vload(hist_ub, 128)
        c0 = reg.vload(cumsum_ub, 0)
        c1 = reg.vload(cumsum_ub, 128)
        hw00, hw01 = reg.vinterleave(h0, z)
        hw10, hw11 = reg.vinterleave(h1, z)
        cw00, cw01 = reg.vinterleave(c0, z)
        cw10, cw11 = reg.vinterleave(c1, z)
        reg.vstore(hist32, 0, reg.vreinterpret_lanes(hw00, dtypes.int32), m32)
        reg.vstore(hist32, 64, reg.vreinterpret_lanes(hw01, dtypes.int32), m32)
        reg.vstore(hist32, 128, reg.vreinterpret_lanes(hw10, dtypes.int32), m32)
        reg.vstore(hist32, 192, reg.vreinterpret_lanes(hw11, dtypes.int32), m32)
        reg.vstore(cumsum32, 0, reg.vreinterpret_lanes(cw00, dtypes.int32), m32)
        reg.vstore(cumsum32, 64, reg.vreinterpret_lanes(cw01, dtypes.int32), m32)
        reg.vstore(cumsum32, 128, reg.vreinterpret_lanes(cw10, dtypes.int32), m32)
        reg.vstore(cumsum32, 192, reg.vreinterpret_lanes(cw11, dtypes.int32), m32)


@jit
def prescan_pass(slot, b8_stage, hist_ub, cumsum_ub, excl_acc, n, ds, shift, xormask):
    """单 tile 预扫描（= ASC HistogramB{16,32,64}VF + ComputeSumChist 泛化）。

    slot: 通道槽——ds=2 原生 (TILE,) vdt（§19.31）；其余 u32 打包
    (TILE*ds//4,) int32；n: 逻辑元素数；
    ds: 值字节数（2/4/8，烘焙）；shift: 本轮字节位 r*8（烘焙）；
    xormask: b8 域 twiddle 掩码（运行期标量或 int，0 = 恒等；D1 desc meta 化）。
    提取链内联（@jit 嵌套调用未验证，§17.2-#16 教训）。
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
        xm8 = reg.vdups(
            xormask, dtypes.uint8
        )  # D1：twiddle 运行期化（轮级广播，探针 B）
        for i in cannbotdsl.range(nblk):
            bm, c8 = reg.update_mask(c8, elem_bits=8)
            # ── 字节提取（按 ds 分链，全部内联）──
            if const_expr(ds == 2):
                # 原生 u16 lane（= ASC HistogramB16VF：LoadAlign<u16> +
                # ShiftRights<u16, maskB16>；§19.31 U1 翻案后回归——
                # 原 u32 打包域为 U1 误判产物，指令数等价但形态对齐）
                q0 = reg.vload(slot, i * 256 + 0 * VL16)
                q1 = reg.vload(slot, i * 256 + 1 * VL16)
                if const_expr(shift > 0):
                    q0 = reg.vshr(q0, shift, mask=m16)
                    q1 = reg.vshr(q1, shift, mask=m16)
                b8, _j2 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(q0, dtypes.uint8),
                    reg.vreinterpret_lanes(q1, dtypes.uint8),
                )
            elif const_expr(ds == 4):
                q0 = reg.vload(slot, i * VL8 + 0 * VL32)
                q1 = reg.vload(slot, i * VL8 + 1 * VL32)
                q2 = reg.vload(slot, i * VL8 + 2 * VL32)
                q3 = reg.vload(slot, i * VL8 + 3 * VL32)
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
                # 8B：8×u32 → deinterleave 分 lo/hi 流 → 4B 链取字节 r%4
                q0 = reg.vload(slot, i * 512 + 0 * VL32)
                q1 = reg.vload(slot, i * 512 + 1 * VL32)
                q2 = reg.vload(slot, i * 512 + 2 * VL32)
                q3 = reg.vload(slot, i * 512 + 3 * VL32)
                q4 = reg.vload(slot, i * 512 + 4 * VL32)
                q5 = reg.vload(slot, i * 512 + 5 * VL32)
                q6 = reg.vload(slot, i * 512 + 6 * VL32)
                q7 = reg.vload(slot, i * 512 + 7 * VL32)
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
                    s2 = shift
                    w0, w1, w2, w3 = lo0, lo1, lo2, lo3
                else:
                    s2 = shift - 32
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
            reg.vstore(b8_stage, i * VL8, b8, bm)
            hist0 = reg.vadd(
                hist0, reg.vhistogram_frequency(b8, mask=bm, bin=0), mask=m16
            )
            hist1 = reg.vadd(
                hist1, reg.vhistogram_frequency(b8, mask=bm, bin=1), mask=m16
            )
            cum0 = reg.vhistogram_accumulate(cum0, b8, mask=bm, bin=0)
            cum1 = reg.vhistogram_accumulate(cum1, b8, mask=bm, bin=1)
        # tile 排他前缀 = 包含前缀 − 本 tile 计数（= ASC exclusiveSum = chist − hist）
        es0 = reg.vsub(cum0, hist0, mask=m16)
        es1 = reg.vsub(cum1, hist1, mask=m16)
        reg.vstore(hist_ub, 0, hist0, m16)
        reg.vstore(hist_ub, 128, hist1, m16)
        reg.vstore(cumsum_ub, 0, es0, m16)
        reg.vstore(cumsum_ub, 128, es1, m16)
        # 宽化（Interleave(es, 0) = ASC 同构）+ 累积进 blockExcl（int32 计数承载）
        z = reg.vdups(0, dtypes.uint16, mask=m16)
        w00, w01 = reg.vinterleave(es0, z)
        w10, w11 = reg.vinterleave(es1, z)
        e0 = reg.vload(excl_acc, 0 * VL32)
        e1 = reg.vload(excl_acc, 1 * VL32)
        e2 = reg.vload(excl_acc, 2 * VL32)
        e3 = reg.vload(excl_acc, 3 * VL32)
        e0 = reg.vadd(e0, reg.vreinterpret_lanes(w00, dtypes.int32), mask=m32)
        e1 = reg.vadd(e1, reg.vreinterpret_lanes(w01, dtypes.int32), mask=m32)
        e2 = reg.vadd(e2, reg.vreinterpret_lanes(w10, dtypes.int32), mask=m32)
        e3 = reg.vadd(e3, reg.vreinterpret_lanes(w11, dtypes.int32), mask=m32)
        reg.vstore(excl_acc, 0 * VL32, e0, m32)
        reg.vstore(excl_acc, 1 * VL32, e1, m32)
        reg.vstore(excl_acc, 2 * VL32, e2, m32)
        reg.vstore(excl_acc, 3 * VL32, e3, m32)


# ══════════ i64 计数变体（§19.21，N > 2^30 路径；探针 P1a/P1b 实证）══════════
# DSL 限制 #32/#33 规避：vdups-int64 缺失 → interleave+reinterpret 常量构造；
# MTE atomic-int64 缺失 → 调用方以 simt.atomic_add 替代（excl 发布段）。

AGG_FLAG_B64 = 2**62  # = ASC AGGREGATE_READY_MASK_B64（bit62，计数 62 位）


def _dup_i64(value_lo32, value_hi32):
    """i64 常量向量构造（vdups-int64 缺失的 workaround，P1b 实证）。

    interleave(lo32, hi32) → [lo,hi,lo,hi...] → reinterpret i64 →
    每 lane = (hi<<32)|lo。仅可用全掩码（常量广播无尾块）。
    """
    m32 = reg.full_mask(elem_bits=32)
    lo = reg.vdups(value_lo32 & 0xFFFFFFFF, dtypes.int32, mask=m32)
    hi = reg.vdups(value_hi32 & 0xFFFFFFFF, dtypes.int32, mask=m32)
    r0, _r1 = reg.vinterleave(lo, hi)
    return reg.vreinterpret_lanes(r0, dtypes.int64)


@jit
def publish_agg_pass_c8(hist64, slot64):
    """AGG 标志发布（i64 计数版）：hist64 + 2^62 → int64 槽。

    256 bins = 8× i64 寄存器（32 lane/寄存器）。vadd-int64 + B64 掩码
    均为 P1a 实证支持项。
    """
    with vf(mode="simd"):
        m64 = reg.full_mask(elem_bits=64)
        f = _dup_i64(0, 2**30)  # 2^62 = (2^30 << 32) | 0
        for k in range(8):
            reg.vstore(
                slot64, k * 32, reg.vadd(reg.vload(hist64, k * 32), f, mask=m64), m64
            )


@jit
def widen_hist_pass_c8(hist_ub, cumsum_ub, hist64, cum64):
    """u16 hist/cumsum（256）值扩展宽化为 int64（双重 interleave，P5 推导）。

    u16 --interleave(0)--> u32 --interleave(0)--> i64，每级位宽翻倍、
    值不变（= widen_hist_pass 的 i64 同构，ASC EncodeHistVF 前半）。
    256 bins：每源寄存器（128 u16）→ 4 i64 寄存器，共 8 store。
    """
    with vf(mode="simd"):
        m16 = reg.full_mask(elem_bits=16)
        m32 = reg.full_mask(elem_bits=32)
        m64 = reg.full_mask(elem_bits=64)
        z16 = reg.vdups(0, dtypes.uint16, mask=m16)
        z32 = reg.vdups(0, dtypes.int32, mask=m32)

        def _w2_64(v, dst, base):
            r0, r1 = reg.vinterleave(v, z16)  # → 2×64 u32 值
            w = reg.vreinterpret_lanes(r0, dtypes.int32)
            s0, s1 = reg.vinterleave(w, z32)  # → 2×32 i64 值
            reg.vstore(dst, base, reg.vreinterpret_lanes(s0, dtypes.int64), m64)
            reg.vstore(dst, base + 32, reg.vreinterpret_lanes(s1, dtypes.int64), m64)
            w = reg.vreinterpret_lanes(r1, dtypes.int32)
            s0, s1 = reg.vinterleave(w, z32)
            reg.vstore(dst, base + 64, reg.vreinterpret_lanes(s0, dtypes.int64), m64)
            reg.vstore(dst, base + 96, reg.vreinterpret_lanes(s1, dtypes.int64), m64)

        _w2_64(reg.vload(hist_ub, 0), hist64, 0)
        _w2_64(reg.vload(hist_ub, 128), hist64, 128)
        _w2_64(reg.vload(cumsum_ub, 0), cum64, 0)
        _w2_64(reg.vload(cumsum_ub, 128), cum64, 128)


@jit
def prescan_pass_c8(
    slot, b8_stage, hist_ub, cumsum_ub, excl_acc, n, ds, shift, xormask
):
    """单 tile 预扫描（i64 计数版）——提取链/直方图与 u32 版全同，
    仅尾部排他前缀累积进 int64 excl_acc（双重 interleave 宽化）。
    """
    nblk = ceil_div(n, VL8)
    with vf(mode="simd"):
        m16 = reg.full_mask(elem_bits=16)
        m32 = reg.full_mask(elem_bits=32)
        m64 = reg.full_mask(elem_bits=64)
        hist0 = reg.vdups(0, dtypes.uint16, mask=m16)
        hist1 = reg.vdups(0, dtypes.uint16, mask=m16)
        cum0 = reg.vdups(0, dtypes.uint16, mask=m16)
        cum1 = reg.vdups(0, dtypes.uint16, mask=m16)
        c8 = reg.mask_counter(n)
        xm8 = reg.vdups(
            xormask, dtypes.uint8
        )  # D1：twiddle 运行期化（轮级广播，探针 B）
        for i in cannbotdsl.range(nblk):
            bm, c8 = reg.update_mask(c8, elem_bits=8)
            if const_expr(ds == 2):
                # §19.31：原生 u16 lane（= ASC HistogramB16VF 同构）
                q0 = reg.vload(slot, i * 256 + 0 * VL16)
                q1 = reg.vload(slot, i * 256 + 1 * VL16)
                if const_expr(shift > 0):
                    q0 = reg.vshr(q0, shift, mask=m16)
                    q1 = reg.vshr(q1, shift, mask=m16)
                b8, _j2 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(q0, dtypes.uint8),
                    reg.vreinterpret_lanes(q1, dtypes.uint8),
                )
            elif const_expr(ds == 4):
                q0 = reg.vload(slot, i * VL8 + 0 * VL32)
                q1 = reg.vload(slot, i * VL8 + 1 * VL32)
                q2 = reg.vload(slot, i * VL8 + 2 * VL32)
                q3 = reg.vload(slot, i * VL8 + 3 * VL32)
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
                q0 = reg.vload(slot, i * 512 + 0 * VL32)
                q1 = reg.vload(slot, i * 512 + 1 * VL32)
                q2 = reg.vload(slot, i * 512 + 2 * VL32)
                q3 = reg.vload(slot, i * 512 + 3 * VL32)
                q4 = reg.vload(slot, i * 512 + 4 * VL32)
                q5 = reg.vload(slot, i * 512 + 5 * VL32)
                q6 = reg.vload(slot, i * 512 + 6 * VL32)
                q7 = reg.vload(slot, i * 512 + 7 * VL32)
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
                    s2 = shift
                    w0, w1, w2, w3 = lo0, lo1, lo2, lo3
                else:
                    s2 = shift - 32
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
            b8 = reg.vbitwise_xor(b8, xm8, mask=bm)
            reg.vstore(b8_stage, i * VL8, b8, bm)
            hist0 = reg.vadd(
                hist0, reg.vhistogram_frequency(b8, mask=bm, bin=0), mask=m16
            )
            hist1 = reg.vadd(
                hist1, reg.vhistogram_frequency(b8, mask=bm, bin=1), mask=m16
            )
            cum0 = reg.vhistogram_accumulate(cum0, b8, mask=bm, bin=0)
            cum1 = reg.vhistogram_accumulate(cum1, b8, mask=bm, bin=1)
        es0 = reg.vsub(cum0, hist0, mask=m16)
        es1 = reg.vsub(cum1, hist1, mask=m16)
        reg.vstore(hist_ub, 0, hist0, m16)
        reg.vstore(hist_ub, 128, hist1, m16)
        reg.vstore(cumsum_ub, 0, es0, m16)
        reg.vstore(cumsum_ub, 128, es1, m16)
        # 尾部（i64 版）：双重 interleave 宽化 es → vadd 进 int64 excl_acc
        z16 = reg.vdups(0, dtypes.uint16, mask=m16)
        z32 = reg.vdups(0, dtypes.int32, mask=m32)

        def _acc64(v, base):
            r0, r1 = reg.vinterleave(v, z16)
            w = reg.vreinterpret_lanes(r0, dtypes.int32)
            s0, s1 = reg.vinterleave(w, z32)
            e0 = reg.vadd(
                reg.vload(excl_acc, base),
                reg.vreinterpret_lanes(s0, dtypes.int64),
                mask=m64,
            )
            reg.vstore(excl_acc, base, e0, m64)
            e1 = reg.vadd(
                reg.vload(excl_acc, base + 32),
                reg.vreinterpret_lanes(s1, dtypes.int64),
                mask=m64,
            )
            reg.vstore(excl_acc, base + 32, e1, m64)
            w = reg.vreinterpret_lanes(r1, dtypes.int32)
            s0, s1 = reg.vinterleave(w, z32)
            e0 = reg.vadd(
                reg.vload(excl_acc, base + 64),
                reg.vreinterpret_lanes(s0, dtypes.int64),
                mask=m64,
            )
            reg.vstore(excl_acc, base + 64, e0, m64)
            e1 = reg.vadd(
                reg.vload(excl_acc, base + 96),
                reg.vreinterpret_lanes(s1, dtypes.int64),
                mask=m64,
            )
            reg.vstore(excl_acc, base + 96, e1, m64)

        _acc64(es0, 0)
        _acc64(es1, 128)
