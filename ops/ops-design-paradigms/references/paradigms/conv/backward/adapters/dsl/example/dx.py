# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""B16 NCDHW dX; bias-free ConvTranspose reuses exactly this device program.

Teaching subset: groups=1, pad=0, dilation=1, strideD=1, Kh*Kw<=9,
Co multiple of 16; whole expanded dY plane and all weight depth planes fit L1.
Weight transformation is transpose_and_reverse (also transpose_only for 1x1).
No host weight permutation or numerical preprocessing is hidden in prepare().
"""

import cannbotdsl as cbd
from cannbotdsl import Tensor, TensorSpec, Dim, MemLoc, dtypes, make_layout
from cannbotdsl.ops.memcpy import cube_fill_l1_zero
import torch

from common import (
    Tile,
    Invocation,
    align16,
    ceildiv,
    check_tile,
    check_tensors,
    dtype_of,
    resources,
    wave_util,
)


class InputGradient:
    def __init__(self, dt, bm, bn, bk):
        self.dt, self.bm, self.bn, self.bk = dt, bm, bn, bk

    @cbd.jit
    def load_dy(self, dy: Tensor, slot, batch, oz, sh, sw):
        _, co, od, oh, ow = dy.shape
        eh, ew = (oh - 1) * sh + 1, (ow - 1) * sw + 1
        if sh > 1 or sw > 1:
            # FMatrix padding only covers OUTSIDE the feature map. It cannot
            # initialize the holes INSIDE an expanded plane. Clear only here.
            cube_fill_l1_zero(
                slot,
                offset_elems=0,
                repeat=1,
                blk_num=ceildiv(eh * ew * co * 2, 32),
                dst_gap=0,
            )
            cbd.cube_sync_pipe(cbd.PIPE.MTE2)
        src = cbd.from_tensor(
            dy,
            layout=make_layout((oh, ow, co), stride=(ow, 1, od * oh * ow)),
            offset=(batch * co * od + oz) * oh * ow * 2,
        )
        # H is the matrix batch axis (dnNum). SH/SW are destination strides,
        # so all H rows are emitted by one mem_copy, no scalar H loop.
        dst = cbd.from_tensor(
            slot,
            layout=make_layout((oh, ow, co)),
            data_format="nz",
            physical_layout=make_layout(
                (oh, co // 16, ceildiv(ow, 16), 16, 16),
                stride=(sh * ew * 16, eh * ew * 16, (sw * 16) * 16, sw * 16, 1),
            ),
        )
        cbd.mem_copy(dst, src)

    @cbd.jit
    def load_weights(self, w: Tensor, store, ci_start, nv):
        co, ci, kd, kh, kw = w.shape
        taps = kh * kw
        if nv < self.bn:
            cube_fill_l1_zero(
                store,
                offset_elems=0,
                repeat=1,
                blk_num=kd * taps * co * self.bn // 16,
                dst_gap=0,
            )
            cbd.cube_sync_pipe(cbd.PIPE.MTE2)
        for kz in range(kd):
            src = cbd.from_tensor(
                w,
                layout=make_layout(
                    (co, taps, nv), stride=(ci * kd * taps, 1, kd * taps)
                ),
                offset=(ci_start * kd + kz) * taps * 2,
            )
            # P1 from the adapter: bytes [Ci1,tap,Co1,Co0,Ci0]. The batch
            # axis of this view is Co, not an extra leading matrix dimension.
            dst = cbd.from_tensor(
                store,
                layout=make_layout((co, taps, nv)),
                data_format="nz",
                physical_layout=make_layout(
                    (co, self.bn // 16, ceildiv(taps, 16), 16, 16),
                    stride=(16, taps * co * 16, (co * 16) * 16, co * 16, 1),
                ),
                offset=kz * self.bn * co * taps * 2,
            )
            cbd.mem_copy(dst, src)

    @cbd.jit
    def load_b0(self, store, bw, co, kz, taps, kstart, ku):
        # This per-K-fractal loop matches ASC's transposed Load2D granularity;
        # it is NOT a replacement for the one-instruction GM parent packing.
        for j in range(ku // 16):
            r = kstart // 16 + j
            src = cbd.from_tensor(
                store,
                layout=make_layout((16, self.bn)),
                data_format="nz",
                physical_layout=make_layout(
                    (self.bn // 16, 1, 16, 16), stride=(taps * co * 16, 256, 16, 1)
                ),
                offset=(
                    kz * self.bn * co * taps
                    + (r // taps) * 256
                    + (taps - 1 - r % taps) * co * 16
                )
                * 2,
            )
            dst = cbd.from_tensor(
                bw,
                layout=make_layout((16, self.bn)),
                data_format="zn",
                physical_layout=make_layout(
                    (1, self.bn // 16, 16, 16), stride=(256, 256, 16, 1)
                ),
                offset=j * self.bn * 16 * 2,
            )
            cbd.mem_copy(dst, src)

    @cbd.kernel
    def kernel(self, dy: Tensor, w: Tensor, out: Tensor, sh, sw, chunk):
        n, co, od, oh, ow = dy.shape
        _, ci, di, hi, wi = out.shape
        _, _, kd, kh, kw = w.shape
        taps = kh * kw
        eh, ew = (oh - 1) * sh + 1, (ow - 1) * sw + 1
        # output_padding is derived from the requested output, not hardcoded.
        extra_h, extra_w = hi - (eh + kh - 1), wi - (ew + kw - 1)
        nt, mt = ceildiv(ci, self.bn), ceildiv(hi * wi, self.bm)
        chunks = ceildiv(mt, chunk)
        root = cbd.L1.view(512 * 1024)
        a1 = cbd.make_ring_buffer(
            tuple(
                cbd.from_tensor(
                    root,
                    dtype=self.dt,
                    layout=make_layout((131072 // 2,)),
                    offset=i * 262144,
                    data_format="nd",
                )
                for i in range(2)
            )
        )
        weights = cbd.make_buffer(
            cbd.from_tensor(
                root,
                dtype=self.dt,
                layout=make_layout((131072 // 2,)),
                offset=131072,
                data_format="nd",
            )
        )
        a0 = cbd.RingBuffer(
            MemLoc.L0A, self.dt, make_layout((self.bm, self.bk)), depth=2
        )
        b0 = cbd.RingBuffer(
            MemLoc.L0B, self.dt, make_layout((self.bn, self.bk)), depth=2
        )
        c0 = cbd.Buffer(MemLoc.L0C, dtypes.float32, make_layout((self.bm, self.bn)))
        for task in range(
            cbd.get_block_idx(), n * di * nt * chunks, cbd.get_block_num()
        ):
            ci_start = task % nt * self.bn
            first_m = task // nt % chunks * chunk
            z = task // (nt * chunks) % di
            batch = task // (nt * chunks * di)
            nv = min(self.bn, ci - ci_start)
            # Load complete K weights ONCE per task, reuse across M blocks.
            # Increasing chunk reduces reloads but also reduces independent
            # tasks. Host evaluates both; do not maximize residency blindly.
            self.load_weights(w, weights, ci_start, nv)
            for mi in range(first_m, min(mt, first_m + chunk)):
                pos = mi * self.bm
                mv = min(self.bm, hi * wi - pos)
                mc = align16(mv)
                # Tail M uses a compact physical pitch. Keep Load3D, MMAD
                # and Fixpipe on THIS same mc; the allocation still fits BM.
                acc = cbd.from_tensor(
                    c0, layout=make_layout((mc, self.bn)), data_format="nz"
                )
                first = True
                for stage in range(kd + 1):
                    if stage < kd:
                        oz = z - stage
                        if oz >= 0 and oz < od:
                            self.load_dy(dy, a1.next_write(), batch, oz, sh, sw)
                    if stage > 0:
                        kz = stage - 1
                        oz = z - kz
                        if oz >= 0 and oz < od:
                            parent = a1.next_read()
                            for kstart in range(0, co * taps, self.bk):
                                ku = min(self.bk, co * taps - kstart)
                                aw = cbd.from_tensor(
                                    a0.next_write(),
                                    layout=make_layout((mc, self.bk)),
                                    data_format="nz",
                                )
                                bw = b0.next_write()
                                # Forward stride is already implemented by
                                # L1 holes. Load3D stride MUST remain (1,1).
                                cbd.load3d(
                                    aw,
                                    parent,
                                    feature=cbd.Load3DFeatureMap(
                                        height=eh,
                                        width=ew,
                                        channels=co,
                                        pad_top=kh - 1,
                                        pad_bottom=kh - 1 + extra_h,
                                        pad_left=kw - 1,
                                        pad_right=kw - 1 + extra_w,
                                    ),
                                    filter=(kh, kw),
                                    stride=(1, 1),
                                    window=cbd.Load3DWindow(
                                        m_start=pos,
                                        m_extent=mv,
                                        k_start=kstart,
                                        k_extent=ku,
                                    ),
                                    destination=cbd.Load3DDestination(
                                        k_stride_fractals=mc // 16
                                    ),
                                )
                                self.load_b0(weights, bw, co, kz, taps, kstart, ku)
                                rhs = cbd.from_tensor(
                                    b0.next_read(),
                                    layout=make_layout((self.bn, ku)),
                                    data_format="nz",
                                )
                                lhs = cbd.from_tensor(
                                    a0.next_read(),
                                    layout=make_layout((mc, self.bk)),
                                    data_format="nz",
                                )
                                cbd.matmul(
                                    acc, lhs[None, :ku], rhs[None, :ku], init=first
                                )
                                first = False
                # strideD=1 + valid output geometry guarantees some kd is
                # valid. Do not extend this to depth holes without defining
                # zero output: an untouched L0C must never be copied out.
                dst = cbd.from_tensor(
                    out,
                    layout=make_layout((mv, nv), stride=(1, di * hi * wi)),
                    offset=(((batch * ci + ci_start) * di + z) * hi * wi + pos) * 2,
                )
                cbd.mem_copy(dst, acc[:mv, :nv].convert(self.dt))

    @cbd.host
    def launch(self, dy, w, out, sh, sw, chunk, blocks):
        self.kernel[blocks](dy, w, out, sh, sw, chunk)


def compile_program(dt, bm, bn, bk):
    op = InputGradient(dt, bm, bn, bk)
    dim_names = ("n", "ci", "co", "di", "hi", "wi", "od", "oh", "ow", "kd", "kh", "kw")
    n, ci, co, di, hi, wi, od, oh, ow, kd, kh, kw = [Dim(name) for name in dim_names]
    # Layout arithmetic uses i64 shape values. Keep stride scalars i64 too,
    # so affine DMA pitches do not cross a narrowing/extension boundary.
    return cbd.compile(
        op.launch,
        TensorSpec((n, co, od, oh, ow), dt),
        TensorSpec((co, ci, kd, kh, kw), dt),
        TensorSpec((n, ci, di, hi, wi), dt),
        *[dtypes.int64] * 4,
    )


def prepare(dy, w, input_shape, stride=(1, 1, 1), blocks=None, tile=None):
    check_tensors(dy, w)
    cores = resources(blocks, device=dy.device)
    n, co, od, oh, ow = dy.shape
    wc, ci, kd, kh, kw = w.shape
    bn, bci, di, hi, wi = tuple(input_shape)
    sd, sh, sw = tuple(stride)
    if (
        (bn, bci) != (n, ci)
        or wc != co
        or sd != 1
        or min(sh, sw) < 1
        or co % 16
        or kh * kw > 9
    ):
        raise ValueError("expected groups=1, Co%16=0, strideD=1, Kh*Kw<=9")
    if min(di, hi, wi) <= 0 or (
        di - kd + 1,
        (hi - kh) // sh + 1,
        (wi - kw) // sw + 1,
    ) != (
        od,
        oh,
        ow,
    ):
        raise ValueError("input_shape is inconsistent with valid forward convolution")
    eh, ew = (oh - 1) * sh + 1, (ow - 1) * sw + 1
    if (
        max(eh, ew) > 4095
        or co * eh * ew * 2 > 131072
        or max(kh - 1 + sh - 1, kw - 1 + sw - 1) > 255
    ):
        raise ValueError(
            "expanded plane/padding exceeds example limits; use local row bands"
        )
    choices = []
    for bm in (32, 64, 128):
        for bc in (16, 32, 64):
            for bk in (64, 128):
                if tile is not None and (bm, bc, bk) != tuple(tile):
                    continue
                if (
                    not check_tile(Tile(bm, bc, bk))
                    or kd * kh * kw * co * bc * 2 > 131072
                ):
                    continue
                # Respect generic NCDHW transform selection: the small-kernel
                # reverse_only branch is not implemented in this example.
                if kd * kh * kw != 1 and kd * kh * kw <= 32 // ci:
                    continue
                mt = ceildiv(hi * wi, bm)
                for chunk in sorted({1, min(2, mt), min(4, mt), mt}):
                    tasks = n * di * ceildiv(ci, bc) * ceildiv(mt, chunk)
                    waves = ceildiv(tasks, cores)
                    mac = waves * chunk * bm * bc * co * kh * kw * min(kd, od)
                    traffic = (
                        waves
                        * (kd * kh * kw * co * bc + chunk * min(kd, od) * eh * ew * co)
                        * 2
                    )
                    calls = waves * chunk * min(kd, od) * ceildiv(co * kh * kw, bk)
                    choices.append(((mac, traffic, calls), bm, bc, bk, chunk, tasks))
    if not choices:
        raise ValueError("no resident transpose-and-reverse plan in example scope")
    score, bm, bc, bk, chunk, tasks = min(choices)
    count = min(cores, tasks)
    out = torch.empty(input_shape, dtype=dy.dtype, device=dy.device)
    p = compile_program(dtype_of(dy), bm, bc, bk)
    plan = dict(
        operator="dx",
        tile=(bm, bc, bk),
        m_blocks_per_task=chunk,
        tasks=tasks,
        blocks=count,
        requested_blocks=cores,
        launch_wave_util=wave_util(tasks, count),
        quota_wave_util=wave_util(tasks, cores),
        weight_bytes=kd * kh * kw * co * bc * 2,
        cube_compute_cycles_est=score[0] / 4096,
        gm_bytes_wave_est=score[1],
        mma_calls_wave_est=score[2],
        ranking="padded per-wave MAC, GM bytes, then MMAD count; not a latency prediction",
    )
    return Invocation(p, (dy, w, out, sh, sw, chunk, count), out, plan)


def prepare_transpose(x, w, stride=(1, 1, 1), output_padding=(0, 0, 0), **kwargs):
    """Bias-free ConvTranspose3d: same program, only Host output-shape derivation."""
    if len(output_padding) != 3 or any(
        v < 0 or v >= s for v, s in zip(output_padding, stride)
    ):
        raise ValueError("output_padding must be in [0,stride)")
    shape = (
        x.shape[0],
        w.shape[1],
        *(
            (a - 1) * s + k + p
            for a, s, k, p in zip(x.shape[2:], stride, w.shape[2:], output_padding)
        ),
    )
    return prepare(x, w, shape, stride=stride, **kwargs)
