# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""B16 NCDHW dW: multi-C1 Load3D, resident row bands, batched NZ2DN output.

Teaching subset: groups=1, stride=dilation=1, pad=0, Kh*Kw<=9;
full-width row bands fit L1, total reduction<=65535. Reject, never silently fall back.
The owner computes the complete reduction, so each output is written ONCE:
no atomic add, GM preclear, or global barrier is needed in this subset.
Long-K segmented atomic output belongs to the generic initialization protocol.
"""

import cannbotdsl as cbd
from cannbotdsl import Tensor, TensorSpec, Dim, MemLoc, dtypes, make_layout
import torch
from cannbotdsl.ops.memcpy import cube_fill_l1_zero

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


class WeightGradient:
    def __init__(self, dtype, bm, ci_tile, bk, ncap):
        self.dt, self.bm, self.ct, self.bk, self.nc = dtype, bm, ci_tile, bk, ncap

    @cbd.jit
    def load_plane(
        self, x: Tensor, dy: Tensor, xb, yb, batch, oz, kz, c, m, hstart, rows
    ):
        _, ci, di, hi, wi = x.shape
        _, co, od, oh, ow = dy.shape
        cv, mv = min(self.ct, ci - c), min(self.bm, co - m)
        # kh = Hi-Ho+1 for the valid stride-1 teaching subset.
        height = rows + hi - oh
        area = height * wi
        points = rows * ow
        # MMAD keeps the static BM/CI pitch even on tails. Define the
        # padded rows/channels by clearing the active tail parent before DMA;
        # complete parents need no preclear. This is a conservative tail policy.
        if cv < self.ct:
            cube_fill_l1_zero(
                xb,
                offset_elems=0,
                repeat=1,
                blk_num=ceildiv(align16(area) * self.ct * 2, 32),
                dst_gap=0,
            )
            cbd.cube_sync_pipe(cbd.PIPE.MTE2)
        if mv < self.bm:
            cube_fill_l1_zero(
                yb,
                offset_elems=0,
                repeat=1,
                blk_num=align16(points) * self.bm // 16,
                dst_gap=0,
            )
            cbd.cube_sync_pipe(cbd.PIPE.MTE2)
        # Source is CHW; permutation does NOT pack bytes. This DN2NZ does.
        # Physical order [C1,H*W,C0], not a dense HWC reinterpretation.
        src = cbd.from_tensor(
            x,
            layout=make_layout((cv, area), stride=(di * hi * wi, 1)),
            offset=((((batch * ci + c) * di + oz + kz) * hi + hstart) * wi) * 2,
        )
        fmap = cbd.from_tensor(
            xb,
            layout=make_layout((area, cv)),
            data_format="nz",
            physical_layout=make_layout(
                (self.ct // 16, ceildiv(area, 16), 16, 16),
                stride=(area * 16, 256, 16, 1),
            ),
        )
        cbd.mem_copy(fmap, src.permute(1, 0))
        grad = cbd.from_tensor(
            dy,
            layout=make_layout((mv, points), stride=(od * oh * ow, 1)),
            offset=((((batch * co + m) * od + oz) * oh + hstart) * ow) * 2,
        )
        parent = cbd.from_tensor(
            yb, layout=make_layout((self.bm, align16(points))), data_format="nz"
        )
        cbd.mem_copy(parent[:mv, :points], grad)

    @cbd.kernel
    def kernel(self, x: Tensor, dy: Tensor, out: Tensor, band_rows, splits):
        batch, ci, di, hi, wi = x.shape
        _, co, od, oh, ow = dy.shape
        _, _, kd, kh, kw = out.shape
        bands = ceildiv(oh, band_rows)
        taps = kh * kw
        cols = self.ct * taps
        mt, nt = ceildiv(co, self.bm), ceildiv(ci, self.ct)
        root = cbd.L1.view(512 * 1024)
        # Two parent generations in different L1 halves. These are raw storage
        # slots; every consuming view below supplies its actual physical pitch.
        xf = cbd.make_ring_buffer(
            tuple(
                cbd.from_tensor(
                    root,
                    dtype=self.dt,
                    layout=make_layout((196608 // 2,)),
                    offset=i * 262144,
                    data_format="nd",
                )
                for i in range(2)
            )
        )
        yg = cbd.make_ring_buffer(
            tuple(
                cbd.from_tensor(
                    root,
                    dtype=self.dt,
                    layout=make_layout((65536 // 2,)),
                    offset=i * 262144 + 196608,
                    data_format="nd",
                )
                for i in range(2)
            )
        )
        a0 = cbd.RingBuffer(
            MemLoc.L0A, self.dt, make_layout((self.bm, self.bk)), depth=2
        )
        b0 = cbd.RingBuffer(
            MemLoc.L0B, self.dt, make_layout((self.nc, self.bk)), depth=2
        )
        acc = cbd.Buffer(MemLoc.L0C, dtypes.float32, make_layout((self.bm, self.nc)))
        # splits=1 is the original owner-computes-all path. For split-K,
        # out's first axis is [split, co] flattened; no two tasks share output.
        band_count = batch * od * bands
        # Use the actual launch size: a one-block launch must still visit every
        # task. A cached core quota as the step would skip work on smaller launches.
        for task in range(
            cbd.get_block_idx(), mt * nt * kd * splits, cbd.get_block_num()
        ):
            part = task % splits
            job = task // splits
            begin, end = band_count * part // splits, band_count * (part + 1) // splits
            kz = job % kd
            c = job // kd % nt * self.ct
            m = job // (kd * nt) * self.bm
            mv, cv = min(self.bm, co - m), min(self.ct, ci - c)
            # Prologue + steady state + drain. Issue the NEXT parent before
            # current MMAD/Fixpipe consumers can stall the scalar issue stream.
            for step in range(begin, end + 1):
                if step < end:
                    hs = step % bands * band_rows
                    nr = min(band_rows, oh - hs)
                    self.load_plane(
                        x,
                        dy,
                        xf.next_write(),
                        yg.next_write(),
                        step // (od * bands),
                        step // bands % od,
                        kz,
                        c,
                        m,
                        hs,
                        nr,
                    )
                if step > begin:
                    xread, yread = xf.next_read(), yg.next_read()
                    nr = min(band_rows, oh - (step - 1) % bands * band_rows)
                    for pos in range(0, nr * ow, self.bk):
                        live = min(self.bk, nr * ow - pos)
                        ap = cbd.from_tensor(
                            yread,
                            layout=make_layout((self.bm, self.bk)),
                            offset=pos * self.bm * 2,
                            data_format="nz",
                        )
                        cbd.mem_copy(a0.next_write()[None, :live], ap[None, :live])
                        bw = b0.next_write()
                        # dW: Load3D.m = reduction spatial axis; Load3D.k =
                        # output columns [ci1, kh, kw, ci0]. channels is the
                        # PARENT channel count, not this output k_extent.
                        bview = cbd.from_tensor(
                            bw, layout=make_layout((cols, self.bk)), data_format="nz"
                        )
                        cbd.load3d(
                            bview,
                            xread,
                            feature=cbd.Load3DFeatureMap(
                                height=nr + kh - 1, width=wi, channels=self.ct
                            ),
                            filter=(kh, kw),
                            config_bank="b",
                            window=cbd.Load3DWindow(
                                m_start=pos, m_extent=live, k_extent=cols
                            ),
                            destination=cbd.Load3DDestination(
                                k_stride_fractals=cols // 16
                            ),
                        )
                        br = cbd.from_tensor(
                            b0.next_read(),
                            layout=make_layout((cols, self.bk)),
                            data_format="nz",
                        )
                        # Keep acc's BM pitch through ALL K blocks and planes.
                        cbd.matmul(
                            acc,
                            a0.next_read()[None, :live],
                            br[:cols, :live],
                            init=(step == begin + 1 and pos == 0),
                        )
            # Batch axis=co within one co0 group, matrix M=tap, N=ci0.
            # The physical tap stride is BM/16*256, NOT always 256!
            # Reinterpreting this same C storage does not transpose any data.
            for cf in range(ceildiv(mv, 16)):
                cm = min(16, mv - cf * 16)
                for c1 in range(ceildiv(cv, 16)):
                    cn = min(16, cv - c1 * 16)
                    source = cbd.from_tensor(
                        acc,
                        layout=make_layout((cm, taps, cn)),
                        data_format="nz",
                        physical_layout=make_layout(
                            (16, 1, 1, 16, 16),
                            stride=(
                                16,
                                16 * self.bm * 16,
                                16 * self.bm * 16,
                                self.bm * 16,
                                1,
                            ),
                        ),
                        offset=(cf * 256 + c1 * taps * self.bm * 16) * 4,
                    )
                    dest = cbd.from_tensor(
                        out,
                        layout=make_layout(
                            (cm, taps, cn), stride=(ci * kd * taps, 1, kd * taps)
                        ),
                        offset=(
                            ((part * co + m + cf * 16) * ci + c + c1 * 16) * kd + kz
                        )
                        * taps
                        * 4,
                    )
                    # One batched Fixpipe per (co0 group, ci0 group), not one
                    # per tap or per scalar co. Output is FP32; no extra cast.
                    cbd.mem_copy(dest, source)

    @cbd.host
    def launch(self, x, dy, out, band_rows, splits, blocks):
        self.kernel[blocks](x, dy, out, band_rows, splits)


def compile_program(dt, bm, ct, bk, ncap):
    op = WeightGradient(dt, bm, ct, bk, ncap)
    dim_names = ("n", "ci", "co", "di", "hi", "wi", "od", "oh", "ow", "kd", "kh", "kw")
    n, ci, co, di, hi, wi, od, oh, ow, kd, kh, kw = [Dim(name) for name in dim_names]
    return cbd.compile(
        op.launch,
        TensorSpec((n, ci, di, hi, wi), dt),
        TensorSpec((n, co, od, oh, ow), dt),
        TensorSpec((Dim("partial_co"), ci, kd, kh, kw), dtypes.float32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int32,
    )


def plan_candidates(x, dy, kernel_size, cores, tile=None):
    """Shared legal row-band candidates; split-K applies its own K limit."""
    check_tensors(x, dy)
    n, ci, di, hi, wi = x.shape
    b, co, od, oh, ow = dy.shape
    kd, kh, kw = tuple(kernel_size)
    if (
        n != b
        or min(kd, kh, kw) <= 0
        or kh * kw > 9
        or (di - kd + 1, hi - kh + 1, wi - kw + 1) != (od, oh, ow)
    ):
        raise ValueError("expected valid stride-1 convolution, groups=1, Kh*Kw<=9")
    if max(hi, wi) > 4095:
        raise ValueError(
            "reduction/window outside this teaching subset; use segmented generic path"
        )
    choices = []
    # Candidate search includes partial improvements even if NONE fills all
    # cores. ci is not fixed at 16. Reject by actual buffers before ranking.
    for bm in (16, 32, 64, 128):
        for ct in (16, 32, 64):
            for bk in (32, 64, 128):
                if tile is not None and (bm, ct, bk) != tuple(tile):
                    continue
                nc = ct * kh * kw
                if not check_tile(Tile(bm, nc, bk)):
                    continue
                rows = min(oh, 196608 // (wi * ct * 2) - kh + 1, 65536 // (ow * bm * 2))
                while rows > 0 and (
                    align16((rows + kh - 1) * wi) * ct * 2 > 196608
                    or align16(rows * ow) * bm * 2 > 65536
                ):
                    rows -= 1
                if rows < 1:
                    continue
                bands = ceildiv(oh, rows)
                tasks = ceildiv(co, bm) * ceildiv(ci, ct) * kd
                waves = ceildiv(tasks, cores)
                full, tail = divmod(oh, rows)
                rounded_k = full * align16(rows * ow)
                if tail:
                    rounded_k += align16(tail * ow)
                padded_mac = waves * bm * nc * rounded_k * n * od
                traffic = (
                    waves
                    * ((oh + bands * (kh - 1)) * wi * ct + oh * ow * bm)
                    * 2
                    * n
                    * od
                )
                calls = (
                    waves
                    * n
                    * od
                    * (full * ceildiv(rows * ow, bk) + ceildiv(tail * ow, bk))
                )
                choices.append(
                    ((padded_mac, traffic, calls), bm, ct, bk, nc, tasks, rows)
                )
    if not choices:
        raise ValueError(
            "no legal candidate: check requested tile or use generic width splitting"
        )
    return choices


def prepare(x, dy, kernel_size, blocks=None, tile=None):
    cores = resources(blocks, device=x.device)
    choices = plan_candidates(x, dy, kernel_size, cores, tile)
    n, ci, di, hi, wi = x.shape
    _, co, od, oh, ow = dy.shape
    kd, kh, kw = kernel_size
    if n * od * oh * ow > 65535:
        raise ValueError("reduction exceeds single-owner example; use dw_splitk")
    score, bm, ct, bk, nc, tasks, rows = min(choices)
    count = min(cores, tasks)
    out = torch.empty((co, ci, kd, kh, kw), device=x.device, dtype=torch.float32)
    p = compile_program(dtype_of(x), bm, ct, bk, nc)
    plan = dict(
        operator="dw",
        tile=(bm, ct, bk),
        band_rows=rows,
        l0_n_capacity=nc,
        tasks=tasks,
        blocks=count,
        requested_blocks=cores,
        launch_wave_util=wave_util(tasks, count),
        quota_wave_util=wave_util(tasks, cores),
        reduction=n * od * oh * ow,
        cube_compute_cycles_est=score[0] / 4096,
        gm_bytes_wave_est=score[1],
        mma_calls_wave_est=score[2],
        ranking="padded per-wave MAC, GM bytes, then MMAD count; not a latency prediction",
    )
    return Invocation(p, (x, dy, out, rows, 1, count), out, plan)
