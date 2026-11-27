# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Deterministic cross-core split-K extension of dw.py, B16 -> FP32.

Each task owns one (output tile, contiguous band range). The cube stage writes
FP32 partials in final weight order. A separate vector stage merges in split
index order: no atomics, GM preclear, inter-core flags or global barriers.
This extends the teaching example, not generic's workspace-free MN_StreamK.
"""

import cannbotdsl as cbd
from cannbotdsl import Tensor, TensorSpec, Dim, MemLoc, dtypes, make_layout
import torch

import dw
from common import Invocation, ceildiv, align16, dtype_of, resources, wave_util

MAX_SPLITS = 128
CHUNK = 256
MAX_WORKSPACE = 64 * 1024 * 1024


class SplitWeight(dw.WeightGradient):
    @cbd.kernel
    def merge(self, partial: Tensor, out: Tensor, splits):
        co, ci, kd, kh, kw = out.shape
        size = co * ci * kd * kh * kw
        # 128 KiB input + 1 KiB output, within 950's UB capacity.
        stage = cbd.Buffer(MemLoc.UB, dtypes.float32, make_layout((MAX_SPLITS, CHUNK)))
        result = cbd.Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK,)))
        lane = cbd.get_block_idx()
        lanes = cbd.get_block_num()
        for task in range(lane, ceildiv(size, CHUNK), lanes):
            pos = task * CHUNK
            live = min(CHUNK, size - pos)
            src = cbd.from_tensor(
                partial,
                layout=make_layout((splits, live), stride=(size, 1)),
                offset=pos * 4,
            )
            # One 2D DMA loads all partials for this contiguous output strip.
            # No tiny scalar-co/tap copies; the last strip may be unaligned.
            cbd.mem_copy(stage[:splits, :live], src)
            # Four 64-FP32 register accumulators; counts are prepared outside
            # VF. One zero-based unit-step split loop, no conditionals inside.
            # Explicit root offsets also make the four load addresses auditable.
            remaining = dtypes.uint16(splits - 1)
            live1, live2, live3 = (
                max(0, live - 64),
                max(0, live - 128),
                max(0, live - 192),
            )
            with cbd.vf():
                mask0, _ = cbd.reg.update_mask(live, elem_bits=32)
                mask1, _ = cbd.reg.update_mask(live1, elem_bits=32)
                mask2, _ = cbd.reg.update_mask(live2, elem_bits=32)
                mask3, _ = cbd.reg.update_mask(live3, elem_bits=32)
                a = cbd.reg.vload(stage, 0)
                b = cbd.reg.vload(stage, 64)
                c = cbd.reg.vload(stage, 128)
                d = cbd.reg.vload(stage, 192)
                for s in range(remaining):
                    offset = (dtypes.int32(s) + 1) * CHUNK
                    a = cbd.reg.vadd(a, cbd.reg.vload(stage, offset + 0), mask=mask0)
                    b = cbd.reg.vadd(b, cbd.reg.vload(stage, offset + 64), mask=mask1)
                    c = cbd.reg.vadd(c, cbd.reg.vload(stage, offset + 128), mask=mask2)
                    d = cbd.reg.vadd(d, cbd.reg.vload(stage, offset + 192), mask=mask3)
                cbd.reg.vstore(result, 64, a, mask0, post_update=True)
                cbd.reg.vstore(result, 64, b, mask1, post_update=True)
                cbd.reg.vstore(result, 64, c, mask2, post_update=True)
                cbd.reg.vstore(result, 64, d, mask3, post_update=True)
            dst = cbd.from_tensor(out, layout=make_layout((live,)), offset=pos * 4)
            cbd.mem_copy(dst, result[:live])

    @cbd.host
    def run(self, x, dy, partial, out, rows, splits, blocks, merge_blocks):
        # Same stream: stage completion orders GM writes before the merge.
        # Keep both kernels in one compiled host program; no Python per split.
        self.kernel[blocks](x, dy, partial, rows, splits)
        self.merge[merge_blocks](partial, out, splits)


def compile_program(dt, bm, ct, bk, nc):
    op = SplitWeight(dt, bm, ct, bk, nc)
    n, ci, co, di, hi, wi, od, oh, ow, kd, kh, kw, pc = [
        Dim(s)
        for s in (
            "n",
            "ci",
            "co",
            "di",
            "hi",
            "wi",
            "od",
            "oh",
            "ow",
            "kd",
            "kh",
            "kw",
            "pc",
        )
    ]
    return cbd.compile(
        op.run,
        TensorSpec((n, ci, di, hi, wi), dt),
        TensorSpec((n, co, od, oh, ow), dt),
        TensorSpec((pc, ci, kd, kh, kw), dtypes.float32),
        TensorSpec((co, ci, kd, kh, kw), dtypes.float32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int32,
        dtypes.int32,
    )


def partition_lengths(n, od, oh, ow, rows, splits, padded=False, bk=None):
    """Exact K count of each band range, including every plane's short band."""
    bands = ceildiv(oh, rows)
    count = n * od * bands
    full, tail = divmod(oh, rows)
    main = align16(rows * ow) if padded else rows * ow
    last = align16(tail * ow) if padded else tail * ow
    if bk is not None:
        main, last = ceildiv(rows * ow, bk), ceildiv(tail * ow, bk)
    plane = full * main + last

    def prefix(b):
        planes, band = divmod(b, bands)
        return planes * plane + min(band, full) * main

    return [
        prefix(count * (s + 1) // splits) - prefix(count * s // splits)
        for s in range(splits)
    ]


def prepare(x, dy, kernel_size, blocks=None, tile=None, splits=None):
    """Select from legal tiles/splits; explicit splits enables controlled A/B.

    The model ranks worst-core padded MAC, then workspace traffic and MMAD
    count. It does not predict absolute latency. Auto split needs >=4096 K
    per part and >=25% less compute work than the best legal unsplit tile;
    long-K inputs must split regardless. Measure the FULL two-stage result.
    """
    cores = resources(blocks, device=x.device)
    candidates = dw.plan_candidates(x, dy, kernel_size, cores, tile)
    n, ci, _, _, _ = x.shape
    _, co, od, oh, ow = dy.shape
    kd, kh, kw = kernel_size
    output_elems = co * ci * kd * kh * kw
    options = []
    for _, bm, ct, bk, nc, tasks, rows in candidates:
        bands = n * od * ceildiv(oh, rows)
        limit = min(MAX_SPLITS, bands, MAX_WORKSPACE // (output_elems * 4))
        counts = [splits] if splits is not None else range(1, max(1, limit) + 1)
        for s in counts:
            if not isinstance(s, int) or s < 1 or s > bands or s > MAX_SPLITS:
                continue
            if s > 1 and s > limit:
                continue
            lengths = partition_lengths(n, od, oh, ow, rows, s)
            if max(lengths) > 65535:
                continue  # Conservative example segment policy, not an ISA limit.
            if splits is None and s > 1 and min(lengths) < 4096:
                continue
            padded = partition_lengths(n, od, oh, ow, rows, s, padded=True)
            # Task order is output-major, split-minor. Accumulate actual work
            # per lane, including uneven split lengths and the final wave.
            mma_parts = partition_lengths(n, od, oh, ow, rows, s, bk=bk)
            loads = [0] * min(cores, tasks * s)
            calls = [0] * len(loads)
            for task in range(tasks * s):
                loads[task % len(loads)] += padded[task % s]
                calls[task % len(loads)] += mma_parts[task % s]
            mac = max(loads) * bm * nc
            extra_bytes = 2 * s * output_elems * 4 if s > 1 else 0
            score = (mac, extra_bytes, max(calls))
            options.append((score, bm, ct, bk, nc, tasks, rows, s, lengths))
    if not options:
        raise ValueError(
            "no legal split/tile: need nonempty bands, K<=65535/part, <=128 parts, workspace<=64MiB"
        )
    best = min(options)
    unsplit = [p for p in options if p[7] == 1]
    if splits is None and unsplit:
        base = min(unsplit)
        if best[0][0] > base[0][0] * 0.75:
            best = base
    score, bm, ct, bk, nc, owners, rows, s, lengths = best
    if s == 1:
        call = dw.prepare(x, dy, kernel_size, blocks=cores, tile=(bm, ct, bk))
        call.plan.update(
            splits=1, workspace_bytes=0, device_stages=1, split_k_lengths=lengths
        )
        return call
    partial = torch.empty(
        (s * co, ci, kd, kh, kw), device=x.device, dtype=torch.float32
    )
    out = torch.empty((co, ci, kd, kh, kw), device=x.device, dtype=torch.float32)
    count = min(cores, owners * s)
    # Pure AIV launch counts vector cores, not AIC/AIV pairs.
    vector_quota = min(cbd.get_platform_info().vector_core_num, 2 * cores)
    if vector_quota <= 0:
        raise RuntimeError("cannot query current-stream AIV quota")
    merge_blocks = min(vector_quota, ceildiv(output_elems, CHUNK))
    program = compile_program(dtype_of(x), bm, ct, bk, nc)
    plan = dict(
        operator="dw_splitk",
        tile=(bm, ct, bk),
        l0_n_capacity=nc,
        band_rows=rows,
        splits=s,
        output_owners=owners,
        tasks=owners * s,
        blocks=count,
        requested_blocks=cores,
        merge_blocks=merge_blocks,
        vector_quota=vector_quota,
        device_stages=2,
        quota_wave_util=wave_util(owners * s, cores),
        launch_wave_util=wave_util(owners * s, count),
        split_k_lengths=lengths,
        reduction=n * od * oh * ow,
        workspace_bytes=partial.numel() * 4,
        merge_traffic_bytes=(s + 1) * output_elems * 4,
        extra_gm_bytes_est=score[1],
        cube_compute_cycles_est=score[0] / 4096,
        mma_calls_worst_core=score[2],
        ranking="worst-core padded MAC, workspace traffic, MMAD count; latency must be measured",
    )
    return Invocation(
        program, (x, dy, partial, out, rows, s, count, merge_blocks), out, plan
    )
