# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-double Golden, runtime-shape reuse, repeatability and optional device timing.

Run from this directory. --output owns caches/dumps/results; it is never written
into the skill tree by default. These smoke tolerances are NOT a replacement for
ops-test-kit acceptance of a production operator.
"""

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import statistics
import sys


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=("dw", "dx", "all"), default="all")
    p.add_argument(
        "--dtype", choices=("fp16", "bf16", "float16", "bfloat16", "all"), default="all"
    )
    p.add_argument("--blocks", type=int, default=None)
    p.add_argument("--device", default="npu:0")
    p.add_argument(
        "--output", type=Path, default=Path("/tmp/conv-backward-dsl-example")
    )
    p.add_argument("--profile", action="store_true")
    p.add_argument(
        "--case", action="append", help="run only the named case(s), e.g. dw_fp16_3"
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--auto-tile",
        action="store_true",
        help="use analytical candidate ranking instead of layout-stress tiles",
    )
    group.add_argument(
        "--tile",
        type=int,
        nargs=3,
        metavar=("M", "CI_OR_N", "K"),
        help="one candidate for a controlled comparison",
    )
    return p.parse_args()


def device_time(call, path, repeats=5):
    from torch_npu.profiler import profile, ProfilerActivity, tensorboard_trace_handler

    for _ in range(3):
        call()
    torch.npu.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        on_trace_ready=tensorboard_trace_handler(
            str(path.parent / "profiles"), analyse_flag=False
        ),
    ) as prof:
        for _ in range(repeats):
            call()
        torch.npu.synchronize()
    prof.export_chrome_trace(str(path))
    content = json.loads(path.read_text())
    events = content if isinstance(content, list) else content["traceEvents"]
    # Device task identity, not name-only matching (which may catch Host calls).
    kernels = [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("dur") is not None
        and str(e.get("args", {}).get("Task Type", "")).startswith("KERNEL")
    ]
    # Include all stages: split-K is partial + deterministic merge.
    stages = call.plan.get("device_stages", 1)
    if len(kernels) != repeats * stages:
        raise RuntimeError(
            f"expected {repeats * stages} device tasks, got {len(kernels)}; inspect {path}"
        )
    kernels.sort(key=lambda e: Decimal(str(e["ts"])))
    groups = []
    for start in range(0, len(kernels), stages):
        end = start + stages
        groups.append(kernels[start:end])
    samples = [sum(e["dur"] for e in group) for group in groups]
    spans = [
        float(
            max(Decimal(str(e["ts"])) + Decimal(str(e["dur"])) for e in group)
            - Decimal(str(group[0]["ts"]))
        )
        for group in groups
    ]
    return dict(
        device_us=statistics.median(samples),
        samples_us=samples,
        device_span_us=statistics.median(spans),
        stage_us=[
            statistics.median(g[i]["dur"] for g in groups) for i in range(stages)
        ],
        kernels=sorted({e["name"] for e in kernels}),
        trace=str(path),
    )


def check(call, ref, name, output, profile, reused=None):
    # Golden already uses actual B16-quantized inputs, then CPU float64 math.
    call.output.fill_(float("nan"))
    if reused is None:
        actual = call().cpu()
    else:
        reused.program(*call.args)
        actual = call.output.cpu()
    target = ref.to(actual.dtype)
    if actual.dtype == torch.float32:
        atol, rtol = 2e-4, 2e-3
    elif actual.dtype == torch.float16:
        atol, rtol = 2e-4, 2e-3
    else:
        atol, rtol = 2e-3, 2e-2
    torch.testing.assert_close(actual, target, atol=atol, rtol=rtol)
    # One owner and fixed reduction order: reproducibility is part of example.
    again = call().cpu()
    if not torch.equal(
        actual.contiguous().view(torch.uint8), again.contiguous().view(torch.uint8)
    ):
        raise AssertionError(f"{name}: bitwise repeatability failed")
    rec = dict(
        case=name,
        tensor_shapes=[list(t.shape) for t in call.args[:3]],
        passed=True,
        plan=call.plan,
        max_abs=float((actual.float() - target.float()).abs().max()),
        atol=atol,
        rtol=rtol,
        reused_program=reused is not None,
        bitwise_repeat=True,
    )
    if profile:
        rec.update(device_time(call, output / (name + ".trace.json")))
    print(json.dumps(rec), flush=True)
    return rec


def run(args):
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    kinds = (
        [torch.float16, torch.bfloat16]
        if args.dtype == "all"
        else [torch.float16 if args.dtype in ("fp16", "float16") else torch.bfloat16]
    )
    records = []
    for dt in kinds:
        label = "fp16" if dt == torch.float16 else "bf16"
        calls = []
        try:
            torch.manual_seed(42)
            if args.op in ("all", "dw"):
                previous = {}
                for idx, (xs, co, ks) in enumerate(
                    [
                        ((1, 33, 3, 7, 9), 35, (2, 3, 3)),
                        ((2, 47, 4, 9, 11), 49, (2, 3, 3)),
                        ((1, 64, 1, 32, 1026), 128, (1, 1, 3)),
                        ((1, 33, 1, 7, 502), 35, (1, 3, 3)),
                        ((2, 17, 1, 3, 5), 17, (1, 1, 1)),
                    ]
                ):
                    if args.case and f"dw_{label}_{idx}" not in args.case:
                        continue
                    x = (torch.randn(xs) * 0.1).to(dt)
                    ds = (xs[0], co, *(s - k + 1 for s, k in zip(xs[2:], ks)))
                    dy = (torch.randn(ds) * 0.1).to(dt)
                    call = dw.prepare(
                        x.to(args.device),
                        dy.to(args.device),
                        ks,
                        blocks=args.blocks,
                        tile=None if args.auto_tile else (args.tile or (32, 32, 32)),
                    )
                    calls.append(call)
                    ref = torch.nn.grad.conv3d_weight(
                        x.double(), (co, xs[1], *ks), dy.double()
                    )
                    key = (dt, tuple(call.plan["tile"]), call.plan.get("l0_n_capacity"))
                    reuse = previous.get(key)
                    records.append(
                        check(
                            call,
                            ref,
                            f"dw_{label}_{idx}",
                            args.output,
                            args.profile,
                            reuse,
                        )
                    )
                    previous.setdefault(key, call)
            if args.op in ("all", "dx"):
                previous = {}
                for idx, (ds, ci, ks, st, opad) in enumerate(
                    [
                        ((1, 32, 2, 5, 7), 33, (2, 3, 3), (1, 1, 1), (0, 0, 0)),
                        ((5, 48, 3, 6, 9), 47, (2, 3, 3), (1, 1, 1), (0, 0, 0)),
                        ((1, 32, 2, 4, 5), 17, (2, 3, 3), (1, 2, 2), (0, 1, 1)),
                        ((1, 32, 1, 5, 7), 17, (1, 1, 1), (1, 1, 1), (0, 0, 0)),
                        ((2, 48, 3, 6, 9), 47, (2, 3, 3), (1, 1, 1), (0, 0, 0)),
                    ]
                ):
                    if args.case and f"dx_{label}_{idx}" not in args.case:
                        continue
                    dy = (torch.randn(ds) * 0.1).to(dt)
                    w = (torch.randn((ds[1], ci, *ks)) * 0.1).to(dt)
                    shape = (
                        ds[0],
                        ci,
                        *(
                            (d - 1) * s + k + p
                            for d, s, k, p in zip(ds[2:], st, ks, opad)
                        ),
                    )
                    # Transpose is only the Host binding; it reuses dx.kernel.
                    if idx == 2:
                        call = dx.prepare_transpose(
                            dy.to(args.device),
                            w.to(args.device),
                            stride=st,
                            output_padding=opad,
                            blocks=args.blocks,
                            tile=None
                            if args.auto_tile
                            else (args.tile or (32, 32, 64)),
                        )
                    else:
                        call = dx.prepare(
                            dy.to(args.device),
                            w.to(args.device),
                            shape,
                            stride=st,
                            blocks=args.blocks,
                            tile=None
                            if args.auto_tile
                            else (args.tile or (32, 32, 64)),
                        )
                    calls.append(call)
                    ref = torch.nn.grad.conv3d_input(
                        shape, w.double(), dy.double(), stride=st
                    )
                    key = (dt, tuple(call.plan["tile"]), call.plan.get("l0_n_capacity"))
                    reuse = previous.get(key)
                    records.append(
                        check(
                            call,
                            ref,
                            f"dx_{label}_{idx}",
                            args.output,
                            args.profile,
                            reuse,
                        )
                    )
                    previous.setdefault(key, call)
        finally:
            for call in calls:
                call.close()
    if not records:
        raise ValueError("no cases selected; check --case names")
    args.output.joinpath("results.json").write_text(json.dumps(records, indent=2))
    print(f"{len(records)} cases passed; {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    args = parse()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # Some runtime debug/profiler files use cwd independently of dump_dir.
    os.chdir(args.output)
    sys.dont_write_bytecode = True
    os.environ.setdefault("CANNBOTDSL_NPU_ARCH", "dav-3510")
    os.environ.setdefault("CANNBOTDSL_CACHE_DIR", str(args.output / "cache"))
    os.environ.setdefault("CANNBOTDSL_DUMP_ASCENDC", "1")
    os.environ.setdefault("CANNBOTDSL_DUMP_DIR", str(args.output / "dump"))
    import torch
    import torch_npu
    import dw
    import dx

    run(args)
