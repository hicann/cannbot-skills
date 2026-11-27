# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate and time the full deterministic dW split-K path, including merge."""

import argparse
import json
import os
from pathlib import Path
import sys


def run(args):
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    verify.torch = torch
    records, programs = [], {}
    # Small K, short bands/channel tails, compute-heavy few owners, long K.
    cases = [
        ("small", (2, 17, 1, 3, 5), 17, (1, 1, 1)),
        ("tails", (2, 33, 4, 7, 502), 35, (2, 3, 3)),
        ("wide", (1, 16, 1, 32, 1026), 16, (1, 1, 3)),
        ("long", (5, 16, 1, 64, 258), 16, (1, 1, 3)),
    ]
    kinds = (
        [torch.float16, torch.bfloat16]
        if args.dtype == "all"
        else [torch.float16 if args.dtype in ("fp16", "float16") else torch.bfloat16]
    )
    try:
        for dt in kinds:
            for label, xs, co, ks in cases:
                if args.case and label not in args.case:
                    continue
                torch.manual_seed(42)
                x = (torch.randn(xs) * 0.1).to(dt)
                ds = (xs[0], co, *(d - k + 1 for d, k in zip(xs[2:], ks)))
                dy = (torch.randn(ds) * 0.1).to(dt)
                ref = torch.nn.grad.conv3d_weight(
                    x.double(), (co, xs[1], *ks), dy.double()
                )
                xn, yn = x.to(args.device), dy.to(args.device)
                call = dw_splitk.prepare(
                    xn, yn, ks, blocks=args.blocks, tile=args.tile, splits=args.splits
                )
                name = f"{label}_{str(dt).split('.')[-1]}"
                key = (
                    dt,
                    tuple(call.plan["tile"]),
                    call.plan["l0_n_capacity"],
                    call.plan["device_stages"],
                )
                old = programs.get(key)
                if call.plan["device_stages"] == 2:
                    call.args[2].fill_(
                        float("nan")
                    )  # Detect holes, not production initialization.
                try:
                    rec = verify.check(call, ref, name, args.output, args.profile, old)
                    if call.plan["device_stages"] == 2:
                        assert torch.isfinite(call.args[2]).all().item(), (
                            "partial contains unwritten values"
                        )
                        rec["partial_fully_written"] = True
                    # Negation is exact in B16; scaling could round subnormal inputs.
                    xn.neg_()
                    call.output.fill_(float("nan"))
                    if call.plan["device_stages"] == 2:
                        call.args[2].fill_(float("nan"))
                    torch.testing.assert_close(
                        call().cpu(), (-ref).float(), atol=2e-4, rtol=2e-3
                    )
                    xn.neg_()
                    rec["changed_input_passed"] = True
                    if (
                        args.compare
                        and call.plan["device_stages"] == 2
                        and call.plan["reduction"] <= 65535
                    ):
                        # Model-selected unsplit candidate; not an exhaustive timing search.
                        base = dw.prepare(xn, yn, ks, blocks=args.blocks)
                        try:
                            torch.testing.assert_close(
                                base().cpu(), ref.float(), atol=2e-4, rtol=2e-3
                            )
                            rec["unsplit_plan"] = base.plan
                            if args.profile:
                                rec["unsplit"] = verify.device_time(
                                    base, args.output / (name + "_unsplit.trace.json")
                                )
                                rec["speedup_vs_unsplit"] = (
                                    rec["unsplit"]["device_us"] / rec["device_us"]
                                )
                        finally:
                            base.close()
                    records.append(rec)
                finally:
                    if old is None:
                        programs[key] = call
                    else:
                        call.close()
        if not records:
            raise ValueError("no selected cases")
        (args.output / "results.json").write_text(json.dumps(records, indent=2))
        print(
            f"{len(records)} split-K cases passed; {args.output / 'results.json'}",
            flush=True,
        )
    finally:
        for call in programs.values():
            call.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--blocks", type=int, default=32)
    p.add_argument("--device", default="npu:0")
    p.add_argument(
        "--dtype", choices=("fp16", "bf16", "float16", "bfloat16", "all"), default="all"
    )
    p.add_argument(
        "--case", action="append", choices=("small", "tails", "wide", "long")
    )
    p.add_argument(
        "--splits", type=int, help="force split count; default uses geometric selection"
    )
    p.add_argument("--tile", type=int, nargs=3)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--output", type=Path, default=Path("/tmp/conv-dw-splitk-example"))
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output)
    sys.dont_write_bytecode = True
    os.environ.setdefault("CANNBOTDSL_NPU_ARCH", "dav-3510")
    os.environ.setdefault("CANNBOTDSL_CACHE_DIR", str(args.output / "cache"))
    os.environ.setdefault("CANNBOTDSL_DUMP_ASCENDC", "1")
    os.environ.setdefault("CANNBOTDSL_DUMP_DIR", str(args.output / "dump"))
    import torch
    import torch_npu
    import dw
    import dw_splitk
    import verify

    run(args)
