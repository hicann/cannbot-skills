#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Kernel runtime determinism check.

Runs the AscendC kernel at `current_task/` twice on the same inputs (seeded),
element-wise bit-exact diff per case. Outputs JSON suitable for embedding in
`verification.json` under the `determinism` field.

Usage:
    python3 determinism_check.py [--out PATH] [--task-dir DIR]

Defaults:
    --task-dir /root/AscendOpGenAgent/current_task
    --out (stdout)

Exit codes:
    0 = deterministic (all cases bit-exact across two runs)
    1 = non-deterministic (at least one case differs)
    2 = infrastructure error (import/load/NPU)
"""
import argparse
import json
import os
import sys
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", default="/root/AscendOpGenAgent/current_task")
    ap.add_argument("--out", default=None, help="write JSON report to this path (stdout if omitted)")
    ap.add_argument("--cases", type=str, default=None,
                    help="comma-separated case indices to check (default: all)")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
    except Exception as e:
        print(f"[determinism_check] import failed: {e}", file=sys.stderr)
        return 2

    sys.path.insert(0, args.task_dir)
    sys.path.insert(0, os.path.join(args.task_dir, "kernel", "build"))

    try:
        from model import get_input_groups  # noqa: PLC0415
        from model_new_ascendc import ModelNew  # noqa: PLC0415
    except Exception as e:
        print(f"[determinism_check] model import failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 2

    import torch

    # Seed-deterministic input generation (benchmark convention)
    torch.manual_seed(0)
    inputs_run1 = get_input_groups()
    torch.manual_seed(0)
    inputs_run2 = get_input_groups()

    # Sanity: inputs must be identical across seeded runs
    for i, (a, b) in enumerate(zip(inputs_run1, inputs_run2)):
        for j, (x, y) in enumerate(zip(a, b)):
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                if not torch.equal(x, y):
                    print(f"[determinism_check] input case {i}[{j}] non-deterministic "
                          f"under seed — benchmark framework issue", file=sys.stderr)
                    return 2

    model = ModelNew().npu().eval()
    cases_to_check = list(range(len(inputs_run1)))
    if args.cases:
        cases_to_check = [int(x) for x in args.cases.split(",")]

    n_identical = 0
    n_diff = 0
    drift_detail = []

    def _to_cpu_list(out):
        """Normalize model output to a list of CPU tensors (handles single tensor, tuple, list)."""
        if isinstance(out, torch.Tensor):
            return [out.cpu()]
        if isinstance(out, (tuple, list)):
            return [t.cpu() if isinstance(t, torch.Tensor) else t for t in out]
        raise TypeError(f"Unsupported model output type: {type(out)}")

    with torch.no_grad():
        for i in cases_to_check:
            inp = inputs_run1[i]
            args_npu = [t.npu() if isinstance(t, torch.Tensor) else t for t in inp]
            # Run 1
            outs1 = _to_cpu_list(model(*[a.clone() if isinstance(a, torch.Tensor) else a for a in args_npu]))
            # Run 2 (fresh clones, same input values)
            outs2 = _to_cpu_list(model(*[a.clone() if isinstance(a, torch.Tensor) else a for a in args_npu]))

            if len(outs1) != len(outs2):
                n_diff += 1
                drift_detail.append({"case": i, "n_diff_elements": -1,
                                    "note": f"tuple length mismatch run1={len(outs1)} run2={len(outs2)}"})
                continue

            case_diff_count = 0
            case_max_d = 0.0
            case_shapes = []
            case_dtypes = []
            for o1, o2 in zip(outs1, outs2):
                if not isinstance(o1, torch.Tensor):
                    continue
                case_shapes.append(list(o1.shape))
                case_dtypes.append(str(o1.dtype))
                if torch.equal(o1, o2):
                    continue
                both_nan = torch.isnan(o1) & torch.isnan(o2)
                exact_match = (o1 == o2) | both_nan
                sub_count = int((~exact_match).sum().item())
                case_diff_count += sub_count
                if sub_count > 0:
                    delta = (o1.to(torch.float32) - o2.to(torch.float32)).abs()
                    finite_delta = delta[torch.isfinite(delta)]
                    max_d = float(finite_delta.max().item()) if finite_delta.numel() > 0 else float("inf")
                    case_max_d = max(case_max_d, max_d)

            if case_diff_count == 0:
                n_identical += 1
            else:
                n_diff += 1
                total_elems = sum(o.numel() for o in outs1 if isinstance(o, torch.Tensor))
                drift_detail.append({
                    "case": i,
                    "n_diff_elements": case_diff_count,
                    "ratio_pct": 100.0 * case_diff_count / max(total_elems, 1),
                    "max_abs_diff": case_max_d,
                    "shapes": case_shapes,
                    "dtypes": case_dtypes,
                })

    report = {
        "observed_deterministic": (n_diff == 0),
        "n_cases_checked": len(cases_to_check),
        "n_identical_cases": n_identical,
        "n_diff_cases": n_diff,
        "method": "run kernel twice on seed-identical inputs, element-wise bit-exact diff (NaN-aware)",
        "drift_detail": drift_detail[:10],  # cap to 10 cases for readability
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    else:
        print(json.dumps(report, indent=2))

    return 0 if n_diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
