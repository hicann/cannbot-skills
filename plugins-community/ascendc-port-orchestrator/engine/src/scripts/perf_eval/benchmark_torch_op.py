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

"""Torch-op benchmark via torch_npu.profiler (PR #103 methodology).

Reads <op>_perf_cases.jsonl (from atk_loader.py), constructs tensors per case,
invokes the **torch-level** op (torch.gather / F.adaptive_avg_pool3d /
torch_npu.npu_apply_adam_w / ...) on NPU, profiles with warmup=5/active=5,
extracts per-case kernel time from op_statistic.csv.

This script runs on EITHER A3 or A5 — same input cases, same code, same
profiler schedule. Cross-host ratio = (A3_us / A5_us) on a per-case basis.
That's the apples-to-apples methodology PR #103 mandates.

Note: the kernel actually run is whichever .so the host's CANN install
registers. For port_a3_to_a5 archives where our build isn't installed
(P89 pending), this measures upstream-vs-upstream cross-arch — still
useful, but NOT a claim of "our port outperforms upstream".

Usage (run on target NPU host inside CANN-sourced container):
    python3 benchmark_torch_op.py \\
        --cases <op>_perf_cases.jsonl \\
        --op-call <op_kind> \\
        --device npu:0 \\
        --trace-root /tmp/profiler_<op> \\
        --out-json <op>_<host>_perf.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

WARMUP = 5
ACTIVE = 5
WAIT = 0

DTYPE_MAP = {
    "fp16": "float16", "fp32": "float32", "bf16": "bfloat16",
    "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
    "uint8": "uint8", "bool": "bool",
}


def make_tensor(spec: dict, device: str, torch):
    dt = getattr(torch, DTYPE_MAP[spec["dtype"]])
    shape = spec["shape"]
    if dt.is_floating_point:
        # range_values may be {name:'nd', mean:[a,b], std:[c,d]} — use mid value
        return torch.randn(*shape, dtype=dt, device=device)
    elif dt == torch.bool:
        return torch.randint(0, 2, shape, device=device).bool()
    else:
        rv = spec.get("range_values")
        lo, hi = 0, 100
        if isinstance(rv, list) and len(rv) == 2 and all(isinstance(x, (int, float)) for x in rv):
            lo, hi = int(rv[0]), int(rv[1])
        # For gather index: hi must fit dim_size; clamp loosely
        return torch.randint(lo, max(lo + 1, hi), shape, dtype=dt, device=device)


def gather_index_safe(self_t, dim, index_t, torch):
    """torch.gather requires index in [0, self.size(dim))."""
    dim_size = self_t.size(dim if dim >= 0 else self_t.dim() + dim)
    if index_t.dtype != torch.int64:
        index_t = index_t.to(torch.int64)
    return index_t.abs() % max(dim_size, 1)


def build_dual_calls(op_kind: str, case: dict, device: str, torch, torch_npu):
    """Build (candidate_call, baseline_call) on SAME `device` per PR #103.

    For port_a3_to_a5 mode: both candidate and baseline currently dispatch
    to upstream .so (until P89 registers our build). Ratio ~ 1.0× by
    construction — explicit, not hidden. Caller writes this expectation
    into the report.

    Returns (cand_fn, base_fn, op_name_for_report).
    """
    inputs = {e["name"]: e for e in case["inputs"]}
    if op_kind == "adaptive_avg_pool3d":
        x_spec = inputs["input"]
        x = make_tensor(x_spec, device, torch)
        out_size_vals = [e["value"] for e in case["inputs"] if e["name"] == "outputSize"]
        out_size = tuple(out_size_vals) if len(out_size_vals) == 3 else int(out_size_vals[0])
        import torch.nn.functional as F
        # Candidate: torch.ops.npu.* if it exists; else use F.* (same .so endpoint)
        # Baseline: F.adaptive_avg_pool3d (canonical reference)
        cand_op = getattr(torch.ops.npu, "adaptive_avg_pool3d",
                          getattr(torch.ops.aten, "adaptive_avg_pool3d", None))

        def cand():
            if cand_op is not None and hasattr(cand_op, "__call__"):
                try:
                    return cand_op(x, out_size)
                except Exception:
                    return F.adaptive_avg_pool3d(x, out_size)
            return F.adaptive_avg_pool3d(x, out_size)

        def base():
            return F.adaptive_avg_pool3d(x, out_size)
        return cand, base, "AdaptiveAvgPool3d"

    elif op_kind == "gather":
        self_spec = inputs["self"]
        idx_spec = inputs["index"]
        dim = inputs["dim"]["value"]
        self_t = make_tensor(self_spec, device, torch)
        idx_t = make_tensor(idx_spec, device, torch)
        idx_t = gather_index_safe(self_t, dim, idx_t, torch)
        # Candidate = baseline = torch.gather (port hasn't replaced upstream)

        def cand():
            return torch.gather(self_t, dim, idx_t)

        def base():
            return torch.gather(self_t, dim, idx_t)
        return cand, base, "Gather"

    elif op_kind == "apply_adam_w":
        # Schema (CANN 9.0.0, discovered via TypeError probe):
        # npu_apply_adam_w(Scalar beta1_power, Scalar beta2_power, Scalar lr,
        #   Scalar weight_decay, Scalar beta1, Scalar beta2, Scalar epsilon,
        #   Tensor grad, Tensor? max_grad_norm, bool? amsgrad, bool? maximize)
        #   -> (Tensor, Tensor, Tensor)
        tensor_specs = [(n, e) for n, e in inputs.items() if e["type"] == "tensor"]
        flt = [(n, e) for n, e in tensor_specs if e["dtype"] in ("fp32", "fp16", "bf16")]
        if not flt:
            return None, None, None
        grad = make_tensor(flt[0][1], device, torch)

        def cand():
            return torch_npu.npu_apply_adam_w(
                0.9, 0.999, 0.001, 0.01, 0.9, 0.999, 1e-8,
                grad, None, False, False,
            )
        # No "small-op decomposition" baseline on NPU for adam — skip
        return cand, cand, "ApplyAdamW"

    else:
        raise ValueError(f"unknown op_kind: {op_kind}")


def run_one_case(call, trace_dir: str, torch, torch_npu) -> float:
    """Measure kernel time via torch.npu.Event (NPU stream timing).

    Why Event instead of torch_npu.profiler: profiler post-processing
    (op_statistic.csv emission) is broken on the CANN 9.0.0 install at hand
    — trace_view.json comes out truncated and CPU-only, no NPU kernel data.
    torch.npu.Event.elapsed_time is the canonical NPU-side stream timing
    primitive; preserves PR #103 spirit (NPU-
    side timing, symmetric across A3/A5) without depending on profiler infra.

    Schedule still 5 warmup + 5 active per PR #103 convention.
    """
    # Warmup
    for _ in range(WARMUP):
        try:
            call()
        except Exception as e:
            print(f"  warmup raised: {type(e).__name__}: {e}", file=sys.stderr)
            return float("nan")
    torch.npu.synchronize()

    # Active: per-step Event timing, return MEAN over ACTIVE steps
    times_us = []
    for _ in range(ACTIVE):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        try:
            call()
        except Exception as e:
            print(f"  active raised: {type(e).__name__}: {e}", file=sys.stderr)
            return float("nan")
        end.record()
        torch.npu.synchronize()
        ms = start.elapsed_time(end)  # milliseconds
        times_us.append(ms * 1000.0)
    if not times_us:
        return float("nan")
    return sum(times_us) / len(times_us)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--op-call", required=True,
                    choices=["adaptive_avg_pool3d", "gather", "apply_adam_w"])
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--trace-root", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--host-label", default=os.environ.get("HOSTNAME", "unknown"))
    args = ap.parse_args()

    try:
        import torch
        import torch_npu
    except ImportError as e:
        sys.exit(f"FATAL: torch/torch_npu unavailable: {e}")

    cases = [json.loads(ln) for ln in args.cases.read_text().splitlines() if ln.strip()]
    args.trace_root.mkdir(parents=True, exist_ok=True)

    results = []
    for case in cases:
        try:
            cand_call, base_call, op_name = build_dual_calls(
                args.op_call, case, args.device, torch, torch_npu)
        except Exception as e:
            results.append({"case_id": case["case_id"], "skipped": True,
                            "reason": f"build_dual_calls: {type(e).__name__}: {e}"})
            print(f"[skip] case {case['case_id']}: {e}", file=sys.stderr)
            continue
        if cand_call is None:
            results.append({"case_id": case["case_id"], "skipped": True,
                            "reason": "case schema unmappable"})
            continue
        trace_dir_c = str(args.trace_root / f"case_{case['case_id']:04d}" / "candidate")
        trace_dir_b = str(args.trace_root / f"case_{case['case_id']:04d}" / "baseline")
        os.makedirs(trace_dir_c, exist_ok=True)
        os.makedirs(trace_dir_b, exist_ok=True)
        t0 = time.time()
        cand_us = run_one_case(cand_call, trace_dir_c, torch, torch_npu)
        base_us = run_one_case(base_call, trace_dir_b, torch, torch_npu)
        wall = time.time() - t0
        ratio = (base_us / cand_us) if (cand_us and cand_us == cand_us and cand_us > 0) else float("nan")
        results.append({
            "case_id": case["case_id"],
            "scale_bucket": case["scale_bucket"],
            "numel": case["numel"],
            "dtype": next((e["dtype"] for e in case["inputs"] if e["type"] == "tensor"), "?"),
            "candidate_us": cand_us,
            "baseline_us": base_us,
            "ratio_base_over_cand": ratio,
            "wall_s": wall,
        })
        print(f"case {case['case_id']:>3} {case['scale_bucket']} numel={case['numel']:>10} "
              f"dtype={results[-1]['dtype']:>5}: cand={cand_us:.2f}us "
              f"base={base_us:.2f}us ratio={ratio:.3f} wall={wall:.1f}s",
              flush=True)

    valid = []
    for result in results:
        candidate_us = result.get("candidate_us")
        baseline_us = result.get("baseline_us")
        if (
            not result.get("skipped")
            and candidate_us == candidate_us
            and baseline_us == baseline_us
        ):
            valid.append(result)
    ratios = [r["ratio_base_over_cand"] for r in valid
              if r["ratio_base_over_cand"] == r["ratio_base_over_cand"]]
    out = {
        "host": args.host_label,
        "op_call": args.op_call,
        "cases_file": str(args.cases),
        "profiler_config": {"warmup": WARMUP, "active": ACTIVE, "wait": WAIT, "repeat": 1},
        "method": (
            f"torch.npu.Event.elapsed_time warmup={WARMUP} active={ACTIVE} "
            f"on NPU stream, PR #103 dual-path same-NPU (candidate vs baseline). "
            f"Profiler unavailable on CANN 9.0.0 install; Event is symmetric "
            f"NPU-side primitive; method byte-equal must hold both A3 and A5."
        ),
        "report_format": "PR #103 dual-path",
        "candidate_path": (
            "torch.ops.npu.* if registered else falls back to baseline "
            "(port-as-wiring state when P89 pending)"
        ),
        "baseline_path": "torch native API (F.* / torch.gather) on same NPU device",
        "results": results,
        "summary": {
            "valid_count": len(valid),
            "skipped_count": sum(1 for r in results if r.get("skipped")),
            "mean_ratio_base_over_cand": (sum(ratios) / len(ratios)) if ratios else None,
            "ratio_interpretation_warning": (
                "If candidate and baseline dispatch to the same .so (P89 pending), "
                "ratio MUST be ~1.0× by construction. Any deviation is measurement "
                "noise, not optimization. Report as 'port-as-wiring' state, not "
                "kernel speedup."
            ),
        },
    }
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote: {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
