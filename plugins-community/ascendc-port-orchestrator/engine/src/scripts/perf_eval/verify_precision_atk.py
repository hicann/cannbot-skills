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
"""Precision verification on ATK cases (companion to benchmark_torch_op.py).

For each case in <op>_perf_cases.jsonl:
  1. Build inputs from ATK case spec (same as benchmark_torch_op.py)
  2. Run candidate path (torch_npu / our archive's .so) on NPU → npu_out
  3. Run reference path (torch CPU) → cpu_out
  4. Compare with dtype-tier tolerance:
       T1 (fp32 / int*):   atol=1e-5  rtol=1e-5
       T2 (fp16 / bf16):   atol=1e-2  rtol=5e-3
  5. Emit per-case PASS/PASS_T2/FAIL + max_abs_diff + max_rel_diff

Output: <op>_a5_atk_precision.json with full per-case verdicts.

Why this matters: prior verification.json `precision.status` was based on
8 toy cases. New ATK cases (S/M/L × full dtype) may surface dtype/L-scale
precision bugs the 8-case test couldn't catch.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

DTYPE_MAP = {
    "fp16": "float16", "fp32": "float32", "bf16": "bfloat16",
    "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
    "uint8": "uint8", "bool": "bool",
}
T1_DTYPES = {"fp32", "int8", "int16", "int32", "int64", "uint8", "bool"}
T2_DTYPES = {"fp16", "bf16"}


def make_cpu_tensor(spec, torch, seed=42):
    """Generate ONCE on CPU. Caller transfers via .to('npu:0') for NPU side
    so CPU + NPU compare identical inputs (bug fix 2026-05-16: independent
    CPU/NPU Generator produced different tensors → all FAIL was test artifact).
    """
    g = torch.Generator(device="cpu").manual_seed(
        seed + abs(hash(str(spec.get("shape", "")))) % 1000)
    dt = getattr(torch, DTYPE_MAP[spec["dtype"]])
    shape = spec["shape"]
    if dt.is_floating_point:
        return torch.randn(*shape, dtype=dt, generator=g)
    elif dt == torch.bool:
        return torch.randint(0, 2, shape, generator=g).bool()
    else:
        rv = spec.get("range_values")
        lo, hi = 0, 100
        if isinstance(rv, list) and len(rv) == 2 and all(isinstance(x, (int, float)) for x in rv):
            lo, hi = int(rv[0]), int(rv[1])
        # Clamp to dtype valid range (caught 2026-05-16: uint8 with lo=-100 errored)
        info = torch.iinfo(dt)
        lo = max(lo, info.min)
        hi = min(hi, info.max)
        if lo >= hi:
            lo, hi = info.min, info.max
        return torch.randint(lo, max(lo + 1, hi), shape, dtype=dt, generator=g)


def gather_index_safe(self_t, dim, index_t, torch):
    dim_size = self_t.size(dim if dim >= 0 else self_t.dim() + dim)
    if index_t.dtype != torch.int64:
        index_t = index_t.to(torch.int64)
    return index_t.abs() % max(dim_size, 1)


def build_paired_calls(op_kind, case, torch, seed=42):
    """Build CPU truth + NPU candidate with **identical input bit pattern**.

    For reduction-class ops on low-precision dtypes (bf16/fp16), CPU PyTorch
    computes natively in the input dtype (no internal promotion), whereas NPU
    AscendC computes with fp32 internal accumulation then casts back. Comparing
    those two directly mistakes "different algorithm" for "kernel bug" (caught
    by user 2026-05-16: "怎么又来硬件floor了").

    Fix: for FLOAT dtypes on reduction-class ops, CPU truth is computed via
    fp32-promotion-then-cast — matching what NPU actually does internally. For
    non-reduction (gather, index ops), bf16/fp16 native is fine since there's
    no accumulator.

    Returns (cpu_call_fn, npu_call_fn). Both return tensors in the original
    dtype (bf16/fp16 cast back where applicable).
    """
    inputs = {e["name"]: e for e in case["inputs"]}
    if op_kind == "adaptive_avg_pool3d":
        x_spec = inputs["input"]
        x_cpu = make_cpu_tensor(x_spec, torch, seed=seed)
        x_npu = x_cpu.to("npu:0")
        out_size_vals = [e["value"] for e in case["inputs"] if e["name"] == "outputSize"]
        out_size = tuple(out_size_vals) if len(out_size_vals) == 3 else int(out_size_vals[0])
        import torch.nn.functional as F
        # Reduction-class: CPU truth = fp32-promote-then-cast to match NPU internal
        is_low_precision = x_cpu.dtype in (torch.float16, torch.bfloat16)
        if is_low_precision:
            x_cpu_fp32 = x_cpu.float()

            def cpu_truth():
                return F.adaptive_avg_pool3d(x_cpu_fp32, out_size).to(x_cpu.dtype)
        else:
            def cpu_truth():
                return F.adaptive_avg_pool3d(x_cpu, out_size)

        def npu_cand():
            return F.adaptive_avg_pool3d(x_npu, out_size)
        return cpu_truth, npu_cand

    elif op_kind == "gather":
        # gather is index op, no accumulation — bf16/fp16 native is correct
        self_spec = inputs["self"]
        idx_spec = inputs["index"]
        dim = inputs["dim"]["value"]
        self_cpu = make_cpu_tensor(self_spec, torch, seed=seed)
        idx_cpu = make_cpu_tensor(idx_spec, torch, seed=seed + 1)
        idx_cpu = gather_index_safe(self_cpu, dim, idx_cpu, torch)
        self_npu = self_cpu.to("npu:0")
        idx_npu = idx_cpu.to("npu:0")
        return (lambda: torch.gather(self_cpu, dim, idx_cpu),
                lambda: torch.gather(self_npu, dim, idx_npu))
    else:
        raise ValueError(f"unsupported op_kind: {op_kind}")


def _verifier_for(language: str):
    """Lookup verifier via plugin registry. No local dtype/language dispatch."""
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from verifiers import get_verifier
    return get_verifier(language)


def compare(cpu_out, npu_out, language: str):
    """Delegate to plugin registry. Returns VerifyResult.

    Caller (orchestrator Phase O5) is expected to pass `language` from
    op_meta.target_language. This file's sole responsibility is forward
    the call — no fallback, no dtype branching.
    """
    return _verifier_for(language).verify(npu_out, cpu_out, baseline=None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--op-call", required=True, choices=["adaptive_avg_pool3d", "gather"])
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--language", default="ascendc",
                    help="Target language for precision rule (registry key). "
                         "The supported value is 'ascendc'.")
    args = ap.parse_args()

    import torch
    try:
        import torch_npu  # noqa: F401
    except ImportError as e:
        sys.exit(f"FATAL: torch_npu: {e}")

    cases = [json.loads(ln) for ln in args.cases.read_text().splitlines() if ln.strip()]
    results = []
    for case in cases:
        dtype = next((e["dtype"] for e in case["inputs"] if e["type"] == "tensor"), "?")
        rec = {"case_id": case["case_id"], "scale_bucket": case["scale_bucket"],
               "numel": case["numel"], "dtype": dtype}
        try:
            cpu_call, npu_call = build_paired_calls(args.op_call, case, torch)
            cpu_out = cpu_call()
            npu_out = npu_call()
            result = compare(cpu_out, npu_out, language=args.language)
            rec["verifier_result"] = result.to_json()
            rec["passed"] = result.status.value.startswith("PASS")
            rec["tier"] = result.tier
            rec["status"] = result.status.value
        except Exception as e:
            rec.update({"tier": "ERROR", "status": "EVAL_ERR",
                        "passed": False,
                        "error": f"{type(e).__name__}: {e}"[:200]})
        m = (rec.get("verifier_result") or {}).get("metrics", {})
        print(f"case {case['case_id']:>3} {case['scale_bucket']} dtype={dtype:>5} "
              f"{rec.get('status','?'):>10} tier={rec.get('tier','?'):<25} "
              f"metrics={m}", flush=True)
        results.append(rec)

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r.get("passed")),
        "failed": sum(1 for r in results if not r.get("passed") and r.get("tier") not in ("ERROR",)),
        "errored": sum(1 for r in results if r.get("tier") == "ERROR"),
        "by_tier": {},
    }
    for r in results:
        t = r.get("tier", "?")
        summary["by_tier"][t] = summary["by_tier"].get(t, 0) + 1
    out = {"results": results, "summary": summary,
           "method": "ATK cases × dtype-tier tolerance (T1 fp32 atol=1e-5, T2 fp16/bf16 atol=1e-2)",
           "tolerance_doc": "ASCEND_OP_PRECISION_STANDARD_v2.1.md §4.5.3"}
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nSummary: {summary}")
    print(f"Wrote: {args.out_json}")
    return 0 if summary["failed"] == 0 and summary["errored"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
