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

"""precision_eval_dual_ref.py — dual-tier precision evaluation.

Tier 1: ours_npu vs cpu_ref (CPU-truth, primary)
Tier 2: ours_npu vs cann_npu (CANN bit-exact fallback, only checked if Tier 1 fails)

A case PASSES iff Tier 1 passes OR (Tier 1 fails AND Tier 2 passes bit-exact).

Tier 2 is "CANN as implementation standard" — useful when CPU-NPU rounding chain
makes Tier 1 unreachable but the kernel correctly implements what NPU vendor does.

Usage:
  python3 precision_eval_dual_ref.py <archive_dir> [--quiet] [--json out.json]

Requires:
  - archive_dir/model.py with Model class (used as CPU reference)
  - archive_dir/model_new_ascendc.py with ModelNew class (our kernel)

Note: For Tier 2, this script computes CANN reference by running the SAME Model
class on NPU. If model.py has been migrated to use Python decomp instead of
torch_npu vendor (per OL-104), then Tier 2 == "ours_npu vs decomp_npu" which
verifies the kernel's NPU compute matches the same Python decomp on NPU.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
try:
    import torch_npu  # noqa: F401
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

EPS = 1e-7
PRECISION_THRESHOLDS = {
    torch.float16: 2 ** -10,
    torch.bfloat16: 2 ** -7,
    torch.float32: 2 ** -13,
}


def compute_mere_mare(a, g):
    a = a.detach().to(torch.float64).cpu()
    g = g.detach().to(torch.float64).cpu()
    diff = (a - g).abs()
    rel = diff / (g.abs() + EPS)
    if rel.numel() == 0:
        return 0.0, 0.0
    return float(rel.mean().item()), float(rel.max().item())


def case_passes_mere_mare(actual, golden):
    dtype = golden.dtype
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
        return bool(torch.equal(actual.cpu(), golden.cpu()))
    if dtype not in PRECISION_THRESHOLDS:
        return False
    thr = PRECISION_THRESHOLDS[dtype]
    mere, mare = compute_mere_mare(actual, golden)
    return (mere < thr) and (mare < 10 * thr)


def case_passes_bit_exact(a, b):
    """Tier 2: bit-exact match between ours and CANN."""
    return bool(torch.equal(a.cpu(), b.cpu()))


def to_dev(v, dev):
    if isinstance(v, torch.Tensor):
        return v.detach().to(dev).contiguous()
    if isinstance(v, (list, tuple)):
        return type(v)(to_dev(x, dev) for x in v)
    if isinstance(v, dict):
        return {k: to_dev(x, dev) for k, x in v.items()}
    return v


def normalize(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        return [t for t in out if isinstance(t, torch.Tensor)]
    if hasattr(out, "_asdict"):
        return [t for t in out._asdict().values() if isinstance(t, torch.Tensor)]
    if out is None:
        return []
    return []


def load_module(p, name):
    spec = importlib.util.spec_from_file_location(name, str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def evaluate(archive: Path, verbose=True):
    op_name = archive.name
    model_py = archive / "model.py"
    new_py = archive / "model_new_ascendc.py"

    mod_ref_cpu = load_module(model_py, f"{op_name}_ref_cpu")
    mod_ref_npu = load_module(model_py, f"{op_name}_ref_npu")
    mod_cand = load_module(new_py, f"{op_name}_cand")

    cls_ref_cpu = mod_ref_cpu.Model
    cls_ref_npu = mod_ref_npu.Model
    cls_cand = mod_cand.ModelNew

    torch.manual_seed(0)
    init = mod_ref_cpu.get_init_inputs() if hasattr(mod_ref_cpu, "get_init_inputs") else []
    get_inputs = getattr(mod_cand, "get_input_groups", None) or getattr(mod_ref_cpu, "get_input_groups", None)
    groups = get_inputs()

    cpu_model = (cls_ref_cpu(*copy.deepcopy(init)) if init else cls_ref_cpu()).cpu().eval()
    npu_ref_model = (cls_ref_npu(*copy.deepcopy(init)) if init else cls_ref_npu())
    if HAS_NPU:
        npu_ref_model = npu_ref_model.npu()
    npu_ref_model = npu_ref_model.eval()
    cand_model = (cls_cand(*copy.deepcopy(init)) if init else cls_cand())
    if HAS_NPU:
        cand_model = cand_model.npu()
    cand_model = cand_model.eval()

    results = []
    for idx, raw in enumerate(groups):
        try:
            cin = to_dev(raw, "cpu")
            nin = to_dev(raw, "npu") if HAS_NPU else cin

            with torch.no_grad():
                cpu_out = cpu_model(*cin) if isinstance(cin, (list, tuple)) else cpu_model(cin)
                # use deepcopy of nin for ref since some ops modify in-place
                nin_for_ref = to_dev(raw, "npu") if HAS_NPU else cin
                ref_out = npu_ref_model(*nin_for_ref) if isinstance(nin_for_ref,
                                         (list, tuple)) else npu_ref_model(nin_for_ref)
                cand_out = cand_model(*nin) if isinstance(nin, (list, tuple)) else cand_model(nin)

            cpu_t = normalize(cpu_out)
            ref_t = normalize(ref_out)
            cand_t = normalize(cand_out)

            if len(cand_t) != len(cpu_t):
                results.append({"case": idx, "tier1": False, "tier2": False, "passed": False,
                               "error": f"out count mismatch ours={len(cand_t)} cpu={len(cpu_t)}"})
                continue

            tier1_ok = True
            for c, r in zip(cand_t, cpu_t):
                cc = c.cpu().to(r.dtype) if r.dtype != c.dtype else c.cpu()
                if not case_passes_mere_mare(cc, r):
                    tier1_ok = False
                    break

            # Tier 2 fallback: only valid if CANN is deterministic for this op.
            # Determinism check: run npu_ref_model again on the same input, compare bit-exact.
            tier2_ok = False
            cann_deterministic = True
            if not tier1_ok and len(ref_t) == len(cand_t):
                # Re-run reference to check determinism
                with torch.no_grad():
                    nin_for_ref2 = to_dev(raw, "npu") if HAS_NPU else cin
                    ref_out2 = npu_ref_model(*nin_for_ref2) if isinstance(nin_for_ref2,
                                             (list, tuple)) else npu_ref_model(nin_for_ref2)
                ref_t2 = normalize(ref_out2)
                if len(ref_t2) == len(ref_t):
                    for r1, r2 in zip(ref_t, ref_t2):
                        if not torch.equal(r1.cpu(), r2.cpu()):
                            cann_deterministic = False
                            break
                else:
                    cann_deterministic = False

                if cann_deterministic:
                    tier2_ok = True
                    for c, r in zip(cand_t, ref_t):
                        cc = c.cpu().to(r.dtype) if r.dtype != c.dtype else c.cpu()
                        if not case_passes_bit_exact(cc, r):
                            tier2_ok = False
                            break

            results.append({"case": idx, "tier1": tier1_ok, "tier2": tier2_ok,
                            "cann_deterministic": cann_deterministic if not tier1_ok else None,
                            "passed": tier1_ok or tier2_ok, "error": None})
            if verbose:
                tag = "PASS-T1" if tier1_ok else ("PASS-T2" if tier2_ok else (
                    "FAIL-NONDET" if not cann_deterministic else "FAIL"))
                print(f"case[{idx}]: {tag}")
        except Exception as e:
            results.append({"case": idx, "tier1": False, "tier2": False, "passed": False,
                            "error": f"{type(e).__name__}: {e}"})
            if verbose:
                traceback.print_exc()

    n_total = len(results)
    n_t1 = sum(1 for r in results if r.get("tier1"))
    n_t2 = sum(1 for r in results if r.get("tier2"))
    n_pass = sum(1 for r in results if r["passed"])
    n_nondet = sum(1 for r in results if r.get("cann_deterministic") is False)
    return {"op": op_name, "n_total": n_total, "n_tier1": n_t1, "n_tier2_only": n_t2,
            "n_cann_nondet": n_nondet,
            "n_passed": n_pass, "n_failed": n_total - n_pass,
            "pass_rate": n_pass / n_total if n_total else 0.0, "results": results}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("archive_dir")
    p.add_argument("--json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    s = evaluate(Path(args.archive_dir).resolve(), verbose=not args.quiet)
    if not args.quiet:
        print(
            f"\n{s['op']}: tier1={s['n_tier1']}/{s['n_total']}  "
            f"+tier2_fallback={s['n_tier2_only']}  "
            f"total_pass={s['n_passed']}/{s['n_total']}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(s, indent=2, default=str))
    return 0 if s["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
