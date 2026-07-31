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

"""precision_eval_cpu_ref.py — MERE/MARE precision eval against CPU ground truth.

Mirrors the Ascend/agent-skills `ascendc-operator-precision-eval` SKILL: same
metric formulas, same per-dtype thresholds, same pass criteria. Differs from
br_430's `utils/verification_ascendc.py` only in REFERENCE OBJECT — production
skill uses CPU as ground truth (`x.cpu().float()` then op then cast back),
br_430 uses CANN/NPU result (which is what we are validating, so it's
circular for fp32 FMA-order disagreements).

Usage:
    python3 precision_eval_cpu_ref.py <archive_dir>
    python3 precision_eval_cpu_ref.py <archive_dir> --json <output.json>
    python3 precision_eval_cpu_ref.py <archive_dir> --quiet

Designed to be invoked from the same docker container as
verification_ascendc.py (i.e. NPU container with torch_npu installed).
The script does NOT touch performance — separate concern, run
performance.py against CANN as before.
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


# ---------------------------------------------------------------------------
# Precision standards — IDENTICAL to Ascend/agent-skills SKILL
# ---------------------------------------------------------------------------
# MERE = mean(|actual-golden|/(|golden|+eps))
# MARE = max (|actual-golden|/(|golden|+eps))
# Pass: MERE < Threshold AND MARE < 10 × Threshold
# ---------------------------------------------------------------------------
EPS = 1e-7
PRECISION_THRESHOLDS = {
    torch.float16: 2 ** -10,    # ≈ 9.77e-4
    torch.bfloat16: 2 ** -7,    # ≈ 7.81e-3
    torch.float32: 2 ** -13,    # ≈ 1.22e-4
    # int / bool: bit-exact required (handled separately)
}


def compute_mere_mare(actual: torch.Tensor, golden: torch.Tensor) -> tuple[float, float]:
    """Return (MERE, MARE) — same formula as production skill."""
    if actual.shape != golden.shape:
        raise ValueError(f"shape mismatch: actual={tuple(actual.shape)} golden={tuple(golden.shape)}")
    a = actual.detach().to(torch.float64).cpu()
    g = golden.detach().to(torch.float64).cpu()
    diff = (a - g).abs()
    rel = diff / (g.abs() + EPS)
    if rel.numel() == 0:
        return 0.0, 0.0
    return float(rel.mean().item()), float(rel.max().item())


def compute_aux_metrics(actual: torch.Tensor, golden: torch.Tensor) -> dict[str, float]:
    """Auxiliary metrics: max-abs-err, mean-abs-err, cosine-similarity."""
    a = actual.detach().to(torch.float32).cpu().flatten()
    g = golden.detach().to(torch.float32).cpu().flatten()
    if a.numel() == 0:
        return {"max_abs_err": 0.0, "mean_abs_err": 0.0, "cosine_sim": 1.0}
    diff = (a - g).abs()
    cs = float(torch.nn.functional.cosine_similarity(a.unsqueeze(
        0), g.unsqueeze(0)).item()) if a.norm() > 0 and g.norm() > 0 else 1.0
    return {
        "max_abs_err": float(diff.max().item()),
        "mean_abs_err": float(diff.mean().item()),
        "cosine_sim": cs,
    }


def case_passes(actual: torch.Tensor, golden: torch.Tensor) -> dict[str, Any]:
    """Compute pass/fail for one tensor pair against the dtype threshold."""
    dtype = golden.dtype
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
        # Integer / bool: bit-exact required
        match = bool(torch.equal(actual.cpu(), golden.cpu()))
        return {
            "dtype": str(dtype).replace("torch.", ""),
            "rule": "bit-exact",
            "passed": match,
            "mere": 0.0 if match else float("inf"),
            "mare": 0.0 if match else float("inf"),
            "threshold": 0.0,
            "mare_threshold": 0.0,
            **compute_aux_metrics(actual.float(), golden.float()),
        }
    if dtype not in PRECISION_THRESHOLDS:
        return {
            "dtype": str(dtype).replace("torch.", ""),
            "rule": "unknown-dtype",
            "passed": False,
            "mere": float("nan"),
            "mare": float("nan"),
            "threshold": float("nan"),
            "mare_threshold": float("nan"),
            **compute_aux_metrics(actual.float(), golden.float()),
        }
    threshold = PRECISION_THRESHOLDS[dtype]
    mare_threshold = 10 * threshold
    mere, mare = compute_mere_mare(actual, golden)
    aux = compute_aux_metrics(actual, golden)
    passed = (mere < threshold) and (mare < mare_threshold)
    return {
        "dtype": str(dtype).replace("torch.", ""),
        "rule": "MERE<T AND MARE<10T",
        "passed": passed,
        "mere": mere,
        "mare": mare,
        "threshold": threshold,
        "mare_threshold": mare_threshold,
        **aux,
    }


# ---------------------------------------------------------------------------
# Reference computation — run Model.forward on CPU
# ---------------------------------------------------------------------------
def to_cpu_for_ref(value: Any, upcast_dtype: torch.dtype | None = None) -> Any:
    """Move value to CPU, optionally upcast float dtypes to fp32 for stable internal compute."""
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and upcast_dtype is not None and value.dtype != upcast_dtype:
            return value.detach().cpu().to(upcast_dtype).contiguous()
        return value.detach().cpu().contiguous()
    if isinstance(value, (list, tuple)):
        cls = type(value)
        return cls(to_cpu_for_ref(v, upcast_dtype) for v in value)
    if isinstance(value, dict):
        return {k: to_cpu_for_ref(v, upcast_dtype) for k, v in value.items()}
    return value


def to_npu(value: Any) -> Any:
    """Move tensors to NPU for the candidate kernel call."""
    if isinstance(value, torch.Tensor):
        return value.detach().npu().contiguous()
    if isinstance(value, (list, tuple)):
        cls = type(value)
        return cls(to_npu(v) for v in value)
    if isinstance(value, dict):
        return {k: to_npu(v) for k, v in value.items()}
    return value


def normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        return [t for t in out if isinstance(t, torch.Tensor)]
    if hasattr(out, "_asdict"):
        return [t for t in out._asdict().values() if isinstance(t, torch.Tensor)]
    return []


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------
def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_model_class(mod, prefer: str):
    if hasattr(mod, prefer):
        return getattr(mod, prefer)
    for attr in dir(mod):
        if attr.startswith("Model") and isinstance(getattr(mod, attr), type):
            return getattr(mod, attr)
    raise AttributeError(f"no Model class in {mod}")


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------
def evaluate(archive_dir: Path, verbose: bool = True) -> dict[str, Any]:
    if not archive_dir.is_dir():
        raise FileNotFoundError(f"archive dir not found: {archive_dir}")

    model_py = archive_dir / "model.py"
    new_py = archive_dir / "model_new_ascendc.py"
    if not model_py.is_file():
        raise FileNotFoundError(f"missing {model_py}")
    if not new_py.is_file():
        raise FileNotFoundError(f"missing {new_py}")

    op_name = archive_dir.name
    if verbose:
        print(f"=== Precision eval (CPU reference) for {op_name} ===")

    # Load modules
    ref_mod = load_module(model_py, f"{op_name}_ref")
    cand_mod = load_module(new_py, f"{op_name}_cand")
    ref_cls = find_model_class(ref_mod, "Model")
    cand_cls = find_model_class(cand_mod, "ModelNew")

    # Init harness once with seed 0 (same as br_430 line 306)
    torch.manual_seed(0)
    init_inputs = []
    if hasattr(cand_mod, "get_init_inputs"):
        init_inputs = cand_mod.get_init_inputs()
    elif hasattr(ref_mod, "get_init_inputs"):
        init_inputs = ref_mod.get_init_inputs()

    # Get input cases — try cand first then ref (matches br_430 priority)
    get_inputs = getattr(cand_mod, "get_input_groups", None) or getattr(ref_mod, "get_input_groups", None)
    if get_inputs is None:
        raise AttributeError("no get_input_groups in either module")
    input_groups = get_inputs()

    # Construct models — ref on CPU, cand on NPU
    ref_model = ref_cls(*copy.deepcopy(init_inputs)).cpu().eval() if init_inputs else ref_cls().cpu().eval()
    cand_model = (cand_cls(*copy.deepcopy(init_inputs)) if init_inputs else cand_cls())
    if HAS_NPU:
        cand_model = cand_model.npu()
    cand_model = cand_model.eval()

    # Per-case eval
    results = []
    for idx, raw_inputs in enumerate(input_groups):
        try:
            cpu_inputs = to_cpu_for_ref(raw_inputs)  # keep original dtype (matches br_430)
            npu_inputs = to_npu(raw_inputs) if HAS_NPU else cpu_inputs

            with torch.no_grad():
                # CPU reference: run on CPU exactly as defined
                ref_out = ref_model(*cpu_inputs) if isinstance(cpu_inputs, (list, tuple)) else ref_model(cpu_inputs)
                # NPU candidate
                cand_out = cand_model(*npu_inputs) if isinstance(npu_inputs, (list, tuple)) else cand_model(npu_inputs)

            ref_tensors = normalize_outputs(ref_out)
            cand_tensors = normalize_outputs(cand_out)

            if len(ref_tensors) != len(cand_tensors):
                results.append({
                    "case": idx, "passed": False,
                    "error": f"output count mismatch: ref={len(ref_tensors)} cand={len(cand_tensors)}",
                    "outputs": [],
                })
                if verbose:
                    print(f"case[{idx}]: count mismatch — ref={len(ref_tensors)} cand={len(cand_tensors)}")
                continue

            per_out = []
            all_passed = True
            for j, (rt, ct) in enumerate(zip(ref_tensors, cand_tensors)):
                # Ensure same dtype for comparison: cast cand to ref dtype
                ct_cast = ct.cpu().to(rt.dtype) if rt.dtype != ct.dtype else ct.cpu()
                m = case_passes(ct_cast, rt)
                m["output_idx"] = j
                m["shape"] = list(rt.shape)
                per_out.append(m)
                if not m["passed"]:
                    all_passed = False

            results.append({"case": idx, "passed": all_passed, "outputs": per_out, "error": None})
            if verbose:
                if all_passed:
                    summary = ", ".join(
                        f"o{m['output_idx']} MERE={m['mere']:.2e} MARE={m['mare']:.2e}" for m in per_out)
                    print(f"case[{idx}]: PASS  {summary}")
                else:
                    summary = ", ".join(
                        f"o{m['output_idx']} MERE={m['mere']:.2e} "
                        f"MARE={m['mare']:.2e} [{m['dtype']} "
                        f"thr={m['threshold']:.2e}]"
                        for m in per_out
                        if not m["passed"]
                    )
                    print(f"case[{idx}]: FAIL  {summary}")
        except Exception as e:
            results.append({"case": idx, "passed": False, "error": f"{type(e).__name__}: {e}", "outputs": []})
            if verbose:
                print(f"case[{idx}]: ERROR  {type(e).__name__}: {e}")
                traceback.print_exc()

    # Summary
    n_total = len(results)
    n_passed = sum(1 for r in results if r["passed"])
    summary = {
        "op": op_name,
        "archive": str(archive_dir),
        "reference": "CPU (Model.forward on CPU)",
        "metric": "MERE/MARE per Ascend/agent-skills SKILL",
        "thresholds": {str(k).replace("torch.", ""): v for k, v in PRECISION_THRESHOLDS.items()},
        "n_total": n_total,
        "n_passed": n_passed,
        "n_failed": n_total - n_passed,
        "pass_rate": (n_passed / n_total) if n_total else 0.0,
        "results": results,
    }

    if verbose:
        print(f"\n--- Summary: {op_name} ---")
        print(f"  total: {n_total}  passed: {n_passed}  failed: {n_total - n_passed}  rate: {summary['pass_rate']:.2%}")

    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive_dir", help="op archive dir (e.g. output/.../1_BatchMatmul)")
    p.add_argument("--json", help="write json report to this path")
    p.add_argument("--quiet", action="store_true", help="suppress per-case stdout")
    args = p.parse_args(argv)

    archive = Path(args.archive_dir).resolve()
    summary = evaluate(archive, verbose=not args.quiet)

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2, default=str))
        print(f"wrote json: {args.json}")
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
