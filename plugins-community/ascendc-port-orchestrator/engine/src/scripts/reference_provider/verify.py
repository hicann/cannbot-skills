# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""A5-side verification harness for external-reference op-gen mode.

Loads reference_dataset.pt, imports ModelNew from a kernel dir, runs each case
through ModelNew, bit-exact compares against the pre-computed reference outputs.

Usage:
    python3 verify.py <kernel_dir> [--ref /path/to/reference_dataset.pt]

    kernel_dir    must contain model_new_ascendc.py + kernel/build/*.so
    --ref         defaults to /root/<op>_pilot/reference_dataset.pt (see --help);
                  set via env var AFAP_REF or CLI.

Exit: 0 if all cases PASS bit-exact; 1 if any FAIL; 2 on setup error.

PARAMETERIZATION — supports two ModelNew call conventions:
  1. positional: `model_new(*inputs.values())` — arg order from dataset inputs dict
  2. keyword:    `model_new(**inputs)` — dict-splat — use --call-mode kwargs

Pick the convention matching your pybind signature. The call convention is
orthogonal to the kernel's semantics; it's just Python plumbing.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import sys

import torch


def load_model_new(kernel_dir: pathlib.Path):
    mod_path = kernel_dir / "model_new_ascendc.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"{mod_path} not found")
    sys.path.insert(0, str(kernel_dir))
    spec = importlib.util.spec_from_file_location("_model_new", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ModelNew


def to_device_recursive(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device_recursive(v, device) for k, v in obj.items()}
    return obj


def _tolerance_defaults(dtype: torch.dtype) -> tuple[float, float]:
    """Cross-platform reduction-order RE tolerances (P-P58 sub-case).

    For point-wise ops, bit-exact is the norm (rtol=0/atol=0). For reductions
    — where different reduction trees produce different
    rounding paths — scaled drift is expected. Defaults here follow typical
    "sum of N fp32 terms" heuristics:
      fp32: rtol=1e-5 (≈84x eps), atol=1e-6 (small-magnitude floor)
      fp16: rtol=1e-3 (≈10x eps),  atol=1e-4
      bf16: rtol=1e-2 (≈1.3x eps), atol=1e-3
      fp64: rtol=1e-12,            atol=1e-14
    CLI flags --rtol/--atol override per-run.
    """
    return {
        torch.float32: (1e-5, 1e-6),
        torch.float16: (1e-3, 1e-4),
        torch.bfloat16: (1e-2, 1e-3),
        torch.float64: (1e-12, 1e-14),
    }.get(dtype, (1e-5, 1e-6))


def _coarser_float_dtype(d1: torch.dtype, d2: torch.dtype) -> torch.dtype:
    """Return the lower-precision (looser-tolerance) of two floating dtypes.

    task#15(iii) fp16-aware evaluator (OL-109 / OL-83 / P-P58): when a kernel
    output and the oracle reference differ in float precision (e.g. fp16-FA
    kernel vs fp32 oracle), the comparison tolerance MUST be keyed to the
    COARSER dtype — a low-precision output cannot be held to high-precision
    tolerance (doing so spuriously FAILs a numerically-correct fp16 kernel;
    this caused the contested FA-A3 pass count). NOT bar-lowering: it is the
    correct precision-aware tolerance for the lower-precision operand.

    Precision rank by mantissa bits: fp64 > fp32 > fp16 > bf16.
    """
    rank = {torch.float64: 3, torch.float32: 2, torch.float16: 1, torch.bfloat16: 0}
    return d1 if rank.get(d1, 2) <= rank.get(d2, 2) else d2


def run_one_case(model_new, case: dict, device, call_mode: str,
                 tolerance_mode: bool = False,
                 rtol_override: float | None = None,
                 atol_override: float | None = None) -> tuple[bool, str]:
    inputs_dev = to_device_recursive(case["inputs"], device)
    ref_outputs = case["outputs"]  # dict: name -> tensor (on CPU)

    if call_mode == "positional":
        cand = model_new(*inputs_dev.values())
    else:  # kwargs
        cand = model_new(**inputs_dev)

    # Normalize cand to dict with same keys as ref_outputs
    if isinstance(cand, torch.Tensor):
        if len(ref_outputs) != 1:
            return False, f"ModelNew returned single tensor but reference has {len(ref_outputs)} outputs"
        cand = {next(iter(ref_outputs.keys())): cand}
    elif isinstance(cand, (tuple, list)):
        names = list(ref_outputs.keys())
        if len(cand) != len(names):
            return False, f"ModelNew returned {len(cand)} tensors, reference has {len(names)}"
        cand = dict(zip(names, cand))
    elif not isinstance(cand, dict):
        return False, f"ModelNew returned unsupported type: {type(cand).__name__}"

    for name, ref_t in ref_outputs.items():
        if name not in cand:
            return False, f"ModelNew missing output '{name}'"
        c_t = cand[name].detach().to("cpu")
        if c_t.shape != ref_t.shape:
            return False, f"{name} shape mismatch cand={c_t.shape} ref={ref_t.shape}"
        cross_precision = False
        if c_t.dtype != ref_t.dtype:
            # task#15(iii): a lower-precision kernel output vs a higher-precision
            # oracle (e.g. fp16-FA vs fp32-oracle) is a LEGITIMATE comparison in
            # tolerance-mode — compare in common precision, tolerance keyed to the
            # coarser dtype (see _coarser_float_dtype). Only float-vs-float; a
            # float-vs-int/bool mismatch is still a real contract bug → hard-fail.
            if tolerance_mode and c_t.is_floating_point() and ref_t.is_floating_point():
                cross_precision = True
            else:
                return False, f"{name} dtype mismatch cand={c_t.dtype} ref={ref_t.dtype}"
        # Bit-exact compare with NaN-awareness (case_gen production tier may emit NaN/Inf).
        # Two tensors are "bit-exact" here iff: NaN positions identical AND Inf positions
        # identical AND (non-special) values bit-identical.
        c_nan = torch.isnan(c_t)
        r_nan = torch.isnan(ref_t)
        c_inf = torch.isinf(c_t)
        r_inf = torch.isinf(ref_t)
        if not torch.equal(c_nan, r_nan):
            return False, f"{name} NaN-mask mismatch: n_cand_nan={int(c_nan.sum())} n_ref_nan={int(r_nan.sum())}"
        if not torch.equal(c_inf, r_inf):
            return False, f"{name} Inf-mask mismatch: n_cand_inf={int(c_inf.sum())} n_ref_inf={int(r_inf.sum())}"
        finite_mask = ~(c_nan | c_inf)
        c_finite = c_t[finite_mask]
        r_finite = ref_t[finite_mask]
        if cross_precision:
            # Promote both to fp32 so equal/diff/allclose are well-defined across
            # the precision gap; tolerance is keyed to the coarser dtype below.
            c_finite = c_finite.to(torch.float32)
            r_finite = r_finite.to(torch.float32)
        if torch.equal(c_finite, r_finite):
            continue  # bit-exact on this output; next output
        # Not bit-exact: either tolerance-mode accepts, or we fail
        diff = (c_finite - r_finite).abs()
        max_diff = float(diff.max().item()) if diff.numel() else 0.0
        n_unequal = int((c_finite != r_finite).sum().item())
        flat_diff = (c_finite != r_finite).nonzero().flatten()
        idx = int(flat_diff[0].item()) if flat_diff.numel() else -1
        if tolerance_mode:
            # task#15(iii): cross-precision compares key tolerance to the COARSER
            # dtype (fp16-FA vs fp32-oracle → fp16 tolerance, not fp32). Same-dtype
            # path unchanged (coarser(X,X)==X) so non-FA ops are unaffected.
            tol_dtype = (_coarser_float_dtype(c_t.dtype, ref_t.dtype)
                         if cross_precision else ref_t.dtype)
            rtol, atol = _tolerance_defaults(tol_dtype)
            if rtol_override is not None:
                rtol = rtol_override
            if atol_override is not None:
                atol = atol_override
            if torch.allclose(c_finite, r_finite, rtol=rtol, atol=atol):
                # passed within tolerance — report drift for info
                return True, (f"{name} within-tolerance rtol={rtol:g} atol={atol:g} "
                              f"max_abs_diff={max_diff:.3g} n_unequal={n_unequal}/{c_finite.numel()}")
            return False, (f"{name} OVER-tolerance rtol={rtol:g} atol={atol:g} "
                           f"unequal_elements={n_unequal}/{c_finite.numel()} "
                           f"max_abs_diff={max_diff:.3g} first_finite_idx={idx} "
                           f"ref={r_finite.flatten()[idx].item():.9g} "
                           f"cand={c_finite.flatten()[idx].item():.9g}")
        return False, (f"{name} unequal_elements={n_unequal}/{c_finite.numel()} "
                       f"max_abs_diff={max_diff:.3g} first_finite_idx={idx} "
                       f"ref={r_finite.flatten()[idx].item():.9g} "
                       f"cand={c_finite.flatten()[idx].item():.9g}")
    return True, ("within-tolerance" if tolerance_mode else "bit-exact")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel_dir")
    ap.add_argument("--ref", default=os.environ.get("AFAP_REF"), help="path to reference_dataset.pt")
    ap.add_argument("--call-mode", choices=["positional", "kwargs"], default="positional")
    ap.add_argument("--tolerance-mode", action="store_true",
                    help="Accept cross-platform reduction-order drift within per-dtype defaults (P-P58).")
    ap.add_argument("--rtol", type=float, default=None, help="Override rtol in tolerance mode.")
    ap.add_argument("--atol", type=float, default=None, help="Override atol in tolerance mode.")
    args = ap.parse_args()

    kd = pathlib.Path(args.kernel_dir).resolve()
    if not kd.is_dir():
        print(f"not a dir: {kd}", file=sys.stderr)
        sys.exit(2)

    ref_path = pathlib.Path(args.ref) if args.ref else None
    if ref_path is None or not ref_path.exists():
        print("reference_dataset.pt not found (--ref / AFAP_REF). Set via --ref or env AFAP_REF.", file=sys.stderr)
        sys.exit(2)

    try:
        import torch_npu  # noqa: F401
        device = torch.device("npu:0")
    except Exception as e:
        print(f"torch_npu not available: {e}", file=sys.stderr)
        sys.exit(3)

    ref = torch.load(ref_path, weights_only=False)
    print(f"reference: {ref['reference_source']['platform']} {ref['reference_source']['device']} "
          f"(torch={ref['reference_source']['torch_version']})", flush=True)
    print(f"kernel_dir: {kd}", flush=True)
    print(f"cases: {len(ref['cases'])}", flush=True)

    ModelNew = load_model_new(kd)
    model = ModelNew().to(device) if hasattr(ModelNew, "to") else ModelNew()

    n_pass = n_fail = 0
    fails = []
    print("-" * 72, flush=True)
    for case in ref["cases"]:
        ok, msg = run_one_case(model, case, device, args.call_mode,
                               tolerance_mode=args.tolerance_mode,
                               rtol_override=args.rtol, atol_override=args.atol)
        tag = "PASS" if ok else "FAIL"
        print(f"case[{case['idx']:2d}] {case['name']:20s} shape={case['shape']} : {tag}  {msg}", flush=True)
        n_pass += int(ok)
        n_fail += int(not ok)
        if not ok:
            fails.append(case["idx"])
    print("-" * 72, flush=True)
    mode_label = "within-tolerance" if args.tolerance_mode else "bit-exact"
    print(f"Result: {n_pass}/{n_pass+n_fail} PASS ({mode_label} vs external reference)")
    if fails:
        print(f"failures: {fails}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
