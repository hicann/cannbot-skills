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

"""CDV per-case error collector — runs IN the A3 container (NPU + built kernel).

Op-agnostic (reads forward_spec.forward + BACKWARD_SPEC). For each (case × dtype):
  * REPRESENTATIVE: K randn draws (different seeds) → large statistical sample.
  * EDGE: zeros / large / small / boundary value-profiles (bug-finding).
Golden = fp64 torch.autograd.grad on the SAME dtype-quantized inputs the kernel
sees (inputs cast to dtype, then upcast to fp64) → isolates the KERNEL's numeric
error (not input-quantization). Kernel = model_new_ascendc.ModelNew on NPU.

Per (case,dtype,profile,output): MERE (mean rel err), MARE (max rel err),
RMSE, max_abs_diff. Writes per_case_errors.json.

T2 (double-baseline ratio, ADDITIVE — owner-requested "需要补", 2026-06-14): for each
record ALSO compute the comparator error vs the SAME fp64 golden. The comparator is the
same-precision CPU torch autograd backward and is stored under the canonical
baseline_{mere,mare,rmse} field names with competitor_kind="torch_same_dtype_cpu".
T2 is purely additive: T1 (same-dtype absolute threshold) stays strict and unchanged.

Usage (in container, cwd = deployed task dir): python3 cdv_collect.py [K] [out.json]
"""
import json
import sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import torch_npu  # noqa
import model_new_ascendc as _mna  # noqa
from forward_spec import forward as FWD  # noqa

# Resolve the kernel-wrapper class robustly. The OL-160 anti-cheat scanner
# (finalize_pipeline.py:3807) already accepts BOTH names —
# `node.name in ("ModelNew", "Model")` — so a consumer tool MUST accept both too,
# else it (not the safety net) is the brittle link. `ModelNew` is the canonical
# preferred name; `Model` is tolerated (some backward archives emit it).
_KERNEL_CLASS_NAMES = ("ModelNew", "Model")  # mirror finalize_pipeline.py:3807
ModelNew = next((getattr(_mna, n) for n in _KERNEL_CLASS_NAMES if hasattr(_mna, n)), None)
if ModelNew is None:
    raise ImportError(
        f"model_new_ascendc.py defines none of {_KERNEL_CLASS_NAMES} "
        "(expected the kernel-wrapper nn.Module)")

# BACKWARD_SPEC lives in the forward spec module copy; fall back to backward_ref.json
try:
    from forward_spec import BACKWARD_SPEC as SPEC  # noqa
except Exception:
    SPEC = None

_DT = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
REP_PROFILE = "randn"
EDGE_PROFILES = ["zeros", "large", "small", "boundary"]


def _resolve_shape(shape_decl, binding):
    return [binding[d] if isinstance(d, str) else int(d) for d in shape_decl]


def _apply_profile(t, profile, seed):
    g = torch.Generator().manual_seed(seed + 7)
    if profile == "randn":
        return t
    if profile == "zeros":
        mask = torch.rand(t.shape, generator=g) < 0.4
        return t.masked_fill(mask, 0.0)
    if profile == "large":
        return t * 100.0
    if profile == "small":
        return t * 1e-3
    if profile == "boundary":
        out = t.clone().flatten()
        vals = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5])
        n = max(1, out.numel() // 10)
        idx = torch.randint(0, out.numel(), (n,), generator=g)
        out[idx] = vals[torch.randint(0, len(vals), (n,), generator=g)]
        return out.reshape(t.shape)
    return t


def _metrics(kernel_t, golden_f64):
    k = kernel_t.detach().double().flatten()
    r = golden_f64.detach().double().flatten()
    d = (k - r).abs()
    denom = r.abs() + 1e-12
    rel = d / denom
    # restrict relative error to elements where |golden| is not negligible (small-value
    # handled separately by abs); use a magnitude floor at 2^-20 of the max |golden|.
    floor = r.abs().max().item() * (2 ** -20) if r.numel() else 0.0
    sig = r.abs() > max(floor, 1e-12)
    rel_sig = rel[sig] if sig.any() else rel
    return {
        "mere": float(rel_sig.mean().item()),
        "mare": float(rel_sig.max().item()),
        "rmse": float(torch.sqrt((d * d).mean()).item()),
        "max_abs_diff": float(d.max().item()),
    }


def _golden_fp64(fwd_args_dt, wrt, input_order, gy_dt):
    """fp64 autograd grads on the SAME dtype-quantized inputs (upcast to fp64)."""
    named = dict(zip(input_order, fwd_args_dt))
    leaves, cast = [], {}
    for name, t in named.items():
        if name in wrt:
            leaf = t.detach().double().requires_grad_(True)
            cast[name] = leaf
            leaves.append(leaf)
        elif torch.is_floating_point(t):
            cast[name] = t.detach().double()
        else:
            cast[name] = t.detach()
    with torch.enable_grad():
        y = FWD(**cast)
        outs = tuple(y) if isinstance(y, (tuple, list)) else (y,)
        gos = (gy_dt.detach().double(),) if gy_dt is not None else tuple(torch.ones_like(o) for o in outs)
        grads = torch.autograd.grad(outs, leaves, grad_outputs=gos, allow_unused=False)
    return list(grads)  # fp64


# T2 double-baseline comparator. The same-precision CPU torch-autograd result is
# explicitly labelled competitor_kind="torch_same_dtype_cpu". It runs the same
# forward+autograd in the kernel dtype `dt`, then downcasts grads to `dt` so the
# comparator carries the same rounding budget as the kernel.
COMPETITOR_KIND = "torch_same_dtype_cpu"


def _competitor_same_dtype(fwd_args_dt, wrt, input_order, gy_dt, dt):
    """SAME-dtype CPU torch autograd backward (the T2 标杆 / competitor).

    Inputs are already dtype-quantized to `dt` (identical to what the kernel sees and what
    the fp64 golden upcasts from). We run torch's own forward+autograd.grad in fp32
    (autograd needs a differentiable float dtype; fp16/bf16 autograd is unstable on CPU),
    then DOWNCAST each grad to `dt` so the competitor incurs the same dtype-rounding the
    kernel output does. Result returned upcast to fp64 for metric comparison vs the same
    fp64 golden. This is the apples-to-apples same-precision competitor of v2.1 §135.
    """
    named = dict(zip(input_order, fwd_args_dt))
    leaves, cast = [], {}
    for name, t in named.items():
        if name in wrt:
            leaf = t.detach().float().requires_grad_(True)
            cast[name] = leaf
            leaves.append(leaf)
        elif torch.is_floating_point(t):
            cast[name] = t.detach().float()
        else:
            cast[name] = t.detach()
    with torch.enable_grad():
        y = FWD(**cast)
        outs = tuple(y) if isinstance(y, (tuple, list)) else (y,)
        gos = (gy_dt.detach().float(),) if gy_dt is not None else tuple(torch.ones_like(o) for o in outs)
        grads = torch.autograd.grad(outs, leaves, grad_outputs=gos, allow_unused=False)
    # downcast each grad to the kernel dtype (same rounding budget), then upcast for metrics
    return [g.detach().to(dt).double() for g in grads]


def collect_cases(K=20):
    """In-container: build the per-(output×case) RAW-array `cases` list for the cannbot
    `precision_cannbot_adapter.grade_batch` (owner 2026-06-18: 完全照抄 cannbot — backward
    uses the SAME single judge as pass_a/benchmark/port_a3).

    Mirrors main()'s draw/golden/competitor/kernel loop EXACTLY, but returns RAW np.ndarray
    triples per (output×case) instead of pre-reduced scalar metrics: grade_batch computes
    MARE/MERE/ratio + small_value + inf_nan INTERNALLY, so do NOT pre-reduce or apply the
    2^-20 magnitude floor here (that floor is for the legacy scalar path only). Each case:
    {npu, golden(fp64), third_party(=same-precision-CPU competitor), dtype, output, is_edge}.
    is_edge = (profile != REP_PROFILE) → cannbot excludes edge from the statistical verdict
    (bug-find stream). golden/competitor are fp64 torch tensors → .numpy() = float64.
    """
    spec = SPEC or json.loads((_HERE / "backward_ref.json").read_text()).get("spec", {})
    input_order = list(spec["inputs"].keys())
    wrt = spec["wrt"]
    cases = spec["cases"]
    dtypes = spec["dtypes"]
    explicit = spec.get("grad_output", "explicit") == "explicit"
    base_seed = spec.get("seed", 1234)

    from backward_cdv_grade import _as_np  # bf16-safe coercion (numpy has no bf16 → lossless fp32 upcast)
    model = ModelNew()
    out_cases = []
    for case in cases:
        for dtn in dtypes:
            dt = _DT[dtn]
            shapes = {n: _resolve_shape(spec["inputs"][n]["shape"], case) for n in input_order}
            probe = {n: torch.randn(shapes[n]) for n in input_order}
            yo = FWD(**probe)
            outs0 = tuple(yo) if isinstance(yo, (tuple, list)) else (yo,)
            oshape = tuple(outs0[0].shape)

            def draw(seed, profile, shapes=shapes, dt=dt, oshape=oshape):
                gg = torch.Generator().manual_seed(seed)
                fwd_args = []
                for i, n in enumerate(input_order):
                    t = torch.randn(shapes[n], generator=gg)
                    t = _apply_profile(t, profile, seed + i)
                    fwd_args.append(t.to(dt))
                gy = torch.randn(oshape, generator=torch.Generator().manual_seed(
                    seed + 999)).to(dt) if explicit else None
                return fwd_args, gy

            plan = [(base_seed + 1000 + k, REP_PROFILE) for k in range(K)] + \
                   [(base_seed + 5000 + j, p) for j, p in enumerate(EDGE_PROFILES)]
            for seed, profile in plan:
                fwd_args, gy = draw(seed, profile)
                golden = _golden_fp64([t.clone() for t in fwd_args], wrt, input_order, gy)
                competitor = _competitor_same_dtype([t.clone() for t in fwd_args], wrt, input_order, gy, dt)
                npu_args = [t.npu() for t in fwd_args] + ([gy.npu()] if gy is not None else [])
                kres = model(*npu_args)
                kres = kres if isinstance(kres, (tuple, list)) else (kres,)
                for oi, wname in enumerate(wrt):
                    out_cases.append({
                        "npu": _as_np(kres[oi]),               # raw kernel output (bf16-safe)
                        "golden": _as_np(golden[oi]),          # fp64 truth
                        # ONE tensor, TWO consumers: the §135 same-precision-CPU autograd is BOTH the
                        # 生态-DEFAULT native_output (compare.py carve-out) AND the optional 商用 标杆.
                        "native": _as_np(competitor[oi]),      # 生态 DEFAULT: CPU-same-precision baseline
                        "native_kind": "cpu_same_precision",   # provenance guard tag (adapter honors only this)
                        "third_party": _as_np(competitor[oi]),  # optional independent ratio baseline
                        "dtype": dtn,
                        "output": wname,
                        "is_edge": (profile != REP_PROFILE),
                    })
    return out_cases


def cases_from_records(records, *, model=None):
    """VERIFY-SIDE cannbot `cases` builder — from STORED backward_cpu_truth.pt records
    (owner 2026-06-18: 完全照抄 cannbot — feeds the SAME single judge `grade_batch` via
    backward_cdv_grade.grade_cases that pass_a/benchmark/port_a3 use). TWO case-sources,
    ONE judge:
      * collect_cases (above)      = COLLECT/PROVISION side — RE-DRAWS inputs while
        building the truth dataset.
      * cases_from_records (here)  = VERIFY side — LOADS each record's OWN stored
        inputs/grads from `backward_cpu_truth.pt` and does **NOT re-seed** (守 no-reseed:
        the inputs are edge-profiled — re-drawing torch.randn would produce the WRONG
        inputs and mismatch the stored fp64 truth).

    Per status=='ok' record: npu = run the kernel on the record's OWN inputs (+grad_outputs)
    on NPU; golden = `record["grads"]` (fp64 truth already computed by phase_o25_backward);
    competitor = same-precision-CPU autograd on the SAME record inputs (mirror
    `_competitor_same_dtype`). One case per (output × record) via the pure
    `backward_cdv_grade._build_record_cases` (CPU-unit-testable).

    NPU/torch-runtime. Single-output-gy contract (mirrors the existing
    `_competitor_same_dtype`/`_golden_fp64` single-gy assumption that collect_cases already
    relies on; multi-output support is a separate follow-up and is not introduced here).
    """
    from backward_cdv_grade import _build_record_cases  # CPU-clean (no torch_npu)

    spec = SPEC or json.loads((_HERE / "backward_ref.json").read_text()).get("spec", {})
    input_order = list(spec["inputs"].keys())
    wrt = spec["wrt"]
    model = model if model is not None else ModelNew()
    out_cases = []
    for rec in records:
        if rec.get("status") != "ok":
            continue  # skipped (§5.4 non-finite truth) records carry no grads
        dtn = rec["dtype"]
        dt = _DT[dtn]
        profile = rec.get("profile", REP_PROFILE)
        inputs = rec["inputs"]                          # dict name->tensor (already dtype-quantized)
        fwd_args = [inputs[n].to(dt) for n in input_order]
        gy = rec.get("grad_outputs")                    # single tensor | None (single-output contract)
        golden = rec["grads"]                           # dict name->fp64 truth
        # Compute the same-precision CPU autograd once and reuse it as both the native
        # baseline and the optional independent comparator.
        native_seq = _competitor_same_dtype([t.clone() for t in fwd_args], wrt, input_order, gy, dt)
        competitor, competitor_kind = native_seq, COMPETITOR_KIND
        npu_args = [t.npu() for t in fwd_args] + ([gy.npu()] if gy is not None else [])
        kres = model(*npu_args)
        kres = kres if isinstance(kres, (tuple, list)) else (kres,)
        out_cases.extend(_build_record_cases(
            kres, golden, competitor, wrt, dtn, profile,
            competitor_kind=competitor_kind, native_grads=native_seq))
    return out_cases


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out = sys.argv[2] if len(sys.argv) > 2 else "per_case_errors.json"
    spec = SPEC or json.loads((_HERE / "backward_ref.json").read_text()).get("spec", {})
    input_order = list(spec["inputs"].keys())
    wrt = spec["wrt"]
    cases = spec["cases"]
    dtypes = spec["dtypes"]
    explicit = spec.get("grad_output", "explicit") == "explicit"
    base_seed = spec.get("seed", 1234)

    recs = []
    model = ModelNew()
    for case in cases:
        for dtn in dtypes:
            dt = _DT[dtn]
            shapes = {n: _resolve_shape(spec["inputs"][n]["shape"], case) for n in input_order}
            # determine output shape via a probe forward (fp32)
            probe = {n: torch.randn(shapes[n]) for n in input_order}
            yo = FWD(**probe)
            outs0 = tuple(yo) if isinstance(yo, (tuple, list)) else (yo,)
            oshape = tuple(outs0[0].shape)

            def draw(seed, profile, shapes=shapes, dt=dt, oshape=oshape):
                gg = torch.Generator().manual_seed(seed)
                fwd_args = []
                for i, n in enumerate(input_order):
                    t = torch.randn(shapes[n], generator=gg)
                    t = _apply_profile(t, profile, seed + i)
                    fwd_args.append(t.to(dt))
                gy = torch.randn(oshape, generator=torch.Generator().manual_seed(
                    seed + 999)).to(dt) if explicit else None
                return fwd_args, gy

            plan = [(base_seed + 1000 + k, REP_PROFILE) for k in range(K)] + \
                   [(base_seed + 5000 + j, p) for j, p in enumerate(EDGE_PROFILES)]
            for seed, profile in plan:
                fwd_args, gy = draw(seed, profile)
                golden = _golden_fp64([t.clone() for t in fwd_args], wrt, input_order, gy)
                # T2 competitor (标杆): same-dtype CPU torch autograd vs the SAME fp64 golden.
                competitor = _competitor_same_dtype([t.clone() for t in fwd_args], wrt, input_order, gy, dt)
                npu_args = [t.npu() for t in fwd_args] + ([gy.npu()] if gy is not None else [])
                kres = model(*npu_args)
                kres = kres if isinstance(kres, (tuple, list)) else (kres,)
                for oi, wname in enumerate(wrt):
                    m = _metrics(kres[oi].cpu(), golden[oi])
                    cm = _metrics(competitor[oi], golden[oi])  # competitor err vs SAME golden
                    recs.append({"profile": profile, "dtype": dtn, "shape": shapes[input_order[0]],
                                 "output": wname, "ours_mere": m["mere"], "ours_mare": m["mare"],
                                 "ours_rmse": m["rmse"], "ours_max_abs_diff": m["max_abs_diff"],
                                 # T2 double-baseline same-precision CPU comparator.
                                 "competitor_kind": COMPETITOR_KIND,
                                 "baseline_mere": cm["mere"], "baseline_mare": cm["mare"],
                                 "baseline_rmse": cm["rmse"]})
    with open(out, "w") as report_file:
        json.dump(recs, report_file)
    nrep = sum(1 for r in recs if r["profile"] == REP_PROFILE)
    print(f"COLLECTED {len(recs)} records ({nrep} representative, {len(recs)-nrep} edge) -> {out} "
          f"[+T2 competitor={COMPETITOR_KIND}: baseline_* fields populated for double_baseline_ratio]")


if __name__ == "__main__":
    main()
