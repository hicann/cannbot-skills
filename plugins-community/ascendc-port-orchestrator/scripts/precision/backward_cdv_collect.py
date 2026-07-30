#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CDV per-case error collector — runs IN the A3 container (NPU + built kernel).

Op-agnostic (reads forward_spec.forward + BACKWARD_SPEC). For each (case × dtype):
  * REPRESENTATIVE: K randn draws (different seeds) → large statistical sample.
  * EDGE: zeros / large / small / boundary value-profiles (bug-finding).
Golden = fp64 torch.autograd.grad on the SAME dtype-quantized inputs the kernel
sees (inputs cast to dtype, then upcast to fp64) → isolates the KERNEL's numeric
error (not input-quantization). Kernel = model_new_ascendc.ModelNew on NPU.

Per (case,dtype,profile,output): MERE (mean rel err), MARE (max rel err),
RMSE, max_abs_diff. Writes per_case_errors.json.

T2 (double-baseline ratio, ADDITIVE): for each record also compute a declared
same-precision CPU torch-autograd comparator against the same fp64 golden. The
metrics use `baseline_{mere,mare,rmse}` and record
`competitor_kind="torch_same_dtype_cpu"`. T2 is purely additive: T1 remains the
strict primary gate.

Usage (in container, cwd = deployed task dir): python3 cdv_collect.py [K] [out.json]
"""
import json
import sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import torch_npu  # noqa: E402,F401  (side-effect import: registers Ascend NPU backend)
import model_new_ascendc as _mna  # noqa: E402  (import after sys.path.insert)
from forward_spec import forward as FWD  # noqa: E402

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
    from forward_spec import BACKWARD_SPEC as SPEC  # noqa: E402
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
# explicitly labelled `competitor_kind="torch_same_dtype_cpu"`. It runs the same forward
# and autograd computation in the
# kernel dtype `dt` (fp32-internal compute is fine — that mirrors how a vendor same-dtype
# impl accumulates), then DOWNCASTS grads to `dt` so the competitor carries the SAME
# dtype-rounding error budget the kernel does, measured against the SAME fp64 golden.
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
            output_shape = tuple(outs0[0].shape)

            def draw(seed, profile, *, _shapes=shapes, _dtype=dt, _output_shape=output_shape):
                gg = torch.Generator().manual_seed(seed)
                fwd_args = []
                for i, n in enumerate(input_order):
                    t = torch.randn(_shapes[n], generator=gg)
                    t = _apply_profile(t, profile, seed + i)
                    fwd_args.append(t.to(_dtype))
                gy = torch.randn(
                    _output_shape,
                    generator=torch.Generator().manual_seed(seed + 999),
                ).to(_dtype) if explicit else None
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
                                 "baseline_mere": cm["mere"], "baseline_mare": cm["mare"], "baseline_rmse": cm["rmse"]})
    with open(out, "w") as output_file:
        json.dump(recs, output_file)
    nrep = sum(1 for r in recs if r["profile"] == REP_PROFILE)
    print(f"COLLECTED {len(recs)} records ({nrep} representative, {len(recs)-nrep} edge) -> {out} "
          f"[+T2 competitor={COMPETITOR_KIND}: baseline_* fields populated for double_baseline_ratio]")


if __name__ == "__main__":
    main()
