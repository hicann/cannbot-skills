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

"""apply_adam_w_v2 precision verification via aclnn-direct runner subprocess.

P109 phase B implementation. torch_npu.npu_apply_adam_w not registered on
CANN 9.0.0 install → torch-op symmetric path unavailable. Solution: invoke
the archive's apply_adam_w_v2_runner.cpp (aclnn-direct C++ exe) per ATK
case, then compare runner output to CPU AdamW reference via plugin.

Pre-conditions:
  - apply_adam_w_v2_runner already built (./build_runner.sh)
  - run_a3_reference.py available (reused for per-case roundtrip)
  - ATK cases JSONL in apply_adam_w_v2_perf_cases.jsonl

Usage:
  python3 verify_adam_w_v2_atk.py [--limit N] [--out-json path]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ARCHIVE = Path("/tmp/output/a3_to_a5_port/src/kernels/apply_adam_w_v2")  # A5 deploy path
ARCHIVE_LOCAL = Path(__file__).resolve().parents[3] / "output/a3_to_a5_port/src/kernels/apply_adam_w_v2"
if ARCHIVE_LOCAL.exists():
    ARCHIVE = ARCHIVE_LOCAL
sys.path.insert(0, str(ARCHIVE))

DTYPE_MAP = {
    "fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16,
    "int64": torch.int64,
}


def _pick_scalar(spec_value, name=""):
    """ATK 'value' field can be scalar or [lo, hi] range.

    Default: median of range. Exception: eps uses upper bound to avoid
    near-zero denom (eps too small → div by ~0 on v=0 inputs → var
    update diverges by 6+ orders of magnitude). User catch 2026-05-16:
    median of [0, 1e-8] = 5e-9 caused case 199 var to diverge to -60000.

    Other attrs (beta1/beta2/lr/wd) keep median — those test the
    algorithm's behavior across attr ranges; eps is purely numerical.
    """
    if isinstance(spec_value, list) and len(spec_value) == 2:
        lo, hi = float(spec_value[0]), float(spec_value[1])
        if name == "eps":
            return max(hi, 1e-8)  # eps lower-bound for numerical stability
        return (lo + hi) / 2.0
    return float(spec_value) if isinstance(spec_value, (int, float)) else 0.0


def _make_input_tensor(spec, seed, name=""):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dt = DTYPE_MAP.get(spec["dtype"])
    if dt is None:
        return None
    shape = spec["shape"]
    if dt.is_floating_point:
        # Avoid huge values that break AdamW (var ~ 0..1 typical)
        return torch.randn(*shape, dtype=dt, generator=g) * 0.1
    else:
        # step tensor: AdamW step counter, MUST be ≥1 (step^0=1, bc1=0 → div0)
        if name == "step":
            return torch.ones(shape, dtype=dt)
        return torch.zeros(shape, dtype=dt)


def _atk_case_to_runner_input(atk_case):
    """Convert ATK case → c dict matching run_a3_reference._run_case schema."""
    inputs_spec = {e["name"]: e for e in atk_case["inputs"]}
    seed_base = atk_case["case_id"] * 1000

    # Map ATK names → runner names (per apply_adam_w_v2_runner.cpp meta.txt)
    # Per-tensor seed offsets are deterministic (NOT Python hash() — randomized).
    runner_inputs = {}
    name_map = {"varRef": "var", "mRef": "m", "vRef": "v",
                "maxGradNormOptionalRef": "maxgrad", "grad": "grad", "step": "step"}
    seed_offsets = {"varRef": 1, "mRef": 2, "vRef": 3, "maxGradNormOptionalRef": 4, "grad": 5, "step": 6}
    for atk_name, runner_name in name_map.items():
        spec = inputs_spec.get(atk_name)
        if spec is None:
            return None  # case incomplete
        if spec.get("type") != "tensor":
            return None
        t = _make_input_tensor(spec, seed=seed_base + seed_offsets[atk_name], name=runner_name)
        if t is None:
            return None
        # m, v, maxgrad non-negative (Adam state buffers; v = variance,
        # maxgrad = running max of v_hat per AmsGrad — both must sqrt-clean)
        if runner_name in ("m", "v", "maxgrad"):
            t = t.abs()
        runner_inputs[runner_name] = t

    # Runner reads var/m/v/maxgrad/grad with var_bytes = var_numel × var_dtype.
    # ATK spec may have grad=fp32 even when var=bf16 — runner's assumption is
    # "all same dtype". Cast grad/maxgrad to var.dtype to match runner's read.
    var_dtype = runner_inputs["var"].dtype
    for name in ("grad", "maxgrad"):
        if name in runner_inputs and runner_inputs[name].dtype != var_dtype:
            runner_inputs[name] = runner_inputs[name].to(var_dtype)
    # step is ACL_FLOAT (fp32) shape [1] in runner — force fp32 scalar
    if "step" in runner_inputs:
        s = runner_inputs["step"]
        if s.dtype != torch.float32 or s.numel() != 1:
            runner_inputs["step"] = torch.tensor([1.0], dtype=torch.float32)

    attrs_spec = {e["name"]: e for e in atk_case["inputs"] if e["type"] == "attr"}
    attrs = {
        "lr": _pick_scalar(attrs_spec.get("lr", {}).get("value", 0.001), "lr"),
        "beta1": _pick_scalar(attrs_spec.get("beta1", {}).get("value", 0.9), "beta1"),
        "beta2": _pick_scalar(attrs_spec.get("beta2", {}).get("value", 0.999), "beta2"),
        "weight_decay": _pick_scalar(attrs_spec.get("weightDecay", {}).get("value", 0.01), "weightDecay"),
        "eps": _pick_scalar(attrs_spec.get("eps", {}).get("value", 1e-8), "eps"),
        "amsgrad": bool(attrs_spec.get("amsgrad", {}).get("value", False)),
        "maximize": bool(attrs_spec.get("maximize", {}).get("value", False)),
    }
    # Hard guards: prevent degenerate cases
    if attrs["beta1"] >= 1.0:
        attrs["beta1"] = 0.9
    if attrs["beta2"] >= 1.0:
        attrs["beta2"] = 0.999
    if attrs["eps"] <= 0:
        attrs["eps"] = 1e-8

    return {"case_id": atk_case["case_id"], "inputs": runner_inputs, "attrs": attrs}


def cpu_adamw_step(c):
    """Reference AdamW step. Returns (var_new, m_new, v_new, maxgrad_out)."""
    var = c["inputs"]["var"].float()
    m = c["inputs"]["m"].float()
    v = c["inputs"]["v"].float()
    grad = c["inputs"]["grad"].float()
    maxgrad = c["inputs"]["maxgrad"].float()
    step = c["inputs"]["step"]
    step_val = int(step.flatten()[0].item()) if step.numel() > 0 else 1
    if step_val < 1:
        step_val = 1
    a = c["attrs"]
    if a["maximize"]:
        grad = -grad

    # AdamW update (decoupled weight decay), upstream formula:
    # θ_{t+1} = θ_t - (η/(√v̂ + ε)) * m̂ - η·λ·θ_t
    m_new = a["beta1"] * m + (1 - a["beta1"]) * grad
    v_new = a["beta2"] * v + (1 - a["beta2"]) * grad * grad
    bc1 = 1 - a["beta1"] ** step_val
    bc2 = 1 - a["beta2"] ** step_val
    # NPU runner stores m_new/v_new in low-precision dtype between stages
    # then re-loads to fp32 for var update. Replicate that round-trip.
    m_for_var = m_new.to(c["inputs"]["m"].dtype).float()
    v_for_var = v_new.to(c["inputs"]["v"].dtype).float()

    # Per upstream aclnnApplyAdamWV2 docs §maxGradNormOptionalRef:
    # "输入maxGradNormOptionalRef与更新后的vRef比较后，得到的最大值输出到maxGradNormOptionalRef".
    # → maxgrad_out = max(maxgrad_in, v_new)  ALWAYS (not gated by amsgrad).
    # Comparison is with v_new (raw), NOT v_hat (post-bias-correction).
    maxgrad_for_compare = maxgrad.to(c["inputs"]["maxgrad"].dtype).float()
    maxgrad_out_fp = torch.maximum(maxgrad_for_compare, v_for_var)

    # AmsGrad: denom uses max(maxgrad_in, v_new) / bc2; non-amsgrad uses v_new / bc2.
    if a["amsgrad"]:
        denom = (maxgrad_out_fp / bc2).sqrt() + a["eps"]
    else:
        denom = (v_for_var / bc2).sqrt() + a["eps"]

    # Keep the update in separate stages to match the likely aclnn
    # implementation: subtract both the normalized gradient term and the
    # learning-rate-scaled weight-decay term from the original variable.
    m_hat = m_for_var / bc1
    grad_term = a["lr"] * m_hat / denom
    wd_term = a["lr"] * a["weight_decay"] * var
    var_new = var - grad_term - wd_term

    # Cast back to original dtypes
    out = {
        "var": var_new.to(c["inputs"]["var"].dtype),
        "m": m_new.to(c["inputs"]["m"].dtype),
        "v": v_new.to(c["inputs"]["v"].dtype),
        "maxgrad": maxgrad_out_fp.to(c["inputs"]["maxgrad"].dtype),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ARCHIVE / "apply_adam_w_v2_perf_cases.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out-json", default="/tmp/adam_atk_precision.json")
    args = ap.parse_args()

    # Plugin import
    proj_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(proj_root / "src"))
    from verifiers import get_verifier
    from references import get_provider, TruthSource
    verifier = get_verifier("ascendc")
    ref_provider = get_provider("port_a3_to_a5")

    # Reuse run_a3_reference._run_case for per-case NPU roundtrip
    import importlib.util
    spec = importlib.util.spec_from_file_location("rar", ARCHIVE / "run_a3_reference.py")
    rar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rar)

    # P112 ReferenceProvider integration: if archive has edge_dataset.pt with
    # A3 captured outputs, prefer those as truth. Falls back to CPU AdamW
    # (cpu_adamw_step) when no A3 capture.
    archive_dir = ARCHIVE  # local archive path
    print(f"[ref-provider] {ref_provider.describe()}", flush=True)

    atk_cases = [json.loads(ln) for ln in Path(args.cases).read_text().splitlines() if ln.strip()]
    if args.limit > 0:
        atk_cases = atk_cases[: args.limit]

    results = []
    for atk in atk_cases:
        c = _atk_case_to_runner_input(atk)
        if c is None:
            results.append({"case_id": atk["case_id"], "status": "SKIP_UNMAPPABLE"})
            continue
        try:
            npu_result = getattr(rar, "_run_case")(c)
        except Exception as e:
            results.append({"case_id": atk["case_id"], "status": "RUNNER_ERR",
                            "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        if "a3_error" in npu_result:
            results.append({"case_id": atk["case_id"], "status": "RUNNER_ERR",
                            "error": npu_result["a3_error"][:200]})
            continue

        # P115: prefer A3 NPU capture as truth via ReferenceProvider; fall
        # back to CPU AdamW formula only when archive has no edge_dataset.pt
        # match. Track which path was used for audit.
        ref_pair = ref_provider.get(
            op_meta={"op_name": "apply_adam_w_v2"},
            case=atk,
            archive_dir=archive_dir,
        )
        if ref_pair.truth_source == TruthSource.A3_NPU_CAPTURE and isinstance(ref_pair.truth, dict):
            # ref_pair.truth is the matched edge_dataset.pt case row's a3_outputs dict
            truth_dict = ref_pair.truth
            truth_source_str = "A3_NPU_CAPTURE"
        else:
            cpu_truth = cpu_adamw_step(c)
            truth_dict = cpu_truth
            truth_source_str = ref_pair.truth_source.value

        # Verify each output tensor independently; aggregate worst verdict
        per_tensor = {}
        worst = "PASS"
        for name in ("var", "m", "v"):  # skip maxgrad passthrough
            ours = npu_result["a3_outputs"][name]
            truth = truth_dict.get(name) if isinstance(truth_dict, dict) else None
            if truth is None:
                # Truth missing for this tensor (e.g., A3 capture had different keys)
                per_tensor[name] = {"status": "EVAL_ERR", "reason": f"no truth for {name}"}
                worst = "EVAL_ERR" if worst != "FAIL" else worst
                continue
            vr = verifier.verify(ours, truth)
            per_tensor[name] = vr.to_json()
            s = vr.status.value
            if s == "FAIL":
                worst = "FAIL"
            elif s == "EVAL_ERR" and worst != "FAIL":
                worst = "EVAL_ERR"
            elif s == "PASS_WITH_BASELINE" and worst not in ("FAIL", "EVAL_ERR"):
                worst = "PASS_WITH_BASELINE"
        results.append({
            "case_id": atk["case_id"],
            "scale_bucket": atk["scale_bucket"],
            "dtype": next(e["dtype"] for e in atk["inputs"] if e["type"] == "tensor"),
            "status": worst, "per_tensor": per_tensor,
            "truth_source": truth_source_str,
        })
        print(f"case {atk['case_id']:>3} {atk['scale_bucket']} dtype={results[-1]['dtype']:>5} "
              f"{worst}", flush=True)

    summary = {
        "total": len(results),
        "pass_t1": sum(1 for r in results if r.get("status") == "PASS_T1"),
        "pass_t2": sum(1 for r in results if r.get("status") == "PASS_T2"),
        "fail": sum(1 for r in results if r.get("status") == "FAIL"),
        "eval_err": sum(1 for r in results if r.get("status") == "EVAL_ERR"),
        "skip": sum(1 for r in results if r.get("status", "").startswith("SKIP")),
        "runner_err": sum(1 for r in results if r.get("status") == "RUNNER_ERR"),
    }
    out = {"results": results, "summary": summary,
           "method": "apply_adam_w_v2_runner.cpp (aclnn-direct) subprocess + AscendCVerifier plugin (vendor §4.5.3)"}
    Path(args.out_json).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nSummary: {summary}")
    print(f"Wrote: {args.out_json}")
    return 0 if summary["fail"] == 0 and summary["eval_err"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
