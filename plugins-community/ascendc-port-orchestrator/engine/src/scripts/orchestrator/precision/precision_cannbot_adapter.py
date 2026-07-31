# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
a5_ops precision-standard adapter — DEFAULT = 生态 (ecosystem) cann-bench compare.py.

OWNER-DIRECTED CORRECTION (2026-06-30): the real cannbot op-gen grader is the cann-bench
ecosystem grader `compare.py` (生态 standard), NOT the cannbot DESCRIPTION-library formulas
(商用 dual-baseline ratio + an invented ①③ absolute floor). This adapter previously drifted to
the description-library + ①③; it now grades BY DEFAULT through the VERBATIM-vendored cann-bench
grader at `cannbench_grader/` (byte-identical to gitcode.com/cann/cann-bench@007855b — see
cannbench_grader/PROVENANCE.md). This is the SINGLE precision-verdict entry-point for a5_ops:
all modes (benchmark / port_a3 / backward in-process) call THIS grade()/grade_batch().

Routing (float, the default op_class):
  DEFAULT  生态  → vendored compare.py (compare_tensors): golden=fp64 CPU, output cast back to its
                  native dtype, native_output = CPU-same-precision reference (when provisioned).
                  compare.py is the COMPLETE grader (Stage-1 MERE/MARE + Stage-2 small-value /
                  cancellation carve-outs + inf/nan) — no extra composition is layered on it.
  OPTIONAL 商用  → route="commercial" WITH a third_party 标杆 → legacy dual-baseline ratio
                  (cannbot_standard/mare_mere_rmse_ratio). NON-DEFAULT ("商用 is later" per official
                  docs). The invented ①③ degenerate-competitor→absolute fallback is RETIRED.

Other op_classes (integer / non_compute / quantization / random) keep their existing
cannbot_standard scripts — they are NOT the drifted float metric (Phase-1 scope boundary;
candidate for Phase-2 consolidation onto compare.py's integer/bit-exact paths).

NOTE: `cannbot_standard/` is the cannbot DESCRIPTION-library, NOT the grader; the grader is
`cannbench_grader/compare.py`.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any, Optional

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cannbot_standard", "scripts")
_PREC_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(mod_name: str):
    """Load a verbatim cannbot DESCRIPTION-library script by filename (商用/quant/integer routes)."""
    path = os.path.join(_SCRIPTS_DIR, mod_name + ".py")
    spec = importlib.util.spec_from_file_location("cannbot_" + mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Verbatim cannbot DESCRIPTION-library modules (loaded lazily; used only by non-default routes).
_M: dict = {}


def _m(name: str):
    if name not in _M:
        _M[name] = _load(name)
    return _M[name]


# ---- the REAL 生态 grader: VERBATIM-vendored cann-bench compare.py (cannbench_grader/) ----
_GRADER: dict = {}


def _grader():
    """Import the byte-identical vendored cann-bench grader package (self-contained closure)."""
    if not _GRADER:
        import sys
        if _PREC_DIR not in sys.path:
            sys.path.insert(0, _PREC_DIR)
        import cannbench_grader as G  # noqa
        _GRADER["G"] = G
    return _GRADER["G"]


_TORCH_DTYPE = None


def _torch_dtype_map():
    global _TORCH_DTYPE
    if _TORCH_DTYPE is None:
        import torch
        _TORCH_DTYPE = {
            "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
            "float64": torch.float64, "fp64": torch.float64, "double": torch.float64,
            "int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
        }
    return _TORCH_DTYPE


def _grade_ecosystem(npu_output, golden_output, *, dtype=None, native_output=None) -> dict:
    """生态 DEFAULT: grade one (output) tensor via the VERBATIM-vendored cann-bench compare.py.

    npu_output / golden_output / native_output may be np.ndarray or torch.Tensor. golden is the
    fp64 CPU truth; npu_output is cast back to its NATIVE dtype (declared `dtype` wins; this recovers
    bf16/fp16 from the lossless fp32 transport used because numpy has no bf16). native_output is the
    CPU-SAME-PRECISION reference (when provisioned); None ⇒ compare.py's stricter carve-out baseline.

    Returns a dict = CompareResult.to_dict() + {"is_pass": passed} so _primary_metric reads `mare`
    and grade_batch reads is_pass — the SAME keys the prior 生态 path exposed under criteria["float"].
    """
    import numpy as np
    import torch
    G = _grader()
    tdt = _torch_dtype_map()

    def _to_t(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        return torch.as_tensor(np.asarray(x))

    dt = (str(dtype).lower() if dtype is not None else "")
    target = tdt.get(dt)

    out_t = _to_t(npu_output)
    if target is not None and out_t is not None and out_t.dtype != target:
        out_t = out_t.to(target)            # cast back to native dtype → correct per-dtype threshold
    gold_t = _to_t(golden_output)
    if gold_t is not None and gold_t.is_floating_point():
        gold_t = gold_t.double()            # fp64 CPU golden
    nat_t = _to_t(native_output)
    if nat_t is not None and target is not None and nat_t.dtype != target:
        nat_t = nat_t.to(target)

    dtype_label = dt or str(out_t.dtype).replace("torch.", "")
    res = G.compare_tensors(out_t, gold_t, dtype=dtype_label, native_output=nat_t)
    d = res.to_dict()
    d["is_pass"] = bool(res.passed)
    d["grader"] = "cannbench_compare.py"
    return d


def grade(
    npu_output,
    golden_output,
    *,
    third_party_output=None,
    op_class: str = "float",
    precision_level: str = "L1",
    input_dtype=None,
    output_dtype=None,
    dtype=None,
    native_output=None,
    native_kind=None,
    route: str = "ecosystem",
    is_before_910a: bool = False,
) -> dict[str, Any]:
    """
    Single precision verdict. DEFAULT float route = 生态 vendored compare.py.

    Args:
      npu_output:     our kernel output (np.ndarray or torch.Tensor)
      golden_output:  CPU-fp64 golden (the truth) — REQUIRED
      third_party_output: independent numerical baseline (np). Used only by the optional ratio route
                          (route="commercial"); ignored by the 生态 default.
      op_class:       'float' | 'integer' | 'quantization' | 'non_compute' | 'random'
      dtype/output_dtype: declared output dtype (picks the compare.py per-dtype threshold).
      native_output:  CPU-SAME-PRECISION reference for compare.py's small-value/cancel carve-outs
                      (REAL re-run at native dtype on CPU; None ⇒ stricter carve-out baseline). 生态 only.
      native_kind:    PROVENANCE GUARD (codex #3) — native_output is HONORED only when this is the
                      sanctioned tag "cpu_same_precision". A native_output with a missing/other tag is
                      DROPPED to None (recorded) so a wrong/unverified baseline can't silently relax
                      the carve-out. (The case builders set native_kind="cpu_same_precision" only for a
                      genuine CPU-same-dtype autograd reference.)
      route:          "ecosystem" (default, 生态 compare.py) | "commercial" (opt-in 商用 ratio; needs 标杆).
      precision_level: L0/L1/L2 (商用 ratio + quant only).
      is_before_910a: inf/nan rule flag (A5/950 = False → strict).

    Returns dict: {is_pass, validator, scenario, criteria, provenance}. is_pass=None ⇒ inconclusive.
    """
    use_commercial = (op_class == "float" and route == "commercial"
                      and third_party_output is not None)

    # ---- native_output PROVENANCE GUARD (codex #3) ----
    native_used = native_output
    native_dropped_reason = None
    if native_output is not None and native_kind not in _grader().NATIVE_CARVEOUT_WHITELIST:
        native_dropped_reason = (
            f"native_output supplied with native_kind={native_kind!r} not in "
            f"NATIVE_CARVEOUT_WHITELIST → DROPPED (unverified provenance; carve-out stays strict, "
            "never relaxed by a wrong baseline)")
        native_used = None

    prov = {
        "standard": "cann-bench 生态 compare.py (vendored, byte-identical @007855b)",
        "op_class": op_class,
        "scenario": None,
        "validator": None,
        # references list ONLY what the SELECTED route actually consumes (audit honesty):
        #  - 生态 default: fp64 golden + (native carve-out baseline if used). third_party IGNORED.
        #  - 商用 route: fp64 golden + third_party 标杆. native is NOT consumed by the ratio path,
        #    so it is NOT listed even if a tensor was supplied (low-sev codex: no over-claim).
        "references": ["fp64_cpu_golden"]
                      + (["cpu_same_dtype_native"]
                         if (native_used is not None and not use_commercial) else [])
                      + (["third_party_标杆"] if use_commercial else []),
    }
    if native_dropped_reason:
        prov["native_dropped"] = native_dropped_reason
    criteria: dict[str, Any] = {}

    # ---- random class: distribution-conformance, not a per-element verdict ----
    if op_class == "random":
        _m("random_distribution_check")
        prov["validator"] = "random_distribution_check"
        return {"is_pass": None, "validator": prov["validator"], "scenario": "random",
                "criteria": {"note": "random-generation class: use random_distribution_check separately"},
                "provenance": prov}

    # ============================================================================
    # FLOAT DEFAULT (生态) — VERBATIM-vendored cann-bench compare.py is the COMPLETE grader.
    # Takes effect unless the caller explicitly opts into 商用 AND provides a 标杆.
    # ============================================================================
    if op_class == "float" and not use_commercial:
        res = _grade_ecosystem(npu_output, golden_output,
                               dtype=(dtype or output_dtype), native_output=native_used)
        res["native_kind"] = "cpu_same_precision" if native_used is not None else None
        if native_dropped_reason:
            res["native_dropped"] = native_dropped_reason
        criteria["float"] = res
        prov["validator"] = "cannbench_compare.py (生态单标杆Threshold — vendored cann-bench@007855b)"
        prov["scenario"] = "生态"
        main_pass = res.get("is_pass")
        if main_pass is not None:
            main_pass = bool(main_pass)
        return {"is_pass": main_pass, "validator": prov["validator"],
                "scenario": "生态", "criteria": criteria, "provenance": prov}

    # ============================================================================
    # Non-default routes / non-float classes — cannbot_standard DESCRIPTION-library scripts
    # (+ special_cases composition). These are NOT the drifted float metric.
    # ============================================================================
    if op_class == "non_compute":
        bm = _m("bitwise_match")
        ok = bool(bm.bitwise_match(npu_output, golden_output)) if hasattr(bm, "bitwise_match") else None
        prov["validator"] = "bitwise_match"
        prov["standard"] = "cannbot ops-precision-standard (description-library)"
        criteria["bitwise"] = ok
        main_pass = ok

    elif op_class == "integer":
        res = _m("integer_compute_check").check_integer_compute(npu_output, golden_output)
        prov["validator"] = "integer_compute_check"
        prov["standard"] = "cannbot ops-precision-standard (description-library)"
        criteria["integer"] = res
        main_pass = res.get("is_pass")

    elif op_class == "quantization":
        res = _m("quantization_check").check_quantization(
            npu_output, golden_output, third_party_output,
            input_dtype=input_dtype, output_dtype=output_dtype, precision_level=precision_level)
        prov["validator"] = "quantization_check"
        prov["standard"] = "cannbot ops-precision-standard (description-library)"
        prov["scenario"] = "商用" if third_party_output is not None else "生态"
        criteria["quantization"] = res
        main_pass = res.get("is_pass")

    else:  # float + route=="commercial" + third_party present → OPTIONAL 商用 双标杆 Ratio
        res = _m("mare_mere_rmse_ratio").check_precision_ratio(
            npu_output, golden_output, third_party_output, precision_level=precision_level)
        prov["validator"] = "mare_mere_rmse_ratio (商用双标杆Ratio — OPTIONAL non-default route)"
        prov["standard"] = "cannbot ops-precision-standard (description-library, OPTIONAL 商用 route)"
        prov["scenario"] = "商用"
        criteria["float"] = res
        main_pass = res.get("is_pass")

    # ---- special_cases.md compose (commercial-float / quant / integer / non_compute only) ----
    svc = _m("small_value_check")
    try:
        use_sv = bool(svc.should_use_small_value_standard(npu_output, golden_output))
    except Exception:
        use_sv = False
    if use_sv:
        if third_party_output is not None:
            sv = svc.check_small_value_precision(npu_output, golden_output, third_party_output)
            criteria["small_value"] = sv
            if sv.get("is_pass") is False:
                main_pass = False
        else:
            criteria["small_value"] = {"is_pass": None,
                                       "note": "小值域 present but no 标杆 — small-value ratio needs third_party"}

    try:
        inf = _m("inf_nan_check").check_inf_nan_consistency(
            npu_output, golden_output, benchmark_output=third_party_output, is_before_910a=is_before_910a)
        criteria["inf_nan"] = inf
        if inf.get("has_special_values"):
            if inf.get("is_pass") is False:
                main_pass = False
            elif inf.get("is_pass") is None and main_pass is True:
                main_pass = None
    except Exception as e:
        criteria["inf_nan"] = {"is_pass": None, "error": str(e)}

    prov["scenario"] = prov["scenario"] or ("商用" if third_party_output is not None else "生态")
    if main_pass is not None:
        main_pass = bool(main_pass)
    return {"is_pass": main_pass, "validator": prov["validator"],
            "scenario": prov["scenario"], "criteria": criteria, "provenance": prov}


def _primary_metric(case_result: dict) -> Optional[float]:
    """Primary error metric for the statistical aggregate: 商用 ⇒ mare_ratio; 生态 ⇒ mare (absolute)."""
    fl = case_result.get("criteria", {}).get("float", {})
    if "mare_ratio" in fl:
        return float(fl["mare_ratio"])
    if fl.get("mare") is not None:
        return float(fl["mare"])
    return None


def grade_batch(cases: list, *, op_class: str = "float", precision_level: str = "L1",
                route: str = "ecosystem", is_before_910a: bool = False,
                confidence_level: float = 0.95) -> dict[str, Any]:
    """
    Statistical-aggregate verdict over many cases. DEFAULT route = 生态 compare.py.

    cases: list of dicts, each {npu, golden, third_party=None, native=None, native_kind=None,
                                dtype=None, input_dtype=None, output_dtype=None, is_edge=False,
                                competitor_kind=None}.
      - native: CPU-SAME-PRECISION reference for compare.py carve-outs (生态). Honored only with
        native_kind="cpu_same_precision" (provenance guard); else None (compare.py stricter, never looser).
      - third_party: 标杆 for the OPTIONAL 商用 ratio (route="commercial").
      - competitor_kind: honest 标杆 provenance tag (binding infra) — threaded into the per-case result.

    生态 verdict: per-case compare.py is authoritative; the op passes iff ALL representative cases pass
    (the bootstrap CI is reported as supplementary statistics, not as an overriding gate — compare.py
    already applies the principled small-value/cancellation carve-outs per case).
    商用 verdict (route="commercial"): bootstrap-median ratio CI within the tier gate at scale, else
    small-sample per-case-all-pass. FAIL-CLOSED (codex #1): route="commercial" REQUIRES every
    representative case to carry a 标杆 (all 商用) — a mixed 商用/生态 batch is REFUSED so an absolute
    MARE can never enter the ratio bootstrap and be gated against the ratio threshold.
    """
    boot = _m("bootstrap_median")
    rep_results, edge_results, metrics = [], [], []
    scenario = None
    for c in cases:
        r = grade(c["npu"], c["golden"], third_party_output=c.get("third_party"),
                  op_class=op_class, precision_level=precision_level,
                  input_dtype=c.get("input_dtype"), output_dtype=c.get("output_dtype"),
                  dtype=c.get("dtype"), native_output=c.get("native"),
                  native_kind=c.get("native_kind"),
                  route=route, is_before_910a=is_before_910a)
        # AUDIT HONESTY (binding infra, KEPT): persist WHICH 标杆 graded this case.
        if "competitor_kind" in c:
            r["competitor_kind"] = c.get("competitor_kind")
        if c.get("is_edge"):
            edge_results.append(r)  # bug-find / robustness stream — not in statistical verdict
            continue
        rep_results.append(r)
        scenario = scenario or r["scenario"]
        m = _primary_metric(r)
        if m is not None:
            metrics.append(m)

    n_rep = len(rep_results)
    n_pass = sum(1 for r in rep_results if r["is_pass"])
    pass_rate = (n_pass / n_rep) if n_rep else 0.0

    # ---- FAIL-CLOSED mixed-scenario guard (codex #1) ----
    # route="commercial" gates the ratio bootstrap against the 商用 tier threshold; _primary_metric
    # returns absolute `mare` for any case that fell back to 生态 (no 标杆). Mixing absolute + ratio
    # in one bootstrap under a ratio gate is unsound → REFUSE rather than silently mis-gate.
    rep_scenarios = sorted({r["scenario"] for r in rep_results})
    if route == "commercial" and n_rep and "生态" in rep_scenarios:
        n_eco = sum(1 for r in rep_results if r["scenario"] == "生态")
        return {
            "is_pass": None, "scenario": "mixed", "verdict_basis": "REFUSED_mixed_scenario",
            "n_representative": n_rep, "n_edge": len(edge_results),
            "fail_closed_reason": (
                f"route='commercial' but {n_eco}/{n_rep} representative case(s) lack a 标杆 "
                "(fell back to 生态 absolute) → refusing to mix absolute MARE into the 商用 ratio "
                "bootstrap. Supply a 标杆 for every case or grade under route='ecosystem'."),
            "rep_scenarios": rep_scenarios,
            "competitor_kinds": [], "competitor_kind_counts": {},
            "pass_rate": pass_rate, "bootstrap_valid": False,
            "bootstrap_median": None, "ci_lower": None, "ci_upper": None, "gate": None,
            "per_case": rep_results, "edge": edge_results,
            "provenance": {"standard": "cann-bench 生态 compare.py (vendored)",
                           "aggregate": "REFUSED: mixed 商用/生态 under commercial route",
                           "scenario": "mixed", "precision_level": precision_level, "route": route},
        }

    # gate: 商用 ⇒ MARE-ratio tier gate; 生态 ⇒ absolute MARE gate = 10× dtype threshold.
    if scenario == "商用":
        gate = {"L0": 10.0, "L1": 5.0, "L2": 2.0}.get(precision_level, 10.0)
    else:
        dt = cases[0].get("dtype") if cases else None
        try:
            gate = float(_grader().get_threshold(dt or "float32")) * 10.0
        except Exception:
            gate = None

    stat = boot.bootstrap_median(metrics, confidence_level=confidence_level) if metrics else {}
    bootstrap_valid = bool(stat.get("is_valid", False)) and stat.get("ci_upper") is not None
    ci_upper = stat.get("ci_upper")

    if scenario == "商用" and bootstrap_valid and gate is not None:
        is_pass = bool(ci_upper <= gate)                 # 商用 statistical 达标 (ratio CI within gate)
        verdict_basis = "bootstrap_median_ci"
    elif n_rep:
        is_pass = (n_pass == n_rep)                      # per-case authoritative (生态) / small-N (商用)
        verdict_basis = ("per_case_all_pass (生态 compare.py per-case authoritative)"
                         if scenario != "商用"
                         else "per_case_all_pass (bootstrap below N≥200 scale)")
    else:
        is_pass = None
        verdict_basis = "no_representative_cases"

    # AUDIT (binding infra, KEPT): which 标杆(s) graded this batch, with per-kind representative counts.
    competitor_kind_counts: dict[str, int] = {}
    for r in rep_results:
        k = r.get("competitor_kind")
        if k is not None:
            competitor_kind_counts[k] = competitor_kind_counts.get(k, 0) + 1

    return {
        "is_pass": is_pass, "scenario": scenario, "verdict_basis": verdict_basis,
        "n_representative": n_rep, "n_edge": len(edge_results),
        "competitor_kinds": sorted(competitor_kind_counts),
        "competitor_kind_counts": competitor_kind_counts,
        "pass_rate": pass_rate, "bootstrap_valid": bootstrap_valid,
        "bootstrap_median": stat.get("median"), "ci_lower": stat.get("ci_lower"),
        "ci_upper": ci_upper, "gate": gate,
        "per_case": rep_results, "edge": edge_results,
        "provenance": {"standard": ("cann-bench 生态 compare.py (vendored)" if scenario != "商用"
                                    else "cannbot 商用 ratio (description-library, OPTIONAL route)"),
                       "aggregate": ("生态: per-case compare.py authoritative (all representative must "
                                     "pass); bootstrap CI reported as supplementary"
                                     if scenario != "商用" else
                                     "商用: bootstrap-median ratio CI within tier gate at scale; "
                                     "small-N fallback=per-case-all-pass"),
                       "scenario": scenario, "precision_level": precision_level, "route": route},
    }


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    g = rng.standard_normal(4096).astype(np.float32)
    npu = (g + rng.standard_normal(4096).astype(np.float32) * 1e-4).astype(np.float32)
    tp = (g + rng.standard_normal(4096).astype(np.float32) * 1e-4).astype(np.float32)
    print("生态 default (no route):", grade(npu, g, op_class="float")["is_pass"],
          grade(npu, g, op_class="float")["scenario"])
    print("生态 default (tp present, ignored):", grade(npu, g, third_party_output=tp, op_class="float")["scenario"])
    print("商用 opt-in (route=commercial):",
          grade(npu, g, third_party_output=tp, op_class="float", route="commercial")["is_pass"],
          grade(npu, g, third_party_output=tp, op_class="float", route="commercial")["scenario"])
