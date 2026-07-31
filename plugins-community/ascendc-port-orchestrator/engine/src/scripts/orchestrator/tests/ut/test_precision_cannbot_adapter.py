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
Regression tests for the a5_ops precision adapter (owner-directed correction 2026-06-30).

NEW CONTRACT: the DEFAULT float grader is the VERBATIM-vendored cann-bench 生态 `compare.py`
(`cannbench_grader/`). The 商用 dual-baseline ratio is an OPTIONAL non-default route
(`route="commercial"`); the invented ①③ degenerate-competitor→absolute fallback is RETIRED.
These tests lock the 生态-default routing + the native_output channel + the vendored wiring.
"""
import logging
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch  # noqa: F401  collection-time import: keeps torch in the per-test sys.modules baseline
              # (DEBT-47 conftest tears down NEW modules after each test; the 生态 grader uses torch)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "precision"))
import precision_cannbot_adapter as A  # noqa: E402


def _g(n=4096, seed=0):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


# ── vendored grader provenance (byte-identical cann-bench compare.py) ──────────────────────────
def test_vendored_grader_present_and_byte_identical():
    base = os.path.join(os.path.dirname(A.__file__), "cannbench_grader")
    assert os.path.exists(os.path.join(base, "compare.py")), "missing vendored compare.py"
    assert os.path.exists(os.path.join(base, "thresholds.py")), "missing vendored thresholds.py"
    assert os.path.exists(os.path.join(base, "PROVENANCE.md"))
    # thresholds.py: byte-identical to gitcode.com/cann/cann-bench@007855b (0 edits).
    # compare.py: GRADING LOGIC verbatim @007855b PLUS the sanctioned, verdict-neutral changes documented
    # in PROVENANCE.md (diagnostic exposure and device-neutral CPU normalization). The pin still guards
    # against any FURTHER unauthorized edit; the grading-verdict-unchanged property is separately proven
    # by test_precision_eval_port_a3_two_tier.test_compare_py_additive_field_verdict_consistency.

    def md5(path):
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    assert md5(os.path.join(base, "compare.py")) == "3f683648c42a77c9862ad5fa1b0701f8"
    assert md5(os.path.join(base, "thresholds.py")) == "be42d99ac574aef6a32f84bdbb4a4440"


def test_verbatim_description_library_scripts_present():
    """cannbot_standard scripts still back the OPTIONAL 商用/quant/integer routes."""
    base = os.path.join(os.path.dirname(A.__file__), "cannbot_standard", "scripts")
    for f in ("mare_mere_rmse_ratio", "mare_mere_threshold", "small_value_check", "inf_nan_check"):
        assert os.path.exists(os.path.join(base, f + ".py")), f"missing verbatim cannbot script {f}"


# ── 生态 is the DEFAULT (even when a 标杆 is present) ──────────────────────────────────────────
def test_float_default_is_ecosystem_no_baseline():
    g = _g()
    out = A.grade(g.copy(), g, op_class="float")
    assert out["scenario"] == "生态"
    assert "compare.py" in out["validator"] or "生态" in out["validator"]
    assert out["is_pass"] is True            # exact match passes the vendored grader
    assert out["criteria"]["float"].get("grader") == "cannbench_compare.py"


def test_float_default_ignores_third_party_stays_ecosystem():
    """The demotion: a third_party 标杆 present does NOT auto-route to 商用 anymore — 生态 is default."""
    g = _g()
    npu = (g + np.random.default_rng(1).standard_normal(g.shape).astype(np.float32) * 1e-4).astype(np.float32)
    tp = (g + np.random.default_rng(2).standard_normal(g.shape).astype(np.float32) * 1e-4).astype(np.float32)
    out = A.grade(npu, g, third_party_output=tp, op_class="float")  # no route → ecosystem
    assert out["scenario"] == "生态"


def test_commercial_is_opt_in_only():
    """商用 双标杆Ratio only when explicitly route='commercial' AND a 标杆 is supplied."""
    g = _g()
    npu = (g + np.random.default_rng(1).standard_normal(g.shape).astype(np.float32) * 1e-4).astype(np.float32)
    tp = (g + np.random.default_rng(2).standard_normal(g.shape).astype(np.float32) * 1e-4).astype(np.float32)
    out = A.grade(npu, g, third_party_output=tp, op_class="float", route="commercial")
    assert out["scenario"] == "商用"
    assert "商用" in out["validator"] or "ratio" in out["validator"].lower()
    # route='commercial' but NO 标杆 → cannot ratio → falls back to 生态 default
    out2 = A.grade(npu, g, op_class="float", route="commercial")
    assert out2["scenario"] == "生态"


def test_key_property_fails_absolute_but_passes_ratio_optin():
    """The cannbot point still holds via the OPT-IN ratio: an op failing the 生态 absolute threshold
    can pass the 商用 ratio when its error is comparable to a mature 标杆.
    """
    g = _g()
    npu = (g + np.random.default_rng(1).standard_normal(g.shape).astype(np.float32) * 1e-3).astype(np.float32)
    tp = (g + np.random.default_rng(2).standard_normal(g.shape).astype(np.float32) * 1e-3).astype(np.float32)
    eco = A.grade(npu, g, op_class="float")                                   # default 生态
    com = A.grade(npu, g, third_party_output=tp, op_class="float", route="commercial")
    assert eco["is_pass"] is False, "noisy fp32 should fail the tight 生态 absolute threshold"
    assert com["is_pass"] is True, "comparable-to-标杆 should pass the opt-in 商用 ratio"


# ── native_output channel (CPU-same-precision) — REAL compare.py carve-out behavior ────────────
def test_native_output_channel_relaxes_small_value():
    """compare.py small-value carve-out: native_output=None ⇒ CPU baseline exact ⇒ NPU must be exact
    (stricter). A REAL CPU-same-precision native_output with the SAME small-value error ⇒ ratio ≤2 ⇒ PASS.
    This is the Phase-2 reference-provisioning hook; we never fabricate native_output.
    """
    g = np.full(64, 1e-8, dtype=np.float64)              # all in the fp32 small-value region (<2^-14)
    npu = (g + 1e-6).astype(np.float32)                  # error > small_value_error in every element
    assert A.grade(npu, g, op_class="float", dtype="float32")["is_pass"] is False  # native=None → strict
    native = npu.copy()                                  # CPU-same-precision ref with the same error
    # native_output is honored ONLY with the sanctioned provenance tag (codex #3 guard).
    r = A.grade(npu, g, op_class="float", dtype="float32",
                native_output=native, native_kind="cpu_same_precision")
    assert r["is_pass"] is True
    assert r["criteria"]["float"].get("native_kind") == "cpu_same_precision"


def test_native_output_dropped_without_provenance_tag():
    """codex #3 guard: a native_output with a missing/wrong native_kind is DROPPED (NOT used) so a
    wrong baseline can't silently relax the carve-out — verdict stays at the strict native=None result.
    """
    g = np.full(64, 1e-8, dtype=np.float64)
    npu = (g + 1e-6).astype(np.float32)
    # no native_kind → dropped → strict FAIL (same as native=None)
    r_notag = A.grade(npu, g, op_class="float", dtype="float32", native_output=npu.copy())
    assert r_notag["is_pass"] is False
    assert "native_dropped" in r_notag["provenance"]
    assert "cpu_same_dtype_native" not in r_notag["provenance"]["references"]
    # wrong tag → also dropped
    r_wrong = A.grade(npu, g, op_class="float", dtype="float32",
                      native_output=npu.copy(), native_kind="untrusted_backend")
    assert r_wrong["is_pass"] is False


# ── other op_classes unchanged (NOT the drifted float metric) ──────────────────────────────────
def test_integer_path_unchanged():
    a = np.arange(100, dtype=np.int32)
    out = A.grade(a.copy(), a, op_class="integer")
    assert out["validator"] == "integer_compute_check"
    assert out["is_pass"] is True


def test_references_provenance_recorded():
    g = _g()
    out = A.grade(g.copy(), g, op_class="float")
    assert "fp64_cpu_golden" in out["provenance"]["references"]
    g2 = _g()
    out2 = A.grade(g2.copy(), g2, native_output=g2.copy(), native_kind="cpu_same_precision",
                   op_class="float", dtype="float32")
    assert "cpu_same_dtype_native" in out2["provenance"]["references"]
    # codex #4: third_party is NOT a reference under the 生态 default (it is ignored), even if present
    out3 = A.grade(g2.copy(), g2, third_party_output=g2.copy(), op_class="float")
    assert "third_party_标杆" not in out3["provenance"]["references"]
    # ...but IS a reference under the opt-in 商用 route (where it is actually consumed)
    out4 = A.grade(g2.copy(), g2, third_party_output=g2.copy(), op_class="float", route="commercial")
    assert "third_party_标杆" in out4["provenance"]["references"]


# ── grade_batch ───────────────────────────────────────────────────────────────────────────────
def _case(seed, noise, n=256, edge=False, tp=True, dtype="bfloat16", near_zero=True):
    rng = np.random.default_rng(seed)
    g = (rng.standard_normal(n) if near_zero else np.abs(rng.standard_normal(n)) + 0.5).astype(np.float32)
    npu = (g + np.random.default_rng(seed + 7).standard_normal(n).astype(np.float32) * noise).astype(np.float32)
    third = (g + np.random.default_rng(seed + 13).standard_normal(n).astype(np.float32)
             * noise).astype(np.float32) if tp else None
    return {"npu": npu, "golden": g, "third_party": third, "dtype": dtype, "is_edge": edge}


def test_grade_batch_default_ecosystem_scenario():
    """Default grade_batch route = 生态: per-case compare.py authoritative; tp present is ignored."""
    cases = [_case(i, 1e-4, near_zero=False) for i in range(6)]
    out = A.grade_batch(cases, op_class="float")
    assert out["scenario"] == "生态"
    assert out["n_representative"] == 6
    assert out["verdict_basis"].startswith("per_case_all_pass")
    assert out["is_pass"] is True            # non-near-zero clean bf16 batch passes the vendored grader


def test_grade_batch_commercial_small_sample_fallback():
    """route='commercial', <200 cases ⇒ bootstrap circuit-breaks; verdict → per-case all-pass + flag."""
    cases = [_case(i, 1e-3, near_zero=False) for i in range(6)] + [_case(99, 5e-1, edge=True)]
    out = A.grade_batch(cases, op_class="float", precision_level="L1", route="commercial")
    assert out["scenario"] == "商用"
    assert out["n_representative"] == 6 and out["n_edge"] == 1
    assert out["bootstrap_valid"] is False
    assert out["verdict_basis"].startswith("per_case_all_pass")
    assert out["is_pass"] in (True, False)


def test_grade_batch_commercial_bootstrap_valid_at_scale():
    """route='commercial', ≥200 representative cases ⇒ bootstrap ratio CI verdict is valid."""
    cases = [_case(i, 1e-3, n=64, near_zero=False) for i in range(220)]
    out = A.grade_batch(cases, op_class="float", precision_level="L1", route="commercial")
    assert out["n_representative"] == 220
    assert out["bootstrap_valid"] is True
    assert out["verdict_basis"] == "bootstrap_median_ci"
    assert out["ci_upper"] is not None and out["gate"] == 5.0
    assert out["is_pass"] is True            # ratio ~1 ≪ L1 gate 5


def test_grade_batch_competitor_kind_threaded():
    """Binding infra KEPT: competitor_kind is threaded into the per-case result + batch counts."""
    cases = [dict(_case(i, 1e-4, near_zero=False), competitor_kind="independent_baseline") for i in range(4)]
    out = A.grade_batch(cases, op_class="float")
    assert out["competitor_kinds"] == ["independent_baseline"]
    assert out["competitor_kind_counts"] == {"independent_baseline": 4}
    assert all(pc.get("competitor_kind") == "independent_baseline" for pc in out["per_case"])


def test_grade_batch_commercial_mixed_scenario_refused():
    """codex #1 fail-close: route='commercial' with a rep case lacking 标杆 (→ falls back to 生态
    absolute per-case) is REFUSED — an absolute MARE must never enter the 商用 ratio bootstrap.
    """
    with_tp = _case(0, 1e-4, near_zero=False, tp=True)
    no_tp = _case(1, 1e-4, near_zero=False, tp=False)   # no 标杆 → 生态 fallback inside grade()
    out = A.grade_batch([with_tp, no_tp], op_class="float", route="commercial")
    assert out["is_pass"] is None
    assert out["verdict_basis"] == "REFUSED_mixed_scenario"
    assert "生态" in out["rep_scenarios"] and "商用" in out["rep_scenarios"]
    assert "fail_closed_reason" in out


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            logging.info(f"ok  {fn.__name__}")
            passed += 1
        except Exception:
            logging.info(f"FAIL {fn.__name__}")
            traceback.print_exc()
    logging.info(f"{passed}/{len(fns)} ok")
