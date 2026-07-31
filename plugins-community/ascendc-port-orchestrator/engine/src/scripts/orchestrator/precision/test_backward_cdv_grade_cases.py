# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression tests for backward `grade_cases` — the cannbot single-judge wiring.

Owner 2026-06-18 (完全照抄 cannbot): backward's in-process verify routes its per-
(output×case) RAW arrays through the SAME canonical `precision_cannbot_adapter.grade_batch`
that pass_a/benchmark/port_a3 use (one source of truth, no backward-private statistics).
`grade_cases` is the thin backward-side mapper (grade_batch → verification.json verdict,
fail-closed).

Guardrails pinned (all CPU, no NPU — exercises the GRADE wiring; the collector + full
backward e2e are NPU-runtime and verified separately on A3):
  * clean batch (ours≈golden, no near-zero) → verdict PASS, scenario 生态 (DEFAULT vendored compare.py)
  * bad batch (ours ≫ golden) → verdict FAIL
  * empty cases → REJECT_FAIL_CLOSED (never fabricate PASS)
  * edge cases excluded from the statistical verdict (bug-find stream only)
  * no third_party competitor → 生态 absolute path (still graded)

NOTE (2026-06-30): grade_cases now DEFAULTS to the 生态 vendored cann-bench compare.py; the
third_party 标杆 is only consumed by the OPTIONAL 商用 route (not exercised by grade_cases default).
"""
import sys
from pathlib import Path

import numpy as np
import torch  # noqa: F401  collection-time import (DEBT-47 conftest): the 生态 grader uses torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backward_cdv_grade import grade_cases, _build_record_cases, _as_np  # noqa: E402


def _cases(noise, n=20, *, competitor=True, edge=0, near_zero=False, seed=0):
    """n per-(output×case) raw-array dicts. golden avoids near-zero unless near_zero=True
    (near-zero golden blows up relative error — the same sensitivity selective_scan/GDN hit)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        g = (rng.standard_normal(256) if near_zero
             else np.abs(rng.standard_normal(256)) + 0.5).astype(np.float64)
        npu = (g + rng.standard_normal(256) * noise).astype(np.float32)
        tp = (g + rng.standard_normal(256) * 1e-4).astype(np.float32) if competitor else None
        out.append({"npu": npu, "golden": g, "third_party": tp,
                    "dtype": "bfloat16", "is_edge": (i < edge)})
    return out


def test_clean_batch_passes_ecosystem_default():
    out = grade_cases(_cases(1e-4), "fag")
    assert out["verdict"] == "PASS"
    assert out["cannbot"]["scenario"] == "生态"  # DEFAULT = vendored compare.py (生态), 标杆 not auto-used
    assert out["cannbot"]["pass_rate"] == 1.0


def test_bad_batch_fails():
    out = grade_cases(_cases(1.0), "fag")  # ours ~1.0 abs err ≫ competitor 1e-4
    assert out["verdict"] == "FAIL"
    assert out["cannbot"]["is_pass"] is False


def test_empty_cases_reject_fail_closed():
    out = grade_cases([], "fag")
    assert out["verdict"] == "REJECT_FAIL_CLOSED"
    assert out["cannbot"] is None
    assert "no cases" in out["fail_closed_reasons"][0]


def test_edge_cases_excluded_from_verdict():
    # 5 edge + 15 representative, all clean → verdict on representatives only
    out = grade_cases(_cases(1e-4, n=20, edge=5), "fag")
    assert out["cannbot"]["n_representative"] == 15
    assert out["cannbot"]["n_edge"] == 5
    assert out["verdict"] == "PASS"


def test_no_competitor_routes_ecosystem_absolute():
    # no third_party → 生态 absolute threshold path (still graded, not crashed)
    out = grade_cases(_cases(1e-4, competitor=False), "fag")
    assert out["cannbot"]["scenario"] == "生态"
    assert out["verdict"] in ("PASS", "FAIL")  # graded by absolute threshold, not REJECT


def test_provenance_marks_single_source_of_truth():
    out = grade_cases(_cases(1e-4), "fag")
    p = out["criterion_provenance"]
    assert "grade_batch" in p["standard"]
    assert p["single_source_of_truth"] == "shared grade_batch with pass_a/benchmark/port_a3"


# ── _build_record_cases — the PURE (CPU-testable) verify-side case mapper ──────────
# cases_from_records (backward_cdv_collect) is NPU-runtime (runs the kernel + autograd);
# its pure case-dict construction is isolated here so it is testable without a container.

def _rec_arrays(wrt, *, competitor=True):
    """Per-record per-output arrays: npu wrt-ordered seq, golden dict name->fp64,
    competitor wrt-ordered seq (or None). Distinct values so mis-mapping is visible."""
    npu = [np.full(8, float(oi), np.float32) for oi in range(len(wrt))]
    golden = {w: np.full(8, 100.0 + oi, np.float64) for oi, w in enumerate(wrt)}
    comp = ([np.full(8, 200.0 + oi, np.float64) for oi in range(len(wrt))]
            if competitor else None)
    return npu, golden, comp


def test_build_record_cases_one_per_output_and_fields():
    wrt = ["x", "w"]
    npu, golden, comp = _rec_arrays(wrt)
    cases = _build_record_cases(npu, golden, comp, wrt, "bfloat16", "randn")
    assert len(cases) == len(wrt)                       # one case per (output × record)
    for oi, c in enumerate(cases):
        assert set(c) == {"npu", "golden", "third_party", "native", "native_kind",
                          "dtype", "output", "is_edge", "competitor_kind"}
        assert c["competitor_kind"] is None                # default when caller omits it
        assert c["native"] is None and c["native_kind"] is None  # no native_grads supplied → None
        assert c["output"] == wrt[oi] and c["dtype"] == "bfloat16"
        assert c["npu"][0] == float(oi)                 # wrt-ordered npu
        assert c["golden"][0] == 100.0 + oi             # golden pulled from dict BY NAME
        assert c["third_party"][0] == 200.0 + oi        # competitor wrt-ordered


def test_build_record_cases_is_edge_from_profile():
    wrt = ["g"]
    npu, golden, comp = _rec_arrays(wrt)
    # representative profile → is_edge False (counts toward statistical verdict)
    assert _build_record_cases(npu, golden, comp, wrt, "float32", "randn")[0]["is_edge"] is False
    assert _build_record_cases(npu, golden, comp, wrt, "float32", "representative")[0]["is_edge"] is False
    # every edge value-profile → is_edge True (bug-find stream, excluded from verdict)
    for p in ("zeros", "large", "small", "boundary"):
        assert _build_record_cases(npu, golden, comp, wrt, "float32", p)[0]["is_edge"] is True


def test_build_record_cases_no_competitor_third_party_none():
    wrt = ["g"]
    npu, golden, _ = _rec_arrays(wrt, competitor=False)
    c = _build_record_cases(npu, golden, None, wrt, "float32", "randn")[0]
    assert c["third_party"] is None                     # 生态 absolute path downstream


def test_build_record_cases_chains_into_grade_cases():
    """The pure builder's output is exactly the dict shape grade_cases/grade_batch consume."""
    wrt = ["x", "w"]
    cases = []
    for _ in range(20):                                  # ≥ bootstrap rep floor
        npu, golden, comp = _rec_arrays(wrt)
        npu = [golden[w] for w in wrt]                   # npu == golden ⇒ zero error ⇒ clean
        cases += _build_record_cases(npu, golden, comp, wrt, "bfloat16", "randn")
    out = grade_cases(cases, "fag")
    assert out["verdict"] == "PASS" and out["cannbot"]["scenario"] == "生态"


def test_as_np_coerces_torch_tensor():
    t = torch.arange(4, dtype=torch.float32)
    a = _as_np(t)
    assert isinstance(a, np.ndarray) and a.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert _as_np(None) is None


def test_as_np_bf16_upcasts_lossless():
    """numpy has no bfloat16 → _as_np must upcast bf16 tensors to fp32 (lossless: fp32 ⊃ bf16).
    Without this, .numpy() raises 'unsupported ScalarType BFloat16' on a bf16 kernel output.
    """
    t = torch.tensor([1.5, -2.25, 0.0], dtype=torch.bfloat16)  # exact in bf16
    a = _as_np(t)
    assert isinstance(a, np.ndarray) and a.dtype == np.float32
    assert a.tolist() == [1.5, -2.25, 0.0]


def test_build_record_cases_bf16_npu_and_competitor():
    """bf16 kernel output + bf16 competitor must build cases without numpy bf16 error."""
    wrt = ["x"]
    npu = [torch.tensor([1.0, 2.0], dtype=torch.bfloat16)]   # bf16 kernel output
    golden = {"x": np.array([1.0, 2.0], np.float64)}
    comp = [torch.tensor([1.0, 2.0], dtype=torch.bfloat16)]  # bf16 competitor
    c = _build_record_cases(npu, golden, comp, wrt, "bfloat16", "randn")[0]
    assert c["npu"].dtype == np.float32 and c["third_party"].dtype == np.float32
    assert c["dtype"] == "bfloat16" and c["is_edge"] is False


# ── Phase 2: native_output (CPU-same-precision) wiring through _build_record_cases → grade_batch ──
def test_native_grads_threaded_and_relaxes_small_value():
    """_build_record_cases emits `native`(+native_kind='cpu_same_precision') from the CPU-same-
    precision grads; grade_batch consumes it → a small-value case that FAILs native=None PASSes
    with native (compare.py ratio≤2 carve-out). This proves the 生态 grader RECEIVES native_output
    on the backward path (codex #3 / Phase 2).
    """
    from precision_cannbot_adapter import grade_batch
    wrt = ["g"]
    gold = {"g": np.full(64, 1e-8, np.float64)}              # small-value region (<2^-14 fp32)
    npu = [np.full(64, 1e-8 + 1e-6, np.float32)]             # error > small_value_error
    native = [np.full(64, 1e-8 + 1e-6, np.float64)]          # CPU-same-precision, SAME error

    cases_with = _build_record_cases(npu, gold, None, wrt, "float32", "randn", native_grads=native)
    assert cases_with[0]["native"] is not None
    assert cases_with[0]["native_kind"] == "cpu_same_precision"
    out_with = grade_batch(cases_with, op_class="float")
    assert out_with["per_case"][0]["is_pass"] is True        # relaxed via native carve-out

    cases_no = _build_record_cases(npu, gold, None, wrt, "float32", "randn", native_grads=None)
    assert cases_no[0]["native"] is None and cases_no[0]["native_kind"] is None
    out_no = grade_batch(cases_no, op_class="float")
    assert out_no["per_case"][0]["is_pass"] is False         # strict native=None


def test_grade_cases_persists_bounded_raw_outputs(tmp_path):
    """Part B / criterion 3: grade_cases persists a TOTAL-BOUNDED subset of raw npu/golden/native
    arrays so a failed op can be re-graded offline with NO NPU re-run.
    """
    p = str(tmp_path / "npu_outputs.pt")
    cases = _cases(1e-4, n=20, edge=3)                       # 17 representative + 3 edge (clean → all PASS)
    out = grade_cases(cases, "fag", persist_outputs_path=p, persist_cap=10)
    prov = out["criterion_provenance"]["persisted_outputs"]
    assert prov["schema"] == "backward_npu_outputs_v1"
    assert prov["n_total"] == 20
    assert prov["n_persisted"] == 10                         # TOTAL cap (not cap+edge)
    assert prov["capped"] is True
    assert prov["all_failures_kept"] is True                 # no failures here, trivially true
    blob = torch.load(p, weights_only=False)
    assert blob["op"] == "fag" and len(blob["cases"]) == 10
    assert blob["cases"][0]["npu"] is not None and blob["cases"][0]["golden"] is not None


def test_persist_keeps_failing_cases_first_under_cap(tmp_path):
    """main gate (keep-failures-first): a cap that would drop a FAILING case defeats the re-grade
    purpose. With many passes + a few fails and a tight cap, ALL failing cases MUST survive.
    """
    # 2 FAILING representative (npu ≫ golden) + 12 clean PASS + 1 edge; cap=4
    fail = [{"npu": np.full(64, 9.0, np.float32), "golden": np.full(64, 1.0, np.float64),
             "third_party": None, "dtype": "float32", "is_edge": False} for _ in range(2)]
    ok = [{"npu": np.full(64, 1.0, np.float32), "golden": np.full(64, 1.0, np.float64),
           "third_party": None, "dtype": "float32", "is_edge": False} for _ in range(12)]
    edge = [{"npu": np.full(64, 1.0, np.float32), "golden": np.full(64, 1.0, np.float64),
             "third_party": None, "dtype": "float32", "is_edge": True}]
    p = str(tmp_path / "npu_outputs.pt")
    out = grade_cases(ok + fail + edge, "fag", persist_outputs_path=p, persist_cap=4)
    prov = out["criterion_provenance"]["persisted_outputs"]
    assert prov["n_failing_representative"] == 2
    assert prov["n_failing_persisted"] == 2                  # BOTH failures survive the tight cap
    assert prov["all_failures_kept"] is True
    assert prov["n_persisted"] == 4                          # total bound respected
    blob = torch.load(p, weights_only=False)
    # the 2 failing cases (npu==9.0) are present despite cap < n_total
    n_fail_in_blob = sum(1 for c in blob["cases"] if float(c["npu"].flat[0]) == 9.0)
    assert n_fail_in_blob == 2


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
