# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""task#25 — port_a3 reference-capture content validation + task#24 entry-detection.

Surfaced by the §6 cube-MIX confirm-run (fused_quant_mat_mul, 2026-06-01): the
A3 reference producer declared `verdict=READY` over edge_dataset.pt with 0/73
a3_outputs captured + a3_baseline_perf.json median_ms_per_case={}, wasting a
406s / $2.28 worker spawn that had to self-detect the empty baseline.

Main decisions (2026-06-01, locked):
  (1) Empty/partial capture → CAPTURE_INCOMPLETE, FAIL-FAST. NOT fallback-eligible
      (no silent CPU-truth degrade — fp32 truth is a misleading oracle for quant
      ops = a fake-pass per the no-CPU-fallback rule).
  (2) READY requires FULL capture: n_captured == n_total (partial = coverage fraud).

Pins:
  A. _count_a3_outputs handles BOTH edge_dataset.pt schemas (dict + list-of-cases).
  B. _validate_a3_capture: red (empty/partial/empty-perf) / green (full) across
     both schemas + both tiers (manifest / torch fallback).
  C. derive_aclnn_entry content-fallback (task#24): mat_mul_v3-style API-named
     example resolves on a UNIQUE aclnn-symbol match; ambiguous (>1) → None.
  D. Source-pin: both READY sites route through _validate_a3_capture, and
     CAPTURE_INCOMPLETE is deliberately NOT in orchestrator's _fallback_eligible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors
import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o25_a3_ref as p25a3  # noqa: E402

torch = pytest.importorskip("torch")  # Tier-2 fallback + dataset fixtures need torch


# ---------------------------------------------------------------------------
# Fixtures: write the two edge_dataset.pt schemas + perf
# ---------------------------------------------------------------------------
def _write_perf(ws: Path, n: int) -> None:
    """a3_baseline_perf.json with n per-case medians (n==0 → empty dict = the bug)."""
    median = {str(i): 0.05 for i in range(n)}
    (ws / "a3_baseline_perf.json").write_text(
        json.dumps({"op": "x", "median_ms_per_case": median, "schema_version": 1})
    )


def _write_dataset_list(ws: Path, n_total: int, n_captured: int) -> None:
    """Schema 2 (per-op run_a3_reference.py): list of case dicts; first n_captured
    carry a3_outputs, the rest carry an a3_error (mirrors §6 fused_quant_mat_mul)."""
    cases = []
    for i in range(n_total):
        c = {"case_id": i, "inputs": {"a": torch.zeros(2)}}
        if i < n_captured:
            c["a3_outputs"] = torch.ones(2)
        else:
            c["a3_error"] = "aclnn call returned non-zero"
        cases.append(c)
    torch.save(cases, ws / "edge_dataset.pt")


def _write_dataset_dict(ws: Path, n_total: int, captured: bool) -> None:
    """Schema 1 (CPU-truth synth): {"inputs":[...], "a3_outputs":[...]}."""
    payload = {"inputs": [torch.zeros(2) for _ in range(n_total)]}
    if captured:
        payload["a3_outputs"] = [torch.ones(2) for _ in range(n_total)]
    torch.save(payload, ws / "edge_dataset.pt")


def _write_dataset_int_keyed(ws: Path, n_total: int, n_captured: int) -> None:
    """Schema 3 (the shape aog-a3-author emitted for flash_attention_score,
    2026-07-03): an INT-KEYED dict {0:case, 1:case, ..., N-1:case} — NO top-level
    a3_outputs/inputs/cases key, so it falsely read as 0 cases (CAPTURE_INCOMPLETE)
    before _coerce_case_list. First n_captured carry a3_outputs, rest carry a3_error."""
    ds = {}
    for i in range(n_total):
        c = {"name": f"case{i}"}
        if i < n_captured:
            c["a3_outputs"] = {"attention_out": torch.ones(2)}
        else:
            c["a3_error"] = "aclnn returned non-zero"
        ds[i] = c
    torch.save(ds, ws / "edge_dataset.pt")


# ---------------------------------------------------------------------------
# A. _count_a3_outputs — both schemas
# ---------------------------------------------------------------------------
def test_count_list_schema_full(tmp_path):
    _write_dataset_list(tmp_path, 8, 8)
    ds = torch.load(tmp_path / "edge_dataset.pt", weights_only=False)
    assert getattr(p25a3, '_count_a3_outputs')(ds) == (8, 8)


def test_count_list_schema_partial(tmp_path):
    _write_dataset_list(tmp_path, 73, 0)  # the §6 case
    ds = torch.load(tmp_path / "edge_dataset.pt", weights_only=False)
    assert getattr(p25a3, '_count_a3_outputs')(ds) == (0, 73)


def test_count_list_schema_error_cases_not_counted(tmp_path):
    """A case with a3_outputs BUT a non-empty a3_error does not count as captured."""
    cases = [
        {"case_id": 0, "a3_outputs": torch.ones(2)},
        {"case_id": 1, "a3_outputs": torch.ones(2), "a3_error": "stale partial write"},
    ]
    assert getattr(p25a3, '_count_a3_outputs')(cases) == (1, 2)


def test_count_dict_schema(tmp_path):
    _write_dataset_dict(tmp_path, 5, captured=True)
    ds = torch.load(tmp_path / "edge_dataset.pt", weights_only=False)
    assert getattr(p25a3, '_count_a3_outputs')(ds) == (5, 5)
    _write_dataset_dict(tmp_path, 5, captured=False)
    ds = torch.load(tmp_path / "edge_dataset.pt", weights_only=False)
    assert getattr(p25a3, '_count_a3_outputs')(ds) == (0, 5)


# ---------------------------------------------------------------------------
# A'. _coerce_case_list + int-keyed dict (FA regression, 2026-07-03)
# ---------------------------------------------------------------------------
def test_coerce_int_keyed_dict_to_list():
    ds = {0: {"a3_outputs": 1}, 1: {"a3_outputs": 2}, 2: {"a3_outputs": 3}}
    out = getattr(p25a3, '_coerce_case_list')(ds)
    assert isinstance(out, list) and len(out) == 3
    assert [c["a3_outputs"] for c in out] == [1, 2, 3]  # order preserved


def test_coerce_str_int_keyed_dict():
    out = getattr(p25a3, '_coerce_case_list')({"0": {"x": 1}, "1": {"x": 2}})
    assert isinstance(out, list) and [c["x"] for c in out] == [1, 2]


def test_coerce_leaves_documented_schemas_untouched():
    s1 = {"inputs": [1, 2], "a3_outputs": [3, 4]}
    assert getattr(p25a3, '_coerce_case_list')(s1) is s1          # Schema 1 aligned-lists
    lst = [{"a3_outputs": 1}]
    assert getattr(p25a3, '_coerce_case_list')(lst) is lst        # list stays list
    nc = {0: "a", 2: "b"}
    assert getattr(p25a3, '_coerce_case_list')(nc) is nc          # non-contiguous → don't guess
    other = {"foo": 1, "bar": 2}
    assert getattr(p25a3, '_coerce_case_list')(other) is other    # non-int keys → untouched
    assert getattr(p25a3, '_coerce_case_list')({}) == {}          # empty → untouched


def test_count_int_keyed_schema_full(tmp_path):
    """FA regression: int-keyed dict of 43 fully-captured cases counts 43/43,
    NOT the false 0/0 that produced the CAPTURE_INCOMPLETE false-fail.
    """
    _write_dataset_int_keyed(tmp_path, 43, 43)
    ds = torch.load(tmp_path / "edge_dataset.pt", weights_only=False)
    assert getattr(p25a3, '_count_a3_outputs')(ds) == (43, 43)


def test_count_int_keyed_schema_partial(tmp_path):
    """Coercion does NOT weaken the coverage-fraud gate: error cases still uncounted."""
    _write_dataset_int_keyed(tmp_path, 10, 6)
    ds = torch.load(tmp_path / "edge_dataset.pt", weights_only=False)
    assert getattr(p25a3, '_count_a3_outputs')(ds) == (6, 10)


def test_validate_green_full_capture_int_keyed(tmp_path):
    """End-to-end: full int-keyed capture + perf → READY (Tier-2 torch path)."""
    _write_dataset_int_keyed(tmp_path, 43, 43)
    _write_perf(tmp_path, 43)
    ok, msg = getattr(p25a3, '_validate_a3_capture')(tmp_path)
    assert ok, msg


def test_validate_red_partial_int_keyed(tmp_path):
    """Partial int-keyed capture → CAPTURE_INCOMPLETE (fraud gate intact post-coerce)."""
    _write_dataset_int_keyed(tmp_path, 43, 40)
    _write_perf(tmp_path, 43)
    ok, msg = getattr(p25a3, '_validate_a3_capture')(tmp_path)
    assert not ok and "40/43" in msg


# ---------------------------------------------------------------------------
# A''. provision_cpu_truth — 生态 T1 golden auto-gen (DEBT-199, 2026-07-03)
# ---------------------------------------------------------------------------
def _write_model_and_inputs(ws: Path, model_src: str, n: int = 3) -> None:
    (ws / "model.py").write_text(model_src)
    # REAL case_gen structure: WRAPPED {idx,name,shape,inputs:{...},meta} — NOT flat.
    # DEBT-199 v2 regression: the flat-vs-nested bug (model.forward(**whole_wrapper) →
    # TypeError: unexpected keyword 'idx') hid behind a flat smoke fixture in v1. Both
    # provision_cpu_truth AND provision_native_capture must unwrap case['inputs'].
    cases = [{"idx": i, "name": f"c{i}", "shape": [1, 2, 4, 8],
              "inputs": {"q": torch.randn(1, 2, 4, 8, dtype=torch.float16),
                         "k": torch.randn(1, 2, 4, 8, dtype=torch.float16),
                         "v": torch.randn(1, 2, 4, 8, dtype=torch.float16),
                         "scale": 0.35},
              "meta": {"effective_scalars": {"scale": 0.35}}} for i in range(n)]
    torch.save({"cases": cases}, ws / "edge_inputs.pt")


_FA_LIKE_MODEL = '''
import torch
class Model(torch.nn.Module):
    def forward(self, q, k, v, scale):
        s = (q @ k.transpose(-1, -2)) * scale
        p = torch.softmax(s, dim=-1)
        return p @ v, s.amax(dim=-1)   # multi-output like FA (attention_out, rowmax)
'''


def test_case_model_kwargs_unwraps_nested_only():
    """DEBT-199 v2: unwrap the wrapped case to case['inputs'] (the model.forward args); leave
    flat/positional cases untouched.
    """
    nested = {"idx": 0, "name": "x", "shape": [2], "inputs": {"a": 1, "b": 2}, "meta": {}}
    assert getattr(p25a3, '_case_model_kwargs')(nested) == {"a": 1, "b": 2}
    assert getattr(p25a3, '_case_model_kwargs')({"a": 1, "b": 2}) == {"a": 1, "b": 2}   # no wrapper markers
    assert getattr(p25a3, '_case_model_kwargs')([1, 2, 3]) == [1, 2, 3]                 # positional


def test_provision_cpu_truth_fp64_golden(tmp_path):
    """DEBT-199: when A3 succeeds, the harness auto-provisions cpu_truth_outputs.pt at fp64
    (the 生态 T1 golden) so O5 grades ours-vs-cpu_truth, not the vs-A3 T3 fallback.
    Uses the REAL nested case_gen structure (v1 flat smoke missed the wrapper-navigation bug).
    """
    _write_model_and_inputs(tmp_path, _FA_LIKE_MODEL, n=3)
    ok, msg = p25a3.provision_cpu_truth(tmp_path)
    assert ok, msg
    blob = torch.load(tmp_path / "cpu_truth_outputs.pt", weights_only=False)
    assert blob["golden_kind"] == "cpu_fp64" and blob["dtype"] == "float64"
    assert len(blob["outputs"]) == 3
    o0 = blob["outputs"][0]
    first = o0[0] if isinstance(o0, tuple) else o0     # FA multi-output handled uniformly
    assert first.dtype == torch.float64                # genuinely fp64, not an fp16 cast


def test_provision_cpu_truth_fail_closed_on_npu_only(tmp_path):
    """A genuinely NPU-only op (model.forward raises on CPU) leaves cpu_truth ABSENT — never a
    partial file that would masquerade as a 生态 golden (fail-closed, mirrors native_capture P3).
    """
    (tmp_path / "model.py").write_text(
        "import torch\n"
        "class Model(torch.nn.Module):\n"
        "    def forward(self, q, k, v, scale):\n"
        "        raise RuntimeError('NPU-only op, cannot run on CPU')\n"
    )
    _write_model_and_inputs(tmp_path, (tmp_path / "model.py").read_text(), n=2)
    ok, msg = p25a3.provision_cpu_truth(tmp_path)
    assert not ok and not (tmp_path / "cpu_truth_outputs.pt").exists()
    assert "FAIL-CLOSED" in msg


def test_provision_cpu_truth_does_not_touch_a3_outputs(tmp_path):
    """cpu_truth is a SEPARATE artifact — provisioning it must NOT overwrite a real a3_outputs
    (A3 stays the T2 competitor).
    """
    _write_dataset_list(tmp_path, 3, 3)   # a real edge_dataset.pt with a3_outputs
    before = (tmp_path / "edge_dataset.pt").read_bytes()
    _write_model_and_inputs(tmp_path, _FA_LIKE_MODEL, n=3)
    ok, _ = p25a3.provision_cpu_truth(tmp_path)
    assert ok
    assert (tmp_path / "edge_dataset.pt").read_bytes() == before   # a3_outputs untouched
    assert (tmp_path / "cpu_truth_outputs.pt").is_file()


def test_provision_native_capture_navigates_nested_case(tmp_path):
    """DEBT-199 v2: provision_native_capture shares the SAME flat-vs-nested bug (disk-confirmed
    43/43-fail on FA with the identical TypeError). The wrapped case must navigate to
    case['inputs'] → native_capture.pt written (not a silent fail-closed native=None).
    """
    _write_model_and_inputs(tmp_path, _FA_LIKE_MODEL, n=3)
    ok, msg = p25a3.provision_native_capture(tmp_path)
    assert ok, msg
    assert (tmp_path / "native_capture.pt").is_file()
    blob = torch.load(tmp_path / "native_capture.pt", weights_only=False)
    assert blob["native_kind"] in ("cpu_same_precision", "cpu_fp32_fallback")
    assert len(blob["outputs"]) == 3


# ---------------------------------------------------------------------------
# B. _validate_a3_capture — red/green across schemas + tiers
# ---------------------------------------------------------------------------
def test_validate_green_full_capture_list(tmp_path):
    _write_dataset_list(tmp_path, 8, 8)
    _write_perf(tmp_path, 8)
    ok, reason = getattr(p25a3, '_validate_a3_capture')(tmp_path)
    assert ok is True, reason


def test_validate_green_full_capture_dict(tmp_path):
    _write_dataset_dict(tmp_path, 5, captured=True)
    _write_perf(tmp_path, 5)
    ok, reason = getattr(p25a3, '_validate_a3_capture')(tmp_path)
    assert ok is True, reason


def test_validate_red_zero_outputs_the_s6_bug(tmp_path):
    """The exact §6 signature: 0/73 outputs + non-empty perf → reject."""
    _write_dataset_list(tmp_path, 73, 0)
    _write_perf(tmp_path, 73)
    ok, reason = getattr(p25a3, '_validate_a3_capture')(tmp_path)
    assert ok is False
    assert "0/73" in reason


def test_validate_red_partial_capture(tmp_path):
    """Decision (2): n_captured != n_total → reject even though some captured."""
    _write_dataset_list(tmp_path, 73, 40)
    _write_perf(tmp_path, 73)
    ok, reason = getattr(p25a3, '_validate_a3_capture')(tmp_path)
    assert ok is False
    assert "40/73" in reason


def test_validate_red_empty_perf(tmp_path):
    """Outputs full but perf empty ({}) → reject (the other half of §6)."""
    _write_dataset_list(tmp_path, 8, 8)
    _write_perf(tmp_path, 0)  # median_ms_per_case == {}
    ok, reason = getattr(p25a3, '_validate_a3_capture')(tmp_path)
    assert ok is False
    assert "median_ms_per_case" in reason


def test_validate_red_missing_files(tmp_path):
    ok, reason = getattr(p25a3, '_validate_a3_capture')(tmp_path)
    assert ok is False
    assert "absent" in reason


def test_validate_self_reported_manifest_does_not_replace_dataset_count(tmp_path):
    """Coverage comes from captured bytes, not the runner's self-report."""
    _write_dataset_list(tmp_path, 8, 8)
    _write_perf(tmp_path, 8)
    (tmp_path / "a3_capture_manifest.json").write_text(
        json.dumps({"n_total": 8, "n_captured": 8})
    )
    ok, reason = getattr(p25a3, '_validate_a3_capture')(tmp_path)
    assert ok is True
    assert "8/8" in reason


def test_validate_self_reported_partial_manifest_cannot_override_full_dataset(tmp_path):
    """A stale runner manifest cannot turn a complete captured dataset red."""
    _write_dataset_list(tmp_path, 8, 8)  # dataset looks full…
    _write_perf(tmp_path, 8)
    (tmp_path / "a3_capture_manifest.json").write_text(
        json.dumps({"n_total": 8, "n_captured": 3})  # …but manifest says 3/8
    )
    ok, reason = getattr(p25a3, '_validate_a3_capture')(tmp_path)
    assert ok is True
    assert "8/8" in reason


def test_validate_tier2_when_manifest_absent_cannot_be_bypassed(tmp_path):
    """OL-160: a runner that OMITS the manifest cannot bypass the gate — Tier-2
    torch.load still rejects an empty capture.
    """
    _write_dataset_list(tmp_path, 73, 0)
    _write_perf(tmp_path, 73)
    assert not (tmp_path / "a3_capture_manifest.json").exists()
    ok, reason = getattr(p25a3, '_validate_a3_capture')(tmp_path)
    assert ok is False
    assert "0/73" in reason


# ---------------------------------------------------------------------------
# C. derive_aclnn_entry content-fallback (task#24)
# ---------------------------------------------------------------------------
def _mk_example(op_dir: Path, fname: str, api: str) -> None:
    ex = op_dir / "examples"
    ex.mkdir(parents=True, exist_ok=True)
    (ex / fname).write_text(
        f"// generated test\nint main() {{ aclnn{api}GetWorkspaceSize(...); "
        f"aclnn{api}(...); return 0; }}\n"
    )


def test_entry_content_fallback_unique_match(tmp_path):
    """mat_mul_v3-style: dir name never matches; resolve via UNIQUE aclnn symbol."""
    op_dir = tmp_path / "mat_mul_v3"
    _mk_example(op_dir, "test_aclnn_mm.cpp", "Mm")
    _mk_example(op_dir, "test_aclnn_matmul.cpp", "Matmul")
    _mk_example(op_dir, "test_aclnn_addmm.cpp", "Addmm")
    entry = p25a3.derive_aclnn_entry(op_dir)
    assert entry is not None and entry.name == "test_aclnn_matmul.cpp"


def test_entry_name_match_takes_precedence_over_content(tmp_path):
    """If the canonical dir-named entry exists, it wins (content-fallback is last)."""
    op_dir = tmp_path / "fused_quant_mat_mul"
    _mk_example(op_dir, "test_aclnn_fused_quant_mat_mul.cpp", "FusedQuantMatmul")
    _mk_example(op_dir, "test_aclnn_mm.cpp", "Mm")
    entry = p25a3.derive_aclnn_entry(op_dir)
    assert entry is not None and entry.name == "test_aclnn_fused_quant_mat_mul.cpp"


def test_entry_content_fallback_ambiguous_returns_none(tmp_path):
    """>1 candidates exact-match the normalized op name → DO NOT guess → None."""
    op_dir = tmp_path / "foo"
    _mk_example(op_dir, "test_aclnn_a.cpp", "Foo")
    _mk_example(op_dir, "test_aclnn_b.cpp", "Foo")  # two files, same API → ambiguous
    assert p25a3.derive_aclnn_entry(op_dir) is None


def test_entry_no_examples_returns_none(tmp_path):
    op_dir = tmp_path / "bar"
    op_dir.mkdir()
    assert p25a3.derive_aclnn_entry(op_dir) is None


def test_entry_real_mat_mul_v3_integration():
    """Integration against the real ops-nn layout (skips if CANN tree absent)."""
    real = Path("/home/npu_user/workspace/cann/ops-nn/matmul/mat_mul_v3")
    if not real.is_dir():
        pytest.skip("CANN ops-nn tree not present")
    entry = p25a3.derive_aclnn_entry(real)
    assert entry is not None and entry.name == "test_aclnn_matmul.cpp"


# ---------------------------------------------------------------------------
# D. Source-pins (guard the wiring against future drift)
# ---------------------------------------------------------------------------
def test_only_fresh_post_exec_path_can_declare_ready():
    """No cached READY path may bypass the post-exec capture validator."""
    src = (_reorg_paths.ORCH_DIR / "phase_o25_a3_ref.py").read_text()
    assert src.count("_validate_a3_capture(workspace)") == 1, (
        "expected one fresh post-exec READY site; an extra call may reintroduce "
        "a cached-capture acceptance path"
    )
    assert 'rep.verdict = "CAPTURE_INCOMPLETE"' in src


def test_capture_incomplete_not_fallback_eligible():
    """Every non-READY live-capture verdict, including incomplete, blocks."""
    src = (_reorg_paths.ORCH_DIR / "fsm_phase_o25_dispatch.py").read_text()
    assert 'if o25_a3.verdict != "READY":' in src
    assert "_fallback_eligible" not in src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
