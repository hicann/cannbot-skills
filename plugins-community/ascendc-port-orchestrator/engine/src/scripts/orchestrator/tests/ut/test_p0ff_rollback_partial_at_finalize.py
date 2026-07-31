# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0ff (2026-05-05): rollback gate at finalize state.

Origin: today's sweep finalized 27 ops by promoting workspace -> archive,
but 3 of them (14_adaptive_instance_norm_bwd, 19_IndexPut, 23_hyenafft)
had precision.status=PARTIAL with NO persist_verdict. They should NOT
have been at "done"; the pipeline wasn't legitimately exhausted.

Rule: precision.status in {PARTIAL, FAIL} requires persist_verdict=
PARTIAL_PERSIST AND probe_report.md to advance to finalize. Otherwise
rollback to the appropriate non-terminal state to fill the gap.

Rollback target priority:
  1. no probe_report.md  -> await_probe
  2. probe done, no cann_strategy_inference.md  -> await_researcher
  3. researcher done, no optimization_log.md  -> await_optimizer
  4. all artifacts present, no persist_verdict  -> await_worker (re-evaluate)

PASS / PASS_WITHIN_TOLERANCE always eligible.
PARTIAL + persist_verdict=PARTIAL_PERSIST + probe_report.md -> eligible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import finalize_pipeline as fp  # noqa: E402


_KB_WRITEUP_OK = """# knowledge_update.md — TestOp

## Context
Test op for finalize gate.

## Findings
1. **Test finding**: this is a fixture for the P0aax KB-content gate.
   When kw spawns and exits with PASS, knowledge_update.md must exist
   with this structure or finalize rolls back to await_worker.

## KB-promotable patterns (proposed)
None for fixture.

## Cited KB items
None — fixture only.

## Anti-patterns avoided
None applicable.
"""


_AUDIT_DOC_OK = """# aog-self-critic — TestOp post-worker audit

**Verdict: ✅ PASS** with no findings.

C18 delegation: clean. C13 denominator: 50/50 verified. C25/C26 anti-overfit: PASS.
"""


def _seed(ws: Path, prec: dict, probe: bool = False, research: bool = False,
          opt_log: bool = False, kb_writeup: bool = True,
          audit_doc: bool = True, deleg_marker: bool = True,
          full_verification: bool = True):
    ws.mkdir(parents=True, exist_ok=True)
    (ws / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 1,
        "op": "test_op",
        "opgen_mode": "backward",
    }))
    # P0aba: PASS gate now also requires pass_b.status + perf info in
    # verification.json. Build a fixture that satisfies that for the legacy
    # tests; explicit gate-test cases override.
    v = {
        "precision": prec,
        "harness_pristine": {
            "state": "CLEAN", "o5_verdict": "VERIFIED",
            "sampled_at": "o5_post_verify",
        },
    }
    if full_verification and prec.get("status") in ("PASS", "PASS_WITHIN_TOLERANCE"):
        prec.setdefault("pass_b", {"status": "PASS", "tier1_pass": 10, "total": 10,
                                    "method": "test fixture inline"})
        v["performance"] = {
            "ratio": 1.0, "ratio_baseline": "Path A — CPU-truth (test fixture)",
            "status": "PASS",
            "independent_re_measure": {"status": "N/A", "reason": "test fixture"},
        }
    (ws / "verification.json").write_text(json.dumps(v))
    if probe:
        (ws / "probe_report.md").write_text("# probe report\n" + "x" * 200)
    if research:
        (ws / "cann_strategy_inference.md").write_text("# researcher\n" + "x" * 200)
    if opt_log:
        (ws / "optimization_log.md").write_text("# optimizer log\n" + "x" * 200)
    # P0aax: PASS handoffs need knowledge_update.md to be eligible. Default
    # to seeding it so legacy tests stay green; tests that exercise the gate
    # explicitly pass kb_writeup=False.
    if kb_writeup:
        (ws / "knowledge_update.md").write_text(_KB_WRITEUP_OK)
    # P0aba: PASS handoffs also need post-worker self-critic audit doc.
    if audit_doc:
        (ws / "audit_self_critic_post_worker.md").write_text(_AUDIT_DOC_OK)
    # P0aba: delegation scan marker.
    if deleg_marker:
        (ws / ".delegation_scan_passed").write_text(
            "scanner=scan_delegation_cheating.py violations=0 ts=test\n"
        )
    # PB-33 (2026-05-14): op_host/ completeness gate — seed minimum 3 files
    # so eligible-path tests don't trip OP_HOST_COMPLETENESS gate. Tests
    # targeting OP_HOST_COMPLETENESS explicitly should override.
    op_host = ws / "op_host"
    op_host.mkdir(exist_ok=True)
    if not list(op_host.glob("*.cpp")) and not list(op_host.glob("*.h")):
        (op_host / "stub_def.cpp").write_text("// stub op_def\n")
        (op_host / "stub_tiling.cpp").write_text("// stub tiling\n")
        (op_host / "stub_tiling.h").write_text("// stub tiling header\n")
    # DEBT-NEW (2026-05-14, OL-160): _check_universal_entrypoints gate
    # requires canonical Python entry-point file names at workspace root
    # for ANY PASS verdict (mode-agnostic). Seed minimum stubs so legacy
    # tests stay green; tests targeting the gate override.
    if not (ws / "model_new_ascendc.py").is_file():
        (ws / "model_new_ascendc.py").write_text(
            "import torch.nn as nn\n"
            "class ModelNew(nn.Module):\n    def forward(self,x): return x\n"
            "if __name__ == '__main__':\n    pass\n"
        )
    if not (ws / "model.py").is_file():
        (ws / "model.py").write_text(
            "import torch.nn as nn\n"
            "class Model(nn.Module):\n    def forward(self,x): return x\n"
        )
    kernel = ws / "kernel"
    kernel.mkdir(exist_ok=True)
    (kernel / "libtest_op.so").write_bytes(b"compiled-extension")
    (ws / "verify_test_op.py").write_text(
        "from model_new_ascendc import ModelNew\n"
        "candidate = ModelNew()\n"
        "output = candidate(inputs)\n"
    )


def test_pass_eligible(tmp_path):
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}})
    elig = fp.check_finalize_eligibility(tmp_path)
    assert elig["eligible"]
    assert elig["rollback_state"] is None


def test_backward_pass_within_tolerance_requires_exact_pass_a(tmp_path):
    _seed(tmp_path, {"status": "PASS_WITHIN_TOLERANCE",
                     "pass_a": {"status": "PASS_WITHIN_TOLERANCE",
                                "tier1_pass": 47, "total": 50}})
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert elig["gate"] == fp.GateID.PASS_COUNT.value


# ---- P0aax KB-content gate (2026-05-07) ----

def test_pass_without_kb_writeup_rolls_back(tmp_path):
    """The 6_QuantMatmul regression: PASS but no knowledge_update.md → rollback."""
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          kb_writeup=False)
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert elig["rollback_state"] == "await_worker"
    assert "knowledge_update.md missing" in elig["reason"]


def test_pass_with_tiny_kb_writeup_rolls_back(tmp_path):
    """File exists but too short — gate rejects."""
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          kb_writeup=False)
    (tmp_path / "knowledge_update.md").write_text("tiny")
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert elig["rollback_state"] == "await_worker"
    assert "100" in elig["reason"]  # mentions the byte threshold


def test_pass_with_kb_writeup_lacking_findings_rolls_back(tmp_path):
    """File ≥ 100 bytes but no `## Findings` section → gate rejects."""
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          kb_writeup=False)
    (tmp_path / "knowledge_update.md").write_text("# random text\n" + "x" * 200)
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert elig["rollback_state"] == "await_worker"
    assert "Findings" in elig["reason"]


def test_pass_within_tolerance_without_kb_writeup_rolls_back(tmp_path):
    """PASS_WITHIN_TOLERANCE also gated by KB-content rule."""
    _seed(tmp_path, {"status": "PASS_WITHIN_TOLERANCE",
                     "pass_a": {"status": "PASS_WITHIN_TOLERANCE",
                                "tier1_pass": 47, "total": 50}},
          kb_writeup=False)
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert elig["rollback_state"] == "await_worker"


# ---- P0aba audit gate (2026-05-07) ----

def test_pass_without_self_critic_audit_doc_rolls_back(tmp_path):
    """The 6_QuantMatmul finalize gap: PASS but audit_self_critic_post_worker.md
    missing → rollback. User caught manually 2026-05-07.
    """
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          audit_doc=False)
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert elig["rollback_state"] == "await_worker"
    assert "audit_self_critic_post_worker.md" in elig["reason"]


def test_pass_with_self_critic_fail_rolls_back(tmp_path):
    """Audit doc says ❌ block → rollback."""
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          audit_doc=False)
    (tmp_path / "audit_self_critic_post_worker.md").write_text(
        "# self-critic\n\n**Verdict: ❌ block** — C18 delegation found in pybind11.cpp"
    )
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert elig["rollback_state"] == "await_worker"


def test_pass_with_partial_audit_and_waiver_eligible(tmp_path):
    """Audit doc says PARTIAL with explicit waiver → eligible."""
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          audit_doc=False)
    (tmp_path / "audit_self_critic_post_worker.md").write_text(
        "# self-critic\n\n**Verdict: PARTIAL** — C13 denominator confirmed but no "
        "independent re-verifier. Waiver: bit-exact MERE=0 across all cases + 3-run "
        "determinism re-confirmed mitigates risk for Path A op."
    )
    elig = fp.check_finalize_eligibility(tmp_path)
    assert elig["eligible"]


def test_pass_without_delegation_marker_rolls_back(tmp_path):
    """No .delegation_scan_passed marker AND no verification.json field → rollback."""
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          deleg_marker=False)
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert elig["rollback_state"] == "await_worker"
    assert "delegation_scan" in elig["reason"]


def test_pass_without_delegation_marker_blocks_even_with_verification_field(tmp_path):
    """P0aba codex 2026-05-07 #2: verification.json self-claim alone is
    NOT accepted. Worker can't bypass scanner by writing audit field —
    must produce real .delegation_scan_passed marker.
    """
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          deleg_marker=False, full_verification=False)
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
            "pass_b": {"status": "N/A", "reason": "Path A test fixture"},
        },
        "performance": {"status": "PASS", "ratio": 1.0,
                        "ratio_baseline": "Path A — CPU-truth (test fixture)"},
        "audit": {"delegation_scan": {"violations": 0, "scanner": "test"}},
    }))
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert ".delegation_scan_passed" in elig["reason"]


def test_pass_with_stale_delegation_marker_rolls_back(tmp_path):
    """P0aba codex 2026-05-07 #2 freshness: marker mtime older than newest
    kernel file mtime → rolls back.
    """
    import os
    import time
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}})
    # Make marker old, kernel file new
    marker = tmp_path / ".delegation_scan_passed"
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(exist_ok=True)
    new_kernel = kernel_dir / "kernel.h"
    new_kernel.write_text("// edited after scan ran")
    # Set marker mtime to 100s ago, kernel file to now
    old_ts = time.time() - 100
    os.utime(marker, (old_ts, old_ts))
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert "STALE" in elig["reason"]


def test_pass_with_pass_b_silent_skip_rolls_back(tmp_path):
    """precision.pass_b without explicit status → rollback (silent-skip)."""
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          full_verification=False)
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
            # pass_b absent — silent skip
        },
        "performance": {"status": "PASS", "ratio": 1.0},
    }))
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert "pass_b" in elig["reason"]


def test_pass_with_pass_b_na_no_reason_rolls_back(tmp_path):
    """precision.pass_b.status=N/A but no reason → rollback."""
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          full_verification=False)
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
            "pass_b": {"status": "N/A"},  # no reason
        },
        "performance": {"status": "PASS", "ratio": 1.0,
                        "ratio_baseline": "Path A — CPU-truth"},
    }))
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert "reason" in elig["reason"].lower()


def test_pass_with_pass_b_na_with_reason_eligible(tmp_path):
    """Path A: pass_b N/A with reason='OL-68 Case A...' → eligible."""
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          full_verification=False)
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
            "pass_b": {"status": "N/A",
                       "reason": "OL-68 Case A — torch_npu reference unrunnable"},
        },
        "performance": {"status": "PASS", "ratio": 1.0,
                        "ratio_baseline": "Path A — CPU-truth"},
        "audit": {"delegation_scan": {"violations": 0}},
    }))
    elig = fp.check_finalize_eligibility(tmp_path)
    assert elig["eligible"]


def test_pass_with_perf_no_independent_remeasure_rolls_back(tmp_path):
    """Non-Path-A perf.status=PASS without independent_re_measure → rollback.

    2026-05-27 fixup: original fixture only seeded `ratio` + `ratio_baseline`,
    which now trips P0ee METHODOLOGY_DECLARATION (a NEWER gate added after
    this test was written: "ratio > 1.0× without performance.method →
    default-deny"). To exercise this test's original intent — the
    `independent_re_measure` gate — the fixture must declare a valid
    `performance.method` so P0ee passes and the older gate fires.
    """
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          full_verification=False)
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
            "pass_b": {"status": "PASS", "tier1_pass": 10, "total": 10,
                       "method": "edge_dataset"},
        },
        "performance": {
            "status": "PASS",
            "ratio": 1.5,
            "ratio_baseline": "torch_npu.npu_X (vs CANN)",
            # Declare method so P0ee METHODOLOGY_DECLARATION + P97
            # PERF_METHODOLOGY_ASYMMETRY both pass — gap under test is
            # independent_re_measure, not methodology.
            "method": "same_wrapper symmetric=true method_symmetric",
        },
        # No independent_re_measure field — the gap under test
        "audit": {"delegation_scan": {"violations": 0}},
    }))
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert "independent_re_measure" in elig["reason"]


# ---- P0aba codex hardening (2026-05-07) ----

def test_pass_with_empty_audit_doc_rolls_back(tmp_path):
    """Codex finding: audit doc must be substantive, not just present."""
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          audit_doc=False)
    (tmp_path / "audit_self_critic_post_worker.md").write_text("# stub\n")
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert "empty" in elig["reason"].lower() or "trivial" in elig["reason"].lower()


def test_pass_with_no_block_findings_negative_statement_eligible(tmp_path):
    """Codex finding: 'No `❌ block` findings' (negative statement) under a
    PASS verdict must NOT trigger has_fail_block. Per-line scoping fixes this.
    """
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          audit_doc=False)
    (tmp_path / "audit_self_critic_post_worker.md").write_text(
        "# audit\n\n**Verdict: ✅ PASS** with 1 WARN.\n\n"
        "Detail: No `❌ block` findings on the kernel. C18 PASS, C25 PASS.\n"
    )
    elig = fp.check_finalize_eligibility(tmp_path)
    assert elig["eligible"]


def test_pass_b_pass_without_counts_rolls_back(tmp_path):
    """Codex finding: pass_b PASS without tier1_pass/total counts is a
    self-claim string. Require concrete denominator.
    """
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          full_verification=False)
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
            "pass_b": {"status": "PASS", "method": "claimed but no numbers"},
        },
        "performance": {"status": "PASS", "ratio": 1.0,
                        "ratio_baseline": "Path A — CPU-truth"},
        "audit": {"delegation_scan": {"violations": 0}},
    }))
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert "counts missing" in elig["reason"].lower() or "denominator" in elig["reason"].lower()


def test_pass_with_perf_independent_re_measure_empty_dict_rolls_back(tmp_path):
    """Codex finding: presence-only check accepts empty {} — must require contents."""
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          full_verification=False)
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
            "pass_b": {"status": "PASS", "tier1_pass": 10, "total": 10},
        },
        "performance": {"status": "PASS", "ratio": 1.5,
                        "ratio_baseline": "torch_npu (vs CANN)",
                        "independent_re_measure": {}},  # empty dict
        "audit": {"delegation_scan": {"violations": 0}},
    }))
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]


# ---- P0abd benchmark Pass A coverage gate (2026-05-07) ----

def test_pass_a_coverage_shortfall_rolls_back(tmp_path):
    """The 3_FusionAttention regression (2026-05-07): benchmark has 50 cases,
    kw ran 1, reported tier1_pass=0/total=1. Gate must read benchmark JSON
    case count and reject silent shortfall.
    """
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 1, "total": 1}})
    # Stage benchmark JSON with 50 cases in the workspace.
    bench = tmp_path / f"{tmp_path.name}.json"
    bench_lines = [json.dumps({"inputs": [{"name": "x", "shape": [4]}]})
                   for _ in range(50)]
    bench.write_text("\n".join(bench_lines) + "\n")
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert elig["rollback_state"] == "await_worker"
    assert "P0abd" in elig["reason"]
    assert "1/50" in elig["reason"] or ("total=1" in elig["reason"] and "50 cases" in elig["reason"])


def test_pass_a_coverage_full_eligible(tmp_path):
    """Full coverage (total ≥ benchmark count) → eligible."""
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}})
    bench = tmp_path / f"{tmp_path.name}.json"
    bench_lines = [json.dumps({"inputs": []}) for _ in range(50)]
    bench.write_text("\n".join(bench_lines) + "\n")
    elig = fp.check_finalize_eligibility(tmp_path)
    assert elig["eligible"]


def test_pass_a_coverage_shortfall_with_explicit_skips_eligible(tmp_path):
    """Explicit skipped_cases list with reasons → allowed."""
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {
                         "status": "PASS",
                         "tier1_pass": 47, "total": 47,
                         "skipped_cases": [
                             {"case_idx": 12, "reason": "OL-68 Case A — torch_npu rejected"},
                             {"case_idx": 23, "reason": "shape exceeds NPU UB capacity"},
                             {"case_idx": 41, "reason": "torch_npu segfaults on this combo"},
                         ],
                     }})
    bench = tmp_path / f"{tmp_path.name}.json"
    bench_lines = [json.dumps({"inputs": []}) for _ in range(50)]
    bench.write_text("\n".join(bench_lines) + "\n")
    elig = fp.check_finalize_eligibility(tmp_path)
    assert elig["eligible"]


def test_pass_a_coverage_shortfall_with_too_few_skips_rolls_back(tmp_path):
    """Skipped_cases list shorter than the actual shortfall → still gate."""
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {
                         "status": "PASS", "tier1_pass": 30, "total": 30,
                         "skipped_cases": [
                             {"case_idx": 5, "reason": "explicit reason"},
                         ],  # only 1 entry but 20 cases missing
                     }})
    bench = tmp_path / f"{tmp_path.name}.json"
    bench_lines = [json.dumps({"inputs": []}) for _ in range(50)]
    bench.write_text("\n".join(bench_lines) + "\n")
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]


def test_pass_a_coverage_skips_without_reason_rolls_back(tmp_path):
    """Skipped_cases with no reason field → still gate."""
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {
                         "status": "PASS", "tier1_pass": 49, "total": 49,
                         "skipped_cases": [{"case_idx": 7}],  # no reason
                     }})
    bench = tmp_path / f"{tmp_path.name}.json"
    bench_lines = [json.dumps({"inputs": []}) for _ in range(50)]
    bench.write_text("\n".join(bench_lines) + "\n")
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]


def test_pass_a_coverage_no_benchmark_json_skips_gate(tmp_path):
    """When workspace lacks <op>.json, gate is unverifiable → skipped (other
    gates handle the rest).
    """
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 5, "total": 5}})
    # Don't stage benchmark JSON — gate has nothing to compare against.
    elig = fp.check_finalize_eligibility(tmp_path)
    assert elig["eligible"]


def test_pass_a_coverage_partial_persist_also_gated(tmp_path):
    """PARTIAL_PERSIST handoffs ALSO need full coverage (silent skip is
    silent skip regardless of status). 3_FusionAttention PARTIAL_PERSIST
    with 1/50 was the actual incident.
    """
    _seed(tmp_path, {"status": "PARTIAL", "persist_verdict": "PARTIAL_PERSIST",
                     "pass_a": {"status": "FAIL", "tier1_pass": 0, "total": 1}},
          probe=True)
    bench = tmp_path / f"{tmp_path.name}.json"
    bench_lines = [json.dumps({"inputs": []}) for _ in range(50)]
    bench.write_text("\n".join(bench_lines) + "\n")
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert "P0abd" in elig["reason"]


def test_pass_with_perf_independent_re_measure_ran_no_ratio_rolls_back(tmp_path):
    """Codex finding: ran=true but no ratio — incomplete record."""
    _seed(tmp_path, {"status": "PASS",
                     "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
          full_verification=False)
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
            "pass_b": {"status": "PASS", "tier1_pass": 10, "total": 10},
        },
        "performance": {"status": "PASS", "ratio": 1.5,
                        "ratio_baseline": "torch_npu (vs CANN)",
                        "independent_re_measure": {"ran": True}},  # ran but no ratio
        "audit": {"delegation_scan": {"violations": 0}},
    }))
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert "ratio" in elig["reason"].lower()


def test_partial_with_persist_verdict_and_probe_eligible(tmp_path):
    """Legitimate PARTIAL_PERSIST: status=PARTIAL, persist=PARTIAL_PERSIST,
    probe_report.md present.
    """
    _seed(tmp_path, {"status": "PARTIAL", "persist_verdict": "PARTIAL_PERSIST",
                     "pass_a": {"status": "PARTIAL", "tier1_pass": 47, "total": 50}},
          probe=True)
    elig = fp.check_finalize_eligibility(tmp_path)
    assert elig["eligible"]


def test_partial_persist_without_delegation_marker_rolls_back(tmp_path):
    """DEBT-211 PARTIAL_PERSIST hole: the delegation-scan marker gate used to
    run ONLY on the PASS branch (via POST_WORKER_AUDIT). The PARTIAL_PERSIST
    branch ran only plugin extra-checks, so a host-compute delegation could
    ship via PARTIAL_PERSIST undetected. A PARTIAL_PERSIST archive whose
    delegation scan never produced a clean marker must now roll back.

    (Its sibling `test_partial_with_persist_verdict_and_probe_eligible` seeds
    the marker by default and stays eligible — so this proves the gate is live
    on the branch, not that the branch is broken.)
    """
    _seed(tmp_path, {"status": "PARTIAL", "persist_verdict": "PARTIAL_PERSIST",
                     "pass_a": {"status": "PARTIAL", "tier1_pass": 47, "total": 50}},
          probe=True, deleg_marker=False)
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"], (
        "PARTIAL_PERSIST with no .delegation_scan_passed marker was accepted — "
        "the DEBT-211 PARTIAL_PERSIST delegation hole has reopened"
    )
    assert elig["rollback_state"] == "await_worker"
    assert ".delegation_scan_passed" in elig["reason"]


def test_partial_persist_with_stale_delegation_marker_rolls_back(tmp_path):
    """Freshness also enforced on the PARTIAL_PERSIST branch: a marker older
    than a rebuilt kernel source (here in op_kernel/, a port_a3 C++ dir now
    covered by DEBT-211) must roll back.
    """
    import os
    import time
    _seed(tmp_path, {"status": "PARTIAL", "persist_verdict": "PARTIAL_PERSIST",
                     "pass_a": {"status": "PARTIAL", "tier1_pass": 47, "total": 50}},
          probe=True)
    ok = tmp_path / "op_kernel"
    ok.mkdir(exist_ok=True)
    (ok / "rebuilt.cpp").write_text("// rebuilt after the scan ran")
    marker = tmp_path / ".delegation_scan_passed"
    old_ts = time.time() - 100
    os.utime(marker, (old_ts, old_ts))
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert "STALE" in elig["reason"]


def test_partial_with_persist_verdict_no_probe_rejects(tmp_path):
    """PARTIAL_PERSIST claimed but no probe evidence -> rollback to await_probe."""
    _seed(tmp_path, {"status": "PARTIAL", "persist_verdict": "PARTIAL_PERSIST",
                     "pass_a": {"status": "PARTIAL", "tier1_pass": 47, "total": 50}})
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert elig["rollback_state"] == "await_probe"


def test_partial_no_persist_no_probe_rolls_back_to_probe(tmp_path):
    """The 14_adaptive_instance_norm_bwd / 23_hyenafft case: PARTIAL with
    no persist_verdict and no probe_report.md -> rollback to await_probe.
    """
    _seed(tmp_path, {"status": "PARTIAL",
                     "pass_a": {"status": "PARTIAL", "tier1_pass": 22, "total": 50}})
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert elig["rollback_state"] == "await_probe"


def test_partial_no_persist_with_probe_rolls_back_to_researcher(tmp_path):
    """Probe done but no researcher report -> rollback to await_researcher."""
    _seed(tmp_path, {"status": "PARTIAL",
                     "pass_a": {"status": "PARTIAL", "tier1_pass": 22, "total": 50}},
          probe=True)
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert elig["rollback_state"] == "await_researcher"


def test_partial_with_research_no_optimizer_rolls_back_to_optimizer(tmp_path):
    """Probe + researcher done but no optimizer attempt -> await_optimizer."""
    _seed(tmp_path, {"status": "PARTIAL",
                     "pass_a": {"status": "PARTIAL", "tier1_pass": 22, "total": 50}},
          probe=True, research=True)
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert elig["rollback_state"] == "await_optimizer"


def test_partial_full_pipeline_no_persist_rolls_back_to_worker(tmp_path):
    """Probe + researcher + optimizer done but no persist_verdict -> worker
    re-evaluates and explicitly emits PARTIAL_PERSIST or fixes.
    """
    _seed(tmp_path, {"status": "PARTIAL",
                     "pass_a": {"status": "PARTIAL", "tier1_pass": 47, "total": 50}},
          probe=True, research=True, opt_log=True)
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert elig["rollback_state"] == "await_worker"


def test_fail_no_persist_rolls_back(tmp_path):
    """precision.status=FAIL is treated like PARTIAL for rollback purposes."""
    _seed(tmp_path, {"status": "FAIL",
                     "pass_a": {"status": "FAIL", "tier1_pass": 0, "total": 50}})
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]


def test_missing_verification_rolls_back_to_worker(tmp_path):
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert elig["rollback_state"] == "await_worker"


def test_malformed_verification_rolls_back_to_worker(tmp_path):
    (tmp_path / "verification.json").write_text("{ not json }")
    elig = fp.check_finalize_eligibility(tmp_path)
    assert not elig["eligible"]
    assert elig["rollback_state"] == "await_worker"
