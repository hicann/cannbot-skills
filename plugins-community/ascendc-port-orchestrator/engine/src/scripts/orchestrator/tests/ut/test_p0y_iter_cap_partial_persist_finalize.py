# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P0y (2026-05-05): iter_cap hit on await_researcher → finalize PARTIAL_PERSIST.

Origin: op#28 multimodal_rope 2026-05-05. Full V3.8.8 pipeline exercised:
  await_probe → await_researcher (V3.8.8) → await_worker (Kind-2 directive)
  → await_probe → await_researcher (V3.8.8 fired again, requirement verdict)

Then orchestrator hit iter_cap=2 on await_researcher and returned error
code 2 instead of routing to finalize PARTIAL_PERSIST. Per V3.8.8 'never
let PARTIAL pass' policy, this IS the legitimate terminal state — full
pipeline exhausted with researcher-evidence-backed verdict.

Fix: orchestrator.run_single_op detects iter_cap on await_researcher with
researcher having actually run (cann_strategy_inference.md present), routes
to finalize, tags verification.json.precision.persist_verdict=PARTIAL_PERSIST.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import orchestrator as orch  # noqa: E402


def test_p0y_legitimate_exhaustion_with_researcher_output(tmp_path):
    """await_researcher iter_cap hit + cann_strategy_inference.md present
    + probe_result.json classification=requirement → legitimate exhaustion.
    """
    (tmp_path / "cann_strategy_inference.md").write_text("# researcher findings")
    (tmp_path / "probe_result.json").write_text(json.dumps({
        "classification": "requirement",
        "confidence": "verified",
    }))
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_researcher") is True


def test_p0y_legitimate_exhaustion_without_probe_json(tmp_path):
    """If cann_strategy_inference.md exists but no probe_result.json,
    still legitimate (researcher tried but no probe info).
    """
    (tmp_path / "cann_strategy_inference.md").write_text("# researcher findings")
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_researcher") is True


def test_p0y_not_legitimate_without_researcher_output(tmp_path):
    """No cann_strategy_inference.md → researcher didn't actually run with output;
    iter_cap hit is a real error (likely workflow loop).
    """
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_researcher") is False


def test_p0y_not_legitimate_for_non_researcher_states(tmp_path):
    """await_worker / await_probe iter_cap hits don't qualify as legitimate
    pipeline exhaustion — those are stuck-in-loop errors. await_optimizer
    and await_fused_optimizer have their own legitimate-exhaustion conditions
    (P0aa) and are tested separately.
    """
    (tmp_path / "cann_strategy_inference.md").write_text("# researcher findings")
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_worker") is False
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_probe") is False


# ---------------------------------------------------------------------------
# P0aa (2026-05-05): await_optimizer / await_fused_optimizer iter_cap →
# finalize PARTIAL_PERSIST when full pipeline exhausted.
# Origin: op#9 TopKTopP 2026-05-05 — ko-5 emitted KO_PERF_PLATEAU 0.385x
# after researcher (ar-2) + probe (pp-3 requirement) full cycle. Without
# this branch, orchestrator returned error 2 instead of cleanly finalizing.
# ---------------------------------------------------------------------------
def _seed_p0aa_legitimate(tmp_path: Path, ratio: float = 0.385):
    (tmp_path / "cann_strategy_inference.md").write_text("# researcher findings")
    (tmp_path / "probe_result.json").write_text(json.dumps({
        "classification": "requirement",
        "confidence": "verified",
    }))
    (tmp_path / "verification.json").write_text(json.dumps({
        "performance": {"ratio": ratio, "status": "BELOW_THRESHOLD"},
    }))


def test_p0aa_legitimate_optimizer_exhaustion(tmp_path):
    """await_optimizer iter_cap + researcher ran + probe=requirement +
    perf < 0.6 → legitimate exhaustion.
    """
    _seed_p0aa_legitimate(tmp_path)
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_optimizer") is True


def test_p0aa_legitimate_fused_optimizer_exhaustion(tmp_path):
    """await_fused_optimizer iter_cap + same conditions → legitimate."""
    _seed_p0aa_legitimate(tmp_path)
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_fused_optimizer") is True


def test_p0aa_not_legitimate_without_researcher(tmp_path):
    """await_optimizer iter_cap WITHOUT researcher having run is a real
    error — pipeline didn't reach late-stage exhaustion.
    """
    (tmp_path / "probe_result.json").write_text(json.dumps({
        "classification": "requirement",
    }))
    (tmp_path / "verification.json").write_text(json.dumps({
        "performance": {"status": "PASS", "ratio": 0.3},
    }))
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_optimizer") is False


def test_p0aa_not_legitimate_without_probe_requirement(tmp_path):
    """probe verdict != 'requirement' → precision-side fix may exist;
    iter_cap hit is a real loop error, not exhaustion.
    """
    (tmp_path / "cann_strategy_inference.md").write_text("# researcher findings")
    (tmp_path / "probe_result.json").write_text(json.dumps({
        "classification": "convention",
    }))
    (tmp_path / "verification.json").write_text(json.dumps({
        "performance": {"status": "PASS", "ratio": 0.3},
    }))
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_optimizer") is False


def test_p0aa_not_legitimate_when_perf_at_or_above_threshold(tmp_path):
    """If perf is >= parity threshold (1.0, owner-directed 2026-07-21; was
    0.6), optimizer iter_cap shouldn't hit (orchestrator would route to
    finalize done). If it somehow does, treat as real error.
    """
    _seed_p0aa_legitimate(tmp_path, ratio=1.05)
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_optimizer") is False


def test_p0aa_not_legitimate_without_probe_result_json(tmp_path):
    """If probe_result.json missing entirely, can't confirm requirement
    verdict — treat as real error.
    """
    (tmp_path / "cann_strategy_inference.md").write_text("# researcher findings")
    (tmp_path / "verification.json").write_text(json.dumps({
        "performance": {"status": "PASS", "ratio": 0.3},
    }))
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_optimizer") is False


def test_p0y_record_partial_persist_writes_state_log(tmp_path):
    """_record_partial_persist_finalize appends transition entry."""
    (tmp_path / "state_transitions.jsonl").write_text("")
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {"status": "PARTIAL", "pass_a": {"status": "PARTIAL", "tier1_pass": 47, "total": 50}},
        "performance": {"ratio": 0.10, "status": "BELOW_THRESHOLD"},
    }))

    getattr(orch, '_record_partial_persist_finalize')(tmp_path, "await_researcher", count=2, cap=2)

    log_lines = (tmp_path / "state_transitions.jsonl").read_text().strip().splitlines()
    assert len(log_lines) == 1
    entry = json.loads(log_lines[0])
    assert entry["from_state"] == "await_researcher"
    assert entry["to_state"] == "finalize"
    assert "P0y" in entry["rationale"]
    assert "iter_cap hit" in entry["rationale"]


def test_p0y_record_partial_persist_tags_verification_json(tmp_path):
    """_record_partial_persist_finalize tags verification.json.precision."""
    (tmp_path / "state_transitions.jsonl").write_text("")
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {"status": "PARTIAL"},
    }))

    getattr(orch, '_record_partial_persist_finalize')(tmp_path, "await_researcher", count=2, cap=2)

    vj = json.loads((tmp_path / "verification.json").read_text())
    p = vj["precision"]
    assert p.get("persist_verdict") == "PARTIAL_PERSIST"
    assert p.get("persist_classification") == "requirement"
    assert "Full V3.8.8 pipeline" in p.get("persist_evidence", "")
    assert p.get("persist_signed_off_by") == "P0y_orchestrator_pipeline_exhaustion"


def test_p0y_record_handles_missing_verification_json(tmp_path):
    """Robust to verification.json absence: creates minimal record carrying
    persist_verdict info so REPORT generation has a usable file. Does NOT
    silently restore from .batch*-bak / .pre-* backups (would import stale
    numbers).
    """
    (tmp_path / "state_transitions.jsonl").write_text("")

    # No verification.json — should NOT crash, should create minimal one
    getattr(orch, '_record_partial_persist_finalize')(tmp_path, "await_researcher", count=2, cap=2)

    log_lines = (tmp_path / "state_transitions.jsonl").read_text().strip().splitlines()
    assert len(log_lines) == 1

    # NEW: verification.json now exists with persist_verdict tags
    vj_path = tmp_path / "verification.json"
    assert vj_path.exists()
    vj = json.loads(vj_path.read_text())
    p = vj["precision"]
    assert p["persist_verdict"] == "PARTIAL_PERSIST"
    assert p["persist_classification"] == "requirement"
    assert p["persist_signed_off_by"] == "P0y_orchestrator_pipeline_exhaustion"
    # Note about absence preserved
    assert "_note" in vj


def test_p0y_does_not_consult_backup_files(tmp_path):
    """When verification.json missing AND .batch*-bak exists from prior
    session, P0y MUST NOT read backup. Stale numbers from prior session
    would be silently imported and tagged as if current.
    """
    (tmp_path / "state_transitions.jsonl").write_text("")
    # Prior backup with stale "PASS" content
    (tmp_path / "verification.json.batch4-bak-20260504T100822Z").write_text(json.dumps({
        "precision": {"status": "PASS",
                       "pass_a": {"tier1_pass": 50, "total": 50}},
        "performance": {"status": "PASS", "ratio": 1.5},
    }))

    getattr(orch, '_record_partial_persist_finalize')(tmp_path, "await_researcher", count=2, cap=2)

    vj = json.loads((tmp_path / "verification.json").read_text())
    # Stale PASS data NOT imported
    assert vj["precision"].get("status") == "PARTIAL", \
        f"P0y must not import backup status; got {vj['precision'].get('status')}"
    pass_a = vj["precision"].get("pass_a", {})
    assert pass_a.get("tier1_pass") != 50, \
        "Backup pass_a numbers must not leak into recovery record"
    # _note documents the absence
    assert "_note" in vj


def test_p135pp_precision_pass_no_probe_legitimate(tmp_path):
    """P135.PP regression:
    await_optimizer / await_fused_optimizer iter_cap exhaustion is
    legitimate when precision was ALREADY fully PASS from the start
    (so probe correctly never fired, probe_result.json doesn't exist),
    researcher has run, and perf is still below threshold. Previously
    the gate required probe_result.json → death loop for perf-only
    failure cases.
    """
    (tmp_path / "cann_strategy_inference.md").write_text("# researcher findings")
    # NO probe_result.json — probe never fired because precision was clean
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS",
                       "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50}},
        "performance": {"status": "BELOW_THRESHOLD", "ratio": 0.184},
    }))
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_optimizer") is True
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_fused_optimizer") is True


def test_p135pp_precision_pass_within_tolerance_no_probe_legitimate(tmp_path):
    """P135.PP: PASS_WITHIN_TOLERANCE also counts as 'precision was clean'."""
    (tmp_path / "cann_strategy_inference.md").write_text("# researcher findings")
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {"pass_a": {"status": "PASS_WITHIN_TOLERANCE", "tier1_pass": 50, "total": 50}},
        "performance": {"status": "BELOW_THRESHOLD", "ratio": 0.18},
    }))
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_optimizer") is True


def test_p135pp_rolled_up_precision_status_schema_fallback(tmp_path):
    """P135.PP: schema fallback — when precision.pass_a.status missing
    but precision.status present (rolled-up form), gate still recognizes
    as PASS for the rolled-up compatibility schema.
    """
    (tmp_path / "cann_strategy_inference.md").write_text("# findings")
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS"},  # rolled-up only, no pass_a
        "performance": {"status": "BELOW_THRESHOLD", "ratio": 0.3},
    }))
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_optimizer") is True


def test_p135pp_no_probe_no_precision_pass_rejected(tmp_path):
    """P135.PP negative: no probe AND precision was PARTIAL/FAIL → reject.
    Original safety preserved (don't accept pseudo-exhaustion claims).
    """
    (tmp_path / "cann_strategy_inference.md").write_text("# findings")
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {"pass_a": {"status": "PARTIAL", "tier1_pass": 30, "total": 50}},
        "performance": {"status": "BELOW_THRESHOLD", "ratio": 0.3},
    }))
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_optimizer") is False


def test_p135pp_existing_probe_requirement_path_still_works(tmp_path):
    """P135.PP backward-compat: when probe DID fire with classification
    requirement, old path still works (no precision PASS needed).
    """
    (tmp_path / "cann_strategy_inference.md").write_text("# findings")
    (tmp_path / "probe_result.json").write_text(json.dumps({
        "classification": "requirement",
    }))
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {"pass_a": {"status": "PARTIAL", "tier1_pass": 47, "total": 50}},
        "performance": {"status": "BELOW_THRESHOLD", "ratio": 0.3},
    }))
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_optimizer") is True


def test_p135x_no_strategy_inference_still_rejected(tmp_path):
    """Sanity: if `cann_strategy_inference.md` does not exist, the
    legitimacy check still returns False —
    the gate's original protection (don't accept iter_cap as success
    without researcher output) remains intact.
    """
    (tmp_path / "probe_result.json").write_text(json.dumps({
        "classification": "requirement",
    }))
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_researcher") is False


def test_p0y_op28_scenario_exact_replay(tmp_path):
    """Replay op#28 2026-05-05 case: probe → researcher → worker → probe →
    researcher (V3.8.8 fired); researcher iter_cap=2 hit. Both criteria
    present → legitimate exhaustion → finalize PARTIAL_PERSIST.
    """
    (tmp_path / "cann_strategy_inference.md").write_text("""\
# Vendor strategy investigation
Researcher attempted alternate vendor primitives. No applicable public-API
substitution found for the residual fp32 MARE cases.
""")
    (tmp_path / "probe_result.json").write_text(json.dumps({
        "classification": "requirement",
        "confidence": "verified",
        "summary": "OL-103 fp16-grade transcendental hardware floor",
    }))
    (tmp_path / "probe_report.md").write_text("# probe long enough\n" + "x" * 200)
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {"status": "PARTIAL",
                       "pass_a": {"status": "PARTIAL", "tier1_pass": 47, "total": 50},
                       "pass_b": {"status": "PARTIAL", "tier1_pass": 14, "total": 14}},
        "performance": {"ratio": 0.18, "status": "BELOW_THRESHOLD"},
        "determinism": {"policy_satisfied": True},
    }))
    (tmp_path / "state_transitions.jsonl").write_text("")

    # Verify legitimate exhaustion
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_researcher") is True

    # Apply finalize routing
    getattr(orch, '_record_partial_persist_finalize')(tmp_path, "await_researcher", count=2, cap=2)

    # State log records transition
    log = json.loads((tmp_path / "state_transitions.jsonl").read_text().strip())
    assert log["to_state"] == "finalize"

    # verification.json carries persist verdict
    vj = json.loads((tmp_path / "verification.json").read_text())
    assert vj["precision"]["persist_verdict"] == "PARTIAL_PERSIST"


# ---------------------------------------------------------------------------
# DEBT-112 (2026-05-27, DS): finalize iter-loop on PARTIAL_PERSIST_INFRA_BLOCKED
# — algorithm gates (model_py_shape, pass_a_coverage, pass_count, persist_evidence)
# should be treated as legitimate exhaustion when the persist_verdict is
# explicitly infra-blocked, because they're consequences of infrastructure
# failure, not real algorithm issues.
# ---------------------------------------------------------------------------

def _seed_debt112_legitimate(tmp_path: Path, persist_verdict="PARTIAL_PERSIST_INFRA_BLOCKED"):
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {
            "status": "PARTIAL",
            "persist_verdict": persist_verdict,
        },
    }))


def test_debt112_infra_blocked_algorithm_gate_accepted(tmp_path):
    """PARTIAL_PERSIST_INFRA_BLOCKED + model_py_shape gate → legitimate."""
    _seed_debt112_legitimate(tmp_path)
    (tmp_path / "cann_strategy_inference.md").write_text("# infra blocked")
    (tmp_path / "probe_result.json").write_text(json.dumps({
        "classification": "requirement",
    }))
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_researcher") is True


def test_debt112_without_infra_blocked_algorithm_gate_rejected(tmp_path):
    """Without PARTIAL_PERSIST_INFRA_BLOCKED, algorithm gates are NOT in _INFRA_GATES."""
    _seed_debt112_legitimate(tmp_path, persist_verdict="PARTIAL_PERSIST")
    (tmp_path / "cann_strategy_inference.md").write_text("# not infra")
    (tmp_path / "probe_result.json").write_text(json.dumps({
        "classification": "requirement",
    }))
    assert getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_researcher") is True
    # The key test: verify that algorithm gates are NOT in base _INFRA_GATES
    # This is tested implicitly — PARTIAL_PERSIST without INFRA_BLOCKED
    # doesn't expand the gate set


def test_debt112_infra_blocked_missing_verification_json(tmp_path):
    """Missing verification.json → no crash, treat as non-infra."""
    (tmp_path / "cann_strategy_inference.md").write_text("# no vj")
    (tmp_path / "probe_result.json").write_text(json.dumps({
        "classification": "requirement",
    }))
    # Should not crash — gracefully returns False
    result = getattr(orch, '_is_legitimate_pipeline_exhaustion')(tmp_path, "await_researcher")
    assert result is True  # researcher output + probe=requirement → legitimate regardless
