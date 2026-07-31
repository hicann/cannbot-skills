# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P96 regression tests — infra-baseline paper-over + retry-without-cap +
port_a3 pass_b schema gates.

User directive 2026-05-15 Discord 18:06Z + 18:10Z + 19:14Z: codify the
discipline that env problems are STRUCTURAL signals, not transient
nuisances. See:
- src/skills/references/shared/ANTI_PRESSURE_PROTOCOLS.md §P9
- src/skills/aog-self-critic/SKILL.md (C-INFRA-* catalog)
- docs/baseline/environment_baseline.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import finalize_pipeline as fp


# ── C-INFRA-BASELINE-PAPER-OVER ─────────────────────────────────────────


def _seed_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "test_op"
    ws.mkdir()
    return ws


def test_no_paper_over_keywords_gate_inactive(tmp_path):
    """Clean workspace → gate doesn't fire."""
    ws = _seed_workspace(tmp_path)
    (ws / "PROGRESS.md").write_text("Worker ran 8/8 PASS, perf 2.0x.\n")
    assert getattr(fp, '_check_infra_paper_over')(ws) is None


@pytest.mark.parametrize("phrase", [
    "Replaced libophost_nn.so with built version",
    "replace libopapi.so to test dispatch",
    "Manual install of .o files into CANN tree",
    "Bypass --pkg step since gmake errored",
    "Bypassing build pipeline for verification",
    "Hand-edit binary_info_config to add entry",
    "Merge binary_info_config keys",
])
def test_paper_over_keyword_in_progress_rejected(tmp_path, phrase):
    """Each paper-over phrase triggers reject."""
    ws = _seed_workspace(tmp_path)
    (ws / "PROGRESS.md").write_text(f"# PROGRESS\n[16:30] {phrase}\n")
    result = getattr(fp, '_check_infra_paper_over')(ws)
    assert result is not None
    assert "C-INFRA-BASELINE-PAPER-OVER" in result


def test_npu_error_code_without_escalation_rejected(tmp_path):
    """NPU baseline error code in PROGRESS without INFRA_BASELINE_VIOLATED
    handoff → reject.
    """
    ws = _seed_workspace(tmp_path)
    (ws / "PROGRESS.md").write_text(
        "# PROGRESS\nDispatch returned 561103 (kernel not registered).\n"
        "Continuing with retry...\n"
    )
    result = getattr(fp, '_check_infra_paper_over')(ws)
    assert result is not None
    assert "561103" in result or "INFRA_BASELINE_VIOLATED" in result


def test_npu_error_with_escalation_accepted(tmp_path):
    """NPU error code WITH INFRA_BASELINE_VIOLATED handoff → gate inactive."""
    ws = _seed_workspace(tmp_path)
    (ws / "PROGRESS.md").write_text(
        "# PROGRESS\nDispatch returned 561103 (kernel not registered).\n"
        "→ orchestrator: await_user_decision — INFRA_BASELINE_VIOLATED "
        "<CANN install missing ascend950 binary>\n"
    )
    assert getattr(fp, '_check_infra_paper_over')(ws) is None


def test_libophost_rollback_phrase_rejected(tmp_path):
    """The 2026-05-15 gather_elements_v2 specific pattern: 'libophost rollback'."""
    ws = _seed_workspace(tmp_path)
    (ws / "PROGRESS.md").write_text(
        "Rollback libophost_nn.so after symbol mismatch discovered.\n"
    )
    result = getattr(fp, '_check_infra_paper_over')(ws)
    assert result is not None


# ── C-INFRA-RETRY-WITHOUT-CAP ───────────────────────────────────────────


def test_no_retries_gate_inactive(tmp_path):
    """Clean workspace, no retry keywords → gate inactive."""
    ws = _seed_workspace(tmp_path)
    (ws / "PROGRESS.md").write_text("Worker ran once, PASS.\n")
    assert getattr(fp, '_check_infra_retry_budget')(ws) is None


def test_low_retries_gate_inactive(tmp_path):
    """Retry counts 1-3 are within budget → gate inactive."""
    ws = _seed_workspace(tmp_path)
    (ws / "PROGRESS.md").write_text(
        "Retry 1: API hiccup\nRetry 2: same error\nRetry 3: resolved\n"
    )
    assert getattr(fp, '_check_infra_retry_budget')(ws) is None


def test_high_retries_without_tracked_counter_rejected(tmp_path):
    """Retry >= 4 with NO orchestrator-tracked counter → reject."""
    ws = _seed_workspace(tmp_path)
    (ws / "PROGRESS.md").write_text(
        "Retry 4: same error\nRetry 5: same error\nattempt #6: still failing\n"
    )
    result = getattr(fp, '_check_infra_retry_budget')(ws)
    assert result is not None
    assert "C-INFRA-RETRY-WITHOUT-CAP" in result


def test_high_retries_with_tracked_counter_and_handoff_accepted(tmp_path):
    """Retry >= 4 BUT properly tracked + escalated → gate inactive."""
    ws = _seed_workspace(tmp_path)
    (ws / "PROGRESS.md").write_text(
        "Retry 4: same error → emitting INFRA_TRANSIENT_RETRY_EXHAUSTED\n"
    )
    (ws / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 1,
        "op": "test_op",
        "transient_retry_count": 4,
    }))
    assert getattr(fp, '_check_infra_retry_budget')(ws) is None


def test_high_retries_with_tracked_counter_but_no_handoff_rejected(tmp_path):
    """Retry >= 4, tracked, but no handoff → still reject (worker should escalate)."""
    ws = _seed_workspace(tmp_path)
    (ws / "PROGRESS.md").write_text(
        "Retry 4: same error\nRetry 5: keep going\n"
    )
    (ws / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 1,
        "op": "test_op",
        "transient_retry_count": 5,
    }))
    result = getattr(fp, '_check_infra_retry_budget')(ws)
    assert result is not None


# ── C-PORT-A3-PASS-B-SCHEMA ─────────────────────────────────────────────


def test_backward_mode_port_gate_inactive(tmp_path):
    """Backward mode is not subject to the migration-only Pass-B gate."""
    ws = _seed_workspace(tmp_path)
    (ws / "run_pass_b.py").write_text("# backward verifier\n")
    vj = {"mode": "backward", "precision": {"pass_b": {"status": "PASS"}}}
    assert getattr(fp, '_check_port_a3_pass_b_schema')(ws, vj) is None


def test_port_a3_with_run_pass_b_file_rejected(tmp_path):
    """port_a3 mode + run_pass_b.py at workspace root → reject."""
    ws = _seed_workspace(tmp_path)
    (ws / "run_pass_b.py").write_text("# bad: should not exist in port_a3\n")
    vj = {"mode": "port_a3_to_a5", "precision": {"pass_b": {"status": "N/A"}}}
    result = getattr(fp, '_check_port_a3_pass_b_schema')(ws, vj)
    assert result is not None
    assert "run_pass_b.py" in result


def test_port_a3_with_pass_b_pass_status_rejected(tmp_path):
    """port_a3 mode + pass_b.status=PASS → reject (should be N/A)."""
    ws = _seed_workspace(tmp_path)
    vj = {"mode": "port_a3_to_a5", "precision": {"pass_b": {"status": "PASS"}}}
    result = getattr(fp, '_check_port_a3_pass_b_schema')(ws, vj)
    assert result is not None
    assert "PASS" in result


def test_port_a3_with_pass_b_canonical_na_accepted(tmp_path):
    """port_a3 mode + pass_b.status=N/A with canonical reason → accept."""
    ws = _seed_workspace(tmp_path)
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {
            "pass_b": {
                "status": "N/A",
                "reason": (
                    "port_a3_to_a5 mode: pass_b is subsumed by pass_a — "
                    "edge_dataset.pt['a3_outputs'] IS the truth source..."
                ),
                "method": "n/a — port_a3 mode pass_b not applicable",
            }
        },
    }
    assert getattr(fp, '_check_port_a3_pass_b_schema')(ws, vj) is None


def test_port_a3_with_benchmark_method_template_rejected(tmp_path):
    """port_a3 mode but pass_b.method references benchmark template → reject."""
    ws = _seed_workspace(tmp_path)
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {
            "pass_b": {
                "status": "N/A",
                "method": "precision_eval_two_tier.py CPU fp64 truth vs ModelNew.forward",
            }
        },
    }
    result = getattr(fp, '_check_port_a3_pass_b_schema')(ws, vj)
    assert result is not None
    assert "precision_eval_two_tier" in result


# ── GateID stability ────────────────────────────────────────────────────


def test_gate_ids_stable():
    """Lock the gate ID strings."""
    assert fp.GateID.INFRA_BASELINE_PAPER_OVER.value == "infra_baseline_paper_over"
    assert fp.GateID.INFRA_RETRY_WITHOUT_CAP.value == "infra_retry_without_cap"
    assert fp.GateID.PORT_A3_PASS_B_SCHEMA.value == "port_a3_pass_b_schema"


# ── P135.SL: gate self-loop on events.jsonl + anti-pattern sections ────


def test_p135sl_gate_skips_own_rollback_in_events_jsonl(tmp_path):
    """Gate's own rejection rationale lands in orchestrator_events.jsonl
    on finalize ROLLBACK. Next scan must NOT re-detect its own audit
    trail. foreach_neg 2026-05-18 incident: kw-7 rolled back, kw-8
    cleaned worker docs but events.jsonl line 40 still had the rationale
    → gate self-loop, orchestrator paused at await_user_decision.
    """
    # events.jsonl contains the gate's own rollback line — quotes the
    # trigger phrases back as part of the rationale.
    rollback_line = (
        '{"ts": "2026-05-18T10:38:52Z", "event": "orchestrator.transition", '
        '"data": {"from_state": "finalize", "to_state": "await_worker", '
        '"rationale": "P0ff rollback: C-INFRA-BASELINE-PAPER-OVER (P96): '
        'worker performed structural env workaround instead of escalating '
        'to preflight. Violations: workspace docs contain paper-over '
        'keywords [(\'knowledge_update.md\', \'replacing libophost\')]"}, '
        '"source": "orchestrator"}'
    )
    (tmp_path / "orchestrator_events.jsonl").write_text(
        '{"ts": "2026-05-18T10:00Z", "event": "orchestrator.start", "data": {}, "source": "orchestrator"}\n'
        + rollback_line + "\n"
    )
    # Worker docs are clean (post-kw-8 cleanup state)
    (tmp_path / "knowledge_update.md").write_text("# kw-8 clean output\n\nNo paper-over content.\n")
    result = getattr(fp, '_check_infra_paper_over')(tmp_path)
    assert result is None, (
        f"Gate should skip its own audit-trail in events.jsonl, "
        f"but returned: {result}"
    )


def test_p135sl_gate_skips_anti_patterns_section_in_knowledge_update(tmp_path):
    """knowledge_update.md §Anti-patterns avoided legitimately names the
    anti-pattern (to say 'we did NOT do this') as part of generalizing
    the lesson. Gate must NOT trigger on documentation that mentions
    the trigger by reference in an anti-pattern section.
    """
    (tmp_path / "knowledge_update.md").write_text(
        "# Findings\n\nKernel correctly handles non-aligned tail via DataCopyPad.\n\n"
        "## Anti-patterns avoided\n\n"
        "- Did NOT use 'replacing libophost' to paper over the build error.\n"
        "- Did NOT bypass --pkg or hand-edit binary_info_config.json.\n\n"
        "## Cited KB items\n\n"
        "- OL-167 DataCopy alignment principle\n"
    )
    result = getattr(fp, '_check_infra_paper_over')(tmp_path)
    assert result is None, (
        f"Gate should skip §Anti-patterns sections, but returned: {result}"
    )


def test_p135sl_gate_still_catches_real_paper_over_outside_filtered_sections(tmp_path):
    """Sanity: filters don't disable the gate. Real paper-over content in
    PROGRESS.md or in non-anti-pattern sections of knowledge_update.md
    still triggers.
    """
    # Real paper-over in PROGRESS.md (no filter applies)
    (tmp_path / "PROGRESS.md").write_text(
        "iter 3 build error: replacing libophost_nn.so resolved it"
    )
    result = getattr(fp, '_check_infra_paper_over')(tmp_path)
    assert result is not None, "Gate must still catch real paper-over in PROGRESS.md"
    assert "paper-over" in result.lower() or "PAPER-OVER" in result


def test_p135sl_gate_catches_paper_over_outside_anti_pattern_section(tmp_path):
    """Real paper-over content in knowledge_update.md ## Findings section
    (NOT under §Anti-patterns) still triggers.
    """
    (tmp_path / "knowledge_update.md").write_text(
        "# Findings\n\nFixed build by replacing libophost_nn.so manually.\n\n"
        "## Anti-patterns avoided\n\n- N/A.\n"
    )
    result = getattr(fp, '_check_infra_paper_over')(tmp_path)
    assert result is not None, "Real paper-over outside anti-pattern section must still trigger"


def test_p135sl_gate_catches_paper_over_in_non_rollback_events(tmp_path):
    """Sanity: events.jsonl entries that are NOT the gate's own audit
    trail (e.g., legitimate worker emit lines) still get scanned for
    paper-over content.
    """
    (tmp_path / "orchestrator_events.jsonl").write_text(
        '{"ts": "2026-05-18T10:00Z", "event": "worker.action", "data": '
        '{"description": "Replacing libophost manually to fix build"}, '
        '"source": "worker"}\n'
    )
    result = getattr(fp, '_check_infra_paper_over')(tmp_path)
    assert result is not None, (
        "Non-rollback events.jsonl entries with real paper-over content "
        "must still trigger the gate"
    )


# ── P135.VC (2026-05-18 task #21): pass_b coverage silent-skip gate ──


def _seed_pass_b_artifacts(ws: Path):
    """Helper: create pass_b_runner.py + edge_dataset.pt in workspace."""
    (ws / "pass_b_runner.py").write_text("# placeholder runner")
    (ws / "edge_dataset.pt").write_text("# placeholder dataset (bytes)")


def test_p135vc_pass_b_artifacts_present_but_pass_b_none_rejected(tmp_path):
    """Workspace has pass_b_runner.py + edge_dataset.pt but
    verification.json.precision.pass_b is None → REJECT (silent skip).
    """
    _seed_pass_b_artifacts(tmp_path)
    vj = {
        "mode": "backward",
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 15, "total": 15},
            # pass_b absent
        },
    }
    result = getattr(fp, '_check_pass_b_coverage')(tmp_path, vj)
    assert result is not None
    assert "P135.VC" in result or "PASS_B_COVERAGE_SILENT_SKIP" in result


def test_p135vc_pass_b_artifacts_present_but_pass_b_empty_dict_rejected(tmp_path):
    """pass_b is empty dict {} (no status, no counts) → REJECT."""
    _seed_pass_b_artifacts(tmp_path)
    vj = {
        "mode": "backward",
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 15, "total": 15},
            "pass_b": {},  # empty
        },
    }
    result = getattr(fp, '_check_pass_b_coverage')(tmp_path, vj)
    assert result is not None


def test_p135vc_pass_b_status_pass_accepted(tmp_path):
    """pass_b.status='PASS' → accept (worker ran the runner)."""
    _seed_pass_b_artifacts(tmp_path)
    vj = {
        "mode": "backward",
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 15, "total": 15},
            "pass_b": {"status": "PASS", "tier1_pass": 138, "total": 138},
        },
    }
    assert getattr(fp, '_check_pass_b_coverage')(tmp_path, vj) is None


def test_p135vc_pass_b_status_na_with_reason_accepted(tmp_path):
    """pass_b.status='N/A' is a deliberate claim, not silent-skip — accept."""
    _seed_pass_b_artifacts(tmp_path)
    vj = {
        "mode": "backward",
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS"},
            "pass_b": {"status": "N/A", "reason": "runner-fundamentally-blocked"},
        },
    }
    assert getattr(fp, '_check_pass_b_coverage')(tmp_path, vj) is None


def test_p135vc_pass_b_counts_only_accepted(tmp_path):
    """pass_b has tier1_pass + total but no status — counts are evidence
    the runner DID run. Accept (other gates audit the count consistency).
    """
    _seed_pass_b_artifacts(tmp_path)
    vj = {
        "mode": "backward",
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS"},
            "pass_b": {"tier1_pass": 100, "total": 138},
        },
    }
    assert getattr(fp, '_check_pass_b_coverage')(tmp_path, vj) is None


def test_p135vc_port_a3_mode_bypasses_gate(tmp_path):
    """port_a3_to_a5 mode pass_b is degenerate by design — gate doesn't fire
    even when pass_b_runner.py + edge_dataset.pt present (handled by sibling
    port_a3 pass_b schema gate).
    """
    _seed_pass_b_artifacts(tmp_path)
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS"},
            # pass_b absent — but port_a3 mode bypasses this gate
        },
    }
    assert getattr(fp, '_check_pass_b_coverage')(tmp_path, vj) is None


def test_p135vc_no_pass_b_runner_bypasses_gate(tmp_path):
    """No pass_b_runner.py in workspace → nothing to enforce. Different gap
    class (missing runner) than this gate handles.
    """
    # Only edge_dataset, no runner
    (tmp_path / "edge_dataset.pt").write_text("# placeholder")
    vj = {
        "mode": "backward",
        "precision": {"status": "PASS", "pass_a": {"status": "PASS"}},
    }
    assert getattr(fp, '_check_pass_b_coverage')(tmp_path, vj) is None


def test_p135vc_no_edge_dataset_bypasses_gate(tmp_path):
    """No edge_dataset.pt → nothing to verify against. Bypass."""
    (tmp_path / "pass_b_runner.py").write_text("# placeholder")
    vj = {
        "mode": "backward",
        "precision": {"status": "PASS", "pass_a": {"status": "PASS"}},
    }
    assert getattr(fp, '_check_pass_b_coverage')(tmp_path, vj) is None


def test_p135vc_gate_id_stable():
    """Lock GateID enum value."""
    assert fp.GateID.PASS_B_COVERAGE_SILENT_SKIP.value == "pass_b_coverage_silent_skip"
