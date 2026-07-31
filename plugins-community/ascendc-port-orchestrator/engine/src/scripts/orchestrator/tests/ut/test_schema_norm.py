# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Unit tests for schema_norm.py.

Run: python3 -m pytest src/scripts/orchestrator/tests/test_schema_norm.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import schema_norm as sn  # noqa: E402


@pytest.fixture
def ws(tmp_path):
    # P0qq (2026-05-06): seed Self-introspection so tests targeting
    # downstream precision gates aren't blocked by the introspection gate.
    (tmp_path / "PROGRESS.md").write_text(
        "# test\n\n## Self-introspection (test-fixture)\n\n"
        "### Pressure modes I felt\nP1.\n\n"
        "### Decisions I almost rationalized\nnone\n\n"
        "### Verifications I might have skipped\nnone\n\n"
        "### Confidence calibration\nprecision: HIGH\nperf: HIGH\narchitectural fit: HIGH\n"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# SAFE key alias rewrites (state_transitions.jsonl)
# ---------------------------------------------------------------------------
def test_from_to_aliases_rewritten(ws):
    """Worker wrote `from`/`to` keys → auto-rewrite to `from_state`/`to_state`."""
    log_text = json.dumps({
        "ts": "2026-05-04T01:00:00Z",
        "from": "await_worker",
        "to": "await_optimizer",
        "verdict": "perf low",
    }) + "\n"
    (ws / "state_transitions.jsonl").write_text(log_text)

    report = sn.normalize_workspace(ws, fail_strict=False)

    assert len(report.events) >= 2  # 2 SAFE rewrites: from→from_state, to→to_state
    safe_events = [e for e in report.events if e.category == "SAFE"]
    assert any(e.before == "from" and e.after == "from_state" for e in safe_events)
    assert any(e.before == "to" and e.after == "to_state" for e in safe_events)

    # Log should now have canonical keys
    new_log = (ws / "state_transitions.jsonl").read_text()
    entry = json.loads(new_log.strip())
    assert "from_state" in entry
    assert "to_state" in entry
    assert "from" not in entry
    assert "to" not in entry


# ---------------------------------------------------------------------------
# TERMINAL state alias: done → finalize
# ---------------------------------------------------------------------------
# P0dd (2026-05-05): `done` was previously a free-form alias for `finalize`.
# After P0dd, `done` is a real terminal state in YAML and must NOT be aliased.
# These tests now use `partial_persist` (still aliased) to exercise the
# evidence gate. The legacy `done` cases are kept as no-op tests since
# `done` log entries are now canonical.

def test_done_state_is_canonical_no_normalization(ws):
    """P0dd: log entries with to_state="done" are canonical and must NOT be
    rewritten. Tests `done`-as-real-terminal-state contract.
    """
    log_text = json.dumps({
        "ts": "2026-05-04T01:00:00Z",
        "from_state": "finalize",
        "to_state": "done",
        "handoff": "→ orchestrator: pipeline_done",
    }) + "\n"
    (ws / "state_transitions.jsonl").write_text(log_text)
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS", "pass_a": {"status": "PASS"}},
        "performance": {"status": "PASS", "ratio": 0.85},
    }))
    report = sn.normalize_workspace(ws, fail_strict=True)
    # No TERMINAL_AUTO events for `done`
    assert not any(
        e.category == "TERMINAL_AUTO" and e.before == "done" for e in report.events
    )
    # Log entry unchanged
    entry = json.loads((ws / "state_transitions.jsonl").read_text().strip())
    assert entry["to_state"] == "done"


def test_partial_persist_with_pass_evidence_auto_normalizes(ws):
    """`partial_persist` → `finalize` when verification.json + probe_report.md."""
    log_text = json.dumps({
        "ts": "2026-05-04T01:00:00Z",
        "from_state": "await_worker",
        "to_state": "partial_persist",
        "handoff": "→ orchestrator: PARTIAL_PERSIST",
    }) + "\n"
    (ws / "state_transitions.jsonl").write_text(log_text)
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PARTIAL"},
        "performance": {"status": "PASS", "ratio": 0.85},
    }))
    (ws / "probe_report.md").write_text("# probe\n" + "x" * 200)

    report = sn.normalize_workspace(ws, fail_strict=False)
    auto_events = [e for e in report.events if e.category == "TERMINAL_AUTO"]
    assert len(auto_events) == 1
    assert auto_events[0].before == "partial_persist"
    assert auto_events[0].after == "finalize"


def test_partial_persist_without_evidence_rejected(ws):
    """`partial_persist` without probe_report.md → REJECT."""
    log_text = json.dumps({
        "ts": "2026-05-04T01:00:00Z",
        "from_state": "await_worker",
        "to_state": "partial_persist",
        "handoff": "→ orchestrator: PARTIAL_PERSIST",
    }) + "\n"
    (ws / "state_transitions.jsonl").write_text(log_text)
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PARTIAL"},
    }))
    with pytest.raises(sn.SchemaNormalizationError):
        sn.normalize_workspace(ws, fail_strict=True)


# ---------------------------------------------------------------------------
# TERMINAL state alias: partial_persist → finalize
# ---------------------------------------------------------------------------
def test_partial_persist_with_probe_report_auto(ws):
    """partial_persist with probe_report.md present → finalize OK."""
    log_text = json.dumps({
        "ts": "2026-05-04T01:00:00Z",
        "from_state": "await_worker",
        "to_state": "partial_persist",
        "handoff": "PARTIAL_PERSIST OL-110 fail-floor",
    }) + "\n"
    (ws / "state_transitions.jsonl").write_text(log_text)
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PARTIAL"},
    }))
    (ws / "probe_report.md").write_text(
        "# Probe report\n\n## Classification\nType: requirement (OL-110 fail-floor)\n\n"
        "## Recommendation\nShip at fail-floor with Tier-2 evidence.\n"
    )

    report = sn.normalize_workspace(ws, fail_strict=True)
    auto = [e for e in report.events if e.category == "TERMINAL_AUTO"]
    assert any(e.before == "partial_persist" for e in auto)


def test_partial_persist_without_evidence_rejected(ws):
    """partial_persist with PARTIAL but NO probe_report.md and NO Tier-2 evidence → REJECT."""
    log_text = json.dumps({
        "ts": "2026-05-04T01:00:00Z",
        "from_state": "await_worker",
        "to_state": "partial_persist",
        "handoff": "PARTIAL",
    }) + "\n"
    (ws / "state_transitions.jsonl").write_text(log_text)
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PARTIAL"},
    }))
    # NO probe_report.md, NO pass_b two-tier evidence

    with pytest.raises(sn.SchemaNormalizationError):
        sn.normalize_workspace(ws, fail_strict=True)


def test_partial_persist_wrong_status_rejected(ws):
    """partial_persist requires precision.status=PARTIAL, not PASS."""
    log_text = json.dumps({
        "ts": "2026-05-04T01:00:00Z",
        "from_state": "await_worker",
        "to_state": "partial_persist",
        "handoff": "?",
    }) + "\n"
    (ws / "state_transitions.jsonl").write_text(log_text)
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS"},  # not PARTIAL
    }))

    with pytest.raises(sn.SchemaNormalizationError) as exc_info:
        sn.normalize_workspace(ws, fail_strict=True)
    assert "PASS" in str(exc_info.value) or "PARTIAL" in str(exc_info.value)


# ---------------------------------------------------------------------------
# DROP category: await_orchestrator
# ---------------------------------------------------------------------------
def test_await_orchestrator_drops_entry(ws):
    """to_state=await_orchestrator (not a YAML state) → entry dropped."""
    log_text = "\n".join([
        json.dumps({
            "ts": "2026-05-04T01:00:00Z",
            "from_state": "await_worker",
            "to_state": "await_probe",
            "handoff": "@aog-precision-probe",
            "matched_transition_index": 4,
            "rationale": "match"
        }),
        json.dumps({
            "ts": "2026-05-04T01:30:00Z",
            "from_state": "await_probe",
            "to_state": "await_orchestrator",  # invalid
            "handoff": "probe done",
        }),
    ]) + "\n"
    (ws / "state_transitions.jsonl").write_text(log_text)

    report = sn.normalize_workspace(ws, fail_strict=False)
    drops = [e for e in report.events if e.category == "DROP"]
    assert len(drops) == 1
    assert drops[0].before == "await_orchestrator"

    # The dropped entry should be gone from the log
    new_log = (ws / "state_transitions.jsonl").read_text()
    assert "await_orchestrator" not in new_log


# ---------------------------------------------------------------------------
# verification.json performance key aliases
# ---------------------------------------------------------------------------
def test_overall_speedup_aliased_to_ratio(ws):
    """Worker wrote performance.overall_speedup → mirror to performance.ratio."""
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS"},
        "performance": {"overall_speedup": 1.15, "median": 0.85},
    }))

    report = sn.normalize_workspace(ws, fail_strict=False)
    safe = [e for e in report.events if e.category == "SAFE"]
    assert any(e.before == "overall_speedup" and e.after == "ratio" for e in safe)
    assert any(e.before == "median" and e.after == "ratio_median" for e in safe)

    new = json.loads((ws / "verification.json").read_text())
    assert new["performance"]["ratio"] == 1.15
    assert new["performance"]["ratio_median"] == 0.85
    # Original keys preserved (backward compat)
    assert new["performance"]["overall_speedup"] == 1.15


# ---------------------------------------------------------------------------
# event log persistence
# ---------------------------------------------------------------------------
def test_normalization_events_logged(ws):
    log_text = json.dumps({"ts": "x", "from": "a", "to": "b"}) + "\n"
    (ws / "state_transitions.jsonl").write_text(log_text)

    sn.normalize_workspace(ws, fail_strict=False)

    event_log = ws / ".schema_normalizations.log"
    assert event_log.exists()
    entries = [json.loads(l) for l in event_log.read_text().strip().splitlines()]
    assert any(e["category"] == "SAFE" for e in entries)


def test_event_log_appends_not_overwrites(ws):
    """Multiple normalize calls APPEND to event log, not replace."""
    log_text_v1 = json.dumps({"ts": "x", "from": "a", "to": "b"}) + "\n"
    (ws / "state_transitions.jsonl").write_text(log_text_v1)
    sn.normalize_workspace(ws, fail_strict=False)

    n_after_first = len((ws / ".schema_normalizations.log").read_text().splitlines())
    assert n_after_first >= 2

    # Run again — first call already canonicalized, so this might be a no-op
    sn.normalize_workspace(ws, fail_strict=False)
    n_after_second = len((ws / ".schema_normalizations.log").read_text().splitlines())
    # Should be ≥ first (append-only)
    assert n_after_second >= n_after_first
