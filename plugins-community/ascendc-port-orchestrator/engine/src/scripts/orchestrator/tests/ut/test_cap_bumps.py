# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Test --bump-cap audited override (DEBT-077 #61).

Covers:
- _parse_bump_caps validates COUNTER:DELTA syntax
- _parse_bump_caps rejects negative / >5 deltas
- _parse_bump_caps rejects unknown counters
- _audit_bump_caps appends JSONL entry
- state_executor.iter_cap reads .cap_bumps.jsonl + sums per-counter deltas
- multiple bumps for same counter accumulate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))
import orchestrator as orch  # noqa: E402
import state_executor as se  # noqa: E402


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "PROGRESS.md").write_text("# fresh\n")
    return tmp_path


# ---------------------------------------------------------------------------
# _parse_bump_caps
# ---------------------------------------------------------------------------
def test_parse_bump_caps_single():
    assert getattr(orch, '_parse_bump_caps')(["worker:2"]) == {"worker": 2}


def test_parse_bump_caps_multiple():
    result = getattr(orch, '_parse_bump_caps')(["worker:2", "probe:1"])
    assert result == {"worker": 2, "probe": 1}


def test_parse_bump_caps_rejects_missing_colon():
    with pytest.raises(ValueError, match="expected COUNTER:DELTA"):
        getattr(orch, '_parse_bump_caps')(["worker2"])


def test_parse_bump_caps_rejects_unknown_counter():
    with pytest.raises(ValueError, match="unknown counter"):
        getattr(orch, '_parse_bump_caps')(["fake:1"])


def test_parse_bump_caps_rejects_zero_or_negative():
    with pytest.raises(ValueError, match="positive"):
        getattr(orch, '_parse_bump_caps')(["worker:0"])
    with pytest.raises(ValueError, match="positive"):
        getattr(orch, '_parse_bump_caps')(["worker:-1"])


def test_parse_bump_caps_rejects_too_large():
    """Bumps >5 should require a YAML edit, not a runtime override."""
    with pytest.raises(ValueError, match=r"> 5"):
        getattr(orch, '_parse_bump_caps')(["worker:10"])


def test_parse_bump_caps_rejects_non_int_delta():
    with pytest.raises(ValueError, match="not int"):
        getattr(orch, '_parse_bump_caps')(["worker:abc"])


def test_parse_bump_caps_empty():
    assert getattr(orch, '_parse_bump_caps')([]) == {}


# ---------------------------------------------------------------------------
# _audit_bump_caps + state_executor read
# ---------------------------------------------------------------------------
def test_audit_writes_jsonl_entry(ws):
    getattr(orch, '_audit_bump_caps')(ws, {"worker": 2})
    log = ws / ".cap_bumps.jsonl"
    assert log.exists()
    entries = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert len(entries) == 1
    e = entries[0]
    assert e["bumps"] == {"worker": 2}
    assert e["actor"] == "user_cli"
    assert "ts" in e
    assert "rationale" in e


def test_audit_appends_multiple_runs(ws):
    """Two separate orchestrator runs each leave an audit entry."""
    getattr(orch, '_audit_bump_caps')(ws, {"worker": 1})
    getattr(orch, '_audit_bump_caps')(ws, {"probe": 2})
    entries = [json.loads(l) for l in (ws / ".cap_bumps.jsonl").read_text().splitlines() if l.strip()]
    assert len(entries) == 2


def test_audit_skips_when_no_bumps(ws):
    getattr(orch, '_audit_bump_caps')(ws, {})
    assert not (ws / ".cap_bumps.jsonl").exists()


# ---------------------------------------------------------------------------
# state_executor.iter_cap reads bumps
# ---------------------------------------------------------------------------
def test_iter_cap_no_bumps_returns_yaml_value(ws):
    # await_worker iter_cap=9 in YAML
    assert se.iter_cap("await_worker", workspace=ws) == 9


def test_iter_cap_with_bumps(ws):
    """Bump worker by 2 — effective cap should be 11."""
    getattr(orch, '_audit_bump_caps')(ws, {"worker": 2})
    assert se.iter_cap("await_worker", workspace=ws) == 11


def test_iter_cap_cumulative_bumps(ws):
    """Two bumps to the same counter accumulate."""
    getattr(orch, '_audit_bump_caps')(ws, {"worker": 1})
    getattr(orch, '_audit_bump_caps')(ws, {"worker": 2})
    assert se.iter_cap("await_worker", workspace=ws) == 9 + 3


def test_iter_cap_unrelated_bump_doesnt_affect(ws):
    """Bumping probe doesn't change worker cap."""
    getattr(orch, '_audit_bump_caps')(ws, {"probe": 2})
    assert se.iter_cap("await_worker", workspace=ws) == 9


def test_iter_cap_workspace_none_returns_base(ws):
    """workspace=None → no bumps applied even if file exists."""
    getattr(orch, '_audit_bump_caps')(ws, {"worker": 5})
    assert se.iter_cap("await_worker") == 9
    assert se.iter_cap("await_worker", workspace=ws) == 14


def test_at_iter_cap_honors_bumps(ws, monkeypatch):
    """at_iter_cap uses bumped cap. Simulate: 10 worker iters in log + cap=9 base.
    Without bump: at_iter_cap=True. With bump=2: at_iter_cap=False.
    """
    # Synthesize state_transitions with 10 worker entries
    log_lines = []
    for i in range(10):
        log_lines.append(json.dumps({
            "ts": f"2026-05-04T05:00:{i:02d}Z",
            "from_state": "init", "to_state": "await_worker",
            "handoff": "", "matched_transition_index": 0, "rationale": "",
            "iter_counts_snapshot": {},
        }))
    (ws / "state_transitions.jsonl").write_text("\n".join(log_lines) + "\n")

    # Without bump: 10 >= 9 → at cap
    assert se.at_iter_cap(ws, "await_worker") is True
    # With bump=2: cap becomes 11; 10 < 11 → not at cap
    getattr(orch, '_audit_bump_caps')(ws, {"worker": 2})
    assert se.at_iter_cap(ws, "await_worker") is False
