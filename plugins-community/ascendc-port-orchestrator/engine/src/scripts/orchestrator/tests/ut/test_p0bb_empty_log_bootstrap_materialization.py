# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0bb (2026-05-05): bootstrap canonicalization in state_machine.get_current_state.

Origin: op#5_Cumsum 2026-05-05. After resume cleared agent_died marker,
state_transitions.jsonl was empty. Orchestrator iter=0 inferred state via
P0o PROGRESS.md bootstrap (returned 'await_worker'). Worker spawned, ran
1618s, appended "→ orchestrator: done" to PROGRESS.md tail. The next
current_state() call re-bootstrapped from the NEW PROGRESS.md tail and
returned 'finalize' — different answer from the same workspace state.
state_machine.next("finalize", "done") then errored with no transition
match.

Fix (proper, not workaround): bootstrap is no longer a pure read.
get_current_state persists its inference to state_transitions.jsonl
when bootstrap fires from PROGRESS.md handoff. Subsequent calls read
the canonical log instead of re-bootstrapping. Drift class eliminated
in one place; both orchestrator and resume callers benefit without
needing per-call materialization.

The persisted chain is `init→initial_state` then
`initial_state→bootstrap_target` with the PROGRESS.md handoff. This
preserves audit trail showing what was inferred and why.

Truly fresh workspaces (empty log + no PROGRESS handoff) get NO log
entry — return initial state, no persist. Avoids spurious entries
on inspection-only calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))
import state_executor  # noqa: E402
import state_machine as _sm  # noqa: E402


def test_p0bb_fresh_workspace_no_persist(tmp_path):
    """Empty log + no PROGRESS.md handoff → return initial, no log entry.
    Inspection tools (diagnose, scan) shouldn't generate spurious entries.
    """
    (tmp_path / "PROGRESS.md").write_text("# fresh\n")
    state = state_executor.current_state(tmp_path)
    assert state == "await_worker"  # YAML phase_o4_initial_state
    log = tmp_path / "state_transitions.jsonl"
    assert (not log.exists()) or log.stat().st_size == 0


def test_p0bb_canonicalizes_when_progress_handoff_present(tmp_path):
    """Empty log + PROGRESS.md tail with canonical handoff → bootstrap
    persists the chain (init→initial AND initial→target) with the handoff.
    """
    (tmp_path / "PROGRESS.md").write_text(
        "# op\n\n## Phase\n\n→ orchestrator: done — Pass A 51/51\n"
    )
    target = state_executor.current_state(tmp_path)
    assert target != "await_worker", (
        "with a 'done' handoff, bootstrap should route past initial state"
    )

    log_path = tmp_path / "state_transitions.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2, f"expected 2-entry chain (init→initial, initial→target); got {len(lines)}"

    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["from_state"] == "init"
    assert first["to_state"] == "await_worker"  # initial
    assert second["from_state"] == "await_worker"
    assert second["to_state"] == target
    assert "→ orchestrator: done" in second["handoff"]
    assert "P0bb" in first["rationale"]
    assert "P0bb" in second["rationale"]


def test_p0bb_subsequent_reads_return_same_state(tmp_path):
    """After bootstrap canonicalization, current_state must be stable
    even if PROGRESS.md changes (the original drift bug).
    """
    (tmp_path / "PROGRESS.md").write_text(
        "# op\n\n→ orchestrator: done — Pass A 51/51\n"
    )
    first = state_executor.current_state(tmp_path)

    # Simulate worker rewriting PROGRESS.md tail (the case that broke
    # op#5_Cumsum: bootstrap re-fires, returns different target)
    (tmp_path / "PROGRESS.md").write_text(
        "# op\n\n→ orchestrator: PARTIAL_PERSIST — different evidence\n"
    )
    second = state_executor.current_state(tmp_path)

    assert first == second, (
        f"current_state must be stable across PROGRESS.md edits after "
        f"canonicalization; got {first} → {second}"
    )


def test_p0bb_log_nonempty_no_bootstrap(tmp_path):
    """If log already has entries, bootstrap doesn't fire — current_state
    just reads tail. PROGRESS.md is irrelevant.
    """
    (tmp_path / "PROGRESS.md").write_text(
        "# op\n\n→ orchestrator: done — would route to finalize via bootstrap\n"
    )
    log_path = tmp_path / "state_transitions.jsonl"
    log_path.write_text(json.dumps({
        "ts": "t", "from_state": "await_worker", "to_state": "await_probe",
        "handoff": "", "matched_transition_index": 0, "rationale": "prior",
        "iter_counts_snapshot": {},
    }) + "\n")

    state = state_executor.current_state(tmp_path)
    assert state == "await_probe"

    # Log unchanged (no bootstrap canonicalization)
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1


def test_p0bb_op5_cumsum_scenario_e2e(tmp_path):
    """Full op#5_Cumsum scenario: worker handoff in PROGRESS.md, log empty.
    First current_state call canonicalizes; subsequent reads stable.
    """
    (tmp_path / "PROGRESS.md").write_text(
        "# op#5_Cumsum\n\n## Phase E\n\n"
        "→ orchestrator: done — Pass A 51/51 PASS_T1; Pass B 10/11 PASS_T1; "
        "det 11/11; perf median 8.42×\n"
    )

    # First read: bootstrap fires
    state1 = state_executor.current_state(tmp_path)
    log_path = tmp_path / "state_transitions.jsonl"
    assert log_path.exists()
    n1 = len(log_path.read_text().strip().splitlines())

    # Second read: log non-empty, no bootstrap, same answer
    state2 = state_executor.current_state(tmp_path)
    n2 = len(log_path.read_text().strip().splitlines())

    assert state1 == state2
    assert n1 == n2, "second read should NOT add log entries"
