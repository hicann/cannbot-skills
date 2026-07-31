# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P0t (2026-05-05): resume recovers from abort caused by P0s parsing bug.

Origin: op#10_layernorm 2026-05-05 kw-2. Worker emitted improvised wrapper
"→ orchestrator: handoff to @aog-kernel-optimizer per V3.8.4 routing".
Pre-P0s extract_canonical_handoff matched the `→ orchestrator:` prefix and
returned the wrapper, which YAML state machine couldn't route → abort.
After P0s landed, extract correctly returns the inner @aog-kernel-optimizer
form. But op#10's state log still has the abort entry from the buggy run.

This test verifies resume.diagnose() classifies as BUGGY_ABORT_RECOVERABLE
when:
- state log tail to_state == "abort"
- rationale contains "worker exited without recognized handoff"
- PROGRESS.md tail has a now-canonical handoff (post-P0s parses correctly)

And resume._replay_buggy_abort_recovery() drops the abort entry, replays
state_machine.next() with the canonical handoff, and the new transition
lands on the right next state (await_optimizer for @aog-kernel-optimizer).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))
import resume  # noqa: E402


@pytest.fixture
def ws_with_buggy_abort(tmp_path):
    """Workspace with state_log: await_worker → abort (handoff-parsing bug)
    AND PROGRESS.md tail with the worker's actual @aog-kernel-optimizer handoff.
    """
    log_lines = [
        json.dumps({
            "ts": "2026-05-05T07:24:39Z",
            "from_state": "await_researcher", "to_state": "await_worker",
            "handoff": "→ orchestrator: research_done — directive at .../optimization_directive.md",
            "matched_transition_index": 0,
            "rationale": "Researcher wrote Kind-2 directive directly",
            "iter_counts_snapshot": {},
        }),
        json.dumps({
            "ts": "2026-05-05T08:14:38Z",
            "from_state": "await_worker", "to_state": "abort",
            "handoff": (
                "→ orchestrator: handoff to @aog-kernel-optimizer per V3.8.4 routing "
                "(precision PASS + det PASS + perf 0.19× < 0.6× threshold)."
            ),
            "matched_transition_index": 18,
            "rationale": "worker exited without recognized handoff — contract violation",
            "iter_counts_snapshot": {"researcher": 1, "worker": 1},
        }),
    ]
    (tmp_path / "state_transitions.jsonl").write_text("\n".join(log_lines) + "\n")
    # PROGRESS.md tail with the canonical (post-P0s parseable) form
    (tmp_path / "PROGRESS.md").write_text("""\
# PROGRESS — test_op
Mode: backward

### [08:14] kw-2 (final exit)

@aog-kernel-optimizer — kw-2 implemented researcher Kind-2 directive at 3 single-pass sites.
HARD GATE PRESERVED + Pass B precision IMPROVED to 16/16 bit-exact. Specific tuning idea
for ko-3: batched A=K Normalize.
""")
    # verification.json — minimal but valid for state machine condition checks
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS",
                       "pass_a": {"status": "PASS", "tier1_pass": 60, "total": 60},
                       "pass_b": {"status": "PASS", "tier1_pass": 16, "total": 16}},
        "performance": {"ratio": 0.19, "status": "BELOW_THRESHOLD"},
        "determinism": {"policy_satisfied": True,
                         "n_identical_cases": 60, "n_cases_checked": 60},
    }))
    return tmp_path


def test_p0t_diagnose_classifies_buggy_abort_recoverable(ws_with_buggy_abort):
    """state log tail = abort + rationale matches handoff-violation pattern +
    PROGRESS.md has canonical form → BUGGY_ABORT_RECOVERABLE.
    """
    status = resume.diagnose("test_op", workspace=ws_with_buggy_abort)
    assert status.action == resume.ResumeAction.BUGGY_ABORT_RECOVERABLE, \
        f"Expected BUGGY_ABORT_RECOVERABLE, got {status.action}"
    assert status.last_handoff is not None
    assert status.last_handoff.startswith("@aog-kernel-optimizer")


def test_p0t_diagnose_genuine_abort_not_recoverable(tmp_path):
    """abort with different rationale (e.g. build stuck) → NOT recoverable,
    falls through to NONE_TERMINAL.
    """
    log_lines = [
        json.dumps({
            "ts": "2026-05-05T08:14:38Z",
            "from_state": "await_worker", "to_state": "abort",
            "handoff": "@orchestrator: build stuck",
            "matched_transition_index": 0,
            "rationale": "build stuck with stable signature after 3 attempts",
            "iter_counts_snapshot": {},
        }),
    ]
    (tmp_path / "state_transitions.jsonl").write_text("\n".join(log_lines) + "\n")
    (tmp_path / "PROGRESS.md").write_text("# stub\n")

    status = resume.diagnose("test_op", workspace=tmp_path)
    assert status.action == resume.ResumeAction.NONE_TERMINAL, \
        f"Genuine abort should fall to NONE_TERMINAL, got {status.action}"


def test_p0t_replay_drops_abort_and_appends_routing(ws_with_buggy_abort):
    """_replay_buggy_abort_recovery archives original log, drops abort entry,
    replays state_machine.next() with canonical handoff, appends new
    transition to log.
    """
    log_path = ws_with_buggy_abort / "state_transitions.jsonl"
    pre_lines = log_path.read_text().strip().splitlines()
    assert len(pre_lines) == 2
    assert json.loads(pre_lines[-1])["to_state"] == "abort"

    ok = getattr(resume, '_replay_buggy_abort_recovery')(ws_with_buggy_abort)
    assert ok is True

    # Original log archived
    archived = list(ws_with_buggy_abort.glob("state_transitions.jsonl.pre-p0t-recovery-*"))
    assert len(archived) == 1

    # Current log: abort dropped, new transition appended
    post_lines = log_path.read_text().strip().splitlines()
    assert len(post_lines) == 2  # original 1st entry + new replayed routing
    last = json.loads(post_lines[-1])
    assert last["from_state"] == "await_worker"
    assert last["to_state"] == "await_optimizer", \
        f"@aog-kernel-optimizer should route to await_optimizer, got {last['to_state']}"
    assert "@aog-kernel-optimizer" in last["handoff"]


def test_p0t_no_canonical_in_progress_keeps_abort(tmp_path):
    """If PROGRESS.md tail has NO canonical form (genuine wrong handoff),
    classification falls through to NONE_TERMINAL. Don't auto-recover what
    we can't parse.
    """
    log_lines = [
        json.dumps({
            "ts": "2026-05-05T08:14:38Z",
            "from_state": "await_worker", "to_state": "abort",
            "handoff": "→ orchestrator: gibberish that's not parseable",
            "matched_transition_index": 18,
            "rationale": "worker exited without recognized handoff — contract violation",
            "iter_counts_snapshot": {},
        }),
    ]
    (tmp_path / "state_transitions.jsonl").write_text("\n".join(log_lines) + "\n")
    (tmp_path / "PROGRESS.md").write_text("# stub\nNo handoff line here either.\n")

    status = resume.diagnose("test_op", workspace=tmp_path)
    # Either NONE_TERMINAL (current_state=abort terminal) or just not BUGGY_ABORT
    assert status.action != resume.ResumeAction.BUGGY_ABORT_RECOVERABLE
