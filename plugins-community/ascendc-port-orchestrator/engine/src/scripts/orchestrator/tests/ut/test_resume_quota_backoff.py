# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression: resume.py classifies quota/usage-limit failures as
QUOTA_BACKOFF (no auto-retry), not FW-transient (immediate retry).

Empirical anchor: 2026-05-18 cold-start of 1_GELU / 5_Cumsum / 6_Histc
each hit `corporate proxy Notification` on a ko or precision-probe spawn.
The pre-fix resume.py classified them as FW-transient (the broad
"claude (stream-json) exited 1" pattern matched) and immediately
re-invoked the orchestrator — which spawned again, hit the same
upstream quota, exhausted the retry budget within seconds, and
surfaced AGENT_DIED with no real progress.

Fix: QUOTA_BACKOFF classification BEFORE FW-transient check. A quota
error gets surfaced to the user with the explicit "wait for quota
reset, then re-invoke manually" advice; no auto-retry.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import resume  # noqa: E402


def _make_workspace_with_died_marker(tmp_path, reason: str, state: str = "await_optimizer") -> pathlib.Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "PROGRESS.md").write_text("# fresh\n")
    (ws / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 1,
        "op": "test_op",
        "opgen_mode": "backward",
    }))
    # Minimal state_transitions.jsonl so state_executor can read current state
    (ws / "state_transitions.jsonl").write_text(
        json.dumps({"ts": "2026-05-18T00:00:00Z", "from_state": "await_worker",
                    "to_state": state, "handoff": "test"}) + "\n"
    )
    # Place the died marker with the given reason
    marker = ws / f".agent_died_at_{state}"
    marker.write_text(json.dumps({
        "ts": "2026-05-18T00:00:01Z",
        "state": state,
        "reason": reason,
    }))
    return ws


def test_his_proxy_notification_classified_as_quota_backoff(tmp_path) -> None:
    """The empirical 2026-05-18 1_GELU/5_Cumsum/6_Histc failure mode."""
    ws = _make_workspace_with_died_marker(
        tmp_path,
        reason="claude (stream-json) exited 1; stdout last line: API Error: corporate proxy Notification",
    )
    status = resume.diagnose("test_op", workspace=ws)
    assert status.action == resume.ResumeAction.QUOTA_BACKOFF, (
        f"corporate proxy Notification must be QUOTA_BACKOFF, got {status.action}"
    )
    assert "quota" in status.summary.lower() or "usage limit" in status.summary.lower()


def test_monthly_usage_limit_classified_as_quota_backoff(tmp_path) -> None:
    ws = _make_workspace_with_died_marker(
        tmp_path,
        reason="claude --print exited 1; tail: You've hit your org's monthly usage limit",
    )
    status = resume.diagnose("test_op", workspace=ws)
    assert status.action == resume.ResumeAction.QUOTA_BACKOFF


def test_rate_limit_classified_as_quota_backoff(tmp_path) -> None:
    ws = _make_workspace_with_died_marker(
        tmp_path,
        reason="HTTP 429 rate limit exceeded",
    )
    status = resume.diagnose("test_op", workspace=ws)
    assert status.action == resume.ResumeAction.QUOTA_BACKOFF


def test_fw_transient_not_quota_still_classified_as_auto_recoverable(tmp_path) -> None:
    """Pre-fix behavior preserved for actual FW-transient (no quota markers)."""
    ws = _make_workspace_with_died_marker(
        tmp_path,
        reason="API returned an empty or malformed response",
    )
    status = resume.diagnose("test_op", workspace=ws)
    assert status.action == resume.ResumeAction.AUTO_RECOVERABLE, (
        f"genuine FW-transient must stay AUTO_RECOVERABLE, got {status.action}"
    )


def test_unknown_failure_still_classified_as_agent_died(tmp_path) -> None:
    """Pre-fix behavior preserved: unrecognized failure → AGENT_DIED."""
    ws = _make_workspace_with_died_marker(
        tmp_path,
        reason="some completely novel error mode we haven't seen before",
    )
    status = resume.diagnose("test_op", workspace=ws)
    assert status.action == resume.ResumeAction.AGENT_DIED


def test_quota_takes_precedence_over_fw_transient(tmp_path) -> None:
    """Hybrid message containing BOTH `claude (stream-json) exited 1`
    AND `corporate proxy Notification` → must classify as QUOTA_BACKOFF (more
    specific), NOT as FW-transient (broader, would auto-retry into
    the quota wall).
    """
    ws = _make_workspace_with_died_marker(
        tmp_path,
        # Real-world wrapper-level message includes both substrings.
        reason="claude (stream-json) exited 1; tail: corporate proxy Notification — request rejected upstream",
    )
    status = resume.diagnose("test_op", workspace=ws)
    assert status.action == resume.ResumeAction.QUOTA_BACKOFF, (
        f"quota-pattern hybrid message must classify QUOTA_BACKOFF, "
        f"got {status.action}"
    )


def test_execute_quota_backoff_exits_without_reinvoke(tmp_path, caplog) -> None:
    """QUOTA_BACKOFF must NOT re-invoke the orchestrator subprocess.
    Returns non-zero exit code to signal the orchestrator should pause.

    2026-05-27 (zero-UT-failure rule): switched from capsys to caplog after
    the logging refactor (`ec913af0`) converted resume.py print() calls to
    log.info(). resume.py uses the `a5_orchestrator.resume` logger
    (propagate=False), so we explicitly attach caplog's handler to the root
    namespace.
    """
    import logging

    ws = _make_workspace_with_died_marker(
        tmp_path,
        reason="corporate proxy Notification",
    )
    _capture_logger = logging.getLogger("a5_orchestrator")
    _capture_logger.addHandler(caplog.handler)
    _capture_logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="a5_orchestrator"):
        rc = resume.execute("test_op", workspace=ws, dry_run=False)
    assert rc == 2, "QUOTA_BACKOFF should exit non-zero so user notices"
    assert (
        "quota_backoff" in caplog.text.lower() or "QUOTA_BACKOFF" in caplog.text
    )


def test_quota_pattern_set_complete() -> None:
    """The _QUOTA_PATTERNS tuple must contain the empirically-observed
    quota signatures. Needles are lowercase (matched case-insensitively via
    _is_quota_reason) so a capitalized marker reason is not missed.
    """
    pats = getattr(resume, '_QUOTA_PATTERNS')
    assert all(p == p.lower() for p in pats), "needles must be lowercase (case-insensitive match)"
    assert "corporate proxy notification" in pats
    assert "you've hit your org's monthly usage limit" in pats
    assert "session limit" in pats  # back issue #2 (2026-07-17)
    assert any("rate" in p for p in pats)


# --- back issue #2 (2026-07-17): 5-hour SESSION limit was misclassified ----
# The session limit surfaces as "You've hit your session limit · resets <t>"
# wrapped inside a generic "claude (stream-json) exited 1" reason. Before the
# fix, "session limit" was not a _QUOTA_PATTERN, so it matched the FW-transient
# branch -> AUTO_RECOVERABLE -> auto-retried into the same limit -> burned
# FW_AUTO_RETRY_CAP -> escalated to AGENT_DIED "nothing recoverable". Real
# symptom (back account-05), wrong mechanism (back's report said the marker
# archive lost the recovery signal; the classify tree proves non-terminal +
# no-marker -> RESUMABLE, so that was not the cause).

_SESSION_LIMIT_REASON = (
    "claude (stream-json) exited 1 for agent='aog-kernel-worker'\n"
    "stderr: \nprogress_tail: Now let's run the build.\n"
    "You've hit your session limit · resets 9:40am (America/Vancouver)"
)


def test_session_limit_classified_as_quota_backoff(tmp_path) -> None:
    ws = _make_workspace_with_died_marker(tmp_path, reason=_SESSION_LIMIT_REASON)
    status = resume.diagnose("test_op", workspace=ws)
    assert status.action == resume.ResumeAction.QUOTA_BACKOFF, (
        f"session-limit death must be QUOTA_BACKOFF (not FW-transient auto-retry), "
        f"got {status.action}"
    )


def test_session_limit_budget_not_burned(tmp_path) -> None:
    """G-a (main hard gate): behavior-correct, not just classification-correct.
    A session-limit resume must NOT consume the FW auto-retry budget — the
    QUOTA_BACKOFF path returns before _increment_retry_count, so the retry
    counter for the state stays 0 (unlike the mis-classified auto-retry that
    burned the budget).
    """
    ws = _make_workspace_with_died_marker(tmp_path, reason=_SESSION_LIMIT_REASON)
    rc = resume.execute("test_op", workspace=ws, dry_run=False)
    assert rc == 2, "QUOTA_BACKOFF should exit non-zero so the operator notices"
    assert getattr(resume, '_get_retry_count')(ws, "await_optimizer") == 0, (
        "session-limit QUOTA_BACKOFF must not burn the FW retry budget"
    )


def test_session_limit_quota_backoff_surfaces_reset(tmp_path) -> None:
    """G-b (main hard gate): QUOTA_BACKOFF must surface an explicit
    wait-for-reset action, not silently wait forever.
    """
    ws = _make_workspace_with_died_marker(tmp_path, reason=_SESSION_LIMIT_REASON)
    status = resume.diagnose("test_op", workspace=ws)
    low = status.summary.lower()
    assert "reset" in low and ("re-invoke" in low or "wait" in low), (
        f"QUOTA_BACKOFF summary must tell the operator to wait for reset + "
        f"re-invoke; got: {status.summary}"
    )


def test_quota_match_is_case_insensitive(tmp_path) -> None:
    """The pre-existing needles are lowercase; a capitalized marker reason
    (as providers actually emit) must still classify QUOTA_BACKOFF.
    """
    ws = _make_workspace_with_died_marker(
        tmp_path, reason="You've Hit Your Monthly Usage Limit")
    status = resume.diagnose("test_op", workspace=ws)
    assert status.action == resume.ResumeAction.QUOTA_BACKOFF


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
