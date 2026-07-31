# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for FW transient error retry in agent_transport (P0k, Day 4 finding).

Huawei firewall occasionally returns HTTP 200 with empty/malformed body,
surfacing as the claude CLI error:
  "API Error: API returned an empty or malformed response (HTTP 200) —
   check for a proxy or gateway intercepting the request"

Retry policy: wait 120s, retry up to 3 times. Non-FW errors propagate.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import agent_transport as at  # noqa: E402


# ---------------------------------------------------------------------------
# _is_fw_transient_error
# ---------------------------------------------------------------------------
def test_fw_transient_canonical_error():
    """The exact error message reported by user."""
    msg = (
        "claude (stream-json) exited 1 for agent='aog-kernel-worker'\n"
        "stderr: API Error: API returned an empty or malformed response "
        "(HTTP 200) — check for a proxy or gateway intercepting the request"
    )
    assert getattr(at, "_is_fw_transient_error")(msg) is True


def test_fw_transient_partial_match_proxy_phrase():
    """If only the second sentinel phrase appears."""
    msg = "Error during claude execution: check for a proxy or gateway intercepting the request"
    assert getattr(at, "_is_fw_transient_error")(msg) is True


def test_fw_transient_partial_match_empty_response():
    msg = "API returned an empty or malformed response"
    assert getattr(at, "_is_fw_transient_error")(msg) is True


def test_fw_transient_unrelated_quota_error():
    """Quota errors must NOT be classified as FW-transient."""
    msg = "claude exited 1: You've hit your org's monthly usage limit"
    assert getattr(at, "_is_fw_transient_error")(msg) is False


def test_fw_transient_unrelated_real_crash():
    """Real subprocess crashes / agent infinite loops must NOT retry."""
    msg = "claude exited 137 (SIGKILL)"
    assert getattr(at, "_is_fw_transient_error")(msg) is False


def test_fw_transient_empty_input():
    assert getattr(at, "_is_fw_transient_error")("") is False


# 2026-05-27 corporate proxy transient detection (owner directive after the
# 3_FusionAttention spawn observed "API Error: corporate proxy Notification" via
# stream-json event then exit 1).
def test_fw_transient_his_proxy_notification():
    """corporate proxy Notification surfaces via progress_lines tail; the
    exception message now includes those, so detector must catch it.
    """
    msg = (
        "claude (stream-json) exited 1 for agent='aog-kernel-worker'\n"
        "stderr: \n"
        "progress_tail: [aog-kernel-worker-1] | API Error: corporate proxy Notification"
    )
    assert getattr(at, "_is_fw_transient_error")(msg) is True


def test_fw_transient_his_proxy_short_form():
    """Just the proxy-name token (defensive — future variants)."""
    assert getattr(at, "_is_fw_transient_error")("corporate proxy Notification mid-stream") is True


def test_fw_transient_api_error_prefix_only():
    """Pattern accepts the bare 'API Error' prefix as transient.
    Tradeoff: may over-match real API errors, but the alternative is
    missing corporate proxy variants. The retry cap (3) + 30s wait bounds
    the cost of a false positive.
    """
    msg = "progress_tail: API Error: <some claude internal>"
    assert getattr(at, "_is_fw_transient_error")(msg) is True


def test_fw_retry_wait_shortened():
    """2026-05-27 owner directive: corporate proxy clears in seconds, not
    minutes. Default wait reduced from 120s to 30s.
    """
    assert at.DEFAULT_FW_RETRY_WAIT_SEC == 30
    assert at.DEFAULT_FW_MAX_RETRIES == 3
    assert getattr(at, "_is_fw_transient_error")(None) is False


# ---------------------------------------------------------------------------
# _run_streaming_with_fw_retry retry behavior
# ---------------------------------------------------------------------------
def _fake_envelope(agent_type="test"):
    """Build a minimal AgentResult for fake successful runs."""
    return at.AgentResult(
        agent_type=agent_type, success=True, is_error=False,
        output_text="ok", duration_ms=1000, cost_usd=0.5,
        session_id="abc", terminal_reason="completed", raw_envelope={},
        tool_uses=[], progress_lines=[],
    )


def test_retry_returns_immediately_on_success(monkeypatch):
    """No retry needed if first attempt succeeds."""
    n_calls = []

    def fake_run_streaming(*args, **kwargs):
        n_calls.append(1)
        return _fake_envelope()
    monkeypatch.setattr(at, "_run_streaming", fake_run_streaming)

    sleeps = []
    result = getattr(at, "_run_streaming_with_fw_retry")(
        ["claude"], "test", prompt="test", tee_path=None, timeout_sec=60,
        cwd=None, progress_callback=None,
        sleep_fn=lambda s: sleeps.append(s),
    )
    assert result.success is True
    assert len(n_calls) == 1
    assert sleeps == []  # no retry, no sleep


def test_retry_succeeds_on_second_attempt(monkeypatch):
    """First attempt FW error → wait → second attempt succeeds."""
    n = [0]

    def fake_run_streaming(*args, **kwargs):
        n[0] += 1
        if n[0] == 1:
            raise at.AgentTransportError(
                "claude (stream-json) exited 1 for agent='test'\n"
                "stderr: API Error: API returned an empty or malformed response (HTTP 200)"
            )
        return _fake_envelope()
    monkeypatch.setattr(at, "_run_streaming", fake_run_streaming)

    sleeps = []
    result = getattr(at, "_run_streaming_with_fw_retry")(
        ["claude"], "test", prompt="test", tee_path=None, timeout_sec=60,
        cwd=None, progress_callback=None,
        sleep_fn=lambda s: sleeps.append(s),
    )
    assert result.success is True
    assert n[0] == 2  # retried once
    assert sleeps == [30]  # waited the default (DEFAULT_FW_RETRY_WAIT_SEC=30 after 2026-05-27 owner directive)


def test_retry_exhausts_after_max_retries(monkeypatch):
    """All FW errors → exhaust retries → propagate the last AgentTransportError."""
    n = [0]

    def fake_run_streaming(*args, **kwargs):
        n[0] += 1
        raise at.AgentTransportError(
            "API returned an empty or malformed response (HTTP 200)"
        )
    monkeypatch.setattr(at, "_run_streaming", fake_run_streaming)

    sleeps = []
    with pytest.raises(at.AgentTransportError, match="empty or malformed"):
        getattr(at, "_run_streaming_with_fw_retry")(
            ["claude"], "test", prompt="test", tee_path=None, timeout_sec=60,
            cwd=None, progress_callback=None,
            max_retries=3,
            sleep_fn=lambda s: sleeps.append(s),
        )
    # max_retries=3 means up to 4 attempts (initial + 3 retries)
    assert n[0] == 4
    # 3 sleeps (one before each retry), NOT 4 (no sleep after final failure)
    assert sleeps == [30, 30, 30]


def test_non_fw_error_propagates_immediately(monkeypatch):
    """Quota error / real crash → no retry, raise immediately."""
    n = [0]

    def fake_run_streaming(*args, **kwargs):
        n[0] += 1
        raise at.AgentTransportError(
            "claude exited 1: You've hit your org's monthly usage limit"
        )
    monkeypatch.setattr(at, "_run_streaming", fake_run_streaming)

    sleeps = []
    with pytest.raises(at.AgentTransportError, match="usage limit"):
        getattr(at, "_run_streaming_with_fw_retry")(
            ["claude"], "test", prompt="test", tee_path=None, timeout_sec=60,
            cwd=None, progress_callback=None,
            sleep_fn=lambda s: sleeps.append(s),
        )
    assert n[0] == 1  # NOT retried
    assert sleeps == []


def test_timeout_expired_propagates_no_retry(monkeypatch):
    """Wall-clock timeout → propagate, don't retry (would burn another full
    timeout cycle).
    """
    import subprocess as sp
    n = [0]

    def fake_run_streaming(*args, **kwargs):
        n[0] += 1
        raise sp.TimeoutExpired(["claude"], 60)
    monkeypatch.setattr(at, "_run_streaming", fake_run_streaming)

    with pytest.raises(sp.TimeoutExpired):
        getattr(at, "_run_streaming_with_fw_retry")(
            ["claude"], "test", prompt="test", tee_path=None, timeout_sec=60,
            cwd=None, progress_callback=None,
            sleep_fn=lambda s: None,
        )
    assert n[0] == 1


def test_custom_max_retries_and_wait(monkeypatch):
    """Test injectable retry counts + wait values (used by tests; future
    operators could tune via env var if needed).
    """
    n = [0]

    def fake_run_streaming(*args, **kwargs):
        n[0] += 1
        if n[0] < 2:
            raise at.AgentTransportError("API returned an empty or malformed response")
        return _fake_envelope()
    monkeypatch.setattr(at, "_run_streaming", fake_run_streaming)

    sleeps = []
    result = getattr(at, "_run_streaming_with_fw_retry")(
        ["claude"], "test", prompt="test", tee_path=None, timeout_sec=60,
        cwd=None, progress_callback=None,
        max_retries=1, wait_sec=5,
        sleep_fn=lambda s: sleeps.append(s),
    )
    assert result.success is True
    assert sleeps == [5]


def test_zero_max_retries_means_no_retry_at_all(monkeypatch):
    """max_retries=0 → only 1 attempt total, no retries."""
    n = [0]

    def fake_run_streaming(*args, **kwargs):
        n[0] += 1
        raise at.AgentTransportError("API returned an empty or malformed response")
    monkeypatch.setattr(at, "_run_streaming", fake_run_streaming)

    with pytest.raises(at.AgentTransportError):
        getattr(at, "_run_streaming_with_fw_retry")(
            ["claude"], "test", prompt="test", tee_path=None, timeout_sec=60,
            cwd=None, progress_callback=None,
            max_retries=0,
            sleep_fn=lambda s: None,
        )
    assert n[0] == 1
