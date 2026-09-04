# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""A.5/A.7 backend-side unit tests for the opencode streaming path.

A.5 (2_FFN_evo abort, 2026-08-29): the provider error event
  {"type":"error","error":{"name":"UnknownError","data":{"message":
   "Unexpected server error. Check server logs for details."}}}
reached the FSM as a bare is_error with no structured status, so no retry
could trust it.  The backend now backfills Envelope.api_error_status from
STRUCTURED error-event fields only (never message text), and the FSM retry
consumer keys on that field alone.

A.7: candidate-tree stall watchdog — a chatty-but-motionless worker stream
is SIGTERMed when the candidate scope digest has not changed for
AOG_TREE_STALL_TIMEOUT_SEC.

Run: cd src/scripts/orchestrator && python3 -m pytest tests/ut/test_opencode_stream_error_status.py -v
"""
from __future__ import annotations

import json
import stat
import sys
import textwrap
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from backends import opencode_backend as ob  # noqa: E402
from backends.opencode_backend import OpencodeBackend, _TreeStallWatchdog  # noqa: E402

# ``_extract_stream_error`` is a protected static helper of OpencodeBackend.  Resolve it
# by name once here so the cases below exercise it as a plain callable.
extract_stream_error = getattr(OpencodeBackend, "_extract_stream_error")


# ---------------------------------------------------------------------------
# A.5 — _extract_stream_error (structured fields only, never text)
# ---------------------------------------------------------------------------
def test_extract_stream_error_reads_status_code_from_data():
    event = {
        "type": "error",
        "error": {"name": "APIError", "data": {"message": "boom", "statusCode": 503}},
    }
    assert extract_stream_error(event) == ("APIError", 503)


def test_extract_stream_error_accepts_alternate_status_keys():
    for container_key in ("status", "code"):
        event = {"type": "error", "error": {"name": "APIError", "data": {container_key: 429}}}
        assert extract_stream_error(event) == ("APIError", 429)
    event = {"type": "error", "error": {"name": "APIError", "status": 500, "data": {}}}
    assert extract_stream_error(event) == ("APIError", 500)


def test_extract_stream_error_text_only_error_has_no_status():
    """The exact 2_FFN_evo shape: message text must NOT become a status."""
    event = {
        "type": "error",
        "timestamp": 1787975099695,
        "sessionID": "ses_x",
        "error": {
            "name": "UnknownError",
            "data": {"message": "Unexpected server error. Check server logs for details.",
                     "ref": "err_70fe9780"},
        },
    }
    name, status = extract_stream_error(event)
    assert name == "UnknownError"
    assert status is None


def test_extract_stream_error_ignores_non_error_events():
    assert extract_stream_error({"type": "text", "part": {}}) == (None, None)
    assert extract_stream_error({"type": "step_finish"}) == (None, None)


def test_extract_stream_error_rejects_bool_and_non_numeric_status():
    event = {"type": "error", "error": {"name": "APIError", "data": {"statusCode": True}}}
    assert extract_stream_error(event) == ("APIError", None)


# ---------------------------------------------------------------------------
# A.5 — end-to-end envelope backfill through a fake opencode subprocess
# ---------------------------------------------------------------------------
def _fake_opencode(tmp_path: Path, script_body: str) -> Path:
    fake = tmp_path / "opencode"
    fake.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(script_body))
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake


def _dispatch_error_event(tmp_path: Path, event: dict):
    """Dispatch a streaming agent turn against a fake opencode that emits ``event`` once.

    The fake prints the single provider event to stdout and exits non-zero, which is
    the shape both A.5 envelope cases below need.
    """
    fake = _fake_opencode(
        tmp_path,
        "import json\n"
        "import sys\n"
        "\n"
        "sys.stdin.read()\n"
        f"print(json.dumps({event!r}), flush=True)\n"
        "sys.exit(1)\n",
    )
    backend = OpencodeBackend(opencode_bin=str(fake))
    return backend.dispatch(
        "aog-kernel-worker", "Return marker",
        kind="agent", mode="streaming", cwd=tmp_path, timeout=10,
    )


def test_streaming_envelope_backfills_api_error_status(tmp_path):
    """Provider error event WITH a structured status → Envelope.api_error_status set."""
    env = _dispatch_error_event(tmp_path, {
        "type": "error", "sessionID": "ses_api",
        "error": {"name": "APIError",
                  "data": {"message": "server exploded", "statusCode": 503}},
    })
    assert env.is_error
    assert env.api_error_status == 503
    assert env.raw_envelope["api_error_status"] == 503
    assert env.raw_envelope["error_name"] == "APIError"


def test_streaming_envelope_text_only_error_keeps_status_none(tmp_path):
    """Text-only provider error leaves Envelope.api_error_status unset (A.5 acceptance).

    An "Unexpected server error" message with NO structured status field must not
    become a status, so the FSM retry consumer cannot fire on it.
    """
    env = _dispatch_error_event(tmp_path, {
        "type": "error", "sessionID": "ses_unknown",
        "error": {"name": "UnknownError",
                  "data": {"message": "Unexpected server error. Check server logs for details.",
                           "ref": "err_abc"}},
    })
    assert env.is_error
    assert env.api_error_status is None
    assert env.raw_envelope["error_name"] == "UnknownError"


# ---------------------------------------------------------------------------
# A.7 — _TreeStallWatchdog tracker logic
# ---------------------------------------------------------------------------
def _watchdog(**kwargs):
    defaults = {"timeout_sec": 10, "check_interval_sec": 1, "workspace": Path("/nonexistent")}
    defaults.update(kwargs)
    return _TreeStallWatchdog(**defaults)


def test_tree_stall_watchdog_fires_after_unchanged_window(monkeypatch):
    watchdog = _watchdog()
    monkeypatch.setattr(_TreeStallWatchdog, "_digest", lambda self: "shaA")
    watchdog.prime(100.0)
    assert watchdog.armed
    assert watchdog.poll(100.5) is False   # inside the check interval
    assert watchdog.poll(101.5) is False   # checked, unchanged, but window not over
    assert watchdog.poll(111.0) is True    # unchanged for >= timeout_sec


def test_tree_stall_watchdog_digest_change_resets_window(monkeypatch):
    digests = iter(["shaA", "shaA", "shaB", "shaB", "shaB"])
    monkeypatch.setattr(_TreeStallWatchdog, "_digest", lambda self: next(digests))
    watchdog = _watchdog()
    watchdog.prime(100.0)  # consumes shaA (prime uses _digest directly)
    assert watchdog.poll(101.0) is False   # shaA unchanged
    assert watchdog.poll(102.0) is False   # shaB — change resets changed_at to 102
    assert watchdog.poll(111.5) is False   # only 9.5s since the change
    assert watchdog.poll(112.5) is True    # >= 10s since the change


def test_tree_stall_watchdog_failopen_on_digest_error(monkeypatch):
    def boom(self):
        raise RuntimeError("workspace gone")
    monkeypatch.setattr(_TreeStallWatchdog, "_digest", boom)
    watchdog = _watchdog()
    watchdog.prime(100.0)
    # prime() catches the failure and disarms rather than killing spawns.
    assert watchdog.armed is False
    assert watchdog.poll(1000.0) is False


def test_new_tree_stall_watchdog_only_arms_for_streaming_agents(tmp_path, monkeypatch):
    backend = OpencodeBackend(opencode_bin="/nonexistent/opencode")
    # Protected static helper — resolved by name off the live instance.
    new_watchdog = getattr(backend, "_new_tree_stall_watchdog")
    assert new_watchdog("skill", "streaming", tmp_path / "t.jsonl", 0.0) is None
    assert new_watchdog("agent", "foreground", tmp_path / "t.jsonl", 0.0) is None
    monkeypatch.setenv(ob.TREE_STALL_TIMEOUT_ENV, "0")
    assert new_watchdog("agent", "streaming", tmp_path / "t.jsonl", 0.0) is None


def test_streaming_tree_stall_raises_and_sigterms(tmp_path, monkeypatch):
    """Chatty stream + unmoved candidate tree → CandidateTreeStallTimeout."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "model_new_ascendc.py").write_text("# candidate entry\n")
    fake = _fake_opencode(
        tmp_path,
        r'''
        import json
        import sys
        import time

        sys.stdin.read()
        print(json.dumps({
            "type": "text", "sessionID": "ses_stall",
            "part": {"type": "text", "text": "still thinking"},
        }), flush=True)
        time.sleep(60)
        ''',
    )
    monkeypatch.setenv(ob.TREE_STALL_TIMEOUT_ENV, "1")
    monkeypatch.setenv(ob.TREE_STALL_CHECK_INTERVAL_ENV, "1")
    backend = OpencodeBackend(opencode_bin=str(fake))
    with pytest.raises(ob.CandidateTreeStallTimeout) as exc_info:
        backend.dispatch(
            "aog-kernel-worker", "Return marker",
            kind="agent", mode="streaming", cwd=tmp_path, timeout=60,
            silence_timeout=60,
            tee_path=workspace / ".cc_stream_log_aog-kernel-worker_1.jsonl",
        )
    assert exc_info.value.agent_type == "aog-kernel-worker"
    assert exc_info.value.stall_seconds >= 1.0
    assert exc_info.value.tree_sha256  # baseline digest captured at stream start
