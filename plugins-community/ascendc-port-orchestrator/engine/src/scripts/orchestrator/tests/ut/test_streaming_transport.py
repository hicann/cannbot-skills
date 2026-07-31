# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for stream-json transport (V2 #2 + #3, DEBT-077 Day 4).

Uses parse_stream_json_events pure function for hermetic testing
(no real claude CLI needed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import agent_transport as at  # noqa: E402


# Sample stream-json events (matches claude CLI --output-format stream-json schema)
def _line(d):
    return json.dumps(d)


def _build_stream(*events):
    return "\n".join(_line(e) for e in events) + "\n"


# ---------------------------------------------------------------------------
# parse_stream_json_events
# ---------------------------------------------------------------------------
def test_parse_empty_stream():
    final, tools, progress = at.parse_stream_json_events("")
    assert final is None
    assert tools == []
    assert progress == []


def test_parse_simple_result_only():
    text = _build_stream(
        {"type": "system", "subtype": "init"},
        {"type": "result", "subtype": "success", "result": "done", "duration_ms": 1234},
    )
    final, tools, progress = at.parse_stream_json_events(text)
    assert final is not None
    assert final["subtype"] == "success"
    assert final["result"] == "done"


def test_parse_extracts_tool_use():
    text = _build_stream(
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "tu_1", "name": "Bash",
             "input": {"command": "ls"}},
        ]}},
        {"type": "result", "subtype": "success", "result": ""},
    )
    final, tools, progress = at.parse_stream_json_events(text)
    assert len(tools) == 1
    assert tools[0]["tool_name"] == "Bash"
    assert tools[0]["input"]["command"] == "ls"


def test_parse_extracts_multiple_tool_uses():
    text = _build_stream(
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"path": "x"}},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "..."},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "tu_2", "name": "Edit", "input": {"file_path": "y"}},
        ]}},
        {"type": "result", "subtype": "success", "result": "done"},
    )
    final, tools, progress = at.parse_stream_json_events(text)
    assert len(tools) == 2
    assert [t["tool_name"] for t in tools] == ["Read", "Edit"]


def test_parse_extracts_progress_text():
    text = _build_stream(
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "starting Phase A\nKB Manifest LOADED"},
        ]}},
        {"type": "result", "subtype": "success", "result": ""},
    )
    final, tools, progress = at.parse_stream_json_events(text)
    assert "starting Phase A" in progress
    assert "KB Manifest LOADED" in progress


def test_parse_skips_malformed_lines():
    text = (
        _line({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "line1"},
        ]}}) + "\n"
        "this is not json\n"
        + _line({"type": "result", "subtype": "success", "result": ""}) + "\n"
    )
    final, tools, progress = at.parse_stream_json_events(text)
    assert final is not None
    assert "line1" in progress


def test_parse_mixed_content_block():
    text = _build_stream(
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "I will run ls now."},
            {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}},
        ]}},
        {"type": "result", "subtype": "success", "result": "done"},
    )
    final, tools, progress = at.parse_stream_json_events(text)
    assert len(tools) == 1
    assert "I will run ls now." in progress


def test_parse_result_envelope_has_full_data():
    text = _build_stream(
        {"type": "result", "subtype": "success", "result": "done",
         "duration_ms": 5000, "total_cost_usd": 1.23, "session_id": "abc",
         "permission_denials": [], "num_turns": 7,
         "stop_reason": "end_turn", "terminal_reason": "completed"},
    )
    final, _, _ = at.parse_stream_json_events(text)
    assert final["duration_ms"] == 5000
    assert final["total_cost_usd"] == 1.23
    assert final["num_turns"] == 7


# ---------------------------------------------------------------------------
# AgentResult dataclass with new fields
# ---------------------------------------------------------------------------
def test_agent_result_has_tool_uses_field():
    res = at.AgentResult(
        agent_type="test", success=True, is_error=False, output_text="",
        duration_ms=0, cost_usd=0, session_id="", terminal_reason="",
        raw_envelope={}, tool_uses=[{"tool_name": "Bash"}],
        progress_lines=["line1"],
    )
    assert res.tool_uses == [{"tool_name": "Bash"}]
    assert res.progress_lines == ["line1"]


def test_agent_result_defaults_empty_lists():
    """tool_uses + progress_lines default to None per dataclass; _parse_envelope
    converts to [].
    """
    text = _build_stream(
        {"type": "result", "subtype": "success", "result": "x"},
    )
    final, tu, pl = at.parse_stream_json_events(text)
    res = getattr(at, "_parse_envelope")("test", final, tool_uses=tu, progress_lines=pl)
    assert res.tool_uses == []
    assert res.progress_lines == []


def test_parse_envelope_coerces_none_lists():
    """Defensive: caller passes None → empty lists, not None."""
    res = getattr(at, "_parse_envelope")("test", {"subtype": "success", "result": ""},
                              tool_uses=None, progress_lines=None)
    assert res.tool_uses == []
    assert res.progress_lines == []


# ---------------------------------------------------------------------------
# tee_path forensic write — integration-style with a fake stream
# ---------------------------------------------------------------------------
def test_tee_path_writes_raw_stream(tmp_path, monkeypatch):
    """Verify _run_streaming writes raw stream-json to tee_path.

    We mock subprocess.Popen to return a controlled stream + immediate exit.
    """
    import io
    import subprocess as sp

    stream_text = _build_stream(
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "step 1"},
        ]}},
        {"type": "result", "subtype": "success", "result": "ok"},
    )

    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = io.StringIO(stream_text)
            self.stdin = io.StringIO()
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(sp, "Popen", FakePopen)
    monkeypatch.setattr(at.subprocess, "Popen", FakePopen)

    tee = tmp_path / "stream.log"
    result = getattr(at, "_run_streaming")(
        ["claude", "--print"], "test_agent",
        prompt="test", tee_path=tee, timeout_sec=60, cwd=None, progress_callback=None,
    )
    assert tee.exists()
    text = tee.read_text()
    assert "step 1" in text
    assert "result" in text  # the final event line
    assert result.success is True
    assert "step 1" in result.progress_lines


def test_progress_callback_invoked_per_event(monkeypatch):
    import io
    import subprocess as sp

    stream_text = _build_stream(
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "hi"},
        ]}},
        {"type": "result", "subtype": "success", "result": ""},
    )

    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = io.StringIO(stream_text)
            self.stdin = io.StringIO()
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(at.subprocess, "Popen", FakePopen)

    seen_types = []

    def cb(event):
        seen_types.append(event.get("type"))

    getattr(at, "_run_streaming")(["claude"], "test", prompt="test", tee_path=None, timeout_sec=60,
                      cwd=None, progress_callback=cb)
    # Each non-empty line should have triggered the callback
    assert "system" in seen_types
    assert "assistant" in seen_types
    assert "result" in seen_types


def test_streaming_raises_when_no_result_event(monkeypatch):
    """Defensive: if claude exits 0 without emitting `result`, that's a bug."""
    import io
    import subprocess as sp

    stream_text = _build_stream(
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "hi"},
        ]}},
    )

    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = io.StringIO(stream_text)
            self.stdin = io.StringIO()
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(at.subprocess, "Popen", FakePopen)
    with pytest.raises(at.AgentTransportError, match="without a final result"):
        getattr(at, "_run_streaming")(["claude"], "test", prompt="test", tee_path=None, timeout_sec=60,
                          cwd=None, progress_callback=None)


# ---------------------------------------------------------------------------
# P0m (2026-05-05): post-result-event hang from orphan child stdout fd
# ---------------------------------------------------------------------------
def test_p0m_post_result_grace_window_breaks_loop(monkeypatch):
    """Simulate orphan child holding stdout fd: after `result` event, stream
    appears to never EOF. _run_streaming must break the loop after
    RESULT_EVENT_GRACE_SEC and call killpg, returning the parsed envelope.
    """
    import io
    import time as _time

    stream_text = _build_stream(
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "doing work"},
        ]}},
        {"type": "result", "subtype": "success", "result": "done"},
    )

    class HangingStream:
        """StringIO-like: emits all events, then blocks forever on next read.
        Mimics orphan-child-fd-holding-pipe behavior.
        """

        def __init__(self, text):
            self.lines = list(io.StringIO(text))
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index < len(self.lines):
                line = self.lines[self.index]
                self.index += 1
                return line
            # After exhausting events, simulate hang by sleeping then yielding
            # nothing — but we need the iter to actually stop, so use the
            # P0m grace-window break instead. Returning StopIteration here
            # would simulate natural EOF (the bug-free case).
            # For this test, we want to verify the grace window kicks in even
            # when the iterator IS still yielding (slowly).
            _time.sleep(0.5)  # simulate slow drip
            return ""  # empty line, parser skips it

    killpg_calls = []

    def fake_killpg(pgid, sig):
        killpg_calls.append((pgid, sig))

    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = HangingStream(stream_text)
            self.stdin = io.StringIO()
            self.stderr = io.StringIO("")
            self.returncode = 0
            self.pid = 99999
            self._poll_count = 0

        def wait(self, timeout=None):
            return 0

        def poll(self):
            # Always return None — simulating "still alive" (orphan kept fd open)
            return None

    monkeypatch.setattr(at.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(at.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(at.os, "killpg", fake_killpg)
    # Compress the grace window so the test runs fast
    monkeypatch.setattr(at, "RESULT_EVENT_GRACE_SEC", 1)

    result = getattr(at, "_run_streaming")(
        ["claude"], "test_p0m",
        prompt="test",
        tee_path=None, timeout_sec=60, cwd=None, progress_callback=None,
    )
    # Got the result envelope despite "hung" stream
    assert result.success is True
    # killpg was called (process still alive after grace window)
    assert len(killpg_calls) >= 1, f"expected killpg call, got {killpg_calls}"
    # First killpg should be SIGTERM (we may also see SIGKILL escalation in finally)
    import signal as _sig
    assert killpg_calls[0][1] == _sig.SIGTERM


def test_p0aal_select_polling_yields_none_on_silence():
    """P0aal (2026-05-19): _select_polling_lines generator must yield None
    (poll-tick) when select() reports no data within poll_interval, allowing
    the consumer to evaluate grace timer + wall-clock — even when the
    subprocess goes silent post-`result` event.

    Pre-fix bug: `for raw in proc.stdout:` blocked forever inside readline()
    when no more lines arrived, so the in-loop grace check never ran.
    clipped_swiglu 2026-05-19 hung 9h.
    """
    import os as _os
    import time as _time

    # Create a real pipe; never write to write-end → read-end blocks forever
    r, w = _os.pipe()
    try:
        rf = _os.fdopen(r, "r")

        def always_alive():
            return True

        gen = getattr(at, "_select_polling_lines")(
            rf, r, always_alive,
            poll_interval=0.05,  # 50ms — test runs fast
        )

        # First call: select waits 50ms with no data, yields None (poll tick)
        t0 = _time.time()
        first = next(gen)
        elapsed = _time.time() - t0
        assert first is None, f"expected poll-tick None, got {first!r}"
        # Should have waited ~50ms not blocked forever
        assert elapsed < 1.0, f"polling took {elapsed:.2f}s — appears to be blocking"

        # Generator continues to yield None on subsequent ticks
        second = next(gen)
        assert second is None
    finally:
        try:
            _os.close(w)
        except OSError:
            pass


def test_p0aal_select_polling_yields_line_on_data():
    """When pipe has data, generator yields the line (not None)."""
    import os as _os

    r, w = _os.pipe()
    try:
        rf = _os.fdopen(r, "r")
        wf = _os.fdopen(w, "w")
        wf.write("hello world\n")
        wf.flush()

        gen = getattr(at, "_select_polling_lines")(
            rf, r, lambda: True,
            poll_interval=0.05,
        )
        # First call: select sees data ready → readline returns
        line = next(gen)
        assert line is not None
        assert "hello world" in line
    finally:
        try:
            wf.close()
        except OSError:
            pass


def test_p0aal_select_polling_returns_when_proc_dead_and_pipe_empty():
    """When proc has exited AND no buffered data, generator returns
    cleanly without infinite loop. P0aal safety: the generator must
    eventually surface EOF; otherwise the orchestrator could wait
    forever even after the subprocess died.
    """
    import os as _os

    r, w = _os.pipe()
    rf = _os.fdopen(r, "r")
    _os.close(w)  # Close write end immediately → reader sees EOF

    proc_exited = [True]  # Proc already exited before first call

    gen = getattr(at, "_select_polling_lines")(
        rf, r, lambda: not proc_exited[0],
        poll_interval=0.05,
    )

    # Drain — should yield zero or one None poll-tick then return cleanly
    out = []
    safety = 10
    for item in gen:
        out.append(item)
        safety -= 1
        if safety <= 0:
            break

    assert safety > 0, "generator did not terminate (safety limit hit)"
    assert all(x is None or x == "" for x in out), \
        f"unexpected non-tick yields: {[x for x in out if x is not None and x != '']}"


def test_p0aal_2_silence_watchdog_raises_after_timeout(monkeypatch):
    """P0aal-2 (2026-05-19): mid-work stdout-silence watchdog. When the
    subprocess emits NO stream events for STREAM_SILENCE_TIMEOUT_SEC
    AND the `result` event has not yet been seen, _run_streaming must
    SIGTERM the process group + raise StreamSilenceTimeout.

    The orchestrator main loop catches StreamSilenceTimeout distinctly
    from generic Exception to enable bounded auto-respawn (up to
    STREAM_SILENCE_RETRY_MAX).
    """
    import io as _io
    import os as _os
    import signal as _sig

    killpg_calls = []

    def fake_killpg(pgid, sig):
        killpg_calls.append((pgid, sig))

    # Real pipe — `_select_polling_lines` path. Never write to it →
    # silence-watchdog must fire.
    r, w = _os.pipe()

    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = _os.fdopen(r, "r")
            self.stdin = _io.StringIO()
            self.stderr = _io.StringIO("")
            self.returncode = 0
            self.pid = 88888

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return None  # "still alive" — keeps polling

    monkeypatch.setattr(at.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(at.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(at.os, "killpg", fake_killpg)
    # Compress silence timeout so test runs fast
    monkeypatch.setattr(at, "STREAM_SILENCE_TIMEOUT_SEC", 1)
    monkeypatch.setattr(at, "STREAM_POLL_INTERVAL_SEC", 0.2)

    try:
        with pytest.raises(at.StreamSilenceTimeout) as exc_info:
            getattr(at, "_run_streaming")(
                ["claude"], "test_p0aal_2",
                prompt="test",
                tee_path=None, timeout_sec=60, cwd=None, progress_callback=None,
            )
        e = exc_info.value
        assert e.agent_type == "test_p0aal_2"
        assert e.silent_seconds >= 0.9  # at least the silence timeout
        # killpg called as part of the silence kill
        assert any(sig == _sig.SIGTERM for pgid, sig in killpg_calls), \
            f"expected SIGTERM, got {killpg_calls}"
    finally:
        try:
            _os.close(w)
        except OSError:
            pass


def test_p0aal_2_silence_does_not_fire_post_result(monkeypatch):
    """Post-result silence is handled by RESULT_EVENT_GRACE_SEC (P0aal-1),
    NOT by the StreamSilenceTimeout watchdog. The two mechanisms must NOT
    interfere — once result is seen, silence-watchdog stops checking.
    """
    import io as _io

    killpg_calls = []

    def fake_killpg(pgid, sig):
        killpg_calls.append((pgid, sig))

    # Stream that emits `result` then iterator slow-drips empty lines
    # (HangingStream — no fileno, falls back to iterator path which
    # doesn't have silence-watchdog wired anyway, but tests the
    # contract that StreamSilenceTimeout does NOT get raised here)
    stream_text = _build_stream(
        {"type": "result", "subtype": "success", "result": "done"},
    )

    class HangingStream:
        """Yield the result event, then slowly produce empty lines."""

        def __init__(self, text):
            self.lines = list(_io.StringIO(text))
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index < len(self.lines):
                line = self.lines[self.index]
                self.index += 1
                return line
            import time as _time
            _time.sleep(0.1)
            return ""

    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = HangingStream(stream_text)
            self.stdin = _io.StringIO()
            self.stderr = _io.StringIO("")
            self.returncode = 0
            self.pid = 77777

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return None

    monkeypatch.setattr(at.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(at.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(at.os, "killpg", fake_killpg)
    monkeypatch.setattr(at, "RESULT_EVENT_GRACE_SEC", 1)
    monkeypatch.setattr(at, "STREAM_SILENCE_TIMEOUT_SEC", 100)  # large — silence shouldn't fire

    # Should return normally via RESULT_EVENT_GRACE path, NOT raise
    # StreamSilenceTimeout
    result = getattr(at, "_run_streaming")(
        ["claude"], "test_post_result",
        prompt="test",
        tee_path=None, timeout_sec=30, cwd=None, progress_callback=None,
    )
    assert result.success is True
