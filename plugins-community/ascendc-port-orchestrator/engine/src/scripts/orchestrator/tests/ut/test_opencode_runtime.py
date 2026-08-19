# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""UT for backends/opencode_runtime.py (G5 self-check barrier + G6 process-group cleanup).

No live opencode binary: subprocess.run is monkeypatched. Process-cleanup tests only
assert against the pgid this test itself fabricates (G6 UT scope rule — no global
process scanning).
"""
from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import backends.opencode_runtime as ort

parse_version_token = getattr(ort, "_parse_version_token")
version_ge = getattr(ort, "_version_ge")


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    ort.reset_runtime_state()
    monkeypatch.delenv("AOG_OPENCODE_MIN_VERSION", raising=False)
    yield
    ort.reset_runtime_state()


# ---- version parsing (G5/B9: numeric-segment compare, prerelease truncation) ------------

def test_parse_version_token_plain():
    assert parse_version_token("1.18.18") == (1, 18, 18)
    assert parse_version_token("v1.18.9") == (1, 18, 9)
    assert parse_version_token("1.18") == (1, 18)
    assert parse_version_token("0.11.0") == (0, 11, 0)


def test_parse_version_token_prerelease_truncated():
    assert parse_version_token("1.18.18-beta.3") == (1, 18, 18)
    assert parse_version_token("1.18.18+build7") == (1, 18, 18)


def test_parse_version_token_garbage():
    assert parse_version_token("") is None
    assert parse_version_token("dev") is None


def test_version_ge_numeric_not_string():
    # The B9 anchor: string compare would call 1.18.9 >= 1.18.18; numeric must not.
    assert version_ge((1, 18, 9), (1, 18, 18)) is False
    assert version_ge((1, 18, 18), (1, 18, 18)) is True
    assert version_ge((1, 19, 0), (1, 18, 18)) is True
    assert version_ge((2, 0, 0), (1, 18, 18)) is True


# ---- self-check outcomes (fail-closed) ----------------------------------------------------

def _fake_run(responses, calls):
    def fake(cmd, **kw):
        calls.append(list(cmd))
        kind = "version" if "--version" in cmd else "probe"
        resp = responses[kind]
        if isinstance(resp, Exception):
            raise resp
        return SimpleNamespace(returncode=resp[0], stdout=resp[1], stderr="")
    return fake


def test_runtime_check_ok_memoized(monkeypatch):
    calls = []
    monkeypatch.setattr(ort.subprocess, "run", _fake_run(
        {"version": (0, "1.18.18"), "probe": (0, "OK")}, calls))
    monkeypatch.setattr(ort, "pick_js_runtime", lambda: "node")
    r1 = ort.ensure_opencode_runtime("opencode")
    assert r1.ok and r1.version == (1, 18, 18)
    r2 = ort.ensure_opencode_runtime("opencode")
    assert r2 is r1  # memoized: second call must not re-probe
    assert len(calls) == 2


def test_runtime_check_old_version_warns_and_is_memoized(monkeypatch):
    calls = []
    monkeypatch.setattr(ort.subprocess, "run", _fake_run(
        {"version": (0, "1.18.9"), "probe": (0, "OK")}, calls))
    monkeypatch.setattr(ort, "pick_js_runtime", lambda: "node")
    r1 = ort.ensure_opencode_runtime("opencode")
    assert r1.ok is True
    assert any("below the recommended" in warning for warning in r1.warnings)
    r2 = ort.ensure_opencode_runtime("opencode")
    assert r2 is r1
    assert len(calls) == 2  # version advisory does not skip the real safety-net probe


def test_runtime_check_probe_fail_refuses(monkeypatch):
    monkeypatch.setattr(ort.subprocess, "run", _fake_run(
        {"version": (0, "1.18.18"), "probe": (1, "FAIL: door open")}, []))
    monkeypatch.setattr(ort, "pick_js_runtime", lambda: "node")
    r = ort.ensure_opencode_runtime("opencode")
    assert r.ok is False and "door open" in r.reason


def test_runtime_check_probe_skip_warns_not_refuses(monkeypatch):
    monkeypatch.setattr(ort.subprocess, "run", _fake_run(
        {"version": (0, "1.18.18"), "probe": (2, "SKIP: no symlink perm")}, []))
    monkeypatch.setattr(ort, "pick_js_runtime", lambda: "node")
    r = ort.ensure_opencode_runtime("opencode")
    assert r.ok is True and any("SKIP" in w for w in r.warnings)


@pytest.mark.parametrize("probe_result", [
    (1, "OK"),
    (0, "SKIP: no symlink perm"),
    (2, "OK"),
    (2, "FAIL: door open"),
])
def test_runtime_check_rejects_mismatched_probe_exit_and_output(monkeypatch, probe_result):
    """Installer and runtime share one narrow `(rc, output)` acceptance table."""
    monkeypatch.setattr(ort.subprocess, "run", _fake_run(
        {"version": (0, "1.18.18"), "probe": probe_result}, []))
    monkeypatch.setattr(ort, "pick_js_runtime", lambda: "node")
    r = ort.ensure_opencode_runtime("opencode")
    assert r.ok is False
    assert "safety-net probe failed" in r.reason


def test_runtime_check_without_js_runtime_refuses(monkeypatch):
    monkeypatch.setattr(ort.subprocess, "run", _fake_run(
        {"version": (0, "1.18.18")}, []))
    monkeypatch.setattr(ort, "pick_js_runtime", lambda: None)
    r = ort.ensure_opencode_runtime("opencode")
    assert r.ok is False
    assert "node/bun" in r.reason


def test_runtime_check_version_timeout_warns_if_safety_net_passes(monkeypatch):
    monkeypatch.setattr(ort.subprocess, "run", _fake_run(
        {
            "version": subprocess.TimeoutExpired(["opencode", "--version"], 60),
            "probe": (0, "OK"),
        }, []))
    monkeypatch.setattr(ort, "pick_js_runtime", lambda: "node")
    r = ort.ensure_opencode_runtime("opencode")
    assert r.ok is True
    assert any("timed out" in warning and "advisory" in warning for warning in r.warnings)


def test_runtime_check_min_version_env_override(monkeypatch):
    monkeypatch.setenv("AOG_OPENCODE_MIN_VERSION", "1.18.9")
    monkeypatch.setattr(ort.subprocess, "run", _fake_run(
        {"version": (0, "1.18.9"), "probe": (0, "OK")}, []))
    monkeypatch.setattr(ort, "pick_js_runtime", lambda: "node")
    r = ort.ensure_opencode_runtime("opencode")
    assert r.ok is True and r.version == (1, 18, 9)


def test_concurrent_dispatch_single_probe(monkeypatch):
    calls = []
    lock = threading.Lock()

    def slow_fake(cmd, **kw):
        with lock:
            calls.append(list(cmd))
        time.sleep(0.05)
        if "--version" in cmd:
            return SimpleNamespace(returncode=0, stdout="1.18.18", stderr="")
        return SimpleNamespace(returncode=0, stdout="OK", stderr="")
    monkeypatch.setattr(ort.subprocess, "run", slow_fake)
    monkeypatch.setattr(ort, "pick_js_runtime", lambda: "node")
    results = []
    threads = [threading.Thread(target=lambda: results.append(
        ort.ensure_opencode_runtime("opencode"))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r.ok for r in results)
    assert len(calls) == 2  # exactly one probe round despite 8 racers


# ---- js runtime rule (node → bun, shared with init.sh) ------------------------------------

def test_pick_js_runtime_node_preferred(monkeypatch):
    monkeypatch.setattr(ort.shutil, "which", lambda name: "/bin/node" if name == "node" else None)
    assert ort.pick_js_runtime() == "/bin/node"


def test_pick_js_runtime_bun_fallback(monkeypatch):
    monkeypatch.setattr(ort.shutil, "which", lambda name: "/bin/bun" if name == "bun" else None)
    assert ort.pick_js_runtime() == "/bin/bun"


# ---- process-group cleanup (G6) -----------------------------------------------------------

class _FakeProc:
    def __init__(self, pid, poll=None, wait_raise=None):
        self.pid = pid
        self._poll = poll
        self._wait_raise = wait_raise
        self.waited = []

    def poll(self):
        return self._poll() if callable(self._poll) else self._poll

    def wait(self, timeout=None):
        self.waited.append(timeout)
        if self._wait_raise is not None:
            raise self._wait_raise
        return 0


def test_terminate_group_killpg_only_darwin(monkeypatch):
    """darwin: killpg primary path, then wait after SIGKILL."""
    monkeypatch.setattr(sys, "platform", "darwin")
    sent = []
    monkeypatch.setattr(ort.os, "getpgid", lambda pid: 4242)
    monkeypatch.setattr(ort.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    proc = _FakeProc(pid=123, poll=None, wait_raise=subprocess.TimeoutExpired(["x"], 2))
    ort.terminate_process_group(proc)
    assert sent[0] == (4242, signal.SIGTERM)
    assert sent[1] == (4242, signal.SIGKILL)
    assert proc.waited == [2.0, 2.0]


def test_terminate_group_already_reaped(monkeypatch):
    monkeypatch.setattr(ort.os, "getpgid", lambda pid: (_ for _ in ()).throw(
        ProcessLookupError()))
    called = []
    monkeypatch.setattr(ort.os, "killpg", lambda pgid, sig: called.append(sig))
    ort.terminate_process_group(_FakeProc(pid=123, poll=0))
    assert called == []


def test_terminate_group_never_raises(monkeypatch):
    monkeypatch.setattr(ort.os, "getpgid", lambda pid: 4242)
    monkeypatch.setattr(ort.os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(
        OSError("nope")))
    assert ort.terminate_process_group(_FakeProc(pid=123, poll=0)) is None
