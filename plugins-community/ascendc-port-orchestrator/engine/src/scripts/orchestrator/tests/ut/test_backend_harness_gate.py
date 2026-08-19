# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""G1 process-level gate: the opencode backend must never spawn `claude` or read the
Claude config dir.

Probe surface (all monkeypatched): subprocess.Popen / subprocess.run / os.system /
os.execv* / os.posix_spawn / asyncio.create_subprocess_exec, plus config-dir file/stat
access against a sentinel CLAUDE_CONFIG_DIR. A POSITIVE CONTROL asserts the recorder
actually observed the opencode spawn, so a vacuous pass is impossible.

Whitelist (out of scope of THIS test by design): resolve_harness's CLAUDECODE fingerprint
reads, cc_backend itself, and claude-side tests — this gate exercises only the opencode
backend dispatch path.
"""
from __future__ import annotations

import asyncio
import builtins
import os
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # orchestrator/
from backends.opencode_backend import OpencodeBackend  # noqa: E402


@pytest.fixture
def recorders(monkeypatch, tmp_path):
    """Install spawn-family recorders + a sentinel config-dir access recorder."""
    spawns = []
    sentinel_reads = []
    # Point both Claude's explicit config variable and its conventional home at
    # the same non-existent directory.  That proves the OpenCode path does not
    # fall back to `~/.claude` after ignoring CLAUDE_CONFIG_DIR.
    home = tmp_path / "sentinel-home"
    sentinel = home / ".claude"  # must not exist

    def record_argv(argv):
        spawns.append([str(a) for a in (argv or [])])
        raise FileNotFoundError("recorded; no real spawn in this test")

    class FakePopen:
        def __init__(self, argv, **kw):
            record_argv(argv)

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: record_argv(argv))
    monkeypatch.setattr(os, "system", lambda cmd: record_argv([cmd]))
    for name in ("execl", "execle", "execlp", "execlpe", "execv", "execve",
                 "execvp", "execvpe", "posix_spawn", "posix_spawnp"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, lambda *a, **k: record_argv(list(a)))
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        lambda *a, **k: record_argv(list(a)))

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(sentinel))
    monkeypatch.setenv("HOME", str(home))
    # This gate targets the dispatch/command path itself.  G5 runtime probing
    # has dedicated tests and would otherwise make the spawn recorder pass
    # without reaching `opencode run`.
    monkeypatch.setenv("AOG_OPENCODE_SKIP_RUNTIME_CHECK", "1")

    _path_open = Path.open
    _path_read = Path.read_text
    _path_stat = Path.stat
    _builtin_open = builtins.open
    _os_stat = os.stat
    _os_scandir = os.scandir

    def _is_sentinel(p):
        return str(p).startswith(str(sentinel))

    def _rec(p, label, fn, *a, **k):
        if _is_sentinel(p):
            sentinel_reads.append(f"{label}:{p}")
        return fn(p, *a, **k)

    def _path_open_rec(self, *a, **k):
        return _rec(self, "Path.open", _path_open, *a, **k)

    def _path_read_rec(self, *a, **k):
        return _rec(self, "Path.read_text", _path_read, *a, **k)

    def _path_stat_rec(self, *a, **k):
        return _rec(self, "Path.stat", _path_stat, *a, **k)

    monkeypatch.setattr(Path, "open", _path_open_rec)
    monkeypatch.setattr(Path, "read_text", _path_read_rec)
    monkeypatch.setattr(Path, "stat", _path_stat_rec)
    monkeypatch.setattr(builtins, "open",
                        lambda p, *a, **k: _rec(p, "open", _builtin_open, *a, **k))
    monkeypatch.setattr(os, "stat",
                        lambda p, *a, **k: _rec(p, "os.stat", _os_stat, *a, **k))
    monkeypatch.setattr(os, "scandir",
                        lambda p=".", *a, **k: _rec(p, "os.scandir", _os_scandir, *a, **k))

    def prove_sentinel_read():
        """Prove the recorder is live before asserting the production path is clean."""
        try:
            (sentinel / "recorder-probe").read_text()
        except FileNotFoundError:
            pass
        assert sentinel_reads, "sentinel read recorder did not observe its positive control"
        sentinel_reads.clear()

    return {
        "spawns": spawns,
        "sentinel_reads": sentinel_reads,
        "prove_sentinel_read": prove_sentinel_read,
    }


def _dispatch(backend, mode):
    try:
        return backend.dispatch("aog-kernel-worker", "gate probe", kind="agent",
                                mode=mode, timeout=5)
    except FileNotFoundError:
        return None  # recorder refused the spawn — argv was captured first


def test_opencode_dispatch_never_spawns_claude(tmp_path, recorders):
    backend = OpencodeBackend(opencode_bin=str(tmp_path / "opencode"))
    for mode in ("streaming", "foreground"):
        _dispatch(backend, mode)
    assert recorders["spawns"], "positive control failed: recorder saw no spawn"
    joined = [" ".join(a) for a in recorders["spawns"]]
    assert any(len(argv) >= 2 and argv[1] == "run" for argv in recorders["spawns"]), \
        "positive control failed: recorder did not observe an opencode run"
    assert not any("claude" in j.lower() for j in joined), \
        f"opencode backend spawned claude: {joined}"


def test_opencode_dispatch_never_reads_claude_config_dir(tmp_path, recorders):
    backend = OpencodeBackend(opencode_bin=str(tmp_path / "opencode"))
    recorders["prove_sentinel_read"]()
    _dispatch(backend, "streaming")
    assert recorders["sentinel_reads"] == [], \
        f"opencode dispatch read the sentinel claude config dir: {recorders['sentinel_reads']}"
