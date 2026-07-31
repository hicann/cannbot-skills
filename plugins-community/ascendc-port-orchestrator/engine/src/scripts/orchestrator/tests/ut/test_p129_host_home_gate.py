# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression tests for the P129 A3_HOST_HOME mount validation gate.

Per docs/handovers/SESSION_HANDOVER_2026_05_17.md + ROADMAP P129: orchestrator
must hard-fail (or warn-and-continue under advisory conditions) when the
.ascendc_env A3_HOST_HOME value disagrees with the live container mount,
so that scp pushes go to a path the container can read.

The gate is implemented as `_validate_a3_host_home_mount(a3_host, a3_container,
expected_host_home)` in orchestrator.py. Returns 0 on success or advisory
fallback, 18 on hard mismatch.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

import orchestrator as orch  # noqa: E402


def _fake_run(returncode: int, stdout: str = "", stderr: str = ""):
    """Build a subprocess.CompletedProcess-like result for monkeypatching."""
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_gate_passes_when_mount_matches(monkeypatch, capsys):
    """Happy path: docker inspect returns /home/npu_user_opus, env says same."""
    monkeypatch.setattr(
        orch.subprocess,
        "run",
        lambda *args, **kwargs: _fake_run(0, stdout="/home/npu_user_opus\n"),
    )
    rc = getattr(orch, '_validate_a3_host_home_mount')(
        "198.51.100.70", "npu-a3", "/home/npu_user_opus"
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "ERROR" not in out, "should not print ERROR on happy path"


def test_container_home_config_driven_in_inspect_command(monkeypatch):
    """② genericize (independent review re-sync catch 2026-07-01): a config-driven `container_home`
    (non-npu_user) MUST appear in the docker-inspect `.Destination` filter instead of the
    hardcoded /home/npu_user — else a scrubbed / non-npu_user cannbot deployment inspects a
    Destination the container doesn't have → false rc=18. Locks validation.py's param-inject.
    """
    captured = {}

    def _capture(*args, **kwargs):
        captured["cmd"] = args[0] if args else kwargs.get("args")
        return _fake_run(0, stdout="/data/acme_opus\n")

    monkeypatch.setattr(orch.subprocess, "run", _capture)
    rc = getattr(orch, '_validate_a3_host_home_mount')(
        "198.51.100.70", "npu-a3", "/data/acme_opus", container_home="/data/acme"
    )
    assert rc == 0
    cmd_str = " ".join(captured["cmd"])
    assert "/data/acme" in cmd_str, "custom container_home must reach the docker-inspect Destination filter"
    assert "npu_user" not in cmd_str, "hardcoded npu_user must NOT appear when container_home is overridden"


def test_gate_fails_on_default_mount_when_env_expects_slice(monkeypatch, capsys):
    """The exact 2026-05-16/17 P129 incident: env expects _opus slice,
    container has default 1:1 mount instead. MUST hard-fail.
    """
    monkeypatch.setattr(
        orch.subprocess,
        "run",
        lambda *args, **kwargs: _fake_run(0, stdout="/home/npu_user\n"),
    )
    rc = getattr(orch, '_validate_a3_host_home_mount')(
        "198.51.100.70", "npu-a3", "/home/npu_user_opus"
    )
    assert rc == 18, "mismatch must return exit code 18 (P129 gate)"
    out = capsys.readouterr().out
    assert "P129 mount gate FAILED" in out
    assert "/home/npu_user_opus" in out
    assert "/home/npu_user" in out


def test_gate_advisory_on_ssh_failure(monkeypatch, capsys):
    """docker inspect rc != 0 → advisory warn, return 0 (not hard-fail).
    Reason: SSH or docker transient failure shouldn't block op-gen by itself;
    downstream operations will surface the issue with concrete context.
    """
    monkeypatch.setattr(
        orch.subprocess,
        "run",
        lambda *args, **kwargs: _fake_run(255, stderr="ssh: connect to host failed"),
    )
    rc = getattr(orch, '_validate_a3_host_home_mount')(
        "198.51.100.70", "npu-a3", "/home/npu_user_opus"
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "WARN" in out and "advisory" in out


def test_gate_advisory_on_ssh_timeout(monkeypatch, capsys):
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=15)
    monkeypatch.setattr(orch.subprocess, "run", _raise_timeout)
    rc = getattr(orch, '_validate_a3_host_home_mount')(
        "198.51.100.70", "npu-a3", "/home/npu_user_opus"
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "timeout" in out.lower()


def test_gate_noop_on_empty_inputs():
    """When caller has not configured A3 yet, gate is a no-op (the
    upstream A3_HOST + A3_CONTAINER existence check is the actual gate
    for missing config; the mount gate only fires once both exist).
    """
    assert getattr(orch, '_validate_a3_host_home_mount')("", "npu-a3", "/home/npu_user_opus") == 0
    assert getattr(orch, '_validate_a3_host_home_mount')("198.51.100.70", "", "/home/npu_user_opus") == 0
    assert getattr(orch, '_validate_a3_host_home_mount')("198.51.100.70", "npu-a3", "") == 0


def test_gate_handles_extra_whitespace_in_docker_output(monkeypatch):
    """docker inspect Go-template output sometimes has trailing newline /
    whitespace. Gate must strip before comparing — without strip, every
    valid setup would be reported as mismatch.
    """
    monkeypatch.setattr(
        orch.subprocess,
        "run",
        lambda *args, **kwargs: _fake_run(0, stdout="  /home/npu_user_opus  \n"),
    )
    rc = getattr(orch, '_validate_a3_host_home_mount')(
        "198.51.100.70", "npu-a3", "/home/npu_user_opus"
    )
    assert rc == 0


def test_gate_exception_returns_advisory(monkeypatch, capsys):
    """Any unexpected subprocess.run exception → advisory (not hard-fail).
    Prevents transient infra hiccups from blocking the orchestrator.
    """
    def _boom(*args, **kwargs):
        raise OSError("disk full reading docker socket")
    monkeypatch.setattr(orch.subprocess, "run", _boom)
    rc = getattr(orch, '_validate_a3_host_home_mount')(
        "198.51.100.70", "npu-a3", "/home/npu_user_opus"
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "WARN" in out
