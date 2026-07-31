# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P0kk Step 2.1 (2026-05-06): real SSH-based runner for O5 post-verify.

Step 2 shipped phase_o5 with a stubbed default runner. Step 2.1 replaces
the stub with phase_o5_runner.ssh_runner — reads .ascendc_env, SSH+docker
exec to run workspace verifier scripts on A5, parses JSON output.

Tests mock subprocess.run so no real SSH happens. Real-op validation
happens when an actual op runs through finalize.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o5  # noqa: E402
import phase_o5_runner  # noqa: E402


def _seed_env_and_workspace(ws: Path, *, scripts: list[str] = None):
    """Workspace + parent .ascendc_env. Listed scripts get stubbed presence."""
    scripts = scripts or ["run_pass_b.py"]
    ws.mkdir(parents=True, exist_ok=True)
    for s in scripts:
        (ws / s).write_text(f"# {s} stub\n")
    env_path = ws.parent / ".ascendc_env"
    env_path.write_text(
        "A5_HOST=test-host\nA5_USER=root\nA5_PASSWORD=test\n"
        "A5_CONTAINER=test-container\nCANN_PATH=/test/cann\n"
        "BENCHMARK_ROOT=/root/AscendOpGenAgent\n"
    )


def test_read_ascendc_env_prefers_explicit_env_path(tmp_path, monkeypatch):
    ws = tmp_path / "workspace" / "op"
    ws.mkdir(parents=True)
    (ws.parent / ".ascendc_env").write_text("TARGET=a5\nA5_HOST=wrong-host\n")
    env_path = tmp_path / "a3.env"
    env_path.write_text(
        "TARGET=a3\nA3_HOST=a3-host\nA3_CONTAINER=a3-container\n"
        "SOC_VERSION=Ascend910_9382\n"
    )
    monkeypatch.setenv("ASCENDC_ENV_PATH", str(env_path))

    env = getattr(phase_o5_runner, "_read_ascendc_env")(ws)

    assert env["TARGET"] == "a3"
    assert env["A3_HOST"] == "a3-host"
    assert env["SOC_VERSION"] == "Ascend910_9382"


def test_read_ascendc_env_accepts_deploy_env_file_alias(tmp_path, monkeypatch):
    ws = tmp_path / "workspace" / "op"
    ws.mkdir(parents=True)
    env_path = tmp_path / "deploy.env"
    env_path.write_text("TARGET=a3\nA3_HOST=deploy-a3\n")
    monkeypatch.setenv("ASCENDC_ENV_FILE", str(env_path))

    env = getattr(phase_o5_runner, "_read_ascendc_env")(ws)

    assert env["TARGET"] == "a3"
    assert env["A3_HOST"] == "deploy-a3"


# ---------------------------------------------------------------------------
# JSON tail parsing
# ---------------------------------------------------------------------------
def test_parse_json_tail_handles_pure_json():
    out = getattr(phase_o5_runner, "_try_parse_json_tail")('{"tier1_pass": 50, "total": 50}')
    assert out == {"tier1_pass": 50, "total": 50}


def test_parse_json_tail_skips_log_prefix():
    """Verifier scripts often print logs before the final JSON summary."""
    text = "[info] running tests...\n" + json.dumps({"tier1_pass": 47, "total": 50})
    out = getattr(phase_o5_runner, "_try_parse_json_tail")(text)
    assert out == {"tier1_pass": 47, "total": 50}


def test_parse_json_tail_returns_none_on_no_json():
    assert getattr(phase_o5_runner, "_try_parse_json_tail")("just plain text") is None


def test_parse_json_tail_returns_none_on_empty():
    assert getattr(phase_o5_runner, "_try_parse_json_tail")("") is None


# ---------------------------------------------------------------------------
# Verifier output normalization
# ---------------------------------------------------------------------------
def test_normalize_canonical_fields_pass_through():
    out = getattr(phase_o5_runner, "_normalize_verifier_output")(
        {"tier1_pass": 50, "total": 50, "status": "PASS"}, "pass_b"
    )
    assert out == {"tier1_pass": 50, "total": 50, "status": "PASS"}


def test_normalize_legacy_n_pass_n_total_mapped():
    """Legacy field names get mapped to canonical."""
    out = getattr(phase_o5_runner, "_normalize_verifier_output")(
        {"n_pass": 47, "n_total": 50, "status": "FAIL"}, "pass_b"
    )
    assert out["tier1_pass"] == 47
    assert out["total"] == 50


def test_normalize_determinism_extra_fields():
    out = getattr(phase_o5_runner, "_normalize_verifier_output")(
        {"policy_satisfied": True, "n_identical_cases": 50, "n_cases_checked": 50},
        "determinism",
    )
    assert out["policy_satisfied"] is True
    assert out["n_identical_cases"] == 50


# ---------------------------------------------------------------------------
# ssh_runner — fully mocked subprocess
# ---------------------------------------------------------------------------
def test_ssh_runner_no_ascendc_env_returns_error(tmp_path, monkeypatch):
    """No .ascendc_env → runner_error, doesn't crash.
    Block both candidate paths (workspace-relative + project-root fallback).
    """
    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "run_pass_b.py").write_text("# stub")
    # Force _read_ascendc_env to find nothing by monkeypatching it
    monkeypatch.setattr(phase_o5_runner, "_read_ascendc_env", lambda _ws=None: {})
    result = phase_o5_runner.ssh_runner(ws, "test_op")
    assert result.runner_error is not None
    assert "ascendc_env" in result.runner_error or "A5_HOST" in result.runner_error


def test_ssh_runner_no_run_pass_b_returns_error(tmp_path):
    """workspace lacks any Pass B verifier candidate → runner_error names them all."""
    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws.parent / ".ascendc_env").write_text("A5_HOST=test\nA5_USER=root\n")
    result = phase_o5_runner.ssh_runner(ws, "test_op")
    assert result.runner_error is not None
    # P0aaf: error message now lists all tried candidates so user can rename
    # whichever the worker actually produced.
    assert "pass_b_runner.py" in result.runner_error
    assert "run_pass_b.py" in result.runner_error


def test_p0aaf_ssh_runner_finds_pass_b_runner_alias(tmp_path, monkeypatch):
    """P0aaf (2026-05-06): worker filename audit found 4/85 canonical kernels
    have `pass_b_runner.py` and 0/85 have `run_pass_b.py` — the file the
    runner originally hardcoded. Verify alias lookup picks pass_b_runner.py
    when run_pass_b.py absent, and the SSH command actually invokes the
    found script (not the legacy name).
    """
    ws = tmp_path / "test_op"
    _seed_env_and_workspace(ws, scripts=["pass_b_runner.py"])
    monkeypatch.chdir(tmp_path)

    fake_stdout = json.dumps({"tier1_pass": 12, "total": 12, "status": "PASS"})
    captured_cmd = []

    def fake_run(cmd, **kw):
        captured_cmd.append(cmd)
        return MagicMock(returncode=0, stdout=fake_stdout, stderr="")
    monkeypatch.setattr(phase_o5_runner.subprocess, "run", fake_run)

    result = phase_o5_runner.ssh_runner(ws, "test_op")
    assert result.runner_error is None, result.runner_error
    assert result.pass_b == {"tier1_pass": 12, "total": 12, "status": "PASS"}
    # Confirm the SSH cmd includes pass_b_runner.py (the alias), not
    # run_pass_b.py (the legacy hardcoded name). The runner makes multiple
    # subprocess calls (workspace tar + scp push + ssh exec); search across
    # all captured commands for the verifier invocation.
    all_cmd_text = "\n".join(" ".join(c) for c in captured_cmd)
    assert "pass_b_runner.py" in all_cmd_text
    assert "run_pass_b.py" not in all_cmd_text


def test_p0aaf_pass_a_alias_pass_a_runner(tmp_path, monkeypatch):
    """P0aaf: pass_a_runner.py also recognized (worker uses both naming
    conventions; runner accepts either).
    """
    ws = tmp_path / "test_op"
    _seed_env_and_workspace(ws, scripts=["pass_b_runner.py", "pass_a_runner.py"])
    monkeypatch.chdir(tmp_path)

    fake_pass_b = json.dumps({"tier1_pass": 50, "total": 50, "status": "PASS"})
    # task#82: this fixture seeds pass_a_runner.py → detected as port_a3, so the
    # pass_a summary MUST carry the native two-tier fields (the engagement gate
    # rejects a single-tier port_a3 pass_a).
    fake_pass_a = json.dumps({"tier1_pass": 16, "tier2_pass": 0,
                              "tier1_pass_inclusive": 16, "total": 16,
                              "tier2_status": "N/A_ALL_T1", "status": "PASS"})
    captured_cmds = []

    def fake_run(cmd, **kw):
        cmd_str = " ".join(cmd)
        captured_cmds.append(cmd_str)
        if "pass_a_runner.py" in cmd_str:
            return MagicMock(returncode=0, stdout=fake_pass_a, stderr="")
        return MagicMock(returncode=0, stdout=fake_pass_b, stderr="")
    monkeypatch.setattr(phase_o5_runner.subprocess, "run", fake_run)

    result = phase_o5_runner.ssh_runner(ws, "test_op")
    assert result.runner_error is None
    assert result.pass_b["tier1_pass"] == 50
    assert result.pass_a["tier1_pass"] == 16
    # Both scripts were invoked over SSH
    assert any("pass_b_runner.py" in c for c in captured_cmds)
    assert any("pass_a_runner.py" in c for c in captured_cmds)


def test_p0aaf_det_alias_det_check_py(tmp_path, monkeypatch):
    """P0aaf: det_check.py recognized (workers produce this name not run_det_check.py)."""
    ws = tmp_path / "test_op"
    _seed_env_and_workspace(ws, scripts=["pass_b_runner.py", "det_check.py"])
    monkeypatch.chdir(tmp_path)

    fake_pass_b = json.dumps({"tier1_pass": 50, "total": 50, "status": "PASS"})
    fake_det = json.dumps({
        "policy_satisfied": True, "n_identical_cases": 50, "n_cases_checked": 50,
    })

    def fake_run(cmd, **kw):
        cmd_str = " ".join(cmd)
        if "det_check.py" in cmd_str:
            return MagicMock(returncode=0, stdout=fake_det, stderr="")
        return MagicMock(returncode=0, stdout=fake_pass_b, stderr="")
    monkeypatch.setattr(phase_o5_runner.subprocess, "run", fake_run)

    result = phase_o5_runner.ssh_runner(ws, "test_op")
    assert result.runner_error is None
    assert result.determinism is not None
    assert result.determinism["policy_satisfied"] is True


def test_p0aaf_find_verifier_candidate_priority(tmp_path):
    """P0aaf: when multiple candidates exist, _find_verifier returns the FIRST
    in the priority list (matters for distinguishing canonical vs legacy).
    """
    ws = tmp_path / "test_op"
    ws.mkdir()
    # Both present — pass_b_runner.py is listed first in the runner's list,
    # so it should win over run_pass_b.py.
    (ws / "pass_b_runner.py").write_text("# canonical")
    (ws / "run_pass_b.py").write_text("# legacy")
    found = getattr(phase_o5_runner, "_find_verifier")(
        ws, ["pass_b_runner.py", "run_pass_b.py"]
    )
    assert found == "pass_b_runner.py"


def test_ssh_runner_parses_pass_b_json(tmp_path, monkeypatch):
    """Happy path: run_pass_b.py JSON output → MeasuredResult.pass_b populated."""
    ws = tmp_path / "test_op"
    _seed_env_and_workspace(ws, scripts=["run_pass_b.py"])
    monkeypatch.chdir(tmp_path)

    fake_stdout = "[info] verifying...\n" + json.dumps({
        "op": "test_op", "tier1_pass": 50, "total": 50, "status": "PASS",
        "cases": [],
    })

    def fake_run(cmd, **kw):
        return MagicMock(returncode=0, stdout=fake_stdout, stderr="")
    monkeypatch.setattr(phase_o5_runner.subprocess, "run", fake_run)

    result = phase_o5_runner.ssh_runner(ws, "test_op")
    assert result.runner_error is None
    assert result.pass_b == {
        "tier1_pass": 50, "total": 50, "status": "PASS"
    }


def _route_subprocess(*, verifier_returncode: int, verifier_stdout: str):
    """Build a fake `subprocess.run` that distinguishes the pre-O5 workspace
    sync calls (scp / docker cp / docker exec untar) from the actual verifier
    invocation. Sync-side calls always succeed (rc=0); the verifier ssh exec
    returns the caller-supplied returncode + stdout.

    Routing rule: verifier ssh exec is the call whose argv contains
    "python3" (the runner invokes `python3 <verifier>.py` over ssh). Anything
    else is part of the workspace sync and gets rc=0.
    """
    def fake_run(cmd, **kw):
        cmd_str = " ".join(str(c) for c in cmd) if isinstance(cmd, list) else str(cmd)
        if "python3" in cmd_str:
            return MagicMock(returncode=verifier_returncode,
                             stdout=verifier_stdout, stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")
    return fake_run


def test_ssh_runner_handles_verifier_fail_with_json(tmp_path, monkeypatch):
    """run_pass_b.py exits non-zero (FAIL) but JSON output still parseable."""
    ws = tmp_path / "test_op"
    _seed_env_and_workspace(ws, scripts=["run_pass_b.py"])
    monkeypatch.chdir(tmp_path)

    fake_stdout = json.dumps({
        "tier1_pass": 47, "total": 50, "status": "FAIL"
    })
    monkeypatch.setattr(
        phase_o5_runner.subprocess, "run",
        _route_subprocess(verifier_returncode=1, verifier_stdout=fake_stdout),
    )

    result = phase_o5_runner.ssh_runner(ws, "test_op")
    # FAIL with parseable output → not a runner error, just measured FAIL
    assert result.runner_error is None
    assert result.pass_b == {
        "tier1_pass": 47, "total": 50, "status": "FAIL"
    }


def test_ssh_runner_ssh_failure_surfaces_as_runner_error(tmp_path, monkeypatch):
    """ssh exits non-zero with no parseable output → runner_error."""
    ws = tmp_path / "test_op"
    _seed_env_and_workspace(ws, scripts=["run_pass_b.py"])
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, **kw):
        return MagicMock(returncode=255, stdout="",
                         stderr="ssh: connect to host: Connection refused")
    monkeypatch.setattr(phase_o5_runner.subprocess, "run", fake_run)

    result = phase_o5_runner.ssh_runner(ws, "test_op")
    assert result.runner_error is not None
    assert "Connection refused" in result.runner_error or "exit 255" in result.runner_error


def test_ssh_runner_timeout_surfaces_as_runner_error(tmp_path, monkeypatch):
    """SSH timeout on verifier exec → runner_error, doesn't crash."""
    ws = tmp_path / "test_op"
    _seed_env_and_workspace(ws, scripts=["run_pass_b.py"])
    monkeypatch.chdir(tmp_path)

    import subprocess as sp

    def fake_run(cmd, **kw):
        cmd_str = " ".join(str(c) for c in cmd) if isinstance(cmd, list) else str(cmd)
        # Workspace sync calls succeed; only the verifier ssh exec times out
        if "python3" in cmd_str:
            raise sp.TimeoutExpired(cmd, 600)
        return MagicMock(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(phase_o5_runner.subprocess, "run", fake_run)

    result = phase_o5_runner.ssh_runner(ws, "test_op")
    assert result.runner_error is not None
    assert "timeout" in result.runner_error.lower()


def test_ssh_runner_invokes_pass_a_when_edge_verify_present(tmp_path, monkeypatch):
    """If edge_verify.py exists, runner invokes it for pass_a measurement."""
    ws = tmp_path / "test_op"
    _seed_env_and_workspace(ws, scripts=["run_pass_b.py", "edge_verify.py"])
    monkeypatch.chdir(tmp_path)

    call_count = {"n": 0}

    def fake_run(cmd, **kw):
        call_count["n"] += 1
        # Return different counts for pass_a vs pass_b based on which script
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
        if "edge_verify.py" in cmd_str:
            # task#82: edge_verify.py seeds port_a3 detection → pass_a must carry
            # the native two-tier fields (engagement gate rejects single-tier).
            return MagicMock(returncode=0,
                             stdout=json.dumps({"tier1_pass": 31, "tier2_pass": 0,
                                                "tier1_pass_inclusive": 31, "total": 31,
                                                "tier2_status": "N/A_ALL_T1",
                                                "status": "PASS"}),
                             stderr="")
        if "run_pass_b.py" in cmd_str:
            return MagicMock(returncode=0,
                             stdout=json.dumps({"tier1_pass": 11, "total": 11, "status": "PASS"}),
                             stderr="")
        return MagicMock(returncode=0, stdout="{}", stderr="")
    monkeypatch.setattr(phase_o5_runner.subprocess, "run", fake_run)

    result = phase_o5_runner.ssh_runner(ws, "test_op")
    assert result.runner_error is None
    assert result.pass_a is not None
    assert result.pass_b is not None
    assert result.pass_a["tier1_pass"] == 31
    assert result.pass_b["tier1_pass"] == 11
    assert call_count["n"] >= 2


def test_ssh_runner_invokes_determinism_when_run_det_present(tmp_path, monkeypatch):
    """run_det_check.py present → determinism measured."""
    ws = tmp_path / "test_op"
    _seed_env_and_workspace(ws, scripts=["run_pass_b.py", "run_det_check.py"])
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, **kw):
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
        if "run_det_check.py" in cmd_str:
            return MagicMock(returncode=0,
                             stdout=json.dumps({
                                 "policy_satisfied": True,
                                 "n_identical_cases": 50,
                                 "n_cases_checked": 50,
                             }),
                             stderr="")
        return MagicMock(returncode=0,
                         stdout=json.dumps({"tier1_pass": 50, "total": 50, "status": "PASS"}),
                         stderr="")
    monkeypatch.setattr(phase_o5_runner.subprocess, "run", fake_run)

    result = phase_o5_runner.ssh_runner(ws, "test_op")
    assert result.determinism is not None
    assert result.determinism["policy_satisfied"] is True


# ---------------------------------------------------------------------------
# End-to-end with phase_o5.post_verify_for_finalize
# ---------------------------------------------------------------------------
def test_runner_used_by_post_verify_detects_lying_worker(tmp_path, monkeypatch):
    """A backward worker claiming 50/50 is rejected when O5 measures 0/50."""
    ws = tmp_path / "test_op"
    _seed_env_and_workspace(ws, scripts=["run_pass_b.py"])
    (ws / ".opgen_state.json").write_text(json.dumps({
        "op": "test_op",
        "opgen_mode": "backward",
    }))
    monkeypatch.chdir(tmp_path)
    # Worker's lying claim
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS",
                      "pass_b": {"status": "PASS", "tier1_pass": 50, "total": 50}},
        "performance": {"status": "PASS", "ratio": 1.5},
    }))

    # SSH verifier returns the truth: 0/50 (workspace sync calls succeed)
    monkeypatch.setattr(
        phase_o5_runner.subprocess, "run",
        _route_subprocess(
            verifier_returncode=1,
            verifier_stdout=json.dumps(
                {"tier1_pass": 0, "total": 50, "status": "FAIL"}
            ),
        ),
    )

    rep = phase_o5.post_verify_for_finalize(
        ws, "test_op", runner=phase_o5_runner.ssh_runner,
    )
    assert rep.verdict == "MISMATCH"
    assert any("pass_b.tier1_pass" in m and "0" in m and "50" in m for m in rep.mismatches)


# ---------------------------------------------------------------------------
# P0vv (2026-05-06): target-aware host resolution
# ---------------------------------------------------------------------------
def test_ssh_runner_uses_a3_host_when_target_a3(tmp_path, monkeypatch):
    """When .ascendc_env has TARGET=a3, runner reads A3_HOST not A5_HOST."""
    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "run_pass_b.py").write_text("# stub")
    monkeypatch.setattr(phase_o5_runner, "_read_ascendc_env", lambda _ws=None: {
        "TARGET": "a3",
        "A3_HOST": "a3-host.example",
        "A3_USER": "root",
        "A3_PASSWORD": "",
        "A3_CONTAINER": "npu-a3",
        "CANN_PATH": "/usr/local/Ascend/cann",
        "BENCHMARK_ROOT": "/home/npu_user/workspace/AscendOpGenAgent_ds",
    })
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout='{"tier1_pass":50,"total":50,"status":"PASS"}', stderr="")
    monkeypatch.setattr(phase_o5_runner.subprocess, "run", fake_run)
    result = phase_o5_runner.ssh_runner(ws, "test_op")
    assert result.runner_error is None
    # The SSH command must target a3-host.example with a3 container
    cmd_str = " ".join(captured.get("cmd") or [])
    assert "a3-host.example" in cmd_str
    assert "npu-a3" in cmd_str


def test_ssh_runner_falls_back_to_a5_when_target_unset(tmp_path, monkeypatch):
    """Legacy .ascendc_env (no TARGET) → falls back to A5_HOST (preserves
    behavior for forks that haven't added TARGET=a5 yet).
    """
    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "run_pass_b.py").write_text("# stub")
    monkeypatch.setattr(phase_o5_runner, "_read_ascendc_env", lambda _ws=None: {
        "A5_HOST": "a5-legacy.example",
        "A5_USER": "root",
    })
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout='{"tier1_pass":50,"total":50,"status":"PASS"}', stderr="")
    monkeypatch.setattr(phase_o5_runner.subprocess, "run", fake_run)
    result = phase_o5_runner.ssh_runner(ws, "test_op")
    assert result.runner_error is None
    cmd_str = " ".join(captured.get("cmd") or [])
    assert "a5-legacy.example" in cmd_str


def test_ssh_runner_errors_when_no_target_host_or_a5_fallback(tmp_path, monkeypatch):
    """TARGET=a3 set but neither A3_HOST nor A5_HOST present → clear error."""
    ws = tmp_path / "test_op"
    ws.mkdir()
    monkeypatch.setattr(phase_o5_runner, "_read_ascendc_env", lambda _ws=None: {
        "TARGET": "a3",
        "A3_USER": "root",
    })
    result = phase_o5_runner.ssh_runner(ws, "test_op")
    assert result.runner_error is not None
    assert "A3_HOST" in result.runner_error
