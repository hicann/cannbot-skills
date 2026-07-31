# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-STAGE-COLLISION (2026-06-16): parallel op-gen runs sharing the same
$HOME must NOT clobber each other's LOCAL staging dir.

Incident: independent review's account-01 celu run clobbered/co-killed account-02's
selective_scan run because deploy_to_npu.sh's LOCAL_TASK default
(`$HOME/workspace/AscendOpGenAgent/current_task`) was BOTH account-agnostic
AND lane-agnostic — two parallel runs on the same $HOME resolved to the SAME
local tar/scp staging dir.

Fix: the LOCAL_TASK *default* (the `${LOCAL_TASK:-...}` fallback only) now bakes
in BOTH $USER (account isolation) AND LANE (lane isolation):
    $HOME/workspace/AscendOpGenAgent/current_task_${USER}_lane${LANE}
An explicit LOCAL_TASK env override still wins (backward-compat, Kimi-style /
lane-wrapper lane>0).

These tests exercise the real deploy_to_npu.sh resolution path via the
DEPLOY_RESOLVE_ONLY=1 dry-run hook (now also prints the RESOLVED_STAGING line),
plus the writer-side deploy_to_npu_lane.sh lane-0 consistency. No hardware.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors
import pytest

_HERE = Path(__file__).resolve()
# <repo>/src/scripts/orchestrator/tests/test_*.py → repo root is parents[4]
_REPO = _reorg_paths.REPO_ROOT
_DEPLOY = _REPO / "src" / "scripts" / "deploy_to_npu.sh"
_DEPLOY_LANE = _REPO / "src" / "scripts" / "deploy_to_npu_lane.sh"
_BASH = shutil.which("bash")


def _bash() -> str:
    if _BASH is None:
        pytest.skip("bash executable not found")
    return _BASH


# Minimal a5 env (default target). Staging resolution does not depend on the
# per-target HOST/CONTAINER values, only on HOME/USER/LANE/BENCHMARK_ROOT.
_ENV = """\
TARGET=a5
CANN_PATH=/opt/cann
SOC_VERSION=Ascend950PR_9579
A5_HOST=a5host
A5_CONTAINER=a5c
A5_CANN_PATH=/opt/cann
A5_SOC_VERSION=Ascend950PR_9579
"""

_A3_ENV = """\
TARGET=a3
CANN_PATH=/opt/cann-a5
SOC_VERSION=Ascend950PR_9579
BENCHMARK_ROOT=/root/AscendOpGenAgent
A5_HOST=a5host
A5_CONTAINER=a5c
A5_CANN_PATH=/opt/cann-a5
A5_SOC_VERSION=Ascend950PR_9579
A3_HOST=a3host
A3_CONTAINER=a3c
A3_CANN_PATH=/opt/cann-a3
A3_SOC_VERSION=Ascend910_9382
"""


def _resolve_staging(tmp_path: Path, *, user: str, lane: str | None,
                     home: str | None = None,
                     local_task: str | None = None,
                     benchmark_root: str | None = None) -> dict:
    """Run deploy_to_npu.sh in DEPLOY_RESOLVE_ONLY mode, parse RESOLVED_STAGING."""
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(_ENV)
    env = {
        "ASCENDC_ENV_FILE": str(env_file),
        "DEPLOY_RESOLVE_ONLY": "1",
        "PATH": "/usr/bin:/bin",
        "HOME": home if home is not None else str(tmp_path),
        "USER": user,
    }
    if lane is not None:
        env["LANE"] = lane
    if local_task is not None:
        env["LOCAL_TASK"] = local_task
    if benchmark_root is not None:
        env["BENCHMARK_ROOT"] = benchmark_root
    proc = subprocess.run(
        [_bash(), str(_DEPLOY)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    out = proc.stdout + proc.stderr
    m = re.search(r"^RESOLVED_STAGING (.+)$", out, re.MULTILINE)
    assert m, f"no RESOLVED_STAGING line (rc={proc.returncode}):\n{out}"
    return dict(kv.split("=", 1) for kv in m.group(1).split())


@pytest.mark.skipif(not _DEPLOY.is_file(), reason="deploy_to_npu.sh not present")
def test_two_accounts_same_home_no_collision(tmp_path):
    """The incident: two OS accounts sharing one $HOME, both at lane 0, must NOT
    resolve LOCAL_TASK to the same path.
    """
    a = _resolve_staging(tmp_path, user="account01", lane="0")
    b = _resolve_staging(tmp_path, user="account02", lane="0")
    assert a["LOCAL_TASK"] != b["LOCAL_TASK"], (
        f"account collision: {a['LOCAL_TASK']} == {b['LOCAL_TASK']}")
    assert "account01" in a["LOCAL_TASK"]
    assert "account02" in b["LOCAL_TASK"]


@pytest.mark.skipif(not _DEPLOY.is_file(), reason="deploy_to_npu.sh not present")
def test_two_lanes_same_account_no_collision(tmp_path):
    """Same account, different NPU lane, must NOT collide on the local default."""
    a = _resolve_staging(tmp_path, user="acc", lane="0")
    b = _resolve_staging(tmp_path, user="acc", lane="2")
    assert a["LOCAL_TASK"] != b["LOCAL_TASK"], (
        f"lane collision: {a['LOCAL_TASK']} == {b['LOCAL_TASK']}")
    assert a["LOCAL_TASK"].endswith("_lane0")
    assert b["LOCAL_TASK"].endswith("_lane2")


@pytest.mark.skipif(not _DEPLOY.is_file(), reason="deploy_to_npu.sh not present")
def test_account_and_lane_both_in_default(tmp_path):
    """The default path must contain BOTH $USER and the lane discriminator."""
    r = _resolve_staging(tmp_path, user="alice", lane="1")
    assert "alice" in r["LOCAL_TASK"], r["LOCAL_TASK"]
    assert "lane1" in r["LOCAL_TASK"], r["LOCAL_TASK"]


@pytest.mark.skipif(not _DEPLOY.is_file(), reason="deploy_to_npu.sh not present")
def test_explicit_local_task_override_wins(tmp_path):
    """HARD backward-compat: an explicit LOCAL_TASK env override must still win
    verbatim, ignoring the $USER/lane isolation default (Kimi-style override).
    """
    r = _resolve_staging(tmp_path, user="whoever", lane="3",
                         local_task="/custom/path")
    assert r["LOCAL_TASK"] == "/custom/path", r["LOCAL_TASK"]


@pytest.mark.skipif(not _DEPLOY.is_file(), reason="deploy_to_npu.sh not present")
def test_lane_unset_defaults_to_lane0(tmp_path):
    """Direct deploy_to_npu.sh invocation (no LANE) defaults the lane to 0."""
    r = _resolve_staging(tmp_path, user="bob", lane=None)
    assert r["LOCAL_TASK"].endswith("_lane0"), r["LOCAL_TASK"]
    assert "bob" in r["LOCAL_TASK"]


@pytest.mark.skipif(not _DEPLOY_LANE.is_file(), reason="deploy_to_npu_lane.sh not present")
def test_writer_lane0_matches_deploy_default(tmp_path):
    """CONSISTENCY: deploy_to_npu_lane.sh lane-0 must sync the workspace into the
    SAME isolated path deploy_to_npu.sh tars — else deploy finds an empty dir.

    The lane wrapper's lane-0 branch prints
    `[deploy/lane0] synced ASCENDC_WORKSPACE=<src> → LOCAL_TASK=<dst>` (when a
    workspace is provided). Assert <dst> equals deploy_to_npu.sh's resolved
    LOCAL_TASK default for the same (USER, lane=0, HOME).
    """
    # Build a fake workspace so _sync_workspace_to_local_task runs + echoes the dst.
    home = tmp_path / "home"
    ws = tmp_path / "ws"
    (ws / "kernel").mkdir(parents=True)
    (ws / "model.py").write_text("# stub\n")
    (home / "workspace").mkdir(parents=True)
    # The lane-0 path does not read an env file before syncing, so HOME/USER suffice.

    deploy_default = _resolve_staging(
        tmp_path, user="carol", lane="0", home=str(home),
    )["LOCAL_TASK"]

    proc = subprocess.run(
        [_bash(), str(_DEPLOY_LANE), "--lane", "0"],
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
            "USER": "carol",
            "ASCENDC_WORKSPACE": str(ws),
            # Make the inner deploy exit early (empty/cleaned) — we only need the
            # wrapper's sync echo, not a real deploy. The wrapper execs deploy
            # which will fail on SSH, but the sync line is printed BEFORE exec.
            "DEPLOY_RESOLVE_ONLY": "1",
            "ASCENDC_ENV_FILE": str(home / "nonexistent_env"),
        },
        capture_output=True, text=True, timeout=30,
    )
    out = proc.stdout + proc.stderr
    m = re.search(r"→ LOCAL_TASK=(\S+)", out)
    assert m, f"no sync echo from lane wrapper:\n{out}"
    writer_dst = m.group(1)
    assert writer_dst == deploy_default, (
        f"writer/deploy path divergence:\n  writer  = {writer_dst}\n"
        f"  deploy  = {deploy_default}")


@pytest.mark.skipif(not _DEPLOY_LANE.is_file(), reason="deploy_to_npu_lane.sh not present")
def test_lane0_wrapper_sync_removes_excluded_stale_files(tmp_path):
    """The workspace -> LOCAL_TASK sync must be an exact filtered mirror.

    rsync --delete alone does not remove files hidden by exclude rules. That
    left stale root pybind11.cpp/kernel.h/kernels.cpp and dot-state in
    LOCAL_TASK, which then got deployed and built instead of the current
    workspace.
    """
    home = tmp_path / "home"
    ws = tmp_path / "ws"
    (ws / "kernel").mkdir(parents=True)
    (ws / "kernel" / "kernel.h").write_text("// current\n")
    (ws / "kernel" / "kernels.cpp").write_text("// current\n")
    (ws / "model.py").write_text("# current\n")
    (ws / "model_new_ascendc.py").write_text("# current\n")
    (ws / ".opgen_state.json").write_text('{"source": "hidden state"}\n')
    (home / "workspace").mkdir(parents=True)

    deploy_default = Path(_resolve_staging(
        tmp_path, user="carol", lane="0", home=str(home),
    )["LOCAL_TASK"])
    deploy_default.mkdir(parents=True)
    (deploy_default / "pybind11.cpp").write_text("// stale root pybind\n")
    (deploy_default / "kernel.h").write_text("// stale root kernel\n")
    (deploy_default / "kernels.cpp").write_text("// stale root kernels\n")
    (deploy_default / ".opgen_state.json").write_text("{}\n")
    (deploy_default / "kernel").mkdir(exist_ok=True)
    (deploy_default / "kernel" / "old.cpp").write_text("// stale nested\n")

    proc = subprocess.run(
        [_bash(), str(_DEPLOY_LANE), "--lane", "0"],
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
            "USER": "carol",
            "ASCENDC_WORKSPACE": str(ws),
            "DEPLOY_RESOLVE_ONLY": "1",
            "ASCENDC_ENV_FILE": str(home / "nonexistent_env"),
        },
        capture_output=True, text=True, timeout=30,
    )
    out = proc.stdout + proc.stderr
    assert "synced ASCENDC_WORKSPACE" in out, out

    files = sorted(p.relative_to(deploy_default).as_posix()
                   for p in deploy_default.rglob("*") if p.is_file())
    assert files == [
        "kernel/kernel.h",
        "kernel/kernels.cpp",
        "model.py",
        "model_new_ascendc.py",
    ]


@pytest.mark.skipif(not _DEPLOY_LANE.is_file(), reason="deploy_to_npu_lane.sh not present")
def test_lane0_wrapper_normalizes_sourced_shared_benchmark_root(tmp_path):
    """If the caller has already sourced .ascendc_env, BENCHMARK_ROOT=/root/...
    arrives as a pre-set env var. The lane0 wrapper must still deploy to the
    lane0 root, matching phase_o5_runner's O5 read path.
    """
    home = tmp_path / "home"
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(_A3_ENV)

    proc = subprocess.run(
        [_bash(), str(_DEPLOY_LANE), "--lane", "0"],
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
            "USER": "carol",
            "ASCENDC_ENV_FILE": str(env_file),
            "DEPLOY_RESOLVE_ONLY": "1",
            "BENCHMARK_ROOT": "/root/AscendOpGenAgent",
        },
        capture_output=True, text=True, timeout=30,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    m = re.search(r"^RESOLVED_STAGING (.+)$", out, re.MULTILINE)
    assert m, out
    resolved = dict(kv.split("=", 1) for kv in m.group(1).split())
    assert resolved["REMOTE_TASK"] == (
        "/home/npu_user/workspace/AscendOpGenAgent_lane0/current_task"
    )
    assert "/root/AscendOpGenAgent/current_task" not in resolved["REMOTE_TASK"]
