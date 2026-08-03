# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Keep DEBT-159 remote current-task isolation consistent across lanes.

Every remote-touching step (deploy_to_npu.sh REMOTE_TASK/BUILD_ROOT and
phase_o5_runner._lane_aware_benchmark_root), with NO step falling back to the SHARED
`/root/AscendOpGenAgent/current_task` for a lane-managed run.

Ground truth (an isolated lane2 deploy log, 2026-06-16): the lane2 pipeline
cleaned BOTH `.../AscendOpGenAgent_lane2/current_task` (correct, isolated) AND
`/root/AscendOpGenAgent/current_task` (WRONG, shared — collides cross-lane). The shared
touch came from a bare `deploy_to_npu.sh` invocation (the worker's manual
`LOCAL_TASK=… deploy_to_npu.sh` fallback to dodge the git-tracked `current_task` symlink)
which set LOCAL_TASK but NOT BENCHMARK_ROOT → REMOTE_TASK defaulted to /root.

Fix: deploy_to_npu.sh derives a lane-aware REMOTE_TASK/BUILD_ROOT default (LANE>0 →
canonical `_lane{N}` remote base; LANE==0 with shared/empty env → `_lane0`; non-shared
env-file root honored verbatim; bare no-LANE → legacy /root), and
phase_o5_runner._lane_aware_benchmark_root mirrors it EXACTLY. So deploy + build + O5
resolve the SAME remote root for a given lane, and lane 0 is isolated too.

These tests exercise:
  1. deploy_to_npu.sh REMOTE_TASK resolution (real script via DEPLOY_RESOLVE_ONLY=1).
  2. phase_o5_runner._lane_aware_benchmark_root (imported).
  3. deploy ↔ O5 consistency (the same lane resolves to the same remote root in both).
No hardware.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors

_HERE = Path(__file__).resolve()
_REPO = _reorg_paths.REPO_ROOT
_DEPLOY = _REPO / "src" / "scripts" / "deploy_to_npu.sh"
_BASH = shutil.which("bash")

sys.path.insert(0, str(_HERE.parents[1]))
from phase_o5_runner import _lane_aware_benchmark_root  # noqa: E402

_SHARED = "/root/AscendOpGenAgent"
_ROOT_BASE = "/home/npu_user/workspace"  # canonical remote (container) home

_ENV_TEMPLATE = """\
TARGET=a5
CANN_PATH=/opt/cann
SOC_VERSION=Ascend950PR_9579
A5_HOST=a5host
A5_CONTAINER=a5c
A5_CANN_PATH=/opt/cann
A5_SOC_VERSION=Ascend950PR_9579
{benchmark_root_line}
"""


def _deploy_remote_task(tmp_path: Path, *, lane: str | None,
                        env_benchmark_root: str | None,
                        injected_benchmark_root: str | None = None,
                        user: str = "acc") -> str:
    """Run deploy_to_npu.sh DEPLOY_RESOLVE_ONLY, return resolved REMOTE_TASK."""
    if _BASH is None:
        pytest.skip("bash executable not found")
    line = f"BENCHMARK_ROOT={env_benchmark_root}" if env_benchmark_root else ""
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(_ENV_TEMPLATE.format(benchmark_root_line=line))
    env = {
        "ASCENDC_ENV_FILE": str(env_file),
        "DEPLOY_RESOLVE_ONLY": "1",
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "USER": user,
    }
    if lane is not None:
        env["LANE"] = lane
    if injected_benchmark_root is not None:
        env["BENCHMARK_ROOT"] = injected_benchmark_root  # caller-injected (_PRE_)
    proc = subprocess.run([_BASH, str(_DEPLOY)], env=env,
                          capture_output=True, text=True, timeout=30)
    out = proc.stdout + proc.stderr
    m = re.search(r"REMOTE_TASK=(\S+)", out)
    assert m, f"no REMOTE_TASK (rc={proc.returncode}):\n{out}"
    return m.group(1)


# ── deploy_to_npu.sh REMOTE_TASK resolution ───────────────────────────────

@pytest.mark.skipif(not _DEPLOY.is_file(), reason="deploy_to_npu.sh not present")
def test_deploy_lane0_shared_env_isolates_not_root(tmp_path):
    """Isolate lane 0 from the shared remote task root.

    The shared env-file default must resolve to _lane0/current_task.
    """
    rt = _deploy_remote_task(tmp_path, lane="0", env_benchmark_root=_SHARED)
    assert rt == f"{_ROOT_BASE}/AscendOpGenAgent_lane0/current_task", rt
    assert "/root/AscendOpGenAgent/current_task" not in rt


@pytest.mark.skipif(not _DEPLOY.is_file(), reason="deploy_to_npu.sh not present")
def test_deploy_lane_n_shared_env_isolates(tmp_path):
    """Lane N>0 with shared /root env → isolated _laneN (the celu(lane2) case)."""
    rt = _deploy_remote_task(tmp_path, lane="2", env_benchmark_root=_SHARED)
    assert rt == f"{_ROOT_BASE}/AscendOpGenAgent_lane2/current_task", rt
    assert "/root/AscendOpGenAgent/current_task" not in rt


@pytest.mark.skipif(not _DEPLOY.is_file(), reason="deploy_to_npu.sh not present")
def test_deploy_two_lanes_no_shared_root(tmp_path):
    """Resolve lane 0 and lane 2 to different remote task paths.

    Neither path may use the shared root.
    """
    rt0 = _deploy_remote_task(tmp_path, lane="0", env_benchmark_root=_SHARED)
    rt2 = _deploy_remote_task(tmp_path, lane="2", env_benchmark_root=_SHARED)
    assert rt0 != rt2, f"lane collision: {rt0} == {rt2}"
    assert "/root/AscendOpGenAgent/current_task" not in rt0
    assert "/root/AscendOpGenAgent/current_task" not in rt2


@pytest.mark.skipif(not _DEPLOY.is_file(), reason="deploy_to_npu.sh not present")
def test_deploy_bare_single_cc_keeps_root_default(tmp_path):
    """Keep the legacy root for a bare single-session deploy.

    This backward-compatible path applies when no lane is set.
    """
    rt = _deploy_remote_task(tmp_path, lane=None, env_benchmark_root=_SHARED)
    assert rt == "/root/AscendOpGenAgent/current_task", rt


@pytest.mark.skipif(not _DEPLOY.is_file(), reason="deploy_to_npu.sh not present")
def test_deploy_noninstance_data_root_honored_lane0(tmp_path):
    """Honor a non-shared per-instance environment root at lane 0.

    This instance-isolated behavior must match O5.
    """
    data = "/data/npu_user/AscendOpGenAgent"
    rt = _deploy_remote_task(tmp_path, lane="0", env_benchmark_root=data)
    assert rt == f"{data}/current_task", rt


@pytest.mark.skipif(not _DEPLOY.is_file(), reason="deploy_to_npu.sh not present")
def test_deploy_data_root_lane_n_still_lane_isolated(tmp_path):
    """Keep a per-instance root lane-isolated when the lane is nonzero.

    Otherwise, lanes would collide on the shared instance root.
    """
    data = "/data/npu_user/AscendOpGenAgent"
    rt = _deploy_remote_task(tmp_path, lane="3", env_benchmark_root=data)
    assert rt == f"{_ROOT_BASE}/AscendOpGenAgent_lane3/current_task", rt


@pytest.mark.skipif(not _DEPLOY.is_file(), reason="deploy_to_npu.sh not present")
def test_deploy_injected_benchmark_root_wins(tmp_path):
    """A caller-injected BENCHMARK_ROOT (lane-wrapper lane>0 export) wins verbatim."""
    inj = f"{_ROOT_BASE}/AscendOpGenAgent_lane2"
    rt = _deploy_remote_task(tmp_path, lane="2", env_benchmark_root=_SHARED,
                             injected_benchmark_root=inj)
    assert rt == f"{inj}/current_task", rt


# ── deploy ↔ O5 consistency (the load-bearing invariant) ───────────────────

@pytest.mark.skipif(not _DEPLOY.is_file(), reason="deploy_to_npu.sh not present")
@pytest.mark.parametrize("lane", ["0", "1", "2", "3"])
@pytest.mark.parametrize("env_root", [_SHARED, "/data/npu_user/AscendOpGenAgent", None])
def test_deploy_matches_o5_for_same_lane(tmp_path, lane, env_root):
    """Resolve deploy and O5 to the same remote root for a given lane.

    This keeps worker build output and O5 verification in the same task path.
    """
    deploy_rt = _deploy_remote_task(tmp_path, lane=lane, env_benchmark_root=env_root)
    o5_env = {"BENCHMARK_ROOT": env_root} if env_root else {}
    o5_root = _lane_aware_benchmark_root(o5_env, int(lane))
    assert deploy_rt == f"{o5_root}/current_task", (
        f"deploy↔O5 divergence (lane={lane}, env={env_root}):\n"
        f"  deploy = {deploy_rt}\n  o5     = {o5_root}/current_task")


# ── phase_o5_runner._lane_aware_benchmark_root direct ──────────────────────

def test_o5_lane0_shared_isolates():
    assert _lane_aware_benchmark_root({"BENCHMARK_ROOT": _SHARED}, 0) == \
        f"{_ROOT_BASE}/AscendOpGenAgent_lane0"


def test_o5_lane0_empty_isolates():
    assert _lane_aware_benchmark_root({}, 0) == f"{_ROOT_BASE}/AscendOpGenAgent_lane0"


def test_o5_lane0_noninstance_root_honored():
    data = "/data/npu_user/AscendOpGenAgent"
    assert _lane_aware_benchmark_root({"BENCHMARK_ROOT": data}, 0) == data


def test_o5_lane_n_always_isolated():
    # lane N>0 unchanged: always canonical _laneN, ignoring env base.
    assert _lane_aware_benchmark_root({"BENCHMARK_ROOT": _SHARED}, 2) == \
        f"{_ROOT_BASE}/AscendOpGenAgent_lane2"
    assert _lane_aware_benchmark_root({"BENCHMARK_ROOT": "/data/npu_user/AscendOpGenAgent"}, 2) == \
        f"{_ROOT_BASE}/AscendOpGenAgent_lane2"


def test_o5_no_shared_root_for_any_managed_lane():
    """No lane (0..7) ever resolves to the shared /root/AscendOpGenAgent."""
    for lane in range(8):
        for env in ({}, {"BENCHMARK_ROOT": _SHARED}):
            r = _lane_aware_benchmark_root(env, lane)
            assert r != _SHARED, f"lane {lane} env {env} → shared {r}"


def test_o5_host_mode_lane0_isolates():
    env = {"A5_HOST_MODE": "1", "A5_DEPLOY_STAGE_HOST": "/data/npu_user",
           "BENCHMARK_ROOT": _SHARED}
    assert _lane_aware_benchmark_root(env, 0) == "/data/npu_user/AscendOpGenAgent_lane0"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
