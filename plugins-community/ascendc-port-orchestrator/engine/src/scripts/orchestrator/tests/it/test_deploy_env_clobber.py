# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Preserve resolved per-target deployment values.

DEBT-DEPLOY-ENV-CLOBBER (back-agent 2026-05-30): deploy_to_npu.sh must not let
its post-resolve re-source of .ascendc_env (which picks up BENCHMARK_ROOT) clobber
the per-target CANN_PATH/SOC_VERSION that resolve_target.sh resolved.

The bug: for a non-default target (e.g. TARGET=a3) the file's TOP-LEVEL
CANN_PATH/SOC_VERSION are typically the a5 defaults; `set -a; source ENV` re-read
them over the resolved A3 values, so an a3 deploy tried to build with the a5 SOC.

These tests exercise the real deploy_to_npu.sh env-resolution path via the
DEPLOY_RESOLVE_ONLY=1 dry-run hook (prints the resolved target env, exits 0 before
any SSH/build) against a crafted .ascendc_env — no hardware needed.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors

_HERE = Path(__file__).resolve()
# <repo>/src/scripts/orchestrator/tests/test_*.py → repo root is parents[4]
_REPO = _reorg_paths.REPO_ROOT
_DEPLOY = _REPO / "src" / "scripts" / "deploy_to_npu.sh"
_BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(_BASH is None, reason="bash executable not found")

_BUG_ENV = """\
TARGET={target}
CANN_PATH=/opt/A5DEFAULT/cann-a5
SOC_VERSION=Ascend950PR_9579
A5_HOST=a5host
A5_CONTAINER=a5c
A5_CANN_PATH=/opt/A5DEFAULT/cann-a5
A5_SOC_VERSION=Ascend950PR_9579
A3_HOST=198.51.100.70
A3_CONTAINER=npu-a3-back
A3_CANN_PATH=/home/z3/cann/cann-9.0.0
A3_SOC_VERSION=Ascend910_9382
"""


def _run_resolve_only(tmp_path: Path, env_text: str) -> tuple[subprocess.CompletedProcess, str]:
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(env_text)
    proc = subprocess.run(
        [_BASH, str(_DEPLOY)],
        env={
            "ASCENDC_ENV_FILE": str(env_file),
            "DEPLOY_RESOLVE_ONLY": "1",
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
        },
        capture_output=True, text=True, timeout=30,
    )
    return proc, proc.stdout + proc.stderr


def _resolve(tmp_path: Path, target: str) -> dict:
    proc, out = _run_resolve_only(tmp_path, _BUG_ENV.format(target=target))
    m = re.search(r"^RESOLVED (.+)$", out, re.MULTILINE)
    assert m, f"no RESOLVED line (rc={proc.returncode}):\n{out}"
    return dict(kv.split("=", 1) for kv in m.group(1).split())


def _resolve_output(tmp_path: Path, env_text: str) -> str:
    proc, out = _run_resolve_only(tmp_path, env_text)
    assert proc.returncode == 0, out
    return out


@pytest.mark.skipif(not _DEPLOY.is_file(), reason="deploy_to_npu.sh not present")
def test_a3_target_not_clobbered_by_a5_toplevel(tmp_path):
    """Keep resolved source-target values when top-level defaults differ.

    Re-sourcing must not clobber the per-target CANN_PATH or SOC_VERSION.
    """
    r = _resolve(tmp_path, "a3")
    assert r["TARGET"] == "a3"
    assert r["SOC_VERSION"] == "Ascend910_9382", (
        f"SOC clobbered to {r['SOC_VERSION']} (DEBT-DEPLOY-ENV-CLOBBER regressed)")
    assert r["CANN_PATH"] == "/home/z3/cann/cann-9.0.0", (
        f"CANN_PATH clobbered to {r['CANN_PATH']} (DEBT-DEPLOY-ENV-CLOBBER regressed)")
    assert r["CONTAINER"] == "npu-a3-back"


@pytest.mark.skipif(not _DEPLOY.is_file(), reason="deploy_to_npu.sh not present")
def test_a5_default_target_unchanged(tmp_path):
    """TARGET=a5 (top-level == resolved): the clobber-fix is a no-op — a5 values."""
    r = _resolve(tmp_path, "a5")
    assert r["TARGET"] == "a5"
    assert r["SOC_VERSION"] == "Ascend950PR_9579"
    assert r["CANN_PATH"] == "/opt/A5DEFAULT/cann-a5"
    assert r["CONTAINER"] == "a5c"


@pytest.mark.skipif(not _DEPLOY.is_file(), reason="deploy_to_npu.sh not present")
def test_a3_deploy_stage_roots_are_env_overridable(tmp_path):
    out = _resolve_output(tmp_path, _BUG_ENV.format(target="a3") + """\
A3_DEPLOY_STAGE_HOST=/home/npu_user
A3_DEPLOY_STAGE_CONTAINER=/home/npu_user
""")

    m = re.search(r"^RESOLVED_DEPLOY_STAGE (.+)$", out, re.MULTILINE)
    assert m, out
    resolved = dict(kv.split("=", 1) for kv in m.group(1).split())
    assert resolved["DEPLOY_STAGE_HOST"] == "/home/npu_user/ascendc_op_gen_stage"
    assert resolved["DEPLOY_STAGE_CONTAINER"] == "/home/npu_user/ascendc_op_gen_stage"
