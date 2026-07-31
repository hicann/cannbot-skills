# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression test for phase_o5_runner unique remote tar path (P0aca).

22_Nonzero kw-6 diagnosis (2026-05-20): two concurrent op-gen workspaces
(22_Nonzero on lane 0, 31_IOU on lane 1) both invoked
`_resync_workspace_to_container` simultaneously. Both processes scp'd to
the hardcoded remote path `/tmp/o5_sync.tar`, then both docker-cp'd it
into the container, then both `rm`'d it. The second process's untar
saw the file vanish mid-operation and returned:

    docker cp / untar failed: lstat /tmp/o5_sync.tar:
    no such file or directory

Outcome before fix: O5 RUNNER_FAILED → P0aba.O5 gate → await_worker
respawn → worker can't fix it from inside the kernel — it's an
orchestrator-infra race. Hit `await_user_decision` exit-10 pause.

Fix: parameterize the remote tar name with workspace.name + pid so
each process holds its own file. Pattern: `/tmp/o5_sync_{op}_{pid}.tar`.

Mirror the path-construction logic here so the test is independent of
phase_o5_runner's heavy import graph (same approach as
test_phase_o5_runner_oversized_pt.py).
"""
from __future__ import annotations

import logging

import os
import re
from pathlib import Path

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors


def _build_remote_tar_path(workspace: Path, pid: int) -> str:
    """Mirror of the path-construction logic now in
    phase_o5_runner._resync_workspace_to_container (P0aca, 2026-05-20).
    """
    return f"/tmp/o5_sync_{workspace.name}_{pid}.tar"


def test_remote_tar_is_op_unique() -> None:
    """Different op workspaces produce different remote tar paths."""
    pid = 12345
    p1 = _build_remote_tar_path(Path("/foo/bar/22_Nonzero"), pid)
    p2 = _build_remote_tar_path(Path("/foo/bar/31_IOU"), pid)
    assert p1 != p2, f"op-unique invariant violated: {p1} == {p2}"
    assert "22_Nonzero" in p1
    assert "31_IOU" in p2


def test_remote_tar_is_pid_unique() -> None:
    """Two processes running the same op (e.g. reruns) hold different tars."""
    ws = Path("/foo/bar/22_Nonzero")
    p1 = _build_remote_tar_path(ws, 12345)
    p2 = _build_remote_tar_path(ws, 67890)
    assert p1 != p2, f"pid-unique invariant violated: {p1} == {p2}"


def test_remote_tar_not_old_hardcoded_path() -> None:
    """Guard against accidental revert to /tmp/o5_sync.tar bare."""
    p = _build_remote_tar_path(Path("/foo/bar/22_Nonzero"), os.getpid())
    assert p != "/tmp/o5_sync.tar"
    assert re.fullmatch(r"/tmp/o5_sync_[A-Za-z0-9_]+_\d+\.tar", p), p


def test_phase_o5_runner_uses_unique_path_in_source() -> None:
    """Source-level contract pin: phase_o5_runner.py must NOT reference
    the bare `/tmp/o5_sync.tar` path (would re-introduce the race).
    """
    src = (_reorg_paths.ORCH_DIR
           / "phase_o5_runner.py").read_text()
    # Bare hardcoded path must not appear as a string literal anywhere.
    assert '"/tmp/o5_sync.tar"' not in src, \
        "phase_o5_runner.py still has bare /tmp/o5_sync.tar — P0aca race regressed"
    assert "'/tmp/o5_sync.tar'" not in src, \
        "phase_o5_runner.py still has bare /tmp/o5_sync.tar — P0aca race regressed"
    # Either the f-string form OR the variable must be present.
    assert "o5_sync_{workspace.name}" in src or "_remote_tar" in src, \
        "phase_o5_runner.py missing the parameterized remote tar pattern"


if __name__ == "__main__":
    test_remote_tar_is_op_unique()
    test_remote_tar_is_pid_unique()
    test_remote_tar_not_old_hardcoded_path()
    test_phase_o5_runner_uses_unique_path_in_source()
    logging.info("OK")
