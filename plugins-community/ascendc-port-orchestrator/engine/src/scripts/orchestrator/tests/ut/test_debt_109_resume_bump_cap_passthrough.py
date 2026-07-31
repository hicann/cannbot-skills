# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DEBT-109 regression: resume.execute() must pass --bump-cap through to inner orchestrator.

Caught 2026-05-20 on gather_elements_v2 task #51: `src/scripts/orch <op> --resume
--bump-cap worker:3` silently dropped the flag, inner orchestrator hit iter_cap
immediately. Without bump-cap passthrough, --resume on a cap-exhausted workspace
is unrecoverable except by direct orchestrator.py invocation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import resume  # noqa: E402


def _scoped_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace" / "test_op"
    workspace.mkdir(parents=True)
    (workspace / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 1,
        "op": "test_op",
        "opgen_mode": "backward",
    }))
    return workspace


@patch("resume.subprocess.call", return_value=0)
@patch("resume.diagnose")
def test_resume_passes_bump_cap_to_inner_orchestrator(mock_diag, mock_call, tmp_path):
    """When bump_cap=['worker:5'] is passed, the subprocess cmd must include
    --bump-cap worker:5.
    """
    fake = MagicMock()
    fake.action = resume.ResumeAction.RESUMABLE
    fake.workspace = _scoped_workspace(tmp_path)
    fake.current_state = "await_worker"
    mock_diag.return_value = fake

    resume.execute("test_op", workspace=fake.workspace, lane=1,
                   bump_cap=["worker:5"])

    cmd = mock_call.call_args[0][0]
    assert "--bump-cap" in cmd, f"--bump-cap missing from subprocess cmd: {cmd}"
    bc_idx = cmd.index("--bump-cap")
    assert cmd[bc_idx + 1] == "worker:5", (
        f"--bump-cap value wrong: got {cmd[bc_idx + 1]!r}, expected 'worker:5'"
    )


@patch("resume.subprocess.call", return_value=0)
@patch("resume.diagnose")
def test_resume_passes_multiple_bump_caps(mock_diag, mock_call, tmp_path):
    """Multiple bump-cap flags must all be passed through."""
    fake = MagicMock()
    fake.action = resume.ResumeAction.RESUMABLE
    fake.workspace = _scoped_workspace(tmp_path)
    fake.current_state = "await_worker"
    mock_diag.return_value = fake

    resume.execute("test_op", workspace=fake.workspace, lane=1,
                   bump_cap=["worker:5", "optimizer:2"])

    cmd = mock_call.call_args[0][0]
    # Both --bump-cap occurrences with their respective values must be present
    bc_idxs = [i for i, x in enumerate(cmd) if x == "--bump-cap"]
    assert len(bc_idxs) == 2, f"expected 2 --bump-cap, got {len(bc_idxs)} in {cmd}"
    values = {cmd[i + 1] for i in bc_idxs}
    assert values == {"worker:5", "optimizer:2"}, f"got {values}"


@patch("resume.subprocess.call", return_value=0)
@patch("resume.diagnose")
def test_resume_no_bump_cap_means_no_flag(mock_diag, mock_call, tmp_path):
    """Default (no bump_cap) must NOT include --bump-cap in cmd."""
    fake = MagicMock()
    fake.action = resume.ResumeAction.RESUMABLE
    fake.workspace = _scoped_workspace(tmp_path)
    fake.current_state = "await_worker"
    mock_diag.return_value = fake

    resume.execute("test_op", workspace=fake.workspace, lane=1)

    cmd = mock_call.call_args[0][0]
    assert "--bump-cap" not in cmd, f"--bump-cap leaked into cmd: {cmd}"
