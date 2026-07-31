# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression: raw cross-archive read guard (DEBT-HERMETIC-PROBE).

Exercises the real output_read_guard.py PreToolUse hook via subprocess with a crafted
.opgen_state.json (ASCENDC_WORKSPACE override). Exit 2 = blocked, exit 0 = allowed.

The load-bearing case is `test_resume_same_op_other_soc_denied` — op-name scoping alone
would ALLOW the same op's other-SoC port (the cross-SoC cheat owner's rule targets);
this confirms the full-archive-path scoping denies it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_GUARD = Path(__file__).resolve().parents[1] / "output_read_guard.py"


def _run(state: dict, tool_event: dict, tmp_path, agent_id="sub-abc123") -> int:
    """Run the guard via subprocess. By DEFAULT injects an `agent_id` (simulating a
    SUBAGENT), because the guard only restricts subagent context. Pass agent_id=None to
    simulate the MAIN/coordinator agent (which must never be restricted)."""
    (tmp_path / ".opgen_state.json").write_text(json.dumps(state))
    ev = dict(tool_event)
    if agent_id is not None and "agent_id" not in ev:
        ev["agent_id"] = agent_id
    proc = subprocess.run(
        [sys.executable, str(_GUARD)],
        input=json.dumps(ev),
        text=True, capture_output=True,
        env={**os.environ, "ASCENDC_WORKSPACE": str(tmp_path)},
    )
    return proc.returncode


_FRESH = {"op": "flash_attention_score", "mode": "default", "build_archive_dir": ""}
_RESUME = {"op": "flash_attention_score", "mode": "resume",
           "build_archive_dir": "output/a3_to_a5_port/src/kernels/flash_attention_score"}


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_fresh_other_op_output_denied(tmp_path):
    assert _run(_FRESH, {"tool_name": "Read", "tool_input": {
        "file_path": "output/a3_to_a5_port/src/kernels/other_op/op_kernel/x.h"}}, tmp_path) == 2


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_fresh_cann_source_allowed(tmp_path):
    # CANN source is OUTSIDE output/ → never touched (owner-confirmed sanctioned)
    assert _run(_FRESH, {"tool_name": "Read", "tool_input": {
        "file_path": "/home/npu_user/workspace/cann/ops-transformer/flash.h"}}, tmp_path) == 0


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_fresh_own_workspace_allowed(tmp_path):
    assert _run(_FRESH, {"tool_name": "Read", "tool_input": {
        "file_path": "workspace/flash_attention_score/model.py"}}, tmp_path) == 0


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_fresh_bash_cat_output_denied(tmp_path):
    assert _run(_FRESH, {"tool_name": "Bash", "tool_input": {
        "command": "cat output/backward_ops/src/kernels/flash_attention_score/kernel/x.cpp"}}, tmp_path) == 2


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_resume_own_archive_allowed(tmp_path):
    assert _run(_RESUME, {"tool_name": "Read", "tool_input": {
        "file_path": "output/a3_to_a5_port/src/kernels/flash_attention_score/op_kernel/x.h"}}, tmp_path) == 0


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_resume_same_op_other_soc_denied(tmp_path):
    # THE load-bearing case: same op, DIFFERENT SoC project → cross-SoC cheat → DENY.
    assert _run(_RESUME, {"tool_name": "Read", "tool_input": {
        "file_path": "output/backward_ops/src/kernels/flash_attention_score/kernel/x.cpp"}}, tmp_path) == 2


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_non_reader_bash_ignored(tmp_path):
    # a Bash that doesn't read content (no cat/head/etc.) → not our concern, allowed
    assert _run(_FRESH, {"tool_name": "Bash", "tool_input": {
        "command": "ls output/a3_to_a5_port/src/kernels/"}}, tmp_path) == 0


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_main_agent_unrestricted(tmp_path):
    # THE scoping case: main/coordinator agent (NO agent_id in payload) reads another
    # op's archive → ALLOWED. The guard restricts subagents only; main legitimately
    # reads output/ archives for PR review + verification. (Same event that is DENIED
    # for a subagent in test_fresh_other_op_output_denied.)
    assert _run(_FRESH, {"tool_name": "Read", "tool_input": {
        "file_path": "output/a3_to_a5_port/src/kernels/other_op/op_kernel/x.h"}},
        tmp_path, agent_id=None) == 0


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_main_agent_unrestricted_bash_cat(tmp_path):
    # main agent Bash cat of another op's archive → ALLOWED (subagent-scoped guard).
    assert _run(_FRESH, {"tool_name": "Bash", "tool_input": {
        "command": "cat output/backward_ops/src/kernels/flash_attention_score/kernel/x.cpp"}},
        tmp_path, agent_id=None) == 0


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_subagent_explicit_agent_id_denied(tmp_path):
    # explicit subagent agent_id (as the real payload carries) + cross-dir read → DENIED.
    assert _run(_FRESH, {"tool_name": "Read", "agent_id": "a7caf95110c02d0c7",
        "agent_type": "aog-kernel-worker", "tool_input": {
        "file_path": "output/a3_to_a5_port/src/kernels/other_op/op_kernel/x.h"}},
        tmp_path, agent_id=None) == 2


# --------------------------------------------------------------------------------------------
# A2 (2026-07-22): git READ verbs. `git show <sha>:output/.../<op>/model_new_*.py` must be
# DENIED for a fresh/cold subagent (the durable defense-in-depth backstop for the non-git
# export A1). Harness-code git reads + metadata verbs + own-archive resume reads stay ALLOWED.
# --------------------------------------------------------------------------------------------

_GDR_FRESH = {"op": "gated_delta_rule", "opgen_mode": "port_a3_to_a5", "build_archive_dir": ""}
_SHA = "0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b"


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_gdr_output_direct_read_denied(tmp_path):
    # DIRECTORY-CONVENTION prevention (direct side): a fresh/cold subagent CANNOT reach the GDR
    # answer under output/ by a direct file read — it must read customer inputs from input/ instead.
    assert _run(_GDR_FRESH, {"tool_name": "Read", "tool_input": {"file_path":
        "output/a3_to_a5_port/src/kernels/gated_delta_rule/model_new_ascendc.py"}},
        tmp_path) == 2


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_gdr_input_mirror_read_allowed(tmp_path):
    # The positive source: reading the op's customer inputs from the input/ mirror is ALLOWED
    # (input/ is not under output/, so the raw cross-archive guard never touches it).
    assert _run(_GDR_FRESH, {"tool_name": "Read", "tool_input": {"file_path":
        "input/arch22/src/kernels/gated_delta_rule/model.py"}},
        tmp_path) == 0


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_git_show_output_op_answer_denied(tmp_path):
    # THE case A2 targets: read the prior delivered kernel from history → DENIED (fresh subagent).
    assert _run(_GDR_FRESH, {"tool_name": "Bash", "tool_input": {"command":
        f"git show {_SHA}:output/a3_to_a5_port/src/kernels/gated_delta_rule/model_new_ascendc.py"}},
        tmp_path) == 2


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_git_show_src_path_allowed(tmp_path):
    # targeted read of harness code (outside output/) → ALLOWED.
    assert _run(_GDR_FRESH, {"tool_name": "Bash", "tool_input": {"command":
        f"git show {_SHA}:src/scripts/orchestrator/foo.py"}}, tmp_path) == 0


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_git_status_allowed(tmp_path):
    assert _run(_GDR_FRESH, {"tool_name": "Bash", "tool_input": {"command": "git status"}}, tmp_path) == 0


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_git_rev_parse_allowed(tmp_path):
    assert _run(_GDR_FRESH, {"tool_name": "Bash", "tool_input": {"command": "git rev-parse HEAD"}}, tmp_path) == 0


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_git_log_no_patch_allowed(tmp_path):
    # `git log` without -p = commit messages only (no content dump) → ALLOWED.
    assert _run(_GDR_FRESH, {"tool_name": "Bash", "tool_input": {"command": "git log --oneline -5"}}, tmp_path) == 0


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_git_bare_show_fresh_denied(tmp_path):
    # bare commit-ish content dump during fresh gen (could expose the answer) → DENIED.
    assert _run(_GDR_FRESH, {"tool_name": "Bash", "tool_input": {"command": f"git show {_SHA}"}}, tmp_path) == 2


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_git_main_agent_unrestricted(tmp_path):
    # main/coordinator (no agent_id) git-reads the answer from history → ALLOWED (subagent-scoped).
    assert _run(_GDR_FRESH, {"tool_name": "Bash", "tool_input": {"command":
        f"git show {_SHA}:output/a3_to_a5_port/src/kernels/gated_delta_rule/model_new_ascendc.py"}},
        tmp_path, agent_id=None) == 0


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_git_resume_own_archive_read_allowed(tmp_path):
    # --resume reading THIS op's OWN tracked archive via git → ALLOWED (iterate mode).
    assert _run(_RESUME, {"tool_name": "Bash", "tool_input": {"command":
        "git show HEAD:output/a3_to_a5_port/src/kernels/flash_attention_score/op_kernel/x.h"}},
        tmp_path) == 0


@pytest.mark.skipif(not _GUARD.is_file(), reason="output_read_guard.py absent")
def test_git_grep_other_op_output_denied(tmp_path):
    # git grep into another op's archive tree → DENIED (fresh subagent).
    assert _run(_GDR_FRESH, {"tool_name": "Bash", "tool_input": {"command":
        f"git grep EmitSoftmaxOut {_SHA} -- output/a3_to_a5_port/src/kernels/other_op/"}}, tmp_path) == 2
