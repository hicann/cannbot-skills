#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P1-3 + P1-2 (2026-08-28, DSH+v4pro §4.a / FAS P1-3) — kw brief additions.

P1-3: user_decision.md is surfaced to the kernel worker (visibility only —
consumption semantics unchanged).  When the decision was already consumed,
the brief references `.user_decision_consumed.md` and the P0ff extract
`kb_draft_from_user_decision.md` read-only.

P1-2: the NPUKernelBench worker brief carries the replay-legalization
clause — replaying bytes from the worker's OWN stream logs (incl. prior
spawns of the same op) is allowed with a PROGRESS.md source declaration +
SHA self-verification; reading the sealed `.opgen_backups` archive stays
forbidden.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from briefs import kw_brief  # noqa: E402
from briefs.kw_brief import (  # noqa: E402
    _npubench_replay_policy_block,
    _user_decision_visibility_block,
    _worker_brief_context_sections,
)
from briefs._common import AscendCEnv  # noqa: E402
from npubench.npubench_inputs import stage_npubench_inputs  # noqa: E402


def _make_env(**kwargs) -> AscendCEnv:
    """Minimal AscendCEnv for tests (same shape as test_kw_brief_port_a3)."""
    default_items = (
        ("target", "a5"),
        ("host", "198.51.100.35"),
        ("user", "root"),
        ("password", ""),
        ("container", "npu_dev3"),
        ("cann_path", "/data/cann_b103/cann-9.0.0"),
        ("soc_version", "Ascend950PR_9579"),
        ("benchmark_root", "/root/AscendOpGenAgent"),
        ("local_benchmark", "/tmp/bench"),
        ("local_project", "/tmp/proj"),
        ("archive_project", "test_archive"),
        ("build_archive_enabled", False),
        ("opgen_mode", "port_a3_to_a5"),
        ("port_a3_source", ""),
        ("a3_host", ""),
        ("a3_user", "root"),
        ("a3_container", ""),
        ("a3_cann_path", ""),
        ("a3_soc_version", ""),
        ("a3_workspace", ""),
    )
    defaults = dict(default_items)
    defaults.update(kwargs)
    return AscendCEnv(**defaults)


def _make_npubench_workspace(tmp_path: Path, op: str = "add") -> Path:
    """One byte-preserved old-format NPUKernelBench binding (see W5 tests)."""
    workspace = tmp_path.joinpath("workspace", op)
    task_root = tmp_path.joinpath("npu_benchmark", "level1")
    for directory in (workspace, task_root):
        directory.mkdir(parents=True)
    task = task_root.joinpath("3_Add.py")
    task.write_text(
        "\n".join(
            (
                "from pathlib import Path",
                "def get_input_groups():",
                "    return [Path(__file__).with_suffix('.json').read_bytes()]",
            )
        )
        + "\n"
    )
    task.with_suffix(".json").write_bytes(b'{"shape": [8]}\n{"shape": [16]}\n')
    reference = stage_npubench_inputs(
        workspace, npubench_task=task, npubench_root=task_root
    ).state_block()
    state = {"opgen_mode": "port_a3_to_a5", "reference": reference}
    (workspace / ".opgen_state.json").write_text(json.dumps(state))
    return workspace


# ---------------------------------------------------------------------------
# P1-3 — user_decision visibility block
# ---------------------------------------------------------------------------
def test_user_decision_block_empty_without_artifacts(tmp_path):
    assert _user_decision_visibility_block("add", tmp_path) == ""


def test_user_decision_block_live_file(tmp_path):
    (tmp_path / "user_decision.md").write_text("continue with cap 18\n")
    block = _user_decision_visibility_block("add", tmp_path)
    assert "workspace/add/user_decision.md" in block
    assert "read" in block.lower()
    assert "Do not modify" in block


def test_user_decision_block_consumed_references(tmp_path):
    (tmp_path / ".user_decision_consumed.md").write_text("consumed\n")
    (tmp_path / "kb_draft_from_user_decision.md").write_text("draft\n")
    block = _user_decision_visibility_block("add", tmp_path)
    assert ".user_decision_consumed.md" in block
    assert "kb_draft_from_user_decision.md" in block
    assert "read-only" in block


def test_user_decision_block_live_wins_over_consumed(tmp_path):
    (tmp_path / "user_decision.md").write_text("fresh decision\n")
    (tmp_path / ".user_decision_consumed.md").write_text("stale consumed\n")
    block = _user_decision_visibility_block("add", tmp_path)
    assert "workspace/add/user_decision.md" in block
    assert ".user_decision_consumed.md" not in block


def test_legacy_context_sections_include_user_decision(tmp_path):
    ws = tmp_path / "workspace" / "add"
    ws.mkdir(parents=True)
    (ws / "user_decision.md").write_text("steering\n")
    sections = _worker_brief_context_sections(
        "add", ws, env=_make_env(), lane=0, slug="slug"
    )
    assert any("user_decision.md" in section for section in sections)


def test_legacy_context_sections_byte_identical_without_decision(tmp_path):
    ws = tmp_path / "workspace" / "add"
    ws.mkdir(parents=True)
    sections = _worker_brief_context_sections(
        "add", ws, env=_make_env(), lane=0, slug="slug"
    )
    assert not any("USER DECISION" in section for section in sections)


def test_npubench_brief_surfaces_user_decision(tmp_path):
    ws = _make_npubench_workspace(tmp_path)
    (ws / "user_decision.md").write_text("repair mandate\n")
    brief = kw_brief.build_worker_brief(
        op="add", workspace=ws, lane=0, spawn_index=2,
        iter_cap_remaining=3, env=_make_env(),
    )
    assert "workspace/add/user_decision.md" in brief


def test_npubench_brief_references_consumed_decision(tmp_path):
    ws = _make_npubench_workspace(tmp_path)
    (ws / ".user_decision_consumed.md").write_text("consumed\n")
    (ws / "kb_draft_from_user_decision.md").write_text("draft\n")
    brief = kw_brief.build_worker_brief(
        op="add", workspace=ws, lane=0, spawn_index=2,
        iter_cap_remaining=3, env=_make_env(),
    )
    assert ".user_decision_consumed.md" in brief
    assert "kb_draft_from_user_decision.md" in brief


# ---------------------------------------------------------------------------
# P1-2 — stream-log replay legalization clause
# ---------------------------------------------------------------------------
def test_replay_policy_block_content():
    block = _npubench_replay_policy_block()
    # Allowed: own stream-log replay with PROGRESS.md declaration + SHA proof.
    assert "ALLOWED" in block
    assert "PROGRESS.md" in block
    assert "SHA-256" in block
    # Forbidden: the sealed archive.
    assert ".opgen_backups" in block
    assert "SEALED" in block


def test_npubench_brief_carries_replay_policy(tmp_path):
    ws = _make_npubench_workspace(tmp_path)
    brief = kw_brief.build_worker_brief(
        op="add", workspace=ws, lane=0, spawn_index=1,
        iter_cap_remaining=3, env=_make_env(),
    )
    assert "CANDIDATE RE-AUTHORING POLICY" in brief
    assert ".opgen_backups" in brief
