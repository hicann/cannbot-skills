# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0v (2026-05-05): archive stale optimization_directive.md before agent spawn.

Origin: op#9 TopKTopP 2026-05-05. Researcher (ar-2) wrote
cann_strategy_inference.md and emitted PARTIAL_PERSIST handoff. State machine
matched `path_exists: workspace/{op}/optimization_directive.md` (V3.3.4 B-fix
path) on a STALE directive from prior pp-3/ko-1 sessions, routed to
await_worker. Worker correctly diagnosed false-match, refused to spawn yet
another worker iter on already-exhausted paths, emitted prose summary →
abort.

Fix: orchestrator.run_single_op archives stale "optional" outputs (those
that path_exists checks evaluate against in exit_transitions) BEFORE agent
spawn. Per-state archive list:
  await_researcher: optimization_directive.md, research_report.md
  await_optimizer:  optimization_directive.md
  await_fused_optimizer: optimization_directive.md
  await_probe: probe_report.md, probe_result.json

Result: state machine's path_exists check after agent return only sees
files THIS spawn wrote.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import orchestrator as orch  # noqa: E402


def test_p0v_archives_stale_directive_for_await_researcher(tmp_path):
    """Stale optimization_directive.md present before await_researcher spawn
    → archived with .pre-await_researcher-N-<ts>- prefix; path no longer exists.
    """
    directive = tmp_path / "optimization_directive.md"
    directive.write_text("# stale directive from prior session")
    research_report = tmp_path / "research_report.md"
    research_report.write_text("# stale research report")

    getattr(orch, '_archive_stale_outputs_before_spawn')(tmp_path, "await_researcher", spawn_index=2)

    assert not directive.exists(), "stale optimization_directive.md should be archived"
    assert not research_report.exists(), "stale research_report.md should be archived"

    archived = list(tmp_path.glob(".pre-await_researcher-2-*-optimization_directive.md"))
    assert len(archived) == 1, f"expected 1 archived directive, found {archived}"
    archived_research = list(tmp_path.glob(".pre-await_researcher-2-*-research_report.md"))
    assert len(archived_research) == 1


def test_p0v_archives_stale_probe_for_await_probe(tmp_path):
    """await_probe entry archives stale probe_report.md + probe_result.json."""
    (tmp_path / "probe_report.md").write_text("# stale")
    (tmp_path / "probe_result.json").write_text('{"classification": "stale"}')

    getattr(orch, '_archive_stale_outputs_before_spawn')(tmp_path, "await_probe", spawn_index=1)

    assert not (tmp_path / "probe_report.md").exists()
    assert not (tmp_path / "probe_result.json").exists()
    assert len(list(tmp_path.glob(".pre-await_probe-1-*-probe_report.md"))) == 1
    assert len(list(tmp_path.glob(".pre-await_probe-1-*-probe_result.json"))) == 1


def test_p0v_no_op_if_no_stale_files(tmp_path):
    """No stale files → no archive activity, no errors."""
    getattr(orch, '_archive_stale_outputs_before_spawn')(tmp_path, "await_researcher", spawn_index=1)
    archived = list(tmp_path.glob(".pre-*"))
    assert archived == []


def test_p0v_no_op_for_states_without_path_exists_rules(tmp_path):
    """Worker / det_analyzer states don't have path_exists exit_transitions
    that need stale-archive protection. Should be no-op.
    """
    (tmp_path / "optimization_directive.md").write_text("preserved")

    getattr(orch, '_archive_stale_outputs_before_spawn')(tmp_path, "await_worker", spawn_index=1)

    # File preserved (await_worker has no stale-archive list)
    assert (tmp_path / "optimization_directive.md").exists()


def test_p0v_archive_prefix_includes_state_and_spawn_index(tmp_path):
    """Archive filename includes state name + spawn_index for forensics."""
    (tmp_path / "optimization_directive.md").write_text("X")

    getattr(orch, '_archive_stale_outputs_before_spawn')(tmp_path, "await_optimizer", spawn_index=3)

    archived = list(tmp_path.glob(".pre-await_optimizer-3-*-optimization_directive.md"))
    assert len(archived) == 1
    # Confirm timestamp suffix exists
    name = archived[0].name
    assert name.startswith(".pre-await_optimizer-3-")
    assert name.endswith("-optimization_directive.md")


def test_p0v_does_not_archive_user_decision_md(tmp_path):
    """user_decision.md is owned by user — never archive."""
    (tmp_path / "user_decision.md").write_text("next_state: await_researcher")

    getattr(orch, '_archive_stale_outputs_before_spawn')(tmp_path, "await_researcher", spawn_index=1)

    assert (tmp_path / "user_decision.md").exists()


def test_p0v_state_specific_archive_list(tmp_path):
    """await_researcher archives directive + research_report; await_probe doesn't
    touch directive (different state's files).
    """
    (tmp_path / "optimization_directive.md").write_text("X")
    (tmp_path / "probe_report.md").write_text("Y")

    # Enter await_probe — should archive probe_report.md, NOT optimization_directive.md
    getattr(orch, '_archive_stale_outputs_before_spawn')(tmp_path, "await_probe", spawn_index=1)

    assert (tmp_path / "optimization_directive.md").exists()  # untouched (different state)
    assert not (tmp_path / "probe_report.md").exists()  # archived (its state)
