# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""orchestrator `--optimize` — FSM-owned done→optimize re-entry.

Wires the state-machine spec's `optimize` MODE (opgen_state_machine.yaml:242) that the
CLI never implemented. Re-opens an already-VERIFIED op into the optimizer loop WITHOUT
re-generating (the consumer is back's (B) step-2 perf-rewrite, 2026-06-21).

Contract (back/main 2026-06-21) — the inverse of cold-start (which WIPES kw output):
  PRESERVE: kernel/ + verification.json + the driver's optimization_directive.md
    (the optimize-loop tunes/rewrites ON the verified step-1 kernel, not from scratch).
  REQUIRE:  kernel/ + verification.json + optimization_directive.md present — else
    hard-reject (can't optimize a non-existent / un-directed kernel). The flag does NOT
    create the directive (would clobber the driver's).
  RE-ENTER: a FRESH state_transitions.jsonl whose tail is await_optimizer → fresh
    per-state iter-cap; reset lifetime_spawn_count=0 → fresh spawn budget (no --bump-cap).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import orchestrator as orch  # noqa: E402


def _mk_verified_op(ws: Path, *, kernel=True, vj=True, directive=True,
                    prior_spawns=11) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    if kernel:
        (ws / "kernel").mkdir(exist_ok=True)
        (ws / "kernel" / "k.h").write_text("// verified step-1 kernel\n")
    if vj:
        (ws / "verification.json").write_text('{"precision":{"status":"PASS"}}')
    if directive:
        (ws / "optimization_directive.md").write_text(
            "# step-2 directive (driver-authored)\nLever B: stage-fusion (structural)\n")
    # simulate prior terminal state + accumulated spawns
    (ws / "state_transitions.jsonl").write_text(
        '{"ts":"t","from_state":"finalize","to_state":"done","handoff":""}\n')
    # opgen_mode must be CONSISTENT with the built-kernel layout this fixture
    # creates: it lays down `kernel/`, which is the backward workflow layout.
    # (Previously declared `port_a3_to_a5` — an inconsistent
    # combo that was invisible while the --optimize prereq only checked `kernel/`
    # name-blindly; the plugin-aware prereq now asks the mode's kernel_cpp_dirs(),
    # and port_a3 mode expects op_host/+op_kernel/, not kernel/.)
    (ws / ".opgen_state.json").write_text(
        json.dumps({"lifetime_spawn_count": prior_spawns, "opgen_mode": "backward"}))
    (ws / ".finalized-1234").write_text("done")


@pytest.fixture(autouse=True)
def _backup_root(tmp_path, monkeypatch):
    # keep state backups out of $HOME during tests
    monkeypatch.setenv("COLD_START_BACKUP_ROOT", str(tmp_path / "backups"))


def test_optimize_reentry_accepts_valid_target(tmp_path):
    ws = tmp_path / "deformable_conv2d"
    _mk_verified_op(ws)
    ok, msg = getattr(orch, '_optimize_reentry_workspace')(ws)
    assert ok is True, msg
    log = [json.loads(l) for l in (ws / "state_transitions.jsonl").read_text().splitlines()]
    # fresh log entering at await_optimizer (single entry → fresh per-state iter-cap)
    assert len(log) == 1 and log[-1]["to_state"] == "await_optimizer"
    # fresh spawn budget
    st = json.loads((ws / ".opgen_state.json").read_text())
    assert st["lifetime_spawn_count"] == 0
    # PRESERVE verified kernel + driver directive; clear terminal marker
    assert (ws / "kernel" / "k.h").read_text() == "// verified step-1 kernel\n"
    assert (ws / "optimization_directive.md").is_file()
    assert not list(ws.glob(".finalized*"))


def test_optimize_reentry_requires_existing_directive(tmp_path):
    ws = tmp_path / "no_directive"
    _mk_verified_op(ws, directive=False)
    ok, msg = getattr(orch, '_optimize_reentry_workspace')(ws)
    assert ok is False
    assert "optimization_directive.md" in msg
    # must NOT have created the directive (would clobber the driver's)
    assert not (ws / "optimization_directive.md").exists()


def test_optimize_reentry_requires_verified_kernel(tmp_path):
    ws_nk = tmp_path / "no_kernel"
    _mk_verified_op(ws_nk, kernel=False)
    assert getattr(orch, '_optimize_reentry_workspace')(ws_nk)[0] is False
    ws_nv = tmp_path / "no_vj"
    _mk_verified_op(ws_nv, vj=False)
    assert getattr(orch, '_optimize_reentry_workspace')(ws_nv)[0] is False


def test_optimize_reentry_current_state_resolves_to_await_optimizer(tmp_path):
    """End-to-end via the real state-machine reader: get_current_state must return
    await_optimizer after re-entry (this is what run_single_op's snapshot reads).
    """
    sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))
    import state_machine as sm  # noqa: E402
    ws = tmp_path / "iou_v2"
    _mk_verified_op(ws)
    assert getattr(orch, '_optimize_reentry_workspace')(ws)[0] is True
    machine = sm.load_state_machine()
    assert sm.get_current_state(ws, machine) == "await_optimizer"


# --- P0v integration: the re-entry directive must survive pre-spawn archiving ---
# (back's live integration-test on (B) step-2 found _archive_stale_outputs_before_spawn
#  archived the driver's CURRENT optimization_directive.md as if stale → directive-less
#  ko/kw brief. The .optimize_active marker suppresses that for the optimize phase.)

def test_optimize_reentry_drops_optimize_active_marker(tmp_path):
    ws = tmp_path / "iou_v2"
    _mk_verified_op(ws)
    assert getattr(orch, '_optimize_reentry_workspace')(ws)[0] is True
    assert (ws / ".optimize_active").is_file()


@pytest.mark.parametrize("state", ["await_optimizer", "await_researcher", "await_fused_optimizer"])
def test_archive_preserves_directive_when_optimize_active(tmp_path, state):
    """With the .optimize_active marker, optimization_directive.md is NOT archived in any
    of the directive-consuming states (the ko→researcher→kw chain keeps the live directive).
    """
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / "optimization_directive.md").write_text("# driver step-2 directive\n")
    (ws / ".optimize_active").write_text("active")
    getattr(orch, '_archive_stale_outputs_before_spawn')(ws, state, 1)
    assert (ws / "optimization_directive.md").is_file(), \
        f"directive must survive archiving in {state} while --optimize is active"
    assert not list(ws.glob(".pre-*optimization_directive.md"))


def test_archive_still_archives_directive_without_marker(tmp_path):
    """Normal flow (no marker): archiving is unchanged — the directive IS archived in
    await_optimizer (load-bearing for the FSM's path_exists transitions; P0v contract).
    """
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / "optimization_directive.md").write_text("# stale prior-session directive\n")
    getattr(orch, '_archive_stale_outputs_before_spawn')(ws, "await_optimizer", 1)
    assert not (ws / "optimization_directive.md").is_file(), \
        "without --optimize marker, normal P0v archiving must still fire"
    assert list(ws.glob(".pre-await_optimizer-1-*optimization_directive.md"))
