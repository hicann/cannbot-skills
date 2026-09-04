# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""orchestrator --cold-start flag — back up state + kw-output, preserve
true Phase O2.5 inputs.

Origin: 2026-05-05 parallel sweep for op#28_multimodal_rope hit a stuck
state (orchestrator bootstrapped from old "→ orchestrator: done" handoff
in PROGRESS.md, exited at finalize without re-evaluating under T1/T2).

Cold-start contract (P0aav, 2026-05-07 refinement):
  WIPE (back up to .pre-cold-start-<ts>/):
    - State machine artifacts: state_transitions.jsonl, PROGRESS.md
    - kw OUTPUT: kernel/, model_new_ascendc.py, analysis.md, knowledge_update.md
    - Per-iter artifacts: optimization_*, probe_*, self_critic_*
    - Markers: .agent_died_at_*, .kernel_worker_active, .finalized-*
    - User decisions: user_decision.md, .opgen_state.json
    - Stale claim: verification.json
  PRESERVE (true Phase O2.5 prep + benchmark inputs, NOT kw output):
    - model.py (copied from benchmark, not kw output)
    - input_gen.py, edge_inputs.pt, edge_dataset.pt, manifest.json (Phase O2.5)
    - scoped reference-provider state from Phase O2.5
    - op_classification.json (Phase O1.7 output)
    - <op>.json + model.json (benchmark dataset / sibling alias)
    - .ascendc_env (env config)

P0aav rationale: kw OUTPUT (kernel/, model_new_ascendc.py, analysis.md,
knowledge_update.md) is contaminated by prior iteration's KB / brief /
state. Preserving it across cold-start defeats the contract: the next kw
spawn would see partial outputs as resumable state.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import orchestrator as orch  # noqa: E402


def _seed_workspace_with_state(ws: Path):
    """Pre-populate workspace with the three categories cold-start must
    distinguish:

      A. State machine artifacts + kw outputs — these get backed up + wiped
         (per P0aav 2026-05-07 contract).
      B. True Phase O2.5 prep + benchmark inputs — preserved across
         cold-start because regenerating them is wasteful and they are
         input dependencies of the new run, not outputs of the old one.
    """
    # ---- Category A: state + kw output (must be backed up and wiped) ----
    (ws / "state_transitions.jsonl").write_text(json.dumps({
        "ts": "2026-05-04T10:00:00Z",
        "from_state": "init", "to_state": "finalize",
        "handoff": "→ orchestrator: done",
        "matched_transition_index": 0,
        "rationale": "old state",
        "iter_counts_snapshot": {},
    }) + "\n")
    (ws / "PROGRESS.md").write_text("# old progress\n→ orchestrator: done\n")
    (ws / "optimization_directive.md").write_text("old directive")
    (ws / "probe_report.md").write_text("old probe report")
    (ws / "probe_result.json").write_text('{"classification": "requirement"}')
    (ws / "knowledge_update.md").write_text("old kb update")  # kw output
    (ws / "self_critic_report.md").write_text("old critic report")
    (ws / ".agent_died_at_await_worker").write_text('{"reason": "old"}')
    (ws / ".kernel_worker_active").touch()
    # kw OUTPUTS (P0aav: contaminated by prior iter — must be wiped)
    (ws / "kernel").mkdir(exist_ok=True)
    (ws / "kernel" / "main.cpp").write_text("// kw output, must be wiped")
    (ws / "model_new_ascendc.py").write_text("# kw output, must be wiped")
    (ws / "analysis.md").write_text("# kw output, must be wiped")
    # ---- Category B: Phase O2.5 prep + benchmark inputs (preserved) ----
    (ws / "model.py").write_text("# benchmark input, must be preserved")
    (ws / "input_gen.py").write_text("# Phase O2.5, preserved")
    (ws / "manifest.json").write_text('{"coverage_tier": "sign_off"}')
    (ws / "edge_dataset.pt").write_bytes(b"phase O2.5 binary")
    (ws / "ref_runnable.json").write_text('{"verdict": "RUNNABLE"}')


def test_p0u_cold_start_backs_up_state_files(tmp_path, monkeypatch):
    monkeypatch.setenv("COLD_START_BACKUP_ROOT", str(tmp_path / ".bkp"))
    """--cold-start moves state files to .pre-cold-start-<ts>/ and removes
    from workspace root."""
    _seed_workspace_with_state(tmp_path)
    getattr(orch, '_cold_start_reset_workspace')(tmp_path)

    backups = list((tmp_path / ".bkp" / tmp_path.name).glob("pre-cold-start-*")
                   if (tmp_path / ".bkp" / tmp_path.name).exists() else [])
    assert len(backups) == 1
    backup_dir = backups[0]
    assert backup_dir.is_dir()
    # State files in backup
    assert (backup_dir / "state_transitions.jsonl").exists()
    assert (backup_dir / "PROGRESS.md").exists()
    assert (backup_dir / "optimization_directive.md").exists()
    assert (backup_dir / "probe_report.md").exists()
    assert (backup_dir / "probe_result.json").exists()
    assert (backup_dir / ".agent_died_at_await_worker").exists()
    # State files removed from workspace root
    assert not (tmp_path / "state_transitions.jsonl").exists()
    assert not (tmp_path / "PROGRESS.md").exists()
    assert not (tmp_path / "optimization_directive.md").exists()
    assert not (tmp_path / ".agent_died_at_await_worker").exists()
    assert not (tmp_path / ".kernel_worker_active").exists()


def test_debt078_cold_start_wipes_kernel_restore_sources(tmp_path, monkeypatch):
    """DEBT-078 (2026-06-13, owner '物理删 restore 来源'): cold-start must
    physically remove the KERNEL restore sources — `op_kernel/` dir,
    `kernel_snapshot.tgz`, and prior-build `*.so`/`*.tgz` — so a cold-start
    worker MUST regenerate from KB, not restore a prior kernel (fake autonomy).
    They are backed up (not destroyed), and true inputs survive.
    """
    monkeypatch.setenv("COLD_START_BACKUP_ROOT", str(tmp_path / ".bkp"))
    # restore sources (must be wiped)
    (tmp_path / "op_kernel").mkdir()
    (tmp_path / "op_kernel" / "flash_attention_score_kernels.cpp").write_text("// prior gen")
    (tmp_path / "kernel_snapshot.tgz").write_bytes(b"PRIOR-SNAPSHOT")
    (tmp_path / "libfa.so").write_bytes(b"PRIOR-BUILD-SO")
    (tmp_path / "extra_snapshot.tgz").write_bytes(b"X")
    # true input (must survive)
    (tmp_path / "manifest.json").write_text("{}")

    getattr(orch, '_cold_start_reset_workspace')(tmp_path)

    # restore sources gone from workspace
    assert not (tmp_path / "op_kernel").exists()
    assert not (tmp_path / "kernel_snapshot.tgz").exists()
    assert not (tmp_path / "libfa.so").exists()
    assert not (tmp_path / "extra_snapshot.tgz").exists()
    # backed up, not destroyed
    backups = list((tmp_path / ".bkp" / tmp_path.name).glob("pre-cold-start-*"))
    assert len(backups) == 1
    bk = backups[0]
    assert (bk / "op_kernel").exists()
    assert (bk / "kernel_snapshot.tgz").exists()
    assert (bk / "libfa.so").exists()
    # input preserved
    assert (tmp_path / "manifest.json").exists()


def test_p0u_cold_start_preserves_phase_o25_prep(tmp_path):
    """True Phase O2.5 prep + benchmark inputs survive cold-start.
    `kernel/`, `model_new_ascendc.py`, `analysis.md` are kw OUTPUTS, not
    Phase O2.5 prep — they get wiped (P0aav 2026-05-07).
    """
    _seed_workspace_with_state(tmp_path)
    getattr(orch, '_cold_start_reset_workspace')(tmp_path)

    # Preserved: Phase O2.5 prep + benchmark inputs
    assert (tmp_path / "model.py").read_text() == "# benchmark input, must be preserved"
    assert (tmp_path / "input_gen.py").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "edge_dataset.pt").read_bytes() == b"phase O2.5 binary"
    assert (tmp_path / "ref_runnable.json").exists()
    # Wiped: kw outputs (P0aav)
    assert not (tmp_path / "kernel").exists()
    assert not (tmp_path / "model_new_ascendc.py").exists()
    assert not (tmp_path / "analysis.md").exists()


def test_p0u_cold_start_idempotent_with_no_state(tmp_path, monkeypatch):
    monkeypatch.setenv("COLD_START_BACKUP_ROOT", str(tmp_path / ".bkp"))
    """When workspace has only Phase O2.5 prep + no state/output files,
    cold-start creates an empty backup dir and leaves prep untouched.
    `kernel/` here is treated as kw output — if it exists, it gets wiped."""
    (tmp_path / "model.py").write_text("# preserved")
    (tmp_path / "input_gen.py").write_text("# preserved")

    getattr(orch, '_cold_start_reset_workspace')(tmp_path)

    backups = list((tmp_path / ".bkp" / tmp_path.name).glob("pre-cold-start-*")
                   if (tmp_path / ".bkp" / tmp_path.name).exists() else [])
    assert len(backups) == 1
    # Backup is empty (no state or kw output to back up)
    assert list(backups[0].iterdir()) == []
    # Phase O2.5 prep untouched
    assert (tmp_path / "model.py").exists()
    assert (tmp_path / "input_gen.py").exists()


def test_p0u_cold_start_leaves_already_cleaned_markers_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("COLD_START_BACKUP_ROOT", str(tmp_path / ".bkp"))
    """Markers with .cleaned-<ts> suffix (post-recovery archives from P0r/P0t)
    should NOT be moved by cold-start — they're already-archived audit trail."""
    (tmp_path / ".agent_died_at_await_researcher.cleaned-1234567890").write_text('{"old": true}')
    (tmp_path / ".agent_died_at_await_worker").write_text('{"new": true}')

    getattr(orch, '_cold_start_reset_workspace')(tmp_path)

    # Cleaned marker still in place (not moved)
    assert (tmp_path / ".agent_died_at_await_researcher.cleaned-1234567890").exists()
    # New marker moved to backup
    backups = list((tmp_path / ".bkp" / tmp_path.name).glob("pre-cold-start-*")
                   if (tmp_path / ".bkp" / tmp_path.name).exists() else [])
    assert len(backups) == 1
    assert (backups[0] / ".agent_died_at_await_worker").exists()
    assert not (tmp_path / ".agent_died_at_await_worker").exists()


def test_cold_start_migrates_empirical_backup_out_of_workspace(tmp_path, monkeypatch):
    """Cold-start must
    migrate ALL in-workspace backup/snapshot dirs OUT — not just
    `.pre-cold-start-*`. A stale `.empirical_backup_*` snapshot that survives
    in-workspace becomes a RESTORE-TARGET: an agent can restore an old
    block_N=64 kernel instead of generating from the current block_N=128 design,
    so op-gen never produces the
    fast kernel. Any in-workspace backup dir defeats the clean-slate contract.
    """
    monkeypatch.setenv("COLD_START_BACKUP_ROOT", str(tmp_path / ".bkp"))
    # stale snapshot dirs that must NOT survive in-workspace
    (tmp_path / ".empirical_backup_2026_05_26").mkdir()
    (tmp_path / ".empirical_backup_2026_05_26" / "kernel").mkdir()
    (tmp_path / ".empirical_backup_2026_05_26" / "kernel" / "fa.h").write_text(
        "constexpr int FA_BLOCK_N = 64;  // stale slow kernel — must not be restorable")
    (tmp_path / ".pre_kw9_restore_1779565047").mkdir()
    (tmp_path / ".pre_kw9_restore_1779565047" / "old.h").write_text("// stale")
    # preserved Phase O2.5 prep
    (tmp_path / "model.py").write_text("# preserved")

    getattr(orch, '_cold_start_reset_workspace')(tmp_path)

    # No restore-target snapshot dir left in workspace
    assert not (tmp_path / ".empirical_backup_2026_05_26").exists(), \
        "empirical_backup survived cold-start → translator can restore stale block_N=64"
    assert not (tmp_path / ".pre_kw9_restore_1779565047").exists()
    # Migrated OUT to backup root (audit trail preserved, not deleted)
    migrated = list((tmp_path / ".bkp" / tmp_path.name).glob("empirical_backup_2026_05_26-migrated-*"))
    assert len(migrated) == 1, "empirical_backup must be migrated out, not deleted"
    assert (migrated[0] / "kernel" / "fa.h").exists(), "audit content preserved in migrated dir"
    # Phase O2.5 prep untouched
    assert (tmp_path / "model.py").exists()


# ---------------------------------------------------------------------------
# Immutable source stages vs cold-start (2026-08-21)
#
# `.tilelang2ascendc_source` publishes
# under a fail-closed "existing stage must not be replaced" rule.  Cold-start
# is the sole explicit reset mechanism, so it must ARCHIVE a live stage —
# otherwise the second --cold-start on the same op dies at source staging
# (observed: TILELANG2ASCENDC_SOURCE_STAGE_EXISTS on the second run).
# `reference_inputs/` is different: its bundles are content-addressed and
# re-staging the same inputs is idempotent, so cold-start preserves it.
# ---------------------------------------------------------------------------


def _tilelang_project(root: Path) -> Path:
    """Minimal TileLang2AscendC project accepted by tilelang2ascendc_source."""
    source = root / "3_Add"
    (source / "kernel" / "op_host").mkdir(parents=True)
    (source / "kernel" / "op_kernel").mkdir(parents=True)
    (source / "model_new_ascendc.py").write_text(
        "class ModelNew(torch.nn.Module):\n"
        "    def forward(self, x, y):\n"
        "        output = torch.ops.npu.add(x, y)\n"
        "        return output\n",
        encoding="utf-8",
    )
    (source / "kernel" / "register.cpp").write_text(
        'TORCH_LIBRARY_FRAGMENT(npu, m) { m.def("add(Tensor x, Tensor y) -> Tensor"); }\n'
        'TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) { m.impl("add", &Add); }\n',
        encoding="utf-8",
    )
    (source / "kernel" / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "add_library(add SHARED op_host/add.cpp op_kernel/add.cpp register.cpp)\n",
        encoding="utf-8",
    )
    (source / "kernel" / "op_host" / "add.cpp").write_text(
        "void Add() { EXEC_KERNEL_CMD(add_kernel); }\n", encoding="utf-8"
    )
    (source / "kernel" / "op_kernel" / "add.cpp").write_text(
        "__global__ __aicore__ void add_kernel() { AscendC::DataCopy(); }\n",
        encoding="utf-8",
    )
    return source


def test_cold_start_archives_immutable_source_stages(tmp_path, monkeypatch):
    """Cold-start archives `.tilelang2ascendc_source`: removed from the
    workspace, preserved byte-for-byte in the outside-workspace backup.
    `reference_inputs/` is content-addressed and idempotent on re-stage, so
    it is NOT wiped.
    """
    monkeypatch.setenv("COLD_START_BACKUP_ROOT", str(tmp_path / ".bkp"))
    ws = tmp_path / "op_ws"
    (ws / ".tilelang2ascendc_source" / "kernel").mkdir(parents=True)
    (ws / ".tilelang2ascendc_source" / "kernel" / "add.cpp").write_text("// staged tile")
    bundle = ws / "reference_inputs" / "npubench" / ("ab" * 32)
    bundle.mkdir(parents=True)
    (bundle / "bundle_manifest.json").write_text("{}")

    getattr(orch, '_cold_start_reset_workspace')(ws)

    # Immutable stage gone from the worker-visible workspace
    assert not (ws / ".tilelang2ascendc_source").exists()
    # Archived (not deleted) in the outside-workspace backup
    backups = list((tmp_path / ".bkp" / ws.name).glob("pre-cold-start-*"))
    assert len(backups) == 1
    assert (backups[0] / ".tilelang2ascendc_source" / "kernel" / "add.cpp").read_text() == "// staged tile"
    # Content-addressed reference bundles survive (re-staging them is idempotent)
    assert (bundle / "bundle_manifest.json").exists()


def test_cold_start_twice_allows_tilelang_source_restage(tmp_path, monkeypatch):
    """Repro: two consecutive --cold-start runs on the same TileLang2AscendC
    source must not die at source staging on the second run.  The fail-closed
    rule itself is preserved: a bare (non-cold-start) re-stage still refuses
    to replace the live stage.
    """
    import tilelang2ascendc_source as tile_source

    monkeypatch.setenv("COLD_START_BACKUP_ROOT", str(tmp_path / ".bkp"))
    source = _tilelang_project(tmp_path)
    ws = tmp_path / "workspace" / source.name

    # First run stages the immutable source snapshot.
    tile_source.stage_tilelang2ascendc_source_tree(source, ws)
    # Cold-start #1 archives it; cold-start #2 must stage cleanly again.
    getattr(orch, '_cold_start_reset_workspace')(ws)
    stage = tile_source.stage_tilelang2ascendc_source_tree(source, ws)
    assert stage.root == ws / ".tilelang2ascendc_source"

    # Fail-closed semantics intact: without cold-start, re-staging refuses.
    with pytest.raises(tile_source.Tilelang2AscendcSourceError, match="SOURCE_STAGE_EXISTS"):
        tile_source.stage_tilelang2ascendc_source_tree(source, ws)


