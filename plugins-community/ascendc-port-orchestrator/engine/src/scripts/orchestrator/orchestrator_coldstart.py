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
"""orchestrator_coldstart — cold-start workspace reset, extracted from
orchestrator.py (behavior-neutral god-function decomposition, DEBT-201,
2026-07-06).

Pure extraction: byte-identical logic. `_cold_start_reset_workspace`
(P0u / P0hh / P0aav / P94 / DEBT-078) backs up + resets a workspace for a
fresh cold-start re-evaluation. Self-contained over filesystem + workspace
artifacts; NOT monkeypatched; calls no orchestrator-local function. Tests
call it via `orchestrator._cold_start_reset_workspace`, so orchestrator
re-imports it (bottom import) to keep that access path stable."""
from __future__ import annotations
import logging

import datetime as _dt
import json
import os
from pathlib import Path

from logging_config import get_logger

log = get_logger(__name__)


def _cold_start_reset_workspace(workspace: Path) -> None:
    """P0u + P0hh + P0aav (2026-05-07): back up + reset workspace state for fresh
    re-evaluation. Used to regen any op regardless of current status — even
    ops at terminal `done` state.

    P0aav (2026-05-07) — user caught design conflation: previous version
    preserved `kernel/`, `analysis.md`, `model_new_ascendc.py`, `pybind11.cpp`
    as "Phase O2.5 prep" — but those are actually kw OUTPUTS, not Phase
    O2.5 inputs. When kw was killed mid-write (e.g. user interrupt), those
    partial outputs stayed in workspace and the next "cold-start" kw spawn
    saw them as resumable state — defeating cold-start contract.

    Backs up to .pre-cold-start-<ts>/:
    - State machine artifacts: state_transitions.jsonl, PROGRESS.md
    - Per-iter artifacts: optimization_*, probe_*, cann_strategy_*, research_*
    - Worker output: verification.json, analysis.md, kernel/, model_new_ascendc.py,
      compute_reference.py, edge_verify.py, run_pass_*.py, det_check.py,
      probes/ (P0aav — these ARE worker outputs, must be wiped on cold-start)
    - Audit + retry: self_critic_*, .resume_fw_retry_count.json
    - Markers: .agent_died_at_*, .kernel_worker_active, .finalized-*
    - User decisions: user_decision.md, .opgen_state.json
    - Knowledge update output: knowledge_update.md

    Preserves ONLY Phase O2.5 prep + harness-side outputs (NOT kw output):
    - model.py (copied from benchmark, not kw output)
    - input_gen.py, edge_inputs.pt, edge_dataset.pt, manifest.json (Phase O2.5)
    - op_classification.json (Phase O1.7 output)
    - <op>.json + model.json (benchmark dataset / sibling alias)
    - .ascendc_env (env config)

    Rationale (P0aav): cold-start = "fresh evaluation; pretend this op
    has never been touched by a worker." Phase O2.5 prep can be preserved
    because regenerating it is wasteful and not affected by KB updates;
    worker outputs MUST be wiped because they're contaminated by prior
    KB / brief / iteration that may differ from current.
    """
    import shutil
    ts = int(_dt.datetime.now(_dt.timezone.utc).timestamp())
    # P0aav-followup (2026-05-16): backup dir MUST be OUTSIDE workspace
    # so worker (which reads workspace/) cannot find + read the backed-up
    # files. Previous version put backup under workspace/.pre-cold-start-<ts>/
    # — worker naturally explored workspace and Read 9 files from backup
    # (verification.json / pass_a_runner.py / model_new_ascendc.py /
    # PROGRESS.md / analysis.md / model.py / knowledge_update.md /
    # .finalized marker), effectively "resume from prior" not cold-start.
    # User audit 2026-05-16 00:30Z + 01:20Z caught this. Move backup OUT.
    #
    # Backup root: env COLD_START_BACKUP_ROOT overrides; default
    # ~/.opgen_backups/. Per-op subdir for organization. Worker brief
    # only gives workspace path → worker has no reason to scan
    # ~/.opgen_backups/. Tests can override the root.
    op_name = workspace.name
    backup_root_env = os.environ.get("COLD_START_BACKUP_ROOT")
    if backup_root_env:
        backup_root = Path(backup_root_env) / op_name
    else:
        backup_root = Path.home() / ".opgen_backups" / op_name
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_root / f"pre-cold-start-{ts}"
    backup_dir.mkdir(exist_ok=True)

    # 2026-05-16: ALSO migrate any PRE-EXISTING in-workspace backup/snapshot
    # dirs into the new outside-workspace root. Otherwise workers find + Read
    # (or RESTORE from) them, defeating the cold-start clean-slate contract.
    #
    # 2026-05-29 (FA-class restore-backup incident): the migration
    # originally caught ONLY `.pre-cold-start-*`. But a stale `.empirical_backup_*`
    # snapshot survived cold-start in-workspace and the worker
    # RESTORED the old block_N=64 kernel from it byte-for-byte instead of
    # translating fresh from the designer's block_N=128 tile_level — so op-gen
    # never generated the fast kernel. Any in-workspace backup/snapshot dir is a
    # restore-target; migrate ALL of them out, not just `.pre-cold-start-*`.
    _BACKUP_DIR_GLOBS = (
        ".pre-cold-start-*",
        ".empirical_backup*",     # manual/empirical kernel snapshots (restore-target)
        ".pre_kw*_restore_*",     # pre-restore snapshots
        ".pre_iter*", ".iter*_attempt", ".pattern_a*",  # per-iter attempt snapshots
    )
    _seen_legacy: set = set()
    for _glob in _BACKUP_DIR_GLOBS:
        for legacy in list(workspace.glob(_glob)):
            if not legacy.is_dir() or legacy in _seen_legacy:
                continue
            _seen_legacy.add(legacy)
            legacy_new_name = legacy.name.lstrip(".")  # ".empirical_backup-X" → "empirical_backup-X"
            target = backup_root / f"{legacy_new_name}-migrated-{ts}"
            try:
                shutil.move(str(legacy), str(target))
                log.info(f"migrated in-workspace backup {legacy.name} → {target}")
            except Exception as e:
                log.warning(f"failed to migrate {legacy.name}: {e}")

    # State files to back up + remove (these prevent fresh re-evaluation)
    state_patterns = [
        "state_transitions.jsonl",
        "PROGRESS.md",
        "verification.json",  # P0hh: stale claim, not Phase O2.5 prep
        "optimization_directive.md",
        "optimization_log.md",
        "probe_report.md",
        "probe_result.json",
        "cann_strategy_inference.md",
        "research_report.md",
        "self_critic_report.md",
        "knowledge_update.md",
        "user_decision.md",
        "failures_ledger.md",
        ".resume_fw_retry_count.json",
        ".opgen_state.json",
        # P0aav (2026-05-07): worker outputs — were incorrectly preserved as
        # "Phase O2.5 prep". Now backed up so a true cold-start spawn sees
        # workspace as a clean slate (no partial kw artifacts to resume from).
        "analysis.md",
        # NOTE: worker sources `model_new_<variant>.py` (including `_fenced`
        # variants) are cleared by the
        # backend-aware glob added just before the move loop below — NOT hardcoded
        # here (OL-160-class: a hardcoded filename missed variant outputs).
        "pybind11.cpp",
        "compute_reference.py",
        "compute_ref_outputs.py",
        "edge_verify.py",
        "run_pass_a.py",
        "run_pass_b.py",
        "pass_a_runner.py",
        "pass_b_runner.py",
        "run_det_check.py",
        "det_check.py",
        "determinism_check_inplace.py",
        "perf_quick.py",
        "perf_simple.py",
    ]
    # P94 INFRA-BLAME-LOOP fix (2026-05-15T09:17Z): preserve lifetime_spawn_count
    # across cold-start so accumulated cost is visible. Read BEFORE move,
    # write a slim survivor file AFTER all moves are done.
    _lifetime_spawn_count_preserve = 0
    _state_fp_preserve = workspace / ".opgen_state.json"
    if _state_fp_preserve.is_file():
        try:
            _state_obj_preserve = json.loads(_state_fp_preserve.read_text())
            _lifetime_spawn_count_preserve = int(
                _state_obj_preserve.get("lifetime_spawn_count", 0)
            )
        except Exception:
            _lifetime_spawn_count_preserve = 0

    # Variant-aware worker-source clear (OL-160-class fix, 2026-07-22).
    # A hardcoded "model_new_ascendc.py" missed suffixed worker outputs, so a
    # cold-start could preserve prior authored source and reuse it instead of
    # regenerating from scratch. Glob catches every `model_new_*.py`
    # (including `_fenced`/variant snapshots) — these
    # are always worker OUTPUT. The op reference `model.py` is NOT matched by
    # `model_new_*` → correctly preserved as Phase-O2.5 prep.
    _native_worker_sources = sorted(p.name for p in workspace.glob("model_new_*.py"))
    state_patterns = state_patterns + [
        n for n in _native_worker_sources if n not in state_patterns
    ]

    moved = []
    for pat in state_patterns:
        src = workspace / pat
        if src.exists():
            dst = backup_dir / pat
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved.append(pat)

    # P0aav: kernel/ + probes/ subdirs are worker output — wipe on cold-start.
    # DEBT-078 (2026-06-13, owner directive "物理删 restore 来源"): ALSO wipe
    # `op_kernel/` — the port_a3 generated kernel TU dir. Leaving it lets the
    # worker RESTORE a prior op_kernel instead of regenerating from KB (the
    # fake-autonomy cheat). `op_kernel/` is worker OUTPUT (port_a3 reads its A3
    # algorithm spec from the --port-a3 <dir>, NOT from workspace/op_kernel/).
    # A target-archive branch base is valid within a run, but must not leak
    # into a later cold start where its lineage and matching decision may be
    # stale.
    for subdir_name in ("kernel", "op_kernel", "probes", "branched_from_kernel"):
        sub = workspace / subdir_name
        if sub.exists() and sub.is_dir():
            shutil.move(str(sub), str(backup_dir / subdir_name))
            moved.append(f"{subdir_name}/")

    # DEBT-078: physically remove KERNEL-SNAPSHOT / BUILT-ARTIFACT restore sources
    # at workspace root so a cold-start worker MUST regenerate, not restore.
    # `kernel_snapshot.tgz` is the explicit restore tarball; `*.so` is a prior
    # build's compiled kernel; `*.tgz` catches any other snapshot tarball. These
    # are the local restore vectors the kw_brief anti-restore guard names — making
    # the clean-slate STRUCTURAL (file removal) instead of a skippable instruction.
    for snap in (
        list(workspace.glob("kernel_snapshot.tgz"))
        + list(workspace.glob("*.so"))
        + list(workspace.glob("*.tgz"))
    ):
        if snap.is_file():
            try:
                shutil.move(str(snap), str(backup_dir / snap.name))
                moved.append(snap.name)
            except Exception as e:
                log.warning(f"cold-start: failed to migrate restore source {snap.name}: {e}")
    # NOTE (DEBT-078, remote): the prior build's `.so` on the remote A5
    # `current_task/` is ALSO a restore vector, but current_task is SHARED across
    # concurrent agents on the live host — do NOT blind-wipe it here.
    # The fresh build overwrites current_task per-op; remote-restore prevention
    # stays in the kw_brief anti-restore guard until a per-op remote dir exists.

    # Glob patterns for marker files
    for marker in workspace.glob(".agent_died_at_*"):
        if ".cleaned-" in marker.name:
            continue  # leave already-cleaned markers alone
        shutil.move(str(marker), str(backup_dir / marker.name))
        moved.append(marker.name)
    # P0hh: clear .finalized-<hash> markers so re-finalize fires on the
    # fresh run instead of skipping as idempotent.
    for marker in workspace.glob(".finalized-*"):
        shutil.move(str(marker), str(backup_dir / marker.name))
        moved.append(marker.name)
    # Worker active marker (no backup, just remove — it's runtime state)
    for marker in (workspace / ".kernel_worker_active",):
        if marker.exists():
            marker.unlink()
            moved.append(marker.name)
    lineage_marker = workspace / ".branched_from.json"
    if lineage_marker.exists():
        shutil.move(str(lineage_marker), str(backup_dir / lineage_marker.name))
        moved.append(lineage_marker.name)
    log.info(f"--cold-start: backed up {len(moved)} files to {backup_dir.name}")
    log.info(f"moved: {', '.join(moved[:8])}{' ...' if len(moved) > 8 else ''}")

    # P94 INFRA-BLAME-LOOP fix: drop a slim survivor .opgen_state.json
    # carrying just lifetime_spawn_count so cost is visible across
    # cold-start episodes. Full state file went to backup; this stub
    # lets the next session warn at startup when accumulated cost is high.
    if _lifetime_spawn_count_preserve > 0:
        try:
            survivor = {"lifetime_spawn_count": _lifetime_spawn_count_preserve}
            (workspace / ".opgen_state.json").write_text(json.dumps(survivor, indent=2))
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
