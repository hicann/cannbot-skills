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
import stat
from pathlib import Path

from logging_config import get_logger

log = get_logger(__name__)


# Worker/evaluator output entries a repair spawn must not inherit.  Keep the
# list explicit: a broad recursive move could archive the authenticated
# .aclnn_source or reference_inputs trees by mistake.
_NPUBENCH_REPAIR_OUTPUT_NAMES = (
    "kernel",
    "op_kernel",
    "op_host",
    "probes",
    "branched_from_kernel",
    ".npubench_candidate",
    "output",
)


# Engine-owned SHA-256 manifest written into each repair archive (2026-08-28,
# P1-1).  It is what makes an archive restorable: a later engine/infra-class
# repair reset may move the archived tree back into the workspace only after
# verifying every byte against this manifest.  Archives predating the manifest
# (or failing verification) are never restored — the reset falls back to the
# legacy move-out semantics.
_NPUBENCH_REPAIR_MANIFEST_NAME = "repair_manifest.json"
_NPUBENCH_REPAIR_MANIFEST_SCHEMA = "cannbot.npubench_candidate_repair_manifest/v1"


def _npubench_repair_backup_root(workspace: Path) -> Path:
    """Return the out-of-workspace backup root shared by all repair resets."""
    backup_root_env = (
        os.environ.get("NPUBENCH_REPAIR_BACKUP_ROOT")
        or os.environ.get("COLD_START_BACKUP_ROOT")
    )
    return (
        Path(backup_root_env) / workspace.name
        if backup_root_env
        else Path.home() / ".opgen_backups" / workspace.name
    )


def _npubench_repair_archive_dir(workspace: Path) -> Path:
    """Create the out-of-workspace archive directory for one repair reset."""
    import uuid

    backup_root = _npubench_repair_backup_root(workspace)
    backup_root.mkdir(parents=True, exist_ok=True)
    timestamp = int(_dt.datetime.now(_dt.timezone.utc).timestamp())
    archive = backup_root / f"npubench-candidate-repair-{timestamp}-{uuid.uuid4().hex[:12]}"
    archive.mkdir()
    return archive


def _npubench_repair_candidates(workspace: Path) -> list[Path]:
    """Collect the stale candidate outputs to archive out of the workspace.

    npubench_evidence/ is deliberately NOT collected (2026-08-25): the durable
    reports stay in the workspace — the next worker brief reads
    npubench_evidence/preflight_target_receipt.json and must not re-run
    preflight (same contract as the O5 rollback cleanup in
    fsm_phase_finalize._clear_harness_build_artifacts).
    """
    candidates: list[Path] = [workspace / name for name in _NPUBENCH_REPAIR_OUTPUT_NAMES]
    candidates.extend(sorted(workspace.glob("model_new_*.py")))
    candidates.extend(sorted(workspace.glob("pybind11*.cpp")))
    candidates.extend(
        sorted(
            item
            for suffix in ("*.c", "*.cc", "*.cpp", "*.cxx", "*.h", "*.hpp", "*.so", "*.tgz")
            for item in workspace.glob(suffix)
        )
    )
    return candidates


def _npubench_repair_move(candidates: list[Path], archive: Path) -> list[str]:
    """Move each existing candidate entry into the archive; return the names."""
    import shutil

    moved: list[str] = []
    seen: set[Path] = set()
    for source in candidates:
        if source in seen or (not source.exists() and not source.is_symlink()):
            continue
        seen.add(source)
        destination = archive / source.name
        shutil.move(str(source), str(destination))
        moved.append(source.name + ("/" if destination.is_dir() else ""))
    return moved


def _write_npubench_repair_record(workspace: Path, record: dict) -> None:
    """Atomically publish the repair audit record inside the workspace."""
    import uuid

    record_path = workspace / ".npubench_candidate_repair.json"
    temporary = workspace / f".{record_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, record_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _npubench_evaluate_report_flags(workspace: Path) -> tuple[bool, bool]:
    """Return ``(engine_or_infra, candidate)`` flags from the O5 report."""
    try:
        report = json.loads(
            (workspace / "npubench_evidence" / "evaluate_report.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False, False
    status = report.get("status") if isinstance(report, dict) else None
    return status == "ERROR", status == "FAIL"


def _npubench_transition_rollback_kind(line: str) -> tuple[bool, object]:
    """Decode one transition line and return its rollback-kind presence/value."""
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return False, None
    if "rollback_kind" not in entry:
        return False, None
    return True, entry.get("rollback_kind")


def _npubench_transition_flags(workspace: Path) -> tuple[bool, bool]:
    """Return failure-class flags from the newest durable transition record."""
    engine_infra = False
    candidate = False
    try:
        lines = (
            (workspace / "state_transitions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        for line in reversed(lines):
            if not line.strip():
                continue
            has_rollback_kind, rollback_kind = _npubench_transition_rollback_kind(line)
            if not has_rollback_kind:
                continue
            if rollback_kind == "infra":
                engine_infra = True
            elif rollback_kind is not None:
                candidate = True
            break
    except (OSError, UnicodeError):
        return engine_infra, candidate
    return engine_infra, candidate


def _npubench_repair_failure_class(workspace: Path) -> str:
    """Classify the failure that triggered this repair reset (P1-1, 2026-08-28).

    Evidence (read-only; those files are owned by other modules):
    - ``npubench_evidence/evaluate_report.json`` — status ERROR marks an
      engine/infra-class evaluation failure; FAIL marks a real precision
      MISMATCH (candidate class).
    - ``state_transitions.jsonl`` — the most recent entry carrying a
      ``rollback_kind`` field; ``infra`` is engine/infra class, any other
      non-None value is candidate class.

    Returns ``"engine_infra"`` only on positive engine/infra evidence with NO
    contradicting candidate-class signal.  Missing or unreadable evidence is
    fail-closed: ``"unknown"`` keeps the legacy move-out semantics.
    """
    engine_infra, candidate = _npubench_evaluate_report_flags(workspace)
    transition_engine_infra, transition_candidate = _npubench_transition_flags(workspace)
    engine_infra = engine_infra or transition_engine_infra
    candidate = candidate or transition_candidate

    if candidate:
        return "candidate"
    if engine_infra:
        return "engine_infra"
    return "unknown"


def _write_npubench_repair_manifest(archive: Path, moved: list[str]) -> None:
    """Write the engine-owned SHA-256 manifest for one repair archive.

    The manifest is the restore contract: a later engine/infra-class repair
    reset restores this tree only when every archived file still matches.
    Symlinks are recorded and make the archive unrestorable (a restored
    symlink could point outside the workspace).
    """
    import hashlib

    files: dict[str, str] = {}
    symlinks: list[str] = []
    for dirpath, dirnames, filenames in os.walk(archive, followlinks=False):
        for name in dirnames:
            entry = Path(dirpath) / name
            if entry.is_symlink():
                symlinks.append(entry.relative_to(archive).as_posix() + "/")
        for name in filenames:
            entry = Path(dirpath) / name
            rel = entry.relative_to(archive).as_posix()
            if rel == _NPUBENCH_REPAIR_MANIFEST_NAME:
                continue
            if entry.is_symlink():
                symlinks.append(rel)
            else:
                files[rel] = hashlib.sha256(entry.read_bytes()).hexdigest()
    payload = {
        "schema": _NPUBENCH_REPAIR_MANIFEST_SCHEMA,
        "entries": moved,
        "files": files,
        "symlinks": sorted(symlinks),
    }
    (archive / _NPUBENCH_REPAIR_MANIFEST_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _npubench_load_repair_manifest(archive: Path) -> dict | None:
    """Load and sanity-check a repair archive manifest; None = unrestorable."""
    from pathlib import PurePosixPath

    try:
        payload = json.loads(
            (archive / _NPUBENCH_REPAIR_MANIFEST_NAME).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != (
        _NPUBENCH_REPAIR_MANIFEST_SCHEMA
    ):
        return None
    entries = payload.get("entries")
    files = payload.get("files")
    if not _npubench_repair_manifest_shape_valid(
        entries, files, payload.get("symlinks")
    ):
        return None
    for rel in list(files) + [name.rstrip("/") for name in entries]:
        pure = PurePosixPath(rel)
        if pure.is_absolute() or ".." in pure.parts:
            return None
    return payload


def _npubench_repair_manifest_shape_valid(
    entries: object, files: object, symlinks: object,
) -> bool:
    """Validate the manifest container types before path-level checks."""
    return (
        isinstance(entries, list)
        and bool(entries)
        and all(isinstance(name, str) for name in entries)
        and isinstance(files, dict)
        and bool(files)
        and not symlinks
    )


def _npubench_verify_repair_tree(root: Path, files: dict) -> list[str]:
    """Return the manifest paths missing/changed under ``root`` ([] = clean)."""
    import hashlib

    mismatches: list[str] = []
    for rel, expected_sha in sorted(files.items()):
        entry = root / rel
        if entry.is_symlink() or not entry.is_file():
            mismatches.append(rel)
            continue
        if hashlib.sha256(entry.read_bytes()).hexdigest() != expected_sha:
            mismatches.append(rel)
    return mismatches


def _npubench_find_restorable_archive(workspace: Path) -> tuple[Path, dict] | None:
    """Return the newest repair archive whose manifest verifies, newest first.

    Only archives carrying a valid manifest with at least one entry, no
    symlinks, and byte-identical content qualify; anything else (including
    pre-manifest archives) is skipped so the caller can fail closed.
    """
    backup_root = _npubench_repair_backup_root(workspace)
    if not backup_root.is_dir():
        return None
    archives = sorted(
        (
            item
            for item in backup_root.glob("npubench-candidate-repair-*")
            if item.is_dir() and not item.is_symlink()
        ),
        key=lambda item: item.name,
    )
    for archive in reversed(archives):
        payload = _npubench_load_repair_manifest(archive)
        if payload is None:
            continue
        if _npubench_verify_repair_tree(archive, payload["files"]):
            continue
        return archive, payload
    return None


def _npubench_archived_repair_result(
    archive: Path | None, moved: list[str],
) -> dict | None:
    """Return the legacy archived outcome when a new archive was created."""
    if archive is None:
        return None
    return {
        "action": "archived",
        "moved": moved,
        "archive_id": archive.name,
        "reason": "candidate_contract_failure_before_build",
    }


def _npubench_archive_current_outputs(
    workspace: Path,
) -> tuple[list[str], Path | None]:
    """Archive current candidate outputs before attempting a prior-tree restore."""
    current = [
        item
        for item in _npubench_repair_candidates(workspace)
        if item.exists() or item.is_symlink()
    ]
    if not current:
        return [], None
    archive = _npubench_repair_archive_dir(workspace)
    moved = _npubench_repair_move(current, archive)
    _write_npubench_repair_manifest(archive, moved)
    return moved, archive


def _npubench_restore_targets_occupied(workspace: Path, entries: list[str]) -> bool:
    """Return whether a restore destination is occupied by an unarchived entry."""
    for name in entries:
        destination = workspace / name
        if destination.exists() or destination.is_symlink():
            log.warning(
                "NPUBench repair restore aborted: %s still occupied", destination
            )
            return True
    return False


def _npubench_undo_restore(workspace: Path, restored: list[str]) -> None:
    """Remove entries moved into the workspace by a failed restore attempt."""
    import shutil

    for already in restored:
        leftover = workspace / already.rstrip("/")
        if leftover.is_dir() and not leftover.is_symlink():
            shutil.rmtree(leftover)
        elif leftover.exists() or leftover.is_symlink():
            leftover.unlink()


def _npubench_restore_verified_archive(
    workspace: Path, archive: Path, payload: dict,
) -> list[str] | None:
    """Restore and verify one archive, returning restored names or ``None``."""
    import shutil

    entries = [name.rstrip("/") for name in payload["entries"]]
    restored: list[str] = []
    for name in entries:
        source = archive / name
        if not source.exists() or source.is_symlink():
            log.warning(
                "NPUBench repair restore aborted: manifest entry %s missing "
                "from %s",
                name,
                archive,
            )
            _npubench_undo_restore(workspace, restored)
            return None
        shutil.move(str(source), str(workspace / name))
        restored.append(name + ("/" if (workspace / name).is_dir() else ""))

    mismatches = _npubench_verify_repair_tree(workspace, payload["files"])
    if mismatches:
        log.error(
            "NPUBench repair restore verification failed (%d mismatch(es)); "
            "removing the restored tree: %s",
            len(mismatches),
            ", ".join(mismatches[:8]),
        )
        _npubench_undo_restore(workspace, restored)
        return None
    return restored


def _npubench_engine_infra_repair_reset(workspace: Path) -> dict | None:
    """Engine/infra-class repair reset: restore the newest verified archive.

    P1-1 (2026-08-28, DSH+v4pro §4.a): for an engine/infra-class failure the
    previous tree was not the cause, so the ENGINE restores the prior tree
    instead of forcing a clean re-author.  Current workspace candidate
    outputs, if any, are archived first — that makes the newest archive hold
    the latest tree, so "restore the most recent archive" then means "keep
    the current tree" when one exists and "bring back the tree the race
    window removed" when the workspace is empty (14:52 empty-window race).
    The engine owns the archive; the worker never touches ``.opgen_backups``.

    Returns the audit fields for the repair record, or None when nothing was
    archived and nothing is restorable (caller falls back to the legacy
    move-out path).  Any archive that fails manifest/SHA verification is
    skipped; a restore that cannot verify after the move is removed again,
    leaving the fail-closed move-out outcome.
    """
    moved, new_archive = _npubench_archive_current_outputs(workspace)

    restorable = _npubench_find_restorable_archive(workspace)
    if restorable is None:
        return _npubench_archived_repair_result(new_archive, moved)

    archive, payload = restorable
    entries = [name.rstrip("/") for name in payload["entries"]]
    # A destination occupied by something the standard candidate collection
    # did not archive is a collision the restore must not clobber.
    if _npubench_restore_targets_occupied(workspace, entries):
        return _npubench_archived_repair_result(new_archive, moved)

    restored = _npubench_restore_verified_archive(workspace, archive, payload)
    if restored is None:
        return _npubench_archived_repair_result(new_archive, moved)

    return {
        "action": "restored_prior_tree",
        "restored_from": archive.name,
        "restored": restored,
        "restore_sha256_verified": True,
        "moved": moved,
        "archive_id": new_archive.name if new_archive is not None else None,
        "reason": "engine_infra_failure_restore_prior_tree",
    }


def _npubench_repair_archive_record(
    workspace: Path,
    *,
    failure_reason: str | None,
    failure_kind: str | None,
    failure_class: str,
) -> dict:
    """Legacy move-out reset: archive stale candidate outputs (unchanged)."""
    archive = _npubench_repair_archive_dir(workspace)
    moved = _npubench_repair_move(_npubench_repair_candidates(workspace), archive)
    _write_npubench_repair_manifest(archive, moved)

    record = {
        "schema": "cannbot.npubench_candidate_repair/v1",
        "status": "PASS",
        "action": "archived",
        "failure_class": failure_class,
        "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "archive_id": archive.name,
        "moved": moved,
        "reason": "candidate_contract_failure_before_build",
    }
    if failure_kind:
        record["failure_kind"] = failure_kind
    if failure_reason:
        record["failure_reason"] = failure_reason[-8192:]
    _write_npubench_repair_record(workspace, record)
    log.info(
        "NPUBench candidate repair reset: moved %d stale output(s) to %s",
        len(moved),
        archive,
    )
    return record


def _prepare_npubench_candidate_repair(
    workspace: Path,
    *,
    failure_reason: str | None = None,
    failure_kind: str | None = None,
) -> dict:
    """Reset stale NPUBench candidate outputs before a worker repair spawn.

    O5 candidate-contract failures deliberately re-enter ``await_worker``.
    That is a new authoring attempt, but it is not a cold start: the durable
    source stage, frozen task bundle, and state machine must remain intact.
    Moving the previous candidate out of the worker-visible workspace keeps
    the graybox answer gate meaningful on the retry.  The move is recoverable
    and the audit record stays in the workspace so a resumed orchestrator can
    explain why the candidate was reset without exposing the archived source
    to the worker.

    P1-1 (2026-08-28, DSH+v4pro §4.a): branch on the class of the failure
    that triggered this reset.  For an engine/infra-class failure (evaluation
    status=ERROR or an infra rollback) the previous tree was not the cause,
    so the ENGINE restores the newest verified archive back into the
    workspace instead of forcing a clean re-author (the worker still gets
    the failure diagnosis; it never touches ``.opgen_backups``).  Current
    candidate outputs, if any, are archived first so the newest archive
    always holds the latest tree: the restore then keeps the current tree
    when one exists and brings back the tree a race window removed when the
    workspace is empty.  Only a
    candidate-class failure (real precision MISMATCH) keeps the legacy
    move-out + re-author semantics, and missing/unreadable evidence fails
    closed to the same legacy path.  The graybox respawn archive-retry
    (``failure_kind="graybox_answer_gate_respawn"``) always keeps move-out:
    its seal retry requires an answer-free workspace, so restoring there
    would re-trip the gate it is remediating.
    """
    workspace = Path(workspace)
    metadata = workspace.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"NPUBench repair workspace must be a real directory: {workspace}")

    failure_class = _npubench_repair_failure_class(workspace)
    if (
        failure_class == "engine_infra"
        and failure_kind != "graybox_answer_gate_respawn"
    ):
        reset = _npubench_engine_infra_repair_reset(workspace)
        if reset is not None:
            record = {
                "schema": "cannbot.npubench_candidate_repair/v1",
                "status": "PASS",
                "failure_class": failure_class,
                "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                **reset,
            }
            if failure_kind:
                record["failure_kind"] = failure_kind
            if failure_reason:
                record["failure_reason"] = failure_reason[-8192:]
            _write_npubench_repair_record(workspace, record)
            if reset["action"] == "restored_prior_tree":
                log.info(
                    "NPUBench candidate repair reset: engine/infra-class "
                    "failure — restored %d prior-tree entr(ies) from %s "
                    "(SHA-256 verified)",
                    len(reset["restored"]),
                    reset["restored_from"],
                )
            else:
                log.info(
                    "NPUBench candidate repair reset: engine/infra-class "
                    "failure but nothing restorable — moved %d stale "
                    "output(s) to %s",
                    len(reset["moved"]),
                    reset["archive_id"],
                )
            return record

    return _npubench_repair_archive_record(
        workspace,
        failure_reason=failure_reason,
        failure_kind=failure_kind,
        failure_class=failure_class,
    )


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
    - Immutable source stages: .tilelang2ascendc_source
      (fail-closed "stage must not be replaced" — an explicit
      cold-start is the only sanctioned reset, so they are archived, not
      silently reused or deleted)
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
        # Immutable source stage (2026-08-21): `.tilelang2ascendc_source`
        # publishes under a fail-closed rule ("existing stage must not be
        # replaced").  Without archiving it here, a second --cold-start on the
        # same op re-staged the source and died at startup with
        # TILELANG2ASCENDC_SOURCE_STAGE_EXISTS.  Cold-start is the sole
        # explicit reset mechanism, so archiving (not deleting) keeps
        # fail-closed semantics: a bare non-cold-start run still refuses to
        # replace a live stage.
        ".tilelang2ascendc_source",
        # npubench finalize artifacts (2026-08-21, graybox construction):
        # `.npubench_candidate` holds the prior run's built O5 candidate
        # (target/answer-bearing .cpp trees), `.npubench_exec` the exec
        # scratch and `npubench_evidence` the finalize reports.  A cold-start
        # that leaves them in-workspace makes the graybox construction scan
        # reject the next worker spawn (assembled_answer_cpp_reachable>0).
        # They are regenerated by O2.5 preflight / O5 finalize, so archiving
        # them here is the same clean-slate contract as the source stages.
        ".npubench_candidate",
        ".npubench_exec",
        "npubench_evidence",
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
