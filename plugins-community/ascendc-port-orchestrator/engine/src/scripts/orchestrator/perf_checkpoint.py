# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""await_probe precision checkpoint + hill-climbing advance/revert (Task #56).

Problem (root cause, 2026-05-29): on any re-emit loop that re-enters
await_probe (kernel-optimizer / worker respawn), a perf-driven re-emit is
LLM-sampled = non-deterministic. A re-emit chasing perf can BREAK
already-verified precision with no rollback guard (GQA: 11/11 → NaN).
Violates CLAUDE.md "每次改后验精度 AND 性能".

Fix — checkpoint + 3-state hill-climbing advance, atomic over the whole
`kernel/` dir:

  1. CHECKPOINT: first time precision passes (probe → finalize-eligible
     boundary), snapshot `kernel/` → `kernel/.precision_baseline/` and record
     baseline {tier1_pass, total, ratio, status}.
  2. ADVANCE (post-re-emit verify), 3 states:
       - precision REGRESSED  → REVERT to checkpoint (discard re-emit)
       - precision OK + FASTER → UPDATE checkpoint (new best-known-good)
       - precision OK + NOT faster → REVERT to checkpoint (keep best)
     ⇒ checkpoint is ALWAYS the best-correct kernel seen; revert never loses
     a successful improvement.
  3. Reverts are tagged `rollback_kind="perf_regression_revert"` and CONSUME
     iter budget (NOT infra-exempt) — caps the "re-roll the precision dice"
     loop (cf. NODE-5).
  4. N consecutive non-improvements → accept best-known-good, finalize
     PARTIAL (perf-WIP). researcher NOT auto-routed.

Also closes #55: the checkpoint IS the known-good backup, so an in-place
re-emit overwrite can never lose the last good kernel.

Module is pure-filesystem + verification.json; no NPU/network. Non-fatal by
construction — any internal error returns a NOOP result and the caller
proceeds with the existing (unguarded) behavior.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


CHECKPOINT_STATE_FILE = ".perf_checkpoint.json"
BASELINE_DIRNAME = ".precision_baseline"
ROLLBACK_KIND = "perf_regression_revert"

# Dirs inside kernel/ to exclude from the snapshot (build artifacts can be
# hundreds of MB; the baseline dir itself must never recurse into itself).
_SNAPSHOT_EXCLUDE = {"build", BASELINE_DIRNAME}

# Precision statuses that count as "passing" for checkpoint eligibility.
_PRECISION_OK = {"PASS", "PASS_WITHIN_TOLERANCE"}

# Default consecutive-non-improve count before accepting best-known-good.
DEFAULT_NO_IMPROVE_LIMIT = 2

# Perf ratio must improve by at least this much to count as "faster"
# (avoids treating run-to-run contention noise as a real improvement).
_RATIO_IMPROVE_EPS = 1e-3


class Action(str, Enum):
    NOOP = "noop"                          # nothing to do (no baseline + not passing)
    CHECKPOINT_CREATED = "checkpoint_created"
    ADVANCED_FASTER = "advanced_faster"    # new best — checkpoint updated
    REVERTED_REGRESSION = "reverted_regression"   # precision broke → restored
    REVERTED_NO_IMPROVE = "reverted_no_improve"   # not faster → restored


@dataclass
class CheckpointResult:
    action: Action = Action.NOOP
    baseline_tier1_pass: Optional[int] = None
    baseline_total: Optional[int] = None
    baseline_ratio: Optional[float] = None
    current_tier1_pass: Optional[int] = None
    current_ratio: Optional[float] = None
    consecutive_no_improve: int = 0
    reverted: bool = False            # True if kernel/ was restored from baseline
    consumes_budget: bool = False     # True if this iter should count (revert tag)
    rollback_kind: Optional[str] = None
    detail: str = ""


# ---------------------------------------------------------------------------
# verification.json reading
# ---------------------------------------------------------------------------

def _read_verification(workspace: Path) -> Optional[dict]:
    vf = workspace / "verification.json"
    if not vf.is_file():
        return None
    try:
        return json.loads(vf.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _precision_status(vj: dict) -> str:
    return str(vj.get("precision", {}).get("status", "") or "")


def _tier1(vj: dict) -> tuple[Optional[int], Optional[int]]:
    pa = vj.get("precision", {}).get("pass_a", {}) or {}
    t1 = pa.get("tier1_pass")
    tot = pa.get("total")
    try:
        t1 = int(t1) if t1 is not None else None
    except (ValueError, TypeError):
        t1 = None
    try:
        tot = int(tot) if tot is not None else None
    except (ValueError, TypeError):
        tot = None
    return t1, tot


def _perf_ratio(vj: dict) -> Optional[float]:
    r = vj.get("performance", {}).get("ratio")
    try:
        return float(r) if r is not None else None
    except (ValueError, TypeError):
        return None


def is_precision_ok(status: str) -> bool:
    return status in _PRECISION_OK


def precision_regressed(baseline_t1: Optional[int], baseline_ok: bool,
                        cur_status: str, cur_t1: Optional[int]) -> bool:
    """True if current precision is worse than the checkpointed baseline."""
    if baseline_ok and not is_precision_ok(cur_status):
        return True  # was passing, now not
    if baseline_t1 is not None and cur_t1 is not None and cur_t1 < baseline_t1:
        return True  # fewer cases pass than the baseline
    return False


def is_faster(cur_ratio: Optional[float], baseline_ratio: Optional[float]) -> bool:
    """True if current perf strictly improves on baseline (beyond noise eps).

    Higher ratio = faster (ratio = ref_time / kernel_time). If either is
    unknown, treat as NOT faster (conservative — keep the known-good).
    """
    if cur_ratio is None or baseline_ratio is None:
        return False
    return cur_ratio > baseline_ratio + _RATIO_IMPROVE_EPS


# ---------------------------------------------------------------------------
# Atomic whole-dir snapshot / restore
# ---------------------------------------------------------------------------

def _copytree_filtered(src: Path, dst: Path) -> None:
    """Copy src→dst excluding build artifacts + the baseline dir itself."""
    def _ignore(_dir, names):
        return [n for n in names if n in _SNAPSHOT_EXCLUDE]
    shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=False)


def atomic_snapshot(kernel_dir: Path) -> bool:
    """Snapshot kernel/ → kernel/.precision_baseline/ atomically.

    Build into a temp dir in the PARENT (workspace) — NOT inside kernel_dir,
    or copytree would recurse into the temp it is writing — then os.replace
    into place (atomic rename on the same filesystem). Returns True on success.
    """
    if not kernel_dir.is_dir():
        return False
    baseline = kernel_dir / BASELINE_DIRNAME
    parent = kernel_dir.parent
    tmp = Path(tempfile.mkdtemp(prefix=".ckpt_tmp_", dir=str(parent)))
    try:
        snap = tmp / "snap"
        _copytree_filtered(kernel_dir, snap)
        # Replace any existing baseline (small non-atomic window on the
        # rmtree, acceptable — a crash here just re-snapshots next run).
        if baseline.exists():
            shutil.rmtree(baseline, ignore_errors=True)
        os.replace(str(snap), str(baseline))
        return True
    except Exception:
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def atomic_restore(kernel_dir: Path) -> bool:
    """Restore kernel/ contents from kernel/.precision_baseline/ atomically.

    Removes current kernel files (except build/ + the baseline dir) and
    copies baseline contents back. Returns True on success.
    """
    baseline = kernel_dir / BASELINE_DIRNAME
    if not baseline.is_dir():
        return False
    try:
        # Remove current top-level entries except excluded ones.
        for entry in kernel_dir.iterdir():
            if entry.name in _SNAPSHOT_EXCLUDE:
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                try:
                    entry.unlink()
                except OSError:
                    pass
        # Copy baseline contents back (baseline itself has no build/).
        for entry in baseline.iterdir():
            target = kernel_dir / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, target)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Checkpoint state file
# ---------------------------------------------------------------------------

def _read_state(workspace: Path) -> Optional[dict]:
    p = workspace / CHECKPOINT_STATE_FILE
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_state(workspace: Path, state: dict) -> None:
    p = workspace / CHECKPOINT_STATE_FILE
    try:
        p.write_text(json.dumps(state, indent=2))
    except OSError:
        pass


def _snapshot_verification(workspace: Path) -> None:
    """Stash the current verification.json alongside the baseline so a revert
    restores the precision/perf record too (not just the kernel files)."""
    vf = workspace / "verification.json"
    baseline = workspace / "kernel" / BASELINE_DIRNAME
    if vf.is_file() and baseline.is_dir():
        try:
            shutil.copy2(vf, baseline / "verification.json")
        except OSError:
            pass


def _restore_verification(workspace: Path) -> None:
    baseline = workspace / "kernel" / BASELINE_DIRNAME
    saved = baseline / "verification.json"
    vf = workspace / "verification.json"
    if saved.is_file():
        try:
            shutil.copy2(saved, vf)
        except OSError:
            pass


def has_checkpoint(workspace: Path) -> bool:
    return (workspace / "kernel" / BASELINE_DIRNAME).is_dir() and \
           _read_state(workspace) is not None


def should_accept_best_known_good(workspace: Path,
                                  limit: int = DEFAULT_NO_IMPROVE_LIMIT) -> bool:
    """N consecutive non-improvements → caller should accept best-known-good
    and finalize PARTIAL (perf-WIP) instead of looping perf re-emit again.
    """
    st = _read_state(workspace)
    if not st:
        return False
    return int(st.get("consecutive_no_improve", 0)) >= limit


# ---------------------------------------------------------------------------
# Main entry: checkpoint_and_advance
# ---------------------------------------------------------------------------

def checkpoint_and_advance(workspace: Path) -> CheckpointResult:
    """Called when re-entering await_probe after probe updated verification.json.

    See module docstring for the 3-state hill-climbing semantics. Pure
    filesystem + verification.json; never raises (returns NOOP on any error).
    """
    try:
        return _checkpoint_and_advance_impl(workspace)
    except Exception as e:  # never let this break the orch loop
        return CheckpointResult(action=Action.NOOP, detail=f"error: {e!r}")


def _checkpoint_and_advance_impl(workspace: Path) -> CheckpointResult:
    kernel_dir = workspace / "kernel"
    vj = _read_verification(workspace)
    if vj is None:
        return CheckpointResult(action=Action.NOOP, detail="no verification.json")

    cur_status = _precision_status(vj)
    cur_t1, cur_tot = _tier1(vj)
    cur_ratio = _perf_ratio(vj)
    st = _read_state(workspace)

    # ---- No baseline yet ----
    if st is None or not (kernel_dir / BASELINE_DIRNAME).is_dir():
        if not is_precision_ok(cur_status):
            # Nothing verified yet to protect; precision path handles it.
            return CheckpointResult(action=Action.NOOP,
                                    current_tier1_pass=cur_t1, current_ratio=cur_ratio,
                                    detail="precision not yet passing; no checkpoint")
        # First clean PASS → snapshot baseline.
        ok = atomic_snapshot(kernel_dir)
        if not ok:
            return CheckpointResult(action=Action.NOOP, detail="snapshot failed")
        _snapshot_verification(workspace)
        _write_state(workspace, {
            "baseline_tier1_pass": cur_t1,
            "baseline_total": cur_tot,
            "baseline_ratio": cur_ratio,
            "baseline_precision_status": cur_status,
            "consecutive_no_improve": 0,
            "history": [{"event": "checkpoint_created", "tier1": cur_t1,
                         "total": cur_tot, "ratio": cur_ratio}],
        })
        return CheckpointResult(
            action=Action.CHECKPOINT_CREATED,
            baseline_tier1_pass=cur_t1, baseline_total=cur_tot, baseline_ratio=cur_ratio,
            current_tier1_pass=cur_t1, current_ratio=cur_ratio,
            detail="first clean PASS checkpointed",
        )

    # ---- Baseline exists: this is a post-re-emit verify ----
    b_t1 = st.get("baseline_tier1_pass")
    b_ratio = st.get("baseline_ratio")
    b_status = str(st.get("baseline_precision_status", "") or "")
    b_ok = is_precision_ok(b_status)
    n_no_improve = int(st.get("consecutive_no_improve", 0))
    history = list(st.get("history", []))

    # State 1: precision regressed → REVERT.
    if precision_regressed(b_t1, b_ok, cur_status, cur_t1):
        atomic_restore(kernel_dir)
        _restore_verification(workspace)
        n_no_improve += 1
        history.append({"event": "reverted_regression", "cur_status": cur_status,
                        "cur_tier1": cur_t1, "baseline_tier1": b_t1})
        st["consecutive_no_improve"] = n_no_improve
        st["history"] = history
        _write_state(workspace, st)
        return CheckpointResult(
            action=Action.REVERTED_REGRESSION,
            baseline_tier1_pass=b_t1, baseline_ratio=b_ratio,
            current_tier1_pass=cur_t1, current_ratio=cur_ratio,
            consecutive_no_improve=n_no_improve,
            reverted=True, consumes_budget=True, rollback_kind=ROLLBACK_KIND,
            detail=f"precision regressed (cur={cur_status}/{cur_t1} vs baseline={b_status}/{b_t1}); restored",
        )

    # Precision still OK. State 2: faster → UPDATE checkpoint (new best).
    if is_faster(cur_ratio, b_ratio):
        ok = atomic_snapshot(kernel_dir)
        if ok:
            _snapshot_verification(workspace)
        history.append({"event": "advanced_faster", "tier1": cur_t1,
                        "ratio": cur_ratio, "prev_ratio": b_ratio})
        st.update({
            "baseline_tier1_pass": cur_t1,
            "baseline_total": cur_tot,
            "baseline_ratio": cur_ratio,
            "baseline_precision_status": cur_status,
            "consecutive_no_improve": 0,   # progress resets the ladder
            "history": history,
        })
        _write_state(workspace, st)
        return CheckpointResult(
            action=Action.ADVANCED_FASTER,
            baseline_tier1_pass=cur_t1, baseline_ratio=cur_ratio,
            current_tier1_pass=cur_t1, current_ratio=cur_ratio,
            consecutive_no_improve=0,
            detail=f"new best: ratio {b_ratio} → {cur_ratio} at precision {cur_status}",
        )

    # State 3: precision OK but NOT faster → REVERT (keep best-known-good).
    atomic_restore(kernel_dir)
    _restore_verification(workspace)
    n_no_improve += 1
    history.append({"event": "reverted_no_improve", "cur_ratio": cur_ratio,
                    "baseline_ratio": b_ratio})
    st["consecutive_no_improve"] = n_no_improve
    st["history"] = history
    _write_state(workspace, st)
    return CheckpointResult(
        action=Action.REVERTED_NO_IMPROVE,
        baseline_tier1_pass=b_t1, baseline_ratio=b_ratio,
        current_tier1_pass=cur_t1, current_ratio=cur_ratio,
        consecutive_no_improve=n_no_improve,
        reverted=True, consumes_budget=True, rollback_kind=ROLLBACK_KIND,
        detail=f"not faster (cur={cur_ratio} <= baseline={b_ratio}); kept best-known-good",
    )
