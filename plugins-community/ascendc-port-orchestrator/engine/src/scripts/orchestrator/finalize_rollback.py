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

"""finalize_rollback — rollback-history + loop-break detection, extracted from
finalize_pipeline.py (behavior-neutral god-file decomposition, 2026-07-05).
Pure extraction: same logic/behavior. finalize_pipeline re-imports these names
(bottom import) so existing `from finalize_pipeline import record_rollback` etc.
callers (orchestrator.py, tests) are unaffected."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime as _datetime, timezone as _timezone
from pathlib import Path
from typing import Optional

# shared helpers that stay in finalize_pipeline (imported to avoid moving them)
from finalize_pipeline import GateID, _entry_is_perf_family, LOOP_BREAK_K


def _rollback_signature(gate: str, rollback_state: str) -> str:
    """Stable signature: (gate, rollback_state).

    Two rollbacks share a signature iff the same gate fired with the same
    rollback target. Sub-case variation within a single gate (e.g. KB_WRITEUP
    has 3 sub-conditions: missing / too-short / no-Findings) collapses to one
    signature — that's intentional: kw failing repeatedly on any KB sub-case
    is the same loop pattern. Reason text is NOT part of signature; reason
    can vary across retries (different counts, paths, error messages) without
    changing what kw needs to fix.
    """
    return f"{gate}::{rollback_state}"


def _rollback_history_path(workspace: Path) -> Path:
    return workspace / ".rollback_history.jsonl"


def record_rollback(
    workspace: Path,
    *,
    rollback_state: str,
    reason: str,
    gate: str,
    reason_limit: int | None = 1000,
) -> dict:
    """Append a rollback entry to .rollback_history.jsonl and return it.

    `gate` MUST be a `GateID` value (str enum); pass `GateID.X.value` from
    `check_finalize_eligibility`. Signature is computed from (gate,
    rollback_state) — see `_rollback_signature`.
    """
    persisted_reason = reason if reason_limit is None else reason[:reason_limit]
    entry = {
        "ts": _datetime.now(_timezone.utc).isoformat(),
        "gate": gate,
        "rollback_state": rollback_state,
        "reason": persisted_reason,
        "signature": _rollback_signature(gate, rollback_state),
    }
    path = _rollback_history_path(workspace)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def detect_loop_break(workspace: Path) -> Optional[dict]:
    """Read .rollback_history.jsonl tail. If the last 2 entries share a
    signature → loop detected → return last entry with extra
    `loop_detected_at_count: <N>` so caller can route to await_user_decision.
    Returns None when no loop (≤1 entry or last 2 differ).
    """
    path = _rollback_history_path(workspace)
    if not path.exists():
        return None
    try:
        lines = [
            json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except (json.JSONDecodeError, OSError):
        return None
    if len(lines) < 2:
        return None
    last = lines[-1]
    prev = lines[-2]
    if last.get("signature") == prev.get("signature"):
        # Count how many consecutive entries share this signature.
        sig = last["signature"]
        n = 1
        for entry in reversed(lines):
            if entry.get("signature") == sig:
                n += 1
            else:
                break
        return {**last, "loop_detected_at_count": n}
    return None


def _read_rollback_history(workspace: Path) -> "list[dict]":
    """Parse .rollback_history.jsonl into a list of entries (oldest first).

    Returns [] on missing/unreadable/malformed file (best-effort, same
    tolerance as detect_loop_break).
    """
    path = _rollback_history_path(workspace)
    if not path.exists():
        return []
    try:
        return [
            json.loads(ln)
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except (json.JSONDecodeError, OSError):
        return []


def detect_nonconvergent_loop(
    workspace: Path, *, k: int = LOOP_BREAK_K
) -> Optional[dict]:
    """DEBT-192 finalize loop-convergence guard.

    Returns a dict describing a non-convergent rollback loop, else None. Two
    no-progress patterns are recognized (superset of P0abe):

      1. ``same_signature`` — the last two rollbacks share one
         (gate, rollback_state) signature. This is exactly P0abe
         (detect_loop_break), preserved so the byte-identical loop still caps
         at 2 spawns.
      2. ``perf_family`` — the last ``k`` rollbacks are ALL perf-methodology
         family (see _entry_is_perf_family). Catches the alternating-perf-gate
         loop that pattern 1 misses because no two *consecutive* signatures
         match.

    The returned dict is a drop-in superset of detect_loop_break's shape (it
    still carries ``loop_detected_at_count``) plus ``pattern`` and ``count``,
    so existing await_user_decision routing keeps working unchanged.
    """
    # Pattern 1: byte-identical consecutive signatures (P0abe, unchanged).
    same = detect_loop_break(workspace)
    if same is not None:
        return {
            **same,
            "pattern": "same_signature",
            "count": same.get("loop_detected_at_count"),
        }

    # Pattern 2: perf-methodology-family churn (alternation-resilient).
    entries = _read_rollback_history(workspace)
    if len(entries) >= k:
        window = entries[-k:]
        if all(_entry_is_perf_family(e) for e in window):
            last = window[-1]
            return {
                **last,
                "pattern": "perf_family",
                "signature": "perf_methodology_family",
                "count": len(window),
                "loop_detected_at_count": len(window),
            }
    return None


def classify_loop_break_action(workspace: Path, loop: dict) -> dict:
    """Decide what to do about a non-convergent finalize loop (DEBT-192).

    Returns a dict::

        {"action": "coerce_perf_na" | "fail_fast",
         "perf_family": bool, "precision_pass": bool, "port_a3": bool,
         "reason": str}

    ``coerce_perf_na`` is RECOMMENDED (not applied here) iff ALL of:
      - the loop is on the perf-methodology family, AND
      - precision.status is PASS / PASS_WITHIN_TOLERANCE (precision-clean —
        the deliverable is real, only the perf claim can't converge), AND
      - the op is port_a3_to_a5 mode (a precision-focused port where device
        perf is legitimately N/A this session).

    In that case coercing performance.status -> N/A is the honest verdict the
    worker should have emitted (perf never validly measured) — a fail-SAFE on
    the LOOP, not a way to pass an invalid perf claim (the invalid PASS/ratio
    is RETRACTED to N/A, not accepted). Applying that coercion requires the
    port_a3 perf-N/A contract seam (P146 acceptance) and is therefore owned
    there; this classifier only surfaces the recommendation.

    Every other non-convergent loop returns ``fail_fast`` — the engine halts
    with the gate reason (terminal for the autonomous run; workspace preserved
    for inspection), never another respawn.
    """
    vj: dict = {}
    vp = workspace / "verification.json"
    if vp.exists():
        try:
            vj = json.loads(vp.read_text())
        except Exception:
            vj = {}
    prec = vj.get("precision", {}) or {}
    precision_pass = prec.get("status") in ("PASS", "PASS_WITHIN_TOLERANCE")
    port_a3 = (vj.get("mode") == "port_a3_to_a5")
    perf_family = (
        loop.get("pattern") == "perf_family"
        or _entry_is_perf_family(loop)
    )
    coerce_ok = bool(perf_family and precision_pass and port_a3)
    return {
        "action": "coerce_perf_na" if coerce_ok else "fail_fast",
        "perf_family": perf_family,
        "precision_pass": precision_pass,
        "port_a3": port_a3,
        "reason": loop.get("reason", ""),
    }
