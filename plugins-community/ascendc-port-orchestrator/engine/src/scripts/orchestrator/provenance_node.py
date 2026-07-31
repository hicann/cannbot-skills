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

"""DEBT-203 — per-op provenance node (fitness-scored, lineage-aware).

Design: docs/design/DEBT_203_PROVENANCE_TREE_DESIGN.md (main-gated 2026-07-11).

This module is S1 (ADDITIVE-ONLY): compute a `provenance_node` metadata block for an
archived op and inject it into that op's verification.json. It does NOT read the tree,
match similar ops, or seed cold-start (S2/S3) — so it touches no anti-cheat surface.

Key invariants:
- `fitness` is DERIVED from fields verification.json ALREADY carries (precision / performance
  / determinism). It is never a new measurement.
- Branch *eligibility* is CORRECTNESS-ONLY (full precision PASS + determinism satisfied) —
  perf never gates eligibility (main ruling 2026-07-11 Q1/Q3); perf is only a tie-breaker
  in the S2 matcher.
- Injection is idempotent: `created_ts` is preserved from an existing node, everything else
  derives deterministically from verification content — so identical verification content
  yields an identical block (and thus a stable .finalized-<hash> idempotency key).

Naming discipline (collisions in the codebase — see design §1): this is `provenance_node`,
NOT `provenance` (finalize_checks_provenance.py = source-copy detection) and uses `parent_id`,
NOT `pid` (lane_pool.py = OS process id).
"""
from __future__ import annotations
from typing import Any, Optional

SCHEMA_VERSION = 1

# Fitness weights (main-gated defaults; correctness-dominant). Tunable via KB/telemetry later.
_W_PREC = 0.6
_W_PERF = 0.3
_W_DET = 0.1


_PASS_STATUSES = ("PASS", "PASS_WITHIN_TOLERANCE")


def _precision_ratio(verification: dict) -> float:
    """Fraction of precision cases passing. Schema-robust across the two real
    verification.json layouts: current (`precision.pass_a.{tier1_pass,total}`) and
    older-archive (`precision.{pass,total}`). Falls back to the authoritative
    top-level `precision.status` (1.0 on PASS) when neither count shape is present."""
    prec = verification.get("precision") or {}
    # current schema: nested pass_a (tier-1 gate)
    for block_key in ("pass_a", None):
        block = prec.get(block_key) if block_key else prec
        if not isinstance(block, dict):
            continue
        total = block.get("total")
        passed = block.get("tier1_pass", block.get("pass"))
        if total:
            try:
                return max(0.0, min(1.0, float(passed or 0) / float(total)))
            except (TypeError, ValueError, ZeroDivisionError):
                pass
    # no counts found → use the authoritative verdict
    return 1.0 if str(prec.get("status", "")).upper() in _PASS_STATUSES else 0.0


def _perf_ratio(verification: dict) -> float:
    perf = verification.get("performance") or {}
    # schema-robust: sum_ratio (older) → ratio (current) → independent_re_measure.median_ratio
    r = perf.get("sum_ratio")
    if r is None:
        r = perf.get("ratio")
    if r is None:
        r = (perf.get("independent_re_measure") or {}).get("median_ratio")
    if r is None:
        return 0.0
    try:
        return max(0.0, min(1.0, float(r)))
    except (TypeError, ValueError):
        return 0.0


def _det_ok(verification: dict) -> bool:
    det = verification.get("determinism") or {}
    return bool(det.get("policy_satisfied"))


def is_op_buggy(verification: dict) -> bool:
    """True iff the op's recorded verification indicates a broken kernel.

    S1 source signal: precision.status == FAIL. (The buggy-abort FSM state
    resume.py:_try_buggy_abort_recovery is the other signal; wired when that
    context is available — S1 uses the archived verification verdict.)
    """
    prec = verification.get("precision") or {}
    return str(prec.get("status", "")).upper() == "FAIL"


def is_branch_eligible(verification: dict) -> bool:
    """CORRECTNESS-ONLY branch eligibility (main ruling Q1/Q3): a proven op may
    seed a new op ONLY if it fully passed precision AND satisfied determinism.
    Perf does NOT gate eligibility (a correct-but-slow op is a fine seed). A
    partial/FAIL precision or unsatisfied determinism is never eligible.
    """
    if is_op_buggy(verification):
        return False
    prec = verification.get("precision") or {}
    # authoritative correctness verdict = precision.status (present in BOTH schema
    # variants); PARTIAL/FAIL/absent → not eligible. Must also satisfy determinism.
    status = str(prec.get("status", "")).upper()
    return bool(status in _PASS_STATUSES and _det_ok(verification))


def compute_fitness(verification: dict) -> float:
    """Fitness DERIVED from existing verification fields. A FAIL/buggy op is forced
    to 0.0 (never a branch source). Otherwise a correctness-dominant blend.
    """
    if is_op_buggy(verification):
        return 0.0
    prec = _precision_ratio(verification)
    perf = _perf_ratio(verification)
    det = 1.0 if _det_ok(verification) else 0.0
    return round(_W_PREC * prec + _W_PERF * perf + _W_DET * det, 6)


def build_provenance_node(
    op: str,
    source_sha256: str,
    verification: dict,
    signature: dict,
    *,
    created_ts: str,
    parent_id: Optional[str] = None,
    branched: bool = False,
    existing_node: Optional[dict] = None,
) -> dict:
    """Build the provenance_node block. Deterministic except `created_ts`, which is
    preserved from `existing_node` if present (idempotency). S1 callers always pass
    parent_id=None / branched=False (seeding is S3).
    """
    if existing_node and existing_node.get("created_ts"):
        created_ts = existing_node["created_ts"]
    sig = {
        "op_class_tags": list(signature.get("op_class_tags", []) or []),
        "algorithm_classification": signature.get("algorithm_classification"),
    }
    return {
        "node_id": f"{op}@{source_sha256[:12]}" if source_sha256 else f"{op}@unknown",
        "parent_id": parent_id,
        "branched": bool(branched),
        "fitness": compute_fitness(verification),
        "is_buggy": is_op_buggy(verification),
        "signature": sig,
        "created_ts": created_ts,
        "schema_version": SCHEMA_VERSION,
    }


def inject_provenance_node(
    verification: dict,
    op: str,
    source_sha256: str,
    signature: dict,
    *,
    created_ts: str,
    parent_id: Optional[str] = None,
    branched: bool = False,
) -> dict:
    """Return `verification` augmented with a `provenance_node` block (additive only —
    all existing keys/values preserved). Idempotent: re-injecting over an already-
    augmented dict preserves the original created_ts, so the block is stable.
    """
    existing = verification.get("provenance_node")
    verification["provenance_node"] = build_provenance_node(
        op, source_sha256, verification, signature,
        created_ts=created_ts, parent_id=parent_id, branched=branched,
        existing_node=existing if isinstance(existing, dict) else None,
    )
    return verification
