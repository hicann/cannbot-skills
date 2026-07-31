# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression tests for the iter_cap await_optimizer graceful-finalize fix
(2026-07-24).

THE BUG. A route-a run reached a GOOD result (precision PASS, det PASS, perf
0.9056× above the 0.6 finalize floor) but did NOT terminal-finalize. It exited on
`iter_cap hit for await_optimizer (count=5, cap=5)` (exit-2). Root cause: the
optimizer LANDED a kernel whose perf sits in the band between the FINALIZE floor
(0.6×, shippable) and the PARITY optimization target (verification_perf_below_
threshold reads vj.perf.threshold = 1.0 by owner default 2026-07-21). So the
`verification_perf_below_threshold: false → finalize` transition did NOT fire
(perf IS below parity) and control fell through to "keep iterating within
budget". The optimizer re-landed the SAME byte-identical kernel (0 edits) each
re-spawn until iter_cap tripped. detect_loop_break did not catch it (it keys on
`.rollback_history.jsonl`, which forward FSM "keep iterating" transitions never
write) and _is_legitimate_pipeline_exhaustion did not rescue it (its
await_optimizer branch treats only SUB-floor plateaus as legitimate — it returns
False when ratio >= finalize floor).

THE FIX. A new await_optimizer FSM transition (placed right before "keep
iterating within budget") routes to `finalize` when the optimizer has CONVERGED
(byte-identical kernel across >=2 consecutive spawns, via the deterministic
ko_variant_ledger signature ledger) AND the op is already shippable (perf >=
finalize floor, precision PASS, det satisfied). This preserves the real cases:
an optimizer still making edits changes the kernel md5 → not converged → keeps
iterating; a sub-floor op → not above floor → existing PARTIAL_PERSIST/researcher
machinery; a precision/det-failing op → status/det guards keep it out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
# src/scripts/orchestrator on sys.path so ko_variant_ledger / perf_gate resolve;
# src/scripts/workflow for state_machine (the FSM next_state driver).
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parents[3] / "workflow"))

import ko_variant_ledger as kvl  # noqa: E402
import perf_gate  # noqa: E402
import state_machine as sm  # noqa: E402


# ---------------------------------------------------------------------------
# ko_variant_ledger signature-ledger unit tests (the byte-identical machinery)
# ---------------------------------------------------------------------------
def _mk_kernel(ws: Path, body: str) -> None:
    (ws / "kernel").mkdir(exist_ok=True)
    (ws / "kernel" / "k.cpp").write_text(body)


def test_ledger_byte_identical_two_spawns_is_converged(tmp_path):
    """Two consecutive spawns recording a BYTE-IDENTICAL kernel → converged."""
    ws = tmp_path
    _mk_kernel(ws, "// landed v1\n")
    kvl.record_optimizer_kernel_signature(ws, spawn_index=1)
    kvl.record_optimizer_kernel_signature(ws, spawn_index=2)  # 0 edits
    assert kvl.optimizer_kernel_converged(ws) is True


def test_ledger_kernel_changed_is_not_converged(tmp_path):
    """An optimizer STILL MAKING EDITS (kernel changed between spawns) must NOT
    read as converged — the guard against premature finalize.
    """
    ws = tmp_path
    _mk_kernel(ws, "// v1\n")
    kvl.record_optimizer_kernel_signature(ws, spawn_index=1)
    _mk_kernel(ws, "// v2 — real edit\n")
    kvl.record_optimizer_kernel_signature(ws, spawn_index=2)
    assert kvl.optimizer_kernel_converged(ws) is False


def test_ledger_single_spawn_is_not_converged(tmp_path):
    """One recorded signature is not enough to prove convergence."""
    ws = tmp_path
    _mk_kernel(ws, "// v1\n")
    kvl.record_optimizer_kernel_signature(ws, spawn_index=1)
    assert kvl.optimizer_kernel_converged(ws) is False


def test_ledger_absent_is_not_converged(tmp_path):
    """No ledger file → fail-closed (not converged) → legacy behavior."""
    assert kvl.optimizer_kernel_converged(tmp_path) is False


def test_ledger_unresolvable_kernel_records_none_not_converged(tmp_path):
    """No kernel sources → md5 None recorded → never a false convergence even
    across multiple spawns.
    """
    ws = tmp_path  # no kernel/ dir
    assert kvl.record_optimizer_kernel_signature(ws, spawn_index=1) is None
    assert kvl.record_optimizer_kernel_signature(ws, spawn_index=2) is None
    assert kvl.optimizer_kernel_converged(ws) is False


# ---------------------------------------------------------------------------
# End-to-end FSM transition: await_optimizer → finalize (drives the real YAML)
# ---------------------------------------------------------------------------
def _seed_ws(
    ws: Path, *, kernel_body: str, ratio: float | None,
    prec: str = "PASS", det: bool = True, finalize_floor: float = 0.6,
) -> None:
    _mk_kernel(ws, kernel_body)
    (ws / "verification.json").write_text(json.dumps({
        "precision": {
            "status": prec,
            "pass_a": {"status": "PASS", "tier1_pass": 8, "total": 8},
            "pass_b": {"status": "PASS", "tier1_pass": 8, "total": 8},
        },
        # NB: no performance.threshold key → the FSM's
        # verification_perf_below_threshold uses the parity default (1.0), which
        # is exactly what opened the band gap this fix closes.
        "performance": ({"ratio": ratio} if ratio is not None else {}),
        "determinism": {
            "policy_satisfied": det, "n_identical_cases": 8, "n_cases_checked": 8,
        },
    }))
    (ws / "PROGRESS.md").write_text("# fresh\n")
    (ws / "state_transitions.jsonl").write_text("")
    # finalize floor (the shippable bar) resolved by schema_norm._resolve_perf_threshold
    perf_gate.write_profile_marker(ws, finalize_floor)


_LANDED = "→ orchestrator: done — KO_OPTIMIZATION_LANDED, perf 0.91x (was 0.91x)"


def test_converged_landed_above_floor_routes_to_finalize(tmp_path):
    """THE BUG, fixed: perf 0.9056× (in the [0.6 floor, 1.0 parity) band) +
    precision PASS + det satisfied + optimizer converged byte-identical across
    two spawns → the FSM routes to `finalize`, NOT another await_optimizer loop
    to iter_cap exit-2.
    """
    ws = tmp_path
    _seed_ws(ws, kernel_body="// landed 0.9056\n", ratio=0.9056)
    kvl.record_optimizer_kernel_signature(ws, spawn_index=1)
    kvl.record_optimizer_kernel_signature(ws, spawn_index=2)  # byte-identical
    result = sm.next_state(ws, "await_optimizer", _LANDED)
    assert "error" not in result, result
    assert result["next_state"] == "finalize"


def test_optimizer_still_editing_does_not_finalize_early(tmp_path):
    """GUARD: an optimizer still making edits (kernel changes between spawns) is
    NOT converged → keeps iterating (await_optimizer), never a premature
    finalize just because one landed iteration happened.
    """
    ws = tmp_path
    _seed_ws(ws, kernel_body="// v1\n", ratio=0.9056)
    kvl.record_optimizer_kernel_signature(ws, spawn_index=1)
    _mk_kernel(ws, "// v2 — optimizer still improving\n")
    kvl.record_optimizer_kernel_signature(ws, spawn_index=2)
    result = sm.next_state(ws, "await_optimizer", _LANDED)
    assert "error" not in result, result
    assert result["next_state"] == "await_optimizer"


def test_converged_but_subfloor_does_not_false_finalize(tmp_path):
    """GUARD (real-failing preserved): converged byte-identical but perf 0.4×
    is BELOW the 0.6 finalize floor → the new rule does NOT fire; the op is
    handled by the existing sub-floor machinery (here: researcher escalation),
    never a false clean-PASS finalize.
    """
    ws = tmp_path
    _seed_ws(ws, kernel_body="// subfloor\n", ratio=0.4)
    kvl.record_optimizer_kernel_signature(ws, spawn_index=1)
    kvl.record_optimizer_kernel_signature(ws, spawn_index=2)
    result = sm.next_state(ws, "await_optimizer", _LANDED)
    assert "error" not in result, result
    assert result["next_state"] != "finalize"


def test_converged_but_precision_fail_does_not_finalize(tmp_path):
    """GUARD (real-failing preserved): converged + above floor but precision
    FAIL → the precision-status guard keeps it out of finalize.
    """
    ws = tmp_path
    _seed_ws(ws, kernel_body="// precfail\n", ratio=0.9056, prec="FAIL")
    kvl.record_optimizer_kernel_signature(ws, spawn_index=1)
    kvl.record_optimizer_kernel_signature(ws, spawn_index=2)
    result = sm.next_state(ws, "await_optimizer", _LANDED)
    assert "error" not in result, result
    assert result["next_state"] != "finalize"


def test_no_convergence_ledger_preserves_legacy_iteration(tmp_path):
    """GUARD: without any recorded signatures (e.g. the very first optimizer
    spawn) the new rule cannot fire → legacy 'keep iterating within budget'
    behavior is unchanged.
    """
    ws = tmp_path
    _seed_ws(ws, kernel_body="// first\n", ratio=0.9056)
    result = sm.next_state(ws, "await_optimizer", _LANDED)
    assert "error" not in result, result
    assert result["next_state"] == "await_optimizer"
