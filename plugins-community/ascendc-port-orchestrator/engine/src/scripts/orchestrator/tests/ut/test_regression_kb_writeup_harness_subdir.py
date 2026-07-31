# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression test: kb_writeup gate must check .harness/ subdir for
knowledge_update.md (post-promotion location).

Surfaced 2026-05-27 by regression-sweep on archived ops: CLAUDE.md user
direction 2026-05-16 relocated knowledge_update.md to .harness/ subdir
after archive promotion. The gate at finalize_pipeline.check_finalize_
eligibility (kb_writeup branch) ONLY checked workspace root → ALL modern
post-2026-05-16 archives failed re-verification.

This UT pins the layout-agnostic check: gate accepts the writeup at
EITHER workspace root (pre-promotion) OR .harness/ subdir (post-promotion).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent / "plugins"))

import finalize_pipeline as fp  # noqa: E402


def _seed_pass_archive(workspace: Path, *, kb_writeup_at: str) -> None:
    """Seed a workspace looking like a finalized PASS archive. kb_writeup_at
    controls whether knowledge_update.md lives at root or under .harness/."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "verification.json").write_text(json.dumps({
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
            "pass_b": {"status": "PASS", "tier1_pass": 11, "total": 11},
        },
        "performance": {
            "status": "PASS", "ratio": 1.5,
            "method": "same_wrapper symmetric=true method_symmetric",
            "independent_re_measure": {"status": "N/A", "reason": "test"},
        },
        "determinism": {"policy_satisfied": True, "n_identical_cases": 50, "n_cases_checked": 50},
    }))
    body = (
        "## Context\nRegression sweep fixture — modern archive layout.\n\n"
        "## Findings\n- Modern archives place knowledge_update.md in .harness/\n\n"
        "## KB-promotable patterns (proposed)\nNone\n\n"
        "## Cited KB items\n- OL-test\n\n"
        "## Anti-patterns avoided\nNone\n"
    )
    if kb_writeup_at == "root":
        (workspace / "knowledge_update.md").write_text(body)
    elif kb_writeup_at == "harness":
        harness = workspace / ".harness"
        harness.mkdir(exist_ok=True)
        (harness / "knowledge_update.md").write_text(body)
    elif kb_writeup_at == "neither":
        pass  # neither — gate should reject
    else:
        raise ValueError(f"unknown kb_writeup_at: {kb_writeup_at}")
    # Other required artifacts to pass downstream gates
    (workspace / "audit_self_critic_post_worker.md").write_text(
        "## Audit\nNo findings.\n## Verdict\nPASS\n"
    )
    (workspace / ".delegation_scan_passed").write_text("ok")
    # Force scan marker mtime > kernel mtime (fresh)
    kernel = workspace / "kernel"
    kernel.mkdir(exist_ok=True)
    (kernel / "stub.h").write_text("// stub\n")
    import os as _os
    import time as _time
    _time.sleep(0.05)
    _os.utime(workspace / ".delegation_scan_passed", None)


def test_kb_writeup_gate_accepts_root_location(tmp_path):
    """Pre-promotion workspace: knowledge_update.md at root → gate passes
    that file-presence check (later gates may still fail but kb_writeup-
    branch returns reason that does NOT mention 'missing').
    """
    ws = tmp_path / "test_op"
    _seed_pass_archive(ws, kb_writeup_at="root")
    elig = fp.check_finalize_eligibility(ws)
    # The kb_writeup gate fired or not isn't dispositive — what matters
    # is the reason isn't about the file missing. (Other gates downstream
    # may fail in the test fixture; we're specifically checking kb_writeup.)
    reason = elig.get("reason", "")
    assert "knowledge_update.md missing" not in reason, (
        f"kb_writeup gate should not reject when file at workspace root; "
        f"reason: {reason}"
    )


def test_kb_writeup_gate_accepts_harness_subdir_location(tmp_path):
    """Post-promotion archive: knowledge_update.md at .harness/ subdir →
    gate must check that location too. This is the regression catch.
    """
    ws = tmp_path / "test_op"
    _seed_pass_archive(ws, kb_writeup_at="harness")
    elig = fp.check_finalize_eligibility(ws)
    reason = elig.get("reason", "")
    assert "knowledge_update.md missing" not in reason, (
        f"kb_writeup gate must check .harness/ subdir (CLAUDE.md "
        f"2026-05-16 layout rule). Modern archives fail re-verify on "
        f"this without the fix. reason: {reason}"
    )


def test_kb_writeup_gate_rejects_missing_both_locations(tmp_path):
    """Neither location has the file → gate correctly rejects.
    Sanity: the fix is layout-agnostic, not file-absent-tolerant.
    """
    ws = tmp_path / "test_op"
    _seed_pass_archive(ws, kb_writeup_at="neither")
    elig = fp.check_finalize_eligibility(ws)
    assert not elig.get("eligible"), (
        "kb_writeup gate must STILL reject when file absent in both "
        "workspace root AND .harness/ subdir (the fix is layout-agnostic, "
        "not file-presence-tolerant)"
    )
    reason = elig.get("reason", "")
    assert "knowledge_update.md missing" in reason, (
        f"rejection reason should cite missing knowledge_update.md; "
        f"got: {reason}"
    )
    # Reason message should mention both locations were checked (catches
    # the layout-blindness regression class)
    assert "workspace root" in reason or ".harness" in reason or "subdir" in reason, (
        "rejection reason should signal both locations were checked "
        f"(diagnostic improvement); got: {reason}"
    )
