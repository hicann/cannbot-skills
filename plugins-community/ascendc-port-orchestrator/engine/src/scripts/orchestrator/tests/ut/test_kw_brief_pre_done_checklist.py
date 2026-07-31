# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for kw_brief pre-done file-existence checklist (2026-05-21).

Surface: 7_Sum kw-1 (2026-05-21) emitted `done` without writing
pass_a_runner.py + pass_b_runner.py — finalize phase_o5 rolled back →
kw-2 just to write those files (~5-10 min wasted). Same pattern hit
22_Nonzero + 20_Gather earlier in the session.

Fix: hardened Phase E exit-handoff block in kw_brief.py with explicit
pre-done file checklist + bash one-liner the worker can paste.

This test verifies the checklist is present and references the canonical
file names that phase_o5_runner._find_verifier checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "scripts" / "orchestrator"))

from briefs.kw_brief import _exit_handoff_block  # noqa: E402


def test_exit_handoff_block_lists_pass_a_runner():
    """The canonical Pass A runner filename must be enumerated in the checklist."""
    block = _exit_handoff_block()
    assert "pass_a_runner.py" in block, (
        "Phase E exit-handoff checklist must mention pass_a_runner.py "
        "(canonical filename per phase_o5_runner._find_verifier)"
    )


def test_exit_handoff_block_lists_pass_b_runner():
    """The canonical Pass B runner filename must be enumerated in the checklist."""
    block = _exit_handoff_block()
    assert "pass_b_runner.py" in block, (
        "Phase E exit-handoff checklist must mention pass_b_runner.py"
    )


def test_exit_handoff_block_has_pre_done_section():
    """Checklist must be physically present (not buried in a sub-bullet)."""
    block = _exit_handoff_block()
    assert "PRE-DONE FILE-EXISTENCE CHECKLIST" in block


def test_exit_handoff_block_clarifies_verify_py_is_not_substitute():
    """One of the failure modes was workers thinking verify.py == pass_b_runner.py.
    The brief must explicitly say they are different files with different consumers.
    """
    block = _exit_handoff_block()
    assert "verify.py" in block.lower() and "not" in block.lower(), (
        "Brief must clarify verify.py is NOT a substitute for *_runner.py"
    )


def test_exit_handoff_block_references_precision_pass_a_pass_b():
    """verification.json contract for pass_a + pass_b fields must be enumerated."""
    block = _exit_handoff_block()
    assert "tier1_pass" in block
    assert "total" in block


def test_exit_handoff_block_keeps_existing_handoff_options():
    """Backwards compat: existing handoff verbs must still be present."""
    block = _exit_handoff_block()
    # Pre-existing handoffs (must still work)
    assert "→ orchestrator: done" in block
    assert "→ orchestrator: PARTIAL_PERSIST" in block
    assert "→ orchestrator: structural_rewrite_needed" in block
    assert "@aog-precision-probe" in block
    assert "@aog-kernel-optimizer" in block


def test_exit_handoff_block_documents_real_incident():
    """Standing-artifact: link to the 7_Sum incident so future readers
    understand WHY this checklist was added (not just a process change
    for its own sake).
    """
    block = _exit_handoff_block()
    assert "7_Sum" in block or "2026-05-21" in block, (
        "Checklist should cite the empirical anchor for traceability"
    )
