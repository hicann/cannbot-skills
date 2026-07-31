# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""UT-scoped hermetic guards.

The pre-commit gate runs ONLY this ``ut/`` suite and MUST be hermetic — no unit test may
depend on a live ``claude`` subprocess. ``phase_o17_classify`` binds its Backend at import
(``_backend = get_backend()`` → the ``claude_code`` backend) and ``_invoke_claude_skill``
shells out to it. Any ut test that drives ``run_single_op`` through the O17 classify phase
(e.g. the FSM-characterization suite, which already mocks ``spawn_for_state`` + ``fire_critic``
but leaves O17 as an unmocked seam) therefore spawns a real ``claude --print`` child and HANGS
off a provisioned box (headless CI / cannbot-dev / clean checkout). This is the SAME class of
headless-absent env dependency the parent ``tests/conftest.py`` already neutralizes for
``.ascendc_env`` via an autouse hermetic fixture.

This autouse fixture makes the O17 classify skill hermetic in ut: it returns the
"skill unavailable" result (``ok=False``) — exactly what ``_invoke_claude_skill`` yields on a
box without the claude CLI (its ``not_found`` branch) — so O17 deterministically takes its
documented fallback path (cached/intrinsic taxonomy in ``_try_existing_classification`` or an
error classification) instead of blocking on a live subprocess. It is scoped to ``ut/`` only;
backend-requiring integration tests live in ``it/`` where a real backend is expected. Tests that
want a specific O17 classification still monkeypatch ``_invoke_claude_skill`` themselves — the
per-test patch is applied after this fixture and wins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _hermetic_o17_backend(monkeypatch):
    """Neutralize the O17 classify skill's live-backend dispatch for every ut test.

    Mirrors ``tests/conftest.py::_hermetic_ascendc_env`` (neutralize a headless-absent env
    dependency in setup only; tested code unchanged). Returns the honest claude-absent result
    so O17 exercises its real fallback path — this is behavior-preserving for any test that
    does not itself assert on a live O17 classification."""
    orch_dir = str(Path(__file__).resolve().parents[2])  # …/orchestrator
    if orch_dir not in sys.path:
        sys.path.insert(0, orch_dir)
    try:
        import phase_o17_classify as o17
    except ImportError:
        # A ut test that never imports the orchestrator package is unaffected.
        return
    monkeypatch.setattr(
        o17,
        "_invoke_claude_skill",
        lambda workspace, timeout=300: (
            False,
            "",
            "hermetic ut: O17 classify skill disabled (no live claude in the ut gate)",
        ),
        raising=True,
    )


@pytest.fixture(autouse=True)
def _hermetic_harness_pristine(monkeypatch):
    """DEBT-213(b): pin the harness-pristine probe to CLEAN for every ut test.

    Same hermeticity class as ``_hermetic_o17_backend`` above. The probe shells
    out to ``git status`` over the REAL repo, so O5's verdict would otherwise
    depend on whether the developer running pytest happens to have uncommitted
    harness edits — the ut gate would go red for everyone who is mid-change on
    the orchestrator, which is precisely the "check the team learns to ignore"
    outcome DEBT-213 is trying to avoid. Pinning to CLEAN keeps every existing
    O5 test asserting what it means to assert (claim vs re-measurement).

    The probe itself is NOT left untested: ``test_debt_213_harness_pristine.py``
    exercises the real implementation against real throwaway git repos, and
    re-patches ``harness_state`` per-test for the verdict-downgrade cases (the
    per-test patch is applied after this fixture and wins).
    """
    orch_dir = str(Path(__file__).resolve().parents[2])  # …/orchestrator
    if orch_dir not in sys.path:
        sys.path.insert(0, orch_dir)
    try:
        import harness_pristine as hp
    except ImportError:
        return
    monkeypatch.setattr(
        hp,
        "harness_state",
        lambda *a, **k: hp.HarnessState(hp.CLEAN, reason="hermetic ut: probe pinned CLEAN"),
        raising=True,
    )
