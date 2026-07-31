# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DEBT-124 regression: plugin.extra_finalize_checks() fires on
PARTIAL_PERSIST path, not just PASS path.

Pre-fix `finalize_pipeline.check_finalize_eligibility` ran the extras loop
only inside the PASS / PASS_WITHIN_TOLERANCE branch (L2143-2162). The
PARTIAL_PERSIST branch (L2436-2442) short-circuited to `eligible=True`
the moment `probe_report.md` existed → all plugin-registered hooks
(`extra_finalize_checks()`) silently bypassed.

Concrete fallout (ROADMAP §6 DEBT-124): 3_FusionAttention 2026-05-23
finalize — workspace status was PARTIAL_PERSIST, 7 FA-class plugin hooks
(gates 3/4/5/6/8/+v2/+case-6) shipped in PR #125 NEVER FIRED on that
workspace even though they passed their own unit tests and fired correctly
when called manually.

Same shape as DEBT-097's phase_o5 gap (owner 2026-05-27 "独立验证怎么可能
就没了") — verification mechanism exists but the path-specific code never
calls it.

Fix: extracted extras loop to `_run_plugin_extra_finalize_checks()` helper,
called from BOTH PASS and PARTIAL_PERSIST branches.
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
import finalize_dispatch as fd  # noqa: E402
from plugins.base import BasePlugin  # noqa: E402

# DEBT-201 (2026-07-06): the plugin-dispatch cluster (_get_active_plugin,
# _run_plugin_extra_finalize_checks, check_finalize_eligibility) moved to
# finalize_dispatch.py. _run_plugin_extra_finalize_checks / check_finalize_eligibility
# call _get_active_plugin by BARE NAME inside finalize_dispatch, so the patch
# MUST target finalize_dispatch (fd), NOT the finalize_pipeline re-export —
# patching fp._get_active_plugin would rebind only the parent's attribute and
# never reach the intra-cluster bare-name call. fp.<fn> and fd.<fn> are the same
# object (re-export), so the CALL side can use either; only the PATCH must be fd.


def _seed_partial_persist_workspace(tmp_path: Path) -> None:
    """Common seed: PARTIAL_PERSIST verdict + probe_report (the conditions
    that pre-fix short-circuited to eligible=True before hooks ran)."""
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {"status": "PARTIAL", "persist_verdict": "PARTIAL_PERSIST"},
        "performance": {"ratio": 0.5},
    }))
    (tmp_path / "probe_report.md").write_text("# probe report\nstub evidence")
    # DEBT-211: the PARTIAL_PERSIST branch now also runs the delegation-scan
    # marker gate. A legitimately-eligible PARTIAL_PERSIST archive has had its
    # delegation scan run clean (marker present) — so seed it, else the branch
    # rolls back on the marker before the extras-hook logic these tests target.
    (tmp_path / ".delegation_scan_passed").write_text(
        "scanner=scan_delegation_cheating.py violations=0 ts=test\n"
    )


def test_helper_returns_none_when_no_plugin(tmp_path, monkeypatch):
    """If plugin layer is absent, helper returns None (no rejection)."""
    monkeypatch.setattr(fd, "_get_active_plugin", lambda ws: None)
    result = getattr(fp, '_run_plugin_extra_finalize_checks')(tmp_path, {})
    assert result is None


def test_helper_returns_none_when_extras_empty(tmp_path, monkeypatch):
    """BasePlugin default extra_finalize_checks() returns [] → helper
    returns None (no-op, default-pass).
    """
    monkeypatch.setattr(
        fd, "_get_active_plugin", lambda ws: BasePlugin()
    )
    result = getattr(fp, '_run_plugin_extra_finalize_checks')(tmp_path, {})
    assert result is None


def test_helper_returns_rejection_on_violation(tmp_path, monkeypatch):
    """A plugin hook returning truthy string → helper returns the
    canonical eligibility-rejection dict.
    """

    class _StrictPlugin(BasePlugin):
        name = "strict"

        def extra_finalize_checks(self):
            return [
                ("test_gate", lambda ws, v: "violation reason here"),
            ]

    monkeypatch.setattr(
        fd, "_get_active_plugin", lambda ws: _StrictPlugin()
    )
    result = getattr(fp, '_run_plugin_extra_finalize_checks')(tmp_path, {})
    assert result is not None
    assert result["eligible"] is False
    assert result["gate"] == "test_gate"
    assert result["rollback_state"] == "await_worker"
    assert "violation reason here" in result["reason"]


def test_helper_returns_rejection_when_gate_raises(tmp_path, monkeypatch):
    """A plugin hook that RAISES → helper synthesizes a 'raised {type}: {msg}'
    rejection so the rollback diagnostic surfaces the plugin author's bug
    instead of silent-skip.
    """

    class _BrokenPlugin(BasePlugin):
        name = "broken"

        def extra_finalize_checks(self):
            def _raises(ws, v):
                raise RuntimeError("simulated plugin bug")

            return [("broken_gate", _raises)]

    monkeypatch.setattr(
        fd, "_get_active_plugin", lambda ws: _BrokenPlugin()
    )
    result = getattr(fp, '_run_plugin_extra_finalize_checks')(tmp_path, {})
    assert result is not None
    assert result["eligible"] is False
    assert result["gate"] == "broken_gate"
    assert "raised RuntimeError" in result["reason"]
    assert "simulated plugin bug" in result["reason"]


def test_helper_clean_when_all_gates_return_falsy(tmp_path, monkeypatch):
    """Multiple gates returning None / "" → helper returns None (clean
    finalize).
    """

    class _CleanPlugin(BasePlugin):
        name = "clean"

        def extra_finalize_checks(self):
            return [
                ("gate_a", lambda ws, v: None),
                ("gate_b", lambda ws, v: ""),
                ("gate_c", lambda ws, v: False),
            ]

    monkeypatch.setattr(
        fd, "_get_active_plugin", lambda ws: _CleanPlugin()
    )
    result = getattr(fp, '_run_plugin_extra_finalize_checks')(tmp_path, {})
    assert result is None


# ---------------------------------------------------------------------------
# DEBT-124 integration: PARTIAL_PERSIST short-circuit now runs extras
# ---------------------------------------------------------------------------

def test_partial_persist_eligible_when_hooks_pass(tmp_path, monkeypatch):
    """PARTIAL_PERSIST + probe evidence + clean plugin hooks → eligible=True
    (backward compatible — same verdict pre-fix when hooks didn't run at
    all, post-fix when hooks all clean).
    """
    _seed_partial_persist_workspace(tmp_path)

    class _CleanPlugin(BasePlugin):
        name = "clean"

        def extra_finalize_checks(self):
            return [("clean_gate", lambda ws, v: None)]

    monkeypatch.setattr(
        fd, "_get_active_plugin", lambda ws: _CleanPlugin()
    )
    result = fp.check_finalize_eligibility(tmp_path)
    assert result["eligible"] is True
    assert "PARTIAL_PERSIST" in result["reason"]


def test_partial_persist_rejected_when_plugin_hook_violates(
    tmp_path, monkeypatch
):
    """PARTIAL_PERSIST + probe evidence + plugin hook VIOLATION → eligible=
    False (new behavior; pre-fix returned True silently bypassing the hook).
    This is the load-bearing DEBT-124 invariant: PARTIAL_PERSIST archives
    are subject to the same plugin gates as PASS archives.
    """
    _seed_partial_persist_workspace(tmp_path)

    class _StrictPlugin(BasePlugin):
        name = "fa_strict"

        def extra_finalize_checks(self):
            return [
                ("fa_class_case_6", lambda ws, v: "case_6 anti-pattern detected"),
            ]

    monkeypatch.setattr(
        fd, "_get_active_plugin", lambda ws: _StrictPlugin()
    )
    result = fp.check_finalize_eligibility(tmp_path)
    assert result["eligible"] is False, (
        "PARTIAL_PERSIST archive should NOT pass when a plugin hook returns "
        "a violation — pre-fix would have wrongly approved this archive"
    )
    assert result["gate"] == "fa_class_case_6"
    assert "case_6 anti-pattern detected" in result["reason"]


def test_partial_persist_without_probe_evidence_still_rejected(
    tmp_path, monkeypatch
):
    """PARTIAL_PERSIST without probe_report.md → reject immediately at
    the earlier guard (DEBT-124 fix does NOT change this path). Confirms
    we only added the extras call AFTER probe-evidence check.
    """
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {"status": "PARTIAL", "persist_verdict": "PARTIAL_PERSIST"},
        "performance": {"ratio": 0.5},
    }))
    # No probe_report.md

    class _CleanPlugin(BasePlugin):
        name = "clean"

        def extra_finalize_checks(self):
            return [("would_be_clean", lambda ws, v: None)]

    monkeypatch.setattr(
        fd, "_get_active_plugin", lambda ws: _CleanPlugin()
    )
    result = fp.check_finalize_eligibility(tmp_path)
    assert result["eligible"] is False
    assert "no probe_report.md" in result["reason"]
