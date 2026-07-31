# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""independent review review P3 (2026-05-20): two-part defensive hardening of the
handoff routing path.

Part A — Handoff parser dual-form (DEBT-104):
Workers may emit either `→ orchestrator: <kw>` (canonical arrow form, the
brief tells them to) OR `@orchestrator: <kw>` (mention form, mirroring
inter-agent @aog-X handoffs). State machine YAML conditions
exclusively match `→ orchestrator:` via handoff_match — without
normalization, an `@orchestrator: structural_rewrite_needed` handoff
would extract correctly but the downstream handoff_match fails → routing
silently misses.

Fix: orchestrator.extract_canonical_handoff normalizes `@orchestrator:`
→ `→ orchestrator:` (same shape as DEBT-103's `→ aog-X` → `@aog-X`
arrow→at normalization for inter-agent handoffs).

Part B — `op_class=="unknown"` → False (safe default):
state_machine.eval_condition's plugin_method primitive resolves op_class
from snap["op_taxonomy"]["class"]. When op_taxonomy is missing or
detection failed, op_class is "unknown"; the primitive short-circuits to
False at the gate layer so a plugin can't accidentally opt-in on an
unclassified op.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ORCHESTRATOR_DIR = _HERE.parents[2]
sys.path.insert(0, str(_ORCHESTRATOR_DIR))
sys.path.insert(0, str(_ORCHESTRATOR_DIR.parent / "workflow"))

import state_machine as sm  # noqa: E402

_ORCHESTRATOR_SPEC = importlib.util.spec_from_file_location(
    "orchestrator_handoff_ut", _ORCHESTRATOR_DIR / "orchestrator.py"
)
assert _ORCHESTRATOR_SPEC is not None and _ORCHESTRATOR_SPEC.loader is not None
orch = importlib.util.module_from_spec(_ORCHESTRATOR_SPEC)
_ORCHESTRATOR_SPEC.loader.exec_module(orch)


# ──────────────────────────────────────────── Part A: dual-form parser


def test_orchestrator_at_form_normalizes_to_arrow_form():
    """`@orchestrator: <kw>` MUST be returned as `→ orchestrator: <kw>`
    so YAML handoff_match conditions (keyed on the arrow form) hit.
    """
    raw = "Some prose...\n@orchestrator: structural_rewrite_needed — FA case 18 fp32\n"
    out = orch.extract_canonical_handoff(raw)
    assert out.startswith("→ orchestrator:")
    assert "structural_rewrite_needed" in out


def test_orchestrator_at_form_normalizes_done_handoff():
    """The normalization applies uniformly to all valid keywords, not just
    the structural_rewrite sentinel.
    """
    raw = "Final line:\n@orchestrator: done — kernel verified\n"
    out = orch.extract_canonical_handoff(raw)
    assert out.startswith("→ orchestrator: done")


def test_orchestrator_arrow_form_passes_through_unchanged():
    """The arrow form is unchanged by normalization (regression guard:
    the normalization branch must not catch the arrow form).
    """
    raw = "\n→ orchestrator: structural_rewrite_needed — FA case 18 fp32\n"
    out = orch.extract_canonical_handoff(raw)
    # Single arrow prefix (no double-normalization artifacts)
    assert out.startswith("→ orchestrator:")
    assert out.count("→ orchestrator:") == 1


def test_orchestrator_at_form_handoff_match_downstream_compatibility():
    """End-to-end: extracted handoff from @-form must downstream-match
    YAML conditions that key on `→ orchestrator: structural_rewrite_needed`.
    """
    raw = "@orchestrator: structural_rewrite_needed — FA scope-spanning"
    extracted = orch.extract_canonical_handoff(raw)
    # The YAML condition shape (mirrors await_worker.exit_transitions IL clause):
    ctx = {"handoff": extracted, "snapshot": {}, "iter_counts": {}, "ws": None, "sm": {}}
    assert sm.eval_condition(
        {"handoff_match": "→ orchestrator: structural_rewrite_needed"}, ctx
    ) is True


# ──────────────────────────────────────────── Part B: unknown→False gate


def test_plugin_method_returns_false_when_op_class_unknown():
    """plugin_method (string-arg form) MUST short-circuit to False when
    snap.op_taxonomy.class is "unknown" or missing — defense in depth.
    """
    from plugins.base import BasePlugin

    class _AlwaysOptInPlugin(BasePlugin):
        """Hypothetical buggy/future plugin that would opt-in on unknown."""
        name = "test_buggy_optin_plugin"

        def route_check(self, op_class, op_complexity, worker_signal):
            # Deliberately returns True on EVERY input — proves the gate
            # short-circuits BEFORE consulting the plugin when op_class
            # is unknown.
            return True

    ctx = {
        "handoff": "→ orchestrator: structural_rewrite_needed",
        "snapshot": {"op_taxonomy": {"class": "unknown", "complexity": "L4"}},
        "iter_counts": {},
        "ws": None,
        "sm": {},
        "plugin": _AlwaysOptInPlugin(),
    }
    assert sm.eval_condition({"plugin_method": "route_check"}, ctx) is False


def test_plugin_method_returns_false_when_op_taxonomy_missing():
    """Missing op_taxonomy entirely (snap = {}) MUST resolve op_class to
    'unknown' via the default, then short-circuit to False.
    """
    from plugins.base import BasePlugin

    class _AlwaysOptInPlugin(BasePlugin):
        name = "test_buggy_optin_plugin_v2"

        def route_check(self, op_class, op_complexity, worker_signal):
            return True

    ctx = {
        "handoff": "→ orchestrator: structural_rewrite_needed",
        "snapshot": {},  # No op_taxonomy at all
        "iter_counts": {},
        "ws": None,
        "sm": {},
        "plugin": _AlwaysOptInPlugin(),
    }
    assert sm.eval_condition({"plugin_method": "route_check"}, ctx) is False


def test_plugin_method_evaluates_when_op_class_known():
    """Positive control: when op_class is a real string (e.g. FA shape),
    the plugin IS consulted and its return value flows through.
    """
    from plugins.base import BasePlugin

    class _FaOptInPlugin(BasePlugin):
        name = "test_fa_optin"

        def route_check(self, op_class, op_complexity, worker_signal):
            return (
                op_complexity == "L4"
                and "FUSED" in (op_class or "")
                and "SOFTMAX" in (op_class or "")
                and worker_signal == "structural_rewrite_needed"
            )

    ctx = {
        "handoff": "→ orchestrator: structural_rewrite_needed",
        "snapshot": {"op_taxonomy": {
            "class": "FUSED SOFTMAX TRANSCENDENTAL REDUCTION",
            "complexity": "L4",
        }},
        "iter_counts": {},
        "ws": None,
        "sm": {},
        "plugin": _FaOptInPlugin(),
    }
    assert sm.eval_condition({"plugin_method": "route_check"}, ctx) is True


def test_plugin_method_dict_form_unaffected_by_unknown_guard():
    """The unknown-guard applies ONLY to the string-arg form (which
    resolves op_class from snapshot). The dict-arg form passes explicit
    args, so it bypasses the guard — by design.
    """
    from plugins.base import BasePlugin

    class _ExplicitArgsPlugin(BasePlugin):
        name = "test_explicit"

        def some_other_method(self, arg1, arg2):
            return arg1 == arg2

    ctx = {
        "handoff": "",
        "snapshot": {"op_taxonomy": {"class": "unknown"}},  # would trip the guard for string-form
        "iter_counts": {},
        "ws": None,
        "sm": {},
        "plugin": _ExplicitArgsPlugin(),
    }
    # Dict form: explicit args, guard NOT triggered → plugin IS consulted.
    cond = {"plugin_method": {"method": "some_other_method", "args": ["x", "x"]}}
    assert sm.eval_condition(cond, ctx) is True
