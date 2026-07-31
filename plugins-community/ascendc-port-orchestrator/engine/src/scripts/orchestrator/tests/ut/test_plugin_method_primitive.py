# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""eval_condition + next_state plugin_method primitive (S3c, 2026-05-20).

The YAML primitive `plugin_method` dispatches to a method on the active plugin
(passed via ctx["plugin"]) — plugins own paradigm-specific decisions.

These tests pin the primitive's generic contract — backwards-compat for legacy
callers, method dispatch with resolved orchestrator-side args, forward_kwargs
forwarding, and exception isolation. They use neutral stub method names because
the primitive is feature-agnostic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "scripts" / "workflow"))
sys.path.insert(0, str(ROOT / "src" / "scripts" / "orchestrator"))

import state_machine as sm  # noqa: E402
from plugins.base import BasePlugin  # noqa: E402


# ────────────────────────────────────────────────────── _parse_worker_signal


@pytest.mark.parametrize(
    "handoff,expected",
    [
        ("→ orchestrator: structural_rewrite_needed", "structural_rewrite_needed"),
        ("foo bar structural_rewrite_needed baz", "structural_rewrite_needed"),
        ("→ orchestrator: done", "done"),
        ("kw-1 emit done", "done"),
        ("→ orchestrator: PARTIAL_PERSIST — see probe_report.md", "partial_persist"),
        ("partial_persist with evidence", "partial_persist"),
        ("→ orchestrator: abort", "abort"),
        ("kernel build failure abort", "abort"),
        ("", "unknown"),
        ("some unrelated text", "unknown"),
        ("→ orchestrator: ascendc_ready", "unknown"),  # not a worker signal
    ],
)
def test_parse_worker_signal(handoff: str, expected: str):
    assert getattr(sm, "_parse_worker_signal")(handoff) == expected


# ─────────────────────────────────────────── eval_condition: plugin_method


def _ctx(handoff: str = "", plugin=None, op_class: str = "fused_attention",
        op_complexity: str = "L4") -> dict:
    """Minimal ctx for eval_condition unit tests."""
    return {
        "handoff": handoff,
        "snapshot": {"op_taxonomy": {"class": op_class, "complexity": op_complexity}},
        "iter_counts": {},
        "ws": Path("/tmp/nonexistent"),
        "sm": {},
        "plugin": plugin,
    }


def test_plugin_method_returns_false_when_plugin_absent():
    """Backwards-compat: legacy ctx without plugin → condition False."""
    ctx = _ctx(handoff="→ orchestrator: structural_rewrite_needed", plugin=None)
    assert sm.eval_condition({"plugin_method": "nonexistent_method"}, ctx) is False


def test_plugin_method_returns_false_when_method_missing():
    """Plugin present but method doesn't exist → False (with stderr warning)."""
    p = BasePlugin()
    ctx = _ctx(handoff="→ orchestrator: structural_rewrite_needed", plugin=p)
    assert sm.eval_condition({"plugin_method": "nonexistent_method"}, ctx) is False


def test_plugin_method_dispatches_with_resolved_args():
    """Method receives (op_class, op_complexity, worker_signal) from ctx — captured."""
    captured: list[tuple] = []

    class _CapturingPlugin(BasePlugin):
        name = "capturing"

        def route_check(self, op_class, op_complexity, worker_signal):
            captured.append((op_class, op_complexity, worker_signal))
            return True

    p = _CapturingPlugin()
    ctx = _ctx(
        handoff="→ orchestrator: structural_rewrite_needed",
        plugin=p,
        op_class="fused_attention",
        op_complexity="L4",
    )
    result = sm.eval_condition({"plugin_method": "route_check"}, ctx)
    assert result is True
    assert captured == [("fused_attention", "L4", "structural_rewrite_needed")]


def test_plugin_method_handles_exception_in_method():
    """Plugin method raises → caught, logged, returns False (no orchestrator crash)."""

    class _BrokenPlugin(BasePlugin):
        name = "broken"

        def route_check(self, op_class, op_complexity, worker_signal):
            raise RuntimeError("intentional fault for test")

    p = _BrokenPlugin()
    ctx = _ctx(handoff="→ orchestrator: structural_rewrite_needed", plugin=p)
    assert sm.eval_condition({"plugin_method": "route_check"}, ctx) is False


def test_plugin_method_dict_form_explicit_args():
    """Dict form: {method, args} overrides default arg resolution."""
    captured: list[tuple] = []

    class _CapturingPlugin(BasePlugin):
        name = "dictform"

        def custom_check(self, a, b):
            captured.append((a, b))
            return True

    p = _CapturingPlugin()
    ctx = _ctx(plugin=p)
    cond = {"plugin_method": {"method": "custom_check", "args": ["x", 42]}}
    assert sm.eval_condition(cond, ctx) is True
    assert captured == [("x", 42)]


def test_plugin_method_in_all_of_short_circuits_when_plugin_absent():
    """plugin_method inside all_of with another True primitive → still False without plugin."""
    ctx = _ctx(handoff="→ orchestrator: structural_rewrite_needed", plugin=None)
    cond = {
        "all_of": [
            {"handoff_match": "→ orchestrator: structural_rewrite_needed"},
            {"plugin_method": "nonexistent_method"},
        ]
    }
    # handoff_match matches → True; plugin_method False (no plugin) → all_of False
    assert sm.eval_condition(cond, ctx) is False


# ─────────────────────────────────────────── next_state: backwards-compat


def test_next_state_signature_accepts_optional_plugin(tmp_path):
    """next_state(ws, state, handoff) still works without plugin — backwards-compat.

    next_state(ws, state, handoff, plugin=p) accepts the new param.
    2026-05-27: also accepts optional `runtime_kwargs` for the generic
    forward_kwargs mechanism.
    """
    import inspect
    sig = inspect.signature(sm.next_state)
    params = list(sig.parameters.keys())
    assert params == ["ws", "current_state", "handoff", "plugin", "runtime_kwargs"], (
        f"next_state signature changed: {params}"
    )
    # Plugin + runtime_kwargs params must have defaults (backwards-compat).
    assert sig.parameters["plugin"].default is None
    assert sig.parameters["runtime_kwargs"].default is None


def test_unknown_primitive_falls_through_gracefully():
    """Unknown plugin_method method name doesn't crash — stays in the False path."""
    p = BasePlugin()
    ctx = _ctx(plugin=p)
    assert sm.eval_condition({"plugin_method": "method_that_does_not_exist"}, ctx) is False


# ────────────────────── forward_kwargs + runtime_kwargs (2026-05-27) ──────────────────────


def test_plugin_method_dict_form_forward_kwargs_pulls_from_runtime():
    """Dict form + forward_kwargs forwards the named keys from ctx.runtime_kwargs.

    Generic mechanism: evaluator pulls declared names from runtime_kwargs and
    forwards them as kwargs to the plugin method. No feature-specific
    branches in the evaluator.
    """
    captured: dict = {}

    class _SwitchAware(BasePlugin):
        name = "switch_aware"

        def custom_route(self, op_class, op_complexity, worker_signal, *, force_x=False):
            captured["force_x"] = force_x
            captured["args"] = (op_class, op_complexity, worker_signal)
            return force_x

    p = _SwitchAware()
    ctx = _ctx(
        handoff="→ orchestrator: structural_rewrite_needed",
        plugin=p,
        op_class="foreach_neg", op_complexity="L2",  # non-FA op
    )
    ctx["runtime_kwargs"] = {"force_x": True}

    cond = {"plugin_method": {
        "method": "custom_route",
        "forward_kwargs": ["force_x"],
    }}
    # Switch on + plugin uses it → True
    assert sm.eval_condition(cond, ctx) is True
    assert captured["force_x"] is True
    # Args auto-resolved from snapshot (dict form without `args` key)
    assert captured["args"] == ("foreach_neg", "L2", "structural_rewrite_needed")


def test_plugin_method_dict_form_forward_kwargs_missing_in_runtime():
    """forward_kwargs name not in runtime_kwargs → not forwarded (plugin default applies)."""
    captured: dict = {}

    class _SwitchAware(BasePlugin):
        name = "switch_aware_missing"

        def custom_route(self, op_class, op_complexity, worker_signal, *, force_x=False):
            captured["force_x"] = force_x
            return force_x

    p = _SwitchAware()
    ctx = _ctx(handoff="→ orchestrator: structural_rewrite_needed", plugin=p)
    # No runtime_kwargs set → plugin's default kwarg value (False) applies
    cond = {"plugin_method": {
        "method": "custom_route",
        "forward_kwargs": ["force_x"],
    }}
    assert sm.eval_condition(cond, ctx) is False
    assert captured["force_x"] is False
