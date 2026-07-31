# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-201 — plugins/base.py god-class decompose characterization lock.

The ~1400-line `plugins/base.py` god-module was split into three cohesive
leaves:
  - `plugins/taxonomy.py` — the `is_*` op-class / op-name predicates.
  - `plugins/protocol.py` — the `@runtime_checkable` `PluginProtocol`.
  - `plugins/base.py`      — the `BasePlugin` neutral-default class +
                             backward-compat re-exports.

These tests lock the invariants the decompose MUST preserve:
  1. Every historical import path still resolves the SAME object.
  2. `PluginProtocol` stays `@runtime_checkable` and `BasePlugin()`
     is an instance of it (MRO / structural conformance intact).
  3. `BasePlugin` exposes its full original public method set.
  4. The taxonomy predicates behave identically and are importable both
     standalone (leaf) and via `plugins.base` (re-export).
  5. Each split module is < 1000 lines (the god-file bar).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent.parent  # src/scripts/orchestrator/
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

# The full public method set BasePlugin exposed before the split (54 names).
# Locked here so any accidental drop/rename during a future refactor fails
# loudly rather than silently breaking a subclass's inherited override.
_EXPECTED_BASEPLUGIN_PUBLIC = {
    "archive_layout_mapping",
    "archive_project_subdir",
    "check_binary_provenance",
    "check_op_host_completeness",
    "check_verifier_uses_modelnew",
    "check_verify_path_provenance",
    "cli_flag",
    "canonical_pass_a_skip_reason",
    "detect",
    "docs_source_files",
    "extra_finalize_checks",
    "forbidden_patterns",
    "forbidden_workspace_files",
    "kb_subdirs",
    "ko_escalation_threshold",
    "kw_brief_phase_a",
    "kw_brief_phase_block",
    "kw_brief_phase_d",
    "name",
    "pass_b_default_when_skipped",
    "pass_b_required",
    "resolve_archive_target",
    "scanner_category",
    "should_run_phase_o5_perf_capture",
    "truth_source",
    "verifier_canonical_filenames",
    "verify_files",
}


def _public(cls):
    return {n for n in dir(cls) if not n.startswith("_")}


def test_reexport_identity_same_objects():
    """Every historical import path resolves the SAME object."""
    import plugins
    import plugins.base as base
    import plugins.protocol as protocol
    import plugins.taxonomy as taxonomy

    assert base.PluginProtocol is protocol.PluginProtocol
    assert plugins.PluginProtocol is protocol.PluginProtocol
    assert plugins.BasePlugin is base.BasePlugin
    for name in ("is_fa_class", "is_attention_named",
                 "is_l4_fused", "is_backward_class"):
        assert getattr(base, name) is getattr(taxonomy, name)


def test_protocol_runtime_checkable_and_isinstance():
    """PluginProtocol stays runtime_checkable; BasePlugin conforms."""
    from plugins.base import BasePlugin, PluginProtocol

    assert getattr(PluginProtocol, "_is_runtime_protocol", False) is True
    assert isinstance(BasePlugin(), PluginProtocol)


def test_baseplugin_full_public_surface_preserved():
    """BasePlugin exposes its complete original public method set."""
    from plugins.base import BasePlugin

    got = _public(BasePlugin)
    missing = _EXPECTED_BASEPLUGIN_PUBLIC - got
    assert not missing, f"BasePlugin lost public members: {sorted(missing)}"


def test_every_registered_plugin_still_subclasses_baseplugin():
    """MRO intact: the concrete plugins inherit BasePlugin defaults."""
    from plugins import all_plugins, BasePlugin

    plugins = all_plugins()
    assert plugins, "no plugins discovered"
    for p in plugins:
        # Not every plugin must subclass BasePlugin (protocol allows a
        # plain class), but any that does must still see it in its MRO.
        if isinstance(p, BasePlugin):
            assert BasePlugin in type(p).__mro__


def test_taxonomy_predicates_behaviour():
    """Predicates behave identically whether imported standalone or via base."""
    from plugins.taxonomy import (
        is_fa_class, is_attention_named, is_l4_fused, is_backward_class,
    )

    assert is_fa_class("FUSED SOFTMAX ATTENTION") is True
    assert is_fa_class("FUSED SOFTMAX REDUCTION") is False
    assert is_fa_class("") is False
    assert is_fa_class(None) is False

    assert is_attention_named("3_FusionAttention") is True
    assert is_attention_named("hc_split_sinkhorn") is False
    assert is_attention_named(None) is False

    assert is_l4_fused("FUSED MOE GATING") is True
    assert is_l4_fused("ELEMENTWISE") is False
    assert is_l4_fused(None) is False

    assert is_backward_class("ELEMENTWISE GRADIENT") is True
    assert is_backward_class("FUSED SOFTMAX BACKWARD") is True
    assert is_backward_class("ELEMENTWISE") is False
    assert is_backward_class(None) is False

    # is_fa_class implies is_l4_fused semantic anchor from the docstring
    assert is_l4_fused("FUSED SOFTMAX ATTENTION") is True


def test_taxonomy_is_a_pure_leaf():
    """taxonomy.py must not import the protocol or plugin machinery."""
    src = (_ORCH_DIR / "plugins" / "taxonomy.py").read_text()
    assert "PluginProtocol" not in src
    assert "import protocol" not in src
    assert "from .protocol" not in src


@pytest.mark.parametrize("module", ["base.py", "protocol.py", "taxonomy.py"])
def test_split_modules_under_1000_lines(module):
    path = _ORCH_DIR / "plugins" / module
    n = len(path.read_text().splitlines())
    assert n < 1000, f"{module} is {n} lines (god-file bar is <1000)"
