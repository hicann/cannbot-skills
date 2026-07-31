# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-094 phase 1 Layer 1 — protocol conformance.

Every registered plugin MUST satisfy PluginProtocol: have all required
attributes + methods with correct signatures + runtime-isinstance
agreement.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import get_type_hints

import pytest

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent.parent  # src/scripts/orchestrator/
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

from plugins import all_plugins, PluginProtocol  # noqa: E402


REQUIRED_METHODS = (
    "detect",
    "verify_files",
    "forbidden_patterns",
    "scanner_category",
    "check_binary_provenance",
    "check_verify_path_provenance",
    "archive_layout_mapping",
    "archive_project_subdir",
    "resolve_archive_target",
    "check_op_host_completeness",  # DEBT-094 phase 3 (2026-05-18) — PB-33 owned by plugin
    "extra_finalize_checks",  # P87 — option-A extension path
    "kb_subdirs",              # P88 phase 1 — KB reorg foundation
    "kw_brief_phase_a",
    "kw_brief_phase_d",
)

REQUIRED_ATTRS = ("name", "cli_flag")


def test_registry_discovers_required_plugins():
    """The customer registry is an exact allowlist, not an extensible scan."""
    plugins = all_plugins()
    names = [p.name for p in plugins]
    assert names == ["backward", "port_a3_to_a5"]


@pytest.mark.parametrize("plugin", all_plugins(), ids=lambda p: p.name)
def test_plugin_has_required_attrs(plugin):
    for attr in REQUIRED_ATTRS:
        assert hasattr(plugin, attr), f"{plugin.name}: missing attribute {attr!r}"
    assert plugin.name, f"plugin name must be truthy, got {plugin.name!r}"


@pytest.mark.parametrize("plugin", all_plugins(), ids=lambda p: p.name)
def test_plugin_has_required_methods(plugin):
    for method_name in REQUIRED_METHODS:
        assert hasattr(plugin, method_name), (
            f"{plugin.name}: missing method {method_name!r}"
        )
        m = getattr(plugin, method_name)
        assert callable(m), f"{plugin.name}.{method_name} is not callable"


@pytest.mark.parametrize("plugin", all_plugins(), ids=lambda p: p.name)
def test_plugin_satisfies_protocol_isinstance(plugin):
    """Runtime-checkable Protocol — every plugin should pass isinstance."""
    assert isinstance(plugin, PluginProtocol), (
        f"{plugin.name} fails isinstance(PluginProtocol) — missing methods?"
    )


@pytest.mark.parametrize("plugin", all_plugins(), ids=lambda p: p.name)
def test_plugin_detect_signature(plugin):
    """detect() must take exactly 1 positional arg (workspace: Path)."""
    sig = inspect.signature(plugin.detect)
    params = []
    ignored_kinds = (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    )
    for parameter in sig.parameters.values():
        if parameter.kind not in ignored_kinds:
            params.append(parameter)
    assert len(params) == 1, (
        f"{plugin.name}.detect signature {sig} should take 1 arg, got {len(params)}"
    )


def test_plugin_names_are_unique():
    """No two plugins may share a name (registry already enforces, but
    pin via test in case discovery order changes).
    """
    plugins = all_plugins()
    names = [p.name for p in plugins]
    assert len(set(names)) == len(names), (
        f"duplicate plugin names: {names}"
    )
