# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Scope and fail-closed tests for the customer plugin registry."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent.parent
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

from plugins import (  # noqa: E402
    BasePlugin,
    PluginAmbiguityError,
    all_plugins,
    detect_plugin,
    get_plugin,
    register_plugin,
    reset_registry_for_testing,
)


@pytest.fixture(autouse=True)
def _restore_registry():
    original = all_plugins()
    yield
    reset_registry_for_testing()
    for plugin in original:
        register_plugin(plugin)


def _state(workspace: Path, mode: str) -> Path:
    workspace.mkdir()
    (workspace / ".opgen_state.json").write_text(
        json.dumps({"op": workspace.name, "opgen_mode": mode})
    )
    return workspace


def test_registry_is_exact_customer_allowlist():
    assert [plugin.name for plugin in all_plugins()] == [
        "backward",
        "port_a3_to_a5",
    ]
    assert get_plugin("backward").cli_flag == "--backward"
    assert get_plugin("port_a3_to_a5").cli_flag == "--port-a3"
    assert get_plugin("nonexistent_mode") is None


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("backward", "backward"), ("port_a3_to_a5", "port_a3_to_a5")],
)
def test_detect_supported_mode(tmp_path, mode, expected):
    workspace = _state(tmp_path / mode, mode)
    assert detect_plugin(workspace).name == expected


@pytest.mark.parametrize("mode", ["unsupported", "legacy_mode", ""])
def test_removed_mode_is_not_detected(tmp_path, mode):
    workspace = _state(tmp_path / (mode or "empty"), mode)
    assert detect_plugin(workspace) is None


def test_empty_workspace_returns_none(tmp_path):
    assert detect_plugin(tmp_path) is None


def test_duplicate_name_is_rejected():
    class Duplicate(BasePlugin):
        name = "backward"
        cli_flag = "--different"

    with pytest.raises(ValueError, match="Duplicate plugin name"):
        register_plugin(Duplicate())


def test_ambiguity_is_regular_runtime_error_and_never_none(tmp_path):
    class ClaimA(BasePlugin):
        name = "claim_a"
        cli_flag = "--claim-a"

        def detect(self, workspace):
            return True

    class ClaimB(BasePlugin):
        name = "claim_b"
        cli_flag = "--claim-b"

        def detect(self, workspace):
            return True

    reset_registry_for_testing()
    register_plugin(ClaimA())
    register_plugin(ClaimB())

    assert issubclass(PluginAmbiguityError, RuntimeError)
    with pytest.raises(PluginAmbiguityError, match="matched by 2 plugins"):
        detect_plugin(tmp_path)
