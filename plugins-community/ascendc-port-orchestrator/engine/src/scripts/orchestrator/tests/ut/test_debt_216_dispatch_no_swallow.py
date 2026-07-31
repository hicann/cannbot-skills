# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-216 — the dispatch helpers must not launder a failed detect into None.

Two separate failures are covered: an ambiguous match and an exception raised
while inspecting workspace state. Both are ordinary ``Exception`` subclasses,
and both must reach the caller instead of becoming ``None`` at the dispatch
helpers. ``plugins/tests/test_registry.py`` separately locks ambiguity to the
normal ``RuntimeError`` hierarchy.

Both matter for the same reason. `None` from these helpers means "no plugin
claims this workspace", and every caller reads it that way — it skips the
plugin-dispatched gates on purpose and reports clean. A helper that answers
`None` when it actually has no idea makes that report a lie.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ORCH = _HERE.parents[2]
_SCRIPTS = _HERE.parents[3]
for _p in (str(_ORCH), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import finalize_pipeline  # noqa: E402,F401  (import first: finalize_dispatch cycles through it)
import finalize_dispatch  # noqa: E402
import plugins as plugins_mod  # noqa: E402
import scan_delegation_cheating as sdc  # noqa: E402
from plugins import PluginAmbiguityError  # noqa: E402


class _ExplodingPlugin:
    """A plugin whose detect() is broken — the 'we could not tell' case that
    is NOT ambiguity (a permission error, a malformed workspace, a typo)."""
    name = "exploding"
    cli_flag = None

    def detect(self, workspace):
        raise OSError("detect blew up reading the workspace")


@pytest.mark.parametrize("helper", [
    pytest.param(lambda ws: getattr(finalize_dispatch, '_get_active_plugin')(ws), id="finalize_dispatch"),
    pytest.param(lambda ws: getattr(sdc, '_get_active_plugin')(ws), id="scan_delegation_cheating"),
])
def test_broken_detect_is_not_laundered_into_none(helper, tmp_path, monkeypatch):
    """MUTATION GUARD: restore `except Exception: return None` and this dies.

    A detect() that raises has not said "no plugin applies". Returning None
    asserts exactly that, and the caller then skips every plugin-dispatched
    gate and reports clean — the DEBT-216 shape, reached by a different road
    than the double-match.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(plugins_mod, "_PLUGIN_REGISTRY",
                        {"exploding": _ExplodingPlugin()})

    with pytest.raises(OSError):
        result = helper(ws)
        pytest.fail(
            f"detect() raised, but the dispatch helper swallowed it and "
            f"returned {result!r}. None means 'no plugin claims this "
            f"workspace' — it must never mean 'we could not tell'."
        )


@pytest.mark.parametrize("helper", [
    pytest.param(lambda ws: getattr(finalize_dispatch, '_get_active_plugin')(ws), id="finalize_dispatch"),
    pytest.param(lambda ws: getattr(sdc, '_get_active_plugin')(ws), id="scan_delegation_cheating"),
])
def test_ambiguity_reaches_the_caller_through_the_dispatch_helper(helper, tmp_path, monkeypatch):
    """End-to-end at the two sites the ROADMAP names: a double-match must
    reach the caller, not arrive as None.
    """
    ws = tmp_path / "ws"
    ws.mkdir()

    class _Always:
        cli_flag = None

        def __init__(self, name):
            self.name = name

        def detect(self, workspace):
            return True

    monkeypatch.setattr(plugins_mod, "_PLUGIN_REGISTRY",
                        {"a": _Always("a"), "b": _Always("b")})

    with pytest.raises(PluginAmbiguityError, match="matched by 2 plugins"):
        result = helper(ws)
        pytest.fail(
            f"double-match returned {result!r} instead of raising; every "
            f"plugin-dispatched gate would silently skip and report clean"
        )


def test_no_plugin_claims_workspace_still_returns_none(tmp_path, monkeypatch):
    """The other half of the contract: None must keep working for its ONE
    real meaning. Fail-loud that also fails on honest no-mode workspaces
    would just be a different outage — 39 archives legitimately match no
    plugin and must stay on the legacy fallback path.
    """
    ws = tmp_path / "ws"
    ws.mkdir()

    class _Never:
        name = "never"
        cli_flag = None

        def detect(self, workspace):
            return False

    monkeypatch.setattr(plugins_mod, "_PLUGIN_REGISTRY", {"never": _Never()})
    assert getattr(finalize_dispatch, '_get_active_plugin')(ws) is None
    assert getattr(sdc, '_get_active_plugin')(ws) is None
