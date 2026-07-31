# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Registry for the two supported customer workflow plugins.

Importing this package registers only arch migration and backward generation.
Directory scanning is intentionally not used: adding a folder must never expose
a third customer mode by accident.

Public API:
- register_plugin(plugin)
- all_plugins() -> tuple[PluginProtocol, ...]
- get_plugin(name) -> Optional[PluginProtocol]
- detect_plugin(workspace) -> Optional[PluginProtocol]
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import PluginProtocol, BasePlugin

__all__ = [
    "PluginProtocol",
    "BasePlugin",
    "PluginAmbiguityError",
    "register_plugin",
    "all_plugins",
    "get_plugin",
    "detect_plugin",
    "reset_registry_for_testing",
]


class PluginAmbiguityError(RuntimeError):
    """Two or more plugins claim the same workspace — we cannot tell which
    mode it is, so no mode-dispatched gate can run against it.

    This is a regular runtime error so normal exception cleanup, logging, and
    command boundaries still apply.  Production detection sites must either
    propagate it or convert it to an explicit fail-closed result; no caller
    may translate it to ``None``, whose sole meaning is "no plugin applies".
    """


_PLUGIN_REGISTRY: dict[str, PluginProtocol] = {}


def register_plugin(plugin: PluginProtocol) -> None:
    """Register a plugin. Idempotent for the same object; raises on
    name collision OR cli_flag collision with a different object.
    """
    if not getattr(plugin, "name", None):
        raise ValueError(
            f"plugin {plugin!r} has no `name` attribute set; cannot register"
        )
    existing = _PLUGIN_REGISTRY.get(plugin.name)
    if existing is not None:
        if existing is plugin:
            return  # idempotent
        raise ValueError(
            f"Duplicate plugin name {plugin.name!r}: existing={existing!r}, "
            f"new={plugin!r}. Plugin names must be globally unique."
        )
    # P86 review C8: detect cli_flag collisions across plugins. Multiple
    # plugins MAY share the same flag, but the shared flag then needs a
    # DISAMBIGUATION rule at the CLI so the right plugin is dispatched.
    #
    # Shared flags require an explicit disambiguation rule. None are currently
    # sanctioned, so every collision is reported.
    flag = getattr(plugin, "cli_flag", None)
    if flag is not None:
        for other in _PLUGIN_REGISTRY.values():
            if getattr(other, "cli_flag", None) == flag:
                import sys as _sys
                print(
                    f"[plugins] WARNING: cli_flag collision on {flag!r} — "
                    f"{other.name!r} vs {plugin.name!r}. Two plugins share a CLI "
                    f"flag with no disambiguation rule; add one (see "
                    f"_INTENTIONAL_SHARED_CLI_FLAGS) before wiring CLI dispatch.",
                    file=_sys.stderr,
                )
    _PLUGIN_REGISTRY[plugin.name] = plugin


def all_plugins() -> tuple[PluginProtocol, ...]:
    """All registered plugins in deterministic order (sorted by name)."""
    return tuple(_PLUGIN_REGISTRY[n] for n in sorted(_PLUGIN_REGISTRY))


def get_plugin(name: str) -> Optional[PluginProtocol]:
    """Lookup by exact name. None if not registered."""
    return _PLUGIN_REGISTRY.get(name)


def detect_plugin(workspace: Path) -> Optional[PluginProtocol]:
    """Find the unique plugin whose detect(workspace) returns True.

    Returns `None` for exactly one condition: **no plugin claims this
    workspace**. `None` never means "we could not tell" — that case raises
    `PluginAmbiguityError`. Callers must propagate that error or convert it to
    an explicit fail-closed result; they must never turn it into ``None``.
    Callers may therefore read ``None`` as the settled fact "this workspace
    has no mode" and skip mode-dispatched work on purpose, rather than
    skipping it because the answer went missing (DEBT-216).

    Mutual exclusivity is enforced: if 2+ plugins claim the workspace, raise.
    That indicates a detect() logic bug — DO NOT silently pick one; the
    offending plugin's detect rules must be fixed. `plugins/layout.py` holds
    the shared defer helpers that keep the rules exclusive.
    """
    matches = [p for p in all_plugins() if p.detect(workspace)]
    if len(matches) > 1:
        raise PluginAmbiguityError(
            f"Workspace {workspace} matched by {len(matches)} plugins: "
            f"{[p.name for p in matches]}. detect() rules must be "
            f"mutually exclusive; fix the offending plugin's detect logic. "
            f"Until then NO plugin-dispatched gate can run on this "
            f"workspace — it is unGATED, not clean."
        )
    return matches[0] if matches else None


def reset_registry_for_testing() -> None:
    """Test-only: clear the registry so tests can install fakes.
    NEVER call from production code.
    """
    _PLUGIN_REGISTRY.clear()


def _discover_plugins() -> None:
    """Import the explicit allowlist; fail loudly if either required mode breaks."""
    import importlib
    import sys as _sys
    for module_name in ("port_a3", "backward"):
        try:
            importlib.import_module(f"{__name__}.{module_name}")
        except Exception as e:
            print(
                f"[plugins] ERROR: required plugin {module_name!r} failed to import "
                f"({type(e).__name__}: {e})",
                file=_sys.stderr,
            )
            raise


# Register the fixed customer scope at import time.
_discover_plugins()
