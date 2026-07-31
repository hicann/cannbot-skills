# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Canonical KB-root resolver (2026-07-05 skills/ → plugin-root refactor).

The knowledge base used to live at ``engine/src/skills/references/`` and was
resolved throughout the engine as ``<engine_root>/src/skills/references``.
It now lives at ``<plugin_root>/kb/`` (sibling to ``engine/``, ``agents/``,
``hooks/``). Every engine site that needs the KB should resolve it through
``kb_root()`` (or the equivalent ``Path(__file__)...parents[N] / "kb"`` form
for modules that cannot cleanly import this helper), so the path stays correct
regardless of the current working directory.

Layout reminder::

    <plugin_root>/
      engine/
        src/scripts/orchestrator/kb_paths.py   <- this file
      kb/                                       <- the relocated KB
        KB_INDEX.md
        target/ascendc/OPERATIONAL_KNOWLEDGE.md
        shared/{GATE_CONTRACT,ANTI_PRESSURE_PROTOCOLS}.md
        hardware/ ...
"""
from __future__ import annotations

from pathlib import Path

# This file lives below the plugin root in engine/src/scripts/orchestrator.
# Its successive parents are orchestrator, scripts, src, engine, and then the
# plugin root itself.
_PLUGIN_ROOT = Path(__file__).resolve().parents[4]


def plugin_root() -> Path:
    """Absolute path to the plugin root (parent of ``engine/``)."""
    return _PLUGIN_ROOT


def kb_root() -> Path:
    """Absolute path to the relocated knowledge base (``<plugin_root>/kb``)."""
    return _PLUGIN_ROOT / "kb"
