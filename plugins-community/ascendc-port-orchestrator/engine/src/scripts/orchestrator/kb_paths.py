# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Canonical packaged orchestration-rule root resolver.

The knowledge base used to live at ``engine/src/skills/references/`` and was
resolved throughout the engine as ``<engine_root>/src/skills/references``.
Official b-tier knowledge is consumed from cannbot-knowledge through
``briefs.external_kb``. The only knowledge-adjacent content resolved here is
the plugin's packaged ``kb/shared`` orchestration discipline.

Layout reminder::

    <plugin_root>/
      engine/
        src/scripts/orchestrator/kb_paths.py   <- this file
      kb/                                       <- plugin runtime assets only
        shared/              # orchestration discipline, not domain knowledge
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


def shared_kb_root() -> Path:
    """Absolute path to packaged, non-domain ``<plugin_root>/kb/shared`` rules."""
    return _PLUGIN_ROOT / "kb" / "shared"
