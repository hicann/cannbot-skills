#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Skill instruction loader for non-Claude harness backends."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_SAFE_SKILL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_PROMPT_MANAGED_SKILLS = {
    "aog-self-critic",
}


def load_skill_context(target: str) -> Optional[str]:
    """Return local SKILL.md text for `target`, or None if unavailable.

    Claude Code resolves `/skill-name` itself. Codex/opencode do not share that
    resolver, so their backend adapters inline the repo-local skill instructions
    for ordinary skill dispatches. Some orchestrator calls already build a slim,
    self-contained skill prompt; do not re-inline their full SKILL.md here.
    """
    skill_name = target.strip().lstrip("/")
    if not skill_name or not _SAFE_SKILL_RE.match(skill_name):
        return None
    if skill_name in _PROMPT_MANAGED_SKILLS:
        return None

    # Plugin-private skills are packaged beside engine/, not duplicated under
    # engine/src/skills or published into the repository-wide ops/ pool.
    skills_root = Path(__file__).resolve().parents[5] / "skills"
    skill_path = (skills_root / skill_name / "SKILL.md").resolve()
    try:
        skill_path.relative_to(skills_root.resolve())
    except ValueError:
        return None
    if not skill_path.is_file():
        return None
    return (
        f"=== Skill instructions from {skill_path} ===\n\n"
        f"{skill_path.read_text(errors='replace')}\n\n"
        "=== End skill instructions ===\n"
    )
