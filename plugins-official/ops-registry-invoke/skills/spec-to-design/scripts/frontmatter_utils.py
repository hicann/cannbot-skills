# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared YAML frontmatter parsing for PLAN.md validation and assembly."""

from __future__ import annotations

import re


def _parse_frontmatter_fallback(frontmatter_text: str) -> dict:
    """Parse simple ``key: value`` frontmatter without PyYAML.

    Raises ValueError when nested lists are present, since the regex fallback
    cannot represent them.
    """
    if re.search(r"^\s*-\s", frontmatter_text, re.MULTILINE):
        raise ValueError(
            "PyYAML is required to parse PLAN.md frontmatter containing "
            "nested lists. Install with: pip install PyYAML"
        )
    frontmatter: dict = {}
    for line in frontmatter_text.splitlines():
        match = re.match(r"^(\w+):\s*(.+)$", line.strip())
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        try:
            frontmatter[key] = int(value)
        except ValueError:
            frontmatter[key] = value
    return frontmatter


def _load_frontmatter(frontmatter_text: str) -> dict:
    """Load frontmatter via PyYAML, falling back to a regex parser if absent."""
    try:
        import yaml
    except ImportError as exc:
        try:
            return _parse_frontmatter_fallback(frontmatter_text)
        except ValueError as err:
            raise ValueError(str(err)) from exc

    try:
        return yaml.safe_load(frontmatter_text) or {}
    except Exception:
        return {}


def parse_yaml_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a markdown document.

    Returns (frontmatter_dict, body_text). If no frontmatter is found,
    returns ({}, original_text).

    Raises ValueError when PyYAML is unavailable and the frontmatter
    contains nested structures (lists, mappings) that the regex fallback
    cannot parse.
    """
    if not text.startswith("---"):
        return {}, text
    end_match = re.search(r"\n---\s*\n", text[3:])
    if not end_match:
        return {}, text
    frontmatter_text = text[3:end_match.start() + 3]
    body = text[end_match.end() + 3:]
    return _load_frontmatter(frontmatter_text), body
