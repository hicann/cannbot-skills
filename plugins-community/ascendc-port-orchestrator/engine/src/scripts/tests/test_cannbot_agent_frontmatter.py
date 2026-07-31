# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Every shipped agent's YAML frontmatter must parse — regression anchor.

Found 2026-07-25 by running `claude plugin validate` (the official manifest validator)
for the first time, while checking whether the marketplace path could be exercised. It
reported, on the tagged v3.18.0 tree:

    frontmatter: YAML frontmatter failed to parse: YAML Parse error: Unexpected token.
    At runtime this agent loads with empty metadata (all frontmatter fields silently
    dropped).

`aog-kernel-optimizer.md` had an unquoted `description:` containing `Spawn hint: ...`.
YAML reads the second colon-space as a nested mapping, the block fails, and **every**
field goes — including `tools:`, which declares what that agent is allowed to use. It
does not error at runtime; the agent simply loads with nothing.

Ten sibling agents were fine, several of them because their description happens to be
quoted. Nothing enforced it, so the difference was accidental.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_AGENTS = Path(__file__).resolve().parents[4] / "agents"
_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)


def _agent_files():
    return sorted(_AGENTS.glob("*.md")) if _AGENTS.is_dir() else []


def test_agents_directory_is_found():
    """Positive control: an empty glob would make every test below vacuously pass."""
    assert _agent_files(), f"no agent .md files under {_AGENTS} — the check is scanning nothing"


@pytest.mark.parametrize("path", _agent_files(), ids=lambda p: p.name)
def test_agent_frontmatter_parses(path: Path):
    m = _FRONTMATTER.match(path.read_text(encoding="utf-8"))
    assert m, f"{path.name}: no YAML frontmatter block"
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:
        pytest.fail(
            f"{path.name}: frontmatter does not parse -> at runtime this agent loads "
            f"with EMPTY metadata and every field, including `tools:`, is dropped "
            f"silently. Usually an unquoted value containing ': ' — quote the whole "
            f"string. YAML said: {e}"
        )
    assert isinstance(data, dict), f"{path.name}: frontmatter is not a mapping"
    for field in ("name", "description"):
        assert data.get(field), f"{path.name}: `{field}` missing or empty after parse"
