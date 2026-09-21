#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Resolve the full cannbot-knowledge checkout installed for this project.

This is a small compatibility CLI for ``okf_kb.sh``. It intentionally shares
the orchestrator's root resolver: explicit ``CANNBOT_KNOWLEDGE_ROOT`` first,
then the nearest installer-owned ``.cannbot/knowledge.env``. Marketplace skill
packages and the plugin-local legacy cards are not knowledge roots.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ORCHESTRATOR = Path(__file__).resolve().parents[1] / "orchestrator"
if str(_ORCHESTRATOR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATOR))

from briefs.external_kb import external_kb_root  # noqa: E402


INSTALL_HINT = (
    "cannbot-knowledge is not installed for this project. Run:\n"
    "  bash /path/to/cannbot-knowledge/install.sh <claude|opencode> "
    "<operator-project> consumer\n"
    "then launch port from that project (or explicitly export "
    "CANNBOT_KNOWLEDGE_ROOT=/path/to/cannbot-knowledge)."
)


def knowledge_root() -> Path | None:
    """Return the validated full checkout root, never a plugin-local fallback."""
    return external_kb_root()


def knowledge_query_script() -> Path | None:
    root = knowledge_root()
    if root is None:
        return None
    script = (
        root
        / ".agents"
        / "skills"
        / "knowledge-query"
        / "scripts"
        / "knowledge_query.py"
    )
    return script if script.is_file() else None


def main(argv: list[str]) -> int:
    capability = argv[1] if len(argv) > 1 else ""
    if capability == "root":
        resolved = knowledge_root()
    elif capability == "query":
        resolved = knowledge_query_script()
    else:
        print("usage: okf_engine.py {root|query}", file=sys.stderr)
        return 64
    if resolved is None:
        print(INSTALL_HINT, file=sys.stderr)
        return 2
    print(resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
