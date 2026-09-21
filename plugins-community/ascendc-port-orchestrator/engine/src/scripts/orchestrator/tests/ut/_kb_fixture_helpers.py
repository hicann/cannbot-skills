# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Shared fixture-repo builders for the external-KB (OKF v0.2) contract tests."""
from __future__ import annotations

from pathlib import Path


def _valid_root(base: Path, name: str = "cannbot-knowledge") -> Path:
    """Build a minimal valid cannbot-knowledge checkout root under ``base``.

    Layout gates ``external_kb.external_kb_root()``'s validation: knowledge
    Bundle index + governance Registry + the knowledge-query script (a json
    argv echo, so the okf_engine shell-wrapper test can execute it).
    """
    root = base / name
    (root / "knowledge").mkdir(parents=True)
    (root / "knowledge" / "index.md").write_text("# knowledge\n")
    registry = root / "governance" / "schemas" / "registries.yaml"
    registry.parent.mkdir(parents=True)
    registry.write_text("domains: {}\n")
    query = (
        root
        / ".agents"
        / "skills"
        / "knowledge-query"
        / "scripts"
        / "knowledge_query.py"
    )
    query.parent.mkdir(parents=True)
    query.write_text(
        "import json, sys\n"
        "print(json.dumps(sys.argv[1:]))\n"
    )
    return root
