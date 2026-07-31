# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Stable on-disk anchors for tests after the ut/it/ct reorg (2026-06).

Background
----------
Many test files locate DATA on disk — shipped docs (GATE_CONTRACT.md, SKILL.md),
the FSM yaml (workflows/opgen_state_machine.yaml), KB references, .py module
sources to grep / importlib-load, deploy shell scripts, lint scripts + fixtures,
real output/<op>/REPORT.md — by computing a path from their own __file__, e.g.
``Path(__file__).resolve().parents[4]`` (repo root) or
``Path(__file__).resolve().parent.parent`` (the orchestrator/ dir), under the
assumption the file lived DIRECTLY in orchestrator/tests/.

The ut/it/ct split moved every such file one directory deeper
(tests/ -> tests/{ut,it,ct}/), shifting those fixed-depth computations by one
and breaking the data path (the module-IMPORT paths are handled separately and
centrally by tests/conftest.py).

This module computes the anchors ONCE from a STABLE marker — it lives in
orchestrator/tests/_reorg_paths.py regardless of which subdir imports it, and
walks up to the repo root identified by the ``src/skills/references`` directory.
Depth therefore no longer matters: a future move of the test files won't break
these anchors. Tests import this and use ``_reorg_paths.REPO_ROOT`` etc. instead
of a fragile ``Path(__file__)...parent`` chain.

(tests/ is placed on sys.path by tests/conftest.py so ``import _reorg_paths``
resolves from any subdir.)
"""
from __future__ import annotations

from pathlib import Path


def _find_repo_root() -> Path:
    """Walk up from this file to the repo root (the dir holding src/skills/references)."""
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "src" / "skills" / "references").is_dir():
            return ancestor
    # Fallback by known structure: _reorg_paths.py is at
    # <repo>/src/scripts/orchestrator/tests/_reorg_paths.py → parents[4] == <repo>.
    return here.parents[4]


REPO_ROOT: Path = _find_repo_root()
SRC_DIR: Path = REPO_ROOT / "src"
SCRIPTS_DIR: Path = SRC_DIR / "scripts"
ORCH_DIR: Path = SCRIPTS_DIR / "orchestrator"
WORKFLOW_DIR: Path = SCRIPTS_DIR / "workflow"

__all__ = ["REPO_ROOT", "SRC_DIR", "SCRIPTS_DIR", "ORCH_DIR", "WORKFLOW_DIR"]
