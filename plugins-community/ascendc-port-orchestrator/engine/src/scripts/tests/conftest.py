# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Shared fixtures for src/scripts/tests/.

Exists for one reason: make this suite hermetic against a leaked git
environment. See `_strip_git_env` below.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _strip_git_env():
    """Strip GIT_* from the environment for every test in this directory.

    Mirrors the DEBT-47b fixture in src/scripts/orchestrator/tests/conftest.py
    (`_isolate_global_state`); the reasoning is identical, so the two should
    stay in sync.

    Git exports GIT_DIR / GIT_INDEX_FILE into any process it spawns — a hook
    (GIT_DIR, always) or a partial commit (GIT_INDEX_FILE, a temp index) — and
    those variables OVERRIDE a subprocess's cwd. Any test helper that builds a
    throwaway repo does so with
    `subprocess.run(["git", ...], cwd=tmp_path)`, which is correct when a human
    runs pytest and actively destructive when anything with a git env does:
    `git init` resolves to the REAL repo's git dir, and because no GIT_WORK_TREE
    accompanies it, re-initializes that repo as BARE (core.bare false -> true,
    which breaks `git status` in every linked worktree). The subsequent
    `git add -A` + `git commit -qm x` then land in the real repo, moving HEAD
    and leaving junk commits named `x`.

    Stripping GIT_* makes git resolve the repo from cwd/-C, i.e. the tmp repo,
    which is what the fixtures always meant. The suite still passes either way,
    so this damage is silent — hence a callee-side fixture rather than trusting
    every caller to sanitize first.

    Restoring afterwards keeps the strip from leaking out of the test.
    """
    snapshot = dict(os.environ)
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        del os.environ[key]

    yield

    for key in [k for k in os.environ if k not in snapshot]:
        del os.environ[key]
    os.environ.update(snapshot)
