# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for deploy_to_npu.sh OP_SLUG extraction (P0i).

Day 4 op#10 finding: ${ASCENDC_WORKSPACE##*/} returns empty when var ends
with `/`. Caused archive to land at /build_archive//<batch>/iter_*.

Fix: use `basename` which handles trailing slashes correctly. These tests
verify the slug extraction logic invoked via bash subprocess.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

_BASH = shutil.which("bash")


def _bash() -> str:
    if _BASH is None:
        pytest.skip("bash executable not found")
    return _BASH


def _slug(workspace_value: str) -> str:
    """Mirror the deploy_to_npu.sh OP_SLUG line under test (basename approach)."""
    # Pass workspace_value as $1 so we don't interpolate it into the shell
    # script body (avoids quoting hell + shell injection in tests).
    result = subprocess.run(
        [_bash(), "-c", 'basename "$1" 2>/dev/null', "_", workspace_value],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip()


def _v1_slug(workspace_value: str) -> str:
    """The OLD broken version: bash parameter expansion ${var##*/}."""
    result = subprocess.run(
        [_bash(), "-c", 'echo "${1##*/}"', "_", workspace_value],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Documenting the V1 bug (regression-detector)
# ---------------------------------------------------------------------------
def test_v1_bug_returns_empty_on_trailing_slash():
    """The bug we're fixing: trailing slash → empty slug."""
    assert _v1_slug("/home/x/workspace/10_layernorm/") == ""


def test_v1_bug_works_normally_without_trailing_slash():
    """V1 worked correctly when path had no trailing slash — that's why
    op#17/op#22 archived correctly but op#10 did not.
    """
    assert _v1_slug("/home/x/workspace/22_Nonzero") == "22_Nonzero"


# ---------------------------------------------------------------------------
# New basename-based behavior
# ---------------------------------------------------------------------------
def test_basename_normal_path():
    assert _slug("/home/x/workspace/22_Nonzero") == "22_Nonzero"


def test_basename_trailing_slash():
    assert _slug("/home/x/workspace/10_layernorm/") == "10_layernorm"


def test_basename_multiple_trailing_slashes():
    assert _slug("/home/x/workspace/10_layernorm///") == "10_layernorm"


def test_basename_double_internal_slashes():
    """Double internal slashes (//) are normalized by basename."""
    assert _slug("/home/x/workspace//10_layernorm") == "10_layernorm"


def test_basename_relative_path():
    """basename of relative path without leading / still works."""
    assert _slug("workspace/22_Nonzero") == "22_Nonzero"


def test_basename_just_dirname():
    """Bare dir name."""
    assert _slug("17_AdamW") == "17_AdamW"


def test_basename_empty_returns_empty():
    """Empty string → basename returns empty (handled by defensive check)."""
    assert _slug("") == ""


def test_basename_root_returns_slash():
    """Pure root '/' → basename returns '/'. Caller treats this as bad input."""
    assert _slug("/") == "/"


def test_basename_dot_returns_dot():
    """Cwd '.' → basename returns '.'. Caller treats this as bad input."""
    assert _slug(".") == "."


# ---------------------------------------------------------------------------
# Defensive sentinel check (what the script does with empty result)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ws,expected_skip", [
    ("/home/x/workspace/22_Nonzero", False),
    ("/home/x/workspace/10_layernorm/", False),  # FIXED by basename
    ("", True),
    ("/", True),
    (".", True),
])
def test_skip_logic_matches_script(ws, expected_skip):
    """Verify the SAME skip decision as the deploy_to_npu.sh
    `[ -z "$OP_SLUG" ] || [ "$OP_SLUG" = "/" ] || [ "$OP_SLUG" = "." ]` test.
    """
    slug = _slug(ws)
    skip = slug == "" or slug == "/" or slug == "."
    assert skip == expected_skip, f"ws={ws!r} slug={slug!r}"


def test_basename_with_special_chars():
    """Op names can have underscores, digits, capital letters."""
    assert _slug("/home/x/workspace/19_FusedResidualRmsNormBackward") == \
        "19_FusedResidualRmsNormBackward"


def test_basename_with_legacy_lowercase_slug():
    """Legacy LLM-orchestrator-era slugs with all-lowercase + underscores."""
    assert _slug("/home/x/workspace/10_layernorm") == "10_layernorm"
