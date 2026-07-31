# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for check_tag_version_consistency (cut-tag / VERSION.md banner gate).

Covers the real 2026-07-23 incident (v3.17.0 tag cut while banner said V3.16.1)
plus the fail-safe non-block paths and the CLI exit codes.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "check_tag_version_consistency.py"
_spec = importlib.util.spec_from_file_location("check_tag_version_consistency", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(mod)


def _banner(v: str) -> str:
    return f"# a5_ops Version Info\n\n## Current version: **{v}** (2026-07-23)\n\nnotes\n"


# --- banner_version -------------------------------------------------------

def test_banner_version_extracts_and_normalizes():
    assert mod.banner_version(_banner("V3.17.0")) == "3.17.0"
    assert mod.banner_version(_banner("v3.17.0")) == "3.17.0"
    assert mod.banner_version(_banner("V3.17")) == "3.17"


def test_banner_version_missing_returns_none():
    assert mod.banner_version("no banner here\njust text\n") is None
    assert mod.banner_version("") is None


# --- tag_to_version -------------------------------------------------------

def test_tag_to_version_full_ref_and_bare():
    assert mod.tag_to_version("refs/tags/v3.17.0") == "3.17.0"
    assert mod.tag_to_version("v3.17.0") == "3.17.0"
    assert mod.tag_to_version("v3.17") == "3.17"


def test_tag_to_version_non_version_tag_is_none():
    assert mod.tag_to_version("refs/tags/checkpoint-foo") is None
    assert mod.tag_to_version("checkpoint-foo") is None


def test_tag_to_version_branch_ref_is_none():
    assert mod.tag_to_version("refs/heads/main") is None
    assert mod.tag_to_version("garbage") is None
    assert mod.tag_to_version("") is None


# --- check ----------------------------------------------------------------

def test_check_match_ok():
    ok, msg = mod.check("v3.17.0", _banner("V3.17.0"))
    assert ok is True
    assert "3.17.0" in msg


def test_check_real_incident_blocks():
    # The 2026-07-23 incident: tag v3.17.0 pushed while banner still said V3.16.1.
    ok, msg = mod.check("v3.17.0", _banner("V3.16.1"))
    assert ok is False
    assert "3.17.0" in msg and "3.16.1" in msg


def test_check_patch_normalization_both_directions():
    ok1, _ = mod.check("v3.17", _banner("V3.17.0"))
    assert ok1 is True
    ok2, _ = mod.check("v3.17.0", _banner("V3.17"))
    assert ok2 is True


def test_check_non_version_tag_is_noop():
    ok, msg = mod.check("refs/tags/checkpoint-foo", _banner("V3.16.1"))
    assert ok is True
    assert "not a version tag" in msg


def test_check_branch_ref_is_noop():
    ok, _ = mod.check("refs/heads/main", _banner("V3.16.1"))
    assert ok is True


def test_check_banner_missing_is_failsafe_pass():
    ok, msg = mod.check("v3.17.0", "no banner here at all\n")
    assert ok is True
    assert "could not verify" in msg.lower()


# --- CLI (main) -----------------------------------------------------------

def test_cli_mismatch_exits_1(tmp_path: Path, capsys):
    vf = tmp_path / "VERSION.md"
    vf.write_text(_banner("V3.16.1"), encoding="utf-8")
    rc = mod.main(["v3.17.0", str(vf)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "MISMATCH" in out


def test_cli_match_exits_0(tmp_path: Path, capsys):
    vf = tmp_path / "VERSION.md"
    vf.write_text(_banner("V3.17.0"), encoding="utf-8")
    rc = mod.main(["v3.17.0", str(vf)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out


def test_cli_non_version_tag_exits_0(tmp_path: Path):
    vf = tmp_path / "VERSION.md"
    vf.write_text(_banner("V3.16.1"), encoding="utf-8")
    assert mod.main(["checkpoint-foo", str(vf)]) == 0


def test_cli_unreadable_file_is_failsafe_exit_0(tmp_path: Path):
    missing = tmp_path / "nope.md"
    assert mod.main(["v3.17.0", str(missing)]) == 0
