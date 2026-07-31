# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Unit tests for the FA-class `tile_size_consistency` finalize gate.

Whitebox root-cause (2026-05-29): designer emits authoritative block_N in
design/tile_level/*.py but the translator hand-writes the kernel's FA_BLOCK_N
with no enforcement, silently shipping the slow verbatim sibling tile. The gate
under test asserts emitted-kernel tile constant == design block_N.

Tests call `finalize_check_tile_size_consistency(tmp_path, {})` directly.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "scripts" / "orchestrator"))

from plugins._fa_class_gate import (  # noqa: E402
    finalize_check_tile_size_consistency,
)


def _write_tile_level(ws: Path, body: str) -> None:
    d = ws / "design" / "tile_level"
    d.mkdir(parents=True, exist_ok=True)
    (d / "fa_tile.py").write_text(body)


def _write_kernel(ws: Path, body: str, name: str = "fa_cube.h") -> None:
    d = ws / "kernel"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body)


def test_tuple_form_mismatch_returns_violation(tmp_path):
    _write_tile_level(tmp_path, "block_M, block_N = 64, 128\n")
    _write_kernel(tmp_path, "constexpr uint32_t FA_BLOCK_N = 64;\n")
    res = finalize_check_tile_size_consistency(tmp_path, {})
    assert res is not None
    assert "128" in res
    assert "64" in res


def test_match_returns_none(tmp_path):
    _write_tile_level(tmp_path, "block_M, block_N = 64, 128\n")
    _write_kernel(tmp_path, "constexpr uint32_t FA_BLOCK_N = 128;\n")
    assert finalize_check_tile_size_consistency(tmp_path, {}) is None


def test_design_tile_level_absent_returns_none(tmp_path):
    # No design/tile_level/ at all.
    _write_kernel(tmp_path, "constexpr uint32_t FA_BLOCK_N = 64;\n")
    assert finalize_check_tile_size_consistency(tmp_path, {}) is None


def test_kernel_dir_absent_returns_none(tmp_path):
    _write_tile_level(tmp_path, "block_M, block_N = 64, 128\n")
    # No kernel/ dir.
    assert finalize_check_tile_size_consistency(tmp_path, {}) is None


def test_kernel_no_tile_constant_returns_none(tmp_path):
    _write_tile_level(tmp_path, "block_M, block_N = 64, 128\n")
    _write_kernel(tmp_path, "constexpr uint32_t SOME_OTHER = 32;\n")
    assert finalize_check_tile_size_consistency(tmp_path, {}) is None


def test_single_form_parse_match(tmp_path):
    _write_tile_level(tmp_path, "block_N = 128\n")
    _write_kernel(tmp_path, "constexpr uint32_t FA_BLOCK_N = 128;\n")
    assert finalize_check_tile_size_consistency(tmp_path, {}) is None


def test_single_form_parse_mismatch(tmp_path):
    _write_tile_level(tmp_path, "block_N = 128\n")
    _write_kernel(tmp_path, "constexpr uint32_t BLOCK_N = 64;\n")
    res = finalize_check_tile_size_consistency(tmp_path, {})
    assert res is not None
    assert "128" in res and "64" in res


def test_last_assignment_is_operative(tmp_path):
    # Two design assignments; the LAST one (128) is authoritative.
    _write_tile_level(tmp_path, "block_N = 64\nblock_N = 128\n")
    _write_kernel(tmp_path, "constexpr uint32_t FA_BLOCK_N = 128;\n")
    assert finalize_check_tile_size_consistency(tmp_path, {}) is None


def test_empty_tile_level_dir_returns_none(tmp_path):
    (tmp_path / "design" / "tile_level").mkdir(parents=True)
    _write_kernel(tmp_path, "constexpr uint32_t FA_BLOCK_N = 64;\n")
    assert finalize_check_tile_size_consistency(tmp_path, {}) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
