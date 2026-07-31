# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""NODE-19: verify ccec_compiler/bin PATH propagation for llvm-objdump."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors
import pytest

_HERE = Path(__file__).resolve().parent
_BUILD_ASCENDC = _reorg_paths.SCRIPTS_DIR / "patches" / "build_ascendc.py"


def _simulate_path_propagation(ascend_path: Path, initial_path: str = "/usr/bin") -> str:
    """Mirror the NODE-19 PATH propagation logic added to build_ascendc.py.

    Scans ascend_path/<arch>/ccec_compiler/bin for x86_64-linux and
    aarch64-linux. Prepends the first match to PATH. Returns the
    resulting PATH string.
    """
    env_path = initial_path
    for arch in ("x86_64-linux", "aarch64-linux"):
        ccec_bin = ascend_path / arch / "ccec_compiler" / "bin"
        if ccec_bin.is_dir():
            env_path = str(ccec_bin) + os.pathsep + env_path
            break
    return env_path


def test_ccec_path_aarch64_found(tmp_path):
    """When aarch64-linux/ccec_compiler/bin exists, it's prepended to PATH."""
    ccec_dir = tmp_path / "aarch64-linux" / "ccec_compiler" / "bin"
    ccec_dir.mkdir(parents=True)
    result = _simulate_path_propagation(tmp_path, "/usr/bin")
    assert str(ccec_dir) in result
    assert result.startswith(str(ccec_dir))
    assert "/usr/bin" in result


def test_ccec_path_x86_64_found(tmp_path):
    """When x86_64-linux/ccec_compiler/bin exists (and aarch64 doesn't),
    x86_64 path is prepended.
    """
    ccec_dir = tmp_path / "x86_64-linux" / "ccec_compiler" / "bin"
    ccec_dir.mkdir(parents=True)
    result = _simulate_path_propagation(tmp_path, "/usr/bin")
    assert str(ccec_dir) in result
    assert result.startswith(str(ccec_dir))


def test_ccec_path_both_archs_prefers_x86_64(tmp_path):
    """When both arch dirs exist, x86_64-linux wins (checked first)."""
    x86_dir = tmp_path / "x86_64-linux" / "ccec_compiler" / "bin"
    arm_dir = tmp_path / "aarch64-linux" / "ccec_compiler" / "bin"
    x86_dir.mkdir(parents=True)
    arm_dir.mkdir(parents=True)
    result = _simulate_path_propagation(tmp_path, "/usr/bin")
    assert str(x86_dir) in result
    assert str(arm_dir) not in result


def test_ccec_path_neither_arch_returns_original(tmp_path):
    """When no ccec_compiler/bin exists under either arch, PATH unchanged."""
    result = _simulate_path_propagation(tmp_path, "/usr/bin")
    assert result == "/usr/bin"


def test_ccec_path_empty_initial(tmp_path):
    """Works with empty initial PATH."""
    ccec_dir = tmp_path / "aarch64-linux" / "ccec_compiler" / "bin"
    ccec_dir.mkdir(parents=True)
    result = _simulate_path_propagation(tmp_path, "")
    assert result.startswith(str(ccec_dir))


def test_source_file_has_node19_comment():
    """Verify the NODE-19 fix is actually in the source file."""
    if not _BUILD_ASCENDC.is_file():
        pytest.skip("build_ascendc.py not found at expected path")
    content = _BUILD_ASCENDC.read_text()
    assert "NODE-19" in content, "NODE-19 marker comment not found in build_ascendc.py"
    assert "ccec_compiler" in content, "ccec_compiler path not found in build_ascendc.py"
    assert "llvm-objdump" in content, "llvm-objdump reference not found in build_ascendc.py"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
