# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DEBT-197: deploy_to_npu.sh local self-heal must provision the FULL build baseline
trio, not just the entrypoint.

build_ascendc.py does `from build_capabilities import ...` after inserting its own dir on
sys.path, and consumes cann_stubs/ as the DEBT-110 baseline include. The old self-heal
provisioned ONLY build_ascendc.py, so a truly fresh container hit
`ModuleNotFoundError: No module named 'build_capabilities'` — an empty-stderr phantom
compile failure. These tests drive the local self-heal in isolation (no NPU / no remote
host) via the DEBT197_SELFHEAL_TEST_ONLY hook in deploy_to_npu.sh and assert the whole
dependency closure lands and is importable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_SH = REPO_ROOT / "src" / "scripts" / "deploy_to_npu.sh"
_BASH = shutil.which("bash")

TRIO = ("build_ascendc.py", "build_capabilities", "cann_stubs")


def _bash() -> str:
    if _BASH is None:
        pytest.skip("bash executable not found")
    return _BASH


def _make_fake_patches(patch_dir: Path, *, build_capabilities_as_file: bool = False,
                       omit: tuple[str, ...] = ()) -> None:
    """Stage a minimal but realistic patches/ bundle.

    build_capabilities is a real importable package (has __init__.py) unless overridden.
    """
    patch_dir.mkdir(parents=True, exist_ok=True)
    if "build_ascendc.py" not in omit:
        (patch_dir / "build_ascendc.py").write_text(
            "import sys, os\n"
            "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
            "from build_capabilities import MARKER\n"
            "print('build_ascendc stub', MARKER)\n"
        )
    if "build_capabilities" not in omit:
        if build_capabilities_as_file:
            # Deliberately wrong shape: a file, not a package dir → guard must fire.
            (patch_dir / "build_capabilities").write_text("not a package\n")
        else:
            pkg = patch_dir / "build_capabilities"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("MARKER = 'debt197-ok'\n")
            (pkg / "baseline.py").write_text("VALUE = 1\n")
    if "cann_stubs" not in omit:
        stubs = patch_dir / "cann_stubs"
        stubs.mkdir()
        (stubs / "platform_ascendc.h").write_text("// stub header\n")
        (stubs / "tiling").mkdir()
        (stubs / "tiling" / "tiling.h").write_text("// nested stub\n")


def _run_selfheal(tmp_path: Path, patch_dir: Path, build_root: Path) -> subprocess.CompletedProcess:
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(
        DEBT197_SELFHEAL_TEST_ONLY="1",
        DEBT197_SELFHEAL_TEST_BUILD_ROOT=str(build_root),
        DEBT197_SELFHEAL_TEST_PATCH_DIR=str(patch_dir),
        ASCENDC_WORKSPACE=str(marker_dir),
    )
    proc = subprocess.run(
        [_bash(), str(DEPLOY_SH)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    proc.stderr_marker = (marker_dir / ".last_build.stderr")  # type: ignore[attr-defined]
    return proc


def _import_build_capabilities(utils: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(utils)!r}); "
            "import build_capabilities; print(build_capabilities.MARKER)",
        ],
        text=True,
        capture_output=True,
    )


def test_selfheal_provisions_full_trio_and_import_works(tmp_path: Path) -> None:
    patch_dir = tmp_path / "patches"
    build_root = tmp_path / "AscendOpGenAgent"
    (build_root / "utils").mkdir(parents=True)  # fresh, empty utils/
    _make_fake_patches(patch_dir)

    proc = _run_selfheal(tmp_path, patch_dir, build_root)
    assert proc.returncode == 0, proc.stderr + proc.stdout

    utils = build_root / "utils"
    for piece in TRIO:
        assert (utils / piece).exists(), f"{piece} not provisioned: {proc.stdout}"
    assert (utils / "build_capabilities").is_dir()
    assert (utils / "build_capabilities" / "__init__.py").is_file()
    assert (utils / "cann_stubs").is_dir()
    assert (utils / "cann_stubs" / "tiling" / "tiling.h").is_file()  # recursive copy

    # Proof the ModuleNotFoundError is gone: build_capabilities imports off utils/.
    imp = _import_build_capabilities(utils)
    assert imp.returncode == 0, imp.stderr
    assert "debt197-ok" in imp.stdout


def test_selfheal_is_idempotent(tmp_path: Path) -> None:
    patch_dir = tmp_path / "patches"
    build_root = tmp_path / "AscendOpGenAgent"
    (build_root / "utils").mkdir(parents=True)
    _make_fake_patches(patch_dir)

    first = _run_selfheal(tmp_path, patch_dir, build_root)
    assert first.returncode == 0, first.stderr + first.stdout

    # Mutate a provisioned file to prove the re-run does NOT overwrite existing pieces.
    sentinel = build_root / "utils" / "build_capabilities" / "__init__.py"
    sentinel.write_text("MARKER = 'debt197-ok'\n# sentinel-untouched\n")

    second = _run_selfheal(tmp_path, patch_dir, build_root)
    assert second.returncode == 0, second.stderr + second.stdout
    # No duplication, no error, existing piece left untouched (idempotent skip).
    assert "sentinel-untouched" in sentinel.read_text()
    for piece in TRIO:
        assert (build_root / "utils" / piece).exists()


def test_selfheal_fires_when_only_entrypoint_present(tmp_path: Path) -> None:
    """A container can have build_ascendc.py but not the deps — self-heal must still fire
    and provision the missing dirs (trigger = ANY piece missing).
    """
    patch_dir = tmp_path / "patches"
    build_root = tmp_path / "AscendOpGenAgent"
    utils = build_root / "utils"
    utils.mkdir(parents=True)
    _make_fake_patches(patch_dir)
    # Pre-place ONLY the entrypoint (deps absent) — the old code would no-op here.
    (utils / "build_ascendc.py").write_text("# preexisting entrypoint\n")

    proc = _run_selfheal(tmp_path, patch_dir, build_root)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (utils / "build_capabilities").is_dir()
    assert (utils / "cann_stubs").is_dir()
    # Entrypoint left untouched (idempotent skip).
    assert "preexisting entrypoint" in (utils / "build_ascendc.py").read_text()


def test_missing_bundled_piece_fails_with_clear_marker(tmp_path: Path) -> None:
    """If the bundle lacks a required piece, keep the existing FAIL path with a clear,
    non-empty stderr marker (never an empty-stderr phantom failure).
    """
    patch_dir = tmp_path / "patches"
    build_root = tmp_path / "AscendOpGenAgent"
    (build_root / "utils").mkdir(parents=True)
    _make_fake_patches(patch_dir, omit=("build_capabilities",))

    proc = _run_selfheal(tmp_path, patch_dir, build_root)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    marker_txt = proc.stderr_marker.read_text()  # type: ignore[attr-defined]
    assert marker_txt.strip(), "stderr marker must not be empty (no phantom failure)"
    assert "build_capabilities" in marker_txt


def test_guard_fires_when_build_capabilities_not_a_package_dir(tmp_path: Path) -> None:
    """Guard: if build_capabilities lands as something other than a package DIR, emit a
    clear diagnostic instead of letting build_ascendc.py raise ModuleNotFoundError.
    """
    patch_dir = tmp_path / "patches"
    build_root = tmp_path / "AscendOpGenAgent"
    (build_root / "utils").mkdir(parents=True)
    _make_fake_patches(patch_dir, build_capabilities_as_file=True)

    proc = _run_selfheal(tmp_path, patch_dir, build_root)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    marker_txt = proc.stderr_marker.read_text()  # type: ignore[attr-defined]
    assert "guard" in marker_txt.lower()
    assert "build_capabilities" in marker_txt
    assert "ModuleNotFoundError" in marker_txt
