# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Golden byte-identical backward-compat gate for build_ascendc._generate_cmakelists.

The load-bearing proof of 零行为变化 for the build-capability extension mechanism
(docs/design/BUILD_CAPABILITY_EXTENSION_DESIGN.md §3): after DEBT-20/20.1 (per-source
COMPILE_DEFINITIONS) and DEBT-110 (cann_stubs include + conditional nnopbase/opapi link)
are refactored into BASELINE CMakeContribution modules, the generated CMakeLists text
MUST be byte-identical to the pre-refactor output for every op that declares NO
`capabilities`.

Four fixtures (design §3):
  (a) plain      — op with no build_overrides.json
  (b) legacy     — legacy list-form per_source_defines (DEBT-20)
  (c) per_pass   — per-compile-pass aic/aiv defines (DEBT-20.1)
  (d) cann_stubs — cann_stubs / host-tiling context with a catlass include (DEBT-110)

Goldens are captured from the PRE-refactor code by running this file as a script
(`python3 test_golden_cmakelists.py --capture`) and committed under tests/goldens/.
The test then renders the SAME fixtures with the refactored code and asserts the
normalized text is byte-identical to each golden. Absolute (tmp / repo) paths are
normalized to stable tokens so the goldens are machine-independent.
"""
from __future__ import annotations

import logging

import importlib.util
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_PATCHES_DIR = _HERE.parent
_GOLDEN_DIR = _HERE / "goldens"
_BUILD_ASCENDC = _PATCHES_DIR / "build_ascendc.py"

_FIXTURES = ("plain", "legacy", "per_pass", "cann_stubs")


def _load_build_ascendc():
    # patches/ on sys.path so the module's `from build_capabilities import ...`
    # (present post-refactor; absent pre-refactor) resolves either way.
    if str(_PATCHES_DIR) not in sys.path:
        sys.path.insert(0, str(_PATCHES_DIR))
    spec = importlib.util.spec_from_file_location("_build_ascendc_golden", _BUILD_ASCENDC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_overrides(kernel_dir: Path, kind: str) -> None:
    import json

    if kind == "plain":
        return
    if kind == "legacy":
        payload = {"per_source_defines": {"x_kernels.cpp": ["ASCENDC_MATMUL_AICORE", "FOO=1"]}}
    elif kind == "per_pass":
        payload = {
            "per_source_defines": {
                "x_kernels.cpp": {
                    "global": ["GLOBAL_FLAG=1"],
                    "aic": ["SPLIT_CORE_CUBE=1"],
                    "aiv": ["SPLIT_CORE_VEC=1"],
                }
            }
        }
    elif kind == "cann_stubs":
        # host-tiling / CUBE context: a global define + a catlass include dir present.
        payload = {"per_source_defines": {"x_kernels.cpp": {"global": ["ENABLE_HOST_TILING=1"]}}}
    else:
        raise ValueError(kind)
    (kernel_dir / "build_overrides.json").write_text(json.dumps(payload), encoding="utf-8")


def _setup_fixture(tmp_path: Path, kind: str) -> tuple[Path, Path, list[Path]]:
    kernel_dir = tmp_path / kind / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "x_kernels.cpp").write_text("// stub\n", encoding="utf-8")
    (kernel_dir / "pybind11.cpp").write_text("// pybind\n", encoding="utf-8")
    if kind == "cann_stubs":
        # host-tiling context brings a task-local catlass include tree.
        (kernel_dir / "catlass" / "include").mkdir(parents=True, exist_ok=True)
    _write_overrides(kernel_dir, kind)
    build_dir = kernel_dir / "build"
    sources = [kernel_dir / "x_kernels.cpp"]
    return kernel_dir, build_dir, sources


def _cann_stubs_dir(mod) -> Path:
    # Both pre- and post-refactor resolve cann_stubs as patches/cann_stubs.
    return _PATCHES_DIR / "cann_stubs"


def _normalize(text: str, kernel_dir: Path, build_dir: Path, cann_stubs: Path) -> str:
    # Longest / most-specific paths first (build_dir is under kernel_dir).
    text = text.replace(str(build_dir), "<BUILD_DIR>")
    text = text.replace(str(kernel_dir), "<KERNEL_DIR>")
    text = text.replace(str(cann_stubs), "<CANN_STUBS>")
    return text


def _render(mod, tmp_path: Path, kind: str) -> str:
    kernel_dir, build_dir, sources = _setup_fixture(tmp_path, kind)
    text = getattr(mod, '_generate_cmakelists')(
        kernel_dir=kernel_dir,
        build_dir=build_dir,
        module_name="x_ext",
        sources=sources,
        ascend_path=Path("/opt/cann"),
    )
    return _normalize(text, kernel_dir, build_dir, _cann_stubs_dir(mod))


@pytest.mark.parametrize("kind", _FIXTURES)
def test_golden_byte_identical(kind: str, tmp_path: Path):
    golden_path = _GOLDEN_DIR / f"{kind}.CMakeLists.golden"
    assert golden_path.is_file(), (
        f"missing golden {golden_path}; regenerate with "
        f"`python3 {Path(__file__).name} --capture` against the pre-refactor code"
    )
    mod = _load_build_ascendc()
    actual = _render(mod, tmp_path, kind)
    expected = golden_path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"CMakeLists for fixture '{kind}' is NOT byte-identical to golden "
        f"(no capabilities declared → must equal pre-refactor output)"
    )


def _capture() -> None:
    import tempfile

    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    mod = _load_build_ascendc()
    for kind in _FIXTURES:
        with tempfile.TemporaryDirectory() as td:
            text = _render(mod, Path(td), kind)
        (_GOLDEN_DIR / f"{kind}.CMakeLists.golden").write_text(text, encoding="utf-8")
        logging.info(f"[capture] wrote goldens/{kind}.CMakeLists.golden ({len(text)} bytes)")


if __name__ == "__main__":
    if "--capture" in sys.argv:
        _capture()
    else:
        raise SystemExit("run via pytest, or pass --capture to (re)generate goldens")
