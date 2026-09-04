# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression: build_ascendc.py's CMake torch/torch_npu path probe must use
importlib.find_spec, NOT `import torch_npu`.

2026-06-20 (FA-grad .141 V351 clean-runtime lane): `import torch_npu` at
cmake-configure loads the torch_npu `_C` extension, which ABI-trips on a
clean-runtime container whose torch_npu wheel is ABI-paired to a device CANN
that differs from the build toolchain (the .post5/B060 case → `free(): invalid
pointer` / libruntime ErrorManager symbol mismatch). `find_spec(...).origin`
returns the package dir WITHOUT importing `_C`, so the include/lib paths resolve
even when the runtime import would crash, and works identically on a fully-paired
container. This test locks the fix so the next clean-runtime customer keeps it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_BUILD_ASCENDC = (
    Path(__file__).resolve().parent.parent / "patches" / "build_ascendc.py"
)


def _load_build_ascendc():
    spec = importlib.util.spec_from_file_location("_build_ascendc_under_test", _BUILD_ASCENDC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gen_cmake(tmp_path: Path) -> str:
    mod = _load_build_ascendc()
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "x_kernels.cpp").write_text("// stub\n")
    return getattr(mod, '_generate_cmakelists')(
        kernel_dir=kernel_dir,
        build_dir=tmp_path / "build",
        module_name="x_ext",
        sources=[kernel_dir / "x_kernels.cpp"],
        ascend_path=Path("/opt/cann"),
    )


def _probe_command_lines(cmake: str) -> list[str]:
    """The `execute_process(COMMAND python3 -c "...")` probe lines (not comments).

    CMake comments start with `#`; the probe is an `execute_process` line. We key on
    the literal `python3 -c` so the assertions test the COMMAND, not the explanatory
    comment prose (which legitimately mentions the old `import torch_npu` form).
    """
    return [ln for ln in cmake.splitlines()
            if "python3 -c" in ln and "execute_process" in ln]


def test_torch_npu_probe_uses_find_spec_not_import(tmp_path: Path):
    cmake = _gen_cmake(tmp_path)
    probes = _probe_command_lines(cmake)
    npu_probes = [ln for ln in probes if "torch_npu" in ln]
    assert npu_probes, "expected a torch_npu path probe in the generated CMake"
    for ln in npu_probes:
        # The torch_npu path probe COMMAND must use find_spec, never `import torch_npu`
        # (which loads _C and ABI-crashes on a clean-runtime container).
        assert "find_spec('torch_npu')" in ln, (
            f"torch_npu probe must use importlib.find_spec('torch_npu'): {ln!r}"
        )
        assert "import torch_npu" not in ln, (
            f"torch_npu probe COMMAND must NOT `import torch_npu` (loads _C → ABI "
            f"crash on .post5/B060 clean-runtime): {ln!r}"
        )


def test_torch_probe_also_uses_find_spec(tmp_path: Path):
    cmake = _gen_cmake(tmp_path)
    probes = _probe_command_lines(cmake)
    torch_probes = [ln for ln in probes if "find_spec('torch')" in ln]
    assert torch_probes, "expected a torch path probe using find_spec('torch')"
    # No probe COMMAND should use the bare `import torch;` form.
    for ln in probes:
        assert "import torch;" not in ln, f"probe must not bare-import torch: {ln!r}"


def test_pybind_links_torch_python_before_torch_npu(tmp_path: Path):
    """The generated extension must retain PyTorch's Python-facing symbols."""
    cmake = _gen_cmake(tmp_path)
    link_block = cmake.split(
        "target_link_libraries(pybind11_lib PRIVATE", 1
    )[1].split("\n)", 1)[0].split()

    assert "torch" in link_block
    assert "torch_python" in link_block
    assert "torch_npu" in link_block
    assert link_block.index("torch") < link_block.index("torch_python")
    assert link_block.index("torch_python") < link_block.index("torch_npu")


def test_missing_torch_python_has_a_configure_time_error(tmp_path: Path):
    """A reduced torch install must fail clearly instead of at module import."""
    cmake = _gen_cmake(tmp_path)

    assert 'file(GLOB TORCH_PYTHON_LIBRARIES "${TORCH_PATH}/lib/libtorch_python.so*")' in cmake
    assert "if(NOT TORCH_PYTHON_LIBRARIES)" in cmake
    assert "the generated torch::Tensor binding cannot be linked" in cmake


def test_pybind_headers_come_from_torch_not_an_optional_python_package(tmp_path: Path):
    """Target CANN containers ship PyTorch's headers, not necessarily pybind11's CLI."""
    cmake = _gen_cmake(tmp_path)

    assert "python3 -m pybind11 --includes" not in cmake
    assert "PYBIND11_INC" not in cmake
    assert "sysconfig.get_path('include')" in cmake
    assert "${PYTHON_INCLUDE_DIR}/Python.h" in cmake
    assert "${PYTHON_INCLUDE_DIR}" in cmake
