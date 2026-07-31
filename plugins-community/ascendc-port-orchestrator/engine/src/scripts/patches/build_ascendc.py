#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = SCRIPT_DIR.parent

# build_capabilities/ lives alongside this script (patches/). Make it importable
# whether build_ascendc runs as a script or is loaded via importlib (tests).
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_capabilities import (  # noqa: E402
    BuildContext,
    CMakeContribution,
    baseline_contributions,
    load_capabilities,
    merge,
    registry,
    run_prebuild_steps,
)
from build_capabilities.baseline_per_source_defines import (  # noqa: E402
    needs_target_directory,
)


def _resolve_task_dir(op: str) -> Path:
    op_path = Path(op)
    if op_path.is_dir():
        return op_path.resolve()

    direct = WORKDIR / op
    if direct.is_dir():
        return direct

    raise FileNotFoundError(f"Cannot find task directory for op '{op}'")


def _detect_ascend_path() -> Path:
    for env_name in ("ASCEND_INSTALL_PATH", "ASCEND_HOME_PATH"):
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser().resolve()

    candidates = [
        Path.home() / "Ascend" / "ascend-toolkit" / "latest",
        Path("/usr/local/Ascend/ascend-toolkit/latest"),
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()

    return candidates[-1]


def _find_kernel_sources(kernel_dir: Path) -> list[Path]:
    sources = sorted(
        path for path in kernel_dir.glob("*.cpp")
        if path.name != "pybind11.cpp"
    )
    if not sources:
        raise FileNotFoundError(f"No kernel .cpp sources found in {kernel_dir}")
    return sources


def _extract_pybind_module_name(pybind_path: Path) -> str:
    content = pybind_path.read_text(encoding="utf-8")
    match = re.search(r"PYBIND11_MODULE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,", content)
    if not match:
        raise ValueError(f"Unable to detect PYBIND11_MODULE name from {pybind_path}")
    return match.group(1)


def _format_cmake_list(items: list[str], indent: int = 4) -> str:
    prefix = " " * indent
    return "\n".join(f"{prefix}{item}" for item in items)


def _generate_cmakelists(
    kernel_dir: Path,
    build_dir: Path,
    module_name: str,
    sources: list[Path],
    ascend_path: Path,
    merged: "CMakeContribution | None" = None,
) -> str:
    # The single structural extension point (BUILD_CAPABILITY_EXTENSION_DESIGN.md §2B):
    # a merged CMakeContribution folded into the template at defined injection points.
    # `merged=None` (legacy callers) means "just the baseline contributions" — the
    # behavior-neutral backward-compat path (DEBT-20/20.1 + DEBT-110), proven
    # byte-identical by tests/test_golden_cmakelists.py.
    if merged is None:
        ctx = BuildContext(
            kernel_dir=kernel_dir,
            sources=tuple(sources),
            ascend_path=ascend_path,
            script_dir=SCRIPT_DIR,
        )
        merged = merge(baseline_contributions(ctx))

    include_dirs = [kernel_dir]
    catlass_include = kernel_dir / "catlass" / "include"
    if catlass_include.is_dir():
        include_dirs.append(catlass_include)

    task_catlass_include = kernel_dir.parent / "catlass" / "include"
    if task_catlass_include.is_dir() and task_catlass_include not in include_dirs:
        include_dirs.append(task_catlass_include)

    # DEBT-110 cann_stubs include is now contributed by the baseline capability
    # (build_capabilities.baseline_cann_host_tiling) and arrives via
    # merged.include_dirs — appended AFTER the base includes, preserving order.
    include_dirs.extend(merged.include_dirs)

    all_sources = list(sources) + list(merged.extra_sources)
    source_lines = [str(path) for path in all_sources]
    include_lines = [str(path) for path in include_dirs]

    # DEBT-20 + DEBT-20.1 (2026-05-21): per-source-file compile defines for
    # KFC/Matmul isolation. Loaded from kernel/build_overrides.json. Emitted
    # AFTER ascendc_library() so the aic_obj / aiv_obj sub-targets exist.
    # Non-breaking: empty dict if file absent or malformed.
    #   - "global" defines emitted on the source itself (both passes see it)
    #   - "aic" defines scoped to TARGET_DIRECTORY aic_obj (AIC compile pass only)
    #   - "aiv" defines scoped to TARGET_DIRECTORY aiv_obj (AIV compile pass only)
    #
    # DEBT-20.1 follow-up: in CANN 9.0.0
    # ascendc.cmake DYNAMIC_MODE, aic_obj/aiv_obj live in a sub-cmake-project
    # scope; TARGET_DIRECTORY refs from parent fail with "non-existent target".
    # Wrapped in `if(TARGET ...)` guards so the emit becomes a defensive no-op
    # in that environment. RECOMMENDED ALTERNATIVE: kernel source self-defines
    # SPLIT_CORE_CUBE/VEC based on bisheng-provided __DAV_C220_CUBE__ /
    # __DAV_C220_VEC__ macros (see docs/design/FA_CLASS_DESIGN_NOTES.md#cann-fa-build-launch-recipe-2026-05-21).
    # DEBT-20/20.1 per-source defines now arrive via the baseline capability
    # (build_capabilities.baseline_per_source_defines) as merged.defines. The
    # emission logic below is unchanged (same shape, same output).
    per_source_defines = merged.defines
    override_lines: list[str] = []
    for src_path in all_sources:
        passes = per_source_defines.get(src_path.name)
        if not passes:
            continue
        if passes.get("global"):
            override_lines.append(
                f'set_source_files_properties("{src_path}" PROPERTIES '
                f'COMPILE_DEFINITIONS "{";".join(passes["global"])}")'
            )
        if passes.get("aic"):
            override_lines.append(
                f'if(TARGET aic_obj)\n'
                f'    set_source_files_properties("{src_path}" '
                f'TARGET_DIRECTORY aic_obj PROPERTIES '
                f'COMPILE_DEFINITIONS "{";".join(passes["aic"])}")\n'
                f'endif()'
            )
        if passes.get("aiv"):
            override_lines.append(
                f'if(TARGET aiv_obj)\n'
                f'    set_source_files_properties("{src_path}" '
                f'TARGET_DIRECTORY aiv_obj PROPERTIES '
                f'COMPILE_DEFINITIONS "{";".join(passes["aiv"])}")\n'
                f'endif()'
            )
    if override_lines:
        per_source_block = (
            "\n# DEBT-20/20.1 per-source-file COMPILE_DEFINITIONS (build_overrides.json)\n"
            + "\n".join(override_lines)
            + "\n"
        )
    else:
        per_source_block = ""

    # DEBT-20.1 requires cmake ≥ 3.18 for set_source_files_properties TARGET_DIRECTORY.
    # Bump min version when per-pass defines are in use; stay 3.16 otherwise.
    cmake_min_version = "3.18.0" if needs_target_directory(per_source_defines) else "3.16.0"

    # Extra pybind11_lib links contributed by capabilities (empty for baseline →
    # byte-identical). Raw cmake_fragment injected right after the base link block
    # (DEBT-110's find_library block for the baseline; escape hatch for capabilities).
    link_libs_block = "".join(f"\n  {lib}" for lib in merged.link_libs)
    cmake_fragment = merged.cmake_fragment

    return f"""cmake_minimum_required(VERSION {cmake_min_version})
project(Ascend_C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOC_VERSION "${{SOC_VERSION}}" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "${{ASCEND_CANN_PACKAGE_PATH}}" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "{build_dir}")

if(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/ascendc_devkit/tikcpp/samples/cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/x86_64-linux/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/x86_64-linux/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/aarch64-linux/tikcpp/ascendc_kernel_cmake)
    # A3-ARM (2026-06-30, rms_norm_backward_add_a3 on Ascend910/aarch64): ARM CANN puts
    # ascendc_kernel_cmake under the arch subdir, not tools/compiler/devkit. Without this
    # branch the generated CMakeLists FATAL_ERROR'd on A3. (x86_64-linux above for symmetry.)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/aarch64-linux/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${{ASCENDC_CMAKE_DIR}}/ascendc.cmake)

ascendc_library(kernels STATIC
{_format_cmake_list(source_lines)}
)
{per_source_block}
ascendc_include_directories(kernels PRIVATE
{_format_cmake_list(include_lines)}
    ${{ASCEND_CANN_PACKAGE_PATH}}/include
    ${{ASCEND_CANN_PACKAGE_PATH}}/include/experiment/runtime
    ${{ASCEND_CANN_PACKAGE_PATH}}/include/experiment/msprof
)

add_library(pybind11_lib SHARED "{kernel_dir / 'pybind11.cpp'}")
target_link_libraries(pybind11_lib PRIVATE
  kernels
  torch
  torch_python
  torch_npu
  m
  dl{link_libs_block}
)
{cmake_fragment}# Resolve torch / torch_npu paths via importlib.find_spec, NOT `import torch_npu`
# (2026-06-20, FA-grad .141 V351 clean-runtime lane): `import torch_npu` loads the
# torch_npu `_C` extension, which ABI-trips at cmake-configure on a clean-runtime
# container whose torch_npu wheel is ABI-paired to a device CANN that differs from
# the build toolchain (the .post5/B060 case: `free(): invalid pointer` / libruntime
# ErrorManager symbol mismatch). find_spec(...).origin returns the package dir
# WITHOUT importing `_C`, so the include/lib paths resolve even when the runtime
# import would crash. Works identically on a fully-paired container.
execute_process(COMMAND python3 -c "import os,importlib.util as u; print(os.path.dirname(u.find_spec('torch').origin))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_PATH
)
message("TORCH_PATH is ${{TORCH_PATH}}")
file(GLOB TORCH_PYTHON_LIBRARIES "${{TORCH_PATH}}/lib/libtorch_python.so*")
if(NOT TORCH_PYTHON_LIBRARIES)
  message(FATAL_ERROR
    "libtorch_python.so was not found under ${{TORCH_PATH}}/lib; "
    "the generated torch::Tensor binding cannot be linked in this environment.")
endif()
set(ENV{{ASCEND_HOME_PATH}} ${{ASCEND_CANN_PACKAGE_PATH}})
execute_process(COMMAND python3 -c "import os,importlib.util as u; print(os.path.dirname(u.find_spec('torch_npu').origin))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_NPU_PATH
)
message("TORCH_NPU_PATH is ${{TORCH_NPU_PATH}}")
target_link_directories(pybind11_lib PRIVATE
  ${{TORCH_PATH}}/lib
  ${{TORCH_NPU_PATH}}/lib
)
target_include_directories(pybind11_lib PRIVATE
  "{kernel_dir}"
  ${{TORCH_NPU_PATH}}/include
  ${{TORCH_PATH}}/include
  ${{TORCH_PATH}}/include/torch/csrc/api/include
)
execute_process(COMMAND python3 -m pybind11 --includes
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYBIND11_INC
)
string(REPLACE " " ";" PYBIND11_INC ${{PYBIND11_INC}})
target_compile_options(pybind11_lib PRIVATE
  ${{PYBIND11_INC}}
  -D_GLIBCXX_USE_CXX11_ABI=1
)

execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
)
set_target_properties(pybind11_lib PROPERTIES
  OUTPUT_NAME {module_name}
  PREFIX ""
  SUFFIX "${{PYTHON_EXTENSION_SUFFIX}}"
)
"""


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"[build_ascendc] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def build(task: str, soc_version: str, build_type: str, clean: bool) -> Path:
    task_dir = _resolve_task_dir(task)

    kernel_dir = task_dir / "kernel"
    pybind_path = kernel_dir / "pybind11.cpp"
    if not kernel_dir.is_dir():
        raise FileNotFoundError(f"Kernel directory not found: {kernel_dir}")
    if not pybind_path.is_file():
        raise FileNotFoundError(f"Missing pybind11 entry: {pybind_path}")

    sources = _find_kernel_sources(kernel_dir)
    module_name = _extract_pybind_module_name(pybind_path)
    build_dir = kernel_dir / "build"
    cmake_dir = build_dir / "_autogen_cmake"
    ascend_path = _detect_ascend_path()

    # Build-capability extension (BUILD_CAPABILITY_EXTENSION_DESIGN.md §2B):
    # fold the baseline contributions (DEBT-20/20.1 + DEBT-110) into one merged
    # CMakeContribution, run any prebuild codegen steps BEFORE cmake configure,
    # then generate. (STEP 2 layers declared named capabilities from
    # build_overrides.json into this merge.)
    ctx = BuildContext(
        kernel_dir=kernel_dir,
        sources=tuple(sources),
        ascend_path=ascend_path,
        soc=soc_version,
        script_dir=SCRIPT_DIR,
    )
    # STEP 2 named-capability resolution: baseline contributions ALWAYS apply
    # first; declared named capabilities (from build_overrides.json's
    # "capabilities":[...]) resolve through the registry and are merged after.
    # GUARDRAIL: an op with no "capabilities" → declared==[] → contribs==[] →
    # merge is baseline-only → byte-identical to STEP-1 (golden gate proves it).
    # An unknown declared name FAILS LOUDLY via registry.get() — never a silent
    # no-op (canonical-boundary invariant, build_capabilities/__init__.py §2A).
    declared = load_capabilities(kernel_dir)
    contribs = [registry.get(name)(ctx) for name in declared]
    merged = merge(baseline_contributions(ctx) + contribs)
    run_prebuild_steps(merged.prebuild_steps, ctx)

    if clean and build_dir.exists():
        shutil.rmtree(build_dir)

    cmake_dir.mkdir(parents=True, exist_ok=True)
    cmakelists_path = cmake_dir / "CMakeLists.txt"
    cmakelists_path.write_text(
        _generate_cmakelists(
            kernel_dir=kernel_dir,
            build_dir=build_dir,
            module_name=module_name,
            sources=sources,
            ascend_path=ascend_path,
            merged=merged,
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["ASCEND_HOME_PATH"] = str(ascend_path)

    # NODE-19 (2026-05-28): CANN's extract_host_stub.py calls bare llvm-objdump
    # which lives under ASCEND_HOME_PATH/<arch>/ccec_compiler/bin. Propagate this
    # to PATH so cmake subprocesses find it without manual pre-export.
    # Arch-adaptive: check both x86_64-linux and aarch64-linux.
    for _arch in ("x86_64-linux", "aarch64-linux"):
        _ccec_bin = ascend_path / _arch / "ccec_compiler" / "bin"
        if _ccec_bin.is_dir():
            env["PATH"] = str(_ccec_bin) + os.pathsep + env.get("PATH", "")
            break

    cmake_configure = [
        "cmake",
        "-S",
        str(cmake_dir),
        "-B",
        str(build_dir),
        f"-DSOC_VERSION={soc_version}",
        f"-DASCEND_CANN_PACKAGE_PATH={ascend_path}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]
    cmake_build = ["cmake", "--build", str(build_dir), "-j"]

    _run(cmake_configure, cwd=task_dir, env=env)
    _run(cmake_build, cwd=task_dir, env=env)
    return build_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AscendC kernels for a task without task-local run.sh")
    parser.add_argument("task", help="Task directory name or path")
    parser.add_argument("-v", "--soc-version", default="Ascend910B2", help="Ascend SoC version")
    parser.add_argument("--build-type", default="Debug", help="CMake build type")
    parser.add_argument("--clean", action="store_true", help="Remove kernel/build before configuring")
    args = parser.parse_args()

    build_dir = build(
        task=args.task,
        soc_version=args.soc_version,
        build_type=args.build_type,
        clean=args.clean,
    )
    print(f"[build_ascendc] Build completed: {build_dir}")


if __name__ == "__main__":
    main()
