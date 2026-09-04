# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Build an authored TileLang2AscendC ``kernel/`` project on the A5 target.

The generator owns the CMake recipe in the candidate project.  This wrapper
only selects the target environment and build directory; it never rewrites
the authored source tree and never invokes the legacy generated-CMake builder.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path, help="candidate workspace")
    parser.add_argument("-v", "--soc-version", required=True, help="target SoC version")
    parser.add_argument("--build-type", default="Release", choices=("Debug", "Release"))
    parser.add_argument("--clean", action="store_true", help="compatibility alias; clean is the default")
    parser.add_argument("--reuse-build", action="store_true", help="reuse kernel/build (unsafe for a fresh candidate)")
    return parser


def _real_dir(path: Path) -> bool:
    """Report whether *path* is a real directory and never a symlink."""
    return not path.is_symlink() and path.is_dir()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.clean and args.reuse_build:
        raise SystemExit("--clean and --reuse-build are mutually exclusive")
    parallelism = os.environ.get("CANNBOT_TILELANG_BUILD_PARALLELISM", "4")
    if not parallelism.isascii() or not parallelism.isdecimal() or int(parallelism) <= 0:
        raise SystemExit("CANNBOT_TILELANG_BUILD_PARALLELISM must be a positive integer")
    workspace = args.task.expanduser().resolve()
    kernel = workspace / "kernel"
    cmake_file = kernel / "CMakeLists.txt"
    if not _real_dir(workspace) or not _real_dir(kernel):
        raise SystemExit(f"invalid TileLang2AscendC candidate workspace: {workspace}")
    if cmake_file.is_symlink() or not cmake_file.is_file():
        raise SystemExit(f"TileLang2AscendC CMakeLists.txt is missing: {cmake_file}")
    build_dir = kernel / "build"
    if build_dir.is_symlink() or (build_dir.exists() and not build_dir.is_dir()):
        raise SystemExit(f"refusing unsafe TileLang2AscendC build path: {build_dir}")
    if not args.reuse_build and build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SOC_VERSION"] = args.soc_version
    cann_path = env.get("ASCEND_HOME_PATH") or env.get("ASCEND_INSTALL_PATH")
    if cann_path:
        env["ASCEND_HOME_PATH"] = cann_path
        env["ASCEND_CANN_PACKAGE_PATH"] = cann_path
    configure = [
        "cmake",
        "-S", str(kernel),
        "-B", str(build_dir),
        f"-DSOC_VERSION={args.soc_version}",
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
    ]
    if cann_path:
        configure.append(f"-DASCEND_CANN_PACKAGE_PATH={cann_path}")
    subprocess.run(configure, cwd=workspace, env=env, check=True)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", parallelism],
        cwd=workspace,
        env=env,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
