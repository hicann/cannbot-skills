#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Shared kernel setup logic for archive_tasks templates.

Each task's kernel/setup.py delegates to `kernel_setup(op_name, kernel_dir=__file__)`.
"""

import os
import glob
import subprocess
from pathlib import Path

import setuptools
from setuptools.command.build_ext import build_ext
from torch_npu.utils.cpp_extension import NpuExtension


def kernel_setup(op_name, kernel_dir, ext_name=None, description=None):
    """Run setuptools.setup() with a standard AscendC kernel build configuration.

    Args:
        op_name: Operator name used as the Python package name.
        kernel_dir: Path to the kernel directory (use __file__ from the calling setup.py).
        ext_name: NpuExtension module name. Defaults to "{op_name}_ext".
        description: Optional one-line description for the package.
    """
    here = Path(kernel_dir).resolve().parent
    build_dir = here / "build"
    if ext_name is None:
        ext_name = f"{op_name}_ext"
    if description is None:
        description = f"{op_name} AscendC kernel"

    class _BuildExt(build_ext):
        def run(self):
            so_files = glob.glob(str(build_dir / f"{ext_name}*.so"))
            if not so_files:
                os.makedirs(build_dir, exist_ok=True)
                soc_ver = os.environ.get("SOC_VERSION", "Ascend910B2")
                ascend = os.environ.get("ASCEND_HOME_PATH", "")
                subprocess.check_call(
                    ["cmake", str(here),
                     f"-DSOC_VERSION={soc_ver}",
                     f"-DASCEND_CANN_PACKAGE_PATH={ascend}",
                     "-DCMAKE_BUILD_TYPE=Release"],
                    cwd=build_dir)
                subprocess.check_call(
                    ["make", f"-j{os.cpu_count()}"], cwd=build_dir)

            self.build_lib = str(build_dir)
            super().run()

    setuptools.setup(
        name=op_name,
        version="0.1.0",
        description=description,
        ext_modules=[NpuExtension(ext_name, sources=[])],
        cmdclass={"build_ext": _BuildExt},
        license="BSD 3-Clause",
        python_requires=">=3.8",
    )
