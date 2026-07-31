# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension


PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
USE_NINJA = os.getenv('USE_NINJA') == '1'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_opp_paths():
    """Parse include and library directories from ASCEND_CUSTOM_OPP_PATH."""
    env_val = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    paths = [p.strip() for p in env_val.split(":") if p.strip()]

    include_dirs = []
    library_dirs = []

    for p in paths:
        inc = os.path.join(p, "op_api", "include")
        if os.path.exists(inc):
            include_dirs.append(inc)
        lib = os.path.join(p, "op_api", "lib")
        if os.path.exists(lib):
            library_dirs.append(lib)

    return include_dirs, library_dirs


opp_inc, opp_lib = parse_opp_paths()

source_files = glob.glob(os.path.join(BASE_DIR, "*.cpp"), recursive=False)

ext_modules = [
    NpuExtension(
        name="py_ext._C",
        sources=source_files,
        include_dirs=opp_inc,
        library_dirs=opp_lib,
        extra_compile_args=[
            "-O2",
            "-fPIC",
            "-I" + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include", "third_party", "acl", "inc"),
        ],
    )
]

setup(
    name="py_ext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)
