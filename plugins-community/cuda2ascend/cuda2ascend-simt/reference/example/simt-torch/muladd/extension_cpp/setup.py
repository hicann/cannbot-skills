# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import os
import glob
import sysconfig
from shutil import which
from distutils.errors import CompileError

from setuptools import find_packages, setup
from setuptools import Extension
from setuptools.command.build_ext import build_ext

import torch
import torch_npu
import torch.utils.cpp_extension as cpp_extension

library_name = "extension_cpp"

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
EXTENSIONS_DIR = os.path.join(BASE_DIR, library_name, "csrc")
NPU_ARCH = os.getenv("NPU_ARCH", "dav-3510")


def get_dependency_paths():
    python_include = sysconfig.get_config_var("INCLUDEPY")
    python_lib = sysconfig.get_config_var("LIBDIR")
    torch_include_paths = cpp_extension.include_paths()
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

    torch_npu_path = os.path.dirname(torch_npu.__file__)
    torch_npu_include = os.path.join(torch_npu_path, "include")
    torch_npu_acl_include = os.path.join(
        torch_npu_path, "include", "third_party", "acl", "inc"
    )
    torch_npu_lib = os.path.join(torch_npu_path, "lib")

    include_dirs = [
        *torch_include_paths,
        python_include,
        torch_npu_include,
        torch_npu_acl_include,
    ]
    library_dirs = [python_lib, torch_lib, torch_npu_lib]

    return {"include_dirs": include_dirs, "library_dirs": library_dirs}


class AscendBuildExtension(build_ext):
    def _check_bisheng_compiler(self):
        if not which("bisheng"):
            raise RuntimeError("bisheng command not found")

    def build_extension(self, ext):
        self._check_bisheng_compiler()
        dep_paths = get_dependency_paths()
        ext_fullpath = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_fullpath), exist_ok=True)

        use_cxx11_abi = torch._C._GLIBCXX_USE_CXX11_ABI
        abi_value = "1" if use_cxx11_abi else "0"
        debug_mode = os.getenv("DEBUG", "0") == "1"
        opt_flag = "-O0" if debug_mode else "-O3"

        compile_cmd = [
            "bisheng",
            "-x",
            "asc",
            "--enable-simt",
            f"--npu-arch={NPU_ARCH}",
            "-shared",
            "-fPIC",
            "-std=c++17",
            opt_flag,
            f"-D_GLIBCXX_USE_CXX11_ABI={abi_value}",
            *ext.sources,
        ]

        if debug_mode:
            compile_cmd.append("-g")

        for include_dir in dep_paths["include_dirs"]:
            compile_cmd.append(f"-I{include_dir}")

        for library_dir in dep_paths["library_dirs"]:
            compile_cmd.append(f"-L{library_dir}")

        compile_cmd.extend(
            [
                "-ltorch_npu",
                "-ltorch_python",
                "-ltorch_cpu",
                "-ltorch",
                "-lc10",
                "-o",
                ext_fullpath,
            ]
        )

        try:
            self.spawn(compile_cmd)
        except Exception as exc:
            raise CompileError(str(exc)) from exc


def get_extensions():
    sources = list(glob.glob(os.path.join(EXTENSIONS_DIR, "*.asc")))
    sources += list(glob.glob(os.path.join(EXTENSIONS_DIR, "simt", "*.asc")))
    return [
        Extension(
            name=f"{library_name}._C",
            sources=sources,
            language="asc",
        )
    ]


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch", "torch_npu"],
    description="Example of PyTorch C++ and Ascend SIMT extensions",
    url="https://github.com/pytorch/extension-cpp",
    cmdclass={"build_ext": AscendBuildExtension},
)
