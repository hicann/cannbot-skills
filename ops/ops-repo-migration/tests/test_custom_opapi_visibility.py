# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host-only ELF regression: independent libraries must own independent C++ state."""

import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    sys.platform.startswith("linux")
    and all(shutil.which(tool) for tool in ("cmake", "c++", "nm")),
    "requires Linux ELF, CMake, a C++ compiler and nm; no CANN/NPU required",
)
class CustomOpapiVisibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        cls.env = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL="2", MAKEFLAGS="-j2")
        cls.env.pop("LD_PRELOAD", None)
        cls.env.pop("LD_LIBRARY_PATH", None)
        asset = (
            Path(__file__).resolve().parents[1] / "assets/custom-opapi-visibility.cmake"
        )
        shutil.copyfile(asset, cls.root / "isolate.cmake")
        for name, size in (("custom", 17), ("system", 29)):
            (cls.root / f"{name}.cpp").write_text(
                "#include <vector>\n"
                f"std::vector<int> registry_state({size}, 1);\n"
                'extern "C" __attribute__((visibility("default")))\n'
                f"int {name}_api() {{ return registry_state.size(); }}\n"
            )
        (cls.root / "main.cpp").write_text(
            'extern "C" int custom_api();\n'
            'extern "C" int system_api();\n'
            "int main() { return custom_api() == 17 && system_api() == 29 ? 0 : 7; }\n"
        )
        (cls.root / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.16)\n"
            "project(symbol_isolation LANGUAGES CXX)\n"
            "set(CMAKE_POSITION_INDEPENDENT_CODE ON)\n"
            "set(OPHOST_NAME arbitrary_package_v4)\n"
            "set(ENABLE_CUSTOM ON)\n"
            # Missing object targets (e.g. another build route) are not modified.
            "include(isolate.cmake)\n"
            "add_library(${OPHOST_NAME}_opapi_obj OBJECT custom.cpp)\n"
            "include(isolate.cmake)\n"
            "add_library(custom SHARED $<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>)\n"
            "add_library(system_copy SHARED system.cpp)\n"
            "add_executable(isolated main.cpp)\n"
            "target_link_libraries(isolated PRIVATE custom system_copy)\n"
            # Negative control: the same code without the migration build fix.
            "add_library(unisolated SHARED custom.cpp)\n"
            "add_executable(colliding main.cpp)\n"
            "target_link_libraries(colliding PRIVATE unisolated system_copy)\n"
            # The custom-only adaptation must not alter built-in build mode.
            "set(OPHOST_NAME second_package)\n"
            "set(ENABLE_CUSTOM OFF)\n"
            "add_library(${OPHOST_NAME}_opapi_obj OBJECT custom.cpp)\n"
            "include(isolate.cmake)\n"
            "add_library(builtin_mode SHARED $<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>)\n"
        )
        cls.run_command(["cmake", "-S", str(cls.root), "-B", str(cls.root / "build")])
        cls.run_command(
            ["cmake", "--build", str(cls.root / "build"), "--parallel", "2"]
        )

    @classmethod
    def run_command(cls, command):
        result = subprocess.run(
            command, env=cls.env, capture_output=True, text=True, timeout=60
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return result.stdout

    def symbols(self, name):
        return self.run_command(
            ["nm", "-D", "--defined-only", str(self.root / f"build/lib{name}.so")]
        )

    def test_private_state_and_public_abi_survive_normal_exit(self):
        symbols = self.symbols("custom")
        self.assertIn(" custom_api\n", symbols)
        self.assertNotIn(" registry_state\n", symbols)
        self.assertIn(" registry_state\n", self.symbols("system_copy"))
        self.run_command([str(self.root / "build/isolated")])

    def test_unisolated_negative_control_reproduces_double_free(self):
        def disable_core():
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        result = subprocess.run(
            [str(self.root / "build/colliding")],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=10,
            preexec_fn=disable_core,
        )
        self.assertEqual(result.returncode, -signal.SIGABRT, result.stderr)

    def test_builtin_mode_is_unchanged(self):
        self.assertIn(" registry_state\n", self.symbols("builtin_mode"))


if __name__ == "__main__":
    unittest.main()
