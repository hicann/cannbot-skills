#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Offline regression of actual helpers; no compiler, CANN library or NPU required."""

import ast
import contextlib
import importlib.util
import io
import os
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


def device_elf(payload=b""):
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    header[16:20] = b"\x02\x00\x29\x10"
    return bytes(header) + payload


def tensor(shape=(1024,), dtype=0, fmt=2):
    return types.SimpleNamespace(
        origin_shape=types.SimpleNamespace(dims=list(shape)),
        storage_shape=types.SimpleNamespace(dims=list(shape)),
        data_type=dtype,
        origin_format=fmt,
        storage_format=fmt,
        expand_dims_type=types.SimpleNamespace(full_size=0),
        format=types.SimpleNamespace(origin_format=fmt, storage_format=fmt),
    )


class TemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ge = types.ModuleType("ge")
        custom = types.ModuleType("ge.custom_op")
        runtime = types.ModuleType("ge.runtime")
        custom.register_op = lambda **kwargs: lambda fn: fn
        custom.register_op_impl = lambda **kwargs: lambda impl: impl
        custom.AnnotatedKernelLaunchInfo = lambda **kwargs: types.SimpleNamespace(
            **kwargs
        )
        custom.get_compile_platform_info = lambda: None
        custom.get_declare_launch_args_ctx = lambda: None
        runtime.Tensor = runtime.TensorDesc = object
        with patch.dict(
            sys.modules, {"ge": ge, "ge.custom_op": custom, "ge.runtime": runtime}
        ):
            cls.template = load_module(
                "ge_template_under_test", ROOT / "templates/python_custom_op.py"
            )

    def test_fresh_build_after_source_and_dependency_change(self):
        module = self.template
        commands = []
        pending = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "asc/include").mkdir(parents=True)
            source, dependency = root / "kernel.asc", root / "kernel.h"
            source.write_text("first")
            dependency.write_text("dependency-1")

            def compiler(command, check):
                commands.append(command)
                if command[0] == "bisheng":
                    out = Path(command[command.index("-o") + 1])
                    out.write_bytes(b"host")
                    pending[str(out)] = source.read_bytes() + dependency.read_bytes()
                else:
                    self.assertIn("--only-section=.aicore_binary", command)
                    self.assertEqual(command[1:3], ["-O", "binary"])
                    Path(command[-1]).write_bytes(device_elf(pending[command[-2]]))

            with (
                patch.dict(
                    os.environ,
                    {"ASCEND_HOME_PATH": directory, "MY_OP_BUILD_DIR": directory},
                ),
                patch.object(module.subprocess, "run", side_effect=compiler),
            ):
                first = module._compile_kernel(source, "soc", "2201")
                source.write_text("second")
                second = module._compile_kernel(source, "soc", "2201")
                dependency.write_text("dependency-2")
                third = module._compile_kernel(source, "soc", "2201")
            self.assertNotEqual(first, second)
            self.assertNotEqual(second, third)
            objects = [c[c.index("-o") + 1] for c in commands if c[0] == "bisheng"]
            self.assertEqual(len(set(objects)), 3)
            self.assertTrue(all(not Path(p).exists() for p in objects))

    def test_host_object_and_empty_binary_are_rejected(self):
        with self.assertRaises(ValueError):
            self.template._check_device_elf(b"")
        host = bytearray(device_elf())
        host[16:20] = b"\x01\x00\x3e\x00"
        with self.assertRaises(ValueError):
            self.template._check_device_elf(bytes(host))
        self.template._check_device_elf(device_elf())

    def test_fixed_kernel_contract(self):
        validate = self.template._validate_tensors
        self.assertEqual(validate(tensor(), tensor(), tensor(), 1.0), 1024)
        for shape in [(0,), (1025,), (-1,), (), (1024 * 65536,)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                validate(tensor(shape), tensor(shape), tensor(shape), 1.0)
        for changed in [tensor(dtype=1), tensor(fmt=0), tensor(shape=(2048,))]:
            with self.assertRaises(ValueError):
                validate(tensor(), changed, tensor(), 1.0)
        with self.assertRaises(ValueError):
            validate(tensor(), tensor(), tensor(), 2.0)

    def test_artifact_key_covers_all_tensor_metadata_and_attrs(self):
        key = self.template._tensor_key
        baseline = key(tensor(), tensor(), tensor(), 1.0)
        variants = [
            key(tensor(dtype=1), tensor(), tensor(), 1.0),
            key(tensor(), tensor(fmt=0), tensor(), 1.0),
            key(tensor(), tensor(), tensor(shape=(2048,)), 1.0),
            key(tensor(), tensor(), tensor(), 2.0),
        ]
        self.assertTrue(all(k != baseline for k in variants))

    def test_cross_platform_reuse_is_rejected(self):
        impl = self.template.MyPythonCustomOp()

        def platform(soc):
            return types.SimpleNamespace(
                get_soc_version=lambda: soc, get_platform_resource=lambda *args: "2201"
            )

        with (
            patch.object(
                self.template,
                "get_compile_platform_info",
                return_value=platform("soc1"),
            ),
            patch.object(self.template, "_compile_kernel", return_value=device_elf()),
        ):
            impl.compile(tensor(), tensor(), tensor(), alpha=1.0)
        with patch.object(
            self.template, "get_compile_platform_info", return_value=platform("soc2")
        ):
            with self.assertRaises(ValueError):
                impl.compile(tensor(), tensor(), tensor(), alpha=1.0)

    def test_portable_rejects_unprepared_or_empty_state(self):
        impl = self.template.AddCustomPortable()
        with self.assertRaises(RuntimeError):
            impl.serialize()
        impl.deserialize(b"state-v1")
        self.assertEqual(impl.serialize(), b"state-v1")
        with self.assertRaises(ValueError):
            impl.deserialize(b"")
        self.assertEqual(impl.serialize(), b"state-v1")


class KernelInspectorTests(unittest.TestCase):
    def test_device_and_host_formats(self):
        module = load_module("kernel_inspector", ROOT / "scripts/inspect_kernel.py")
        self.assertEqual(module.inspect_elf(device_elf())["machine"], "0x1029")
        for data in (b"not an ELF", device_elf()[:32]):
            with self.assertRaises(ValueError):
                module.inspect_elf(data)

    def test_symbol_requires_defined_global_function(self):
        module = load_module("kernel_inspector", ROOT / "scripts/inspect_kernel.py")
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "device.o"
            binary.write_bytes(device_elf())
            args = ["inspect_kernel.py", str(binary), "--symbol", "kernel"]
            for ndx, expected in [("1", 0), ("UND", 1)]:
                result = types.SimpleNamespace(
                    stdout=f" 1: 0000 4 FUNC GLOBAL DEFAULT {ndx} kernel\n"
                )
                with (
                    patch.object(sys, "argv", args),
                    patch.object(module.shutil, "which", return_value="readelf"),
                    patch.object(module.subprocess, "run", return_value=result),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(module.main(), expected)


class ExampleVerificationTests(unittest.TestCase):
    def test_output_verification_checks_values_shape_dtype_and_nan(self):
        source = ROOT / "assets/host_cpu_add/src/run.py"
        tree = ast.parse(source.read_text())
        fn = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "verify_output"
        )
        import math

        namespace = {
            "math": math,
            "EXPECTED_VALUE": 3.0,
            "TOLERANCE": 1e-6,
            "DataType": types.SimpleNamespace(DT_FLOAT=0),
        }
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        for values, shape, dtype, good in [
            ([3.0], [1], 0, True),
            ([4.0], [1], 0, False),
            ([float("nan")], [1], 0, False),
            ([3.0], [2], 0, False),
            ([3.0], [1], 1, False),
        ]:
            output = types.SimpleNamespace(data=values, shape=shape, data_type=dtype)
            with contextlib.redirect_stdout(io.StringIO()):
                if good:
                    namespace["verify_output"]([output])
                else:
                    with self.assertRaises(RuntimeError):
                        namespace["verify_output"]([output])


class CmakeLayoutTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cmake"), "cmake is unavailable")
    def test_absolute_implementation_and_prototype_translation_unit(self):
        # 仅验证实际CMake配置/编译单元，不声称使用假目录完成CANN编译。
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "src").mkdir()
            (project / "target-cann/include").mkdir(parents=True)
            for name in (
                "CMakeLists.txt",
                "proto_reg.h",
                "proto_reg.cpp",
                "my_eager_execute_op.cpp",
            ):
                dest = (
                    project / name
                    if name == "CMakeLists.txt"
                    else project / "src" / name
                )
                shutil.copyfile(ROOT / "templates" / name, dest)
            command = [
                "cmake",
                "-S",
                str(project),
                "-B",
                str(project / "build"),
                "-DASCEND_HOME_PATH=" + str(project / "target-cann"),
                "-DCUSTOM_OP_IMPLEMENTATION="
                + str(project / "src/my_eager_execute_op.cpp"),
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            units = json.loads((project / "build/compile_commands.json").read_text())
            self.assertEqual(
                {Path(item["file"]).name for item in units},
                {"proto_reg.cpp", "my_eager_execute_op.cpp"},
            )


if __name__ == "__main__":
    unittest.main()
