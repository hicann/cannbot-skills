#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check portable source references and template interfaces without loading an NPU runtime."""

import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
ST = ROOT / "ops/tilelang-op-test-design/Ascend950/scripts"
REFERENCES = ROOT / "ops/tilelang-performance-best-practices/Ascend950/references"


class SourcePathTests(unittest.TestCase):
    def discover(self, root, *args):
        return subprocess.run(
            [
                sys.executable,
                str(ST / "discover_tests.py"),
                "--repo-root",
                str(root),
                *args,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_discovery_supports_both_source_layouts_and_generated_operators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in (
                "tests/moe",
                "testing/ascend/language",
                "examples/ascend",
                "operators/add",
            ):
                path = root / directory / "test_sample.py"
                path.parent.mkdir(parents=True)
                # Static discovery must not import the module.
                path.write_text(
                    'raise RuntimeError("must not import")\ndef test_sample():\n    assert 1 == 1\n'
                )
            embedded = root / "examples/ascend/example_sample.py"
            embedded.write_text("def test_embedded():\n    assert 2 == 2\n")
            result = self.discover(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["summary"]["matching_test_functions"], 5)
            self.assertEqual(report["summary"]["package_test_functions"], 4)
            self.assertEqual(report["summary"]["embedded_test_functions"], 1)
            self.assertTrue(all((root / path).is_dir() for path in report["roots"]))
            explicit = self.discover(root, "--root", "operators/add/test_sample.py")
            self.assertEqual(
                json.loads(explicit.stdout)["summary"]["matching_test_functions"], 1
            )

    def test_discovery_does_not_succeed_without_a_test_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.discover(Path(tmp))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--root", result.stderr)

    def test_evidence_records_actual_ops_test_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test_sample.py").write_text(
                "def test_sample():\n    assert 1 == 1\n"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ST / "pytest_st_evidence.py"),
                    "--repo-root",
                    tmp,
                    "--output",
                    "evidence.json",
                    "--collect-only",
                    "--",
                    "test_sample.py",
                ],
                env={
                    **os.environ,
                    "OPS_TILELANG_TEST_LEVEL": "2",
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / "evidence.json").read_text())
            self.assertEqual(
                report["environment"]["selected_variables"]["OPS_TILELANG_TEST_LEVEL"],
                "2",
            )
            self.assertEqual(report["automated_conclusion"], "COLLECTED_NOT_EXECUTED")

    def test_no_legacy_checkout_references(self):
        roots = list((ROOT / "ops").glob("tilelang*/Ascend950"))
        roots.append(ROOT / "plugins-official/tilelang-op-orchestrator")
        forbidden = re.compile(
            r"tile[_-]?kernels|tilelang[-_]deepseek|\bTK_TEST_LEVEL\b|/home/|/Users/",
            re.I,
        )
        failures = []
        for root in roots:
            for path in root.rglob("*"):
                if not path.is_file() or any(
                    p in path.parts
                    for p in ("repositories", ".preparation", "__pycache__")
                ):
                    continue
                if path.suffix not in {".md", ".py", ".json", ".yaml", ".toml", ".sh"}:
                    continue
                for n, line in enumerate(path.read_text().splitlines(), 1):
                    if forbidden.search(line):
                        failures.append(f"{path.relative_to(ROOT)}:{n}")
        self.assertEqual(failures, [])


class TemplateInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(REFERENCES))

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(REFERENCES))

    def test_transpose_adapters_forward_full_shapes_to_supplied_factory(self):
        names = (
            "transpose_base",
            "transpose_big_dim",
            "transpose_cut_one_axis",
            "transpose_cut_two_axis",
            "transpose_small_shape",
            "transpose_tensor_move",
            "transpose_with_gather",
            "transpose_n_last",
        )
        shapes = {
            "transpose_big_dim": (4096, 192),
            "transpose_cut_one_axis": (64, 256),
            "transpose_small_shape": (64, 128),
        }
        dtype = types.SimpleNamespace(bytes=2)
        for name in names:
            module = importlib.import_module(f"conversion.templates.dav3510.{name}")
            shape = shapes.get(name, (192, 256))
            args = (2, *shape, dtype) if name == "transpose_n_last" else (*shape, dtype)
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "kernel_factory"):
                    module.build(*args)
                seen = []
                kernel = object()

                def factory(*shape):
                    seen.append(shape)
                    return kernel

                self.assertIs(module.build(*args, kernel_factory=factory), kernel)
                self.assertEqual(seen, [(*shape, dtype)])
        base = importlib.import_module("conversion.templates.dav3510.transpose_base")
        with self.assertRaises(ValueError):
            base.build(65, 256, "float16", kernel_factory=lambda *args: None)

    def test_template_core_counts_are_explicit_before_dsl_construction(self):
        # Only exercise Python argument validation. This is not a lowering or kernel test.
        fake = types.ModuleType("tilelang")
        fake.language = types.ModuleType("tilelang.language")
        fake.jit = lambda **kwargs: lambda function: function
        fake.next_power_of_2 = lambda value: 1 << (value - 1).bit_length()
        cases = (
            ("broadcast.code.broadcast_add_kernel", "build", (3, 129)),
            ("reduce.templates.dav310.kernel_utils", "euclidean_norm", (2, 257)),
            ("reduce.templates.dav310.kernel_utils", "softmax_full_load", (2, 257)),
            ("scan.templates.dav310.scan_base", "build", (2, 257)),
            ("rope.code.rope_vf_common", "build", (2, 2, 128)),
        )
        with patch.dict(
            sys.modules, {"tilelang": fake, "tilelang.language": fake.language}
        ):
            try:
                for name, function, args in cases:
                    module = importlib.import_module(name)
                    for cores in (None, 0, -1, True, 1.5):
                        with self.subTest(name=name, cores=cores):
                            with self.assertRaisesRegex(ValueError, "num_cores"):
                                getattr(module, function)(*args, num_cores=cores)
            finally:
                for name, _, _ in cases:
                    sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
