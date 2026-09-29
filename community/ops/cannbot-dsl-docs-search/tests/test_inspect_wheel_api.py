#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Exercise public export resolution through the wheel inspection CLI."""

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/inspect_wheel_api.py"


class PublicExportTests(unittest.TestCase):
    def inspect(self, package, exports):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheel = root / "sample-1.0-py3-none-any.whl"
            prefix = package.replace(".", "/")
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(f"{prefix}/__init__.py", exports)
                archive.writestr(
                    f"{prefix}/bridge.py",
                    "from .core import original as intermediate\n",
                )
                archive.writestr(
                    f"{prefix}/core.py",
                    """
def original(value):
    return value

def stable(value):
    return value

def submodule_only():
    pass

class Original:
    def public(self, value):
        return value

    def _private(self):
        pass
""",
                )
                archive.writestr(
                    "sample-1.0.dist-info/METADATA", "Name: sample\nVersion: 1.0\n"
                )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--wheel",
                    str(wheel),
                    "--package",
                    package,
                    "--output-dir",
                    str(root / "output"),
                ],
                cwd=root,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["static_export_unresolved_count"], 0)
            index = yaml.safe_load((root / summary["yaml_index"]).read_text())
            docs = "\n".join(
                (root / path).read_text() for path in summary["api_documents"]
            )
            return index["apis"], docs

    def test_renamed_function_and_class_members(self):
        for package in ("cannbotdsl", "vendor.dsl"):
            with self.subTest(package=package):
                apis, docs = self.inspect(
                    package,
                    """
from .core import original as renamed, stable, Original as Renamed
__all__ = ["renamed", "stable", "Renamed"]
""",
                )
                expected = {
                    f"{package}.{name}"
                    for name in ("renamed", "stable", "Renamed", "Renamed.public")
                }
                self.assertEqual({api["symbol"] for api in apis}, expected)
                for symbol in expected:
                    self.assertIn(symbol, docs)

    def test_chained_and_multiple_aliases(self):
        apis, _ = self.inspect(
            "cannbotdsl",
            """
from .bridge import intermediate as renamed
from .core import original as second, stable
__all__ = ["renamed", "second", "stable"]
""",
        )
        aliased = next(
            api for api in apis if api["definition"] == "cannbotdsl.core.original"
        )
        self.assertEqual(
            set(aliased["public_imports"]), {"cannbotdsl.renamed", "cannbotdsl.second"}
        )
        self.assertEqual(len(apis), 2)

    def test_lazy_module_attribute_export(self):
        apis, _ = self.inspect(
            "cannbotdsl",
            """
__all__ = ["stable"]
def __getattr__(name):
    if name == "stable":
        from . import core
        return getattr(core, name)
""",
        )
        self.assertEqual([api["symbol"] for api in apis], ["cannbotdsl.stable"])

    def test_nested_guarded_star_export(self):
        apis, _ = self.inspect(
            "cannbotdsl",
            """
__all__ = ["stable"]
try:
    if True:
        from .core import *
except ImportError:
    pass
""",
        )
        self.assertEqual([api["symbol"] for api in apis], ["cannbotdsl.stable"])


if __name__ == "__main__":
    unittest.main()
