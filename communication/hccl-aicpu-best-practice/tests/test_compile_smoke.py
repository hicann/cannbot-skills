#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""仅假编译器，不调用 HCCL 编译。"""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "smoke", Path(__file__).resolve().parents[1] / "scripts/compile_smoke.py"
)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.target = self.repo / "ring.cc"
        self.target.write_text("fixture")
        self.build = self.repo / "build"
        self.build.mkdir()
        self.compiler = self.root / "compiler"
        self.compiler.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\nexit 0\n')
        self.compiler.chmod(0o755)
        self.database = self.build / "compile_commands.json"
        self.cache = self.build / "CMakeCache.txt"
        self.cache.write_text(
            "CMAKE_HOME_DIRECTORY:INTERNAL="
            + str(self.repo)
            + "\nASCEND_INSTALL_PATH:PATH=/fake-cann\n"
        )
        self.entry = {
            "directory": str(self.build),
            "file": str(self.target),
            "arguments": [
                str(self.compiler),
                "-MD",
                "-MF",
                "old.d",
                "-o",
                "old.o",
                "-c",
                str(self.target),
            ],
        }
        self.save()

    def save(self):
        self.database.write_text(json.dumps([self.entry]))

    def run_smoke(self, **kwargs):
        return smoke.run(
            self.repo, self.target, self.database, cann="/fake-cann", **kwargs
        )

    def test_safe_syntax_arguments_and_check_only(self):
        report = self.run_smoke(check_only=True)
        self.assertEqual(report["status"], "READY")
        self.assertEqual(report["configuration"], "host")
        self.assertEqual(
            report["argv"], [str(self.compiler), str(self.target), "-fsyntax-only"]
        )
        self.assertEqual(self.run_smoke()["status"], "PASS")
        self.assertFalse((self.build / "old.o").exists())
        self.assertFalse((self.build / "old.d").exists())
        self.compiler.write_text("#!/bin/sh\nexit 3\n")
        self.assertEqual(self.run_smoke()["status"], "FAIL")

    def test_missing_database_or_unproven_configuration_skip(self):
        self.cache.write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/other-repo\n")
        self.assertEqual(self.run_smoke()["status"], "SKIP")
        self.database.unlink()
        self.assertEqual(self.run_smoke()["status"], "SKIP")

    def test_reference_requires_explicit_same_directory_file(self):
        reference = self.repo / "mesh.cc"
        reference.write_text("fixture")
        self.entry["file"] = str(reference)
        self.entry["arguments"][-1] = str(reference)
        self.save()
        self.assertEqual(self.run_smoke()["status"], "SKIP")
        report = self.run_smoke(reference=reference)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["reference"], str(reference))
        self.assertIn(str(self.target), report["argv"])
        with self.assertRaises(ValueError):
            self.run_smoke(reference=self.root / "absent.cc")

    def test_compiler_extra_output_modes_skip(self):
        original = self.entry["arguments"][:]
        for option in (
            "-Wp,-MD,old.d",
            "-save-temps",
            "-fprofile-arcs",
            "-fdump-tree-all",
            "-MJold.json",
            "-gsplit-dwarf",
            "-Xclang",
            "--coverage",
            "-fopt-info=old.txt",
            "-fdiagnostics-format=sarif-file",
            "--output=old.o",
        ):
            self.entry["arguments"] = original + [option]
            self.save()
            self.assertEqual(self.run_smoke()["status"], "SKIP", option)

    def test_shell_and_response_commands_skip(self):
        self.entry["arguments"] += ["&&", "touch", "bad"]
        self.save()
        self.assertEqual(self.run_smoke()["status"], "SKIP")
        self.entry["arguments"] = [str(self.compiler), "@flags", "-c", str(self.target)]
        self.save()
        self.assertEqual(self.run_smoke()["status"], "SKIP")


if __name__ == "__main__":
    unittest.main()
