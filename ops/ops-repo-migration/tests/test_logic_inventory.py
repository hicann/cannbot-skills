# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Source inventory regressions: semantic edits and failed tools must not match."""

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_logic_unchanged.sh"


@unittest.skipUnless(shutil.which("gcc"), "source comparison requires gcc")
class LogicInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = self.root / "source with spaces"
        self.dst = self.root / "target with spaces"
        self.src.mkdir()
        self.dst.mkdir()

    def pair(self, before, after, name="kernel.cpp"):
        (self.src / name).write_text(before)
        (self.dst / name).write_text(after)

    def run_inventory(self, env=None):
        return subprocess.run(
            [str(SCRIPT), str(self.src), str(self.dst)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            shell=False,
        )

    def test_comments_only_match(self):
        self.pair("int x = 1; // old\n", "/* license */\nint x = 1; // new\n")
        result = self.run_inventory()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("IDENTICAL=1 DIFFERS=0", result.stdout)

    def test_shell_metacharacters_in_paths_are_literal(self):
        source = self.root / "source ; $(false) & [data]"
        target = self.root / "target ; $(false) & [data]"
        self.src.rename(source)
        self.dst.rename(target)
        self.src, self.dst = source, target
        self.pair("int x = 1;\n", "int x = 1;\n")
        result = self.run_inventory()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("IDENTICAL=1 DIFFERS=0", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_literal_and_token_whitespace_changes_are_differences(self):
        for before, after in [
            ('const char* p = "a b";\n', 'const char* p = "ab";\n'),
            ("int z = x + ++y;\n", "int z = x++ + y;\n"),
            ('auto p = R"tag(a b)tag";\n', 'auto p = R"tag(ab)tag";\n'),
        ]:
            with self.subTest(before=before):
                self.pair(before, after)
                result = self.run_inventory()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("IDENTICAL=0 DIFFERS=1", result.stdout)

    def test_implementation_extensions_are_included(self):
        for suffix in [".cc", ".cxx", ".hpp", ".hxx", ".asc"]:
            self.pair("int x = 1;\n", "int x = 2;\n", "kernel" + suffix)
        result = self.run_inventory()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("IDENTICAL=0 DIFFERS=5", result.stdout)

    def test_failed_preprocessor_is_not_an_identical_empty_hash(self):
        self.pair("int x = 1;\n", "int x = 2;\n")
        stub = self.root / "gcc"
        stub.write_text("#!/bin/sh\nexit 42\n")
        stub.chmod(0o755)
        env = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ["PATH"])
        result = self.run_inventory(env)
        self.assertEqual(result.returncode, 1)
        self.assertIn("IDENTICAL=0", result.stdout)
        self.assertIn("ERRORS=1", result.stdout)

    def test_malformed_source_is_not_identical(self):
        self.pair("/* unterminated\n", "/* also unterminated\n")
        result = self.run_inventory()
        self.assertEqual(result.returncode, 1)
        self.assertIn("IDENTICAL=0", result.stdout)

    def test_temporary_files_are_cleaned_after_success_and_failure(self):
        temporary_parent = self.root / "temporary parent"
        temporary_parent.mkdir()
        protected = temporary_parent / "keep.txt"
        protected.write_text("unrelated data")
        env = dict(os.environ, TMPDIR=str(temporary_parent))
        for source, expected in [("int x = 1;\n", 0), ("/* unterminated\n", 1)]:
            with self.subTest(expected=expected):
                self.pair(source, source)
                result = self.run_inventory(env)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual(list(temporary_parent.iterdir()), [protected])
                self.assertEqual(protected.read_text(), "unrelated data")
                self.assertEqual((self.src / "kernel.cpp").read_text(), source)

    def test_cleanup_does_not_remove_unexpected_nested_content(self):
        self.pair("int x = 1;\n", "int x = 1;\n")
        temporary = self.root / "managed temporary"
        nested = temporary / "unexpected"
        nested.mkdir(parents=True)
        protected = nested / "keep.txt"
        protected.write_text("unrelated data")
        stub = self.root / "mktemp"
        stub.write_text(
            "#!/bin/sh\nprintf '%s\\n' " + shlex.quote(str(temporary)) + "\n"
        )
        stub.chmod(0o755)
        env = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ["PATH"])
        result = self.run_inventory(env)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(protected.read_text(), "unrelated data")
        self.assertEqual(list(temporary.iterdir()), [nested])

    def test_moved_files_are_reported_separately(self):
        (self.src / "before.cpp").write_text("int x = 1;\n")
        (self.dst / "after.cpp").write_text("int x = 1;\n")
        result = self.run_inventory()
        self.assertEqual(result.returncode, 0)
        self.assertIn(
            "IDENTICAL=0 DIFFERS=0 ONLY-IN-SRC=1 ONLY-IN-DST=1", result.stdout
        )


if __name__ == "__main__":
    unittest.main()
