# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile a tiny isolated C++ fixture; no CANN or HCCL build is needed."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "hccl-aicpu-best-practice/scripts/compile_probe.py"
)
SPEC = importlib.util.spec_from_file_location("probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@unittest.skipUnless(
    shutil.which("g++") and shutil.which("git"), "requires g++ and git"
)
class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hccl-compile-probe-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.source = self.repo / "src"
        self.source.mkdir()
        self.reference = self.source / "ins_temp_all_reduce_mesh_1D_one_shot.cc"
        self.reference.write_text("int reference;\n")
        (self.source / "test_header.h").write_text("#define PROBE_VALUE 1\n")
        self.git(self.repo, "init", "-q")
        self.git(self.repo, "add", "src")
        self.git(
            self.repo,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        )
        self.worktree = self.root / "worktree"
        self.git(
            self.repo,
            "clone",
            "-q",
            "--no-hardlinks",
            str(self.repo),
            str(self.worktree),
        )
        # A worktree-only header change makes include rebasing observable in compilation.
        (self.worktree / "src/test_header.h").write_text("#define PROBE_VALUE 2\n")
        self.target = self.worktree / "src/probe.cc"
        self.target.write_text(
            '#include <test_header.h>\nstatic_assert(PROBE_VALUE == 2, "mixed trees");\nint probe;\n'
        )
        self.build = self.repo / "build"
        self.build.mkdir()
        self.dep = self.build / "old.d"
        self.dep.write_text("original dependency file\n")
        self.obj = self.build / "old.o"
        self.obj.write_text("original object file\n")
        self.output = self.root / "probe.o"
        self.database = self.build / "compile_commands.json"
        self.entry = {
            "directory": str(self.build),
            "file": "../src/" + self.reference.name,
            "arguments": [
                shutil.which("g++"),
                "-Werror",
                "-std=c++17",
                "-I../src",
                "-MD",
                "-MF",
                "old.d",
                "-MTold.o",
                "-o",
                "old.o",
                "-c",
                str(self.reference),
            ],
        }
        self.save()

    def git(self, path, *args):
        return subprocess.run(
            ["git", "-C", str(path), *args], check=True, capture_output=True
        )

    def save(self):
        self.database.write_text(json.dumps([self.entry]))

    def prepare(self):
        return probe.prepare(
            self.repo, self.worktree, self.target, self.output, self.database
        )

    def test_real_compile_preserves_original_outputs_and_uses_worktree_headers(self):
        result = subprocess.run(
            [
                "python3",
                "-B",
                str(SCRIPT),
                "--repo",
                str(self.repo),
                "--worktree",
                str(self.worktree),
                "--target",
                str(self.target),
                "--output",
                str(self.output),
                "--database",
                str(self.database),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertGreater(self.output.stat().st_size, 0)
        self.assertEqual(self.dep.read_text(), "original dependency file\n")
        self.assertEqual(self.obj.read_text(), "original object file\n")
        self.assertFalse((self.root / "probe.d").exists())

    def test_separate_and_joined_include_paths_rebased(self):
        self.entry["arguments"][3:4] = [
            "-I",
            "../src",
            "-isystem../src",
            "-iquote",
            "../src",
            "-include../src/test_header.h",
            "-imacros",
            "../src/test_header.h",
            "-I" + str(self.build),
        ]
        self.save()
        argv, _ = self.prepare()
        self.assertIn(str(self.worktree / "src"), argv)
        self.assertIn("-isystem" + str(self.worktree / "src"), argv)
        self.assertIn("-include" + str(self.worktree / "src/test_header.h"), argv)
        self.assertIn("-I" + str(self.build), argv)

    def test_dirty_original_and_mismatched_head_rejected(self):
        self.reference.write_text("int changed;\n")
        with self.assertRaisesRegex(ValueError, "未提交修改"):
            self.prepare()
        self.git(self.repo, "add", "src")
        self.git(
            self.repo,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "changed",
        )
        with self.assertRaisesRegex(ValueError, "HEAD 不一致"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_forwarded_dependency_output_rejected(self):
        self.entry["arguments"].append("-Wp,-MD," + str(self.dep))
        self.save()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.dep.read_text(), "original dependency file\n")

    def test_existing_and_in_repo_outputs_rejected(self):
        self.output = self.obj
        with self.assertRaises(ValueError):
            self.prepare()
        self.output = self.root / "existing.o"
        self.output.write_text("preserve")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.output.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
