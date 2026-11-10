# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Preflight route checks using a synthetic environment, without executing an operator."""

import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/preflight.sh"


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="migration env ")
        self.addCleanup(self.temp.cleanup)
        self.env = Path(self.temp.name)
        (self.env / "set_env.sh").write_text(
            f"export ASCEND_HOME_PATH={shlex.quote(str(self.env))}\n"
            f"export ASCEND_OPP_PATH={shlex.quote(str(self.env / 'opp'))}\n"
        )

    def run_preflight(self, require_geir=False, repo=None):
        return subprocess.run(
            ["bash", str(SCRIPT)]
            + (["--require-geir"] if require_geir else [])
            + [str(self.env)]
            + ([str(repo), "math", "review_probe", "ops-math"] if repo else []),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_aclnn_only_does_not_require_geir_header(self):
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_geir_route_requires_header(self):
        result = self.run_preflight(require_geir=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        header = self.env / "opp/built-in/op_proto/inc/array_ops.h"
        header.parent.mkdir(parents=True)
        header.touch()
        result = self.run_preflight(require_geir=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_environment_load_failure_still_blocks_aclnn(self):
        (self.env / "set_env.sh").write_text("return 1\n")
        result = self.run_preflight()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def make_repo(self):
        repo = self.env / "ops-math"
        subprocess.run(
            ["git", "init", "-q", str(repo)], capture_output=True, check=True
        )
        (repo / "math").mkdir()
        return repo

    def git_status(self, repo):
        return subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_clean_git_checkout_is_reported_clean(self):
        repo = self.make_repo()
        result = self.run_preflight(repo=repo)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[ OK ] git 工作区干净", result.stdout)
        self.assertNotIn("[FAIL]", result.stdout)

    def test_dirty_git_checkout_preserves_staged_and_unstaged_changes(self):
        repo = self.make_repo()
        staged = repo / "staged.txt"
        staged.write_text("staged content\n")
        subprocess.run(
            ["git", "-C", str(repo), "add", "staged.txt"],
            capture_output=True,
            check=True,
        )
        staged.write_text("unstaged content\n")
        for index in range(8):
            (repo / f"untracked_{index}.txt").write_text(f"keep {index}\n")
        before = self.git_status(repo)
        self.assertEqual(before.returncode, 0, before.stderr)
        self.assertEqual(len(before.stdout.splitlines()), 9)
        result = self.run_preflight(repo=repo)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[WARN] git 工作区不干净", result.stdout)
        self.assertNotIn("[ OK ] git 工作区干净", result.stdout)
        self.assertIn("untracked_3.txt", result.stdout)
        self.assertNotIn("untracked_4.txt", result.stdout)
        self.assertEqual(self.git_status(repo).stdout, before.stdout)
        self.assertEqual(staged.read_text(), "unstaged content\n")
        saved = subprocess.run(
            ["git", "-C", str(repo), "show", ":staged.txt"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(saved.stdout, "staged content\n")
        for index in range(8):
            self.assertEqual(
                (repo / f"untracked_{index}.txt").read_text(), f"keep {index}\n"
            )

    def test_git_status_failure_blocks_preflight(self):
        repo = self.make_repo()
        index = repo / ".git/index"
        corrupt_index = b"not-a-valid-index\n"
        index.write_bytes(corrupt_index)
        subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            check=True,
        )
        status = self.git_status(repo)
        self.assertEqual(status.returncode, 128)
        result = self.run_preflight(repo=repo)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("[FAIL] git status failed", result.stdout)
        self.assertIn(status.stderr.strip(), result.stdout)
        self.assertNotIn("[ OK ] git 工作区干净", result.stdout)
        self.assertNotIn("[WARN] git 工作区不干净", result.stdout)
        self.assertEqual(index.read_bytes(), corrupt_index)


if __name__ == "__main__":
    unittest.main()
