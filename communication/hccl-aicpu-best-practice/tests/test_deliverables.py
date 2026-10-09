# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Delivery checks use disposable Git repositories and explicit artifact paths."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_deliverables.py"
spec = importlib.util.spec_from_file_location("deliverables", SCRIPT)
delivery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        (self.repo / ".gitignore").write_text("*.md\n")
        (self.repo / "kernel.cc").write_text("source")
        self.git("add", ".")
        self.git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        )

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args], stderr=subprocess.PIPE
        )

    def manifest(self, path, mode="git"):
        return {"files": [{"path": str(path), "role": "spec", "delivery": mode}]}

    def test_ignored_spec_is_detected_without_git_mutation(self):
        (self.repo / "spec.md").write_text("spec")
        before = self.git("status", "--porcelain")
        result = delivery.inspect(self.repo, self.manifest("spec.md"))
        self.assertFalse(result["passed"])
        self.assertTrue(result["files"][0]["ignored"])
        self.assertIsNotNone(result["files"][0]["sha256"])
        self.assertEqual(before, self.git("status", "--porcelain"))
        # The checker does not stage; emulate the author's delivery decision.
        self.git("add", "-f", "spec.md")
        self.assertTrue(delivery.inspect(self.repo, self.manifest("spec.md"))["passed"])

    def test_artifact_hash_changes_and_missing_file_fails(self):
        path = self.root / "review.md"
        self.assertFalse(
            delivery.inspect(self.repo, self.manifest(path, "artifact"))["passed"]
        )
        path.write_text("v1")
        first = delivery.inspect(self.repo, self.manifest(path, "artifact"))
        path.write_text("v2")
        second = delivery.inspect(self.repo, self.manifest(path, "artifact"))
        self.assertTrue(second["passed"])
        self.assertNotEqual(first["files"][0]["sha256"], second["files"][0]["sha256"])
        with self.assertRaises(ValueError):
            delivery.inspect(self.repo, self.manifest(path))

    def test_missing_tracked_file_and_symlink_fail(self):
        (self.repo / "kernel.cc").unlink()
        self.assertFalse(
            delivery.inspect(self.repo, self.manifest("kernel.cc"))["passed"]
        )
        (self.repo / "link").symlink_to(self.repo / ".gitignore")
        self.assertFalse(
            delivery.inspect(self.repo, self.manifest("link", "artifact"))["passed"]
        )

    def test_cli_exit_and_exclusive_output(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(self.manifest("spec.md")))
        output = self.root / "result.json"
        cmd = [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--repo",
            str(self.repo),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ]
        self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 1)
        before = output.read_bytes()
        self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 2)
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
