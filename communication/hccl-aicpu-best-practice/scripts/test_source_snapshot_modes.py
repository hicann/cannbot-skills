# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Real CLI round trips preserve source permissions and existing contracts."""

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("source_snapshot.py")


class SnapshotModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        previous_umask = os.umask(0o022)
        self.addCleanup(os.umask, previous_umask)
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        }
        self.env.update(
            GIT_AUTHOR_NAME="Snapshot Test",
            GIT_AUTHOR_EMAIL="snapshot@example.invalid",
            GIT_COMMITTER_NAME="Snapshot Test",
            GIT_COMMITTER_EMAIL="snapshot@example.invalid",
        )
        self.command("git", "init", str(self.repo))
        self.file = self.repo / "source.cpp"
        self.file.write_bytes(b"int main() { return 0; }\n")
        self.file.chmod(0o644)
        self.commit()

    def command(self, *args):
        return subprocess.run(
            args, env=self.env, check=True, capture_output=True, text=True
        ).stdout

    def commit(self):
        self.command("git", "-C", str(self.repo), "add", ".")
        self.command("git", "-C", str(self.repo), "commit", "-m", "Snapshot fixture")

    def round_trip(self):
        snapshot = self.root / "snapshot"
        restored = self.root / "restored"
        self.command(
            sys.executable,
            str(SCRIPT),
            "capture",
            "--repo",
            str(self.repo),
            "--output",
            str(snapshot),
        )
        self.command(sys.executable, str(SCRIPT), "verify", str(snapshot))
        self.command(
            sys.executable,
            str(SCRIPT),
            "restore",
            str(snapshot),
            "--repository",
            str(self.repo),
            "--output",
            str(restored),
        )
        manifest = json.loads((snapshot / "snapshot.json").read_text())
        self.assertEqual(manifest["schema_version"], 1)
        return restored

    def assert_mode_and_bytes(self, restored, name, mode, expected):
        path = restored / name
        self.assertEqual(path.read_bytes(), expected)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), mode)

    def test_owner_only_permission_change(self):
        self.file.chmod(0o600)
        self.assertEqual(
            self.command("git", "-C", str(self.repo), "status", "--porcelain"), ""
        )
        self.assert_mode_and_bytes(
            self.round_trip(), "source.cpp", 0o600, self.file.read_bytes()
        )

    def test_group_read_permission_change(self):
        self.file.chmod(0o640)
        self.assertEqual(self.command("git", "-C", str(self.repo), "diff", "HEAD"), "")
        self.assert_mode_and_bytes(
            self.round_trip(), "source.cpp", 0o640, self.file.read_bytes()
        )

    def test_executable_permission_change_without_git_diff(self):
        self.file.chmod(0o755)
        self.commit()
        self.file.chmod(0o700)
        self.assertEqual(self.command("git", "-C", str(self.repo), "diff", "HEAD"), "")
        self.assert_mode_and_bytes(
            self.round_trip(), "source.cpp", 0o700, self.file.read_bytes()
        )

    def test_content_and_permission_change(self):
        self.file.write_bytes(b"int main() { return 7; }\n")
        self.file.chmod(0o640)
        self.assert_mode_and_bytes(
            self.round_trip(), "source.cpp", 0o640, self.file.read_bytes()
        )

    def test_clean_regular_file(self):
        self.assert_mode_and_bytes(
            self.round_trip(), "source.cpp", 0o644, self.file.read_bytes()
        )

    def test_untracked_binary_permission(self):
        data = b"untracked\x00\xff"
        (self.repo / "extra.bin").write_bytes(data)
        (self.repo / "extra.bin").chmod(0o600)
        self.assert_mode_and_bytes(self.round_trip(), "extra.bin", 0o600, data)

    def test_deleted_tracked_file_stays_deleted(self):
        self.file.unlink()
        self.assertFalse((self.round_trip() / "source.cpp").exists())

    def test_unchanged_tracked_symlink_remains_supported(self):
        (self.repo / "source-link").symlink_to("source.cpp")
        self.commit()
        restored = self.round_trip()
        self.assertTrue((restored / "source-link").is_symlink())
        self.assertEqual(os.readlink(restored / "source-link"), "source.cpp")


if __name__ == "__main__":
    unittest.main()
