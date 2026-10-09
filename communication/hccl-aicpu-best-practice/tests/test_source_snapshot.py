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

"""Offline recovery tests using temporary Git repositories only."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


snapshot = module("source_snapshot")
archive = module("archive_package")


class SourceSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init")
        self.git("config", "user.name", "Snapshot Test")
        self.git("config", "user.email", "snapshot@example.invalid")
        for name, data in [
            ("tracked", b"base\n"),
            ("deleted", b"remove me\n"),
            ("binary", b"\x00base"),
        ]:
            (self.repo / name).write_bytes(data)
        (self.repo / ".gitignore").write_text("ignored\n")
        self.git("add", ".")
        self.git("commit", "-m", "base")
        self.out = self.root / "snapshot"

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args], stderr=subprocess.PIPE
        )

    def changes(self):
        (self.repo / "tracked").write_text("changed\n")
        (self.repo / "tracked").chmod(0o755)
        (self.repo / "deleted").unlink()
        (self.repo / "binary").write_bytes(b"\x00changed\xff")
        (self.repo / "new").mkdir()
        (self.repo / "new/kernel.cc").write_text("new implementation\n")
        (self.repo / "new/run.sh").write_text("#!/bin/sh\n")
        (self.repo / "new/run.sh").chmod(0o755)
        (self.repo / "ignored").write_text("excluded")

    def test_recover_tracked_deleted_binary_untracked_and_modes(self):
        self.changes()
        before = self.git("status", "--porcelain")
        snapshot.capture(self.repo, self.out)
        restored = snapshot.restore(self.out, self.repo, self.root / "restored")
        for name in ("tracked", "binary", "new/kernel.cc", "new/run.sh"):
            self.assertEqual(
                (restored / name).read_bytes(), (self.repo / name).read_bytes()
            )
            self.assertEqual(
                (restored / name).stat().st_mode & 0o777,
                (self.repo / name).stat().st_mode & 0o777,
            )
        self.assertFalse((restored / "deleted").exists())
        self.assertFalse((restored / "ignored").exists())
        self.assertEqual(before, self.git("status", "--porcelain"))

    def test_capture_and_restore_refuse_existing_and_in_repo(self):
        snapshot.capture(self.repo, self.out)
        with self.assertRaises(ValueError):
            snapshot.capture(self.repo, self.out)
        with self.assertRaises(ValueError):
            snapshot.capture(self.repo, self.repo / "snapshot")
        with self.assertRaises(ValueError):
            snapshot.restore(self.out, self.repo, self.repo)
        with self.assertRaises(ValueError):
            snapshot.restore(self.out, self.repo, self.repo / "restored")

    def test_legacy_snapshot_without_tracked_modes_remains_restorable(self):
        self.changes()
        snapshot.capture(self.repo, self.out)
        manifest_path = self.out / "snapshot.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("tracked_modes")
        manifest_path.write_text(json.dumps(manifest))

        restored = snapshot.restore(self.out, self.repo, self.root / "restored")
        self.assertEqual((restored / "tracked").read_text(), "changed\n")
        self.assertTrue((restored / "tracked").stat().st_mode & 0o100)

    def test_untracked_symlink_rejected_without_reading_target(self):
        (self.repo / "external").symlink_to(self.root / "absent")
        with self.assertRaisesRegex(ValueError, "symlink"):
            snapshot.capture(self.repo, self.out)
        self.assertFalse(self.out.exists())

    def test_tampered_blob_and_unsafe_manifest_rejected(self):
        self.changes()
        snapshot.capture(self.repo, self.out)
        manifest = snapshot.verify(self.out)
        blob = self.out / manifest["untracked"][0]["blob"]
        blob.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            snapshot.verify(self.out)
        manifest["untracked"][0]["path"] = "../outside"
        (self.out / "snapshot.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "unsafe"):
            snapshot.verify(self.out)

    def test_invalid_tracked_mode_rejected(self):
        self.changes()
        snapshot.capture(self.repo, self.out)
        manifest_path = self.out / "snapshot.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tracked_modes"][0]["mode"] = 0o1000
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "invalid file mode"):
            snapshot.verify(self.out)

    def test_missing_base_rejected_before_clone(self):
        snapshot.capture(self.repo, self.out)
        manifest_path = self.out / "snapshot.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["base_commit"] = "0" * 40
        manifest_path.write_text(json.dumps(manifest))
        restored = self.root / "restored"
        with self.assertRaises(subprocess.CalledProcessError):
            snapshot.restore(self.out, self.repo, restored)
        self.assertFalse(restored.exists())

    def archive_args(self):
        build = self.repo / "build_out"
        build.mkdir()
        for name in ("pkg.run", "build.log", "host.so", "device.so"):
            (build / name).write_bytes(name.encode())
        (build / "rc").write_text("0\n")
        return SimpleNamespace(
            repo=str(self.repo),
            package=str(build / "pkg.run"),
            build_log=str(build / "build.log"),
            exit_code_file=str(build / "rc"),
            output=str(self.root / "archive"),
            clean_root=[],
            host_library=str(build / "host.so"),
            device_library=str(build / "device.so"),
            source_id="fixture",
            environment_id="fixture",
            build_command="fixture",
            source_snapshot=str(self.out),
        )

    def test_archive_copies_recoverable_snapshot_and_manifest_hash(self):
        self.changes()
        snapshot.capture(self.repo, self.out)
        a = self.archive_args()
        path = archive.archive(a)
        artifact = json.loads(path.read_text())
        saved = path.parent / artifact["source_snapshot"]["path"]
        self.assertEqual(
            artifact["source_snapshot"]["manifest_sha256"],
            archive.digest(saved / "snapshot.json"),
        )
        restored = snapshot.restore(saved, self.repo, self.root / "restored")
        self.assertEqual(
            (restored / "new/kernel.cc").read_text(), "new implementation\n"
        )

    def test_legacy_archive_does_not_claim_source_recovery(self):
        a = self.archive_args()
        a.source_snapshot = None
        self.assertNotIn("source_snapshot", json.loads(archive.archive(a).read_text()))


if __name__ == "__main__":
    unittest.main()
