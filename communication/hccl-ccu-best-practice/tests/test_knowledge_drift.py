#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Drift hints must be actionable, bounded, read-only and never confer trust."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
from check_knowledge import check_cards


class DriftTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "src/ops/all_gather").mkdir(parents=True)
        self.source = self.root / "framework.cc"
        self.source.write_text("void Helper() {\n    Original();\n}\n")
        self.git("init", "-q")
        self.git("add", "framework.cc")
        self.git(
            "-c",
            "user.name=Skill Test",
            "-c",
            "user.email=skill-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        )
        self.head = self.git("rev-parse", "HEAD").strip()
        self.baseline = self.root / "knowledge.json"
        self.data = {
            "schema_version": 1,
            "reviewed_heads": {"hccl": self.head},
            "cards": {
                "framework": {
                    "scope": "fixture",
                    "reference": "framework.md",
                    "files": [
                        {
                            "repo": "hccl",
                            "path": "framework.cc",
                            "sha256": hashlib.sha256(
                                self.source.read_bytes()
                            ).hexdigest(),
                            "symbols": ["Helper"],
                        }
                    ],
                },
                "example": {
                    "scope": "optional sample",
                    "reference": "example.md",
                    "optional": True,
                    "files": [
                        {"repo": "hccl", "path": "missing.cc", "sha256": "a" * 64}
                    ],
                },
            },
        }
        self.write()

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    def write(self):
        self.baseline.write_text(json.dumps(self.data))

    def card(self, **kwargs):
        return check_cards(
            {"hccl": self.root},
            "all_gather",
            self.baseline,
            only={"framework"},
            explain=True,
            **kwargs,
        )["cards"][0]

    def test_matching_card_needs_no_diff(self):
        card = self.card()
        self.assertEqual("MATCH", card["status"])
        self.assertNotIn("review_hint", card["files"][0])

    def test_changed_symbol_is_located_without_clearing_drift(self):
        self.source.write_text(self.source.read_text().replace("Original", "Changed"))
        card = self.card()
        self.assertEqual("CHANGED", card["status"])
        hint = card["files"][0]["review_hint"]
        self.assertEqual("DIFF_AVAILABLE", hint["status"])
        self.assertIn("Helper", hint["nearby_symbols"])
        self.assertIn("+    Changed();", hint["diff"])
        self.assertEqual(2, hint["changed_ranges"][0]["current_start"])

    def test_wrong_baseline_blob_is_not_used_as_evidence(self):
        self.data["cards"]["framework"]["files"][0]["sha256"] = "0" * 64
        self.write()
        card = self.card()
        self.assertEqual("CHANGED", card["status"])
        self.assertEqual(
            "BASELINE_UNAVAILABLE", card["files"][0]["review_hint"]["status"]
        )

    def test_missing_commit_does_not_invent_diff(self):
        self.data["reviewed_heads"]["hccl"] = "f" * 40
        self.write()
        self.source.write_text("changed\n")
        self.assertEqual(
            "BASELINE_UNAVAILABLE", self.card()["files"][0]["review_hint"]["status"]
        )

    def test_per_card_revision_overrides_global_revision(self):
        self.data["reviewed_heads"]["hccl"] = "f" * 40
        self.data["cards"]["framework"]["reviewed_heads"] = {"hccl": self.head}
        self.write()
        self.source.write_text("changed\n")
        self.assertEqual(
            "DIFF_AVAILABLE", self.card()["files"][0]["review_hint"]["status"]
        )

    def test_output_is_bounded_and_read_only(self):
        self.source.write_text("\n".join(f"new line {i}" for i in range(100)))
        before = (
            self.source.read_bytes(),
            (self.root / ".git/index").read_bytes(),
            self.git("rev-parse", "HEAD"),
        )
        hint = self.card(diff_lines=5)["files"][0]["review_hint"]
        self.assertEqual(5, len(hint["diff"]))
        self.assertTrue(hint["truncated"])
        self.assertEqual(
            before,
            (
                self.source.read_bytes(),
                (self.root / ".git/index").read_bytes(),
                self.git("rev-parse", "HEAD"),
            ),
        )
        with self.assertRaises(ValueError):
            self.card(diff_lines=0)

    def test_cli_selection_excludes_unused_optional_sample(self):
        command = [
            sys.executable,
            str(SKILL / "scripts/check_knowledge.py"),
            "--hccl-root",
            str(self.root),
            "--hcomm-root",
            str(self.root),
            "--op",
            "all_gather",
            "--baseline",
            str(self.baseline),
            "--cards",
            "framework",
            "--summary",
            "--explain",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            ["framework"], [c["card_id"] for c in json.loads(result.stdout)["cards"]]
        )
        result = subprocess.run(
            command + ["--cards", "unknown"], capture_output=True, text=True
        )
        self.assertEqual(2, result.returncode)


if __name__ == "__main__":
    unittest.main()
