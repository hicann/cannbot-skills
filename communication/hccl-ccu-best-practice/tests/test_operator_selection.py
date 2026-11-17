#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Typos must not hide source drift; report missing operator knowledge explicitly."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]


class OperatorSelectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(
            ["git", "init", "-q", str(self.repo)],
            check=True,
            env={
                **os.environ,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
            },
        )
        (self.repo / "src/ops/all_reduce").mkdir(parents=True)
        (self.repo / "src/ops/other_op").mkdir(parents=True)
        self.common = self.repo / "common.h"
        self.common.write_text("reviewed common")
        self.op_source = self.repo / "src/ops/all_reduce/algorithm.cc"
        self.op_source.write_text("reviewed operator")

        def entry(path):
            return dict(
                repo="hccl",
                path=str(path.relative_to(self.repo)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )

        self.baseline = self.root / "baseline.json"
        self.baseline.write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    cards={
                        "common": dict(
                            scope="common fixture",
                            reference="fixture.md",
                            ops=[],
                            files=[entry(self.common)],
                        ),
                        "all-reduce": dict(
                            scope="operator fixture",
                            reference="fixture.md",
                            ops=["all_reduce"],
                            files=[entry(self.op_source)],
                        ),
                    },
                )
            )
        )

    def run_cli(self, op, *extra):
        return subprocess.run(
            [
                sys.executable,
                "-B",
                str(SKILL / "scripts/check_knowledge.py"),
                "--hccl-root",
                str(self.repo),
                "--hcomm-root",
                str(self.repo),
                "--op",
                op,
                "--baseline",
                str(self.baseline),
                "--summary",
                *extra,
            ],
            capture_output=True,
            text=True,
        )

    def test_valid_operator_checks_its_card_and_still_rejects_drift(self):
        result = self.run_cli("all_reduce")
        self.assertEqual(0, result.returncode, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual("CHECKED", data["operator"]["status"])
        self.assertEqual(["all-reduce"], data["operator"]["checked_cards"])
        self.assertTrue(data["operator"]["directory_verified"])
        self.op_source.write_text("changed operator")
        result = self.run_cli("all_reduce")
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertEqual("CHANGED", json.loads(result.stdout)["cards"][1]["status"])

    def test_misspelling_and_hvm_alias_cannot_turn_drift_into_success(self):
        self.op_source.write_text("changed operator")
        for op in ("allreduce", "all_redcue", "missing", "../all_reduce", ""):
            with self.subTest(op=op):
                result = self.run_cli(op)
                self.assertEqual(2, result.returncode, result.stdout)
                self.assertIn("operator", result.stderr)
        self.assertEqual(2, self.run_cli("allreduce", "--cards", "common").returncode)

    def test_existing_uncovered_operator_is_explicit_and_not_a_pass(self):
        result = self.run_cli("other_op")
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertEqual("NO_CARD", json.loads(result.stdout)["operator"]["status"])
        self.assertEqual([], json.loads(result.stdout)["operator"]["checked_cards"])

    def test_explicit_common_selection_discloses_operator_omission(self):
        self.op_source.write_text("changed operator")
        result = self.run_cli("all_reduce", "--cards", "common")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            "NOT_SELECTED", json.loads(result.stdout)["operator"]["status"]
        )
        self.assertEqual([], json.loads(result.stdout)["operator"]["checked_cards"])
        result = self.run_cli("other_op", "--cards", "common")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("NO_CARD", json.loads(result.stdout)["operator"]["status"])

    def test_operator_symlink_cannot_escape_repository(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.repo / "src/ops/escaped").symlink_to(outside, target_is_directory=True)
        result = self.run_cli("escaped")
        self.assertEqual(2, result.returncode)
        self.assertIn("escapes repository", result.stderr)


if __name__ == "__main__":
    unittest.main()
