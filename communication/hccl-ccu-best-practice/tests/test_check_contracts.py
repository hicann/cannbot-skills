#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Unified checks retain independent failures and the original source guards."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
from check_arith_ops import GENERATOR_DIR
from check_contracts import run_checks


class CombinedTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.launch = self.root / "launch.json"
        self.launch.write_bytes(
            (SKILL / "assets/launch-contract.example.json").read_bytes()
        )
        self.sync = self.root / "sync.json"
        self.sync.write_bytes(
            (SKILL / "assets/sync-contract.example.json").read_bytes()
        )
        self.baseline = self.root / "knowledge.json"
        files = {}
        for name in ("sync-limits", "generator-arith"):
            paths = (
                ["src/base_comm/resources/ccu/ccu_transport/ccu_transport_.h"]
                if name == "sync-limits"
                else [
                    GENERATOR_DIR + "/ccu_ins_generator_v1.h",
                    GENERATOR_DIR + "/ccu_ins_generator_v2.h",
                ]
            )
            entries = []
            for relative in paths:
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    "INIT_CKE_NUM = 4;\nINIT_XN_NUM = 4;\n"
                    if name == "sync-limits"
                    else "repTypeInstrCount = {{CcuRepType::ASSIGN,1},{CcuRepType::ADD,1}};\n"
                )
                entries.append(
                    {
                        "repo": "hcomm",
                        "path": relative,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            files[name] = {
                "scope": "isolated fixture",
                "reference": "test.md",
                "files": entries,
            }
        self.baseline.write_text(json.dumps({"schema_version": 1, "cards": files}))

    def check(self, **kwargs):
        return run_checks(self.launch, self.sync, self.root, self.baseline, **kwargs)

    def test_all_three_real_checkers_run_and_bind_contract_bytes(self):
        result = self.check()
        self.assertTrue(result["ok"], result)
        self.assertEqual({"launch", "sync", "arith"}, set(result["checks"]))
        self.assertEqual(
            hashlib.sha256(self.sync.read_bytes()).hexdigest(),
            result["checks"]["arith"]["contract_sha256"],
        )

    def test_launch_failure_does_not_hide_arithmetic_or_sync_failure(self):
        self.launch.write_text("{}")
        data = json.loads(self.sync.read_text())
        data["compute_ops"]["operations"][0]["rep"] = "MUL"
        data["pools"][0]["xn_indices"].append(8)
        self.sync.write_text(json.dumps(data))
        result = self.check()
        self.assertFalse(result["ok"])
        self.assertTrue(all(not r["ok"] for r in result["checks"].values()))

    def test_source_drift_is_still_rejected(self):
        path = self.root / GENERATOR_DIR / "ccu_ins_generator_v1.h"
        path.write_text(path.read_text() + "// drift\n")
        result = self.check()
        self.assertFalse(result["ok"])
        self.assertTrue(result["checks"]["launch"]["ok"])
        self.assertIn("UNVERIFIED", result["checks"]["arith"]["stdout"])

    def test_missing_checker_is_failure(self):
        result = self.check(scripts=self.root / "missing-checkers")
        self.assertFalse(result["ok"])
        self.assertTrue(
            all("missing checker" in row["error"] for row in result["checks"].values())
        )

    def test_timeout_does_not_skip_the_remaining_checks(self):
        with patch(
            "check_contracts.subprocess.run",
            side_effect=subprocess.TimeoutExpired("checker", 30),
        ) as run:
            result = self.check()
        self.assertEqual(3, run.call_count)
        self.assertFalse(result["ok"])

    def test_contract_change_during_check_is_not_bound_as_success(self):
        def change_contract(*args, **kwargs):
            self.launch.write_text(self.launch.read_text() + "\n")
            return subprocess.CompletedProcess(args[0], 0, "ok", "")

        with patch("check_contracts.subprocess.run", side_effect=change_contract):
            result = self.check()
        self.assertFalse(result["ok"])
        self.assertIn("changed while", result["checks"]["launch"]["error"])

    def test_cli_and_read_only_behavior(self):
        before = {
            str(p.relative_to(self.root)): p.read_bytes()
            for p in self.root.rglob("*")
            if p.is_file()
        }
        result = subprocess.run(
            [
                sys.executable,
                str(SKILL / "scripts/check_contracts.py"),
                "--launch",
                str(self.launch),
                "--sync",
                str(self.sync),
                "--hcomm-root",
                str(self.root),
                "--baseline",
                str(self.baseline),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertTrue(json.loads(result.stdout)["ok"])
        self.assertEqual(
            before,
            {
                str(p.relative_to(self.root)): p.read_bytes()
                for p in self.root.rglob("*")
                if p.is_file()
            },
        )


if __name__ == "__main__":
    unittest.main()
