#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Regressions for source-scoped knowledge reuse and actual sync budgets."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
from check_knowledge import check_cards
from check_sync_contract import SyncError, read_limits, validate_contract


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(
            (SKILL / "assets/sync-contract.example.json").read_text()
        )
        self.limits = {"notify_slots": 4, "xn_slots": 4, "mask_bits": 16}
        self.signals = self.contract["pools"][0]["epochs"][0]["signals"]

    def test_reviewed_ring_upper_bound_and_deduplication(self):
        self.assertEqual(
            3, validate_contract(self.contract, self.limits)["pools_checked"]
        )

    def test_original_post_slot_five_is_rejected(self):
        self.signals[-1]["slot"] = 5
        with self.assertRaisesRegex(SyncError, "actual 4"):
            validate_contract(self.contract, self.limits)

    def test_rank_65_overflows_reserved_post_bit(self):
        self.contract["pools"][1]["epochs"][0]["signals"][-2]["mask"] = 65535
        with self.assertRaisesRegex(SyncError, "overlapping"):
            validate_contract(self.contract, self.limits)

    def test_n2_merged_channel_collisions_are_rejected(self):
        self.signals[1]["mask"] = 2
        with self.assertRaisesRegex(SyncError, "overlapping"):
            validate_contract(self.contract, self.limits)

    def test_xn_limit(self):
        self.contract["pools"][0]["xn_indices"].append(4)
        with self.assertRaisesRegex(SyncError, "XN index"):
            validate_contract(self.contract, self.limits)

    def test_mask_zero_overflow_and_boolean(self):
        for mask in (0, 65536, True):
            with self.subTest(mask=mask), self.assertRaises(SyncError):
                contract = copy.deepcopy(self.contract)
                contract["pools"][0]["epochs"][0]["signals"][0]["mask"] = mask
                validate_contract(contract, self.limits)

    def test_cross_epoch_reuse_needs_explicit_proof(self):
        epoch = copy.deepcopy(self.contract["pools"][0]["epochs"][0])
        epoch["name"] = "replay"
        self.contract["pools"][0]["epochs"].append(epoch)
        with self.assertRaisesRegex(SyncError, "reuse_proof"):
            validate_contract(self.contract, self.limits)
        epoch["reuse_proof"] = (
            "Design section R: all consumers acknowledge before any next-epoch producer. Reviewer must verify."
        )
        self.assertEqual(
            "consistent", validate_contract(self.contract, self.limits)["status"]
        )

    def test_missing_evidence_or_schema_cannot_pass(self):
        for document in (
            {},
            {"schema_version": 1, "pools": []},
            {"schema_version": True, "pools": []},
        ):
            with self.subTest(document=document), self.assertRaises(SyncError):
                validate_contract(document, self.limits)
        self.contract["pools"][0]["source_evidence"] = ""
        with self.assertRaises(SyncError):
            validate_contract(self.contract, self.limits)


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.header = (
            self.root / "src/base_comm/resources/ccu/ccu_transport/ccu_transport_.h"
        )
        self.header.parent.mkdir(parents=True)
        self.header.write_text(
            "static constexpr uint32_t INIT_CKE_NUM = 4;\nstatic constexpr uint32_t INIT_XN_NUM = 4;\n"
        )
        self.baseline = self.root / "knowledge.json"
        self.data = {
            "schema_version": 1,
            "cards": {
                "sync-limits": {
                    "ops": [],
                    "scope": "test fixture",
                    "reference": "test.md",
                    "files": [
                        {
                            "repo": "hcomm",
                            "path": str(self.header.relative_to(self.root)),
                            "sha256": hashlib.sha256(
                                self.header.read_bytes()
                            ).hexdigest(),
                            "symbols": ["INIT_CKE_NUM"],
                        }
                    ],
                }
            },
        }
        self.write_baseline()

    def write_baseline(self):
        self.baseline.write_text(json.dumps(self.data))

    def test_exact_bytes_allow_reuse(self):
        result = check_cards({"hcomm": self.root}, "all_gather", self.baseline)
        self.assertEqual("MATCH", result["cards"][0]["status"])
        self.assertEqual(4, read_limits(self.root, self.baseline)[0]["notify_slots"])

    def test_uncommitted_change_invalidates_knowledge_and_limits(self):
        self.header.write_text(
            self.header.read_text().replace("CKE_NUM = 4", "CKE_NUM = 8")
        )
        result = check_cards({"hcomm": self.root}, "all_gather", self.baseline)
        self.assertEqual("CHANGED", result["cards"][0]["status"])
        with self.assertRaisesRegex(SyncError, "UNVERIFIED"):
            read_limits(self.root, self.baseline)

    def test_missing_source_is_not_match(self):
        self.header.unlink()
        self.assertEqual(
            "UNVERIFIED",
            check_cards({"hcomm": self.root}, "all_gather", self.baseline)["cards"][0][
                "status"
            ],
        )

    def test_missing_repository_is_unverified(self):
        self.assertEqual(
            "UNVERIFIED",
            check_cards({}, "all_gather", self.baseline)["cards"][0]["status"],
        )

    def test_path_escape_and_empty_scope_are_rejected(self):
        card = self.data["cards"]["sync-limits"]
        card["files"][0]["path"] = "../outside.h"
        self.write_baseline()
        with self.assertRaisesRegex(ValueError, "escapes"):
            check_cards({"hcomm": self.root}, "all_gather", self.baseline)
        card["files"] = []
        self.write_baseline()
        with self.assertRaisesRegex(ValueError, "empty"):
            check_cards({"hcomm": self.root}, "all_gather", self.baseline)

    def test_other_operator_card_is_not_loaded(self):
        self.data["cards"]["sync-limits"]["ops"] = ["all_reduce"]
        self.write_baseline()
        self.assertEqual(
            [], check_cards({"hcomm": self.root}, "all_gather", self.baseline)["cards"]
        )

    def test_cli_uses_source_backed_limits(self):
        isolated = self.root / "isolated-skill"
        shutil.copytree(
            SKILL / "scripts",
            isolated / "scripts",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        result = subprocess.run(
            [
                sys.executable,
                str(isolated / "scripts/check_sync_contract.py"),
                str(SKILL / "assets/sync-contract.example.json"),
                "--hcomm-root",
                str(self.root),
                "--baseline",
                str(self.baseline),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertEqual(4, json.loads(result.stdout)["limits"]["notify_slots"])
        self.assertEqual([], list(isolated.rglob("__pycache__")))


if __name__ == "__main__":
    unittest.main()
