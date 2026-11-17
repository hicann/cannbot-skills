#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Guard generator-specific arithmetic, source drift and version evidence."""

import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
from check_arith_ops import GENERATOR_DIR, read_capabilities, validate_contract


class ArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tables = {
            "V1": {"ASSIGN", "ADD"},
            "V2": {"ASSIGN", "ADD", "MUL", "SUB", "AND", "SHL"},
        }
        self.document = {
            "schema_version": 1,
            "compute_ops": {
                "target_generator": "UNKNOWN",
                "operations": [
                    {
                        "expression": "offset += doubled",
                        "rep": "ADD",
                        "source_evidence": "kernel::BuildOffset line 20",
                    }
                ],
            },
        }

    def evidence(self, content):
        path = self.root / "generator.log"
        path.write_text(content)
        return {
            "path": "generator.log",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def test_unknown_generator_uses_intersection(self):
        result = validate_contract(self.document, self.tables, self.root)
        self.assertEqual(["ADD", "ASSIGN"], result["available_arithmetic_reps"])

    def test_mul_is_rejected_for_v1_and_unconfirmed_v2(self):
        for target in ("V1", "UNKNOWN", "V2"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.document["compute_ops"]["target_generator"] = target
                self.document["compute_ops"]["operations"][0]["rep"] = "MUL"
                validate_contract(self.document, self.tables, self.root)

    def test_source_backed_v2_with_matching_evidence_allows_mul(self):
        compute = self.document["compute_ops"]
        compute.update(
            target_generator="V2",
            target_evidence=self.evidence("[CcuKernelMgr] Init CcuInsGeneratorV2\n"),
        )
        compute["operations"][0].update(
            expression="offset = stride * hostIndex", rep="MUL"
        )
        self.assertEqual(
            "consistent",
            validate_contract(self.document, self.tables, self.root)["status"],
        )

    def test_v2_claim_cannot_use_v1_evidence(self):
        self.document["compute_ops"].update(
            target_generator="V2",
            target_evidence=self.evidence("Init CcuInsGeneratorV1\n"),
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            validate_contract(self.document, self.tables, self.root)

    def test_mixed_or_non_runtime_evidence_is_rejected(self):
        for content in (
            "operator* exists and compiles",
            "Init CcuInsGeneratorV1\nInit CcuInsGeneratorV2",
        ):
            with (
                self.subTest(content=content),
                self.assertRaisesRegex(ValueError, "unambiguous"),
            ):
                self.document["compute_ops"].update(
                    target_generator="V2", target_evidence=self.evidence(content)
                )
                validate_contract(self.document, self.tables, self.root)

    def test_evidence_content_change_invalidates_hash(self):
        self.document["compute_ops"].update(
            target_generator="V2",
            target_evidence=self.evidence("Init CcuInsGeneratorV2\n"),
        )
        (self.root / "generator.log").write_text("Init CcuInsGeneratorV1\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_contract(self.document, self.tables, self.root)

    def test_control_flow_is_not_misclassified_as_arithmetic(self):
        self.document["compute_ops"]["operations"][0]["rep"] = "JUMP_EQ"
        with self.assertRaisesRegex(ValueError, "unreviewed arithmetic"):
            validate_contract(self.document, self.tables, self.root)

    def test_empty_mapping_needs_reason_and_missing_mapping_fails(self):
        with self.assertRaisesRegex(ValueError, "compute_ops"):
            validate_contract({"schema_version": 1}, self.tables, self.root)
        self.document["compute_ops"]["operations"] = []
        with self.assertRaisesRegex(ValueError, "no_runtime_ops_reason"):
            validate_contract(self.document, self.tables, self.root)
        self.document["compute_ops"]["no_runtime_ops_reason"] = (
            "This kernel only forwards previously computed arguments."
        )
        self.assertEqual(
            0,
            validate_contract(self.document, self.tables, self.root)[
                "operations_checked"
            ],
        )

    def test_missing_expression_evidence_is_rejected(self):
        self.document["compute_ops"]["operations"][0]["source_evidence"] = ""
        with self.assertRaisesRegex(ValueError, "source_evidence"):
            validate_contract(self.document, self.tables, self.root)

    def test_actual_tables_are_extracted_and_drift_is_rejected(self):
        entries = []
        for version, operations in self.tables.items():
            path = self.root / GENERATOR_DIR / f"ccu_ins_generator_{version.lower()}.h"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "std::unordered_map<CcuRepType, uint32_t> repTypeInstrCount = {"
                + ",".join("{CcuRepType::" + rep + ",1}" for rep in sorted(operations))
                + "};\n"
            )
            entries.append(
                {
                    "repo": "hcomm",
                    "path": str(path.relative_to(self.root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        baseline = self.root / "knowledge.json"
        baseline.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cards": {
                        "generator-arith": {
                            "scope": "test tables",
                            "reference": "test.md",
                            "files": entries,
                        }
                    },
                }
            )
        )
        tables, _ = read_capabilities(self.root, baseline)
        self.assertEqual(self.tables, tables)
        path.write_text(path.read_text() + "// unreviewed change\n")
        with self.assertRaisesRegex(ValueError, "UNVERIFIED"):
            read_capabilities(self.root, baseline)

    def test_cli_rejects_mul_in_v1(self):
        # The real reviewed source baseline is exercised by the maintainer smoke run;
        # this CLI fixture checks rejection through source parsing and argument wiring.
        entries = []
        for version in ("v1", "v2"):
            path = self.root / GENERATOR_DIR / f"ccu_ins_generator_{version}.h"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "repTypeInstrCount = {{CcuRepType::ASSIGN,1},{CcuRepType::ADD,1}};"
            )
            entries.append(
                {
                    "repo": "hcomm",
                    "path": str(path.relative_to(self.root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        baseline = self.root / "baseline.json"
        baseline.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cards": {
                        "generator-arith": {
                            "scope": "fixture",
                            "reference": "test.md",
                            "files": entries,
                        }
                    },
                }
            )
        )
        self.document["compute_ops"]["operations"][0]["rep"] = "MUL"
        contract = self.root / "sync-contract.json"
        contract.write_text(json.dumps(self.document))
        isolated = self.root / "isolated-skill"
        shutil.copytree(
            SKILL / "scripts",
            isolated / "scripts",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        result = subprocess.run(
            [
                sys.executable,
                str(isolated / "scripts/check_arith_ops.py"),
                str(contract),
                "--hcomm-root",
                str(self.root),
                "--baseline",
                str(baseline),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(1, result.returncode)
        self.assertIn("MUL is not supported", json.loads(result.stdout)["error"])
        self.assertEqual([], list(isolated.rglob("__pycache__")))


if __name__ == "__main__":
    unittest.main()
