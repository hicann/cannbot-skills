#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Whitebox CSV validation must produce cases executable by the fixed runner."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_skills import PAIR, fixture, row, run_script, write_rows


class WhiteboxContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        _, self.rows = fixture(self.work)
        self.case = next(r for r in self.rows if r["case_id"] == "WB-B0")

    def validate(self):
        path = write_rows(self.work, self.rows)
        before = path.read_bytes()
        result = run_script("build_whitebox_cases.py", self.work)
        self.assertEqual(path.read_bytes(), before, "validation must not rewrite CSV")
        return result

    def test_active_expected_is_required_and_supported(self):
        for expected in ("", " ", "match_goldn", "raises_error:shape_mismatch"):
            with self.subTest(expected=expected):
                self.case["expected"] = expected
                result = self.validate()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("WB-B0: expected must be", result.stderr)

    def test_invalid_assertion_contracts_are_rejected_before_execution(self):
        condition = {
            "kind": "input_conditions",
            "checks": [{"input": "x", "metric": "numel", "relation": "gt", "value": 0}],
        }
        observable = {"kind": "output_shape", "shape": [2, 8]}
        cases = [
            ("match_golden", "{", "invalid execution contract"),
            ("match_golden", "{}", "assertions must be a list"),
            ("match_golden", '[{"kind":"unknown"}]', "unknown assertion kind"),
            ("assertions", "[]", "lacks observable behavior"),
            ("assertions", json.dumps([condition]), "lacks observable behavior"),
            (
                "assertions",
                json.dumps([{"kind": "execution", "repeat": 2, "bitwise": True}]),
                "lacks observable behavior",
            ),
            (
                "assertions",
                json.dumps(
                    [
                        observable,
                        {
                            "kind": "execution",
                            "calls": [{"inputs": PAIR, "assertions": []}],
                        },
                    ]
                ),
                "lacks observable behavior",
            ),
        ]
        for expected, assertions, message in cases:
            with self.subTest(expected=expected, assertions=assertions):
                self.case.update(expected=expected, assertions_json=assertions)
                result = self.validate()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("WB-B0:", result.stderr)
                self.assertIn(message, result.stderr)

    def test_validated_modes_execute_and_excluded_blank_is_skipped(self):
        # The fixture includes an excluded branch with no expected outcome.
        nan_inputs = {
            "x": {"shape": [8], "dtype": "float32", "fill": "nan_one"},
            "y": {"shape": [8], "dtype": "float32", "fill": "zeros"},
        }
        for branch, inputs, expected, assertions in (
            ("BNan", nan_inputs, "match_golden_nan", []),
            (
                "BBehavior",
                PAIR,
                "assertions",
                [{"kind": "output_shape", "shape": [2, 8]}],
            ),
            (
                "BCombined",
                PAIR,
                "match_golden",
                [{"kind": "input_unchanged", "inputs": ["x", "y"]}],
            ),
            (
                "BSequence",
                PAIR,
                "assertions",
                [
                    {"kind": "output_dtype", "dtype": "float32"},
                    {"kind": "execution", "calls": [{"inputs": PAIR}]},
                ],
            ),
        ):
            self.rows.append(
                row(
                    "whitebox",
                    f"WB-{branch}",
                    inputs,
                    branch_ids=branch,
                    expected=expected,
                    assertions_json=json.dumps(assertions),
                    note="return_value",
                    design_ref="Stage-Elementwise",
                    condition=branch,
                )
            )
        self.assertEqual(self.validate().returncode, 0)
        prepared = run_script("prepare_golden.py", self.work)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        test = self.work / "operators/add/test"
        target = test / "target_add.py"
        target.write_text("def run(x, y):\n    return x + y\n", encoding="utf-8")
        self.run_whitebox(test, expected_count=5)

        # The same explicit outcome contract applies to destination arguments.
        self.rows = [self.case]
        self.case.update(
            note="dst_args",
            output_json=json.dumps({"z": {"shape": [2, 8], "dtype": "float32"}}),
        )
        self.assertEqual(self.validate().returncode, 0)
        target.write_text("def run(x, y, z):\n    z.copy_(x + y)\n", encoding="utf-8")
        self.run_whitebox(test, expected_count=1)

    def run_whitebox(self, test, expected_count):
        result = subprocess.run(
            [
                sys.executable,
                str(test / "test_add.py"),
                "--sheet",
                "whitebox",
                "--device",
                "cpu",
            ],
            env=dict(os.environ, CANNBOTDSL_OP_MODULE="target_add"),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        with (test / "testcase_output/whitebox/verify_result_whitebox.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            results = list(csv.DictReader(handle))
        self.assertEqual(len(results), expected_count)
        self.assertTrue(all(r["status"] == "passed" for r in results), results)


if __name__ == "__main__":
    unittest.main()
