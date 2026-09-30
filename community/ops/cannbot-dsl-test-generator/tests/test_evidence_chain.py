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

"""Behavioral checks for test input contracts and lifecycle regressions."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_skills import fixture, write_rows, run_script

ASSETS = Path(__file__).resolve().parents[1] / "assets"
sys.path.insert(0, str(ASSETS))
from case_contract import check_probe


class EvidenceChainTests(unittest.TestCase):
    def setup_case(self, work):
        _, rows = fixture(work)
        tdd = next(row for row in rows if row["sheet"] == "tdd")
        unit_path = work / "design/units/U01.yaml"
        return rows, tdd, unit_path, json.loads(unit_path.read_text())

    def run_target(self, work, code, probe=None):
        prepared = run_script("prepare_golden.py", work)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        (work / "target_add.py").write_text(code)
        env = dict(os.environ, PYTHONPATH=str(work), CANNBOTDSL_OP_MODULE="target_add")
        if probe:
            (work / "probe.py").write_text(probe)
            env["CANNBOTDSL_PROBE_MODULE"] = "probe"
        return subprocess.run(
            [
                sys.executable,
                str(work / "operators/add/test/test_add.py"),
                "--sheet",
                "tdd",
                "--device",
                "cpu",
            ],
            env=env,
            text=True,
            capture_output=True,
        )

    def test_effective_count_is_materialized_not_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            rows, tdd, path, native = self.setup_case(work)
            check = {
                "input": "x",
                "metric": "unique_count",
                "axis": -1,
                "where": {"min": 0, "max": 129},
                "relation": "ge",
                "value": 129,
            }
            native["unit"]["verification_obligations"][0]["input_conditions"] = [check]
            path.write_text(json.dumps(native))
            recipes = {
                name: {"shape": [2, 129], "dtype": "float32", "fill": "zeros"}
                for name in ("x", "y")
            }
            tdd["inputs_json"] = json.dumps(recipes)
            tdd["assertions_json"] = json.dumps(
                [{"kind": "input_conditions", "checks": [check]}]
            )
            write_rows(work, rows)
            failed = run_script("build_tdd_cases.py", work)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("input condition not triggered", failed.stderr)
            recipes["x"].update(fill="arange", axis=-1)
            tdd["inputs_json"] = json.dumps(recipes)
            write_rows(work, rows)
            passed = run_script("build_tdd_cases.py", work)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            green = self.run_target(work, "def run(x, y):\n    return x+y\n")
            self.assertEqual(green.returncode, 0, green.stderr)
            recipes["x"]["fill"] = "zeros"
            tdd["inputs_json"] = json.dumps(recipes)
            write_rows(work, rows)
            rejected = self.run_target(
                work, 'def run(x, y):\n    raise RuntimeError("should not execute")\n'
            )
            self.assertNotEqual(rejected.returncode, 0)
            detail = (
                work / "operators/add/test/testcase_output/tdd/verify_result_tdd.csv"
            ).read_text()
            self.assertIn("input condition not triggered", detail)
            self.assertNotIn("should not execute", detail)

    def test_missing_trigger_binding_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _, _, path, native = self.setup_case(work)
            native["unit"]["verification_obligations"][0]["input_conditions"] = [
                {
                    "input": "x",
                    "metric": "dimension",
                    "axis": 1,
                    "relation": "ge",
                    "value": 129,
                }
            ]
            path.write_text(json.dumps(native))
            result = run_script("build_tdd_cases.py", work)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("input conditions not bound", result.stderr)

    def test_only_supplied_test_contract_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _, _, path, native = self.setup_case(work)
            # Planning metadata is opaque to the test generator.
            native["unit"]["requires"] = ["outside-test-contract"]
            native["unit"]["design_constraints"] = [{"caller-owned": "metadata"}]
            native["unit"]["implementation"] = {"feasibility_checks": "caller-owned"}
            path.write_text(json.dumps(native))
            result = run_script("build_tdd_cases.py", work)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_duplicate_and_malformed_obligations_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _, _, path, native = self.setup_case(work)
            obligations = native["unit"]["verification_obligations"]
            for invalid in (
                [obligations[0], obligations[0]],
                ["bad"],
                [{"id": [], "kind": "numerical"}],
            ):
                with self.subTest(obligations=invalid):
                    native["unit"]["verification_obligations"] = invalid
                    path.write_text(json.dumps(native))
                    result = run_script("build_tdd_cases.py", work)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("TypeError", result.stderr)

    def test_explicit_values_and_dimension_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            rows, tdd, _, _ = self.setup_case(work)
            tdd["inputs_json"] = json.dumps(
                {
                    "x": {
                        "shape": [2, 3],
                        "dtype": "float32",
                        "fill": "values",
                        "values": [[-1, 0, 1], [-1, 1, 2]],
                    },
                    "y": {"shape": [2, 3], "dtype": "float32", "fill": "zeros"},
                }
            )
            tdd["assertions_json"] = json.dumps(
                [
                    {
                        "kind": "input_conditions",
                        "checks": [
                            {
                                "input": "x",
                                "metric": "count",
                                "axis": 1,
                                "where": {
                                    "min": 0,
                                    "max": {"input": "y", "dimension": 1},
                                },
                                "relation": "eq",
                                "value": 2,
                            }
                        ],
                    }
                ]
            )
            write_rows(work, rows)
            result = run_script("build_tdd_cases.py", work)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = self.run_target(work, "def run(x, y):\n    return x+y\n")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_partial_order_allows_independent_interleaving(self):
        checks = [
            {
                "op": "count",
                "where": {"event": "consume", "domain": "cube"},
                "relation": "eq",
                "value": 2,
            },
            {
                "op": "before",
                "first": {"event": "ready"},
                "second": {"event": "consume"},
                "keys": ["slot", "generation"],
            },
        ]

        def ready(slot):
            return {"event": "ready", "slot": slot, "generation": 1}

        def consume(slot):
            return {"event": "consume", "domain": "cube", "slot": slot, "generation": 1}

        for events in (
            [ready(0), ready(1), consume(0), consume(1)],
            [ready(1), consume(1), ready(0), consume(0)],
        ):
            check_probe(events, checks)
        with self.assertRaises(AssertionError):
            check_probe([consume(0), ready(0), ready(1), consume(1)], checks)
        with self.assertRaises(AssertionError):
            check_probe([ready(0), ready(1), consume(0)], checks)

    def test_numeric_and_structural_evidence_share_real_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            rows, tdd, path, native = self.setup_case(work)
            native["unit"]["verification_obligations"].append(
                {"id": "VO-STRUCT", "kind": "structural"}
            )
            path.write_text(json.dumps(native))
            tdd["obligation_ids"] += ",VO-STRUCT"
            tdd["assertions_json"] = json.dumps(
                [
                    {
                        "kind": "probe_predicates",
                        "checks": [
                            {
                                "op": "count",
                                "where": {"event": "compute", "domain": "required"},
                                "relation": "eq",
                                "value": 1,
                            }
                        ],
                    }
                ]
            )
            write_rows(work, rows)
            result = run_script("build_tdd_cases.py", work)
            self.assertEqual(result.returncode, 0, result.stderr)
            probe = "events=[]\ndef reset():\n    events.clear()\ndef snapshot():\n    return events.copy()\n"
            green = self.run_target(
                work,
                'import probe\ndef run(x,y):\n    probe.events.append({"event":"compute","domain":"required"})\n    return x+y\n',
                probe,
            )
            self.assertEqual(green.returncode, 0, green.stderr)
            red = self.run_target(
                work,
                'import probe\ndef run(x,y):\n    probe.events.append({"event":"compute","domain":"other"})\n    return x+y\n',
                probe,
            )
            self.assertNotEqual(red.returncode, 0)

    def test_sequence_catches_stale_shape_and_repeats_in_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            rows, tdd, _, _ = self.setup_case(work)
            extra = {
                name: {"shape": [3, 4], "dtype": "float32", "fill": "ones"}
                for name in ("x", "y")
            }
            tdd["assertions_json"] = json.dumps(
                [
                    {
                        "kind": "execution",
                        "repeat": 3,
                        "bitwise": True,
                        "calls": [{"inputs": extra, "seed": 7}],
                    }
                ]
            )
            write_rows(work, rows)
            checked = run_script("build_tdd_cases.py", work)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            good = self.run_target(work, "def run(x,y):\n    return x+y\n")
            self.assertEqual(good.returncode, 0, good.stderr)
            detail = (
                work / "operators/add/test/testcase_output/tdd/verify_result_tdd.csv"
            ).read_text()
            self.assertIn("repetition", detail)
            bad = self.run_target(
                work,
                "cached=None\ndef run(x,y):\n    global cached\n    if cached is None:\n        cached=x+y\n    return cached\n",
            )
            self.assertNotEqual(bad.returncode, 0)

    def test_bitwise_repeat_detects_signed_zero_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            rows, tdd, _, _ = self.setup_case(work)
            tdd["inputs_json"] = json.dumps(
                {
                    name: {"shape": [2, 8], "dtype": "float32", "fill": "zeros"}
                    for name in ("x", "y")
                }
            )
            tdd["assertions_json"] = json.dumps(
                [{"kind": "execution", "repeat": 2, "bitwise": True}]
            )
            write_rows(work, rows)
            result = self.run_target(
                work,
                "calls=0\ndef run(x,y):\n    global calls\n    calls+=1\n    value=x+y\n    return value if calls%2 else -value\n",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "changed bitwise",
                (
                    work
                    / "operators/add/test/testcase_output/tdd/verify_result_tdd.csv"
                ).read_text(),
            )

    def test_invalid_sequence_inputs_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            rows, tdd, _, _ = self.setup_case(work)
            tdd["assertions_json"] = json.dumps(
                [{"kind": "execution", "calls": [{"inputs": {"unknown": 1}}]}]
            )
            write_rows(work, rows)
            result = run_script("build_tdd_cases.py", work)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
