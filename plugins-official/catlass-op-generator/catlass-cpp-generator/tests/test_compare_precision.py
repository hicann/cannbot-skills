# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression checks for complete-case precision acceptance."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/catlass-cpp-test/scripts"))

from compare_precision import compare, compare_case


SKILL_ROOT = ROOT / "skills/catlass-cpp-test"
POLICY_PATH = ROOT / "skills/catlass-cpp-reference/templates/precision-policy.json"
HAS_TORCH = importlib.util.find_spec("torch") is not None


class PrecisionCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        tail = np.zeros(1000, dtype=bool)
        tail[-1] = True
        self.arguments = {
            "case": "tail_case",
            "expected_outputs": {
                "out": {"shape": [1000], "dtype": "float16", "required_regions": ["tail"]},
            },
            "actual": {"out": np.ones(1000, dtype=np.float16)},
            "golden": {"out": np.ones(1000, dtype=np.float64)},
            "valid_masks": {"out": np.ones(1000, dtype=bool)},
            "regions": {"out": {"tail": tail}},
            "policy": self.policy,
        }

    def test_high_precision_golden_passes_without_matching_actual_dtype(self) -> None:
        result = compare_case(**self.arguments)
        self.assertTrue(result["pass"])
        self.assertEqual("PASS", result["status"])
        self.assertEqual("float16", result["outputs"]["out"]["actual_dtype"])
        self.assertEqual("float64", result["outputs"]["out"]["golden_dtype"])

    def test_tail_error_passes_global_ratio_but_fails_required_region(self) -> None:
        self.arguments["actual"]["out"][-1] = 1.05
        diagnostic = compare(
            self.arguments["actual"]["out"], self.arguments["golden"]["out"],
            dtype="float16", policy=self.policy,
        )
        self.assertTrue(diagnostic["pass"])
        result = compare_case(**self.arguments)
        self.assertFalse(result["pass"])
        self.assertEqual("FAIL", result["status"])
        self.assertFalse(result["outputs"]["out"]["regions"]["tail"]["pass"])
        self.arguments["regions"]["out"] = {}
        incomplete = compare_case(**self.arguments)
        self.assertEqual("CONFIG_ERROR", incomplete["status"])
        self.assertFalse(incomplete["pass"])
        self.assertIn("tail", incomplete["failure_reason"])

    def test_float64_golden_error_survives_float32_rounding(self) -> None:
        result = compare_case(
            {"out": np.array([1048576.0], dtype=np.float32)},
            {"out": np.array([1048576.02], dtype=np.float64)},
            case="high_precision_golden", policy=self.policy,
            expected_outputs={
                "out": {"shape": [1], "dtype": "float32", "required_regions": []},
            },
            valid_masks={"out": np.array([True])}, regions={"out": {}},
        )
        self.assertFalse(result["pass"])
        self.assertAlmostEqual(0.02, result["outputs"]["out"]["global"]["max_abs"])

    def test_missing_extra_or_renamed_actual_output_fails(self) -> None:
        for actual in ({}, {"other": self.arguments["actual"]["out"]},
                       {**self.arguments["actual"], "extra": np.ones(1)}):
            with self.subTest(names=list(actual)):
                result = compare_case(**{**self.arguments, "actual": actual})
                self.assertEqual("FAIL", result["status"])
                self.assertFalse(result["pass"])

    def test_failure_in_second_output_fails_case(self) -> None:
        self.arguments["expected_outputs"]["state"] = {
            "shape": [1], "dtype": "float32", "required_regions": [],
        }
        self.arguments["actual"]["state"] = np.array([1.5], dtype=np.float32)
        self.arguments["golden"]["state"] = np.array([1.0], dtype=np.float64)
        self.arguments["valid_masks"]["state"] = np.array([True])
        self.arguments["regions"]["state"] = {}
        result = compare_case(**self.arguments)
        self.assertFalse(result["pass"])
        self.assertTrue(result["outputs"]["out"]["pass"])
        self.assertFalse(result["outputs"]["state"]["pass"])

    def test_raw_dtype_cannot_be_replaced_by_a_policy_label(self) -> None:
        for actual_dtype, declared in ((np.int32, "float16"), (np.float32, "float16"),
                                       (np.float32, "bfloat16")):
            with self.subTest(actual_dtype=actual_dtype, declared=declared):
                args = copy.deepcopy(self.arguments)
                args["actual"]["out"] = args["actual"]["out"].astype(actual_dtype)
                args["expected_outputs"]["out"]["dtype"] = declared
                result = compare_case(**args)
                self.assertEqual("FAIL", result["status"])
                self.assertEqual("dtype_mismatch", result["outputs"]["out"]["failure_reason"])
                self.assertFalse(result["pass"])

    def test_shape_is_checked_against_independent_expectation(self) -> None:
        self.arguments["actual"]["out"] = np.ones(10, dtype=np.float16)
        result = compare_case(**self.arguments)
        self.assertEqual("FAIL", result["status"])
        self.assertFalse(result["pass"])
        # Actual and golden agreeing with each other cannot override the plan.
        self.arguments["golden"]["out"] = np.ones(10, dtype=np.float64)
        self.assertFalse(compare_case(**self.arguments)["pass"])

    def test_missing_test_inputs_are_configuration_errors(self) -> None:
        for key in ("expected_outputs", "golden", "valid_masks", "regions"):
            with self.subTest(key=key):
                result = compare_case(**{**self.arguments, key: {}})
                self.assertEqual("CONFIG_ERROR", result["status"])
                self.assertFalse(result["pass"])
        args = copy.deepcopy(self.arguments)
        del args["expected_outputs"]["out"]["required_regions"]
        self.assertEqual("CONFIG_ERROR", compare_case(**args)["status"])
        args["golden"]["out"] = np.ones(1000, dtype=np.int32)
        args["expected_outputs"]["out"]["required_regions"] = ["tail"]
        self.assertEqual("CONFIG_ERROR", compare_case(**args)["status"])

    def test_invalid_masks_never_count_as_pass(self) -> None:
        for mask in (np.zeros(1000, dtype=bool), np.ones(999, dtype=bool),
                     np.ones(1000, dtype=np.int32)):
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                args = copy.deepcopy(self.arguments)
                args["valid_masks"]["out"] = mask
                result = compare_case(**args)
                self.assertEqual("CONFIG_ERROR", result["status"])
                self.assertFalse(result["pass"])
        args = copy.deepcopy(self.arguments)
        args["valid_masks"]["out"][-1] = False
        self.assertEqual("CONFIG_ERROR", compare_case(**args)["status"])
        args = copy.deepcopy(self.arguments)
        args["regions"]["out"]["tail"][:] = False
        self.assertEqual("CONFIG_ERROR", compare_case(**args)["status"])

    def test_padding_is_excluded_by_explicit_valid_mask(self) -> None:
        self.arguments["expected_outputs"]["out"]["required_regions"] = []
        self.arguments["regions"]["out"] = {}
        self.arguments["actual"]["out"][-1] = np.nan
        self.arguments["golden"]["out"][-1] = np.nan
        self.arguments["valid_masks"]["out"][-1] = False
        result = compare_case(**self.arguments)
        self.assertTrue(result["pass"])
        self.assertEqual(999, result["outputs"]["out"]["global"]["elements"])
        self.arguments["valid_masks"]["out"][-1] = True
        self.assertFalse(compare_case(**self.arguments)["pass"])

    def test_global_ratio_and_max_abs_must_both_pass(self) -> None:
        for indices, value in (([0], 1.25), (list(range(20)), 1.01)):
            with self.subTest(value=value):
                args = copy.deepcopy(self.arguments)
                args["actual"]["out"][indices] = value
                result = compare_case(**args)
                self.assertFalse(result["pass"])
                self.assertTrue(result["outputs"]["out"]["regions"]["tail"]["pass"])

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required for native bfloat16")
    def test_native_torch_bfloat16_keeps_original_dtype(self) -> None:
        import torch

        self.arguments["actual"]["out"] = torch.ones(1000, dtype=torch.bfloat16)
        self.arguments["golden"]["out"] = torch.ones(1000, dtype=torch.float64)
        self.arguments["expected_outputs"]["out"]["dtype"] = "bfloat16"
        result = compare_case(**self.arguments)
        self.assertTrue(result["pass"])
        self.assertEqual("bfloat16", result["outputs"]["out"]["actual_dtype"])
        self.arguments["actual"]["out"] = self.arguments["actual"]["out"].float().numpy()
        self.assertFalse(compare_case(**self.arguments)["pass"])


class PrecisionCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="precision case ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case_dir = self.root / "case data"
        self.case_dir.mkdir()
        self.other_cwd = self.root / "unrelated directory"
        self.other_cwd.mkdir()
        self.case_path = self.case_dir / "case.json"
        self.report_path = self.case_dir / "report.json"
        np.save(self.case_dir / "actual.npy", np.ones(1000, dtype=np.float16))
        np.save(self.case_dir / "golden.npy", np.ones(1000, dtype=np.float64))
        np.save(self.case_dir / "valid.npy", np.ones(1000, dtype=bool))
        tail = np.zeros(1000, dtype=bool)
        tail[-1] = True
        np.save(self.case_dir / "tail.npy", tail)
        self.payload = {
            "case": "tail_cli",
            "expected_outputs": {
                "out": {"shape": [1000], "dtype": "float16", "required_regions": ["tail"]},
            },
            "actual": {"out": "actual.npy"}, "golden": {"out": "golden.npy"},
            "valid_masks": {"out": "valid.npy"}, "regions": {"out": {"tail": "tail.npy"}},
        }

    def run_case(self, expected_exit: int) -> dict:
        self.case_path.write_text(json.dumps(self.payload), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-X", "utf8", str(SKILL_ROOT / "scripts/compare_precision.py"),
             "--case", str(self.case_path), "--policy", str(POLICY_PATH),
             "--report", str(self.report_path)],
            cwd=self.other_cwd, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(expected_exit, result.returncode, result.stdout + result.stderr)
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report, json.loads(result.stdout))
        return report

    def test_cli_resolves_arrays_relative_to_case_file_and_reports_failures(self) -> None:
        self.assertEqual("PASS", self.run_case(0)["status"])
        actual = np.ones(1000, dtype=np.float16)
        actual[-1] = 1.05
        np.save(self.case_dir / "actual.npy", actual)
        self.assertEqual("FAIL", self.run_case(1)["status"])
        self.payload["regions"]["out"] = {}
        self.assertEqual("CONFIG_ERROR", self.run_case(2)["status"])

    def test_cli_missing_mask_or_array_produces_incomplete_configuration_report(self) -> None:
        self.payload.pop("valid_masks")
        self.assertEqual("CONFIG_ERROR", self.run_case(2)["status"])
        self.payload["valid_masks"] = {"out": "missing.npy"}
        self.assertEqual("CONFIG_ERROR", self.run_case(2)["status"])
        (self.case_dir / "missing.npy").write_bytes(b"")
        self.assertEqual("CONFIG_ERROR", self.run_case(2)["status"])

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required for native bfloat16")
    def test_cli_loads_native_bfloat16_tensor_and_rejects_non_tensor_payload(self) -> None:
        import torch

        tensor_path = self.case_dir / "actual.pt"
        torch.save(torch.ones(1000, dtype=torch.bfloat16), tensor_path)
        self.payload["actual"]["out"] = tensor_path.name
        self.payload["expected_outputs"]["out"]["dtype"] = "bfloat16"
        self.assertEqual("PASS", self.run_case(0)["status"])
        torch.save({"out": torch.ones(1000)}, tensor_path)
        self.assertEqual("CONFIG_ERROR", self.run_case(2)["status"])


if __name__ == "__main__":
    unittest.main()
