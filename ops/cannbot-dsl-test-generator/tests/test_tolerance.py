#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Precision-policy boundaries and runner packaging."""

import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_tolerance import check_spec_tolerance, default_spec_tolerance
from test_skills import fixture, run_script


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTRACT = load_module("tolerance_contract", ROOT / "assets/case_contract.py")
CACHE = load_module("tolerance_cache", ROOT / "assets/case_cache.py")
with mock.patch.dict(sys.modules, {
    "cannbotdsl_case_contract": CONTRACT,
    "cannbotdsl_case_cache": CACHE,
}):
    RUNNER = load_module("tolerance_runner", ROOT / "assets/test.py")

LIMITS = {
    "metric": "mean_relative", "rtol": 0.125,
    "max_relative_rtol": 0.5, "epsilon": 1.0, "atol": 0.0,
}


class JointRelativeTests(unittest.TestCase):
    def compare(self, actual, expected, limits=None, equal_nan=False):
        dtype = str(expected.dtype).removeprefix("torch.")
        return RUNNER._compare(actual, expected, {dtype: limits or LIMITS}, equal_nan=equal_nan)

    def test_joint_pass_without_an_allclose_gate(self):
        expected = torch.zeros(100, dtype=torch.float64)
        actual = expected.clone()
        actual[0] = 0.25
        result = self.compare(actual, expected)[0]
        self.assertEqual(result["mere"], 0.0025)
        self.assertEqual(result["mare"], 0.25)
        self.assertTrue(result["passed"])

    def test_mean_pass_does_not_hide_max_failure(self):
        actual = torch.zeros(100, dtype=torch.float64)
        actual[0] = 1.0
        with self.assertRaises(AssertionError):
            self.compare(actual, torch.zeros_like(actual))

    def test_max_pass_does_not_hide_mean_failure(self):
        actual = torch.full((100,), 0.25, dtype=torch.float64)
        with self.assertRaises(AssertionError):
            self.compare(actual, torch.zeros_like(actual))

    def test_both_threshold_boundaries_are_strict(self):
        expected = torch.zeros(8, dtype=torch.float64)
        for actual in (torch.full_like(expected, 0.125), torch.tensor(
            [0.5, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64
        )):
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                self.compare(actual, expected)

    def test_nonfinite_masks_and_empty_outputs(self):
        for actual, expected in (([float("inf")], [float("-inf")]),
                                 ([0.0], [float("nan")]),
                                 ([float("inf")], [0.0])):
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                self.compare(torch.tensor(actual), torch.tensor(expected))
        matched = torch.tensor([float("inf"), float("-inf"), 0.0])
        self.assertTrue(self.compare(matched, matched)[0]["passed"])
        nan = torch.tensor([float("nan")])
        with self.assertRaises(AssertionError):
            self.compare(nan, nan)
        self.assertTrue(self.compare(nan, nan, equal_nan=True)[0]["passed"])
        self.assertTrue(self.compare(torch.empty(0), torch.empty(0))[0]["passed"])

    def test_legacy_comparators_preserve_behavior(self):
        zero = torch.tensor([0.0])
        self.compare(zero, zero, {"metric": "max_relative", "rtol": 0, "atol": 0})
        with self.assertRaises(AssertionError):
            self.compare(-zero, zero, {"metric": "bitwise_equal"})

    def test_invalid_policies_fail_without_inventing_thresholds(self):
        variants = [
            {**LIMITS, "metric": "unknown"}, {**LIMITS, "rtol": True},
            {**LIMITS, "rtol": float("nan")}, {**LIMITS, "rtol": float("inf")},
            {**LIMITS, "rtol": -1}, {**LIMITS, "rtol": 0},
            {**LIMITS, "epsilon": 0}, {**LIMITS, "atol": 0.01},
            {**LIMITS, "backend": "unknown"},
            {**LIMITS, "backend": "cann_bench"},
        ]
        missing = copy.deepcopy(LIMITS)
        del missing["max_relative_rtol"]
        variants.append(missing)
        for policy in variants:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                CONTRACT.validate_tolerance({"float32": policy})

    def test_unsupported_backend_never_falls_back(self):
        for backend in ("cann_bench", "unknown"):
            policy = {**LIMITS, "backend": backend}
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "unsupported backend"):
                self.compare(torch.ones(1), torch.ones(1), policy)

    def test_missing_output_dtype_is_reported(self):
        with self.assertRaisesRegex(ValueError, "missing output dtype"):
            check_spec_tolerance({
                "numerical_tolerance": {"per_dtype": {"float32": LIMITS}},
                "outputs": [{"dtype_set": ["float16"]}],
            })


class PackagingTests(unittest.TestCase):
    def test_preflight_needs_only_spec_and_does_not_rewrite_it(self):
        with tempfile.TemporaryDirectory(prefix="tolerance spec spaces ") as tmp:
            spec = Path(tmp) / "spec.yaml"
            policy = {"numerical_tolerance": {"per_dtype": {"float32": copy.deepcopy(LIMITS)}}}
            spec.write_text(yaml.safe_dump(policy))
            original = spec.read_bytes()
            command = [sys.executable, str(ROOT / "scripts/check_tolerance.py"), "--spec", str(spec)]
            result = subprocess.run(command, capture_output=True, text=True, cwd="/")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "pass")
            self.assertEqual(spec.read_bytes(), original)
            self.assertEqual(list(Path(tmp).iterdir()), [spec])
            policy["numerical_tolerance"]["per_dtype"]["float32"]["metric"] = "unknown"
            spec.write_text(yaml.safe_dump(policy))
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("unsupported metric", failed.stdout)

    def test_prepare_packages_joint_comparison_for_a_relocated_runner(self):
        with tempfile.TemporaryDirectory(prefix="joint runner spaces ") as tmp:
            work = Path(tmp)
            fixture(work)
            spec_path = work / "design/doc/spec.yaml"
            spec = yaml.safe_load(spec_path.read_text())
            limits = {"metric": "mean_relative", "rtol": 0.001,
                      "max_relative_rtol": 0.01, "atol": 0.0}
            spec["numerical_tolerance"]["per_dtype"] = {"float32": limits}
            spec_path.write_text(yaml.safe_dump(spec))
            test_dir = work / "operators/add/test"
            golden = test_dir / "add_golden.py"
            source = golden.read_text()
            line = next(line for line in source.splitlines() if line.startswith("TOLERANCE ="))
            golden.write_text(source.replace(line, f"TOLERANCE = { {'float32': limits}!r}"))
            result = run_script("prepare_golden.py", work)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            import shutil

            relocated = work / "relocated test"
            shutil.copytree(test_dir, relocated)
            (relocated / "target_add.py").write_text("def run(x, y):\n    return x + y\n")
            env = {**os.environ, "CANNBOTDSL_OP_MODULE": "target_add"}
            result = subprocess.run([
                sys.executable, str(relocated / "test_add.py"), "--sheet", "blackbox",
                "--case-id", "BB-core", "--device", "cpu",
            ], env=env, cwd="/", capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class StagedRelativeTests(unittest.TestCase):
    def policy(self, mode="native_or_truncated"):
        return {"metric": "staged_relative", "rtol": 2**-10,
                "max_relative_rtol": 10 * 2**-10,
                "fallback": {"reference_mode": mode}}

    def compare(self, a, g, native=None, mode="native_or_truncated", equal_nan=False):
        return RUNNER._compare(a, g, {"float16": self.policy(mode)},
                               equal_nan=equal_nan, native=native)

    def test_default_requires_native_but_explicit_legacy_modes_remain(self):
        policy = default_spec_tolerance({"outputs": [{"dtype_set": ["float16"]}]})
        limits = policy["per_dtype"]["float16"]
        self.assertEqual(CONTRACT.staged_policy("float16", limits)["reference_mode"], "native")
        implicit = dict(limits)
        implicit.pop("fallback")
        self.assertEqual(CONTRACT.staged_policy("float16", implicit)["reference_mode"], "native")
        for mode in ["truncated", "native_or_truncated"]:
            self.assertEqual(CONTRACT.staged_policy("float16", self.policy(mode))["reference_mode"], mode)
        with self.assertRaisesRegex(ValueError, "NATIVE_REFERENCE_FUNCTION"):
            RUNNER._compare(torch.zeros(3, dtype=torch.float16), torch.full((3,), .01, dtype=torch.float16),
                            {"float16": implicit})

    def test_default_preparation_and_cache_loading_reject_missing_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            fixture(work)
            spec_path = work / "design/doc/spec.yaml"
            spec = yaml.safe_load(spec_path.read_text())
            spec["numerical_tolerance"] = {"per_dtype": {"float32": {
                "metric": "staged_relative", "rtol": 2**-13, "max_relative_rtol": 10 * 2**-13}}}
            spec_path.write_text(yaml.safe_dump(spec))
            golden = work / "operators/add/test/add_golden.py"
            source = golden.read_text()
            old = next(line for line in source.splitlines() if line.startswith("TOLERANCE ="))
            golden.write_text(source.replace(old, f"TOLERANCE = {spec['numerical_tolerance']['per_dtype']!r}"))
            result = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_golden.py"),
                                    "--spec", str(spec_path), "--design", str(work / "design/doc/DESIGN.md"),
                                     "--test-dir", str(golden.parent)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("NATIVE_REFERENCE_FUNCTION", result.stdout + result.stderr)
            with self.assertRaisesRegex(ValueError, "NATIVE_REFERENCE_FUNCTION"):
                CACHE.load_golden(golden)

    def test_default_stage_two_is_not_a_blanket_relaxation(self):
        limits = default_spec_tolerance({"outputs": [{"dtype_set": ["float16"]}]})["per_dtype"]
        for value, actual in [(0., .0001), (.01, 0.), (1., 1.1)]:
            with self.subTest(value=value):
                g = torch.full((3,), value, dtype=torch.float16)
                a = torch.full_like(g, actual)
                with self.assertRaises(AssertionError):
                    RUNNER._compare(a, g, limits, native=lambda: g)
                result = RUNNER._compare(a, g, limits, native=lambda: a)[0]
                self.assertEqual(result["stage"], 2)
                self.assertEqual(result["reference"], "native")

    def test_stage1_short_circuits_native(self):
        x = torch.ones(4, dtype=torch.float16)
        native = mock.Mock(side_effect=RuntimeError("must not run"))
        self.assertEqual(self.compare(x, x, native)[0]["stage"], 1)
        native.assert_not_called()

    def test_normal_fallback_uses_same_precision_error_counts(self):
        g = torch.ones(8, dtype=torch.float16)
        a = g * 1.1
        result = self.compare(a, g, lambda: a)[0]
        self.assertEqual(result["stage"], 2)
        self.assertEqual(result["normal_error_count"], 8)
        with self.assertRaises(AssertionError):
            self.compare(a, g, lambda: g)
        with self.assertRaises(AssertionError):
            self.compare(a, g, lambda: a, mode="truncated")

    def test_small_cancel_counts_and_boundaries(self):
        for g, a in [(0., 0.0001), (0.01, 0.)]:
            with self.subTest(g=g):
                expected = torch.full((3,), g, dtype=torch.float16)
                actual = torch.full((3,), a, dtype=torch.float16)
                self.assertEqual(self.compare(actual[:2], expected[:2])[0]["stage"], 2)
                with self.assertRaises(AssertionError):
                    self.compare(actual, expected)

    def test_native_failures_do_not_fall_back(self):
        g = torch.ones(4, dtype=torch.float16)
        a = g * 2
        with self.assertRaises(ValueError):
            self.compare(a, g, mode="native")
        with self.assertRaises(RuntimeError):
            self.compare(a, g, mock.Mock(side_effect=RuntimeError("native failed")))
        with self.assertRaises(ValueError):
            self.compare(a, g, lambda: g.float())

    def test_native_nan_mismatch_fails_without_truncated_fallback(self):
        # Two small-value errors would pass against a clean reference. A broken
        # native reference must raise instead of being silently replaced by it.
        expected = torch.zeros(2, dtype=torch.float16)
        actual = torch.full_like(expected, .0001)
        for mode in ("native", "native_or_truncated"):
            with self.subTest(mode=mode):
                clean = mock.Mock(return_value=expected.clone())
                self.assertEqual(self.compare(actual, expected, clean, mode=mode)[0]["stage"], 2)
                clean.assert_called_once_with()
                broken = mock.Mock(return_value=torch.full_like(expected, float("nan")))
                with self.assertRaisesRegex(ValueError, "native reference NaN positions mismatch"):
                    self.compare(actual, expected, broken, mode=mode)
                broken.assert_called_once_with()

    def test_nonfinite_and_empty_outputs(self):
        inf = torch.tensor([float("inf")], dtype=torch.float16)
        finite = torch.tensor([65504.], dtype=torch.float16)
        self.assertEqual(self.compare(inf, finite)[0]["stage"], 1)
        self.assertEqual(self.compare(inf, inf)[0]["stage"], 1)
        with self.assertRaises(AssertionError):
            self.compare(inf, -inf)
        nan = torch.tensor([float("nan")], dtype=torch.float16)
        self.assertTrue(self.compare(nan, nan)[0]["passed"])
        self.assertTrue(self.compare(nan, nan, equal_nan=True)[0]["passed"])
        empty = torch.empty(0, dtype=torch.float16)
        self.assertTrue(self.compare(empty, empty)[0]["passed"])
        x = torch.ones(2, 3, dtype=torch.float16).t()
        self.assertTrue(self.compare(x, x.contiguous())[0]["passed"])
        with self.assertRaises(AssertionError):
            self.compare(x * 2, x.contiguous())

    def test_finite_overflows_are_not_dropped_as_empty_comparisons(self):
        maximum = torch.finfo(torch.float64).max
        policy = default_spec_tolerance({"outputs": [{"dtype_set": ["float64"]}]})["per_dtype"]
        policy["float64"]["fallback"] = {"reference_mode": "truncated"}
        for a, g in [([maximum] * 3, [0.] * 3), ([maximum], [-maximum]),
                     ([0., maximum], [0., -maximum])]:
            with self.subTest(a=a):
                actual, expected = (torch.tensor(v, dtype=torch.float64) for v in (a, g))
                with self.assertRaises(AssertionError):
                    RUNNER._compare(actual, expected, policy)
        a = torch.tensor([maximum], dtype=torch.float64)
        g = -a
        self.assertEqual(RUNNER._staged_relative_errors(a, g, 1e-7).item(), 2.)
        self.assertEqual(RUNNER._staged_relative_errors(a, g, maximum).item(), 1.)
        self.assertTrue(RUNNER._compare(a, a, policy)[0]["passed"])
        self.assertTrue(RUNNER._compare(torch.empty(0, dtype=torch.float64),
                                        torch.empty(0, dtype=torch.float64), policy)[0]["passed"])
        native = {"float64": {**policy["float64"], "fallback": {"reference_mode": "native"}}}
        self.assertEqual(RUNNER._compare(a, g, native, native=lambda: a)[0]["stage"], 2)

    def test_denominator_overflow_preserves_a_finite_difference(self):
        from decimal import Decimal, localcontext
        a = torch.tensor([torch.finfo(torch.float64).max], dtype=torch.float64)
        g = torch.nextafter(a, torch.zeros_like(a))
        epsilon = g.item()
        with localcontext() as context:
            context.prec = 100
            exact = float(abs(Decimal.from_float(a.item()) - Decimal.from_float(g.item())) /
                          (abs(Decimal.from_float(g.item())) + Decimal.from_float(epsilon)))
        for sign in [1, -1]:
            with self.subTest(sign=sign):
                got = RUNNER._staged_relative_errors(sign * a, sign * g, epsilon).item()
                self.assertAlmostEqual(got, exact, delta=exact * 1e-15)
                policy = {"float64": {"metric": "staged_relative", "rtol": 8e-17,
                                      "max_relative_rtol": 9e-17, "epsilon": epsilon}}
                self.assertEqual(RUNNER._compare(sign * a, sign * g, policy)[0]["stage"], 1)
                with self.assertRaises(AssertionError):
                    RUNNER._compare(
                        sign * a, sign * g,
                        {"float64": {**policy["float64"], "rtol": 4e-17, "max_relative_rtol": 4e-17}},
                        native=lambda: sign * g,
                    )

    def test_misplaced_staged_options_fail_preflight_and_comparison(self):
        for extra in [{"reference_mode": "native"}, {"error_ratio_limit": .5}, {"unknown": 1}]:
            with self.subTest(extra=extra):
                policy = {**self.policy(), **extra}
                with self.assertRaisesRegex(ValueError, "unsupported staged_relative fields"):
                    check_spec_tolerance({"numerical_tolerance": {"per_dtype": {"float16": policy}}})
                with self.assertRaisesRegex(ValueError, "unsupported staged_relative fields"):
                    RUNNER._compare(torch.ones(1, dtype=torch.float16),
                                    torch.ones(1, dtype=torch.float16), {"float16": policy})
        # Existing single-stage policy metadata is outside this tightening.
        legacy = {**LIMITS, "legacy_metadata": "kept"}
        self.assertTrue(RUNNER._compare(torch.ones(1), torch.ones(1), {"float32": legacy})[0]["passed"])
        correct = {"float16": {**self.policy(), "fallback": {"error_ratio_limit": .5}}}
        with self.assertRaises(AssertionError):
            RUNNER._compare(
                torch.tensor([.0001], dtype=torch.float16), torch.zeros(1, dtype=torch.float16), correct,
                native=lambda: torch.zeros(1, dtype=torch.float16),
            )

    def test_misplaced_policy_is_rejected_before_behavior_only_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            fixture(work)
            path = work / "design/doc/spec.yaml"
            spec = yaml.safe_load(path.read_text())
            policy = {"metric": "staged_relative", "rtol": .001, "max_relative_rtol": .01}
            spec["numerical_tolerance"]["per_dtype"] = {"float32": policy}
            path.write_text(yaml.safe_dump(spec))
            test = work / "operators/add/test"
            golden = test / "add_golden.py"
            original = golden.read_text()
            limits = next(line for line in original.splitlines() if line.startswith("TOLERANCE ="))
            source = original.replace(limits, f"TOLERANCE = {{'float32': {policy!r}}}")
            source += "\ndef native_reference(x, y):\n    return x + y\nNATIVE_REFERENCE_FUNCTION = native_reference\n"
            golden.write_text(source)
            prepared = run_script("prepare_golden.py", work)
            self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
            wrong = {**policy, "reference_mode": "native"}
            spec["numerical_tolerance"]["per_dtype"]["float32"] = wrong
            path.write_text(yaml.safe_dump(spec))
            prepared = run_script("prepare_golden.py", work)
            self.assertNotEqual(prepared.returncode, 0)
            self.assertIn("unsupported staged_relative fields", prepared.stdout + prepared.stderr)
            path_before = path.read_bytes()
            checked = subprocess.run([sys.executable, str(ROOT / "scripts/check_tolerance.py"),
                                      "--spec", str(path)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 1, checked.stdout + checked.stderr)
            self.assertEqual(json.loads(checked.stdout)["status"], "fail")
            self.assertEqual(path.read_bytes(), path_before)
            golden.write_text(
                source.replace(
                    f"TOLERANCE = {{'float32': {policy!r}}}",
                    f"TOLERANCE = {{'float32': {wrong!r}}}",
                )
            )
            (work / "candidate.py").write_text("def run(x,y):\n    return x+y\n")
            with (test / "testcase.csv").open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "sheet",
                        "case_id",
                        "status",
                        "inputs_json",
                        "expected",
                        "assertions_json"])
                writer.writeheader()
                writer.writerow(dict(sheet="tdd", case_id="behavior", status="active", expected="assertions",
                                     inputs_json=json.dumps(
                                         {name: {"shape": [2, 4], "dtype": "float32"} for name in ["x", "y"]}),
                                     assertions_json=json.dumps([{"kind": "output_shape", "shape": [2, 4]}])))
            env = dict(os.environ, CANNBOTDSL_OP_MODULE="candidate", PYTHONPATH=str(work))
            for selection in [["--sheet", "tdd"], ["--all"]]:
                result = subprocess.run([sys.executable, str(test / "test_add.py"), *selection, "--device", "cpu"],
                                        capture_output=True, text=True, env=env)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("unsupported staged_relative fields", result.stdout + result.stderr)
            self.assertFalse(list((test / "testcase_output").rglob("verify_result*.csv")))

    def test_noncontiguous_tensor_list_keeps_lazy_native_reference(self):
        g = torch.ones(3, 4, dtype=torch.float16).t()
        a = g * 1.1
        self.assertFalse(a.is_contiguous())
        callback = mock.Mock(return_value=[a.clone(), torch.empty(0, dtype=g.dtype)])
        policy = {"float16": self.policy("native")}
        records = RUNNER._compare([a, torch.empty(0, dtype=g.dtype)],
                                  [g, torch.empty(0, dtype=g.dtype)], policy, native=callback)
        self.assertTrue(all(record["passed"] for record in records))
        callback.assert_called_once()
        with self.assertRaises(AssertionError):
            RUNNER._compare([g * 2], [g], policy, native=lambda: [g])

    def test_preflight_rejects_partial_policy_and_bad_fallback(self):
        spec = {"numerical_tolerance": {"per_dtype": {"float16": self.policy()}}}
        check_spec_tolerance(spec)
        spec["numerical_tolerance"]["standard_stages"] = {"stage2": {}}
        with self.assertRaises(ValueError):
            check_spec_tolerance(spec)
        for value in [None, {"unknown": 1}, {"error_ratio_limit": 0}, {"reference_mode": "other"}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                CONTRACT.validate_tolerance({"float16": {**self.policy(), "fallback": value}})

    def test_native_uses_same_materialization_and_is_shared_by_outputs(self):
        from types import SimpleNamespace
        calls = []
        inputs = {"x": torch.ones(8, dtype=torch.float16), "scale": 3}

        def reference(x, scale):
            calls.append(scale)
            result = x * scale * 1.1
            x.zero_()
            return {"z": result, "other": result.clone()}
        golden = SimpleNamespace(INPUT_NAMES=["x"], ATTR_DEFAULTS={"scale": 2}, REQUIRED_ATTR_NAMES=[],
                                 NATIVE_REFERENCE_FUNCTION=reference, TOLERANCE={"float16": self.policy("native")})
        cache = SimpleNamespace(golden=golden, validate_output=lambda value, *_: value)
        expected = {name: inputs["x"] * 3 for name in ["z", "other"]}
        actual = {name: value * 1.1 for name, value in expected.items()}
        records = RUNNER._compare_outputs(actual, expected, cache, inputs, None)
        self.assertEqual(calls, [3])
        self.assertTrue(torch.equal(inputs["x"], torch.ones_like(inputs["x"])))
        self.assertTrue(all(record["stage"] == 2 for record in records))

    def test_missing_contract_selects_defaults_but_explicit_policy_is_preserved(self):
        spec = {"outputs": [{"dtype_set": ["float16", "bfloat16", "int32"]}]}
        selected = check_spec_tolerance(spec)["per_dtype"]
        self.assertEqual(selected["float16"]["metric"], "staged_relative")
        self.assertEqual(selected["float16"]["rtol"], 2**-10)
        self.assertEqual(selected["int32"]["metric"], "bitwise_equal")
        explicit = {"numerical_tolerance": {"per_dtype": {"float16": LIMITS}}}
        self.assertEqual(check_spec_tolerance(explicit)["per_dtype"]["float16"]["metric"], "mean_relative")
        with self.assertRaises(ValueError):
            check_spec_tolerance({"outputs": []})

    def test_default_proposal_does_not_replace_an_explicit_contract(self):
        spec = {"outputs": [{"dtype_set": ["float16", "int32"]}],
                "numerical_tolerance": {"source": "explicit", "per_dtype": {
                    "float16": {"metric": "max_relative", "rtol": .001, "atol": .001},
                    "int32": {"metric": "bitwise_equal"}}}}
        original = copy.deepcopy(spec)
        proposed = default_spec_tolerance(spec)
        self.assertEqual(proposed["mode"], "default_proposal")
        self.assertEqual(proposed["per_dtype"]["float16"]["metric"], "staged_relative")
        self.assertEqual(check_spec_tolerance(spec)["per_dtype"]["float16"]["metric"], "max_relative")
        self.assertEqual(spec, original)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.yaml"
            path.write_text(yaml.safe_dump(spec))
            before = path.read_bytes()
            result = subprocess.run([sys.executable, str(ROOT / "scripts/check_tolerance.py"),
                                     "--spec", str(path), "--defaults"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual(json.loads(result.stdout)["mode"], "default_proposal")
            self.assertEqual(path.read_bytes(), before)

    def test_default_policy_crosses_spec_validator_golden_and_official_runner(self):
        spec_skill = ROOT.parent / "ops-spec-gen"
        if not spec_skill.is_dir():
            self.skipTest("cross-skill regression requires the repository checkout")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            fixture(work)
            spec = yaml.safe_load((spec_skill / "examples/add/spec.yaml").read_text())
            # This is an explicit caller decision to use the DSL default policy;
            # initial registry values are not treated as a confirmed standard.
            proposed = default_spec_tolerance(spec)
            spec["numerical_tolerance"] = {key: proposed[key] for key in ["source", "per_dtype"]}
            spec_path = work / "design/doc/spec.yaml"
            spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
            checked = subprocess.run([sys.executable, str(spec_skill / "scripts/validate_spec.py"),
                                      str(spec_path), "--json"], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertEqual(check_spec_tolerance(spec)["per_dtype"], proposed["per_dtype"])
            test = work / "operators/add/test"
            golden = test / "add_golden.py"
            source = golden.read_text()
            old_limits = next(line for line in source.splitlines() if line.startswith("TOLERANCE ="))
            source = source.replace(old_limits, f"TOLERANCE = {proposed['per_dtype']!r}")
            source += "\ndef native_reference(x, y):\n    return x + y\nNATIVE_REFERENCE_FUNCTION = native_reference\n"
            golden.write_text(source)
            run_script("prepare_golden.py", work)
            with (test / "testcase.csv").open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "sheet",
                        "case_id",
                        "status",
                        "inputs_json",
                        "seed",
                        "expected",
                        "assertions_json"])
                writer.writeheader()
                for sheet in ["tdd", "whitebox", "blackbox", "performance"]:
                    writer.writerow(dict(
                        sheet=sheet, case_id=sheet + "-default", status="active", seed=42,
                        expected="match_golden", assertions_json="[]",
                        inputs_json=json.dumps({name: {"shape": [2, 4], "dtype": "float16"} for name in ["x", "y"]}),
                    ))
            (work / "candidate.py").write_text("def run(x,y):\n    return x+y\n")
            env = dict(os.environ, CANNBOTDSL_OP_MODULE="candidate", PYTHONPATH=str(work))
            for selection in [["--sheet", "tdd"], ["--all"]]:
                result = subprocess.run([sys.executable, str(test / "test_add.py"), *selection,
                                         "--device", "cpu"], capture_output=True, text=True, env=env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for sheet in ["tdd", "whitebox", "blackbox", "performance"]:
                files = list((test / "testcase_output" / sheet).glob("verify_result*.csv"))
                self.assertTrue(files)
                with files[0].open(encoding="utf-8-sig") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual([row["status"] for row in rows], ["passed"])
                self.assertIn("staged_relative", rows[0]["detail"])
            (work / "candidate.py").write_text("def run(x,y):\n    return (x+y)*2\n")
            result = subprocess.run([sys.executable, str(test / "test_add.py"), "--all", "--device", "cpu"],
                                    capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_formal_runner_csv_and_modes_use_final_decision(self):
        # Exercise the installed public CLI, not only the comparison helper.
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            fixture(work)
            spec_path = work / "design/doc/spec.yaml"
            spec = yaml.safe_load(spec_path.read_text())
            spec["numerical_tolerance"]["per_dtype"] = {"float16": self.policy("native")}
            spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
            for entry in spec["inputs"]:
                entry["dtype_set"] = ["float16"]
            spec["dtype_policy"]["supported_combinations"] = [
                {"inputs": {"x": "float16", "y": "float16"}, "outputs": {"z": "float16"}}]
            spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
            test = work / "operators/add/test"
            golden = test / "add_golden.py"
            source = golden.read_text()
            old_limits = next(line for line in source.splitlines() if line.startswith("TOLERANCE ="))
            source = source.replace(old_limits, f"TOLERANCE = {{'float16': {self.policy('native')!r}}}")
            source += (
                "\n_ORIGINAL = GOLDEN_FUNCTION\ndef native_reference(**kwargs):\n"
                "    return _ORIGINAL(**kwargs) * 1.1\nNATIVE_REFERENCE_FUNCTION = native_reference\n"
            )
            golden.write_text(source)
            run_script("prepare_golden.py", work)
            with (test / "testcase.csv").open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "sheet",
                        "case_id",
                        "status",
                        "inputs_json",
                        "seed",
                        "expected",
                        "assertions_json"])
                writer.writeheader()
                for sheet in ["tdd", "whitebox", "blackbox", "performance"]:
                    writer.writerow(dict(
                        sheet=sheet, case_id="c1", status="active", seed=42,
                        expected="match_golden", assertions_json="[]",
                        inputs_json=json.dumps({name: {"shape": [8], "dtype": "float16"} for name in ["x", "y"]}),
                    ))
            module = work / "candidate.py"
            module.write_text("def run(x, y):\n    return (x+y) * 1.1\n")
            env = dict(os.environ, CANNBOTDSL_OP_MODULE="candidate", PYTHONPATH=str(work))
            for selection in [["--sheet", "tdd"], ["--all"], ["--sheet", "tdd",
                                                              "--case-id", "c1"], ["--sheet", "tdd", "--grep", "^c1$"]]:
                result = subprocess.run([sys.executable, str(test / "test_add.py"), *selection,
                                        "--device", "cpu"], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                files = list((test / "testcase_output/tdd").glob("verify_result*.csv"))
                self.assertTrue(any('stage' in f.read_text() and 'passed' in f.read_text() for f in files))
            for sheet in ["tdd", "whitebox", "blackbox", "performance"]:
                files = list((test / "testcase_output" / sheet).glob("verify_result*.csv"))
                self.assertTrue(files, sheet)
                with files[-1].open(encoding="utf-8-sig") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(rows[0]["status"], "passed")
                self.assertIn('"stage": 2', rows[0]["detail"])
            golden.write_text(
                golden.read_text().replace(
                    "return _ORIGINAL(**kwargs) * 1.1",
                    "return _ORIGINAL(**kwargs)"))
            result = subprocess.run([sys.executable, str(test / "test_add.py"), "--all",
                                    "--device", "cpu"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
