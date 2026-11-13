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

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_skills import fixture, run_script, write_rows

ASSETS = Path(__file__).resolve().parents[1] / "assets"
sys.path.insert(0, str(ASSETS))
from case_contract import validate_assertions, validate_checks, check_inputs
import case_cache
import case_contract


class EvidenceChainTests(unittest.TestCase):
    def setup_case(self, work):
        _, rows = fixture(work)
        tdd = next(row for row in rows if row["sheet"] == "tdd")
        unit_path = work / "design/units/U01.yaml"
        return rows, tdd, unit_path, json.loads(unit_path.read_text())

    def run_target(self, work, code):
        prepared = run_script("prepare_golden.py", work)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        (work / "target_add.py").write_text(code)
        env = dict(os.environ, PYTHONPATH=str(work), CANNBOTDSL_OP_MODULE="target_add")
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

    def test_multiple_of_and_membership_share_prepare_and_runtime_semantics(self):
        for relation, value in (("multiple_of", 4), ("in", [4, 8])):
            with self.subTest(relation=relation), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                rows, tdd, path, native = self.setup_case(work)
                check = {"input": "x", "metric": "dimension", "axis": 1,
                         "relation": relation, "value": value}
                native["unit"]["verification_obligations"][0]["input_conditions"] = [check]
                path.write_text(json.dumps(native))
                tdd["assertions_json"] = json.dumps(
                    [{"kind": "input_conditions", "checks": [check]}])
                write_rows(work, rows)
                prepared = run_script("build_tdd_cases.py", work)
                self.assertEqual(prepared.returncode, 0, prepared.stderr)
                passed = self.run_target(work, "def run(x, y):\n    return x+y\n")
                self.assertEqual(passed.returncode, 0, passed.stderr)
                recipes = {name: {"shape": [2, 10], "dtype": "float32", "fill": "zeros"}
                           for name in ("x", "y")}
                tdd["inputs_json"] = json.dumps(recipes)
                write_rows(work, rows)
                rejected = run_script("build_tdd_cases.py", work)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("input condition not triggered", rejected.stderr)
                runtime = self.run_target(
                    work, 'def run(x, y):\n    raise RuntimeError("target must not run")\n')
                self.assertNotEqual(runtime.returncode, 0)
                result = (work / "operators/add/test/testcase_output/tdd/"
                          "verify_result_tdd.csv").read_text()
                self.assertIn("input condition not triggered", result)
                self.assertNotIn("target must not run", result)

    def test_predicate_configuration_rejects_invalid_divisors_and_lists(self):
        invalid = [("multiple_of", v) for v in (0, -1, True, 4.0, None, "4", [4])]
        invalid += [("in", v) for v in
                    ([], 8, "8", None, [True], ["8"], [[8]], [None],
                     [float("nan")], [float("inf")])]
        invalid += [("not_in", [8]), (["in"], [8])]
        for relation, value in invalid:
            with self.subTest(relation=relation, value=value):
                with self.assertRaises(ValueError):
                    validate_checks([{"input": "x", "metric": "numel",
                                      "relation": relation, "value": value}])
        validate_checks([{"input": "x", "metric": "numel",
                          "relation": "in", "value": [0, 8, 8.0]}])

    def test_new_predicates_preserve_zero_counts_and_large_integer_divisors(self):
        import torch
        for metric in ("dimension", "numel", "count", "unique_count"):
            for relation, value in (("multiple_of", 2 ** 2000), ("in", [0, 8])):
                with self.subTest(metric=metric, relation=relation):
                    check = {"input": "x", "metric": metric,
                             "relation": relation, "value": value}
                    if metric == "dimension":
                        check["axis"] = 1
                    records = check_inputs({"x": torch.empty(2, 0)}, [check])
                    self.assertEqual(records[0]["observed"], [0])
        for metric in ("count", "unique_count"):
            records = check_inputs({"x": torch.tensor([[1, 1, 2], [1, 2, 2]])},
                                   [{"input": "x", "metric": metric, "axis": 1,
                                     "relation": "in", "value": [2, 3]}])
            self.assertEqual(len(records[0]["observed"]), 2)

    def test_unit_preflight_needs_only_units_and_uses_the_formal_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _, _, _, native = self.setup_case(work)
            isolated = work / "isolated"
            units = isolated / "units"
            units.mkdir(parents=True)
            checks = [{"input": "x", "metric": "dimension", "axis": 1,
                       "relation": "multiple_of", "value": 4},
                      {"input": "x", "metric": "dimension", "axis": 1,
                       "relation": "in", "value": [4, 8]}]
            native["unit"]["verification_obligations"][0]["input_conditions"] = checks
            path = units / "U01.yaml"
            path.write_text(json.dumps(native))
            command = [sys.executable, str(ASSETS.parent / "scripts/build_tdd_cases.py"),
                       "--check-unit-contracts", "--units-dir", str(units)]
            snapshot = path.read_bytes()
            passed = subprocess.run(command, cwd=isolated, text=True, capture_output=True)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertEqual(path.read_bytes(), snapshot)
            self.assertEqual(sorted(p.relative_to(isolated).as_posix()
                                    for p in isolated.rglob("*")),
                             ["units", "units/U01.yaml"])
            for relation, value in (("multiple_of", 0), ("in", []), ("not_in", [8])):
                with self.subTest(relation=relation):
                    checks[0].update(relation=relation, value=value)
                    path.write_text(json.dumps(native))
                    preflight = subprocess.run(command, cwd=isolated, text=True,
                                               capture_output=True)
                    formal_native = json.loads(path.read_text())
                    (work / "design/units/U01.yaml").write_text(json.dumps(formal_native))
                    formal = run_script("build_tdd_cases.py", work)
                    self.assertNotEqual(preflight.returncode, 0)
                    self.assertNotEqual(formal.returncode, 0)
                    self.assertIn("U01/VO-U01", preflight.stderr)
                    self.assertIn("U01/VO-U01", formal.stderr)

    def test_new_predicate_must_be_bound_not_just_satisfied_by_shape(self):
        for relation, value in (("multiple_of", 4), ("in", [4, 8])):
            with self.subTest(relation=relation), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                _, _, path, native = self.setup_case(work)
                native["unit"]["verification_obligations"][0]["input_conditions"] = [
                    {"input": "x", "metric": "dimension", "axis": 1,
                     "relation": relation, "value": value}]
                path.write_text(json.dumps(native))
                failed = run_script("build_tdd_cases.py", work)
                self.assertNotEqual(failed.returncode, 0)
                self.assertIn("input conditions not bound", failed.stderr)

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

    def test_removed_probe_assertions_rejected(self):
        for kind in ("probe_events", "probe_predicates"):
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(ValueError, "review_evidence"),
            ):
                validate_assertions([{"kind": kind}])
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            rows, tdd, _, _ = self.setup_case(work)
            tdd["assertions_json"] = json.dumps(
                [{"kind": "probe_events", "events": ["add"]}]
            )
            write_rows(work, rows)
            rejected = run_script("build_tdd_cases.py", work)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("review_evidence", rejected.stderr)
            result = self.run_target(
                work, 'def run(x,y):\n    raise RuntimeError("must not run")\n'
            )
            self.assertNotEqual(result.returncode, 0)
            detail = (
                work / "operators/add/test/testcase_output/tdd/verify_result_tdd.csv"
            ).read_text()
            self.assertIn("review_evidence", detail)
            self.assertNotIn("must not run", detail)

    def test_input_unchanged_catches_mutation_even_with_correct_output(self):
        with tempfile.TemporaryDirectory(prefix="dsl input checks ") as tmp:
            work = Path(tmp)
            rows, tdd, _, _ = self.setup_case(work)
            tdd["assertions_json"] = json.dumps(
                [{"kind": "input_unchanged", "inputs": ["x", "y"]}]
            )
            write_rows(work, rows)
            checked = run_script("build_tdd_cases.py", work)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            good = self.run_target(work, "def run(x,y):\n    return x+y\n")
            self.assertEqual(good.returncode, 0, good.stderr)
            bad = self.run_target(
                work, "def run(x,y):\n    out=x+y\n    x.zero_()\n    return out\n"
            )
            self.assertNotEqual(bad.returncode, 0)
            detail = (
                work / "operators/add/test/testcase_output/tdd/verify_result_tdd.csv"
            ).read_text()
            self.assertIn("input_unchanged failed: x", detail)
            self.assertFalse((work / "operators/add/probe").exists())

    def test_input_unchanged_contract_rejects_bad_names(self):
        for names in ([], ["x", "x"], [1], "x", [""]):
            with (
                self.subTest(names=names),
                self.assertRaisesRegex(ValueError, "input_unchanged"),
            ):
                validate_assertions([{"kind": "input_unchanged", "inputs": names}])
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            rows, tdd, _, _ = self.setup_case(work)
            tdd["assertions_json"] = json.dumps(
                [{"kind": "input_unchanged", "inputs": ["missing"]}]
            )
            write_rows(work, rows)
            result = run_script("build_tdd_cases.py", work)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("input_unchanged", result.stderr)

    def test_input_unchanged_checks_converted_tensor_and_error_path(self):
        import case_contract
        import torch

        spec = importlib.util.spec_from_file_location(
            "runner_under_test", ASSETS / "test.py"
        )
        runner = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "cannbotdsl_case_contract": case_contract,
                "cannbotdsl_case_cache": case_cache,
            },
        ):
            spec.loader.exec_module(runner)

        original = {"x": torch.tensor([float("nan"), 0.0])}

        class ExpectedError(Exception):
            code = "invalid_input"

        def preserve(x):
            raise ExpectedError()

        def mutate(x):
            x[1] = -0.0
            raise ExpectedError()

        # Simulate a device transfer producing a separate Tensor: inspecting the
        # original host inputs would miss the mutation of the actual argument.
        with patch.object(
            runner,
            "_to_device",
            side_effect=lambda values, device: {
                key: value.clone() for key, value in values.items()
            },
        ):
            runner._invoke(
                runner.Invocation(preserve, None, "cpu"),
                original,
                {},
                expected_error_code="invalid_input",
                unchanged=["x"],
            )
            with self.assertRaisesRegex(AssertionError, "input_unchanged failed: x"):
                runner._invoke(
                    runner.Invocation(mutate, None, "cpu"),
                    original,
                    {},
                    expected_error_code="invalid_input",
                    unchanged=["x"],
                )
        self.assertFalse(torch.signbit(original["x"][1]).item())

    def test_input_unchanged_checks_aicpu_preprocessed_tensor(self):
        import case_contract
        import torch

        spec = importlib.util.spec_from_file_location(
            "runner_under_test", ASSETS / "test.py"
        )
        runner = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "cannbotdsl_case_contract": case_contract,
                "cannbotdsl_case_cache": case_cache,
            },
        ):
            spec.loader.exec_module(runner)

        original = {"x": torch.tensor([1.0])}

        def preserve(x):
            return x.clone()

        def mutate(x):
            x.add_(1)
            return x.clone()

        with (
            patch.object(
                runner, "_to_device", side_effect=lambda values, device: values
            ),
            patch.object(
                sys.modules[__name__],
                "_aicpu_preprocess",
                side_effect=lambda values: {"x": values["x"] + 1},
                create=True,
            ) as preprocess,
        ):
            for device, expected in (("cpu", 1.0), ("npu", 2.0)):
                with self.subTest(device=device):
                    result = runner._invoke(
                        runner.Invocation(preserve, None, device),
                        original,
                        {},
                        unchanged=["x"],
                    )
                    self.assertEqual(result.item(), expected)
            preprocess.assert_called_once()
            with self.assertRaisesRegex(AssertionError, "input_unchanged failed: x"):
                runner._invoke(
                    runner.Invocation(mutate, None, "npu"),
                    original,
                    {},
                    unchanged=["x"],
                )
        self.assertEqual(original["x"].item(), 1.0)

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


class FixedRunnerOutputTargetingTests(unittest.TestCase):
    """Tuple/dict operator outputs with optional index/output targeting."""

    @classmethod
    def setUpClass(cls):
        sys.modules.setdefault("cannbotdsl_case_cache", case_cache)
        sys.modules.setdefault("cannbotdsl_case_contract", case_contract)
        spec = importlib.util.spec_from_file_location(
            "fixed_runner_probe", ASSETS / "test.py"
        )
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_tuple_output_assertions(self):
        import torch

        outputs = (
            torch.zeros(2, 3),
            torch.ones(4, 5),
            torch.zeros(4, 5),
        )
        observe = self.runner._observe_assertion
        record = observe(
            outputs, {}, {"kind": "output_shape", "shape": [4, 5], "index": 1}, "cpu"
        )
        self.assertEqual(record["observed"], [4, 5])
        record = observe(
            outputs,
            {},
            {"kind": "output_dtype", "dtype": "float32", "index": 0},
            "cpu",
        )
        self.assertEqual(record["observed"], "float32")
        record = observe(outputs, {}, {"kind": "output_device", "index": 2}, "cpu")
        self.assertEqual(record["observed"], "cpu")
        record = observe(outputs, {}, {"kind": "output_device"}, "cpu")
        self.assertEqual(record["observed"], ["cpu", "cpu", "cpu"])
        source = torch.zeros(2, 3)
        record = observe(
            outputs,
            {"query": source},
            {"kind": "output_equals_input", "input": "query", "index": 0},
            "cpu",
        )
        self.assertTrue(record["observed"])

    def test_tuple_output_assertions_reject_mismatch(self):
        import torch

        outputs = (torch.zeros(2, 3), torch.ones(4, 5))
        observe = self.runner._observe_assertion
        with self.assertRaises(AssertionError):
            observe(
                outputs, {}, {"kind": "output_shape", "shape": [4, 5]}, "cpu"
            )  # untargeted applies to every output
        with self.assertRaises(AssertionError):
            observe(
                outputs, {}, {"kind": "output_shape", "shape": [4, 5], "index": 2}, "cpu"
            )
        with self.assertRaises(AssertionError):
            observe(
                outputs,
                {},
                {"kind": "output_dtype", "dtype": "float32", "index": "1"},
                "cpu",
            )

    def test_dict_output_and_single_tensor_passthrough(self):
        import torch

        outputs = {"dq": torch.zeros(2, 3), "dk": torch.ones(4, 5)}
        observe = self.runner._observe_assertion
        record = observe(
            outputs, {}, {"kind": "output_dtype", "dtype": "float32", "output": "dk"}, "cpu"
        )
        self.assertEqual(record["observed"], "float32")
        with self.assertRaises(AssertionError):
            observe(
                outputs,
                {},
                {"kind": "output_dtype", "dtype": "float32", "output": "dv"},
                "cpu",
            )
        record = observe(
            outputs["dq"], {}, {"kind": "output_shape", "shape": [2, 3]}, "cpu"
        )
        self.assertEqual(record["observed"], [2, 3])


class AssertionTargetingContractTests(unittest.TestCase):
    def test_index_and_output_fields_validated(self):
        validate_assertions(
            [
                {"kind": "output_shape", "shape": [1], "index": 0},
                {"kind": "output_dtype", "dtype": "float32", "output": "dq"},
                {"kind": "output_device"},
            ]
        )
        for bad in ({"index": -1}, {"index": "0"}, {"index": 1.5}, {"output": ""}, {"output": 3}):
            with self.assertRaises(ValueError):
                validate_assertions([dict({"kind": "output_device"}, **bad)])


class SpecListAttributeTests(unittest.TestCase):
    """Attributes typed list[...] in the spec accept JSON list values."""

    ATTRIBUTE_TYPES = {"actual_seq_qlen": "list[int64]", "head_num": "int"}

    def test_blackbox_accepts_spec_list_attributes(self):
        from build_blackbox_cases import validate_input_recipes

        validate_input_recipes(
            "c1",
            {"actual_seq_qlen": [256, 256], "head_num": 4},
            {},
            None,
            self.ATTRIBUTE_TYPES,
        )
        for bad in (
            {"head_num": [2]},
            {"actual_seq_qlen": {"0": 1}},
            {"actual_seq_qlen": [[1], 2]},
        ):
            with self.assertRaises(ValueError):
                validate_input_recipes("c1", bad, {}, None, self.ATTRIBUTE_TYPES)

    def test_tdd_accepts_spec_list_attributes(self):
        from build_tdd_cases import CaseContext, validate_execution_inputs

        context = CaseContext(
            {"actual_seq_qlen", "head_num"},
            {"actual_seq_qlen", "head_num"},
            {},
            None,
            {},
            self.ATTRIBUTE_TYPES,
        )
        validate_execution_inputs(
            ("U01", 0),
            {"inputs": {"actual_seq_qlen": [4, 4], "head_num": 2}},
            context,
        )
        for bad in ({"head_num": [2]}, {"actual_seq_qlen": [[1], 2]}):
            with self.assertRaises(ValueError):
                validate_execution_inputs(
                    ("U01", 0), {"inputs": bad}, context
                )

    def test_tdd_accepts_match_golden_nan(self):
        from build_tdd_cases import validate_obligation_evidence, validate_unit_evidence

        row = {
            "case_id": "TD-u01-nan",
            "inputs_json": "{}",
            "output_json": "{}",
            "assertions_json": "[]",
            "expected": "match_golden_nan",
        }
        inputs, outputs, assertions, expected = validate_unit_evidence(row, {}, [])
        self.assertEqual(expected, "match_golden_nan")
        validate_obligation_evidence(
            row, {"VO-1": {"kind": "numerical"}}, ["VO-1"], expected, assertions
        )
        with self.assertRaises(ValueError):
            validate_unit_evidence(
                dict(row, expected="golden_plus"), {}, []
            )


if __name__ == "__main__":
    unittest.main()
