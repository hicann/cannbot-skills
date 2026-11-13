#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Independent primary references, declared rounding and cache identity."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
import yaml

from test_tolerance import CACHE, RUNNER
from test_skills import SUPPORT, fixture, run_script, write_rows


class ReferencePolicyTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.calls = 0

        def primary(x):
            self.calls += 1
            return (x + torch.ones_like(x)) - x

        def make_inputs(recipe, seed):
            return {"x": torch.tensor([2048.0], dtype=torch.float16)}

        def simulate(recipe, seed):
            inputs = make_inputs(recipe, seed)
            return inputs, primary(**inputs)
        self.golden = types.SimpleNamespace(
            __cannbotdsl_source_hash__="same-precision",
            INPUT_NAMES=["x"], OUTPUT_NAMES=["y"],
            ATTR_DEFAULTS={}, REQUIRED_ATTR_NAMES=[],
            GOLDEN_FUNCTION=primary, make_inputs=make_inputs, simulate=simulate,
        )
        self.row = {"sheet": "tdd", "case_id": "T1", "expected": "match_golden", "output_json": ""}
        self.call = {"inputs": {"x": {"shape": [1], "dtype": "float16"}}, "seed": 42, "assertions": []}

    def test_declared_same_precision_path_is_the_primary_reference(self):
        cache = CACHE.CaseCache(self.root, self.golden)
        data = cache.get(self.row, self.call, 0)
        self.assertEqual(data["expected"].item(), 0)
        self.assertEqual(data["expected"].dtype, torch.float16)
        high = ((data["inputs"]["x"].double() + 1) - data["inputs"]["x"].double()).half()
        self.assertEqual(high.item(), 1)
        limits = {"float16": {"metric": "bitwise_equal"}}
        self.assertTrue(RUNNER._compare(data["expected"], data["expected"], limits)[0]["equal"])
        with self.assertRaises(AssertionError):
            RUNNER._compare(data["expected"], high, limits)
        again = cache.get(self.row, self.call, 0)
        self.assertTrue(torch.equal(again["expected"], data["expected"]))
        self.assertEqual((self.calls, cache.built, cache.hits), (1, 1, 1))

    def test_reference_source_change_invalidates_cache(self):
        first = CACHE.CaseCache(self.root, self.golden).get(self.row, self.call, 0)
        self.golden.__cannbotdsl_source_hash__ = "high-precision"

        def primary(x):
            x = x.double()
            return ((x + 1) - x).half()
        self.golden.GOLDEN_FUNCTION = primary
        second = CACHE.CaseCache(self.root, self.golden).get(self.row, self.call, 0)
        self.assertNotEqual(first["key"], second["key"])
        self.assertEqual(second["expected"].item(), 1)

    def test_legacy_or_missing_expected_cache_is_rebuilt(self):
        cache = CACHE.CaseCache(self.root, self.golden)
        data = cache.get(self.row, self.call, 0)
        path = self.root / ".case_cache" / f"{data['key']}.pt"
        for version in (2, 3, None):
            changed = copy.deepcopy(data)
            if version is not None:
                changed["version"] = version
            else:
                changed.pop("expected")
            torch.save(changed, path)
            fresh = cache.get(self.row, self.call, 0)
            self.assertEqual(fresh["version"], CACHE.CACHE_VERSION)
            self.assertEqual(fresh["expected"].item(), 0)
        self.assertEqual(self.calls, 4)

    def test_v3_cache_is_not_reused_with_unchanged_golden_source(self):
        with mock.patch.object(CACHE, "CACHE_VERSION", 3):
            old_cache = CACHE.CaseCache(self.root, self.golden)
            old = old_cache.get(self.row, self.call, 0)
            old["expected"] = torch.ones_like(old["expected"])
            torch.save(old, old_cache.root / f"{old['key']}.pt")
        cache = CACHE.CaseCache(self.root, self.golden)
        fresh = cache.get(self.row, self.call, 0)
        self.assertNotEqual(old["key"], fresh["key"])
        self.assertEqual(fresh["expected"].item(), 0)
        self.assertEqual((cache.built, cache.hits), (1, 0))

    def test_reference_and_target_use_one_materialization(self):
        materializations = []
        observed = []

        def make_inputs(recipe, seed):
            value = torch.tensor([float(len(materializations) + 1)])
            materializations.append(value.clone())
            return {"x": value}

        def primary(x):
            observed.append(x.clone())
            return x + 1

        self.golden.make_inputs = make_inputs
        self.golden.GOLDEN_FUNCTION = primary
        self.golden.simulate = mock.Mock(side_effect=AssertionError("must not simulate"))
        self.golden.TOLERANCE = {"float32": {"metric": "bitwise_equal"}}
        row = {**self.row, "inputs_json": '{"x": {"shape": [1], "dtype": "float32"}}',
               "assertions_json": "[]", "seed": "42"}

        def target(x):
            observed.append(x.clone())
            return x + 1

        cache = CACHE.CaseCache(self.root, self.golden)
        invocation = RUNNER.Invocation(target, self.golden, "cpu")
        for _ in range(2):
            result = RUNNER._case_result(row, "cpu", invocation, cache=cache)
            self.assertEqual(result["status"], "passed", result["detail"])
        self.assertEqual(len(materializations), 1)
        self.assertEqual(len(observed), 3)  # One primary call and two target calls.
        for value in observed:
            torch.testing.assert_close(value, materializations[0])
        self.golden.simulate.assert_not_called()

    def test_reference_mutations_do_not_change_cached_inputs_or_defaults(self):
        self.golden.ATTR_DEFAULTS = {"offsets": [1]}

        def make_inputs(recipe, seed):
            tensor = torch.tensor([1.0])
            return {"x": [tensor, tensor]}

        def primary(x, offsets):
            self.assertIs(x[0], x[1])
            x[0].add_(offsets[0])
            offsets.append(99)
            return x[1]

        self.golden.make_inputs = make_inputs
        self.golden.GOLDEN_FUNCTION = primary
        data = CACHE.CaseCache(self.root, self.golden).get(self.row, self.call, 0)
        self.assertEqual(data["expected"].item(), 2)
        self.assertEqual(data["inputs"]["x"][0].item(), 1)
        self.assertIs(data["inputs"]["x"][0], data["inputs"]["x"][1])
        self.assertEqual(self.golden.ATTR_DEFAULTS, {"offsets": [1]})

    def test_primary_receives_default_required_and_optional_arguments(self):
        self.golden.INPUT_NAMES = ["x", "optional"]
        self.golden.ATTR_DEFAULTS = {"scale": 2}
        self.golden.REQUIRED_ATTR_NAMES = ["shift"]

        def primary(x, shift, scale=2, optional=None):
            return x * scale + shift + (0 if optional is None else optional)

        self.golden.GOLDEN_FUNCTION = primary
        for extra, expected in (({}, 5), ({"scale": 4}, 7),
                                ({"optional": torch.tensor([2.0])}, 7)):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                self.golden.make_inputs = lambda recipe, seed: {
                    "x": torch.tensor([1.0]), "shift": 3, **extra
                }
                data = CACHE.CaseCache(tmp, self.golden).get(self.row, self.call, 0)
                self.assertEqual(data["expected"].item(), expected)

    def test_missing_primary_or_required_argument_fails_without_fallback(self):
        self.golden.simulate = mock.Mock(return_value=({}, torch.zeros(1)))
        self.golden.GOLDEN_FUNCTION = None
        with self.assertRaisesRegex(ValueError, "GOLDEN_FUNCTION"):
            CACHE.CaseCache(self.root, self.golden).get(self.row, self.call, 0)
        self.golden.GOLDEN_FUNCTION = lambda x, required: x
        with self.assertRaises(TypeError):
            CACHE.CaseCache(self.root, self.golden).get(self.row, self.call, 0)
        self.golden.simulate.assert_not_called()
        self.assertFalse(list((self.root / ".case_cache").glob("*.pt")))

    def test_behavior_case_does_not_compute_reference(self):
        self.row["expected"] = "assertions"
        data = CACHE.CaseCache(self.root, self.golden).get(self.row, self.call, 0)
        self.assertIsNone(data["expected"])
        self.assertEqual(self.calls, 0)

    def test_nested_outputs_use_each_declared_metric(self):
        value = torch.ones(2, dtype=torch.float16)
        expected = {"pair": (value, value)}
        self.assertEqual(len(RUNNER._compare(expected, expected, {"float16": {"metric": "bitwise_equal"}})), 2)
        with self.assertRaises(AssertionError):
            RUNNER._compare({"pair": (value,)}, expected, {"float16": {"metric": "bitwise_equal"}})


class ReferencePackagingTests(unittest.TestCase):
    def test_missing_reference_output_fails_preparation_and_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            fixture(work)
            test_dir = work / "operators/add/test"
            path = test_dir / "add_golden.py"
            path.write_text(path.read_text().replace("return torch.add(x, y)", "return ()"))
            prepared = run_script("prepare_golden.py", work)
            self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
            built = run_script("build_tdd_cases.py", work)
            self.assertNotEqual(built.returncode, 0, built.stdout + built.stderr)
            self.assertIn("output.z", built.stdout + built.stderr)
            (work / "missing_target.py").write_text("def run(x, y):\n    return ()\n")
            result = subprocess.run([
                sys.executable, str(test_dir / "test_add.py"), "--sheet", "blackbox",
                "--case-id", "BB-core", "--device", "cpu",
            ], env={**os.environ, "PYTHONPATH": str(work), "CANNBOTDSL_OP_MODULE": "missing_target"},
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            with (test_dir / "testcase_output/blackbox/verify_result_blackbox_BB-core.csv").open(
                    encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "failed")
            self.assertIn("output.z", row["detail"])

    def test_packaged_empty_tensor_has_a_comparison_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _, rows = fixture(work)
            row = next(row for row in rows if row["case_id"] == "BB-core")
            row["inputs_json"] = '{"x":{"shape":[0,3],"dtype":"float32"},"y":{"shape":[0,3],"dtype":"float32"}}'
            write_rows(work, rows)
            prepared = run_script("prepare_golden.py", work)
            self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
            built = run_script("build_tdd_cases.py", work)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            test_dir = work / "operators/add/test"
            (work / "empty_tensor_target.py").write_text("def run(x, y):\n    return x+y\n")
            result = subprocess.run([
                sys.executable, str(test_dir / "test_add.py"), "--sheet", "blackbox",
                "--case-id", "BB-core", "--device", "cpu",
            ], env={**os.environ, "PYTHONPATH": str(work), "CANNBOTDSL_OP_MODULE": "empty_tensor_target"},
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (test_dir / "testcase_output/blackbox/verify_result_blackbox_BB-core.csv").open(
                    encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            import json
            records = json.loads(row["detail"])[0]["records"]
            self.assertEqual(row["status"], "passed")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["dtype"], "float32")

    def test_packaged_dtype_contract_and_multiple_output_count_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            fixture(work)
            test_dir = work / "operators/add/test"
            path = test_dir / "add_golden.py"
            original = path.read_text()
            # The comparator supports fp16, but these fp32 inputs require fp32 output.
            policy = {"float32": {"metric": "max_relative", "rtol": 1e-6, "atol": 1e-6},
                      "float16": {"metric": "bitwise_equal"}}
            spec_path = work / "design/doc/spec.yaml"
            spec = yaml.safe_load(spec_path.read_text())
            spec["numerical_tolerance"]["per_dtype"] = policy
            spec_path.write_text(yaml.safe_dump(spec))
            line = next(line for line in original.splitlines() if line.startswith("TOLERANCE ="))
            path.write_text(original.replace(line, f"TOLERANCE = {policy!r}").replace(
                "return torch.add(x, y)", "return torch.add(x, y).half()"))
            prepared = run_script("prepare_golden.py", work)
            self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
            built = run_script("build_tdd_cases.py", work)
            self.assertNotEqual(built.returncode, 0)
            self.assertIn("dtype combination", built.stdout + built.stderr)
            spec["outputs"].append({"name": "extra"})
            spec["dtype_policy"]["supported_combinations"][0]["outputs"]["extra"] = "float32"
            spec_path.write_text(yaml.safe_dump(spec))
            path.write_text(original.replace(line, f"TOLERANCE = {policy!r}").replace(
                "OUTPUT_NAMES = ['z']", "OUTPUT_NAMES = ['z', 'extra']"))
            prepared = run_script("prepare_golden.py", work)
            self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
            built = run_script("build_tdd_cases.py", work)
            self.assertNotEqual(built.returncode, 0)
            self.assertIn("2 ordered outputs", built.stdout + built.stderr)

    def test_divergent_simulate_cannot_make_zero_implementation_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            fixture(work)
            test_dir = work / "operators/add/test"
            golden = test_dir / "add_golden.py"
            original = golden.read_text()
            self.assertEqual(original.count("return inputs, GOLDEN_FUNCTION(**kwargs)"), 1)
            golden.write_text(original.replace(
                "return inputs, GOLDEN_FUNCTION(**kwargs)",
                'return inputs, torch.zeros_like(inputs["x"])',
            ))
            prepared = run_script("prepare_golden.py", work)
            self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
            target = work / "binding_target.py"
            env = {**os.environ, "PYTHONPATH": str(work), "CANNBOTDSL_OP_MODULE": "binding_target"}
            command = [sys.executable, str(test_dir / "test_add.py"),
                       "--sheet", "blackbox", "--case-id", "BB-core", "--device", "cpu"]
            result_path = test_dir / "testcase_output/blackbox/verify_result_blackbox_BB-core.csv"
            for body, code, status in (("torch.zeros_like(x)", 1, "failed"),
                                       ("torch.add(x, y)", 0, "passed")):
                with self.subTest(target=body):
                    target.write_text("import torch\ndef run(x, y):\n    return " + body + "\n")
                    result = subprocess.run(command, env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                    with result_path.open(encoding="utf-8-sig", newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]["status"], status)

    def test_prepare_accepts_either_independent_primary_precision(self):
        for body in ("torch.add(x, y)", "(x.double() + y.double()).to(x.dtype)"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                fixture(work)
                path = work / "operators/add/test/add_golden.py"
                path.write_text(path.read_text().replace("return torch.add(x, y)", "return " + body))
                result = run_script("prepare_golden.py", work)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class OutputContractTests(unittest.TestCase):
    def validate(self, value, outputs, materialized_inputs=None, **sections):
        spec = CACHE._output_spec({"outputs": outputs, **sections})
        return CACHE.validate_outputs(value, [item["name"] for item in outputs], materialized_inputs or {}, spec)

    def test_empty_containers_do_not_count_as_numerical_comparisons(self):
        for value in ((), [], {}, {"nested": ()}, (torch.ones(1), [])):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                RUNNER._compare(value, value, {"float32": {"metric": "bitwise_equal"}})
        for limits in ({"metric": "bitwise_equal"},
                       {"metric": "max_relative", "rtol": 0, "atol": 0},
                       {"metric": "mean_relative", "rtol": 0.1, "max_relative_rtol": 0.1}):
            with self.subTest(limits=limits):
                self.assertEqual(len(RUNNER._compare(torch.empty(0, 3), torch.empty(0, 3),
                                                     {"float32": limits})), 1)

    def test_required_outputs_cannot_be_omitted_or_replaced_by_containers(self):
        tensor = torch.ones(1)
        outputs = [{"name": "a"}, {"name": "b"}]
        for value in ((), (tensor,), tensor, {"a": tensor}, {"a": tensor, "b": None},
                      {"a": tensor, "b": []}, {"a": tensor, "wrong": tensor}, (tensor, 3)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(value, outputs)
        self.assertEqual(set(self.validate((tensor, tensor), outputs)), {"a", "b"})
        self.assertEqual(set(self.validate({"b": tensor, "a": tensor}, outputs)), {"a", "b"})
        for value in ((), [], {}, None):
            with self.subTest(single=value), self.assertRaises(ValueError):
                self.validate(value, [{"name": "a"}])

    def test_outputs_must_match_one_complete_dtype_combination(self):
        outputs = [{"name": "a"}, {"name": "b"}]
        policy = {"supported_combinations": [
            {"inputs": {"x": "float32"}, "outputs": {"a": "float32", "b": "int64"}},
            {"inputs": {"x": "float32"}, "outputs": {"a": "float16", "b": "float32"}},
        ]}
        inputs = {"x": torch.ones(1)}
        self.validate((torch.ones(1), torch.ones(1, dtype=torch.int64)), outputs,
                      inputs, dtype_policy=policy)
        with self.assertRaisesRegex(ValueError, "dtype combination"):
            self.validate((torch.ones(1), torch.ones(1)), outputs, inputs, dtype_policy=policy)

    def test_explicit_output_rank_and_buffer_shape_dtype_are_checked(self):
        spec = CACHE._output_spec({"outputs": [{"name": "y", "shape_bounds": {"output_rank": 2}}]})
        with self.assertRaisesRegex(ValueError, "output_rank"):
            CACHE.validate_outputs(torch.ones(2), ["y"], {}, spec)
        recipe = {"y": {"shape": [0, 3], "dtype": "float32"}}
        CACHE.validate_outputs(torch.empty(0, 3), ["y"], {}, spec, recipe)
        for value in (torch.empty(0, 4), torch.empty(0, 3, dtype=torch.float16)):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "buffer shape/dtype"):
                CACHE.validate_outputs(value, ["y"], {}, spec, recipe)

    def test_empty_tensor_list_requires_declared_role_and_valid_length(self):
        for length in ({"kind": "fixed", "value": 0}, {"kind": "range", "min": 0, "max": 2},
                       {"kind": "unconstrained"}, {"kind": "same_as", "ref": "input.xs"},
                       {"kind": "expression", "value": "len(attr.parts) + shape.x[0] * 0"}):
            with self.subTest(length=length):
                self.validate([], [{"name": "ys", "role": "tensor_list", "list_length": length}],
                              {"xs": [], "x": torch.ones(2)},
                              attributes=[{"name": "parts", "default": []}])
        for length in ({"kind": "fixed", "value": 1}, {"kind": "range", "min": 1, "max": 2},
                       {"kind": "expression", "value": "__import__(\"os\")"}):
            with self.subTest(length=length), self.assertRaises(ValueError):
                self.validate([], [{"name": "ys", "role": "tensor_list", "list_length": length}])

    def test_optional_presence_obeys_existing_semantic_cases(self):
        outputs = [{"name": "y", "optional": True}]
        sections = {"attributes": [{"name": "need", "default": True}], "semantic_cases": [
            {"when": "attr.need == true", "outputs": {"present": ["y"]}},
            {"when": "not attr.need", "outputs": {"absent": ["y"]}},
            {"when": "input.x.is_present and input.x.dtype == \"float16\"", "outputs": {"present": ["y"]}},
        ]}
        with self.assertRaisesRegex(ValueError, "missing required"):
            self.validate(None, outputs, **sections)
        self.validate({}, outputs, {"need": False, "x": None}, **sections)
        with self.assertRaisesRegex(ValueError, "must be absent"):
            self.validate(torch.ones(1), outputs, {"need": False, "x": None}, **sections)

    def test_authorized_absent_or_empty_list_has_explicit_records(self):
        golden = types.SimpleNamespace(OUTPUT_NAMES=["y"], __cannbotdsl_source_hash__="outputs",
                                       TOLERANCE={"float32": {"metric": "bitwise_equal"}})
        for value, output, metric in ((None, {"name": "y", "optional": True}, "output_presence"),
                                      ([], {"name": "y", "role": "tensor_list", "list_length": {
                                          "kind": "fixed", "value": 0}}, "output_list_length")):
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as tmp:
                cache = CACHE.CaseCache(tmp, golden, spec={"outputs": [output]})
                records = RUNNER._compare_outputs(value, value, cache, {}, {})
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["metric"], metric)

    def test_invalid_cached_output_is_rebuilt_and_revalidated(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary = mock.Mock(side_effect=lambda x: x + 1)
            golden = types.SimpleNamespace(
                OUTPUT_NAMES=["y"], INPUT_NAMES=["x"],
                ATTR_DEFAULTS={}, REQUIRED_ATTR_NAMES=[], GOLDEN_FUNCTION=primary,
                make_inputs=lambda recipe, seed: {"x": torch.ones(1)}, __cannbotdsl_source_hash__="cache",
            )
            row = {"sheet": "tdd", "case_id": "T", "expected": "match_golden"}
            call = {"inputs": {}, "seed": 42, "assertions": []}
            cache = CACHE.CaseCache(tmp, golden)
            data = cache.get(row, call, 0)
            data["expected"] = ()
            path = cache.root / f"{data['key']}.pt"
            torch.save(data, path)
            self.assertEqual(cache.get(row, call, 0)["expected"].item(), 2)
            self.assertEqual(primary.call_count, 2)
            torch.save(data, path)
            primary.side_effect = lambda x: ()
            with self.assertRaisesRegex(ValueError, "expected Tensor"):
                cache.get(row, call, 0)

    def test_output_contract_changes_invalidate_cache_without_new_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            golden = types.SimpleNamespace(
                OUTPUT_NAMES=["y"], INPUT_NAMES=["x"],
                ATTR_DEFAULTS={}, REQUIRED_ATTR_NAMES=[], GOLDEN_FUNCTION=lambda x: x,
                make_inputs=lambda recipe, seed: {"x": torch.ones(1)}, __cannbotdsl_source_hash__="same",
            )
            row = {"sheet": "tdd", "case_id": "T", "expected": "match_golden"}
            call = {"inputs": {}, "seed": 42, "assertions": []}
            first = CACHE.CaseCache(tmp, golden).get(row, call, 0)
            second = CACHE.CaseCache(tmp, golden, spec={"outputs": [
                {"name": "y", "dtype_set": ["float32"]}]}).get(row, call, 0)
            self.assertNotEqual(first["key"], second["key"])
            self.assertEqual(first["version"], second["version"])

    def test_output_checks_skip_conditions_with_no_output_obligations(self):
        self.validate(torch.ones(1), [{"name": "y"}], {"xs": []}, semantic_cases=[{
            "when": 'input.xs.dtype == "float32"', "inputs": {"required": ["xs"]}, "outputs": {},
        }])

    def test_output_conditions_resolve_tensorlist_dtype_without_guessing(self):
        sections = {"inputs": [{"name": "xs", "role": "tensor_list", "dtype_set": ["float32", "float16"]}],
                    "semantic_cases": [
                        {"when": 'input.xs.dtype == "float32"', "outputs": {"present": ["y"]}},
                        {"when": 'input.xs.dtype == "float16"', "outputs": {"absent": ["y"]}},
        ]}
        outputs = [{"name": "y", "optional": True}]
        self.validate(torch.ones(1), outputs, {"xs": [torch.ones(1)]}, **sections)
        with self.assertRaisesRegex(ValueError, "missing required"):
            self.validate(None, outputs, {"xs": [torch.ones(1)]}, **sections)
        self.validate(None, outputs, {"xs": [torch.ones(1, dtype=torch.float16)]}, **sections)
        sections["inputs"][0]["dtype_set"] = ["float32"]
        self.validate(torch.ones(1), outputs, {"xs": []}, **sections)
        with self.assertRaisesRegex(ValueError, "missing required"):
            self.validate(None, outputs, {"xs": []}, **sections)
        sections["inputs"][0]["dtype_set"] = ["float32", "float16"]
        for members in ([], [torch.ones(1), torch.ones(1, dtype=torch.float16)]):
            with self.subTest(ambiguous=members), self.assertRaisesRegex(ValueError, "cannot determine one dtype"):
                self.validate(None, outputs, {"xs": members}, **sections)

    def test_optional_none_snapshot_preserves_presence(self):
        before = RUNNER._clone_cpu({"a": torch.empty(0, 3), "aux": None})
        RUNNER._bitwise_equal(RUNNER._clone_cpu(before), before)
        with self.assertRaisesRegex(AssertionError, "presence changed"):
            RUNNER._bitwise_equal({"a": torch.empty(0, 3), "aux": torch.ones(1)}, before)


class DestinationCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        namespace = {"torch": torch}
        exec(SUPPORT, namespace)
        self.golden = types.SimpleNamespace(
            __cannbotdsl_source_hash__="destination", INPUT_NAMES=["x"], OUTPUT_NAMES=["a", "aux"],
            ATTR_DEFAULTS={"keep_aux": False}, REQUIRED_ATTR_NAMES=[],
            GOLDEN_FUNCTION=lambda x, keep_aux=False: (x + 1, x * 2 if keep_aux else None),
            make_inputs=namespace["make_inputs"],
            TOLERANCE={"float32": {"metric": "bitwise_equal"}},
        )
        self.spec = {"outputs": [{"name": "a"}, {"name": "aux", "optional": True}],
                     "attributes": [{"name": "keep_aux", "default": False}],
                     "semantic_cases": [
                         {"when": "attr.keep_aux", "outputs": {"present": ["aux"]}},
                         {"when": "not attr.keep_aux", "outputs": {"absent": ["aux"]}},
        ]}
        self.tensor = {"shape": [2], "dtype": "float32", "fill": "ones"}

    def row(self, outputs, inputs=None):
        return {"sheet": "blackbox", "case_id": "destination", "expected": "match_golden", "seed": "42",
                "inputs_json": json.dumps(inputs or {"x": self.tensor}), "output_json": json.dumps(outputs),
                "note": "dst_args", "assertions_json": json.dumps([
                    {"kind": "execution", "repeat": 2, "bitwise": True}])}

    def run_case(self, row, target, cache=None):
        cache = cache or CACHE.CaseCache(self.root, self.golden, spec=self.spec)
        return RUNNER._case_result(row, "cpu", RUNNER.Invocation(target, self.golden, "cpu"), cache=cache)

    def test_optional_destination_is_omitted_and_repeat_keeps_none_slot(self):
        calls = []

        def target(x, a):
            calls.append(True)
            self.assertEqual(torch.count_nonzero(a).item(), 0)
            a.add_(x + 1)
        row = self.row({"a": self.tensor})
        result = self.run_case(row, target)
        self.assertEqual(result["status"], "passed", result["detail"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(json.loads(result["detail"])[0]["records"]), 2)
        row["output_json"] = json.dumps({"a": self.tensor, "aux": None})
        result = self.run_case(row, target)
        self.assertEqual(result["status"], "passed", result["detail"])

    def test_present_and_forbidden_optional_buffers_cannot_be_skipped(self):
        target = mock.Mock()
        for row in (self.row({}, {"x": self.tensor}),
                    self.row({"a": self.tensor}, {"x": self.tensor, "keep_aux": True}),
                    self.row({"a": self.tensor, "aux": self.tensor})):
            with self.subTest(row=row):
                result = self.run_case(row, target)
                self.assertEqual(result["status"], "failed")
        target.assert_not_called()

        def target_present(x, a, aux, keep_aux=False):
            a.copy_(x + 1)
            aux.copy_(x * 2)
        result = self.run_case(self.row({"a": self.tensor, "aux": self.tensor},
                                        {"x": self.tensor, "keep_aux": True}), target_present)
        self.assertEqual(result["status"], "passed", result["detail"])

    def test_all_optional_destinations_can_be_absent(self):
        self.spec = {"outputs": [{"name": "a", "optional": True}, {"name": "aux", "optional": True}]}
        self.golden.GOLDEN_FUNCTION = lambda x, keep_aux=False: (None, None)
        result = self.run_case(self.row({}), lambda x: None)
        self.assertEqual(result["status"], "passed", result["detail"])

    def test_optional_return_none_repeats_but_required_none_still_fails(self):
        self.golden.OUTPUT_NAMES = ["aux"]
        self.golden.GOLDEN_FUNCTION = lambda x, keep_aux=False: None
        self.spec = {"outputs": [{"name": "aux", "optional": True}]}
        row = self.row({})
        row["note"] = "return_value"
        result = self.run_case(row, lambda x: None)
        self.assertEqual(result["status"], "passed", result["detail"])
        self.spec["outputs"][0]["optional"] = False
        result = self.run_case(row, lambda x: None)
        self.assertEqual(result["status"], "failed")
        self.assertIn("missing required output", result["detail"])

    def test_optional_buffer_presence_matches_primary_even_without_guard(self):
        self.spec = {"outputs": [{"name": "a"}, {"name": "aux", "optional": True}]}
        self.golden.GOLDEN_FUNCTION = lambda x, keep_aux=False: (x + 1, x * 2)
        target = mock.Mock()
        result = self.run_case(self.row({"a": self.tensor}), target)
        self.assertEqual(result["status"], "failed")
        self.assertIn("reference output presence", result["detail"])
        target.assert_not_called()

    def test_tensorlist_buffers_are_zeroed_on_build_hit_and_each_repeat(self):
        self.golden.OUTPUT_NAMES = ["ys"]
        self.golden.GOLDEN_FUNCTION = mock.Mock(side_effect=lambda x, keep_aux=False: [x + 1, x - 1])
        self.spec = {"outputs": [{"name": "ys", "role": "tensor_list", "list_length": {"kind": "fixed", "value": 2}}]}
        row = self.row({"ys": {"tensors": [self.tensor, self.tensor]}})
        cache = CACHE.CaseCache(self.root, self.golden, spec=self.spec)
        call = {"inputs": {"x": self.tensor}, "seed": 42, "assertions": []}
        data = cache.get(row, call, 0)
        for member in data["outputs"]["ys"]:
            self.assertEqual(torch.count_nonzero(member).item(), 0)
            member.fill_(7)
        torch.save(data, cache.root / f"{data['key']}.pt")
        restored = cache.get(row, call, 0)
        for member in restored["outputs"]["ys"]:
            self.assertEqual(torch.count_nonzero(member).item(), 0)
        self.assertEqual(self.golden.GOLDEN_FUNCTION.call_count, 1)

        def target(x, ys):
            for member in ys:
                self.assertEqual(torch.count_nonzero(member).item(), 0)
            ys[0].add_(x + 1)
            ys[1].add_(x - 1)
        result = self.run_case(row, target, cache)
        self.assertEqual(result["status"], "passed", result["detail"])

    def test_empty_tensorlist_and_empty_tensor_destinations_still_pass(self):
        self.golden.OUTPUT_NAMES = ["ys"]
        for members in ([], [{"shape": [0, 3], "dtype": "float32"}]):
            with self.subTest(members=members), tempfile.TemporaryDirectory() as tmp:
                self.golden.GOLDEN_FUNCTION = lambda x, keep_aux=False: [torch.empty(item["shape"]) for item in members]
                self.spec = {"outputs": [{"name": "ys", "role": "tensor_list", "list_length": {
                    "kind": "fixed", "value": len(members)}}]}
                result = self.run_case(self.row({"ys": {"tensors": members}}), lambda x, ys: None,
                                       CACHE.CaseCache(tmp, self.golden, spec=self.spec))
                self.assertEqual(result["status"], "passed", result["detail"])
                self.assertTrue(json.loads(result["detail"])[0]["records"])

    def test_error_call_does_not_gain_output_dtype_prerequisites(self):
        self.golden.OUTPUT_NAMES = ["a"]
        self.golden.GOLDEN_FUNCTION = mock.Mock(side_effect=AssertionError("must not run Golden"))
        self.spec = {"outputs": [{"name": "a"}], "dtype_policy": {"supported_combinations": [
            {"inputs": {"x": "float32"}, "outputs": {"a": "float32"}}]}}
        row = self.row({"a": self.tensor}, {"x": {**self.tensor, "dtype": "float16"}})
        row["expected"] = "raises_error:dtype_not_supported"
        called = []

        def target(x, a):
            called.append(True)
            error = ValueError("unsupported dtype")
            error.code = "dtype_not_supported"
            raise error
        result = self.run_case(row, target)
        self.assertEqual(result["status"], "passed", result["detail"])
        self.assertEqual(len(called), 1)
        self.golden.GOLDEN_FUNCTION.assert_not_called()


class DestinationPackagingTests(unittest.TestCase):
    def test_whitebox_tdd_and_runner_share_optional_tensorlist_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _, rows = fixture(work)
            spec_path = work / "design/doc/spec.yaml"
            spec = yaml.safe_load(spec_path.read_text())
            spec["outputs"] = [{"name": "z", "role": "tensor_list", "list_length": {"kind": "fixed", "value": 1}},
                               {"name": "aux", "optional": True}]
            spec["dtype_policy"]["supported_combinations"][0]["outputs"]["aux"] = "float32"
            spec_path.write_text(yaml.safe_dump(spec))
            test = work / "operators/add/test"
            path = test / "add_golden.py"
            path.write_text(path.read_text().replace("OUTPUT_NAMES = ['z']", "OUTPUT_NAMES = ['z', 'aux']").replace(
                "return torch.add(x, y)", "return ([torch.add(x, y)], None)"))
            for row in rows:
                if row["sheet"] in ("whitebox", "tdd") and row["status"] == "active":
                    row["note"] = "dst_args"
                    row["output_json"] = json.dumps(
                        {"z": {"tensors": [{"shape": [2, 8], "dtype": "float32", "fill": "ones"}]}})
                    row["assertions_json"] = '[{"kind":"execution","repeat":2,"bitwise":true}]'
            write_rows(work, rows)
            for script in ("prepare_golden.py", "build_whitebox_cases.py", "build_tdd_cases.py"):
                result = run_script(script, work)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            (work / "dst_target.py").write_text(
                "import torch\ndef run(x, y, z):\n    assert torch.count_nonzero(z[0]) == 0\n    z[0].add_(x+y)\n")
            for sheet in ("whitebox", "tdd"):
                result = subprocess.run(
                    [sys.executable, str(test / "test_add.py"), "--sheet", sheet, "--device", "cpu"],
                    env={**os.environ, "PYTHONPATH": str(work), "CANNBOTDSL_OP_MODULE": "dst_target"},
                    text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # All outputs absent: keep the existing style note, without forcing
            # a TDD-to-whitebox association just to supply nonexistent buffers.
            for output in spec["outputs"]:
                output["optional"] = True
            spec_path.write_text(yaml.safe_dump(spec))
            path.write_text(path.read_text().replace("return ([torch.add(x, y)], None)", "return (None, None)"))
            for row in rows:
                if row["sheet"] in ("whitebox", "tdd") and row["status"] == "active":
                    row["output_json"] = "{}"
                    if row["sheet"] == "tdd":
                        row["branch_ids"] = ""
            write_rows(work, rows)
            for script in ("prepare_golden.py", "build_whitebox_cases.py", "build_tdd_cases.py"):
                result = run_script(script, work)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            (work / "absent_target.py").write_text("def run(x, y):\n    return None\n")
            result = subprocess.run(
                [sys.executable, str(test / "test_add.py"), "--sheet", "tdd", "--device", "cpu"],
                env={**os.environ, "PYTHONPATH": str(work), "CANNBOTDSL_OP_MODULE": "absent_target"},
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            spec["outputs"][0]["optional"] = False
            spec_path.write_text(yaml.safe_dump(spec))
            for row in rows:
                if row["sheet"] == "whitebox" and row["status"] == "active":
                    row["output_json"] = '{"aux":null}'
            write_rows(work, rows)
            result = run_script("build_whitebox_cases.py", work)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("required spec outputs", result.stdout + result.stderr)


class PipelineContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        namespace = {"torch": torch}
        exec(SUPPORT, namespace)
        self.golden = types.SimpleNamespace(
            __cannbotdsl_source_hash__="pipeline", INPUT_NAMES=["x"], OUTPUT_NAMES=["a"],
            ATTR_DEFAULTS={}, REQUIRED_ATTR_NAMES=[], make_inputs=namespace["make_inputs"],
            GOLDEN_FUNCTION=mock.Mock(side_effect=AssertionError("behavior case must not run Golden")),
            TOLERANCE={"float32": {"metric": "bitwise_equal"}, "int64": {"metric": "bitwise_equal"}},
        )
        self.spec = {"outputs": [{"name": "a"}]}
        self.inputs = {"x": {"shape": [2], "dtype": "float32", "fill": "ones"}}

    def row(self, expected="assertions", bitwise=False, outputs=None):
        assertions = [{"kind": "input_unchanged", "inputs": ["x"]}]
        if bitwise:
            assertions.append({"kind": "execution", "repeat": 2, "bitwise": True})
        return {"sheet": "blackbox", "case_id": "pipeline", "status": "active", "expected": expected,
                "seed": "42", "inputs_json": json.dumps(self.inputs), "output_json": json.dumps(outputs or {}),
                "assertions_json": json.dumps(assertions)}

    def run_case(self, row, target, cache=None):
        cache = cache or CACHE.CaseCache(self.root, self.golden, spec=self.spec)
        return RUNNER._case_result(row, "cpu", RUNNER.Invocation(target, self.golden, "cpu"), cache=cache)

    def test_behavior_case_cannot_omit_required_output_after_preparation(self):
        row = self.row()
        cache = CACHE.CaseCache(self.root, self.golden, spec=self.spec)
        cache.prepare([row], RUNNER.case_calls)
        result = self.run_case(row, lambda x: None, cache)
        self.assertEqual(result["status"], "failed")
        self.assertIn("missing required output", result["detail"])
        self.golden.GOLDEN_FUNCTION.assert_not_called()

    def test_behavior_case_checks_destination_after_target_mutates_shape(self):
        row = self.row(outputs={"a": {"shape": [2], "dtype": "float32"}})
        result = self.run_case(row, lambda x, a: a.resize_(0))
        self.assertEqual(result["status"], "failed")
        self.assertIn("buffer shape/dtype mismatch", result["detail"])
        self.golden.GOLDEN_FUNCTION.assert_not_called()

    def test_valid_behavior_outputs_do_not_add_a_numerical_gate(self):
        result = self.run_case(self.row(), lambda x: x + 99)
        self.assertEqual(result["status"], "passed", result["detail"])
        self.spec["outputs"][0]["shape_bounds"] = {"output_rank": 2}
        result = self.run_case(self.row(), lambda x: torch.empty(0, 3))
        self.assertEqual(result["status"], "passed", result["detail"])
        self.golden.GOLDEN_FUNCTION.assert_not_called()

    def test_behavior_case_obeys_forbidden_optional_output(self):
        self.spec = {"outputs": [{"name": "a", "optional": True}], "semantic_cases": [
            {"when": "true", "outputs": {"absent": ["a"]}}]}
        result = self.run_case(self.row(), lambda x: x)
        self.assertEqual(result["status"], "failed")
        self.assertIn("must be absent", result["detail"])
        result = self.run_case(self.row(), lambda x: None)
        self.assertEqual(result["status"], "passed", result["detail"])

    def test_torch_return_type_is_cached_as_safe_plain_tuple(self):
        self.golden.OUTPUT_NAMES = ["values", "indices"]
        self.golden.GOLDEN_FUNCTION = mock.Mock(side_effect=lambda x: torch.max(x, dim=0))
        self.spec = {"outputs": [{"name": "values"}, {"name": "indices"}]}
        row = self.row(expected="match_golden", bitwise=True)
        cache = CACHE.CaseCache(self.root, self.golden, spec=self.spec)
        call = {"inputs": self.inputs, "seed": 42, "assertions": [{"kind": "input_unchanged", "inputs": ["x"]}]}
        first = cache.get(row, call, 0)
        second = cache.get(row, call, 0)
        self.assertIs(type(first["expected"]), tuple)
        self.assertEqual((cache.built, cache.hits), (1, 1))
        stored = torch.load(cache.root / f"{first['key']}.pt", map_location="cpu", weights_only=True)
        self.assertIs(type(stored["expected"]), tuple)
        torch.testing.assert_close(first["expected"], second["expected"])
        result = self.run_case(row, lambda x: torch.max(x, dim=0), cache)
        self.assertEqual(result["status"], "passed", result["detail"])
        self.assertEqual(self.golden.GOLDEN_FUNCTION.call_count, 1)
        # A legacy v4 custom return container is rebuilt once, then hits again.
        stored["expected"] = torch.max(first["inputs"]["x"], dim=0)
        torch.save(stored, cache.root / f"{first['key']}.pt")
        cache.get(row, call, 0)
        cache.get(row, call, 0)
        self.assertEqual(cache.built, 2)
        self.assertEqual(self.golden.GOLDEN_FUNCTION.call_count, 2)

    def test_equivalent_optional_and_ordered_representations_repeat(self):
        self.golden.OUTPUT_NAMES = ["a", "aux"]
        self.golden.GOLDEN_FUNCTION = lambda x: {"a": x, "aux": None}
        self.spec = {"outputs": [{"name": "a"}, {"name": "aux", "optional": True}]}
        for first_form in (lambda x: {"a": x}, lambda x: (x, None)):
            with self.subTest(first_form=first_form), tempfile.TemporaryDirectory() as tmp:
                calls = []

                def target(x):
                    calls.append(True)
                    return first_form(x) if len(calls) == 1 else {"a": x, "aux": None}
                cache = CACHE.CaseCache(tmp, self.golden, spec=self.spec)
                result = self.run_case(self.row(expected="match_golden", bitwise=True), target, cache)
                self.assertEqual(result["status"], "passed", result["detail"])
                self.assertEqual(len(calls), 2)

    def test_equivalent_tensorlist_containers_repeat(self):
        self.golden.OUTPUT_NAMES = ["ys"]
        self.golden.GOLDEN_FUNCTION = lambda x: [x]
        self.spec = {"outputs": [{"name": "ys", "role": "tensor_list", "list_length": {"kind": "fixed", "value": 1}}]}
        calls = []

        def target(x):
            calls.append(True)
            return [x] if len(calls) == 1 else (x,)
        result = self.run_case(self.row(expected="match_golden", bitwise=True), target)
        self.assertEqual(result["status"], "passed", result["detail"])

    def test_normalization_still_detects_bit_and_presence_changes(self):
        calls = []

        def sign_change(x):
            calls.append(True)
            return torch.tensor([0.0 if len(calls) == 1 else -0.0])
        result = self.run_case(self.row(bitwise=True), sign_change)
        self.assertEqual(result["status"], "failed")
        self.assertIn("changed bitwise", result["detail"])
        self.golden.OUTPUT_NAMES = ["a", "aux"]
        self.spec = {"outputs": [{"name": "a"}, {"name": "aux", "optional": True}]}
        calls.clear()

        def presence_change(x):
            calls.append(True)
            return {"a": x, "aux": None if len(calls) == 1 else x}
        result = self.run_case(self.row(bitwise=True), presence_change)
        self.assertEqual(result["status"], "failed")
        self.assertIn("presence changed", result["detail"])


class FinalCsvContractTests(unittest.TestCase):
    def test_tdd_does_not_inherit_unmaterialized_branch_destinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _, rows = fixture(work)
            buffers = {"z": {"shape": [2, 8], "dtype": "float32"}}
            for row in rows:
                if row["sheet"] == "whitebox" and row["status"] == "active":
                    row["note"] = "dst_args"
                    row["output_json"] = json.dumps(buffers)
            tdd = next(row for row in rows if row["sheet"] == "tdd")
            for note in ("", "dst_args"):
                tdd["note"] = note
                csv_path = write_rows(work, rows)
                before = csv_path.read_bytes()
                prepared = run_script("prepare_golden.py", work)
                self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
                result = run_script("build_tdd_cases.py", work)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(tdd["case_id"], result.stdout + result.stderr)
                self.assertIn("output_json", result.stdout + result.stderr)
                self.assertEqual(csv_path.read_bytes(), before)
                self.assertFalse((work / "operators/add/test/.case_cache").exists())
            tdd["output_json"] = json.dumps(buffers)
            csv_path = write_rows(work, rows)
            before = csv_path.read_bytes()
            result = run_script("build_tdd_cases.py", work)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(csv_path.read_bytes(), before)
            (work / "csv_target.py").write_text("def run(x, y, z):\n    z.copy_(x+y)\n")
            result = subprocess.run(
                [sys.executable, str(work / "operators/add/test/test_add.py"), "--sheet", "tdd", "--device", "cpu"],
                env={**os.environ, "PYTHONPATH": str(work), "CANNBOTDSL_OP_MODULE": "csv_target"},
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_packaged_behavior_case_rejects_missing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _, rows = fixture(work)
            unit = work / "design/units/U01.yaml"
            data = json.loads(unit.read_text())
            data["unit"]["verification_obligations"][0]["kind"] = "semantic"
            unit.write_text(json.dumps(data))
            tdd = next(row for row in rows if row["sheet"] == "tdd")
            tdd["expected"] = "assertions"
            tdd["assertions_json"] = '[{"kind":"input_unchanged","inputs":["x"]}]'
            write_rows(work, rows)
            for script in ("prepare_golden.py", "build_tdd_cases.py"):
                result = run_script(script, work)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            (work / "behavior_target.py").write_text("def run(x, y):\n    return None\n")
            result = subprocess.run(
                [sys.executable, str(work / "operators/add/test/test_add.py"), "--sheet", "tdd", "--device", "cpu"],
                env={**os.environ, "PYTHONPATH": str(work), "CANNBOTDSL_OP_MODULE": "behavior_target"},
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            with (work / "operators/add/test/testcase_output/tdd/verify_result_tdd.csv").open(
                encoding="utf-8-sig", newline="",
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertIn("missing required output", row["detail"])


if __name__ == "__main__":
    unittest.main()
