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

"""User case compatibility, shared-file updates and correctness-only execution."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_skills import fixture, row, PAIR, SPEC, run_script, write_rows

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from casebook import COLUMNS
from update_case_group import update_group
from build_performance_cases import validate_performance
from build_blackbox_cases import validate_input_recipes


class PerformanceTests(unittest.TestCase):
    def perf(self, **changes):
        return row("performance", "PERF-user-1", PAIR, expected="match_golden",
                   design_ref="math_semantics", note="user attachment: case 1", **changes)

    def test_validate_and_execute_with_shared_golden(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            fixture(work)
            target = work / "operators/add/test"
            update_group(target, "performance", [self.perf()])
            self.assertEqual(run_script("build_performance_cases.py", work).returncode, 0)
            self.assertEqual(run_script("build_blackbox_cases.py", work).returncode, 0)
            prepared = run_script("prepare_golden.py", work)
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            cache_result = run_script("build_tdd_cases.py", work)
            self.assertEqual(cache_result.returncode, 0, cache_result.stdout + cache_result.stderr)
            self.assertTrue(list((target / ".case_cache").glob("*.pt")))
            (target / "target_add.py").write_text("def run(x, y):\n    return x + y\n")
            env = {**os.environ, "CANNBOTDSL_OP_MODULE": "target_add"}
            command = [sys.executable, str(target / "test_add.py"), "--sheet", "performance", "--device", "cpu"]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = target / "testcase_output/performance/verify_result_performance.csv"
            with output.open() as handle:
                self.assertEqual(list(csv.DictReader(handle))[0]["status"], "passed")
            # A wrong implementation must produce an actual failure, not timing success.
            (target / "target_add.py").write_text("def run(x, y):\n    return x - y + 100\n")
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_empty_group_is_optional_and_preserves_existing_cases(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            fixture(work)
            update_group(work / "operators/add/test", "performance", [])
            self.assertEqual(run_script("build_performance_cases.py", work).returncode, 0)
            self.assertEqual(run_script("build_blackbox_cases.py", work).returncode, 0)

    def test_invalid_user_inputs_and_refs_rejected(self):
        for key, value in [("inputs_json", json.dumps({"unknown": PAIR["x"]})),
                           ("design_ref", "does_not_exist"), ("expected", "raises_error:shape_mismatch")]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as td:
                work = Path(td)
                fixture(work)
                item = self.perf()
                item[key] = value
                update_group(work / "operators/add/test", "performance", [item])
                self.assertNotEqual(run_script("build_performance_cases.py", work).returncode, 0)

    def test_concurrent_group_updates_preserve_both(self):
        script = Path(__file__).resolve().parents[1] / "scripts/update_case_group.py"
        with tempfile.TemporaryDirectory() as td:
            processes = []
            for sheet, item in [("performance", self.perf()),
                                ("blackbox", row("blackbox", "BB-core", PAIR))]:
                process = subprocess.Popen(
                    [sys.executable, str(script), "--test-dir", td, "--sheet", sheet],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                processes.append((process, item))
            for process, item in processes:
                process.stdin.write(json.dumps([item]))
                process.stdin.close()
            for process, _ in processes:
                self.assertEqual(process.wait(timeout=10), 0, process.stderr.read())
                process.stdout.close()
                process.stderr.close()
            with (Path(td) / "testcase.csv").open() as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames), COLUMNS)
                self.assertEqual({r["sheet"] for r in reader}, {"blackbox", "performance"})

    def test_failed_update_leaves_original_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            update_group(td, "performance", [self.perf()])
            before = (Path(td) / "testcase.csv").read_bytes()
            with self.assertRaises(ValueError):
                update_group(td, "blackbox", [self.perf()])
            self.assertEqual((Path(td) / "testcase.csv").read_bytes(), before)

    def list_attribute_spec(self, **changes):
        import yaml

        spec = yaml.safe_load(SPEC)
        spec["attributes"] = [{"name": "groups", "type": "list[int64]", **changes}]
        return spec

    def test_declared_int64_lists_allow_empty_duplicates_and_signed_limits(self):
        for value in ([], [0], [2, 2, -1], [-(2**63), 2**63 - 1]):
            with self.subTest(value=value):
                item = self.perf(inputs_json=json.dumps({**PAIR, "groups": value}))
                self.assertEqual(validate_performance([item], self.list_attribute_spec()), 1)
        # Optional/default attributes may still be omitted; no business constraint
        # (such as nonempty, positive or monotone) is inferred from the list type.
        self.assertEqual(validate_performance(
            [self.perf()], self.list_attribute_spec(default=[])), 1)

    def test_declared_int64_lists_reject_wrong_container_elements_and_overflow(self):
        for value in (None, 1, "1,2", {}, [True], [False], [1.0], ["1"],
                      [None], [[1]], [{"value": 1}], [2**63], [-(2**63) - 1],
                      [float("nan")], [float("inf")]):
            with self.subTest(value=value):
                item = self.perf(inputs_json=json.dumps({**PAIR, "groups": value}))
                with self.assertRaisesRegex(ValueError, r"PERF-user-1.groups: list\[int64\]"):
                    validate_performance([item], self.list_attribute_spec())
        with self.assertRaisesRegex(ValueError, "omit required attributes"):
            validate_performance([self.perf()], self.list_attribute_spec())

    def test_supported_non_integer_list_attributes_keep_upstream_support(self):
        for kind, values in (("list[float32]", [1.25, -2.0]),
                             ("list[bool]", [True, False]),
                             ("list[string]", ["a", "b"])):
            for value in ([], values):
                with self.subTest(kind=kind, value=value):
                    item = self.perf(inputs_json=json.dumps({**PAIR, "groups": value}))
                    self.assertEqual(validate_performance(
                        [item], self.list_attribute_spec(type=kind)), 1)

    def test_list_attribute_support_does_not_relax_other_input_roles(self):
        item = self.perf(inputs_json=json.dumps({**PAIR, "groups": [1]}))
        with self.assertRaisesRegex(ValueError, "outside spec"):
            import yaml
            validate_performance([item], yaml.safe_load(SPEC))
        for kind in ("int64", "string", "list[unknown]"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "JSON scalar"):
                validate_performance([item], self.list_attribute_spec(type=kind))
        with self.assertRaises(ValueError):
            validate_input_recipes("P", {"x": [1]},
                                   {"x": {"role": "tensor"}}, None)
        tensor_list = {"xs": {"tensors": [PAIR["x"]]}}
        validate_input_recipes("P", tensor_list,
                               {"xs": {"role": "tensor_list",
                                       "list_length": {"kind": "fixed", "value": 1},
                                       "dtype_set": ["float32"], "rank_range": [0, 8]}}, None)
        with self.assertRaises(ValueError):
            validate_input_recipes("P", {"xs": [1]},
                                   {"xs": {"role": "tensor_list",
                                           "list_length": {"kind": "fixed", "value": 1},
                                           "dtype_set": ["float32"], "rank_range": [0, 8]}}, None)
        # The old four-argument interface still rejects untyped collections.
        with self.assertRaisesRegex(ValueError, "JSON scalar"):
            validate_input_recipes("P", {"groups": [1]}, {}, None)

    def test_blackbox_and_performance_cli_share_list_attribute_validation(self):
        import yaml

        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            _, rows = fixture(work)
            (work / "design/doc/spec.yaml").write_text(
                yaml.safe_dump(self.list_attribute_spec()))
            for item in rows:
                if item["status"] == "active":
                    inputs = json.loads(item["inputs_json"])
                    item["inputs_json"] = json.dumps({**inputs, "groups": [1, 2]})
            rows.append(self.perf(inputs_json=json.dumps({**PAIR, "groups": [1, 2]})))
            write_rows(work, rows)
            for script in ("build_blackbox_cases.py", "build_performance_cases.py"):
                result = run_script(script, work)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for item in rows:
                if item["case_id"] in ("BB-core", "PERF-user-1"):
                    inputs = json.loads(item["inputs_json"])
                    item["inputs_json"] = json.dumps({**inputs, "groups": [True]})
            write_rows(work, rows)
            for script in ("build_blackbox_cases.py", "build_performance_cases.py"):
                result = run_script(script, work)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("list[int64]", result.stderr)

    def test_list_attributes_reach_reference_target_and_cache_identity(self):
        import yaml
        import torch
        from case_cache import CaseCache, load_golden

        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            fixture(work)
            spec = self.list_attribute_spec()
            (work / "design/doc/spec.yaml").write_text(yaml.safe_dump(spec))
            test_dir = work / "operators/add/test"
            golden_path = test_dir / "add_golden.py"
            golden_path.write_text(golden_path.read_text().replace(
                "REQUIRED_ATTR_NAMES = []", "REQUIRED_ATTR_NAMES = ['groups']").replace(
                "def add_golden(x, y):\n    return torch.add(x, y)",
                "def add_golden(x, y, groups):\n"
                "    assert type(groups) is list\n"
                "    return torch.add(x, y) + sum(groups)"))
            item = self.perf(inputs_json=json.dumps({**PAIR, "groups": [1, 2]}))
            self.assertEqual(validate_performance([item], spec), 1)
            cache = CaseCache(test_dir, load_golden(golden_path), spec=spec)
            call = {"inputs": json.loads(item["inputs_json"]), "seed": 42, "assertions": []}
            first = cache.get(item, call, 0)
            repeated = cache.get(item, call, 0)
            self.assertEqual((cache.built, cache.hits), (1, 1))
            self.assertEqual(first["inputs"]["groups"], [1, 2])
            self.assertTrue(torch.equal(first["expected"], repeated["expected"]))
            call["inputs"]["groups"] = [1, 5]
            changed = cache.get(item, call, 0)
            self.assertEqual((cache.built, cache.hits), (2, 1))
            self.assertTrue(torch.allclose(changed["expected"], first["expected"] + 3))
            item["inputs_json"] = json.dumps(call["inputs"])
            write_rows(work, [item])
            prepared = run_script("prepare_golden.py", work)
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            (test_dir / "target_add.py").write_text(
                "def run(x, y, groups):\n"
                "    assert type(groups) is list and groups == [1, 5]\n"
                "    return x + y + sum(groups)\n")
            result = subprocess.run(
                [sys.executable, str(test_dir / "test_add.py"),
                 "--sheet", "performance", "--device", "cpu"],
                env={**os.environ, "CANNBOTDSL_OP_MODULE": "target_add"},
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class AllCasesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dsl all cases spaces ")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        _, rows = fixture(self.work)
        self.rows = [
            item
            for item in rows
            if item["case_id"] in {"BB-core", "WB-B0", "WB-B-excluded", "TD-u01-main"}
        ]
        self.rows.append(row("performance", "PERF-user-1", PAIR, expected="match_golden"))
        write_rows(self.work, self.rows)
        prepared = run_script("prepare_golden.py", self.work)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        self.test_dir = self.work / "operators/add/test"
        self.log = self.work / "calls.log"
        (self.test_dir / "all_target.py").write_text(
            "import os\nfrom pathlib import Path\n"
            "log = Path(os.environ['CANNBOTDSL_TEST_LOG'])\n"
            "with log.open('a') as stream:\n"
            "    stream.write(f'import {os.getpid()}\\n')\n"
            "calls = 0\n"
            "def run(x, y):\n"
            "    global calls\n"
            "    calls += 1\n"
            "    with log.open('a') as stream:\n"
            "        stream.write(f'call {os.getpid()}\\n')\n"
            "    if os.environ.get('CANNBOTDSL_TEST_FAIL') and calls == 1:\n"
            "        return x + y + 100\n"
            "    return x + y\n"
        )
        self.env = {
            **os.environ,
            "CANNBOTDSL_OP_MODULE": "all_target",
            "CANNBOTDSL_TEST_LOG": str(self.log),
        }

    def invoke(self, *arguments):
        return subprocess.run(
            [sys.executable, str(self.test_dir / "test_add.py"),
             "--all", "--device", "cpu", *arguments],
            cwd=self.work.parent, env=self.env, capture_output=True, text=True, timeout=60,
        )

    def result_paths(self):
        return {
            sheet: self.test_dir / f"testcase_output/{sheet}/verify_result_{sheet}.csv"
            for sheet in ("blackbox", "whitebox", "tdd", "performance")
        }

    def read_results(self):
        results = []
        for path in self.result_paths().values():
            if path.exists():
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    results.extend(csv.DictReader(handle))
        return results

    def test_all_active_cases_share_one_target_load_and_keep_group_outputs(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        results = self.read_results()
        self.assertEqual(
            {(item["sheet"], item["case_id"]) for item in results},
            {(item["sheet"], item["case_id"]) for item in self.rows if item["status"] == "active"},
        )
        self.assertEqual(len(results), 4)
        self.assertTrue(all(item["status"] == "passed" for item in results))
        events = [line.split() for line in self.log.read_text().splitlines()]
        self.assertEqual([event[0] for event in events], ["import", *["call"] * 4])
        self.assertEqual(len({event[1] for event in events}), 1)
        for sheet, path in self.result_paths().items():
            with path.open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual({item["sheet"] for item in csv.DictReader(handle)}, {sheet})

    def test_case_failure_does_not_skip_later_groups(self):
        self.env["CANNBOTDSL_TEST_FAIL"] = "1"
        result = self.invoke()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(
            [item["status"] for item in self.read_results()],
            ["failed", "passed", "passed", "passed"],
        )

    def test_empty_performance_group_removes_old_result_without_faking_pass(self):
        write_rows(self.work, [item for item in self.rows if item["sheet"] != "performance"])
        path = self.result_paths()["performance"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old success")
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(path.exists())
        self.assertEqual(len(self.read_results()), 3)
        self.assertIn("skipped performance: no active cases", result.stdout)

    def test_unexecutable_all_run_invalidates_all_old_results(self):
        for fault in ("empty", "invalid_csv", "missing_golden", "cpu_guard"):
            with self.subTest(fault=fault):
                write_rows(self.work, self.rows if fault != "empty" else [])
                if fault == "invalid_csv":
                    (self.test_dir / "testcase.csv").write_text("bad_header\n")
                golden = self.test_dir / "add_golden.py"
                source = golden.read_text()
                if fault == "missing_golden":
                    golden.unlink()
                for path in self.result_paths().values():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("old success")
                try:
                    result = self.invoke(*(["--require-accelerator"] if fault == "cpu_guard" else []))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(any(path.exists() for path in self.result_paths().values()))
                    self.assertFalse(self.log.exists())
                finally:
                    golden.write_text(source)

    def test_all_rejects_group_and_case_filters(self):
        for arguments in (("--sheet", "tdd"), ("--case-id", "TD-u01-main"), ("--grep", ".*")):
            with self.subTest(arguments=arguments):
                result = self.invoke(*arguments)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(self.log.exists())
