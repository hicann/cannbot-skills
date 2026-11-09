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

import csv
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from prepare_golden import SUPPORT

SPEC = """schema_version: 1
op: {name: add, error_codes: [shape_mismatch]}
inputs:
  - {name: x, role: tensor, dtype_set: [float32], rank_range: [0, 8]}
  - {name: y, role: tensor, dtype_set: [float32], rank_range: [0, 8]}
attributes: []
outputs:
  - {name: z}
dtype_policy:
  supported_combinations:
    - {inputs: {x: float32, y: float32}, outputs: {z: float32}}
math_semantics: {formula: "z = x + y"}
boundary_conditions:
  - {case: aligned, machine_check: {kind: matches_oracle}}
  - {case: incompatible, machine_check: {kind: raises_error, error_type: shape_mismatch}}
extreme_inputs:
  - {case: nan, machine_check: {kind: nan_propagates}}
numerical_tolerance:
  per_dtype:
    float32: {rtol: 1.0e-6, atol: 1.0e-6, metric: max_relative}
"""
DESIGN = """# add
Algorithm-Add: elementwise addition.
Stage-Elementwise: add x and y.
"""
COLUMNS = (
    "sheet",
    "case_id",
    "source",
    "branch_ids",
    "unit_id",
    "inputs_json",
    "output_json",
    "seed",
    "expected",
    "assertions_json",
    "obligation_ids",
    "note",
    "status",
    "design_ref",
    "condition",
    "exclude_reason",
)
PAIR = {
    "x": {"shape": [2, 8], "dtype": "float32"},
    "y": {"shape": [2, 8], "dtype": "float32"},
}


def row(sheet, case_id, inputs=None, **fields):
    value = {column: "" for column in COLUMNS}
    value.update(
        sheet=sheet,
        source=sheet,
        case_id=case_id,
        status="active" if inputs is not None else "excluded",
        inputs_json=json.dumps(inputs) if inputs is not None else "",
        output_json="{}",
        seed="42",
        assertions_json="[]",
    )
    value.update(fields)
    return value


def write_rows(work, rows):
    path = work / "operators/add/test/testcase.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def fixture(work):
    (work / "design/doc").mkdir(parents=True, exist_ok=True)
    (work / "design/doc/spec.yaml").write_text(SPEC, encoding="utf-8")
    (work / "design/doc/DESIGN.md").write_text(DESIGN, encoding="utf-8")
    unit_dir = work / "design/units"
    unit_dir.mkdir()
    native = {
        "unit": {
            "unit_id": "U01",
            "title": "add",
            "kind": "minimal_operator",
            "design_sources": [
                {"file": "algorithm.md", "section": "Stage-Elementwise"}
            ],
            "depends_on": [],
            "scope": {"source_files": ["src/add.py"]},
            "verification_obligations": [
                {
                    "id": "VO-U01",
                    "kind": "numerical",
                    "design_sources": ["algorithm.md"],
                }
            ],
        }
    }
    (unit_dir / "U01.yaml").write_text(json.dumps(native), encoding="utf-8")
    golden = (
        "import torch\n"
        "OP_NAME = 'add'\n"
        "INPUT_NAMES = ['x', 'y']\nOPTIONAL_INPUT_NAMES = []\nOUTPUT_NAMES = ['z']\n"
        "ATTR_DEFAULTS = {}\nREQUIRED_ATTR_NAMES = []\n"
        "TOLERANCE = {'float32': {'rtol': 1e-6, 'atol': 1e-6, 'metric': 'max_relative'}}\n"
        "DESIGN_REFS = ('Algorithm-Add', 'Stage-Elementwise')\n"
        "def add_golden(x, y):\n    return torch.add(x, y)\n"
        "GOLDEN_FUNCTION = add_golden\n" + SUPPORT
    )
    (work / "operators/add/test/add_golden.py").parent.mkdir(
        parents=True, exist_ok=True
    )
    (work / "operators/add/test/add_golden.py").write_text(golden, encoding="utf-8")
    rows = [
        row(
            "blackbox",
            "BB-core",
            PAIR,
            expected="match_golden",
            note="aligned addition",
            design_ref="math_semantics,dtype_policy.supported_combinations[0],boundary_conditions[0]",
            condition="L0",
        ),
        row(
            "blackbox",
            "BB-nan",
            {
                "x": {"shape": [8], "dtype": "float32", "fill": "nan_one"},
                "y": {"shape": [8], "dtype": "float32", "fill": "zeros"},
            },
            expected="match_golden_nan",
            note="NaN propagation",
            design_ref="extreme_inputs[0]",
            condition="L1",
        ),
        row(
            "blackbox",
            "BB-bad-shape",
            {
                "x": {"shape": [2, 3], "dtype": "float32"},
                "y": {"shape": [2, 4], "dtype": "float32"},
            },
            expected="raises_error:shape_mismatch",
            note="invalid shapes",
            design_ref="boundary_conditions[1]",
            condition="L2",
        ),
        row(
            "whitebox",
            "WB-B0",
            PAIR,
            branch_ids="B0",
            expected="match_golden",
            note="return_value",
            design_ref="Stage-Elementwise",
            condition="tile path",
        ),
        row(
            "whitebox",
            "WB-B-excluded",
            branch_ids="B-excluded",
            design_ref="Stage-Elementwise",
            condition="unreachable path",
            exclude_reason="not exposed by public inputs",
        ),
        row(
            "tdd",
            "TD-u01-main",
            PAIR,
            unit_id="u01",
            branch_ids="B0",
            obligation_ids="VO-U01",
            expected="match_golden",
        ),
    ]
    path = write_rows(work, rows)
    return path, rows


def run_script(name, work, *extra):
    args = [
        "--spec",
        str(work / "design/doc/spec.yaml"),
        "--test-dir",
        str(work / "operators/add/test"),
    ]
    if name != "build_blackbox_cases.py":
        args += ["--design", str(work / "design/doc/DESIGN.md")]
    if name == "build_tdd_cases.py":
        args += ["--units-dir", str(work / "design/units")]
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args, *extra],
        text=True,
        capture_output=True,
    )


class ExplicitPathTests(unittest.TestCase):
    def test_inputs_and_output_can_use_independent_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            fixture(work)
            source = work / "design/doc"
            spec = work / "inputs/operator.yaml"
            design = work / "notes/algorithm.md"
            units = work / "unit-cards"
            target = work / "output/test-assets"
            spec.parent.mkdir()
            design.parent.mkdir()
            shutil.move(source / "spec.yaml", spec)
            shutil.move(source / "DESIGN.md", design)
            shutil.move(work / "design/units", units)
            target.parent.mkdir(parents=True)
            shutil.move(work / "operators/add/test", target)
            base = ["--spec", str(spec), "--test-dir", str(target)]
            commands = [
                ("build_blackbox_cases.py", base),
                ("prepare_golden.py", base + ["--design", str(design)]),
                ("build_whitebox_cases.py", base + ["--design", str(design)]),
                (
                    "build_tdd_cases.py",
                    base + ["--design", str(design), "--units-dir", str(units)],
                ),
            ]
            for name, args in commands:
                result = subprocess.run(
                    [sys.executable, str(SCRIPTS / name), *args],
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(
                    result.returncode, 0, f"{name}: {result.stderr} {result.stdout}"
                )
            # Execute from a custom directory, then relocate and rename the runner.
            for directory in (target, work / "relocated/assets"):
                if directory != target:
                    shutil.copytree(target, directory)
                    (directory / "test_add.py").rename(directory / "entry.py")
                entry = directory / (
                    "test_add.py" if directory == target else "entry.py"
                )
                (directory / "target_add.py").write_text(
                    "def run(x, y):\n    return x + y\n"
                )
                env = dict(
                    os.environ,
                    CANNBOTDSL_OP_MODULE="target_add",
                    CANNBOTDSL_OP_ENTRY="run",
                )
                for sheet, case_id in (
                    ("blackbox", "BB-core"),
                    ("whitebox", "WB-B0"),
                    ("tdd", "TD-u01-main"),
                ):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(entry),
                            "--sheet",
                            sheet,
                            "--case-id",
                            case_id,
                            "--device",
                            "cpu",
                        ],
                        cwd=work,
                        env=env,
                        text=True,
                        capture_output=True,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stderr + result.stdout
                    )
                    report = (
                        directory
                        / "testcase_output"
                        / sheet
                        / f"verify_result_{sheet}_{case_id}.csv"
                    )
                    with report.open() as handle:
                        self.assertEqual(
                            next(csv.DictReader(handle))["status"], "passed"
                        )


class BatchSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dsl batch spaces ")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        _, rows = fixture(self.work)
        for case_id, uid in (
            ("TD-u01-second", "u01"),
            ("TD-u02-main", "u02"),
            ("TD-u010-main", "u010"),
        ):
            rows.append(row("tdd", case_id, PAIR, unit_id=uid, expected="match_golden"))
        rows.append(
            row("tdd", "TD-u01-excluded", unit_id="u01", exclude_reason="fixture")
        )
        rows.append(
            row("blackbox", "TD-u01-other-sheet", PAIR, expected="match_golden")
        )
        write_rows(self.work, rows)
        prepared = run_script("prepare_golden.py", self.work)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        self.test_dir = self.work / "operators/add/test"
        self.log = self.work / "calls.log"
        (self.work / "batch_target.py").write_text(
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
            "        raise RuntimeError('fixture failure')\n"
            "    return x + y\n",
            encoding="utf-8",
        )
        self.env = {
            **os.environ,
            "PYTHONPATH": str(self.work),
            "CANNBOTDSL_OP_MODULE": "batch_target",
            "CANNBOTDSL_TEST_LOG": str(self.log),
        }

    def invoke(self, *selection):
        return subprocess.run(
            [
                sys.executable,
                str(self.test_dir / "test_add.py"),
                "--sheet",
                "tdd",
                "--device",
                "cpu",
                *selection,
            ],
            cwd=self.work.parent,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def result_path(self, expression):
        digest = hashlib.sha256(expression.encode("utf-8")).hexdigest()[:16]
        return (
            self.test_dir / f"testcase_output/tdd/verify_result_tdd_grep_{digest}.csv"
        )

    def read_results(self, expression):
        with self.result_path(expression).open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            return list(csv.DictReader(handle))

    def test_unit_batch_runs_all_bound_cases_in_one_process(self):
        expression = "^TD-u01-"
        result = self.invoke("--grep", expression)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = self.read_results(expression)
        self.assertEqual(
            [item["case_id"] for item in rows], ["TD-u01-main", "TD-u01-second"]
        )
        self.assertTrue(all(item["status"] == "passed" for item in rows))
        events = [line.split() for line in self.log.read_text().splitlines()]
        self.assertEqual([event[0] for event in events], ["import", "call", "call"])
        self.assertEqual(len({event[1] for event in events}), 1)

    def test_multi_unit_batch_has_separate_results(self):
        expression = "^TD-(?:u01|u02)-"
        single = self.result_path("^TD-u01-")
        single.write_text("previous unit batch", encoding="utf-8")
        full = single.parent / "verify_result_tdd.csv"
        full.write_text("previous full sheet", encoding="utf-8")
        result = self.invoke("--grep", expression)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [item["case_id"] for item in self.read_results(expression)],
            ["TD-u01-main", "TD-u01-second", "TD-u02-main"],
        )
        self.assertEqual(single.read_text(), "previous unit batch")
        self.assertEqual(full.read_text(), "previous full sheet")

    def test_failed_batch_finishes_and_replaces_stale_success(self):
        expression = "^TD-u01-"
        self.result_path(expression).write_text("old success", encoding="utf-8")
        self.env["CANNBOTDSL_TEST_FAIL"] = "1"
        result = self.invoke("--grep", expression)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(
            [item["status"] for item in self.read_results(expression)],
            ["failed", "passed"],
        )

    def test_invalid_or_empty_selection_removes_stale_results(self):
        for expression in ("[", "^TD-missing-", "^TD-u01-excluded$", "other-sheet$"):
            with self.subTest(expression=expression):
                path = self.result_path(expression)
                path.write_text("old success", encoding="utf-8")
                result = self.invoke("--grep", expression)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(
                    "invalid --grep" if expression == "[" else "no cases selected",
                    result.stderr,
                )
                self.assertFalse(path.exists())
        self.assertFalse(self.log.exists())

    def test_grep_and_case_id_are_mutually_exclusive(self):
        result = self.invoke("--grep", "^TD-u01-", "--case-id", "TD-u01-main")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("not allowed with argument", result.stderr)
        self.assertFalse(self.log.exists())


class CpuCaseCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dsl CPU cache spaces ")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        _, self.rows = fixture(self.work)
        prepared = run_script("prepare_golden.py", self.work)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        self.test_dir = self.work / "operators/add/test"
        self.cache_module = self.load_module(
            "cache_under_test", self.test_dir / "cannbotdsl_case_cache.py"
        )
        contract = self.load_module(
            "contract_under_test", self.test_dir / "cannbotdsl_case_contract.py"
        )
        with mock.patch.dict(
            sys.modules,
            {
                "cannbotdsl_case_cache": self.cache_module,
                "cannbotdsl_case_contract": contract,
            },
        ):
            self.runner = self.load_module(
                "runner_under_test", self.test_dir / "test_add.py"
            )
        self.golden = self.cache_module.load_golden(self.test_dir / "add_golden.py")
        self.cache = self.cache_module.CaseCache(self.test_dir, self.golden)
        self.case = next(r for r in self.rows if r["sheet"] == "tdd")
        self.target = ModuleType("cache_target")
        self.target.run = lambda x, y: x + y
        self.counts = {"inputs": 0, "golden": 0}
        make_inputs, simulate = self.golden.make_inputs, self.golden.simulate

        def counted_inputs(*args, **kwargs):
            self.counts["inputs"] += 1
            return make_inputs(*args, **kwargs)

        def counted_golden(*args, **kwargs):
            self.counts["golden"] += 1
            return simulate(*args, **kwargs)

        self.golden.make_inputs = counted_inputs
        self.golden.simulate = counted_golden

    @staticmethod
    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def call(self):
        return {
            "inputs": json.loads(self.case["inputs_json"]),
            "seed": int(self.case["seed"]),
            "assertions": json.loads(self.case["assertions_json"]),
        }

    def run_case(self):
        invocation = self.runner.Invocation(self.target.run, self.golden, "cpu")
        return self.runner._case_result(self.case, "cpu", invocation, cache=self.cache)

    def test_generation_prepares_all_active_groups_and_reuses_entries(self):
        result = run_script("build_tdd_cases.py", self.work)
        self.assertEqual(result.returncode, 0, result.stderr)
        active = sum(r["status"] == "active" for r in self.rows)
        self.assertEqual(len(list(self.cache.root.glob("*.pt"))), active)
        original = {p.name: p.stat().st_mtime_ns for p in self.cache.root.glob("*.pt")}
        repeated = run_script("build_tdd_cases.py", self.work)
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertIn(f"built=0 hits={active}", repeated.stdout)
        self.assertEqual(
            original,
            {p.name: p.stat().st_mtime_ns for p in self.cache.root.glob("*.pt")},
        )

    def test_external_cache_is_shared_by_generation_and_execution(self):
        self.assertEqual(self.cache.root, self.test_dir / ".case_cache")
        root = self.work / "external cache with spaces"
        with mock.patch.dict(os.environ, {"CANNBOTDSL_CASE_CACHE_DIR": str(root)}):
            result = run_script("build_tdd_cases.py", self.work)
            self.assertEqual(result.returncode, 0, result.stderr)
            cache = self.cache_module.CaseCache(self.test_dir, self.golden)
            self.assertEqual(cache.root, root)
            cache.get(self.case, self.call(), 0)
            self.assertEqual((cache.built, cache.hits), (0, 1))
        self.assertTrue(list(root.glob("*.pt")))
        self.assertFalse((self.test_dir / ".case_cache").exists())

    def test_hit_skips_input_generation_and_golden(self):
        first = self.cache.get(self.case, self.call(), 0)
        self.assertEqual(self.counts, {"inputs": 2, "golden": 1})
        second = self.cache.get(self.case, self.call(), 0)
        self.assertEqual(self.counts, {"inputs": 2, "golden": 1})
        self.cache_module.torch.testing.assert_close(first["inputs"], second["inputs"])
        self.cache_module.torch.testing.assert_close(
            first["expected"], second["expected"]
        )
        self.assertEqual((self.cache.built, self.cache.hits), (1, 1))

    def test_recipe_seed_and_assertion_changes_rebuild_only_changed_call(self):
        call = self.call()
        self.cache.get(self.case, call, 0)
        for change in ("seed", "inputs", "assertions"):
            altered = copy.deepcopy(call)
            if change == "seed":
                altered["seed"] += 1
            elif change == "inputs":
                altered["inputs"]["x"]["fill"] = "zeros"
            else:
                altered["assertions"] = [{"kind": "output_dtype", "dtype": "float32"}]
            self.cache.get(self.case, altered, 0)
        self.cache.get(self.case, call, 0)
        self.assertEqual((self.cache.built, self.cache.hits), (4, 1))

    def test_golden_edit_with_same_size_and_mtime_invalidates_cache(self):
        first = self.cache.get(self.case, self.call(), 0)
        path = self.test_dir / "add_golden.py"
        stat = path.stat()
        path.write_text(path.read_text().replace("torch.add(x, y)", "torch.sub(x, y)"))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        golden = self.cache_module.load_golden(path)
        cache = self.cache_module.CaseCache(self.test_dir, golden)
        second = cache.get(self.case, self.call(), 0)
        self.assertNotEqual(first["key"], second["key"])
        self.assertFalse(
            self.cache_module.torch.equal(first["expected"], second["expected"])
        )
        self.assertEqual(cache.built, 1)

    def test_corrupt_or_mismatched_entry_rebuilt_using_restricted_cpu_load(self):
        data = self.cache.get(self.case, self.call(), 0)
        path = next(self.cache.root.glob("*.pt"))
        for broken in (b"truncated cache", None):
            if broken is not None:
                path.write_bytes(broken)
            else:
                data["key"] = "stale key"
                self.cache_module.torch.save(data, path)
            with mock.patch.object(
                self.cache_module.torch, "load", wraps=self.cache_module.torch.load
            ) as load:
                result = self.cache.get(self.case, self.call(), 0)
            load.assert_called_once_with(path, map_location="cpu", weights_only=True)
            self.assertNotEqual(result["key"], "stale key")
        self.assertEqual(self.cache.built, 3)
        self.assertFalse(list(self.cache.root.glob("*.tmp")))

    def test_write_failure_propagates_and_cleans_temporary_file(self):
        with mock.patch.object(
            self.cache_module.torch, "save", side_effect=OSError("disk full")
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.cache.get(self.case, self.call(), 0)
        self.assertFalse(list(self.cache.root.iterdir()))

    def test_sequence_repeats_use_fresh_inputs_and_still_call_target(self):
        self.case["assertions_json"] = json.dumps(
            [
                {
                    "kind": "execution",
                    "repeat": 3,
                    "bitwise": True,
                    "calls": [{"inputs": PAIR, "seed": 43}],
                }
            ]
        )
        calls = []

        def target(x, y):
            calls.append(True)
            return x.add_(y)

        self.target.run = target
        first = self.run_case()
        second = self.run_case()
        self.assertEqual(first["status"], "passed", first["detail"])
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 12)
        self.assertEqual(self.counts, {"inputs": 4, "golden": 2})
        self.assertEqual(len(list(self.cache.root.glob("*.pt"))), 2)

    def test_implementation_changes_and_input_mutation_are_not_cached_as_success(self):
        self.case["assertions_json"] = json.dumps(
            [{"kind": "input_unchanged", "inputs": ["x"]}]
        )
        first = self.run_case()
        self.assertEqual(first["status"], "passed", first["detail"])
        self.target.run = lambda x, y: x.add_(y)
        failed = self.run_case()
        self.assertEqual(failed["status"], "failed")
        self.assertIn("input_unchanged failed", failed["detail"])
        self.target.run = lambda x, y: x + y
        self.assertEqual(self.run_case(), first)
        self.assertEqual(self.counts, {"inputs": 2, "golden": 1})

    def test_dst_args_and_multiple_outputs_get_fresh_zero_buffers(self):
        self.golden.OUTPUT_NAMES = ["out_add", "out_sub"]
        self.golden.GOLDEN_FUNCTION = lambda x, y: (x + y, x - y)
        self.case["output_json"] = json.dumps(
            {name: PAIR["x"] for name in self.golden.OUTPUT_NAMES}
        )
        self.case["assertions_json"] = json.dumps(
            [{"kind": "execution", "repeat": 2, "bitwise": True}]
        )
        torch = self.cache_module.torch

        def target(x, y, out_add, out_sub):
            self.assertEqual(torch.count_nonzero(out_add).item(), 0)
            self.assertEqual(torch.count_nonzero(out_sub).item(), 0)
            out_add.add_(x + y)
            out_sub.add_(x - y)

        self.target.run = target
        first = self.run_case()
        self.assertEqual(first["status"], "passed", first["detail"])
        self.assertEqual(self.run_case(), first)
        self.assertEqual(self.counts, {"inputs": 3, "golden": 1})

    def test_nan_result_detail_preserved_on_hit(self):
        inputs = copy.deepcopy(PAIR)
        inputs["x"]["fill"] = "nan_one"
        self.case["inputs_json"] = json.dumps(inputs)
        self.case["expected"] = "match_golden_nan"
        first = self.run_case()
        self.assertEqual(first["status"], "passed", first["detail"])
        self.assertEqual(self.run_case(), first)
        self.assertEqual(self.counts, {"inputs": 2, "golden": 1})

    def test_behavior_and_expected_error_do_not_compute_golden(self):
        self.case["expected"] = "assertions"
        self.case["assertions_json"] = json.dumps(
            [{"kind": "output_shape", "shape": [2, 8]}]
        )
        self.assertEqual(self.run_case()["status"], "passed")
        self.case["expected"] = "raises_error:shape_mismatch"
        self.case["assertions_json"] = "[]"

        def target(x, y):
            error = ValueError("fixture error")
            error.code = "shape_mismatch"
            raise error

        self.target.run = target
        first = self.run_case()
        self.assertEqual(first["status"], "passed", first["detail"])
        self.assertEqual(self.run_case(), first)
        self.assertEqual(self.counts, {"inputs": 2, "golden": 0})

    def test_tensor_lists_optional_values_and_noncontiguous_layout(self):
        self.case["expected"] = "assertions"
        call = {
            "inputs": {
                "items": {"tensors": [PAIR["x"], PAIR["y"]]},
                "optional": None,
                "scale": 0.25,
            },
            "seed": 42,
            "assertions": [],
        }
        first = self.cache.get(self.case, call, 0)
        second = self.cache.get(self.case, call, 0)
        self.cache_module.torch.testing.assert_close(first["inputs"], second["inputs"])
        self.assertIsNone(second["inputs"]["optional"])
        self.assertEqual(second["inputs"]["scale"], 0.25)
        base = self.golden.make_inputs
        self.golden.make_inputs = lambda *a, **kw: {
            name: value.t() for name, value in base(*a, **kw).items()
        }
        self.case["case_id"] = "TD-u01-strided"
        data = self.cache.get(self.case, self.call(), 0)
        loaded = self.cache.get(self.case, self.call(), 0)
        cloned = copy.deepcopy(loaded["inputs"])
        self.assertEqual(data["inputs"]["x"].stride(), cloned["x"].stride())
        self.assertFalse(cloned["x"].is_contiguous())


class DirectArtifactTests(unittest.TestCase):
    def test_failed_rerun_does_not_keep_success_and_paths_are_portable(self):
        with tempfile.TemporaryDirectory(prefix="dsl runner spaces ") as tmp:
            work = Path(tmp)
            fixture(work)
            prepared = run_script("prepare_golden.py", work)
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            test = work / "operators/add/test"
            target = test / "target_add.py"
            target.write_text("def run(x, y):\n    return x + y\n")
            env = dict(os.environ, CANNBOTDSL_OP_MODULE="target_add")
            command = [
                sys.executable,
                str(test / "test_add.py"),
                "--sheet",
                "whitebox",
                "--device",
                "cpu",
            ]
            report = test / "testcase_output/whitebox/verify_result_whitebox.csv"

            def invoke(*extra):
                return subprocess.run(
                    command + list(extra),
                    cwd=work.parent,
                    env=env,
                    capture_output=True,
                    text=True,
                )

            passed = invoke()
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertTrue(report.is_file())
            # Failure before loading cases must invalidate a previous success.
            rejected = invoke("--require-accelerator")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(report.exists())
            report.write_text("old success", encoding="utf-8")
            target.write_text(
                "def run(x, y):\n    raise RuntimeError('failed target')\n"
            )
            failed = invoke()
            self.assertNotEqual(failed.returncode, 0)
            with report.open(encoding="utf-8-sig") as handle:
                self.assertTrue(
                    all(r["status"] != "passed" for r in csv.DictReader(handle))
                )
            self.assertNotIn("old success", report.read_text())
            # A missing delivered dependency must not be mistaken for target success.
            target.write_text("import missing_delivered_error_matcher\n")
            missing = invoke()
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("missing_delivered_error_matcher", report.read_text())
            (test / "testcase.csv").unlink()
            self.assertNotEqual(invoke().returncode, 0)
            self.assertFalse(report.exists())
            self.assertFalse(list(report.parent.glob(".verify-*.tmp")))

    def test_complete_validation_has_no_intermediate_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            fixture(work)
            for name in (
                "build_blackbox_cases.py",
                "prepare_golden.py",
                "build_whitebox_cases.py",
                "build_tdd_cases.py",
            ):
                result = run_script(name, work)
                self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")
            self.assertFalse((work / "operators/add/test/intermediate").exists())
            test = work / "operators/add/test"
            expected = (
                (ROOT / "assets/test.py")
                .read_text()
                .replace('"__CANNBOTDSL_OP_NAME__"', repr("add"))
            )
            self.assertEqual((test / "test_add.py").read_text(), expected)
            for sheet in ("blackbox", "whitebox", "tdd"):
                self.assertTrue((test / "testcase_output" / sheet).is_dir())

    def test_blackbox_coverage_and_exclusion_are_read_from_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            path, rows = fixture(work)
            rows = [row for row in rows if row["case_id"] != "BB-bad-shape"]
            rows.append(
                row(
                    "blackbox",
                    "BB-exclude-shape",
                    design_ref="boundary_conditions[1]",
                    exclude_reason="target API does not expose this shape",
                )
            )
            write_rows(work, rows)
            result = run_script("build_blackbox_cases.py", work, "--coverage")
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(
                report["exclusions"]["boundary_conditions[1]"],
                "target API does not expose this shape",
            )
            self.assertFalse((path.parent / "intermediate").exists())

    def test_uncovered_spec_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _, rows = fixture(work)
            write_rows(work, [row for row in rows if row["case_id"] != "BB-bad-shape"])
            result = run_script("build_blackbox_cases.py", work)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("uncovered blackbox spec paths", result.stderr)

    def test_unknown_golden_design_reference_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            fixture(work)
            golden = work / "operators/add/test/add_golden.py"
            golden.write_text(
                golden.read_text().replace("Stage-Elementwise", "Stage-Missing"),
                encoding="utf-8",
            )
            result = run_script("prepare_golden.py", work)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("DESIGN_REFS", result.stderr)

    def test_obligation_kind_requires_matching_evidence(self):
        for kind in ("performance", "unknown", "semantic", "device"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                _, rows = fixture(work)
                native_path = work / "design/units/U01.yaml"
                native = json.loads(native_path.read_text())
                native["unit"]["verification_obligations"][0]["kind"] = kind
                native_path.write_text(json.dumps(native), encoding="utf-8")
                rejected = run_script("build_tdd_cases.py", work)
                self.assertNotEqual(rejected.returncode, 0, rejected.stdout)
                if kind in ("performance", "unknown"):
                    self.assertIn(
                        "unsupported verification obligation kind", rejected.stderr
                    )
                    continue
                tdd = next(row for row in rows if row["sheet"] == "tdd")
                tdd["expected"] = "assertions"
                assertion = (
                    {"kind": "output_device"}
                    if kind == "device"
                    else {"kind": "output_shape", "shape": [2, 8]}
                )
                tdd["assertions_json"] = json.dumps([assertion])
                write_rows(work, rows)
                accepted = run_script("build_tdd_cases.py", work)
                self.assertEqual(accepted.returncode, 0, accepted.stderr)

    def test_missing_tdd_obligation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _, rows = fixture(work)
            write_rows(work, [row for row in rows if row["sheet"] != "tdd"])
            result = run_script("build_tdd_cases.py", work)
            self.assertNotEqual(result.returncode, 0)

    def test_fixed_runner_executes_direct_csv_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            fixture(work)
            prepared = run_script("prepare_golden.py", work)
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            (work / "target_add.py").write_text(
                "import torch\ndef run(x, y):\n    return torch.add(x, y)\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["PYTHONPATH"] = str(work)
            env["CANNBOTDSL_OP_MODULE"] = "target_add"
            result = subprocess.run(
                [
                    sys.executable,
                    str(work / "operators/add/test/test_add.py"),
                    "--sheet",
                    "blackbox",
                    "--case-id",
                    "BB-core",
                    "--device",
                    "cpu",
                ],
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("total=1 failed=0", result.stdout)

    def test_legacy_structural_unit_requires_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            fixture(work)
            path = work / "design/units/U01.yaml"
            native = json.loads(path.read_text())
            native["unit"]["verification_obligations"][0]["kind"] = "structural"
            path.write_text(json.dumps(native))
            result = run_script("build_tdd_cases.py", work)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("U01/VO-U01", result.stderr)
            self.assertIn("review_evidence", result.stderr)

    def test_excluded_whitebox_is_not_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            fixture(work)
            result = run_script("prepare_golden.py", work)
            self.assertEqual(result.returncode, 0, result.stderr)
            import ast

            tree = ast.parse((ROOT / "assets/test.py").read_text(encoding="utf-8"))
            function = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "_rows"
            )
            context = {"CASEBOOK": work / "operators/add/test/testcase.csv", "csv": csv}
            exec(
                compile(
                    ast.Module(body=[function], type_ignores=[]), "<runner>", "exec"
                ),
                context,
            )
            self.assertEqual(
                [item["case_id"] for item in context["_rows"]("whitebox", None)],
                ["WB-B0"],
            )
            with self.assertRaisesRegex(ValueError, "no cases selected"):
                context["_rows"]("whitebox", "WB-B-excluded")
