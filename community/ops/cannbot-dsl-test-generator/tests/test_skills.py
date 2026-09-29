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
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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


class DirectArtifactTests(unittest.TestCase):
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

    def test_structural_unit_requires_real_probe_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _, rows = fixture(work)
            native_path = work / "design/units/U01.yaml"
            native = json.loads(native_path.read_text())
            native["unit"]["kind"] = "pipeline_framework"
            native["unit"]["verification_obligations"][0]["kind"] = "structural"
            native_path.write_text(json.dumps(native), encoding="utf-8")
            tdd = next(row for row in rows if row["sheet"] == "tdd")
            tdd["expected"] = "assertions"
            tdd["assertions_json"] = json.dumps(
                [{"kind": "output_shape", "shape": [2, 8]}]
            )
            write_rows(work, rows)
            rejected = run_script("build_tdd_cases.py", work)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("probe_events", rejected.stderr)
            tdd["assertions_json"] = json.dumps(
                [{"kind": "probe_events", "events": ["stage-0"]}]
            )
            write_rows(work, rows)
            accepted = run_script("build_tdd_cases.py", work)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)

    def test_runner_loads_nested_probe_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _, rows = fixture(work)
            tdd = next(row for row in rows if row["sheet"] == "tdd")
            tdd["expected"] = "assertions"
            tdd["assertions_json"] = json.dumps(
                [{"kind": "probe_events", "events": ["add"]}]
            )
            write_rows(work, rows)
            self.assertEqual(run_script("prepare_golden.py", work).returncode, 0)
            probe = work / "operators/add/probe"
            probe.mkdir()
            for package in (work / "operators", probe.parent, probe):
                (package / "__init__.py").touch()
            (probe / "add_probe.py").write_text(
                "events = []\ndef reset(): events.clear()\ndef snapshot(): return list(events)\n"
            )
            (probe.parent / "add.py").write_text(
                "from .probe import add_probe\ndef run(x, y):\n"
                "    add_probe.events.append('add')\n    return x + y\n"
            )

            env = dict(
                os.environ,
                PYTHONPATH=str(work),
                CANNBOTDSL_OP_MODULE="operators.add.add",
                CANNBOTDSL_PROBE_MODULE="operators.add.probe.add_probe",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(work / "operators/add/test/test_add.py"),
                    "--sheet",
                    "tdd",
                    "--device",
                    "cpu",
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

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
