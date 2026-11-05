#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU regression and round replay for evaluation evidence consumption."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "community/ops/ascendc-evolution-world-model/scripts"
sys.path.insert(0, str(SCRIPTS))

import check_round_artifacts as artifacts  # noqa: E402
import evaluation_contract as contract  # noqa: E402
import wm_ledger  # noqa: E402
import wm_ops  # noqa: E402


def flat(speedup=1.35):
    return {
        "compilation_success": True,
        "precision_passed": True,
        "speedup": speedup,
        "cv_pct": 3.0,
        "n_cases_total": 3,
        "evolved": {"pipeline": {"aiv_vec_ratio": 0.6}},
    }


def comparison(speedup=1.35):
    return {
        "comparison": {
            "compilation_success": True,
            "precision_passed": True,
            "speedup": speedup,
            "cv_pct": 3.0,
        },
        "n_cases_total": 3,
    }


def multi():
    return {
        "shape_results": {
            "target": [
                {
                    "compilation_success": True,
                    "precision_passed": True,
                    "speedup": 2.0,
                    "cv_pct": 2.0,
                },
                {
                    "compilation_success": True,
                    "precision_passed": True,
                    "speedup": 1.35,
                    "cv_pct": 2.0,
                },
            ],
            "generalization": [],
        },
        "aggregate": {"target_geo_mean_speedup": 1.64, "target_min_speedup": 1.35},
        "n_cases_total": 3,
    }


class EvaluationContractTest(unittest.TestCase):
    def test_equivalent_formats(self):
        for factory in (flat, comparison, multi):
            with self.subTest(format=factory.__name__):
                data = factory()
                original = copy.deepcopy(data)
                result = contract.normalize_eval_result(data)
                self.assertTrue(result["compile"])
                self.assertTrue(result["precision"])
                self.assertEqual(result["speedup"], 1.35)
                self.assertEqual(result["measurement_quality"], "good")
                self.assertEqual(data, original)

    def test_flat_with_pipeline_does_not_become_comparison(self):
        result = contract.normalize_eval_result(flat())
        self.assertEqual(result["_source"], "flat")
        self.assertEqual(result["pipeline"], {"aiv_vec_ratio": 0.6})

    def test_false_precision_is_not_overridden(self):
        data = comparison()
        data["comparison"]["precision_passed"] = False
        data["evolved"] = {"precision_passed": True}
        self.assertFalse(contract.normalize_eval_result(data)["precision"])

    def test_nested_precision_fallback(self):
        data = comparison()
        del data["comparison"]["precision_passed"]
        data["evolved"] = {"precision_passed": False}
        self.assertFalse(contract.normalize_eval_result(data)["precision"])

    def test_quality_boundaries_and_unknown(self):
        cases = [
            (0, "good"),
            (4.99, "good"),
            (5, "acceptable"),
            (14.99, "acceptable"),
            (15, "noisy"),
            (20, "noisy"),
            (-1, "unknown"),
            (None, "unknown"),
            (True, "unknown"),
            (float("nan"), "unknown"),
            (float("inf"), "unknown"),
        ]
        for cv, expected in cases:
            with self.subTest(cv=cv):
                self.assertEqual(contract.measurement_quality({"cv_pct": cv}), expected)

    def test_explicit_quality_has_priority(self):
        self.assertEqual(
            contract.measurement_quality({"measurement_quality": "good", "cv_pct": 20}),
            "good",
        )
        for value in ("unknown", "invalid", [], 0):
            with self.subTest(quality=value):
                self.assertEqual(
                    contract.measurement_quality(
                        {"measurement_quality": value, "cv_pct": 0}
                    ),
                    "unknown",
                )

    def test_invalid_evidence_cannot_pass(self):
        cases = [
            None,
            [],
            {"foo": 1},
            {"comparison": []},
            {"shape_results": {}},
            {"shape_results": {"target": [None]}},
            {"shape_results": {"target": []}},
            dict(flat(), evolved=[]),
            dict(flat(), aggregate="bad"),
        ]
        for field, value in [
            ("compilation_success", "false"),
            ("precision_passed", 1),
            ("speedup", True),
            ("speedup", -1),
            ("speedup", 0),
            ("speedup", float("nan")),
            ("speedup", float("inf")),
        ]:
            data = flat()
            data[field] = value
            cases.append(data)
        for data in cases:
            with self.subTest(data=data):
                self.assertIsNone(contract.normalize_eval_result(data))

    def test_failed_evaluation_may_have_no_timing(self):
        data = flat()
        data.update(compilation_success=False, precision_passed=False, speedup=None)
        result = contract.normalize_eval_result(data)
        self.assertFalse(result["compile"])
        self.assertIsNone(result["speedup"])

    def test_multi_shape_uses_slowest_target(self):
        data = multi()
        data["aggregate"]["target_min_speedup"] = 1.8
        self.assertEqual(contract.normalize_eval_result(data)["speedup"], 1.35)
        del data["aggregate"]["target_min_speedup"]
        self.assertEqual(contract.normalize_eval_result(data)["speedup"], 1.35)

    def test_multi_shape_checks_every_target(self):
        data = multi()
        data["shape_results"]["target"][1]["precision_passed"] = False
        data["comparison"] = comparison()["comparison"]
        self.assertFalse(contract.normalize_eval_result(data)["precision"])

    def test_generalization_precision_failure_is_not_success(self):
        data = multi()
        data["shape_results"]["generalization"] = [
            {"compilation_success": True, "precision_passed": False, "speedup": 1.1}
        ]
        result = contract.normalize_eval_result(data)
        self.assertFalse(result["precision"])
        self.assertIn("generalization_precision_failed", result["_warnings"])

    def test_multi_shape_uses_worst_quality(self):
        data = multi()
        data["comparison"] = {"measurement_quality": "good"}
        data["shape_results"]["target"][1]["cv_pct"] = 25.0
        self.assertEqual(
            contract.normalize_eval_result(data)["measurement_quality"], "noisy"
        )
        del data["comparison"]
        del data["shape_results"]["target"][1]["cv_pct"]
        self.assertEqual(
            contract.normalize_eval_result(data)["measurement_quality"], "unknown"
        )

    def test_nested_null_pipeline_does_not_revive_top_level(self):
        data = flat()
        data["evolved"]["pipeline"] = None
        data["pipeline"] = {"aiv_vec_ratio": 0.6}
        self.assertIsNone(contract.normalize_eval_result(data)["pipeline"])


class EvaluationRoundTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / "round_1"
        (self.root / "shared").mkdir()
        (self.root / "shared/probe.json").write_text("{}\n" * 3)
        self.nodes = {
            "root": {
                "id": "root",
                "parent_id": None,
                "status": "completed",
                "score": 1.0,
                "difficulty": 1,
                "depth": 0,
                "children": [],
                "strategy_combination": [],
                "mode": "completed",
            }
        }
        self.wm = {
            "best_score": 1.0,
            "stagnation_count": 0,
            "target_speedup": 1.2,
            "decision_tree": {"nodes": self.nodes},
            "session": {"op_name": "probe"},
        }
        self.mapping = {}

    def add_variant(self, data):
        slot = str(len(self.mapping))
        nid = "n" + slot
        self.mapping[slot] = nid
        self.nodes["root"]["children"].append(nid)
        self.nodes[nid] = {
            "id": nid,
            "parent_id": "root",
            "status": "in_progress",
            "score": None,
            "difficulty": 2,
            "depth": 1,
            "children": [],
            "strategy_combination": ["P1"],
            "mode": "strategy_guided",
        }
        directory = self.results / f"parallel_{slot}"
        directory.mkdir(parents=True)
        (directory / "evaluation_results.json").write_text(json.dumps(data))
        return nid, directory

    def refine(self):
        return wm_ops.refine(self.wm, 1, str(self.results), self.mapping)

    def test_round_agrees_with_artifact_gate_and_lineage(self):
        for factory in (flat, comparison, multi):
            self.add_variant(factory())
        result = self.refine()
        self.assertIn("3/3 passed", result["round_summary"])
        wm_ledger._maybe_write_ledger(
            str(self.root / "world_model.json"),
            self.wm,
            1,
            self.mapping,
            str(self.results),
        )
        entries = [
            json.loads(line)
            for line in (self.root / "artifacts/lineage.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(entries), 3)
        for slot, nid in self.mapping.items():
            gate = artifacts._check_eval_json(str(self.results / f"parallel_{slot}"))
            entry = next(item for item in entries if item["node_id"] == nid)
            self.assertTrue(gate["valid"])
            self.assertEqual(entry["status"], "passed")
            self.assertEqual(entry["speedup"], self.nodes[nid]["score"])
            self.assertEqual(entry["measurement_quality"], "good")
        self.assertEqual(self.wm["best_score"], 1.35)
        self.assertIn("1.350", (self.root / "artifacts/attempt-ledger.md").read_text())

    def test_noisy_three_percent_gain_does_not_clear_stagnation(self):
        data = comparison(1.03)
        data["comparison"]["cv_pct"] = 20.0
        nid, _ = self.add_variant(data)
        result = self.refine()
        self.assertEqual(result["worst_quality"], "noisy")
        self.assertEqual(self.wm["stagnation_count"], 1)
        self.assertEqual(self.nodes[nid]["measurement_quality"], "noisy")

    def test_unknown_quality_uses_conservative_branch_thresholds(self):
        data = flat(1.1)
        del data["cv_pct"]
        nid, _ = self.add_variant(data)
        self.refine()
        children = [self.nodes[c] for c in self.nodes[nid]["children"]]
        self.assertEqual(self.nodes[nid]["measurement_quality"], "unknown")
        self.assertFalse(any("深度探索" in c.get("description", "") for c in children))

    def test_missing_quality_does_not_clear_small_gain_stagnation(self):
        data = flat(1.03)
        del data["cv_pct"]
        self.add_variant(data)
        self.assertEqual(self.refine()["worst_quality"], "unknown")
        self.assertEqual(self.wm["stagnation_count"], 1)

    def test_explicit_good_quality_preserves_existing_thresholds(self):
        data = comparison(1.03)
        data["comparison"].update(measurement_quality="good", cv_pct=20)
        self.add_variant(data)
        self.refine()
        self.assertEqual(self.wm["stagnation_count"], 0)

    def test_invalid_slot_does_not_abort_successful_sibling(self):
        invalid, directory = self.add_variant(flat())
        (directory / "evaluation_results.json").write_text('{"partial":')
        valid, _ = self.add_variant(flat())
        self.refine()
        self.assertEqual(self.nodes[invalid]["status"], "failed")
        self.assertEqual(self.nodes[valid]["status"], "passed")

    def test_invalid_numeric_speedup_never_reaches_best(self):
        for value in (float("nan"), float("inf"), True, -1):
            self.add_variant(flat(value))
        self.refine()
        self.assertEqual(self.wm["best_score"], 1.0)
        self.assertTrue(
            all(self.nodes[n]["status"] == "failed" for n in self.mapping.values())
        )

    def test_case_subset_and_missing_count_remain_rejected(self):
        subset = flat()
        subset["n_cases_total"] = 2
        missing = flat()
        del missing["n_cases_total"]
        for data in (subset, missing):
            self.add_variant(data)
        self.refine()
        for nid in self.mapping.values():
            self.assertEqual(self.nodes[nid]["status"], "failed")
            self.assertIn("case_", self.nodes[nid]["failure_reason"])
        self.assertEqual(self.wm["best_score"], 1.0)

    def test_case_rejection_is_not_a_passed_lineage_entry(self):
        data = flat()
        data["n_cases_total"] = 2
        nid, _ = self.add_variant(data)
        self.refine()
        self.assertFalse(wm_ledger._is_lineage_passed(self.nodes[nid]))
        row = wm_ledger._build_ledger_row(
            1, "0", nid, self.nodes[nid], str(self.results)
        )
        self.assertNotIn("1.350", row)

    def test_null_comparison_failure_is_handled(self):
        data = flat()
        data.update(comparison=None, precision_passed=False)
        nid, _ = self.add_variant(data)
        self.refine()
        self.assertEqual(self.nodes[nid]["status"], "failed")

    def test_same_run_canonical_score_is_recorded_in_lineage(self):
        nid, directory = self.add_variant(flat(1.49))
        (directory / "performance.json").write_text(
            json.dumps({"geomean_ref_us": 68, "geomean_asc_us": 100})
        )
        (self.root / "baseline_evaluation.json").write_text(
            json.dumps({"base_time_ms_avg": 9999})
        )
        self.refine()
        self.assertAlmostEqual(self.nodes[nid]["score"], 0.68)
        entry = wm_ledger._build_lineage_entry(
            nid,
            self.nodes[nid],
            wm_ledger._LineageContext(self.nodes, 1, str(self.results)),
            "0",
        )
        self.assertEqual(entry["speedup"], 0.68)
        self.assertEqual(entry["reported_speedup"], 1.49)
        self.assertEqual(self.wm["best_score"], 1.0)

    def test_precision_failure_cannot_be_rescued_by_fast_canonical_time(self):
        data = flat()
        data["precision_passed"] = False
        nid, directory = self.add_variant(data)
        (directory / "performance.json").write_text(
            json.dumps({"geomean_ref_us": 100, "geomean_asc_us": 1})
        )
        self.refine()
        self.assertEqual(self.nodes[nid]["status"], "failed")
        self.assertEqual(self.wm["best_score"], 1.0)
        self.assertFalse(self.nodes[nid]["precision_passed"])

    def test_profiling_and_precision_warning_use_same_format_contract(self):
        for factory in (flat, comparison, multi):
            self.add_variant(factory())
        self.assertEqual(
            wm_ops._collect_missing_profiling(str(self.results), self.mapping),
            ["0", "1", "2"],
        )
        failed = comparison()
        failed["comparison"]["precision_passed"] = False
        self.add_variant(failed)
        self.assertEqual(
            wm_ops._count_precision_failures(str(self.results), self.mapping), (4, 1)
        )
        self.assertNotIn(
            "3", wm_ops._collect_missing_profiling(str(self.results), self.mapping)
        )

    def test_cli_refine_writes_consistent_artifacts(self):
        self.add_variant(flat())
        path = self.root / "world_model.json"
        path.write_text(json.dumps(self.wm))
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "wm_ops.py"),
                "refine",
                "--wm-path",
                str(path),
                "--round",
                "1",
                "--results-dir",
                str(self.results),
                "--parallel-map",
                json.dumps(self.mapping),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        saved = json.loads(path.read_text())
        self.assertEqual(saved["decision_tree"]["nodes"]["n0"]["status"], "passed")
        self.assertTrue((self.root / "artifacts/lineage.jsonl").exists())

    def test_real_multi_shape_producer_preserves_regression_gating(self):
        script = (
            REPO / "community/ops/ascendc-ops-evaluation/scripts/evaluate_ops_direct.py"
        )
        spec = importlib.util.spec_from_file_location("contract_eval_producer", script)
        producer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = producer
        spec.loader.exec_module(producer)
        baseline, evolved, precision = [], [], {}
        for name, speedup in (("fast", 2.0), ("slow", 0.9)):
            base = {
                "name": name,
                "group": "target",
                "time_us": 100.0,
                "precision_passed": True,
                "pipeline": {},
                "bottleneck": "unknown",
                "cv_pct": 2.0,
            }
            baseline.append(base)
            evolved.append(dict(base, time_us=100.0 / speedup))
            precision[name] = (True, "ok")
        shapes, compile_ok = producer.merge_per_shape_results(
            baseline, evolved, precision
        )
        aggregate = producer.compute_aggregate(
            shapes["target"], shapes["generalization"], 1.2
        )
        legacy = producer._synthesize_legacy_fields(shapes, aggregate)
        data = dict(
            legacy,
            shape_results=shapes,
            aggregate=aggregate,
            n_cases_total=3,
            gating=producer.determine_gating(
                aggregate, precision_ok=True, compile_ok=compile_ok
            ),
        )
        nid, _ = self.add_variant(data)
        self.refine()
        self.assertEqual(self.nodes[nid]["status"], "passed")
        self.assertAlmostEqual(self.nodes[nid]["score"], 0.9)
        self.assertFalse(self.nodes[nid]["parent_eligible"])
        self.assertEqual(self.nodes[nid]["gating"], "target_regression")


if __name__ == "__main__":
    unittest.main()
