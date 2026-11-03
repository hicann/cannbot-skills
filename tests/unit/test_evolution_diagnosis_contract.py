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
"""CPU-only regressions for the evolution diagnosis → facts → strategy contract."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "community/ops"
WORLD_MODEL = OPS / "ascendc-evolution-world-model/scripts"
STRATEGIES = OPS / "ascendc-evolution-strategies"
AIC_LABELS = {"cube_bound", "fixp_bound", "mte1_stall"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class EvolutionDiagnosisContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiling = load_module(
            "profiling_evidence", WORLD_MODEL / "profiling_evidence.py"
        )
        cls.generator = load_module(
            "diagnosis_index_generator", STRATEGIES / "scripts/generate_index_json.py"
        )
        cls.query = load_module(
            "diagnosis_strategy_query", STRATEGIES / "scripts/query_strategies.py"
        )
        cls.index = json.loads((STRATEGIES / "references/INDEX.json").read_text())

    def test_aic_labels_are_accepted_and_unknown_labels_still_rejected(self):
        report = self.profiling.validate_labels(sorted(AIC_LABELS))
        self.assertTrue(report["valid"], report)
        report = self.profiling.validate_labels(
            ["mte2_stall", "cube_bound", "not_a_pipe"]
        )
        self.assertFalse(report["valid"])
        self.assertEqual(report["unknown"], ["not_a_pipe"])

    def test_aic_vocabulary_reaches_generators_queries_and_gain_mapping(self):
        for vocabulary in (
            self.profiling.KNOWN_BOTTLENECK_LABELS,
            self.generator.BOTTLENECK_LABELS,
            self.query.VALID_BOTTLENECKS,
            self.index["vocabulary"]["bottleneck_labels"],
        ):
            with self.subTest(vocabulary=vocabulary):
                self.assertTrue(AIC_LABELS <= set(vocabulary))
        self.assertEqual(
            set(self.profiling.LABEL_TO_FACTS_RATIO_KEYS),
            self.profiling.KNOWN_BOTTLENECK_LABELS,
        )

    def test_aic_pipeline_facts_normalize_ratios_and_percentages(self):
        for scale in (1, 100):
            data = {
                "pipeline": {
                    "aic_cube_ratio": 0.8 * scale,
                    "aic_fixpipe_ratio": 0.2 * scale,
                    "aic_mte1_ratio": 0.4 * scale,
                    "aiv_vec_ratio": 0.1 * scale,
                }
            }
            before = copy.deepcopy(data)
            with self.subTest(scale=scale):
                facts = self.profiling.extract_facts(data)
                for key, expected in (
                    ("cube_ratio", 0.8),
                    ("fixpipe_ratio", 0.2),
                    ("mte1_ratio", 0.4),
                ):
                    self.assertAlmostEqual(facts[key], expected)
                self.assertEqual(facts["dominant_pipe"], "cube")
                self.assertEqual(data, before)

    def test_cube_metric_aliases_and_explicit_percent_units(self):
        cases = [
            ({"aic_mac_ratio": "0.7"}, 0.7),
            ({"cube_utilization(%)": 70}, 0.7),
            ({"cube_utilization(%)": 0.5}, 0.005),
            ({"cube_utilization(%)": 1}, 0.01),
            ({"aic_cube_ratio": 0, "aic_mac_ratio": 0.8, "cube_utilization(%)": 90}, 0),
            ({"aic_cube_ratio": None, "aic_mac_ratio": 0.6}, 0.6),
            ({"aic_cube_ratio": "N/A", "aic_mac_ratio": 0.6}, 0.6),
        ]
        for pipeline, expected in cases:
            with self.subTest(pipeline=pipeline):
                self.assertAlmostEqual(
                    self.profiling.extract_facts({"pipeline": pipeline})["cube_ratio"],
                    expected,
                )

    def test_missing_aic_metrics_remain_unknown(self):
        facts = self.profiling.extract_facts(
            {
                "pipeline": {
                    "aiv_vec_ratio": 0.6,
                    "aic_mac_ratio": "N/A",
                    "aic_fixpipe_ratio": None,
                }
            }
        )
        for key in ("cube_ratio", "fixpipe_ratio", "mte1_ratio"):
            self.assertIsNone(facts[key])
        self.assertEqual(facts["dominant_pipe"], "vec")
        self.assertIsNone(self.profiling.extract_facts({}))

    def test_zero_aic_metrics_are_observed_zero(self):
        facts = self.profiling.extract_facts(
            {
                "pipeline": {
                    "aic_cube_ratio": 0,
                    "aic_fixpipe_ratio": 0,
                    "aic_mte1_ratio": 0,
                }
            }
        )
        self.assertEqual(
            [facts[k] for k in ("cube_ratio", "fixpipe_ratio", "mte1_ratio")], [0, 0, 0]
        )
        self.assertIsNone(facts["dominant_pipe"])

    def test_each_aic_pipe_can_be_dominant(self):
        for field, pipe in (
            ("aic_cube_ratio", "cube"),
            ("aic_fixpipe_ratio", "fixpipe"),
            ("aic_mte1_ratio", "mte1"),
        ):
            with self.subTest(field=field):
                facts = self.profiling.extract_facts(
                    {"pipeline": {field: 0.8, "aiv_vec_ratio": 0.2}}
                )
                self.assertEqual(facts["dominant_pipe"], pipe)

    def test_deep_profiling_keeps_priority_and_fills_missing_aic_metrics(self):
        facts = self.profiling.extract_facts(
            {
                "profiling_analysis": {
                    "aic_cube_ratio": 0,
                    "aic_mte1_ratio": 0.3,
                    "pattern_type": "periodic",
                },
                "pipeline": {
                    "aic_cube_ratio": 0.9,
                    "aic_fixpipe_ratio": 0.2,
                    "aic_mte1_ratio": 0.8,
                },
            }
        )
        self.assertEqual(facts["_source"], "deep_profiling")
        self.assertEqual(facts["pattern_type"], "periodic")
        self.assertEqual(facts["cube_ratio"], 0)
        self.assertEqual(facts["mte1_ratio"], 0.3)
        self.assertEqual(facts["fixpipe_ratio"], 0.2)

    def test_csv_synthesis_does_not_hide_original_aic_metrics(self):
        pipeline = {
            "aic_cube_ratio": 0.8,
            "aic_fixpipe_ratio": 0.2,
            "aic_mte1_ratio": 0.4,
            "aiv_vec_ratio": 0.1,
        }
        analysis = self.profiling.synthesize_analysis_from_pipeline(pipeline)
        facts = self.profiling.extract_facts(
            {"profiling_analysis": analysis, "pipeline": pipeline}
        )
        self.assertEqual(facts["cube_ratio"], 0.8)
        self.assertEqual(facts["fixpipe_ratio"], 0.2)
        self.assertEqual(facts["mte1_ratio"], 0.4)

    def test_vector_facts_and_gain_remain_compatible(self):
        facts = self.profiling.extract_facts(
            {
                "pipeline": {
                    "aiv_mte2_ratio": 0.6,
                    "aiv_vec_ratio": 0.2,
                    "aiv_scalar_ratio": 0.1,
                }
            }
        )
        self.assertEqual(facts["dominant_pipe"], "mte2")
        self.assertEqual(facts["mte2_ratio"], 0.6)
        self.assertAlmostEqual(
            self.profiling.compute_expected_gain(["P1"], facts, ["mte2_stall"]), 1.5
        )

    def test_new_labels_retrieve_real_public_strategies_and_use_measured_gain(self):
        cases = [
            ("cube_bound", "aic_cube_ratio", "P122", 0.5),
            ("mte1_stall", "aic_mte1_ratio", "P122", 0.4),
            ("fixp_bound", "aic_fixpipe_ratio", "P66", 0.2),
        ]
        for label, metric, strategy, ratio in cases:
            with self.subTest(label=label):
                facts = self.profiling.extract_facts({"pipeline": {metric: ratio}})
                candidates = self.profiling.match_strategies_by_labels([label])
                self.assertIn(strategy, candidates["candidate_ids"])
                self.assertNotIn(label, candidates.get("unknown_labels", []))
                gain = self.profiling.compute_expected_gain([strategy], facts, [label])
                self.assertAlmostEqual(gain, 1 / (1 - ratio) - 1)
                self.assertEqual(
                    self.profiling.compute_expected_gain([strategy], {}, [label]), 0
                )

    def test_checked_in_index_matches_generated_metadata(self):
        okf = self.generator.discover_okf()
        self.assertEqual(self.generator.discover_cards() + okf, self.index["cards"])
        self.assertEqual(len(okf), self.index["stats"]["okf"])
        self.assertEqual(
            self.generator.BOTTLENECK_LABELS,
            self.index["vocabulary"]["bottleneck_labels"],
        )

    def test_fixpipe_query_cli_accepts_label_and_retrieves_alignment_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(STRATEGIES / "scripts/query_strategies.py"),
                    "--bottleneck",
                    "fixp_bound",
                ],
                cwd=tmp,
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertGreater(report["total_after_filter"], 0)
        self.assertIn("P66", report["matched"])

    def test_strict_diagnosis_cli_accepts_new_labels_and_rejects_unknown(self):
        for label, expected_code in [(label, 0) for label in sorted(AIC_LABELS)] + [
            ("not_a_pipe", 2)
        ]:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                wm_path = Path(tmp) / "world_model.json"
                wm_path.write_text(
                    json.dumps(
                        {
                            "decision_tree": {
                                "nodes": {
                                    "n1": {
                                        "status": "passed",
                                        "solution_ref": "round_1/parallel_0",
                                        "diagnosis": {
                                            "bottleneck_labels": [label],
                                            "confidence": 0.8,
                                            "diagnosis_text": (
                                                "Measured pipeline utilization supports "
                                                "inspecting this bottleneck."
                                            ),
                                            "next_round_hint": {
                                                "prefer": [],
                                                "avoid": [],
                                                "rationale": "Inspect measured pipeline data.",
                                            },
                                        },
                                    }
                                }
                            }
                        }
                    )
                )
                result = subprocess.run(
                    [
                        sys.executable,
                        str(WORLD_MODEL / "wm_ops.py"),
                        "validate-diagnosis",
                        "--wm-path",
                        str(wm_path),
                        "--strict",
                    ],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    timeout=30,
                )
                self.assertEqual(
                    result.returncode, expected_code, result.stderr + result.stdout
                )
                self.assertEqual(json.loads(result.stdout)["valid"], expected_code == 0)


if __name__ == "__main__":
    unittest.main()
