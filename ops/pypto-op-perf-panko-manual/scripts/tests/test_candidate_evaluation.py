# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""A completed frozen evaluation must settle before another model turn."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent))
from _support import CHIP_ARGS, file_hash, init_argv, read_state, run_cli  # noqa: E402
import evaluate_candidate as trial  # noqa: E402
import session_worker  # noqa: E402
from bayesian_optimization.evaluator import RealEvaluator  # noqa: E402


KERNEL = """import pypto
@pypto.frontend.jit()
def demo(a, b):
    pypto.set_vec_tile_shapes(64, 128)
    return a + b
"""


class CandidateEvaluationCase(unittest.TestCase):
    eval_budget = 2
    source_name = None

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.op = Path(temp.name) / "demo"
        self.op.mkdir()
        self.source = self.op / (self.source_name or "demo_impl.py")
        self.source.write_text(KERNEL)
        opts = {"checkpoint_handoffs": True, "eval_budget": self.eval_budget,
                "bayesian_optimization_tile_search": False}
        override = self.source if self.source_name else None
        _, rc = run_cli(*init_argv(self.op, override, *CHIP_ARGS, "--device", "0", "--config", json.dumps(opts)))
        self.assertEqual(rc, 0)
        common = ("--op-dir", str(self.op))
        selected, rc = run_cli("select", *common)
        self.assertEqual(rc, 0)
        pending = selected["pending_code"]
        self.source.write_text(KERNEL.replace("64, 128", "32, 128"))
        _, rc = run_cli(
            "ack-code", *common, "--parent", pending["action_id"], "--base-hash", pending["base_hash"],
            "--intent", pending["intent"], "--summary", "changed the vector tile row count",
        )
        self.assertEqual(rc, 0)
        self.args = argparse.Namespace(op_dir=self.op, parent=pending["action_id"], code_hash=file_hash(self.source),
                                       test_command="python3 test_demo.py", lint_root=self.op, device="0")
        self.measurement = Mock()
        original = trial.invoke

        def run(script, *args):
            if script == "evaluation_session.py":
                return self.measurement(*args)
            return original(script, *args)

        mock = patch.object(trial, "invoke", side_effect=run)
        mock.start()
        self.addCleanup(mock.stop)
        self.addCleanup(self.close_workers)

    def close_workers(self):
        session_worker.close_worker(self.op / "optimization" / ".harness")


class CandidateEvaluationTest(CandidateEvaluationCase):
    def test_measures_conventional_path_when_init_omits_op_file(self):
        self.assertIsNone(read_state(self.op)["op_file"])
        self.measurement.return_value = {"s": 1, "metrics": {"p": 200}}
        result = trial.settle(self.args)
        self.assertTrue(result["record"]["improved"])
        arguments = self.measurement.call_args.args
        self.assertEqual(arguments[arguments.index("--op-file") + 1], self.source)

    def test_timeout_is_charged_once_and_restores_baseline(self):
        self.measurement.return_value = {"s": 0, "metrics": {"reason": "candidate timeout"}}
        settled = trial.settle(self.args)
        self.assertEqual(settled["s"], 0)
        self.assertFalse(settled["record"]["improved"])
        self.assertEqual(self.source.read_text(), KERNEL)
        self.assertTrue(trial.settle(self.args)["stop"])
        self.measurement.assert_called_once()
        state = read_state(self.op)
        self.assertEqual(state["progress"]["evals_used"], 2)
        self.assertEqual(state["progress"]["stop_reason"], "eval_budget")
        self.assertEqual(state["eval_cache"][self.args.code_hash]["s"], 0)

    def test_record_interruption_reuses_measurement_and_frozen_settings(self):
        self.measurement.return_value = {"s": 1, "metrics": {"p": 1.5, "aic_util": 31.0}}
        original = trial.controller

        def interrupted(directory, command, *args):
            if command == "record":
                raise OSError("interrupted before record")
            return original(directory, command, *args)

        with patch.object(trial, "controller", side_effect=interrupted), self.assertRaises(OSError):
            trial.settle(self.args)
        self.assertEqual(read_state(self.op)["progress"]["evals_used"], 1)
        with patch.object(self.args, "device", "1"), self.assertRaisesRegex(ValueError, "configuration"):
            trial.settle(self.args)
        with patch.object(self.args, "test_command", "python3 another_case.py"), self.assertRaisesRegex(
            ValueError, "another evaluation configuration"
        ):
            trial.settle(self.args)
        settled = trial.settle(self.args)
        self.assertTrue(settled["record"]["improved"])
        self.measurement.assert_called_once()
        state = read_state(self.op)
        self.assertEqual(state["eval_cache"][self.args.code_hash]["m"]["aic_util"], 31.0)
        self.assertEqual(state["progress"]["evals_used"], 2)

    def test_raw_receipt_requires_a_fresh_measurement_before_record(self):
        start = read_state(self.op)["progress"]["_start_ts"]
        receipt = self.op / "optimization" / "measurements" / f"{start}-{self.args.code_hash}.json"
        receipt.parent.mkdir()
        raw = {"s": 1, "metrics": {"p": 1.5}}
        receipt.write_text(json.dumps(raw), encoding="utf-8")
        self.measurement.side_effect = [
            {"s": 0, "metrics": {"fault_kind": "device", "reason": "device busy"}},
            {"s": 0, "metrics": {"reason": "candidate timeout"}},
        ]

        with self.assertRaisesRegex(ValueError, "environment fault"):
            trial.settle(self.args)
        self.assertEqual(json.loads(receipt.read_text()), raw)
        self.assertEqual(read_state(self.op)["progress"]["evals_used"], 1)

        settled = trial.settle(self.args)
        saved = json.loads(receipt.read_text())
        self.assertEqual(settled["s"], 0)
        self.assertEqual(saved["binding"]["command"], self.args.test_command)
        self.assertEqual(saved["metrics"]["reason"], "candidate timeout")
        self.assertEqual(self.source.read_text(), KERNEL)
        state = read_state(self.op)
        self.assertEqual(state["eval_cache"][self.args.code_hash]["s"], 0)
        self.assertEqual(state["progress"]["evals_used"], 2)
        self.assertTrue(trial.settle(self.args)["stop"])
        self.assertEqual(self.measurement.call_count, 2)

    def test_active_search_uses_recorded_result_before_raw_receipt(self):
        state = read_state(self.op)
        baseline = state["nodes"]["closed"][0]["code_hash"]
        start = state["progress"]["_start_ts"]
        receipt = self.op / "optimization" / "measurements" / f"{start}-{baseline}.json"
        receipt.parent.mkdir()
        receipt.write_text(json.dumps({"s": 0, "metrics": {}}), encoding="utf-8")

        with patch.object(self.args, "code_hash", baseline):
            hit = trial.settle(self.args)
        self.assertTrue(hit["hit"])
        self.assertEqual((hit["s"], hit["p"]), (1, 500))
        self.measurement.assert_not_called()
        self.assertEqual(read_state(self.op)["progress"]["evals_used"], 1)

    def test_environment_fault_keeps_handoff_unbilled(self):
        self.measurement.return_value = {"s": 0, "metrics": {"fault_kind": "device", "reason": "device busy"}}
        with self.assertRaisesRegex(ValueError, "environment fault"):
            trial.settle(self.args)
        self.assertFalse(list((self.op / "optimization/measurements").glob("*.json")))
        self.assertEqual(read_state(self.op)["progress"]["evals_used"], 1)

    def test_concurrent_retries_share_the_receipt_and_budget(self):
        self.measurement.return_value = {"s": 1, "metrics": {"p": 1.5}}
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(trial.settle, (self.args, self.args)))
        self.measurement.assert_called_once()
        self.assertEqual(sum(row.get("s") == 1 for row in results), 1)
        self.assertEqual(read_state(self.op)["progress"]["evals_used"], 2)

    def test_frozen_evaluator_identifies_timeout_stage(self):
        evaluator = RealEvaluator("demo", str(self.op), str(self.source), "0", "python3 test_demo.py")
        with patch.object(evaluator, "_run_test", return_value=124):
            score, metrics = evaluator.evaluate_current()
        self.assertEqual((score, metrics["reason"], metrics["test_rc"]), (0, "test_command timed out", 124))
        timeout = subprocess.TimeoutExpired("measure", 300, output=b"partial profile")
        with (
            patch.object(evaluator, "_run_test", return_value=0),
            patch.object(evaluator, "_lint_ok", return_value=True),
            patch("subprocess.run", side_effect=timeout),
        ):
            score, metrics = evaluator.evaluate_current()
        self.assertEqual(
            (score, metrics["reason"], metrics["detail"]), (0, "perf measurement timed out", "partial profile")
        )

    def test_stale_generation_cannot_record(self):
        args = ("record", "--op-dir", str(self.op), "--parent", self.args.parent,
                "--code-hash", self.args.code_hash, "--s", "0", "--run-start", "0")
        result, rc = run_cli(*args)
        self.assertEqual((rc, result["error"]), (3, "stale_run"))
        self.assertEqual(read_state(self.op)["progress"]["evals_used"], 1)

    def test_expired_search_returns_the_controller_stop_without_late_credit(self):
        self.measurement.return_value = {"s": 1, "metrics": {"p": 1.5}}
        replies = [
            {"stop": False},
            trial.CommandError("record", '{"error":"stop_required"}', 3),
            {"stop": True, "reason": "wall_clock"},
        ]
        with patch.object(trial, "controller", side_effect=replies):
            reply = trial.settle(self.args)
        self.assertIs(reply["stop"], True)
        self.assertFalse(reply["recorded"])
        self.assertEqual(reply["reason"], "wall_clock")
        self.assertTrue(Path(reply["receipt"]).is_file())
        self.assertEqual(read_state(self.op)["progress"]["evals_used"], 1)


class DuplicateCandidateTest(CandidateEvaluationCase):
    eval_budget = 4
    source_name = "alternative_impl.py"

    def test_repeated_source_releases_its_handoff_without_another_measurement(self):
        candidate = self.source.read_bytes()
        self.measurement.return_value = {"s": 1, "metrics": {"p": 650}}
        first = trial.settle(self.args)
        self.assertFalse(first["record"]["improved"])
        common = ("--op-dir", str(self.op), "--parent", self.args.parent)
        intent = "try the narrower tile again"
        handoff, rc = run_cli("handoff", *common, "--intent", intent)
        self.assertEqual(rc, 0, handoff)
        self.source.write_bytes(candidate)
        ack, rc = run_cli("ack-code", *common, "--base-hash", handoff["base_hash"],
                          "--intent", intent, "--summary", "reapplied the earlier tile")
        self.assertEqual(rc, 0, ack)

        settled = trial.settle(self.args)
        self.assertTrue(settled["record"]["duplicate"])
        state = read_state(self.op)
        self.assertFalse(state["progress"].get("pending_code"))
        self.assertEqual(self.source.read_text(), KERNEL)
        self.assertEqual(state["progress"]["evals_used"], 2)
        self.assertEqual(len(state["code_handoffs"]), 1)

        next_code, rc = run_cli("handoff", *common, "--intent", "try a wider tile")
        self.assertEqual(rc, 0, next_code)
        replay = trial.settle(self.args)
        self.assertTrue(replay["hit"])
        self.assertEqual(read_state(self.op)["progress"]["pending_code"], next_code)
        self.measurement.assert_called_once()
