# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The snapshot chain is a precondition, not a best effort.

Every cycle is anchored to `nodes/<code_hash>.py`: `select` restores the global
best into the working file and tells the coder to apply the delta on top of it,
`record` rolls a loser back to it, `stop` delivers it. When a link in that chain
is missing the harness used to carry on and report the failure in a field
nothing reads, so which program a candidate was actually built on depended on
the caller's path and on residue from earlier runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from _support import (
    CHIP_ARGS,
    HARNESS,
    feasible_argv,
    file_hash,
    harness,
    init_argv,
    read_state,
    record_argv,
    run_cli,
)
import harness_session

KERNEL = """import pypto


@pypto.frontend.jit(runtime_options={"device_sched_mode": 1})
def demo(a, b):
    pypto.set_vec_tile_shapes(64, 128)
    return a + b
"""


def _run(*args):
    """(payload, returncode). Refusals exit non-zero but still emit JSON."""
    return run_cli(*args)


class SnapshotContractTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.op_dir = Path(self._tmp.name) / "demo"
        self.op_dir.mkdir(parents=True)
        self.op_file = self.op_dir / "demo_impl.py"
        self.op_file.write_text(KERNEL, encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    # ── init ──────────────────────────────────────────────────────────────────

    def test_init_derives_the_baseline_hash_from_the_file(self):
        """`--preopt-hash` is optional. Without it the run used to start with an
        empty root hash, an empty eval cache and no snapshot at all.
        """
        res, rc = self._init()
        self.assertEqual(rc, 0)
        self.assertTrue(res["initialized"])
        self.assertTrue(res["preopt_snapshot"])
        digest = file_hash(self.op_file)
        self.assertEqual(res["preopt_hash"], digest)
        self.assertEqual([p.stem for p in self._nodes()], [digest])

    def test_init_refuses_a_hash_that_does_not_match_the_file(self):
        """The caller believes it measured one program; the file is another. The
        preopt latency every speedup is normalised against is then not the
        baseline's.
        """
        res, rc = self._init("--preopt-hash", "deadbeef")
        self.assertEqual(rc, 3)
        self.assertFalse(res["initialized"])
        self.assertEqual(res["error"], "preopt_hash_mismatch")

    def test_init_accepts_a_hash_that_does_match(self):
        digest = file_hash(self.op_file)
        res, rc = self._init("--preopt-hash", digest)
        self.assertEqual(rc, 0)
        self.assertTrue(res["initialized"])

    def test_init_refuses_an_unreadable_op_file(self):
        res, rc = self._init(op_file=self.op_dir / "missing.py")
        self.assertEqual(rc, 3)
        self.assertEqual(res["error"], "op_file_unreadable")

    def test_a_refused_init_leaves_no_state_behind(self):
        """A half-built run whose baseline is missing is worse than no run."""
        self._init("--preopt-hash", "deadbeef")
        self.assertFalse((self.op_dir / "optimization" / "search_state.json").exists())

    # ── select ────────────────────────────────────────────────────────────────

    def test_resume_keeps_edited_bytes_and_records_the_same_action_once(self):
        self._init("--config", '{"eval_budget":3,"bayesian_optimization_tile_search":false}')
        op_args = ("--op-dir", str(self.op_dir))
        selected, rc = _run("select", *op_args, "--resume")
        self.assertEqual(rc, 0)
        pending = selected["pending_code"]
        self.assertTrue(pending["coder"])
        self.assertEqual(pending["base_hash"], file_hash(self.op_file))
        self.op_file.write_text(KERNEL.replace("64, 128", "32, 128"))
        digest = file_hash(self.op_file)
        resumed, rc = _run("select", *op_args)
        self.assertEqual((rc, resumed["pending_code"]), (0, pending))
        self.assertEqual(file_hash(self.op_file), digest)
        self.assertEqual(read_state(self.op_dir)["progress"]["evals_used"], 1)
        record = ("record", *op_args, "--code-hash", digest, "--s", "1", "--p", "400")
        refused, rc = _run(*record, "--parent", "another-action")
        self.assertEqual((rc, refused["error"]), (3, "code_pending"))
        unacknowledged, rc = _run(*record, "--parent", pending["action_id"])
        self.assertEqual((rc, unacknowledged["error"]), (3, "code_ack_required"))
        ack = ("ack-code", *op_args, "--parent", pending["action_id"], "--base-hash", pending["base_hash"])
        mismatch, rc = _run(*ack, "--intent", "unrelated arithmetic change", "--summary", "replace division")
        self.assertEqual((rc, mismatch["error"]), (3, "code_intent_mismatch"))
        acknowledged, rc = _run(*ack, "--intent", pending["intent"], "--summary", "set the vector tile to 32 rows")
        self.assertEqual((rc, acknowledged["candidate_hash"]), (0, digest))
        self.op_file.write_text(KERNEL.replace("64, 128", "16, 128"))
        changed, rc = _run(*record, "--parent", pending["action_id"])
        self.assertEqual((rc, changed["error"]), (3, "candidate_changed_since_ack"))
        self.op_file.write_text(KERNEL.replace("64, 128", "32, 128"))
        self.assertEqual(_run(*record, "--parent", pending["action_id"])[1], 0)
        state = read_state(self.op_dir)
        self.assertEqual(state["code_handoffs"][0]["intent"], pending["intent"])
        self.assertEqual(state["code_handoffs"][0]["evaluated_hash"], digest)
        progress = state["progress"]
        self.assertEqual(progress["evals_used"], 2)
        self.assertIsNone(progress.get("pending_code"))

        args = ("handoff", *op_args, "--parent", pending["action_id"], "--intent", "use 16 rows")
        next_code, rc = _run(*args)
        self.assertEqual((rc, next_code["base_hash"]), (0, digest))
        self.assertEqual(_run(*args), (next_code, 0))
        self.assertEqual(_run("select", *op_args)[0]["pending_code"], next_code)
        self.assertEqual(_run("handoff", *op_args, "--parent", pending["action_id"], "--intent", "other")[1], 3)

    def test_controller_written_tile_resumes_without_dispatching_an_editor(self):
        self._init("--config", '{"eval_budget":2}')
        args = ["select", "--op-dir", str(self.op_dir), "--resume"]

        def write_tile(*_):
            self.op_file.write_text(KERNEL.replace("64, 128", "32, 128"))
            return {"applied": True, "code_hash": file_hash(self.op_file)}

        with (
            patch.object(harness, "_tile_bayes_propose", side_effect=write_tile),
            patch.object(harness_session.importlib, "import_module", return_value=harness),
        ):
            result = harness_session.execute({"op_dir": str(self.op_dir)}, args)
        self.assertEqual(result["returncode"], 0)
        pending = json.loads(result["stdout"])["pending_code"]
        self.assertFalse(pending["coder"])
        self.assertEqual(pending["candidate_hash"], file_hash(self.op_file))
        restored, rc = _run(*args)
        self.assertEqual((rc, restored["pending_code"]), (0, pending))
        self.assertEqual(file_hash(self.op_file), pending["candidate_hash"])

    def test_select_refuses_when_the_best_cannot_be_restored(self):
        self._init()
        for snap in self._nodes():
            snap.unlink()
        with self.op_file.open("a", encoding="utf-8") as f:
            f.write("\n# drifted\n")
        res, rc = _run("select", "--op-dir", str(self.op_dir))
        self.assertEqual(rc, 3)
        self.assertIsNone(res["selected"])
        self.assertEqual(res["error"], "restore_failed")

    def test_select_still_works_when_the_chain_is_intact(self):
        self._init()
        res, rc = _run("select", "--op-dir", str(self.op_dir))
        self.assertEqual(rc, 0)
        self.assertIsNotNone(res["selected"])
        self.assertTrue(res["selected"]["restored"])
        self.assertTrue(res["selected"]["restore_code_hash"])

    # ── stop ──────────────────────────────────────────────────────────────────

    def test_stop_refuses_to_record_a_stop_it_cannot_deliver(self):
        """`stop_reason` is sticky and the completion gate reads it, so recording
        it over some other program would complete Stage 7 on a kernel the search
        never chose.
        """
        self._init()
        for snap in self._nodes():
            snap.unlink()
        with self.op_file.open("a", encoding="utf-8") as f:
            f.write("\n# drifted\n")
        res, rc = _run("stop", "--op-dir", str(self.op_dir), "--force", "manual")
        self.assertEqual(rc, 3)
        self.assertFalse(res["stop"])
        self.assertEqual(res["error"], "restore_failed")
        state = json.loads(
            (self.op_dir / "optimization" / "search_state.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIsNone(state["progress"]["stop_reason"])  # not recorded

    def test_stop_does_not_refuse_when_the_file_already_is_the_best(self):
        """No false refusal: a missing snapshot does not matter if the working
        file is already the program that would have been restored.
        """
        self._init()
        for snap in self._nodes():
            snap.unlink()
        res, rc = _run("stop", "--op-dir", str(self.op_dir), "--force", "manual")
        self.assertEqual(rc, 0)
        self.assertTrue(res["stop"])

    # ── record ────────────────────────────────────────────────────────────────

    def test_record_halts_when_a_loser_cannot_be_rolled_back(self):
        """A refinement submits candidate after candidate for one action without
        passing through `select`, so a break here would go unnoticed until the
        next select -- with everything in between built on a rejected program.
        """
        self._init()
        _run("select", "--op-dir", str(self.op_dir))
        for snap in self._nodes():
            snap.unlink()
        with self.op_file.open("a", encoding="utf-8") as f:
            f.write("\n# a losing candidate\n")
        digest = file_hash(self.op_file)
        res, rc = _run(*record_argv(self.op_dir, digest, "900"))
        self.assertEqual(rc, 3)
        self.assertFalse(res["improved"])
        self.assertEqual(res["error"], "restore_failed")
        # The measurement was paid for, so it is still recorded.
        state = json.loads(
            (self.op_dir / "optimization" / "search_state.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn(digest, state["eval_cache"])

    def test_record_is_quiet_when_the_chain_is_intact(self):
        self._init()
        _run("select", "--op-dir", str(self.op_dir))
        with self.op_file.open("a", encoding="utf-8") as f:
            f.write("\n# a losing candidate\n")
        digest = file_hash(self.op_file)
        res, rc = _run(*record_argv(self.op_dir, digest, "900"))
        self.assertEqual(rc, 0)
        self.assertNotIn("error", res)
        # and the loser really was rolled back
        self.assertEqual(self.op_file.read_text(encoding="utf-8"), KERNEL)

    def test_budget_stop_delivers_the_last_improving_candidate(self):
        self._init("--config", '{"eval_budget":2}')
        _run("select", "--op-dir", str(self.op_dir))
        self.op_file.write_text(KERNEL.replace("64, 128", "32, 128"), encoding="utf-8")
        digest = file_hash(self.op_file)
        record, rc = _run(*record_argv(self.op_dir, digest, "400"))
        self.assertEqual(rc, 0)
        self.assertTrue(record["improved"])
        stop, rc = _run("stop", "--op-dir", str(self.op_dir))
        self.assertEqual(rc, 0)
        self.assertEqual(stop["reason"], "eval_budget")
        self.assertEqual(stop["best_code_hash"], digest)
        self.assertEqual(file_hash(self.op_file), digest)

    def test_baseline_uses_the_entire_budget(self):
        self._init("--config", '{"eval_budget":1}')
        op_args = ("--op-dir", str(self.op_dir))
        selection, rc = _run("select", *op_args)
        self.assertEqual(rc, 3)
        self.assertIsNone(selection["selected"])
        self.assertEqual(selection["reason"], "eval_budget")
        self.op_file.write_text(KERNEL.replace("a + b", "a - b"), encoding="utf-8")
        feasible, rc = _run(*feasible_argv(self.op_dir, self.op_file))
        self.assertEqual(rc, 3)
        self.assertEqual(feasible["error"], "stop_required")
        block, rc = _run("block", *op_args, "--parent", "u1", "--test-command", "true")
        self.assertEqual(rc, 3)
        self.assertEqual(block["error"], "stop_required")
        digest = file_hash(self.op_file)
        record, rc = _run(*record_argv(self.op_dir, digest, "400"))
        self.assertEqual(rc, 3)
        self.assertEqual(record["reason"], "eval_budget")
        self.assertNotIn(digest, read_state(self.op_dir)["eval_cache"])
        stop, rc = _run("stop", *op_args)
        self.assertEqual(rc, 0)
        self.assertEqual(stop["evals_used"], 1)
        self.assertEqual(stop["reason"], "eval_budget")
        self.assertEqual(self.op_file.read_text(encoding="utf-8"), KERNEL)

    def test_session_writes_terminal_when_baseline_reaches_target(self):
        session = [sys.executable, str(HARNESS.with_name("harness_session.py")),
                   "--op-dir", str(self.op_dir)]
        self.addCleanup(subprocess.run, [*session, "--close"], capture_output=True)
        initial = subprocess.run(
            [*session, "--", *init_argv(self.op_dir, self.op_file, *CHIP_ARGS, "--p-ref", "500")],
            capture_output=True, text=True,
        )
        self.assertEqual(initial.returncode, 0, initial.stderr)
        self.assertTrue(json.loads(initial.stdout)["initialized"])
        marker = self.op_dir / "optimization" / "controller-terminal.json"
        terminal = json.loads(marker.read_text(encoding="utf-8"))
        self.assertEqual(terminal["reason"], "success")
        self.assertEqual(terminal["delivered_code_hash"], file_hash(self.op_file))
        self.assertEqual(read_state(self.op_dir)["progress"]["evals_used"], 1)

    def test_session_records_its_last_candidate_and_keeps_the_terminal_on_resume(self):
        session = [sys.executable, str(HARNESS.with_name("harness_session.py")),
                   "--op-dir", str(self.op_dir)]
        self.addCleanup(subprocess.run, [*session, "--close"], capture_output=True)

        def run_session(*args):
            result = subprocess.run([*session, "--", *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

        run_session(*init_argv(self.op_dir, self.op_file, *CHIP_ARGS,
                              "--config", '{"eval_budget":2}'))
        optimization = self.op_dir / "optimization"
        marker = optimization / "controller-terminal.json"
        self.assertFalse(marker.exists())
        op_args = ("--op-dir", str(self.op_dir))
        self.assertIsNotNone(run_session("select", *op_args)["selected"])
        self.op_file.write_text(KERNEL.replace("64, 128", "32, 128"), encoding="utf-8")
        digest = file_hash(self.op_file)
        self.assertTrue(run_session(*record_argv(self.op_dir, digest, "400"))["improved"])
        terminal = json.loads(marker.read_text(encoding="utf-8"))
        self.assertEqual(terminal["reason"], "eval_budget")
        self.assertEqual(terminal["delivered_code_hash"], digest)
        verdict = run_session("stop", *op_args)
        self.assertEqual(verdict["evals_used"], 2)
        self.assertEqual(verdict["delivered_code_hash"], digest)
        self.assertEqual(json.loads(marker.read_text(encoding="utf-8")), terminal)
        report = (optimization / "demo_optimization.md").read_text(encoding="utf-8")
        self.assertEqual(report.count("## Final best"), 1)

    def test_block_winner_is_charged_once_when_recorded_at_the_budget_limit(self):
        response = self._run_final_block(400.0, best_was_incumbent=False)
        result = json.loads(response["stdout"])
        self.assertIsNone(response.get("terminal"))
        self.assertEqual(result["device_trials"], 1)
        self.assertEqual(result["charged"], 0)
        self.assertEqual(read_state(self.op_dir)["progress"]["evals_used"], 1)
        digest = result["code_hash"]
        self.assertEqual(digest, file_hash(self.op_file))
        record, rc = _run(*record_argv(self.op_dir, digest, str(result["best_latency"])))
        self.assertEqual(rc, 0)
        self.assertTrue(record["improved"])
        stop, rc = _run("stop", "--op-dir", str(self.op_dir))
        self.assertEqual(rc, 0)
        self.assertEqual(stop["evals_used"], 2)
        self.assertEqual(stop["delivered_code_hash"], digest)

    def test_block_without_a_winner_stops_at_the_budget_limit(self):
        response = self._run_final_block(900.0, best_was_incumbent=True)
        self.assertEqual(json.loads(response["stdout"])["charged"], 1)
        terminal = response["terminal"]
        self.assertEqual(terminal["reason"], "eval_budget")
        self.assertEqual(self.op_file.read_text(encoding="utf-8"), KERNEL)
        self.assertEqual(terminal["delivered_code_hash"], file_hash(self.op_file))
        self.assertEqual(read_state(self.op_dir)["progress"]["evals_used"], 2)

    def test_resumed_block_winner_is_recorded_even_after_the_wall_clock_limit(self):
        response = self._run_final_block(400.0, best_was_incumbent=False, wall_clock_limit=1)
        digest = json.loads(response["stdout"])["code_hash"]
        self.op_file.write_text(KERNEL, encoding="utf-8")
        op_args = ("--op-dir", str(self.op_dir))

        def do_not_dispatch(*args):
            self.fail("a pending winner must not dispatch another block")

        repeated_block = self._block_response(do_not_dispatch)
        self.assertEqual(repeated_block["returncode"], 3, repeated_block["stderr"])
        refusal = json.loads(repeated_block["stdout"])
        self.assertEqual(refusal["error"], "block_winner_pending")
        self.assertEqual(refusal["pending_record"], dict(parent="u1", code_hash=digest, s=1, p=400.0))

        for args in (("select", *op_args), ("stop", *op_args),
                     tuple(record_argv(self.op_dir, digest, "401"))):
            with self.subTest(command=args[0]):
                payload, rc = _run(*args)
                self.assertEqual(rc, 3)
                self.assertEqual(payload["error"], "block_winner_pending")
        state = read_state(self.op_dir)
        self.assertGreaterEqual(state["progress"]["wall_clock_s"], 1)
        self.assertEqual(state["progress"]["evals_used"], 1)

        record, rc = _run(*record_argv(self.op_dir, digest, "400"))
        self.assertEqual(rc, 0)
        self.assertTrue(record["improved"])
        stop, rc = _run("stop", *op_args)
        self.assertEqual(rc, 0)
        self.assertEqual(stop["reason"], "wall_clock")
        self.assertEqual(stop["evals_used"], 2)
        self.assertEqual(stop["delivered_code_hash"], digest)
        self.assertEqual(file_hash(self.op_file), digest)

    def test_replayed_block_winner_does_not_spend_another_evaluation(self):
        response = self._run_final_block(400.0, best_was_incumbent=False, replayed=True)
        payload = json.loads(response["stdout"])
        self.assertEqual(payload["device_trials"], 0)
        record, rc = _run(*record_argv(self.op_dir, payload["code_hash"], "400"))
        self.assertEqual(rc, 0)
        self.assertTrue(record["improved"])
        self.assertEqual(read_state(self.op_dir)["progress"]["evals_used"], 1)

    def _run_final_block(self, latency, *, best_was_incumbent, replayed=False, wall_clock_limit=None):
        config = {"eval_budget": 2}
        if wall_clock_limit is not None:
            config["wall_clock_limit_s"] = wall_clock_limit
        self._init("--config", json.dumps(config))
        if wall_clock_limit is not None:
            state = read_state(self.op_dir)
            state["progress"]["_start_ts"] -= wall_clock_limit
            path = self.op_dir / "optimization" / "search_state.json"
            path.write_text(json.dumps(state), encoding="utf-8")
        candidate = {"vec#0": {"d0": 32, "d1": 128}}

        def offline_block(op_file, evaluator, chip, options):
            self.assertEqual(options.max_trials, 1)
            if not replayed:
                options.on_trial({"status": "ok", "config": candidate, "latency": latency})
            return {
                "best_config": candidate,
                "best_latency": options.current_latency if best_was_incumbent else latency,
                "best_was_incumbent": best_was_incumbent,
                "n_trials": 1,
            }

        result = self._block_response(offline_block)
        self.assertEqual(result["returncode"], 0, result["stderr"])
        return result

    def _block_response(self, objective):
        block_args = [
            "block", "--op-dir", str(self.op_dir), "--parent", "u1",
            "--test-command", "true", "--max-trials", "10",
        ]
        with (
            patch.object(harness.bayesian.block, "run", side_effect=objective),
            patch.object(harness, "confirm_chip", return_value=None),
            patch.object(harness_session.importlib, "import_module", return_value=harness),
        ):
            return harness_session.execute({"op_dir": str(self.op_dir)}, block_args)

    def _init(self, *extra, op_file=None):
        return _run(
            *init_argv(self.op_dir, op_file or self.op_file, *CHIP_ARGS, *extra)
        )

    def _nodes(self):
        return sorted((self.op_dir / "optimization" / "nodes").glob("*.py"))


if __name__ == "__main__":
    unittest.main()
