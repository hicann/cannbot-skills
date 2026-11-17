#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Operator differences, semantic/allocation roles, scope isolation and fail-closed CLI."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
SCRIPT = SKILL / "scripts/preview_cases.py"
spec = importlib.util.spec_from_file_location("collective_preview", SCRIPT)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class CollectivePreviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.common = self.root / "common.cc"
        self.common.write_text("reviewed common source\n")
        self.profile = self.root / "profile.json"

        def entry(p):
            return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()}

        self.data = dict(
            model="hccl-test-collectives-v1",
            common_sources=entry(self.common),
            operators={},
        )
        for op in model.OPS:
            p = self.root / (op + ".cc")
            p.write_text("reviewed " + op + "\n")
            self.data["operators"][op] = dict(binary=op + "_test", sources=entry(p))
        self.write_profile()

    def write_profile(self):
        self.profile.write_text(json.dumps(self.data))

    def run_model(self, op, sizes=(1048576, 1048579), **kwargs):
        return model.preview(
            self.root,
            kwargs.pop("rank_count", 8),
            kwargs.pop("dtype", "int32"),
            sizes,
            op=op,
            profile_path=self.profile,
            **kwargs,
        )

    def test_allreduce_does_not_inherit_allgather_rank_alignment(self):
        out = self.run_model("all_reduce")
        self.assertTrue(out["ok"])
        self.assertEqual([262144, 262145], [r["count_per_rank"] for r in out["cases"]])
        self.assertEqual([], out["duplicate_effective_inputs"])
        self.assertEqual(1048580, out["cases"][1]["effective_total_bytes"])
        self.assertEqual(16777280, out["cases"][1]["aggregate_input_output_bytes"])
        self.assertEqual("allreduce", out["execution_requirements"]["hvm_operator"])

    def test_reducescatter_uses_receive_count_and_input_block_span(self):
        out = self.run_model("reduce_scatter")
        row = out["cases"][1]
        self.assertEqual(32768, row["count_per_rank"])
        self.assertEqual(1048576, row["input_bytes_per_rank"])
        self.assertEqual(131072, row["output_bytes_per_rank"])
        self.assertEqual(9437184, row["aggregate_input_output_bytes"])
        self.assertEqual(1, len(out["duplicate_effective_inputs"]))

    def test_broadcast_alias_counts_one_allocation_per_rank(self):
        out = self.run_model("broadcast", (1,), root_rank=7)
        row = out["cases"][0]
        self.assertEqual(1, row["count_per_rank"])
        self.assertEqual(0, row["allocated_recv_bytes_per_rank"])
        self.assertTrue(row["broadcast_buffer_alias"])
        self.assertEqual(32, row["aggregate_input_output_bytes"])
        self.assertEqual(
            dict(input_bytes=0, output_bytes=4), row["semantic_bytes"]["non_root"]
        )
        self.assertEqual(["-r", "7"], out["execution_requirements"]["hccl_test_flags"])
        self.assertFalse(out["execution_requirements"]["hvm_defaults_suffice"])
        self.assertNotIn("non_dtype_aligned_slice_expressible", row)

    def test_reduce_allocations_are_not_reduced_to_root_only_output(self):
        out = self.run_model("reduce", (1,), root_rank=4, reduce_op="max")
        row = out["cases"][0]
        self.assertEqual(64, row["aggregate_input_output_bytes"])
        self.assertEqual(4, row["allocated_recv_bytes_per_rank"])
        self.assertEqual(0, row["semantic_bytes"]["non_root"]["output_bytes"])
        self.assertEqual(
            ["-r", "4", "-o", "max"], out["execution_requirements"]["hccl_test_flags"]
        )
        self.assertFalse(out["execution_requirements"]["hvm_defaults_suffice"])

    def test_scatter_only_root_semantic_input_but_every_rank_allocates(self):
        out = self.run_model("scatter", (1,), root_rank=7)
        row = out["cases"][0]
        self.assertEqual(1, row["count_per_rank"])
        self.assertEqual(32, row["allocated_send_bytes_per_rank"])
        self.assertEqual(288, row["aggregate_input_output_bytes"])
        self.assertEqual(0, row["semantic_bytes"]["non_root"]["input_bytes"])
        self.assertEqual(32, row["semantic_bytes"]["root"]["input_bytes"])

    def test_alignment_transition_applies_only_to_split_operations(self):
        n = 16 * 1024 * 1024
        for op in model.OPS:
            rows = self.run_model(op, (n, n + 1, n + 4096))["cases"]
            expected = (
                [n, n, n + 4096] if op in model.SPLIT_OPS else [n, n + 1, n + 4096]
            )
            self.assertEqual(expected, [r["normalized_tool_bytes"] for r in rows], op)

    def test_per_case_and_aggregate_budgets_are_independent(self):
        out = self.run_model(
            "scatter", (1,), max_total_bytes=32, max_aggregate_bytes=287
        )
        row = out["cases"][0]
        self.assertFalse(out["ok"])
        self.assertTrue(row["within_budget"])
        self.assertFalse(row["within_aggregate_budget"])
        for op in model.OPS:
            row = self.run_model(op, (1,))["cases"][0]
            self.assertIsNone(row["within_budget"])
            self.assertIsNone(row["within_aggregate_budget"])

    def test_only_selected_operator_and_common_source_are_checked(self):
        (self.root / "scatter.cc").write_text("changed")
        self.assertTrue(self.run_model("reduce")["ok"])
        bad = self.run_model("scatter")
        self.assertFalse(bad["ok"])
        self.assertEqual([], bad["cases"])
        self.common.unlink()
        for op in model.OPS:
            out = self.run_model(op)
            self.assertFalse(out["ok"])
            self.assertEqual([], out["cases"])
            self.assertEqual("MISSING", out["sources"][0]["status"])

    def test_flags_validate_applicability_not_algorithm_support(self):
        for op in model.ROOTED_OPS:
            for root in (-1, 8, True):
                with self.assertRaises(ValueError):
                    self.run_model(op, root_rank=root)
        for op in set(model.OPS) - model.ROOTED_OPS:
            with self.assertRaises(ValueError):
                self.run_model(op, root_rank=0)
        for op in set(model.OPS) - model.REDUCE_OPS:
            with self.assertRaises(ValueError):
                self.run_model(op, reduce_op="sum")
        # Tool accepts prod; the source card separately excludes the existing Mesh implementation.
        self.assertTrue(self.run_model("all_reduce", reduce_op="prod")["ok"])
        with self.assertRaises(ValueError):
            self.run_model("reduce", reduce_op="average")
        with self.assertRaises(ValueError):
            self.run_model("allreduce")

    def test_loop_count_is_based_on_api_count_not_concatenated_span(self):
        for op, expected in [("all_reduce", 8), ("reduce_scatter", 1), ("scatter", 1)]:
            row = self.run_model(
                op, (64,), max_count_per_loop=2, loop_basis="reviewed fixture executor"
            )["cases"][0]
            self.assertEqual(expected, row["predicted_loop_times"])
            self.assertFalse(row["fastlaunch_buffer_change_proven"])
        with self.assertRaises(ValueError):
            self.run_model("reduce", max_count_per_loop=1)

    def test_zero_and_collapsed_alignment_are_not_positive_count_tests(self):
        for op in model.OPS:
            out = self.run_model(op, (0,))
            self.assertFalse(out["ok"])
            self.assertNotIn("count_per_rank", out["cases"][0])
        out = self.run_model("scatter", (16 * 1024 * 1024 + 1,), rank_count=100000)
        self.assertFalse(out["ok"])
        self.assertEqual(0, out["cases"][0]["count_per_rank"])

    def test_escaping_profile_and_unreviewed_dtype_rejected(self):
        self.data["common_sources"] = {"../outside.cc": "0" * 64}
        self.write_profile()
        with self.assertRaises(ValueError):
            self.run_model("reduce")
        with self.assertRaises(ValueError):
            self.run_model("all_reduce", dtype="int128")
        with self.assertRaises(ValueError):
            self.run_model("reduce", (2**63,))
        with self.assertRaises(ValueError):
            self.run_model("scatter", rank_count=2**31)

    def test_binary_hash_remains_unbound_and_preview_readonly(self):
        binary = self.root / "test_bin"
        binary.write_bytes(b"fixture binary")
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        out = self.run_model("broadcast", test_binary=binary)
        self.assertEqual(
            hashlib.sha256(binary.read_bytes()).hexdigest(),
            out["test_binary"]["sha256"],
        )
        self.assertIn("UNVERIFIED", out["test_binary"]["source_binding"])
        self.assertEqual(
            before,
            {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()},
        )

    def test_unknown_source_cli_fails_closed_without_caches(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        run = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--op",
                "reduce",
                "--hccl-test-root",
                str(self.root),
                "--rank-count",
                "8",
                "--dtype",
                "int32",
                "--sizes",
                "64",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(1, run.returncode)
        self.assertEqual([], json.loads(run.stdout)["cases"])
        self.assertEqual(
            before,
            {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()},
        )


class OperatorCardSelectionTests(unittest.TestCase):
    def test_new_cards_do_not_pull_allgather_or_other_operators(self):
        sys.path.insert(0, str(SKILL / "scripts"))
        from check_knowledge import check_cards

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = json.loads(
                (SKILL / "references/knowledge-baseline.json").read_text()
            )
            # Isolated files with original source scope; no access to target repos needed.
            for card in data["cards"].values():
                for entry in card["files"]:
                    p = root / entry["repo"] / entry["path"]
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text("source fixture")
                    entry["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps(data))
            roots = {"hccl": root / "hccl", "hcomm": root / "hcomm"}
            for op in model.OPS:
                (roots["hccl"] / "src/ops" / op).mkdir(parents=True, exist_ok=True)
            for op in set(model.OPS) - {"all_gather"}:
                cards = check_cards(roots, op, baseline)["cards"]
                self.assertEqual(
                    {"sync-limits", "generator-arith", op.replace("_", "-") + "-sole"},
                    {c["card_id"] for c in cards},
                )
                self.assertTrue(all(c["status"] == "MATCH" for c in cards))
            path = (
                root
                / "hccl/src/ops/scatter/algorithm/executor/ins_v2_scatter_sole_executor.cc"
            )
            path.write_text("changed scatter only")
            self.assertEqual(
                "CHANGED",
                check_cards(roots, "scatter", baseline)["cards"][-1]["status"],
            )
            self.assertTrue(
                all(
                    c["status"] == "MATCH"
                    for c in check_cards(roots, "reduce", baseline)["cards"]
                )
            )


if __name__ == "__main__":
    unittest.main()
