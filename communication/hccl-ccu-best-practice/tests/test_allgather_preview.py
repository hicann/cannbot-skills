#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Boundary predictions, fail-closed source checks and read-only behavior."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/preview_allgather_cases.py"
spec = importlib.util.spec_from_file_location("preview_allgather", SCRIPT)
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


class PreviewTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        source = self.root / "source.cc"
        source.write_text("isolated reviewed source fixture\n")
        self.profile = self.root / "profile.json"
        self.profile.write_text(
            json.dumps(
                dict(
                    model="hccl-test-allgather-align-ceil-v1",
                    sources={
                        "source.cc": hashlib.sha256(source.read_bytes()).hexdigest()
                    },
                )
            )
        )

    def run_preview(self, sizes, **kw):
        return preview.preview(
            self.root,
            kw.pop("ranks", 8),
            kw.pop("dtype", "int32"),
            sizes,
            profile_path=self.profile,
            **kw,
        )

    def test_nondivisible_input_is_same_effective_case(self):
        result = self.run_preview([1048576, 1048579])
        self.assertTrue(result["ok"])
        self.assertEqual([32768, 32768], [r["count_per_rank"] for r in result["cases"]])
        self.assertEqual(1048576, result["cases"][1]["normalized_tool_bytes"])
        self.assertEqual(1, len(result["duplicate_effective_inputs"]))
        self.assertFalse(result["cases"][1]["non_dtype_aligned_slice_expressible"])

    def test_small_input_uses_two_ceil_operations_not_integer_truncation(self):
        row = self.run_preview([1])["cases"][0]
        self.assertEqual(8, row["normalized_tool_bytes"])
        self.assertEqual(1, row["count_per_rank"])
        self.assertEqual(32, row["effective_total_bytes"])
        self.assertEqual(4, row["input_bytes_per_rank"])
        self.assertEqual(288, row["aggregate_input_output_bytes"])

    def test_alignment_transition_above_16mib(self):
        n = 16 * 1024 * 1024
        rows = self.run_preview([n - 1, n, n + 1, n + 4095, n + 4096])["cases"]
        self.assertEqual(
            [n - 8, n, n, n, n + 4096], [r["normalized_tool_bytes"] for r in rows]
        )
        self.assertEqual(
            [n, n, n, n, n + 4096], [r["effective_total_bytes"] for r in rows]
        )

    def test_zero_and_expansion_over_budget_fail_without_execution(self):
        zero = self.run_preview([0])
        self.assertFalse(zero["ok"])
        self.assertNotIn("count_per_rank", zero["cases"][0])
        budget = self.run_preview([1], max_total_bytes=8)
        self.assertFalse(budget["ok"])
        self.assertFalse(budget["cases"][0]["within_budget"])

    def test_loop_threshold_requires_basis_and_is_only_prediction(self):
        with self.assertRaises(ValueError):
            self.run_preview([64], max_count_per_loop=2)
        rows = self.run_preview(
            [64, 96], max_count_per_loop=2, loop_basis="fixture executor count bound"
        )["cases"]
        self.assertEqual([1, 2], [r["predicted_loop_times"] for r in rows])
        self.assertTrue(all("requires actual" in r["loop_coverage"] for r in rows))
        self.assertTrue(all(not r["fastlaunch_buffer_change_proven"] for r in rows))

    def test_changed_or_missing_source_never_emits_guessed_predictions(self):
        (self.root / "source.cc").write_text("changed")
        result = self.run_preview([64])
        self.assertFalse(result["ok"])
        self.assertEqual([], result["cases"])
        self.assertEqual("CHANGED", result["sources"][0]["status"])
        (self.root / "source.cc").unlink()
        self.assertEqual("MISSING", self.run_preview([64])["sources"][0]["status"])

    def test_invalid_and_overflow_values(self):
        for kwargs in [dict(ranks=0), dict(dtype="bf16"), dict(ranks=True)]:
            with self.assertRaises(ValueError):
                self.run_preview([64], **kwargs)
        for sizes in [[], [-1], [True], [2**64], [2**64 - 1]]:
            with self.assertRaises(ValueError):
                self.run_preview(sizes)

    def test_predictions_and_failed_cli_are_read_only(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.run_preview([64, 1024, 1048579])
        cli = subprocess.run(
            [
                sys.executable,
                "-B",
                str(SCRIPT),
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
        self.assertEqual(1, cli.returncode)
        self.assertEqual([], json.loads(cli.stdout)["cases"])
        self.assertEqual(
            before,
            {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()},
        )


if __name__ == "__main__":
    unittest.main()
