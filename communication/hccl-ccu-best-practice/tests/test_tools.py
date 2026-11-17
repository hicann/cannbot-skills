#!/usr/bin/env python3
# Copyright (c) 2026 CANNBot contributors
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Regression tests for source provenance and declared launch-ABI validation."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, SKILL / "scripts" / f"{name}.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


abi = module("check_launch_contract")
sources = module("inspect_sources")


class LaunchContractTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(
            (SKILL / "assets" / "launch-contract.example.json").read_text()
        )
        self.kernel = self.data["kernels"][0]

    def test_reviewed_example(self):
        self.assertEqual(abi.validate_contract(self.data), 1)

    def test_multiple_kernels(self):
        second = copy.deepcopy(self.kernel)
        second["name"] = "SecondKernel"
        second["fast_launch"] = {
            "enabled": False,
            "reason": "This path does not support replay.",
        }
        self.data["kernels"].append(second)
        self.assertEqual(abi.validate_contract(self.data), 2)

    def test_launch_count_is_slots_not_bytes(self):
        self.kernel["launch_count"] *= 8
        with self.assertRaisesRegex(abi.ContractError, "64-bit slots"):
            abi.validate_contract(self.data)

    def test_load_index_name_mismatch(self):
        self.kernel["loads"][0]["index"] = 1
        with self.assertRaisesRegex(abi.ContractError, "must be 'outputAddr'"):
            abi.validate_contract(self.data)

    def test_out_of_range_load(self):
        self.kernel["loads"][0]["index"] = 15
        with self.assertRaisesRegex(abi.ContractError, "outside"):
            abi.validate_contract(self.data)

    def test_multiple_consumers_of_one_slot_are_valid(self):
        self.kernel["loads"].append(copy.deepcopy(self.kernel["loads"][0]))
        self.assertEqual(abi.validate_contract(self.data), 1)

    def test_unused_slot_needs_reason(self):
        self.kernel["loads"].pop()
        with self.assertRaisesRegex(abi.ContractError, "unconsumed"):
            abi.validate_contract(self.data)
        self.kernel["unused_args"] = {
            "goSize.residual": "Reserved padding for this variant."
        }
        self.assertEqual(abi.validate_contract(self.data), 1)

    def test_fast_launch_cannot_send_cache_metadata(self):
        self.kernel["fast_launch"]["launch_count"] = 17
        with self.assertRaisesRegex(abi.ContractError, "cache metadata"):
            abi.validate_contract(self.data)

    def test_cache_prefix_order(self):
        cached = self.kernel["fast_launch"]["cached_args"]
        cached[0], cached[1] = cached[1], cached[0]
        with self.assertRaisesRegex(abi.ContractError, "launch slice"):
            abi.validate_contract(self.data)

    def reduce_cache(self):
        self.data = json.loads(
            (SKILL / "assets/launch-contract.reduce.example.json").read_text()
        )
        self.kernel = self.data["kernels"][0]
        return self.kernel["fast_launch"]

    def test_reduce_cache_header_and_physical_update_indices(self):
        fast = self.reduce_cache()
        self.assertEqual(20, self.kernel["launch_count"])
        self.assertEqual("launchSlotCount", fast["cached_args"][0])
        self.assertEqual([1, 2], [item["index"] for item in fast["updates"]])
        self.assertEqual(1, abi.validate_contract(self.data))

    def test_missing_offset_does_not_silently_strip_reduce_header(self):
        del self.reduce_cache()["cache_launch_offset"]
        with self.assertRaisesRegex(abi.ContractError, "launch slice"):
            abi.validate_contract(self.data)

    def test_cache_offset_must_be_unsigned_integer_and_in_bounds(self):
        fast = self.reduce_cache()
        for invalid in (-1, True, 1.0, "1", 2**32, len(fast["cached_args"]) + 1):
            fast["cache_launch_offset"] = invalid
            with self.subTest(offset=invalid), self.assertRaises(abi.ContractError):
                abi.validate_contract(self.data)

    def test_offset_cannot_shift_or_truncate_launch_args(self):
        fast = self.reduce_cache()
        for offset in (0, 2, len(fast["cached_args"]) - 1):
            fast["cache_launch_offset"] = offset
            with self.subTest(offset=offset), self.assertRaises(abi.ContractError):
                abi.validate_contract(self.data)

    def test_updates_use_physical_cache_indices_not_launch_indices(self):
        self.reduce_cache()["updates"][0]["index"] = 0
        with self.assertRaisesRegex(abi.ContractError, "launchSlotCount"):
            abi.validate_contract(self.data)

    def test_offset_does_not_waive_metadata_count_or_cache_capacity(self):
        fast = self.reduce_cache()
        fast["launch_count"] += 1
        with self.assertRaisesRegex(abi.ContractError, "cache metadata"):
            abi.validate_contract(self.data)
        fast["launch_count"] -= 1
        fast["cache_capacity"] = len(fast["cached_args"]) - 1
        with self.assertRaisesRegex(abi.ContractError, "capacity"):
            abi.validate_contract(self.data)

    def test_zero_offset_remains_compatible(self):
        self.kernel["fast_launch"]["cache_launch_offset"] = 0
        self.assertEqual(1, abi.validate_contract(self.data))

    def test_cache_capacity(self):
        self.kernel["fast_launch"]["cache_capacity"] = 16
        with self.assertRaisesRegex(abi.ContractError, "capacity"):
            abi.validate_contract(self.data)

    def test_alias_flag_must_be_updated(self):
        self.kernel["fast_launch"]["updates"].pop()
        with self.assertRaisesRegex(
            abi.ContractError, "missing updates.*isInputOutputEqual"
        ):
            abi.validate_contract(self.data)

    def test_duplicate_update_rejected(self):
        updates = self.kernel["fast_launch"]["updates"]
        updates.append(copy.deepcopy(updates[0]))
        with self.assertRaisesRegex(abi.ContractError, "duplicate update"):
            abi.validate_contract(self.data)

    def test_unclassified_replay_argument(self):
        del self.kernel["fast_launch"]["invariant_args"]["token"]
        with self.assertRaisesRegex(abi.ContractError, "classified"):
            abi.validate_contract(self.data)

    def test_duplicate_argument_names_rejected(self):
        self.kernel["launch_args"][1] = "inputAddr"
        with self.assertRaisesRegex(abi.ContractError, "duplicate names"):
            abi.validate_contract(self.data)

    def test_invariant_and_update_contradiction(self):
        self.kernel["fast_launch"]["updates"].append({"index": 2, "name": "token"})
        with self.assertRaisesRegex(abi.ContractError, "contradict"):
            abi.validate_contract(self.data)

    def test_invalid_structures(self):
        mutations = [
            lambda d: d.update(schema_version=True),
            lambda d: d.update(kernels={}),
            lambda d: d["kernels"][0].update(launch_count=True),
            lambda d: d["kernels"][0].update(launch_count=-1),
            lambda d: d["kernels"][0].update(launch_count=15.0),
            lambda d: d["kernels"][0].update(launch_count=2**32),
            lambda d: d["kernels"][0].update(launch_args=[{}]),
            lambda d: d["kernels"][0].update(loads=None),
            lambda d: d["kernels"][0]["fast_launch"].update(enabled="false"),
            lambda d: d["kernels"].append(copy.deepcopy(d["kernels"][0])),
            lambda d: d["kernels"][0].update(typo="ignored?"),
        ]
        for mutate in mutations:
            data = copy.deepcopy(self.data)
            mutate(data)
            with self.subTest(data=data), self.assertRaises(abi.ContractError):
                abi.validate_contract(data)

    def test_cli_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text('{"schema_version": 1, "schema_version": 1, "kernels": []}')
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(abi.main([str(path)]), 1)
            self.assertIn("duplicate JSON key", stderr.getvalue())


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.roots = {name: self.root / name for name in ("hccl", "hcomm")}
        self.paths = {
            "hccl": "src/ops/demo/algorithm/template/ccu/ccu_temp_demo.cc",
            "hcomm": "include/ccu/ccu_launch.h",
        }
        self.entries = []
        for name, root in self.roots.items():
            root.mkdir()
            self.git(root, "init", "--quiet")
            path = root / self.paths[name]
            path.parent.mkdir(parents=True)
            path.write_text("// fixture Symbol\n", encoding="utf-8")
            self.git(root, "add", ".")
            self.git(
                root,
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--quiet",
                "-m",
                "fixture",
            )
            self.entries.append(
                {
                    "repo": name,
                    "path": self.paths[name],
                    "required": True,
                    "symbols": ["Symbol"],
                    "sha256": sources.sha256(path),
                }
            )
        self.source_map = {"schema_version": 1, "files": self.entries}

    @staticmethod
    def git(root, *args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
            },
        ).stdout.strip()

    def inspect(self):
        return sources.inspect(
            self.roots["hccl"], self.roots["hcomm"], "demo", self.source_map
        )

    def run_cli(self, *extra):
        path = self.root / "source-map.json"
        path.write_text(json.dumps(self.source_map))
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return sources.main(
                [
                    "--hccl-root",
                    str(self.roots["hccl"]),
                    "--hcomm-root",
                    str(self.roots["hcomm"]),
                    "--op",
                    "demo",
                    "--source-map",
                    str(path),
                    *extra,
                ]
            )

    def test_source_identity_and_inventory(self):
        result = self.inspect()
        self.assertTrue(all(s["status"] == "unchanged" for s in result["sources"]))
        self.assertEqual(result["inventory"]["files"][0]["path"], self.paths["hccl"])
        self.assertEqual(result["sources"][0]["symbols"]["Symbol"], [1])
        self.assertEqual(self.run_cli("--check-baseline"), 0)

    def test_dirty_content_detected_without_head_change(self):
        before = self.inspect()["repositories"]["hccl"]["head"]
        (self.roots["hccl"] / self.paths["hccl"]).write_text("// edited Symbol\n")
        result = self.inspect()
        self.assertEqual(before, result["repositories"]["hccl"]["head"])
        self.assertEqual(result["sources"][0]["status"], "changed")
        self.assertTrue(result["repositories"]["hccl"]["scoped_status"])
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.run_cli("--check-baseline"), 1)

    def test_missing_required_source_fails(self):
        (self.roots["hcomm"] / self.paths["hcomm"]).unlink()
        self.assertEqual(self.run_cli(), 1)

    def test_optional_sample_missing_reports_drift(self):
        self.entries[1]["required"] = False
        (self.roots["hcomm"] / self.paths["hcomm"]).unlink()
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.run_cli("--check-baseline"), 1)

    def test_map_cannot_escape_repository(self):
        self.entries[0]["path"] = "../outside.cc"
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.inspect()

    def test_symlink_cannot_escape_repository(self):
        path = self.roots["hccl"] / self.paths["hccl"]
        external = self.root / "outside.cc"
        external.write_text("outside")
        path.unlink()
        path.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.inspect()

    def test_repo_must_be_worktree_root(self):
        with self.assertRaisesRegex(ValueError, "worktree root"):
            sources.repository(self.roots["hccl"] / "src")

    def test_no_index_rewrite(self):
        index = self.roots["hccl"] / ".git" / "index"
        previous = (index.stat().st_mtime_ns, index.read_bytes())
        self.inspect()
        self.assertEqual(previous, (index.stat().st_mtime_ns, index.read_bytes()))

    def test_unrelated_lfs_like_asset_does_not_run_filter(self):
        repo = self.roots["hcomm"]
        (repo / "unrelated.png").write_text("image fixture")
        self.git(repo, "add", "unrelated.png")
        (repo / ".gitattributes").write_text("*.png filter=broken\n")
        self.git(repo, "config", "filter.broken.clean", "false")
        self.git(repo, "config", "filter.broken.required", "true")
        result = self.inspect()
        self.assertEqual(result["sources"][1]["status"], "unchanged")

    def test_legacy_layout_is_identified(self):
        old = self.roots["hccl"] / self.paths["hccl"]
        new = self.roots["hccl"] / "src/ops/demo/template/ccu/ccu_temp_demo.cc"
        new.parent.mkdir(parents=True)
        old.rename(new)
        old.parent.rmdir()
        result = sources.inventory(self.roots["hccl"], "demo")
        self.assertIn("legacy", result["layout"])

    def test_invalid_or_missing_operator(self):
        for op in ("../demo", "", "missing"):
            with self.subTest(op=op), self.assertRaises(ValueError):
                sources.inventory(self.roots["hccl"], op)

    def test_plugin_task_report_is_created_once_without_source_or_index_changes(self):
        task = self.roots["hccl"] / "src/ops/demo/docs/portable-task-001"
        task.mkdir(parents=True)
        output = task / "source-report.json"
        source = self.roots["hccl"] / self.paths["hccl"]
        index = self.roots["hccl"] / ".git/index"
        before = source.read_bytes(), index.read_bytes(), index.stat().st_mtime_ns
        self.assertEqual(self.run_cli("--output", str(output), "--summary"), 0)
        data = json.loads(output.read_text())
        self.assertEqual("demo", data["inventory"]["op"])
        self.assertIn("sources", data)
        self.assertIn("operator", data["knowledge"])
        saved = output.read_bytes()
        self.assertEqual(self.run_cli("--output", str(output)), 2)
        self.assertEqual(saved, output.read_bytes())
        self.assertEqual(
            before, (source.read_bytes(), index.read_bytes(), index.stat().st_mtime_ns)
        )

    def test_task_exception_rejects_other_operator_filename_and_hcomm(self):
        paths = [
            self.roots["hccl"] / "src/ops/demo/docs/task/other.json",
            self.roots["hccl"] / "src/ops/other/docs/task/source-report.json",
            self.roots["hccl"] / "src/ops/demo/docs/invalid task/source-report.json",
            self.roots["hcomm"] / "src/ops/demo/docs/task/source-report.json",
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self.subTest(path=path):
                self.assertEqual(self.run_cli("--output", str(path)), 2)
                self.assertFalse(path.exists())

    def test_redesign_creates_a_new_revision_and_keeps_old_evidence(self):
        task = self.roots["hccl"] / "src/ops/demo/docs/task-001"
        task.mkdir(parents=True)
        original = task / "source-report.json"
        self.assertEqual(0, self.run_cli("--output", str(original)))
        saved = original.read_bytes()
        source = self.roots["hccl"] / self.paths["hccl"]
        source.write_text(source.read_text() + "// next revision\n")
        revised = task / "source-report-r2.json"
        self.assertEqual(0, self.run_cli("--output", str(revised), "--summary"))
        self.assertEqual(saved, original.read_bytes())
        self.assertEqual(
            "changed", json.loads(revised.read_text())["sources"][0]["status"]
        )
        saved_revision = revised.read_bytes()
        self.assertEqual(2, self.run_cli("--output", str(revised)))
        self.assertEqual(saved_revision, revised.read_bytes())
        for filename in (
            "source-report-r0.json",
            "source-report-r-2.json",
            "source-report-r../2.json",
        ):
            self.assertEqual(2, self.run_cli("--output", str(task / filename)))

    def test_source_inventory_records_explicit_reviewed_baseline(self):
        baseline = self.root / "reviewed.json"
        baseline.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cards": {
                        "sync-limits": {
                            "scope": "reviewed fixture",
                            "reference": "fixture.md",
                            "files": [self.entries[1]],
                        }
                    },
                }
            )
        )
        output = self.root / "source-report.json"
        self.assertEqual(
            0, self.run_cli("--baseline", str(baseline), "--output", str(output))
        )
        knowledge = json.loads(output.read_text())["knowledge"]
        self.assertEqual(sources.sha256(baseline), knowledge["baseline_sha256"])
        self.assertEqual("MATCH", knowledge["cards"][0]["status"])

    def test_task_symlink_cannot_redirect_report_into_source_or_hcomm(self):
        docs = self.roots["hccl"] / "src/ops/demo/docs"
        docs.mkdir()
        targets = [
            self.roots["hccl"] / self.paths["hccl"],
            self.roots["hcomm"] / self.paths["hcomm"],
        ]
        for i, source in enumerate(targets):
            task = docs / ("task-" + str(i))
            task.mkdir()
            output = task / "source-report.json"
            output.symlink_to(source)
            before = source.read_bytes()
            self.assertEqual(self.run_cli("--output", str(output)), 2)
            self.assertEqual(before, source.read_bytes())
        task_link = docs / "linked-task"
        source_dir = (self.roots["hccl"] / self.paths["hccl"]).parent
        task_link.symlink_to(source_dir, target_is_directory=True)
        self.assertEqual(
            self.run_cli("--output", str(task_link / "source-report.json")), 2
        )
        self.assertFalse((source_dir / "source-report.json").exists())

    def test_report_never_overwrites_or_writes_into_sources(self):
        output = self.root / "report.json"
        self.assertEqual(self.run_cli("--output", str(output)), 0)
        previous = output.read_bytes()
        self.assertEqual(self.run_cli("--output", str(output)), 2)
        self.assertEqual(output.read_bytes(), previous)
        inside_repo = self.roots["hccl"] / "report.json"
        self.assertEqual(self.run_cli("--output", str(inside_repo)), 2)
        self.assertFalse(inside_repo.exists())


if __name__ == "__main__":
    unittest.main()
