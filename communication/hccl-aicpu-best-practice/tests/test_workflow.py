#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""临时仓库/假构建离线验证，不调用真实 HCCL 或 Checker。"""

import argparse
import importlib.util
import json
import os
from unittest.mock import patch
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/workflow.py"
SPEC = importlib.util.spec_from_file_location("workflow", SCRIPT)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.email", "offline@example.invalid")
        self.git("config", "user.name", "Offline Test")
        (self.repo / "build.sh").write_text('echo "$@"\necho called >> calls\nexit 0\n')
        source = self.repo / "src/reduce_scatter"
        source.mkdir(parents=True)
        (source / "executor.cc").write_text("void Executor() { temp.GetRes(); }\n")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")
        self.archive = self.root / "archive"
        self.archive.mkdir()
        for name, text in [
            ("package.run", "fake package"),
            ("build.log", "fake log"),
            ("build.exit_code", "0\n"),
        ]:
            (self.archive / name).write_text(text)
        self.artifact = dict(
            schema_version=1,
            source_id="old",
            environment_id="fake",
            build_command="fake",
            package="package.run",
            package_sha256=workflow.digest(self.archive / "package.run"),
            build_log_sha256=workflow.digest(self.archive / "build.log"),
            build_exit_code=0,
            expected_libraries={"host": "hosthash", "device": "devicehash"},
        )
        workflow.write_new(self.archive / "artifact.json", self.artifact)
        self.cann = self.root / "cann"
        self.cann.mkdir()
        (self.cann / "set_env.sh").write_text("export SMOKE_CANN_LOADED=yes\n")
        toolbin = self.cann / "toolkit/toolchain/hcc/bin"
        toolbin.mkdir(parents=True)
        for tool in ("gcc", "g++", "ar", "ranlib", "strip", "ld", "nm", "objcopy"):
            executable = toolbin / ("aarch64-target-linux-gnu-" + tool)
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o755)
        self.base = self.make_baseline("base", "0")
        self.args = argparse.Namespace(
            repo=str(self.repo),
            artifact=str(self.archive / "artifact.json"),
            baseline=[str(self.base)],
            output=str(self.root / "candidate"),
            jobs=2,
            clean_root=[],
            require_new_selector=False,
            check_only=False,
            cann=str(self.cann),
        )

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args], stderr=subprocess.PIPE
        )

    def make_baseline(self, name, selector, fail=False):
        root = self.root / name
        root.mkdir()
        log = root / "case.log"
        text = (
            "the selected algo type is old_ring\n[CHECKER_RUN_SUMMARY] "
            + ("Failed" if fail else "All Success (Total Op: 1, Big Graph: 1)")
            + "\n[CMD DONE] mpirun (exit=0)\n"
        )
        log.write_text(text)
        manifest = dict(
            schema_version=1,
            role="baseline",
            dry_run=False,
            source_id="old",
            declared_artifact=self.artifact,
            expected_cases=["case"],
            environment={"HCCL_USE_NEW_SELECTOR": selector, "HCCL_ALGO": ""},
            installed_libraries=[
                {"kind": "host", "sha256": "hosthash"},
                {"kind": "device", "sha256": "devicehash"},
            ],
        )
        workflow.write_new(root / "manifest.json", manifest)
        row = workflow.evidence_module().parse_log(text)
        row.update(
            case_id="case",
            raw_log=str(log),
            raw_log_sha256=workflow.digest(log),
            test_binary_sha256="fakebin",
        )
        (root / "results.jsonl").write_text(json.dumps(row) + "\n")
        return root

    def alter_manifest(self, **fields):
        path = self.base / "manifest.json"
        data = json.loads(path.read_text())
        data.update(fields)
        path.write_text(json.dumps(data))

    def test_preflight_changes_include_untracked_and_environment(self):
        added = self.repo / "src/reduce_scatter/new.h"
        added.write_text("int GetThreadNum();\n")
        env = self.root / "environment.txt"
        env.write_text("version 1")
        a = argparse.Namespace(
            repo=str(self.repo),
            op="reducescatter",
            output=str(self.root / "scan1.json"),
            previous=None,
            environment_file=[str(env)],
        )
        first = workflow.preflight(a)
        self.assertIn("src/reduce_scatter/new.h", first["files"])
        self.assertEqual(first["locations"][0]["locations"][0]["line"], 1)
        a.previous, a.output = a.output, str(self.root / "scan2.json")
        added.write_text("int GetThreadNum() { return 2; }\n")
        env.write_text("version 2")
        second = workflow.preflight(a)
        self.assertEqual(second["changed_files"], ["src/reduce_scatter/new.h"])
        self.assertEqual(second["changed_environment_files"], [str(env)])
        with self.assertRaises(FileExistsError):
            workflow.preflight(a)

    def test_preflight_includes_common_mapping_and_cost_without_op_name(self):
        names = [
            "src/common/alg_parse.cc",
            "src/ops/op_common/selector/cost_model.cc",
            "src/ops/op_common/algorithm/template/alg_v2_template_base.h",
        ]
        for name in names:
            p = self.repo / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("CalcCostCoeff();\n")
        a = argparse.Namespace(
            repo=str(self.repo),
            op="reduce_scatter",
            output=str(self.root / "scan.json"),
            previous=None,
            environment_file=[],
        )
        result = workflow.preflight(a)
        for name in names:
            self.assertIn(name, result["files"])

    def test_build_once_preserves_log_exit_identity(self):
        result = workflow.build(self.args)
        out = Path(self.args.output)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(
            (out / "build.log").read_text().strip(),
            "--pkg --full -j2 -p " + str(self.cann),
        )
        self.assertEqual((out / "build.exit_code").read_text(), "0\n")
        self.assertTrue((out / "source.diff").is_file())
        self.assertTrue((out / "completion.json").is_file())
        self.assertTrue((out / "source-snapshot/snapshot.json").is_file())
        with self.assertRaises(ValueError):
            workflow.build(self.args)
        self.assertEqual((self.repo / "calls").read_text(), "called\n")

    def test_failed_build_keeps_evidence_and_cannot_repeat(self):
        (self.repo / "build.sh").write_text("echo failed\nexit 9\n")
        result = workflow.build(self.args)
        self.assertEqual(result["exit_code"], 9)
        self.assertEqual(
            (Path(self.args.output) / "build.exit_code").read_text(), "9\n"
        )
        with self.assertRaises(ValueError):
            workflow.build(self.args)

    def test_check_only_no_build_no_output_and_fail_not_waived(self):
        self.args.baseline = [str(self.make_baseline("failed", "0", fail=True))]
        self.args.check_only = True
        result = workflow.build(self.args)
        self.assertFalse(result["baseline_all_passed"])
        self.assertEqual(result["baseline"][0]["failures"], 1)
        self.assertFalse(Path(self.args.output).exists())
        self.assertFalse((self.repo / "calls").exists())

    def test_requires_role_and_archive_identity(self):
        path = self.base / "manifest.json"
        data = json.loads(path.read_text())
        del data["role"]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "role=baseline"):
            workflow.gate(self.args)
        self.alter_manifest(role="test")
        with self.assertRaisesRegex(ValueError, "role=baseline"):
            workflow.gate(self.args)
        self.alter_manifest(
            role="baseline",
            declared_artifact=dict(self.artifact, environment_id="different"),
        )
        with self.assertRaisesRegex(ValueError, "旧包不一致"):
            workflow.gate(self.args)

    def test_default_baseline_requires_selector_and_no_override(self):
        self.args.require_new_selector = True
        with self.assertRaisesRegex(ValueError, "selector=1"):
            workflow.gate(self.args)
        self.args.baseline.append(str(self.make_baseline("newselector", "1")))
        workflow.gate(self.args)
        self.alter_manifest(
            environment={
                "HCCL_USE_NEW_SELECTOR": "0",
                "HCCL_ALGO": "reducescatter:sole{ring}",
            }
        )
        with self.assertRaisesRegex(ValueError, "HCCL_ALGO"):
            workflow.gate(self.args)

    def test_cleanup_roots_protect_archive_baseline_rawlogs_and_output(self):
        self.args.clean_root = [str(self.archive)]
        with self.assertRaisesRegex(ValueError, "清理目录"):
            workflow.gate(self.args)
        self.args.clean_root = [str(self.base)]
        with self.assertRaisesRegex(ValueError, "清理目录"):
            workflow.gate(self.args)
        self.args.clean_root = []
        self.args.output = str(self.repo / "build_out/candidate")
        with self.assertRaisesRegex(ValueError, "清理目录"):
            workflow.gate(self.args)
        self.args.output = str(self.root / "candidate")
        raw = self.repo / "build/raw.log"
        raw.parent.mkdir()
        raw.write_bytes((self.base / "case.log").read_bytes())
        rows = self.base / "results.jsonl"
        record = json.loads(rows.read_text())
        record["raw_log"] = str(raw)
        rows.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "清理目录"):
            workflow.gate(self.args)

    def test_missing_or_modified_archive_rejected(self):
        (self.archive / "package.run").write_text("changed")
        with self.assertRaisesRegex(ValueError, "旧包"):
            workflow.gate(self.args)

    def test_archive_logs_exit_and_symlink_cleanup_checked(self):
        rc = self.archive / "build.exit_code"
        rc.write_text("5\n")
        with self.assertRaisesRegex(ValueError, "退出码"):
            workflow.gate(self.args)
        rc.write_text("0\n")
        log = self.archive / "build.log"
        log.write_text("changed log")
        with self.assertRaisesRegex(ValueError, "构建日志"):
            workflow.gate(self.args)
        log.write_text("fake log")
        clean = self.repo / "build"
        clean.mkdir()
        link = self.root / "looks-persistent"
        link.symlink_to(clean, target_is_directory=True)
        self.args.output = str(link / "candidate")
        with self.assertRaisesRegex(ValueError, "清理目录"):
            workflow.gate(self.args)

    def test_environment_check_precedes_gate_and_sources_child_only(self):
        self.args.artifact = "missing-artifact"
        self.args.cann = str(self.root / "missing-cann")
        self.args.check_only = True
        with self.assertRaisesRegex(ValueError, "CANN 环境"):
            workflow.build(self.args)
        self.args.cann = str(self.cann / "set_env.sh")
        env, report = workflow.build_environment(self.args)
        self.assertEqual(env["SMOKE_CANN_LOADED"], "yes")
        self.assertEqual(report["cann"], str(self.cann))
        self.assertNotEqual(os.environ.get("SMOKE_CANN_LOADED"), "yes")
        self.args.cann = None
        with patch.dict(
            os.environ, {"ASCEND_OPP_PATH": str(self.cann / "opp")}, clear=True
        ):
            env, report = workflow.build_environment(self.args)
        self.assertEqual(report["origin"], "ASCEND_OPP_PATH")
        missing = self.cann / "toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++"
        missing.unlink()
        self.args.cann = str(self.cann)
        with self.assertRaisesRegex(ValueError, r"device-g\+\+"):
            workflow.build_environment(self.args)

    def test_task_inventory_only_explicit_directory_and_metadata(self):
        task = self.root / "previous-task"
        task.mkdir()
        (task / "spec.md").write_text("contract")
        (task / "package.run").write_text("large package ignored")
        (task / "external").symlink_to(self.repo, target_is_directory=True)
        first = workflow.task_inventory(task)
        self.assertEqual([row["path"] for row in first["entries"]], ["spec.md"])
        (task / "spec.md").write_text("new contract")
        self.assertNotEqual(
            first["fingerprint"], workflow.task_inventory(task)["fingerprint"]
        )

    def test_all_in_repo_output_paths_rejected_before_mutation(self):
        for check_only in (True, False):
            for output in (self.repo, self.repo / "artifacts/candidate"):
                self.args.check_only = check_only
                self.args.output = str(output)
                with self.assertRaisesRegex(ValueError, "源码仓之外"):
                    workflow.build(self.args)
                self.assertFalse((self.repo / "artifacts").exists())
                self.assertFalse((self.repo / "calls").exists())

    def test_empty_baseline_rejected(self):
        self.alter_manifest(expected_cases=[])
        (self.base / "results.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "不能为空"):
            workflow.gate(self.args)


if __name__ == "__main__":
    unittest.main()
