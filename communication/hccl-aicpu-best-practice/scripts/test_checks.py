# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Offline behavior tests; all generated fixtures live in a temporary directory."""

import contextlib
import hashlib
import io
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest

import check_template
import list_templates
import check_sources
import new_template
from cmake_utils import has_source

SKILL = Path(__file__).resolve().parent.parent


class SourceContractTests(unittest.TestCase):
    def test_changes_are_scoped_and_include_working_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = dict(
                schema_version=1,
                source_commits={"hccl": "recorded", "hcomm": "recorded"},
                shared={},
                hcomm={},
                operators={"all_reduce": {}, "all_gather": {}},
            )
            for group, name in [
                (baseline["shared"], "wrapper.h"),
                (baseline["hcomm"], "hcomm.h"),
                (baseline["operators"]["all_reduce"], "ar.cc"),
                (baseline["operators"]["all_gather"], "ag.cc"),
            ]:
                (root / name).write_bytes(b"original")
                group[name] = hashlib.sha256(b"original").hexdigest()

            def check():
                return check_sources.check(baseline, root, root, "all_reduce")

            self.assertEqual(check()["status"], "MATCH")
            (root / "ag.cc").write_bytes(b"unrelated change")
            self.assertEqual(check()["status"], "MATCH")
            (root / "ar.cc").write_bytes(b"uncommitted change")
            changed = check()
            self.assertEqual(changed["status"], "CHANGED")
            self.assertEqual(changed["groups"]["hccl_operator"]["changed"], ["ar.cc"])
            self.assertEqual(changed["groups"]["hccl_operator"]["status"], "CHANGED")
            self.assertEqual(changed["groups"]["hccl_shared"]["status"], "MATCH")
            (root / "ar.cc").write_bytes(b"original")
            (root / "wrapper.h").unlink()
            self.assertEqual(check()["groups"]["hccl_shared"]["missing"], ["wrapper.h"])
            (root / "wrapper.h").write_bytes(b"original")
            partial = check_sources.check(baseline, root, None, "all_reduce")
            self.assertEqual(partial["status"], "UNVERIFIED")
            self.assertEqual(partial["groups"]["hcomm_api"]["unverified"], ["hcomm.h"])
            self.assertEqual(partial["groups"]["hcomm_api"]["status"], "UNVERIFIED")
            with self.assertRaises(ValueError):
                check_sources.check(baseline, root, root, "unknown_op")

    def test_custom_algorithm_identity_does_not_become_mesh(self):
        for cls in (
            "InsTempAllReduceRing",
            "InsTempAllReduceRingOneShot",
            "InsTempAllGatherTreeTwoShot",
        ):
            self.assertEqual(new_template.infer_algo_type(cls, "barebone"), "UNKNOWN")
        self.assertEqual(
            new_template.infer_algo_type("InsTempAllReduceMesh1DOneShot", "barebone"),
            "MESH_ONESHOT",
        )
        self.assertEqual(
            new_template.infer_algo_type("InsTempAllReduceNHR", "barebone"), "NHR"
        )

    def test_optional_executor_contract_is_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base").write_text("base")
            (root / "parallel").write_text("parallel")
            base = {"base": hashlib.sha256(b"base").hexdigest()}
            files = {"parallel": hashlib.sha256(b"parallel").hexdigest()}
            baseline = dict(
                schema_version=1,
                source_commits={},
                shared=base,
                hcomm=base,
                operators={"all_gather": base, "all_reduce": base},
                contracts={
                    "ag-parallel": dict(
                        op="all_gather",
                        source_commit="base",
                        reference="card.md",
                        files=files,
                    )
                },
            )

            def check(op="all_gather", contracts=()):
                return check_sources.check(baseline, root, root, op, contracts)

            self.assertEqual(check(contracts=["ag-parallel"])["status"], "MATCH")
            (root / "parallel").write_text("changed")
            self.assertEqual(check()["status"], "MATCH")
            result = check(contracts=["ag-parallel"])
            self.assertEqual(result["status"], "CHANGED")
            self.assertEqual(
                result["groups"]["hccl_contract:ag-parallel"]["changed"], ["parallel"]
            )
            (root / "parallel").write_text("parallel")
            (root / "base").write_text("unrelated-to-parallel")
            result = check(contracts=["ag-parallel"])
            self.assertEqual(result["status"], "CHANGED")
            self.assertEqual(
                result["groups"]["hccl_contract:ag-parallel"]["status"], "MATCH"
            )
            for op, name in [("all_gather", "unknown"), ("all_reduce", "ag-parallel")]:
                with self.assertRaises(ValueError):
                    check(op, [name])


class TemplateTests(unittest.TestCase):
    def test_output_placement_gate_catches_missing_branch(self):
        copy = (
            "DataSlice src(tempAlgParams.buffInfo.hcclBuff.addr, "
            "tempAlgParams.buffInfo.hcclBuffBaseOff, size, count); "
            "DataSlice dst(tempAlgParams.buffInfo.outputPtr, "
            "tempAlgParams.buffInfo.outBuffBaseOff, size, count); "
            "return LocalCopy(thread, src, dst);"
        )
        for cls, code, expected in [
            ("InsTempScatterRing", copy, True),
            ("InsTempScatterRing", copy + "\n// buffInfo.outBuffType is checked", True),
            (
                "InsTempScatterRing",
                copy + "\nif (tempAlgParams.buffInfo.outBuffType == "
                "BufferType::HCCL_BUFFER) return HCCL_SUCCESS;",
                False,
            ),
            ("InsTempScatterDpuInter", copy, False),
            ("InsTempScatterOmniPipe", copy, False),
        ]:
            with self.subTest(cls=cls, code=code):
                report = check_template.Report("probe", strict_new=True)
                check_template.check_semantics(report, cls, code, "")
                hit = [
                    level for level, rid, _, _ in report.items if rid == "R-FLOW-022"
                ]
                self.assertEqual(hit, ["ERROR"] if expected else [])
        self.assertFalse(
            check_template.has_scratch_to_output_copy(
                "DataSlice src(tempAlgParams.buffInfo.inputPtr, 0, size, count); "
                "DataSlice dst(tempAlgParams.buffInfo.outputPtr, 0, size, count); "
                "auto scratch = tempAlgParams.buffInfo.hcclBuffBaseOff; "
                "return LocalCopy(thread, src, dst);"
            )
        )
        split_methods = (
            "HcclResult InsTempScatterRing::KeepChunk() {" + copy + "}\n"
            "bool InsTempScatterRing::CheckSupport() {"
            "return tempAlgParams.buffInfo.outBuffType == BufferType::OUTPUT; }"
        )
        self.assertTrue(
            check_template.output_copy_without_guard(
                split_methods, "InsTempScatterRing"
            )
        )
        self.assertEqual(
            check_template.effective_level("R-FLOW-022", "WARN", False), "WARN"
        )

    def test_sizeof_scalar_is_not_pointer_size(self):
        for code, expected in [
            ("void F() { u32 value = 0; auto n = sizeof(value); }", False),
            ("void F() { u32 *value = nullptr; auto n = sizeof(value); }", True),
        ]:
            report = check_template.Report("probe")
            check_template.check_semantics(report, "Probe", code, "")
            self.assertEqual(
                any(
                    rid == "R-SAFE-010" and level == "ERROR"
                    for level, rid, _, _ in report.items
                ),
                expected,
            )

    def test_strict_scope_and_missing_file(self):
        self.assertEqual(
            check_template.effective_level("R-SAFE-004", "WARN", False), "WARN"
        )
        self.assertEqual(
            check_template.effective_level("R-SAFE-004", "WARN", True), "ERROR"
        )
        self.assertGreater(
            check_template.check_one("/tmp", "/tmp/absent-hccl-probe.cc").errors, 0
        )

    def test_named_cost_penalty_is_not_a_validator_escape(self):
        report = check_template.Report("probe")
        check_template.check_semantics(
            report,
            "Probe",
            "constexpr float PENALTY = 1.0e9f;\n"
            "auto Probe::CalcCostCoeff(P p) { return {{0.0f, 0.0f, PENALTY, 0.0f}}; }",
            "",
        )
        self.assertTrue(any(rid == "R-DEC-009" for _, rid, _, _ in report.items))

    def explicit_report(self, body):
        report = check_template.Report("probe", strict_new=True)
        check_template.check_explicit_only(
            report,
            "Probe",
            "std::vector<CostModelParam> Probe::CalcCostCoeff(P p) {" + body + "}",
            "",
        )
        return report

    def test_explicit_only_rejects_interface_derived_unconditional_cost(self):
        # Reduced from opencode_glm5.3_rs_ring.diff: cost-only ifs do not gate candidates.
        report = self.explicit_report(
            "// Only HCCL_ALGO explicit routing; if (!enabled) return {};\n"
            "float A = 0.0f; float B = 0.0f; float C = 0.0f; float D = 0.0f;"
            "CostModelManager::Global()->CalcMeshParam(p.dataRatio, A);"
            "if (p.inputBuffer != p.scratchBuffer) {"
            "CostModelManager::Global()->CalcLocalCopyParams(p.dataRatio, B); }"
            "std::vector<CostModelParam> params; params.push_back({A, B, C, D}); return params;"
        )
        self.assertTrue(
            any(
                level == "ERROR" and rid == "R-DEC-015"
                for level, rid, _, _ in report.items
            )
        )
        self.assertEqual(
            self.explicit_report("return {{0.0f, 0.0f, 0.0f, 0.0f}};").errors, 1
        )

    def test_explicit_only_never_proves_guard_semantics(self):
        for body in [
            "if (p.rankSize == 0) return {}; return {{0.0f, 0.0f, 0.0f, 0.0f}};",
            "return ExistingHelper(p);",
            "std::vector<CostModelParam> params; if (p.enabled) "
            "params.push_back({0.0f, 0.0f, 0.0f, 0.0f}); return params;",
            "RETURN_IF_DISABLED(p); return {{0.0f, 0.0f, 0.0f, 0.0f}};",
            "return {};",
        ]:
            with self.subTest(body=body):
                report = self.explicit_report(body)
                self.assertEqual(report.errors, 0)
                self.assertTrue(
                    any(
                        level == "WARN" and "人工复核" in message
                        for level, _, message, _ in report.items
                    )
                )
                self.assertFalse(any(level == "OK" for level, _, _, _ in report.items))

    def test_explicit_only_cli_requires_one_target(self):
        for args in [[], ["--all", "probe.cc"], ["one.cc", "two.cc"]]:
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(SKILL / "scripts/check_template.py"),
                    "--explicit-only",
                ]
                + args,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("--explicit-only 必须指定单个目标", result.stderr)

    def test_cmake_registration(self):
        entry = "${CMAKE_CURRENT_SOURCE_DIR}/probe.cc"
        for body in [
            "# " + entry,
            "#[[\n" + entry + "\n]]",
            "#[=[\n" + entry + "\n]=]",
        ]:
            self.assertFalse(has_source("set(src_list\n" + body + "\n)", entry, "host"))
        self.assertFalse(has_source("set(other\n" + entry + "\n)", entry, "host"))
        self.assertTrue(
            has_source('list(APPEND src_list\n"' + entry + '"\n)', entry, "host")
        )

    def test_generated_templates_in_both_layouts(self):
        for layout in ["algorithm/template/aicpu", "template/aicpu"]:
            for pattern, cls, filename in [
                (
                    "all-reduce-mesh-oneshot",
                    "InsTempAllReduceMesh1DProbe",
                    "ins_temp_all_reduce_mesh_1D_probe",
                ),
                (
                    "barebone",
                    "InsTempAllReduceNhrProbe",
                    "ins_temp_all_reduce_nhr_probe",
                ),
            ]:
                with (
                    self.subTest(layout=layout, pattern=pattern),
                    tempfile.TemporaryDirectory(prefix="hccl-generator-test-") as tmp,
                ):
                    repo = Path(tmp)
                    out = repo / "src/ops/all_reduce" / layout
                    out.mkdir(parents=True)
                    common = repo / "src/common"
                    common.mkdir()
                    (common / "alg_parse.h").write_text(
                        "enum class AlgoType : unsigned { MESH, MESH_ONESHOT, NHR, UNKNOWN };"
                    )
                    entry = "${CMAKE_CURRENT_SOURCE_DIR}/" + filename + ".cc"
                    local = out / "CMakeLists.txt"
                    local.write_text("set(src_list\n# " + entry + "\n)\n")
                    device_entry = (
                        "${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/"
                        + layout
                        + "/"
                        + filename
                        + ".cc"
                    )
                    device = repo / "src/scatter_aicpu_kernel.cmake"
                    device.write_text(
                        "add_library(scatter_aicpu_kernel SHARED\n# "
                        + device_entry
                        + "\n)\n"
                    )
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-B",
                            str(SKILL / "scripts/new_template.py"),
                            "--repo",
                            tmp,
                            "--op",
                            "all_reduce",
                            "--class",
                            cls,
                            "--pattern",
                            pattern,
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertTrue(has_source(local.read_text(), entry, "host"))
                    self.assertTrue(
                        has_source(device.read_text(), device_entry, "device")
                    )
                    report = check_template.check_one(
                        tmp, str(out / (filename + ".cc")), strict_new=True
                    )
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        report.dump()
                    self.assertEqual(report.errors, 0, output.getvalue())
                    executor = out.parent.parent / "executor"
                    executor.mkdir()
                    (executor / "parallel.cc").write_text(
                        "REGISTER_EXECUTOR_BY_TWO_TEMPS(Cmd, Name, Exec, Topo, "
                        + cls
                        + ", "
                        + cls
                        + ");\n"
                        "void PrepareResForTemplate() { temp.GetRes(request); }"
                    )
                    resource_report = check_template.check_one(
                        tmp, str(out / (filename + ".cc")), strict_new=True
                    )
                    self.assertTrue(
                        any(
                            level == "ERROR" and rid == "R-IFACE-014"
                            for level, rid, _, _ in resource_report.items
                        )
                    )
                    source = out / (filename + ".cc")
                    source.write_text(
                        source.read_text()
                        + "\nHcclResult "
                        + cls
                        + "::GetRes(AlgResourceRequest &r) const { return HCCL_SUCCESS; }\n"
                    )
                    resource_report = check_template.check_one(
                        tmp, str(source), strict_new=True
                    )
                    self.assertFalse(
                        any(
                            rid == "R-IFACE-014" and level == "ERROR"
                            for level, rid, _, _ in resource_report.items
                        )
                    )
                    device.write_text(
                        "add_library(scatter_aicpu_kernel SHARED\n# "
                        + device_entry
                        + "\n)\n"
                    )
                    report = check_template.check_one(
                        tmp, str(out / (filename + ".cc"))
                    )
                    self.assertTrue(
                        any(
                            level == "ERROR" and rid == "R-CMAKE-002"
                            for level, rid, _, _ in report.items
                        )
                    )


class GeneratorBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hccl-generator-boundary-")
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        self.out = self.repo / "src/ops/all_reduce/algorithm/template/aicpu"
        self.out.mkdir(parents=True)
        self.local = self.out / "CMakeLists.txt"
        self.device = self.repo / "src/scatter_aicpu_kernel.cmake"
        self.local.write_text("set(src_list\n)\n")
        self.device.write_text("add_library(scatter_aicpu_kernel SHARED\n)\n")
        self.entry = "${CMAKE_CURRENT_SOURCE_DIR}/ins_temp_probe.cc"
        self.device_entry = "${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/algorithm/template/aicpu/ins_temp_probe.cc"

    def generate(self, *extra):
        return subprocess.run(
            [
                sys.executable,
                "-B",
                str(SKILL / "scripts/new_template.py"),
                "--repo",
                str(self.repo),
                "--op",
                "all_reduce",
                "--class",
                "InsTempProbe",
                *extra,
            ],
            capture_output=True,
            text=True,
        )

    @unittest.skipUnless(
        shutil.which("g++"), "requires g++ for offline generated-stub behavior"
    )
    def test_unimplemented_custom_template_fails_without_mutating_resources(self):
        result = self.generate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # Minimal API fixture intentionally has no Mesh, NHR, transfer or reduction functions.
        # This verifies the generated stub's behavior, not compatibility with real HCCL headers.
        (self.out / "alg_v2_template_base.h").write_text("""
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#define HCCL_ERROR(...) ((void)0)
namespace ops_hccl {
using u32 = uint32_t; using u64 = uint64_t; using HcclComm = void*;
enum class HcclResult { HCCL_SUCCESS, HCCL_E_INTERNAL };
enum class AlgoType { UNKNOWN };
enum class BufferType { INPUT, OUTPUT };
struct TemplateProp { AlgoType algoType; };
struct OpParam {}; struct TopoInfoWithNetLayerDetails {}; struct TemplateDataParams {};
struct CalcCostCoeffParam {}; struct CostModelParam {};
struct AlgResourceRequest { int sentinel = 17; };
struct TemplateResource { int sentinel = 23; };
class InsAlgTemplateBase {
public:
    InsAlgTemplateBase() = default;
    InsAlgTemplateBase(const OpParam&, u32, const std::vector<std::vector<u32>>&) {}
    virtual ~InsAlgTemplateBase() = default;
    virtual std::string Describe() const = 0;
    virtual HcclResult CalcRes(HcclComm, const OpParam&, const TopoInfoWithNetLayerDetails*, AlgResourceRequest&) = 0;
    virtual HcclResult KernelRun(const OpParam&, const TemplateDataParams&, TemplateResource&) = 0;
    virtual u64 CalcScratchMultiple(BufferType, BufferType) = 0;
    virtual void GetNotifyIdxMainToSub(std::vector<u32>&) = 0;
    virtual void GetNotifyIdxSubToMain(std::vector<u32>&) = 0;
protected:
    u32 templateRankSize_ = 0;
};
}
""")
        for name in ("executor_base.h", "alg_data_trans_wrapper.h"):
            (self.out / name).write_text("#pragma once\n")
        (self.out / "main.cc").write_text("""
#include "ins_temp_probe.h"
#include <cassert>
using namespace ops_hccl;
int main() {
    InsTempProbe probe;
    AlgResourceRequest request;
    TemplateResource resources;
    assert(probe.CalcCostCoeff({}).empty());
    assert(probe.CalcRes(nullptr, {}, nullptr, request) != HcclResult::HCCL_SUCCESS);
    assert(request.sentinel == 17);
    assert(probe.KernelRun({}, {}, resources) != HcclResult::HCCL_SUCCESS);
    assert(resources.sentinel == 23);
}
""")
        binary = self.out / "probe"
        compiled = subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(self.out / "ins_temp_probe.cc"),
                str(self.out / "main.cc"),
                "-o",
                str(binary),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
        subprocess.run([str(binary)], check=True, timeout=10)

    def test_bracket_comment_blocks_are_ignored(self):
        for path in (self.local, self.device):
            active = path.read_text()
            path.write_text("#[=[\n" + active + "]=]\n" + active)
        result = self.generate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(has_source(self.local.read_text(), self.entry, "host"))
        self.assertTrue(
            has_source(self.device.read_text(), self.device_entry, "device")
        )
        self.assertEqual(
            list_templates.cmake_status(
                str(self.repo),
                "all_reduce",
                "all_reduce/algorithm",
                "ins_temp_probe",
                "aicpu",
            ),
            "HD",
        )
        self.local.write_text("#[[\n" + self.local.read_text() + "\n]]")
        self.device.write_text("# " + self.device_entry)
        self.assertEqual(
            list_templates.cmake_status(
                str(self.repo),
                "all_reduce",
                "all_reduce/algorithm",
                "ins_temp_probe",
                "aicpu",
            ),
            "--",
        )

    def test_guard_requires_correct_branch(self):
        blocks = [
            (self.local, "list(APPEND src_list\n)\n"),
            (self.device, "target_sources(scatter_aicpu_kernel PRIVATE\n)\n"),
        ]
        for path, block in blocks:
            path.write_text(
                "if(OTHER)\n" + block + "endif()\n"
                "if(NOT HCCL_CANN_COMPAT_850)\nelse()\n" + block + "endif()\n"
            )
        before = [p.read_bytes() for p, _ in blocks]
        self.assertNotEqual(self.generate("--compat-guard").returncode, 0)
        self.assertFalse((self.out / "ins_temp_probe.cc").exists())
        self.assertEqual(before, [p.read_bytes() for p, _ in blocks])
        for path, block in blocks:
            path.write_text(
                path.read_text()
                + "if(NOT HCCL_CANN_COMPAT_850)\n"
                + block
                + "endif()\n"
            )
        result = self.generate("--compat-guard")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(
            has_source(self.local.read_text(), self.entry, "host", compat_guard=True)
        )
        self.assertTrue(
            has_source(
                self.device.read_text(), self.device_entry, "device", compat_guard=True
            )
        )

    def test_paths_and_identifiers_rejected_without_writes(self):
        before = {p: p.read_bytes() for p in (self.local, self.device)}
        for flag, value in [
            ("--file", str(Path(self.tmp.name) / "escape")),
            ("--file", "../escape"),
            ("--file", "probe.cc"),
            ("--op", "../all_reduce"),
            ("--class", "../Escape"),
        ]:
            with self.subTest(flag=flag, value=value):
                self.assertNotEqual(self.generate(flag, value, "--force").returncode, 0)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["CMakeLists.txt"])
        self.assertFalse((Path(self.tmp.name) / "escape.cc").exists())

    def test_symlink_targets_rejected_even_with_force(self):
        external = Path(self.tmp.name) / "external"
        external.write_text("unchanged")
        for target in (self.out / "ins_temp_probe.cc", self.local, self.device):
            with self.subTest(target=target):
                old = target.read_bytes() if target.exists() else None
                target.unlink(missing_ok=True)
                target.symlink_to(external)
                self.assertNotEqual(self.generate("--force").returncode, 0)
                self.assertEqual(external.read_text(), "unchanged")
                target.unlink()
                if old is not None:
                    target.write_bytes(old)
        actual = self.out.with_name("actual")
        self.out.rename(actual)
        self.out.symlink_to(actual, target_is_directory=True)
        self.assertNotEqual(self.generate("--force").returncode, 0)
        self.assertFalse((actual / "ins_temp_probe.cc").exists())


if __name__ == "__main__":
    unittest.main()
