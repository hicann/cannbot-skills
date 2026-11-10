# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU-only regression tests; fixtures contain no runnable operator implementation."""

import itertools
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from migration_contract import (
    check_operator,
    cmake_calls,
    example_files,
    profile,
    validate_decision,
)


class MigrationContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture_count = 0

    def write(self, path, body=""):
        path.parent.mkdir(parents=True, exist_ok=True)
        prefix = (
            "#"
            if path.name == "CMakeLists.txt" or path.suffix in {".py", ".cmake"}
            else "//"
        )
        path.write_text(
            f"{prefix} Copyright (c) 2026 Huawei Technologies Co., Ltd.\n{body}\n"
        )

    def fixture(
        self,
        repo,
        op="sample",
        optype="Sample",
        socs=("ascend910b", "ascend950"),
        manual_api=True,
        api_stem=None,
    ):
        config = profile(repo)
        self.fixture_count += 1
        root = (
            Path(self.temp.name)
            / str(self.fixture_count)
            / repo
            / config["classes"][0]
            / op
        )
        arches = [
            {"ascend910b": "arch22", "ascend310p": "arch20"}.get(soc, "arch35")
            for soc in socs
        ]
        api_stem = api_stem or op
        self.write_source_files(root, optype, socs, arches)
        if manual_api:
            for stem in [api_stem, api_stem + "_grad"]:
                self.write(root / f"op_api/aclnn_{stem}.cpp")
                self.write(root / f"tests/ut/op_api/test_aclnn_{stem}.cpp")
        self.write_build_cmake(root, config, socs, arches, manual_api)
        if config["explicit_tests"]:
            self.write_ut_cmake(root, repo, arches)
        return root

    def write_source_files(self, root, optype, socs, arches):
        op = root.name
        for path in [
            f"op_kernel/{op}_apt.cpp",
            f"tests/ut/op_host/test_{op}_infershape.cpp",
            f"examples/test_aclnn_{op}.cpp",
            f"examples/test_geir_{op}.cpp",
        ]:
            self.write(root / path)
        for arch in set(arches):
            self.write(root / f"op_host/{arch}/{op}_tiling.cpp")
            self.write(root / f"tests/ut/op_host/{arch}/test_{op}_tiling.cpp")
        self.write(root / f"op_graph/{op}_proto.h", f"REG_OP({optype})")
        self.write(
            root / f"op_host/{op}_def.cpp",
            f"OP_ADD({optype});\n"
            + "\n".join(f'AddConfig("{soc}", config);' for soc in socs)
            + f'\nExtendCfgInfo("opFile.value", "{op}_apt");',
        )

    def write_build_cmake(self, root, config, socs, arches, manual_api):
        op = root.name
        aclnn_mode = "aclnn_exclude" if manual_api else "aclnn"
        if config["tiling_lists"]:
            self.write(
                root / "CMakeLists.txt",
                "set(SUPPORT_COMPUTE_UNIT "
                + " ".join(f'"{soc}"' for soc in socs)
                + ")\nset(SUPPORT_TILING_DIR "
                + " ".join(f'"{arch}"' for arch in arches)
                + f")\nadd_all_modules_sources(OPTYPE {op} ACLNNTYPE {aclnn_mode}\n"
                + "    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}\n"
                + "    TILING_DIR ${SUPPORT_TILING_DIR})\n",
            )
        else:
            self.write(
                root / "CMakeLists.txt",
                "add_subdirectory(op_host)\nadd_subdirectory(tests)",
            )
            self.write(
                root / "op_host/CMakeLists.txt",
                "add_op_to_compiled_list()\n"
                + f"add_modules_sources_with_soc(OPTYPE {op} ACLNNTYPE {aclnn_mode}"
                + (
                    " OP_API_INDEPENDENT ON OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api"
                    if manual_api
                    else ""
                )
                + ")\n"
                + f"target_sources({'op_host_aclnnExc' if manual_api else 'op_host_aclnn'} PRIVATE {op}_def.cpp)\n",
            )

    def write_ut_cmake(self, root, repo, arches):
        p = root / "CMakeLists.txt"
        if repo != "ops-transformer":
            p.write_text(p.read_text() + "\nadd_subdirectory(tests)\n")
        self.write(root / "tests/CMakeLists.txt", "add_subdirectory(ut)")
        self.write(
            root / "tests/ut/CMakeLists.txt",
            "add_subdirectory(op_host)\nadd_subdirectory(op_api)",
        )
        host_sources = (
            "add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} "
            "MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})\n"
        )
        tiling_sources = (
            "add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} "
            "MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR}/"
        )
        self.write(
            root / "tests/ut/op_host/CMakeLists.txt",
            host_sources
            + "\n".join(tiling_sources + arch + ")" for arch in set(arches)),
        )
        self.write(
            root / "tests/ut/op_api/CMakeLists.txt",
            "add_modules_ut_sources(UT_NAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})",
        )

    def check(self, root, repo, soc="ascend950", source_optype=None):
        with self.assertLogs("migration_contract", level="INFO"):
            return check_operator(root, soc, repo, source_optype)

    def test_four_repo_layouts_and_selected_second_soc(self):
        for repo in ["ops-math", "ops-nn", "ops-cv", "ops-transformer"]:
            with self.subTest(repo=repo):
                self.assertEqual(self.check(self.fixture(repo), repo), 0)

    def test_operator_names_types_api_modes_and_socs_are_not_sample_constants(self):
        identities = [
            ("vector_affine_v7", "VectorAffineV7", "affine_vector"),
            ("gn_training_reduce_v2", "GNTrainingReduceV2", "gn_train_reduce"),
            ("roi_align_v3", "ROIAlignV3", "roi_align"),
            ("kv_cache_pack_v2", "KVCachePackV2", "cache_pack"),
        ]
        repositories = ["ops-math", "ops-nn", "ops-cv", "ops-transformer"]
        for repo, identity, manual_api in itertools.product(
            repositories, identities, [False, True]
        ):
            op, optype, api_stem = identity
            root = self.fixture(
                repo, op, optype, manual_api=manual_api, api_stem=api_stem
            )
            for soc in ["ascend910b", "ascend950"]:
                with self.subTest(repo=repo, op=op, manual_api=manual_api, soc=soc):
                    self.assertEqual(self.check(root, repo, soc, optype), 0)

    def test_a2_only_operator_has_no_a5_requirement(self):
        for repo in ["ops-math", "ops-nn", "ops-cv", "ops-transformer"]:
            with self.subTest(repo=repo):
                root = self.fixture(repo, socs=("ascend910b",))
                self.assertEqual(self.check(root, repo, "ascend910b", "Sample"), 0)
                self.assertEqual(self.check(root, repo, "ascend950", "Sample"), 1)

    def test_source_registration_preserves_acronyms_and_rejects_stale_types(self):
        for repo in ["ops-math", "ops-nn", "ops-cv", "ops-transformer"]:
            with self.subTest(repo=repo):
                root = self.fixture(repo, "gn_reduce_v2", "GNReduceV2")
                self.assertEqual(self.check(root, repo, source_optype="GNReduceV2"), 0)
                self.assertEqual(self.check(root, repo, source_optype="GnReduceV2"), 1)
                proto = root / "op_graph/gn_reduce_v2_proto.h"
                self.write(proto, "REG_OP(OldTemplate)\n// REG_OP(GNReduceV2)")
                self.assertEqual(self.check(root, repo, source_optype="GNReduceV2"), 1)

    def test_opdef_filename_must_match_directory(self):
        root = self.fixture("ops-math")
        old = root / "op_host/sample_def.cpp"
        old.rename(old.with_name("stale_template_def.cpp"))
        self.assertEqual(self.check(root, "ops-math"), 1)

    def test_example_interface_name_can_differ_from_operator_name(self):
        for repo in ["ops-math", "ops-nn", "ops-cv", "ops-transformer"]:
            with self.subTest(repo=repo):
                root = self.fixture(repo)
                old = root / "examples/test_aclnn_sample.cpp"
                old.rename(old.with_name("test_aclnn_public_api_v2.cpp"))
                self.assertEqual(self.check(root, repo), 0)

    def test_nn_a2_example_directory_is_collected(self):
        root = self.fixture("ops-nn", socs=("ascend910b",))
        old = root / "examples/test_aclnn_sample.cpp"
        (old.parent / "arch22").mkdir()
        old.rename(old.parent / "arch22" / old.name)
        self.assertEqual(self.check(root, "ops-nn", "ascend910b"), 0)

    def test_a5_only_example_cannot_satisfy_other_soc(self):
        for repo in ["ops-nn", "ops-cv", "ops-transformer"]:
            with self.subTest(repo=repo):
                root = self.fixture(repo)
                old = root / "examples/test_aclnn_sample.cpp"
                (old.parent / "arch35").mkdir()
                old.rename(old.parent / "arch35" / old.name)
                self.assertEqual(self.check(root, repo, "ascend950"), 0)
                self.assertEqual(self.check(root, repo, "ascend910b"), 1)

    def test_cli_requires_soc_and_checks_source_registration(self):
        root = self.fixture(
            "ops-nn", "gn_reduce_v2", "GNReduceV2", socs=("ascend910b",)
        )
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        for command in [
            [str(scripts / "post_migration_check.sh"), str(root)],
            [
                sys.executable,
                str(scripts / "migration_contract.py"),
                "check",
                str(root),
            ],
        ]:
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        command = [
            str(scripts / "post_migration_check.sh"),
            str(root),
            "ascend910b",
            "ops-nn",
        ]
        for optype, expected in [("GNReduceV2", 0), ("GnReduceV2", 1)]:
            result = subprocess.run(
                command + [optype], capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_cli_profile_stdout_is_json(self):
        script = Path(__file__).resolve().parents[1] / "scripts/migration_contract.py"
        result = subprocess.run(
            [sys.executable, str(script), "profile", "ops-math"],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), profile("ops-math"))
        self.assertEqual(
            result.stdout,
            json.dumps(profile("ops-math"), ensure_ascii=False, indent=2) + "\n",
        )
        self.assertEqual(result.stderr, "")

    def test_cli_check_preserves_diagnostics_and_failure_exit(self):
        root = self.fixture("ops-nn")
        script = Path(__file__).resolve().parents[1] / "scripts/post_migration_check.sh"
        command = [str(script), str(root), "ascend950", "ops-nn", "Sample"]
        for remove_proto in [False, True]:
            if remove_proto:
                (root / "op_graph/sample_proto.h").unlink()
            result = subprocess.run(
                command, capture_output=True, text=True, check=False, shell=False
            )
            self.assertEqual(result.returncode, int(remove_proto), result.stdout)
            self.assertIn("== ops-nn:", result.stdout)
            self.assertIn("FAIL=", result.stdout)
            self.assertEqual("[FAIL]" in result.stdout, remove_proto)
            self.assertEqual(result.stderr, "")

    def test_math_tiling_suffix_is_not_valid_for_other_repos(self):
        for repo in ["ops-math", "ops-nn", "ops-cv", "ops-transformer"]:
            with self.subTest(repo=repo):
                root = self.fixture(repo)
                old = root / "tests/ut/op_host/arch35/test_sample_tiling.cpp"
                old.rename(old.with_name("test_sample_tiling_arch35.cpp"))
                self.assertEqual(self.check(root, repo), 0 if repo == "ops-math" else 1)

    def test_existing_entry_must_resolve_to_staged_operator_directory(self):
        for repo in ["ops-math", "ops-nn", "ops-cv", "ops-transformer"]:
            with self.subTest(repo=repo):
                root = self.fixture(repo)
                entry = root / "op_kernel/sample_apt.cpp"
                entry.rename(entry.with_name("sample_arch35.cpp"))
                definition = root / "op_host/sample_def.cpp"
                definition.write_text(
                    definition.read_text().replace("sample_apt", "sample_arch35")
                )
                self.assertEqual(self.check(root, repo), 1)

    def test_nested_entry_is_not_a_root_kernel_entry(self):
        root = self.fixture("ops-cv")
        entry = root / "op_kernel/sample_apt.cpp"
        (entry.parent / "arch35").mkdir()
        entry.rename(entry.parent / "arch35" / entry.name)
        self.assertEqual(self.check(root, "ops-cv"), 1)

    def test_missing_explicit_traversal_fails(self):
        for repo in ["ops-cv", "ops-transformer"]:
            root = self.fixture(repo)
            (root / "tests/CMakeLists.txt").unlink()
            self.assertEqual(self.check(root, repo), 1)

    def test_explicit_ut_chain_checks_each_required_child(self):
        edges = [
            ("CMakeLists.txt", "tests"),
            ("tests/CMakeLists.txt", "ut"),
            ("tests/ut/CMakeLists.txt", "op_host"),
            ("tests/ut/CMakeLists.txt", "op_api"),
        ]
        for repo, (name, child) in itertools.product(
            ["ops-cv", "ops-transformer"], edges
        ):
            with self.subTest(repo=repo, edge=(name, child)):
                root = self.fixture(repo)
                path = root / name
                path.write_text(
                    path.read_text().replace(
                        f"add_subdirectory({child})", "add_subdirectory(unrelated)"
                    )
                )
                self.assertEqual(self.check(root, repo), 1)

    def test_explicit_ut_chain_accepts_equivalent_literal_paths(self):
        for repo, prefix in itertools.product(
            ["ops-cv", "ops-transformer"],
            ["./", "${CMAKE_CURRENT_SOURCE_DIR}/", "${CMAKE_CURRENT_LIST_DIR}/"],
        ):
            with self.subTest(repo=repo, prefix=prefix):
                root = self.fixture(repo)
                for path in [
                    root / "CMakeLists.txt",
                    *root.glob("tests/**/CMakeLists.txt"),
                ]:
                    content = path.read_text()
                    for child in ["tests", "ut", "op_host", "op_api"]:
                        content = content.replace(
                            f"add_subdirectory({child})",
                            f'if(ENABLE_TEST)\nadd_subdirectory("{prefix}{child}")\nendif()',
                        )
                    path.write_text(content)
                self.assertEqual(self.check(root, repo), 0)

    def test_generated_aclnn_does_not_require_op_api_ut_edge(self):
        for repo in ["ops-cv", "ops-transformer"]:
            with self.subTest(repo=repo):
                root = self.fixture(repo, manual_api=False)
                path = root / "tests/ut/CMakeLists.txt"
                path.write_text(
                    path.read_text().replace("add_subdirectory(op_api)", "")
                )
                self.assertEqual(self.check(root, repo), 0)

    def test_block_commented_host_ut_registration_is_rejected(self):
        for repo, equals in itertools.product(
            ["ops-cv", "ops-transformer"], ["", "=", "=="]
        ):
            with self.subTest(repo=repo, delimiter=equals):
                root = self.fixture(repo)
                path = root / "tests/ut/op_host/CMakeLists.txt"
                self.write(path, f"#[{equals}[\n{path.read_text()}\n]{equals}]")
                self.assertEqual(self.check(root, repo), 1)

    def test_block_commented_op_api_edge_is_rejected(self):
        for repo in ["ops-cv", "ops-transformer"]:
            with self.subTest(repo=repo):
                root = self.fixture(repo)
                self.write(
                    root / "tests/ut/CMakeLists.txt",
                    "add_subdirectory(op_host)\n#[=[\nadd_subdirectory(op_api)\n]=]",
                )
                self.assertEqual(self.check(root, repo), 1)

    def test_cmake_comments_preserve_quoted_hash_and_delimiter_text(self):
        text = (
            'set(NOTE "# not a comment; #[=[ not a bracket comment")\n'
            "#[=[\nadd_subdirectory(ignored)\n]] is not the closing delimiter\n]=]\n"
            'add_subdirectory("path#name") # add_subdirectory(ignored)\n'
        )
        self.assertEqual(cmake_calls(text, "add_subdirectory"), ['"path#name"'])

    def test_unterminated_cmake_bracket_comment_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unterminated.*comment"):
            cmake_calls("#[==[\nadd_subdirectory(op_api)\n]=]", "add_subdirectory")

    def test_cli_rejects_unterminated_cmake_comment(self):
        root = self.fixture("ops-cv")
        path = root / "tests/ut/op_api/CMakeLists.txt"
        path.write_text(path.read_text() + "\n#[=[\nadd_modules_ut_sources(fake)\n")
        script = Path(__file__).resolve().parents[1] / "scripts/migration_contract.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "check",
                str(root),
                "--soc",
                "ascend950",
                "--repo",
                "ops-cv",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("unterminated", result.stdout + result.stderr)

    def test_license_headers_cover_source_and_cmake_extensions(self):
        for suffix in [
            ".c",
            ".cc",
            ".cpp",
            ".cxx",
            ".h",
            ".hh",
            ".hpp",
            ".hxx",
            ".asc",
            ".py",
            ".cmake",
        ]:
            with self.subTest(suffix=suffix):
                root = self.fixture("ops-math")
                path = root / "op_host" / ("helper" + suffix)
                path.write_text("\n")
                self.assertEqual(self.check(root, "ops-math"), 1)
                self.write(path)
                self.assertEqual(self.check(root, "ops-math"), 0)
        (root / "binary.bin").write_bytes(b"not a source file")
        self.assertEqual(self.check(root, "ops-math"), 0)

    @unittest.skipUnless(
        shutil.which("find"), "requires find for repository discovery rule"
    )
    def test_math_example_discovery_matches_recursive_find(self):
        root = self.fixture("ops-math")
        old = root / "examples/test_aclnn_sample.cpp"
        old.unlink()
        for directory in ["arch35", "nested/with space"]:
            self.write(root / "examples" / directory / "test_aclnn_public_api.cpp")
        result = subprocess.run(
            [
                "find",
                str(root / "examples"),
                "-type",
                "f",
                "-path",
                "*/examples/*",
                "-name",
                "test_aclnn_*.cpp",
                "-print0",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        expected = {Path(item) for item in result.stdout.split("\0") if item}
        self.assertEqual(len(expected), 2)
        for soc in ["ascend910b", "ascend950"]:
            self.assertEqual(
                set(example_files(root, soc, profile("ops-math"), "aclnn")), expected
            )
            self.assertEqual(self.check(root, "ops-math", soc), 0)

    def test_nn_310p_ut_directory_differs_from_example_directory(self):
        root = self.fixture("ops-nn", socs=("ascend310p",))
        host = root / "tests/ut/op_host"
        (host / "arch20").rename(host / "arch22")
        example = root / "examples/test_aclnn_sample.cpp"
        (example.parent / "arch20").mkdir()
        example.rename(example.parent / "arch20" / example.name)
        self.assertEqual(self.check(root, "ops-nn", "ascend310p"), 0)
        (host / "arch22").rename(host / "arch20")
        self.assertEqual(self.check(root, "ops-nn", "ascend310p"), 1)

    def test_zero_tiling_tests_fails(self):
        root = self.fixture("ops-nn")
        (root / "tests/ut/op_host/arch35/test_sample_tiling.cpp").unlink()
        self.assertEqual(self.check(root, "ops-nn"), 1)

    def test_misaligned_soc_lists_fail(self):
        root = self.fixture("ops-math")
        path = root / "CMakeLists.txt"
        path.write_text(path.read_text().replace('"arch22" "arch35"', '"arch35"'))
        self.assertEqual(self.check(root, "ops-math"), 1)

    def test_uncollected_wrong_architecture_fails(self):
        root = self.fixture("ops-math")
        old = root / "tests/ut/op_host/arch35/test_sample_tiling.cpp"
        other = old.parent.parent / "arch99/test_sample_tiling.cpp"
        other.parent.mkdir()
        old.rename(other)
        self.assertEqual(self.check(root, "ops-math"), 1)

    def test_transformer_op_api_must_be_wired(self):
        root = self.fixture("ops-transformer")
        path = root / "op_host/CMakeLists.txt"
        path.write_text(
            path.read_text().replace("OP_API_INDEPENDENT ON", "OP_API_INDEPENDENT OFF")
        )
        self.assertEqual(self.check(root, "ops-transformer"), 1)

    def test_transformer_missing_compiled_directory_registration_fails(self):
        root = self.fixture("ops-transformer")
        self.assertEqual(root.parent.name, "attention")
        path = root / "op_host/CMakeLists.txt"
        path.write_text(path.read_text().replace("add_op_to_compiled_list()", ""))
        self.assertEqual(self.check(root, "ops-transformer"), 1)

    def test_transformer_glob_collected_class_needs_no_extra_registration(self):
        root = self.fixture("ops-transformer")
        root.parent.rename(root.parent.with_name("mc2"))
        root = root.parent.with_name("mc2") / root.name
        path = root / "op_host/CMakeLists.txt"
        path.write_text(path.read_text().replace("add_op_to_compiled_list()", ""))
        self.assertEqual(self.check(root, "ops-transformer"), 0)

    def test_missing_tiling_implementation_fails(self):
        root = self.fixture("ops-transformer")
        (root / "op_host/arch35/sample_tiling.cpp").unlink()
        self.assertEqual(self.check(root, "ops-transformer"), 1)

    def test_transformer_missing_open_project_registration_fails(self):
        root = self.fixture("ops-transformer")
        path = root / "op_host/CMakeLists.txt"
        path.write_text(
            path.read_text().replace(
                "target_sources(op_host_aclnnExc PRIVATE sample_def.cpp)", ""
            )
        )
        self.assertEqual(self.check(root, "ops-transformer"), 1)

    def test_ops_rand_is_not_supported(self):
        with self.assertRaises(ValueError):
            profile("ops-rand")

    def decision(self, **changes):
        result = {
            "decision": "RUN",
            "target_repo": "ops-cv",
            "op_class": "image",
            "reason_code": "category_match",
            "reason": "image resampling",
            "evidence": {"category": "Interpolation", "routing_rule": "resampling"},
            "source_package": "/work/op",
        }
        result.update(changes)
        return result

    def test_supported_decisions(self):
        for repo in ["ops-math", "ops-nn", "ops-cv", "ops-transformer"]:
            validate_decision(
                self.decision(target_repo=repo, op_class=profile(repo)["classes"][0])
            )

    def test_unsupported_explicit_target_retained(self):
        validate_decision(
            self.decision(
                decision="SKIP",
                target_repo="ops-rand",
                op_class=None,
                reason_code="unsupported_target",
            )
        )
        with self.assertRaises(ValueError):
            validate_decision(
                self.decision(
                    decision="SKIP",
                    target_repo=None,
                    op_class=None,
                    reason_code="unsupported_target",
                )
            )

    def test_invalid_decisions_fail(self):
        for changes in [
            {"target_repo": "ops-rand"},
            {"op_class": "objetect"},
            {"target_repo": "ops-nn", "op_class": "image"},
            {"source_package": "../op"},
            {"evidence": {}},
            {"decision": "RUN", "reason_code": "user_opt_out"},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_decision(self.decision(**changes))


if __name__ == "__main__":
    unittest.main()
