# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "changelog.py"
SPEC = importlib.util.spec_from_file_location("changelog_tool", SCRIPT)
changelog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = changelog
SPEC.loader.exec_module(changelog)


class ChangelogFragmentTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo = Path(self.tempdir.name)
        self.fragments = self.repo / "docs/changelog.d"
        self.fragments.mkdir(parents=True)
        (self.fragments / "README.md").write_text("fragments\n", encoding="utf-8")
        self.old_text = (
            "## 🔥 更新日志\n\n"
            "> **记录规则**：单条不超过 200 字。\n\n"
            "### 【2026-11-01】\n"
            "#### 变更 Changes\n"
            "- 【测试】既有条目。\n"
        )
        (self.repo / "CHANGELOG.md").write_text(self.old_text, encoding="utf-8")

    def tearDown(self):
        self.tempdir.cleanup()

    def fragment(self, name, text="- 【测试】示例条目。"):
        path = self.fragments / name
        path.write_text(f"{text}\n", encoding="utf-8")
        return path

    def test_aggregate_groups_entries_and_clears_fragments(self):
        self.fragment("124.added.md", "- 【测试】新增能力。")
        self.fragment("123.fixed.md", "- 【测试】修复问题。")
        changed = changelog.aggregate(self.repo, "2026-11-03")
        self.assertTrue(changed)
        text = (self.repo / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertEqual(text.count(changelog.GENERATED_HEADER), 1)
        self.assertIn("#### 新特性 New Features\n- 【测试】新增能力。", text)
        self.assertIn("#### 缺陷修复 Bug Fixes\n- 【测试】修复问题。", text)
        self.assertLess(text.index("2026-11-03"), text.index("2026-11-01"))
        self.assertEqual(list(self.fragments.glob("[0-9]*.md")), [])

    def test_git_diff_disables_rename_detection(self):
        with patch.object(
            changelog.subprocess, "run", return_value=Mock(stdout="")
        ) as run:
            changelog.changed_files(self.repo, "base", "head")
        command = run.call_args[0][0]
        self.assertIn("--no-renames", command)

    def test_invalid_filename_is_rejected(self):
        self.fragment("topic.fixed.md")
        with self.assertRaisesRegex(ValueError, "文件名"):
            changelog.load_fragments(self.repo)

    def test_multiple_entries_in_one_fragment_are_rejected(self):
        self.fragment("123.fixed.md", "- 第一条。\n- 第二条。")
        with self.assertRaisesRegex(ValueError, "只能包含一条"):
            changelog.load_fragments(self.repo)

    def test_unknown_fragment_kind_is_rejected(self):
        self.fragment("123.feature.md")
        with self.assertRaisesRegex(ValueError, "文件名"):
            changelog.load_fragments(self.repo)

    def test_fragment_body_over_200_chars_is_rejected(self):
        self.fragment("123.fixed.md", "- " + "甲" * 201)
        with self.assertRaisesRegex(ValueError, "200"):
            changelog.load_fragments(self.repo)

    def test_fragment_body_at_200_chars_is_accepted(self):
        self.fragment("123.fixed.md", "- " + "甲" * 200)
        fragments = changelog.load_fragments(self.repo)
        self.assertEqual(len(fragments), 1)

    def test_empty_fragment_is_rejected(self):
        self.fragment("123.fixed.md", "   ")
        with self.assertRaises(ValueError):
            changelog.load_fragments(self.repo)

    def test_pr_number_must_match_current_pr(self):
        self.fragment("123.fixed.md")
        with self.assertRaisesRegex(ValueError, "不一致"):
            changelog.load_fragments(self.repo, expected_pr=124)

    def test_direct_changelog_edit_is_rejected(self):
        with patch.object(changelog, "changed_files", return_value=["CHANGELOG.md"]):
            with self.assertRaisesRegex(ValueError, "禁止直接修改"):
                changelog.check(self.repo, "base", "head", 123)

    def test_valid_fragment_without_changelog_edit_passes(self):
        self.fragment("123.fixed.md")
        with patch.object(
            changelog, "changed_files", return_value=["docs/changelog.d/123.fixed.md"]
        ):
            changelog.check(self.repo, "base", "head", 123)

    def test_existing_other_pr_fragment_does_not_block_current_pr(self):
        self.fragment("999.fixed.md", "- 【测试】存量条目。")
        self.fragment("123.fixed.md")
        with patch.object(
            changelog, "changed_files", return_value=["docs/changelog.d/123.fixed.md"]
        ):
            changelog.check(self.repo, "base", "head", 123)

    def test_changing_other_pr_fragment_is_rejected(self):
        self.fragment("999.fixed.md")
        with patch.object(
            changelog, "changed_files", return_value=["docs/changelog.d/999.fixed.md"]
        ):
            with self.assertRaisesRegex(ValueError, "不一致"):
                changelog.check(self.repo, "base", "head", 123)

    def test_release_requires_deleted_fragment(self):
        with patch.object(
            changelog, "changed_file_statuses", return_value=[("M", "CHANGELOG.md")]
        ):
            with self.assertRaisesRegex(ValueError, "至少要删除"):
                changelog.verify_release(self.repo, "base", "head")

    def test_release_rejects_fragment_modification(self):
        changes = [("M", "CHANGELOG.md"), ("M", "docs/changelog.d/123.fixed.md")]
        with patch.object(changelog, "changed_file_statuses", return_value=changes):
            with self.assertRaisesRegex(ValueError, "只能删除"):
                changelog.verify_release(self.repo, "base", "head")

    def test_release_uses_head_tree_instead_of_merge_workspace(self):
        self.fragment("999.fixed.md", "- 【测试】目标分支并发条目。")
        changes = [("M", "CHANGELOG.md"), ("D", "docs/changelog.d/123.fixed.md")]
        release_text = f"{changelog.GENERATED_HEADER}\n\n" + self.old_text.replace(
            "### 【2026-11-01】",
            "### 【2026-11-03】\n\n#### 缺陷修复 Bug Fixes\n"
            "- 【测试】聚合条目。\n\n### 【2026-11-01】",
        )

        def restore(_repo, commit, destination):
            destination.mkdir(parents=True)
            text = self.old_text if commit == "base" else release_text
            (destination / "CHANGELOG.md").write_text(text, encoding="utf-8")
            if commit == "base":
                fragment = destination / "docs/changelog.d/123.fixed.md"
                fragment.parent.mkdir(parents=True)
                fragment.write_text("- 【测试】聚合条目。\n", encoding="utf-8")

        with patch.object(changelog, "changed_file_statuses", return_value=changes):
            with patch.object(changelog, "restore_tree", side_effect=restore) as mocked:
                changelog.verify_release(self.repo, "base", "head")
        self.assertEqual([call.args[1] for call in mocked.call_args_list], ["base", "head"])

    def test_duplicate_date_does_not_delete_fragment(self):
        self.fragment("123.fixed.md")
        (self.repo / "CHANGELOG.md").write_text(
            self.old_text.replace("2026-11-01", "2026-11-03"), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "拒绝重复聚合"):
            changelog.aggregate(self.repo, "2026-11-03")
        self.assertTrue((self.fragments / "123.fixed.md").exists())
        self.assertNotIn(
            changelog.GENERATED_HEADER,
            (self.repo / "CHANGELOG.md").read_text(encoding="utf-8"),
        )

    def test_no_fragments_is_noop(self):
        changed = changelog.aggregate(self.repo, "2026-11-03")
        self.assertFalse(changed)
        self.assertEqual(
            (self.repo / "CHANGELOG.md").read_text(encoding="utf-8"), self.old_text
        )

    def test_header_is_not_duplicated_on_next_release(self):
        self.fragment("123.fixed.md")
        changelog.aggregate(self.repo, "2026-11-03")
        self.fragment("124.docs.md", "- 【测试】补充文档。")
        changelog.aggregate(self.repo, "2026-11-04")
        text = (self.repo / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertEqual(text.count(changelog.GENERATED_HEADER), 1)
        self.assertIn("#### 文档 Documentation", text)


if __name__ == "__main__":
    unittest.main()
