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
"""pr_review_filter.py 的单元测试。

执行：python3 -m unittest scripts.test_pr_review_filter（或在本目录内 python3 -m unittest test_pr_review_filter）
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pr_review_filter as prf  # noqa: E402


def issue(file, line, item="punctuation-check", level="一般"):
    """模拟 aggregate.js 输出契约：file_name 为裸文件名，location 为 绝对路径:行号。"""
    name = file.rsplit("/", 1)[-1]
    loc = f"{file}:{line}" if line else file
    return {
        "rule_group": "通用规则",
        "rule_item": item,
        "rule_type": "",
        "rule_name": "",
        "snippet": "",
        "description": "d",
        "rule_level": level,
        "file_name": name,
        "line_number": line,
        "location": loc,
        "suggestion": "s",
    }


DIFF = """diff --git a/docs/README.md b/docs/README.md
index 1111111..2222222 100644
--- a/docs/README.md
+++ b/docs/README.md
@@ -10,5 +10,6 @@
 line10
 line11
+added12
 line13
 line14
+line15
diff --git "a/docs/\\346\\226\\260\\345\\242\\236\\346\\226\\207\\346\\241\\243.md" "b/docs/\\346\\226\\260\\345\\242\\236\\346\\226\\207\\346\\241\\243.md"
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/"docs/\\346\\226\\260\\345\\242\\236\\346\\226\\207\\346\\241\\243.md"
@@ -0,0 +1,4 @@
+l1
+l2
+l3
+l4
diff --git a/docs/gone.md b/docs/gone.md
deleted file mode 100644
index 4444444..0000000
--- a/docs/gone.md
+++ /dev/null
@@ -1,3 +0,0 @@
-a
-b
-c
diff --git a/docs/no-count.md b/docs/no-count.md
index 5555555..6666666 100644
--- a/docs/no-count.md
+++ b/docs/no-count.md
@@ -7 +7 @@
-old
+new
"""


class UnquotePathTest(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(prf.unquote_path("docs/README.md"), "docs/README.md")

    def test_quoted_utf8_bytes(self):
        self.assertEqual(
            prf.unquote_path(
                '"docs/\\346\\226\\260\\345\\242\\236\\346\\226\\207\\346\\241\\243.md"'
            ),
            "docs/新增文档.md",
        )

    def test_quoted_ascii_escapes(self):
        self.assertEqual(prf.unquote_path('"a\\"b\\\\c"'), 'a"b\\c')


class ParseDiffTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fd, cls.path = tempfile.mkstemp(suffix=".diff")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(DIFF)
        cls.ranges = prf.parse_diff(cls.path)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.path)

    def test_files(self):
        self.assertEqual(
            set(self.ranges), {"docs/README.md", "docs/新增文档.md", "docs/no-count.md"}
        )

    def test_hunk_ranges(self):
        self.assertEqual(self.ranges["docs/README.md"], [[10, 15]])
        self.assertEqual(self.ranges["docs/新增文档.md"], [[1, 4]])
        self.assertEqual(self.ranges["docs/no-count.md"], [[7, 7]])

    def test_deleted_file_skipped(self):
        self.assertNotIn("docs/gone.md", self.ranges)


class FileMatchesTest(unittest.TestCase):
    def test_exact_and_suffix(self):
        self.assertTrue(prf.file_matches("docs/README.md", "docs/README.md"))
        self.assertTrue(
            prf.file_matches("/repo/worktree/docs/README.md", "docs/README.md")
        )

    def test_path_normalize(self):
        self.assertTrue(
            prf.file_matches(".\\repo\\docs\\README.md", "./docs/README.md")
        )

    def test_mismatch(self):
        self.assertFalse(prf.file_matches("docs/other/README.md", "docs/README.md"))
        self.assertFalse(prf.file_matches("docs", "docs/README.md"))


class IssuePathTest(unittest.TestCase):
    def test_location_preferred(self):
        i = {"file_name": "README.md", "location": "/w/repo/docs/README.md:12"}
        self.assertEqual(prf.issue_path(i), "/w/repo/docs/README.md")

    def test_location_without_line(self):
        i = {"file_name": "README.md", "location": "/w/repo/docs/README.md"}
        self.assertEqual(prf.issue_path(i), "/w/repo/docs/README.md")

    def test_fallback_file_name(self):
        self.assertEqual(
            prf.issue_path({"file_name": "README.md", "location": ""}), "README.md"
        )
        self.assertEqual(prf.issue_path({}), "")

    def test_location_line_only_edge(self):
        self.assertEqual(
            prf.issue_path({"file_name": "a.md", "location": "12"}), "a.md"
        )


class IssueKeptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ranges = {"docs/README.md": [[10, 15]]}

    def test_in_hunk(self):
        ok, path = prf.issue_kept(issue("/w/docs/README.md", 12), self.ranges, 3)
        self.assertTrue(ok)
        self.assertEqual(path, "docs/README.md")

    def test_context_edge_inclusive(self):
        self.assertTrue(
            prf.issue_kept(issue("/w/docs/README.md", 18), self.ranges, 3)[0]
        )
        self.assertTrue(
            prf.issue_kept(issue("/w/docs/README.md", 7), self.ranges, 3)[0]
        )

    def test_beyond_context_dropped(self):
        self.assertFalse(
            prf.issue_kept(issue("/w/docs/README.md", 19), self.ranges, 3)[0]
        )
        self.assertFalse(
            prf.issue_kept(issue("/w/docs/README.md", 6), self.ranges, 3)[0]
        )

    def test_file_level_kept_when_changed(self):
        self.assertTrue(
            prf.issue_kept(issue("/w/docs/README.md", ""), self.ranges, 3)[0]
        )

    def test_unknown_file_dropped(self):
        self.assertFalse(
            prf.issue_kept(issue("/w/docs/other.md", 12), self.ranges, 3)[0]
        )

    def test_context_zero_strict(self):
        self.assertTrue(
            prf.issue_kept(issue("/w/docs/README.md", 15), self.ranges, 0)[0]
        )
        self.assertFalse(
            prf.issue_kept(issue("/w/docs/README.md", 16), self.ranges, 0)[0]
        )


class RenderBodyTest(unittest.TestCase):
    def test_inline_no_location(self):
        body = prf.render_body(
            issue("docs/README.md", "12", item="标点符号", level="严重"),
            "docs/README.md",
            12,
            True,
        )
        self.assertEqual(
            body, "[严重] [通用规则] [标点符号]\n\n**问题**：d\n\n**建议**：s"
        )
        self.assertNotIn("README.md", body)

    def test_with_rule_name(self):
        i = issue("docs/README.md", "12")
        i["rule_name"] = "中英文符号混用"
        body = prf.render_body(i, "docs/README.md", 12, True)
        self.assertTrue(
            body.startswith("[一般] [通用规则] [punctuation-check] [中英文符号混用]\n")
        )

    def test_rule_name_empty_omitted(self):
        i = issue("docs/README.md", "12")
        i["rule_name"] = "  "
        self.assertTrue(
            prf.render_body(i, "docs/README.md", 12, True).startswith(
                "[一般] [通用规则] [punctuation-check]\n"
            )
        )

    def test_group_empty_omitted(self):
        i = issue("docs/README.md", "12")
        i["rule_group"] = ""
        i["rule_name"] = "中英文符号混用"
        self.assertTrue(
            prf.render_body(i, "docs/README.md", 12, True).startswith(
                "[一般] [punctuation-check] [中英文符号混用]\n"
            )
        )

    def test_plain_keeps_location(self):
        body = prf.render_body(
            issue("docs/README.md", "12"), "docs/README.md", 12, False
        )
        self.assertTrue(
            body.startswith("[一般] [通用规则] [punctuation-check] docs/README.md:12\n")
        )

    def test_file_level_no_line(self):
        body = prf.render_body(
            issue("docs/README.md", "", item="filename-check"),
            "docs/README.md",
            None,
            False,
        )
        self.assertEqual(
            body.split("\n")[0], "[一般] [通用规则] [filename-check] docs/README.md"
        )

    def test_multiline_collapsed(self):
        i = issue("docs/README.md", "3")
        i["description"] = "第一行\n第二行"
        body = prf.render_body(i, "docs/README.md", 3, True)
        self.assertIn("第一行 第二行", body)
        self.assertNotIn("第一行\n第二行", body)


class ResolveTokenTest(unittest.TestCase):
    def test_config_file(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump({"token": "t-abc", "baseUrl": "https://example.com/api"}, fh)
            path = fh.name
        try:
            args = prf.build_parser().parse_args(
                [
                    "submit",
                    "--comments",
                    "x.json",
                    "--repo",
                    "o/r",
                    "--pr",
                    "1",
                    "--config",
                    path,
                ]
            )
            token, base = prf.resolve_token(args)
            self.assertEqual(token, "t-abc")
            self.assertEqual(base, "https://example.com/api")
        finally:
            os.remove(path)

    def test_config_default_base_url(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump({"token": "t-abc"}, fh)
            path = fh.name
        try:
            args = prf.build_parser().parse_args(
                [
                    "submit",
                    "--comments",
                    "x.json",
                    "--repo",
                    "o/r",
                    "--pr",
                    "1",
                    "--config",
                    path,
                ]
            )
            token, base = prf.resolve_token(args)
            self.assertEqual(token, "t-abc")
            self.assertEqual(base, prf.DEFAULT_BASE_URL)
        finally:
            os.remove(path)

    def test_env_fallback(self):
        args = prf.build_parser().parse_args(
            [
                "submit",
                "--comments",
                "x.json",
                "--repo",
                "o/r",
                "--pr",
                "1",
                "--config",
                "/nonexistent/config.json",
            ]
        )
        old = os.environ.get("GITCODE_TOKEN")
        os.environ["GITCODE_TOKEN"] = "t-env"
        try:
            token, base = prf.resolve_token(args)
            self.assertEqual(token, "t-env")
            self.assertEqual(base, prf.DEFAULT_BASE_URL)
        finally:
            if old is None:
                os.environ.pop("GITCODE_TOKEN", None)
            else:
                os.environ["GITCODE_TOKEN"] = old

    def test_missing_token_exits(self):
        args = prf.build_parser().parse_args(
            [
                "submit",
                "--comments",
                "x.json",
                "--repo",
                "o/r",
                "--pr",
                "1",
                "--config",
                "/nonexistent/config.json",
            ]
        )
        old = os.environ.pop("GITCODE_TOKEN", None)
        try:
            with self.assertRaises(SystemExit) as ctx:
                prf.resolve_token(args)
            self.assertEqual(ctx.exception.code, 2)
        finally:
            if old is not None:
                os.environ["GITCODE_TOKEN"] = old


class DedupTest(unittest.TestCase):
    def test_dedup_issues_full_fields(self):
        a = issue("/w/docs/README.md", "12")
        b = issue("/w/docs/README.md", "12")
        c = issue("/w/docs/README.md", "12", item="aicheck")
        kept, dup = prf.dedup_issues([a, b, c])
        self.assertEqual(dup, 1)
        self.assertEqual(len(kept), 2)

    def test_dedup_comments_same_body(self):
        c1 = {"file": "a.md", "line": 3, "body": "x"}
        c2 = {"file": "a.md", "line": 3, "body": "x"}
        c3 = {"file": "a.md", "line": 5, "body": "x"}
        self.assertEqual(len(prf.dedup_comments([c1, c2, c3])), 2)

    def test_dedup_against_existing_file_line(self):
        existing = [{"path": "a.md", "position": {"new_line": 3}, "body": "旧评论"}]
        entries = [
            {"file": "a.md", "line": 3, "body": "新评论"},
            {"file": "a.md", "line": 9, "body": "另一条"},
        ]
        kept, skipped = prf.dedup_against_existing(entries, existing)
        self.assertEqual(len(kept), 1)
        self.assertEqual(skipped[0]["line"], 3)

    def test_dedup_against_existing_body(self):
        existing = [
            {"path": "old.md", "position": {"new_line": 1}, "body": "第 1 行  问题：x"}
        ]
        entries = [{"file": "a.md", "line": 9, "body": "第1行问题：x"}]
        kept, _ = prf.dedup_against_existing(entries, existing)
        self.assertEqual(len(kept), 0)

    def test_dedup_against_existing_none(self):
        entries = [{"file": "a.md", "line": 9, "body": "x"}]
        kept, skipped = prf.dedup_against_existing(entries, None)
        self.assertEqual((len(kept), len(skipped)), (1, 0))

    def test_dedup_against_existing_proximity_same_rule(self):
        existing = [
            {
                "path": "a.md",
                "position": {"new_line": 3},
                "body": "[一般] [通用规则] [punctuation-check] [中英文符号混用] 正文",
            }
        ]
        entries = [
            {
                "file": "a.md",
                "line": 6,
                "body": "新评论",
                "rule_item": "punctuation-check",
                "rule_name": "中英文符号混用",
            },
            {
                "file": "a.md",
                "line": 6,
                "body": "新评论",
                "rule_item": "codespell-check",
                "rule_name": "拼写错误",
            },
            {
                "file": "a.md",
                "line": 20,
                "body": "远处",
                "rule_item": "punctuation-check",
                "rule_name": "中英文符号混用",
            },
            {
                "file": "b.md",
                "line": 4,
                "body": "他文件",
                "rule_item": "punctuation-check",
                "rule_name": "中英文符号混用",
            },
        ]
        kept, skipped = prf.dedup_against_existing(entries, existing)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["body"], "新评论")
        self.assertIn("疑似同一问题", skipped[0]["skip_reason"])
        self.assertEqual(len(kept), 3)

    def test_entry_rule_tags_from_body(self):
        entry = {"body": "[严重] [通用规则] [punctuation-check] [中英文符号混用] 正文"}
        tags = prf.entry_rule_tags(entry)
        self.assertIn("punctuation-check", tags)
        self.assertIn("中英文符号混用", tags)
        self.assertNotIn(prf.GENERIC_RULE_GROUP, tags)

    def test_dedup_against_existing_proximity_disabled(self):
        existing = [
            {"path": "a.md", "position": {"new_line": 3}, "body": "x punctuation-check"}
        ]
        entries = [
            {"file": "a.md", "line": 5, "body": "y", "rule_item": "punctuation-check"}
        ]
        kept, _ = prf.dedup_against_existing(entries, existing, proximity=0)
        self.assertEqual(len(kept), 1)

    def test_load_record_missing(self):
        self.assertEqual(prf.load_record("/nonexistent/path/record.json"), [])


class FilterEndToEndTest(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            diff_path = os.path.join(tmp, "p.diff")
            result_path = os.path.join(tmp, "r.json")
            out_path = os.path.join(tmp, "o.json")
            with open(diff_path, "w", encoding="utf-8") as fh:
                fh.write(DIFF)
            issues = [
                issue("/w/docs/README.md", "12"),
                issue("/w/docs/README.md", "12"),
                issue("/w/docs/README.md", "17", item="aicheck", level="提示"),
                issue("/w/docs/README.md", ""),
                issue(
                    "/w/docs/新增文档.md", "3", item="compliance-check", level="严重"
                ),
                issue("/w/docs/gone.md", "2"),
                issue("/w/docs/README.md", "25"),
                issue(
                    "/w/docs/other/zzz.md", "2", item="codespell-check", level="提示"
                ),
            ]
            with open(result_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"summary": {"mode": "full"}, "issues": issues},
                    fh,
                    ensure_ascii=False,
                )
            args = prf.build_parser().parse_args(
                [
                    "filter",
                    "--diff",
                    diff_path,
                    "--result",
                    result_path,
                    "--output",
                    out_path,
                    "--comments",
                    os.path.join(tmp, "c.json"),
                ]
            )
            prf.cmd_filter(args)
            with open(out_path, encoding="utf-8") as fh:
                out = json.load(fh)
            with open(os.path.join(tmp, "c.json"), encoding="utf-8") as fh:
                comments = json.load(fh)
        self.assertEqual(
            out["stats"],
            {
                "total": 8,
                "kept": 4,
                "dedup_removed": 1,
                "dropped_existing": 3,
                "files_in_diff": 3,
            },
        )
        kept_keys = [(i["file_name"], i["line_number"]) for i in out["issues"]]
        self.assertIn(("README.md", "12"), kept_keys)
        self.assertIn(("README.md", ""), kept_keys)
        self.assertIn(("新增文档.md", "3"), kept_keys)
        self.assertNotIn(("README.md", "25"), kept_keys)
        self.assertEqual(len(comments), 4)
        self.assertTrue(comments[0]["inline"])
        self.assertFalse(comments[2]["inline"])


if __name__ == "__main__":
    unittest.main()
