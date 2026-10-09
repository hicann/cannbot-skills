# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Offline rule regressions; fixtures are syntax probes, not HCCL implementations."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import review_template as rt

FIXTURES = Path(__file__).resolve().parent.parent / "assets/fixtures"


class ReviewTests(unittest.TestCase):
    def facts(self, name="good", op="AllReduce", source=None):
        path = FIXTURES / ("ins_temp_all_reduce_ring_" + name + ".cc")
        cls = "InsTempAllReduceRing" + name.title()
        cc = path.read_text() if source is None else source
        hh = path.with_suffix(".h").read_text()
        renamed = cls.replace("AllReduce", op)
        # filebase must match the renamed class so infer_op does not pick up a
        # stale "all_reduce" from the fixture filename when testing other ops
        synthetic_path = "ins_temp_" + op.lower() + "_ring_" + name + ".cc"
        return rt.Facts(
            "",
            synthetic_path,
            renamed,
            "InsAlgTemplateBase",
            rt.strip_comments(cc.replace(cls, renamed)),
            rt.strip_comments(hh.replace(cls, renamed)),
        )

    def ring_findings(self, facts):
        rv = rt.Review("fixture", "ring", "")
        # X01 requires actual target-repo registrations; do not fake their API.
        with patch.object(rt, "algo_gap"):
            rt.check_ring(facts, rv, set())
        return rv.findings

    def test_rotating_positive_and_negative(self):
        self.assertFalse(
            [f for f in self.ring_findings(self.facts()) if f.sev == rt.MAJOR]
        )
        self.assertIn("R02", {f.rule for f in self.ring_findings(self.facts("bad"))})
        source = (FIXTURES / "ins_temp_all_reduce_ring_good.cc").read_text()
        mutated = source.replace("myRankIdx_ - step", "myRankIdx_")
        self.assertIn(
            "R03", {f.rule for f in self.ring_findings(self.facts(source=mutated))}
        )

    def test_chain_broadcast_not_subject_to_rotating_rules(self):
        # Deliberately no step/chunk indices; the checker must request manual
        # endpoint/ordering review rather than claim that the chain is verified.
        source = """const u32 nextIdx = (myRankIdx_ + 1) % templateRankSize_;
        if (myRankIdx_ != rootIdx) { ReceiveComplete(); }
        if (nextIdx != rootIdx) { Forward(); }"""
        findings = self.ring_findings(self.facts(op="Broadcast", source=source))
        self.assertEqual({f.rule for f in findings}, {"R08"})
        self.assertEqual(findings[0].sev, rt.INFO)
        # A broken chain also remains unverified: regex does not prove reachability.
        broken = source.replace("ReceiveComplete();", "")
        self.assertIn(
            "R08",
            {
                f.rule
                for f in self.ring_findings(self.facts(op="Broadcast", source=broken))
            },
        )

    def test_unknown_shape_not_silently_passed(self):
        findings = self.ring_findings(self.facts(op="Custom"))
        self.assertIn("R08", {f.rule for f in findings})
        self.assertNotIn("R02", {f.rule for f in findings})

    def test_component_semantics_override_parent_operation(self):
        facts = self.facts(op="Broadcast")
        facts.op = "all_reduce"
        self.assertEqual(rt.ring_shape(facts), "broadcast")
        rv = rt.Review("fixture", "ring", "")
        facts.scratch_expr = "7"
        rt.check_common(facts, rv)
        self.assertNotIn("R05", {f.rule for f in rv.findings})

    def test_scatter_relay_requires_output_placement_review(self):
        copy = (
            "DataSlice src(tempAlgParams.buffInfo.hcclBuff.addr, "
            "tempAlgParams.buffInfo.hcclBuffBaseOff, size, count); "
            "DataSlice dst(tempAlgParams.buffInfo.outputPtr, "
            "tempAlgParams.buffInfo.outBuffBaseOff, size, count); "
            "return LocalCopy(thread, src, dst);"
        )
        facts = self.facts(op="Scatter", source=copy)
        rv = rt.Review("fixture", "ring", "")
        rt.check_common(facts, rv)
        self.assertIn("C16", {f.rule for f in rv.findings})
        self.assertIn("R09", {f.rule for f in self.ring_findings(facts)})
        guarded = self.facts(
            op="Scatter",
            source=copy + "\nif (tempAlgParams.buffInfo.outBuffType == "
            "BufferType::HCCL_BUFFER) return HCCL_SUCCESS;",
        )
        rv = rt.Review("fixture", "ring", "")
        rt.check_common(guarded, rv)
        self.assertNotIn("C16", {f.rule for f in rv.findings})
        self.assertIn("R09", {f.rule for f in self.ring_findings(guarded)})
        split_methods = (
            "HcclResult InsTempScatterRing::KeepChunk() {" + copy + "}\n"
            "bool InsTempScatterRing::CheckSupport() {"
            "return tempAlgParams.buffInfo.outBuffType == BufferType::OUTPUT; }"
        )
        self.assertTrue(
            rt.output_copy_without_guard(split_methods, "InsTempScatterRing")
        )

    def test_fenced_headings_keep_sync_body(self):
        for fence in ("```", "~~~~", "   ````"):
            text = (
                "## 6. 数据流\n" + fence + "text\n# 早退\n"
                "## 7. 伪标题\nsync main -> sub\nsync sub -> main\n"
                + fence
                + "\n### 子节\nbody\n## 7. 边界\noutside\n"
            )
            with self.subTest(fence=fence):
                section = rt.markdown_section(text, 6, "数据流")
                self.assertIn("sync sub -> main", section)
                self.assertIn("body", section)
                self.assertNotIn("outside", section)
                self.assertIsNone(rt.markdown_section(text, 7, "伪标题"))

    def test_shorter_or_different_fence_does_not_close(self):
        text = "## 6. 数据流\n````\n```\n~~~\n## 7. 伪标题\nsync main -> sub\n````\n## 7. 边界\n"
        self.assertIn("sync main -> sub", rt.markdown_section(text, 6, "数据流"))

    def test_s03_still_detects_missing_sync(self):
        facts = self.facts()
        spec = (
            "## 4. Buffer 布局\n| scratch 倍数 | 1 |\n"
            "## 5. 切片\n| RPT | N |\n## 6. 数据流\n"
            "```text\n# 早退\n" + "sync main -> sub\nsync sub -> main\n" * 2 + "```\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.md"
            for text, mismatch in (
                (spec, False),
                (spec.replace("sync sub -> main\n", "", 1), True),
            ):
                path.write_text(text)
                rv = rt.Review("fixture", "ring", "")
                rt.check_spec(facts, rv, str(path))
                self.assertEqual("S03" in {f.rule for f in rv.findings}, mismatch)

    def test_s03_single_thread_mentions_are_not_actions(self):
        facts = self.facts(source="HcclResult Run() { return HCCL_SUCCESS; }")
        spec = (
            "## 4. Buffer 布局\n| scratch 倍数 | 1 |\n"
            "## 5. 切片\n| RPT | 1 |\n## 6. 数据流\n"
            "单线程，没有 `sync main -> sub` 和 `sync sub -> main`。\n"
            "```text\n# 不需要 sync main -> sub\n"
            "// 不需要 sync sub -> main\n"
            "T0: LocalCopy IN[0,S] -> OUT[0,S] # sync main -> sub 是说明\n```\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.md"
            path.write_text(spec)
            rv = rt.Review("fixture", "ring", "")
            rt.check_spec(facts, rv, str(path))
            self.assertFalse({f.rule for f in rv.findings} & {"S00", "S03"})

    def test_sync_fences_comments_and_uncertain_syntax(self):
        for fence in ("```", "~~~~", "   ````"):
            section = (
                fence + "text\nif N > 1:\n  sync main -> sub # begin\n"
                "/* sync sub -> main */\n  sync sub -> main // end\n" + fence
            )
            self.assertEqual(rt.spec_sync_counts(section), (1, 1))
        for section in (
            "sync main -> sub",
            "```\nsync main -> sub",
            "```\nif N > 1: sync main -> sub\n```",
            "```\n```",
        ):
            with self.subTest(section=section), self.assertRaises(ValueError):
                rt.spec_sync_counts(section)


if __name__ == "__main__":
    unittest.main()
