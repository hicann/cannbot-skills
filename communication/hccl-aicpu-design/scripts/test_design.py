# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Offline regression tests for AICPU algorithm design and Dataflow Spec tools."""

import json
from pathlib import Path
import tempfile
import unittest

import check_layout
import check_spec

SKILL = Path(__file__).resolve().parent.parent


class LayoutTests(unittest.TestCase):
    def document(self, cases):
        return "```layout-check\n" + json.dumps({"cases": cases}) + "\n```\n"

    def case(self, formula, variables, expected, length=256, capacity=16384):
        return dict(
            formula=formula,
            vars=variables,
            expected=expected,
            length=length,
            capacity=capacity,
            element_size=4,
        )

    def test_ring_rank_term_mutation(self):
        # Independent oracle: server 2 owns complete blocks 8..11, each 1024 bytes.
        cases = [
            self.case(
                "b+r*IRS+i*ISS", dict(b=0, r=r, IRS=1024, i=2, ISS=4096), expected
            )
            for r, expected in enumerate([8192, 9216, 10240, 11264])
        ]
        self.assertEqual(check_layout.validate(self.document(cases)), [])
        for case in cases:
            case["formula"] = "b+r*IRS"
        self.assertEqual(len(check_layout.validate(self.document(cases))), 4)

    def test_asymmetric_and_zero_strides(self):
        cases = []
        # Sequence layout from the user-provided example: tight input, strided output.
        for r, (input_expected, output_expected) in enumerate(
            [(0, 2048), (256, 6144), (512, 10240)]
        ):
            cases.append(
                self.case(
                    "b+r*IRS+i*ISS", dict(b=0, r=r, IRS=256, i=2, ISS=0), input_expected
                )
            )
            cases.append(
                self.case(
                    "b+r*ORS+i*OSS",
                    dict(b=0, r=r, ORS=4096, i=2, OSS=1024),
                    output_expected,
                )
            )
        # Nonzero chunk base and tail; zero output stride is legitimate for ReduceScatter.
        cases.append(
            self.case("b+r*T+i*S", dict(b=64, r=0, T=0, i=2, S=1024), 2112, length=12)
        )
        cases.append(
            self.case("b+r*T+i*S", dict(b=64, r=0, T=0, i=2, S=0), 64, length=12)
        )
        self.assertEqual(check_layout.validate(self.document(cases)), [])

    def test_bad_offsets_and_unsafe_formula(self):
        for case in [
            self.case("b", dict(b=128), 128, capacity=256),
            self.case("b", dict(b=1), 1),
            self.case("b+1", dict(b=check_layout.U64_MAX), 0),
            self.case("b-1", dict(b=0), 0),
            self.case('__import__("os")', {}, 0),
        ]:
            self.assertTrue(check_layout.validate(self.document([case])))

    def test_reference_examples(self):
        doc = (SKILL / "assets/dataflow-spec.example.md").read_text()
        self.assertEqual(check_layout.validate(doc), [])

    def test_missing_layout_block(self):
        self.assertTrue(check_layout.validate("no examples"))


class SpecTests(unittest.TestCase):
    def test_fenced_headings_preserve_section_body(self):
        for fence in ("```", "~~~~", "   ````"):
            text = (
                "## 6. 数据流\n" + fence + "text\n# 早退\n"
                "## 7. 伪标题\nsync main -> sub\n"
                + fence
                + "\n### 子节\nsync sub -> main\n## 7. 边界\noutside\n"
            )
            with self.subTest(fence=fence):
                section = check_spec.section_body(text, 6, "数据流")
                self.assertIn("sync main -> sub", section)
                self.assertIn("sync sub -> main", section)
                self.assertNotIn("outside", section)
                self.assertIsNone(check_spec.section_body(text, 7, "伪标题"))

    def setUp(self):
        self.example = (SKILL / "assets/dataflow-spec.example.md").read_text()
        self.tmp = tempfile.TemporaryDirectory(prefix="hccl-spec-test-")
        self.addCleanup(self.tmp.cleanup)

    def check(self, text):
        path = Path(self.tmp.name) / "spec.md"
        path.write_text(text)
        return check_spec.check(str(path)).errors

    def test_example(self):
        self.assertEqual(self.check(self.example), 0)

    def test_invalid_operands(self):
        for old, new in [
            ("LocalCopy  IN[0,S]", "LocalCopy  INVALID"),
            ("SendRecvBatchWrite  IN[0,S]", "SendRecvBatchWrite  INVALID"),
            ("into OUT[0,S]", "into INVALID"),
        ]:
            with self.subTest(new=new):
                self.assertIn(old, self.example)
                self.assertGreater(self.check(self.example.replace(old, new)), 0)

    def test_repeat_comment_does_not_satisfy_loop(self):
        layered = self.example.replace(
            "| 所属层级 | 单层（只做 level0） |", "| 所属层级 | level0-intra |"
        )
        self.assertGreater(self.check(layered), 0)
        self.assertGreater(
            self.check(
                layered.replace(
                    "assert threadNum == N", "# RPT repeat for\nassert threadNum == N"
                )
            ),
            0,
        )
        self.assertEqual(
            self.check(
                layered.replace(
                    "assert threadNum == N",
                    "repeat for rpt in 0..RPT-1:\nassert threadNum == N",
                )
            ),
            0,
        )

    def test_decision_fields_required(self):
        text = "\n".join(
            line for line in self.example.splitlines() if "| 启用方式 |" not in line
        )
        self.assertGreater(self.check(text), 0)

    def test_routing_contract_variants(self):
        for value in ("不变", "不改变", "保持不变", "保持现有算法不变"):
            self.assertEqual(
                self.check(self.example.replace("`不变`", "`" + value + "`")), 0
            )
        changed = self.example.replace("`不变`", "`按批准范围改变`")
        self.assertGreater(self.check(changed), 0)
        fields = (
            "| 选路变更依据 | 上游已批准调整阈值 |\n"
            "| 允许变更范围 | AllReduce FP32 8rank 小数据 |\n"
            "| 选路验收矩阵 | cases.json，包含旧/新算法和边界 |\n"
        )
        changed = changed.replace("## 2.", fields + "\n## 2.")
        self.assertEqual(self.check(changed), 0)
        self.assertGreater(
            self.check(changed.replace("`自动选路`", "`标准配置显式选中`")), 0
        )
        self.assertGreater(self.check(self.example.replace("`不变`", "`可能不变`")), 0)

    def test_default_routing_cannot_be_waived_by_benchmark(self):
        text = self.example.replace("`不变`", "`改变`").replace(
            "无阈值变更（逆向存量 spec）", "benchmark: measured"
        )
        self.assertGreater(self.check(text), 0)


if __name__ == "__main__":
    unittest.main()
