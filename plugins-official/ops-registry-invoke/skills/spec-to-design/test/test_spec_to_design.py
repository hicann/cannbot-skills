# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for spec-to-design scripts: frontmatter parsing, plan assembly, and validation."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from frontmatter_utils import parse_yaml_frontmatter
import assemble_design
import validate_design
import validate_completeness

# Public aliases for module-level helpers under test, so tests reference stable
# names instead of touching the underscore-prefixed members of another module.
strip_conditional_blocks = getattr(assemble_design, "_strip_conditional_blocks")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_workflow_validator_module(module_name):
    script_path = (
        Path(__file__).resolve().parents[3]
        / "workflow"
        / "resources"
        / "validate-workflow-state.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ParseYamlFrontmatterTest(unittest.TestCase):
    def test_valid_frontmatter(self):
        text = "---\niteration_count: 2\niterations:\n  - id: 1\n---\n\n# Body"
        fm, body = parse_yaml_frontmatter(text)
        self.assertEqual(fm["iteration_count"], 2)
        self.assertIsInstance(fm["iterations"], list)
        self.assertEqual(len(fm["iterations"]), 1)
        self.assertIn("# Body", body)

    def test_no_frontmatter(self):
        text = "# Just a heading\n\nSome content"
        fm, body = parse_yaml_frontmatter(text)
        self.assertEqual(fm, {})
        self.assertIn("# Just a heading", body)

    def test_unclosed_frontmatter(self):
        text = "---\niteration_count: 1\n# No closing delimiter"
        fm, body = parse_yaml_frontmatter(text)
        self.assertEqual(fm, {})
        self.assertEqual(body, text)

    def test_empty_frontmatter(self):
        text = "---\n---\n\n# Body"
        fm, body = parse_yaml_frontmatter(text)
        self.assertEqual(fm, {})
        self.assertIn("# Body", body)

    def test_string_iteration_count(self):
        text = '---\niteration_count: "3"\n---\n\n# Body'
        fm, body = parse_yaml_frontmatter(text)
        self.assertEqual(fm["iteration_count"], "3")

    def test_shared_identity(self):
        """parse_yaml_frontmatter must be the same function object across modules."""
        self.assertIs(parse_yaml_frontmatter, assemble_design.parse_yaml_frontmatter)
        self.assertIs(parse_yaml_frontmatter, validate_design.parse_yaml_frontmatter)


class RenderFrontmatterYamlTest(unittest.TestCase):
    def test_iteration_count_1(self):
        yaml_text = assemble_design.render_frontmatter_yaml(1, "float16")
        fm, body = self._parse(yaml_text)
        self.assertEqual(fm["iteration_count"], 1)
        self.assertEqual(len(fm["iterations"]), 1)
        self.assertEqual(fm["iterations"][0]["wave1"]["a1_p"], [])
        self.assertEqual(fm["iterations"][0]["goal"], "全功能实现 + 全覆盖")
        self.assertEqual(body.strip(), "")

    def test_iteration_count_2(self):
        yaml_text = assemble_design.render_frontmatter_yaml(2, "float16")
        fm, _ = self._parse(yaml_text)
        self.assertEqual(fm["iteration_count"], 2)
        self.assertEqual(len(fm["iterations"]), 2)
        self.assertEqual(len(fm["iterations"][0]["wave1"]["a1_p"]), 1)
        self.assertEqual(fm["iterations"][0]["wave1"]["a1_p"][0]["dtype"], "float16")
        self.assertEqual(fm["iterations"][1]["wave1"]["a1_p"], [])

    def test_iteration_count_3(self):
        yaml_text = assemble_design.render_frontmatter_yaml(3, "bfloat16")
        fm, _ = self._parse(yaml_text)
        self.assertEqual(fm["iteration_count"], 3)
        self.assertEqual(len(fm["iterations"]), 3)
        self.assertEqual(fm["iterations"][0]["wave1"]["a1_p"][0]["dtype"], "bfloat16")
        self.assertEqual(fm["iterations"][1]["wave1"]["a1_p"][0]["dtype"], "float32")
        self.assertEqual(fm["iterations"][2]["wave1"]["a1_p"], [])

    def test_schema_completeness(self):
        for ic in [1, 2, 3]:
            yaml_text = assemble_design.render_frontmatter_yaml(ic, "float16")
            fm, _ = self._parse(yaml_text)
            for idx, it in enumerate(fm["iterations"]):
                for field in ("id", "goal", "wave1", "wave2", "acceptance"):
                    self.assertIn(field, it, f"ic={ic} iter[{idx}] missing {field}")
                self.assertIn("a1_main", it["wave1"])
                self.assertIn("a1_p", it["wave1"])
                self.assertIn("b", it["wave1"])
                self.assertIn("a2", it["wave2"])
                self.assertIn("ut", it["acceptance"])
                self.assertIn("st", it["acceptance"])

    def test_dtype_substitution(self):
        for dtype in ["float16", "float32", "bfloat16", "int8"]:
            yaml_text = assemble_design.render_frontmatter_yaml(2, dtype)
            self.assertIn(dtype, yaml_text)

    def _parse(self, yaml_text):
        fm, body = parse_yaml_frontmatter(yaml_text)
        return fm, body


class StripConditionalBlocksTest(unittest.TestCase):
    def test_removes_ge2_block_for_ic1(self):
        text = "<!-- BEGIN iteration_count >= 2 -->\nBLOCK\n<!-- END iteration_count >= 2 -->"
        result = strip_conditional_blocks(text, 1)
        self.assertNotIn("BLOCK", result)

    def test_keeps_ge2_block_for_ic2(self):
        text = "<!-- BEGIN iteration_count >= 2 -->\nBLOCK\n<!-- END iteration_count >= 2 -->"
        result = strip_conditional_blocks(text, 2)
        self.assertIn("BLOCK", result)

    def test_keeps_ge2_block_for_ic3(self):
        text = "<!-- BEGIN iteration_count >= 2 -->\nBLOCK\n<!-- END iteration_count >= 2 -->"
        result = strip_conditional_blocks(text, 3)
        self.assertIn("BLOCK", result)

    def test_removes_eq3_block_for_ic2(self):
        text = "<!-- BEGIN iteration_count = 3 -->\nBLOCK\n<!-- END iteration_count = 3 -->"
        result = strip_conditional_blocks(text, 2)
        self.assertNotIn("BLOCK", result)

    def test_keeps_eq3_block_for_ic3(self):
        text = "<!-- BEGIN iteration_count = 3 -->\nBLOCK\n<!-- END iteration_count = 3 -->"
        result = strip_conditional_blocks(text, 3)
        self.assertIn("BLOCK", result)

    def test_ge1_always_kept(self):
        text = "<!-- BEGIN iteration_count >= 1 -->\nALWAYS\n<!-- END iteration_count >= 1 -->"
        for ic in [1, 2, 3]:
            result = strip_conditional_blocks(text, ic)
            self.assertIn("ALWAYS", result, f"ic={ic}")

    def test_real_template_ic1(self):
        templ_path = SCRIPTS_DIR.parent / "templates" / "PLAN.md.templ"
        templ_text = templ_path.read_text(encoding="utf-8")
        _, body = parse_yaml_frontmatter(templ_text)
        result = strip_conditional_blocks(body, 1)
        self.assertNotIn("## 迭代一穿刺列表", result)
        self.assertNotIn("## 迭代二整合目标", result)
        self.assertIn("## 迭代{N}全覆盖目标", result)
        self.assertIn("## 穿刺结果判定", result)

    def test_real_template_ic2(self):
        templ_path = SCRIPTS_DIR.parent / "templates" / "PLAN.md.templ"
        templ_text = templ_path.read_text(encoding="utf-8")
        _, body = parse_yaml_frontmatter(templ_text)
        result = strip_conditional_blocks(body, 2)
        self.assertIn("## 迭代一穿刺列表", result)
        self.assertNotIn("## 迭代二整合目标", result)
        self.assertIn("## 迭代{N}全覆盖目标", result)

    def test_real_template_ic3(self):
        templ_path = SCRIPTS_DIR.parent / "templates" / "PLAN.md.templ"
        templ_text = templ_path.read_text(encoding="utf-8")
        _, body = parse_yaml_frontmatter(templ_text)
        result = strip_conditional_blocks(body, 3)
        self.assertIn("## 迭代一穿刺列表", result)
        self.assertIn("## 迭代二整合目标", result)
        self.assertIn("## 迭代{N}全覆盖目标", result)


class RenderDefaultPlanTest(unittest.TestCase):
    def test_no_template_uses_ic3(self):
        plan = assemble_design.render_default_plan_no_template("test_op", "float16")
        fm, body = parse_yaml_frontmatter(plan)
        self.assertEqual(fm["iteration_count"], 3)
        self.assertEqual(len(fm["iterations"]), 3)
        self.assertIn("# test_op 迭代执行计划", body)
        self.assertIn("## 迭代一穿刺列表", body)
        self.assertIn("## 迭代三全覆盖目标", body)

    def test_with_template_ic1(self):
        templ_path = SCRIPTS_DIR.parent / "templates" / "PLAN.md.templ"
        templ_text = templ_path.read_text(encoding="utf-8")
        ic1_templ = templ_text.replace("iteration_count: 3", "iteration_count: 1")
        spec_text = "dtype_set: [float16, float32]"
        plan = assemble_design.render_default_plan("my_op", spec_text, ic1_templ)
        fm, body = parse_yaml_frontmatter(plan)
        self.assertEqual(fm["iteration_count"], 1)
        self.assertNotIn("## 迭代一穿刺列表", body)
        self.assertNotIn("## 迭代二整合目标", body)
        self.assertIn("## 全覆盖目标", body)
        self.assertNotIn("{N}", body)

    def test_with_template_ic2(self):
        templ_path = SCRIPTS_DIR.parent / "templates" / "PLAN.md.templ"
        templ_text = templ_path.read_text(encoding="utf-8")
        ic2_templ = templ_text.replace("iteration_count: 3", "iteration_count: 2")
        spec_text = "dtype_set: [float16]"
        plan = assemble_design.render_default_plan("my_op", spec_text, ic2_templ)
        fm, body = parse_yaml_frontmatter(plan)
        self.assertEqual(fm["iteration_count"], 2)
        self.assertIn("## 迭代一穿刺列表", body)
        self.assertIn("迭代二整合与全覆盖目标", body)
        self.assertNotIn("## 迭代二整合目标", body)

    def test_with_template_ic3(self):
        templ_path = SCRIPTS_DIR.parent / "templates" / "PLAN.md.templ"
        templ_text = templ_path.read_text(encoding="utf-8")
        spec_text = "dtype_set: [bfloat16]"
        plan = assemble_design.render_default_plan("my_op", spec_text, templ_text)
        fm, body = parse_yaml_frontmatter(plan)
        self.assertEqual(fm["iteration_count"], 3)
        self.assertIn("## 迭代一穿刺列表", body)
        self.assertIn("## 迭代二整合目标", body)
        self.assertIn("迭代三全覆盖目标", body)

    def test_dtype_substitution_in_template(self):
        templ_path = SCRIPTS_DIR.parent / "templates" / "PLAN.md.templ"
        templ_text = templ_path.read_text(encoding="utf-8")
        spec_text = "dtype_set: [int8, float16]"
        plan = assemble_design.render_default_plan("op", spec_text, templ_text)
        self.assertIn("int8", plan)


class ValidatePlanTest(unittest.TestCase):
    def test_valid_plan_ic1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = self._write_plan(tmpdir, "test_op", 1, self._minimal_body(1))
            errors = validate_design.validate_plan(plan_path, "test_op")
            self.assertEqual(errors, [], f"Unexpected errors: {errors}")

    def test_valid_plan_ic2(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = self._write_plan(tmpdir, "test_op", 2, self._minimal_body(2))
            errors = validate_design.validate_plan(plan_path, "test_op")
            self.assertEqual(errors, [], f"Unexpected errors: {errors}")

    def test_valid_plan_ic3(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = self._write_plan(tmpdir, "test_op", 3, self._minimal_body(3))
            errors = validate_design.validate_plan(plan_path, "test_op")
            self.assertEqual(errors, [], f"Unexpected errors: {errors}")

    def test_missing_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "PLAN.md"
            write_text(plan_path, "# test_op 迭代执行计划\n\n## 修订记录\n")
            errors = validate_design.validate_plan(plan_path, "test_op")
            self.assertTrue(any("missing YAML frontmatter" in e for e in errors))

    def test_wrong_iteration_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "PLAN.md"
            write_text(plan_path, "---\niteration_count: 5\niterations: []\n---\n\n# test_op 迭代执行计划\n")
            errors = validate_design.validate_plan(plan_path, "test_op")
            self.assertTrue(any("must be 1, 2, or 3" in e for e in errors))

    def test_iterations_length_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "PLAN.md"
            write_text(
                plan_path,
                "---\niteration_count: 2\niterations:\n  - id: 1\n    goal: x\n"
                "    wave1: {}\n    wave2: {}\n    acceptance: {}\n---\n\n"
                "# test_op 迭代执行计划\n",
            )
            errors = validate_design.validate_plan(plan_path, "test_op")
            self.assertTrue(any("length" in e for e in errors))

    def _write_plan(self, tmpdir, op_name, iteration_count, body_sections):
        fm_yaml = assemble_design.render_frontmatter_yaml(iteration_count, "float16")
        body = f"# {op_name} 迭代执行计划\n\n" + body_sections
        plan_path = Path(tmpdir) / "PLAN.md"
        write_text(plan_path, fm_yaml + body)
        return plan_path

    def _minimal_body(self, iteration_count):
        sections = ["## 修订记录\n\n| v | r | t | a |\n|---|---|---|---|\n"]
        if iteration_count >= 2:
            sections.append(
                "## 迭代一穿刺列表\n\n| t | k | d | s | m | v |\n"
                "|---|---|---|---|---|---|\n"
            )
        if iteration_count == 3:
            sections.append("## 迭代二整合目标\n\ncontent\n")
            sections.append(
                "## 迭代二穿刺列表\n\n| t | v | i | d | e |\n"
                "|---|---|---|---|---|\n"
            )
        if iteration_count == 1:
            sections.append("## 全覆盖目标\n\n| d | c | s | n |\n|---|---|---|---|\n")
        if iteration_count == 2:
            sections.append(
                "## 迭代二整合与全覆盖目标\n\n| d | c | s | n |\n|---|---|---|---|\n"
            )
        if iteration_count == 3:
            sections.append(
                "## 迭代三全覆盖目标\n\n| d | c | s | n |\n|---|---|---|---|\n"
            )
        sections.append("## 穿刺结果判定\n\n| s | c | h |\n|---|---|---|\n")
        return "\n".join(sections)


class ValidatePlanCompletenessTest(unittest.TestCase):
    def test_ic1_requires_no_iteration_terms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fm = assemble_design.render_frontmatter_yaml(1, "float16")
            body = (
                "# op 迭代执行计划\n\n## 修订记录\n\n| v | r | t | a |\n"
                "|---|---|---|---|\n\n## 全覆盖目标\n\n"
                "TilingKey Dtype Memory Strategy\n\n## 穿刺结果判定\n\n"
                "| s | c | h |\n|---|---|---|\n"
            )
            plan_path = Path(tmpdir) / "PLAN.md"
            write_text(plan_path, fm + body)
            errors = validate_completeness.validate_plan_completeness(plan_path)
            self.assertFalse(any("迭代二" in e for e in errors))
            self.assertFalse(any("迭代三" in e for e in errors))

    def test_ic3_requires_all_iteration_terms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fm = assemble_design.render_frontmatter_yaml(3, "float16")
            body = (
                "# op 迭代执行计划\n\n## 修订记录\n\n| v | r | t | a |\n"
                "|---|---|---|---|\n\n## 迭代一穿刺列表\n\n"
                "TilingKey Dtype Memory Strategy\n\n## 迭代二整合目标\n\n"
                "迭代一 迭代二\n\n## 迭代二穿刺列表\n\n## 迭代三全覆盖目标\n\n"
                "迭代三\n\n## 穿刺结果判定\n\n| s | c | h |\n|---|---|---|\n"
            )
            plan_path = Path(tmpdir) / "PLAN.md"
            write_text(plan_path, fm + body)
            errors = validate_completeness.validate_plan_completeness(plan_path)
            self.assertFalse(any("missing required term" in e for e in errors))


class ReadIterationCountTest(unittest.TestCase):
    def test_reads_from_plan_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            op_dir = Path(tmpdir)
            docs_dir = op_dir / "docs"
            docs_dir.mkdir()
            write_text(
                docs_dir / "PLAN.md",
                "---\niteration_count: 2\niterations: []\n---\n\n# op\n",
            )
            mod = load_workflow_validator_module("vws")
            v = mod.WorkflowValidator(op_dir)
            self.assertEqual(v.read_iteration_count(), 2)

    def test_defaults_to_3_when_no_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            op_dir = Path(tmpdir)
            mod = load_workflow_validator_module("vws2")
            v = mod.WorkflowValidator(op_dir)
            self.assertEqual(v.read_iteration_count(), 3)


if __name__ == "__main__":
    unittest.main()
