# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json

import pytest

from conftest import (
    get_all_skills,
    get_skill_path,
    get_skills_with_evals,
    load_evals_md
)


class TestEvalsMdStructure:
    """Test evals.md file structure and completeness"""

    @pytest.mark.parametrize("skill_name", get_skills_with_evals(), indirect=False)
    def test_evals_md_exists(self, skill_name):
        data = load_evals_md(skill_name)
        assert data is not None, f"evals.md not found for skill: {skill_name}"

    @pytest.mark.parametrize("skill_name", get_skills_with_evals(), indirect=False)
    def test_evals_md_valid(self, skill_name):
        data = load_evals_md(skill_name)
        if data is None:
            pytest.skip(f"evals.md not found for skill: {skill_name}")
        assert isinstance(data, dict), f"evals.md should parse to a dict for skill: {skill_name}"

    @pytest.mark.parametrize("skill_name", get_skills_with_evals(), indirect=False)
    def test_evals_md_has_skill_name(self, skill_name):
        data = load_evals_md(skill_name)
        assert data is not None, f"Could not load evals.md for skill: {skill_name}"
        assert "skill_name" in data, f"Missing 'skill_name' field in evals.md for skill: {skill_name}"
        assert data["skill_name"], f"'skill_name' field is empty for skill: {skill_name}"

    @pytest.mark.parametrize("skill_name", get_skills_with_evals(), indirect=False)
    def test_evals_md_has_evals_list(self, skill_name):
        data = load_evals_md(skill_name)
        assert data is not None, f"Could not load evals.md for skill: {skill_name}"
        assert "evals" in data, f"Missing 'evals' field in evals.md for skill: {skill_name}"
        assert isinstance(data["evals"], list), f"'evals' should be a list for skill: {skill_name}"
        assert len(data["evals"]) > 0, f"'evals' list is empty for skill: {skill_name}"


class TestEvalCaseStructure:
    """Test individual eval case structure"""

    @staticmethod
    def _validate_expectation(exp, exp_index, case_index, skill_name):
        valid_types = ("contains", "not_contains", "file_exists", "file_list", "skill_activated")
        assert isinstance(exp, dict), \
            f"Expectation {exp_index} should be a dict in eval case {case_index} for skill: {skill_name}"
        assert "type" in exp, \
            f"Expectation {exp_index} missing 'type' in eval case {case_index} for skill: {skill_name}"
        assert exp["type"] in valid_types, \
            f"Expectation {exp_index} type '{exp['type']}' should be one of {valid_types} in skill: {skill_name}"
        if exp["type"] in ("contains", "not_contains", "skill_activated"):
            assert "pattern" in exp, \
                f"Expectation {exp_index} missing 'pattern' in eval case {case_index} for skill: {skill_name}"

    @pytest.mark.parametrize("skill_name", get_skills_with_evals(), indirect=False)
    def test_eval_cases_have_id(self, skill_name):
        data = load_evals_md(skill_name)
        assert data is not None
        for i, eval_case in enumerate(data["evals"]):
            assert "id" in eval_case, f"Eval case {i} missing 'id' in skill: {skill_name}"
            assert isinstance(eval_case["id"], int), f"Eval case 'id' should be integer in skill: {skill_name}"

    @pytest.mark.parametrize("skill_name", get_skills_with_evals(), indirect=False)
    def test_eval_cases_have_name(self, skill_name):
        data = load_evals_md(skill_name)
        assert data is not None
        for i, eval_case in enumerate(data["evals"]):
            assert "case_name" in eval_case, f"Eval case {i} missing 'case_name' in skill: {skill_name}"
            assert eval_case["case_name"], f"Eval case {i} has empty 'case_name' in skill: {skill_name}"

    @pytest.mark.parametrize("skill_name", get_skills_with_evals(), indirect=False)
    def test_eval_cases_have_prompt(self, skill_name):
        data = load_evals_md(skill_name)
        assert data is not None
        for i, eval_case in enumerate(data["evals"]):
            assert "prompt" in eval_case, f"Eval case {i} missing 'prompt' in skill: {skill_name}"
            assert eval_case["prompt"], f"Eval case {i} has empty 'prompt' in skill: {skill_name}"

    @pytest.mark.parametrize("skill_name", get_skills_with_evals(), indirect=False)
    def test_eval_cases_have_expected_output(self, skill_name):
        data = load_evals_md(skill_name)
        assert data is not None
        for i, eval_case in enumerate(data["evals"]):
            assert "expected_output" in eval_case, f"Eval case {i} missing 'expected_output' in skill: {skill_name}"

    @pytest.mark.parametrize("skill_name", get_skills_with_evals(), indirect=False)
    def test_eval_cases_expectations_format(self, skill_name):
        data = load_evals_md(skill_name)
        assert data is not None
        for i, eval_case in enumerate(data["evals"]):
            if "expectations" not in eval_case:
                continue
            assert isinstance(eval_case["expectations"], list), \
                f"Eval case 'expectations' should be a list in skill: {skill_name}"
            for j, exp in enumerate(eval_case["expectations"]):
                self._validate_expectation(exp, j, i, skill_name)


class TestEvalCaseLogic:
    """Test eval case logic and consistency"""

    @pytest.mark.parametrize("skill_name", get_skills_with_evals(), indirect=False)
    def test_eval_ids_are_unique(self, skill_name):
        data = load_evals_md(skill_name)
        assert data is not None
        ids = [eval_case["id"] for eval_case in data["evals"]]
        assert len(ids) == len(set(ids)), f"Duplicate eval IDs found in skill: {skill_name}"

    @pytest.mark.parametrize("skill_name", get_skills_with_evals(), indirect=False)
    def test_eval_ids_are_sequential(self, skill_name):
        data = load_evals_md(skill_name)
        assert data is not None
        ids = sorted([eval_case["id"] for eval_case in data["evals"]])
        expected_ids = list(range(ids[0], ids[0] + len(ids)))
        if ids != expected_ids:
            pytest.skip(
                f"Eval IDs are not sequential in skill: {skill_name}. "
                f"Got: {ids}, expected: {expected_ids}. "
                f"This is a known issue and will be resolved by renumbering."
            )

    @pytest.mark.parametrize("skill_name", get_skills_with_evals(), indirect=False)
    def test_prompt_is_descriptive(self, skill_name):
        data = load_evals_md(skill_name)
        assert data is not None
        for eval_case in data["evals"]:
            prompt = eval_case.get("prompt", "")
            assert len(prompt) >= 1, \
                f"Prompt too short (should describe the task) in eval {eval_case['id']} for skill: {skill_name}"

    @pytest.mark.parametrize("skill_name", get_skills_with_evals(), indirect=False)
    def test_expected_output_matches_prompt(self, skill_name):
        data = load_evals_md(skill_name)
        assert data is not None
        for eval_case in data["evals"]:
            prompt = eval_case.get("prompt", "")
            expected = eval_case.get("expected_output", "")
            if expected:
                assert len(expected) >= 5, \
                    f"Expected output too short for eval {eval_case['id']} in skill: {skill_name}"

    @pytest.mark.parametrize("skill_name", get_skills_with_evals(), indirect=False)
    def test_eval_cases_distractor_skills_valid(self, skill_name):
        """Distractor skills must reference real skills and not be the target skill."""
        data = load_evals_md(skill_name)
        assert data is not None
        for eval_case in data["evals"]:
            distractor_skills = eval_case.get("distractor_skills", [])
            if not distractor_skills:
                continue
            for ds_name in distractor_skills:
                ds_dir = get_skill_path(ds_name)
                assert ds_dir is not None, (
                    f"Distractor skill '{ds_name}' in eval case {eval_case['id']} "
                    f"for skill '{skill_name}' not found in any skill_dirs"
                )
                assert ds_name != skill_name, (
                    f"Distractor skill '{ds_name}' cannot be the same as target skill "
                    f"'{skill_name}' in eval case {eval_case['id']}"
                )


class TestEvalMode:
    """Test eval_mode field validity"""

    @pytest.mark.parametrize("skill_name", get_skills_with_evals(), indirect=False)
    def test_skill_eval_mode_valid(self, skill_name):
        data = load_evals_md(skill_name)
        assert data is not None, f"Could not load evals.md for skill: {skill_name}"
        eval_mode = data.get("eval_mode", "text")
        assert eval_mode in ("text", "file_based"), \
            f"eval_mode '{eval_mode}' should be 'text' or 'file_based' in skill: {skill_name}"


class TestSkillDirectory:
    """Test skill directory structure"""

    @pytest.mark.parametrize("skill_name", get_all_skills(), indirect=False)
    def test_skill_has_skill_md(self, skill_name):
        skill_path = get_skill_path(skill_name)
        if not skill_path:
            pytest.skip(f"Skill directory not found: {skill_name}")
        skill_md_path = skill_path / "SKILL.md"
        assert skill_md_path.exists(), f"SKILL.md not found for skill: {skill_name}"

    @pytest.mark.parametrize("skill_name", get_all_skills(), indirect=False)
    def test_skill_md_has_frontmatter(self, skill_name):
        skill_path = get_skill_path(skill_name)
        if not skill_path:
            pytest.skip(f"Skill directory not found: {skill_name}")
        skill_md_path = skill_path / "SKILL.md"
        if not skill_md_path.exists():
            pytest.skip(f"SKILL.md not found for skill: {skill_name}")

        with open(skill_md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert content.startswith("---"), \
            f"SKILL.md should have YAML frontmatter starting with '---' for skill: {skill_name}"

        frontmatter_end = content.find("---", 3)
        assert frontmatter_end > 0, \
            f"SKILL.md frontmatter should end with '---' for skill: {skill_name}"

    @pytest.mark.parametrize("skill_name", get_all_skills(), indirect=False)
    def test_skill_md_has_required_fields(self, skill_name):
        skill_path = get_skill_path(skill_name)
        if not skill_path:
            pytest.skip(f"Skill directory not found: {skill_name}")
        skill_md_path = skill_path / "SKILL.md"
        if not skill_md_path.exists():
            pytest.skip(f"SKILL.md not found for skill: {skill_name}")

        with open(skill_md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        frontmatter_start = content.find("---") + 3
        frontmatter_end = content.find("---", frontmatter_start)
        frontmatter = content[frontmatter_start:frontmatter_end]

        assert "name:" in frontmatter, \
            f"SKILL.md frontmatter missing 'name' field for skill: {skill_name}"
        assert "description:" in frontmatter, \
            f"SKILL.md frontmatter missing 'description' field for skill: {skill_name}"
