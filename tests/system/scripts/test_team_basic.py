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
    get_all_teams,
    get_team_path,
    get_teams_with_evals,
    load_team_evals_md,
)


class TestTeamEvalsMdStructure:
    """Test team evals.md file structure and completeness"""

    @pytest.mark.parametrize("team_name", get_teams_with_evals(), indirect=False)
    def test_team_evals_md_exists(self, team_name):
        data = load_team_evals_md(team_name)
        assert data is not None, f"evals.md not found for team: {team_name}"

    @pytest.mark.parametrize("team_name", get_teams_with_evals(), indirect=False)
    def test_team_evals_md_valid(self, team_name):
        data = load_team_evals_md(team_name)
        if data is None:
            pytest.skip(f"evals.md not found for team: {team_name}")
        assert isinstance(data, dict), f"evals.md should parse to a dict for team: {team_name}"

    @pytest.mark.parametrize("team_name", get_teams_with_evals(), indirect=False)
    def test_team_evals_md_has_team_name(self, team_name):
        data = load_team_evals_md(team_name)
        assert data is not None, f"Could not load evals.md for team: {team_name}"
        assert "team_name" in data, f"Missing 'team_name' field in evals.md for team: {team_name}"
        assert data["team_name"], f"'team_name' field is empty for team: {team_name}"

    @pytest.mark.parametrize("team_name", get_teams_with_evals(), indirect=False)
    def test_team_evals_md_has_evals_list(self, team_name):
        data = load_team_evals_md(team_name)
        assert data is not None, f"Could not load evals.md for team: {team_name}"
        assert "evals" in data, f"Missing 'evals' field in evals.md for team: {team_name}"
        assert isinstance(data["evals"], list), f"'evals' should be a list for team: {team_name}"
        assert len(data["evals"]) > 0, f"'evals' list is empty for team: {team_name}"


class TestTeamEvalCaseStructure:
    """Test individual team eval case structure"""

    @staticmethod
    def _validate_expectation(exp, exp_index, case_index, team_name):
        valid_types = ("contains", "not_contains", "file_exists", "file_list", "file_contains", "skill_activated")
        assert isinstance(exp, dict), \
            f"Expectation {exp_index} should be a dict in eval case {case_index} for team: {team_name}"
        assert "type" in exp, \
            f"Expectation {exp_index} missing 'type' in eval case {case_index} for team: {team_name}"
        assert exp["type"] in valid_types, \
            f"Expectation {exp_index} type '{exp['type']}' should be one of {valid_types} in team: {team_name}"
        if exp["type"] in ("contains", "not_contains", "file_contains", "skill_activated"):
            assert "pattern" in exp, \
                f"Expectation {exp_index} missing 'pattern' in eval case {case_index} for team: {team_name}"

    @pytest.mark.parametrize("team_name", get_teams_with_evals(), indirect=False)
    def test_team_eval_cases_have_id(self, team_name):
        data = load_team_evals_md(team_name)
        assert data is not None
        for i, eval_case in enumerate(data["evals"]):
            assert "id" in eval_case, f"Eval case {i} missing 'id' in team: {team_name}"
            assert isinstance(eval_case["id"], int), f"Eval case 'id' should be integer in team: {team_name}"

    @pytest.mark.parametrize("team_name", get_teams_with_evals(), indirect=False)
    def test_team_eval_cases_have_name(self, team_name):
        data = load_team_evals_md(team_name)
        assert data is not None
        for i, eval_case in enumerate(data["evals"]):
            assert "case_name" in eval_case, f"Eval case {i} missing 'case_name' in team: {team_name}"
            assert eval_case["case_name"], f"Eval case {i} has empty 'case_name' in team: {team_name}"

    @pytest.mark.parametrize("team_name", get_teams_with_evals(), indirect=False)
    def test_team_eval_cases_have_prompt(self, team_name):
        data = load_team_evals_md(team_name)
        assert data is not None
        for i, eval_case in enumerate(data["evals"]):
            assert "prompt" in eval_case, f"Eval case {i} missing 'prompt' in team: {team_name}"
            assert eval_case["prompt"], f"Eval case {i} has empty 'prompt' in team: {team_name}"

    @pytest.mark.parametrize("team_name", get_teams_with_evals(), indirect=False)
    def test_team_eval_cases_have_expected_output(self, team_name):
        data = load_team_evals_md(team_name)
        assert data is not None
        for i, eval_case in enumerate(data["evals"]):
            assert "expected_output" in eval_case, \
                f"Eval case {i} missing 'expected_output' in team: {team_name}"

    @pytest.mark.parametrize("team_name", get_teams_with_evals(), indirect=False)
    def test_team_eval_cases_expectations_format(self, team_name):
        data = load_team_evals_md(team_name)
        assert data is not None
        for i, eval_case in enumerate(data["evals"]):
            if "expectations" not in eval_case:
                continue
            assert isinstance(eval_case["expectations"], list), \
                f"Eval case 'expectations' should be a list in team: {team_name}"
            for j, exp in enumerate(eval_case["expectations"]):
                self._validate_expectation(exp, j, i, team_name)


class TestTeamEvalCaseLogic:
    """Test team eval case logic and consistency"""

    @pytest.mark.parametrize("team_name", get_teams_with_evals(), indirect=False)
    def test_team_eval_ids_are_unique(self, team_name):
        data = load_team_evals_md(team_name)
        assert data is not None
        ids = [eval_case["id"] for eval_case in data["evals"]]
        assert len(ids) == len(set(ids)), f"Duplicate eval IDs found in team: {team_name}"

    @pytest.mark.parametrize("team_name", get_teams_with_evals(), indirect=False)
    def test_team_eval_ids_are_sequential(self, team_name):
        data = load_team_evals_md(team_name)
        assert data is not None
        ids = sorted([eval_case["id"] for eval_case in data["evals"]])
        expected_ids = list(range(ids[0], ids[0] + len(ids)))
        assert ids == expected_ids, \
            f"Eval IDs should be sequential starting from {ids[0]} in team: {team_name}. Got: {ids}"

    @pytest.mark.parametrize("team_name", get_teams_with_evals(), indirect=False)
    def test_team_prompt_is_descriptive(self, team_name):
        data = load_team_evals_md(team_name)
        assert data is not None
        for eval_case in data["evals"]:
            prompt = eval_case.get("prompt", "")
            assert len(prompt) >= 1, \
                f"Prompt too short in eval {eval_case['id']} for team: {team_name}"

    @pytest.mark.parametrize("team_name", get_teams_with_evals(), indirect=False)
    def test_team_expected_output_matches_prompt(self, team_name):
        data = load_team_evals_md(team_name)
        assert data is not None
        for eval_case in data["evals"]:
            expected = eval_case.get("expected_output", "")
            if expected:
                assert len(expected) >= 5, \
                    f"Expected output too short for eval {eval_case['id']} in team: {team_name}"


class TestTeamEvalMode:
    """Test team eval_mode field validity"""

    @pytest.mark.parametrize("team_name", get_teams_with_evals(), indirect=False)
    def test_team_eval_mode_valid(self, team_name):
        data = load_team_evals_md(team_name)
        assert data is not None, f"Could not load evals.md for team: {team_name}"
        eval_mode = data.get("eval_mode", "text")
        assert eval_mode in ("text", "file_based"), \
            f"eval_mode '{eval_mode}' should be 'text' or 'file_based' in team: {team_name}"


class TestTeamDirectory:
    """Test team directory structure"""

    @pytest.mark.parametrize("team_name", get_all_teams(), indirect=False)
    def test_team_has_agents_md(self, team_name):
        team_path = get_team_path(team_name)
        if not team_path:
            pytest.skip(f"Team directory not found: {team_name}")
        agents_md_path = team_path / "AGENTS.md"
        assert agents_md_path.exists(), f"AGENTS.md not found for team: {team_name}"

    @pytest.mark.parametrize("team_name", get_all_teams(), indirect=False)
    def test_team_agents_md_has_frontmatter(self, team_name):
        team_path = get_team_path(team_name)
        if not team_path:
            pytest.skip(f"Team directory not found: {team_name}")
        agents_md_path = team_path / "AGENTS.md"
        if not agents_md_path.exists():
            pytest.skip(f"AGENTS.md not found for team: {team_name}")

        with open(agents_md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert content.startswith("---"), \
            f"AGENTS.md should have YAML frontmatter starting with '---' for team: {team_name}"

        frontmatter_end = content.find("---", 3)
        assert frontmatter_end > 0, \
            f"AGENTS.md frontmatter should end with '---' for team: {team_name}"

    @pytest.mark.parametrize("team_name", get_all_teams(), indirect=False)
    def test_team_agents_md_has_required_fields(self, team_name):
        team_path = get_team_path(team_name)
        if not team_path:
            pytest.skip(f"Team directory not found: {team_name}")
        agents_md_path = team_path / "AGENTS.md"
        if not agents_md_path.exists():
            pytest.skip(f"AGENTS.md not found for team: {team_name}")

        with open(agents_md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        frontmatter_start = content.find("---") + 3
        frontmatter_end = content.find("---", frontmatter_start)
        frontmatter = content[frontmatter_start:frontmatter_end]

        assert "description:" in frontmatter, \
            f"AGENTS.md frontmatter missing 'description' field for team: {team_name}"
        assert "mode:" in frontmatter, \
            f"AGENTS.md frontmatter missing 'mode' field for team: {team_name}"

    @pytest.mark.parametrize("team_name", get_all_teams(), indirect=False)
    def test_team_has_plugin_json(self, team_name):
        team_path = get_team_path(team_name)
        if not team_path:
            pytest.skip(f"Team directory not found: {team_name}")
        plugin_json_path = team_path / ".claude-plugin" / "plugin.json"
        assert plugin_json_path.exists(), f"plugin.json not found for team: {team_name}"

    @pytest.mark.parametrize("team_name", get_all_teams(), indirect=False)
    def test_team_plugin_json_valid(self, team_name):
        team_path = get_team_path(team_name)
        if not team_path:
            pytest.skip(f"Team directory not found: {team_name}")
        plugin_json_path = team_path / ".claude-plugin" / "plugin.json"
        if not plugin_json_path.exists():
            pytest.skip(f"plugin.json not found for team: {team_name}")

        with open(plugin_json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"plugin.json is invalid JSON for team: {team_name}: {e}")

        assert "name" in data, f"plugin.json missing 'name' field for team: {team_name}"
        assert "version" in data, f"plugin.json missing 'version' field for team: {team_name}"

    @pytest.mark.parametrize("team_name", get_all_teams(), indirect=False)
    def test_team_has_init_sh(self, team_name):
        team_path = get_team_path(team_name)
        if not team_path:
            pytest.skip(f"Team directory not found: {team_name}")
        init_sh_path = team_path / "init.sh"
        assert init_sh_path.exists(), f"init.sh not found for team: {team_name}"
