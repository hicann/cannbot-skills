# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Test SKILL.md structure and internal reference integrity.

Migrated from scripts/validate_skills.py into pytest for better
discovery, parametrization, and CI reporting.
"""

import re
from pathlib import Path

import pytest

from conftest import SKILLS_DIR

# Discover all skill directories
SKILL_DIRS = sorted(
    d for d in SKILLS_DIR.iterdir()
    if d.is_dir() and not d.name.startswith(".")
)


@pytest.fixture(params=SKILL_DIRS, ids=[d.name for d in SKILL_DIRS])
def skill_dir(request):
    return request.param


class TestSkillMdExists:
    @staticmethod
    def test_skill_md_present(skill_dir):
        assert (skill_dir / "SKILL.md").exists(), (
            f"{skill_dir.name}/ missing SKILL.md"
        )


class TestSkillMdContent:
    @staticmethod
    def test_not_suspiciously_short(skill_dir):
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            pytest.skip("no SKILL.md")
        content = skill_md.read_text(encoding="utf-8")
        assert len(content.strip()) >= 50, (
            f"{skill_dir.name}/SKILL.md too short ({len(content.strip())} chars)"
        )

    @staticmethod
    def test_not_too_long(skill_dir):
        """SKILL.md should be under 500 lines (Anthropic progressive disclosure guideline)."""
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            pytest.skip("no SKILL.md")
        lines = skill_md.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 500, (
            f"{skill_dir.name}/SKILL.md too long ({len(lines)} lines, max 500)"
        )

    @staticmethod
    def test_description_quality(skill_dir):
        """YAML frontmatter description must be non-empty (>= 5 words)."""
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            pytest.skip("no SKILL.md")
        content = skill_md.read_text(encoding="utf-8")
        # Handle both inline and multiline (>) YAML descriptions
        fm_match = re.search(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not fm_match:
            pytest.skip("no YAML frontmatter")
        fm = fm_match.group(1)
        # Multiline description
        dm = re.search(r"description:\s*>\s*\n(.*?)(?=\n\w|\Z)", fm, re.DOTALL)
        if dm:
            desc = " ".join(dm.group(1).split())
        else:
            # Inline description
            dm = re.search(r"description:\s*(.+)", fm)
            if not dm:
                pytest.fail(f"{skill_dir.name}/SKILL.md missing description in frontmatter")
            desc = dm.group(1).strip().strip("\"'")
        word_count = len(desc.split())
        assert word_count >= 5, (
            f"{skill_dir.name}/SKILL.md description too short ({word_count} words, min 5)"
        )
        assert word_count < 100, (
            f"{skill_dir.name}/SKILL.md description too long ({word_count} words, max 99)"
        )


class TestScriptPaths:
    """SKILL.md must use ${CLAUDE_SKILL_DIR} for script paths, not hardcoded relative paths.

    See: https://code.claude.com/docs/en/skills
    Plugin skills run in the user's project directory, so relative paths to
    plugin-bundled scripts will fail. ${CLAUDE_SKILL_DIR} is auto-resolved
    by Claude Code to the skill's absolute directory path.
    """

    # Pattern: any command referencing scripts/ with a hardcoded skill path
    # Matches: bash skills/..., python3 skills/..., .venv/bin/python3 skills/..., etc.
    HARDCODED_SCRIPT_RE = re.compile(
        r"(?:bash|python3?|sh|\.venv/bin/python3?)\s+skills/[^/]+/scripts/", re.MULTILINE
    )

    def test_no_hardcoded_script_paths(self, skill_dir):
        """Scripts in code blocks must use ${CLAUDE_SKILL_DIR}, not relative paths.

        Plugin skills run in the user's project directory, so
        'bash skills/foo/scripts/bar.sh' will fail with file not found.
        Use 'bash ${CLAUDE_SKILL_DIR}/scripts/bar.sh' instead.
        """
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            pytest.skip("no SKILL.md")
        content = skill_md.read_text(encoding="utf-8")
        # Only check inside code blocks (``` ... ```)
        code_blocks = re.findall(r"```[^\n]*\n(.*?)```", content, re.DOTALL)
        violations = []
        for block in code_blocks:
            for match in self.HARDCODED_SCRIPT_RE.finditer(block):
                violations.append(match.group(0).strip())
        assert not violations, (
            f"{skill_dir.name}/SKILL.md has hardcoded script paths in code blocks "
            f"(use ${{CLAUDE_SKILL_DIR}}/scripts/ instead): {violations}"
        )


class TestInternalLinks:
    @staticmethod
    def test_reference_links_resolve(skill_dir):
        """All [text](references/...) links in SKILL.md must point to existing files."""
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            pytest.skip("no SKILL.md")
        content = skill_md.read_text(encoding="utf-8")
        links = re.findall(r"\[.*?\]\((references/[^)]+)\)", content)
        broken = []
        for link in links:
            target = skill_dir / link
            if not target.exists():
                broken.append(link)
        assert not broken, (
            f"{skill_dir.name}/SKILL.md has broken links: {broken}"
        )

    @staticmethod
    def test_no_orphan_references(skill_dir):
        """Reference files should be linked from SKILL.md."""
        skill_md = skill_dir / "SKILL.md"
        refs_dir = skill_dir / "references"
        if not skill_md.exists() or not refs_dir.exists():
            pytest.skip("no SKILL.md or references/")
        content = skill_md.read_text(encoding="utf-8")
        links = re.findall(r"\[.*?\]\((references/[^)]+)\)", content)
        linked = {(skill_dir / link).resolve() for link in links}

        orphans = []
        for ref_file in refs_dir.glob("*.md"):
            if ref_file.resolve() not in linked:
                orphans.append(ref_file.name)
        # Warn only — orphans in subdirectories are expected
        if orphans:
            pytest.warns(UserWarning, match="orphan")

    @staticmethod
    def test_long_references_have_toc(skill_dir):
        """Reference files over 100 lines should have a TOC (Anthropic guideline)."""
        refs_dir = skill_dir / "references"
        if not refs_dir.exists():
            pytest.skip("no references/")
        poorly_structured = []
        for ref_file in refs_dir.glob("*.md"):
            content = ref_file.read_text(encoding="utf-8")
            lines = content.splitlines()
            if len(lines) > 100:
                toc_links = len(re.findall(r"\[.*?\]\(#.*?\)", content))
                if toc_links < 3:
                    poorly_structured.append(
                        f"{ref_file.name} ({len(lines)} lines, {toc_links} TOC links)")
        assert not poorly_structured, (
            f"{skill_dir.name}/ has long reference files lacking structure: {poorly_structured}"
        )
