# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for kb_load_audit.py (V2 #4)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import kb_load_audit as kba  # noqa: E402


# Fixture: minimal brief + tool_use traces
SAMPLE_BRIEF = """\
adamw-kw-1 — kernel-worker spawn

OP: 17_AdamW

# KB MANIFEST (load all of these BEFORE Phase A)

Op-class tags: scatter-gather, reduction

Required reading (paths are relative to src/skills/references/):
  - src/skills/references/KB_INDEX.md
  - src/skills/references/target/ascendc/OPERATIONAL_KNOWLEDGE.md
  - src/skills/references/target/ascendc/PLATFORM_BUGS.md
  - src/skills/references/hardware/target/ascend950pr.md
  - src/skills/references/target/ascendc/patterns/PATTERN_INDEX.md

# PHASES (cold-start aog-kernel-worker)

A. KB Manifest LOAD ...
"""


# ---------------------------------------------------------------------------
# extract_declared_sections
# ---------------------------------------------------------------------------
def test_extract_declared_basic():
    sections = kba.extract_declared_sections(SAMPLE_BRIEF)
    assert "src/skills/references/KB_INDEX.md" in sections
    assert "src/skills/references/target/ascendc/OPERATIONAL_KNOWLEDGE.md" in sections
    assert "src/skills/references/hardware/target/ascend950pr.md" in sections
    assert len(sections) == 5


def test_extract_declared_no_kb_manifest():
    """No KB MANIFEST block → empty list."""
    brief = "hello world\nno manifest here"
    assert kba.extract_declared_sections(brief) == []


def test_extract_declared_strips_anchor():
    """`X.md#Section` → just `X.md`."""
    brief = """
# KB MANIFEST

Required reading:
  - src/skills/references/target/ascendc/OPERATIONAL_KNOWLEDGE.md#OL-67

# PHASES
"""
    sections = kba.extract_declared_sections(brief)
    assert sections == ["src/skills/references/target/ascendc/OPERATIONAL_KNOWLEDGE.md"]


def test_extract_declared_only_inside_manifest_block():
    """Path mentions OUTSIDE # KB MANIFEST should not be picked up."""
    brief = """
# KB MANIFEST

Required reading:
  - src/skills/references/target/ascendc/OPERATIONAL_KNOWLEDGE.md

# PHASES
A. Read src/skills/references/some_other_file.md (this is in PHASES, not manifest)
"""
    sections = kba.extract_declared_sections(brief)
    # Note: current implementation extracts only from KB MANIFEST → next # heading
    assert "src/skills/references/target/ascendc/OPERATIONAL_KNOWLEDGE.md" in sections
    # The PHASES one might be picked up as "Required reading" line in PHASES
    # Verify our slicing stops at next # header.
    assert "src/skills/references/some_other_file.md" not in sections


# ---------------------------------------------------------------------------
# extract_read_paths
# ---------------------------------------------------------------------------
def test_extract_read_paths_from_read_tool():
    tool_uses = [
        {"tool_name": "Read", "input": {"file_path": "/abs/src/skills/references/KB_INDEX.md"}},
        {"tool_name": "Bash", "input": {"command": "ls"}},
    ]
    paths = kba.extract_read_paths(tool_uses)
    assert "src/skills/references/KB_INDEX.md" in paths


def test_extract_read_paths_grep():
    tool_uses = [
        {"tool_name": "Grep", "input": {"pattern": "OL-67",
                                          "path": "src/skills/references/target/ascendc/OPERATIONAL_KNOWLEDGE.md"}},
    ]
    paths = kba.extract_read_paths(tool_uses)
    assert "src/skills/references/target/ascendc/OPERATIONAL_KNOWLEDGE.md" in paths


def test_extract_read_paths_skips_non_kb():
    tool_uses = [
        {"tool_name": "Read", "input": {"file_path": "/tmp/something.txt"}},
        {"tool_name": "Read", "input": {"file_path": "workspace/22_Nonzero/kernel/x.h"}},
    ]
    assert kba.extract_read_paths(tool_uses) == []


def test_extract_read_paths_empty():
    assert kba.extract_read_paths([]) == []
    assert kba.extract_read_paths(None) == []


# ---------------------------------------------------------------------------
# Cross-reference coverage for the KB-load audit
# ---------------------------------------------------------------------------
def test_audit_full_coverage():
    tool_uses = [
        {"tool_name": "Read", "input": {"file_path": "/abs/src/skills/references/KB_INDEX.md"}},
        {
            "tool_name": "Read",
            "input": {"file_path": "/abs/src/skills/references/target/ascendc/OPERATIONAL_KNOWLEDGE.md"},
        },
        {"tool_name": "Read", "input": {"file_path": "/abs/src/skills/references/target/ascendc/PLATFORM_BUGS.md"}},
        {"tool_name": "Read", "input": {"file_path": "/abs/src/skills/references/hardware/target/ascend950pr.md"}},
        {
            "tool_name": "Read",
            "input": {"file_path": "/abs/src/skills/references/target/ascendc/patterns/PATTERN_INDEX.md"},
        },
    ]
    audit = kba.audit_kb_load(SAMPLE_BRIEF, tool_uses)
    assert audit.coverage_pct == 100
    assert audit.missing == []
    assert len(audit.covered) == 5


def test_audit_partial_coverage():
    tool_uses = [
        {"tool_name": "Read", "input": {"file_path": "src/skills/references/KB_INDEX.md"}},
        {"tool_name": "Read", "input": {"file_path": "src/skills/references/target/ascendc/OPERATIONAL_KNOWLEDGE.md"}},
    ]
    audit = kba.audit_kb_load(SAMPLE_BRIEF, tool_uses)
    assert audit.coverage_pct == 40  # 2/5
    assert "src/skills/references/target/ascendc/PLATFORM_BUGS.md" in audit.missing
    assert "src/skills/references/hardware/target/ascend950pr.md" in audit.missing


def test_audit_zero_coverage():
    audit = kba.audit_kb_load(SAMPLE_BRIEF, [])
    assert audit.coverage_pct == 0
    assert len(audit.missing) == 5


def test_audit_unexpected_reads():
    """Agent read KB sections not declared in brief — flag as unexpected."""
    tool_uses = [
        {"tool_name": "Read", "input": {"file_path": "src/skills/references/KB_INDEX.md"}},
        {"tool_name": "Read", "input": {"file_path": "src/skills/references/EXTRA_NOTES.md"}},
    ]
    audit = kba.audit_kb_load(SAMPLE_BRIEF, tool_uses)
    assert "src/skills/references/EXTRA_NOTES.md" in audit.unexpected


def test_audit_no_brief_manifest_returns_100_pct():
    """If brief has no KB MANIFEST block, coverage is trivially 100%."""
    audit = kba.audit_kb_load("no manifest here", [])
    assert audit.coverage_pct == 100
    assert audit.declared_sections == []


def test_audit_subpath_counts_as_covered():
    """Reading a file under a declared dir prefix counts as covered."""
    brief = """
# KB MANIFEST

Required reading:
  - src/skills/references/target/ascendc/patterns/PATTERN_INDEX.md

# PHASES
"""
    tool_uses = [
        {"tool_name": "Read", "input": {"file_path": "src/skills/references/target/ascendc/patterns/PATTERN_INDEX.md"}},
    ]
    audit = kba.audit_kb_load(brief, tool_uses)
    assert audit.coverage_pct == 100
