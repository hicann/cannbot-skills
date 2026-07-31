# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for V2 #5 live stdout — progress callback printer + line-buffer.

The orchestrator's UI value depends on stdout flushing live during the
17min worker spawn (not all at once at process exit). Two pieces:

  1. orchestrator.py reconfigure stdout/stderr to line-buffered at module load
  2. agent_dispatch._make_progress_printer prints terse one-liner per tool_use
     event (Bash, Read, Edit, ...) so user sees live phase activity

This test file covers (2). Piece (1) is verified by visual inspection on
real ops; sys.stdout.line_buffering is a process-level attribute and
mocking pytest's captured stdout would be brittle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import agent_dispatch as ad  # noqa: E402


@pytest.fixture
def cb():
    return getattr(ad, '_make_progress_printer')("aog-kernel-worker", 1)


def _bash_event(command, description=""):
    return {
        "type": "assistant",
        "message": {"content": [
            {"type": "tool_use", "name": "Bash",
             "input": {"command": command, "description": description}}
        ]},
    }


def _tool_event(tool_name, **inp):
    return {
        "type": "assistant",
        "message": {"content": [
            {"type": "tool_use", "name": tool_name, "input": inp}
        ]},
    }


def _text_event(text):
    return {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
    }


# ---------------------------------------------------------------------------
# Tool-specific line formats
# ---------------------------------------------------------------------------
def test_bash_uses_description_when_present(cb, capsys):
    cb(_bash_event("ls -la", "list workspace"))
    out = capsys.readouterr().out
    assert "Bash:" in out
    assert "list workspace" in out


def test_bash_falls_back_to_command(cb, capsys):
    cb(_bash_event("ls -la"))
    out = capsys.readouterr().out
    assert "Bash:" in out
    assert "ls -la" in out


def test_bash_truncates_long_command(cb, capsys):
    cb(_bash_event("x" * 500))
    out = capsys.readouterr().out
    # Output line should not be 500+ chars
    assert len(out.splitlines()[0]) < 200


def test_read_shows_short_path(cb, capsys):
    cb(_tool_event("Read", file_path="/abs/some/very/long/path/to/file.md"))
    out = capsys.readouterr().out
    assert "Read:" in out
    assert "to/file.md" in out  # last 2 components


def test_edit_shows_basename(cb, capsys):
    cb(_tool_event("Edit", file_path="/abs/path/kernel.h"))
    out = capsys.readouterr().out
    assert "Edit:" in out
    assert "kernel.h" in out


def test_write_shows_basename(cb, capsys):
    cb(_tool_event("Write", file_path="/abs/path/PROGRESS.md"))
    out = capsys.readouterr().out
    assert "Write:" in out
    assert "PROGRESS.md" in out


def test_grep_shows_pattern(cb, capsys):
    cb(_tool_event("Grep", pattern="OL-67", path="src/skills/references/"))
    out = capsys.readouterr().out
    assert "Grep:" in out
    assert "OL-67" in out


def test_skill_shows_skill_name(cb, capsys):
    cb(_tool_event("Skill", skill="aog-self-critic"))
    out = capsys.readouterr().out
    assert "Skill:" in out
    assert "aog-self-critic" in out


def test_unknown_tool_just_shows_name(cb, capsys):
    cb(_tool_event("WeirdTool", x=1))
    out = capsys.readouterr().out
    assert "WeirdTool" in out


# ---------------------------------------------------------------------------
# Text blocks
# ---------------------------------------------------------------------------
def test_text_first_line_printed(cb, capsys):
    cb(_text_event("Phase A: KB Manifest LOAD\nReading OL-67..."))
    out = capsys.readouterr().out
    assert "Phase A: KB Manifest LOAD" in out
    # Second line should NOT be printed (we only show first line)
    assert "Reading OL-67" not in out


def test_empty_text_silent(cb, capsys):
    cb(_text_event(""))
    cb(_text_event("\n\n   \n"))
    out = capsys.readouterr().out
    assert out == "" or out.strip() == ""


def test_long_text_truncated(cb, capsys):
    cb(_text_event("a" * 500))
    out = capsys.readouterr().out
    line = out.splitlines()[0] if out else ""
    assert len(line) < 200


# ---------------------------------------------------------------------------
# Non-assistant events ignored
# ---------------------------------------------------------------------------
def test_system_event_silent(cb, capsys):
    cb({"type": "system", "subtype": "init"})
    assert capsys.readouterr().out == ""


def test_user_event_silent(cb, capsys):
    cb({"type": "user", "message": {"content": [{"type": "tool_result"}]}})
    assert capsys.readouterr().out == ""


def test_result_event_silent(cb, capsys):
    """The result envelope is not progress; orchestrator parses + reports it
    separately.
    """
    cb({"type": "result", "subtype": "success"})
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Prefix uses agent_type + spawn_index
# ---------------------------------------------------------------------------
def test_prefix_includes_agent_type_and_index(capsys):
    cb = getattr(ad, '_make_progress_printer')("aog-precision-probe", 3)
    cb(_bash_event("ls"))
    out = capsys.readouterr().out
    assert "aog-precision-probe-3" in out


# ---------------------------------------------------------------------------
# Mixed-content blocks (text + tool_use in same message)
# ---------------------------------------------------------------------------
def test_mixed_content_both_printed(cb, capsys):
    event = {
        "type": "assistant",
        "message": {"content": [
            {"type": "text", "text": "I will run ls now."},
            {"type": "tool_use", "name": "Bash",
             "input": {"command": "ls", "description": "list dir"}},
        ]},
    }
    cb(event)
    out = capsys.readouterr().out
    assert "I will run ls now." in out
    assert "Bash:" in out
    assert "list dir" in out
