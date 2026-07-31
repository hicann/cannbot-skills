# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Q1 slim self-critic prompt — INLINE catalog extraction tests.

Validates the Q1 optimization (2026-05-21, main agent vote) that cuts
per-fire token cost 64-86% by extracting just the subset items into the
prompt instead of invoking the /aog-self-critic skill (which auto-loads
the full 1485-LOC SKILL.md).

Per-trigger expected reduction (measured against real SKILL.md):
- pre_phase_o4_first_spawn: 12 items → ~64% reduction
- post_iter_cap_warning: 4 items → ~86% reduction
- pre_finalize: 6 items → ~76% reduction
- pre_commit: 5 items → ~80% reduction
Average: ~76%.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "scripts" / "orchestrator"))

import critic_invoke  # noqa: E402


# ---- Catalog extraction primitives ----


def test_extract_preamble_truncates_at_first_item():
    """preamble must end BEFORE the first ### Cn / ### Tn heading."""
    md = (
        "---\nname: aog-self-critic\n---\n\n"
        "# /aog-self-critic\n\n"
        "## When to invoke\n\nSome prose.\n\n"
        "## Checks\n\n"
        "### C1: First item\n\nbody1\n\n"
        "### C2: Second item\n\nbody2\n"
    )
    preamble = getattr(critic_invoke, '_extract_preamble')(md)
    assert "C1" not in preamble, f"preamble bled into C1: {preamble[-100:]}"
    assert "When to invoke" in preamble
    assert "Some prose" in preamble


def test_extract_items_returns_only_requested():
    """Subset filter — extract C1 and C3 but not C2."""
    md = (
        "### C1: First\n\nbody1\n\n"
        "### C2: Second\n\nbody2\n\n"
        "### C3: Third\n\nbody3\n"
    )
    items = getattr(critic_invoke, '_extract_items')(md, ["C1", "C3"])
    assert set(items.keys()) == {"C1", "C3"}
    assert "body1" in items["C1"]
    assert "body3" in items["C3"]
    assert "body2" not in items.get("C1", "")
    assert "body2" not in items.get("C3", "")


def test_extract_items_handles_compound_ids():
    """C-INFRA-RETRY-WITHOUT-CAP-style IDs must be extractable."""
    md = (
        "### C13: Numeric\n\nn-body\n\n"
        "### C-INFRA-RETRY-WITHOUT-CAP: Compound\n\nc-body\n\n"
        "### C14: Next\n\n14-body\n"
    )
    items = getattr(critic_invoke, '_extract_items')(md, ["C-INFRA-RETRY-WITHOUT-CAP"])
    assert "C-INFRA-RETRY-WITHOUT-CAP" in items
    assert "c-body" in items["C-INFRA-RETRY-WITHOUT-CAP"]


def test_extract_items_terminates_at_h2():
    """Last item's body should end at next ## section (Usage etc), not run to EOF."""
    md = (
        "### C1: Final item\n\nbody1\nmore-body1\n\n"
        "## Usage\n\nUsage prose\n"
    )
    items = getattr(critic_invoke, '_extract_items')(md, ["C1"])
    assert "body1" in items["C1"]
    assert "more-body1" in items["C1"]
    assert "Usage prose" not in items["C1"], "item bled into ## Usage"


def test_extract_items_silently_omits_missing():
    """If subset asks for an item not in SKILL.md, omit silently (caller decides)."""
    md = "### C1: Real\n\nbody1\n"
    items = getattr(critic_invoke, '_extract_items')(md, ["C1", "C99"])
    assert set(items.keys()) == {"C1"}


# ---- Slim catalog assembly ----


def test_build_slim_catalog_ascendc_real_skill_md():
    """End-to-end against the real SKILL.md. Validates the extraction
    pipeline works against the production catalog format.
    """
    subset = ["C13", "C14", "C18", "C23", "C26", "C30"]  # pre_finalize
    slim = getattr(critic_invoke, '_build_slim_catalog')("ascendc", subset)
    assert slim is not None
    # Each requested item header should appear
    for item_id in subset:
        assert f"### {item_id}:" in slim, f"{item_id} missing from slim catalog"
    # Items NOT requested should NOT appear as headers (substring match is
    # too aggressive — items mention other items in body)
    for absent_id in ["C2", "C5", "C11", "C19", "C20", "C29"]:
        assert f"### {absent_id}:" not in slim, (
            f"{absent_id} bled into slim catalog (extraction lost item-scope)"
        )


def test_build_slim_catalog_ascendc_pre_finalize_size_reduction():
    """Measured reduction: pre_finalize subset should be ~76% smaller than
    full SKILL.md.
    """
    skill_md = getattr(critic_invoke, '_load_skill_md')("ascendc")
    if skill_md is None:
        pytest.skip("ascendc SKILL.md not readable in test env")
    full_size = len(skill_md)
    subset = ["C13", "C14", "C18", "C23", "C26", "C30"]
    slim = getattr(critic_invoke, '_build_slim_catalog')("ascendc", subset)
    assert slim is not None
    slim_size = len(slim)
    reduction = (full_size - slim_size) / full_size
    assert reduction >= 0.50, (
        f"Expected ≥50% reduction for pre_finalize; got {reduction:.1%} "
        f"(full={full_size}, slim={slim_size})"
    )


def test_build_slim_catalog_unknown_backend_returns_none():
    """Unknown backend → no SKILL.md → return None → caller falls back."""
    slim = getattr(critic_invoke, '_build_slim_catalog')("nonexistent", ["X1"])
    assert slim is None


def test_build_slim_catalog_empty_subset_returns_none():
    """If subset is empty (unknown trigger), bail out and let caller use
    the full-skill fallback path.
    """
    slim = getattr(critic_invoke, '_build_slim_catalog')("ascendc", [])
    assert slim is None


def test_build_slim_catalog_no_items_match_returns_none():
    """If subset items don't exist in SKILL.md, bail to fallback."""
    slim = getattr(critic_invoke, '_build_slim_catalog')("ascendc", ["Z99", "Z100"])
    assert slim is None


# ---- fire_critic prompt-shape end-to-end ----


def _capture_prompt(monkeypatch, tmp_path: Path):
    """Helper: monkeypatch subprocess.run to capture the prompt arg."""
    captured: dict[str, str] = {}

    def fake_run(cmd, **kwargs):
        # cmd format: ["claude", "--print", "--output-format", "json",
        #              "--permission-mode", "bypassPermissions", prompt]
        captured["prompt"] = cmd[-1]
        result = MagicMock()
        result.returncode = 0
        result.stdout = '{"type": "result", "subtype": "success", "result": "stub"}'
        result.stderr = ""
        return result

    monkeypatch.setattr(critic_invoke.subprocess, "run", fake_run)

    # Stub state_executor.snapshot
    class _FakeSnap:
        op = "test_op"
        current_state = "await_worker"
        iter_counts = {"worker": 0}
        iter_caps = {"worker": 6}
        last_handoff = ""
    monkeypatch.setattr(critic_invoke.state_executor, "snapshot",
                        lambda ws: _FakeSnap())

    workspace = tmp_path / "test_op__backward"
    workspace.mkdir()
    (workspace / ".opgen_state.json").write_text(
        '{"opgen_mode": "backward", "backend": "ascendc"}'
    )
    return workspace, captured


def test_fire_critic_uses_inline_catalog_for_known_trigger(monkeypatch, tmp_path):
    """For a known trigger + backend with a working SKILL.md, the prompt
    must NOT contain '/aog-self-critic' invocation (which would auto-load
    full SKILL.md) and MUST contain the inline catalog header marker.
    """
    workspace, captured = _capture_prompt(monkeypatch, tmp_path)

    # The community plugin ships the AscendC critic catalog.
    critic_invoke.fire_critic(workspace, "pre_finalize", backend="ascendc")

    prompt = captured["prompt"]
    # Should NOT invoke the skill (that would defeat the optimization)
    assert "Invoke the /aog-self-critic" not in prompt
    # Should have the slim-catalog markers
    assert "=== CATALOG (slim subset" in prompt
    assert "do NOT load any external skill file" in prompt
    # Should have the slim items inlined as headers
    assert "### T" in prompt or "### C" in prompt


def test_fire_critic_unknown_trigger_falls_back(monkeypatch, tmp_path):
    """Unknown trigger → no subset → fall back to /skill invocation. Validates
    backward-compat for trigger names not in the subset map.
    """
    workspace, captured = _capture_prompt(monkeypatch, tmp_path)

    # Add an unknown trigger to bypass the validate; or use a real trigger
    # but stub _get_subset_for to return empty.
    monkeypatch.setattr(critic_invoke, "_get_subset_for",
                        lambda trigger, backend="ascendc": [])

    critic_invoke.fire_critic(workspace, "pre_finalize", backend="ascendc")

    prompt = captured["prompt"]
    # Empty subset → fallback path 2 (full /skill invocation)
    assert "Invoke the /aog-self-critic" in prompt
    assert "=== CATALOG (slim subset" not in prompt


def test_fire_critic_prompt_token_reduction(monkeypatch, tmp_path):
    """Measured: the inline-catalog prompt should be substantially smaller
    than the equivalent /skill-invocation prompt would be PLUS the SKILL.md
    that the LLM would auto-load.

    Direct comparison: prompt length INLINE vs the full SKILL.md size.
    """
    workspace, captured = _capture_prompt(monkeypatch, tmp_path)

    critic_invoke.fire_critic(workspace, "pre_finalize", backend="ascendc")

    inline_prompt = captured["prompt"]
    skill_md = getattr(critic_invoke, '_load_skill_md')("ascendc")
    if skill_md is None:
        pytest.skip("SKILL.md not in test env")

    # inline prompt should be smaller than full SKILL.md
    # (smaller AND covers the critical preamble + 6 items, vs SKILL.md with all 46 items)
    assert len(inline_prompt) < len(skill_md), (
        f"inline prompt ({len(inline_prompt)} chars) not smaller than full "
        f"SKILL.md ({len(skill_md)} chars) — optimization didn't fire"
    )
