# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Unit tests for op_taxonomy.lookup (v3 — reads op_classification.json).

Background: pre-2026-05-07 lookup() consulted a hardcoded `OP_TAGS` dict
keyed by benchmark name. That heuristic was retired (P0aak): lookup now
reads `workspace/<op>/op_classification.json` produced by the LLM-driven
`/aog-op-classify` skill in Phase O1.7. When no classification artifact is
present, lookup returns just `DEFAULT_KB_SECTIONS` and marks
`is_untagged_fallback=True`.

`OP_TAGS` and `TAG_KB_SECTIONS` are retained in the module for emergency
rollback only; tests below verify the current production behavior, not
the retired dict path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from briefs import op_taxonomy as ot  # noqa: E402


def test_lookup_without_workspace_returns_fallback():
    """No workspace → no classification possible → fallback with defaults."""
    t = ot.lookup("any_op")
    assert t.is_untagged_fallback is True
    assert t.tags == []
    # Still gets DEFAULT_KB_SECTIONS bookshelf
    for default_path in ot.DEFAULT_KB_SECTIONS:
        assert default_path in t.kb_sections


def test_lookup_with_workspace_but_no_classification_returns_fallback(tmp_path):
    """Workspace exists but no op_classification.json → fallback."""
    ws = tmp_path / "test_op"
    ws.mkdir()
    t = ot.lookup("test_op", workspace=ws)
    assert t.is_untagged_fallback is True
    assert t.tags == []
    assert "KB_INDEX.md" in t.kb_sections


def test_lookup_with_classification_returns_tags_and_paths(tmp_path):
    """When op_classification.json is present, lookup reads tags +
    kb_recommendations into the OpTaxonomy result.
    """
    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "op_classification.json").write_text(json.dumps({
        "op_class_tags": ["scatter-gather", "reduction"],
        "kb_recommendations": [
            {"path": "OPERATIONAL_KNOWLEDGE.md#OL-67"},
            {"path": "OPERATIONAL_KNOWLEDGE.md#OL-90"},
        ],
    }))
    t = ot.lookup("test_op", workspace=ws)
    assert t.is_untagged_fallback is False
    assert "scatter-gather" in t.tags
    assert "reduction" in t.tags
    assert "OPERATIONAL_KNOWLEDGE.md#OL-67" in t.kb_sections
    assert "OPERATIONAL_KNOWLEDGE.md#OL-90" in t.kb_sections


def test_lookup_dedupes_kb_sections(tmp_path):
    """If classification recommends a path that's already in
    DEFAULT_KB_SECTIONS, the merged list contains it once.
    """
    ws = tmp_path / "test_op"
    ws.mkdir()
    # Pick a default-set path to verify dedup
    duplicate_default = ot.DEFAULT_KB_SECTIONS[0]
    (ws / "op_classification.json").write_text(json.dumps({
        "op_class_tags": ["foo"],
        "kb_recommendations": [{"path": duplicate_default}],
    }))
    t = ot.lookup("test_op", workspace=ws)
    occurrences = [s for s in t.kb_sections if s == duplicate_default]
    assert len(occurrences) == 1


def test_lookup_corrupt_classification_falls_back(tmp_path):
    """A malformed op_classification.json must not crash; treat as missing
    and fall back to defaults.
    """
    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "op_classification.json").write_text("{ this is not valid json }}}")
    t = ot.lookup("test_op", workspace=ws)
    assert t.is_untagged_fallback is True
    assert t.tags == []


def test_default_kb_sections_are_valid_paths():
    """Every entry in DEFAULT_KB_SECTIONS must look like a path under
    src/skills/references/ (no extraneous formatting).
    """
    for s in ot.DEFAULT_KB_SECTIONS:
        assert "/" in s or s.endswith(".md"), (
            f"DEFAULT_KB_SECTIONS entry {s!r} does not look like a path"
        )


def test_tag_kb_sections_paths_have_known_prefixes():
    """Legacy TAG_KB_SECTIONS paths point to real KB files. Sanity check on
    the retained dict (used only for emergency rollback).
    """
    allowed_prefixes = (
        "OPERATIONAL_KNOWLEDGE.md",
        "ERROR_CORRECTIONS.md",
        "PLATFORM_BUGS.md",
        "patterns/",
        "hardware/",
        "KB_INDEX.md",
        # W8 (2026-05-12, ROADMAP §1.5): new KB directory for arch22→arch35 port
        # artifact layout reference (used by `a3_to_a5_port` tag).
        "ops_nn_layout/",
    )
    for tag, sections in ot.TAG_KB_SECTIONS.items():
        for s in sections:
            path_part = s.split("#")[0]
            assert any(
                path_part == p or path_part.startswith(p)
                for p in allowed_prefixes
            ), f"tag {tag!r} references unknown KB path {s!r}"


def test_coverage_report_reflects_legacy_dict_state():
    """coverage_report runs against the retained OP_TAGS / TAG_KB_SECTIONS
    dicts (legacy diagnostic). Asserts on actual current sizes — when the
    dict shrinks (more deprecation), this test surfaces the change.
    """
    rep = ot.coverage_report()
    assert rep["n_ops_in_taxonomy"] == len(ot.OP_TAGS)
    assert rep["n_tags_defined"] == len(ot.TAG_KB_SECTIONS)
    assert "tag_usage" in rep
    assert "unused_tags" in rep
    assert "ops_with_zero_tags" in rep


# ---------------------------------------------------------------------------
# P0abj (2026-05-08): target-aware hardware-spec dispatch
# ---------------------------------------------------------------------------
def test_default_kb_sections_a5_includes_ascend950pr():
    """A5 target → manifest includes ascend950pr.md (back-compat default)."""
    sections = ot.default_kb_sections("a5")
    assert "hardware/target/ascend950pr.md" in sections
    assert "hardware/target/ascend910b.md" not in sections
    assert "hardware/target/ascend910c.md" not in sections


def test_default_kb_sections_a3_includes_ascend910c():
    """A3 target → manifest includes ascend910c.md (V220 single-die)."""
    sections = ot.default_kb_sections("a3")
    assert "hardware/target/ascend910c.md" in sections
    assert "hardware/target/ascend950pr.md" not in sections
    assert "hardware/target/ascend910b.md" not in sections


def test_default_kb_sections_a2_includes_ascend910b():
    """A2 target → manifest includes ascend910b.md (V220 single-die)."""
    sections = ot.default_kb_sections("a2")
    assert "hardware/target/ascend910b.md" in sections
    assert "hardware/target/ascend950pr.md" not in sections


def test_default_kb_sections_ds_suffix_normalized():
    """DS-env target a3-ds normalizes to a3 → loads ascend910c.md (DS hw
    isolation suffix doesn't change hardware family).
    """
    a3_ds = ot.default_kb_sections("a3-ds")
    a3 = ot.default_kb_sections("a3")
    assert a3_ds == a3, (
        f"a3-ds should resolve to same sections as a3; got "
        f"a3-ds={a3_ds} vs a3={a3}"
    )
    assert "hardware/target/ascend910c.md" in a3_ds


def test_default_kb_sections_unknown_target_falls_back_to_a5():
    """Unknown target → fall back to A5 specs (warn-don't-error policy
    keeps op-gen on a brand-new chip workable).
    """
    sections = ot.default_kb_sections("zz_future_chip")
    assert "hardware/target/ascend950pr.md" in sections


def test_default_kb_sections_case_insensitive():
    """Target case shouldn't matter — A3, a3, A3-DS all map identically."""
    s1 = ot.default_kb_sections("A3")
    s2 = ot.default_kb_sections("a3")
    s3 = ot.default_kb_sections("A3-DS")
    assert s1 == s2 == s3


def test_default_kb_sections_constant_back_compat():
    """The historical `DEFAULT_KB_SECTIONS` constant should still exist
    and equal default_kb_sections('a5') — back-compat for any caller
    that imports it directly.
    """
    assert ot.DEFAULT_KB_SECTIONS == ot.default_kb_sections("a5")


def test_lookup_target_dispatch(tmp_path):
    """lookup() honors target arg — A3 op-gen gets ascend910c.md, NOT
    ascend950pr.md (the regression case the fix targets).
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    res_a5 = ot.lookup("test_op", workspace=workspace, target="a5")
    res_a3 = ot.lookup("test_op", workspace=workspace, target="a3")
    assert "hardware/target/ascend950pr.md" in res_a5.kb_sections
    assert "hardware/target/ascend910c.md" in res_a3.kb_sections
    # a3 must NOT include the A5 spec
    assert "hardware/target/ascend950pr.md" not in res_a3.kb_sections


# ---------------------------------------------------------------------------
# P0abj followup (2026-05-09): fail-fast on missing KB-manifest files
# ---------------------------------------------------------------------------
def test_validate_manifest_paths_passes_when_all_present(tmp_path):
    """All paths exist on disk → no exception raised."""
    refs = tmp_path / "refs"
    (refs / "hardware" / "target").mkdir(parents=True)
    (refs / "hardware" / "target" / "ascend910c.md").write_text("a3 hw")
    (refs / "KB_INDEX.md").write_text("idx")
    sections = ["KB_INDEX.md", "hardware/target/ascend910c.md"]
    # Must not raise
    ot.validate_manifest_paths(sections, references_root=refs)


def test_validate_manifest_paths_raises_when_missing(tmp_path):
    """Missing file → KBManifestMissingError with actionable message."""
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "KB_INDEX.md").write_text("idx")
    sections = ["KB_INDEX.md", "hardware/target/ascend910c.md"]
    with pytest.raises(ot.KBManifestMissingError) as exc:
        ot.validate_manifest_paths(sections, references_root=refs)
    msg = str(exc.value)
    assert "ascend910c.md" in msg
    assert "MISSING" in msg
    assert "Resolution" in msg
    assert str(refs) in msg


def test_validate_manifest_paths_strips_anchors(tmp_path):
    """`OPERATIONAL_KNOWLEDGE.md#OL-103` anchor is intra-doc; should resolve
    to OPERATIONAL_KNOWLEDGE.md (file existence) not the anchor.

    P88 KB reorg: legacy bare `OPERATIONAL_KNOWLEDGE.md` resolves to
    `target/ascendc/OPERATIONAL_KNOWLEDGE.md`. Test fixture mirrors
    the new layout.
    """
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "target" / "ascendc").mkdir(parents=True)
    (refs / "target" / "ascendc" / "OPERATIONAL_KNOWLEDGE.md").write_text("ok")
    sections = ["OPERATIONAL_KNOWLEDGE.md#OL-103"]
    ot.validate_manifest_paths(sections, references_root=refs)


def test_validate_real_a5_manifest_passes_in_repo():
    """Sanity: real ascend950pr.md + base sections exist in the actual
    repo. Catches the case where someone deletes a file but doesn't
    update DEFAULT_KB_SECTIONS.
    """
    sections = ot.default_kb_sections("a5")
    ot.validate_manifest_paths(sections)  # uses real _REFERENCES_ROOT


def test_validate_real_a3_manifest_passes_in_repo():
    """Same for A3 — ascend910c.md must exist in repo."""
    sections = ot.default_kb_sections("a3")
    ot.validate_manifest_paths(sections)


def test_validate_real_a2_manifest_passes_in_repo():
    """Same for A2 — ascend910b.md must exist."""
    sections = ot.default_kb_sections("a2")
    ot.validate_manifest_paths(sections)


def test_kb_manifest_block_fails_fast_on_missing_hw_md(tmp_path, monkeypatch):
    """User scenario: TARGET=a3 but ascend910c.md was removed → user
    expects fail-fast at brief-construction with clear error, not
    agent-side 404 mid-spawn.
    """
    sys.path.insert(0, str(_HERE.parent.parent))
    import briefs._common as bc  # noqa: E402

    refs = tmp_path / "refs"
    (refs / "hardware" / "target").mkdir(parents=True)
    # Stub everything EXCEPT ascend910c.md (the missing file)
    for f in getattr(ot, '_DEFAULT_KB_SECTIONS_BASE'):
        full = refs / f
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text("stub")
    # Don't create ascend910c.md
    monkeypatch.setattr(ot, "_REFERENCES_ROOT", refs)

    with pytest.raises(ot.KBManifestMissingError) as exc:
        bc.kb_manifest_block("test_op", workspace=tmp_path, target="a3")
    assert "ascend910c.md" in str(exc.value)
