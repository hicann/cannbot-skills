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
present, lookup returns just `default_kb_sections(target)` and marks
`is_untagged_fallback=True`.

OKF-only 迁移（2026-08）：legacy manifest 渲染链路（路径重写 / 落盘校验 /
DEFAULT_KB_SECTIONS 别名）已删除；b-tier 唯一路径是 OKF 检索，lookup 的
tags 产出作为 OKF 查询词来源保留。

`OP_TAGS` and `TAG_KB_SECTIONS` are retained in the module for emergency
rollback only; tests below verify the current production behavior, not
the retired dict path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from briefs import op_taxonomy as ot  # noqa: E402


def test_lookup_without_workspace_returns_fallback():
    """No workspace → no classification possible → fallback with defaults."""
    t = ot.lookup("any_op")
    assert t.is_untagged_fallback is True
    assert t.tags == []
    # Still gets the default_kb_sections bookshelf
    for default_path in ot.default_kb_sections("a5"):
        assert default_path in t.kb_sections


def test_lookup_with_workspace_but_no_classification_returns_fallback(tmp_path):
    """Workspace exists but no op_classification.json → fallback."""
    ws = tmp_path / "test_op"
    ws.mkdir()
    t = ot.lookup("test_op", workspace=ws)
    assert t.is_untagged_fallback is True
    assert t.tags == []
    assert "shared/ALWAYS_LOADED_RULES.md" in t.kb_sections


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
    default_kb_sections, the merged list contains it once.
    """
    ws = tmp_path / "test_op"
    ws.mkdir()
    # Pick a default-set path to verify dedup
    duplicate_default = ot.default_kb_sections("a5")[0]
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
    """Every entry in default_kb_sections must look like a path under
    kb/ (no extraneous formatting).
    """
    for s in ot.default_kb_sections("a5"):
        assert "/" in s or s.endswith(".md"), (
            f"default_kb_sections entry {s!r} does not look like a path"
        )


def test_default_kb_sections_okf_only_no_legacy_entries():
    """OKF-only 迁移 pin：默认集合不再含 legacy 导航索引与 target 目录条目
    （知识指针由 OKF 检索产出，默认集合只剩来源无关的纪律文档 + 硬件规格）。
    """
    for target in ("a5", "a3", "a2"):
        for s in ot.default_kb_sections(target):
            assert "KB_INDEX" not in s
            assert not s.startswith("target/"), (
                f"default_kb_sections({target!r}) still carries legacy entry {s!r}"
            )


# ---------------------------------------------------------------------------
# P0abj (2026-05-08): target-aware hardware-spec dispatch
# ---------------------------------------------------------------------------
def test_default_kb_sections_a5_includes_ascend950pr():
    """A5 target → manifest includes ascend950pr.md (back-compat default)."""
    sections = ot.default_kb_sections("a5")
    assert "okf/runbooks/hardware/target-ascend950pr.md" in sections
    assert "okf/runbooks/hardware/target-ascend910b.md" not in sections
    assert "okf/runbooks/hardware/target-ascend910c.md" not in sections


def test_default_kb_sections_a3_includes_ascend910c():
    """A3 target → manifest includes ascend910c.md (V220 single-die)."""
    sections = ot.default_kb_sections("a3")
    assert "okf/runbooks/hardware/target-ascend910c.md" in sections
    assert "okf/runbooks/hardware/target-ascend950pr.md" not in sections
    assert "okf/runbooks/hardware/target-ascend910b.md" not in sections


def test_default_kb_sections_a2_includes_ascend910b():
    """A2 target → manifest includes ascend910b.md (V220 single-die)."""
    sections = ot.default_kb_sections("a2")
    assert "okf/runbooks/hardware/target-ascend910b.md" in sections
    assert "okf/runbooks/hardware/target-ascend950pr.md" not in sections


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
    assert "okf/runbooks/hardware/target-ascend910c.md" in a3_ds


def test_default_kb_sections_unknown_target_falls_back_to_a5():
    """Unknown target → fall back to A5 specs (warn-don't-error policy
    keeps op-gen on a brand-new chip workable).
    """
    sections = ot.default_kb_sections("zz_future_chip")
    assert "okf/runbooks/hardware/target-ascend950pr.md" in sections


def test_default_kb_sections_case_insensitive():
    """Target case shouldn't matter — A3, a3, A3-DS all map identically."""
    s1 = ot.default_kb_sections("A3")
    s2 = ot.default_kb_sections("a3")
    s3 = ot.default_kb_sections("A3-DS")
    assert s1 == s2 == s3


def test_lookup_target_dispatch(tmp_path):
    """lookup() honors target arg — A3 op-gen gets ascend910c.md, NOT
    ascend950pr.md (the regression case the fix targets).
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    res_a5 = ot.lookup("test_op", workspace=workspace, target="a5")
    res_a3 = ot.lookup("test_op", workspace=workspace, target="a3")
    assert "okf/runbooks/hardware/target-ascend950pr.md" in res_a5.kb_sections
    assert "okf/runbooks/hardware/target-ascend910c.md" in res_a3.kb_sections
    # a3 must NOT include the A5 spec
    assert "okf/runbooks/hardware/target-ascend950pr.md" not in res_a3.kb_sections


