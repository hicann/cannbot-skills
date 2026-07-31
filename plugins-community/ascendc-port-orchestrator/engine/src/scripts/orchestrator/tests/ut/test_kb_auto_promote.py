# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for kb_auto_promote — Mode 5 auto-promotion pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.scripts.orchestrator import kb_auto_promote as kap


def _setup_kb(tmp_path: Path) -> Path:
    """Build minimal KB tree under tmp_path."""
    kb = tmp_path / "src" / "skills" / "references"
    (kb / "patterns" / "unverified").mkdir(parents=True)
    (kb / "patterns" / "PATTERN_INDEX.md").write_text("# Pattern Index\n\n")
    (kb / "OPERATIONAL_KNOWLEDGE.md").write_text("# OL\n\n")
    (kb / "ERROR_CORRECTIONS.md").write_text("# EC\n\n")
    (kb / "PLATFORM_BUGS.md").write_text("# PB\n\n")
    return kb


def _add_candidate(kb: Path, candidate_id: str, title: str, body: str,
                  metadata: dict = None) -> None:
    cands_md = kb / "patterns" / "unverified" / "candidates.md"
    block = f"\n## {candidate_id}: {title}\n"
    for k, v in (metadata or {}).items():
        block += f"`{k}: {v}`\n"
    block += f"\n{body}\n"
    if cands_md.exists():
        cands_md.write_text(cands_md.read_text() + block)
    else:
        cands_md.write_text(block)


def _add_pending(kb: Path, run_id: str, candidate_id: str, op: str = "test_op") -> None:
    marker = kb / "patterns" / "unverified" / f".kb_promotion_pending-{run_id}-{candidate_id}"
    marker.write_text(json.dumps({
        "run_id": run_id, "op": op, "candidate_id": candidate_id, "ts": 1,
    }))


def test_scan_no_markers(tmp_path):
    kb = _setup_kb(tmp_path)
    markers = kap.scan_pending_markers(kb)
    assert markers == []


def test_scan_finds_pending(tmp_path):
    kb = _setup_kb(tmp_path)
    _add_pending(kb, "run1", "CAND-FA1")
    _add_pending(kb, "run1", "CAND-FA2")
    markers = kap.scan_pending_markers(kb)
    assert len(markers) == 2


def test_load_candidate_basic(tmp_path):
    kb = _setup_kb(tmp_path)
    _add_candidate(kb, "CAND-FOO1", "Generic principle for fused ops",
                   "**Trigger**: condition X.\n\n**Recommendation**: do Y.",
                   metadata={"applies_to": "op_class=fused_multi_stage"})
    cand = kap.load_candidate(kb / "patterns" / "unverified" / "candidates.md", "CAND-FOO1")
    assert cand is not None
    assert cand.candidate_id == "CAND-FOO1"
    assert "Generic principle" in cand.title
    assert "op_class=fused_multi_stage" in cand.metadata.get("applies_to", "")


def test_c36_passes_with_op_class(tmp_path):
    cand = kap.Candidate(
        candidate_id="CAND-FOO", title="Generic ping-pong slot rotation",
        body="...", metadata={"applies_to": "op_class=multi_stage_pipeline"}
    )
    ok, reason = kap.check_c36_generalize(cand)
    assert ok, reason


def test_c36_fails_op_name_in_title(tmp_path):
    cand = kap.Candidate(
        candidate_id="CAND-FA5",
        title="3_FusionAttention multi-output workspace contract",
        body="...", metadata={}
    )
    ok, reason = kap.check_c36_generalize(cand)
    assert not ok
    assert "C36" in reason


def test_n_gram_overlap_identical_high(tmp_path):
    text = "the quick brown fox jumps over the lazy dog and runs away fast"
    overlap = kap.n_gram_overlap_ratio(text, text)
    assert overlap == 1.0


def test_n_gram_overlap_disjoint_zero(tmp_path):
    a = "the quick brown fox jumps over the lazy dog and runs"
    b = "completely unrelated python programming language tutorial book chapter"
    overlap = kap.n_gram_overlap_ratio(a, b)
    assert overlap < 0.05


def test_find_duplicate_detects_high_overlap(tmp_path):
    kb = _setup_kb(tmp_path)
    canonical = kb / "OPERATIONAL_KNOWLEDGE.md"
    canonical.write_text("# OL\n\n## OL-99: existing principle\n\n"
                         "the quick brown fox jumps over the lazy dog repeatedly with vigor.\n")
    cand = kap.Candidate(
        candidate_id="CAND-X", title="newish",
        body="the quick brown fox jumps over the lazy dog repeatedly with vigor.",
        metadata={}
    )
    dup = kap.find_duplicate(cand, kb, threshold=0.20)
    assert dup is not None
    assert dup[0] == "OL-99"


def test_find_duplicate_returns_none_when_low_overlap(tmp_path):
    kb = _setup_kb(tmp_path)
    cand = kap.Candidate(
        candidate_id="CAND-Y", title="entirely new principle for memory layout",
        body="completely novel content about memory layout strategies.",
        metadata={}
    )
    dup = kap.find_duplicate(cand, kb)
    assert dup is None


def test_allocate_next_id_empty(tmp_path):
    f = tmp_path / "PATTERN_INDEX.md"
    f.write_text("# PI\n\n")
    new_id = kap.allocate_next_id(f, "P-P")
    assert new_id == "P-P1"


def test_allocate_next_id_increments(tmp_path):
    f = tmp_path / "PATTERN_INDEX.md"
    f.write_text("# PI\n\n## P-P1: first\n\n## P-P3: third\n\n## P-P88: legacy\n")
    new_id = kap.allocate_next_id(f, "P-P")
    assert new_id == "P-P89"


def test_promote_candidate_dry_run(tmp_path):
    kb = _setup_kb(tmp_path)
    _add_candidate(kb, "CAND-X", "Some principle", "body content")
    cand = kap.load_candidate(kb / "patterns" / "unverified" / "candidates.md", "CAND-X")
    target = kb / "patterns" / "PATTERN_INDEX.md"
    audit = kap.promote_candidate(cand, kb, kb / "patterns" / "unverified" / "candidates.md",
                                  target, "P-P50", dry_run=True)
    assert audit["dry_run"] is True
    assert audit["would_new_id"] == "P-P50"


def test_promote_candidate_actual(tmp_path):
    kb = _setup_kb(tmp_path)
    _add_candidate(kb, "CAND-X", "Some principle", "body content for the principle")
    cand = kap.load_candidate(kb / "patterns" / "unverified" / "candidates.md", "CAND-X")
    target = kb / "patterns" / "PATTERN_INDEX.md"
    cands_md = kb / "patterns" / "unverified" / "candidates.md"
    kap.promote_candidate(cand, kb, cands_md, target, "P-P50", dry_run=False)
    target_text = target.read_text()
    assert "## P-P50:" in target_text
    assert "Some principle" in target_text
    cands_text = cands_md.read_text()
    assert "CAND-X" not in cands_text


def test_run_auto_promote_no_codex_promotes_clean_candidate(tmp_path):
    kb = _setup_kb(tmp_path)
    # Clean candidate with op_class metadata, no dups
    _add_candidate(kb, "CAND-NEW1", "Principle for some pattern",
                   "Detailed body about the pattern's mechanics and triggers.",
                   metadata={"applies_to": "op_class=fused_multi_stage"})
    _add_pending(kb, "run42", "CAND-NEW1")

    rpt = kap.run_auto_promote(kb, skip_codex=True)
    assert rpt.markers_processed == 1
    assert len(rpt.promoted) == 1
    assert rpt.promoted[0].candidate_id == "CAND-NEW1"
    assert rpt.promoted[0].target_id == "P-P1"


def test_run_auto_promote_blocks_op_name_titled(tmp_path):
    kb = _setup_kb(tmp_path)
    # Bad candidate: op-name in title, no applies_to
    _add_candidate(kb, "CAND-BAD1", "3_FusionAttention specific contract",
                   "body without op_class metadata")
    _add_pending(kb, "run42", "CAND-BAD1")

    rpt = kap.run_auto_promote(kb, skip_codex=True)
    assert len(rpt.blocked) >= 1
    assert any("C36" in r.reason for r in rpt.blocked)


def test_p0acv_blocked_marker_unlinks_pending(tmp_path, monkeypatch):
    """P0acv: when codex returns REJECT (or NEEDS_REVISION / load fail),
    the .kb_promotion_pending-* marker MUST be removed so the next run
    doesn't re-process the same candidate (which would re-spawn codex
    indefinitely and hang pre-commit hooks).
    """
    kb = _setup_kb(tmp_path)
    _add_candidate(kb, "CAND-X", "Generic principle",
                   "Body content.",
                   metadata={"applies_to": "op_class=fused_multi_stage"})
    _add_pending(kb, "run42", "CAND-X")

    pending_dir = kb / "patterns" / "unverified"
    pending = list(pending_dir.glob(".kb_promotion_pending-*"))
    assert len(pending) == 1, "fixture should have 1 pending marker"

    monkeypatch.setattr(kap, "codex_review",
                        lambda c, **kw: ("reject", "codex says no"))

    rpt = kap.run_auto_promote(kb, skip_codex=False)
    assert len(rpt.blocked) == 1
    # Pending marker must be gone
    pending_after = list(pending_dir.glob(".kb_promotion_pending-*"))
    assert pending_after == [], (
        f"P0acv regression: pending marker leaked across block — "
        f"next run will re-process forever: {pending_after}"
    )
    # Blocked audit marker should now exist
    blocked = list(pending_dir.glob(".kb_promotion_blocked-*"))
    assert len(blocked) == 1


def test_p0acu_claude_fallback_approves_promotes(tmp_path, monkeypatch):
    """P0acu: codex unavailable → claude fallback APPROVE → promote with
    provenance stamp '[reviewer=claude_fallback]' in audit feedback.
    """
    kb = _setup_kb(tmp_path)
    _add_candidate(kb, "CAND-NEW1", "Some generic fused-op principle",
                   "Detailed body about the pattern.",
                   metadata={"applies_to": "op_class=fused_multi_stage"})
    _add_pending(kb, "run42", "CAND-NEW1")

    monkeypatch.setattr(kap, "codex_review",
                        lambda c, **kw: ("unavailable", "codex CLI not found"))
    monkeypatch.setattr(kap, "claude_fallback_review",
                        lambda c, **kw: ("claude_fallback_approve",
                                         "VERDICT: APPROVE — looks generic"))

    rpt = kap.run_auto_promote(kb, skip_codex=False)
    assert len(rpt.promoted) == 1
    # Audit field codex_review should be "approve" (normalized)
    assert rpt.promoted[0].codex_review == "approve"


def test_p0acu_claude_fallback_rejects_blocks(tmp_path, monkeypatch):
    """P0acu: codex unavailable → claude fallback REJECT → block."""
    kb = _setup_kb(tmp_path)
    _add_candidate(kb, "CAND-NEW2", "Some generic fused-op principle",
                   "Body with logical contradiction to existing KB.",
                   metadata={"applies_to": "op_class=fused_multi_stage"})
    _add_pending(kb, "run42", "CAND-NEW2")

    monkeypatch.setattr(kap, "codex_review",
                        lambda c, **kw: ("unavailable", "codex quota"))
    monkeypatch.setattr(kap, "claude_fallback_review",
                        lambda c, **kw: ("claude_fallback_reject",
                                         "VERDICT: REJECT — contradicts P-P42"))

    rpt = kap.run_auto_promote(kb, skip_codex=False)
    assert len(rpt.promoted) == 0
    assert len(rpt.blocked) == 1
    assert "REJECT" in rpt.blocked[0].reason


def test_p0acu_both_unavailable_hard_blocks(tmp_path, monkeypatch):
    """P0acu: when even claude fallback returns 'unavailable', hard-block per
    C40 fail-closed — no review = no promotion.
    """
    kb = _setup_kb(tmp_path)
    _add_candidate(kb, "CAND-NEW3", "Some principle",
                   "Body.", metadata={"applies_to": "op_class=foo"})
    _add_pending(kb, "run42", "CAND-NEW3")

    monkeypatch.setattr(kap, "codex_review",
                        lambda c, **kw: ("unavailable", "codex down"))
    monkeypatch.setattr(kap, "claude_fallback_review",
                        lambda c, **kw: ("unavailable", "claude transport down"))

    rpt = kap.run_auto_promote(kb, skip_codex=False)
    assert len(rpt.promoted) == 0
    assert len(rpt.blocked) == 1
    assert "both codex AND claude fallback unavailable" in rpt.blocked[0].reason


def test_p0act_codex_unavailable_blocks_when_no_fallback(tmp_path, monkeypatch):
    """P0act + P0acu: when codex returns 'unavailable' AND the claude fallback
    is also unavailable (no agent_transport / both down), the production path
    MUST block. C40 fail-closed.

    The original P0act behavior — "codex unavailable always blocks" — was
    relaxed by P0acu to "codex unavailable → claude fallback → block only
    if both fail". This test pins the remaining hard-block path.
    """
    kb = _setup_kb(tmp_path)
    _add_candidate(kb, "CAND-NEW1", "Some generic fused-op principle",
                   "Detailed body about the pattern's mechanics.",
                   metadata={"applies_to": "op_class=fused_multi_stage"})
    _add_pending(kb, "run42", "CAND-NEW1")

    monkeypatch.setattr(kap, "codex_review",
                        lambda c, **kw: ("unavailable", "codex CLI not found"))
    monkeypatch.setattr(kap, "claude_fallback_review",
                        lambda c, **kw: ("unavailable", "agent_transport unavailable"))

    rpt = kap.run_auto_promote(kb, skip_codex=False)
    assert rpt.markers_processed == 1
    assert len(rpt.promoted) == 0
    assert len(rpt.blocked) == 1
    assert "both codex AND claude fallback unavailable" in rpt.blocked[0].reason


def test_parse_codex_revision_extracts_title_and_body():
    fb = """NEEDS_REVISION
Reason text here.

REVISED_TITLE: Fixed-up title goes here
REVISED_BODY:
Body content
spans
multiple lines
"""
    result = kap.parse_codex_revision(fb)
    assert result is not None
    title, body = result
    assert title == "Fixed-up title goes here"
    assert "spans" in body
    assert "multiple lines" in body


def test_parse_codex_revision_returns_none_when_missing():
    fb = "NEEDS_REVISION\nReason: something is wrong (no revision attached)"
    assert kap.parse_codex_revision(fb) is None


def test_apply_revision_preserves_metadata():
    cand = kap.Candidate(
        candidate_id="CAND-X", title="old title", body="old body",
        metadata={"applies_to": "op_class=foo"}
    )
    revised = kap.apply_revision(cand, "new title", "new body content")
    assert revised.candidate_id == "CAND-X"
    assert revised.title == "new title"
    assert revised.body == "new body content"
    assert revised.metadata.get("applies_to") == "op_class=foo"  # preserved


def test_persist_revision_overwrites_block(tmp_path):
    cand_md = tmp_path / "candidates.md"
    cand_md.write_text(
        "# Candidates\n\n"
        "## CAND-X: original title\n"
        "`applies_to: op_class=test`\n\n"
        "Original body content.\n\n"
        "## CAND-Y: another\n\n"
        "Other body.\n"
    )
    revised = kap.Candidate(
        candidate_id="CAND-X", title="REVISED title",
        body="REVISED body — much more thoughtful.",
        metadata={"applies_to": "op_class=test"}
    )
    getattr(kap, '_persist_revision')(cand_md, revised)
    new_text = cand_md.read_text()
    assert "## CAND-X: REVISED title" in new_text
    assert "REVISED body" in new_text
    assert "Original body content." not in new_text
    assert "## CAND-Y: another" in new_text  # other block preserved
    assert "Other body." in new_text


def test_audit_report_renders(tmp_path):
    rpt = kap.PromotionBatchReport(markers_processed=3)
    rpt.promoted.append(kap.PromotionResult(
        candidate_id="CAND-A", outcome="promoted", target_id="P-P1",
        codex_review="approve"))
    rpt.merged.append(kap.PromotionResult(
        candidate_id="CAND-B", outcome="merged", target_id="OL-99"))
    rpt.blocked.append(kap.PromotionResult(
        candidate_id="CAND-C", outcome="blocked", reason="C36"))
    rpt.finished_ts = rpt.started_ts + 5
    md = kap.render_audit_report(rpt)
    assert "Markers processed: **3**" in md
    assert "CAND-A" in md
    assert "P-P1" in md
    assert "C36" in md
