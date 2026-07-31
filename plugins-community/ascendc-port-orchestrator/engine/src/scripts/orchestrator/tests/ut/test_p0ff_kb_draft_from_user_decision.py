# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P0ff (2026-05-23, owner directive 20:48Z + independent review/DS endorsement):
user_decision.md → kb_draft_from_user_decision.md auto-extraction.

The customer-side reproducibility gap: user_decision.md is session-state only,
doesn't ship in fresh customer install. Strategic insight there evaporates
post-session. Fix: orchestrator auto-extracts the directive content into
kb_draft_from_user_decision.md, which aog-knowledge-maintain Mode 1 reads
and promotes to canonical KB. Future customer cold-start of similar op
inherits the insight via worker brief's kb_manifest_block.

Test coverage:
  - structured mode: user_decision.md has `kb_distillation:` YAML block
  - heuristic mode: only prose `reason:`, classified by keyword
  - idempotency: re-running extraction doesn't double-append
  - trivial decision (no strategic content): no extraction
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from orchestrator import _extract_kb_draft_from_user_decision  # noqa: E402


def _seed_workspace(tmp_path: Path, user_decision_content: str) -> Path:
    ws = tmp_path / "ws_op"
    ws.mkdir(parents=True)
    (ws / "user_decision.md").write_text(user_decision_content)
    return ws


def test_structured_kb_distillation_block_extracted(tmp_path: Path) -> None:
    """When user_decision.md has structured kb_distillation: block, extraction
    captures the structured fields verbatim.
    """
    content = """next_state: await_worker
reason: |
  researcher intervention on port_a3_to_a5 paradigm.

kb_distillation:
  rule: "port_a3_to_a5 mode does NOT depend on aclnn dispatcher registration"
  evidence: "swi_glu commit 82134c34 pybind11.cpp lines 25-32 — aclrtlaunch direct"
  applies_to: "port_a3_to_a5 mode, fused-activation matmul-family ops"
  anti_pattern_caught: "researcher 'vendor gap' framing based on aclnn 561000 probe"
  kb_target: OL
"""
    ws = _seed_workspace(tmp_path, content)
    result = _extract_kb_draft_from_user_decision(ws, "fused_quant_mat_mul")
    assert result is not None
    body = result.read_text()
    assert "mode=structured" in body
    assert "**kb_target**: OL" in body
    assert "port_a3_to_a5 mode does NOT depend on aclnn" in body
    assert "82134c34" in body  # evidence anchor preserved
    assert "vendor gap" in body  # anti-pattern preserved


def test_heuristic_mode_classifies_by_keyword_vendor(tmp_path: Path) -> None:
    """Without structured block, vendor/561xxx keywords classify as PB candidate."""
    content = """next_state: await_user_decision
reason: |
  CANN binary not registered for V351 — aclnnFusedQuantMatmul returned 561000.
  No builtin op desc info. Vendor hasn't shipped the variant on the new SoC.
"""
    ws = _seed_workspace(tmp_path, content)
    result = _extract_kb_draft_from_user_decision(ws, "test_op")
    assert result is not None
    body = result.read_text()
    assert "mode=heuristic" in body
    assert "**kb_target**: PB" in body
    assert "NEEDS_DISTILLATION" in body  # heuristic flags for owner re-write


def test_heuristic_mode_classifies_by_keyword_pattern(tmp_path: Path) -> None:
    """precedent / paradigm / pattern keywords classify as OL candidate."""
    content = """next_state: await_worker
reason: |
  Follow the swi_glu precedent. Same V220→V351 port pattern via pybind11.
  This is the same paradigm as commit 82134c34.
"""
    ws = _seed_workspace(tmp_path, content)
    result = _extract_kb_draft_from_user_decision(ws, "test_op")
    body = result.read_text()
    assert "mode=heuristic" in body
    assert "**kb_target**: OL" in body


def test_heuristic_mode_classifies_by_keyword_build_error(tmp_path: Path) -> None:
    """build error / compile fail / 错误码 keywords classify as EC candidate."""
    content = """next_state: await_worker
reason: |
  Build error: linker error on matmul_intf.h transitive include. Compile fail
  with 错误码 4017. EC-N candidate.
"""
    ws = _seed_workspace(tmp_path, content)
    body = _extract_kb_draft_from_user_decision(ws, "test_op").read_text()
    assert "**kb_target**: EC" in body


def test_trivial_user_decision_no_extraction(tmp_path: Path) -> None:
    """One-line user_decision.md (just next_state) is not strategic — skip."""
    content = "next_state: abort\nreason: out of budget\n"
    ws = _seed_workspace(tmp_path, content)
    result = _extract_kb_draft_from_user_decision(ws, "test_op")
    assert result is None
    assert not (ws / "kb_draft_from_user_decision.md").exists()


def test_idempotency_no_double_append(tmp_path: Path) -> None:
    """Re-running extraction on same user_decision.md doesn't double-write."""
    content = """next_state: await_worker
reason: |
  Some strategic directive long enough to trigger extraction.
  More text to push past the 100-char threshold for real strategic content.

kb_distillation:
  rule: "test rule"
  evidence: "test evidence"
  applies_to: "test scope"
  anti_pattern_caught: "test anti"
  kb_target: P-P
"""
    ws = _seed_workspace(tmp_path, content)
    r1 = _extract_kb_draft_from_user_decision(ws, "test_op")
    size1 = r1.stat().st_size
    r2 = _extract_kb_draft_from_user_decision(ws, "test_op")
    size2 = r2.stat().st_size
    assert size1 == size2, "second extraction should be no-op (idempotency guard)"


def test_kb_target_field_missing_defaults_to_candidate(tmp_path: Path) -> None:
    """If structured block has no kb_target, default to 'candidate' slot."""
    content = """next_state: await_worker
reason: |
  Long enough strategic content to trigger extraction beyond 100 bytes
  threshold check. More text more text more text more text more text more.

kb_distillation:
  rule: "some rule without kb_target field"
  evidence: "some evidence"
  applies_to: "some scope"
"""
    ws = _seed_workspace(tmp_path, content)
    body = _extract_kb_draft_from_user_decision(ws, "test_op").read_text()
    assert "**kb_target**: candidate" in body


def test_extraction_includes_verbatim_directive(tmp_path: Path) -> None:
    """kb_draft preserves original user_decision.md content (capped at 8KB)
    so reviewers can verify the rule against the source directive.
    """
    content = """next_state: await_worker
reason: |
  Strategic content with anchors: commit 82134c34, OL-185 reference,
  port_a3 paradigm extends to fused activation matmul-class ops.

kb_distillation:
  rule: "test rule for verbatim check"
  evidence: "anchor in commit 82134c34"
  applies_to: "port_a3 fused activation"
  anti_pattern_caught: "vendor gap mis-framing"
  kb_target: OL
"""
    ws = _seed_workspace(tmp_path, content)
    body = _extract_kb_draft_from_user_decision(ws, "test_op").read_text()
    # Verbatim section present
    assert "## Provenance directive (verbatim" in body
    # Original directive content preserved
    assert "commit 82134c34" in body
    assert "OL-185 reference" in body


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
