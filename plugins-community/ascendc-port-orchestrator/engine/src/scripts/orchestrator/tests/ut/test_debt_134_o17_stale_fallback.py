# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-134 regression: Phase O1.7 first-invocation skill timeout must fall
back to a stale-but-valid op_classification.json (intrinsic taxonomy) instead
of returning empty op_class_tags.

Pre-fix, the stale-fallback (P135.O17c L3b) only fired on the SECOND invocation
(after the first timeout sha-pinned an error file). The fix extends it to the
FIRST timeout.

Run: python3 -m pytest src/scripts/orchestrator/tests/test_debt_134_o17_stale_fallback.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o17_classify as o17  # noqa: E402


def test_first_invocation_timeout_uses_stale_classification(tmp_path, monkeypatch):
    ws = tmp_path / "3_FusionAttention"
    ws.mkdir()
    # source file so _compute_source_sha256 has content to hash
    (ws / "model.py").write_text("class Model:\n    pass\n")
    # seed a valid op_classification.json with non-empty tags + a STALE
    # source_sha256 (mismatches current → L1 sha-match would miss anyway).
    (ws / "op_classification.json").write_text(json.dumps({
        "op": "3_FusionAttention",
        "op_class_tags": ["fused-attention"],
        "kb_recommendations": ["EC-57"],
        "rationale": "prior valid classification",
        "source_signatures_observed": [],
        "source_sha256": "0" * 64,
    }))
    # simulate the 120s skill timeout on first invocation
    monkeypatch.setattr(
        o17, "_invoke_claude_skill",
        lambda ws_, timeout=120: (False, "", "timeout after 120s"),
    )

    # force=True skips the L1/L2 cache short-circuits → reaches skill invoke
    result = o17.classify(ws, force=True)

    # DEBT-134: reuse stale intrinsic taxonomy, not empty tags
    assert result.op_class_tags == ["fused-attention"], \
        f"expected stale tags reused, got {result.op_class_tags}"
    assert "first invocation" in (result.error or ""), \
        f"error should note the first-invocation fallback, got {result.error!r}"
    # error file + sha pin written so the NEXT invocation short-circuits via L3a
    assert (ws / "op_classification.error").exists()
    assert (ws / "op_classification.error.sha256").exists()


def test_first_invocation_timeout_no_stale_returns_error(tmp_path, monkeypatch):
    """Symmetry: when NO prior op_classification.json exists, a first-invocation
    timeout still returns the empty-tags error (unchanged behavior).
    """
    ws = tmp_path / "99_NoCacheOp"
    ws.mkdir()
    (ws / "model.py").write_text("class Model:\n    pass\n")
    monkeypatch.setattr(
        o17, "_invoke_claude_skill",
        lambda ws_, timeout=120: (False, "", "timeout after 120s"),
    )
    result = o17.classify(ws, force=True)
    assert result.op_class_tags == []
    assert "skill subprocess failed" in (result.error or "")
