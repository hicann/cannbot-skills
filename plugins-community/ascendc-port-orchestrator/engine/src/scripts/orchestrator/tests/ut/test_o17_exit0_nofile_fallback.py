# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""O1.7-glitch fix (scan 2026-07-24, main-assigned #4): when `aog-op-classify`
exits 0 WITHOUT writing op_classification.json (the classifier narrated / asked a
clarifying question instead of writing — happens when there is no benchmark source
in scope to classify, e.g. an `--optimize` re-entry op lacking a
<op>.py/<op>.json; surfaced 2026-07-24 on selective_scan_fwd_simd re-optimize),
classify() must write a deterministic **sha-matched fallback** op_classification.json.

Pre-fix: returned a bare error WITHOUT caching → (a) every orchestrator resume
re-invoked the flaky 120s LLM subprocess, (b) downstream saw no op_classification.json.
Post-fix (verified here): the fallback file always exists, re-runs on the same source
read it cached (no re-invoke), downstream uses default KB (empty tags), and a source
change re-triggers a fresh classify.

Run: python3 -m pytest src/scripts/orchestrator/tests/ut/test_o17_exit0_nofile_fallback.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o17_classify as o17  # noqa: E402


def test_exit0_no_file_writes_sha_matched_fallback(tmp_path, monkeypatch):
    ws = tmp_path / "selective_scan_fwd_simd"
    ws.mkdir()
    (ws / "model.py").write_text("class Model:\n    pass\n")  # source for sha

    calls = {"n": 0}

    def fake_invoke(ws_, timeout=120):
        # exit 0 (success=True), classifier NARRATED instead of writing a file,
        # stdout has NO "Unknown command" (the genuine no-source glitch).
        calls["n"] += 1
        return (True, "I need more context — which op variant do you want to classify?", "")

    monkeypatch.setattr(o17, "_invoke_claude_skill", fake_invoke)

    result = o17.classify(ws, force=True)

    # (a) graceful fallback, NOT a hard error; empty tags → downstream default KB
    assert result.op_class_tags == [], f"expected empty tags, got {result.op_class_tags}"
    assert result.error is None, f"fallback must be graceful (no error), got {result.error!r}"

    # (b) deterministic fallback file written + sha-matched
    cls_path = ws / "op_classification.json"
    assert cls_path.exists(), "fallback op_classification.json must be written"
    data = json.loads(cls_path.read_text())
    assert data["source"] == "o17_fallback_unclassified"
    assert data["op_class_tags"] == []
    cur_sha = getattr(o17, '_compute_source_sha256')(ws)
    assert data["source_sha256"] == cur_sha, "fallback must be sha-matched so re-runs read it cached"

    # (c) re-run on the SAME source reads the cached fallback → NO re-invoke
    n_after_first = calls["n"]
    result2 = o17.classify(ws)  # no force → cache short-circuit
    assert calls["n"] == n_after_first, "re-run must NOT re-invoke the flaky skill (fallback cached)"
    assert result2.op_class_tags == []


def test_source_change_retriggers_classify_after_fallback(tmp_path, monkeypatch):
    """A real source change (sha change) must re-trigger classify, not stay stuck
    on the fallback — the fallback is scoped to the exact source content.
    """
    ws = tmp_path / "selective_scan_fwd_simd"
    ws.mkdir()
    (ws / "model.py").write_text("class Model:\n    pass\n")

    calls = {"n": 0}

    def fake_invoke(ws_, timeout=120):
        calls["n"] += 1
        return (True, "clarifying question, no file written", "")

    monkeypatch.setattr(o17, "_invoke_claude_skill", fake_invoke)

    o17.classify(ws, force=True)          # writes fallback for source v1
    first_calls = calls["n"]
    (ws / "model.py").write_text("class Model:\n    y = 1\n")  # source changes → sha changes
    o17.classify(ws)                      # no force: sha mismatch → must re-invoke
    assert calls["n"] == first_calls + 1, "source change must re-trigger classify, not stay cached"
