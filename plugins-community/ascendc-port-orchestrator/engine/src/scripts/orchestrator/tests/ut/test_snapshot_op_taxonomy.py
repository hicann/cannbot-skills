# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""state_machine.snapshot surfaces op_class via schema_norm._detect_op_class.

Empirical catch (2026-05-20 S5 cold-start on 3_FusionAttention):
state_machine.snapshot() previously read only verification.json; it didn't surface
op_class_tags. As a result the `plugin_method` primitive's resolution in eval_condition
passed `op_class="unknown"` to the active plugin method, which returned False,
blocking the intended paradigm route.

Main agent's pointer (2026-05-20): use the canonical `schema_norm._detect_op_class`
(also used by ko_escalation_threshold) instead of re-implementing tag-set heuristic.
The function returns an uppercase space-joined tag string like "FUSED SOFTMAX
TRANSCENDENTAL REDUCTION REFERENCE-UB" — plugin gate uses substring checks (per
existing ko_escalation_threshold convention).

These tests pin:
- snapshot populates snap["op_taxonomy"] (class + complexity)
- FA-class derivation: {"FUSED", "SOFTMAX"} ⊂ op_class string → complexity="L4"
- Generic op without FA tags → class via _detect_op_class, complexity="unknown"
- Missing op_classification.json → defaults preserved
- Bad JSON → defaults preserved (no crash)
- End-to-end plugin_method dispatch with realistic FA workspace
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "scripts" / "workflow"))
sys.path.insert(0, str(ROOT / "src" / "scripts" / "orchestrator"))

import state_machine as sm  # noqa: E402
import schema_norm  # noqa: E402


def _ws_with_classification(tmp_path: Path, tags: list[str]) -> Path:
    ws = tmp_path / "ws_test"
    ws.mkdir()
    (ws / "op_classification.json").write_text(json.dumps({
        "op": "test_op",
        "schema_version": 3,
        "op_class_tags": tags,
    }))
    return ws


def test_snapshot_defaults_when_no_op_classification(tmp_path):
    """No op_classification.json → op_taxonomy.class falls through to schema_norm
    heuristic (which uses workspace name); complexity stays 'unknown'.
    """
    ws = tmp_path / "ws_no_class"
    ws.mkdir()
    snap = sm.snapshot(ws)
    assert "op_taxonomy" in snap
    assert snap["op_taxonomy"]["complexity"] == "unknown"  # no FA tag combo


def test_snapshot_fa_tags_yield_l4_complexity(tmp_path):
    """Canonical FA structural signature: tags contain FUSED + SOFTMAX → complexity=L4."""
    ws = _ws_with_classification(tmp_path, [
        "fused", "softmax", "transcendental", "reduction", "reference-ub"
    ])
    snap = sm.snapshot(ws)
    op_class = snap["op_taxonomy"]["class"]
    assert "FUSED" in op_class
    assert "SOFTMAX" in op_class
    assert snap["op_taxonomy"]["complexity"] == "L4"


def test_snapshot_fused_without_softmax_not_l4(tmp_path):
    """Fused but not FA (no SOFTMAX) → complexity stays 'unknown'."""
    ws = _ws_with_classification(tmp_path, ["fused", "quant", "matmul"])
    snap = sm.snapshot(ws)
    assert "FUSED" in snap["op_taxonomy"]["class"]
    assert snap["op_taxonomy"]["complexity"] == "unknown"


def test_snapshot_softmax_without_fused_not_l4(tmp_path):
    """Softmax-only (not a fused op) → complexity stays 'unknown'."""
    ws = _ws_with_classification(tmp_path, ["softmax", "reduction"])
    snap = sm.snapshot(ws)
    assert "SOFTMAX" in snap["op_taxonomy"]["class"]
    assert snap["op_taxonomy"]["complexity"] == "unknown"


def test_snapshot_elementwise_op_unknown_complexity(tmp_path):
    """Pointwise / single-op tags → complexity unknown (not a fused-class candidate)."""
    ws = _ws_with_classification(tmp_path, ["elementwise", "unary"])
    snap = sm.snapshot(ws)
    assert snap["op_taxonomy"]["complexity"] == "unknown"


def test_snapshot_malformed_op_classification_doesnt_crash(tmp_path):
    """Bad JSON in op_classification.json → defaults preserved, no crash."""
    ws = tmp_path / "ws_bad_json"
    ws.mkdir()
    (ws / "op_classification.json").write_text("{not valid json,,,")
    snap = sm.snapshot(ws)
    assert "op_taxonomy" in snap
    assert snap["op_taxonomy"]["complexity"] == "unknown"


def test_detect_op_class_public_alias():
    """schema_norm.detect_op_class is a public alias for _detect_op_class."""
    assert schema_norm.detect_op_class is getattr(schema_norm, '_detect_op_class')


def test_eval_condition_plugin_method_routes_fa_through_plugin_gate(tmp_path):
    """End-to-end: FA workspace → snapshot → ctx → plugin_method → IL gate fires.

    Mirrors S5 empirical scenario: 3_FusionAttention workspace with FA op_class_tags +
    BenchmarkPlugin-shaped opt-in should trigger the IL gate.
    """
    from plugins.base import BasePlugin

    class _OptInPlugin(BasePlugin):
        name = "test_plugin"

        def route_check(self, op_class, op_complexity, worker_signal):
            return (
                op_complexity == "L4"
                and "FUSED" in (op_class or "")
                and "SOFTMAX" in (op_class or "")
                and worker_signal == "structural_rewrite_needed"
            )

    ws = _ws_with_classification(tmp_path, [
        "fused", "softmax", "transcendental", "reduction"
    ])

    snap = sm.snapshot(ws)
    ctx = {
        "handoff": "→ orchestrator: structural_rewrite_needed — FA scope-spanning",
        "snapshot": snap,
        "iter_counts": {},
        "ws": ws,
        "sm": {},
        "plugin": _OptInPlugin(),
    }
    assert sm.eval_condition({"plugin_method": "route_check"}, ctx) is True


def test_eval_condition_plugin_method_skips_non_fa_op(tmp_path):
    """Non-FA op + correct signal → plugin returns False → gate skips."""
    from plugins.base import BasePlugin

    class _OptInPlugin(BasePlugin):
        name = "test_plugin"

        def route_check(self, op_class, op_complexity, worker_signal):
            return (
                op_complexity == "L4"
                and "FUSED" in (op_class or "")
                and "SOFTMAX" in (op_class or "")
                and worker_signal == "structural_rewrite_needed"
            )

    # Workspace has fused tag but NOT softmax → not FA → not L4 per snapshot heuristic
    ws = _ws_with_classification(tmp_path, ["fused", "quant", "matmul"])

    snap = sm.snapshot(ws)
    ctx = {
        "handoff": "→ orchestrator: structural_rewrite_needed",
        "snapshot": snap,
        "iter_counts": {},
        "ws": ws,
        "sm": {},
        "plugin": _OptInPlugin(),
    }
    assert sm.eval_condition({"plugin_method": "route_check"}, ctx) is False


def test_classify_falls_back_to_stale_valid_json_when_skill_times_out(tmp_path):
    """When `op_classification.error` is cached for current sha AND a stale
    `op_classification.json` exists with valid tags, classify() should fall
    back to the stale-but-valid JSON rather than returning empty tags.

    Op_class is the operator's INTRINSIC taxonomy — stable across kernel
    iterations of the SAME op. source_sha tracks regen-relevant changes,
    NOT taxonomy validity. The cached-error path was incorrectly returning
    `tags=[]` even when valid stale classification existed.

    Caught 2026-05-23 white-box debug of workspace/3_FusionAttention FA
    orch run: op_classification.json had valid FA tags from a prior kernel
    iter, but cached error sha pinned current → returned tags=[] →
    `is_fa_class()`=False → all 7 PR #127 FA-class finalize gates silently
    disabled.
    """
    import json as _json
    import sys as _sys
    _here = Path(__file__).resolve()
    _sys.path.insert(0, str(_here.parent.parent.parent))
    from phase_o17_classify import classify, _compute_source_sha256

    ws = tmp_path / "3_FusionAttention"
    ws.mkdir()
    # Synthetic kernel + model so source-sha256 is computable
    kdir = ws / "kernel"
    kdir.mkdir()
    (kdir / "k.h").write_text("// kernel header v2 (post-iter)\n")
    (ws / "model.py").write_text("# model v2\n")

    cur_sha = _compute_source_sha256(ws)

    # Stale-but-valid op_classification.json from PRIOR kernel sha
    (ws / "op_classification.json").write_text(_json.dumps({
        "op": "3_FusionAttention",
        "schema_version": 3,
        "source_sha256": "deadbeef" * 8,  # OLD sha, DIFFERENT from cur_sha
        "op_class_tags": ["fused", "softmax", "transcendental", "reduction"],
        "kb_recommendations": [],
        "rationale": "FA-class taxonomy from prior kernel iter",
    }))

    # Cached error pinning current sha (simulates "skill timed out, don't retry")
    (ws / "op_classification.error").write_text("skill subprocess failed: timeout after 120s")
    (ws / "op_classification.error.sha256").write_text(cur_sha)

    result = classify(ws)
    # Pre-fix: would return tags=[] (BUG — disables op_class-aware paths)
    # Post-fix: returns stale-valid tags with note in error field
    assert result.op_class_tags == ["fused", "softmax", "transcendental", "reduction"], (
        f"Expected stale FA-class tags fallback; got {result.op_class_tags!r}"
    )
    # Error field documents the fallback transparently (post-refactor
    # uses shared `_try_existing_classification` helper message format)
    err_lower = (result.error or "").lower()
    assert "reused intrinsic taxonomy" in err_lower or "stale" in err_lower
    assert "timeout" in err_lower or "timed out" in err_lower


def test_classify_falls_back_on_cached_sentinel_path_too(tmp_path):
    """independent review catch (msg `DISCORD_ID_REDACTED` 2026-05-23): the cached-
    sentinel path at phase_o17_classify.py:182 (skill-unavailable sentinel)
    had the SAME empty-tags bug as the cached-error path. Both paths
    should now fall back to existing op_classification.json via the
    shared `_try_existing_classification` helper.

    Trigger: workspace has stale-but-valid op_classification.json AND a
    skill-unavailable sentinel pinning current sha. Pre-fix: classify()
    returned tags=[] with "skill unavailable (cached sentinel)". Post-fix:
    returns the stale-valid tags with the cache event captured in error.
    """
    import json as _json
    import sys as _sys
    _here = Path(__file__).resolve()
    _sys.path.insert(0, str(_here.parent.parent.parent))
    from phase_o17_classify import (
        classify, _compute_source_sha256, _SKILL_UNAVAILABLE_SENTINEL,
    )

    ws = tmp_path / "ws_sentinel"
    ws.mkdir()
    kdir = ws / "kernel"
    kdir.mkdir()
    (kdir / "k.h").write_text("// kernel v3\n")
    (ws / "model.py").write_text("# model v3\n")

    cur_sha = _compute_source_sha256(ws)

    # Stale-but-valid op_classification.json (different sha)
    (ws / "op_classification.json").write_text(_json.dumps({
        "op": "ws_sentinel",
        "schema_version": 3,
        "source_sha256": "cafebabe" * 8,
        "op_class_tags": ["elementwise", "reference-ub"],
        "kb_recommendations": [],
    }))
    # Skill-unavailable sentinel pinning current sha
    (ws / _SKILL_UNAVAILABLE_SENTINEL).write_text(cur_sha)

    result = classify(ws)
    assert result.op_class_tags == ["elementwise", "reference-ub"], (
        f"sentinel-path fallback missing; got tags={result.op_class_tags!r}"
    )
    assert "skill unavailable" in (result.error or "").lower()
    assert "reused intrinsic taxonomy" in (result.error or "").lower()


def test_classify_returns_empty_when_neither_cached_json_nor_error(tmp_path):
    """When NEITHER a cached classification JSON nor a cached error exists,
    classify() invokes the skill (which will fail in test env) and returns
    empty tags with subprocess-error in the error field. This is the
    pre-existing behavior and the fallback in test_classify_falls_back_*
    must NOT change it.
    """
    import sys as _sys
    _here = Path(__file__).resolve()
    _sys.path.insert(0, str(_here.parent.parent.parent))
    from phase_o17_classify import classify

    ws = tmp_path / "ws_empty"
    ws.mkdir()
    (ws / "model.py").write_text("# empty model\n")

    # Use 1s timeout so test doesn't hang; subprocess will fail fast
    result = classify(ws, timeout=1)
    assert result.op_class_tags == []
