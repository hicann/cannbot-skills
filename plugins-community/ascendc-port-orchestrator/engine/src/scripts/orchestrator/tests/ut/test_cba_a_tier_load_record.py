# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Unit tests for §5.2 C2 — the objective tier-a LOAD record.

Two halves:
  A) brief-time: _cba_tier_a_routes_block writes workspace/a_tier_manifest.json naming
     the SURFACED tier-a skills (harness-objective LOAD evidence, independent of the
     worker's self-report). Idempotent (created_ts preserved) + fail-open.
  B) finalize: _inject_a_tier_loaded merges the manifest into verification.json as
     `a_tier_loaded` (additive + fail-open). Absent manifest => no-op (config-gated off).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))       # orchestrator/ on path
import finalize_pipeline as _fp  # noqa: E402,F401  (import first: resolves the re-export cycle)
import finalize_dispatch as fd   # noqa: E402

from briefs.brief_kb import _cba_tier_a_routes_block  # noqa: E402  (orchestrator/ on path)


def _route(tmp, routes):
    (tmp / ".cba_required_routes.json").write_text(json.dumps(routes))


# ── A) brief-time LOAD record ────────────────────────────────────────────────
def test_surface_writes_load_record(tmp_path):
    _route(tmp_path, [{"topic": "code-review", "skill": "ops-code-reviewer", "reference_hint": "检视条例"}])
    block = _cba_tier_a_routes_block("13_Cat", tmp_path)
    assert "ops-code-reviewer" in block            # surfaced in the brief (USE prompt)
    mf = tmp_path / "a_tier_manifest.json"
    assert mf.exists()                             # objective LOAD record written
    rec = json.loads(mf.read_text())
    assert rec["op"] == "13_Cat" and rec["schema_version"] == 1 and rec["created_ts"]
    assert rec["surfaced"] == [{
        "topic": "code-review", "skill": "ops-code-reviewer",
        "kind": "REQUIRED", "reason": "cba-route", "reference_hint": "检视条例"}]


def test_load_record_idempotent_created_ts(tmp_path):
    _route(tmp_path, [{"topic": "t", "skill": "s"}])
    _cba_tier_a_routes_block("op", tmp_path)
    ts1 = json.loads((tmp_path / "a_tier_manifest.json").read_text())["created_ts"]
    _cba_tier_a_routes_block("op", tmp_path)       # re-brief (each agent rebuilds the brief)
    ts2 = json.loads((tmp_path / "a_tier_manifest.json").read_text())["created_ts"]
    assert ts1 == ts2                              # preserved => stable across re-briefs


def test_no_routes_no_load_record(tmp_path):
    # config-gated OFF: no route file => no manifest written (default a5_ops path)
    assert _cba_tier_a_routes_block("op", tmp_path) == ""
    assert not (tmp_path / "a_tier_manifest.json").exists()


# ── B) finalize merge into verification.json ─────────────────────────────────
def test_finalize_merges_a_tier_loaded(tmp_path):
    (tmp_path / "a_tier_manifest.json").write_text(json.dumps({
        "op": "13_Cat", "schema_version": 1, "created_ts": "2026-01-01T00:00:00Z",
        "surfaced": [{"topic": "code-review", "skill": "ops-code-reviewer", "kind": "REQUIRED",
                      "reason": "cba-route", "reference_hint": ""}]}))
    (tmp_path / "verification.json").write_text(json.dumps({"op": "13_Cat", "verdict": "PASS_T1"}))
    getattr(fd, '_inject_a_tier_loaded')("13_Cat", tmp_path)
    v = json.loads((tmp_path / "verification.json").read_text())
    assert v["a_tier_loaded"] == [{"topic": "code-review", "skill": "ops-code-reviewer", "kind": "REQUIRED"}]
    assert v["verdict"] == "PASS_T1"               # additive: existing fields untouched


def test_finalize_noop_without_manifest(tmp_path):
    # config-gated OFF: no manifest => verification.json untouched (no a_tier_loaded key)
    (tmp_path / "verification.json").write_text(json.dumps({"verdict": "PASS_T1"}))
    getattr(fd, '_inject_a_tier_loaded')("op", tmp_path)
    v = json.loads((tmp_path / "verification.json").read_text())
    assert "a_tier_loaded" not in v


def test_finalize_fail_open_on_bad_manifest(tmp_path):
    (tmp_path / "a_tier_manifest.json").write_text("{not json")
    (tmp_path / "verification.json").write_text(json.dumps({"verdict": "PASS_T1"}))
    getattr(fd, '_inject_a_tier_loaded')("op", tmp_path)       # must not raise
    assert json.loads((tmp_path / "verification.json").read_text())["verdict"] == "PASS_T1"
