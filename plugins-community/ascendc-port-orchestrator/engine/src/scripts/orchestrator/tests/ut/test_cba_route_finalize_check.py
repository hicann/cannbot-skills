# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Unit tests for §5.2 C4 — finalize LOAD∧USE cross-check.

Covers: USE via worker stream log (transcript-level, reusing cba_route_gate), USE via
CBA_USED marker (PROGRESS.md fallback), LOAD-without-USE (missing_use populated),
config-gated no-op (no manifest), and the finalize injection into
verification.json.a_tier_cross_check (additive).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))        # orchestrator/ on path
import finalize_pipeline as _fp  # noqa: E402,F401   (import first: resolves re-export cycle)
import finalize_dispatch as fd   # noqa: E402
import cba_route_finalize_check as cc  # noqa: E402


def _manifest(ws, skills):
    (ws / "a_tier_manifest.json").write_text(json.dumps({
        "op": "13_Cat", "schema_version": 1, "created_ts": "2026-01-01T00:00:00Z",
        "surfaced": [{"topic": f"t-{s}", "skill": s, "kind": "REQUIRED",
                      "reason": "cba-route", "reference_hint": ""} for s in skills]}))


def _stream_log(ws, skills, errored=()):
    lines = []
    for s in skills:
        tid = f"tu_{s}"
        lines.append(json.dumps({"message": {"content": [
            {"type": "tool_use", "id": tid, "name": "Skill", "input": {"skill": s}}]}}))
        if s in errored:
            lines.append(json.dumps({"message": {"content": [
                {"type": "tool_result", "tool_use_id": tid, "is_error": True,
                 "content": "No such tool available: Skill"}]}}))
    (ws / ".cc_stream_log_kw_1.jsonl").write_text("\n".join(lines))


def test_load_and_use_via_stream_log(tmp_path):
    _manifest(tmp_path, ["ops-code-reviewer"])
    _stream_log(tmp_path, ["ops-code-reviewer"])
    r = cc.check_a_tier_load_use(tmp_path)
    assert r["checked"] == 1 and r["missing_use"] == []
    assert r["results"][0] == {"topic": "t-ops-code-reviewer", "skill": "ops-code-reviewer",
                               "loaded": True, "used": True}


def test_load_without_use_is_missing(tmp_path):
    _manifest(tmp_path, ["ops-code-reviewer"])
    _stream_log(tmp_path, ["some-other-skill"])   # required skill NOT invoked
    r = cc.check_a_tier_load_use(tmp_path)
    assert r["missing_use"] == [["t-ops-code-reviewer", "ops-code-reviewer"]]
    assert r["results"][0]["used"] is False


def test_use_via_cba_used_marker_fallback(tmp_path):
    _manifest(tmp_path, ["ops-code-reviewer"])
    (tmp_path / "PROGRESS.md").write_text(
        "some progress\nCBA_USED tier=a topic=code-review skill=ops-code-reviewer\n")
    r = cc.check_a_tier_load_use(tmp_path)
    assert r["missing_use"] == [] and r["results"][0]["used"] is True


def test_errored_skill_call_not_counted_as_use(tmp_path):
    _manifest(tmp_path, ["ops-code-reviewer"])
    _stream_log(tmp_path, ["ops-code-reviewer"], errored=["ops-code-reviewer"])
    r = cc.check_a_tier_load_use(tmp_path)
    assert r["missing_use"] == [["t-ops-code-reviewer", "ops-code-reviewer"]]   # errored != used


def test_config_gated_noop_without_manifest(tmp_path):
    r = cc.check_a_tier_load_use(tmp_path)
    assert r == {"ok": True, "checked": 0, "results": [], "missing_use": []}


def test_finalize_injects_cross_check_record(tmp_path):
    _manifest(tmp_path, ["ops-code-reviewer"])
    _stream_log(tmp_path, ["ops-code-reviewer"])
    (tmp_path / "verification.json").write_text(json.dumps({"verdict": "PASS_T1"}))
    getattr(fd, '_inject_a_tier_cross_check')("13_Cat", tmp_path)
    v = json.loads((tmp_path / "verification.json").read_text())
    assert v["verdict"] == "PASS_T1"                       # additive
    assert v["a_tier_cross_check"]["checked"] == 1
    assert v["a_tier_cross_check"]["missing_use"] == []    # LOAD∧USE both satisfied


def test_finalize_noop_without_manifest(tmp_path):
    (tmp_path / "verification.json").write_text(json.dumps({"verdict": "PASS_T1"}))
    getattr(fd, '_inject_a_tier_cross_check')("op", tmp_path)
    v = json.loads((tmp_path / "verification.json").read_text())
    assert "a_tier_cross_check" not in v                   # config-gated off => untouched
