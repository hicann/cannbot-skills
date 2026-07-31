# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Unit tests for §5.2 C1 — config-gated op-class -> tier-a route auto-populate.

Covers: applies_to selection ("all" / op-class intersect / no-match), config-gating
(env unset / file absent / non-list => no-op), fail-open, and the round-trip
(populate -> brief_kb surfaces the route -> a_tier_manifest LOAD record).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))       # orchestrator/ on path
import cba_route_populate as cp  # noqa: E402
from briefs.brief_kb import _cba_tier_a_routes_block  # noqa: E402


_CFG = [
    {"topic": "code-review", "skill": "ops-code-reviewer", "reference_hint": "检视条例",
     "applies_to": ["all"]},
    {"topic": "simt-vs-simd", "skill": "ascendc-simt-best-practices", "applies_to": ["simt"]},
]


# ── pure selector ────────────────────────────────────────────────────────────
def test_select_all_applies_to_every_op():
    r = cp.select_routes(_CFG, ["elementwise"])
    assert [x["skill"] for x in r] == ["ops-code-reviewer"]   # "all" matches; simt does not


def test_select_op_class_intersect():
    r = cp.select_routes(_CFG, ["simt", "reduction"])
    assert {x["skill"] for x in r} == {"ops-code-reviewer", "ascendc-simt-best-practices"}


def test_select_no_match_when_tags_disjoint_and_no_all():
    r = cp.select_routes([{"topic": "t", "skill": "s", "applies_to": ["simt"]}], ["elementwise"])
    assert r == []


def test_select_skips_malformed_entries():
    r = cp.select_routes(["not-a-dict", {"topic": "t", "skill": "s", "applies_to": "all"}], ["x"])
    assert r == []   # applies_to must be a list; string "all" is malformed -> skipped


# ── config-gating (populate) ─────────────────────────────────────────────────
def test_populate_writes_route_file_on_match(tmp_path):
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps(_CFG))
    ws = tmp_path / "ws"
    ws.mkdir()
    sel = cp.populate_cba_routes("13_Cat", ws, ["elementwise"], config_path=str(cfg))
    assert [x["skill"] for x in sel] == ["ops-code-reviewer"]
    rf = ws / ".cba_required_routes.json"
    assert rf.exists() and json.loads(rf.read_text())[0]["skill"] == "ops-code-reviewer"


def test_populate_noop_when_env_unset_and_no_conventional(tmp_path, monkeypatch):
    monkeypatch.delenv("AOG_CBA_ROUTE_CONFIG", raising=False)
    monkeypatch.setattr(cp, "_CONVENTIONAL", tmp_path / "nope.json")   # isolate from any shipped file
    ws = tmp_path / "ws"
    ws.mkdir()
    assert cp.populate_cba_routes("op", ws, ["x"]) == []
    assert not (ws / ".cba_required_routes.json").exists()   # config-gated OFF => no file


def test_conventional_path_fallback(tmp_path, monkeypatch):
    # no config_path arg + no env => falls back to the bundled conventional path
    monkeypatch.delenv("AOG_CBA_ROUTE_CONFIG", raising=False)
    conv = tmp_path / "cba_routes.json"
    conv.write_text(json.dumps(_CFG))
    monkeypatch.setattr(cp, "_CONVENTIONAL", conv)
    ws = tmp_path / "ws"
    ws.mkdir()
    sel = cp.populate_cba_routes("op", ws, ["elementwise"])
    assert [x["skill"] for x in sel] == ["ops-code-reviewer"]
    assert (ws / ".cba_required_routes.json").exists()


def test_env_override_wins_over_conventional(tmp_path, monkeypatch):
    conv = tmp_path / "conv.json"
    conv.write_text(json.dumps([{"topic": "c", "skill": "conv-skill", "applies_to": ["all"]}]))
    envf = tmp_path / "env.json"
    envf.write_text(json.dumps([{"topic": "e", "skill": "env-skill", "applies_to": ["all"]}]))
    monkeypatch.setattr(cp, "_CONVENTIONAL", conv)
    monkeypatch.setenv("AOG_CBA_ROUTE_CONFIG", str(envf))
    ws = tmp_path / "ws"
    ws.mkdir()
    sel = cp.populate_cba_routes("op", ws, ["x"])
    assert [x["skill"] for x in sel] == ["env-skill"]        # env override beats conventional


def test_conventional_path_is_module_relative_not_cwd():
    # robustness: the conventional path resolves off the module file, not cwd
    assert getattr(cp, '_CONVENTIONAL').is_absolute()
    assert getattr(cp, '_CONVENTIONAL').name == "cba_routes.json"
    assert getattr(cp, '_CONVENTIONAL').parent == Path(cp.__file__).resolve().parent


def test_populate_noop_when_no_match(tmp_path):
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps([{"topic": "t", "skill": "s", "applies_to": ["simt"]}]))
    ws = tmp_path / "ws"
    ws.mkdir()
    assert cp.populate_cba_routes("op", ws, ["elementwise"], config_path=str(cfg)) == []
    assert not (ws / ".cba_required_routes.json").exists()


def test_populate_fail_open_on_bad_config(tmp_path):
    cfg = tmp_path / "routes.json"
    cfg.write_text("{not json")
    ws = tmp_path / "ws"
    ws.mkdir()
    assert cp.populate_cba_routes("op", ws, ["x"], config_path=str(cfg)) == []   # no raise


# ── round-trip: C1 populate -> brief surfaces -> C2 LOAD record ──────────────
def test_c1_populate_feeds_c2_load_record(tmp_path):
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps(_CFG))
    ws = tmp_path / "ws"
    ws.mkdir()
    cp.populate_cba_routes("13_Cat", ws, ["elementwise"], config_path=str(cfg))
    block = _cba_tier_a_routes_block("13_Cat", ws)      # brief surfaces the populated route
    assert "ops-code-reviewer" in block
    rec = json.loads((ws / "a_tier_manifest.json").read_text())   # C2 LOAD record written
    assert [s["skill"] for s in rec["surfaced"]] == ["ops-code-reviewer"]
