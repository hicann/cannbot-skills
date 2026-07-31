# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""M2 OKF read-path (_okf_reference_block / kb_manifest_block): DEFAULT-ON, exclusive, fail-safe.

OKF is the DEFAULT b-tier knowledge source (user 2026-07-14) and is MUTUALLY EXCLUSIVE with the legacy
KB_INDEX manifest — the brief has one or the other, never both. Invariants locked here:
  1. gate default-ON: unset ASCENDC_PORT_OKF → enabled; opt out only with {0,false,off,no}.
  2. exclusivity: enabled → OKF block (or a LOUD empty-marker), NEVER the legacy "# KB MANIFEST";
     opted out (=0) → legacy manifest, NEVER the OKF block.
  3. exclusivity drops only KNOWLEDGE POINTERS — the source-agnostic discipline + per-target hw spec
     (ALWAYS_LOADED_RULES, ANTI-PRESSURE CHECKPOINT, the chip spec) STAY in OKF mode.
  4. fail-loud-no-fallback: OKF enabled but empty/broken retrieval → marker, NEVER a silent legacy fallback.
  5. NEVER raises — any subprocess / JSON / shape failure returns "".
The mocked tests force the kbq/index preconditions True so the mocked subprocess is actually reached.
"""
import json
import subprocess
from pathlib import Path
import pytest
from briefs import _common


def _force_okf_ready(monkeypatch):
    """Make _okf_reference_block reach the (mocked) subprocess: pretend the query script + index exist.
    Also pin CANNBOT_OKF_ENGINE_ROOT so okf_engine's resolver returns on its FIRST (env) candidate under
    the is_file mock — keeps these behavior tests independent of the real filesystem's plugin layout.
    (Real resolver-discovery coverage — env vs sibling vs marketplace dir names, consumer/contributor
    split — lives in test_okf_engine.py against real tmpdirs.)"""
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setenv("CANNBOT_OKF_ENGINE_ROOT", "/fake/cannbot-knowledge")


# --- 1. gate default-ON ---------------------------------------------------
def test_okf_enabled_default_on_and_optout(monkeypatch):
    monkeypatch.delenv("ASCENDC_PORT_OKF", raising=False)
    assert getattr(_common, '_okf_enabled')() is True
    for off in ("0", "false", "off", "no", "OFF", "False"):
        monkeypatch.setenv("ASCENDC_PORT_OKF", off)
        assert getattr(_common, '_okf_enabled')() is False, off
    for on in ("1", "true", "on", "yes", "", "anything"):
        monkeypatch.setenv("ASCENDC_PORT_OKF", on)
        assert getattr(_common, '_okf_enabled')() is True, on


def test_okf_block_empty_when_opted_out(monkeypatch):
    monkeypatch.setenv("ASCENDC_PORT_OKF", "0")
    assert getattr(_common, '_okf_reference_block')("13_Cat", None, "a5") == ""


def test_okf_block_empty_when_engine_missing(monkeypatch):
    """RFC #381: the engine is EXTERNAL (cannbot-knowledge). If it isn't installed, okf_engine resolves
    None → this returns "" and the caller emits the loud marker — NEVER a silent legacy fallback.
    is_file→False simulates cannbot-knowledge absent (no knowledge_query.py at any candidate root).
    """
    monkeypatch.setenv("ASCENDC_PORT_OKF", "1")
    monkeypatch.delenv("CANNBOT_OKF_ENGINE_ROOT", raising=False)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    assert getattr(_common, '_okf_reference_block')("13_Cat", None, "a5") == ""
    # and end-to-end the manifest turns the empty result into the loud marker, not the legacy manifest.
    monkeypatch.delenv("ASCENDC_PORT_OKF", raising=False)
    out = _common.kb_manifest_block("13_Cat", None, "a5")
    assert "knowledge-query 无返回" in out and "# KB MANIFEST" not in out


# --- 2+3. exclusivity + discipline, deterministic (mocked subprocess) ------
def _mock_hits(monkeypatch, payload):
    class _CP:
        returncode = 0
        stdout = payload
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _CP())


def test_okf_block_populated_on_valid_hits(monkeypatch):
    monkeypatch.setenv("ASCENDC_PORT_OKF", "1")
    _force_okf_ready(monkeypatch)
    _mock_hits(monkeypatch, json.dumps({"hits": [
        {"path": "runbooks/field-notes/build/ec-13.md", "title": "SyncFunc", "kind": "field_note", "score": 9}]}))
    blk = getattr(_common, '_okf_reference_block')("13_Cat", None, "a5")
    assert "# OKF 知识卡片" in blk and "ec-13.md" in blk


def test_kb_manifest_default_on_success_is_okf_with_scaffold(monkeypatch):
    """Default (OKF on) + hits → OKF block, NO legacy '# KB MANIFEST', discipline + per-target hw kept."""
    monkeypatch.delenv("ASCENDC_PORT_OKF", raising=False)
    _force_okf_ready(monkeypatch)
    _mock_hits(monkeypatch, json.dumps({"hits": [
        {"path": "runbooks/field-notes/build/ec-13.md", "title": "t", "kind": "field_note", "score": 9}]}))
    out = _common.kb_manifest_block("13_Cat", None, "a5")
    assert "# OKF 知识卡片" in out and "# KB MANIFEST" not in out           # exclusive OKF
    assert "shared/ALWAYS_LOADED_RULES.md" in out                          # BLOCKER regression guard
    assert "ANTI-PRESSURE CHECKPOINT" in out                              # discipline kept
    assert "ascend950pr.md" in out                                        # per-target hw spec kept (a5)


def test_kb_manifest_default_on_target_routes_hw_spec(monkeypatch):
    monkeypatch.delenv("ASCENDC_PORT_OKF", raising=False)
    _force_okf_ready(monkeypatch)
    _mock_hits(monkeypatch, json.dumps({"hits": [
        {"path": "runbooks/field-notes/build/ec-13.md", "title": "t", "kind": "field_note", "score": 9}]}))
    assert "ascend910c.md" in _common.kb_manifest_block("13_Cat", None, "a3")   # a3 → 910c
    assert "ascend910b.md" in _common.kb_manifest_block("13_Cat", None, "a2")   # a2 → 910b


# --- 4. fail-loud, NO silent legacy fallback ------------------------------
def test_kb_manifest_default_on_empty_is_marker_no_fallback(monkeypatch):
    """OKF enabled but retrieval empty → LOUD marker, NEVER the legacy manifest; discipline still there."""
    monkeypatch.delenv("ASCENDC_PORT_OKF", raising=False)
    _force_okf_ready(monkeypatch)
    _mock_hits(monkeypatch, json.dumps({"hits": []}))
    out = _common.kb_manifest_block("13_Cat", None, "a5")
    assert "knowledge-query 无返回" in out            # loud marker
    assert "# KB MANIFEST" not in out            # NO silent legacy fallback
    assert "shared/ALWAYS_LOADED_RULES.md" in out  # discipline still present


def test_kb_manifest_optout_is_legacy_not_okf(monkeypatch):
    monkeypatch.setenv("ASCENDC_PORT_OKF", "0")
    out = _common.kb_manifest_block("13_Cat", None, "a5")
    assert "# KB MANIFEST" in out and "# OKF 知识卡片" not in out


# --- 5. never raises (subprocess actually reached via _force_okf_ready) ----
@pytest.mark.parametrize("payload", ['{"hits":"x"}', '{"hits":[1,2]}', '[1,2,3]', 'not json', '{}'])
def test_okf_block_never_raises_on_malformed_output(monkeypatch, payload):
    monkeypatch.setenv("ASCENDC_PORT_OKF", "1")
    _force_okf_ready(monkeypatch)
    _mock_hits(monkeypatch, payload)
    assert getattr(_common, '_okf_reference_block')("13_Cat", None, "a5") == ""


def test_okf_block_empty_on_subprocess_error(monkeypatch):
    monkeypatch.setenv("ASCENDC_PORT_OKF", "1")
    _force_okf_ready(monkeypatch)

    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="knowledge_query", timeout=30)

    monkeypatch.setattr(subprocess, "run", _boom)
    assert getattr(_common, '_okf_reference_block')("13_Cat", None, "a5") == ""


def test_kb_manifest_subprocess_error_is_marker_no_fallback(monkeypatch):
    """OKF enabled but subprocess raises → still marker + no legacy fallback (fail-loud end-to-end)."""
    monkeypatch.delenv("ASCENDC_PORT_OKF", raising=False)
    _force_okf_ready(monkeypatch)

    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="knowledge_query", timeout=30)

    monkeypatch.setattr(subprocess, "run", _boom)
    out = _common.kb_manifest_block("13_Cat", None, "a5")
    assert "knowledge-query 无返回" in out and "# KB MANIFEST" not in out
