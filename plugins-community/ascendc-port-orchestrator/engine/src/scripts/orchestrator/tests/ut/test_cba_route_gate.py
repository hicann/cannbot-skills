# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Unit tests for cba_route_gate — the CBA tier-a USE gate (§5.2, ported from
a5-cannbot-export/validation). Proves a required community skill was REALLY invoked
via the `Skill` tool in a worker transcript (not a prose claim), errored calls excluded.
"""
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

# cba_route_gate.py lives at orchestrator/ (ut/ -> tests/ -> orchestrator/)
_ORCH_DIR = Path(__file__).resolve().parents[2]
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))
import cba_route_gate as g


@pytest.fixture(autouse=True)
def _cc_backend_by_default(monkeypatch):
    """Keep legacy transcript fixtures independent of host harness environment."""
    from backends.cc_backend import CCBackend
    monkeypatch.setattr(g, "get_backend", lambda: CCBackend())


def _mk(tmp, skills, errored=()):
    p = tmp / "t.jsonl"
    lines = []
    for s in skills:
        tid = f"tu_{s}"
        lines.append(json.dumps({"message": {"content": [
            {"type": "tool_use", "id": tid, "name": "Skill", "input": {"skill": s}}]}}))
        if s in errored:
            lines.append(json.dumps({"message": {"content": [
                {"type": "tool_result", "tool_use_id": tid, "is_error": True,
                 "content": "No such tool available: Skill"}]}}))
    lines.append(json.dumps({"message": {"content": [
        {"type": "tool_use", "name": "Read", "input": {"file_path": "x"}}]}}))
    p.write_text("\n".join(lines))
    return p


def test_pass_when_skill_invoked(tmp_path):
    t = _mk(tmp_path, ["ascendc-api-best-practices"])
    r = g.check(t, {"datacopypad-padding": "ascendc-api-best-practices"})
    assert r.ok and not r.missing


def test_fail_when_skill_missing(tmp_path):
    t = _mk(tmp_path, ["some-other-skill"])
    r = g.check(t, {"datacopypad-padding": "ascendc-api-best-practices"})
    assert not r.ok and r.missing == [("datacopypad-padding", "ascendc-api-best-practices")]


def test_fail_when_no_skills(tmp_path):
    t = _mk(tmp_path, [])
    r = g.check(t, {"x": "ascendc-api-best-practices"})
    assert not r.ok


def test_errored_skill_call_does_not_count(tmp_path):
    # a Skill call whose tool_result errored ("No such tool available") is NOT real usage
    t = _mk(tmp_path, ["ops-code-reviewer"], errored=["ops-code-reviewer"])
    r = g.check(t, {"code-review": "ops-code-reviewer"})
    assert not r.ok and ("code-review", "ops-code-reviewer") in r.missing


def test_opencode_transcript_parsed_by_active_backend(tmp_path, monkeypatch):
    """G7: with the opencode backend active, its NDJSON transcript parser feeds the gate."""
    from backends.opencode_backend import OpencodeBackend
    monkeypatch.setattr(g, "get_backend", lambda: OpencodeBackend(opencode_bin="opencode"))
    t = tmp_path / "oc.jsonl"
    t.write_text("\n".join([
        json.dumps({"type": "text", "timestamp": "2026-08-17T00:00:00Z", "sessionID": "ses_1",
                    "part": {"id": "part_text", "type": "text", "text": "hi"}}),
        json.dumps({"type": "tool_use", "timestamp": "2026-08-17T00:00:01Z", "sessionID": "ses_1",
                    "part": {"id": "part_ok", "type": "tool", "tool": "skill",
                             "state": {"status": "completed", "input": {
                                 "name": "ascendc-api-best-practices"}}}}),
        json.dumps({"type": "tool_use", "timestamp": "2026-08-17T00:00:02Z", "sessionID": "ses_1",
                    "part": {"id": "part_bash", "type": "tool", "tool": "bash",
                             "state": {"status": "completed", "input": {"command": "ls"}}}}),
        json.dumps({"type": "tool_use", "timestamp": "2026-08-17T00:00:03Z", "sessionID": "ses_1",
                    "part": {"id": "part_error", "type": "tool", "tool": "skill",
                             "state": {"status": "error", "input": {
                                 "name": "ops-code-reviewer"}}}}),
    ]))
    r = g.check(t, {"datacopypad-padding": "ascendc-api-best-practices"})
    assert r.ok, r.missing
    assert "ops-code-reviewer" not in r.invoked_skills  # errored call excluded


def test_opencode_accepts_native_part_under_new_envelope_name(tmp_path, monkeypatch):
    """Native part shape, not a hard-coded top-level event list, proves ownership."""
    from backends.opencode_backend import OpencodeBackend
    monkeypatch.setattr(g, "get_backend", lambda: OpencodeBackend(opencode_bin="opencode"))
    t = tmp_path / "oc_new_envelope.jsonl"
    t.write_text(json.dumps({
        "type": "message.part.updated", "sessionID": "ses_1",
        "part": {"id": "part_ok", "type": "tool", "tool": "skill",
                 "state": {"status": "completed", "input": {
                     "name": "ascendc-api-best-practices"}}},
    }))
    r = g.check(t, {"datacopypad-padding": "ascendc-api-best-practices"})
    assert r.ok, r.blocked_note or r.missing


def test_opencode_malformed_skill_event_blocks_not_missing(tmp_path, monkeypatch):
    """A partly native transcript cannot turn malformed evidence into a false missing verdict."""
    from backends.opencode_backend import OpencodeBackend
    monkeypatch.setattr(g, "get_backend", lambda: OpencodeBackend(opencode_bin="opencode"))
    t = tmp_path / "oc_missing_session.jsonl"
    t.write_text("\n".join([
        json.dumps({"type": "text", "sessionID": "ses_1",
                    "part": {"id": "part_text", "type": "text", "text": "started"}}),
        json.dumps({"type": "tool_use", "part": {
            "id": "part_bad", "type": "tool", "tool": "skill",
            "state": {"status": "completed", "input": {
                "name": "ascendc-api-best-practices"}},
        }}),
    ]))
    r = g.check(t, {"datacopypad-padding": "ascendc-api-best-practices"})
    assert not r.ok
    assert r.missing == []
    assert "session id" in r.blocked_note


def test_opencode_unrelated_nonterminal_skill_does_not_block_proven_route(tmp_path, monkeypatch):
    from backends.opencode_backend import OpencodeBackend
    monkeypatch.setattr(g, "get_backend", lambda: OpencodeBackend(opencode_bin="opencode"))
    t = tmp_path / "oc_unrelated_pending.jsonl"
    rows = [
        {"id": "part_required", "status": "completed", "name": "ascendc-api-best-practices"},
        {"id": "part_other", "status": "running", "name": "ops-code-reviewer"},
    ]
    t.write_text("\n".join(json.dumps({
        "type": "tool_use", "sessionID": "ses_1",
        "part": {"id": row["id"], "type": "tool", "tool": "skill",
                 "state": {"status": row["status"], "input": {"name": row["name"]}}},
    }) for row in rows))
    r = g.check(t, {"datacopypad-padding": "ascendc-api-best-practices"})
    assert r.ok, r.blocked_note or r.missing


def test_opencode_required_nonterminal_skill_blocks_route_verdict(tmp_path, monkeypatch):
    from backends.opencode_backend import OpencodeBackend
    monkeypatch.setattr(g, "get_backend", lambda: OpencodeBackend(opencode_bin="opencode"))
    t = tmp_path / "oc_required_pending.jsonl"
    t.write_text(json.dumps({
        "type": "tool_use", "sessionID": "ses_1",
        "part": {"id": "part_required", "type": "tool", "tool": "skill",
                 "state": {"status": "running", "input": {
                     "name": "ascendc-api-best-practices"}}},
    }))
    r = g.check(t, {"datacopypad-padding": "ascendc-api-best-practices"})
    assert not r.ok
    assert r.missing == []
    assert "non-terminal required" in r.blocked_note


def test_opencode_late_error_for_same_part_never_counts_as_use(tmp_path, monkeypatch):
    """Terminal events may repeat a part; an error must win over a prior completed update."""
    from backends.opencode_backend import OpencodeBackend
    monkeypatch.setattr(g, "get_backend", lambda: OpencodeBackend(opencode_bin="opencode"))
    t = tmp_path / "oc_repeat.jsonl"
    rows = []
    for status in ("completed", "error"):
        rows.append(json.dumps({
            "type": "tool_use", "timestamp": f"2026-08-17T00:00:0{len(rows)}Z", "sessionID": "ses_1",
            "part": {"id": "part_repeat", "type": "tool", "tool": "skill",
                     "state": {"status": status, "input": {"name": "ops-code-reviewer"}}},
        }))
    t.write_text("\n".join(rows))
    r = g.check(t, {"code-review": "ops-code-reviewer"})
    assert not r.ok
    assert r.missing == [("code-review", "ops-code-reviewer")]


def test_opencode_blocks_foreign_claude_stream_json(tmp_path, monkeypatch):
    """A foreign transcript is unprovable, not a false CBA_MISSING verdict."""
    from backends.opencode_backend import OpencodeBackend
    monkeypatch.setattr(g, "get_backend", lambda: OpencodeBackend(opencode_bin="opencode"))
    t = tmp_path / "foreign.jsonl"
    t.write_text(json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "tu_1", "name": "Skill",
         "input": {"skill": "ascendc-api-best-practices"}},
    ]}}))
    r = g.check(t, {"datacopypad-padding": "ascendc-api-best-practices"})
    assert not r.ok
    assert r.missing == []
    assert "not native opencode" in r.blocked_note


def test_opencode_scalar_jsonl_event_blocks_instead_of_raising(tmp_path, monkeypatch):
    """Malformed JSONL is unprovable evidence, not an exception or a false missing route."""
    from backends.opencode_backend import OpencodeBackend
    monkeypatch.setattr(g, "get_backend", lambda: OpencodeBackend(opencode_bin="opencode"))
    t = tmp_path / "oc_scalar.jsonl"
    t.write_text("[]\n")
    r = g.check(t, {"datacopypad-padding": "ascendc-api-best-practices"})
    assert not r.ok
    assert r.missing == []
    assert "not an object" in r.blocked_note


def test_opencode_non_skill_tool_without_session_blocks_not_missing(tmp_path, monkeypatch):
    """A malformed non-skill event cannot be used to certify a foreign transcript."""
    from backends.opencode_backend import OpencodeBackend
    monkeypatch.setattr(g, "get_backend", lambda: OpencodeBackend(opencode_bin="opencode"))
    t = tmp_path / "oc_non_skill_no_session.jsonl"
    t.write_text(json.dumps({
        "type": "tool_use",
        "part": {"id": "part_bash", "type": "tool", "tool": "bash",
                 "state": {"status": "completed", "input": {"command": "ls"}}},
    }))
    r = g.check(t, {"datacopypad-padding": "ascendc-api-best-practices"})
    assert not r.ok
    assert r.missing == []
    assert "session id" in r.blocked_note


def test_skills_invoked_respects_backend_seam_and_refuses_unprovable(tmp_path, monkeypatch):
    t = tmp_path / "t.jsonl"
    t.write_text("{}\n")
    monkeypatch.setattr(g, "get_backend", lambda: SimpleNamespace(
        transcript_skills=lambda _: SimpleNamespace(
            invoked={"ascendc-api-best-practices"}, parseable=True, note="")))
    assert g.skills_invoked(t) == {"ascendc-api-best-practices"}
    monkeypatch.setattr(g, "get_backend", lambda: SimpleNamespace(
        transcript_skills=lambda _: SimpleNamespace(invoked=set(), parseable=False, note="foreign")))
    with pytest.raises(RuntimeError, match="foreign"):
        g.skills_invoked(t)


def test_blocked_when_transcript_unprovable(tmp_path, monkeypatch):
    """An unprovable native transcript yields an explicit BLOCKED verdict."""
    monkeypatch.setattr(g, "get_backend", lambda: SimpleNamespace(
        transcript_skills=lambda p: SimpleNamespace(
            invoked=set(), parseable=False, note="not a native transcript")))
    t = tmp_path / "t.jsonl"
    t.write_text("garbage\n")
    r = g.check(t, {"x": "ascendc-api-best-practices"})
    assert not r.ok
    assert r.missing == []
    assert "not a native transcript" in r.blocked_note


def test_cli_exit_codes(tmp_path):
    t = _mk(tmp_path, ["ascendc-api-best-practices"])
    assert g.main(["--transcript", str(t), "--require", "dc=ascendc-api-best-practices"]) == 0
    assert g.main(["--transcript", str(t), "--require", "dc=missing-skill"]) == 1
