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
from pathlib import Path

# cba_route_gate.py lives at orchestrator/ (ut/ -> tests/ -> orchestrator/)
_ORCH_DIR = Path(__file__).resolve().parents[2]
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))
import cba_route_gate as g


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


def test_cli_exit_codes(tmp_path):
    t = _mk(tmp_path, ["ascendc-api-best-practices"])
    assert g.main(["--transcript", str(t), "--require", "dc=ascendc-api-best-practices"]) == 0
    assert g.main(["--transcript", str(t), "--require", "dc=missing-skill"]) == 1
