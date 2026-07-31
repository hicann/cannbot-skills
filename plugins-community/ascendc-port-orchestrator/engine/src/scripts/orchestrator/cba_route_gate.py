#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""CBA route gate (§5.2 verification primitive) — proves cannbot tier-a was REALLY used.

codex §5.2: "a visible installed skill is availability only, not a routing guarantee...
add a transcript gate that parses the actual sub-agent JSONL for a `Skill` tool call
naming the required skill. Do NOT accept a prose claim or an analysis.md marker alone."

Given a worker transcript (Claude Code stream JSONL) + a set of REQUIRED tier-a routes
(topic -> community skill name), assert each required skill was actually invoked via the
`Skill` tool. Exit 0 if all satisfied; exit 1 (CBA_MISSING_A_TIER) if any missing.

Pure stdlib. importable + CLI. Unit-tested.
"""
from __future__ import annotations
import logging
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GateResult:
    ok: bool
    invoked_skills: set = field(default_factory=set)
    missing: list = field(default_factory=list)   # (topic, skill) required but not invoked


def skills_invoked(transcript_path: Path) -> set:
    """Return the set of skill names invoked via the `Skill` tool in a CC stream JSONL."""
    invoked: set = set()
    try:
        text = transcript_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return invoked
    skill_calls = {}   # tool_use_id -> skill name
    errored_ids = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        skip_current_item = False
        try:
            ev = json.loads(line)
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
        msg = ev.get("message", {})
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "tool_use" and c.get("name") == "Skill":
                inp = c.get("input", {}) or {}
                for k in ("skill", "command", "name"):
                    v = inp.get(k)
                    if isinstance(v, str) and v:
                        skill_calls[c.get("id")] = v.strip()
                        break
            elif c.get("type") == "tool_result":
                # a Skill call that errored (e.g. "No such tool available: Skill") does NOT count
                body = c.get("content", "")
                if isinstance(body, list):
                    body = " ".join(y.get("text", "") for y in body if isinstance(y, dict))
                if c.get("is_error") or "tool_use_error" in str(body) or "No such tool available" in str(body):
                    errored_ids.add(c.get("tool_use_id"))
    # only count skill calls whose result did NOT error (success = real a-tier usage)
    for tid, name in skill_calls.items():
        if tid not in errored_ids:
            invoked.add(name)
    return invoked


def check(transcript_path: Path, required_routes: dict) -> GateResult:
    """required_routes: {topic_key: cannbot_skill_name}. All must be invoked."""
    invoked = skills_invoked(transcript_path)
    missing = [(t, s) for t, s in required_routes.items() if s not in invoked]
    return GateResult(ok=(not missing), invoked_skills=invoked, missing=missing)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="CBA route gate: assert required cannbot tier-a skills were invoked in a worker transcript.")
    ap.add_argument("--transcript", required=True, type=Path, help="worker stream JSONL")
    ap.add_argument("--require", action="append", default=[], metavar="TOPIC=SKILL",
                    help="required tier-a route topic=cannbot_skill (repeatable)")
    a = ap.parse_args(argv)
    routes = {}
    for r in a.require:
        if "=" in r:
            t, s = r.split("=", 1)
            routes[t.strip()] = s.strip()
    res = check(a.transcript, routes)
    print(f"invoked Skill(s): {sorted(res.invoked_skills) or '(none)'}")
    if res.ok:
        print(f"CBA_ROUTE_GATE PASS — all {len(routes)} required tier-a route(s) invoked")
        return 0
    for t, s in res.missing:
        print(f"CBA_MISSING_A_TIER: topic={t} required cannbot skill={s} NOT invoked")
    return 1


if __name__ == "__main__":
    sys.exit(main())
