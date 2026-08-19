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
import sys
from dataclasses import dataclass, field
from pathlib import Path

from backends import get_backend  # G7: transcript parsing is backend-owned


@dataclass
class GateResult:
    ok: bool
    invoked_skills: set = field(default_factory=set)
    missing: list = field(default_factory=list)   # (topic, skill) required but not invoked
    # G7: the active backend may not prove its native transcript format; BLOCK explicitly.
    blocked_note: str = ""


def skills_invoked(transcript_path: Path) -> set:
    """Return skills invoked in a transcript the active backend can prove."""
    # Backend-owned parsers preserve the historical CC stream-json and OpenCode NDJSON
    # formats. Unprovable input raises here; route enforcement calls check() for BLOCKED.
    try:
        ts = get_backend().transcript_skills(transcript_path)
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )
        raise RuntimeError(f"transcript parse failed: {error}") from error
    if not ts.parseable:
        raise RuntimeError(ts.note or "transcript format is not provable for the active backend")
    return ts.invoked


def check(transcript_path: Path, required_routes: dict) -> GateResult:
    """Verify every required route against a transcript proved by the active backend."""
    # Unprovable native-format input produces BLOCKED, never a fabricated missing list.
    try:
        ts = get_backend().transcript_skills(transcript_path)
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )
        return GateResult(ok=False, blocked_note=f"transcript parse failed: {error}")
    if not ts.parseable:
        return GateResult(ok=False, blocked_note=ts.note)
    invoked = ts.invoked
    required_skills = set(required_routes.values())
    unproven_required = sorted(set(getattr(ts, "unproven", set())) & required_skills)
    if unproven_required:
        return GateResult(
            ok=False,
            invoked_skills=invoked,
            blocked_note=("native transcript has non-terminal required skill call(s): "
                          + ", ".join(unproven_required)),
        )
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
    if res.blocked_note:
        print(f"CBA_ROUTE_GATE BLOCKED — {res.blocked_note}")
        return 1
    for t, s in res.missing:
        print(f"CBA_MISSING_A_TIER: topic={t} required cannbot skill={s} NOT invoked")
    return 1


if __name__ == "__main__":
    sys.exit(main())
