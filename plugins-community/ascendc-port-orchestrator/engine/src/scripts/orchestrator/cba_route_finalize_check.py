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

"""§5.2 C4 — finalize cross-check of tier-a LOAD∧USE.

Cross-checks the objective LOAD record (a_tier_manifest.json, harness-written at
brief-time by brief_kb) against USE evidence: the required community Skill actually
INVOKED in a worker stream log (.cc_stream_log_<agent>_<idx>.jsonl) — the transcript-
level proof, parsed by REUSING cba_route_gate (not a reimplementation) — or a CBA_USED
provenance marker in PROGRESS.md (worker self-report fallback).

Policy: a REQUIRED route that was surfaced (LOAD) but not used (USE) is a WARNING for a
general op — the run still finalizes; the miss is recorded (surfaces silent non-use
without blocking). The C6 proof case HARD-asserts LOAD∧USE on the recorded result.

Config-gated: no a_tier_manifest.json => no-op ({ok, checked:0}). Pure stdlib.
"""
from __future__ import annotations
import logging

import json
from pathlib import Path

import cba_route_gate as _gate  # REUSE the USE parser (no rebuild)


def _used_skills(workspace: Path) -> set:
    """Union of tier-a community skills evidenced as USED: invoked via the `Skill` tool
    in any worker stream log (transcript-level proof, via cba_route_gate), plus any named
    in a CBA_USED marker in PROGRESS.md (self-report fallback)."""
    used: set = set()
    for sl in sorted(workspace.glob(".cc_stream_log_*.jsonl")):
        skip_current_item = False
        try:
            used |= _gate.skills_invoked(sl)
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
    prog = workspace / "PROGRESS.md"
    if prog.exists():
        try:
            for line in prog.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "CBA_USED" in line and "skill=" in line:
                    sk = line.split("skill=", 1)[1].split()[0].strip().strip("`")
                    if sk:
                        used.add(sk)
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
    return used


def check_a_tier_load_use(workspace) -> dict:
    """Cross-check surfaced (LOAD) vs used (USE) tier-a routes.
    Returns {ok, checked, results:[{topic,skill,loaded,used}], missing_use:[[topic,skill]]}.
    Config-gated: no manifest => {ok:True, checked:0}. Never raises (returns the no-op
    shape on any error) — the general policy is warn-not-block; the caller decides.
    """
    workspace = Path(workspace)
    noop = {"ok": True, "checked": 0, "results": [], "missing_use": []}
    mf = workspace / "a_tier_manifest.json"
    if not mf.exists():
        return noop
    try:
        manifest = json.loads(mf.read_text())
        surfaced = manifest.get("surfaced") or [] if isinstance(manifest, dict) else []
    except Exception:
        return noop
    used = _used_skills(workspace)
    results, missing = [], []
    for s in surfaced:
        if not isinstance(s, dict):
            continue
        skill, topic = s.get("skill"), s.get("topic")
        is_required = s.get("kind", "REQUIRED") == "REQUIRED"
        is_used = skill in used
        results.append({"topic": topic, "skill": skill, "loaded": True, "used": is_used})
        if is_required and not is_used:
            missing.append([topic, skill])
    return {"ok": True, "checked": len(results), "results": results, "missing_use": missing}
