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
"""§5.2 C1 — config-gated auto-populate of the per-op tier-a route file.

Runtime op-class -> route resolution (APPLICABILITY: "which routes does op O need").
This is orthogonal to cba_resolver's term-coverage (design-time curation/dedup:
"which tier covers topic T") — the two answer different questions and must not be
conflated (review 2026-07-12).

Reads the deployment's project route-config (path in $AOG_CBA_ROUTE_CONFIG), selects
entries whose `applies_to` matches this op (contains "all" or intersects op_class_tags),
and writes workspace/{op}/.cba_required_routes.json — the file that
brief_kb._cba_tier_a_routes_block consumes. Config-gated OFF by default (env unset /
file absent / no match => no file written => a5_ops mainline generation unaffected).
Deployments (e.g. cannbot) supply the route-config + set the env var. The applies_to
VALUE ("all" vs specific op-classes) is deployment policy DATA, not mechanism.

Pure stdlib; importable + unit-tested.

route-config schema (JSON list):
  [{"topic": str, "skill": str, "reference_hint": str (optional),
    "applies_to": ["all"] | [op-class tag, ...]}]
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_ENV = "AOG_CBA_ROUTE_CONFIG"
# Bundled conventional route-config path — resolved MODULE-RELATIVE (never cwd-relative,
# so the orchestrator finds it no matter where it is launched from). a5_ops mainline
# does NOT ship this file (=> config-gated OFF by default); a deployment (e.g. cannbot)
# ships cba_routes.json here to activate a-tier routing with zero env wiring.
_CONVENTIONAL = Path(__file__).resolve().parent / "cba_routes.json"


def _resolve_config_path(config_path=None):
    """Route-config lookup order (all resolve to a single Path or None):
      1. explicit config_path arg (test override)
      2. $AOG_CBA_ROUTE_CONFIG (deploy/C6/override)
      3. bundled conventional path (module-relative, cwd-independent)
    Returns the first existing path, else None (=> config-gated OFF, no-op)."""
    if config_path:
        p = Path(config_path)
        return p if p.exists() else None
    env = os.environ.get(_ENV, "").strip()
    if env:
        p = Path(env)
        return p if p.exists() else None
    return _CONVENTIONAL if _CONVENTIONAL.exists() else None


def select_routes(entries, op_class_tags) -> list:
    """Pure selector: route entries whose `applies_to` is "all" or intersects the op's
    class tags. Skips malformed entries. Returns the per-op route list (may be empty).
    """
    tags = set(op_class_tags or [])
    out = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        applies = e.get("applies_to") or []
        if not isinstance(applies, list):
            continue
        if "all" in applies or (tags & set(applies)):
            out.append({
                "topic": e.get("topic", "?"),
                "skill": e.get("skill", "?"),
                "reference_hint": e.get("reference_hint", ""),
            })
    return out


def populate_cba_routes(op, workspace, op_class_tags, *, config_path=None) -> list:
    """Config-gated write of workspace/.cba_required_routes.json from the project
    route-config. Returns the selected routes ([] if gated off / no match). Fail-open:
    any error returns [] and writes nothing (never blocks the op-gen flow).
    """
    try:
        p = _resolve_config_path(config_path)
        if p is None:
            return []
        entries = json.loads(p.read_text())
        if not isinstance(entries, list):
            return []
        selected = select_routes(entries, op_class_tags)
        if selected:
            (Path(workspace) / ".cba_required_routes.json").write_text(
                json.dumps(selected, indent=2))
        return selected
    except Exception:
        return []
