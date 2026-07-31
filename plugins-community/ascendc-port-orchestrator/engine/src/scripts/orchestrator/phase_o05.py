# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase O0.5 durable state init (P0mm 2026-05-06).

Plan: docs/design/CONTRACT_AND_MATURITY_NOTES.md#fsm-phase-gap-fix-plan Step 4.

Scope:
    Write `.opgen_state.json` at workspace start with op metadata
    (start ts, current phase tracking pointer). Resume + diagnose paths
    reference this file for cross-session continuity.

Idempotent: re-init updates `last_seen_ts` but preserves `started_ts`
and other origin metadata.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


STATE_FILE = ".opgen_state.json"
SCHEMA_VERSION = 1
SUPPORTED_OPGEN_MODES = frozenset({"port_a3_to_a5", "backward"})


@dataclass
class O05Report:
    verdict: str  # "INITIALIZED" | "REFRESHED"
    summary: str = ""


def init_durable_state(
    workspace: Path,
    op: str,
    *,
    lane: int = 0,
    target: str = "a5",
    opgen_mode: Optional[str] = None,
    backend: str = "ascendc",
) -> O05Report:
    """Initialize or refresh `.opgen_state.json` for a supported workflow.

    On first run: writes new state file with started_ts.
    On re-runs: updates last_seen_ts, increments invocation_count.
    Other origin fields (op, target, started_ts, opgen_mode) are preserved.
    The mode is explicit and immutable: callers must choose arch22→arch35
    migration or backward generation, and a workspace cannot switch between
    them during refresh.

    """
    if opgen_mode not in SUPPORTED_OPGEN_MODES:
        choices = ", ".join(sorted(SUPPORTED_OPGEN_MODES))
        raise ValueError(
            f"opgen_mode must be explicit and supported ({choices}); got {opgen_mode!r}"
        )
    if backend != "ascendc":
        raise ValueError(f"only the AscendC backend is supported; got {backend!r}")

    workspace.mkdir(parents=True, exist_ok=True)
    state_path = workspace / STATE_FILE
    now = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")

    if state_path.exists():
        try:
            existing = json.loads(state_path.read_text())
        except Exception:
            # Malformed — treat as missing, overwrite
            existing = None
    else:
        existing = None

    if existing and isinstance(existing, dict) and existing.get("op") == op:
        # Refresh: bump last_seen + invocation_count, preserve origin
        existing["last_seen_ts"] = now
        existing["invocation_count"] = int(existing.get("invocation_count", 1)) + 1
        existing["lane"] = lane  # current invocation's lane
        declared_mode = existing.get("opgen_mode")
        if declared_mode not in (None, opgen_mode):
            raise ValueError(
                f"workspace mode conflict: state declares {declared_mode!r}, "
                f"invocation requested {opgen_mode!r}"
            )
        declared_backend = existing.get("backend")
        if declared_backend not in (None, "ascendc"):
            raise ValueError(
                f"workspace backend conflict: state declares {declared_backend!r}; "
                "only AscendC is supported"
            )
        existing["opgen_mode"] = opgen_mode
        existing["backend"] = "ascendc"
        # Remove the retired broad-source learning opt-in from legacy state so
        # resume cannot reactivate it after this scoped workflow is installed.
        state_path.write_text(json.dumps(existing, indent=2))
        return O05Report(
            verdict="REFRESHED",
            summary=(
                f"durable state refreshed (invocation #{existing['invocation_count']}, "
                f"mode={existing['opgen_mode']}, backend=ascendc)"
            ),
        )

    # First time, malformed prior, or op-name mismatch (rare) — fresh write
    new_state = {
        "schema_version": SCHEMA_VERSION,
        "op": op,
        "target": target,
        "lane": lane,
        "opgen_mode": opgen_mode,  # W6
        "backend": backend,  # P135.TFc
        "started_ts": now,
        "last_seen_ts": now,
        "invocation_count": 1,
    }
    state_path.write_text(json.dumps(new_state, indent=2))
    return O05Report(
        verdict="INITIALIZED",
        summary=f"durable state initialized (op={op}, target={target}, backend={backend})",
    )


def read_durable_state(workspace: Path) -> Optional[dict]:
    """Read `.opgen_state.json` if present + valid. Returns None otherwise."""
    p = workspace / STATE_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None
