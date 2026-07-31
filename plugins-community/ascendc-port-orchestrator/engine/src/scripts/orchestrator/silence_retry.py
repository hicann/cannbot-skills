# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Per-workspace silence-retry counters.

Mechanically extracted from orchestrator.py (god-file decomposition 2026-06-30, per
ORCHESTRATOR_REFACTOR_AND_UT_SPEC §1). Behavior unchanged — function bodies are verbatim.
Imported back into orchestrator's namespace (`from silence_retry import *names*`) so existing
call-sites and `orchestrator._silence_retry_path`/etc. external access are preserved.

P0aal-2: silence-retry is the budget for how many times a state may be re-spawned after an agent
returned a silence/no-output result. Counter is per-state (silence can hit any agent type — worker /
probe / optimizer / researcher — each gets its own budget).

MONKEYPATCH NOTE (durable — OL-160-class latent-coupling guard): the functions and module-level
constants/logger here are re-imported into orchestrator's namespace, which preserves
`orchestrator.<name>` attribute LOOKUP only — it does NOT rebind THIS module's own globals. A test
that overrides a symbol one of these functions reads must `monkeypatch.setattr(<this_module>,
'<name>', ...)` on THIS module, NOT on `orchestrator` (patching orchestrator silently misses the
binding used here). No current test patches these on orchestrator; this note prevents a future one
from a silent no-op.
"""
from __future__ import annotations

import json
from pathlib import Path


def _silence_retry_path(workspace: Path) -> Path:
    """P0aal-2: per-workspace silence-retry counter file path. JSON dict
    keyed by state name → integer retry count for that state."""
    return workspace / ".silence_retry_counts.json"


def _load_silence_retry_count(workspace: Path, state: str) -> int:
    """Return silence-retry count for the given state. 0 if file absent."""
    p = _silence_retry_path(workspace)
    try:
        d = json.loads(p.read_text()) if p.exists() else {}
        return int(d.get(state, 0))
    except (OSError, ValueError, TypeError):
        return 0


def _bump_silence_retry_count(workspace: Path, state: str) -> int:
    """Increment + persist the silence-retry counter for `state`. Returns
    the new value. Counter is per-state because silence can hit any
    agent type (worker / probe / optimizer / researcher); each gets its
    own budget."""
    p = _silence_retry_path(workspace)
    try:
        d = json.loads(p.read_text()) if p.exists() else {}
    except (OSError, ValueError):
        d = {}
    new_v = int(d.get(state, 0)) + 1
    d[state] = new_v
    try:
        p.write_text(json.dumps(d, indent=2))
    except OSError:
        pass
    return new_v
