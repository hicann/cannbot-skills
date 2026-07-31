# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Named-capability resolution (STEP 2 / Phase 2 of the build-capability extension).

STEP 1 (d9b70144) shipped the (B) interface + (C) baseline refactor: every build
gets `baseline_contributions(ctx)` folded in, proven byte-identical by the golden
gate. STEP 2 (this module) adds the OPT-IN half: a kernel may declare a
`"capabilities": [...]` list in its `build_overrides.json`; each declared name is
resolved through the process-wide `registry` to a factory that returns an ADDITIVE
`CMakeContribution`, and those are merged AFTER the baseline (baseline always first).

`load_capabilities` mirrors `baseline_per_source_defines.load_per_source_defines`
one-for-one: same `build_overrides.json` file, same defensive parsing, and — the
load-bearing GUARDRAIL — it returns `[]` whenever the key is absent/missing/malformed.
An op with no `capabilities` therefore resolves to `declared == []` → `contribs == []`
→ the merge is baseline-only → byte-identical to STEP-1 output. This is what keeps
the four no-capability goldens unchanged.

Canonical-boundary invariant (__init__.py §2B): a declared name that does NOT resolve
FAILS LOUDLY via `registry.get()` (KeyError) — a half-wired capability must never
silently degrade to a no-op build.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_capabilities(kernel_dir: Path) -> list[str]:
    """Load the declared named build capabilities from kernel/build_overrides.json.

    Schema (STEP 2, additive to the DEBT-20 `per_source_defines` block):
        {
            "capabilities": ["example-capability", ...]
        }

    Returns the declared names in declared order. Returns `[]` when the file is
    missing, unreadable, malformed, or has no (or a non-list) `capabilities` key —
    exactly mirroring `load_per_source_defines`'s empty-on-absence contract so a
    no-capability op produces a byte-identical build (the golden guardrail).
    """
    overrides_path = kernel_dir / "build_overrides.json"
    if not overrides_path.is_file():
        return []
    try:
        data = json.loads(overrides_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    raw = data.get("capabilities", [])
    if not isinstance(raw, list):
        return []
    return [name for name in raw if isinstance(name, str) and name]
