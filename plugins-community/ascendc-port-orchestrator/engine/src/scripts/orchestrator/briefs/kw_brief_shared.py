# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Shared leaf helpers for the kw_brief module family.

Extracted from kw_brief.py (DEBT-201 god-file decomposition, 2026-07-06) as a
LEAF that both the parent `kw_brief` and the sibling `kw_brief_port_a3` import,
so the forced-architecture-honor block has a single definition with no import
cycle (kw_brief -> kw_brief_shared <- kw_brief_port_a3, all one-way).

Behavior is BYTE-IDENTICAL to the pre-split kw_brief functions (prompt-template
refactor; golden-locked by test_kw_brief_port_a3_golden.py + the FA/pa3 goldens).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from briefs._common import _detect_forced_architecture


# `_detect_forced_architecture` + `_FORCED_ARCH_TAGS` live in `_common.py` (the
# lower-level module) and are imported above. They MUST NOT be defined here:
# `_common.kb_manifest_block` also needs the detector, and `_common` importing
# back into `kw_brief` would form a circular import (caught in review 2026-06-16
# — order-dependent ImportError under combined pytest collection). The dependency
# edge is one-way: kw_brief → _common.



def _forced_architecture_block(workspace: Optional[Path]) -> str:
    """If the op's architecture is FORCED at classification time, return an
    instruction block telling kw to HONOR it and implement only — do NOT run
    the SIMT_VS_SIMD decision tree, do NOT override the forced choice. Else "".

    Empty for non-forced ops so their briefs stay byte-identical (matches the
    `_backward_perf_c2_block` / `_port_a3_cube_class_mix_block` convention).

    Root cause: kw was
    given a forced-SIMT classification, but during Phase A authoring kw ran the
    SIMT_VS_SIMD decision tree itself, classified the op recurrence→SIMD, and
    OVERRODE the forced SIMT — while precision wasn't even aligned yet, on a
    signal from kw's own decision logic (not ko, not any perf measurement).
    That is overreach + wrong timing. Correct flow: architecture is FIXED at
    classification time; kw only IMPLEMENTS it; precision aligns first; ONLY ko
    (the optimizer), AFTER precision passes and for performance reasons, may
    propose an architecture change.
    """
    forced = _detect_forced_architecture(workspace)
    if forced is None:
        return ""
    return (
        f"# ARCHITECTURE IS FIXED — {forced} (classification-time decision; do NOT override)\n"
        "\n"
        f"The kernel architecture is **FIXED to {forced}** by classification "
        "(`op_classification.json` carries a forced-architecture marker — "
        "`force_simt`/`force_simd`/`forced_arch`, or a bare SIMT/SIMD tag the "
        "port-mode merge preserved). This is a **decision already made**; it is "
        "NOT yours to revisit during authoring.\n"
        "\n"
        f"- **Implement {forced} as specified.** Author the kernel in the fixed "
        "architecture and bring it to precision.\n"
        "- **Do NOT run the SIMT_VS_SIMD decision tree** "
        "(`target/ascendc/SIMT_VS_SIMD_DECISION.md`). The choice is already made; "
        "re-deriving it is overreach. Do NOT re-classify the op (recurrence→SIMD, "
        "scatter→SIMT, etc.) to second-guess the forced choice.\n"
        f"- **Do NOT override to a different architecture.** Switching {forced} to "
        "the other architecture during Phase A is FORBIDDEN — especially before "
        "precision is even aligned, and especially on a signal from your own "
        "decision logic (not the optimizer, not any perf measurement).\n"
        "- **Architecture-change is a ko-stage decision** — if you believe another "
        "architecture is better for PERFORMANCE, that is for `aog-kernel-optimizer` "
        "to propose AFTER precision passes (performance-driven, post-precision), "
        "NOT a kw decision. **Document any such concern in `analysis.md`** (so ko "
        f"can weigh it later) but IMPLEMENT the forced {forced} choice now.\n"
        "\n"
        "Precision-alignment comes first; the forced architecture is the substrate "
        "you align precision on, not a variable you tune."
    )
