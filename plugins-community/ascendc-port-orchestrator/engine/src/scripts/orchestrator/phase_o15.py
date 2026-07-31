# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Phase O1.5 DET_POLICY classification (P0nn 2026-05-06).

Plan: docs/design/CONTRACT_AND_MATURITY_NOTES.md#fsm-phase-gap-fix-plan Step 5.

Scope:
    Classify the op's determinism policy. Stored in .opgen_state.json
    so downstream phases (worker brief, O5 post-verify, da_brief)
    can act on it.

Three policies:
    - required: kernel MUST be bit-identical across runs (e.g.
      training-graph ops where non-det breaks gradient checkpoint)
    - best_effort: bit-identity preferred but tolerable mismatch
      (most inference ops; OL-83 / OL-110 fail-floor cases)
    - n_a: determinism not measurable (Path A / OL-68 case A; CPU
      fallback ops; ops with no NPU run)

Classification source priority:
    1. analysis.md explicit DET_POLICY line (set by worker after
       reading op semantics)
    2. op_taxonomy hint by op-class tag (norm/reduction → required;
       elementwise → best_effort)
    3. default best_effort (safest fallback — won't gate finalize)
"""
from __future__ import annotations
import logging

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


VALID_POLICIES = ("required", "best_effort", "n_a")
DEFAULT_POLICY = "best_effort"

# Op-class → DET_POLICY hint. Conservative: prefer best_effort unless
# the op-class has a clear correctness-critical determinism need.
_TAG_POLICY_MAP = {
    # required: training-path ops where non-det breaks downstream
    "norm-backward": "required",
    "softmax-backward": "required",
    # n_a: ops that genuinely have no measurable determinism axis
    "path-a-cpu-truth": "n_a",
    # best_effort default (everything else inherits)
}


@dataclass
class O15Report:
    verdict: str  # "CLASSIFIED" | "PRESERVED" | "DEFAULTED"
    policy: str
    source: str  # "analysis_md" | "op_taxonomy" | "default" | "preserved"
    summary: str = ""
    # O5-BESTEFFORT-DET-COUNT-TOLERANCE (2026-07-21): author-declared,
    # IMMUTABLE minimum tier1_pass count a best_effort NONDETERMINISTIC op must
    # meet on every run (e.g. 14 of 16). Declared here at O1.5 — never derived
    # from any later measurement — so Phase O5's count reconciliation can
    # tolerate benign stochastic pass-count variance (measured >= det_floor)
    # without the stochastic MISMATCH→respawn churn. None → no tolerance
    # declared → O5 keeps exact-equality (deterministic ops + undeclared
    # best_effort ops are unchanged). Only meaningful when policy=="best_effort".
    det_floor: Optional[int] = None


def classify_det_policy(
    workspace: Path,
    op: str,
    *,
    op_tags: Optional[list[str]] = None,
    explicit: Optional[str] = None,
    explicit_floor: Optional[int] = None,
) -> O15Report:
    """Determine DET_POLICY for op. Priority: explicit > analysis.md
    > op_taxonomy tag > default.

    Args:
        workspace: workspace dir
        op: op name
        op_tags: optional list of op-class tags (e.g. from op_taxonomy)
        explicit: explicit override from caller (e.g. CLI flag)
        explicit_floor: explicit det_floor override from caller

    Returns O15Report with the classified policy, source, and (when declared)
    an immutable det_floor. Priority for det_floor: explicit_floor >
    analysis.md `DET_FLOOR:` line. det_floor is attached to whatever report is
    returned — it is a separate declared input from the policy.
    """
    # det_floor is a declared INPUT, independent of how policy is resolved.
    # Parse it once from analysis.md (explicit override wins).
    det_floor = explicit_floor
    analysis_md = workspace / "analysis.md"
    analysis_text: Optional[str] = None
    if analysis_md.exists():
        try:
            analysis_text = analysis_md.read_text()
        except Exception:
            analysis_text = None
    if det_floor is None and analysis_text is not None:
        det_floor = _parse_det_floor_from_analysis(analysis_text)

    if explicit and explicit in VALID_POLICIES:
        return O15Report(
            verdict="CLASSIFIED",
            policy=explicit,
            source="explicit",
            summary=f"DET_POLICY={explicit} (explicit override)",
            det_floor=det_floor,
        )

    # 1. Read analysis.md for an explicit DET_POLICY line
    if analysis_text is not None:
        try:
            policy = _parse_det_policy_from_analysis(analysis_text)
            if policy:
                return O15Report(
                    verdict="CLASSIFIED",
                    policy=policy,
                    source="analysis_md",
                    summary=f"DET_POLICY={policy} (parsed from analysis.md)",
                    det_floor=det_floor,
                )
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )

    # 2. Op-taxonomy tag mapping
    if op_tags:
        for tag in op_tags:
            if tag in _TAG_POLICY_MAP:
                policy = _TAG_POLICY_MAP[tag]
                return O15Report(
                    verdict="CLASSIFIED",
                    policy=policy,
                    source="op_taxonomy",
                    summary=f"DET_POLICY={policy} (tag={tag!r})",
                    det_floor=det_floor,
                )

    # 3. Default
    return O15Report(
        verdict="DEFAULTED",
        policy=DEFAULT_POLICY,
        source="default",
        summary=f"DET_POLICY={DEFAULT_POLICY} (default — no analysis.md hint)",
        det_floor=det_floor,
    )


_DET_POLICY_RE = re.compile(
    r"DET[_ ]?POLICY\s*[:=]\s*(required|best[_ ]?effort|n[_ /]?a)",
    re.IGNORECASE,
)


def _parse_det_policy_from_analysis(text: str) -> Optional[str]:
    """Extract DET_POLICY value from analysis.md text. Tolerant of
    formatting variants:
      DET_POLICY: required
      DET_POLICY = best_effort
      DETERMINISM POLICY: n/a
    """
    m = _DET_POLICY_RE.search(text)
    if not m:
        return None
    raw = m.group(1).lower().replace(" ", "_").replace("/", "_")
    # Normalize: best-effort / besteffort / best effort → best_effort
    if "best" in raw:
        return "best_effort"
    if raw in ("n_a", "na"):
        return "n_a"
    if raw == "required":
        return "required"
    return None


_DET_FLOOR_RE = re.compile(
    r"DET[_ ]?FLOOR\s*[:=]\s*(\d+)",
    re.IGNORECASE,
)


def _parse_det_floor_from_analysis(text: str) -> Optional[int]:
    """Extract an author-declared DET_FLOOR integer from analysis.md text.
    Tolerant of formatting variants:
      DET_FLOOR: 14
      DET FLOOR = 14
    Returns None when absent or unparseable (→ no tolerance declared)."""
    m = _DET_FLOOR_RE.search(text)
    if not m:
        return None
    try:
        val = int(m.group(1))
    except (TypeError, ValueError):
        return None
    return val if val >= 0 else None


def store_in_durable_state(
    workspace: Path,
    policy: str,
    det_floor: Optional[int] = None,
) -> None:
    """Update .opgen_state.json with the classified det_policy (and, when
    declared, the immutable det_floor). Caller typically chains
    classify_det_policy → store_in_durable_state.

    det_floor is written only when not None so undeclared ops leave the field
    absent (= no tolerance), never a fabricated 0.
    """
    import phase_o05
    state_path = workspace / phase_o05.STATE_FILE
    if not state_path.exists():
        return  # nothing to update; phase_o05 should run first
    try:
        import json
        state = json.loads(state_path.read_text())
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )
    else:
        state["det_policy"] = policy
        if det_floor is not None:
            state["det_floor"] = det_floor
        state_path.write_text(json.dumps(state, indent=2))
