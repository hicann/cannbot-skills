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

"""Validate verification.json against pre-done schema contract.

Replaces the inline `python3 -c "..."` assertion block in kw_brief.py
port_a3 phase block (size-cap pressure 2026-05-23, P0abm/P0cc fold).

Usage: python3 check_verification_schema.py workspace/<op>/verification.json

Exit codes:
  0 — schema valid
  1 — schema violation (message on stderr)

Checks:
  - precision.pass_a.tier1_pass is int
  - precision.pass_a.total is int
  - if precision.pass_a.status in INCLUSIVE_STATUSES:
      precision.pass_a.tier1_pass_inclusive is int (P0cc)
  - precision.pass_b is either:
      {tier1_pass: int, total: int} OR {status: "N/A", reason: <str>}

P0cc dual-count rationale: src/skills/references/shared/GATE_CONTRACT.md §P0cc
EC-59 (incident codification): src/skills/references/target/ascendc/ERROR_CORRECTIONS.md
Memory: feedback_no_patch_fix_harness_for_next_customer.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

INCLUSIVE_STATUSES = (
    "PASS_WITHIN_TOLERANCE",
    "PARTIAL_PASS_WITHIN_TOLERANCE",
    "PARTIAL_PASS",
)


def check(vp: Path) -> tuple[bool, str]:
    if not vp.exists():
        return False, f"verification.json not found: {vp}"
    try:
        v = json.loads(vp.read_text())
    except Exception as e:
        return False, f"verification.json malformed: {e}"
    prec = v.get("precision", {}) or {}
    pa = prec.get("pass_a", {})
    if not isinstance(pa.get("tier1_pass"), int):
        return False, "precision.pass_a.tier1_pass must be int"
    if not isinstance(pa.get("total"), int):
        return False, "precision.pass_a.total must be int"
    pa_status = pa.get("status", "")
    if pa_status in INCLUSIVE_STATUSES:
        if not isinstance(pa.get("tier1_pass_inclusive"), int):
            return False, (
                f"P0cc: precision.pass_a.status='{pa_status}' requires "
                f"integer tier1_pass_inclusive field "
                f"(T1_BIT_EXACT + T2_PASS_WITHIN_TOLERANCE count). "
                f"See src/skills/references/shared/GATE_CONTRACT.md §P0cc."
            )
    # task#82 (2026-06-14): port_a3 native two-tier verdict. When the pass_a block
    # carries the native tier2 fields (emitted by precision_eval_port_a3_two_tier),
    # validate their types so the two-tier schema is well-formed and consistent
    # with benchmark's two-tier shape. The fields are OPTIONAL (older single-tier
    # port_a3 archives + all benchmark archives omit them); validate only when
    # present so existing archives stay valid.
    if "tier2_pass" in pa or "tier2_status" in pa:
        if not isinstance(pa.get("tier2_pass"), int):
            return False, (
                "task#82: precision.pass_a.tier2_pass must be int when the native "
                "two-tier fields are present"
            )
        VALID_TIER2_STATUS = (
            "PASS", "FAIL", "A3_UNAVAILABLE", "N/A_ALL_T1",
            "N/A_T1_PASS", "N/A_NO_CASES",
            # 生态 DEFAULT (owner 2026-06-30): the compare.py path grades vs cpu_truth + native and
            # does NOT use the A3 competitor → T2 not applicable. Per-case emits N/A_ECOSYSTEM; accept
            # it at op-level too for forward-compat (codex #3).
            "N/A_ECOSYSTEM",
        )
        if pa.get("tier2_status") not in VALID_TIER2_STATUS:
            return False, (
                f"task#82: precision.pass_a.tier2_status='{pa.get('tier2_status')}' "
                f"must be one of {VALID_TIER2_STATUS}"
            )
        # When tier2 fields exist, the inclusive count must also exist (T1+T2) so
        # the phase_o5 MISMATCH gate compares like for like.
        if not isinstance(pa.get("tier1_pass_inclusive"), int):
            return False, (
                "task#82: precision.pass_a with native tier2 fields requires "
                "integer tier1_pass_inclusive (tier1_pass + tier2_pass)"
            )
    pb = prec.get("pass_b", {})
    pb_valid = (
        isinstance(pb.get("tier1_pass"), int)
        and isinstance(pb.get("total"), int)
    )
    pb_na = pb.get("status") == "N/A" and pb.get("reason")
    if not (pb_valid or pb_na):
        return False, (
            "pass_b must have integer tier1_pass+total OR "
            "status='N/A' with reason"
        )
    if pb_na and (vp.parent / "pass_b_runner.py").exists():
        return False, (
            "precision.pass_b.status is N/A but pass_b_runner.py exists; "
            "delete the runner for a true N/A mode or make pass_b real PASS/FAIL"
        )
    return True, "schema OK"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <verification.json>", file=sys.stderr)
        return 2
    ok, msg = check(Path(sys.argv[1]))
    print(msg, file=sys.stderr if not ok else sys.stdout)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
