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

"""FA-class fixture EVAL_ERR taxonomy builder — Phase 1 skeleton.

Per design §6 Phase 1b: classifies each fixture EVAL_ERR per evidence_strength
schema (HIGH / MEDIUM / LOW). LOW-confidence cases MUST classify as
RootCause.OTHER per main R1 fold — schema enforces.

**Phase 1 skeleton scope**: this script reads an existing
`canonical_p2t_summary.json` (output of `precision_eval_two_tier.py`) and
classifies each EVAL_ERR case via STATIC traceback substring matching
(no kernel execution needed). The output writes `fixture_eval_err_taxonomy.json`
conforming to `fa_class_schemas.FixtureEvalErrTaxonomy`.

**Phase 1.5 follow-up (NOT IN THIS COMMIT)**: extend to also fire the
kernel on NPU + capture richer EVAL_ERR signatures when canonical_p2t_summary
is stale or missing. Requires A3 container access + msprof, deferred to
Phase 2 / cold-start workflow.

Static classification rules (canonical, per design §3.3 + observed cases):
- 'UB budget' / 'S*D = N > UB budget' → UB_BUDGET / V1_0 / HIGH
- 'dtype torch.bfloat16 not supported' → DTYPE_TEMPLATE / V1_0 / HIGH
- 'dtype torch.float32 not supported' → DTYPE_TEMPLATE / V1_0 / HIGH
- 'pse pre-bias' → PSE / V1_2 / HIGH
- 'GQA (n_q != n_kv)' → GQA_REFERENCE / OUT_OF_SCOPE / HIGH
- 'S*Skv = N > UB scores budget' → UB_BUDGET / V1_0 / HIGH
- non-OutOfScope error (any other exception) → OTHER / OUT_OF_SCOPE / LOW

Usage:
  python3 src/scripts/fa_class_taxonomy_builder.py <workspace> [--p2t-summary <path>] \\
      [--out <path>] [--quiet]

Exit codes:
  0 — taxonomy built + written
  1 — input file unreadable / no eval_err cases / schema validation failed
  2 — usage error
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from fa_class_schemas import (  # noqa: E402
    EvidenceStrength, FixPhase, FixtureEvalErrTaxonomy, RootCause, TaxonomyEntry,
)


# Static classification rules. Order matters — first match wins.
# Each entry: (regex_pattern, root_cause, fix_phase, evidence_strength)
_RULES = [
    (r"UB budget|UB scores budget|S\*D\s*=\s*\d+\*\d+\s*=\s*\d+\s*>",
     RootCause.UB_BUDGET, FixPhase.V1_0, EvidenceStrength.HIGH),
    (r"dtype torch\.bfloat16 not supported", RootCause.DTYPE_TEMPLATE, FixPhase.V1_0, EvidenceStrength.HIGH),
    (r"dtype torch\.float32 not supported", RootCause.DTYPE_TEMPLATE, FixPhase.V1_0, EvidenceStrength.HIGH),
    (r"pse pre-bias", RootCause.PSE, FixPhase.V1_2, EvidenceStrength.HIGH),
    (r"GQA \(n_q != n_kv\)", RootCause.GQA_REFERENCE, FixPhase.OUT_OF_SCOPE, EvidenceStrength.HIGH),
    (r"atten[_\s]?mask|mask broadcast", RootCause.MASK_BROADCAST, FixPhase.V1_0, EvidenceStrength.MEDIUM),
]


def classify_case(case_idx: int, error_text: str) -> TaxonomyEntry:
    """Classify a single EVAL_ERR case. LOW evidence MUST be OTHER per schema."""
    for pattern, root_cause, fix_phase, strength in _RULES:
        if re.search(pattern, error_text, re.IGNORECASE):
            return TaxonomyEntry(
                case_idx=case_idx,
                root_cause=root_cause,
                evidence_strength=strength,
                fix_phase=fix_phase,
                evidence=error_text[:300],
            )
    # No rule matched — LOW evidence → OTHER per schema enforcement
    return TaxonomyEntry(
        case_idx=case_idx,
        root_cause=RootCause.OTHER,
        evidence_strength=EvidenceStrength.LOW,
        fix_phase=FixPhase.OUT_OF_SCOPE,
        evidence=(error_text[:300] or "(empty error string)"),
    )


def build_taxonomy(workspace: Path, p2t_summary_path: Path) -> FixtureEvalErrTaxonomy:
    """Build taxonomy from canonical_p2t_summary.json output."""
    if not p2t_summary_path.exists():
        raise FileNotFoundError(f"p2t summary not found: {p2t_summary_path}")
    d = json.loads(p2t_summary_path.read_text())
    results = d.get("results", [])
    total = d.get("n_total", len(results))

    cases = []
    for r in results:
        if r.get("verdict") != "EVAL_ERR":
            continue
        case_idx = r.get("case_idx")
        if case_idx is None:
            # Some p2t outputs put case_idx in `case` field
            case_idx = r.get("case", {}).get("case_idx") if isinstance(r.get("case"), dict) else None
        if case_idx is None:
            case_idx = -1  # synthetic; will read as "unidentified case"
        error_text = r.get("error", "") or ""
        cases.append(classify_case(case_idx, error_text))

    return FixtureEvalErrTaxonomy(
        op=workspace.name,
        workspace=str(workspace),
        classified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        total_cases=total,
        cases=cases,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FA-class fixture EVAL_ERR taxonomy builder (Phase 1 skeleton, static-rule)"
    )
    parser.add_argument(
        "workspace", type=Path,
        help="Path to workspace dir (e.g. workspace/3_FusionAttention)",
    )
    parser.add_argument(
        "--p2t-summary", type=Path, default=None,
        help="Path to canonical_p2t_summary.json (default: <workspace>/canonical_p2t_summary.json)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output path (default: <workspace>/fixture_eval_err_taxonomy.json)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress non-error stdout output",
    )
    args = parser.parse_args(argv)

    p2t_path = args.p2t_summary or (args.workspace / "canonical_p2t_summary.json")
    out_path = args.out or (args.workspace / "fixture_eval_err_taxonomy.json")

    try:
        taxonomy = build_taxonomy(args.workspace, p2t_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out_path.write_text(taxonomy.to_json(), encoding="utf-8")

    # Summary
    counts = {}
    for c in taxonomy.cases:
        k = (c.root_cause.value, c.fix_phase.value, c.evidence_strength.value)
        counts[k] = counts.get(k, 0) + 1

    if not args.quiet:
        print(f"wrote {out_path}")
        print(f"  total_cases: {taxonomy.total_cases}")
        print(f"  classified: {len(taxonomy.cases)} EVAL_ERR")
        for (cause, phase, strength), n in sorted(counts.items()):
            print(f"    {n:>3}  {cause:<18} {phase:<14} {strength}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
