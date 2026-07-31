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

"""FA-class JSON schemas — frozen contracts for Phase 1 artifacts.

Per `docs/design/FA_CLASS_PROBLEM_SOLUTION_DESIGN.md` §6 Phase 1a
(merged 2026-05-22 commit `cfcd0cd5` PR #122).

Three artifact schemas, each consumed by the FA-class plugin's
finalize hooks + kw_brief_phase_block:

1. **case_variant_map.json** — msprof-derived per-case dispatch capture.
   Records which CANN variant (s1s2_bn2gs1 / s1s2_bn2gs1_sab / etc.)
   each fixture case actually routes to at runtime (UB0/UB1/Block
   tuple from host tiling). Required by Phase 1 gate #8.

2. **fixture_eval_err_taxonomy.json** — per-case root cause + fix phase
   + evidence_strength. Forces honest accounting per main R1 fold:
   LOW-confidence entries MUST classify as OTHER (not predicted-fix).
   Required by Phase 1 gate #7 + #9 (brief-time precondition).

3. **alignment_audit.json** — per-operation alignment check on
   kernel brief markdown. Catches V220 ADDR_MISALIGN / 507035 at
   brief-review time vs A3-test time. Required by DS round-1 addition.

This module is the **single source of truth** for these contracts.
fa_class_taxonomy_builder.py / fa_class_msprof_capture.py /
fa_class_self_check.py / fa_class_alignment_audit.py all import
from here. Tests at src/scripts/tests/test_fa_class_schemas.py.

No runtime dependencies beyond stdlib (dataclasses + json).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ────────────────────────────────────────────────────────────────────
# §1. case_variant_map.json (msprof capture)
# ────────────────────────────────────────────────────────────────────

class DispatchStatus(str, Enum):
    """msprof capture outcome per case."""
    CAPTURED = "captured"           # msprof ran + parsed cleanly
    MSPROF_UNAVAILABLE = "msprof_unavailable"  # msprof binary missing — graceful degrade
    KERNEL_CRASHED = "kernel_crashed"  # kernel binary itself crashed before msprof could record
    PARSE_FAILED = "parse_failed"   # msprof output present but unparseable


@dataclass
class CaseVariantEntry:
    """Per-case msprof dispatch capture."""
    case_idx: int
    status: DispatchStatus
    # Populated when status == CAPTURED:
    kernel_binary_name: Optional[str] = None     # e.g. "flash_attention_score_s1s2_bn2gs1"
    ub0: Optional[int] = None                     # host tiling UB0 axis
    ub1: Optional[int] = None                     # host tiling UB1 axis
    block: Optional[int] = None                   # host tiling Block axis
    # Populated when status indicates failure:
    error: Optional[str] = None                   # human-readable diagnostic

    def to_dict(self) -> dict:
        d = asdict(self)
        # Enum value is already str by inheritance
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class CaseVariantMap:
    """Full case_variant_map.json — list of per-case captures."""
    op: str                          # e.g. "3_FusionAttention"
    workspace: str                   # relative path
    captured_at: str                 # ISO8601 timestamp
    msprof_version: Optional[str]    # None when MSPROF_UNAVAILABLE for all
    cases: list[CaseVariantEntry] = field(default_factory=list)

    @classmethod
    def from_json(cls, text: str) -> "CaseVariantMap":
        d = json.loads(text)
        cases = [
            CaseVariantEntry(
                case_idx=c["case_idx"],
                status=DispatchStatus(c["status"]),
                kernel_binary_name=c.get("kernel_binary_name"),
                ub0=c.get("ub0"),
                ub1=c.get("ub1"),
                block=c.get("block"),
                error=c.get("error"),
            )
            for c in d.get("cases", [])
        ]
        return cls(
            op=d["op"],
            workspace=d["workspace"],
            captured_at=d["captured_at"],
            msprof_version=d.get("msprof_version"),
            cases=cases,
        )

    def to_json(self) -> str:
        return json.dumps({
            "op": self.op,
            "workspace": self.workspace,
            "captured_at": self.captured_at,
            "msprof_version": self.msprof_version,
            "cases": [c.to_dict() for c in self.cases],
        }, indent=2)


# ────────────────────────────────────────────────────────────────────
# §2. fixture_eval_err_taxonomy.json (evidence-graded root cause)
# ────────────────────────────────────────────────────────────────────

class RootCause(str, Enum):
    """Per-case EVAL_ERR root cause classification.

    Per main R1 fold (PR #122 amendment): LOW-confidence claims MUST
    classify as OTHER. Forces explicit evidence-vs-speculation split.
    """
    UB_BUDGET = "UB_BUDGET"            # S*D > UB capacity → needs KV-tiling
    DTYPE_TEMPLATE = "DTYPE_TEMPLATE"  # bf16/fp32/etc unsupported → needs template
    MASK_BROADCAST = "MASK_BROADCAST"  # atten_mask shape mismatch
    PSE = "PSE"                        # pse pre-bias not in kernel
    GQA_REFERENCE = "GQA_REFERENCE"    # Python reference itself crashes (out of scope)
    OTHER = "OTHER"                    # Catch-all for LOW evidence or unknown


class EvidenceStrength(str, Enum):
    """Evidence-grading per case classification (main R1 fold)."""
    HIGH = "HIGH"      # traceback line + exact file:line pointer in evidence
    MEDIUM = "MEDIUM"  # heuristic match (e.g. "S*D=N > UB budget" arithmetic check)
    LOW = "LOW"        # speculation — MUST be reclassified to RootCause.OTHER


class FixPhase(str, Enum):
    """Which v-phase actually fixes this case."""
    V1_0 = "v1.0"      # Path A + KV-tile + bf16/fp32 templates + case 6 hard gate
    V1_1 = "v1.1"      # + SAB variant code
    V1_2 = "v1.2"      # + pse handling
    V2 = "v2"          # IterateAll<false> + extraInfo[3] (perf, not unblock)
    V3 = "v3"          # cross-op Pipeline B
    OUT_OF_SCOPE = "out_of_scope"  # e.g. GQA where reference crashes


@dataclass
class TaxonomyEntry:
    """Per-case classification — schema enforces HIGH/MEDIUM evidence_strength
    is paired with predicted-fix bucket; LOW MUST go to OTHER.
    """
    case_idx: int
    root_cause: RootCause
    evidence_strength: EvidenceStrength
    fix_phase: FixPhase
    evidence: str  # MUST be non-empty traceback line / log signature / file:line

    def __post_init__(self):
        # Enforce main R1 fold: LOW evidence → root_cause MUST be OTHER
        if (
            self.evidence_strength == EvidenceStrength.LOW
            and self.root_cause != RootCause.OTHER
        ):
            raise ValueError(
                f"case_idx={self.case_idx}: evidence_strength=LOW but "
                f"root_cause={self.root_cause.value} (must be OTHER per main R1 fold). "
                f"LOW evidence does not justify a specific predicted-fix bucket."
            )
        # Enforce evidence string non-empty (HIGH/MEDIUM require concrete citation)
        if (
            self.evidence_strength in (EvidenceStrength.HIGH, EvidenceStrength.MEDIUM)
            and not self.evidence.strip()
        ):
            raise ValueError(
                f"case_idx={self.case_idx}: evidence_strength="
                f"{self.evidence_strength.value} but evidence is empty. "
                f"HIGH/MEDIUM requires concrete citation."
            )

    def to_dict(self) -> dict:
        return {
            "case_idx": self.case_idx,
            "root_cause": self.root_cause.value,
            "evidence_strength": self.evidence_strength.value,
            "fix_phase": self.fix_phase.value,
            "evidence": self.evidence,
        }


@dataclass
class FixtureEvalErrTaxonomy:
    """Full fixture_eval_err_taxonomy.json — list of per-case classifications.

    Brief-time precondition (gate #9): file MUST exist + non-empty cases
    list before kw_brief_phase_block composes a FA-class brief.
    """
    op: str
    workspace: str
    classified_at: str          # ISO8601
    total_cases: int            # total fixture size (e.g. 61)
    cases: list[TaxonomyEntry] = field(default_factory=list)

    @classmethod
    def from_json(cls, text: str) -> "FixtureEvalErrTaxonomy":
        d = json.loads(text)
        cases = [
            TaxonomyEntry(
                case_idx=c["case_idx"],
                root_cause=RootCause(c["root_cause"]),
                evidence_strength=EvidenceStrength(c["evidence_strength"]),
                fix_phase=FixPhase(c["fix_phase"]),
                evidence=c["evidence"],
            )
            for c in d.get("cases", [])
        ]
        return cls(
            op=d["op"],
            workspace=d["workspace"],
            classified_at=d["classified_at"],
            total_cases=d["total_cases"],
            cases=cases,
        )

    def to_json(self) -> str:
        return json.dumps({
            "op": self.op,
            "workspace": self.workspace,
            "classified_at": self.classified_at,
            "total_cases": self.total_cases,
            "cases": [c.to_dict() for c in self.cases],
        }, indent=2)


# ────────────────────────────────────────────────────────────────────
# §3. alignment_audit.json (brief-time alignment check, DS round-1 addition)
# ────────────────────────────────────────────────────────────────────

class AlignmentVerdict(str, Enum):
    """Per-operation alignment audit verdict."""
    ALIGNED = "aligned"              # count satisfies 32B alignment for target dtype
    MISALIGNED = "misaligned"        # count NOT 32B-aligned — V220 will hit 507035
    INDETERMINATE = "indeterminate"  # could not extract count parameter for static check


@dataclass
class AlignmentOperation:
    """Per-operation alignment audit entry."""
    line: int                      # line in brief markdown
    operation: str                  # e.g. "DataCopy", "Cast", "Add", "Max", "Mul"
    text: str                       # the source line text (truncated to 200 chars)
    verdict: AlignmentVerdict
    # Populated when ALIGNED or MISALIGNED:
    count: Optional[int] = None              # extracted count parameter
    dtype: Optional[str] = None              # e.g. "fp16", "bf16", "fp32"
    elements_per_32b: Optional[int] = None   # 16 for fp16/bf16, 8 for fp32
    # Populated when MISALIGNED:
    remainder: Optional[int] = None          # count % elements_per_32b (non-zero)
    # Populated when INDETERMINATE:
    reason: Optional[str] = None             # why we couldn't extract

    def to_dict(self) -> dict:
        d = {
            "line": self.line,
            "operation": self.operation,
            "text": self.text,
            "verdict": self.verdict.value,
        }
        for k in ("count", "dtype", "elements_per_32b", "remainder", "reason"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


@dataclass
class AlignmentAudit:
    """Full alignment_audit.json — list of per-operation findings on a brief markdown."""
    brief_path: str
    audited_at: str               # ISO8601
    operations: list[AlignmentOperation] = field(default_factory=list)

    @property
    def n_aligned(self) -> int:
        return sum(1 for o in self.operations if o.verdict == AlignmentVerdict.ALIGNED)

    @property
    def n_misaligned(self) -> int:
        return sum(1 for o in self.operations if o.verdict == AlignmentVerdict.MISALIGNED)

    @property
    def n_indeterminate(self) -> int:
        return sum(
            1 for o in self.operations if o.verdict == AlignmentVerdict.INDETERMINATE
        )

    def to_json(self) -> str:
        return json.dumps({
            "brief_path": self.brief_path,
            "audited_at": self.audited_at,
            "summary": {
                "n_operations": len(self.operations),
                "n_aligned": self.n_aligned,
                "n_misaligned": self.n_misaligned,
                "n_indeterminate": self.n_indeterminate,
            },
            "operations": [o.to_dict() for o in self.operations],
        }, indent=2)


# ────────────────────────────────────────────────────────────────────
# §4. Public API surface — what fa_class_*.py imports
# ────────────────────────────────────────────────────────────────────

__all__ = [
    # case_variant_map
    "DispatchStatus", "CaseVariantEntry", "CaseVariantMap",
    # fixture_eval_err_taxonomy
    "RootCause", "EvidenceStrength", "FixPhase", "TaxonomyEntry",
    "FixtureEvalErrTaxonomy",
    # alignment_audit
    "AlignmentVerdict", "AlignmentOperation", "AlignmentAudit",
]
