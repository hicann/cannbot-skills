# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for `src/scripts/fa_class_schemas.py` — the FA-class JSON schemas.

Per design doc §6 Phase 1c. Verifies:
1. Each schema round-trips through JSON without loss
2. main R1 fold enforcement: LOW evidence_strength → MUST be RootCause.OTHER
3. HIGH/MEDIUM evidence MUST have non-empty evidence string
4. Enum values render as plain strings (not Enum reprs) in JSON output
5. Optional fields gracefully omit when None
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add src/scripts to sys.path so `import fa_class_schemas` works in test context
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from fa_class_schemas import (  # noqa: E402
    AlignmentAudit,
    AlignmentOperation,
    AlignmentVerdict,
    CaseVariantEntry,
    CaseVariantMap,
    DispatchStatus,
    EvidenceStrength,
    FixPhase,
    FixtureEvalErrTaxonomy,
    RootCause,
    TaxonomyEntry,
)


# ────────────────────────────────────────────────────────────────────
# case_variant_map
# ────────────────────────────────────────────────────────────────────

class TestCaseVariantMap:
    @staticmethod
    def test_roundtrip_full_capture():
        m = CaseVariantMap(
            op="3_FusionAttention",
            workspace="workspace/3_FusionAttention",
            captured_at="2026-05-22T12:34:56Z",
            msprof_version="8.0.0",
            cases=[
                CaseVariantEntry(
                    case_idx=3,
                    status=DispatchStatus.CAPTURED,
                    kernel_binary_name="flash_attention_score_s1s2_bn2gs1",
                    ub0=3, ub1=4, block=9,
                ),
            ],
        )
        text = m.to_json()
        m2 = CaseVariantMap.from_json(text)
        assert m2.op == m.op
        assert m2.cases[0].kernel_binary_name == "flash_attention_score_s1s2_bn2gs1"
        assert m2.cases[0].ub0 == 3
        assert m2.cases[0].status == DispatchStatus.CAPTURED

    @staticmethod
    def test_roundtrip_msprof_unavailable():
        m = CaseVariantMap(
            op="3_FusionAttention",
            workspace="workspace/3_FusionAttention",
            captured_at="2026-05-22T12:34:56Z",
            msprof_version=None,
            cases=[
                CaseVariantEntry(
                    case_idx=0,
                    status=DispatchStatus.MSPROF_UNAVAILABLE,
                    error="msprof binary not found in $PATH",
                ),
            ],
        )
        text = m.to_json()
        d = json.loads(text)
        # Optional fields should omit when None
        assert "kernel_binary_name" not in d["cases"][0]
        assert "ub0" not in d["cases"][0]
        # Error field present
        assert d["cases"][0]["error"] == "msprof binary not found in $PATH"
        # Roundtrip
        m2 = CaseVariantMap.from_json(text)
        assert m2.msprof_version is None
        assert m2.cases[0].status == DispatchStatus.MSPROF_UNAVAILABLE

    @staticmethod
    def test_enum_string_in_json():
        m = CaseVariantMap(
            op="op", workspace="ws", captured_at="2026-05-22T00:00:00Z",
            msprof_version=None,
            cases=[CaseVariantEntry(case_idx=0, status=DispatchStatus.CAPTURED,
                                    kernel_binary_name="x", ub0=1, ub1=2, block=3)],
        )
        d = json.loads(m.to_json())
        # status must be plain string "captured", not "DispatchStatus.CAPTURED"
        assert d["cases"][0]["status"] == "captured"


# ────────────────────────────────────────────────────────────────────
# fixture_eval_err_taxonomy
# ────────────────────────────────────────────────────────────────────

class TestTaxonomyEntry:
    @staticmethod
    def test_low_evidence_with_specific_cause_raises():
        """Main R1 fold: LOW evidence + non-OTHER root_cause must raise."""
        with pytest.raises(ValueError, match="LOW.*root_cause=UB_BUDGET"):
            TaxonomyEntry(
                case_idx=0,
                root_cause=RootCause.UB_BUDGET,
                evidence_strength=EvidenceStrength.LOW,
                fix_phase=FixPhase.V1_0,
                evidence="speculative — maybe UB",
            )

    @staticmethod
    def test_low_evidence_with_other_is_ok():
        """LOW + OTHER is the canonical 'I don't know' classification."""
        entry = TaxonomyEntry(
            case_idx=0,
            root_cause=RootCause.OTHER,
            evidence_strength=EvidenceStrength.LOW,
            fix_phase=FixPhase.OUT_OF_SCOPE,
            evidence="no traceback captured",
        )
        assert entry.root_cause == RootCause.OTHER

    @staticmethod
    def test_high_evidence_empty_string_raises():
        with pytest.raises(ValueError, match="evidence is empty"):
            TaxonomyEntry(
                case_idx=0,
                root_cause=RootCause.UB_BUDGET,
                evidence_strength=EvidenceStrength.HIGH,
                fix_phase=FixPhase.V1_0,
                evidence="",
            )

    @staticmethod
    def test_medium_evidence_empty_string_raises():
        with pytest.raises(ValueError, match="evidence is empty"):
            TaxonomyEntry(
                case_idx=0,
                root_cause=RootCause.DTYPE_TEMPLATE,
                evidence_strength=EvidenceStrength.MEDIUM,
                fix_phase=FixPhase.V1_0,
                evidence="   ",  # whitespace only
            )

    @staticmethod
    def test_high_evidence_with_traceback_ok():
        entry = TaxonomyEntry(
            case_idx=5,
            root_cause=RootCause.UB_BUDGET,
            evidence_strength=EvidenceStrength.HIGH,
            fix_phase=FixPhase.V1_0,
            evidence="_OutOfScope: S*D = 128*128 = 16384 > UB budget 8192 (case 5)",
        )
        assert entry.case_idx == 5


class TestFixtureEvalErrTaxonomy:
    @staticmethod
    def test_roundtrip():
        t = FixtureEvalErrTaxonomy(
            op="3_FusionAttention",
            workspace="workspace/3_FusionAttention",
            classified_at="2026-05-22T13:00:00Z",
            total_cases=61,
            cases=[
                TaxonomyEntry(
                    case_idx=4,
                    root_cause=RootCause.UB_BUDGET,
                    evidence_strength=EvidenceStrength.HIGH,
                    fix_phase=FixPhase.V1_0,
                    evidence="case_idx=4 OutOfScope S*D=128*128>8192",
                ),
                TaxonomyEntry(
                    case_idx=11,
                    root_cause=RootCause.DTYPE_TEMPLATE,
                    evidence_strength=EvidenceStrength.HIGH,
                    fix_phase=FixPhase.V1_0,
                    evidence="case_idx=11 OutOfScope dtype torch.bfloat16",
                ),
            ],
        )
        text = t.to_json()
        t2 = FixtureEvalErrTaxonomy.from_json(text)
        assert t2.total_cases == 61
        assert len(t2.cases) == 2
        assert t2.cases[0].root_cause == RootCause.UB_BUDGET
        assert t2.cases[1].fix_phase == FixPhase.V1_0

    @staticmethod
    def test_enum_string_in_json():
        t = FixtureEvalErrTaxonomy(
            op="op", workspace="ws", classified_at="2026-05-22T00:00:00Z",
            total_cases=1,
            cases=[TaxonomyEntry(case_idx=0, root_cause=RootCause.UB_BUDGET,
                                 evidence_strength=EvidenceStrength.HIGH,
                                 fix_phase=FixPhase.V1_0,
                                 evidence="x")],
        )
        d = json.loads(t.to_json())
        assert d["cases"][0]["root_cause"] == "UB_BUDGET"
        assert d["cases"][0]["evidence_strength"] == "HIGH"
        assert d["cases"][0]["fix_phase"] == "v1.0"


# ────────────────────────────────────────────────────────────────────
# alignment_audit
# ────────────────────────────────────────────────────────────────────

class TestAlignmentAudit:
    @staticmethod
    def test_summary_counts():
        a = AlignmentAudit(
            brief_path="workspace/3_FusionAttention/brief.md",
            audited_at="2026-05-22T14:00:00Z",
            operations=[
                AlignmentOperation(line=10, operation="DataCopy", text="DataCopy(d, s, 256)",
                                   verdict=AlignmentVerdict.ALIGNED,
                                   count=256, dtype="fp16", elements_per_32b=16),
                AlignmentOperation(line=20, operation="Add", text="Add(d, a, b, 13)",
                                   verdict=AlignmentVerdict.MISALIGNED,
                                   count=13, dtype="fp16", elements_per_32b=16, remainder=13),
                AlignmentOperation(line=30, operation="Cast", text="Cast(d, s, ...)",
                                   verdict=AlignmentVerdict.INDETERMINATE,
                                   reason="could not extract count from variadic Cast"),
            ],
        )
        assert a.n_aligned == 1
        assert a.n_misaligned == 1
        assert a.n_indeterminate == 1
        d = json.loads(a.to_json())
        assert d["summary"]["n_aligned"] == 1
        assert d["summary"]["n_misaligned"] == 1
        assert d["summary"]["n_indeterminate"] == 1
        assert d["summary"]["n_operations"] == 3

    @staticmethod
    def test_misaligned_includes_remainder():
        op = AlignmentOperation(
            line=5, operation="DataCopy", text="DataCopy(d, s, 17)",
            verdict=AlignmentVerdict.MISALIGNED,
            count=17, dtype="fp16", elements_per_32b=16, remainder=1,
        )
        d = op.to_dict()
        assert d["remainder"] == 1
        assert d["verdict"] == "misaligned"

    @staticmethod
    def test_aligned_no_remainder_field():
        op = AlignmentOperation(
            line=5, operation="DataCopy", text="DataCopy(d, s, 256)",
            verdict=AlignmentVerdict.ALIGNED,
            count=256, dtype="fp16", elements_per_32b=16,
        )
        d = op.to_dict()
        assert "remainder" not in d  # None should be omitted
        assert d["count"] == 256
