# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Tests for `is_fa_class()` / `is_l4_fused()` helpers in `plugins/base.py`.

Per design doc §3.2 + main R1+B1 folds (PR #122 amendment). Verifies:
1. Substring (not enum) matching — uppercase space-joined tag-string
2. Case-insensitivity
3. Graceful on empty / None
4. is_fa_class implies is_l4_fused (subset relation)
5. Specific taxonomy strings observed in current code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add src/scripts to sys.path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from orchestrator.plugins.base import is_fa_class, is_l4_fused  # noqa: E402


class TestIsFaClass:
    @staticmethod
    def test_canonical_fa_op_class_matches():
        # task#31 (2026-06-04): FA-class keys on the ATTENTION structural tag
        assert is_fa_class("FUSED SOFTMAX ATTENTION TRANSCENDENTAL REDUCTION")

    @staticmethod
    def test_fused_softmax_without_attention_rejects():
        # task#31 narrowing: FUSED+SOFTMAX alone is NOT FA-class — this is the
        # Sinkhorn-shaped case (softmax normalization, no QK^T·V attention
        # structure) that the old `FUSED + SOFTMAX` heuristic mis-fired on.
        assert not is_fa_class("FUSED SOFTMAX")

    @staticmethod
    def test_case_insensitive_matches():
        assert is_fa_class("fused softmax attention")
        assert is_fa_class("Fused Softmax Attention Transcendental")

    @staticmethod
    def test_extra_tags_still_match():
        assert is_fa_class("FUSED SOFTMAX ATTENTION EXTRA TAGS")

    @staticmethod
    def test_missing_softmax_rejects():
        assert not is_fa_class("FUSED ELEMENTWISE")
        assert not is_fa_class("FUSED MOE")
        assert not is_fa_class("FUSED REDUCTION")

    @staticmethod
    def test_missing_fused_rejects():
        assert not is_fa_class("SOFTMAX TRANSCENDENTAL")
        assert not is_fa_class("REDUCTION")

    @staticmethod
    def test_empty_rejects():
        assert not is_fa_class("")

    @staticmethod
    def test_none_graceful():
        # Schema occasionally passes None when op_class detection fails
        assert not is_fa_class(None)

    @staticmethod
    def test_implies_l4_fused():
        """is_fa_class(x) MUST imply is_l4_fused(x)."""
        positives = [
            "FUSED SOFTMAX ATTENTION",
            "FUSED SOFTMAX ATTENTION TRANSCENDENTAL REDUCTION",
            "fused softmax attention extra",
        ]
        for s in positives:
            assert is_fa_class(s), f"baseline broke: is_fa_class({s!r})"
            assert is_l4_fused(s), f"is_fa_class({s!r}) but not is_l4_fused"


class TestIsL4Fused:
    @staticmethod
    def test_fa_class_matches():
        assert is_l4_fused("FUSED SOFTMAX TRANSCENDENTAL REDUCTION")

    @staticmethod
    def test_other_fused_classes_match():
        # Per design §3.1: covers fused_attention_grad / fused_moe / fused_norm_matmul
        assert is_l4_fused("FUSED MOE GATING")
        assert is_l4_fused("FUSED NORM MATMUL")
        assert is_l4_fused("FUSED ATTENTION GRAD")

    @staticmethod
    def test_case_insensitive():
        assert is_l4_fused("fused attention grad")
        assert is_l4_fused("Fused Moe")

    @staticmethod
    def test_no_fused_tag_rejects():
        assert not is_l4_fused("ELEMENTWISE")
        assert not is_l4_fused("REDUCTION")
        assert not is_l4_fused("SCAN")
        assert not is_l4_fused("SOFTMAX")  # no FUSED prefix

    @staticmethod
    def test_empty_rejects():
        assert not is_l4_fused("")

    @staticmethod
    def test_none_graceful():
        assert not is_l4_fused(None)

    @staticmethod
    def test_strict_superset_of_fa_class():
        """Verify is_l4_fused covers more than just FA-class."""
        # is_l4_fused True, is_fa_class False
        broader_only = [
            "FUSED MOE",
            "FUSED NORM",
            "FUSED ELEMENTWISE",
        ]
        for s in broader_only:
            assert is_l4_fused(s), f"is_l4_fused({s!r}) should be True"
            assert not is_fa_class(s), f"is_fa_class({s!r}) should be False (no SOFTMAX)"


class TestConformanceWithExistingCode:
    """Verify helpers match the substring patterns used by scoped gates."""

    @staticmethod
    def test_fa_class_uses_attention_tag_not_fused_softmax():
        """task#31 (2026-06-04): is_fa_class keys on the ATTENTION structural tag,
        NOT the old '"FUSED" in op_class AND "SOFTMAX" in op_class' heuristic (which
        mis-fired for the Sinkhorn-shaped `hc_split_sinkhorn`). The FA-class routing
        gate deliberately stays op-name-only (task#28) and is
        no longer equivalent to is_fa_class.
        """
        # ATTENTION-tagged (QK^T·V structure) → FA-class
        assert is_fa_class("FUSED SOFTMAX ATTENTION TRANSCENDENTAL REDUCTION")
        # Sinkhorn-shaped (FUSED+SOFTMAX, no ATTENTION) → NOT FA-class
        assert not is_fa_class("FUSED SOFTMAX TRANSCENDENTAL REDUCTION")
