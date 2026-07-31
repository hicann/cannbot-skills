# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Unit tests for kb_schema.py — NODE-21 Phase A."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from kb_schema import (
    _TIER_SPECIFICITY_ORDER,
    VALID_VALUES,
    format_tag_column,
    kb_entry_applies,
    parse_applies_to,
    parse_tag_column,
    sort_by_specificity,
    specificity_score,
)


# ---- parse_applies_to ----

def test_parse_simple():
    tags = parse_applies_to("paradigm: ascendc\narch_family: arch35")
    assert tags["paradigm"] == "ascendc"
    assert tags["arch_family"] == "arch35"


def test_parse_with_any():
    tags = parse_applies_to("paradigm: ascendc\narch_family: any\nnpu_arch: 3510")
    assert tags["arch_family"] == "any"


def test_parse_with_comments():
    tags = parse_applies_to("# comment\nparadigm: ascendc  # inline\narch_family: arch35")
    assert tags["paradigm"] == "ascendc"


def test_parse_empty():
    assert parse_applies_to("") == {}


def test_parse_strips_plus():
    tags = parse_applies_to("cann: 9.0.0+")
    assert tags["cann"] == "9.0.0"


# ---- kb_entry_applies ----

def test_applies_exact_match():
    entry = {"arch_family": "arch35", "paradigm": "ascendc"}
    target = {"arch_family": "arch35", "paradigm": "ascendc"}
    assert kb_entry_applies(entry, target)


def test_applies_any():
    entry = {"arch_family": "any"}
    target = {"arch_family": "arch35"}
    assert kb_entry_applies(entry, target)


def test_applies_absent_tier():
    entry = {"arch_family": "arch35"}
    target = {"arch_family": "arch35", "npu_arch": "3510"}
    assert kb_entry_applies(entry, target)


def test_applies_mismatch():
    entry = {"arch_family": "arch35"}
    target = {"arch_family": "arch22"}
    assert not kb_entry_applies(entry, target)


def test_applies_entry_more_specific_than_target():
    """Entry declares soc=Ascend950PR, target doesn't know soc → include (conservative)."""
    entry = {"arch_family": "arch35", "soc": "Ascend950PR"}
    target = {"arch_family": "arch35"}
    assert kb_entry_applies(entry, target)


def test_applies_empty_entry():
    """Empty entry = universal scope."""
    assert kb_entry_applies({}, {"arch_family": "arch35"})


def test_applies_multi_tier_match():
    entry = {"paradigm": "ascendc", "arch_family": "arch35", "npu_arch": "3510", "cann": "9.0.0"}
    target = {"paradigm": "ascendc", "arch_family": "arch35", "npu_arch": "3510", "cann": "9.0.0"}
    assert kb_entry_applies(entry, target)


def test_applies_multi_tier_mismatch():
    entry = {"paradigm": "ascendc", "arch_family": "arch35", "npu_arch": "3510"}
    target = {"paradigm": "ascendc", "arch_family": "arch35", "npu_arch": "3003"}
    assert not kb_entry_applies(entry, target)


# ---- specificity_score ----

def test_specificity_more_tiers_higher():
    broad = {"paradigm": "ascendc"}
    narrow = {"paradigm": "ascendc", "arch_family": "arch35", "soc": "Ascend950PR"}
    assert specificity_score(narrow) > specificity_score(broad)


def test_specificity_any_is_zero():
    e1 = {"arch_family": "arch35"}
    e2 = {"arch_family": "any"}
    assert specificity_score(e1) > specificity_score(e2)


def test_specificity_empty_is_zero():
    assert specificity_score({}) == 0


# ---- sort_by_specificity ----

def test_sort_narrow_first():
    entries = [
        ({"paradigm": "ascendc"}, "broad"),
        ({"paradigm": "ascendc", "arch_family": "arch35", "soc": "Ascend950PR"}, "narrow"),
        ({"paradigm": "ascendc", "arch_family": "arch35"}, "mid"),
    ]
    sorted_entries = sort_by_specificity(entries)
    assert sorted_entries[0][1] == "narrow"
    assert sorted_entries[2][1] == "broad"


# ---- format_tag_column / parse_tag_column round-trip ----

def test_format_roundtrip():
    original = {"paradigm": "ascendc", "arch_family": "arch35", "npu_arch": "3510"}
    formatted = format_tag_column(original)
    parsed = parse_tag_column(formatted)
    assert parsed["paradigm"] == "ascendc"
    assert parsed["arch_family"] == "arch35"
    assert parsed["npu_arch"] == "3510"


def test_format_skips_any():
    formatted = format_tag_column({"paradigm": "ascendc", "arch_family": "any"})
    assert "arch_family" not in formatted


def test_format_empty():
    formatted = format_tag_column({})
    assert "paradigm=any" in formatted


def test_parse_tag_column_multi_value():
    parsed = parse_tag_column("npu_arch=3510,5102; arch_family=arch35")
    assert parsed["npu_arch"] == "3510"  # takes first
    assert parsed["arch_family"] == "arch35"


# ---- VALID_VALUES sanity ----

def test_all_tiers_have_any():
    for tier in _TIER_SPECIFICITY_ORDER:
        if tier in VALID_VALUES and VALID_VALUES[tier]:
            assert "any" in VALID_VALUES[tier], f"{tier} missing 'any'"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ---- alias expansion + collision guard (2026-05-28 name-mismatch fix) ----

def test_expand_aliases_a5_family_name_invariant():
    from kb_schema import expand_aliases
    a = expand_aliases("V300")
    b = expand_aliases("arch35")
    c = expand_aliases("Ascend950PR")
    assert a == b == c
    assert "arch35" in a and "v300" in a and "ascend950pr" in a and "3510" in a


def test_expand_aliases_a3_family():
    from kb_schema import expand_aliases
    a = expand_aliases("arch22")
    assert "v220" in a and "ascend910_9382" in a and "3003" in a
    assert "arch35" not in a  # families disjoint


def test_expand_aliases_unknown_term_passthrough():
    from kb_schema import expand_aliases
    assert expand_aliases("WholeReduceMax") == {"wholereducemax"}


def test_alias_match_cross_name_recall():
    """A query for 'V300' must match an entry written with 'arch35' (and vice versa)."""
    from kb_schema import alias_match
    assert alias_match("V300", "this rule applies to arch35 kernels")
    assert alias_match("arch35", "this is a V300 / Ascend950PR thing")
    assert alias_match("Ascend950PR", "guarded by __NPU_ARCH__ == 3510")


def test_alias_match_v300x_collision_guard():
    """CRITICAL: 'V300' (A5) must NOT match 'V300x' (Atlas 200I/500 A2, diff chip)."""
    from kb_schema import alias_match
    assert alias_match("V300", "targets V300x Atlas 200I") is False
    # but real arch35 alias in the same text still matches
    assert alias_match("V300", "V300x is different but this is arch35") is True


def test_alias_match_plain_keyword_substring():
    """Non-alias keywords keep plain substring behavior."""
    from kb_schema import alias_match
    assert alias_match("PipeBarrier", "use PipeBarrier<PIPE_V>")
    assert alias_match("PipeBarrier", "no barrier here") is False
