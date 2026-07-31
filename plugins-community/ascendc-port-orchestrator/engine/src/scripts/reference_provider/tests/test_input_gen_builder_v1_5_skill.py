# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression: aog-input-gen-builder SKILL.md text contains V1.5 guidance.

When case_gen V1.5 added multi-rank / multi-dtype / 2-arg derive /
callable probe_values primitives (commit c2262f86), the skill's BLOCK
behavior on those patterns became wrong — it should USE the primitives,
not BLOCK on them.

This test pins the skill text so any future regression that removes the
V1.5 sections (or restores BLOCK guidance for those primitives) is
caught immediately.
"""
from __future__ import annotations

import pathlib

import pytest

_HERE = pathlib.Path(__file__).resolve()


def _resolve_skill(name: str) -> pathlib.Path:
    plugin_root = _HERE.parents[5]
    return plugin_root / "skills" / "aog-input-gen-builder" / name


_SKILL = _resolve_skill("SKILL.md")


def test_skill_md_exists() -> None:
    assert _SKILL.exists(), f"skill missing at {_SKILL}"


def test_skill_documents_multi_rank() -> None:
    text = _SKILL.read_text()
    assert 'schema["ranks"]' in text or "schema['ranks']" in text, (
        "skill must document the schema['ranks'] field per case_gen V1.5"
    )
    # The skill must NOT advise BLOCKING multi-rank ops anymore
    assert "rank-3 or rank-4 base_shape" not in text or "V1.5" in text


def test_skill_documents_multi_dtype() -> None:
    text = _SKILL.read_text()
    assert 'schema["dtypes"]' in text or "schema['dtypes']" in text


def test_skill_documents_2arg_derive() -> None:
    text = _SKILL.read_text()
    assert "2-arg" in text and "derive" in text
    # Specifically the (base_shape, scalars) signature
    assert "(base_shape, scalars)" in text


def test_skill_documents_callable_probe_values() -> None:
    text = _SKILL.read_text()
    assert "lambda rank" in text or "Rank-dependent probe" in text


def test_skill_provides_worked_5_cumsum_schema() -> None:
    text = _SKILL.read_text()
    assert "5_Cumsum" in text
    # Sanity: cited the ranks + dtypes lists explicitly
    cumsum_section = text.split("Worked schema — 5_Cumsum", 1)
    assert len(cumsum_section) == 2, "missing 5_Cumsum worked-schema section"


def test_skill_provides_worked_9_topk_schema() -> None:
    text = _SKILL.read_text()
    assert "9_TopK" in text
    topk_section = text.split("Worked schema — 9_TopK", 1)
    assert len(topk_section) == 2, "missing 9_TopK worked-schema section"
    # 9_TopK demonstrates cross-scalar dependency via 2-arg derive
    body = topk_section[1][:1500]
    assert "scalars[" in body, "9_TopK example must show 2-arg derive reading scalars"


def test_skill_no_longer_lists_resolved_block_items() -> None:
    """Items 5/6/7 from the original V1 BLOCK list ('rank-3+ base_shape',
    'dim-constraint shapes', 'rank-dependent tuple lengths') were
    partially-or-fully resolved. The original wording 'Rank-3+ base_shape
    with independent dims' as a BLOCK class must NOT remain — case_gen
    V1.5 supports multi-rank including rank-3/4 via _shape_plan.
    """
    text = _SKILL.read_text()
    # The exact V1 phrasing that case_gen V1.5 contradicts:
    forbidden_v1_phrasing = "Rank-3+ base_shape with independent dims"
    if forbidden_v1_phrasing in text:
        pytest.fail(
            f"skill still lists '{forbidden_v1_phrasing}' as BLOCK-class. "
            "case_gen V1.5 supports rank-3/rank-4 via the 'ranks' schema "
            "field + _shape_plan d3_*/d4_* shapes. Either remove this item "
            "from the BLOCK list or move it to a V1.5-resolved section."
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
