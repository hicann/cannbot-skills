# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for C34c copy-shape detector."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

from cann_learn import copy_shape as cs  # noqa: E402


def test_normalize_renames_identifiers_to_ident():
    """Different-named identifiers normalize to IDENT — renamed copy detection."""
    a = cs.normalize_tokens("for (int i = 0; i < N; i++) sum += a[i];")
    b = cs.normalize_tokens("for (int j = 0; j < count; j++) accum += arr[j];")
    # Identifier names differ, but normalized form matches
    assert a == b


def test_normalize_keeps_keywords():
    toks = cs.normalize_tokens("if (x > 0) return; else continue;")
    assert "if" in toks
    assert "return" in toks
    assert "else" in toks
    assert "continue" in toks


def test_normalize_keeps_numeric_literals():
    """Magic numbers preserved — catches '0.044715 from GELU formula' copy."""
    toks = cs.normalize_tokens("x * 0.044715 + 1.0;")
    assert "0.044715" in toks
    assert "1.0" in toks


def test_normalize_keeps_hex_literals():
    toks = cs.normalize_tokens("uint32_t mask = 0xFFFFFFFF;")
    assert "0xFFFFFFFF" in toks


def test_copy_shape_clean_low_score(tmp_path):
    """Candidate that doesn't share contiguous shape with source = low score."""
    src = tmp_path / "internal.h"
    src.write_text("""
namespace internal {
class Reducer {
  void reduce_block(uint8_t* buf, int n) {
    int total = 0;
    for (int i = 0; i < n; i += 4) {
      total += buf[i];
    }
  }
};
}
""")
    cand = "Use AscendC's WholeReduceSum<T> primitive on the LocalTensor."
    result = cs.check("c1", cand, [src])
    # Conceptual prose, no shared shape
    assert result.score < 0.05
    assert result.passed


def test_copy_shape_renamed_copy_detected(tmp_path):
    """Source: `for (int i = 0; i < N; i++) sum += a[i];`
    Candidate (renamed copy): `for (int j = 0; j < count; j++) accum += arr[j];`
    Should detect contiguous-shape match.
    """
    src = tmp_path / "src.cpp"
    src.write_text("""
void reduce(int* a, int N) {
    int sum = 0;
    for (int i = 0; i < N; i++) {
        sum += a[i];
    }
}
""")
    cand = """
void compute(int* arr, int count) {
    int accum = 0;
    for (int j = 0; j < count; j++) {
        accum += arr[j];
    }
}
"""
    result = cs.check("renamed_copy", cand, [src], threshold=0.05)
    # Renamed copy → high score
    assert result.score > 0.3, f"renamed copy should have high score, got {result.score}"
    assert not result.passed


def test_copy_shape_magic_number_copy_detected(tmp_path):
    """Source uses GELU formula constant 0.044715. Candidate uses same constant
    in same code shape → match.
    """
    src = tmp_path / "src.cpp"
    src.write_text("y = x * 0.044715 + 1.0;")
    cand = "result = input * 0.044715 + 1.0;"
    result = cs.check("magic", cand, [src])
    # Magic-number-preserved copy detected
    assert result.score > 0.05
    assert any("0.044715" in m[1] for m in result.sample_matches)


def test_copy_shape_threshold_configurable(tmp_path):
    """Threshold tunable. Score ranges 0..1; threshold 0.5 means
    strict_pass iff score < 0.5.
    """
    src = tmp_path / "src.cpp"
    # Source contains some loop pattern
    src.write_text("for (int i = 0; i < N; i++) a[i] = 0;")
    # Candidate: half-shared shape (loop), half-novel (function call to public API)
    cand = "for (int j = 0; j < M; j++) b[j] = 0; AscendC::WholeReduceSum(buf, 1, M);"

    strict = cs.check("c", cand, [src], threshold=0.05)
    assert not strict.passed  # loop part matches → score > 5%

    # Tunable: score is determinstic; assert score < 1.0 (some non-matching tail)
    assert strict.score < 1.0, "candidate has novel suffix, score should be < 1"


def test_copy_shape_too_short_for_ngram(tmp_path):
    """Candidate shorter than n tokens → score 0.0, vacuous pass."""
    src = tmp_path / "src.cpp"
    src.write_text("any content")
    result = cs.check("short", "x;", [src], n=10)
    assert result.score == 0.0
    assert result.passed


def test_copy_shape_no_source_files(tmp_path):
    """No source files → no matches possible → score 0."""
    result = cs.check("c", "for (int i = 0; i < N; i++) a[i] = 0;", [])
    assert result.score == 0.0
    assert result.passed


def test_copy_shape_sample_matches_capped(tmp_path):
    """sample_matches list capped at 10 even if many matches."""
    src = tmp_path / "src.cpp"
    src.write_text("a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q;")
    cand = "x + y + z + w + v + u + t + s + r + q + p + o + n + m + l + k + j;"
    result = cs.check("c", cand, [src])
    assert len(result.sample_matches) <= 10
