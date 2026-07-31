# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for sanitized public summary JSON schema."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

from cann_learn import summary_schema as ss  # noqa: E402


def _valid_summary():
    return {
        "run_id": "abc123",
        "ts": "2026-05-05T22:00:00Z",
        "op": "10_LayerNorm",
        "module_path_sha256": "a" * 64,
        "files_read_count": 3,
        "files_read_total_bytes": 12345,
        "files_read_hashes": ["b" * 64, "c" * 64, "d" * 64],
        "candidate_count_extracted": 5,
        "candidate_count_kept": 3,
        "candidate_count_dropped_leak": 1,
        "candidate_count_dropped_compile": 0,
        "candidate_count_dropped_copy_shape": 1,
        "candidate_count_overlap_existing": 0,
        "metadata_fix_proposals_count": 0,
        "leak_score": 0.0,
        "copy_shape_score": 0.02,
        "compile_pass_rate": 1.0,
        "self_review_verdict": "PASS",
        "self_review_failures": [],
        "checks": {
            "C34a": {"passed": True},
            "C34b": {"passed": True},
            "C34c": {"passed": True},
            "C35": {"passed": True},
        },
    }


def test_validate_valid_summary_passes():
    res = ss.validate(_valid_summary())
    assert res.valid
    assert res.errors == []


def test_validate_missing_required_key_fails():
    s = _valid_summary()
    del s["self_review_verdict"]
    res = ss.validate(s)
    assert not res.valid
    assert any("self_review_verdict" in e for e in res.errors)


def test_validate_unknown_key_rejected_as_prose_leak():
    s = _valid_summary()
    s["source_notes"] = "vendor uses c310_impl namespace"  # PROSE LEAK
    res = ss.validate(s)
    assert not res.valid
    assert any("source_notes" in e for e in res.errors)


def test_validate_self_review_verdict_must_be_pass_or_fail():
    s = _valid_summary()
    s["self_review_verdict"] = "MAYBE"
    res = ss.validate(s)
    assert not res.valid
    assert any("PASS" in e for e in res.errors)


def test_validate_negative_count_rejected():
    s = _valid_summary()
    s["candidate_count_kept"] = -1
    res = ss.validate(s)
    assert not res.valid


def test_validate_score_out_of_range_rejected():
    s = _valid_summary()
    s["leak_score"] = 1.5
    res = ss.validate(s)
    assert not res.valid
    assert any("0,1" in e for e in res.errors)


def test_validate_invalid_sha256_rejected():
    s = _valid_summary()
    s["files_read_hashes"] = ["not_a_sha256"]
    res = ss.validate(s)
    assert not res.valid


def test_validate_checks_must_have_passed_field():
    s = _valid_summary()
    s["checks"]["C34a"] = {"score": 0.0}  # no 'passed'
    res = ss.validate(s)
    assert not res.valid
    assert any("passed" in e for e in res.errors)


def test_validate_self_review_failures_must_be_list_of_str():
    s = _valid_summary()
    s["self_review_failures"] = [1, 2, 3]  # not strings
    res = ss.validate(s)
    assert not res.valid


def test_validate_optional_keys_allowed():
    s = _valid_summary()
    s["ts_finished"] = "2026-05-05T22:01:00Z"
    s["warning_count"] = 0
    res = ss.validate(s)
    assert res.valid, f"optional keys should be allowed; errors: {res.errors}"


def test_validate_file_passes_for_valid(tmp_path):
    f = tmp_path / "summary.json"
    f.write_text(json.dumps(_valid_summary()))
    res = ss.validate_file(f)
    assert res.valid


def test_validate_file_invalid_json_rejected(tmp_path):
    f = tmp_path / "summary.json"
    f.write_text("{not valid json")
    res = ss.validate_file(f)
    assert not res.valid


def test_validate_file_missing_returns_error(tmp_path):
    res = ss.validate_file(tmp_path / "nope.json")
    assert not res.valid


def test_validate_long_prose_in_unknown_key_caught():
    """Even with structurally valid required fields, an extra prose-typed key
    leaks source notes back to main context.
    """
    s = _valid_summary()
    s["learner_observations"] = (
        "From reading the source files, vendor uses internal::c310_impl::Normalize. "
        "Key insight: A=K dispatch via Reg::LoadAlign<DIST_BRC_B32>."
    )
    res = ss.validate(s)
    assert not res.valid
    assert any("learner_observations" in e for e in res.errors)
