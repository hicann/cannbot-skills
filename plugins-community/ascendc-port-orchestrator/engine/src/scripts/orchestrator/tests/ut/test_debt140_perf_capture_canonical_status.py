# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DEBT-140 follow-up — phase_o5_perf_capture must emit a CANONICAL finalize-gate
status, never the non-canonical "NOT_VERIFIED_SAME_METHOD" that loops finalize forever.

blue repro (live, 2026-06-01, act_quant op-gen on Ascend910_9382 / V220):
- worker precision PASS, O5 VERIFIED (Pass A 24/24 + Pass B 24/24 bit-exact).
- but `phase_o5_perf_capture` could not get profiler device-duration on V220
  ('Failed to get acl to npu flow events') → emitted status="NOT_VERIFIED_SAME_METHOD".
- the finalize-eligibility gate (finalize_pipeline.py:774) only accepts
  PASS / PASS_WITHIN_TOLERANCE / FAIL / BELOW_THRESHOLD (measured, w/ ratio) or
  N/A (with reason). "NOT_VERIFIED_SAME_METHOD" is non-canonical → finalize ROLLBACK
  → same signature 3× → LOOP-BREAK → await_user_decision. No worker edit can fix it
  (phase_o5_perf_capture overwrites the perf block on each finalize).

Fix: the honest "could not measure same-method" outcome IS N/A-with-reason. Map both
not-verified sites (`_not_verified` + the cand_us<=0 guard) to canonical status="N/A"
+ a non-empty `reason` (original signal preserved in `error`). Measured PASS/threshold
paths are unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

import phase_o5_perf_capture as p  # noqa: E402

# the canonical set the finalize gate accepts (finalize_pipeline.py)
_CANONICAL = {"PASS", "PASS_WITHIN_TOLERANCE", "FAIL", "BELOW_THRESHOLD", "N/A", "NA"}


def test_not_verified_emits_canonical_na_with_reason():
    d = getattr(p, '_not_verified')("some-method", "profiler device-duration unavailable on V220")
    assert d["status"] in _CANONICAL, d["status"]
    assert d["status"] == "N/A"
    assert d.get("reason"), "N/A requires a non-empty reason (finalize gate enforces this)"
    # original signal preserved for forensics
    assert "V220" in (d.get("error") or "")


def test_non_positive_candidate_is_canonical_na():
    r = getattr(p, '_build_result')(
        method="m", ref_samples=[1.0, 1.0], cand_samples=[0.0, 0.0], threshold=0.6
    ).to_dict()
    assert r["status"] == "N/A", r["status"]
    assert r.get("reason"), "must document why"


def test_measured_pass_path_unchanged():
    r = getattr(p, '_build_result')(
        method="m", ref_samples=[10.0, 10.0], cand_samples=[2.0, 2.0], threshold=0.6
    ).to_dict()
    assert r["status"] == "PASS"
    assert r["ratio"] == 5.0


def test_measured_below_threshold_still_emitted_with_ratio():
    # ratio 0.5 < threshold 0.6 → BELOW_THRESHOLD (canonical measured class, keeps ratio)
    r = getattr(p, '_build_result')(
        method="m", ref_samples=[1.0, 1.0], cand_samples=[2.0, 2.0], threshold=0.6
    ).to_dict()
    assert r["status"] == "BELOW_THRESHOLD"
    assert isinstance(r["ratio"], (int, float))


if __name__ == "__main__":
    sys.exit(p.__name__ and pytest.main([__file__, "-v"]))
