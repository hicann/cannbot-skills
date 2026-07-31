# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Adversarial regression tests for contract_e2e — proves the anti-cheat gates BITE.

Each test encodes one of independent review's load-bearing review holes and asserts the e2e
driver fail-closes on it (i.e., the gate is not cosmetic). Run: pytest test_contract_e2e.py
"""
import copy
import logging
from contract_validator_poc import Contract
from contract_e2e import grade_e2e

LOGGER = logging.getLogger(__name__)


def _clean_rec(i, dtype="float16", ours_scale=1.0):
    """A representative case where ours ~= baseline (both small errors vs fp64 golden)."""
    base = 2 ** -11  # well inside fp16 rounding band
    return {
        "i": i, "dtype": dtype, "layout": "BSH", "B": 2, "S": 1024, "N": 8, "D": 128,
        "baseline_mare": base, "baseline_mere": base, "baseline_rmse": base,
        "cann_mare": base, "cann_mere": base, "cann_rmse": base,
        "ours_mare": base * ours_scale, "ours_mere": base * ours_scale, "ours_rmse": base * ours_scale,
    }


def _contract(recs, dtype="float16", op_class="numerically_hard"):
    return Contract(op="t", op_class=op_class, dtype=dtype, tier="L1", artifact_path="<test>")


def test_clean_representative_is_shippable():
    recs = [_clean_rec(i) for i in range(50)]
    v = grade_e2e(recs, _contract(recs), case_class="representative")
    assert v.validator == "double_baseline_ratio"   # disk shows baseline+fp64 -> ratio
    assert v.shippable is True and v.blockers == []


def test_refs_are_disk_derived_not_self_declared():
    # independent review #1: contract self-declares ALL refs, but artifact has NO baseline columns ->
    # driver must NOT grant the loose ratio validator.
    recs = [_clean_rec(i) for i in range(20)]
    for r in recs:
        del r["baseline_mare"]
        del r["baseline_mere"]
        del r["baseline_rmse"]   # baseline absent on disk
    c = _contract(recs)
    c.refs_available = ("fp64_golden", "independent_baseline", "same_dtype_vendor")  # lie
    v = grade_e2e(recs, c, case_class="representative")
    assert v.validator != "double_baseline_ratio"   # the lie did NOT buy the loose validator
    assert "independent_baseline" not in v.criterion_provenance["evidence"]


def test_inflated_baseline_rejected():
    # independent review #2: blow up baseline error -> ratio trivially small -> must reject.
    recs = [_clean_rec(i) for i in range(30)]
    for r in recs:
        r["baseline_mare"] *= 500
        r["baseline_mere"] *= 500
        r["baseline_rmse"] *= 500
    v = grade_e2e(recs, _contract(recs), case_class="representative")
    assert "baseline_inflated" in v.blockers and v.shippable is False


def test_aggregate_not_met_blocks_ship():
    # independent review #3: ours much worse than the baseline -> CI-upper exceeds tier -> not shippable.
    recs = [_clean_rec(i, ours_scale=10.0) for i in range(30)]
    v = grade_e2e(recs, _contract(recs), case_class="representative")
    assert "aggregate_not_met" in v.blockers and v.shippable is False


def test_circular_vendor_informational_only():
    # independent review #4: no independent baseline on disk, only same-dtype vendor (CANN) ->
    # selection lands on same_dtype_threshold "circular" -> informational, not certifiable.
    recs = [_clean_rec(i) for i in range(20)]
    for r in recs:
        del r["baseline_mare"]
        del r["baseline_mere"]
        del r["baseline_rmse"]   # no third-party baseline on disk
    c = _contract(recs)
    v = grade_e2e(recs, c, case_class="representative")
    assert "circular" in v.selection_rule          # circular-vendor path taken
    assert v.certifiable is False                   # cannot certify ship without owner override


def test_edge_class_never_shippable():
    recs = [_clean_rec(i) for i in range(20)]
    v = grade_e2e(recs, _contract(recs), case_class="edge")
    assert v.shippable is False   # edge/adversarial never certifies ship


def _fp64_only_rec(i, dtype="float16", mere=2 ** -13):
    """ecosystem single-baseline case: fp64 golden only (no independent baseline, no cann) -> ours_* vs fp64."""
    return {"i": i, "dtype": dtype, "shape": [1024],
            "ours_mare": mere, "ours_mere": mere, "ours_rmse": mere}


def test_single_baseline_ecosystem_certifies():
    # RECONCILED (vs back-agent mul_grad + cannbot): fp64-golden-only -> single_baseline_threshold
    # (ecosystem). Comfortably inside dtype-eps -> certifiable (NOT hard-blocked as before).
    recs = [_fp64_only_rec(i, "float16", mere=2 ** -13) for i in range(40)]  # 2^-13 << fp16 2^-10
    v = grade_e2e(recs, _contract(recs), case_class="representative")
    assert v.validator == "single_baseline_threshold"
    assert "cross_precision_floor_underived" not in v.blockers   # over-conservative block removed
    assert "ecosystem_floor" in v.criterion_provenance           # floor source recorded for audit
    assert v.certifiable is True and v.shippable is True


def test_single_baseline_over_floor_not_shippable():
    # mere ABOVE the dtype-eps floor -> ecosystem aggregate fails -> not shippable (still fail-closed).
    recs = [_fp64_only_rec(i, "float16", mere=2 ** -8) for i in range(40)]   # 2^-8 > fp16 2^-10
    v = grade_e2e(recs, _contract(recs), case_class="representative")
    assert v.validator == "single_baseline_threshold"
    assert "aggregate_not_met" in v.blockers and v.shippable is False


if __name__ == "__main__":
    import sys
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    fails = 0
    for f in fns:
        try:
            f()
            LOGGER.info("PASS-test %s", f.__name__)
        except AssertionError as e:
            fails += 1
            LOGGER.info("FAIL-test %s: %s", f.__name__, e)
    LOGGER.info("%d/%d ok", len(fns) - fails, len(fns))
    sys.exit(1 if fails else 0)
