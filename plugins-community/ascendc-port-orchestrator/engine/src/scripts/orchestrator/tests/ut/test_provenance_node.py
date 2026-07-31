#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DEBT-203 S1 unit tests — provenance_node derivation + injection.

S1 is ADDITIVE-ONLY: compute a `provenance_node` block (node_id/parent_id/branched/
fitness/is_buggy/signature/created_ts) DERIVED from an op's verification.json +
op_classification.json, and inject it. No reading/seeding (that's S2/S3). Fitness is
DERIVED from existing precision/perf/determinism fields — never a new measurement.
"""
import logging
import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))  # src/scripts (for `orchestrator` pkg root)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))         # orchestrator/ (flat imports)

import provenance_node as pn


# ── fitness derivation (correctness-dominant; DERIVED, not invented) ──────────
def test_fitness_full_pass():
    v = {"precision": {"pass": 50, "total": 50, "status": "PASS"},
         "performance": {"sum_ratio": 0.9},
         "determinism": {"policy_satisfied": True}}
    f = pn.compute_fitness(v)
    # 0.6*1(prec) + 0.3*0.9(perf) + 0.1*1(det) = 0.97
    assert abs(f - 0.97) < 1e-6, f


def test_fitness_precision_dominates():
    # perf 0 but precision full + det → still high (0.6 prec weight + 0.1 det)
    v = {"precision": {"pass": 50, "total": 50, "status": "PASS"},
         "performance": {"sum_ratio": 0.0},
         "determinism": {"policy_satisfied": True}}
    f = pn.compute_fitness(v)
    assert 0.68 <= f <= 0.72, f  # 0.6*1 + 0.3*0 + 0.1*1 = 0.7


def test_fitness_fail_precision_is_zero():
    # a FAIL-precision op is forced to fitness 0 (never a branch source)
    v = {"precision": {"pass": 10, "total": 50, "status": "FAIL"},
         "performance": {"sum_ratio": 0.9},
         "determinism": {"policy_satisfied": True}}
    assert pn.compute_fitness(v) == 0.0


def test_fitness_perf_clamped():
    # perf ratio > 1 (beats CANN) clamps to 1, doesn't over-inflate
    v = {"precision": {"pass": 50, "total": 50, "status": "PASS"},
         "performance": {"sum_ratio": 3.0},
         "determinism": {"policy_satisfied": True}}
    assert pn.compute_fitness(v) <= 1.0


# ── eligibility = correctness ONLY (main's ruling Q1/Q3) ──────────────────────
def test_eligible_requires_full_precision_pass():
    passing = {"precision": {"pass": 50, "total": 50, "status": "PASS"},
               "determinism": {"policy_satisfied": True}}
    assert pn.is_branch_eligible(passing) is True
    # partial precision → NOT eligible even if fitness float is decent
    partial = {"precision": {"pass": 49, "total": 50, "status": "PARTIAL"},
               "determinism": {"policy_satisfied": True}}
    assert pn.is_branch_eligible(partial) is False
    # det not satisfied → not eligible
    nodet = {"precision": {"pass": 50, "total": 50, "status": "PASS"},
             "determinism": {"policy_satisfied": False}}
    assert pn.is_branch_eligible(nodet) is False


# ── REAL current schema (nested pass_a + perf.ratio) — regression guard ───────
def test_real_schema_nested_pass_a():
    # the actual 2_SwiGLU verification.json shape: precision.status PASS,
    # counts under precision.pass_a.{tier1_pass,total}, perf uses `ratio` not sum_ratio
    v = {"precision": {"status": "PASS",
                       "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
                       "pass_b": {"status": "PASS", "tier1_pass": 291, "total": 291}},
         "performance": {"status": "BELOW_THRESHOLD", "ratio": 0.346},
         "determinism": {"policy_satisfied": True}}
    # eligible: correctness-only, PASS status + det (perf below threshold is irrelevant)
    assert pn.is_branch_eligible(v) is True
    # precision_ratio reads nested pass_a → 1.0; fitness = 0.6*1 + 0.3*0.346 + 0.1*1 = 0.8038
    assert abs(pn.compute_fitness(v) - 0.8038) < 1e-4, pn.compute_fitness(v)


def test_real_schema_perf_from_independent_re_measure():
    v = {"precision": {"status": "PASS", "pass_a": {"tier1_pass": 10, "total": 10}},
         "performance": {"independent_re_measure": {"median_ratio": 0.5}},
         "determinism": {"policy_satisfied": True}}
    # perf falls back to independent_re_measure.median_ratio
    assert abs(pn.compute_fitness(v) - (0.6 + 0.3 * 0.5 + 0.1)) < 1e-4


# ── is_buggy ──────────────────────────────────────────────────────────────────
def test_is_buggy_on_fail():
    assert pn.is_op_buggy({"precision": {"status": "FAIL"}}) is True
    assert pn.is_op_buggy({"precision": {"status": "PASS"}}) is False


# ── build_provenance_node (S1: parent_id=None, branched=False) ────────────────
def test_build_node_shape():
    v = {"precision": {"pass": 50, "total": 50, "status": "PASS"},
         "performance": {"sum_ratio": 0.5},
         "determinism": {"policy_satisfied": True}}
    sig = {"op_class_tags": ["elementwise"], "algorithm_classification": "single_op"}
    node = pn.build_provenance_node("2_SwiGLU", "abc123def456", v, sig, created_ts="2026-07-11T00:00:00Z")
    assert node["node_id"] == "2_SwiGLU@abc123def456"
    assert node["parent_id"] is None            # S1 never seeds
    assert node["branched"] is False
    assert node["is_buggy"] is False
    assert 0.0 <= node["fitness"] <= 1.0
    assert node["signature"]["op_class_tags"] == ["elementwise"]
    assert node["created_ts"] == "2026-07-11T00:00:00Z"
    assert node["schema_version"] == 1


def test_build_node_preserves_existing_created_ts():
    # idempotency: re-building with an existing node preserves its created_ts
    v = {"precision": {"pass": 50, "total": 50, "status": "PASS"}, "determinism": {"policy_satisfied": True}}
    existing = {"created_ts": "2026-07-01T12:00:00Z"}
    node = pn.build_provenance_node("op", "sha", v, {}, created_ts="2026-07-11T00:00:00Z", existing_node=existing)
    assert node["created_ts"] == "2026-07-01T12:00:00Z"  # preserved, not the new ts


# ── injection into verification dict (backward-compat: additive only) ─────────
def test_inject_is_additive():
    v = {"precision": {"pass": 50, "total": 50, "status": "PASS"},
         "determinism": {"policy_satisfied": True},
         "notes": {"archived_to": "x"}}
    before_keys = set(v.keys())
    out = pn.inject_provenance_node(dict(v), "op", "sha", {"op_class_tags": []}, created_ts="2026-07-11T00:00:00Z")
    # every original key/value preserved; only provenance_node added
    assert set(out.keys()) == before_keys | {"provenance_node"}
    for k in before_keys:
        assert out[k] == v[k]


def test_inject_idempotent_on_identical_content():
    v = {"precision": {"pass": 50, "total": 50, "status": "PASS"}, "determinism": {"policy_satisfied": True}}
    a = pn.inject_provenance_node(dict(v), "op", "sha", {}, created_ts="2026-07-11T00:00:00Z")
    # re-inject over the already-augmented dict with a DIFFERENT ts → created_ts preserved → identical block
    b = pn.inject_provenance_node(dict(a), "op", "sha", {}, created_ts="2099-01-01T00:00:00Z")
    assert a["provenance_node"] == b["provenance_node"]  # stable → idempotent hash


if __name__ == "__main__":
    import traceback
    fails = []
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        try:
            fn()
            logging.info(f"  [PASS] {fn.__name__}")
        except Exception as e:
            fails.append(fn.__name__)
            logging.info(f"  [FAIL] {fn.__name__}: {e}")
            traceback.print_exc()
    logging.info(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}'}")
    sys.exit(1 if fails else 0)
