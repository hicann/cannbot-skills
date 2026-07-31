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

"""DEBT-203 S2 unit tests — find_similar_proven_op matcher (READ-ONLY).

S2 reads the archived provenance_node blocks (written by S1) + op_classification
signatures, and returns the most-similar PROVEN (correctness-eligible, fitness>=min,
non-buggy) op above a similarity threshold — or None. It does NOT seed cold-start
(S3, gated on main's anti-cheat sign-off) and is NOT wired on by default.

Reuses cann_learn/kb_overlap op-class alias normalization for tag canonicalization.
"""
import logging
import sys
import os
import json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import provenance_match as pm


# ── similarity ────────────────────────────────────────────────────────────────
def test_similarity_identical_tags():
    a = {"op_class_tags": ["elementwise", "transcendental"]}
    assert pm.op_signature_similarity(a, dict(a)) >= 0.99


def test_similarity_disjoint_is_zero():
    a = {"op_class_tags": ["elementwise"]}
    b = {"op_class_tags": ["attention", "matmul"]}
    assert pm.op_signature_similarity(a, b) == 0.0


def test_similarity_partial_jaccard():
    a = {"op_class_tags": ["elementwise", "transcendental", "reduction"]}
    b = {"op_class_tags": ["elementwise", "transcendental"]}
    s = pm.op_signature_similarity(a, b)  # |∩|=2 / |∪|=3 = 0.667
    assert 0.6 < s < 0.72, s


def test_similarity_empty_is_zero():
    assert pm.op_signature_similarity({"op_class_tags": []}, {"op_class_tags": ["x"]}) == 0.0


# ── find_similar_proven_op (read-only over an archive corpus) ─────────────────
def _write_archived_op(root, op, *, tags, status, det, fitness, is_buggy=False):
    """Create a minimal archived op dir with verification.json (carrying a
    provenance_node) + op_classification.json — the two files S2 reads."""
    d = root / op
    d.mkdir(parents=True, exist_ok=True)
    (d / "verification.json").write_text(json.dumps({
        "precision": {"status": status},
        "determinism": {"policy_satisfied": det},
        "provenance_node": {
            "node_id": f"{op}@deadbeef0000", "parent_id": None, "branched": False,
            "fitness": fitness, "is_buggy": is_buggy,
            "signature": {"op_class_tags": tags, "algorithm_classification": "single_op"},
            "created_ts": "2026-07-11T00:00:00Z", "schema_version": 1,
        },
    }))
    (d / "op_classification.json").write_text(json.dumps({
        "op": op, "op_class_tags": tags, "algorithm_classification": "single_op",
        "source_sha256": "deadbeef0000",
    }))
    return d


def test_find_returns_best_eligible_match(tmp_path):
    root = tmp_path / "kernels"
    # a strong match: same tags, proven (PASS+det), high fitness
    _write_archived_op(root, "5_GELU", tags=["elementwise", "transcendental"],
                       status="PASS", det=True, fitness=0.95)
    # a weaker match: partial tags
    _write_archived_op(root, "9_Sum", tags=["reduction"],
                       status="PASS", det=True, fitness=0.9)
    query = {"op_class_tags": ["elementwise", "transcendental"], "algorithm_classification": "single_op"}
    ref = pm.find_similar_proven_op(query, archive_root=root, threshold=0.5, min_fitness=0.8)
    assert ref is not None
    assert ref.node_id.startswith("5_GELU@")     # the stronger match wins
    assert ref.similarity >= 0.99


def test_find_excludes_buggy_and_low_fitness_and_failed(tmp_path):
    root = tmp_path / "kernels"
    _write_archived_op(root, "buggy", tags=["elementwise"], status="FAIL", det=True, fitness=0.0, is_buggy=True)
    _write_archived_op(root, "lowfit", tags=["elementwise"], status="PASS", det=True, fitness=0.5)   # < min_fitness
    _write_archived_op(root, "nodet", tags=["elementwise"], status="PASS",
                       det=False, fitness=0.9)   # det fail → not eligible
    query = {"op_class_tags": ["elementwise"]}
    ref = pm.find_similar_proven_op(query, archive_root=root, threshold=0.3, min_fitness=0.8)
    assert ref is None   # none are proven+eligible+fit


def test_find_below_threshold_returns_none(tmp_path):
    root = tmp_path / "kernels"
    _write_archived_op(root, "far", tags=["attention", "matmul", "softmax"], status="PASS", det=True, fitness=0.95)
    query = {"op_class_tags": ["elementwise"]}   # near-disjoint
    assert pm.find_similar_proven_op(query, archive_root=root, threshold=0.5, min_fitness=0.8) is None


def test_find_empty_archive_returns_none(tmp_path):
    root = tmp_path / "kernels"
    root.mkdir(parents=True)
    assert pm.find_similar_proven_op({"op_class_tags": ["x"]},
                                     archive_root=root, threshold=0.5, min_fitness=0.8) is None


def test_find_tie_breaks_by_fitness(tmp_path):
    root = tmp_path / "kernels"
    _write_archived_op(root, "A_lo", tags=["elementwise"], status="PASS", det=True, fitness=0.85)
    _write_archived_op(root, "B_hi", tags=["elementwise"], status="PASS", det=True, fitness=0.98)
    ref = pm.find_similar_proven_op({"op_class_tags": ["elementwise"]},
                                    archive_root=root, threshold=0.5, min_fitness=0.8)
    assert ref.node_id.startswith("B_hi@")   # equal similarity → higher fitness wins


if __name__ == "__main__":
    import traceback
    import tempfile
    from pathlib import Path
    fails = []
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            logging.info(f"  [PASS] {name}")
        except Exception as e:
            fails.append(name)
            logging.info(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
    logging.info(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}'}")
    sys.exit(1 if fails else 0)
