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

"""DEBT-203 S5 follow-up — finalize threads the S3 branch lineage into provenance_node.

When an op was branch-seeded (S3 wrote workspace/.branched_from.json at cold-start),
finalize's _inject_provenance_node must record branched=true + parent_id in the archived
provenance_node (previously always None/False). Telemetry-only; no guard reads it.
"""
import logging
import sys
import os
import json
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import finalize_pipeline  # noqa: E402  (import parent FIRST to resolve the re-export dance)
import finalize_dispatch as fd  # noqa: E402


def _ws(tmp_path, *, with_marker):
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS", "pass_a": {"tier1_pass": 5, "total": 5}},
        "determinism": {"policy_satisfied": True}}))
    (ws / "op_classification.json").write_text(json.dumps({
        "op": "op", "op_class_tags": ["elementwise"], "algorithm_classification": "single_op"}))
    if with_marker:
        (ws / ".branched_from.json").write_text(json.dumps({
            "branched": True, "parent_id": "3_Add@abc123def456", "parent_op": "3_Add"}))
    return ws


def test_finalize_threads_branch_lineage(tmp_path):
    ws = _ws(tmp_path, with_marker=True)
    getattr(fd, '_inject_provenance_node')("op", ws)
    node = json.loads((ws / "verification.json").read_text())["provenance_node"]
    assert node["branched"] is True, "finalize must record branched=true when marker present"
    assert node["parent_id"] == "3_Add@abc123def456", "finalize must record parent_id from the marker"


def test_finalize_no_marker_stays_unbranched(tmp_path):
    ws = _ws(tmp_path, with_marker=False)
    getattr(fd, '_inject_provenance_node')("op", ws)
    node = json.loads((ws / "verification.json").read_text())["provenance_node"]
    assert node["branched"] is False and node["parent_id"] is None  # cold-started op = root, unchanged


def test_finalize_bad_marker_fails_open(tmp_path):
    ws = _ws(tmp_path, with_marker=False)
    (ws / ".branched_from.json").write_text("{ not valid json")
    getattr(fd, '_inject_provenance_node')("op", ws)  # must not raise
    node = json.loads((ws / "verification.json").read_text())["provenance_node"]
    assert node["branched"] is False and node["parent_id"] is None  # fail-open → unbranched


if __name__ == "__main__":
    import traceback
    import tempfile
    fails = []
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        try:
            with tempfile.TemporaryDirectory() as td:
                fn(Path(td))
            logging.info(f"  [PASS] {name}")
        except Exception as e:
            fails.append(name)
            logging.info(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
    logging.info(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}'}")
    sys.exit(1 if fails else 0)
