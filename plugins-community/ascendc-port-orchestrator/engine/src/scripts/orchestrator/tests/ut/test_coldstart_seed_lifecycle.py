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
"""DEBT-203 S5 lifecycle fix — cold-start clears the seed base + marker.

Production-bug fix: branched_from_kernel/ (S3 seed base) + .branched_from.json
(seed marker) survived cold-start, so an enabled same-workspace re-run
(lifetime_spawn_count survives P94 → seed hook won't re-fire) would feed the worker a STALE
seed from the prior run. Cold-start (fresh op-gen start) must now clear both — STRENGTHENING
DEBT-078's anti-restore. (Preserved across-iter WITHIN a run since cold-start isn't called mid-run.)
"""
import logging
import sys
import os
import json
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import orchestrator_coldstart as cs


def _ws(tmp_path):
    ws = tmp_path / "4_Abs"
    ws.mkdir()
    # S3 seed artifacts (the contamination vector)
    (ws / "branched_from_kernel").mkdir()
    (ws / "branched_from_kernel" / "add_kernel.cpp").write_text("// proven seed base\n")
    (ws / ".branched_from.json").write_text(json.dumps({"branched": True, "parent_op": "3_Add"}))
    # worker output (should be wiped) + preserved prep (should survive)
    (ws / "kernel").mkdir()
    (ws / "kernel" / "k.cpp").write_text("// worker out\n")
    (ws / "op_classification.json").write_text(json.dumps({"op": "4_Abs"}))  # preserved prep
    (ws / "input_gen.py").write_text("# prep\n")  # preserved prep
    return ws


def test_coldstart_clears_seed_base_and_marker(tmp_path):
    ws = _ws(tmp_path)
    getattr(cs, '_cold_start_reset_workspace')(ws)
    assert not (ws / "branched_from_kernel").exists(), "cold-start must wipe the seed base branched_from_kernel/"
    assert not (ws / ".branched_from.json").exists(), "cold-start must clear the .branched_from.json marker"
    # worker output wiped, prep preserved (unchanged behavior)
    assert not (ws / "kernel").exists(), "kernel/ (worker output) still wiped"
    assert (ws / "op_classification.json").exists(), "op_classification.json (prep) preserved"
    assert (ws / "input_gen.py").exists(), "input_gen.py (prep) preserved"


def test_coldstart_no_seed_artifacts_is_noop_for_them(tmp_path):
    # a workspace that was never seeded: cold-start proceeds normally (no crash)
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / "kernel").mkdir()
    (ws / "kernel" / "k.cpp").write_text("//\n")
    (ws / "op_classification.json").write_text("{}")
    getattr(cs, '_cold_start_reset_workspace')(ws)  # must not raise
    assert not (ws / "kernel").exists() and (ws / "op_classification.json").exists()


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
