# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression tests for ACLNN_VERIFY_PATH_FRAUD gate
(_check_universal_entrypoints + _check_a5_verify_path_provenance).

User catch 2026-05-14T23:30Z: "your reward hacking make us lost 2 days".
Four port_a3 archives shipped PASS while verify ran PyTorch dispatcher
falling back to AICPU/CPU. Architectural fix: mode-agnostic entry-point
gate enforced by check_finalize_eligibility.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import finalize_pipeline as fp


def _make_ws(tmp_path: Path, vj: dict, *, files: list[str] = (),
             port_a3: bool = False) -> Path:
    """Build a fixture workspace.

    `port_a3=True` seeds `.opgen_state.json` declaring port_a3 mode,
    so `_check_a5_verify_path_provenance` (plugin-dispatched) runs on
    this workspace. Without this flag the gate skips (other modes
    don't have this provenance check by design).
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "verification.json").write_text(json.dumps(vj))
    if port_a3:
        (ws / ".opgen_state.json").write_text(json.dumps({
            "op": "fixture", "opgen_mode": "port_a3_to_a5",
        }))
    for fn in files:
        (ws / fn).write_text("// stub")
    return ws


# Tests for _check_universal_entrypoints (the wired gate)

def test_universal_pass_without_model_new_ascendc_rejected(tmp_path):
    vj = {"precision": {"status": "PASS", "pass_a": {"status": "PASS", "total": 8}},
          "performance": {"status": "PASS", "ratio": 2.5}}
    ws = _make_ws(tmp_path, vj, files=["model.py"])
    result = getattr(fp, '_check_universal_entrypoints')(ws, vj)
    assert result is not None
    assert "model_new_ascendc.py" in result


def test_universal_pass_without_model_py_rejected(tmp_path):
    vj = {"precision": {"status": "PASS", "pass_a": {"status": "PASS", "total": 8}},
          "performance": {"status": "PASS", "ratio": 2.5}}
    ws = _make_ws(tmp_path, vj, files=["model_new_ascendc.py"])
    result = getattr(fp, '_check_universal_entrypoints')(ws, vj)
    assert result is not None
    assert "model.py" in result


def test_universal_perf_pass_without_ratio_rejected(tmp_path):
    vj = {"precision": {"status": "PASS", "pass_a": {"status": "PASS", "total": 8}},
          "performance": {"status": "PASS", "ratio": None}}
    ws = _make_ws(tmp_path, vj, files=["model_new_ascendc.py", "model.py"])
    result = getattr(fp, '_check_universal_entrypoints')(ws, vj)
    assert result is not None
    assert "perf.ratio" in result


def test_universal_perf_pass_with_string_ratio_rejected(tmp_path):
    vj = {"precision": {"status": "PASS", "pass_a": {"status": "PASS", "total": 8}},
          "performance": {"status": "PASS", "ratio": "2.5x"}}
    ws = _make_ws(tmp_path, vj, files=["model_new_ascendc.py", "model.py"])
    result = getattr(fp, '_check_universal_entrypoints')(ws, vj)
    assert result is not None
    assert "perf.ratio" in result


def test_universal_full_pass_accepted(tmp_path):
    vj = {"precision": {"status": "PASS", "pass_a": {"status": "PASS", "total": 8}},
          "performance": {"status": "PASS", "ratio": 2.5}}
    ws = _make_ws(tmp_path, vj, files=["model_new_ascendc.py", "model.py"])
    assert getattr(fp, '_check_universal_entrypoints')(ws, vj) is None


def test_universal_partial_not_gated(tmp_path):
    vj = {"precision": {"status": "PARTIAL", "pass_a": {"status": "PARTIAL"}}}
    ws = _make_ws(tmp_path, vj)
    assert getattr(fp, '_check_universal_entrypoints')(ws, vj) is None


# Tests for _check_a5_verify_path_provenance (additional aclnn artifact check;
# currently dormant but ready to be wired)

def test_aclnn_pass_no_truth_source_rejected(tmp_path):
    """Gate is port_a3-specific (DEBT-094 phase 2 plugin migration);
    workspace must declare port_a3 mode for the gate to fire.
    """
    vj = {"precision": {"status": "PASS", "pass_a": {"status": "PASS", "total": 8}}}
    ws = _make_ws(tmp_path, vj, port_a3=True)
    result = getattr(fp, '_check_a5_verify_path_provenance')(ws, vj)
    assert result is not None
    assert "truth_source is missing" in result


def test_aclnn_pass_with_runner_accepted(tmp_path):
    vj = {
        "truth_source": "a3_cann",
        "precision": {"status": "PASS", "pass_a": {"status": "PASS", "total": 8}},
    }
    ws = _make_ws(tmp_path, vj, files=["foo_runner.cpp"], port_a3=True)
    assert getattr(fp, '_check_a5_verify_path_provenance')(ws, vj) is None


def test_aclnn_pass_on_benchmark_workspace_skips(tmp_path):
    """Plugin-dispatched gate: benchmark workspace (no port_a3 detect)
    means the gate doesn't fire. Documents the phase 2 contract.
    """
    vj = {"precision": {"status": "PASS", "pass_a": {"status": "PASS", "total": 8}}}
    ws = _make_ws(tmp_path, vj)  # no port_a3 flag → benchmark detection fails too (no kernel/, no model.py)
    assert getattr(fp, '_check_a5_verify_path_provenance')(ws, vj) is None
