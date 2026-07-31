# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DEBT-NEW (2026-05-14) regression test — A5 verify-path fraud gate.

User catch 2026-05-14: "your reward hacking make us lost 2 days". 4 of 7
"DONE" port_a3 archives (ctc_loss_v3, foreach_abs, rms_norm_quant,
gather_elements_v2) shipped PASS verdicts without actually executing our
built kernel on A5 — they routed verify through PyTorch dispatcher which
silently fell back to AICPU. The safety net was a name-coupled scanner
that didn't know to look at port_a3 verify files.

User architectural fix 2026-05-14: 文件名列表必须在不同mode都对齐 —
align canonical entry-point file names across all op-gen modes
(`model_new_ascendc.py` + `model.py`) so the safety net targets one set
of names and covers every mode uniformly.

This test enumerates the 4 known-bad fixture archives and asserts the
finalize gate rejects PASS for each. It is THE regression test that
must always pass — if it ever passes a known-bad fixture, the safety
net has regressed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

from finalize_pipeline import (  # type: ignore
    GateID,
    _check_universal_entrypoints,
)


def _make_workspace(
    tmp_path: Path,
    *,
    with_model_new: bool,
    with_model: bool,
    precision_status: str = "PASS",
    perf_status: str | None = "PASS",
    perf_ratio: float | None = 2.5,
    truth_source: str | None = "a3_cann",
) -> tuple[Path, dict]:
    ws = tmp_path / "workspace_op"
    ws.mkdir(parents=True, exist_ok=True)
    if with_model_new:
        (ws / "model_new_ascendc.py").write_text(
            "import torch.nn as nn\n"
            "class ModelNew(nn.Module):\n"
            "    def forward(self, x): return x\n"
            "if __name__ == '__main__':\n    pass\n"
        )
    if with_model:
        (ws / "model.py").write_text(
            "import torch.nn as nn\n"
            "class Model(nn.Module):\n"
            "    def forward(self, x): return x\n"
        )
    vj = {
        "schema_version": 1,
        "op": "fixture_op",
        "truth_source": truth_source,
        "precision": {
            "status": precision_status,
            "pass_a": {
                "status": precision_status, "tier1_pass": 8, "tier2_pass": 0,
                "fail": 0, "total": 8,
            },
        },
        "performance": {
            "status": perf_status, "ratio": perf_ratio,
        },
    }
    return ws, vj


def test_pass_with_both_entrypoints_and_numeric_perf_is_eligible(tmp_path):
    """Happy path: model_new_ascendc.py + model.py + numeric perf.ratio → eligible."""
    ws, vj = _make_workspace(
        tmp_path, with_model_new=True, with_model=True,
        perf_ratio=2.5,
    )
    assert _check_universal_entrypoints(ws, vj) is None


def test_pass_without_model_new_ascendc_rejected(tmp_path):
    """ctc_loss_v3 / foreach_abs / rms_norm_quant / gather_elements_v2 pattern:
    PASS verdict but no model_new_ascendc.py at workspace root → REJECT.

    This is the load-bearing assertion. All 4 known-bad archives had:
    - precision.status == PASS
    - no model_new_ascendc.py (port_a3 mode never emitted it)
    Without model_new_ascendc.py the safety net scanner has nothing to
    inspect and the wiring-only fraud propagates.
    """
    ws, vj = _make_workspace(
        tmp_path, with_model_new=False, with_model=True,
    )
    err = _check_universal_entrypoints(ws, vj)
    assert err is not None, "gate must reject PASS without model_new_ascendc.py"
    assert "model_new_ascendc.py" in err
    assert "文件名列表必须在不同mode都对齐" in err or "PASS" in err


def test_pass_without_model_py_rejected(tmp_path):
    """No model.py → REJECT (no reference Module to compare against)."""
    ws, vj = _make_workspace(
        tmp_path, with_model_new=True, with_model=False,
    )
    err = _check_universal_entrypoints(ws, vj)
    assert err is not None
    assert "model.py" in err


def test_pass_with_perf_ratio_none_rejected(tmp_path):
    """Per user directive 2026-05-14 '不能获取性能数据要及时报错并退出, 绝对不能
    报pass': PASS + perf.status=PASS + perf.ratio=None → REJECT.

    This catches the rms_norm_quant pattern where perf.status=PASS was
    set but the actual measurement was bogus (CPU torch.rsqrt vs A3 NPU
    aclnn). Ratio reported but if it had been None, the gate must catch.
    """
    ws, vj = _make_workspace(
        tmp_path, with_model_new=True, with_model=True,
        perf_status="PASS", perf_ratio=None,
    )
    err = _check_universal_entrypoints(ws, vj)
    assert err is not None, "gate must reject PASS with perf.status=PASS but ratio=None"
    assert "perf.ratio" in err
    # P88c (2026-05-15T08:18Z): message refined to "measured-class statuses
    # REQUIRE numeric ratio" — semantics unchanged.
    assert "ratio" in err.lower() and ("require" in err.lower() or "reward" in err.lower())


def test_perf_status_na_with_ratio_none_allowed(tmp_path):
    """If perf.status='N/A' WITH non-empty reason, ratio=None is fine.
    P88c strictening (2026-05-15T08:18Z): N/A requires explicit reason
    (empty reason is now hidden evasion).
    """
    ws, vj = _make_workspace(
        tmp_path, with_model_new=True, with_model=True,
        perf_status="N/A", perf_ratio=None,
    )
    vj["performance"]["reason"] = "L1 verbatim mirror — testing upstream binary"
    assert _check_universal_entrypoints(ws, vj) is None


def test_non_pass_verdict_skipped(tmp_path):
    """Non-PASS verdicts (PARTIAL / FAIL / N/A) handled by other gates;
    this gate only fires on claimed-PASS to catch reward-hacking.
    """
    ws, vj = _make_workspace(
        tmp_path, with_model_new=False, with_model=False,
        precision_status="PARTIAL",
    )
    assert _check_universal_entrypoints(ws, vj) is None


def test_gate_id_value_stable():
    """Lock the gate ID string so rollback-loop detector keys correctly."""
    assert GateID.ACLNN_VERIFY_PATH_FRAUD.value == "aclnn_verify_path_fraud"
