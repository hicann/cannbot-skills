# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P79 (2026-05-15) — universal_entrypoints gate cross-mode coverage.

User catch (Discord 03:30Z 2026-05-15): "对于其他mode的regression检测也要做".

`_check_universal_entrypoints` was designed mode-agnostic: every op-gen
mode (port_a3_to_a5 / backward) MUST emit
`model.py` + `model_new_ascendc.py` so the safety net scanner targets
one fixed file set and covers every mode uniformly (user directive
2026-05-14: 文件名列表必须在不同mode都对齐).

This test pins the mode-agnostic property: regardless of opgen_mode
declared in .opgen_state.json, the gate behaves identically for the
same input shape.
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

from finalize_pipeline import _check_universal_entrypoints  # type: ignore


_ALL_MODES = ("port_a3_to_a5", "backward")


def _make_ws(tmp_path: Path, mode: str, *,
             with_model: bool, with_model_new: bool) -> tuple[Path, dict]:
    ws = tmp_path / f"ws_{mode}"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(json.dumps({
        "op": "fixture", "opgen_mode": mode,
    }))
    if with_model:
        (ws / "model.py").write_text(
            "import torch.nn as nn\nclass Model(nn.Module):\n    def forward(self,x): return x\n"
        )
    if with_model_new:
        (ws / "model_new_ascendc.py").write_text(
            "import torch.nn as nn\nclass ModelNew(nn.Module):\n    def forward(self,x): return x\n"
            "if __name__ == '__main__':\n    pass\n"
        )
    vj = {
        "precision": {"status": "PASS"},
        "performance": {"status": "PASS", "ratio": 2.5},
    }
    return ws, vj


@pytest.mark.parametrize("mode", _ALL_MODES)
def test_pass_with_both_entrypoints_accepted_all_modes(tmp_path, mode):
    """Happy path holds for every mode."""
    ws, vj = _make_ws(tmp_path, mode, with_model=True, with_model_new=True)
    assert _check_universal_entrypoints(ws, vj) is None, (
        f"mode={mode}: happy path with both entry-points + numeric perf "
        f"must be accepted regardless of mode"
    )


@pytest.mark.parametrize("mode", _ALL_MODES)
def test_missing_model_new_ascendc_rejected_all_modes(tmp_path, mode):
    """Missing model_new_ascendc.py must be rejected for EVERY mode —
    that's the architectural rule (文件名列表必须在不同mode都对齐).

    If a future code change adds a mode exemption, this test
    fails and forces the reviewer to re-confirm the file-name
    alignment invariant.
    """
    ws, vj = _make_ws(tmp_path, mode, with_model=True, with_model_new=False)
    err = _check_universal_entrypoints(ws, vj)
    assert err is not None, (
        f"mode={mode}: PASS without model_new_ascendc.py must be rejected — "
        f"file-name alignment invariant says ALL modes emit this file"
    )
    assert "model_new_ascendc.py" in err


@pytest.mark.parametrize("mode", _ALL_MODES)
def test_missing_model_py_rejected_all_modes(tmp_path, mode):
    """Same invariant for model.py: a PyTorch reference is required in every mode."""
    ws, vj = _make_ws(tmp_path, mode, with_model=False, with_model_new=True)
    err = _check_universal_entrypoints(ws, vj)
    assert err is not None
    assert "model.py" in err


@pytest.mark.parametrize("mode", _ALL_MODES)
def test_numeric_perf_required_all_modes(tmp_path, mode):
    """perf.status=PASS + perf.ratio=None is reward-hacking in every mode."""
    ws, vj = _make_ws(tmp_path, mode, with_model=True, with_model_new=True)
    vj["performance"]["ratio"] = None
    err = _check_universal_entrypoints(ws, vj)
    assert err is not None, (
        f"mode={mode}: PASS + ratio=None must be rejected (hidden non-measurement)"
    )
    assert "perf.ratio" in err


# ── P88c-foreach-abs-gap fixes (user catch 2026-05-15T08:18Z) ──────────

@pytest.mark.parametrize("mode", _ALL_MODES)
def test_perf_status_null_rejected_all_modes(tmp_path, mode):
    """User catch 2026-05-15T08:18Z (foreach_abs): worker set
    perf.status=null + a custom field 'ratio_status=DEFERRED_TO_FINALIZE_PIPELINE'.
    Gate previously only checked PASS+missing-ratio; missed unset/null.
    Now reject silent null.
    """
    ws, vj = _make_ws(tmp_path, mode, with_model=True, with_model_new=True)
    vj["performance"] = {"status": None}  # silent evasion
    err = _check_universal_entrypoints(ws, vj)
    assert err is not None, (
        f"mode={mode}: perf.status=null must be rejected as silent evasion"
    )
    assert "perf.status is unset" in err or "silent" in err.lower() or "None" in err


@pytest.mark.parametrize("mode", _ALL_MODES)
def test_perf_status_na_requires_reason(tmp_path, mode):
    """N/A status without explicit reason is hidden evasion."""
    ws, vj = _make_ws(tmp_path, mode, with_model=True, with_model_new=True)
    vj["performance"] = {"status": "N/A"}  # missing reason
    err = _check_universal_entrypoints(ws, vj)
    assert err is not None
    assert "reason" in err


def test_perf_status_na_with_reason_accepted(tmp_path):
    """N/A + non-empty reason is the legitimate non-measurement path."""
    ws, vj = _make_ws(tmp_path, "port_a3_to_a5", with_model=True, with_model_new=True)
    vj["performance"] = {
        "status": "N/A",
        "reason": "L1 verbatim mirror — testing upstream-built binary, not our authored kernel",
    }
    assert _check_universal_entrypoints(ws, vj) is None


@pytest.mark.parametrize("noncanonical", ["DEFERRED", "PENDING", "TBD", "?"])
def test_noncanonical_perf_status_rejected(tmp_path, noncanonical):
    """Custom non-canonical perf.status values reject — agents must commit
    to PASS / PASS_WITHIN_TOLERANCE / N/A explicitly.
    """
    ws, vj = _make_ws(tmp_path, "port_a3_to_a5", with_model=True, with_model_new=True)
    vj["performance"] = {"status": noncanonical}
    err = _check_universal_entrypoints(ws, vj)
    assert err is not None
    assert "canonical" in err.lower() or "Allowed" in err


# ── P94 attack vector fixes ──────────────────────────────────────────────

def test_p94_t1_by_construction_pass_rejected(tmp_path):
    """WORKER-PRECISION-T1-BY-CONSTRUCTION: tier=T1_BY_CONSTRUCTION
    with precision.status=PASS is fraud. PASS requires measured data,
    not textual reasoning.
    """
    ws, vj = _make_ws(tmp_path, "port_a3_to_a5", with_model=True, with_model_new=True)
    vj["precision"]["pass_a"] = {"status": "PASS", "tier": "T1_BY_CONSTRUCTION"}
    err = _check_universal_entrypoints(ws, vj)
    assert err is not None
    assert "T1_BY_CONSTRUCTION" in err and "FRAUD" in err


def test_p94_t1_by_construction_with_na_status_allowed(tmp_path):
    """T1_BY_CONSTRUCTION with N/A status is fine (no PASS claim)."""
    ws, vj = _make_ws(tmp_path, "port_a3_to_a5", with_model=True, with_model_new=True)
    vj["precision"]["status"] = "PARTIAL"  # not PASS — gate skips entirely
    vj["precision"]["pass_a"] = {"status": "PARTIAL", "tier": "T1_BY_CONSTRUCTION"}
    assert _check_universal_entrypoints(ws, vj) is None


def test_p94_mirror_with_no_execution_rejected(tmp_path):
    """WORKER-MIRROR-MD5-TRIVIAL: mirror tag PLUS textual-derivation
    method = fraud (foreach_abs original pattern).
    """
    ws, vj = _make_ws(tmp_path, "port_a3_to_a5", with_model=True, with_model_new=True)
    vj["build_evidence"] = {"stage": "L1_verbatim_mirror"}
    vj["precision"]["pass_a"] = {
        "status": "PASS",
        "method": (
            "mode_a_source_identity_from_verification_json (pass_a_runner.py "
            "independently re-derives from this block)"
        ),
    }
    err = _check_universal_entrypoints(ws, vj)
    assert err is not None
    assert "MIRROR" in err.upper() or "mirror" in err


def test_p94_mirror_with_real_execution_still_rejected(tmp_path):
    """Real measurement cannot legitimize a target/source mirror."""
    ws, vj = _make_ws(tmp_path, "port_a3_to_a5", with_model=True, with_model_new=True)
    vj["build_evidence"] = {"stage": "L1_verbatim_mirror"}
    vj["precision"]["pass_a"] = {
        "status": "PASS",
        "method": (
            "Canonical Pass A (AUTHORITATIVE per P0abh): precision_eval_two_tier.py "
            "CPU fp64 truth via Model.forward vs ModelNew.forward = 5 PASS_T1 + 3 FAIL."
        ),
    }
    err = _check_universal_entrypoints(ws, vj)
    assert err is not None
    assert "mirror" in err.lower() or "reuse" in err.lower()
