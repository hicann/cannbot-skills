# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Prior-art markers are advisory and cannot replace generation or verification."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import finalize_pipeline as fp


def _seed_workspace(tmp_path: Path, verdict: str | None) -> Path:
    ws = tmp_path / "test_op"
    ws.mkdir()
    vj = {
        "op": "test_op",
        "mode": "port_a3_to_a5",
        "verdict": verdict,
        "precision": {"status": "N/A", "pass_a": {"status": "N/A"}, "pass_b": {"status": "N/A"}},
        "performance": {"status": "N/A", "ratio": None, "reason": "skip path"},
    }
    if verdict is None:
        del vj["verdict"]
    (ws / "verification.json").write_text(json.dumps(vj))
    return ws


def test_bare_skip_upstream_rejected(tmp_path):
    ws = _seed_workspace(tmp_path, "SKIP_UPSTREAM_HAS_REFERENCE")
    r = fp.check_finalize_eligibility(ws)
    assert not r["eligible"]
    assert r["gate"] == fp.GateID.UNKNOWN_PRECISION_STATUS.value


def test_verified_suffix_skip_upstream_rejected(tmp_path):
    ws = _seed_workspace(tmp_path, "SKIP_UPSTREAM_HAS_REFERENCE_VERIFIED")
    r = fp.check_finalize_eligibility(ws)
    assert not r["eligible"]
    assert r["gate"] == fp.GateID.UNKNOWN_PRECISION_STATUS.value


def test_marker_file_skip_upstream_rejected(tmp_path):
    ws = _seed_workspace(tmp_path, None)
    (ws / "SKIP_UPSTREAM_HAS_REFERENCE.md").write_text("# Skip marker\n")
    r = fp.check_finalize_eligibility(ws)
    assert not r["eligible"]
    assert r["gate"] == fp.GateID.UNKNOWN_PRECISION_STATUS.value


def test_unknown_status_without_skip_verdict_rejected(tmp_path):
    """Without any SKIP signal, status=N/A → rollback (unchanged behavior)."""
    ws = _seed_workspace(tmp_path, None)
    r = fp.check_finalize_eligibility(ws)
    assert not r["eligible"]
    assert r["gate"] == fp.GateID.UNKNOWN_PRECISION_STATUS.value


def test_unrelated_skip_verdict_rejected(tmp_path):
    """A different verdict string MUST not bypass the gate."""
    ws = _seed_workspace(tmp_path, "SKIP_BECAUSE_REASONS")
    r = fp.check_finalize_eligibility(ws)
    assert not r["eligible"]
