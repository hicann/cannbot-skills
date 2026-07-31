# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase O0.5 durable-state tests for the two supported customer workflows."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o05  # noqa: E402


def _init(workspace: Path, op: str = "test_op", **kwargs):
    return phase_o05.init_durable_state(
        workspace, op, opgen_mode="port_a3_to_a5", **kwargs
    )


def test_initialized_on_first_run(tmp_path):
    rep = _init(tmp_path, lane=0, target="a5")
    assert rep.verdict == "INITIALIZED"
    state = json.loads((tmp_path / phase_o05.STATE_FILE).read_text())
    assert state["op"] == "test_op"
    assert state["target"] == "a5"
    assert state["lane"] == 0
    assert state["opgen_mode"] == "port_a3_to_a5"
    assert state["backend"] == "ascendc"
    assert state["invocation_count"] == 1
    assert state["schema_version"] == phase_o05.SCHEMA_VERSION


def test_refreshed_on_subsequent_runs(tmp_path):
    _init(tmp_path)
    started = json.loads((tmp_path / phase_o05.STATE_FILE).read_text())["started_ts"]
    rep = _init(tmp_path, lane=2)
    state = json.loads((tmp_path / phase_o05.STATE_FILE).read_text())
    assert rep.verdict == "REFRESHED"
    assert state["invocation_count"] == 2
    assert state["started_ts"] == started
    assert state["lane"] == 2


def test_requires_explicit_supported_mode(tmp_path):
    with pytest.raises(ValueError, match="must be explicit"):
        phase_o05.init_durable_state(tmp_path, "test_op")
    with pytest.raises(ValueError, match="must be explicit"):
        phase_o05.init_durable_state(tmp_path, "test_op", opgen_mode="unsupported")


def test_rejects_non_ascendc_backend(tmp_path):
    with pytest.raises(ValueError, match="only the AscendC backend"):
        phase_o05.init_durable_state(
            tmp_path, "test_op", opgen_mode="backward", backend="unsupported"
        )


def test_rejects_mode_change_on_refresh(tmp_path):
    _init(tmp_path)
    with pytest.raises(ValueError, match="workspace mode conflict"):
        phase_o05.init_durable_state(
            tmp_path, "test_op", opgen_mode="backward"
        )


def test_rejects_legacy_backend_on_refresh(tmp_path):
    (tmp_path / phase_o05.STATE_FILE).write_text(json.dumps({
        "op": "test_op",
        "opgen_mode": "port_a3_to_a5",
        "backend": "unsupported",
        "started_ts": "origin",
    }))
    with pytest.raises(ValueError, match="workspace backend conflict"):
        _init(tmp_path)


def test_op_name_mismatch_reinitializes(tmp_path):
    _init(tmp_path, "first_op")
    rep = _init(tmp_path, "different_op")
    state = json.loads((tmp_path / phase_o05.STATE_FILE).read_text())
    assert rep.verdict == "INITIALIZED"
    assert state["op"] == "different_op"
    assert state["invocation_count"] == 1


def test_malformed_state_file_rewritten_with_explicit_mode(tmp_path):
    (tmp_path / phase_o05.STATE_FILE).write_text("{ not json")
    assert _init(tmp_path).verdict == "INITIALIZED"
    assert phase_o05.read_durable_state(tmp_path)["opgen_mode"] == "port_a3_to_a5"


def test_read_absent_and_malformed(tmp_path):
    assert phase_o05.read_durable_state(tmp_path) is None
    (tmp_path / phase_o05.STATE_FILE).write_text("{ not json")
    assert phase_o05.read_durable_state(tmp_path) is None


def test_creates_workspace_dir(tmp_path):
    ws = tmp_path / "newop"
    assert _init(ws, "newop").verdict == "INITIALIZED"
    assert (ws / phase_o05.STATE_FILE).exists()
