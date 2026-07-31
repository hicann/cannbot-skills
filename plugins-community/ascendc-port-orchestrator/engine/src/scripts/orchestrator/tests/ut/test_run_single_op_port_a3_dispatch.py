# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Top-level scope gates for ``run_single_op``."""
from __future__ import annotations

import json

import pytest

import orchestrator as orch


def _state(workspace, mode):
    workspace.mkdir(parents=True)
    (workspace / ".opgen_state.json").write_text(
        json.dumps({"op": workspace.name, "opgen_mode": mode})
    )


@pytest.mark.parametrize("mode", [None, "unsupported", "legacy_mode"])
def test_run_rejects_unscoped_state_before_phase_side_effects(
    tmp_path, monkeypatch, mode
):
    workspace = tmp_path / "op"
    workspace.mkdir()
    if mode is not None:
        (workspace / ".opgen_state.json").write_text(
            json.dumps({"opgen_mode": mode})
        )
    monkeypatch.setattr(
        orch.phase_o0,
        "check_hook_integrity",
        lambda *_: pytest.fail("phase O0 must not run for an unscoped workspace"),
    )
    assert orch.run_single_op("op", workspace=workspace) == 2


def test_run_rejects_alternate_backend_before_state_change(tmp_path):
    workspace = tmp_path / "op"
    _state(workspace, "backward")
    before = (workspace / ".opgen_state.json").read_bytes()
    assert orch.run_single_op("op", workspace=workspace, backend="alternate") == 2
    assert (workspace / ".opgen_state.json").read_bytes() == before


def test_run_rejects_multi_rank_before_state_change(tmp_path):
    workspace = tmp_path / "op"
    _state(workspace, "port_a3_to_a5")
    before = (workspace / ".opgen_state.json").read_bytes()
    assert orch.run_single_op("op", workspace=workspace, extra_lanes=[1]) == 2
    assert (workspace / ".opgen_state.json").read_bytes() == before
