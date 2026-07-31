# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Scoped Phase O2.5 dispatch tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import fsm_phase_o25_dispatch as dispatch


def _workspace(tmp_path: Path, mode: str | None, **extra) -> Path:
    workspace = tmp_path / "op"
    workspace.mkdir(parents=True)
    if mode is not None:
        state = {"opgen_mode": mode, **extra}
        (workspace / ".opgen_state.json").write_text(json.dumps(state))
    return workspace


@pytest.mark.parametrize("mode", [None, "unsupported", "legacy_mode"])
def test_unsupported_or_missing_mode_fails_closed(tmp_path, mode):
    workspace = _workspace(tmp_path, mode)
    assert dispatch.provision_reference("op", workspace, lane=0) == 2


def test_port_a3_routes_only_to_port_handler(tmp_path, monkeypatch):
    from source_arch import stage_source_tree

    source = tmp_path / "source"
    kernel = source / "op_kernel" / "arch22" / "op.h"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("class Op { void Process() {} };\n")
    workspace = _workspace(tmp_path, None)
    stage = stage_source_tree(source, workspace)
    state = {
        "opgen_mode": "port_a3_to_a5",
        "port_a3_source": str(stage.root),
        "source_stage_manifest": str(stage.manifest),
        "source_stage_digest": stage.digest,
        "graybox_sandbox": True,
        "graybox_arch22_dir": str(stage.root),
    }
    (workspace / ".opgen_state.json").write_text(json.dumps(state))
    seen = []
    build_source = tmp_path / "original" / "ops-nn" / "category" / "op"
    build_source.mkdir(parents=True)
    monkeypatch.setenv(
        getattr(dispatch, '_PORT_A3_BUILD_SOURCE_ENV'),
        str(build_source),
    )
    monkeypatch.setattr(
        dispatch,
        "load_port_a3_build_source",
        lambda ws, *, source_stage_digest: build_source.resolve(),
    )
    monkeypatch.setattr(
        dispatch,
        "_provision_port_a3",
        lambda op, ws, lane, source, build_source: seen.append(
            (op, ws, lane, source, build_source)
        ) or 7,
    )
    assert dispatch.provision_reference("op", workspace, lane=3) == 7
    assert seen == [
        (
            "op",
            workspace,
            3,
            str(stage.root),
            str(build_source.resolve()),
        )
    ]
    assert getattr(dispatch, '_PORT_A3_BUILD_SOURCE_ENV') not in os.environ


def test_port_a3_resume_recovers_private_build_source_without_process_env(
    tmp_path, monkeypatch
):
    from source_arch import stage_source_tree

    source = tmp_path / "source"
    kernel = source / "op_kernel" / "arch22" / "op.h"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("class Op { void Process() {} };\n")
    workspace = _workspace(tmp_path, None)
    stage = stage_source_tree(source, workspace)
    state = {
        "opgen_mode": "port_a3_to_a5",
        "port_a3_source": str(stage.root),
        "source_stage_manifest": str(stage.manifest),
        "source_stage_digest": stage.digest,
        "graybox_sandbox": True,
        "graybox_arch22_dir": str(stage.root),
    }
    (workspace / ".opgen_state.json").write_text(json.dumps(state))
    build_source = tmp_path / "original" / "ops-nn" / "category" / "op"
    build_source.mkdir(parents=True)
    seen = []
    monkeypatch.delenv(getattr(dispatch, '_PORT_A3_BUILD_SOURCE_ENV'), raising=False)
    monkeypatch.setattr(
        dispatch,
        "load_port_a3_build_source",
        lambda ws, *, source_stage_digest: build_source.resolve(),
    )
    monkeypatch.setattr(
        dispatch,
        "_provision_port_a3",
        lambda op, ws, lane, staged, build: seen.append(
            (op, ws, lane, staged, build)
        ),
    )

    assert dispatch.provision_reference("op", workspace, lane=4) is None
    assert seen == [
        (
            "op",
            workspace,
            4,
            str(stage.root),
            str(build_source.resolve()),
        )
    ]


def test_backward_routes_only_to_backward_handler(tmp_path, monkeypatch):
    workspace = _workspace(
        tmp_path, "backward", backward_forward_source="/source/forward.py"
    )
    seen = []
    monkeypatch.setattr(
        dispatch,
        "_provision_backward",
        lambda op, ws, lane, source: seen.append((op, ws, lane, source)),
    )
    assert dispatch.provision_reference("op", workspace, lane=1) is None
    assert seen == [("op", workspace, 1, "/source/forward.py")]


def test_scoped_modes_require_their_customer_source(tmp_path):
    assert dispatch.provision_reference(
        "op", _workspace(tmp_path / "a", "port_a3_to_a5"), lane=0
    ) == 7
    assert dispatch.provision_reference(
        "op", _workspace(tmp_path / "b", "backward"), lane=0
    ) == 7


@pytest.mark.parametrize(
    "verdict",
    [
        "A3_UNREACHABLE",
        "A3_BUSY",
        "BUILD_FAILED",
        "CAPTURE_INCOMPLETE",
        "EXEC_FAILED",
        "INPUT_GEN_FAILED",
        "MISSING_DEPS",
        "MISSING_ENTRY",
        "READY_PROBE_ONLY",
        "RUNNER_MISSING",
    ],
)
def test_port_a3_requires_live_ready(tmp_path, monkeypatch, verdict):
    workspace = _workspace(tmp_path, "port_a3_to_a5")
    source = tmp_path / "source"
    source.mkdir()

    from briefs import _common
    import phase_o25_a3_ref

    monkeypatch.setattr(
        _common,
        "load_env",
        lambda: SimpleNamespace(
            a3_host="a3.test",
            a3_user="root",
            a3_container="cjm_cann",
            a3_cann_path="/opt/cann",
        ),
    )
    calls = []

    def _provision(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(verdict=verdict, errors=[verdict], summary=verdict)

    monkeypatch.setattr(phase_o25_a3_ref, "provision_a3_reference", _provision)
    monkeypatch.setattr(phase_o25_a3_ref, "format_block_message", lambda *_: verdict)

    assert getattr(dispatch, '_provision_port_a3')("op", workspace, 0, source) == 7
    assert calls[0]["probe_only"] is False
    assert not (workspace / ".truth_source_override").exists()
    assert not (workspace / ".a3_ref_unavailable_cpu_truth_deferred").exists()


def test_port_a3_live_ready_proceeds(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path, "port_a3_to_a5")
    source = tmp_path / "source"
    source.mkdir()

    from briefs import _common
    import phase_o25_a3_ref

    monkeypatch.setattr(
        _common,
        "load_env",
        lambda: SimpleNamespace(
            a3_host="a3.test",
            a3_user="root",
            a3_container="cjm_cann",
            a3_cann_path="/opt/cann",
        ),
    )
    monkeypatch.setattr(
        phase_o25_a3_ref,
        "provision_a3_reference",
        lambda **_: SimpleNamespace(verdict="READY", errors=[], summary="live"),
    )
    assert getattr(dispatch, '_provision_port_a3')("op", workspace, 0, source) is None
