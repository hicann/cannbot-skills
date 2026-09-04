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
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import fsm_phase_o25_dispatch as dispatch
from npubench.npubench_inputs import stage_npubench_inputs
from reference_source import explicit_a3_live_binding, explicit_cannbench_binding


def _workspace(tmp_path: Path, mode: str | None, **extra) -> Path:
    workspace = tmp_path / "op"
    workspace.mkdir(parents=True)
    if mode is not None:
        state = {"opgen_mode": mode, **extra}
        (workspace / ".opgen_state.json").write_text(json.dumps(state))
    return workspace


def _migration_workspace(tmp_path: Path, *, reference=None):
    """Create a valid arch22 source stage suitable for O2.5 dispatch."""
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
    if reference is not None:
        state["reference"] = reference
    (workspace / ".opgen_state.json").write_text(json.dumps(state))
    return workspace, state, stage


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
        "reference": explicit_a3_live_binding(),
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
        "reference": explicit_a3_live_binding(),
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


def test_legacy_port_a3_state_requires_explicit_migration_before_a3_setup(
    tmp_path, monkeypatch
):
    workspace, _state, _stage = _migration_workspace(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy state must not enter live-A3 setup")

    monkeypatch.setattr(dispatch, "load_port_a3_build_source", forbidden)
    monkeypatch.setattr(dispatch, "_provision_port_a3", forbidden)

    assert dispatch.provision_reference("op", workspace, lane=9) == 2
    persisted = json.loads((workspace / ".opgen_state.json").read_text())
    assert "reference" not in persisted


def test_explicit_unknown_reference_source_fails_closed_before_a3_setup(
    tmp_path, monkeypatch
):
    workspace, state, _source_stage = _migration_workspace(
        tmp_path, reference={"source": "offline_tensor"}
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unknown reference source must not use A3 setup")

    monkeypatch.setattr(dispatch, "load_port_a3_build_source", forbidden)
    monkeypatch.setattr(dispatch, "_provision_port_a3", forbidden)

    assert dispatch.provision_reference("op", workspace, lane=7) == 2
    assert state["reference"]["source"] == "offline_tensor"


def test_partial_a3_live_binding_fails_closed_before_a3_setup(
    tmp_path, monkeypatch
):
    workspace, _state, _source_stage = _migration_workspace(
        tmp_path, reference={"source": "a3_live"}
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("partial a3_live binding must not enter live-A3 setup")

    monkeypatch.setattr(dispatch, "verify_source_stage", forbidden)
    monkeypatch.setattr(dispatch, "load_port_a3_build_source", forbidden)
    monkeypatch.setattr(dispatch, "_provision_port_a3", forbidden)

    assert dispatch.provision_reference("op", workspace, lane=8) == 2
    persisted = json.loads((workspace / ".opgen_state.json").read_text())
    assert persisted["reference"] == {"source": "a3_live"}


def test_a3_live_routes_through_verify_source_stage_seam(tmp_path, monkeypatch):
    workspace, state, stage = _migration_workspace(
        tmp_path, reference=explicit_a3_live_binding()
    )
    build_source = tmp_path / "original" / "ops-nn" / "category" / "op"
    build_source.mkdir(parents=True)
    monkeypatch.delenv(getattr(dispatch, "_PORT_A3_BUILD_SOURCE_ENV"), raising=False)
    verify_calls = []

    def fake_verify(ws, opst):
        verify_calls.append((ws, dict(opst)))
        return True, "ok", None

    monkeypatch.setattr(dispatch, "verify_source_stage", fake_verify)
    monkeypatch.setattr(
        dispatch,
        "load_port_a3_build_source",
        lambda ws, *, source_stage_digest: build_source.resolve(),
    )
    seen = []
    monkeypatch.setattr(
        dispatch,
        "_provision_port_a3",
        lambda op, ws, lane, source, build: seen.append(
            (op, ws, lane, source, build)
        ),
    )

    assert dispatch.provision_reference("op", workspace, lane=2) is None
    assert verify_calls == [(workspace, state)]
    assert seen == [
        ("op", workspace, 2, str(stage.root), str(build_source.resolve()))
    ]


def test_a3_live_rejected_source_stage_stops_before_a3_setup(tmp_path, monkeypatch):
    workspace, _state, _source_stage = _migration_workspace(
        tmp_path, reference=explicit_a3_live_binding()
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("rejected source stage must not reach A3 setup")

    monkeypatch.setattr(
        dispatch, "verify_source_stage", lambda ws, opst: (False, "tampered", None)
    )
    monkeypatch.setattr(dispatch, "load_port_a3_build_source", forbidden)
    monkeypatch.setattr(dispatch, "_provision_port_a3", forbidden)

    assert dispatch.provision_reference("op", workspace, lane=5) == 7


def _npubench_stage(tmp_path: Path, workspace: Path):
    root = tmp_path / "npu_benchmark"
    level = root / "level1"
    level.mkdir(parents=True)
    task = level / "3_Add.py"
    task.write_text("def get_input_groups():\n    return []\n")
    task.with_suffix(".json").write_text('{"case": 0}\n')
    return stage_npubench_inputs(workspace, npubench_task=task, npubench_root=root)


def test_npubench_routes_before_source_stage_or_a3_setup(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path, None)
    stage = _npubench_stage(tmp_path, workspace)
    state = {
        "opgen_mode": "port_a3_to_a5",
        "port_a3_source": "/must-not-be-read",
        "reference": stage.state_block(),
    }
    (workspace / ".opgen_state.json").write_text(json.dumps(state))
    sentinel = "/must-not-be-consumed-for-npubench"
    monkeypatch.setenv(getattr(dispatch, "_PORT_A3_BUILD_SOURCE_ENV"), sentinel)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("npubench must not read source-stage/A3 helpers")

    monkeypatch.setattr(dispatch, "verify_source_stage", forbidden)
    monkeypatch.setattr(dispatch, "load_port_a3_build_source", forbidden)
    monkeypatch.setattr(dispatch, "_provision_port_a3", forbidden)
    seen = []
    monkeypatch.setitem(
        sys.modules,
        "phase_o25_npubench",
        SimpleNamespace(
            provision_npubench_reference=lambda **kwargs: seen.append(kwargs) or 7
        ),
    )

    assert dispatch.provision_reference("op", workspace, lane=6) == 7
    assert seen == [{"workspace": workspace, "reference": state["reference"], "lane": 6}]
    assert os.environ[getattr(dispatch, "_PORT_A3_BUILD_SOURCE_ENV")] == sentinel


def test_cannbench_persists_nonretryable_unsupported_before_a3_setup(
    tmp_path, monkeypatch
):
    workspace = _workspace(
        tmp_path,
        "port_a3_to_a5",
        port_a3_source="/must-not-be-read",
        reference=explicit_cannbench_binding(),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cannbench must not read source-stage/A3 helpers")

    monkeypatch.setattr(dispatch, "verify_source_stage", forbidden)
    monkeypatch.setattr(dispatch, "load_port_a3_build_source", forbidden)
    monkeypatch.setattr(dispatch, "_provision_port_a3", forbidden)
    emitted = []
    monkeypatch.setattr(dispatch.events, "emit", lambda *args, **kwargs: emitted.append((args, kwargs)))

    assert dispatch.provision_reference("op", workspace, lane=7) == 7
    persisted = json.loads((workspace / ".opgen_state.json").read_text())
    assert persisted["reference"]["provisioning_status"] == {
        "verdict": "UNSUPPORTED_REFERENCE_SOURCE",
        "source": "cannbench",
        "retryable": False,
    }
    assert emitted


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
    ) == 2
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
