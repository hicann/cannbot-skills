# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""CLI and pre-mutation architecture-gate tests for arch22 to arch35."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import orchestrator_cli as cli
import orchestrator_cmds as commands


@pytest.fixture
def ascendc_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(
        "\n".join(
            [
                "TARGET=a5",
                "A5_HOST=a5.test",
                "A5_USER=root",
                "A5_CONTAINER=cjm_cann_a5",
                "A5_CANN_PATH=/opt/cann",
                "A5_SOC_VERSION=Ascend950PR_9579",
                "LOCAL_PROJECT=/tmp/project",
                "A3_HOST=a3.test",
                "A3_USER=root",
                "A3_CONTAINER=cjm_cann_a3",
                "A3_CANN_PATH=/opt/cann",
                "A3_SOC_VERSION=Ascend910_9382",
            ]
        )
        + "\n"
    )
    from briefs import _common

    monkeypatch.setattr(_common, "DEFAULT_ASCENDC_ENV", env_file)
    return env_file


def _op_dir(tmp_path: Path, *, source: str) -> Path:
    op_dir = tmp_path / "ops-nn" / "demo_op"
    (op_dir / "op_host").mkdir(parents=True)
    if source == "arch22":
        kernel = op_dir / "op_kernel" / "arch22" / "demo_op.cpp"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("class DemoOp { void Process() {} };\n")
    elif source == "top_level":
        kernel = op_dir / "op_kernel" / "demo_op.cpp"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("void Process() { DataCopy<int>(0); }\n")
    elif source == "target_only":
        target = op_dir / "op_kernel" / "arch35" / "demo_op.h"
        target.parent.mkdir(parents=True)
        target.write_text("// target implementation\n")
        (op_dir / "op_kernel" / "demo_op.cpp").write_text(
            '#include "arch35/demo_op.h"\n'
        )
    else:
        (op_dir / "op_kernel").mkdir(parents=True)
    return op_dir


def _fake_engine(tmp_path: Path, *, run_rc: int = 0):
    calls = []

    def _run(*args, **kwargs):
        calls.append((args, kwargs))
        return run_rc

    return SimpleNamespace(
        WORKSPACE_ROOT=tmp_path / "workspace",
        run_single_op=_run,
        _detect_max_lane=lambda: 2,
    ), calls


def test_missing_path_and_shape_fail_without_workspace_mutation(tmp_path, monkeypatch):
    engine, calls = _fake_engine(tmp_path)
    monkeypatch.setattr(commands, "_orch", lambda: engine)

    assert getattr(commands, "_cmd_port_a3")(
        port_a3_dir=tmp_path / "missing",
        lane=0,
        plan_only=False,
        cold_start=False,
        cap_bumps={},
    ) == 2
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    assert getattr(commands, "_cmd_port_a3")(
        port_a3_dir=malformed,
        lane=0,
        plan_only=False,
        cold_start=False,
        cap_bumps={},
    ) == 2
    assert calls == []
    assert not engine.WORKSPACE_ROOT.exists()


@pytest.mark.parametrize("source", ["target_only", "unknown"])
def test_source_arch_gate_precedes_cold_start_and_env(
    tmp_path, monkeypatch, source
):
    op_dir = _op_dir(tmp_path, source=source)
    engine, calls = _fake_engine(tmp_path)
    old_workspace = engine.WORKSPACE_ROOT / op_dir.name
    old_workspace.mkdir(parents=True)
    sentinel = old_workspace / "keep.txt"
    sentinel.write_text("preserve")
    monkeypatch.setattr(commands, "_orch", lambda: engine)
    monkeypatch.setattr(
        commands,
        "_cold_start_reset_workspace",
        lambda *_: pytest.fail("cold-start mutation preceded architecture detection"),
    )

    assert getattr(commands, "_cmd_port_a3")(
        port_a3_dir=op_dir,
        lane=0,
        plan_only=False,
        cold_start=True,
        cap_bumps={},
    ) == 2
    assert sentinel.read_text() == "preserve"
    assert calls == []


@pytest.mark.parametrize("source", ["arch22", "top_level"])
def test_plan_persists_arch22_to_arch35_metadata(
    tmp_path, monkeypatch, ascendc_env, source
):
    op_dir = _op_dir(tmp_path, source=source)
    engine, calls = _fake_engine(tmp_path)
    monkeypatch.setattr(commands, "_orch", lambda: engine)

    assert getattr(commands, "_cmd_port_a3")(
        port_a3_dir=op_dir,
        lane=2,
        plan_only=True,
        cold_start=False,
        cap_bumps={},
    ) == 0
    state = json.loads(
        (engine.WORKSPACE_ROOT / op_dir.name / ".opgen_state.json").read_text()
    )
    assert state["opgen_mode"] == "port_a3_to_a5"
    assert state["source_arch"] == "arch22"
    assert state["target_arch"] == "arch35"
    assert state["source_arch_detection"]["supported"] is True
    stage = engine.WORKSPACE_ROOT / op_dir.name / ".source_arch22"
    assert state["port_a3_source"] == str(stage)
    assert state["graybox_arch22_dir"] == str(stage)
    assert state["graybox_sandbox"] is True
    assert state["source_stage_digest"]
    assert str(op_dir) not in json.dumps(state)
    assert calls == []


def test_live_entry_invokes_engine_after_state_seed(
    tmp_path, monkeypatch, ascendc_env
):
    op_dir = _op_dir(tmp_path, source="arch22")
    engine, calls = _fake_engine(tmp_path, run_rc=19)
    monkeypatch.setattr(commands, "_orch", lambda: engine)

    assert getattr(commands, "_cmd_port_a3")(
        port_a3_dir=op_dir,
        lane=1,
        plan_only=False,
        cold_start=False,
        cap_bumps={"worker": 1},
        timing=True,
    ) == 19
    assert calls[0][0][0] == op_dir.name
    assert calls[0][1]["workspace"] == engine.WORKSPACE_ROOT / op_dir.name
    assert calls[0][1]["timing"] is True


def test_live_entry_build_source_is_ephemeral_not_durable(
    tmp_path, monkeypatch, ascendc_env
):
    op_dir = _op_dir(tmp_path, source="arch22").resolve()
    engine, _ = _fake_engine(tmp_path)
    seen_build_sources: list[str | None] = []

    def run_single_op(*args, **kwargs):
        seen_build_sources.append(
            os.environ.get("CANNBOT_PORT_A3_BUILD_SOURCE")
        )
        return 0

    engine.run_single_op = run_single_op
    monkeypatch.setattr(commands, "_orch", lambda: engine)
    monkeypatch.delenv("CANNBOT_PORT_A3_BUILD_SOURCE", raising=False)

    assert getattr(commands, "_cmd_port_a3")(
        port_a3_dir=op_dir,
        lane=2,
        plan_only=False,
        cold_start=False,
        cap_bumps={},
    ) == 0

    assert seen_build_sources == [str(op_dir)]
    assert "CANNBOT_PORT_A3_BUILD_SOURCE" not in os.environ
    state_text = (
        engine.WORKSPACE_ROOT / op_dir.name / ".opgen_state.json"
    ).read_text()
    assert str(op_dir) not in state_text


def test_parser_exposes_only_scoped_start_modes(monkeypatch, tmp_path):
    seen = []
    engine = SimpleNamespace(
        _refuse_if_detached=lambda: None,
        _detect_max_lane=lambda: 2,
        _parse_bump_caps=lambda raw: {},
        _cmd_port_a3=lambda **kwargs: seen.append(kwargs) or 17,
    )
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    source = tmp_path / "op"
    monkeypatch.setattr(
        sys,
        "argv",
        ["orchestrator", "--port-a3", str(source), "--plan", "--timing"],
    )
    assert cli.main() == 17
    assert seen[0]["port_a3_dir"] == source
    assert seen[0]["timing"] is True


def test_timing_is_forwarded_for_scoped_lifecycle(monkeypatch, tmp_path):
    calls = []
    workspace = tmp_path / "workspace" / "demo"
    engine = SimpleNamespace(
        _refuse_if_detached=lambda: None,
        _detect_max_lane=lambda: 2,
        _resolve_workspace=lambda op, backend: workspace,
        run_single_op=lambda *args, **kwargs: calls.append((args, kwargs)) or 0,
    )
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    monkeypatch.setattr(
        sys, "argv", ["orchestrator", "demo", "--plan", "--timing"]
    )

    assert cli.main() == 0
    assert calls[0][1]["timing"] is True


@pytest.mark.parametrize(
    "removed_flag",
    ["--batch", "--port", "--world-size", "--backend"],
)
def test_removed_start_flags_are_not_parseable(monkeypatch, removed_flag):
    engine = SimpleNamespace(_refuse_if_detached=lambda: None)
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    monkeypatch.setattr(sys, "argv", ["orchestrator", removed_flag, "value"])
    with pytest.raises(BaseException) as exc:
        cli.main()
    assert type(exc.value).__name__ == "SystemExit"
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--port-a3", "/tmp/op", "--dry-run"],
        ["--backward", "/tmp/forward.py", "--dry-run"],
        ["demo", "--resume", "--workspace", "/tmp/ws"],
        ["demo", "--resume", "--bump-cap", "worker:1"],
        ["demo", "--resume", "--precision-standard", "ecosystem"],
    ],
)
def test_irrelevant_mode_flags_are_rejected(monkeypatch, argv):
    calls = []
    engine = SimpleNamespace(
        _refuse_if_detached=lambda: None,
        _detect_max_lane=lambda: 2,
        _cmd_port_a3=lambda **kwargs: calls.append(kwargs) or 0,
        _cmd_backward=lambda **kwargs: calls.append(kwargs) or 0,
        _cmd_resume=lambda **kwargs: calls.append(kwargs) or 0,
    )
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    monkeypatch.setattr(sys, "argv", ["orchestrator", *argv])
    assert cli.main() == 2
    assert calls == []


@pytest.mark.parametrize("lane", ["-1", "3"])
def test_lane_outside_detected_range_is_rejected(monkeypatch, lane, tmp_path):
    calls = []
    engine = SimpleNamespace(
        _refuse_if_detached=lambda: None,
        _detect_max_lane=lambda: 2,
        _parse_bump_caps=lambda raw: {},
        _cmd_port_a3=lambda **kwargs: calls.append(kwargs) or 0,
    )
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    monkeypatch.setattr(
        sys,
        "argv",
        ["orchestrator", "--port-a3", str(tmp_path / "op"), "--lane", lane],
    )
    assert cli.main() == 2
    assert calls == []


def test_detected_max_lane_is_forwarded(monkeypatch, tmp_path):
    calls = []
    engine = SimpleNamespace(
        _refuse_if_detached=lambda: None,
        _detect_max_lane=lambda: 2,
        _parse_bump_caps=lambda raw: {},
        _cmd_port_a3=lambda **kwargs: calls.append(kwargs) or 0,
    )
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    monkeypatch.setattr(
        sys,
        "argv",
        ["orchestrator", "--port-a3", str(tmp_path / "op"), "--lane", "2"],
    )
    assert cli.main() == 0
    assert calls[0]["lane"] == 2
