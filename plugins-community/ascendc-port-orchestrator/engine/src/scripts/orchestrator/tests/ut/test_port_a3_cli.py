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
from reference_source import explicit_a3_live_binding


_A5_ENV_LINES = (
    "TARGET=a5",
    "A5_HOST=a5.test",
    "A5_USER=root",
    "A5_CONTAINER=cjm_cann_a5",
    "A5_CANN_PATH=/opt/cann",
    "A5_SOC_VERSION=Ascend950PR_9579",
    "LOCAL_PROJECT=/tmp/project",
)

_A3_ENV_LINES = (
    "A3_HOST=a3.test",
    "A3_USER=root",
    "A3_CONTAINER=cjm_cann_a3",
    "A3_CANN_PATH=/opt/cann",
    "A3_SOC_VERSION=Ascend910_9382",
)


def _write_ascendc_env(tmp_path, monkeypatch, lines) -> Path:
    """Write a ``.ascendc_env`` file and point the brief loader at it."""
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text("\n".join(lines) + "\n")
    from briefs import _common

    monkeypatch.setattr(_common, "DEFAULT_ASCENDC_ENV", env_file)
    return env_file


@pytest.fixture
def ascendc_env(tmp_path, monkeypatch):
    return _write_ascendc_env(
        tmp_path,
        monkeypatch,
        _A5_ENV_LINES + ("PORT_A3_REFERENCE_SOURCE=a3_live",) + _A3_ENV_LINES,
    )


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


def _npubench_task(tmp_path: Path) -> tuple[Path, Path, bytes, bytes]:
    root = tmp_path / "npu_benchmark"
    level = root / "level1"
    level.mkdir(parents=True)
    task = level / "3_Add.py"
    task_bytes = b"def get_input_groups():\n    return []\n"
    sidecar_bytes = b'{"case": 0}\n{"case": 1}\n'
    task.write_bytes(task_bytes)
    task.with_suffix(".json").write_bytes(sidecar_bytes)
    return root, task, task_bytes, sidecar_bytes


@pytest.fixture
def a3_live_env(tmp_path, monkeypatch):
    """A5 + A3 env without PORT_A3_REFERENCE_SOURCE: the CLI flag selects a3_live."""
    return _write_ascendc_env(tmp_path, monkeypatch, _A5_ENV_LINES + _A3_ENV_LINES)


@pytest.fixture
def a5_only_env(tmp_path, monkeypatch):
    return _write_ascendc_env(tmp_path, monkeypatch, _A5_ENV_LINES)


@pytest.fixture
def a5_local_env(tmp_path, monkeypatch):
    return _write_ascendc_env(
        tmp_path,
        monkeypatch,
        (
            "TARGET=a5",
            "A5_CONTAINER=local",
            "A5_CANN_PATH=/opt/cann",
            "A5_SOC_VERSION=Ascend950PR_9579",
            "LOCAL_PROJECT=/tmp/project",
        ),
    )


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


def _run_port_a3(**kwargs) -> int:
    """Invoke the port_a3 CLI entry, filling in the common default arguments."""
    call = {"lane": 0, "plan_only": False, "cold_start": False, "cap_bumps": {}}
    call.update(kwargs)
    return getattr(commands, "_cmd_port_a3")(**call)


def _run_port_a3_npubench(op_dir, task, task_root, **kwargs) -> int:
    """Invoke the CLI entry with an NPUKernelBench reference provider."""
    return _run_port_a3(
        port_a3_dir=op_dir,
        reference_source="npubench",
        npubench_task=task,
        npubench_root=task_root,
        **kwargs,
    )


def _fake_cli_engine(sink, *, rc: int = 0) -> SimpleNamespace:
    """CLI-level stub engine recording the kwargs a parsed argv forwards."""
    return SimpleNamespace(
        _refuse_if_detached=lambda: None,
        _detect_max_lane=lambda: 2,
        _parse_bump_caps=lambda raw: {},
        _cmd_port_a3=lambda **kwargs: sink.append(kwargs) or rc,
    )




def test_missing_path_and_shape_fail_without_workspace_mutation(tmp_path, monkeypatch):
    engine, calls = _fake_engine(tmp_path)
    monkeypatch.setattr(commands, "_orch", lambda: engine)

    assert _run_port_a3(
        port_a3_dir=tmp_path / "missing",
    ) == 2
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    assert _run_port_a3(
        port_a3_dir=malformed,
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

    assert _run_port_a3(
        port_a3_dir=op_dir,
        cold_start=True,
    ) == 2
    assert sentinel.read_text() == "preserve"
    assert calls == []


@pytest.mark.parametrize("source", ["arch22", "top_level"])
def test_plan_validates_arch22_to_arch35_without_workspace_mutation(
    tmp_path, monkeypatch, ascendc_env, source
):
    op_dir = _op_dir(tmp_path, source=source)
    engine, calls = _fake_engine(tmp_path)
    monkeypatch.setattr(commands, "_orch", lambda: engine)

    assert _run_port_a3(
        port_a3_dir=op_dir,
        lane=2,
        plan_only=True,
    ) == 0
    assert not engine.WORKSPACE_ROOT.exists()
    assert calls == []


def test_plan_has_no_cold_start_or_reference_side_effects(
    tmp_path, monkeypatch, a5_only_env
):
    op_dir = _op_dir(tmp_path, source="arch22")
    task_root, task, _task_bytes, _sidecar_bytes = _npubench_task(tmp_path)
    engine, calls = _fake_engine(tmp_path)
    existing = engine.WORKSPACE_ROOT / op_dir.name
    existing.mkdir(parents=True)
    sentinel = existing / "keep.txt"
    sentinel.write_text("preserve")
    monkeypatch.setattr(commands, "_orch", lambda: engine)
    monkeypatch.setattr(
        commands,
        "_cold_start_reset_workspace",
        lambda *_: pytest.fail("--plan must not cold-start a workspace"),
    )

    assert _run_port_a3_npubench(op_dir, task, task_root, lane=2, plan_only=True, cold_start=True) == 0
    assert sentinel.read_text() == "preserve"
    assert not (existing / "reference_inputs").exists()
    assert not (existing / ".source_arch22").exists()
    assert calls == []


def test_bare_port_a3_requires_explicit_reference_source(
    tmp_path, monkeypatch, a5_only_env, capsys
):
    op_dir = _op_dir(tmp_path, source="arch22")
    engine, calls = _fake_engine(tmp_path)
    monkeypatch.setattr(commands, "_orch", lambda: engine)

    assert _run_port_a3(
        port_a3_dir=op_dir,
    ) == 2

    out = capsys.readouterr().out
    assert "requires an explicit reference provider" in out
    assert "--reference-source npubench --npubench-task" in out
    assert "--reference-source a3_live" in out
    assert "model_reference" not in out
    assert not engine.WORKSPACE_ROOT.exists()
    assert calls == []


def test_explicit_a3_live_reference_source_is_accepted_and_bound(
    tmp_path, monkeypatch, a3_live_env
):
    op_dir = _op_dir(tmp_path, source="arch22").resolve()
    engine, calls = _fake_engine(tmp_path)
    monkeypatch.setattr(commands, "_orch", lambda: engine)

    assert _run_port_a3(
        port_a3_dir=op_dir,
        lane=1,
        reference_source="a3_live",
    ) == 0

    assert calls[0][0][0] == op_dir.name
    state = json.loads(
        (engine.WORKSPACE_ROOT / op_dir.name / ".opgen_state.json").read_text()
    )
    assert state["reference"] == explicit_a3_live_binding()


def test_a3_live_requires_a3_host_and_container(
    tmp_path, monkeypatch, a5_only_env, capsys
):
    op_dir = _op_dir(tmp_path, source="arch22")
    engine, calls = _fake_engine(tmp_path)
    monkeypatch.setattr(commands, "_orch", lambda: engine)

    assert _run_port_a3(
        port_a3_dir=op_dir,
        reference_source="a3_live",
    ) == 2

    out = capsys.readouterr().out
    assert "A3_HOST" in out
    assert "A3_CONTAINER" in out
    assert not engine.WORKSPACE_ROOT.exists()
    assert calls == []


def test_live_entry_invokes_engine_after_state_seed(
    tmp_path, monkeypatch, ascendc_env
):
    op_dir = _op_dir(tmp_path, source="arch22")
    engine, calls = _fake_engine(tmp_path, run_rc=19)
    monkeypatch.setattr(commands, "_orch", lambda: engine)

    assert _run_port_a3(
        port_a3_dir=op_dir,
        lane=1,
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

    assert _run_port_a3(
        port_a3_dir=op_dir,
        lane=2,
    ) == 0

    assert seen_build_sources == [str(op_dir)]
    assert "CANNBOT_PORT_A3_BUILD_SOURCE" not in os.environ
    state_text = (
        engine.WORKSPACE_ROOT / op_dir.name / ".opgen_state.json"
    ).read_text()
    assert str(op_dir) not in state_text


def test_npubench_accepts_hostless_explicit_local_a5_target(
    tmp_path, monkeypatch, a5_local_env
):
    op_dir = _op_dir(tmp_path, source="arch22").resolve()
    task_root, task, _task_bytes, _sidecar_bytes = _npubench_task(tmp_path)
    engine, calls = _fake_engine(tmp_path)
    monkeypatch.setattr(commands, "_orch", lambda: engine)

    assert _run_port_a3_npubench(op_dir, task, task_root, lane=1) == 0

    assert calls[0][0][0] == op_dir.name
    state = json.loads((engine.WORKSPACE_ROOT / op_dir.name / ".opgen_state.json").read_text())
    assert state["reference"]["source"] == "npubench"


def test_npubench_entry_stages_original_task_and_skips_a3_build_source(
    tmp_path, monkeypatch, a5_only_env
):
    op_dir = _op_dir(tmp_path, source="arch22").resolve()
    task_root, task, task_bytes, sidecar_bytes = _npubench_task(tmp_path)
    engine, calls = _fake_engine(tmp_path)
    monkeypatch.setattr(commands, "_orch", lambda: engine)
    monkeypatch.setattr(
        commands,
        "record_port_a3_build_source",
        lambda *_args, **_kwargs: pytest.fail("npubench must not persist an A3 build source"),
    )
    monkeypatch.setattr(
        commands,
        "verify_source_stage",
        lambda *_args, **_kwargs: pytest.fail("npubench must not verify an A3 source stage"),
    )
    sentinel = "/must-not-be-consumed-for-npubench"
    monkeypatch.setenv("CANNBOT_PORT_A3_BUILD_SOURCE", sentinel)

    assert _run_port_a3_npubench(op_dir, task, task_root, lane=1) == 0

    assert calls[0][0][0] == op_dir.name
    assert os.environ["CANNBOT_PORT_A3_BUILD_SOURCE"] == sentinel
    state = json.loads((engine.WORKSPACE_ROOT / op_dir.name / ".opgen_state.json").read_text())
    reference = state["reference"]
    assert state["schema_version"] == 3
    assert reference["source"] == "npubench"
    staged_root = engine.WORKSPACE_ROOT / op_dir.name / "reference_inputs" / "npubench" / reference["bundle_sha256"]
    assert (staged_root / "level1" / "3_Add.py").read_bytes() == task_bytes
    assert (staged_root / "level1" / "3_Add.json").read_bytes() == sidecar_bytes


def test_npubench_matrix_rejects_missing_or_mismatched_provider_inputs(
    tmp_path, monkeypatch, a5_only_env, capsys
):
    op_dir = _op_dir(tmp_path, source="arch22")
    task_root, task, _task_bytes, _sidecar_bytes = _npubench_task(tmp_path)
    engine, calls = _fake_engine(tmp_path)
    monkeypatch.setattr(commands, "_orch", lambda: engine)

    assert _run_port_a3(
        port_a3_dir=op_dir,
        reference_source="npubench",
    ) == 2
    assert "requires --npubench-task" in capsys.readouterr().out
    assert not engine.WORKSPACE_ROOT.exists()

    assert _run_port_a3(
        port_a3_dir=op_dir,
        reference_source="a3_live",
        npubench_task=task,
        npubench_root=task_root,
    ) == 2
    assert "require --reference-source npubench" in capsys.readouterr().out
    assert calls == []


def test_cannbench_entry_persists_reserved_binding_without_a3_setup(
    tmp_path, monkeypatch, a5_only_env
):
    op_dir = _op_dir(tmp_path, source="arch22").resolve()
    engine, calls = _fake_engine(tmp_path)
    monkeypatch.setattr(commands, "_orch", lambda: engine)
    monkeypatch.setattr(
        commands,
        "record_port_a3_build_source",
        lambda *_args, **_kwargs: pytest.fail("cannbench must not persist an A3 build source"),
    )

    assert _run_port_a3(
        port_a3_dir=op_dir,
        lane=1,
        reference_source="cannbench",
    ) == 0
    assert calls[0][0][0] == op_dir.name
    state = json.loads((engine.WORKSPACE_ROOT / op_dir.name / ".opgen_state.json").read_text())
    assert state["reference"] == {
        "schema_version": 3,
        "source": "cannbench",
        "semantic_binding": "cannbench_reserved",
        "runner_contract_version": "cannbench/unimplemented",
    }


def test_resume_mode_does_not_verify_port_a3_stage_for_npubench(tmp_path, monkeypatch):
    task_root, task, _task_bytes, _sidecar_bytes = _npubench_task(tmp_path)
    workspace = tmp_path / "workspace" / "op"
    from npubench.npubench_inputs import stage_npubench_inputs

    stage = stage_npubench_inputs(
        workspace, npubench_task=task, npubench_root=task_root
    )
    (workspace / ".opgen_state.json").write_text(
        json.dumps({"opgen_mode": "port_a3_to_a5", "reference": stage.state_block()})
    )
    monkeypatch.setattr(
        commands,
        "verify_source_stage",
        lambda *_args, **_kwargs: pytest.fail("npubench resume must not verify A3 source stage"),
    )

    assert getattr(commands, "_workspace_mode")(workspace) == "port_a3_to_a5"


def test_parser_exposes_only_scoped_start_modes(monkeypatch, tmp_path):
    seen = []
    engine = _fake_cli_engine(seen, rc=17)
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    source = tmp_path / "op"
    monkeypatch.setattr(
        sys,
        "argv",
        ["orchestrator", "--port-a3-ops", str(source), "--plan", "--timing"],
    )
    assert cli.main() == 17
    assert seen[0]["port_a3_dir"] == source
    assert seen[0]["timing"] is True


def test_parser_forwards_npubench_flags(monkeypatch, tmp_path):
    seen = []
    engine = _fake_cli_engine(seen)
    task_root, task, _task_bytes, _sidecar_bytes = _npubench_task(tmp_path)
    source = tmp_path / "op"
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "orchestrator", "--port-a3-ops", str(source),
            "--reference-source", "npubench",
            "--npubench-task", str(task),
            "--npubench-root", str(task_root),
        ],
    )

    assert cli.main() == 0
    assert seen[0]["reference_source"] == "npubench"
    assert seen[0]["npubench_task"] == task
    assert seen[0]["npubench_root"] == task_root


def test_parser_forwards_a3_live_reference_source(monkeypatch, tmp_path):
    seen = []
    engine = _fake_cli_engine(seen)
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "orchestrator", "--port-a3-ops", str(tmp_path / "op"),
            "--reference-source", "a3_live",
        ],
    )

    assert cli.main() == 0
    assert seen[0]["reference_source"] == "a3_live"


def test_reference_flags_require_port_a3(monkeypatch, tmp_path):
    engine = SimpleNamespace(_refuse_if_detached=lambda: None)
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "orchestrator", "--backward", str(tmp_path / "forward.py"),
            "--reference-source", "a3_live",
        ],
    )
    assert cli.main() == 2


def test_status_rejects_reference_flags(monkeypatch, capsys):
    engine = SimpleNamespace(
        _refuse_if_detached=lambda: None,
        _cmd_status=lambda: pytest.fail("--status must not ignore reference flags"),
    )
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    monkeypatch.setattr(
        sys,
        "argv",
        ["orchestrator", "--status", "--reference-source", "a3_live"],
    )

    assert cli.main() == 2
    assert "--reference-source" in capsys.readouterr().out


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
        ["--port-a3-ops", "/tmp/op", "--dry-run"],
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
    engine = _fake_cli_engine(calls)
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    monkeypatch.setattr(
        sys,
        "argv",
        ["orchestrator", "--port-a3-ops", str(tmp_path / "op"), "--lane", lane],
    )
    assert cli.main() == 2
    assert calls == []


def test_detected_max_lane_is_forwarded(monkeypatch, tmp_path):
    calls = []
    engine = _fake_cli_engine(calls)
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    monkeypatch.setattr(
        sys,
        "argv",
        ["orchestrator", "--port-a3-ops", str(tmp_path / "op"), "--lane", "2"],
    )
    assert cli.main() == 0
    assert calls[0]["lane"] == 2


def test_positional_op_without_persisted_workspace_is_rejected(monkeypatch, tmp_path, capsys):
    engine = SimpleNamespace(
        _refuse_if_detached=lambda: None,
        _detect_max_lane=lambda: 2,
        _parse_bump_caps=lambda raw: {},
        _resolve_workspace=lambda op, backend="ascendc": tmp_path / op,
        _read_scoped_opgen_mode=lambda workspace: None,
        run_single_op=lambda *a, **k: 0,
    )
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    monkeypatch.setattr(sys, "argv", ["orchestrator", "3_Add", "--lane", "0"])
    assert cli.main() == 2
    assert "lifecycle-only" in capsys.readouterr().out


def test_positional_workspace_resolution_error_is_user_facing(monkeypatch, capsys):
    def fail_resolution(_op, backend="ascendc"):
        raise ValueError("workspace lookup failed")

    engine = SimpleNamespace(
        _refuse_if_detached=lambda: None,
        _detect_max_lane=lambda: 2,
        _resolve_workspace=fail_resolution,
    )
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    monkeypatch.setattr(sys, "argv", ["orchestrator", "3_Add"])

    assert cli.main() == 2
    output = capsys.readouterr().out
    assert "could not resolve positional workspace" in output
    assert "Traceback" not in output


@pytest.mark.parametrize(
    ("identity_field", "identity_value"),
    [
        ("op", "other_op"),
        ("workspace", "other_workspace"),
    ],
)
def test_positional_continuation_rejects_durable_identity_mismatch(
    monkeypatch, tmp_path, capsys, identity_field, identity_value
):
    workspace = tmp_path / "3_Add"
    workspace.mkdir()
    state = {"opgen_mode": "port_a3_to_a5", identity_field: identity_value}
    (workspace / ".opgen_state.json").write_text(json.dumps(state), encoding="utf-8")
    seen = []
    engine = SimpleNamespace(
        _refuse_if_detached=lambda: None,
        _detect_max_lane=lambda: 2,
        _resolve_workspace=lambda op, backend="ascendc": workspace,
        _read_scoped_opgen_mode=lambda ws: "port_a3_to_a5",
        run_single_op=lambda *a, **k: seen.append((a, k)) or 0,
    )
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    monkeypatch.setattr(sys, "argv", ["orchestrator", "3_Add", "--lane", "0"])

    assert cli.main() == 2
    assert seen == []
    assert "identity" in capsys.readouterr().out


def test_positional_op_with_persisted_mode_continues_state_machine(monkeypatch, tmp_path):
    seen = []
    workspace = tmp_path / "3_Add"
    workspace.mkdir()
    engine = SimpleNamespace(
        _refuse_if_detached=lambda: None,
        _detect_max_lane=lambda: 2,
        _parse_bump_caps=lambda raw: {},
        _resolve_workspace=lambda op, backend="ascendc": workspace,
        _read_scoped_opgen_mode=lambda ws: "port_a3_to_a5",
        run_single_op=lambda *a, **k: seen.append((a, k)) or 0,
    )
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    monkeypatch.setattr(sys, "argv", ["orchestrator", "3_Add", "--lane", "0"])
    assert cli.main() == 0
    assert seen[0][0] == ("3_Add",)
    assert seen[0][1]["workspace"] == workspace
    assert seen[0][1]["plan_only"] is False


def test_positional_resume_forwards_bump_cap(monkeypatch, tmp_path):
    seen = []
    workspace = tmp_path / "3_Add"
    workspace.mkdir()
    engine = SimpleNamespace(
        _refuse_if_detached=lambda: None,
        _detect_max_lane=lambda: 2,
        _parse_bump_caps=lambda raw: {"worker": 5} if raw == ["worker:5"] else {},
        _resolve_workspace=lambda op, backend="ascendc": workspace,
        _read_scoped_opgen_mode=lambda ws: "port_a3_to_a5",
        run_single_op=lambda *a, **k: seen.append((a, k)) or 0,
    )
    monkeypatch.setattr(cli, "_orch", lambda: engine)
    monkeypatch.setattr(
        sys,
        "argv",
        ["orchestrator", "3_Add", "--lane", "0", "--bump-cap", "worker:5"],
    )

    assert cli.main() == 0
    assert seen[0][1]["cap_bumps"] == {"worker": 5}
