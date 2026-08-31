# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Hermetic coverage for the minimal NPUKernelBench A5 target route."""
from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import tarfile
from pathlib import Path

import _reorg_paths  # noqa: F401
import pytest

from npubench import npubench_inputs as inputs
from npubench import npubench_runner as runner
from npubench import npubench_target as target
from npubench.npubench_runner import _default_profiler_summary
from npubench.npubench_target import (
    _REPORT,
    _atomic_json,
    _build_contract,
    _candidate_build_error_payload,
    _commit,
    _DirectBuildTimeout,
    _publish,
    _remote,
    _run_direct_build_process,
    _scp,
    _Target,
    _target_command,
    _target_identity,
    _unlink,
)


def _workspace(tmp_path: Path) -> tuple[Path, dict, Path]:
    source = tmp_path / "source"
    source.mkdir()
    task = source / "3_Add.py"
    task.write_text(
        "class Model:\n"
        "    def __init__(self, *args): pass\n"
        "def get_input_groups(): return [[1]]\n"
        "def get_init_inputs(): return []\n",
        encoding="utf-8",
    )
    task.with_suffix(".json").write_text('{"inputs": []}\n', encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    stage = inputs.stage_npubench_inputs(workspace, npubench_task=task)
    state = inputs.bind_npubench_state({"op": "add"}, stage)
    inputs.atomic_write_state(workspace, state)
    (workspace / "model_new_ascendc.py").write_text(
        "class ModelNew:\n    def __init__(self, *args): pass\n", encoding="utf-8"
    )
    return workspace, dict(state["reference"]), runner.materialize_candidate_snapshot(workspace)


def _endpoint(
    *,
    local: bool = True,
    host_mode: bool = False,
    cann_path: str = "/opt/Ascend/cann",
    visible_device: int = 0,
    env: dict | None = None,
) -> _Target:
    return _Target(
        name="A5",
        host="" if local else "a5.example.test",
        user="tester",
        password="",
        container="local" if local else "a5-container",
        cann_path=cann_path,
        benchmark_root="/benchmark-root",
        host_mode=host_mode,
        visible_device=visible_device,
        ssh_options=(),
        env={"A5_SOC_VERSION": "Ascend950PR"} if env is None else env,
    )


def test_tilelang_build_contract_has_separate_receipt_and_error_schema() -> None:
    contract = _build_contract(target.TILELANG2ASCENDC_SOURCE_KIND)
    assert contract["schema"] == target.TILELANG2ASCENDC_BUILD_RECEIPT_SCHEMA
    assert contract["receipt_path"] == target.TILELANG2ASCENDC_BUILD_RECEIPT_PATH

    payload = _candidate_build_error_payload(
        contract,
        target.TILELANG2ASCENDC_SOURCE_KIND,
        "compile failed",
        target={"container": "local"},
        candidate_digest="a" * 64,
    )
    assert payload["build_mode"] == "controlled_authored_cmake"
    assert payload["candidate_independence_gate"] == "PASS"
    assert (
        payload["candidate_independence_schema"]
        == target.TILELANG2ASCENDC_CANDIDATE_INDEPENDENCE_SCHEMA
    )


def test_direct_build_timeout_terminates_its_process_group(tmp_path: Path, monkeypatch) -> None:
    calls: list[object] = []
    killed: list[tuple[int, signal.Signals]] = []

    class FakeProcess:
        pid = 4321

        @staticmethod
        def communicate(timeout=None):
            calls.append(timeout)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(
                    ["bash", "-lc", "build"], timeout, output="partial stdout", stderr="partial stderr"
                )
            return "terminated stdout", "terminated stderr"

    def fake_popen(*_args, **kwargs):
        assert kwargs["start_new_session"] is True
        return FakeProcess()

    monkeypatch.setattr(target.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(target.os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(_DirectBuildTimeout, match="timed out after 7s") as exc_info:
        _run_direct_build_process(
            "build",
            cwd=tmp_path,
            env={},
            timeout_sec=7,
        )

    assert calls == [7, 5]
    assert killed == [(4321, signal.SIGTERM)]
    assert exc_info.value.stdout == "terminated stdout"
    assert exc_info.value.stderr == "terminated stderr"


def test_local_target_identity_persists_outer_runtime_container() -> None:
    endpoint = _endpoint(env={
        "A5_SOC_VERSION": "Ascend950PR",
        "A5_LOCAL_RUNTIME_CONTAINER": "cjm_cann2",
    })

    identity = _target_identity(endpoint)

    assert identity["container"] == "local"
    assert identity["runtime_container"] == "cjm_cann2"
    assert identity["runtime_observation"]["probe"] == "python-process"
    assert identity["runtime_observation"]["proc_1_cgroup_sha256"]


def _report(verb: str, status: str, binding: dict) -> dict:
    return {
        "schema": f"cannbot.npubench.{verb}/v1",
        "status": status,
        "runner_contract_version": runner.RUNNER_CONTRACT_VERSION,
        "run_id": "a" * 32,
        "timestamp": "2026-01-01T00:00:00Z",
        "binding_sha256": binding["binding_sha256"],
        "evaluation_binding": binding,
    }


def _evaluate_reports(workspace: Path, snapshot: Path) -> dict[str, dict]:
    binding = runner.build_evaluation_binding(workspace, snapshot)
    precision = _report("precision", "PASS", binding)
    precision["pass_a"] = {"status": "PASS", "tier1_pass": 1, "total": 1}
    performance = _report("performance", "FAIL", binding)
    performance.update({"profile_archive": None, "profile_tree_sha256": None})
    evaluate = _report("evaluate", "FAIL", binding)
    evaluate.update({"precision": precision, "performance": performance})
    return {"precision": precision, "performance": performance, "evaluate": evaluate}


def _write_reports(workspace: Path, reports: dict[str, dict]) -> None:
    evidence = workspace / runner.EVIDENCE_DIRNAME
    evidence.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        (evidence / _REPORT[name]).write_text(json.dumps(report), encoding="utf-8")


def _result_tar(tmp_path: Path, reports: dict[str, dict]) -> Path:
    root = tmp_path / "target-result"
    _write_reports(root, reports)
    archive = tmp_path / "target-result.tar"
    with tarfile.open(archive, "w") as tar:
        for name in ("precision", "performance", "evaluate"):
            path = root / runner.EVIDENCE_DIRNAME / _REPORT[name]
            tar.add(path, arcname=f"{runner.EVIDENCE_DIRNAME}/{path.name}")
    return archive


def test_local_preflight_delegates_runner_and_writes_target_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, reference, _ = _workspace(tmp_path)
    binding = runner.build_evaluation_binding(workspace)
    report = _report("preflight", "PASS", binding)

    def fake_preflight(_workspace: Path) -> dict:
        _write_reports(workspace, {"preflight": report})
        return report

    monkeypatch.setattr(target, "_target", lambda *_: _endpoint())
    monkeypatch.setattr(runner, "preflight_workspace", fake_preflight)

    result = target.preflight_npubench_on_target(workspace, reference, lane=0)

    assert result["status"] == "PASS"
    assert result["transport"] == "local_target"
    assert result["target_receipt_path"] == target.PREFLIGHT_RECEIPT_PATH


def test_local_evaluate_receipt_is_finalizer_verifiable_and_rejects_multilane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, reference, snapshot = _workspace(tmp_path)
    reports = _evaluate_reports(workspace, snapshot)

    def fake_evaluate(*_args, **kwargs) -> dict:
        assert kwargs["precision_device"] == kwargs["performance_device"] == 0
        _write_reports(workspace, reports)
        return reports["evaluate"]

    monkeypatch.setattr(target, "_target", lambda *_: _endpoint())
    monkeypatch.setattr(runner, "evaluate_workspace", fake_evaluate)
    result = target.evaluate_npubench_on_target(workspace, reference, snapshot, 0, 0, None)
    final_evidence = {
        "target_execution_receipt": result["target_receipt_path"],
        "target_execution_receipt_sha256": result["target_receipt_sha256"],
    }
    assert target.validate_target_evidence_receipt(workspace, reference, final_evidence, reports)[0]

    rejected = target.evaluate_npubench_on_target(workspace, reference, snapshot, 0, 1, None)
    assert rejected["status"] == "ERROR"
    assert "one precision/performance lane" in rejected["reason"]


def test_stage_and_fake_ssh_protocol_keep_native_bundle_and_token_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npubench.npubench_target import _stage

    workspace, reference, snapshot = _workspace(tmp_path)
    archive = _stage(workspace, reference, snapshot, None)
    try:
        with tarfile.open(archive, "r") as tar:
            names = set(tar.getnames())
    finally:
        archive.unlink()
    bundle = reference["bundle_sha256"]
    assert f"reference_inputs/npubench/{bundle}/3_Add.py" in names
    assert f"reference_inputs/npubench/{bundle}/3_Add.json" in names
    assert f"{runner.SNAPSHOT_DIRNAME}/{snapshot.name}/model_new_ascendc.py" in names
    assert "ops/ops-profiling/scripts/msprof_perf_summary.py" in names
    assert "ops/ops-profiling/scripts/msprof_profile_run.sh" not in names
    assert "model.py" not in names and "test.py" not in names

    endpoint = _endpoint(local=False, host_mode=True)
    commands: list[list[str]] = []

    def fake_run(command, what, timeout=None):
        del what, timeout
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(target, "_run", fake_run)
    remote_root = "/benchmark-root/npubench_target/token"
    from npubench.npubench_target import _pack, _prepare

    _prepare(endpoint, remote_root, "/tmp/stage.tar", "token")
    _pack(endpoint, remote_root, "/tmp/result.tar", "token", "evaluate")
    assert _scp("/tmp/stage.tar", "tester@a5.example.test:/tmp/stage.tar", endpoint)[0] == "scp"
    remote_commands = [command[-1] for command in commands]
    assert all("current_task" not in command for command in remote_commands)
    assert all(remote_root in command for command in remote_commands)
    assert any("evaluate_report.json" in command for command in remote_commands)


def test_target_stage_uses_packaged_profiler_payload(tmp_path: Path) -> None:
    from npubench.npubench_target import _stage

    workspace, reference, snapshot = _workspace(tmp_path)
    archive = _stage(workspace, reference, snapshot, None)
    try:
        with tarfile.open(archive, "r") as tar:
            staged = tar.extractfile("ops/ops-profiling/scripts/msprof_perf_summary.py")
            assert staged is not None
            assert staged.read() == _default_profiler_summary().read_bytes()
    finally:
        archive.unlink()


def _legacy_host_command_text(monkeypatch: pytest.MonkeyPatch) -> str:
    endpoint = _endpoint(local=False, host_mode=True)
    monkeypatch.setitem(endpoint.env, "A5_HOST_PYTHON", "/opt/npu-python/bin/python3")
    command = _target_command(
        endpoint,
        "/benchmark-root/npubench_target/token",
        ["evaluate", "--workspace", "."],
    )
    return command[-1]


def test_legacy_target_host_command_keeps_full_runtime_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _legacy_host_command_text(monkeypatch)
    assert "if [ -f /opt/Ascend/cann/set_env.sh ]; then source /opt/Ascend/cann/set_env.sh || true; fi" in text
    assert "export LD_LIBRARY_PATH=" in text
    assert "export PYTHONPATH=" in text
    assert "export PATH=/opt/npu-python/bin:$PATH" in text
    assert "export ASCEND_RT_VISIBLE_DEVICES=0" in text
    assert "cd /benchmark-root/npubench_target/token" in text
    assert "npubench_runner.py evaluate --workspace ." in text


def test_legacy_target_host_command_keeps_workspace_and_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _legacy_host_command_text(monkeypatch)
    assert "LD_LIBRARY_PATH=" in text
    assert "PYTHONPATH=" in text
    assert "PATH=/opt/npu-python/bin:$PATH" in text
    assert "ASCEND_RT_VISIBLE_DEVICES=" in text
    assert "cd /benchmark-root/npubench_target/token" in text
    assert "npubench_runner.py evaluate --workspace ." in text


def test_remote_fails_closed_when_cleanup_fails(tmp_path: Path, monkeypatch) -> None:
    workspace, reference, snapshot = _workspace(tmp_path)
    stage = tmp_path / "stage.tar"
    stage.write_bytes(b"stage")
    result = tmp_path / "result.tar"
    result.write_bytes(b"result")
    endpoint = _endpoint(local=False)

    monkeypatch.setattr(target, "_stage", lambda *_args, **_kwargs: stage)
    monkeypatch.setattr(
        target,
        "_run",
        lambda command, what, timeout=None: subprocess.CompletedProcess(command, 0, "", ""),
    )
    monkeypatch.setattr(target, "_prepare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(target, "_invoke", lambda *_args, **_kwargs: {"status": "PASS"})
    monkeypatch.setattr(target, "_pack", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(target, "_download", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(target, "_import", lambda *_args, **_kwargs: {"status": "PASS"})
    monkeypatch.setattr(
        target,
        "_cleanup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            target.TargetTransportError("target cleanup failed")
        ),
    )

    with pytest.raises(target.TargetTransportError, match="target cleanup failed"):
        _remote(workspace, reference, endpoint, "preflight", 0, snapshot, None)


def test_remote_cleanup_failure_does_not_mask_primary_error(tmp_path: Path, monkeypatch) -> None:
    workspace, reference, snapshot = _workspace(tmp_path)
    endpoint = _endpoint(local=False)

    def fail_stage(*_args, **_kwargs):
        raise target.TargetTransportError("primary transport failure")

    monkeypatch.setattr(target, "_stage", fail_stage)
    monkeypatch.setattr(
        target,
        "_cleanup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            target.TargetTransportError("cleanup transport failure")
        ),
    )

    with pytest.raises(target.TargetTransportError, match="primary transport failure") as exc_info:
        _remote(workspace, reference, endpoint, "preflight", 0, snapshot, None)

    assert "cleanup transport failure" in " ".join(getattr(exc_info.value, "__notes__", ()))


def test_required_unlink_failure_is_visible(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "material.tar"
    path.write_bytes(b"material")

    def fail_unlink(_self, *args, **kwargs):
        del args, kwargs
        raise PermissionError("material is still in use")

    monkeypatch.setattr(target.Path, "unlink", fail_unlink)

    with pytest.raises(target.CleanupFailure, match="required cleanup failed.*material is still in use"):
        _unlink(path, required=True)


def test_optional_unlink_returns_auditable_failure_without_raising(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "optional.cache"

    def fail_unlink(_self, *args, **kwargs):
        del args, kwargs
        raise OSError("optional cache is busy")

    monkeypatch.setattr(target.Path, "unlink", fail_unlink)

    failure = _unlink(path)

    assert isinstance(failure, target.CleanupFailure)
    assert "optional cache is busy" in str(failure)


def test_optional_unlink_succeeds_normally(tmp_path: Path) -> None:
    path = tmp_path / "optional.cache"
    path.write_bytes(b"cache")

    assert _unlink(path) is None
    assert not path.exists()


def test_atomic_json_fails_closed_when_temporary_cleanup_fails(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "state.json"
    real_unlink = Path.unlink

    def fail_temporary_unlink(path: Path, *args, **kwargs):
        if path.name.startswith(f".{destination.name}."):
            raise OSError("atomic temporary is busy")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(target.Path, "unlink", fail_temporary_unlink)

    with pytest.raises(target.CleanupFailure, match="required cleanup failed.*atomic temporary is busy"):
        _atomic_json(destination, {"status": "PASS"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "PASS"}


def test_atomic_json_preserves_primary_failure_when_cleanup_also_fails(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "state.json"

    def fail_replace(*_args, **_kwargs):
        raise OSError("atomic publish failed")

    def fail_unlink(_self, *args, **kwargs):
        del args, kwargs
        raise OSError("atomic temporary cleanup failed")

    monkeypatch.setattr(target.os, "replace", fail_replace)
    monkeypatch.setattr(target.Path, "unlink", fail_unlink)

    with pytest.raises(OSError, match="atomic publish failed") as exc_info:
        _atomic_json(destination, {"status": "PASS"})

    assert "atomic temporary cleanup failed" in " ".join(
        getattr(exc_info.value, "__notes__", ())
    )


def test_import_fails_closed_when_staging_cleanup_fails(tmp_path: Path, monkeypatch) -> None:
    from npubench.npubench_target import _import

    workspace, _reference, snapshot = _workspace(tmp_path)
    reports = _evaluate_reports(workspace, snapshot)
    archive = _result_tar(tmp_path, reports)

    def fail_remove_tree(*_args, **_kwargs):
        raise OSError("import staging is busy")

    monkeypatch.setattr(target, "_remove_tree", fail_remove_tree)

    with pytest.raises(target.CleanupFailure, match="required cleanup failed.*import staging is busy"):
        _import(workspace, archive, "evaluate", reports["evaluate"], snapshot)


def test_fixed_result_import_receipt_binds_report_bytes(tmp_path: Path) -> None:
    from npubench.npubench_target import _import

    workspace, reference, snapshot = _workspace(tmp_path)
    reports = _evaluate_reports(workspace, snapshot)
    imported = _import(
        workspace, _result_tar(tmp_path, reports), "evaluate", reports["evaluate"], snapshot
    )
    result = _publish(
        workspace, reference, _endpoint(local=False), "evaluate", imported, "ssh_target", 0
    )
    evidence = {
        "target_execution_receipt": result["target_receipt_path"],
        "target_execution_receipt_sha256": result["target_receipt_sha256"],
    }
    assert target.validate_target_evidence_receipt(workspace, reference, evidence, imported)[0]

    (workspace / runner.EVIDENCE_DIRNAME / runner.PRECISION_REPORT_FILENAME).write_text("{}", encoding="utf-8")
    valid, reason = target.validate_target_evidence_receipt(workspace, reference, evidence, imported)
    assert not valid
    assert "precision report digest differs" in reason


def test_result_import_refuses_profile_parent_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    staged = tmp_path / "staged"
    profile = staged / runner.EVIDENCE_DIRNAME / "profiles" / "run-1"
    profile.mkdir(parents=True)
    (profile / "raw.csv").write_text("profile", encoding="utf-8")
    evidence = workspace / runner.EVIDENCE_DIRNAME
    evidence.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (evidence / "profiles").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(target.TargetTransportError, match="profile evidence directory"):
        _commit(
            workspace,
            staged,
            {"performance": {"profile_archive": "npubench_evidence/profiles/run-1"}},
            "evaluate",
        )

    assert not (outside / "run-1").exists()
