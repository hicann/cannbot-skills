# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for build_candidate.build() — Phase 3 of aog-prior-art-verify.

All tests use dependency-injected run_remote/push/pull mocks so no real
SSH is needed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from build_candidate import (  # noqa: E402
    build, write_build_report, BuildReport,
    _construct_build_command, _construct_remote_build_layout,
    _default_pull_files, _default_push_dir, _load_candidate_binding,
)
from stage_candidate import compute_candidate_digest  # noqa: E402


def _make_candidate(workspace: Path, op: str = "op_a",
                    repo_name: str = "ops-nn") -> Path:
    candidate = workspace / ".prior_art_candidate"
    (candidate / "op_kernel" / "arch35").mkdir(parents=True)
    source = candidate / "op_kernel" / "arch35" / f"{op}.h"
    source.write_text("// stub")
    import hashlib
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    files = [{
        "rel_path": f"op_kernel/arch35/{op}.h",
        "build_rel_path": f"norm/{op}/op_kernel/arch35/{op}.h",
        "sha": source_sha,
        "source_type": "upstream_arch35",
        "origin": f"/source/{op}.h",
    }]
    manifest = {
        "schema_version": 2,
        "op": op,
        "candidate_dir": str(candidate),
        "repo_name": repo_name,
        "op_repo_rel_path": f"norm/{op}",
        "sources_staged": ["upstream_arch35"],
        "file_count": 1,
        "files": files,
        "candidate_digest": compute_candidate_digest(op, repo_name, files),
        "warnings": [],
        "errors": [],
    }
    (candidate / "manifest.json").write_text(json.dumps(manifest))
    return candidate


def _fake_pull_creates_artifacts(workspace: Path, op: str,
                                 candidate_digest: str):
    """Build a pull mock that creates a fake .so + binary.json in build_out."""
    def _pull(user, host, container, container_artifacts, host_artifacts, local):
        local.mkdir(parents=True, exist_ok=True)
        (local / f"{op}_kernels.so").write_text("ELF mock")
        (local / f"{op}_binary.json").write_text('{"bin_filename": "x"}')
        (local / "candidate_digest.txt").write_text(candidate_digest + "\n")
        return None
    return _pull


def _digest(candidate: Path) -> str:
    return json.loads((candidate / "manifest.json").read_text())["candidate_digest"]


def _fake_pull_only_marker(candidate_digest: str):
    def _pull(*args):
        local = args[-1]
        local.mkdir(parents=True, exist_ok=True)
        (local / "candidate_digest.txt").write_text(candidate_digest + "\n")
        return None
    return _pull


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_build_success_pulls_so_and_json(tmp_path: Path) -> None:
    workspace = tmp_path / "ws" / "op_a"
    workspace.mkdir(parents=True)
    candidate = _make_candidate(workspace)
    rep = build("op_a", workspace, ops_nn_path="/cann/ops-nn",
                a5_host="h", a5_user="u", a5_container="c",
                run_remote=lambda cmd: (0, "build ok", ""),
                push_dir=lambda *_args: None,
                pull_files=_fake_pull_creates_artifacts(
                    workspace, "op_a", _digest(candidate)
                ))
    assert rep.verdict == "SUCCESS"
    assert rep.so_path is not None and rep.so_path.exists()
    assert rep.binary_json_path is not None
    assert rep.duration_s >= 0
    assert rep.errors == []


def test_build_command_includes_bound_overlay_and_freshness(
        tmp_path: Path) -> None:
    candidate = _make_candidate(tmp_path, op="group_norm_silu_quant")
    binding, errors = _load_candidate_binding(
        candidate, "group_norm_silu_quant", expected_repo_name="ops-nn"
    )
    assert binding is not None, errors
    layout = _construct_remote_build_layout(
        "group_norm_silu_quant", binding.candidate_digest, "a" * 12
    )
    cmd = _construct_build_command(
        "group_norm_silu_quant", "/data/cann_b103/cann-9.0.0/ops-nn",
        binding, layout,
    )
    assert "--ops=group_norm_silu_quant" in cmd
    assert "--soc=ascend950" in cmd
    assert "build.sh" in cmd
    assert "build.log" in cmd  # log piping
    assert "set -euo pipefail" in cmd
    assert "rm -rf --" in cmd and "/repo/build_out" in cmd
    assert "-newer" in cmd
    assert f"{layout['candidate']}/op_kernel/arch35" in cmd
    assert f"{layout['repo']}/norm/group_norm_silu_quant" in cmd


def test_remote_build_layout_paths() -> None:
    layout = _construct_remote_build_layout("my_op", "b" * 64, "c" * 12)
    assert "my_op" in layout["container_root"]
    assert layout["candidate"].startswith("/tmp/aog_prior_art_build/")
    assert layout["host_root"].startswith("/tmp/aog_prior_art_transfer/")
    assert layout["repo"] != "/cann/ops-nn"


def test_default_transport_explicitly_crosses_container_boundary(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def _run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("build_candidate.subprocess.run", _run)
    local_candidate = tmp_path / "candidate"
    local_candidate.mkdir()
    push_error = _default_push_dir(
        local_candidate, "user", "host", "container",
        "/tmp/aog_prior_art_transfer/op-id", "/tmp/aog_prior_art_build/op-id/candidate",
    )
    pull_error = _default_pull_files(
        "user", "host", "container", "/tmp/aog_prior_art_build/op-id/artifacts",
        "/tmp/aog_prior_art_transfer/op-id/artifacts", tmp_path / "pulled",
    )

    assert push_error is None and pull_error is None
    remote_commands = [
        command[-1] for command in calls
        if command and command[0] == "ssh"
    ]
    assert any("docker cp" in command
               and "container:/tmp/aog_prior_art_build/op-id/candidate/" in command
               for command in remote_commands)
    assert any("docker cp container:/tmp/aog_prior_art_build/op-id/artifacts/." in command
               for command in remote_commands)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------
def test_build_no_candidate_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "ws" / "op_a"
    workspace.mkdir(parents=True)
    # NO candidate dir created
    rep = build("op_a", workspace, ops_nn_path="/x",
                a5_host="h", run_remote=lambda c: (0, "", ""))
    assert rep.verdict == "NO_CANDIDATE"
    assert any("stage_candidate" in e for e in rep.errors)


def test_build_scp_push_failed(tmp_path: Path) -> None:
    workspace = tmp_path / "ws" / "op_a"
    workspace.mkdir(parents=True)
    _make_candidate(workspace, repo_name="x")
    rep = build("op_a", workspace, ops_nn_path="/x",
                a5_host="h", run_remote=lambda c: (0, "", ""),
                push_dir=lambda *a, **kw: "scp permission denied")
    assert rep.verdict == "SCP_PUSH_FAILED"
    assert "permission denied" in rep.errors[0]


def test_build_returns_nonzero(tmp_path: Path) -> None:
    workspace = tmp_path / "ws" / "op_a"
    workspace.mkdir(parents=True)
    _make_candidate(workspace, repo_name="x")
    rep = build("op_a", workspace, ops_nn_path="/x",
                a5_host="h",
                run_remote=lambda c: (2, "compile log", "fatal: missing header"),
                push_dir=lambda *a, **kw: None)
    assert rep.verdict == "BUILD_FAILED"
    assert "rc=2" in rep.errors[0]
    assert "fatal" in rep.build_log


def test_build_timeout(tmp_path: Path) -> None:
    workspace = tmp_path / "ws" / "op_a"
    workspace.mkdir(parents=True)
    _make_candidate(workspace, repo_name="x")
    rep = build("op_a", workspace, ops_nn_path="/x",
                a5_host="h",
                run_remote=lambda c: (124, "", "TimeoutExpired"),
                push_dir=lambda *a, **kw: None)
    assert rep.verdict == "TIMEOUT"


def test_build_scp_pull_both_fail(tmp_path: Path) -> None:
    workspace = tmp_path / "ws" / "op_a"
    workspace.mkdir(parents=True)
    _make_candidate(workspace, repo_name="x")
    rep = build("op_a", workspace, ops_nn_path="/x",
                a5_host="h",
                run_remote=lambda c: (0, "ok", ""),
                push_dir=lambda *a, **kw: None,
                pull_files=lambda *a, **kw: "no such file")
    assert rep.verdict == "SCP_PULL_FAILED"


def test_build_success_but_no_so(tmp_path: Path) -> None:
    """rc=0 + pull returns rc=0 but no .so file found → BUILD_FAILED.
    Guards against silent half-success where build.sh logs SUCCESS but the
    artifact set is empty."""
    workspace = tmp_path / "ws" / "op_a"
    workspace.mkdir(parents=True)
    candidate = _make_candidate(workspace, repo_name="x")
    rep = build("op_a", workspace, ops_nn_path="/x",
                a5_host="h",
                run_remote=lambda c: (0, "ok", ""),
                push_dir=lambda *a, **kw: None,
                pull_files=_fake_pull_only_marker(_digest(candidate)))
    assert rep.verdict == "BUILD_FAILED"
    assert any("no .so produced" in e for e in rep.errors)


def test_build_rejects_staged_file_changed_after_manifest(tmp_path: Path) -> None:
    workspace = tmp_path / "ws" / "op_a"
    workspace.mkdir(parents=True)
    candidate = _make_candidate(workspace, repo_name="x")
    (candidate / "op_kernel" / "arch35" / "op_a.h").write_text("tampered")
    invoked = False

    def _run(_command):
        nonlocal invoked
        invoked = True
        return 0, "", ""

    rep = build(
        "op_a", workspace, ops_nn_path="/x", a5_host="h",
        run_remote=_run, push_dir=lambda *args: None,
    )

    assert rep.verdict == "CANDIDATE_INVALID"
    assert invoked is False
    assert any("changed" in error for error in rep.errors)


def test_build_cannot_reuse_stale_local_artifact(tmp_path: Path) -> None:
    workspace = tmp_path / "ws" / "op_a"
    workspace.mkdir(parents=True)
    candidate = _make_candidate(workspace, repo_name="x")
    stale = candidate / "build" / "op_a_kernels.so"
    stale.parent.mkdir()
    stale.write_text("stale library")

    rep = build(
        "op_a", workspace, ops_nn_path="/x", a5_host="h",
        run_remote=lambda command: (0, "ok", ""),
        push_dir=lambda *args: None,
        pull_files=_fake_pull_only_marker(_digest(candidate)),
    )

    assert rep.verdict == "BUILD_FAILED"
    assert not stale.exists()
    assert any("no .so produced" in error for error in rep.errors)


def test_build_rejects_artifact_digest_marker_from_other_candidate(
        tmp_path: Path) -> None:
    workspace = tmp_path / "ws" / "op_a"
    workspace.mkdir(parents=True)
    _make_candidate(workspace, repo_name="x")

    rep = build(
        "op_a", workspace, ops_nn_path="/x", a5_host="h",
        run_remote=lambda command: (0, "ok", ""),
        push_dir=lambda *args: None,
        pull_files=_fake_pull_creates_artifacts(
            workspace, "op_a", "f" * 64
        ),
    )

    assert rep.verdict == "BUILD_FAILED"
    assert any("not bound" in error for error in rep.errors)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_write_build_report_records_verdict(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    rep = BuildReport(op="op_a", verdict="SUCCESS", duration_s=42.5,
                      so_path=tmp_path / "x.so", build_log="ok")
    out = write_build_report(rep, workspace)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["op"] == "op_a"
    assert data["verdict"] == "SUCCESS"
    assert data["duration_s"] == 42.5
    assert data["schema_version"] == 2


def test_write_build_report_truncates_long_log(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    rep = BuildReport(op="op_a", verdict="SUCCESS", build_log="x" * 5000)
    out = write_build_report(rep, workspace)
    data = json.loads(out.read_text())
    assert len(data["build_log_tail"]) <= 2000


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
