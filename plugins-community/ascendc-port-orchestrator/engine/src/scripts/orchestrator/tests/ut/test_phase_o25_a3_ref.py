# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""W4 (2026-05-12, ROADMAP §1.5) — phase_o25_a3_ref unit tests.

Pure-helper tests (no I/O):
- derive_aclnn_entry: finds the canonical path; returns None when absent
- derive_input_gen_source: finds gen_data.py; returns None when absent
- derive_op_dependencies: parses CMakeLists.txt DEPENDENCIES lines correctly
- parse_a3_perf_log: extracts per-case timings from both supported formats

I/O-bound tests (with injected run_remote mock):
- probe_a3_reachable: success / failure cases
- provision_a3_reference: orchestrates pure helpers + probe; writes
  a3_reference_runnable.json with correct verdict for each branch

Integration-shaped tests (real op-dir fixtures):
- ctc_loss_v3 case: has aclnn entry in examples/, no gen_data.py, depends on ctc_loss_v2
- gather_elements_v2 case: has aclnn entry + gen_data.py, no peer deps
"""
from __future__ import annotations

import json
import pickle
import shlex
import sys
import types
from pathlib import Path
from textwrap import dedent

import pytest

try:  # collection-time torch import (DEBT-47 conftest tears down per-test NEW modules; the
    import torch  # noqa: F401   provision_native_capture tests + existing edge_inputs tests use torch)
except ImportError:  # pragma: no cover
    torch = None

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o25_a3_ref as p25a3  # noqa: E402
import a3_ref_provision as _a3prov  # noqa: E402  (post-split: run_a3_reference + scp defaults live here)
import a3_ref_npu as _a3npu  # noqa: E402  (post-split: NPU-smi + idle-lane gating live here)
from briefs import _common as _bc  # noqa: E402  (DEBT-101: canonical .ascendc_env patch point)
from source_arch import stage_source_tree  # noqa: E402


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_derive_aclnn_entry_examples_path(tmp_path):
    """Canonical: examples/test_aclnn_<op>.cpp."""
    op_dir = tmp_path / "ctc_loss_v3"
    (op_dir / "examples").mkdir(parents=True)
    entry = op_dir / "examples" / "test_aclnn_ctc_loss_v3.cpp"
    entry.write_text("// fake")
    assert p25a3.derive_aclnn_entry(op_dir) == entry


def test_derive_aclnn_entry_pytorch_asc(tmp_path):
    """Alternative: pytorch/<op>.asc."""
    op_dir = tmp_path / "top_k_top_p_sample_v2"
    (op_dir / "pytorch").mkdir(parents=True)
    entry = op_dir / "pytorch" / "top_k_top_p_sample_v2.asc"
    entry.write_text("// fake")
    assert p25a3.derive_aclnn_entry(op_dir) == entry


def test_derive_aclnn_entry_tests_ut(tmp_path):
    """Fallback: tests/ut/op_host/test_aclnn_<op>.cpp."""
    op_dir = tmp_path / "weird_op"
    (op_dir / "tests" / "ut" / "op_host").mkdir(parents=True)
    entry = op_dir / "tests" / "ut" / "op_host" / "test_aclnn_weird_op.cpp"
    entry.write_text("// fake")
    assert p25a3.derive_aclnn_entry(op_dir) == entry


def test_derive_aclnn_entry_none_when_missing(tmp_path):
    """Returns None when no aclnn entry exists."""
    op_dir = tmp_path / "no_aclnn_op"
    op_dir.mkdir()
    (op_dir / "op_host").mkdir()
    (op_dir / "op_kernel").mkdir()
    assert p25a3.derive_aclnn_entry(op_dir) is None


def test_derive_aclnn_entry_prefers_examples_over_tests(tmp_path):
    """When both examples/ and tests/ut/op_host/ have aclnn entry, examples wins."""
    op_dir = tmp_path / "ctc_loss_v3"
    (op_dir / "examples").mkdir(parents=True)
    (op_dir / "tests" / "ut" / "op_host").mkdir(parents=True)
    examples_entry = op_dir / "examples" / "test_aclnn_ctc_loss_v3.cpp"
    tests_entry = op_dir / "tests" / "ut" / "op_host" / "test_aclnn_ctc_loss_v3.cpp"
    examples_entry.write_text("// canonical")
    tests_entry.write_text("// fallback")
    assert p25a3.derive_aclnn_entry(op_dir) == examples_entry


def test_staged_snapshot_uses_manifest_op_name_for_derivation(tmp_path):
    """The fixed .source_arch22 basename must not replace the original op name."""
    op_dir = tmp_path / "heaviside"
    (op_dir / "op_kernel" / "arch22").mkdir(parents=True)
    (op_dir / "op_kernel" / "arch22" / "heaviside.h").write_text(
        "class Heaviside { void Process() {} };\n"
    )
    (op_dir / "examples").mkdir()
    entry = op_dir / "examples" / "test_aclnn_heaviside.cpp"
    entry.write_text("// fake")
    data_dir = op_dir / "tests" / "ut" / "op_kernel" / "heaviside_data"
    data_dir.mkdir(parents=True)
    gen_data = data_dir / "gen_data.py"
    gen_data.write_text("# fake")
    (op_dir / "op_host").mkdir()
    (op_dir / "op_host" / "heaviside_def.cpp").write_text(
        'this->Input("x").ParamType(REQUIRED).DataType({DT_FLOAT});\n'
    )
    workspace = tmp_path / "workspace" / "heaviside"

    stage = stage_source_tree(op_dir, workspace)

    assert p25a3.derive_aclnn_entry(stage.root) == (
        stage.root / "examples" / "test_aclnn_heaviside.cpp"
    )
    assert p25a3.derive_input_gen_source(stage.root) == (
        stage.root / "tests" / "ut" / "op_kernel" / "heaviside_data" / "gen_data.py"
    )
    signature = p25a3.parse_op_def_signature(stage.root)
    assert signature is not None
    assert signature["op_name"] == "heaviside"


def test_derive_input_gen_source_present(tmp_path):
    """gen_data.py at tests/ut/op_kernel/<op>_data/."""
    op_dir = tmp_path / "gather_elements_v2"
    data_dir = op_dir / "tests" / "ut" / "op_kernel" / "gather_elements_v2_data"
    data_dir.mkdir(parents=True)
    gen = data_dir / "gen_data.py"
    gen.write_text("# gen")
    assert p25a3.derive_input_gen_source(op_dir) == gen


def test_derive_input_gen_source_absent(tmp_path):
    """Returns None when gen_data.py missing (e.g. ctc_loss_v3 in PR4778)."""
    op_dir = tmp_path / "ctc_loss_v3"
    op_dir.mkdir()
    assert p25a3.derive_input_gen_source(op_dir) is None


def test_derive_op_dependencies_single(tmp_path):
    """Parse a single DEPENDENCIES line (the ctc_loss_v3 case)."""
    op_dir = tmp_path / "ctc_loss_v3"
    (op_dir / "op_host").mkdir(parents=True)
    cmake = op_dir / "op_host" / "CMakeLists.txt"
    cmake.write_text(dedent("""\
        # op_host CMakeLists for ctc_loss_v3
        set(OP_NAME ctc_loss_v3)
        add_op_kernel(
            ${OP_NAME}
            DEPENDENCIES ctc_loss_v2
            SOURCES ${SRC_FILES}
        )
    """))
    assert p25a3.derive_op_dependencies(op_dir) == ["ctc_loss_v2"]


def test_derive_op_dependencies_multiple(tmp_path):
    """Parse multiple DEPENDENCIES tokens on one line."""
    op_dir = tmp_path / "fused_op"
    (op_dir / "op_host").mkdir(parents=True)
    cmake = op_dir / "op_host" / "CMakeLists.txt"
    cmake.write_text(
        "add_op_kernel(my_op DEPENDENCIES peer_a peer_b peer_c SOURCES x.cpp)\n"
    )
    assert p25a3.derive_op_dependencies(op_dir) == ["peer_a", "peer_b", "peer_c"]


def test_derive_op_dependencies_none(tmp_path):
    """Returns [] when no DEPENDENCIES line present."""
    op_dir = tmp_path / "standalone_op"
    (op_dir / "op_host").mkdir(parents=True)
    cmake = op_dir / "op_host" / "CMakeLists.txt"
    cmake.write_text("set(OP_NAME standalone_op)\n")
    assert p25a3.derive_op_dependencies(op_dir) == []


def test_derive_op_dependencies_ignores_comments(tmp_path):
    """Commented-out DEPENDENCIES lines are ignored."""
    op_dir = tmp_path / "op"
    (op_dir / "op_host").mkdir(parents=True)
    cmake = op_dir / "op_host" / "CMakeLists.txt"
    cmake.write_text(dedent("""\
        # DEPENDENCIES old_peer  # legacy comment
        DEPENDENCIES real_peer
    """))
    assert p25a3.derive_op_dependencies(op_dir) == ["real_peer"]


def test_derive_op_dependencies_missing_cmake_returns_empty(tmp_path):
    """No CMakeLists.txt → empty list, not error."""
    op_dir = tmp_path / "broken_op"
    op_dir.mkdir()
    assert p25a3.derive_op_dependencies(op_dir) == []


def test_parse_a3_perf_log_elapsed_ms_format():
    """Format 1: case_<id> elapsed_ms=<float>."""
    log = dedent("""\
        running aclnn UT...
        case_0 elapsed_ms=1.234
        case_1 elapsed_ms=2.567
        done
    """)
    times = p25a3.parse_a3_perf_log(log)
    assert times == {"0": 1.234, "1": 2.567}


def test_parse_a3_perf_log_bracketed_format():
    """Format 2: [case <id>] device_time=<float>ms."""
    log = dedent("""\
        [case 0] device_time=1.5ms
        [case fp16-large] device_time=12.34ms
    """)
    times = p25a3.parse_a3_perf_log(log)
    assert times == {"0": 1.5, "fp16-large": 12.34}


def test_parse_a3_perf_log_empty():
    """No matching lines → empty dict."""
    log = "no timings here\nsome other output"
    assert p25a3.parse_a3_perf_log(log) == {}


# ---------------------------------------------------------------------------
# I/O via mocked run_remote
# ---------------------------------------------------------------------------
def make_canned_runner(scripted: list[tuple[int, str, str]]):
    """Returns a run_remote that pops responses from the scripted list."""
    responses = list(scripted)

    def run(cmd: str) -> tuple[int, str, str]:
        if not responses:
            return 1, "", "no more scripted responses"
        return responses.pop(0)
    return run


def test_probe_a3_reachable_success():
    """A3 probe with bisheng present → reachable."""
    runner = make_canned_runner([
        (0, "PROBE_OK\n/home/.../bin/bisheng\n", ""),
    ])
    ok, msg = p25a3.probe_a3_reachable("root", "1.2.3.4", "fake-c", run_remote=runner)
    assert ok is True
    assert "reachable" in msg.lower()


def test_probe_a3_reachable_ssh_failure():
    """A3 probe with non-zero rc → unreachable."""
    runner = make_canned_runner([
        (255, "", "ssh: connect to host 1.2.3.4 port 22: Connection refused"),
    ])
    ok, msg = p25a3.probe_a3_reachable("root", "1.2.3.4", "fake-c", run_remote=runner)
    assert ok is False
    assert "rc=255" in msg or "Connection refused" in msg


def test_probe_a3_reachable_no_probe_ok_marker():
    """If the runner returns 0 but no PROBE_OK marker, treat as unreachable."""
    runner = make_canned_runner([(0, "weird output", "")])
    ok, msg = p25a3.probe_a3_reachable("root", "h", "c", run_remote=runner)
    assert ok is False


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------
def make_ctc_loss_v3_op_dir(tmp_path):
    """Synthesize a ctc_loss_v3-shaped ops-nn op dir for testing."""
    op_dir = tmp_path / "loss" / "ctc_loss_v3"
    (op_dir / "op_host").mkdir(parents=True)
    (op_dir / "op_kernel" / "arch22").mkdir(parents=True)
    (op_dir / "op_kernel" / "arch22" / "ctc_loss_v3.h").write_text(
        "class CtcLossV3 { void Process() {} };\n"
    )
    (op_dir / "examples").mkdir()
    # aclnn entry
    (op_dir / "examples" / "test_aclnn_ctc_loss_v3.cpp").write_text("// fake")
    # CMakeLists with v2 dependency
    (op_dir / "op_host" / "CMakeLists.txt").write_text(
        "add_op_kernel(ctc_loss_v3 DEPENDENCIES ctc_loss_v2 SOURCES x.cpp)\n"
    )
    # NOTE: no gen_data.py (real ctc_loss_v3 from PR4778 also lacks it)
    return op_dir


def bind_source_stage(op_dir: Path, workspace: Path) -> None:
    """Bind a real source-only arch22 snapshot to migration durable state."""
    stage = stage_source_tree(op_dir, workspace)
    state = {
        "schema_version": 2,
        "opgen_mode": "port_a3_to_a5",
        "source_arch": "arch22",
        "target_arch": "arch35",
        "port_a3_source": str(stage.root),
        "graybox_arch22_dir": str(stage.root),
        "graybox_sandbox": True,
        "source_stage_manifest": str(stage.manifest),
        "source_stage_digest": stage.digest,
    }
    (workspace / ".opgen_state.json").write_text(
        json.dumps(state, indent=2) + "\n"
    )


def write_complete_live_capture(workspace: Path, *, value: float = 1.0) -> None:
    """Write one genuine torch-readable case as a simulated live runner result."""
    inputs = [torch.tensor([value], dtype=torch.float32)] if torch else [value]
    outputs = (
        [torch.tensor([value + 1.0], dtype=torch.float32)]
        if torch
        else [value + 1.0]
    )
    payloads = {
        workspace / "edge_inputs.pt": {"inputs": inputs},
        workspace / "edge_dataset.pt": {
            "inputs": inputs,
            "a3_outputs": outputs,
        },
    }
    for path, payload in payloads.items():
        if torch:
            torch.save(payload, path)
        else:
            with path.open("wb") as stream:
                pickle.dump(payload, stream)
    (workspace / "a3_baseline_perf.json").write_text(
        json.dumps({"median_ms_per_case": {"case_0": 1.234}}) + "\n"
    )


def install_capture_torch_stub(monkeypatch) -> None:
    """Provide only the torch.load seam used by capture gates when torch is absent."""
    if torch is not None:
        return
    fake_torch = types.ModuleType("torch")

    def _load(path, *, weights_only=False):
        del weights_only
        with Path(path).open("rb") as stream:
            return pickle.load(stream)

    fake_torch.load = _load
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def load_capture_dataset(workspace: Path):
    if torch:
        return torch.load(workspace / "edge_dataset.pt", weights_only=False)
    with (workspace / "edge_dataset.pt").open("rb") as stream:
        return pickle.load(stream)


def test_probe_only_is_rejected_as_migration_truth(tmp_path):
    """A reachability probe can never substitute for a fresh live arch22 capture."""
    op_dir = make_ctc_loss_v3_op_dir(tmp_path)
    workspace = tmp_path / "workspace" / "ctc_loss_v3"
    workspace.mkdir(parents=True)
    (workspace / "edge_dataset.pt").write_bytes(b"stale")
    (workspace / "a3_reference_runnable.json").write_text(
        '{"verdict":"READY"}'
    )

    runner = make_canned_runner([])
    rep = p25a3.provision_a3_reference(
        op_dir=op_dir,
        workspace=workspace,
        a3_host="1.2.3.4",
        a3_user="root",
        a3_container="npu-a3",
        run_remote=runner,
        probe_only=True,
    )
    assert rep.verdict == "LIVE_CAPTURE_REQUIRED"
    assert rep.a3_exec_attempted is False

    # JSON artifact written
    art = workspace / "a3_reference_runnable.json"
    assert art.is_file()
    payload = json.loads(art.read_text())
    assert payload["verdict"] == "LIVE_CAPTURE_REQUIRED"
    assert not (workspace / "edge_dataset.pt").exists()
    assert list(workspace.glob("edge_dataset.pt.stale-pre-capture-*"))
    assert list(workspace.glob("a3_reference_runnable.json.stale-pre-capture-*"))


def test_provision_a3_reference_missing_entry(tmp_path):
    """Op dir without aclnn entry → MISSING_ENTRY verdict."""
    op_dir = tmp_path / "no_aclnn_op"
    (op_dir / "op_host").mkdir(parents=True)
    (op_dir / "op_kernel" / "arch22").mkdir(parents=True)
    (op_dir / "op_kernel" / "arch22" / "no_aclnn_op.h").write_text(
        "class NoAclnnOp { void Process() {} };\n"
    )
    workspace = tmp_path / "workspace" / "no_aclnn_op"
    runner = make_canned_runner([])  # should never be called

    rep = p25a3.provision_a3_reference(
        op_dir=op_dir,
        workspace=workspace,
        a3_host="1.2.3.4",
        run_remote=runner,
        probe_only=False,
    )
    assert rep.verdict == "MISSING_ENTRY"
    assert "no aclnn entry" in " ".join(rep.errors)
    # Artifact still written so caller can inspect
    art = workspace / "a3_reference_runnable.json"
    assert art.is_file()
    payload = json.loads(art.read_text())
    assert payload["verdict"] == "MISSING_ENTRY"


def test_stale_capture_quarantined_before_missing_entry(tmp_path):
    op_dir = tmp_path / "no_aclnn_op"
    (op_dir / "op_host").mkdir(parents=True)
    (op_dir / "op_kernel" / "arch22").mkdir(parents=True)
    (op_dir / "op_kernel" / "arch22" / "no_aclnn_op.h").write_text(
        "class NoAclnnOp { void Process() {} };\n"
    )
    workspace = tmp_path / "workspace" / "no_aclnn_op"
    workspace.mkdir(parents=True)
    for name in (
        "edge_dataset.pt",
        "a3_baseline_perf.json",
        "a3_capture_manifest.json",
    ):
        (workspace / name).write_bytes(b"stale")

    rep = p25a3.provision_a3_reference(
        op_dir=op_dir,
        workspace=workspace,
        a3_host="1.2.3.4",
        run_remote=make_canned_runner([]),
        probe_only=False,
    )

    assert rep.verdict == "MISSING_ENTRY"
    for name in (
        "edge_dataset.pt",
        "a3_baseline_perf.json",
        "a3_capture_manifest.json",
    ):
        assert not (workspace / name).exists()
        assert list(workspace.glob(f"{name}.stale-pre-capture-*"))


def test_provision_a3_reference_a3_unreachable(tmp_path):
    """A3 SSH fails → A3_UNREACHABLE verdict, useful recommendation."""
    op_dir = make_ctc_loss_v3_op_dir(tmp_path)
    workspace = tmp_path / "workspace" / "ctc_loss_v3"
    runner = make_canned_runner([
        (255, "", "ssh: connect to host 1.2.3.4 port 22: No route to host"),
    ])
    rep = p25a3.provision_a3_reference(
        op_dir=op_dir,
        workspace=workspace,
        a3_host="1.2.3.4",
        run_remote=runner,
        probe_only=False,
    )
    assert rep.verdict == "A3_UNREACHABLE"
    assert any("ssh" in r.lower() or "docker ps" in r.lower() for r in rep.recommendations)


def test_provision_a3_reference_probe_only_false_build_failure(tmp_path, monkeypatch):
    """P128 (2026-05-17): build is no longer the first step — runner runs first.
    When the runner fails (with or without unshipped-op signature in error),
    verdict is EXEC_FAILED. The build path only triggers when run_a3_reference
    returns an error matching unshipped-op signatures; even then, run_a3_reference
    internally makes multiple run_remote calls (input_gen, push, exec, pull) and
    the signature check is against the aggregate exec_msg. This test asserts the
    happy fallback path: runner fails → verdict reflects the failure mode.

    Hermetic guard (2026-06-03): the probe_only=False path reaches run_a3_reference
    which pushes/pulls the workspace via the module-level _scp_push_dir / _scp_pull_files
    helpers — these are NOT covered by the run_remote injection (provision_a3_reference
    does not forward push_dir/pull_files). Without mocking them, the test fired a REAL
    scp to a3_host="1.2.3.4", which hangs forever on networks that don't fast-reject the
    unreachable host (e.g. WSL2+proxy) — stalling the whole pre-commit unit suite for 15+
    min. Mock them to no-op success so the flow reaches the canned exec-failure as intended.
    Proper fix (DEBT): thread push_dir/pull_files through provision_a3_reference so the
    seam is injectable like run_remote.
    """
    # Post-split: run_a3_reference resolves its scp defaults from a3_ref_provision's
    # own namespace, so patch them there (not on the facade).
    monkeypatch.setattr(_a3prov, "_scp_push_dir", lambda *a, **k: None)
    monkeypatch.setattr(_a3prov, "_scp_pull_files", lambda *a, **k: None)
    op_dir = make_ctc_loss_v3_op_dir(tmp_path)
    workspace = tmp_path / "workspace" / "ctc_loss_v3"
    workspace.mkdir(parents=True)
    (workspace / "run_a3_reference.py").write_text("# fake runner")
    runner = make_canned_runner([
        (0, "PROBE_OK\n/usr/bin/bisheng\n", ""),
        (1, "cmake Error: missing dependency\n", "build.sh: line 42: Error"),
    ])
    rep = p25a3.provision_a3_reference(
        op_dir=op_dir,
        workspace=workspace,
        a3_host="1.2.3.4",
        run_remote=runner,
        probe_only=False,
        npu_busy_check=_noop_busy_check,
    )
    # Runner failed; error doesn't match unshipped-op signature → EXEC_FAILED.
    assert rep.verdict == "EXEC_FAILED"
    assert rep.a3_exec_attempted is True


def test_provision_a3_reference_probe_only_false_runner_missing(tmp_path):
    """P128: runner-missing path is now hit before build (runner is step 4b).
    No build called because we fail before trying any execution.
    """
    op_dir = make_ctc_loss_v3_op_dir(tmp_path)
    workspace = tmp_path / "workspace" / "ctc_loss_v3"
    workspace.mkdir(parents=True)
    # NOTE: no run_a3_reference.py in workspace; aog-a3-author won't run
    # because tmp_path op_dir doesn't have the right shape for it.
    runner = make_canned_runner([
        (0, "PROBE_OK\n/usr/bin/bisheng\n", ""),
    ])
    rep = p25a3.provision_a3_reference(
        op_dir=op_dir,
        workspace=workspace,
        a3_host="1.2.3.4",
        run_remote=runner,
        probe_only=False,
        npu_busy_check=_noop_busy_check,
    )
    assert rep.verdict == "RUNNER_MISSING"
    assert rep.a3_exec_attempted is False
    assert "run_a3_reference.py" in " ".join(rep.errors)


def test_provision_a3_reference_probe_only_false_full_success(tmp_path, monkeypatch):
    """P128: runner succeeds via torch_npu dispatch (shipped aclnn op), no
    build invoked. Expected output files present + content captured → READY.

    task#25: a successful capture now requires populated content, not just file
    presence. A real successful run_a3_reference.py emits a3_capture_manifest.json
    {n_total,n_captured} (Tier-1) + a3_baseline_perf.json with a non-empty
    median_ms_per_case; the validator gates READY on those.
    """
    op_dir = make_ctc_loss_v3_op_dir(tmp_path)
    workspace = tmp_path / "workspace" / "ctc_loss_v3"
    workspace.mkdir(parents=True)
    install_capture_torch_stub(monkeypatch)
    bind_source_stage(op_dir, workspace)
    # Author a fake runner script
    (workspace / "run_a3_reference.py").write_text("# fake runner")

    def _live_capture(*args, **kwargs):
        write_complete_live_capture(workspace)
        return True, "all cases captured on source NPU"

    monkeypatch.setattr(p25a3, "run_a3_reference", _live_capture)
    # Probe is the only remote call; the injected live runner writes fresh files.
    runner = make_canned_runner([(0, "PROBE_OK\n/usr/bin/bisheng\n", "")])
    rep = p25a3.provision_a3_reference(
        op_dir=op_dir,
        workspace=workspace,
        a3_host="1.2.3.4",
        run_remote=runner,
        probe_only=False,
        npu_busy_check=_noop_busy_check,
    )
    assert rep.verdict == "READY"
    assert rep.a3_exec_attempted is True
    assert rep.a3_outputs_path is not None
    assert rep.a3_perf_path is not None
    assert rep.capture_id
    manifest = json.loads((workspace / "a3_capture_manifest.json").read_text())
    assert manifest["schema"] == "a3_capture/v2"
    assert manifest["live_exec"] is True
    assert manifest["source_arch"] == "arch22"


@pytest.mark.parametrize(
    "failure",
    [
        "aclnnHeavisideGetWorkspaceSize failed. ERROR: 561103",
        "runtime reports no binary for Heaviside",
        "kernel-not-registered: Heaviside",
    ],
)
def test_unshipped_runner_failure_uses_run_scoped_custom_opp(
    tmp_path, monkeypatch, failure
):
    op_dir = make_ctc_loss_v3_op_dir(tmp_path)
    workspace = tmp_path / "workspace" / "ctc_loss_v3"
    workspace.mkdir(parents=True)
    install_capture_torch_stub(monkeypatch)
    bind_source_stage(op_dir, workspace)
    (workspace / "run_a3_reference.py").write_text("# fake runner")
    run_calls: list[dict] = []
    build_calls: list[tuple[tuple, dict]] = []

    def fake_run(*args, **kwargs):
        run_calls.append(kwargs)
        if len(run_calls) == 1:
            return False, f"runner exited rc=2: {failure}"
        write_complete_live_capture(workspace)
        return True, "fresh live A3 capture"

    def fake_build(*args, **kwargs):
        build_calls.append((args, kwargs))
        return True, "source package built and installed"

    monkeypatch.setattr(p25a3, "run_a3_reference", fake_run)
    monkeypatch.setattr(p25a3, "build_a3_reference_kernel", fake_build)
    monkeypatch.setattr(p25a3, "_a3_container_home", lambda: "/home/test")

    rep = p25a3.provision_a3_reference(
        op_dir=workspace / ".source_arch22",
        source_build_op_dir=op_dir,
        workspace=workspace,
        a3_host="1.2.3.4",
        run_remote=make_canned_runner(
            [(0, "PROBE_OK\n/usr/bin/bisheng\n", "")]
        ),
        probe_only=False,
        npu_busy_check=_noop_busy_check,
    )

    assert rep.verdict == "READY"
    assert len(run_calls) == 2
    assert run_calls[0].get("custom_opp_env_script") is None
    assert len(build_calls) == 1
    build_args, build_kwargs = build_calls[0]
    assert Path(build_args[0]) == op_dir
    assert ".source_arch22" not in str(build_args[0])
    install_path = build_kwargs["custom_opp_install_path"]
    assert install_path.startswith(
        "/home/test/workspace/a5_ops_a3_to_a5/custom_opp/ctc_loss_v3/"
    )
    assert "/usr/local/Ascend/opp" not in install_path
    assert run_calls[1]["custom_opp_env_script"] == (
        f"{install_path}/vendors/custom_nn/bin/set_env.bash"
    )


def test_stale_fallback_marker_quarantines_cached_capture(tmp_path, monkeypatch):
    """A stale synthetic-truth marker makes every old capture artifact ineligible."""
    op_dir = make_ctc_loss_v3_op_dir(tmp_path)
    workspace = tmp_path / "workspace" / "ctc_loss_v3"
    workspace.mkdir(parents=True)
    bind_source_stage(op_dir, workspace)
    (workspace / "run_a3_reference.py").write_text("# fake runner")
    (workspace / "edge_dataset.pt").write_bytes(b"old")
    (workspace / "a3_baseline_perf.json").write_text(
        '{"median_ms_per_case": {"case_0": 1.0}}'
    )
    (workspace / "a3_capture_manifest.json").write_text(
        '{"n_total": 1, "n_captured": 1}'
    )
    (workspace / ".truth_source_override").write_text("stale\n")
    monkeypatch.setattr(p25a3, "run_a3_reference", lambda *a, **k: (True, "no rewrite"))

    rep = p25a3.provision_a3_reference(
        op_dir=op_dir,
        workspace=workspace,
        a3_host="1.2.3.4",
        run_remote=make_canned_runner([(0, "PROBE_OK\n/usr/bin/bisheng\n", "")]),
        probe_only=False,
        npu_busy_check=_noop_busy_check,
    )

    assert rep.verdict == "CAPTURE_INCOMPLETE"
    assert not (workspace / "edge_dataset.pt").exists()
    assert not (workspace / "a3_baseline_perf.json").exists()
    assert not (workspace / "a3_capture_manifest.json").exists()
    assert list(workspace.glob("*.stale-pre-capture-*"))
    assert (workspace / ".truth_source_override").exists()


def test_fresh_live_capture_replaces_stale_cache_and_clears_marker(
    tmp_path, monkeypatch
):
    op_dir = make_ctc_loss_v3_op_dir(tmp_path)
    workspace = tmp_path / "workspace" / "ctc_loss_v3"
    workspace.mkdir(parents=True)
    install_capture_torch_stub(monkeypatch)
    bind_source_stage(op_dir, workspace)
    (workspace / "run_a3_reference.py").write_text("# fake runner")
    (workspace / "edge_dataset.pt").write_bytes(b"old")
    (workspace / "a3_baseline_perf.json").write_text("{}")
    (workspace / "a3_capture_manifest.json").write_text(
        '{"n_total": 0, "n_captured": 0}'
    )
    (workspace / ".a3_ref_unavailable_cpu_truth_deferred").write_text("stale\n")

    def _fresh_capture(*args, **kwargs):
        write_complete_live_capture(workspace, value=7.0)
        return True, "fresh live capture"

    monkeypatch.setattr(p25a3, "run_a3_reference", _fresh_capture)
    rep = p25a3.provision_a3_reference(
        op_dir=op_dir,
        workspace=workspace,
        a3_host="1.2.3.4",
        run_remote=make_canned_runner([(0, "PROBE_OK\n/usr/bin/bisheng\n", "")]),
        probe_only=False,
        npu_busy_check=_noop_busy_check,
    )

    assert rep.verdict == "READY"
    assert not (workspace / ".a3_ref_unavailable_cpu_truth_deferred").exists()
    captured = load_capture_dataset(workspace)
    captured_value = captured["a3_outputs"][0]
    assert float(captured_value.item() if torch else captured_value) == 8.0
    assert list(workspace.glob("*.stale-pre-capture-*"))


def test_build_a3_reference_kernel_success(tmp_path):
    """W14b: build_a3_reference_kernel returns (True, log) on rc=0 + no Error."""
    op_dir = tmp_path / "loss" / "ctc_loss_v3"
    op_dir.mkdir(parents=True)
    runner = make_canned_runner([(0, "[100%] Built target ctc_loss_v3_kernels\n", "")])
    ok, log = p25a3.build_a3_reference_kernel(
        op_dir, "/home/x/cann/cann-9.0.0", run_remote=runner,
    )
    assert ok is True
    assert "Built target" in log


def test_build_a3_reference_kernel_rc_nonzero(tmp_path):
    """W14b: rc≠0 → build failed."""
    op_dir = tmp_path / "loss" / "ctc_loss_v3"
    op_dir.mkdir(parents=True)
    runner = make_canned_runner([(1, "cmake fatal: missing dep", "")])
    ok, _ = p25a3.build_a3_reference_kernel(
        op_dir, "/home/x/cann/cann-9.0.0", run_remote=runner,
    )
    assert ok is False


def test_build_a3_reference_kernel_quotes_missing_root_diagnostic(tmp_path):
    malicious_root = tmp_path / "root$(printf PWNED > injected-marker)"
    op_dir = malicious_root / "loss" / "ctc_loss_v3"
    op_dir.mkdir(parents=True)
    commands = []

    def runner(command):
        commands.append(command)
        return (
            2,
            "",
            f"ORCHESTRATOR: build.sh not at {malicious_root}/build.sh",
        )

    ok, log = p25a3.build_a3_reference_kernel(
        op_dir, "/home/x/cann/cann-9.0.0", run_remote=runner,
    )

    assert ok is False
    assert len(commands) == 1
    assert shlex.quote(str(malicious_root)) in commands[0]
    assert "root$(printf PWNED > injected-marker)" in log
    assert not (tmp_path / "injected-marker").exists()


def test_build_a3_reference_kernel_warning_as_error(tmp_path):
    """W14b: rc=0 but log contains 'Error' → build failed (defensive)."""
    op_dir = tmp_path / "loss" / "ctc_loss_v3"
    op_dir.mkdir(parents=True)
    runner = make_canned_runner([
        (0, "Warning: foo\nError: bar\n[100%] Built\n", ""),
    ])
    ok, _ = p25a3.build_a3_reference_kernel(
        op_dir, "/home/x/cann/cann-9.0.0", run_remote=runner,
    )
    assert ok is False


def test_build_a3_reference_kernel_uses_explicit_logical_snapshot_name(tmp_path):
    op_dir = tmp_path / "workspace" / "heaviside" / ".source_arch22"
    op_dir.mkdir(parents=True)
    commands = []

    def runner(command):
        commands.append(command)
        return 0, "[100%] Built target heaviside_kernels\n", ""

    ok, _ = p25a3.build_a3_reference_kernel(
        op_dir,
        "/home/x/cann/cann-9.0.0",
        run_remote=runner,
        logical_op_name="heaviside",
    )

    assert ok is True
    assert "--ops=heaviside" in commands[0]
    assert "--ops=.source_arch22" not in commands[0]


def test_build_a3_reference_kernel_installs_only_to_isolated_path(tmp_path):
    op_dir = tmp_path / "activation" / "heaviside"
    op_dir.mkdir(parents=True)
    commands: list[str] = []

    def runner(command):
        commands.append(command)
        return 0, "[100%] Built target heaviside_kernels\nSUCCESS\n", ""

    isolated = "/home/test/workspace/a5_ops_a3_to_a5/custom_opp/heaviside/run-id"
    ok, _ = p25a3.build_a3_reference_kernel(
        op_dir,
        "/opt/cann-9.0.0",
        run_remote=runner,
        logical_op_name="heaviside",
        custom_opp_install_path=isolated,
    )

    assert ok is True
    assert len(commands) == 1
    command = commands[0]
    assert "set -eo pipefail" in command
    assert "cann-ops-nn-custom-linux.$(uname -m).run" in command
    assert f"--install-path={isolated}" in command
    assert f"{isolated}/vendors/custom_nn/bin/set_env.bash" in command
    assert "/usr/local/Ascend/opp" not in command


def _noop_push(local_dir, a3_user, a3_host, remote_dir):
    """No-op scp push stub for unit tests (avoids real ssh)."""
    return None


def _noop_pull(a3_user, a3_host, remote_dir, filenames, local_dir):
    """No-op scp pull stub for unit tests (avoids real ssh).
    Test fixtures pre-create the expected files locally — pull is a no-op."""
    return None


def _noop_busy_check(a3_user, a3_host, npu_id, threshold_pct, *, run_remote=None):
    """No-op busy gate for unit tests (avoids real ssh+npu-smi 15s timeout)."""
    return False, 0.0, "test stub: npu-smi skipped"


def test_run_a3_reference_success(tmp_path):
    """W14b: run_a3_reference returns (True, _) when runner produces both outputs."""
    op_dir = tmp_path / "loss" / "ctc_loss_v3"
    op_dir.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner_script = workspace / "run_a3_reference.py"
    runner_script.write_text("# runner")
    # Pre-create the expected output files (simulating runner success)
    (workspace / "edge_dataset.pt").write_bytes(b"x")
    (workspace / "a3_baseline_perf.json").write_text("{}")

    runner = make_canned_runner([(0, "all cases passed", "")])
    ok, msg = p25a3.run_a3_reference(
        op_dir, workspace, "root", "1.2.3.4", "npu-a3",
        "/home/x/cann/cann-9.0.0",
        runner_script=runner_script, run_remote=runner,
        push_dir=_noop_push, pull_files=_noop_pull,
    )
    assert ok is True
    assert "A3 ref captured" in msg


def test_run_a3_reference_runner_failed(tmp_path):
    """W14b: runner exits rc≠0 → failure."""
    op_dir = tmp_path / "loss" / "ctc_loss_v3"
    op_dir.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner_script = workspace / "run_a3_reference.py"
    runner_script.write_text("# runner")
    runner = make_canned_runner([(1, "", "Traceback (most recent call last)")])
    ok, msg = p25a3.run_a3_reference(
        op_dir, workspace, "root", "1.2.3.4", "npu-a3",
        "/home/x/cann/cann-9.0.0",
        runner_script=runner_script, run_remote=runner,
        push_dir=_noop_push, pull_files=_noop_pull,
    )
    assert ok is False
    assert "rc=1" in msg


def test_run_a3_reference_preserves_bounded_dispatch_root_cause(tmp_path):
    op_dir = tmp_path / "activation" / "heaviside"
    op_dir.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner_script = workspace / "run_a3_reference.py"
    runner_script.write_text("# runner")
    commands: list[str] = []
    secret = "sk-" + "x" * 32
    noisy_tail = "\n".join(f"case {index}: 0/3" for index in range(200))

    def runner(command):
        commands.append(command)
        return (
            2,
            "aclnnHeavisideGetWorkspaceSize failed. ERROR: 561103\n"
            f"ANTHROPIC_API_KEY={secret}\n{noisy_tail}\n",
            "",
        )

    ok, msg = p25a3.run_a3_reference(
        op_dir,
        workspace,
        "root",
        "1.2.3.4",
        "npu-a3",
        "/opt/cann-9.0.0",
        runner_script=runner_script,
        run_remote=runner,
        push_dir=_noop_push,
        pull_files=_noop_pull,
    )

    assert ok is False
    assert "561103" in msg
    assert secret not in msg
    assert "API_KEY=<redacted>" not in msg  # non-diagnostic secret line is omitted
    assert len(msg) < 4500
    assert "grep -E -i -m 8" in commands[0]
    assert ".a3_reference_runner.log" in commands[0]
    assert 'exit "$runner_rc"' in commands[0]


def test_run_a3_reference_sources_only_explicit_custom_opp_on_retry(tmp_path):
    op_dir = tmp_path / "activation" / "heaviside"
    op_dir.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner_script = workspace / "run_a3_reference.py"
    runner_script.write_text("# runner")
    (workspace / "edge_dataset.pt").write_bytes(b"x")
    (workspace / "a3_baseline_perf.json").write_text("{}")
    commands: list[str] = []

    def runner(command):
        commands.append(command)
        return 0, "captured", ""

    env_script = (
        "/home/test/workspace/a5_ops_a3_to_a5/custom_opp/"
        "heaviside/run-id/vendors/custom_nn/bin/set_env.bash"
    )
    ok, _ = p25a3.run_a3_reference(
        op_dir,
        workspace,
        "root",
        "1.2.3.4",
        "npu-a3",
        "/opt/cann-9.0.0",
        runner_script=runner_script,
        run_remote=runner,
        push_dir=_noop_push,
        pull_files=_noop_pull,
        custom_opp_env_script=env_script,
    )

    assert ok is True
    assert len(commands) == 1
    assert f"source {env_script}" in commands[0]
    assert "/usr/local/Ascend/opp/vendors" not in commands[0]


def test_run_a3_reference_quotes_configurable_remote_paths(
    tmp_path, monkeypatch
):
    op_dir = tmp_path / "activation" / "heaviside"
    op_dir.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner_script = workspace / "run_a3_reference.py"
    runner_script.write_text("# runner")
    (workspace / "build_runner.sh").write_text("#!/usr/bin/env bash\n")
    (workspace / "edge_inputs.pt").write_bytes(b"x")
    (workspace / "edge_dataset.pt").write_bytes(b"x")
    (workspace / "a3_baseline_perf.json").write_text("{}")
    commands: list[str] = []
    malicious_home = "/root/a3 space;$(touch /tmp/container-injected)"
    malicious_cann = "/opt/cann space;$(touch /tmp/cann-injected)"
    monkeypatch.setattr(_a3prov, "_a3_container_home", lambda: malicious_home)

    def runner(command):
        commands.append(command)
        return 0, "ok", ""

    ok, _ = p25a3.run_a3_reference(
        op_dir,
        workspace,
        "root",
        "1.2.3.4",
        "npu-a3",
        malicious_cann,
        runner_script=runner_script,
        run_remote=runner,
        push_dir=_noop_push,
        pull_files=_noop_pull,
    )

    assert ok is True
    assert len(commands) == 2
    container_workspace = (
        f"{malicious_home}/workspace/a5_ops_a3_to_a5/workspace/{workspace.name}"
    )
    for command in commands:
        assert f"cd {shlex.quote(container_workspace)} &&" in command
        assert f"source {shlex.quote(malicious_cann)}/set_env.sh" in command
        assert f"cd {container_workspace} &&" not in command
        assert f"source {malicious_cann}/set_env.sh" not in command


def test_run_a3_reference_rejects_multiline_cann_path_before_transport(tmp_path):
    op_dir = tmp_path / "activation" / "heaviside"
    op_dir.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner_script = workspace / "run_a3_reference.py"
    runner_script.write_text("# runner")
    called = []

    ok, msg = p25a3.run_a3_reference(
        op_dir,
        workspace,
        "root",
        "1.2.3.4",
        "npu-a3",
        "/opt/cann\n$(touch /tmp/cann-injected)",
        runner_script=runner_script,
        run_remote=lambda command: called.append(command) or (0, "", ""),
        push_dir=lambda *args: called.append("push") or None,
        pull_files=_noop_pull,
    )

    assert ok is False
    assert "absolute single-line" in msg
    assert called == []


def test_run_a3_reference_missing_outputs(tmp_path):
    """W14b: runner exits 0 but doesn't write edge_dataset.pt → failure."""
    op_dir = tmp_path / "loss" / "ctc_loss_v3"
    op_dir.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner_script = workspace / "run_a3_reference.py"
    runner_script.write_text("# runner")
    # NOTE: not creating edge_dataset.pt
    runner = make_canned_runner([(0, "completed", "")])
    ok, msg = p25a3.run_a3_reference(
        op_dir, workspace, "root", "1.2.3.4", "npu-a3",
        "/home/x/cann/cann-9.0.0",
        runner_script=runner_script, run_remote=runner,
        push_dir=_noop_push, pull_files=_noop_pull,
    )
    assert ok is False
    assert "edge_dataset.pt not produced" in msg


def test_run_a3_reference_push_failure_propagates(tmp_path):
    """2026-05-13 fix: scp push failure surfaces as run_a3_reference error
    (must NOT silently fall through to runner exec which would then fail
    on missing files inside container).
    """
    op_dir = tmp_path / "loss" / "ctc_loss_v3"
    op_dir.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner_script = workspace / "run_a3_reference.py"
    runner_script.write_text("# runner")
    runner = make_canned_runner([(0, "should not be called", "")])
    # Push errors → run_a3_reference returns False immediately

    def failing_push(*args):
        return "scp exited 1: Permission denied"
    ok, msg = p25a3.run_a3_reference(
        op_dir, workspace, "root", "1.2.3.4", "npu-a3",
        "/home/x/cann/cann-9.0.0",
        runner_script=runner_script, run_remote=runner,
        push_dir=failing_push, pull_files=_noop_pull,
    )
    assert ok is False
    assert "scp push" in msg
    assert "Permission denied" in msg


def test_run_a3_reference_pipefail_propagates_exit_code(tmp_path):
    """Regression (GDN chunk_gated_delta_rule, 2026-06-18): the remote build +
    runner commands MUST use `set -o pipefail`. Both end with `... | tail -N`;
    without pipefail the pipeline rc is tail's (0), masking a real build/run
    failure. Symptom: chunk_gated_delta_rule build_runner.sh failed (missing
    aclnn header — op not in A3 CANN 8.5.0) but reported rc=0, so the orchestrator
    thought the build succeeded, the runner found no binary + captured 0 cases,
    and the failure surfaced two steps downstream as a misleading
    CAPTURE_INCOMPLETE — also bypassing the P128 build-from-source fallback that
    keys off the build/runner error message.
    """
    op_dir = tmp_path / "attention" / "chunk_gated_delta_rule"
    op_dir.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner_script = workspace / "run_a3_reference.py"
    runner_script.write_text("# runner")
    # build_runner.sh present → triggers the Path-B build_cmd branch.
    (workspace / "build_runner.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (workspace / "edge_inputs.pt").write_bytes(b"x")
    (workspace / "edge_dataset.pt").write_bytes(b"x")
    (workspace / "a3_baseline_perf.json").write_text("{}")

    seen_cmds: list[str] = []

    def recording_runner(cmd: str) -> tuple[int, str, str]:
        seen_cmds.append(cmd)
        return 0, "ok", ""

    p25a3.run_a3_reference(
        op_dir, workspace, "root", "1.2.3.4", "npu-a3",
        "/home/x/cann/cann-8.5.0",
        runner_script=runner_script, run_remote=recording_runner,
        push_dir=_noop_push, pull_files=_noop_pull,
    )
    build_cmds = [c for c in seen_cmds if "build_runner.sh" in c]
    runner_cmds = [c for c in seen_cmds if runner_script.name in c and "python3" in c]
    assert build_cmds, f"build command not issued; seen={seen_cmds}"
    assert runner_cmds, f"runner command not issued; seen={seen_cmds}"
    for c in build_cmds:
        assert "tail -" in c, f"build cmd should pipe to tail: {c}"
        assert "pipefail" in c, f"build cmd must set pipefail to unmask rc: {c}"
    for c in runner_cmds:
        assert "tail -" in c, f"runner cmd should pipe to tail: {c}"
        assert "pipefail" in c, f"runner cmd must set pipefail to unmask rc: {c}"


# ---------------------------------------------------------------------------
# Format block message
# ---------------------------------------------------------------------------
def test_format_block_message_with_peer_deps(tmp_path):
    """Block message surfaces peer-op dependency note."""
    op_dir = make_ctc_loss_v3_op_dir(tmp_path)
    workspace = tmp_path / "ws"
    runner = make_canned_runner([
        (255, "", "ssh failed"),
    ])
    rep = p25a3.provision_a3_reference(
        op_dir=op_dir, workspace=workspace,
        a3_host="x", run_remote=runner, probe_only=False,
    )
    msg = p25a3.format_block_message("ctc_loss_v3", rep)
    assert "A3_UNREACHABLE" in msg
    assert "ctc_loss_v2" in msg  # peer dep surfaced
    assert "Cross-op router edits" in msg  # links to KB W9


# ---------------------------------------------------------------------------
# NPU-busy gate (2026-05-13 — perf-fairness safeguard)
# ---------------------------------------------------------------------------
_NPU_SMI_SAMPLE_IDLE = """\
+------------------------------------------------------------------------------------------------+
| npu-smi 25.3.rc1                 Version: 25.3.rc1                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip  Phy-ID              | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     Ascend910           | OK            | 197.5       47                0    / 0             |
| 0     0                   | 0000:9D:00.0  | 1           0    / 0          43482/ 65536         |
| 0     Ascend910           | OK            | -           49                0    / 0             |
| 1     1                   | 0000:9F:00.0  | 2           0    / 0          42021/ 65536         |
"""

_NPU_SMI_SAMPLE_BUSY = """\
+---------------------------+---------------+----------------------------------------------------+
| Chip  Phy-ID              | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     0                   | 0000:9D:00.0  | 87          0    / 0          43482/ 65536         |
| 1     1                   | 0000:9F:00.0  | 3           0    / 0          42021/ 65536         |
"""


def test_parse_aicore_pct_idle():
    """Idle NPU 0 → returns 1.0."""
    assert p25a3.parse_aicore_pct(_NPU_SMI_SAMPLE_IDLE, npu_id=0) == 1.0


def test_parse_aicore_pct_busy():
    """Busy NPU 0 → returns 87.0."""
    assert p25a3.parse_aicore_pct(_NPU_SMI_SAMPLE_BUSY, npu_id=0) == 87.0


def test_parse_aicore_pct_chip_1_idle_when_chip_0_busy():
    """When parsing for chip 1, we get 3.0 (chip 1's load), not 87.0 (chip 0's load)."""
    assert p25a3.parse_aicore_pct(_NPU_SMI_SAMPLE_BUSY, npu_id=1) == 3.0


def test_parse_aicore_pct_unknown_chip_returns_zero():
    """Chip id not present → returns 0.0 (caller treats as "skip gate")."""
    assert p25a3.parse_aicore_pct(_NPU_SMI_SAMPLE_BUSY, npu_id=99) == 0.0


def test_read_a3_npu_gate_config_defaults_when_no_env(tmp_path, monkeypatch):
    """No .ascendc_env → npu_id=0, threshold=20. DEBT-101: patch the canonical
    DEFAULT_ASCENDC_ENV to a non-existent path so the reader is isolated from any
    real .ascendc_env on the running agent's checkout.
    """
    monkeypatch.setattr(_bc, "DEFAULT_ASCENDC_ENV", tmp_path / "absent.ascendc_env")
    workspace = tmp_path / "workspace" / "op"
    workspace.mkdir(parents=True)
    npu_id, threshold = getattr(p25a3, "_read_a3_npu_gate_config")(workspace)
    assert (npu_id, threshold) == (0, 20)


def test_read_a3_npu_gate_config_reads_env(tmp_path, monkeypatch):
    """A3_DEFAULT_NPU_ID + A3_AICORE_BUSY_THRESHOLD picked up from .ascendc_env.
    DEBT-101: route through the canonical DEFAULT_ASCENDC_ENV patch point.
    """
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text("A3_DEFAULT_NPU_ID=3\nA3_AICORE_BUSY_THRESHOLD=50\n")
    monkeypatch.setattr(_bc, "DEFAULT_ASCENDC_ENV", env_file)
    workspace = tmp_path / "workspace" / "op"
    workspace.mkdir(parents=True)
    npu_id, threshold = getattr(p25a3, "_read_a3_npu_gate_config")(workspace)
    assert (npu_id, threshold) == (3, 50)


def test_provision_a3_reference_busy_gate_blocks(tmp_path):
    """When chip 0 is busy (AICore=87%), provision returns verdict=A3_BUSY."""
    op_dir = make_ctc_loss_v3_op_dir(tmp_path)
    workspace = tmp_path / "workspace" / "ctc_loss_v3"
    workspace.mkdir(parents=True)
    (workspace / "run_a3_reference.py").write_text("# fake runner")
    runner = make_canned_runner([(0, "PROBE_OK\n/usr/bin/bisheng\n", "")])

    def busy_check(*args, **kwargs):
        return True, 87.0, _NPU_SMI_SAMPLE_BUSY

    rep = p25a3.provision_a3_reference(
        op_dir=op_dir, workspace=workspace, a3_host="x",
        run_remote=runner, probe_only=False,
        npu_busy_check=busy_check,
    )
    assert rep.verdict == "A3_BUSY"
    # Post-Task#50 (2026-05-13): error format now includes the candidate
    # set + observed AICore pcts (e.g. "all candidates [0] above
    # threshold=20%. observed={0: 87.0}"). The legacy "AICore=87%" prose
    # was replaced by the structured observed dict.
    errs = " ".join(rep.errors)
    assert "threshold=20%" in errs
    assert "87" in errs  # observed pct appears in the dict repr
    # Helpful recovery hint surfaced (now points at A3_NPU_RANGE for
    # auto-pick, plus the threshold knob)
    rec_text = " ".join(rep.recommendations)
    assert "A3_NPU_RANGE" in rec_text or "A3_AICORE_BUSY_THRESHOLD" in rec_text


def test_provision_a3_reference_busy_gate_passes_when_idle(tmp_path):
    """When chip 0 AICore=1%, gate passes — flow proceeds.

    P128: under new runner-first order, without a workspace runner script
    the flow fails at RUNNER_MISSING (since aog-a3-author can't author for
    tmp_path op_dir). Verifies the BUSY gate didn't short-circuit — verdict
    is RUNNER_MISSING (next failure), not A3_BUSY.
    """
    op_dir = make_ctc_loss_v3_op_dir(tmp_path)
    workspace = tmp_path / "workspace" / "ctc_loss_v3"
    workspace.mkdir(parents=True)
    runner = make_canned_runner([
        (0, "PROBE_OK\n/usr/bin/bisheng\n", ""),
    ])
    rep = p25a3.provision_a3_reference(
        op_dir=op_dir, workspace=workspace, a3_host="x",
        run_remote=runner, probe_only=False,
        npu_busy_check=_noop_busy_check,
    )
    # Past the gate — verdict reflects the next failure (RUNNER_MISSING), not A3_BUSY.
    assert rep.verdict == "RUNNER_MISSING"


# ---------------------------------------------------------------------------
# NPU range parsing + auto-pick (2026-05-13 Task #50)
# ---------------------------------------------------------------------------
def test_parse_npu_range_single():
    assert p25a3.parse_npu_range("0") == [0]
    assert p25a3.parse_npu_range("5") == [5]


def test_parse_npu_range_dash_range():
    assert p25a3.parse_npu_range("0-3") == [0, 1, 2, 3]
    assert p25a3.parse_npu_range("5-7") == [5, 6, 7]


def test_parse_npu_range_csv():
    assert p25a3.parse_npu_range("0,2,4") == [0, 2, 4]


def test_parse_npu_range_mixed():
    assert p25a3.parse_npu_range("0-1,4-5,7") == [0, 1, 4, 5, 7]


def test_parse_npu_range_empty():
    assert p25a3.parse_npu_range("") == []
    assert p25a3.parse_npu_range(",") == []


def test_parse_npu_range_dedup():
    assert p25a3.parse_npu_range("0,0,1,1,2") == [0, 1, 2]
    assert p25a3.parse_npu_range("0-2,1-3") == [0, 1, 2, 3]


def test_parse_npu_range_invalid_skipped():
    # Garbage tokens are skipped, valid ones kept
    assert p25a3.parse_npu_range("0,abc,2,xyz") == [0, 2]


def test_pick_idle_npu_in_range_picks_lowest_aicore(monkeypatch):
    """When multiple NPUs eligible, pick the lowest-AICore one."""
    fake_smi = """
| 0     0   | 0000:9D:00.0  | 15  0    / 0          43482/ 65536         |
| 1     1   | 0000:9F:00.0  | 3   0    / 0          42021/ 65536         |
| 2     2   | 0000:9B:00.0  | 5   0    / 0          43089/ 65536         |
"""
    monkeypatch.setattr(
        _a3npu, "_run_npu_smi",
        lambda *a, **kw: (0, fake_smi, ""),
    )
    chosen, observed, _ = p25a3.pick_idle_npu_in_range(
        "root", "1.2.3.4", [0, 1, 2], threshold_pct=20,
    )
    assert chosen == 1  # lowest aicore (3%) among candidates
    assert observed == {0: 15.0, 1: 3.0, 2: 5.0}


def test_pick_idle_npu_in_range_all_busy_returns_none(monkeypatch):
    """When all candidates exceed threshold → return None (caller refuses)."""
    fake_smi = """
| 0     0   | 0000:9D:00.0  | 87  0    / 0          43482/ 65536         |
| 1     1   | 0000:9F:00.0  | 65  0    / 0          42021/ 65536         |
"""
    monkeypatch.setattr(
        _a3npu, "_run_npu_smi",
        lambda *a, **kw: (0, fake_smi, ""),
    )
    chosen, observed, _ = p25a3.pick_idle_npu_in_range(
        "root", "1.2.3.4", [0, 1], threshold_pct=20,
    )
    assert chosen is None
    assert observed == {0: 87.0, 1: 65.0}


def test_pick_idle_npu_in_range_npu_smi_unreachable_falls_back(monkeypatch):
    """If npu-smi command itself fails, fall back to first candidate (best-effort)."""
    monkeypatch.setattr(
        _a3npu, "_run_npu_smi",
        lambda *a, **kw: (1, "", "ssh failed"),
    )
    chosen, observed, log = p25a3.pick_idle_npu_in_range(
        "root", "1.2.3.4", [3, 5, 7], threshold_pct=20,
    )
    assert chosen == 3
    assert observed == {}
    assert "npu-smi unavailable" in log


def test_pick_idle_npu_in_range_empty_candidates():
    chosen, observed, log = p25a3.pick_idle_npu_in_range(
        "root", "1.2.3.4", [], threshold_pct=20,
    )
    assert chosen is None
    assert observed == {}
    assert "no candidate" in log


def test_read_a3_npu_range_config_falls_back_to_default(tmp_path, monkeypatch):
    """When A3_NPU_RANGE absent, fall back to [A3_DEFAULT_NPU_ID].
    DEBT-101: route through the canonical DEFAULT_ASCENDC_ENV patch point.
    """
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text("A3_DEFAULT_NPU_ID=2\n")
    monkeypatch.setattr(_bc, "DEFAULT_ASCENDC_ENV", env_file)
    workspace = tmp_path / "ws" / "op"
    workspace.mkdir(parents=True)
    candidates, threshold = getattr(p25a3, "_read_a3_npu_range_config")(workspace)
    assert candidates == [2]
    assert threshold == 20


def test_read_a3_npu_range_config_parses_range(tmp_path, monkeypatch):
    """A3_NPU_RANGE overrides single-chip behavior.
    DEBT-101: route through the canonical DEFAULT_ASCENDC_ENV patch point.
    """
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(
        "A3_DEFAULT_NPU_ID=0\n"
        "A3_NPU_RANGE=0-1,4-5\n"
        "A3_AICORE_BUSY_THRESHOLD=30\n"
    )
    monkeypatch.setattr(_bc, "DEFAULT_ASCENDC_ENV", env_file)
    workspace = tmp_path / "ws" / "op"
    workspace.mkdir(parents=True)
    candidates, threshold = getattr(p25a3, "_read_a3_npu_range_config")(workspace)
    assert candidates == [0, 1, 4, 5]
    assert threshold == 30


# ---------------------------------------------------------------------------
# Prior-art inventory hook
# ---------------------------------------------------------------------------
def test_prior_art_scan_emits_target_inventory_when_opted_in(tmp_path, monkeypatch):
    op_dir = tmp_path / "cann" / "ops-nn" / "norm" / "ada_layer_norm"
    target_dir = op_dir / "op_kernel" / "arch35"
    target_dir.mkdir(parents=True)
    (target_dir / "ada_layer_norm.h").write_text("// target implementation\n")
    workspace = tmp_path / "ws" / "ada_layer_norm"
    workspace.mkdir(parents=True)

    monkeypatch.setenv("OPGEN_PRESTAGE_ARCH35", "1")
    output = getattr(p25a3, "_run_prior_art_scan")(op_dir, workspace)

    assert output is not None and output.exists()
    data = json.loads(output.read_text())
    assert data["has_prior_art"] is True
    assert data["consulted_a5_sources"] is True
    assert "upstream_arch35" in {source["type"] for source in data["sources"]}


def test_prior_art_scan_default_does_not_consult_target_tree(tmp_path, monkeypatch):
    op_dir = tmp_path / "cann" / "ops-nn" / "norm" / "ada_layer_norm"
    target_dir = op_dir / "op_kernel" / "arch35"
    target_dir.mkdir(parents=True)
    (target_dir / "ada_layer_norm.h").write_text("// target implementation\n")
    workspace = tmp_path / "ws" / "ada_layer_norm"
    workspace.mkdir(parents=True)

    monkeypatch.delenv("OPGEN_PRESTAGE_ARCH35", raising=False)
    output = getattr(p25a3, "_run_prior_art_scan")(op_dir, workspace)

    assert output is not None and output.exists()
    data = json.loads(output.read_text())
    assert data["consulted_a5_sources"] is False
    assert "upstream_arch35" not in {source["type"] for source in data["sources"]}


def test_prior_art_scan_is_non_fatal_when_skill_is_unavailable(tmp_path, monkeypatch):
    op_dir = tmp_path / "missing-op"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.delenv("OPGEN_PRESTAGE_ARCH35", raising=False)

    result = getattr(p25a3, "_run_prior_art_scan")(op_dir, workspace)

    if result is not None:
        assert json.loads(result.read_text())["has_prior_art"] is False


# ---------------------------------------------------------------------------
# CPU-truth scaffold (apply_adam_w_quant aftermath, 2026-05-14)
# ---------------------------------------------------------------------------
def test_cpu_truth_scaffold_writes_template(tmp_path):
    op_dir = tmp_path / "src" / "myop"
    op_dir.mkdir(parents=True)
    workspace = tmp_path / "ws" / "myop"
    workspace.mkdir(parents=True)
    out = getattr(p25a3, "_emit_cpu_truth_scaffold")(op_dir, workspace)
    assert out is not None and out.exists()
    body = out.read_text()
    assert "CPU truth template" in body
    assert "cpu_truth_reference.py" in body
    assert "When supplemental CPU truth is not authoritative" in body
    normalized = " ".join(body.split())
    assert "Provenance-tracked prior art or a target archive" in normalized
    assert "Live arch22 capture remains the migration truth" in normalized
    assert "myop" in body


def test_cpu_truth_scaffold_skips_when_truth_exists(tmp_path):
    """If kw already authored cpu_truth_reference.py, scaffold no-ops."""
    op_dir = tmp_path / "src" / "myop"
    op_dir.mkdir(parents=True)
    workspace = tmp_path / "ws" / "myop"
    workspace.mkdir(parents=True)
    (workspace / "cpu_truth_reference.py").write_text("# kw-authored\n")
    getattr(p25a3, "_emit_cpu_truth_scaffold")(op_dir, workspace)
    # template not created when reference already exists
    assert not (workspace / "cpu_truth_template.md").exists()


def test_cpu_truth_scaffold_includes_schema_from_manifest(tmp_path):
    op_dir = tmp_path / "src" / "myop"
    op_dir.mkdir(parents=True)
    workspace = tmp_path / "ws" / "myop"
    workspace.mkdir(parents=True)
    (workspace / "manifest.json").write_text(json.dumps({
        "inputs": ["var", "grad", "m"],
        "outputs": ["var_out", "m_out"],
    }))
    out = getattr(p25a3, "_emit_cpu_truth_scaffold")(op_dir, workspace)
    body = out.read_text()
    assert "var" in body
    assert "grad" in body


def test_staged_snapshot_provision_helpers_use_manifest_op_name(tmp_path, monkeypatch):
    op_dir = tmp_path / "heaviside"
    _source = op_dir / "op_kernel" / "arch22" / "heaviside.h"
    _source.parent.mkdir(parents=True)
    _source.write_text("class Heaviside { void Process() {} };\n")
    workspace = tmp_path / "workspace" / "heaviside"
    workspace.mkdir(parents=True)
    stage = stage_source_tree(op_dir, workspace)
    monkeypatch.delenv("OPGEN_PRESTAGE_ARCH35", raising=False)

    scaffold = getattr(p25a3, "_emit_cpu_truth_scaffold")(stage.root, workspace)
    scan = getattr(p25a3, "_run_prior_art_scan")(stage.root, workspace)

    assert scaffold is not None
    assert "`heaviside`" in scaffold.read_text()
    assert "`.source_arch22`" not in scaffold.read_text()
    assert scan is not None
    assert json.loads(scan.read_text())["op"] == "heaviside"


# ─── P124: local→A3 container path translation ─────────────────────────

def test_translate_zheng_user_home():
    """My local username 'zheng' → A3 container npu_user."""
    local = Path("/home/npu_user/workspace/cann/ops-nn/foreach/foreach_reciprocal")
    out = getattr(p25a3, "_translate_to_a3_container_path")(local)
    assert str(out) == "/home/npu_user/workspace/cann/ops-nn/foreach/foreach_reciprocal"


def test_a3_container_home_config_driven(tmp_path, monkeypatch):
    """A3_CONTAINER_HOME in .ascendc_env overrides the built-in default — the container home
    is a FUNCTIONAL runtime constant, config-driven (genericize) not a source literal, so a
    scrubbed / non-npu_user deployment sets its own and the container mount still matches.
    """
    patched = tmp_path / ".ascendc_env"
    patched.write_text("A3_CONTAINER_HOME=/home/custom_a3\n")
    monkeypatch.setattr(_bc, "DEFAULT_ASCENDC_ENV", patched)
    assert getattr(p25a3, "_a3_container_home")() == "/home/custom_a3"
    out = getattr(p25a3, "_translate_to_a3_container_path")(Path("/home/npu_user/workspace/cann/x"))
    assert str(out) == "/home/custom_a3/workspace/cann/x"


def test_a3_container_home_falls_back_when_unset(tmp_path, monkeypatch):
    """No A3_CONTAINER_HOME key → built-in default (behavior-preserving)."""
    patched = tmp_path / ".ascendc_env"
    patched.write_text("A3_HOST=1.2.3.4\n")
    monkeypatch.setattr(_bc, "DEFAULT_ASCENDC_ENV", patched)
    assert getattr(p25a3, "_a3_container_home")() == getattr(p25a3, "_A3_CONTAINER_HOME_DEFAULT")


def test_translate_arbitrary_user_home():
    """Robust to any local username (not hardcoded to 'zheng')."""
    for user in ("alice", "bob", "owen", "npu_user"):
        local = Path(f"/home/{user}/workspace/cann/ops-nn/op/foo")
        out = getattr(p25a3, "_translate_to_a3_container_path")(local)
        assert str(out) == "/home/npu_user/workspace/cann/ops-nn/op/foo"


def test_translate_passthrough_when_not_under_home():
    """Path NOT under /home/<user>/ returned unchanged."""
    for p in ("/tmp/foo", "/opt/bar", "/usr/local/lib/x"):
        assert getattr(p25a3, "_translate_to_a3_container_path")(Path(p)) == Path(p)


def test_translate_passthrough_when_already_container_path():
    """Idempotent on container paths (translate_x then translate_x = same)."""
    container = Path("/home/npu_user/workspace/cann/ops-nn/foo")
    once = getattr(p25a3, "_translate_to_a3_container_path")(container)
    twice = getattr(p25a3, "_translate_to_a3_container_path")(once)
    assert once == twice == Path("/home/npu_user/workspace/cann/ops-nn/foo")


def test_translate_short_home_no_tail():
    """`/home/<user>` with no tail returned unchanged (no meaningful translation)."""
    for p in ("/home/npu_user", "/home/npu_user/"):
        out = getattr(p25a3, "_translate_to_a3_container_path")(Path(p))
        # Either returned as-is OR with trailing slash normalization
        assert str(out) in (p, p.rstrip("/"))


# ─── DEBT-148: port_a3 edge_inputs.pt generation from op_def signature ──────
#
# These tests prove O2.5 generates deterministic edge_inputs.pt from
# op_host/<op>_def.cpp before the mandatory live arch22 capture.

_CELU_DEF_CPP = dedent("""
    #include "register/op_def_registry.h"
    namespace ops {
    class Celu : public OpDef {
    public:
        explicit Celu(const char* name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("alpha1").AttrType(OPTIONAL).Float(1.0f);
            this->Attr("alpha2").AttrType(OPTIONAL).Float(2.5f);
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        }
    };
    OP_ADD(Celu);
    }
""")


def _make_celu_op_dir(tmp_path):
    """A celu-class fixture: op_host/<op>_def.cpp present, NO aclnn entry,
    NO edge_inputs.pt — exactly the dead-ending shape."""
    op_dir = tmp_path / "celu"
    (op_dir / "op_host").mkdir(parents=True)
    (op_dir / "op_host" / "celu_def.cpp").write_text(_CELU_DEF_CPP)
    return op_dir


def test_parse_op_def_signature_celu(tmp_path):
    """Parser extracts Input + Attr defaults from the GE OpDef (no aclnn needed)."""
    op_dir = _make_celu_op_dir(tmp_path)
    sig = p25a3.parse_op_def_signature(op_dir)
    assert sig is not None
    assert sig["op_name"] == "celu"
    assert [i["name"] for i in sig["inputs"]] == ["x"]
    # First-listed dtype is fp16 for celu; both recorded.
    assert sig["inputs"][0]["dtypes"][:2] == ["float16", "float32"]
    attrs = {a["name"]: a["default"] for a in sig["attrs"]}
    assert attrs == {"alpha1": 1.0, "alpha2": 2.5}
    assert [o["name"] for o in sig["outputs"]] == ["y"]


def test_parse_op_def_signature_absent(tmp_path):
    """No op_host/<op>_def.cpp → None (caller falls through, no regression)."""
    op_dir = tmp_path / "nodef"
    op_dir.mkdir()
    assert p25a3.parse_op_def_signature(op_dir) is None


def test_generate_edge_inputs_produces_kwargs_dict_shape(tmp_path):
    """Generated edge_inputs.pt is list[dict] of Model(**case) kwargs:
    input tensors keyed by Input names + attr defaults merged in.
    """
    pytest.importorskip("torch")
    op_dir = _make_celu_op_dir(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    ok, msg = p25a3.generate_edge_inputs_from_signature(op_dir, ws)
    assert ok, msg
    edge = ws / "edge_inputs.pt"
    assert edge.is_file()
    cases = torch.load(edge, weights_only=False)
    assert isinstance(cases, list) and len(cases) > 0
    c0 = cases[0]
    assert isinstance(c0, dict)
    # tensor input + both attr defaults present (the Model(**case) contract)
    assert "x" in c0 and torch.is_tensor(c0["x"])
    assert c0["alpha1"] == 1.0 and c0["alpha2"] == 2.5
    # provenance marker dropped
    assert (ws / ".edge_inputs_provenance.json").is_file()


def test_ensure_edge_inputs_respects_existing_fixture(tmp_path):
    """If edge_inputs.pt already exists, ensure_edge_inputs is a no-op (True)."""
    pytest.importorskip("torch")
    op_dir = _make_celu_op_dir(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    sentinel = [{"x": torch.zeros(3)}]
    torch.save(sentinel, ws / "edge_inputs.pt")
    ok, msg = p25a3.ensure_edge_inputs(op_dir, ws)
    assert ok and "already present" in msg
    # untouched
    loaded = torch.load(ws / "edge_inputs.pt", weights_only=False)
    assert len(loaded) == 1


def test_ensure_edge_inputs_defers_to_input_gen_py(tmp_path):
    """If workspace ships input_gen.py, that path owns generation — don't pre-empt."""
    op_dir = _make_celu_op_dir(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "input_gen.py").write_text("# owns edge_inputs.pt generation\n")
    ok, msg = p25a3.ensure_edge_inputs(op_dir, ws)
    assert not ok and "input_gen.py present" in msg
    assert not (ws / "edge_inputs.pt").is_file()


# ---------------------------------------------------------------------------
# DEBT-165 port-ENTRY source-arch-completeness gate (architecture-based)
# ---------------------------------------------------------------------------
def _mk_op(tmp_path: Path, op: str, files: dict[str, str]) -> Path:
    src = tmp_path / "cann" / "ops-nn" / "x" / op
    for rel, content in files.items():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return src


def test_entry_gate_rejects_pure_target_arch_shell(tmp_path: Path) -> None:
    """deformable_offsets shape: <op>_apt.cpp #includes only arch35/ + arch35/<op>.h —
    no source-arch algorithm → gate returns (False, ...) so the port is rejected.
    """
    op = "deformable_offsets"
    src = _mk_op(tmp_path, op, {
        f"op_kernel/{op}_apt.cpp": '#include "arch35/deformable_offsets.h"\n',
        f"op_kernel/arch35/{op}.h": "// impl only here\n",
    })
    complete, reason = getattr(p25a3, "_source_arch_complete_gate")(src)
    assert complete is False, reason


def test_source_incomplete_quarantines_stale_ready_capture(tmp_path: Path) -> None:
    op = "deformable_offsets"
    src = _mk_op(tmp_path, op, {
        f"op_kernel/{op}_apt.cpp": '#include "arch35/deformable_offsets.h"\n',
        f"op_kernel/arch35/{op}.h": "// target-only implementation\n",
    })
    workspace = tmp_path / "workspace" / op
    workspace.mkdir(parents=True)
    for name in (
        "edge_dataset.pt",
        "a3_baseline_perf.json",
        "a3_capture_manifest.json",
    ):
        (workspace / name).write_bytes(b"stale")
    (workspace / "a3_reference_runnable.json").write_text(
        '{"verdict":"READY"}'
    )

    rep = p25a3.provision_a3_reference(
        op_dir=src,
        workspace=workspace,
        a3_host="1.2.3.4",
        run_remote=make_canned_runner([]),
    )

    assert rep.verdict == "SOURCE_ARCH_INCOMPLETE"
    for name in (
        "edge_dataset.pt",
        "a3_baseline_perf.json",
        "a3_capture_manifest.json",
    ):
        assert not (workspace / name).exists()
        assert list(workspace.glob(f"{name}.stale-pre-capture-*"))
    runnable = json.loads(
        (workspace / "a3_reference_runnable.json").read_text()
    )
    assert runnable["verdict"] == "SOURCE_ARCH_INCOMPLETE"
    assert list(
        workspace.glob("a3_reference_runnable.json.stale-pre-capture-*")
    )


def test_entry_gate_accepts_v220_entry_and_apt_with_algorithm(tmp_path: Path) -> None:
    """Legit shapes must pass: plain <op>.cpp, AND apt-only that references a
    non-target-arch algorithm lib (gelu/atvoss).
    """
    src_plain = _mk_op(tmp_path, "deformable_conv2d", {
        "op_kernel/arch22/deformable_conv2d.h": "class Op { void Process() {} };\n"
    })
    assert getattr(p25a3, "_source_arch_complete_gate")(src_plain)[0] is True
    src_apt = _mk_op(tmp_path, "gelu", {
        "op_kernel/gelu_apt.cpp": (
            '#include "arch35/gelu_dag.h"\n#include "atvoss/x.h"\n'
            "void Process() { DataCopy<int>(0); }\n"
        ),
        "op_kernel/arch35/gelu_dag.h": "// cfg\n",
    })
    assert getattr(p25a3, "_source_arch_complete_gate")(src_apt)[0] is True


def test_entry_gate_fails_closed_on_missing_dir(tmp_path: Path) -> None:
    """Missing source evidence is rejected before any migration mutation."""
    src = tmp_path / "cann" / "ops-nn" / "x" / "noop"
    src.mkdir(parents=True)
    complete, _ = getattr(p25a3, "_source_arch_complete_gate")(src)
    assert complete is False


def test_entry_gate_report_verdict_documented() -> None:
    """The SOURCE_ARCH_INCOMPLETE verdict must be documented in O25A3Report (the
    orchestrator keys its hard-reject on this exact string).
    """
    assert "SOURCE_ARCH_INCOMPLETE" in (p25a3.O25A3Report.__doc__ or "")


# ── ② PROVIDER-AUTO native_capture (owner 2026-06-30) — the load-bearing #1 fix ──────────────────
def test_provision_native_capture_auto_emits_at_native_dtype(tmp_path):
    """The provider (harness) AUTO-emits native_capture.pt = model.py.Model.forward @ NATIVE dtype on
    CPU over edge_inputs.pt — NOT a cast of fp64 cpu_truth. Without this, the 生态 grader gets
    native=None → the fp32 near-zero false-FAIL silently returns.
    """
    pytest.importorskip("torch")
    ws = tmp_path / "gelu"
    ws.mkdir()
    (ws / "model.py").write_text(
        "import torch, torch.nn as nn\n"
        "class Model(nn.Module):\n"
        "    def forward(self, x): return torch.nn.functional.gelu(x, approximate='none')\n")
    torch.manual_seed(0)
    x = torch.cat([torch.randn(2000) * 2.0, torch.linspace(-12.0, -7.0, 96)]).to(torch.float32)
    torch.save([{"x": x}], ws / "edge_inputs.pt")

    ok, msg = p25a3.provision_native_capture(ws)
    assert ok, msg
    assert (ws / "native_capture.pt").is_file()             # AUTO-emitted, no manual step
    blob = torch.load(ws / "native_capture.pt", weights_only=False)
    assert blob["native_kind"] == "cpu_same_precision"      # tagged
    assert blob["dtype"] == "float32"                        # op's native dtype (not fp64)
    nat = blob["outputs"]
    assert len(nat) == 1 and nat[0].dtype == torch.float32
    # NOT a cast of the fp64 truth: a real fp32 model run differs from cpu_truth.to(fp32)
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", str(ws / "model.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cpu_truth_fp64 = m.Model()(x.double())
    assert nat[0].shape == cpu_truth_fp64.shape             # shape-aligned to cpu_truth
    assert int((nat[0] != cpu_truth_fp64.to(torch.float32)).sum().item()) > 0, \
        "native_capture must be a REAL native-dtype run, not a truncation of the fp64 truth"


def test_provision_native_capture_missing_inputs_is_safe_noop(tmp_path):
    """Best-effort: missing edge_inputs/model → returns (False, reason), no crash, native=None (safe)."""
    ws = tmp_path / "op"
    ws.mkdir()
    ok, msg = p25a3.provision_native_capture(ws)
    assert ok is False and "edge_inputs.pt missing" in msg
    assert not (ws / "native_capture.pt").exists()


def test_provision_a3_reference_wires_native_capture_step():
    """Source-pin: O2.5 provision_a3_reference calls provision_native_capture (Step 0.8) so it is
    auto-run for every forward port_a3 op.
    """
    src = Path(p25a3.__file__).read_text()
    assert "provision_native_capture(workspace)" in src
    assert "Step 0.8" in src


# ── codex01 re-review fixes (owner 2026-06-30): P1 input_gen / P2 fp32-fallback / P3 fail-closed ──
def test_provision_native_capture_input_gen_op_produces_native(tmp_path):
    """P1: an input_gen.py op with edge_inputs.pt initially ABSENT must STILL get native — Step 0.8
    runs input_gen.py first (else input_gen ops fall back to native=None → near-zero false-FAIL).
    """
    pytest.importorskip("torch")
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / "model.py").write_text(
        "import torch, torch.nn as nn\n"
        "class Model(nn.Module):\n    def forward(self, x): return torch.nn.functional.gelu(x)\n")
    (ws / "input_gen.py").write_text(
        "import torch\n"
        "torch.save([{'x': torch.randn(64, dtype=torch.float32)}], 'edge_inputs.pt')\n")
    assert not (ws / "edge_inputs.pt").exists()              # input_gen path — edge absent at Step 0.8
    ok, msg = p25a3.provision_native_capture(ws)
    assert ok, msg
    assert (ws / "native_capture.pt").is_file()             # native PRODUCED for the input_gen op
    assert (ws / "edge_inputs.pt").is_file()                # input_gen.py was run
    blob = torch.load(ws / "native_capture.pt", weights_only=False)
    assert blob["native_kind"] == "cpu_same_precision"


def test_provision_native_capture_fp32_fallback_tag(tmp_path):
    """P2: when the CPU native-dtype run raises (fp16 unsupported) and falls back to fp32-compute, the
    provenance MUST be tagged cpu_fp32_fallback — NOT the same-precision tag (audit must not be misled).
    """
    pytest.importorskip("torch")
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / "model.py").write_text(
        "import torch, torch.nn as nn\n"
        "class Model(nn.Module):\n"
        "    def forward(self, x):\n"
        "        if x.dtype == torch.float16:\n"
        "            raise RuntimeError('fp16 not supported on CPU for this op')\n"
        "        return torch.nn.functional.gelu(x)\n")
    torch.save([{"x": torch.randn(64).to(torch.float16)}], ws / "edge_inputs.pt")
    ok, msg = p25a3.provision_native_capture(ws)
    assert ok, msg
    blob = torch.load(ws / "native_capture.pt", weights_only=False)
    assert blob["native_kind"] == "cpu_fp32_fallback"       # honest fallback tag (NOT cpu_same_precision)
    assert blob["n_fp32_fallback"] == 1


def test_provision_native_capture_partial_failure_is_fail_closed(tmp_path):
    """P3: if ANY case fails to produce native, provision returns False and writes NO file — a partial
    native (None holes) must NOT masquerade as provisioned (that is the silent native=None the fix guards).
    """
    pytest.importorskip("torch")
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / "model.py").write_text(
        "import torch, torch.nn as nn\n"
        "class Model(nn.Module):\n"
        "    def forward(self, x):\n"
        "        if float(x.flatten()[0]) < 0:\n"
        "            raise RuntimeError('boom on this case')\n"
        "        return torch.nn.functional.gelu(x)\n")
    torch.save([{"x": torch.ones(8)}, {"x": -torch.ones(8)}], ws / "edge_inputs.pt")  # case1 raises
    ok, msg = p25a3.provision_native_capture(ws)
    assert ok is False
    assert "FAIL-CLOSED" in msg and "1/2" in msg
    assert not (ws / "native_capture.pt").exists()          # NO partial file written


def test_provision_native_capture_deletes_stale_on_failure(tmp_path):
    """① codex01 (stale-delete): a pre-existing (stale/worker) native_capture.pt MUST be removed even
    when the new provision FAILS — else O5 (local-wins via FORCE_UPDATE_SCRIPTS) force-pushes the stale
    native to the remote as authoritative → the ① re-grade reads stale native → fix silently no-ops.
    """
    pytest.importorskip("torch")
    ws = tmp_path / "op"
    ws.mkdir()
    torch.save({"native_kind": "cpu_same_precision", "outputs": [torch.ones(4)]},
               ws / "native_capture.pt")            # pre-existing STALE native
    # NO model.py / edge_inputs / input_gen.py → provision will FAIL
    ok, msg = p25a3.provision_native_capture(ws)
    assert ok is False
    assert not (ws / "native_capture.pt").exists()  # stale REMOVED on failure (no stale force-push)
