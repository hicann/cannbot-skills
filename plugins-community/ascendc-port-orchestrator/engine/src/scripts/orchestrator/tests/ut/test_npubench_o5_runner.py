# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""O5 bridge coverage for target-owned NPUKernelBench evaluation."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import _reorg_paths  # noqa: F401  (stable sys.path setup for reorganized tests)
import pytest

from a5_target_capability import (
    a5_soc_version,
    is_limited_a5_soc,
    parse_npu_smi_soc,
    soc_product_family,
)
from npubench import npubench_o5_runner as bridge
from npubench import npubench_target as target
import phase_o5  # noqa: F401  (load before replacing bridge-only dependencies)
import reference_source
import events
import fsm_phase_finalize as finalize_fsm
import state_executor
from fsm_context import OrchestratorContext


def _prepare_tilelang2ascendc_workspace(tmp_path, monkeypatch, soc: str) -> Path:
    """Create a workspace whose durable state selects the TileLang2AscendC port route."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".opgen_state.json").write_text(
        '{"port_source":{"kind":"port-aclnn-tilelang2ascendc","tree_sha256":"'
        + "a" * 64
        + '"}}\n',
        encoding="utf-8",
    )
    env_file = workspace.parent / ".ascendc_env"
    env_file.write_text(f"A5_SOC_VERSION={soc}\n", encoding="utf-8")
    monkeypatch.setenv("ASCENDC_ENV_PATH", str(env_file))
    return workspace


def _install_build_only_target(monkeypatch, build_result: dict) -> None:
    """Replace npubench_target with a double whose controlled build returns build_result."""
    fake_target = types.ModuleType("npubench.npubench_target")

    def build_tilelang2ascendc_candidate_on_target(*_args, **_kwargs):
        return dict(build_result)

    fake_target.build_tilelang2ascendc_candidate_on_target = build_tilelang2ascendc_candidate_on_target
    monkeypatch.setitem(sys.modules, "npubench.npubench_target", fake_target)


def _make_local_target():
    """Build the in-process target endpoint shared by receipt-path tests."""
    from npubench.npubench_target import _Target

    return _Target(
        name="A5",
        host="",
        user="tester",
        password="",
        container="local",
        cann_path="/opt/Ascend/cann",
        benchmark_root="/benchmark-root",
        host_mode=False,
        visible_device=0,
        ssh_options=(),
        env={"A5_SOC_VERSION": "Ascend950PR"},
    )


def _leases(device: int) -> dict:
    """The lease record shape the bridge hands to the target transport."""
    return {
        "precision": {"device": device, "token": "precision"},
        "performance": {"device": device, "token": "performance"},
        "parallelism": {"mode": "degraded_single_lane", "reason": "test"},
    }


def _install_o5_bridge_doubles(monkeypatch, snapshot, evaluate, reference, lease_pair) -> None:
    """Install the runner/target/reference/lease doubles shared by the O5 bridge tests."""
    fake_runner = types.ModuleType("npubench.npubench_runner")

    def materialize_candidate_snapshot(_supplied_workspace):
        return snapshot

    fake_runner.materialize_candidate_snapshot = materialize_candidate_snapshot
    # Intentionally no evaluate_workspace attribute: accessing it would prove
    # that the controller-local evaluator has leaked back into the O5 path.
    fake_target = types.ModuleType("npubench.npubench_target")
    fake_target.evaluate_npubench_on_target = evaluate
    monkeypatch.setitem(sys.modules, "npubench.npubench_runner", fake_runner)
    monkeypatch.setitem(sys.modules, "npubench.npubench_target", fake_target)
    monkeypatch.setattr(
        reference_source, "load_durable_state", lambda _supplied_workspace: {"reference": reference}
    )
    monkeypatch.setattr(
        reference_source, "resolve_reference_binding", lambda state: state["reference"]
    )
    monkeypatch.setattr(bridge, "_acquire_leases", lambda *_args, **_kwargs: lease_pair)


def test_non_direct_910_is_not_blocked_by_direct_capability_gate(tmp_path, monkeypatch) -> None:
    """The capability build gate is scoped to explicitly supported source kinds."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".opgen_state.json").write_text('{"port_source":{"kind":"aclnn"}}\n')
    env_file = workspace.parent / ".ascendc_env"
    env_file.write_text("A5_SOC_VERSION=Ascend910B3\n", encoding="utf-8")
    monkeypatch.setenv("ASCENDC_ENV_PATH", str(env_file))

    def sentinel_lease(*_args, **_kwargs):
        raise RuntimeError("sentinel")

    monkeypatch.setattr(bridge, "_acquire_leases", sentinel_lease)

    result = bridge.npubench_verify_runner(workspace, "add", lane=0)

    assert result.rollback_kind == "infra"
    assert "A5_SOC_UNSUPPORTED_FOR_VALIDATION" not in (result.runner_error or "")


def test_tilelang_project_910_builds_before_capability_terminal_gate(tmp_path, monkeypatch) -> None:
    """TileLang2AscendC uses its dedicated controlled build and never leases on Ascend910."""
    workspace = _prepare_tilelang2ascendc_workspace(tmp_path, monkeypatch, "Ascend910B3")
    calls: list[str] = []
    fake_target = types.ModuleType("npubench.npubench_target")

    def build_tilelang(*_args, **_kwargs):
        calls.append("tilelang2ascendc")
        return {"status": "PASS"}

    fake_target.build_tilelang2ascendc_candidate_on_target = build_tilelang
    monkeypatch.setitem(sys.modules, "npubench.npubench_target", fake_target)
    monkeypatch.setattr(
        bridge, "_acquire_leases", lambda *_args, **_kwargs: pytest.fail("must stop before lease/evaluate")
    )

    result = bridge.npubench_verify_runner(workspace, "gelu", lane=0)

    assert calls == ["tilelang2ascendc"]
    assert result.rollback_kind == "target_capability"
    assert "A5_SOC_UNSUPPORTED_FOR_VALIDATION" in (result.runner_error or "")


def test_tilelang_project_build_failure_stops_before_lease(tmp_path, monkeypatch) -> None:
    workspace = _prepare_tilelang2ascendc_workspace(tmp_path, monkeypatch, "Ascend950PR")
    _install_build_only_target(
        monkeypatch,
        {"status": "ERROR", "reason": "controlled build failed"},
    )
    monkeypatch.setattr(
        bridge, "_acquire_leases", lambda *_args, **_kwargs: pytest.fail("must not lease on build failure")
    )

    result = bridge.npubench_verify_runner(workspace, "gelu", lane=0)

    assert result.rollback_kind == "infra"
    assert result.runner_error == "controlled build failed"
    assert result.failure_kind == "target_build"


def test_candidate_failure_kind_requires_authenticated_receipt(tmp_path, monkeypatch) -> None:
    """An in-memory build label cannot route O5 without a valid receipt."""
    workspace = _prepare_tilelang2ascendc_workspace(tmp_path, monkeypatch, "Ascend950PR")
    _install_build_only_target(
        monkeypatch,
        {
            "status": "ERROR",
            "reason": "candidate rejected before build: missing CMakeLists.txt",
            "failure_kind": "candidate_contract",
        },
    )
    monkeypatch.setattr(
        bridge, "_acquire_leases", lambda *_args, **_kwargs: pytest.fail("must not lease")
    )

    result = bridge.npubench_verify_runner(workspace, "gelu", lane=0)

    assert result.failure_kind == "target_build"


def _prepare_generic_kernel_workspace(tmp_path, monkeypatch, soc: str) -> Path:
    """Create a port_a3_to_a5-style workspace: no port_source, generic kernel project."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".opgen_state.json").write_text(
        json.dumps(
            {
                "op": "flash_attention_score",
                "source_stage_digest": "a" * 64,
                "reference": {"source": "npubench"},
            }
        ),
        encoding="utf-8",
    )
    (workspace / "model_new_ascendc.py").write_text(
        "class ModelNew:\n    pass\n", encoding="utf-8"
    )
    (workspace / "kernel").mkdir()
    (workspace / "kernel" / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n", encoding="utf-8"
    )
    env_file = workspace.parent / ".ascendc_env"
    env_file.write_text(f"A5_SOC_VERSION={soc}\n", encoding="utf-8")
    monkeypatch.setenv("ASCENDC_ENV_PATH", str(env_file))
    return workspace


def test_generic_kernel_project_uses_generic_controlled_build(tmp_path, monkeypatch) -> None:
    """A workspace without port_source still gets the engine-controlled build."""
    workspace = _prepare_generic_kernel_workspace(tmp_path, monkeypatch, "Ascend950PR")
    calls: list[str] = []
    fake_target = types.ModuleType("npubench.npubench_target")

    def build_generic(*_args, **_kwargs):
        calls.append("generic")
        return {"status": "PASS"}

    # Intentionally no build_tilelang2ascendc_candidate_on_target attribute:
    # resolving it would prove the generic route leaked into the TileLang2AscendC entry.
    fake_target.build_generic_kernel_project_on_target = build_generic
    monkeypatch.setitem(sys.modules, "npubench.npubench_target", fake_target)

    def sentinel_lease(*_args, **_kwargs):
        raise RuntimeError("sentinel")

    monkeypatch.setattr(bridge, "_acquire_leases", sentinel_lease)

    result = bridge.npubench_verify_runner(workspace, "flash_attention_score", lane=0)

    assert calls == ["generic"]
    assert result.rollback_kind == "infra"
    assert "sentinel" in (result.runner_error or "")


def test_generic_kernel_project_build_failure_stops_before_lease(tmp_path, monkeypatch) -> None:
    workspace = _prepare_generic_kernel_workspace(tmp_path, monkeypatch, "Ascend950PR")
    fake_target = types.ModuleType("npubench.npubench_target")

    def build_generic(*_args, **_kwargs):
        return {"status": "ERROR", "reason": "generic controlled build failed"}

    fake_target.build_generic_kernel_project_on_target = build_generic
    monkeypatch.setitem(sys.modules, "npubench.npubench_target", fake_target)
    monkeypatch.setattr(
        bridge, "_acquire_leases", lambda *_args, **_kwargs: pytest.fail("must not lease on build failure")
    )

    result = bridge.npubench_verify_runner(workspace, "flash_attention_score", lane=0)

    assert result.rollback_kind == "infra"
    assert result.runner_error == "generic controlled build failed"
    # The fake double carries no signed receipt: fail closed as target_build.
    assert result.failure_kind == "target_build"


def test_generic_candidate_receipt_flows_through_bridge(tmp_path, monkeypatch) -> None:
    """The generic route's signed candidate-contract receipt survives real authentication."""
    from npubench.npubench_target import _Target, _receipt_payload_valid

    # Empty kernel/CMakeLists.txt: present enough for the precheck fallback,
    # rejected by the generic pre-build validation, which exercises the real
    # writer -> authenticated reader path.
    workspace = _prepare_generic_kernel_workspace(tmp_path, monkeypatch, "Ascend950PR")
    (workspace / "kernel" / "CMakeLists.txt").write_text("", encoding="utf-8")
    endpoint = _make_local_target()
    monkeypatch.setattr(target, "_target", lambda *_args: endpoint)
    monkeypatch.setattr(
        bridge,
        "_acquire_leases",
        lambda *_args, **_kwargs: pytest.fail("candidate rejection must stop before lease"),
    )

    result = bridge.npubench_verify_runner(workspace, "flash_attention_score", lane=0)

    assert result.runner_error is not None
    assert "kernel/CMakeLists.txt" in result.runner_error
    assert result.failure_kind == "candidate_contract"
    receipt_path = workspace / target.TILELANG2ASCENDC_BUILD_RECEIPT_PATH
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["source_kind"] is None
    assert receipt["failure_kind"] == "candidate_contract"
    assert _receipt_payload_valid(receipt, workspace)


def test_compile_failure_classifier_routes_candidate_rooted_include_cascade() -> None:
    """An SDK-header error cascade rooted below kernel/ is a candidate failure."""
    from npubench.npubench_build_receipt import _classify_controlled_compile_failure

    workspace = "/tmp/ws"
    cascade_stderr = (
        f"In file included from {workspace}/kernel/op_kernel/fusion_attention.cpp:27:\n"
        f"In file included from {workspace}/kernel/utils/fusion_attention_tiling.h:17:\n"
        "In file included from /usr/local/Ascend/cann/include/tiling/tiling_api.h:23:\n"
        "/usr/local/Ascend/cann/include/graph/types.h:90:3: error: expected identifier\n"
        "  DT_INT64 = ::C_DT_INT64,\n"
        "  ^\n"
        "fatal error: too many errors emitted, stopping now [-ferror-limit=]\n"
    )

    assert (
        _classify_controlled_compile_failure("", cascade_stderr)
        == "candidate_contract"
    )
    # gcc folds include chains into indented "from <path>:N," continuations.
    gcc_cascade_stderr = (
        "In file included from "
        "/usr/local/lib/python3.11/site-packages/torch_npu/include/third_party/"
        "acl/inc/graph/operator.h:19,\n"
        "                 from "
        "/usr/local/lib/python3.11/site-packages/torch_npu/include/torch_npu/"
        "csrc/framework/OpCommand.h:6,\n"
        f"                 from {workspace}/kernel/utils/torch_kernel_helper.h:21,\n"
        f"                 from {workspace}/kernel/op_host/fusion_attention.cpp:27:\n"
        "/usr/local/lib/python3.11/site-packages/torch_npu/include/third_party/acl/inc/graph/./ge_error_codes.h:64:19: "
        "error: redefinition of 'const graphStatus ge::GRAPH_RoundUp_Overflow'\n"
    )
    assert (
        _classify_controlled_compile_failure("", gcc_cascade_stderr)
        == "candidate_contract"
    )
    # A diagnostic naming a candidate file directly still takes the same path.
    assert (
        _classify_controlled_compile_failure(
            "", "kernel/op_kernel/gelu.cpp:12:5: error: use of undeclared identifier 'x'"
        )
        == "candidate_contract"
    )
    # Missing SDK headers stay fail-closed even when the candidate pulls them in.
    missing_sdk = (
        f"In file included from {workspace}/kernel/op_kernel/gelu.cpp:3:\n"
        "/usr/local/Ascend/cann/include/tiling/tiling_api.h:23:10: fatal error: "
        "'tiling/missing.h' file not found\n"
    )
    assert _classify_controlled_compile_failure("", missing_sdk) == "target_build"
    # Pure toolchain noise without any candidate evidence stays a target failure.
    assert (
        _classify_controlled_compile_failure("", "cmake: command not found\n")
        == "target_build"
    )


def _real_receipt_workspace(tmp_path, monkeypatch) -> Path:
    """Materialize the on-disk workspace the real receipt writer/reader path needs."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = {
        "op": "gelu",
        "port_source": {
            "kind": "port-aclnn-tilelang2ascendc",
            "tree_sha256": "a" * 64,
            "digest": "a" * 64,
        },
        "source_stage_digest": "a" * 64,
        "reference": {"source": "npubench"},
    }
    (workspace / ".opgen_state.json").write_text(json.dumps(state), encoding="utf-8")
    (workspace / "model_new_ascendc.py").write_text(
        "class ModelNew:\n    pass\n", encoding="utf-8"
    )
    (workspace / "kernel").mkdir()
    env_file = workspace.parent / ".ascendc_env"
    env_file.write_text("A5_SOC_VERSION=Ascend950PR\n", encoding="utf-8")
    monkeypatch.setenv("ASCENDC_ENV_PATH", str(env_file))
    return workspace


def _install_local_target_endpoint(monkeypatch):
    """Bind npubench_target to an in-process A5 endpoint and forbid lease acquisition."""
    endpoint = _make_local_target()
    monkeypatch.setattr(target, "_target", lambda *_args: endpoint)
    monkeypatch.setattr(target, "_source_stage_digest", lambda *_args: "a" * 64)
    monkeypatch.setattr(target, "_verified_tilelang2ascendc_source_manifest", lambda *_args: {"files": []})
    monkeypatch.setattr(
        bridge,
        "_acquire_leases",
        lambda *_args, **_kwargs: pytest.fail("candidate rejection must stop before lease"),
    )
    return endpoint


def _assert_o5_repair_path_rolls_back_to_worker(workspace: Path, monkeypatch) -> None:
    """Drive the signed-receipt result through the production O5 report and finalize dispatch.

    This is the repair path, not a manually constructed O5 stub.
    """
    monkeypatch.setattr(phase_o5, "expected_truth_source", lambda _ws: "npubench")
    monkeypatch.setattr(
        finalize_fsm,
        "_o5_runner_for_workspace",
        lambda *_args, **_kwargs: bridge.npubench_verify_runner,
    )
    monkeypatch.setattr(state_executor, "at_iter_cap", lambda _ws, _state: False)
    monkeypatch.setattr(state_executor, "iter_cap", lambda _state, workspace=None: 9)
    monkeypatch.setattr(events, "emit", lambda *args, **kwargs: None)
    from fsm_phase_finalize import _o5_post_verify

    fsm_result = _o5_post_verify(
        OrchestratorContext(op="gelu", workspace=workspace, lane=0),
        types.SimpleNamespace(iter_counts={}),
    )
    assert fsm_result.action == "continue"
    transition = json.loads(
        (workspace / "state_transitions.jsonl").read_text().splitlines()[-1]
    )
    assert transition["to_state"] == "await_worker"
    assert transition["rollback_kind"] == "algorithm"
    persisted_reason = json.loads(
        (workspace / ".rollback_history.jsonl").read_text().splitlines()[-1]
    )["reason"]
    assert "kernel/CMakeLists.txt" in persisted_reason


def _assert_receipt_replay_is_rejected(workspace: Path, receipt_path: Path, endpoint) -> None:
    """A stale, SoC-rebound or HMAC-tampered receipt must never keep its candidate_contract label."""
    from npubench.npubench_o5_runner import _authenticated_build_failure_kind
    from npubench.npubench_target import _receipt_payload_sha256

    def failure_kind(evaluation, attempt_id: str) -> str:
        return _authenticated_build_failure_kind(
            workspace,
            "port-aclnn-tilelang2ascendc",
            evaluation,
            lane=0,
            expected_attempt_id=attempt_id,
        )

    attempt_one = "1" * 32
    first = target.build_tilelang2ascendc_candidate_on_target(
        workspace, lane=0, build_attempt_id=attempt_one
    )
    assert failure_kind(first, attempt_one) == "candidate_contract"

    # The current target SoC, not the receipt's historical SoC, is part of the
    # authenticated classification.  A target reconfigured to limited 910
    # must invalidate the old candidate-contract receipt.
    endpoint.env["A5_SOC_VERSION"] = "Ascend910B3"
    assert failure_kind(first, attempt_one) == "target_build"
    endpoint.env["A5_SOC_VERSION"] = "Ascend950PR"

    # A newer signed receipt makes the old in-process return stale even though
    # both payloads are individually authentic.
    attempt_two = "2" * 32
    target.build_tilelang2ascendc_candidate_on_target(
        workspace, lane=0, build_attempt_id=attempt_two
    )
    assert failure_kind(first, attempt_one) == "target_build"

    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["failure_kind"] = "candidate_contract"
    tampered["payload_sha256"] = _receipt_payload_sha256(tampered)
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    second = target.build_tilelang2ascendc_candidate_on_target(
        workspace, lane=0, build_attempt_id=attempt_two
    )
    # The target builder rewrites the receipt above, so use a direct HMAC
    # mutation to verify the reader's fail-closed branch independently.
    current = json.loads(receipt_path.read_text(encoding="utf-8"))
    current["receipt_auth_hmac"] = "0" * 64
    receipt_path.write_text(json.dumps(current), encoding="utf-8")
    assert failure_kind(second, attempt_two) == "target_build"


def test_real_candidate_receipt_flows_through_bridge_and_rejects_replay(
    tmp_path, monkeypatch
) -> None:
    """Exercise the real writer -> authenticated reader -> MeasuredResult path."""
    from npubench.npubench_target import _receipt_payload_valid

    workspace = _real_receipt_workspace(tmp_path, monkeypatch)
    endpoint = _install_local_target_endpoint(monkeypatch)

    result = bridge.npubench_verify_runner(workspace, "gelu", lane=0)

    assert result.runner_error is not None
    assert result.failure_kind == "candidate_contract"
    receipt_path = workspace / target.TILELANG2ASCENDC_BUILD_RECEIPT_PATH
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["failure_kind"] == "candidate_contract"
    assert _receipt_payload_valid(receipt, workspace)

    _assert_o5_repair_path_rolls_back_to_worker(workspace, monkeypatch)
    _assert_receipt_replay_is_rejected(workspace, receipt_path, endpoint)


@pytest.mark.parametrize(
    "soc",
    (
        "910A",
        "Ascend910A",
        "910B",
        "Ascend910B2C",
        "Ascend910B3",
        "Ascend910C",
        "Ascend910_9382",
        "Ascend910-V220",
        "ascend910c_superpod",
    ),
)
def test_a5_capability_gate_matches_910_product_variants(soc: str) -> None:
    assert is_limited_a5_soc(soc)


@pytest.mark.parametrize("soc", ("", "Ascend999", "Ascend950PR??", None))
def test_a5_capability_gate_fails_closed_for_invalid_soc(soc: object) -> None:
    assert is_limited_a5_soc(soc)


def test_empty_target_specific_soc_does_not_fallback_to_generic_soc() -> None:
    resolved = a5_soc_version(
        {"A5_SOC_VERSION": "", "SOC_VERSION": "Ascend950PR"}
    )

    assert resolved == ""
    assert is_limited_a5_soc(resolved)


def test_missing_target_specific_soc_does_not_fallback_to_generic_soc() -> None:
    assert a5_soc_version({"SOC_VERSION": "Ascend950PR"}) == ""


@pytest.mark.parametrize(
    ("output", "device", "expected"),
    (
        ("| 0      | Ascend950PR      | OK |\n|        |                  | 0000:71:00.0 |", 0, "Ascend950PR"),
        ("| 0     910B2C              | OK |\n|        0000:71:00.0 |", 0, "910B2C"),
        ("| 0      | Ascend950PR      | OK |\n| 1      | Ascend910B3      | OK |", 1, "Ascend910B3"),
    ),
)
def test_npu_smi_model_parser_binds_requested_device(
    output: str, device: int, expected: str
) -> None:
    assert parse_npu_smi_soc(output, device) == expected


def test_npu_smi_model_family_is_explicit() -> None:
    assert soc_product_family("Ascend950PR_9579") == "ascend950"
    assert soc_product_family("Ascend950DT_9582") == "ascend950"
    assert soc_product_family("Ascend910B3") == "ascend910"
    assert soc_product_family("unknown") is None


def test_a5_capability_gate_does_not_match_910_in_an_a5_suffix() -> None:
    assert not is_limited_a5_soc("Ascend950PR_9107x")


@pytest.mark.parametrize(
    "soc",
    ("Ascend950DT_9582", "950DT_9582", "ascend950dt_9582"),
)
def test_a5_capability_gate_accepts_950dt_sku_full_names(soc: str) -> None:
    assert not is_limited_a5_soc(soc)


def test_o5_freezes_candidate_then_delegates_to_target_transport(tmp_path, monkeypatch) -> None:
    """A remote-capable route must never call the controller evaluator directly."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    snapshot = workspace / ".npubench_candidate" / ("a" * 64)
    snapshot.mkdir(parents=True)
    manifest = workspace / "lease.json"
    manifest.write_text("{}\n", encoding="utf-8")
    leases = _leases(2)
    reference = {"source": "npubench", "bundle_sha256": "b" * 64}
    calls: list[dict] = []

    def evaluate_npubench_on_target(**kwargs):
        calls.append(kwargs)
        binding = {
            "binding_sha256": "c" * 64,
            "candidate_tree_sha256": "a" * 64,
        }
        precision = {
            "status": "PASS",
            "binding_sha256": binding["binding_sha256"],
            "evaluation_binding": binding,
            "pass_a": {"status": "PASS", "tier1_pass": 1, "total": 1},
        }
        performance = {
            "status": "PASS",
            "binding_sha256": binding["binding_sha256"],
            "evaluation_binding": binding,
        }
        return {
            "status": "PASS",
            "binding_sha256": binding["binding_sha256"],
            "evaluation_binding": binding,
            "precision": precision,
            "performance": performance,
            "transport": "ssh_target",
            "target_receipt_path": "npubench_evidence/target_receipt.json",
            "target_receipt_sha256": "d" * 64,
        }

    _install_o5_bridge_doubles(
        monkeypatch, snapshot, evaluate_npubench_on_target, reference, (leases, manifest)
    )
    monkeypatch.setattr(bridge, "_release_leases", lambda *_args, **_kwargs: None)

    result = bridge.npubench_verify_runner(workspace, "add", lane=2)

    assert result.runner_error is None
    assert result.pass_a == {"status": "PASS", "tier1_pass": 1, "total": 1}
    assert result.provider_evidence["evaluate"]["transport"] == "ssh_target"
    assert result.provider_evidence["evaluate"]["candidate_snapshot"] == str(snapshot)
    assert calls == [
        {
            "workspace": workspace,
            "reference": reference,
            "candidate_snapshot": snapshot,
            "precision_device": 2,
            "performance_device": 2,
            "lease_manifest": manifest,
        }
    ]


def test_o5_refuses_a_transport_result_without_a_canonical_receipt(tmp_path) -> None:
    """A transport error cannot briefly become an O5 VERIFIED result."""
    from npubench.npubench_o5_runner import _measured_result_from_evaluation

    manifest = tmp_path / "lease.json"
    manifest.write_text("{}\n", encoding="utf-8")
    result = _measured_result_from_evaluation(
        {"status": "PASS"},
        manifest_path=manifest,
        leases={"precision": {}, "performance": {}, "parallelism": {}},
    )

    assert result.runner_error is not None
    assert "canonical target receipt path" in result.runner_error


def _passing_target_evaluation(**_kwargs) -> dict:
    """A target evaluation payload whose precision and performance both PASS."""
    return {
        "status": "PASS",
        "binding_sha256": "c" * 64,
        "evaluation_binding": {"binding_sha256": "c" * 64},
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 1, "total": 1},
        },
        "performance": {"status": "PASS"},
        "target_receipt_path": "npubench_evidence/target_receipt.json",
        "target_receipt_sha256": "d" * 64,
    }


def test_o5_lease_cleanup_failure_cannot_return_evaluation_pass(
    tmp_path, monkeypatch
) -> None:
    """A PASS evaluation is fail-closed when lease release is uncertain."""
    from phase_o5 import _npubench_o5_report

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    snapshot = workspace / ".npubench_candidate" / ("a" * 64)
    snapshot.mkdir(parents=True)
    manifest = workspace / "lease.json"
    manifest.write_text('{}\n', encoding="utf-8")
    leases = _leases(0)
    reference = {"source": "npubench", "bundle_sha256": "b" * 64}
    _install_o5_bridge_doubles(
        monkeypatch, snapshot, _passing_target_evaluation, reference, (leases, manifest)
    )

    def fail_release(*_args, **_kwargs):
        raise RuntimeError("release unavailable")

    monkeypatch.setattr(bridge, "_release_leases", fail_release)

    result = bridge.npubench_verify_runner(workspace, "add", lane=0)

    assert result.pass_a == {"status": "PASS", "tier1_pass": 1, "total": 1}
    assert result.runner_error is not None
    assert "npubench lease cleanup failed" in result.runner_error
    assert "release unavailable" in result.runner_error
    assert result.rollback_kind == "infra"
    assert result.provider_evidence["lease_cleanup"]["status"] == "ERROR"
    assert result.provider_evidence["lease_cleanup"]["manifest_path"] == str(manifest)
    assert result.provider_evidence["evaluate"]["status"] == "PASS"
    assert "cleanup_failure" in json.loads(manifest.read_text())
    report = _npubench_o5_report(workspace, "add", 0, lambda *_args: result)
    assert report.verdict == "RUNNER_FAILED"
    assert report.rollback_kind == "infra"


def test_o5_lease_cleanup_failure_preserves_evaluation_exception(
    tmp_path, monkeypatch
) -> None:
    """Cleanup failure augments, rather than replaces, an evaluation error."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest = workspace / "lease.json"
    manifest.write_text('{}\n', encoding="utf-8")
    leases = _leases(0)
    reference = {"source": "npubench", "bundle_sha256": "b" * 64}

    def fail_evaluation(**_kwargs):
        raise ValueError("target evaluation exploded")

    _install_o5_bridge_doubles(
        monkeypatch, workspace, fail_evaluation, reference, (leases, manifest)
    )

    def fail_release(*_args, **_kwargs):
        raise RuntimeError("release unavailable")

    monkeypatch.setattr(bridge, "_release_leases", fail_release)

    result = bridge.npubench_verify_runner(workspace, "add", lane=0)

    assert result.pass_a is None
    assert result.runner_error is not None
    assert "npubench evaluation raised ValueError: target evaluation exploded" in result.runner_error
    assert "npubench lease cleanup failed" in result.runner_error
    assert result.rollback_kind == "infra"
    assert result.provider_evidence["lease_cleanup"]["status"] == "ERROR"
