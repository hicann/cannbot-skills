# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Focused finalization regressions for native NPUKernelBench evidence."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import _reorg_paths  # noqa: F401  (stable sys.path setup for reorganized tests)

from npubench import npubench_finalize_contract as contract
import phase_o5
# CodeCheck G.CLS.11: bind the module-under-test's protected helpers by name
# instead of reaching through the module object.  None of them are
# monkeypatched anywhere in the suite, so an import-time binding is safe.
from npubench.npubench_finalize_contract import (
    _validate_controlled_build_receipt,
    _validate_report_contents,
)
from phase_o5 import _persist_npubench_verification


_BINDING_SHA = "a" * 64
_CANDIDATE_SHA = "b" * 64
_TASK_SHA = "c" * 64
_ENTRY_SHA = "d" * 64


def _reference() -> dict[str, object]:
    return {
        "schema_version": 3,
        "source": "npubench",
        "semantic_binding": "npubench_old_format_task_bundle",
        "runner_contract_version": "npubench/v1",
        "bundle_manifest_path": "reference_inputs/npubench/" + "e" * 64 + "/bundle_manifest.json",
        "bundle_manifest_sha256": "f" * 64,
        "bundle_sha256": "e" * 64,
        "task_relative_path": "level1/3_Add.py",
        "task_sha256": _TASK_SHA,
        "sidecar_relative_path": "level1/3_Add.json",
        "sidecar_sha256": "1" * 64,
        "sidecar_encoding": "jsonl",
    }


def _binding(*, candidate_sha: str = _CANDIDATE_SHA) -> dict[str, object]:
    return {
        "source": "npubench",
        "binding_sha256": _BINDING_SHA,
        "candidate_tree_sha256": candidate_sha,
        "candidate_entry_sha256": _ENTRY_SHA,
        "candidate_entry": "model_new_ascendc.py",
        "task_sha256": _TASK_SHA,
        "candidate_digest_scheme": "npubench-candidate-scope/v3",
    }


def _execution_integrity() -> dict[str, object]:
    return {
        "execution_isolation": "process_boundary",
        "tamper_protection": "post_run_hash_check",
        "child_returncode": 0,
    }


def _report(verb: str, binding: dict[str, object], **extra: object) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": f"cannbot.npubench.{verb}/v1",
        "runner_contract_version": "npubench/v1",
        "status": "PASS",
        "binding_sha256": _BINDING_SHA,
        "evaluation_binding": dict(binding),
    }
    result.update(extra)
    return result


def _install_runner(monkeypatch, *, current_binding: dict[str, object] | None = None) -> None:
    fake = types.ModuleType("npubench.npubench_runner")

    def tree_sha256(_path: Path) -> str:
        return _CANDIDATE_SHA

    def build_evaluation_binding(_workspace: Path, candidate_dir: Path) -> dict[str, object]:
        # The finalizer deliberately asks both the frozen snapshot and the
        # current workspace candidate.  Make the latter controllable here.
        if candidate_dir == _workspace and current_binding is not None:
            return dict(current_binding)
        return _binding()

    def verify_evidence_report(_workspace, _report_value, *, expected_verb, candidate_dir):
        assert expected_verb in {"precision", "performance", "evaluate"}
        assert candidate_dir.name == _CANDIDATE_SHA
        return True, "verified"

    # The contract verifies the snapshot's content address through the
    # runner's canonical scope digest (built extension modules under
    # kernel/build are excluded).  The stubbed snapshots carry no excluded
    # paths, so scope digest == full-tree digest here.
    fake.tree_sha256 = tree_sha256
    fake.candidate_tree_sha256 = tree_sha256
    fake.CANDIDATE_DIGEST_SCHEME = "npubench-candidate-scope/v3"
    fake.build_evaluation_binding = build_evaluation_binding
    fake.verify_evidence_report = verify_evidence_report
    monkeypatch.setitem(sys.modules, "npubench.npubench_runner", fake)


def _install_target_receipt_verifier(monkeypatch) -> None:
    """Keep this finalizer unit focused on its canonical receipt gate.

    The transport module owns the full file/tree/target-identity check and has
    its own hermetic tests.  Here we prove that finalization invokes that
    verifier only after enforcing the fixed harness pointer contract.
    """
    fake = types.ModuleType("npubench.npubench_target")

    def validate_target_evidence_receipt(**_kwargs) -> tuple[bool, str]:
        return True, "verified"

    fake.validate_target_evidence_receipt = validate_target_evidence_receipt
    monkeypatch.setitem(sys.modules, "npubench.npubench_target", fake)


def _quick_profile_command() -> list[str]:
    return [
        "python3", "ops/ops-profiling/scripts/msprof_perf_summary.py", "--quick",
        "--warmup", "3", "--device", "0", "--keep-prof", "--repeats", "5",
        "--prof-tag", "run-1", "--output-dir=<native-profiler-adapter>",
    ]


def _native_fixture() -> dict[str, object]:
    return {
        "manifest_sha256": "2" * 64,
        "fixture_sha256": "3" * 64,
        "case_count": 2,
        "valid_case_indices": [0],
        "empty_case_indices": [1],
        "case_fixtures": [
            {"case": 0, "path": "native_perf_cases/case_000000.pt", "sha256": "4" * 64, "empty_tensor": False},
            {"case": 1, "path": "native_perf_cases/case_000001.pt", "sha256": "5" * 64, "empty_tensor": True},
        ],
    }


def _profiler_summary() -> dict[str, object]:
    return {
        "native_input_manifest_sha256": "2" * 64,
        "native_fixture_sha256": "3" * 64,
        "n_cases_total": 2,
        "n_cases_valid": 1,
        "per_case": [
            {
                "case": 0,
                "ref_us": 10.0,
                "asc_us": 5.0,
                "speedup": 2.0,
                "ref_prof_dir": "/tmp/ref",
                "asc_prof_dir": "/tmp/asc",
                "native_case_fixture_sha256": "4" * 64,
                "native_case_path": "native_perf_cases/case_000000.pt",
            },
            {
                "case": 1,
                "skipped": "empty_tensor",
                "ref_us": None,
                "asc_us": None,
                "speedup": None,
                "ref_prof_dir": None,
                "asc_prof_dir": None,
                "native_case_fixture_sha256": "5" * 64,
                "native_case_path": "native_perf_cases/case_000001.pt",
            },
        ],
    }


def _performance_report(binding: dict[str, object], archive: Path) -> dict[str, object]:
    return _report(
        "performance",
        binding,
        **_execution_integrity(),
        fixture_execution_isolation="process_boundary",
        fixture_tamper_protection="post_run_hash_check",
        warm_up=3,
        repeats=5,
        keep_prof=True,
        profiling_mode="quick",
        command=_quick_profile_command(),
        profile_archive="npubench_evidence/profiles/run-1",
        profile_tree_sha256=contract.profile_tree_sha256(archive),
        native_fixture=_native_fixture(),
        profiler_summary=_profiler_summary(),
    )


def _prepare_candidate_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".opgen_state.json").write_text(
        json.dumps({"opgen_mode": "port_a3_to_a5", "reference": _reference()}),
        encoding="utf-8",
    )
    (workspace / "model_new_ascendc.py").write_text("class ModelNew: pass\n", encoding="utf-8")
    snapshot = workspace / ".npubench_candidate" / _CANDIDATE_SHA
    snapshot.mkdir(parents=True)
    candidate = snapshot / "model_new_ascendc.py"
    candidate.write_text("class ModelNew: pass\n", encoding="utf-8")
    os.chmod(candidate, 0o400)
    os.chmod(snapshot, 0o500)
    return workspace


def _write_evidence_reports(evidence_root: Path, reports: dict[str, dict[str, object]]) -> None:
    for name, payload in reports.items():
        (evidence_root / f"{name}_report.json").write_text(json.dumps(payload), encoding="utf-8")


def _verification_record(precision: dict[str, object], performance: dict[str, object]) -> dict[str, object]:
    return {
        "truth_source": "npubench",
        "precision": {"pass_a": precision["pass_a"]},
        "performance": performance,
        "npubench_evidence": {
            "binding_sha256": _BINDING_SHA,
            "precision_report": "npubench_evidence/precision_report.json",
            "performance_report": "npubench_evidence/performance_report.json",
            "evaluate_report": "npubench_evidence/evaluate_report.json",
            "target_execution_receipt": "npubench_evidence/target_receipt.json",
            "target_execution_receipt_sha256": "6" * 64,
        },
    }


def _make_workspace(tmp_path: Path, monkeypatch, *, current_binding=None) -> tuple[Path, dict[str, object]]:
    workspace = _prepare_candidate_workspace(tmp_path)

    monkeypatch.setattr(contract, "verify_npubench_stage", lambda *_: (True, "ok", {}))
    _install_runner(monkeypatch, current_binding=current_binding)
    _install_target_receipt_verifier(monkeypatch)

    archive = workspace / "npubench_evidence" / "profiles" / "run-1"
    archive.mkdir(parents=True)
    (archive / "raw.csv").write_text("profile", encoding="utf-8")
    binding = _binding()
    precision = _report(
        "precision",
        binding,
        pass_a={"status": "PASS", "tier1_pass": 2, "total": 2},
        **_execution_integrity(),
    )
    performance = _performance_report(binding, archive)
    evaluate = _report("evaluate", binding, precision=precision, performance=performance)
    _write_evidence_reports(
        workspace / "npubench_evidence",
        {"precision": precision, "performance": performance, "evaluate": evaluate},
    )
    return workspace, _verification_record(precision, performance)


def test_finalize_accepts_one_complete_parent_published_evidence_unit(tmp_path, monkeypatch):
    workspace, verification = _make_workspace(tmp_path, monkeypatch)

    assert contract.validate_npubench_finalize_evidence(workspace, verification) is None


def _bind_sidecar_descriptor_evidence(workspace: Path, verification: dict[str, object]) -> dict[str, object]:
    """Upgrade the compact fixture to the canonical sidecar adapter contract."""
    identity = {
        "kind": "sidecar_descriptor/v1",
        "provider": None,
        "contract": "npubench-input-adapter/v1",
        "schema": "cannbot.npubench.sidecar_descriptor/v1",
        "case_count": 2,
    }
    evidence_root = workspace / "npubench_evidence"
    precision = json.loads((evidence_root / "precision_report.json").read_text(encoding="utf-8"))
    performance = json.loads((evidence_root / "performance_report.json").read_text(encoding="utf-8"))
    evaluate = json.loads((evidence_root / "evaluate_report.json").read_text(encoding="utf-8"))
    precision["evaluation_binding"]["input_adapter"] = identity
    precision["input_adapter"] = identity
    precision["case_count"] = 2
    precision["cases"] = [{"case": 0}, {"case": 1}]
    performance["native_fixture"]["input_adapter"] = identity
    evaluate["evaluation_binding"]["input_adapter"] = identity
    evaluate["precision"] = precision
    evaluate["performance"] = performance
    for name, payload in (("precision", precision), ("performance", performance), ("evaluate", evaluate)):
        (evidence_root / f"{name}_report.json").write_text(json.dumps(payload), encoding="utf-8")
    verification["performance"] = performance
    verification["precision"] = {"pass_a": precision["pass_a"]}
    return verification


def test_finalize_accepts_sidecar_descriptor_binding(tmp_path, monkeypatch):
    workspace, verification = _make_workspace(tmp_path, monkeypatch)
    _bind_sidecar_descriptor_evidence(workspace, verification)

    assert contract.validate_npubench_finalize_evidence(workspace, verification) is None


def test_finalize_rejects_malformed_sidecar_native_fixture_without_raising(tmp_path, monkeypatch):
    workspace, verification = _make_workspace(tmp_path, monkeypatch)
    _bind_sidecar_descriptor_evidence(workspace, verification)
    performance_path = workspace / "npubench_evidence" / "performance_report.json"
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    performance["native_fixture"] = "not-an-object"
    performance_path.write_text(json.dumps(performance), encoding="utf-8")
    evaluate_path = workspace / "npubench_evidence" / "evaluate_report.json"
    evaluate = json.loads(evaluate_path.read_text(encoding="utf-8"))
    evaluate["performance"] = performance
    evaluate_path.write_text(json.dumps(evaluate), encoding="utf-8")
    verification["performance"] = performance

    error = contract.validate_npubench_finalize_evidence(workspace, verification)

    assert error == "NPUBENCH_EVIDENCE_INVALID: performance native fixture evidence is malformed"


def test_finalize_rejects_current_candidate_drift_after_snapshot(tmp_path, monkeypatch):
    changed = _binding(candidate_sha="9" * 64)
    workspace, verification = _make_workspace(tmp_path, monkeypatch, current_binding=changed)

    error = contract.validate_npubench_finalize_evidence(workspace, verification)

    assert error is not None
    assert "current candidate scope differs" in error


def test_finalize_reports_digest_scheme_drift_for_older_evidence(tmp_path, monkeypatch):
    """Older-scheme evidence yields the distinct scheme-drift message.

    Audit H1: the finalize self-heal keys off that message instead of routing
    the unfixable drift to await_worker.
    """
    workspace, verification = _make_workspace(tmp_path, monkeypatch)
    precision_path = workspace / "npubench_evidence" / "precision_report.json"
    precision = json.loads(precision_path.read_text(encoding="utf-8"))
    precision["evaluation_binding"]["candidate_digest_scheme"] = "npubench-candidate-scope/v2"
    precision_path.write_text(json.dumps(precision), encoding="utf-8")

    error = contract.validate_npubench_finalize_evidence(workspace, verification)

    assert error is not None
    assert "candidate digest scheme drift" in error


def test_finalize_rejects_inline_or_report_binding_substitution(tmp_path, monkeypatch):
    workspace, verification = _make_workspace(tmp_path, monkeypatch)
    verification["npubench_evidence"]["binding_sha256"] = "0" * 64

    error = contract.validate_npubench_finalize_evidence(workspace, verification)

    assert error is not None
    assert "binding_sha256 values differ" in error


def test_finalize_rejects_missing_or_noncanonical_target_receipt_pointer(tmp_path, monkeypatch):
    workspace, verification = _make_workspace(tmp_path, monkeypatch)
    evidence = verification["npubench_evidence"]
    evidence.pop("target_execution_receipt")

    error = contract.validate_npubench_finalize_evidence(workspace, verification)

    assert error is not None
    assert "target execution receipt path is not canonical" in error


def test_finalize_rejects_profile_archive_symlink_or_digest_drift(tmp_path, monkeypatch):
    workspace, verification = _make_workspace(tmp_path, monkeypatch)
    archive = workspace / "npubench_evidence" / "profiles" / "run-1"
    moved = workspace / "elsewhere"
    archive.rename(moved)
    try:
        archive.symlink_to(moved, target_is_directory=True)
    except OSError:
        # Filesystems without symlink support still exercise the same final
        # contract via an archive digest drift.
        archive.mkdir()
        (archive / "changed.csv").write_text("changed", encoding="utf-8")

    error = contract.validate_npubench_finalize_evidence(workspace, verification)

    assert error is not None
    assert "profile archive" in error


_T2A_SOURCE_KIND = "port-aclnn-tilelang2ascendc"
_T2A_INDEPENDENCE_SCHEMA = "cannbot.tilelang2ascendc_candidate_independence/v1"
_T2A_RECEIPT_RELPATH = "npubench_evidence/tilelang2ascendc_build_receipt.json"


def _install_tilelang2ascendc_target_stub(monkeypatch) -> None:
    """Stub the transport module's tilelang2ascendc controlled-build surface."""
    fake = types.ModuleType("npubench.npubench_target")
    fake.TILELANG2ASCENDC_BUILD_RECEIPT_PATH = _T2A_RECEIPT_RELPATH
    fake.TILELANG2ASCENDC_CANDIDATE_INDEPENDENCE_SCHEMA = _T2A_INDEPENDENCE_SCHEMA

    def receipt_payload_valid(_receipt, _workspace) -> bool:
        return True

    def candidate_source_digest(_snapshot) -> str:
        return "8" * 64

    # setattr names the protected members as strings, so the stub keeps the exact
    # private surface the contract consumes without a protected attribute access.
    setattr(fake, "_receipt_payload_valid", receipt_payload_valid)
    setattr(fake, "_candidate_source_digest", candidate_source_digest)
    monkeypatch.setitem(sys.modules, "npubench.npubench_target", fake)


def _tilelang2ascendc_build_state(source_digest: str) -> dict[str, object]:
    return {
        "source_stage_digest": source_digest,
        "port_source": {
            "kind": _T2A_SOURCE_KIND,
            "tree_sha256": source_digest,
            "digest": source_digest,
        },
    }


def _tilelang2ascendc_build_receipt(source_digest: str) -> dict[str, object]:
    return {
        "status": "PASS",
        "source_kind": _T2A_SOURCE_KIND,
        "source_stage_digest": source_digest,
        "build_mode": "controlled_authored_cmake",
        "returncode": 0,
        "candidate_source_sha256": "8" * 64,
        "candidate_independence_gate": "PASS",
        "candidate_independence_schema": _T2A_INDEPENDENCE_SCHEMA,
        "candidate_independence_proof": {"format": "tilelang2ascendc"},
    }


def _make_controlled_build_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    snapshot = workspace / ".npubench_candidate" / _CANDIDATE_SHA
    snapshot.mkdir(parents=True)
    (snapshot / "model_new_ascendc.py").write_text("candidate", encoding="utf-8")
    (workspace / "npubench_evidence").mkdir()
    return workspace


def test_tilelang_controlled_build_receipt_uses_authored_cmake_contract(
    tmp_path, monkeypatch
):
    workspace = _make_controlled_build_workspace(tmp_path)
    source_digest = "7" * 64
    state = _tilelang2ascendc_build_state(source_digest)
    receipt = _tilelang2ascendc_build_receipt(source_digest)
    receipt_path = workspace / _T2A_RECEIPT_RELPATH
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    _install_tilelang2ascendc_target_stub(monkeypatch)

    binding = {"candidate_tree_sha256": _CANDIDATE_SHA}
    assert _validate_controlled_build_receipt(workspace, state, binding) is None

    receipt["build_mode"] = "controlled_generated_cmake"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    error = _validate_controlled_build_receipt(workspace, state, binding)
    assert error == "NPUBENCH_EVIDENCE_INVALID: controlled build receipt is not a PASS for this route"


def test_finalize_rejects_partial_native_profile_coverage_after_report_publish(
    tmp_path, monkeypatch
):
    workspace, verification = _make_workspace(tmp_path, monkeypatch)
    performance_path = workspace / "npubench_evidence" / "performance_report.json"
    evaluate_path = workspace / "npubench_evidence" / "evaluate_report.json"
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    performance["profiler_summary"]["per_case"] = performance["profiler_summary"]["per_case"][:1]
    performance_path.write_text(json.dumps(performance), encoding="utf-8")
    evaluate = json.loads(evaluate_path.read_text(encoding="utf-8"))
    evaluate["performance"] = performance
    evaluate_path.write_text(json.dumps(evaluate), encoding="utf-8")
    verification["performance"] = performance

    error = contract.validate_npubench_finalize_evidence(workspace, verification)

    assert error is not None
    assert "per-case coverage" in error


def _persist_npubench_o5_record(workspace: Path) -> phase_o5.O5Report:
    """Publish a small harness record through the same O5 helper as production."""
    report = phase_o5.O5Report(verdict="VERIFIED", truth_source="npubench")
    _persist_npubench_verification(
        workspace,
        precision={"status": "PASS", "pass_a": {"status": "PASS", "tier1_pass": 1, "total": 1}},
        performance={"status": "PASS", "warm_up": 3, "repeats": 5, "keep_prof": True},
        evaluate={"binding_sha256": _BINDING_SHA},
        evidence={"leases": {"parallelism": {"mode": "degraded_single_lane"}}},
        report=report,
    )
    return report


def test_npubench_o5_publishes_regular_atomic_verification_and_preserves_unrelated_fields(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "verification.json").write_text(json.dumps({"keep": "field", "precision": {"fake": True}}))

    report = _persist_npubench_o5_record(workspace)

    assert report.verdict == "VERIFIED"
    published_path = workspace / "verification.json"
    published = json.loads(published_path.read_text())
    assert published["keep"] == "field"
    assert published["truth_source"] == "npubench"
    assert published["precision"]["pass_a"]["tier1_pass"] == 1
    assert not published_path.is_symlink()
    assert published_path.stat().st_nlink == 1


def test_npubench_o5_refuses_worker_verification_symlink_without_touching_target(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "outside.json"
    target.write_text("outside must not change", encoding="utf-8")
    (workspace / "verification.json").symlink_to(target)

    report = _persist_npubench_o5_record(workspace)

    assert report.verdict == "RUNNER_FAILED"
    assert "safely" in report.summary
    assert target.read_text(encoding="utf-8") == "outside must not change"
    assert (workspace / "verification.json").is_symlink()


def test_npubench_o5_refuses_worker_verification_hardlink_without_touching_target(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "outside.json"
    target.write_text("outside must not change", encoding="utf-8")
    verification = workspace / "verification.json"
    try:
        os.link(target, verification)
    except OSError as exc:
        # The contract is portable; this filesystem simply cannot exercise its
        # hard-link branch locally.
        import pytest

        pytest.skip(f"hard links unavailable: {exc}")

    report = _persist_npubench_o5_record(workspace)

    assert report.verdict == "RUNNER_FAILED"
    assert target.read_text(encoding="utf-8") == "outside must not change"
    assert verification.stat().st_ino == target.stat().st_ino


def test_npubench_harness_stamp_keeps_the_safe_atomic_publish_path(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "verification.json").write_text(json.dumps({"truth_source": "npubench"}))
    monkeypatch.setattr(phase_o5, "expected_truth_source", lambda _workspace: "npubench")
    report = phase_o5.O5Report(verdict="VERIFIED", truth_source="npubench")

    assert phase_o5.record_harness_state(workspace, report) is True

    verification = workspace / "verification.json"
    assert not verification.is_symlink()
    assert verification.stat().st_nlink == 1
    assert json.loads(verification.read_text())["harness_pristine"]["o5_verdict"] == "VERIFIED"


def test_npubench_harness_stamp_does_not_follow_worker_verification_link(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "outside.json"
    target.write_text('{"outside": true}', encoding="utf-8")
    try:
        (workspace / "verification.json").symlink_to(target)
    except OSError as exc:
        import pytest

        pytest.skip(f"symlinks unavailable: {exc}")
    monkeypatch.setattr(phase_o5, "expected_truth_source", lambda _workspace: "npubench")
    report = phase_o5.O5Report(verdict="VERIFIED", truth_source="npubench")

    assert phase_o5.record_harness_state(workspace, report) is False
    assert target.read_text(encoding="utf-8") == '{"outside": true}'


def _passing_precision_report() -> dict[str, object]:
    """Minimal PASS precision payload shared by the deferred/backfill units."""
    return {
        "status": "PASS",
        "binding_sha256": "e" * 64,
        "evaluation_binding": None,
        "input_adapter": None,
        "pass_a": {"status": "PASS", "tier1_pass": 2, "total": 2},
    }


def _deferred_performance_report(*, marker: bool = True) -> dict[str, object]:
    """Deferred performance payload; ``marker`` toggles the perf_deferred flag."""
    report: dict[str, object] = {
        "status": "DEFERRED",
        "binding_sha256": "e" * 64,
        "evaluation_binding": None,
    }
    if marker:
        report["perf_deferred"] = True
    return report


def test_deferred_performance_skips_perf_gates(tmp_path: Path, monkeypatch) -> None:
    """DEFERRED performance + PASS precision passes the finalize contract (P3)."""
    monkeypatch.delenv("CANNBOT_NPUBENCH_PERF_BACKFILL", raising=False)
    precision = _passing_precision_report()
    performance = _deferred_performance_report()
    evaluate = {"status": "PASS"}
    assert _validate_report_contents(precision, performance, evaluate, tmp_path) is None


def test_deferred_performance_requires_marker_and_matching_binding(tmp_path: Path, monkeypatch) -> None:
    """DEFERRED without the marker or with a divergent binding is rejected."""
    monkeypatch.delenv("CANNBOT_NPUBENCH_PERF_BACKFILL", raising=False)
    precision = _passing_precision_report()
    performance = _deferred_performance_report(marker=False)
    assert _validate_report_contents(
        precision, performance, {"status": "PASS"}, tmp_path
    ) == "NPUBENCH_EVIDENCE_INVALID: performance DEFERRED without perf_deferred marker"
    performance["perf_deferred"] = True
    performance["binding_sha256"] = "f" * 64
    assert _validate_report_contents(
        precision, performance, {"status": "PASS"}, tmp_path
    ) == "NPUBENCH_EVIDENCE_INVALID: deferred performance binding differs from precision"


def test_backfill_rejects_still_deferred_performance(tmp_path: Path, monkeypatch) -> None:
    """Backfill refuses a still-DEFERRED performance report.

    Perf backfill (codex review F1, 2026-08-25): under
    CANNBOT_NPUBENCH_PERF_BACKFILL=1 the backfill has to produce a real
    measurement, so a still-DEFERRED report must NOT route to done.
    """
    monkeypatch.setenv("CANNBOT_NPUBENCH_PERF_BACKFILL", "1")
    precision = _passing_precision_report()
    performance = _deferred_performance_report()
    reason = _validate_report_contents(
        precision, performance, {"status": "PASS"}, tmp_path
    )
    assert reason is not None
    assert "still DEFERRED" in reason
    assert "CANNBOT_NPUBENCH_SKIP_PERF" in reason


def test_backfill_requires_measured_performance(tmp_path: Path, monkeypatch) -> None:
    """Backfill requires an explicitly measured performance report.

    A PASS report without measurement_completed=true is not accepted as the
    backfilled measurement.
    """
    monkeypatch.setenv("CANNBOT_NPUBENCH_PERF_BACKFILL", "1")
    precision = _passing_precision_report()
    performance = {"status": "PASS", "binding_sha256": "e" * 64}
    reason = _validate_report_contents(
        precision, performance, {"status": "PASS"}, tmp_path
    )
    assert reason is not None
    assert "measurement_completed" in reason
