# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Unit coverage for the P2-4 precision repeat fingerprint mechanism.

Covers the five 2026-08-28 additions to ``npubench_runner``: in-evaluation
repeats for FAIL cases, the per-case ``repeat_fingerprint`` classification
vocabulary, the reference-stability gate (``reference_unstable_cases`` plus
the scored denominator), the cross-evaluation determinism alert (NEW-P2), and
the reference-side annotations (NEW-P1).  All cases are driven by fake
models/observations; no NPU is required.
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import _reorg_paths  # noqa: F401  (stable sys.path setup for reorganized tests)
import pytest

from npubench import npubench_runner as runner


def _runner_attr(name):
    # Resolve protected runner helpers at call time so monkeypatches remain visible.
    return getattr(runner, name)


# torch must be bound at collection time: the DEBT-47 tests/conftest.py
# isolation fixture restores sys.modules after every test, so a first
# in-test import is evicted afterwards and a second in-test import re-executes
# the torch package — which raises (its C-level functions already carry
# docstrings).  A collection-time import predates every snapshot and survives.
try:
    import torch as _TORCH
except ImportError:
    _TORCH = None


@pytest.fixture(autouse=True)
def _local_runner_configuration(tmp_path: Path, monkeypatch) -> None:
    """Keep runner unit tests local unless a test explicitly configures a target."""
    config = tmp_path / ".ascendc_env"
    config.write_text("A5_CONTAINER=local\n", encoding="utf-8")
    monkeypatch.setenv("ASCENDC_ENV_PATH", str(config))
    for key in ("A5_CONTAINER", "A5_NPU_PYTHON_BIN", "NPU_PYTHON_BIN", "A5_HOST_PYTHON"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv(runner.PRECISION_REPEATS_ENV, raising=False)
    monkeypatch.delenv(runner.REFERENCE_OVERFLOW_INPUT_ABS_ENV, raising=False)


def _torch_or_skip():
    """Keep the fingerprint UT runnable without PyTorch."""
    if _TORCH is None:
        pytest.skip("PyTorch unavailable")
    return _TORCH


def _observation(
    status: str,
    *,
    mere=0.5,
    matched_ratio=0.8,
    candidate_inf_count=0,
    candidate_nan_count=0,
    reference_inf_count=0,
    reference_nan_count=0,
) -> dict:
    return {
        "status": status,
        "mere": mere,
        "matched_ratio": matched_ratio,
        "candidate_inf_count": candidate_inf_count,
        "candidate_nan_count": candidate_nan_count,
        "reference_inf_count": reference_inf_count,
        "reference_nan_count": reference_nan_count,
        "reference_nonfinite_mask_sha256": None,
    }


# --- classification vocabulary ------------------------------------------------


def test_classify_stable_pass_tolerates_metric_drift_between_passes() -> None:
    observations = [
        _observation("PASS", mere=1.0e-6, matched_ratio=1.0),
        _observation("PASS", mere=2.0e-6, matched_ratio=0.9999),
        _observation("PASS", mere=1.0e-6, matched_ratio=1.0),
    ]
    assert _runner_attr("_classify_repeat_fingerprint")(observations) == "stable-pass"


def test_classify_deterministic_fail_requires_bit_identical_failures() -> None:
    observations = [_observation("FAIL")] * 3
    assert _runner_attr("_classify_repeat_fingerprint")(observations) == "deterministic-fail"


def test_classify_deterministic_fail_normalizes_nan_metrics() -> None:
    # nan != nan as floats; the fingerprint key must still treat two nan-MERE
    # observations of one diverged candidate as bit-identical.
    observations = [_observation("FAIL", mere=float("nan"))] * 2
    assert _runner_attr("_classify_repeat_fingerprint")(observations) == "deterministic-fail"


def test_classify_bimodal_on_diverging_failure_values() -> None:
    observations = [_observation("FAIL", mere=1.5), _observation("FAIL", mere=2.5)]
    assert _runner_attr("_classify_repeat_fingerprint")(observations) == "bimodal"


def test_classify_bimodal_on_verdict_flip() -> None:
    observations = [_observation("PASS", mere=1e-6), _observation("FAIL", mere=198202.21875)]
    assert _runner_attr("_classify_repeat_fingerprint")(observations) == "bimodal"


def test_classify_reference_unstable_on_any_reference_nonfinite() -> None:
    # Constant NaN count: no jump needed, a non-finite reference already
    # disqualifies the case from candidate adjudication.
    observations = [_observation("FAIL", reference_nan_count=42816)] * 3
    assert _runner_attr("_classify_repeat_fingerprint")(observations) == "reference-unstable"


def test_classify_reference_unstable_on_count_jump_across_repeats() -> None:
    observations = [
        _observation("FAIL", reference_nan_count=0),
        _observation("FAIL", reference_nan_count=42816),
    ]
    assert _runner_attr("_classify_repeat_fingerprint")(observations) == "reference-unstable"


def test_classify_reference_unstable_wins_over_stable_pass() -> None:
    observations = [_observation("PASS", reference_inf_count=3)] * 2
    assert _runner_attr("_classify_repeat_fingerprint")(observations) == "reference-unstable"


def test_classify_candidate_nonfinite_alone_is_not_reference_unstable() -> None:
    observations = [_observation("FAIL", candidate_inf_count=32385)] * 2
    assert _runner_attr("_classify_repeat_fingerprint")(observations) == "deterministic-fail"


def test_classify_rejects_empty_observations() -> None:
    with pytest.raises(runner.NpuBenchRunnerError, match="at least one observation"):
        _runner_attr("_classify_repeat_fingerprint")([])


# --- env resolvers ------------------------------------------------------------


def test_precision_repeats_default_and_env_override(monkeypatch) -> None:
    assert _runner_attr("_resolve_precision_repeats")() == runner.PRECISION_REPEATS_DEFAULT == 3
    monkeypatch.setenv(runner.PRECISION_REPEATS_ENV, "1")
    assert _runner_attr("_resolve_precision_repeats")() == 1
    monkeypatch.setenv(runner.PRECISION_REPEATS_ENV, "5")
    assert _runner_attr("_resolve_precision_repeats")() == 5


def test_precision_repeats_rejects_invalid_env(monkeypatch) -> None:
    for raw in ("0", "-2", "abc", ""):
        monkeypatch.setenv(runner.PRECISION_REPEATS_ENV, raw)
        with pytest.raises(runner.NpuBenchRunnerError, match="positive integer"):
            _runner_attr("_resolve_precision_repeats")()


# --- tracked case execution with fingerprint ----------------------------------


class _ScriptedModel:
    """Callable model returning scripted per-invocation outputs."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0

    def __call__(self, *args, **kwargs):
        output = self._outputs[min(self.calls, len(self._outputs) - 1)]
        self.calls += 1
        return output.clone() if hasattr(output, "clone") else output


def _run_two_case_fingerprint(reference, candidate, *, repeats):
    torch = _torch_or_skip()
    groups = [
        [torch.tensor([1.0, 2.0, 3.0])],
        [torch.tensor([4.0, 5.0, 6.0])],
    ]
    case_reports, _abort_reason = _runner_attr("_run_precision_cases_with_fingerprint")(
        groups, torch, reference, candidate, "cpu",
        repeats=repeats, overflow_threshold=1.0e4,
    )
    # e4e417fd (2026-08-25): the helper now returns
    # (case_reports, abort_reason); these fingerprint tests only
    # assert on the per-case reports.
    return case_reports


def test_fingerprint_pass_case_observed_once_fail_case_repeated() -> None:
    torch = _torch_or_skip()
    reference = _ScriptedModel([
        torch.tensor([1.0, 2.0, 3.0]),  # case 0
        torch.tensor([4.0, 5.0, 6.0]),  # case 1, first pass
        torch.tensor([4.0, 5.0, 6.0]),  # case 1, rerun 1
        torch.tensor([4.0, 5.0, 6.0]),  # case 1, rerun 2
    ])
    candidate = _ScriptedModel([
        torch.tensor([1.0, 2.0, 3.0]),  # case 0: PASS
        torch.zeros(3),                 # case 1: FAIL, identical across repeats
    ])
    reports = _run_two_case_fingerprint(reference, candidate, repeats=3)

    assert reports[0]["status"] == "PASS"
    assert reports[0]["repeat_fingerprint"]["repeats"] == 1
    assert reports[0]["repeat_fingerprint"]["class"] == "stable-pass"
    assert reports[1]["status"] == "FAIL"
    fingerprint = reports[1]["repeat_fingerprint"]
    assert fingerprint["repeats"] == 3
    assert fingerprint["class"] == "deterministic-fail"
    assert len(fingerprint["observations"]) == 3
    # The quadruple separates NaN from Inf on both sides.
    observation = fingerprint["observations"][0]
    assert observation["candidate_inf_count"] == 0
    assert observation["candidate_nan_count"] == 0
    assert observation["reference_inf_count"] == 0
    assert observation["reference_nan_count"] == 0
    # MERE divides by |golden|+1e-7 with the candidate as golden (the reviewed
    # verifier's argument order): zeros vs [4,5,6] gives (4+5+6)/3 * 1e7.
    assert observation["mere"] == pytest.approx(5.0e7)
    assert observation["reference_nonfinite_mask_sha256"] is not None
    # Reference ran once per observation: 1 (case 0) + 3 (case 1 repeats).
    assert reference.calls == 4
    assert candidate.calls == 4


def test_fingerprint_bimodal_when_candidate_alternates() -> None:
    torch = _torch_or_skip()
    wrong = torch.tensor([100.0, 200.0, 300.0])
    reference = _ScriptedModel([torch.tensor([4.0, 5.0, 6.0])])
    # Drive a single FAIL case whose candidate output alternates.
    groups = [[torch.tensor([4.0, 5.0, 6.0])]]
    candidate = _ScriptedModel([wrong, wrong * 2, wrong])
    reports, _abort_reason = _runner_attr("_run_precision_cases_with_fingerprint")(
        groups, torch, reference, candidate, "cpu",
        repeats=3, overflow_threshold=1.0e4,
    )
    fingerprint = reports[0]["repeat_fingerprint"]
    assert fingerprint["repeats"] == 3
    assert fingerprint["class"] == "bimodal"
    meres = {obs["mere"] for obs in fingerprint["observations"]}
    assert len(meres) == 2


def test_fingerprint_reference_unstable_when_reference_nan_jumps() -> None:
    torch = _torch_or_skip()
    clean = torch.tensor([4.0, 5.0, 6.0])
    poisoned = torch.tensor([4.0, float("nan"), 6.0])
    reference = _ScriptedModel([clean, poisoned, clean])
    candidate = _ScriptedModel([torch.zeros(3)])
    groups = [[torch.tensor([4.0, 5.0, 6.0])]]
    reports, _abort_reason = _runner_attr("_run_precision_cases_with_fingerprint")(
        groups, torch, reference, candidate, "cpu",
        repeats=3, overflow_threshold=1.0e4,
    )
    fingerprint = reports[0]["repeat_fingerprint"]
    assert fingerprint["class"] == "reference-unstable"
    nan_counts = [obs["reference_nan_count"] for obs in fingerprint["observations"]]
    assert nan_counts == [0, 1, 0]
    # The mask sha follows the reference's non-finite footprint.
    shas = {obs["reference_nonfinite_mask_sha256"] for obs in fingerprint["observations"]}
    assert len(shas) == 2


def test_fingerprint_repeats_one_keeps_single_observation() -> None:
    torch = _torch_or_skip()
    reference = _ScriptedModel([torch.tensor([4.0, 5.0, 6.0])])
    candidate = _ScriptedModel([torch.zeros(3)])
    groups = [[torch.tensor([4.0, 5.0, 6.0])]]
    reports, _abort_reason = _runner_attr("_run_precision_cases_with_fingerprint")(
        groups, torch, reference, candidate, "cpu",
        repeats=1, overflow_threshold=1.0e4,
    )
    fingerprint = reports[0]["repeat_fingerprint"]
    assert fingerprint["repeats"] == 1
    assert fingerprint["class"] == "deterministic-fail"
    assert reference.calls == 1


def test_extreme_input_annotates_reference_overflow_risk() -> None:
    torch = _torch_or_skip()
    reference = _ScriptedModel([torch.tensor([1.0]), torch.tensor([1.0])])
    candidate = _ScriptedModel([torch.tensor([1.0]), torch.tensor([1.0])])
    groups = [
        [torch.tensor([1.0, 2.0])],
        [torch.tensor([2.0e5, 1.0])],
    ]
    reports, _abort_reason = _runner_attr("_run_precision_cases_with_fingerprint")(
        groups, torch, reference, candidate, "cpu",
        repeats=1, overflow_threshold=1.0e4,
    )
    assert reports[0]["input_max_abs"] == pytest.approx(2.0)
    assert "reference_overflow_risk" not in reports[0]
    assert reports[1]["input_max_abs"] == pytest.approx(2.0e5)
    assert reports[1]["reference_overflow_risk"] is True


def test_execution_failure_observation_has_none_counts() -> None:
    torch = _torch_or_skip()

    def _exploding_model(*args, **kwargs):
        raise RuntimeError("kernel wedged")

    reference = _ScriptedModel([torch.tensor([1.0])])
    candidate = _exploding_model
    groups = [[torch.tensor([1.0])]]
    reports, _abort_reason = _runner_attr("_run_precision_cases_with_fingerprint")(
        groups, torch, reference, candidate, "cpu",
        repeats=2, overflow_threshold=1.0e4,
    )
    assert reports[0]["status"] == "FAIL"
    assert "case execution failed" in reports[0]["reason"]
    fingerprint = reports[0]["repeat_fingerprint"]
    assert fingerprint["class"] == "deterministic-fail"
    observation = fingerprint["observations"][0]
    assert observation["candidate_inf_count"] is None
    assert observation["reference_inf_count"] == 0


# --- reference-stability gate in the aggregate report -------------------------


def _case_report(index: int, status: str, fingerprint_class: str) -> dict:
    return {
        "case": index,
        "status": status,
        "metrics": {},
        "reason": "" if status == "PASS" else "mismatch",
        "repeat_fingerprint": {
            "repeats": 3,
            "class": fingerprint_class,
            "observations": [],
        },
    }


def test_precision_report_isolates_reference_unstable_cases() -> None:
    binding = {"binding_sha256": "b" * 64, "candidate_tree_sha256": "c" * 64}
    cases = [
        _case_report(0, "PASS", "stable-pass"),
        _case_report(1, "FAIL", "deterministic-fail"),
        _case_report(2, "FAIL", "reference-unstable"),
    ]
    report = _runner_attr("_precision_report")(
        cases,
        binding,
        seed=0,
        seed_events=[],
        input_adapter={},
        device_value="cpu",
        case_count=3,
        repeats=3,
        overflow_threshold=1.0e4,
    )
    # Legacy raw counts are untouched (purely additive contract).
    assert report["status"] == "FAIL"
    assert report["passed_case_count"] == 1
    assert report["failed_case_count"] == 2
    assert report["pass_a"] == {"status": "FAIL", "tier1_pass": 1, "total": 3}
    # The scored view isolates the reference-unstable case from the denominator.
    assert report["reference_unstable_cases"] == [2]
    assert report["scored_case_count"] == 2
    assert report["scored_passed_case_count"] == 1
    assert report["scored_failed_case_count"] == 1
    assert report["precision_repeats"] == 3
    assert report["reference_compute_side"] == "cpu"
    assert report["reference_overflow_threshold"] == 1.0e4


def test_precision_report_marks_device_side_reference() -> None:
    binding = {"binding_sha256": "b" * 64, "candidate_tree_sha256": "c" * 64}
    report = _runner_attr("_precision_report")(
        [_case_report(0, "PASS", "stable-pass")],
        binding,
        seed=0,
        seed_events=[],
        input_adapter={},
        device_value="npu:0",
        case_count=1,
        repeats=1,
        overflow_threshold=1.0e4,
    )
    assert report["reference_compute_side"] == "device"
    assert report["reference_unstable_cases"] == []
    assert report["scored_case_count"] == 1


# --- cross-evaluation determinism alert (NEW-P2) -------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_provider(monkeypatch, manifest: dict) -> None:
    provider = types.ModuleType("npubench.npubench_inputs")

    def verify(workspace: Path, reference: dict):
        assert reference["source"] == "npubench"
        return True, "verified", manifest

    provider.verify_npubench_stage = verify
    monkeypatch.setitem(sys.modules, "npubench.npubench_inputs", provider)


def _workspace(tmp_path: Path, monkeypatch) -> Path:
    workspace = tmp_path / "workspace"
    bundle = workspace / "reference_inputs" / "npubench" / "digest"
    task_dir = bundle / "level1"
    task_dir.mkdir(parents=True)
    task = task_dir / "task.py"
    task.write_text("# placeholder\n", encoding="utf-8")
    sidecar = task.with_suffix(".json")
    sidecar.write_text('{"inputs": []}\n', encoding="utf-8")
    manifest = {
        "schema": "cannbot.npubench.bundle/v1",
        "task_relative_path": str(task.relative_to(bundle)),
        "sidecar_relative_path": str(sidecar.relative_to(bundle)),
        "bundle_sha256": "b" * 64,
    }
    manifest_path = bundle / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    state = {
        "reference": {
            "source": "npubench",
            "bundle_manifest_path": str(manifest_path.relative_to(workspace)),
            "bundle_manifest_sha256": _sha(manifest_path),
            "bundle_sha256": "b" * 64,
            "task_relative_path": manifest["task_relative_path"],
            "sidecar_relative_path": manifest["sidecar_relative_path"],
        }
    }
    (workspace / ".opgen_state.json").write_text(json.dumps(state), encoding="utf-8")
    _install_provider(monkeypatch, manifest)
    return workspace


def _candidate(tmp_path: Path) -> Path:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "model_new_ascendc.py").write_text("# placeholder\n", encoding="utf-8")
    return candidate


def _fake_child_report(context, verdicts: dict) -> dict:
    report = _runner_attr("_base_report")(
        context.verb, status="FAIL", binding=context.binding, run_id=context.run_id
    )
    report.update(
        {
            "case_count": len(verdicts),
            "passed_case_count": sum(status == "PASS" for status in verdicts.values()),
            "failed_case_count": sum(status != "PASS" for status in verdicts.values()),
            "cases": [
                {"case": index, "status": status, "metrics": {}, "reason": ""}
                for index, status in verdicts.items()
            ],
            "child_returncode": 0,
        }
    )
    return report


def _run_fake_precision(workspace: Path, candidate: Path, monkeypatch, verdicts: dict) -> dict:
    monkeypatch.setattr(
        runner,
        "_run_isolated_context",
        lambda context, **_kwargs: _fake_child_report(context, verdicts),
    )
    return runner.run_precision_workspace(workspace, candidate, device=0)


def test_determinism_alert_fires_on_same_binding_verdict_drift(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = _workspace(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path)

    first = _run_fake_precision(workspace, candidate, monkeypatch, {0: "PASS", 1: "FAIL"})
    assert first["determinism_alert"] is False

    second = _run_fake_precision(workspace, candidate, monkeypatch, {0: "PASS", 1: "PASS"})
    assert second["determinism_alert"] is True
    assert second["determinism_alert_detail"]["changed_cases"] == [1]
    assert second["determinism_alert_detail"]["binding_sha256"] == first["binding_sha256"]
    assert second["determinism_alert_detail"]["previous_run_id"] == first["run_id"]
    # The alert is part of both the returned and the persisted report.
    persisted = json.loads(
        (workspace / runner.EVIDENCE_DIRNAME / runner.PRECISION_REPORT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert persisted["determinism_alert"] is True
    assert persisted == second

    history = json.loads(
        (workspace / runner.EVIDENCE_DIRNAME / runner.PRECISION_BINDING_HISTORY_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    entries = history["bindings"][first["binding_sha256"]]
    assert len(entries) == 2
    assert entries[0]["case_verdicts"] == {"0": "PASS", "1": "FAIL"}
    assert entries[1]["case_verdicts"] == {"0": "PASS", "1": "PASS"}

    event_lines = (workspace / "orchestrator_events.jsonl").read_text(encoding="utf-8").splitlines()
    alert_events = [
        json.loads(line)
        for line in event_lines
        if json.loads(line)["event"] == "npubench.precision_determinism_alert"
    ]
    assert len(alert_events) == 1
    assert alert_events[0]["data"]["binding_sha256"] == first["binding_sha256"]
    assert alert_events[0]["data"]["changed_cases"] == [1]
    assert alert_events[0]["data"]["previous_run_id"] == first["run_id"]


def test_determinism_alert_stays_quiet_for_identical_verdicts(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = _workspace(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path)

    first = _run_fake_precision(workspace, candidate, monkeypatch, {0: "FAIL", 1: "FAIL"})
    second = _run_fake_precision(workspace, candidate, monkeypatch, {0: "FAIL", 1: "FAIL"})
    assert first["determinism_alert"] is False
    assert second["determinism_alert"] is False
    assert "determinism_alert_detail" not in second
    assert not (workspace / "orchestrator_events.jsonl").exists()


def test_determinism_history_is_keyed_per_binding(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path)
    other_candidate = tmp_path / "candidate2"
    other_candidate.mkdir()
    (other_candidate / "model_new_ascendc.py").write_text("# different tree\n", encoding="utf-8")

    first = _run_fake_precision(workspace, candidate, monkeypatch, {0: "PASS"})
    # A different candidate tree is a different binding: its first evaluation
    # must not alert against the other binding's history.
    second = _run_fake_precision(workspace, other_candidate, monkeypatch, {0: "FAIL"})
    assert first["binding_sha256"] != second["binding_sha256"]
    assert second["determinism_alert"] is False
