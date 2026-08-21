# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
OPTIMIZE_SCRIPT = ROOT / "skills/catlass-dsl-optimize/scripts/optimize_state.py"
RUN_ID = "run-20260728-000000"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


optimize = load(OPTIMIZE_SCRIPT, "compact_optimize_module")


def run_git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def setup_project(tmp_path, **policy):
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "CATLASS DSL Test"], cwd=root, check=True)
    (root / "src").mkdir()
    kernel = root / "src/operator.py"
    kernel.write_text("TILE = 1\n", encoding="utf-8")
    (root / "bench.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=root, check=True)
    evidence = root / ".catlass-dsl/optimize-runs" / RUN_ID
    evidence.mkdir(parents=True)
    return root.resolve(), evidence.resolve(), kernel.resolve()


def benchmark(path, root, kernel, metric, *, std=0.1, profiling=False, noise=True):
    trace = path.parent / "profiling/candidate"
    workloads = []
    if profiling:
        trace.mkdir(parents=True, exist_ok=True)
        (trace / "kernel_details.csv").write_text(
            "Name,Type,OP State,Duration(us)\nk,k,N/A,10\nk,k,N/A,10\n",
            encoding="utf-8",
        )
        (trace / "step_trace_time.csv").write_text("Computing\n10\n", encoding="utf-8")
        iterations = []
        for index in range(2):
            iteration = trace / "anti_hack" / "iteration-{:04d}".format(index) / "kernel_details.csv"
            iteration.parent.mkdir(parents=True, exist_ok=True)
            iteration.write_text(
                "Name,Type,OP State,Duration(us)\nk,k,N/A,10\n",
                encoding="utf-8",
            )
            iterations.append({
                "iteration": index,
                "kernel_details": str(iteration.relative_to(trace)),
                "sha256": hashlib.sha256(iteration.read_bytes()).hexdigest(),
                "observed_kernel_names": ["k"],
                "observed_launches": 1,
            })
        (trace / "anti_hack_manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "policy": "single-fused-catlass-kernel-v1",
            "profiled_iterations": 2,
            "iterations": iterations,
        }), encoding="utf-8")
        if noise:
            (trace / "trace.json").write_text("{}", encoding="utf-8")
            (trace / "huge.bin").write_bytes(b"noise")
        workloads = [{
            "uuid": "case-1",
            "anti_hack": {
                "status": "passed",
                "policy": "single-fused-catlass-kernel-v1",
                "declared_kernel_names": ["k"],
                "observed_kernel_names": ["k"],
                "profiled_iterations": 2,
                "observed_launches": 2,
                "launches_per_iteration": 1.0,
                "reason": None,
            },
            "performance": {"candidate": {"trace_dir": str(trace)}},
        }]
    return write_json(path, {
        "status": "passed",
        "solution": {"source_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest()},
        "environment": {"config": str(root / "bench.json"), "device": "npu:0" if profiling else "cpu", "arch": "test", "catlass": "test", "cann": "test", "warmup": 1, "trials": 2},
        "anti_hack": ({
            "status": "passed",
            "policy": "single-fused-catlass-kernel-v1",
            "declared_kernel_names": ["k"],
            "observed_kernel_names": ["k"],
            "profiled_iterations": 2,
            "observed_launches": 2,
            "launches_per_iteration": 1.0,
            "reason": None,
        } if profiling else {"status": "not_applicable"}),
        "correctness": {"status": "passed"},
        "performance": {"status": "passed", "candidate": {"mean_ms": metric, "std_ms": std}},
        "profiling": {"status": "passed" if profiling else "not_run"},
        "workloads": workloads,
    })


def initialize(tmp_path, **policy):
    root, evidence, kernel = setup_project(tmp_path, **policy)
    baseline = benchmark(evidence / "baseline/benchmark/result.json", root, kernel, 10.0, profiling=policy.get("profiling") == "yes")
    correctness = evidence / "baseline/correctness/full-test.txt"
    correctness.parent.mkdir(parents=True)
    correctness.write_text("passed\n", encoding="utf-8")
    state = evidence / "state.json"
    initialize_policy = {
        "threshold": policy.get("threshold", 8.0),
        "stall_threshold": policy.get("stall", 3),
        "min_improvement_fraction": 0.03,
        "profiling_required": policy.get("profiling", "no"),
        "required_commands": ["full-test"],
    }
    if "maximum" in policy:
        initialize_policy["max_iterations"] = policy["maximum"]
    result = optimize.initialize(
        state, RUN_ID, root, kernel, baseline, ["full-test={}".format(correctness)], 0,
        **initialize_policy,
    )
    return root, evidence, kernel, state, result


def proposal(path, axis="tile", profiles=None):
    return write_json(path, {
        "hypothesis": "change tile structure",
        "axis_id": axis,
        "expected_effect": "reduce mean latency by five percent",
        "falsification_condition": "mean latency does not improve",
        "knowledge_sources": ["knowledge/optimization/tiling.md"] if not profiles else [],
        "profile_evidence": profiles or [],
    })


def begin(state, evidence, revision, axis="tile", profiles=None):
    return optimize.begin_round(state, revision, proposal(evidence / "proposal-input.json", axis, profiles))


def submit_pass(root, evidence, kernel, state, active, metric, profiling=False):
    trace = Path(active["trace_dir"])
    correctness = trace / "correctness/full-test.txt"
    correctness.parent.mkdir(parents=True)
    correctness.write_text("passed\n", encoding="utf-8")
    bench = benchmark(trace / "benchmark/result.json", root, kernel, metric, profiling=profiling)
    submission = write_json(trace / "submission.json", {
        "round": active["round"], "correctness_status": "passed",
        "command_results": {"full-test": {"status": "passed", "evidence": str(correctness)}},
        "benchmark_result": str(bench),
    })
    return optimize.record_round(state, json.loads(Path(state).read_text())["revision"], submission)


def test_initialize_creates_compact_baseline_and_ako_summary(tmp_path):
    _root, evidence, kernel, state, result = initialize(tmp_path)
    assert result["best_correct"]["kernel_sha256"] == hashlib.sha256(kernel.read_bytes()).hexdigest()
    assert json.loads(state.read_text())["policy"]["max_iterations"] == 30
    assert sorted(path.name for path in (evidence / "baseline").iterdir()) == ["kernel.py", "result.json"]
    assert state == evidence / "state.json"
    text = (evidence / "ITERATIONS.md").read_text()
    assert "# Iteration Log" in text and "| Iter | Title | Score | Passed | Notes |" in text
    assert not (evidence / "worktrees").exists()
    assert not (evidence / "state.json.lock").exists()


def test_initialize_rejects_npu_benchmark_without_anti_hack(tmp_path):
    root, evidence, kernel = setup_project(tmp_path)
    baseline = benchmark(
        evidence / "baseline/benchmark/result.json",
        root,
        kernel,
        10.0,
        profiling=True,
    )
    payload = json.loads(baseline.read_text())
    payload.pop("anti_hack")
    baseline.write_text(json.dumps(payload), encoding="utf-8")
    correctness = evidence / "baseline/correctness/full-test.txt"
    correctness.parent.mkdir(parents=True)
    correctness.write_text("passed\n", encoding="utf-8")
    with pytest.raises(optimize.OptimizeError, match="anti_hack.status=passed"):
        optimize.initialize(
            evidence / "state.json",
            RUN_ID,
            root,
            kernel,
            baseline,
            ["full-test={}".format(correctness)],
            0,
            profiling_required="yes",
            required_commands=["full-test"],
        )


def test_initialize_rejects_kernel_outside_repository(tmp_path):
    root, evidence, kernel = setup_project(tmp_path)
    baseline = benchmark(evidence / "baseline/benchmark/result.json", root, kernel, 10.0)
    correctness = evidence / "baseline/correctness/full-test.txt"
    correctness.parent.mkdir(parents=True)
    correctness.write_text("passed\n")
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n")
    with pytest.raises(optimize.OptimizeError, match="越出 repository_root"):
        optimize.initialize(evidence / "state.json", RUN_ID, root, outside, baseline, ["full-test={}".format(correctness)], 0, threshold=8.0)


def test_accepted_round_uses_kernel_hash_and_compacts_trace(tmp_path):
    root, evidence, kernel, state, initial = initialize(tmp_path, profiling="yes")
    started = begin(state, evidence, initial["revision"])
    active = started["active_round"]
    assert not (evidence / "proposal-input.json").exists()
    kernel.write_text("TILE = 2\n", encoding="utf-8")
    recorded = submit_pass(root, evidence, kernel, state, active, 8.0, profiling=True)
    trace = Path(recorded["iterations"][0]["kernel_snapshot"]).parent
    assert recorded["best_correct"]["kernel_sha256"] == hashlib.sha256(kernel.read_bytes()).hexdigest()
    assert {path.name for path in trace.iterdir()} == {"kernel.py", "proposal.json", "result.json", "profile"}
    profile_files = {path.name for path in (trace / "profile/case-0000").iterdir()}
    assert profile_files == {"kernel_details.csv", "step_trace_time.csv"}
    assert not (trace / "profile/case-0000/anti_hack_manifest.json").exists()
    assert not (trace / "profile/case-0000/anti_hack").exists()
    assert not list(trace.glob("**/trace.json")) and not list(trace.glob("**/huge.bin"))


def test_rejected_and_failed_rounds_restore_best(tmp_path):
    root, evidence, kernel, state, initial = initialize(tmp_path)
    started = begin(state, evidence, initial["revision"])
    kernel.write_text("TILE = 2\n")
    rejected = submit_pass(root, evidence, kernel, state, started["active_round"], 9.9)
    assert kernel.read_text() == "TILE = 1\n"
    started = begin(state, evidence, rejected["revision"], axis="pipeline")
    kernel.write_text("BROKEN = True\n")
    trace = Path(started["active_round"]["trace_dir"])
    submission = write_json(trace / "submission.json", {"round": 2, "correctness_status": "failed", "command_results": {}, "benchmark_result": None})
    failed = optimize.record_round(state, started["revision"], submission)
    assert kernel.read_text() == "TILE = 1\n"
    assert Path(failed["iterations"][1]["kernel_snapshot"]).read_text() == "BROKEN = True\n"


def test_stall_reprofiles_repeatedly_without_stopping_iteration(tmp_path):
    root, evidence, kernel, state, current = initialize(
        tmp_path, stall=2, maximum=5
    )

    for round_number, axis in ((1, "tile"), (2, "pipeline")):
        started = begin(state, evidence, current["revision"], axis=axis)
        kernel.write_text("TILE = {}\n".format(round_number + 1))
        current = submit_pass(
            root, evidence, kernel, state, started["active_round"], 9.9
        )

    assert current["next_action"] == "reprofile"
    assert current["reprofile_required"] is True
    assert current["status"] == "running"

    first_profile = evidence / "fresh-profile-1.csv"
    first_profile.write_text("profile 1\n")
    started = begin(
        state,
        evidence,
        current["revision"],
        axis="vectorize",
        profiles=[str(first_profile)],
    )
    assert started["stall_count"] == 0
    kernel.write_text("TILE = 4\n")
    current = submit_pass(
        root, evidence, kernel, state, started["active_round"], 9.9
    )
    assert current["next_action"] == "begin_round"
    assert current["status"] == "running"

    started = begin(state, evidence, current["revision"], axis="layout")
    kernel.write_text("TILE = 5\n")
    current = submit_pass(
        root, evidence, kernel, state, started["active_round"], 9.9
    )
    assert current["next_action"] == "reprofile"
    assert current["reprofile_required"] is True
    assert current["status"] == "running"

    second_profile = evidence / "fresh-profile-2.csv"
    second_profile.write_text("profile 2\n")
    started = begin(
        state,
        evidence,
        current["revision"],
        axis="prefetch",
        profiles=[str(second_profile)],
    )
    kernel.write_text("TILE = 6\n")
    current = submit_pass(
        root, evidence, kernel, state, started["active_round"], 9.9
    )
    assert current["next_action"] == "begin_finalization"
    assert current["status"] == "failed"
    assert len(current["iterations"]) == 5


def test_round_rejects_changes_outside_the_single_kernel(tmp_path):
    root, evidence, kernel, state, initial = initialize(tmp_path)
    started = begin(state, evidence, initial["revision"])
    kernel.write_text("TILE = 2\n")
    (root / "bench.json").write_text('{"changed": true}\n')
    trace = Path(started["active_round"]["trace_dir"])
    submission = write_json(trace / "submission.json", {
        "round": 1, "correctness_status": "failed", "command_results": {}, "benchmark_result": None,
    })
    with pytest.raises(optimize.OptimizeError, match="越出唯一 kernel"):
        optimize.record_round(state, started["revision"], submission)


def test_profile_whitelist_allows_one_csv_and_rejects_none(tmp_path):
    destination = tmp_path / "compact"
    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "kernel_details.csv").write_text("Duration(us)\n1\n")
    benchmark_value = {"workloads": [{"uuid": "x", "performance": {"candidate": {"trace_dir": str(trace)}}}]}
    records = optimize._compact_profiles(benchmark_value, destination, True)
    assert records[0]["kernel_details.csv"] and records[0]["step_trace_time.csv"] is None
    (trace / "kernel_details.csv").unlink()
    with pytest.raises(optimize.OptimizeError, match="profiling 未生成"):
        optimize._compact_profiles(benchmark_value, tmp_path / "empty", True)


def test_profile_evidence_rejects_two_plus_zero_iteration_manifest(tmp_path):
    root, evidence, kernel = setup_project(tmp_path)
    result_path = benchmark(
        evidence / "benchmark/result.json", root, kernel, 10.0, profiling=True
    )
    result = json.loads(result_path.read_text())
    trace = Path(result["workloads"][0]["performance"]["candidate"]["trace_dir"])
    manifest_path = trace / "anti_hack_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    first = trace / manifest["iterations"][0]["kernel_details"]
    second = trace / manifest["iterations"][1]["kernel_details"]
    first.write_text(
        "Name,Type,OP State,Duration(us)\nk,k,N/A,10\nk,k,N/A,10\n",
        encoding="utf-8",
    )
    second.write_text("Name,Type,OP State,Duration(us)\n", encoding="utf-8")
    for entry, path in zip(manifest["iterations"], (first, second)):
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["iterations"][0]["observed_launches"] = 2
    manifest["iterations"][1]["observed_launches"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(optimize.OptimizeError, match="每个 iteration 必须恰好一次"):
        optimize._profile_kernel_evidence(
            trace, result["workloads"][0]["anti_hack"], "NPU workload 0"
        )


def test_profile_evidence_rejects_duplicate_iteration_fragment_reference(tmp_path):
    root, evidence, kernel = setup_project(tmp_path)
    result_path = benchmark(
        evidence / "benchmark/result.json", root, kernel, 10.0, profiling=True
    )
    result = json.loads(result_path.read_text())
    trace = Path(result["workloads"][0]["performance"]["candidate"]["trace_dir"])
    manifest_path = trace / "anti_hack_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["iterations"][1]["kernel_details"] = manifest["iterations"][0]["kernel_details"]
    manifest["iterations"][1]["sha256"] = manifest["iterations"][0]["sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(optimize.OptimizeError, match="路径非 canonical"):
        optimize._profile_kernel_evidence(
            trace, result["workloads"][0]["anti_hack"], "NPU workload 0"
        )


def test_finalization_leaves_best_uncommitted_and_writes_synthesis(tmp_path):
    root, evidence, kernel, state, initial = initialize(tmp_path, threshold=9.0)
    started = begin(state, evidence, initial["revision"])
    kernel.write_text("TILE = 2\n")
    recorded = submit_pass(root, evidence, kernel, state, started["active_round"], 8.0)
    final_started = optimize.begin_finalization(state, recorded["revision"])
    final_dir = Path(final_started["finalization"]["evidence_dir"])
    correctness = final_dir / "correctness/full-test.txt"
    correctness.parent.mkdir(parents=True)
    correctness.write_text("passed\n")
    final_bench = benchmark(final_dir / "benchmark/result.json", root, kernel, 8.1)
    submission = write_json(final_dir / "submission.json", {
        "command_results": {"full-test": {"status": "passed", "evidence": str(correctness)}},
        "benchmark_result": str(final_bench), "remaining_bottlenecks": "memory latency", "next_steps": "try wider tiles",
    })
    finished = optimize.finalize(state, final_started["revision"], submission)
    assert finished["status"] == "passed"
    assert finished["best_kernel_path"] == str(kernel)
    assert finished["best_kernel_sha256"] == hashlib.sha256(kernel.read_bytes()).hexdigest()
    assert sorted(path.name for path in final_dir.iterdir()) == ["kernel.py", "result.json"]
    assert run_git(root, "status", "--porcelain")
    summary = (evidence / "ITERATIONS.md").read_text()
    assert "Session synthesis" in summary and "memory latency" in summary and "try wider tiles" in summary


def test_status_rejects_non_current_schema(tmp_path):
    state = write_json(tmp_path / "old.json", {"schema": "catlass.dsl.optimization.v3"})
    with pytest.raises(optimize.OptimizeError, match="state schema 非法"):
        optimize.status(state)


def test_state_freezes_configuration_and_detects_tampering(tmp_path):
    _root, _evidence, _kernel, state_path, result = initialize(tmp_path)
    assert result["config_digest"]
    state = json.loads(state_path.read_text())
    assert state["required_commands"] == ["full-test"]
    state["policy"]["threshold"] = 123.0
    state_path.write_text(json.dumps(state))
    with pytest.raises(optimize.OptimizeError, match="配置摘要"):
        optimize.status(state_path)


def test_controller_cli_has_no_contract_input():
    help_text = OPTIMIZE_SCRIPT.read_text(encoding="utf-8")
    skill_text = (ROOT / "skills/catlass-dsl-optimize/SKILL.md").read_text(encoding="utf-8")
    assert "--contract" not in help_text
    assert "CONTRACT.md" not in help_text
    assert "--contract" not in skill_text
    assert "CONTRACT.md" not in skill_text


def test_source_contains_no_git_candidate_lifecycle(tmp_path):
    source = OPTIMIZE_SCRIPT.read_text(encoding="utf-8")
    assert "candidate_commit" not in source
    assert "candidate_worktree" not in source
    assert "optimization-state.json" not in source
    assert "optimization-journal.md" not in source
    assert "git worktree" not in source
    assert "git commit" not in source
