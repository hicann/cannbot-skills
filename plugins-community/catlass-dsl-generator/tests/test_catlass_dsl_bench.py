# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/catlass-dsl-bench/scripts/bench.py"
spec = importlib.util.spec_from_file_location("catlass_dsl_bench", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def write_suite(
    root,
    *,
    solution_body="def run(x, y, alpha):\n    return (x + y) * alpha\n",
    dps=False,
    tolerance=None,
):
    reference = (
        "import torch\n\n"
        "def run(x, y, alpha):\n"
        "    return torch.add(x, y) * alpha\n"
    )
    definition = root / "definition.json"
    definition.write_text(
        json.dumps(
            {
                "name": "scaled_add",
                "op_type": "elementwise",
                "axes": {"N": {"type": "var"}},
                "inputs": {
                    "x": {"shape": ["N"], "dtype": "float32"},
                    "y": {"shape": ["N"], "dtype": "float32"},
                    "alpha": {"shape": None, "dtype": "float32"},
                },
                "outputs": {
                    "output": {"shape": ["N"], "dtype": "float32"}
                },
                "reference": reference,
            }
        ),
        encoding="utf-8",
    )
    default_tolerance = {
        "max_atol": 0.0,
        "max_rtol": 0.0,
        "required_matched_ratio": 1.0,
        "max_error_cap": None,
        "allow_negative_inf": False,
    }
    default_tolerance.update(tolerance or {})
    workload = root / "workload.jsonl"
    workload.write_text(
        "\n".join(
            json.dumps(
                {
                    "uuid": case_id,
                    "axes": {"N": size},
                    "inputs": {
                        "x": {"type": "random"},
                        "y": {"type": "random"},
                        "alpha": {"type": "scalar", "value": 0.5},
                    },
                    "tolerance": default_tolerance,
                }
            )
            for case_id, size in (("main", 16), ("tail", 13))
        )
        + "\n",
        encoding="utf-8",
    )
    solution = root / "solution.json"
    solution.write_text(
        json.dumps(
            {
                "name": "scaled_add_solution",
                "definition": "scaled_add",
                "author": "test",
                "spec": {
                    "languages": ["python"],
                    "target_hardware": ["LOCAL"],
                    "entry_point": "solution.py::run",
                    "destination_passing_style": dps,
                },
                "sources": [
                    {"path": "solution.py", "content": solution_body}
                ],
            }
        ),
        encoding="utf-8",
    )
    return solution, workload, definition


def test_loads_solution_workload_and_definition(tmp_path):
    solution, workload, definition = write_suite(tmp_path)
    loaded_definition = bench.load_definition(definition)
    loaded_solution = bench.load_solution(solution)
    loaded_workloads, path, digest = bench.load_workloads(workload)

    assert loaded_solution["definition"] == loaded_definition["name"]
    assert [item["uuid"] for item in loaded_workloads] == ["main", "tail"]
    assert path == str(workload.resolve())
    assert len(digest) == 64


def test_suite_runs_all_workloads_and_analyzes_timing(tmp_path):
    solution, workload, definition = write_suite(tmp_path)
    result = bench.run_suite(
        solution,
        workload,
        definition,
        tmp_path / "output",
        warmup=1,
        trials=3,
    )

    assert result["status"] == "passed"
    assert result["schema_version"] == 3
    assert result["correctness"] == {"status": "passed", "passed": 2, "total": 2}
    assert result["performance"]["candidate"]["mean_ms"] > 0
    assert result["performance"]["candidate"]["std_ms"] >= 0
    source = json.loads(solution.read_text())["sources"][0]["content"]
    assert result["solution"]["source_sha256"] == hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()
    assert len(result["performance"]["candidate"]["workload_mean_ms"]) == 2
    assert all(
        len(item["performance"]["candidate"]["trials"]) == 3
        for item in result["workloads"]
    )
    assert Path(result["artifacts"]["result_json"]).is_file()
    assert Path(result["artifacts"]["sources"]).is_dir()
    assert result["anti_hack"]["status"] == "not_applicable"
    assert all(
        item["anti_hack"]["status"] == "not_applicable"
        for item in result["workloads"]
    )


def test_benchmark_defaults_to_one_warmup_and_two_trials(tmp_path):
    solution, workload, definition = write_suite(tmp_path)

    result = bench.run_suite(
        solution, workload, definition, tmp_path / "output"
    )

    assert result["environment"]["warmup"] == 1
    assert result["environment"]["trials"] == 2
    assert all(
        len(item["performance"]["candidate"]["trials"]) == 2
        for item in result["workloads"]
    )


def test_parses_profiler_step_trace_as_per_iteration_kernel_time(tmp_path):
    trace = tmp_path / "worker" / "ASCEND_PROFILER_OUTPUT"
    trace.mkdir(parents=True)
    (trace / "step_trace_time.csv").write_text(
        "Device_id,Step,Computing,Free\n0,,250.0,10.0\n",
        encoding="utf-8",
    )

    mean_ms, source = bench.parse_profile_kernel_mean_ms(tmp_path, trials=5)

    assert mean_ms == pytest.approx(0.05)
    assert source == "step_trace_time.Computing"


def test_profiler_kernel_details_is_timing_fallback(tmp_path):
    trace = tmp_path / "worker" / "ASCEND_PROFILER_OUTPUT"
    trace.mkdir(parents=True)
    (trace / "kernel_details.csv").write_text(
        "Name,Duration(us)\nkernel_a,10.0\nkernel_b,30.0\n",
        encoding="utf-8",
    )

    mean_ms, source = bench.parse_profile_kernel_mean_ms(tmp_path, trials=2)

    assert mean_ms == pytest.approx(0.02)
    assert source == "kernel_details.Duration"


def _audit_solution(body, entry="solution.py::run"):
    return bench.audit_solution_sources(
        {
            "spec": {"entry_point": entry},
            "sources": [{"path": "solution.py", "content": body}],
        }
    )


def _valid_catlass_wrapper(run_body):
    return (
        "import catlass as tla\n"
        "import torch\n\n"
        "@tla.kernel\n"
        "def fused_kernel(x, out):\n"
        "    pass\n\n"
        "def run(x):\n"
        + "".join("    " + line + "\n" for line in run_body.splitlines())
    )


def test_source_audit_allows_metadata_and_empty_allocation():
    audit = _audit_solution(
        _valid_catlass_wrapper(
            "shape = x.shape\n"
            "valid = x.is_contiguous() and x.dim() > 0\n"
            "out = torch.empty_like(x)\n"
            "return out"
        )
    )
    assert audit["status"] == "passed"
    assert audit["declared_kernel_names"] == ["fused_kernel"]


@pytest.mark.parametrize(
    "body, operation",
    [
        (
            _valid_catlass_wrapper("return torch.add(x, x)"),
            "torch.add",
        ),
        (
            "from torch import matmul as mm\n"
            + _valid_catlass_wrapper("return mm(x, x)"),
            "torch.matmul",
        ),
        (
            _valid_catlass_wrapper("mm = torch.matmul\nreturn mm(x, x)"),
            "torch.matmul",
        ),
        (
            "import torch.nn.functional as F\n"
            + _valid_catlass_wrapper("return F.gelu(x)"),
            "torch.nn.functional.gelu",
        ),
        (
            _valid_catlass_wrapper("return torch.ops.aten.add.Tensor(x, x)"),
            "torch.ops.aten.add.Tensor",
        ),
        (_valid_catlass_wrapper("return x.matmul(x)"), "Tensor.matmul"),
        (_valid_catlass_wrapper("return x.T"), "Tensor.T"),
        (_valid_catlass_wrapper("return x + x"), "Add"),
    ],
)
def test_source_audit_rejects_torch_computation(body, operation):
    audit = _audit_solution(body)
    assert audit["status"] == "failed"
    assert audit["reason"] == "torch_computation"
    assert operation in {item["operation"] for item in audit["source_violations"]}


def test_source_audit_tracks_tensor_into_helper():
    body = (
        "import catlass as tla\n\n"
        "@tla.kernel\n"
        "def fused_kernel(x, out):\n"
        "    pass\n\n"
        "def helper(value):\n"
        "    return value.sin()\n\n"
        "def run(x):\n"
        "    return helper(x)\n"
    )
    audit = _audit_solution(body)
    assert audit["status"] == "failed"
    assert any(
        item["operation"] == "Tensor.sin"
        for item in audit["source_violations"]
    )


def test_source_audit_allows_multiple_declared_dispatch_targets():
    body = (
        "import catlass as tla\n\n"
        "@tla.kernel\n"
        "def fp16_kernel(x, out):\n"
        "    pass\n\n"
        "@tla.kernel\n"
        "def bf16_kernel(x, out):\n"
        "    pass\n\n"
        "def run(x):\n"
        "    return launch(x)\n"
    )
    audit = _audit_solution(body)
    assert audit["status"] == "passed"
    assert audit["declared_kernel_names"] == ["bf16_kernel", "fp16_kernel"]


def _write_kernel_details(root, names):
    trace = root / "worker" / "ASCEND_PROFILER_OUTPUT"
    trace.mkdir(parents=True)
    (trace / "kernel_details.csv").write_text(
        "Name,Type,OP State,Duration(us)\n"
        + "".join("{0},{0},N/A,10.0\n".format(name) for name in names),
        encoding="utf-8",
    )


def test_profile_anti_hack_accepts_one_declared_kernel_per_trial(tmp_path):
    _write_kernel_details(tmp_path, ["fp16_kernel", "fp16_kernel"])
    audit = bench.parse_profile_anti_hack(
        tmp_path, trials=2, declared_kernel_names=["fp16_kernel", "bf16_kernel"]
    )
    assert audit["status"] == "passed"
    assert audit["observed_launches"] == 2
    assert audit["launches_per_iteration"] == 1.0


def test_profile_anti_hack_rejects_two_stage_trace(tmp_path):
    _write_kernel_details(
        tmp_path,
        [
            "projection_gelu_kernel",
            "second_projection_kernel",
            "projection_gelu_kernel",
            "second_projection_kernel",
        ],
    )
    with pytest.raises(bench.BenchError) as raised:
        bench.parse_profile_anti_hack(
            tmp_path,
            trials=2,
            declared_kernel_names=[
                "projection_gelu_kernel",
                "second_projection_kernel",
            ],
        )
    assert raised.value.category == "hack"
    assert raised.value.details["reason"] == "launch_count_mismatch"
    assert raised.value.details["launches_per_iteration"] == 2.0


@pytest.mark.parametrize(
    "names, declared, reason",
    [
        ([], ["fused_kernel"], "launch_count_mismatch"),
        (["vendor_add", "vendor_add"], ["fused_kernel"], "kernel_not_declared_catlass"),
        (["a", "b"], ["a", "b"], "multiple_kernel_names"),
    ],
)
def test_profile_anti_hack_rejects_unverifiable_launches(
    tmp_path, names, declared, reason
):
    _write_kernel_details(tmp_path, names)
    with pytest.raises(bench.BenchError) as raised:
        bench.parse_profile_anti_hack(tmp_path, trials=2, declared_kernel_names=declared)
    assert raised.value.category == "hack"
    assert raised.value.details["reason"] == reason


def test_profile_anti_hack_requires_kernel_details(tmp_path):
    with pytest.raises(bench.BenchError) as raised:
        bench.parse_profile_anti_hack(
            tmp_path, trials=2, declared_kernel_names=["fused_kernel"]
        )
    assert raised.value.category == "hack"
    assert raised.value.details["reason"] == "kernel_details_file_count"


def test_profile_anti_hack_rejects_vendor_kernel_spoofing_declared_name(tmp_path):
    trace = tmp_path / "worker" / "ASCEND_PROFILER_OUTPUT"
    trace.mkdir(parents=True)
    (trace / "kernel_details.csv").write_text(
        "Name,Type,OP State,Duration(us)\n"
        "fused_kernel,MatMulV3,dynamic,10.0\n",
        encoding="utf-8",
    )
    with pytest.raises(bench.BenchError) as raised:
        bench.parse_profile_anti_hack(
            tmp_path, trials=1, declared_kernel_names=["fused_kernel"]
        )
    assert raised.value.category == "hack"
    assert raised.value.details["reason"] == "kernel_not_catlass_profile"


def test_npu_reference_profile_is_reused_across_run_directories(tmp_path, monkeypatch):
    solution, workload, definition = write_suite(tmp_path)
    original_generate_inputs = bench.generate_inputs
    calls = {"candidate": 0, "reference": 0}
    candidate_invocations = 0

    monkeypatch.setattr(bench, "configure_device", lambda device, torch: lambda: None)
    monkeypatch.setattr(bench, "_device_name", lambda device, torch: "fake-npu")
    monkeypatch.setattr(
        bench,
        "generate_inputs",
        lambda definition, workload, reference_module, device, seed, workload_root: (
            original_generate_inputs(
                definition,
                workload,
                reference_module,
                "cpu",
                seed,
                workload_root,
            )
        ),
    )

    monkeypatch.setattr(
        bench,
        "audit_solution_sources",
        lambda solution: {
            "status": "passed",
            "policy": bench.ANTI_HACK_POLICY,
            "declared_kernel_names": ["fake_kernel"],
            "observed_kernel_names": [],
            "profiled_iterations": 0,
            "observed_launches": 0,
            "launches_per_iteration": None,
            "reason": None,
            "source_violations": [],
        },
    )
    original_stage_solution = bench.stage_solution

    def tracked_stage_solution(solution, output):
        run, sources = original_stage_solution(solution, output)

        def tracked_run(*args):
            nonlocal candidate_invocations
            candidate_invocations += 1
            return run(*args)

        return tracked_run, sources

    monkeypatch.setattr(bench, "stage_solution", tracked_stage_solution)

    def fake_measure_npu(
        function,
        inputs,
        warmup,
        trials,
        synchronize,
        trace_dir,
        declared_kernel_names=None,
    ):
        kind = "reference" if "shared-reference" in str(trace_dir) else "candidate"
        calls[kind] += 1
        result = {
            "mean_ms": 2.0 if kind == "reference" else 1.0,
            "median_ms": 1.0,
            "min_ms": 1.0,
            "p95_ms": 1.0,
            "std_ms": 0.0,
            "trials": [1.0],
            "profiled_iterations": trials,
            "measurement_source": "fake-profiler",
            "trace_dir": str(Path(trace_dir).resolve()),
        }
        if declared_kernel_names is not None:
            result["anti_hack"] = {
                "status": "passed",
                "policy": bench.ANTI_HACK_POLICY,
                "declared_kernel_names": list(declared_kernel_names),
                "observed_kernel_names": ["fake_kernel"],
                "profiled_iterations": trials,
                "observed_launches": trials,
                "launches_per_iteration": 1.0,
                "reason": None,
            }
            result["_last_output"] = function(*inputs)
        return result

    monkeypatch.setattr(bench, "measure_npu", fake_measure_npu)
    cache = tmp_path / "shared-reference"
    first = bench.run_suite(
        solution,
        workload,
        definition,
        tmp_path / "develop-run",
        device="npu:0",
        warmup=0,
        trials=1,
        reference_profile_cache=cache,
    )
    second = bench.run_suite(
        solution,
        workload,
        definition,
        tmp_path / "optimize-run",
        device="npu:0",
        warmup=0,
        trials=1,
        reference_profile_cache=cache,
    )

    assert first["status"] == second["status"] == "passed"
    assert first["profiling"]["reference"]["cache_status"] == "collected"
    assert second["profiling"]["reference"]["cache_status"] == "reused"
    assert first["anti_hack"]["status"] == second["anti_hack"]["status"] == "passed"
    assert calls == {"candidate": 4, "reference": 2}
    assert candidate_invocations == 4
    assert Path(second["artifacts"]["reference_profile"], "manifest.json").is_file()


def test_npu_source_hack_fails_before_profiling(tmp_path, monkeypatch):
    solution, workload, definition = write_suite(tmp_path)
    monkeypatch.setattr(bench, "configure_device", lambda device, torch: lambda: None)
    monkeypatch.setattr(bench, "_device_name", lambda device, torch: "fake-npu")
    result = bench.run_suite(
        solution,
        workload,
        definition,
        tmp_path / "output",
        device="npu:0",
        warmup=0,
        trials=1,
    )
    assert result["status"] == "failed"
    assert result["error"]["category"] == "hack"
    assert result["anti_hack"]["reason"] == "torch_computation"
    assert result["workloads"] == []


def test_dynamic_anti_hack_failure_discards_performance(tmp_path, monkeypatch):
    solution, workload, definition = write_suite(tmp_path)
    original_generate_inputs = bench.generate_inputs
    monkeypatch.setattr(bench, "configure_device", lambda device, torch: lambda: None)
    monkeypatch.setattr(bench, "_device_name", lambda device, torch: "fake-npu")
    monkeypatch.setattr(
        bench,
        "generate_inputs",
        lambda definition, workload, reference_module, device, seed, workload_root: (
            original_generate_inputs(
                definition,
                workload,
                reference_module,
                "cpu",
                seed,
                workload_root,
            )
        ),
    )
    monkeypatch.setattr(
        bench,
        "audit_solution_sources",
        lambda solution: {
            "status": "passed",
            "policy": bench.ANTI_HACK_POLICY,
            "declared_kernel_names": ["fused_kernel"],
            "observed_kernel_names": [],
            "profiled_iterations": 0,
            "observed_launches": 0,
            "launches_per_iteration": None,
            "reason": None,
            "source_violations": [],
        },
    )

    candidate_calls = 0

    def reject_second_candidate(
        function,
        inputs,
        warmup,
        trials,
        synchronize,
        trace_dir,
        declared_kernel_names=None,
    ):
        nonlocal candidate_calls
        if declared_kernel_names is None:
            return {
                "mean_ms": 2.0,
                "median_ms": 2.0,
                "min_ms": 2.0,
                "p95_ms": 2.0,
                "std_ms": 0.0,
                "trials": [2.0] * trials,
                "profiled_iterations": trials,
                "measurement_source": "fake-profiler",
                "trace_dir": str(Path(trace_dir).resolve()),
            }
        assert declared_kernel_names == ["fused_kernel"]
        candidate_calls += 1
        if candidate_calls == 1:
            return {
                "mean_ms": 1.0,
                "median_ms": 1.0,
                "min_ms": 1.0,
                "p95_ms": 1.0,
                "std_ms": 0.0,
                "trials": [1.0] * trials,
                "profiled_iterations": trials,
                "measurement_source": "fake-profiler",
                "trace_dir": str(Path(trace_dir).resolve()),
                "anti_hack": {
                    "status": "passed",
                    "policy": bench.ANTI_HACK_POLICY,
                    "declared_kernel_names": ["fused_kernel"],
                    "observed_kernel_names": ["fused_kernel"],
                    "profiled_iterations": trials,
                    "observed_launches": trials,
                    "launches_per_iteration": 1.0,
                    "reason": None,
                },
                "_last_output": function(*inputs),
            }
        details = {
            "status": "failed",
            "policy": bench.ANTI_HACK_POLICY,
            "declared_kernel_names": ["fused_kernel"],
            "observed_kernel_names": ["stage_a", "stage_b"],
            "profiled_iterations": trials,
            "observed_launches": trials * 2,
            "launches_per_iteration": 2.0,
            "reason": "launch_count_mismatch",
        }
        raise bench.BenchError("hack", "two stages", details=details)

    monkeypatch.setattr(bench, "measure_npu", reject_second_candidate)
    result = bench.run_suite(
        solution,
        workload,
        definition,
        tmp_path / "output",
        device="npu:0",
        warmup=0,
        trials=2,
        reference_profile_cache=tmp_path / "reference-cache",
    )
    assert result["status"] == "failed"
    assert result["anti_hack"]["status"] == "failed"
    assert result["anti_hack"]["reason"] == "launch_count_mismatch"
    assert result["performance"]["status"] == "not_run"
    assert all(item["performance"]["status"] == "not_run" for item in result["workloads"])
    assert result["workloads"][0]["anti_hack"]["status"] == "passed"
    assert result["workloads"][1]["error"]["category"] == "hack"


def test_incorrect_solution_has_no_performance_claim(tmp_path):
    solution, workload, definition = write_suite(
        tmp_path,
        solution_body="def run(x, y, alpha):\n    return (x - y) * alpha\n",
    )
    result = bench.run_suite(
        solution, workload, definition, tmp_path / "output", warmup=0, trials=1
    )

    assert result["status"] == "failed"
    assert result["performance"]["status"] == "not_run"
    assert all(
        item["performance"]["status"] == "not_run"
        for item in result["workloads"]
    )


def test_matched_ratio_and_error_cap_follow_workload(tmp_path):
    body = (
        "import torch\n\n"
        "def run(x, y, alpha):\n"
        "    error = (torch.arange(x.numel()) == 0).to(x.dtype)\n"
        "    return (x + y) * alpha + error\n"
    )
    solution, workload, definition = write_suite(
        tmp_path,
        solution_body=body,
        tolerance={"required_matched_ratio": 0.9},
    )
    accepted = bench.run_suite(
        solution, workload, definition, tmp_path / "accepted", warmup=0, trials=1
    )
    assert accepted["status"] == "passed"

    rows = [json.loads(line) for line in workload.read_text().splitlines()]
    for row in rows:
        row["tolerance"]["max_error_cap"] = 0.5
    workload.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    rejected = bench.run_suite(
        solution, workload, definition, tmp_path / "rejected", warmup=0, trials=1
    )
    assert rejected["status"] == "failed"


def test_destination_passing_solution(tmp_path):
    solution, workload, definition = write_suite(
        tmp_path,
        dps=True,
        solution_body=(
            "def run(x, y, alpha, output):\n"
            "    output.copy_((x + y) * alpha)\n"
        ),
    )
    result = bench.run_suite(
        solution, workload, definition, tmp_path / "output", warmup=0, trials=1
    )
    assert result["status"] == "passed"


def test_custom_inputs_use_definition_entrypoint(tmp_path):
    source = (
        "import torch\n\n"
        "def make_inputs(axes, device):\n"
        "    return {'x': torch.ones(axes['N'], device=device)}\n\n"
        "def run(x):\n"
        "    return x + 1\n"
    )
    definition = tmp_path / "definition.json"
    definition.write_text(
        json.dumps(
            {
                "name": "custom",
                "axes": {"N": {"type": "var"}},
                "custom_inputs_entrypoint": "make_inputs",
                "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
                "outputs": {"output": {"shape": ["N"], "dtype": "float32"}},
                "reference": source,
            }
        ),
        encoding="utf-8",
    )
    workload = tmp_path / "workload.jsonl"
    workload.write_text(
        json.dumps(
            {
                "uuid": "custom-1",
                "axes": {"N": 4},
                "inputs": {"x": {"type": "custom"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    solution = tmp_path / "solution.json"
    solution.write_text(
        json.dumps(
            {
                "name": "custom_solution",
                "definition": "custom",
                "author": "test",
                "spec": {
                    "languages": ["python"],
                    "target_hardware": ["LOCAL"],
                    "entry_point": "solution.py::run",
                    "destination_passing_style": False,
                },
                "sources": [
                    {
                        "path": "solution.py",
                        "content": "def run(x):\n    return x + 1\n",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = bench.run_suite(
        solution, workload, definition, tmp_path / "output", warmup=0, trials=1
    )
    assert result["status"] == "passed"


def test_invalid_schema_is_persisted_as_configuration_failure(tmp_path):
    solution, workload, definition = write_suite(tmp_path)
    data = json.loads(solution.read_text())
    data["spec"]["entry_point"] = "../solution.py::run"
    solution.write_text(json.dumps(data), encoding="utf-8")

    result = bench.run_suite(
        solution, workload, definition, tmp_path / "output", warmup=0, trials=1
    )
    assert result["status"] == "failed"
    assert result["error"]["category"] == "configuration"
    assert (tmp_path / "output/result.json").is_file()


def test_cli_uses_only_three_public_inputs(tmp_path):
    solution, workload, definition = write_suite(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--solution",
            str(solution),
            "--workload",
            str(workload),
            "--definition",
            str(definition),
            "--output",
            str(tmp_path / "output"),
            "--warmup",
            "0",
            "--trials",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    help_text = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "--solution" in help_text
    assert "--workload" in help_text
    assert "--definition" in help_text
    assert "--candidate" not in help_text
    assert "--config" not in help_text


def test_templates_execute_and_runner_has_no_multiprocessing(tmp_path):
    templates = ROOT / "skills/catlass-dsl-bench/templates"
    result = bench.run_suite(
        templates / "solution.json",
        templates / "workload.jsonl",
        templates / "definition.json",
        tmp_path / "output",
        warmup=0,
        trials=1,
    )
    assert result["status"] == "passed"
    source = SCRIPT.read_text(encoding="utf-8")
    assert "multiprocessing" not in source
    assert "_spawn_worker" not in source
    assert "load_config" not in source
