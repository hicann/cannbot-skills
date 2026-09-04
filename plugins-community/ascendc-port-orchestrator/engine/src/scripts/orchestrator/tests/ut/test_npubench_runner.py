# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Hermetic unit coverage for the native old-format NPUKernelBench runner."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import _reorg_paths  # noqa: F401  (stable sys.path setup for reorganized tests)
import pytest

from npubench import npubench_runner as runner


@pytest.fixture(autouse=True)
def _local_runner_configuration(tmp_path: Path, monkeypatch) -> None:
    """Keep runner unit tests local unless a test explicitly configures a target."""
    config = tmp_path / ".ascendc_env"
    config.write_text("A5_CONTAINER=local\n", encoding="utf-8")
    monkeypatch.setenv("ASCENDC_ENV_PATH", str(config))
    for key in ("A5_CONTAINER", "A5_NPU_PYTHON_BIN", "NPU_PYTHON_BIN", "A5_HOST_PYTHON"):
        monkeypatch.delenv(key, raising=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_provider(monkeypatch, manifest: dict) -> None:
    provider = types.ModuleType("npubench.npubench_inputs")

    def verify(workspace: Path, reference: dict):
        assert reference["source"] == "npubench"
        return True, "verified", manifest

    provider.verify_npubench_stage = verify
    monkeypatch.setitem(sys.modules, "npubench.npubench_inputs", provider)


def _workspace(
    tmp_path: Path,
    monkeypatch,
    *,
    package: bool = False,
    relative: bool = False,
    sidecar_suffix: str = ".json",
) -> tuple[Path, Path, Path]:
    workspace = tmp_path.joinpath("workspace")
    bundle = workspace.joinpath("reference_inputs", "npubench", "digest")
    task_dir = bundle.joinpath("pkg" if package else "level1")
    task_dir.mkdir(parents=True)
    if package:
        package_files = {"__init__.py": "", "helper.py": "VALUE = 7\n"}
        for filename, contents in package_files.items():
            (task_dir / filename).write_text(contents, encoding="utf-8")
    task = task_dir.joinpath("task.py")
    source = [
        "from .helper import VALUE\n" if relative else "",
        "class Model:\n",
        "    def __init__(self, *args): pass\n",
        "    def state_dict(self): return {}\n",
        "    def load_state_dict(self, state, strict=True): return None\n",
        "    def to(self, device): return self\n",
        "    def __call__(self, *args, **kwargs): return args\n",
        "def get_input_groups(): return [[1]]\n",
        "def get_init_inputs(): return []\n",
    ]
    task.write_text("".join(source), encoding="utf-8")
    sidecar = task.with_suffix(sidecar_suffix) if sidecar_suffix.startswith(".") else task.with_name(
        f"{task.stem}{sidecar_suffix}"
    )
    # Default is deliberately JSONL under a .json name, as in level1/3_Add.json.
    sidecar.write_text('{"inputs": []}\n{"inputs": []}\n', encoding="utf-8")
    manifest = dict(
        schema="cannbot.npubench.bundle/v1",
        task_relative_path=str(task.relative_to(bundle)),
        sidecar_relative_path=str(sidecar.relative_to(bundle)),
        bundle_sha256="b" * 64,
    )
    manifest_path = bundle.joinpath("bundle_manifest.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    reference = dict(
        source="npubench",
        bundle_manifest_path=str(manifest_path.relative_to(workspace)),
        bundle_manifest_sha256=_sha(manifest_path),
        bundle_sha256="b" * 64,
        task_relative_path=manifest["task_relative_path"],
        sidecar_relative_path=manifest["sidecar_relative_path"],
    )
    state = {"reference": reference}
    (workspace / ".opgen_state.json").write_text(json.dumps(state), encoding="utf-8")
    _install_provider(monkeypatch, manifest)
    return workspace, bundle, task


def _candidate(tmp_path: Path) -> Path:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "model_new_ascendc.py").write_text(
        "class ModelNew:\n"
        "    def __init__(self, *args): pass\n"
        "    def state_dict(self): return {}\n"
        "    def load_state_dict(self, state, strict=True): return None\n"
        "    def to(self, device): return self\n"
        "    def __call__(self, *args, **kwargs): return args\n",
        encoding="utf-8",
    )
    return candidate


def _make_read_only_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix(), reverse=True):
        if path.is_dir():
            os.chmod(path, 0o500)
        else:
            os.chmod(path, 0o400)
    os.chmod(root, 0o500)


def test_preflight_reads_staged_binding_and_jsonl_named_json(tmp_path: Path, monkeypatch) -> None:
    workspace, _, task = _workspace(tmp_path, monkeypatch)

    report = runner.preflight_workspace(workspace, isolated=False)

    assert report["status"] == "PASS"
    assert report["sidecar_encoding"] == "jsonl"
    assert report["case_count"] == 2
    assert Path(report["task_path"]) == task
    assert (workspace / "npubench_evidence" / "preflight_report.json").is_file()


def test_sidecar_descriptor_adapter_accepts_native_task_and_preserves_order(tmp_path: Path, monkeypatch) -> None:
    """The canonical native task has no Python input provider; its JSONL sidecar is authoritative."""
    from npubench.npubench_runner import _resolve_input_groups
    torch = _torch_or_skip()
    workspace, _bundle, task = _workspace(tmp_path, monkeypatch)
    task.write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "class Model(nn.Module):\n"
        "    def forward(self, x, y, alpha=1.0): return x + alpha * y\n",
        encoding="utf-8",
    )
    sidecar = task.with_suffix(".json")
    sidecar.write_text(
        json.dumps({"inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [2, 3]},
            {"name": "y", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 3]},
            {"name": "alpha", "type": "attr", "required": False, "dtype": "float", "value": -0.5},
        ]}) + "\n",
        encoding="utf-8",
    )
    bundle = runner.resolve_staged_bundle(workspace)
    module = runner.load_task_module(task, bundle.root, role="reference")
    groups, identity = _resolve_input_groups(module, bundle=bundle, torch=torch, seed=7)
    assert identity["kind"] == runner.SIDECAR_DESCRIPTOR_ADAPTER
    assert [tuple(item.shape) for item in groups[0][:2]] == [(2, 3), (1, 3)]
    assert [item.dtype for item in groups[0][:2]] == [torch.float16, torch.float16]
    assert groups[0][2] == -0.5


def test_sidecar_descriptor_adapter_rejects_integer_and_bool_dtypes(tmp_path: Path, monkeypatch) -> None:
    from npubench.npubench_runner import _resolve_input_groups
    torch = _torch_or_skip()
    workspace, _bundle, task = _workspace(tmp_path, monkeypatch)
    task.write_text("class Model: pass\n", encoding="utf-8")
    task.with_suffix(".json").write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "name": "i", "type": "tensor", "required": True,
                        "dtype": "int32", "shape": [4], "range": [3, 5],
                    },
                    {"name": "flag", "type": "tensor", "required": True, "dtype": "bool", "shape": [4]},
                    {
                        "name": "ratio", "type": "tensor", "required": True,
                        "dtype": "float32", "shape": [4], "range": [-0.5, 0.5],
                    },
                    {"name": "label", "type": "attr", "required": False, "dtype": "str", "value": "ok"},
                ]
            }
        ) + "\n",
        encoding="utf-8",
    )
    bundle = runner.resolve_staged_bundle(workspace)
    module = runner.load_task_module(task, bundle.root, role="reference")
    with pytest.raises(runner.NpuBenchRunnerError, match="unsupported dtype"):
        _resolve_input_groups(module, bundle=bundle, torch=torch, seed=11)


@pytest.mark.parametrize(
    "descriptor, message",
    [
        ({"name": "x", "type": "tensor", "required": False, "dtype": "float32", "shape": [1]}, "invalid fields"),
        ({"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [0]}, "shape is invalid"),
        (
            {"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [1], "range": [2, 1]},
            "range is invalid",
        ),
        (
            {"name": "x", "type": "tensor", "required": True, "dtype": "bool", "shape": [1], "range": [0, 1]},
            "unsupported dtype",
        ),
        ({"name": "alpha", "type": "attr", "required": False, "dtype": "float"}, "invalid fields"),
    ],
)
def test_sidecar_descriptor_adapter_rejects_ambiguous_or_invalid_inputs(
    tmp_path: Path, monkeypatch, descriptor, message
) -> None:
    from npubench.npubench_runner import _resolve_input_groups
    torch = _torch_or_skip()
    workspace, _bundle, task = _workspace(tmp_path, monkeypatch)
    task.write_text("class Model: pass\n", encoding="utf-8")
    task.with_suffix(".json").write_text(json.dumps({"inputs": [descriptor]}) + "\n", encoding="utf-8")
    bundle = runner.resolve_staged_bundle(workspace)
    module = runner.load_task_module(task, bundle.root, role="reference")
    with pytest.raises(runner.NpuBenchRunnerError, match=message):
        _resolve_input_groups(module, bundle=bundle, torch=torch, seed=0)


def test_sidecar_descriptor_adapter_rejects_unknown_type_before_allocation(tmp_path: Path, monkeypatch) -> None:
    from npubench.npubench_runner import _resolve_input_groups
    torch = _torch_or_skip()
    workspace, _bundle, task = _workspace(tmp_path, monkeypatch)
    task.write_text("class Model: pass\n", encoding="utf-8")
    task.with_suffix(".json").write_text(
        json.dumps({"inputs": [{"name": "x", "type": "sequence", "shape": [2]}]}) + "\n",
        encoding="utf-8",
    )
    bundle = runner.resolve_staged_bundle(workspace)
    module = runner.load_task_module(task, bundle.root, role="reference")
    with pytest.raises(runner.NpuBenchRunnerError, match="unsupported type"):
        _resolve_input_groups(module, bundle=bundle, torch=torch, seed=0)


def test_native_task_binding_identifies_sidecar_adapter(tmp_path: Path, monkeypatch) -> None:
    from npubench.npubench_runner import _canonical_sha256, _input_adapter_contract
    workspace, _bundle, task = _workspace(tmp_path, monkeypatch)
    task.write_text("class Model: pass\n", encoding="utf-8")
    binding = runner.build_evaluation_binding(workspace)
    assert binding["input_adapter"] == {
        "kind": runner.SIDECAR_DESCRIPTOR_ADAPTER,
        "provider": None,
        "contract": runner.INPUT_ADAPTER_CONTRACT_VERSION,
        "schema": runner.SIDECAR_DESCRIPTOR_SCHEMA,
        "case_count": 2,
    }
    assert binding["input_adapter_contract_sha256"] == _canonical_sha256(
        _input_adapter_contract()
    )


def test_input_adapter_binding_ignores_nested_or_conditional_provider_names(tmp_path: Path, monkeypatch) -> None:
    """Only module exports may affect a binding built without task execution."""
    workspace, _bundle, task = _workspace(tmp_path, monkeypatch)
    task.write_text(
        "class Model: pass\n"
        "def wrapper():\n"
        "    def get_input_groups(): return [[1]]\n"
        "    return get_input_groups\n"
        "class Helpers:\n"
        "    def get_inputs(self): return [1]\n"
        "if False:\n"
        "    def get_inputs(): return [1]\n",
        encoding="utf-8",
    )

    binding = runner.build_evaluation_binding(workspace)

    assert binding["input_adapter"]["kind"] == runner.SIDECAR_DESCRIPTOR_ADAPTER
    assert binding["input_adapter"]["case_count"] == 2


def test_real_stage_bind_and_child_preflight_accepts_rebasable_manifest_path(tmp_path: Path, monkeypatch) -> None:
    """Exercise P1's actual portable/rebasable binding, not a fake provider."""
    from npubench import npubench_inputs as inputs

    source = tmp_path / "source"
    source.mkdir()
    task = source / "3_Add.py"
    task.write_text(
        "class Model:\n"
        "    def __init__(self, *args): pass\n"
        "def get_input_groups(): return [[1]]\n",
        encoding="utf-8",
    )
    task.with_suffix(".json").write_text('{"inputs": []}\n', encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    stage = inputs.stage_npubench_inputs(workspace, npubench_task=task)
    state = inputs.bind_npubench_state({"op": "add"}, stage)
    inputs.atomic_write_state(workspace, state)
    # The copied runner must bring its verifier along.  A production child has
    # no guarantee that the controller's source tree is on PYTHONPATH.
    monkeypatch.delenv("PYTHONPATH", raising=False)

    # The production path is a portable independent child process; it must
    # work without an optional host-specific mount-isolation backend.
    report = runner.preflight_workspace(workspace)

    assert report["status"] == "PASS"
    assert report["execution_isolation"] == "process_boundary"
    assert report["tamper_protection"] == "post_run_hash_check"
    assert not Path(state["reference"]["bundle_manifest_path"]).is_absolute()


def test_bundled_native_example_stages_preflights_and_loads_cases(tmp_path: Path) -> None:
    """Keep the documented old-format fixture executable as native input."""
    torch = pytest.importorskip("torch")
    from npubench import npubench_inputs as inputs

    plugin_root = Path(__file__).resolve().parents[6]
    task = plugin_root / "examples" / "npukernelbench-native" / "level1" / "example_add.py"
    root = task.parent.parent
    workspace = tmp_path / "workspace"
    stage = inputs.stage_npubench_inputs(workspace, npubench_task=task, npubench_root=root)
    state = inputs.bind_npubench_state({"op": "add"}, stage)
    inputs.atomic_write_state(workspace, state)

    report = runner.preflight_workspace(workspace, isolated=False)
    assert report["status"] == "PASS"
    assert report["sidecar_encoding"] == "jsonl"
    assert report["case_count"] == 2

    module = runner.load_task_module(stage.task_path, stage.root, role="reference")
    groups = module.get_input_groups()
    assert len(groups) == 2
    assert tuple(groups[0][0].shape) == (2, 3)
    assert tuple(groups[0][1].shape) == (1, 3)
    assert groups[1][2] == -0.5
    assert torch.equal(module.Model()(*groups[0]), groups[0][0] + groups[0][1])


def test_copied_runner_imports_sibling_provider_without_pythonpath(tmp_path: Path, monkeypatch) -> None:
    """The copied public runner must carry its P1 provider into a clean child."""
    from npubench.npubench_runner import (
        _cleanup_execution_context,
        _create_execution_context,
        _scrubbed_task_environment,
    )
    workspace, _, _ = _workspace(tmp_path, monkeypatch)
    bundle = runner.resolve_staged_bundle(workspace)
    binding = runner.build_evaluation_binding(workspace, bundle=bundle)
    context = _create_execution_context(
        workspace, bundle=bundle, candidate_dir=None, binding=binding, verb="preflight"
    )
    source = tmp_path / "source"
    source.mkdir()
    task = source / "task.py"
    task.write_text("VALUE = 1\n", encoding="utf-8")
    task.with_suffix(".json").write_text('{"inputs": []}\n', encoding="utf-8")
    staged_workspace = tmp_path / "staged-by-copied-runner"
    staged_workspace.mkdir()
    environment = _scrubbed_task_environment()
    environment.pop("PYTHONPATH", None)
    try:
        completed = subprocess.run(
            [
                str(Path(sys.executable).resolve()),
                str(context.runner_root / "npubench_runner.py"),
                "stage",
                "--workspace",
                str(staged_workspace),
                "--task",
                str(task),
                "--root",
                str(source),
            ],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
    finally:
        _cleanup_execution_context(context)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "PASS"


@pytest.mark.parametrize("verb", ("preflight", "precision", "fixture"))
def test_non_performance_context_does_not_resolve_profiler(
    tmp_path: Path, monkeypatch, verb: str
) -> None:
    """O2.5 must not depend on the optional repository profiler checkout."""
    from npubench.npubench_runner import _cleanup_execution_context, _create_execution_context
    workspace, _, _ = _workspace(tmp_path, monkeypatch)
    bundle = runner.resolve_staged_bundle(workspace)
    binding = runner.build_evaluation_binding(workspace, bundle=bundle)

    def unexpected_profiler_lookup():
        raise AssertionError("profiler lookup is performance-only")

    monkeypatch.setattr(runner, "_default_profiler_summary", unexpected_profiler_lookup)
    context = _create_execution_context(
        workspace, bundle=bundle, candidate_dir=None, binding=binding, verb=verb
    )
    try:
        request = json.loads(context.request_path.read_text(encoding="utf-8"))
        assert request["profiler_script"] is None
    finally:
        _cleanup_execution_context(context)


def test_loader_preserves_real_file_and_supports_package_relative_import(tmp_path: Path, monkeypatch) -> None:
    workspace, bundle, task = _workspace(tmp_path, monkeypatch, package=True, relative=True)
    del workspace

    module = runner.load_task_module(task, bundle, role="reference")

    assert Path(module.__file__).resolve() == task.resolve()
    assert module.VALUE == 7


def test_loader_fails_closed_for_relative_import_outside_package(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    task = root / "task.py"
    task.write_text("from .helper import value\n", encoding="utf-8")

    with pytest.raises(runner.NpuBenchRunnerError, match="relative import"):
        runner.load_task_module(task, root)


def test_binding_recomputes_candidate_tree_and_detects_change(tmp_path: Path, monkeypatch) -> None:
    workspace, _, _ = _workspace(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path)

    first = runner.build_evaluation_binding(workspace, candidate)
    (candidate / "extra.txt").write_text("changed", encoding="utf-8")
    second = runner.build_evaluation_binding(workspace, candidate)

    assert first["candidate_tree_sha256"] != second["candidate_tree_sha256"]
    assert first["binding_sha256"] != second["binding_sha256"]
    assert first["candidate_entry_sha256"] == second["candidate_entry_sha256"]


def test_candidate_snapshot_stays_finalizer_bindable_after_o5_publish(
    tmp_path: Path,
) -> None:
    import phase_o5

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "model_new_ascendc.py").write_text("VALUE = 1\n", encoding="utf-8")
    (workspace / ".opgen_state.json").write_text("{}\n", encoding="utf-8")
    (workspace / "verification.json").write_text('{"worker": true}\n', encoding="utf-8")
    (workspace / "reference_inputs").mkdir()
    (workspace / "reference_inputs" / "truth.bin").write_bytes(b"not candidate")

    snapshot = runner.materialize_candidate_snapshot(workspace)
    before = runner.tree_sha256(snapshot)
    (workspace / "npubench_evidence").mkdir()
    (workspace / "npubench_evidence" / "precision_report.json").write_text("{}\n", encoding="utf-8")
    report = phase_o5.O5Report(verdict="VERIFIED", truth_source="npubench")
    from phase_o5 import _persist_npubench_verification

    _persist_npubench_verification(
        workspace,
        precision={"status": "PASS", "pass_a": {"status": "PASS"}},
        performance={"status": "PASS", "warm_up": 3, "repeats": 5, "keep_prof": True},
        evaluate={"binding_sha256": "a" * 64},
        evidence={},
        report=report,
    )

    assert (snapshot / "model_new_ascendc.py").is_file()
    assert not (snapshot / "reference_inputs").exists()
    assert not (snapshot / ".opgen_state.json").exists()
    assert not (snapshot / "verification.json").exists()
    assert runner.tree_sha256(snapshot) == before
    # The finalizer recomputes the candidate scope from ``workspace``.  Its
    # digest must still identify the snapshot after O5 publishes this file.
    assert runner.candidate_tree_sha256(workspace) == snapshot.name
    assert snapshot.stat().st_mode & 0o222 == 0


def test_performance_plan_is_quick_w3_r5_keep_prof(tmp_path: Path) -> None:
    script = tmp_path / "msprof_perf_summary.py"
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    command = runner.build_performance_command(
        adapter, device=2, run_id="run-123", profiler_script=script
    )

    assert command[:3] == [sys.executable, str(script), "--quick"]
    assert ["--warmup", "3"] == command[3:5]
    assert ["--device", "2"] == command[5:7]
    assert "--keep-prof" in command
    assert ["--repeats", "5"] == command[8:10]
    assert command[-2:] == ["--prof-tag", "run-123"]
    assert "--native-input-manifest" not in command


def test_default_profiler_is_packaged_with_marketplace_runtime() -> None:
    """The performance dependency must not require a repository checkout."""
    from npubench.npubench_runner import _default_profiler_summary
    profiler = _default_profiler_summary()

    assert profiler.name == "msprof_perf_summary.py"
    assert profiler.parent.name == "vendor"
    assert profiler.is_file()
    assert not profiler.is_symlink()


def _load_module_from_path(path: Path, name: str):
    """Import ``path`` under ``name`` the way the shared profiler loads model modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_native_quick_shim_accepts_input_from_separately_loaded_reference_module(
    tmp_path: Path, monkeypatch
) -> None:
    """The shared profiler loads ``model.py`` twice under different module names."""
    from npubench.npubench_runner import _materialize_native_perf_fixture, _prepare_native_quick_adapter
    torch = _torch_or_skip()
    workspace, bundle, task = _workspace(tmp_path, monkeypatch, sidecar_suffix="_perf_cases.jsonl")
    task.write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "class Model(nn.Module):\n"
        "    def __init__(self, scale):\n"
        "        super().__init__()\n"
        "        self.scale = nn.Parameter(torch.tensor(scale))\n"
        "    def forward(self, x): return x * self.scale\n"
        "def get_input_groups(): return [[torch.tensor([3.0])]]\n"
        "def get_init_inputs(): return [2.0]\n",
        encoding="utf-8",
    )
    candidate = _candidate(tmp_path)
    candidate_entry = candidate / "model_new_ascendc.py"
    candidate_entry.write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "class ModelNew(nn.Module):\n"
        "    def __init__(self, factor):\n"
        "        super().__init__()\n"
        "        self.factor = nn.Parameter(torch.tensor(factor))\n"
        "    def forward(self, x): return x * self.factor\n"
        "def get_init_inputs(): return [4.0]\n",
        encoding="utf-8",
    )
    binding = runner.build_evaluation_binding(workspace, candidate)
    adapter = runner.prepare_adapter_view(
        workspace, candidate, bundle=runner.resolve_staged_bundle(workspace), binding=binding, run_id="shim-test"
    )
    _materialize_native_perf_fixture(
        adapter, runner.resolve_staged_bundle(workspace), candidate, binding=binding, seed=0
    )
    _prepare_native_quick_adapter(adapter, adapter, binding=binding)
    assert not (adapter / f"{task.stem}_perf_cases.jsonl").exists()
    assert (adapter / "npubench_perf_cases.jsonl").is_file()
    fixture = torch.load(
        adapter / runner.NATIVE_PERF_FIXTURE_FILENAME, map_location="cpu", weights_only=True
    )
    assert fixture["init_values"] == [4.0]
    assert "reference_state_dict" not in fixture

    first_reference_load = _load_module_from_path(adapter / "model.py", "shim_reference_inputs")
    separately_loaded_candidate = _load_module_from_path(
        adapter / "model_new_ascendc.py", "shim_candidate_model"
    )
    frozen_input = first_reference_load.get_input_groups()[0][0]

    assert torch.equal(separately_loaded_candidate.ModelNew()(frozen_input), torch.tensor([12.0]))


def test_evaluate_parallel_only_for_two_valid_distinct_leases(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    candidate = _candidate(tmp_path)
    binding = {"binding_sha256": "a" * 64, "source": "npubench"}
    calls: list[str] = []

    def precision(*args, **kwargs):
        calls.append("precision")
        return {
            "schema": "cannbot.npubench.precision/v1",
            "status": "PASS",
            "binding_sha256": binding["binding_sha256"],
            "evaluation_binding": binding,
        }

    def performance(*args, **kwargs):
        calls.append("performance")
        return {
            "schema": "cannbot.npubench.performance/v1",
            "status": "PASS",
            "binding_sha256": binding["binding_sha256"],
            "evaluation_binding": binding,
        }

    leases = tmp_path / "leases.json"
    leases.write_text(
        json.dumps(
            {
                "leases": {
                    "precision": {"device": 0, "token": "p"},
                    "performance": {"device": 1, "token": "q"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "run_precision_workspace", precision)
    monkeypatch.setattr(runner, "run_performance_workspace", performance)

    report = runner.evaluate_workspace(
        workspace,
        candidate,
        precision_device=0,
        performance_device=1,
        lease_manifest=leases,
    )

    assert report["status"] == "PASS"
    assert report["parallelism"] == "parallel_two_lane"
    assert sorted(calls) == ["performance", "precision"]


def test_evaluate_single_lane_keeps_one_binding_for_both_reports(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    candidate = _candidate(tmp_path)
    binding = {"binding_sha256": "b" * 64, "source": "npubench"}

    monkeypatch.setattr(
        runner,
        "run_precision_workspace",
        lambda *args, **kwargs: {
            "status": "PASS",
            "binding_sha256": binding["binding_sha256"],
            "evaluation_binding": binding,
        },
    )
    monkeypatch.setattr(
        runner,
        "run_performance_workspace",
        lambda *args, **kwargs: {
            "status": "PASS",
            "binding_sha256": binding["binding_sha256"],
            "evaluation_binding": binding,
        },
    )

    report = runner.evaluate_workspace(
        workspace, candidate, precision_device=0, performance_device=0
    )

    assert report["status"] == "PASS"
    assert report["parallelism"] == "degraded_single_lane"
    assert report["precision"]["binding_sha256"] == report["performance"]["binding_sha256"]


def test_archive_retained_profiles_is_durable_and_missing_profile_fails(tmp_path: Path) -> None:
    from npubench.npubench_runner import _archive_retained_profiles, _expected_profile_dir
    workspace = tmp_path / "workspace"
    adapter = tmp_path / "adapter"
    workspace.mkdir()
    profile_root = tmp_path / "prof"
    run_id = "run-1"
    ref = _expected_profile_dir(profile_root, run_id, 0, "ref") / "PROF_ref"
    asc = _expected_profile_dir(profile_root, run_id, 0, "asc") / "PROF_asc"
    for raw in (ref, asc):
        raw.mkdir(parents=True)
        (raw / "task_time.csv").write_text("duration\n1\n", encoding="utf-8")
    summary = {
        "profiling_mode": "quick",
        "gate_eligible": True,
        "warmup": 3,
        "repeats": 5,
        "profiler_artifacts_retained": True,
        "per_case": [{"case": 0, "ref_prof_dir": str(ref), "asc_prof_dir": str(asc)}],
    }

    relative, digest = _archive_retained_profiles(
        workspace,
        adapter,
        run_id,
        summary,
        expected_valid_case_indices=[0],
        profile_root=profile_root,
        source_profile_root=profile_root,
    )
    archive = workspace / relative

    assert archive.is_dir()
    assert runner.profile_tree_sha256(archive) == digest
    with pytest.raises(runner.NpuBenchRunnerError, match="ref_prof_dir is invalid"):
        _archive_retained_profiles(
            workspace,
            adapter,
            "run-2",
            {**summary, "per_case": [{"case": 0, "ref_prof_dir": None, "asc_prof_dir": None}]},
            expected_valid_case_indices=[0],
            profile_root=profile_root,
            source_profile_root=profile_root,
        )


def test_strict_performance_rejects_unpinned_custom_profiler(tmp_path: Path, monkeypatch) -> None:
    workspace, _, _ = _workspace(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path)
    custom = tmp_path / "custom_profiler.sh"
    custom.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    report = runner.run_performance_workspace(
        workspace, candidate, device=0, profiler_script=custom
    )

    assert report["status"] == "ERROR"
    assert "requires the repository" in report["reason"]


def test_runner_has_no_a3_provider_import() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "import phase_o25_a3" not in source
    assert "from phase_o25_a3" not in source


def _torch_or_skip():
    """Keep the runner's filesystem/contract UT runnable without PyTorch."""
    return pytest.importorskip("torch")


def test_precision_semantics_source_is_pinned_in_contract() -> None:
    from npubench.npubench_runner import _precision_contract
    contract = _precision_contract()

    assert contract["semantics_source"] == runner.PRECISION_SEMANTICS_SOURCE
    assert contract["semantics_source"]["sha256"] == (
        "855649269af085ee34d093375bf67567b2c1d936ab9253977a163a29f4b6e9a9"
    )
    assert contract["semantics_source"]["reviewed_range"] == "62-873"
    assert contract["integer_lsb_tolerance"] == {"int8": 1, "int16": 1}


def test_precision_quantized_integer_accepts_only_plus_minus_one_lsb() -> None:
    torch = _torch_or_skip()
    reference = torch.tensor([5, -5], dtype=torch.int8)

    passed, metrics, reason = runner.compare_outputs(
        reference,
        torch.tensor([6, -6], dtype=torch.int8),
        torch,
        input_type="float",
        input_dtype=torch.float32,
    )

    assert passed, reason
    assert metrics["kind"] == "integer_quantized"
    assert metrics["lsb_tolerance"] == 1

    failed, failed_metrics, reason = runner.compare_outputs(
        reference,
        torch.tensor([7, -5], dtype=torch.int8),
        torch,
        input_type="float",
        input_dtype=torch.float32,
    )

    assert not failed
    assert failed_metrics["violation_count"] == 1
    assert "±1 LSB" in reason


def test_precision_integer_int8_int16_lsb_and_float_dtype_cast() -> None:
    torch = _torch_or_skip()
    int_reference = torch.tensor([100, -100], dtype=torch.int16)

    passed, metrics, reason = runner.compare_outputs(
        int_reference,
        torch.tensor([101, -99], dtype=torch.int16),
        torch,
        input_type="int",
        input_dtype=torch.int16,
    )

    assert passed, reason
    assert metrics["kind"] == "integer_lsb"
    assert metrics["lsb_tolerance"] == 1

    failed, _metrics, _reason = runner.compare_outputs(
        int_reference,
        torch.tensor([102, -100], dtype=torch.int16),
        torch,
        input_type="int",
        input_dtype=torch.int16,
    )
    assert not failed

    # The reviewed verifier permits float output dtype differences and casts
    # candidate finite values to the reference dtype before its three checks.
    float_reference = torch.tensor([1.0, -2.0], dtype=torch.float32)
    cast_ok, cast_metrics, cast_reason = runner.compare_outputs(
        float_reference,
        torch.tensor([1.0, -2.0], dtype=torch.float16),
        torch,
        input_type="float",
        input_dtype=torch.float32,
    )
    assert cast_ok, cast_reason
    assert cast_metrics["kind"] == "floating"
    assert cast_metrics["dtype"] == "float32"


def test_precision_complex_and_nan_inf_masks_follow_verifier_contract() -> None:
    torch = _torch_or_skip()
    complex_reference = torch.tensor([1.0 + 2.0j, -3.0 + 0.5j], dtype=torch.complex64)

    complex_ok, complex_metrics, complex_reason = runner.compare_outputs(
        complex_reference,
        torch.tensor([1.0 + 2.0j, -3.0 + 0.5j], dtype=torch.complex64),
        torch,
        input_type="float",
        input_dtype=torch.float32,
    )

    assert complex_ok, complex_reason
    assert complex_metrics["kind"] == "complex"
    assert complex_metrics["checks"]["real"]
    assert complex_metrics["checks"]["imag"]

    reference = torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0])
    nan_inf_ok, _metrics, nan_inf_reason = runner.compare_outputs(
        reference,
        torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0]),
        torch,
        input_type="float",
        input_dtype=torch.float32,
    )
    assert nan_inf_ok, nan_inf_reason

    sign_failed, _metrics, sign_reason = runner.compare_outputs(
        reference,
        torch.tensor([float("nan"), float("inf"), float("inf"), 1.0]),
        torch,
        input_type="float",
        input_dtype=torch.float32,
    )
    assert not sign_failed
    assert "Inf sign mismatch" in sign_reason


def test_precision_recurses_nested_outputs_and_ignores_mapping_insertion_order() -> None:
    torch = _torch_or_skip()
    reference = {
        "float": torch.tensor([1.0], dtype=torch.float32),
        "nested": [torch.tensor([2], dtype=torch.int8), ("label",)],
    }
    candidate = {
        "nested": [torch.tensor([3], dtype=torch.int8), ("label",)],
        "float": torch.tensor([1.0], dtype=torch.float32),
    }

    passed, metrics, reason = runner.compare_outputs(
        reference,
        candidate,
        torch,
        input_type="float",
        input_dtype=torch.float32,
    )

    assert passed, reason
    assert set(metrics["children"]) == {"float", "nested"}

    failed, _metrics, failed_reason = runner.compare_outputs(
        reference,
        {"other": torch.tensor([1.0], dtype=torch.float32)},
        torch,
        input_type="float",
        input_dtype=torch.float32,
    )
    assert not failed
    assert "mapping keys differ" in failed_reason


def test_scrubbed_child_environment_removes_a3_and_proxy(monkeypatch) -> None:
    from npubench.npubench_runner import _scrubbed_task_environment
    monkeypatch.setenv("A3_HOST", "private")
    monkeypatch.setenv("CANNBOT_A3_TOKEN", "private")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy")

    environment = _scrubbed_task_environment()

    assert "A3_HOST" not in environment
    assert "CANNBOT_A3_TOKEN" not in environment
    assert "HTTPS_PROXY" not in environment
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_normal_isolated_child_needs_no_os_sandbox_backend(tmp_path: Path, monkeypatch) -> None:
    from npubench.npubench_runner import (
        _base_report, _cleanup_execution_context, _create_execution_context, _run_isolated_context,
    )
    for key in ("A5_NPU_PYTHON_BIN", "NPU_PYTHON_BIN", "A5_HOST_PYTHON"):
        monkeypatch.delenv(key, raising=False)
    workspace, _, _ = _workspace(tmp_path, monkeypatch)
    bundle = runner.resolve_staged_bundle(workspace)
    binding = runner.build_evaluation_binding(workspace, bundle=bundle)
    context = _create_execution_context(
        workspace, bundle=bundle, candidate_dir=None, binding=binding, verb="preflight"
    )
    provider_copy = context.runner_root / "npubench_inputs.py"
    provider_source = Path(runner.__file__).resolve().with_name("npubench_inputs.py")
    assert provider_copy.read_bytes() == provider_source.read_bytes()
    assert not provider_copy.stat().st_mode & 0o222
    captured: dict[str, object] = {}

    def fake_subprocess(command, **kwargs):
        captured["command"] = list(command)
        captured["env"] = kwargs["env"]
        child = _base_report(
            "preflight", status="PASS", binding=binding, run_id=context.run_id
        )
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(child), stderr="")

    # A macOS host without the removed sandbox-exec dependency still gets a
    # normal isolated process boundary.
    monkeypatch.setattr(runner.sys, "platform", "darwin")
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)
    try:
        report = _run_isolated_context(context, subprocess_run=fake_subprocess)
    finally:
        _cleanup_execution_context(context)

    command = captured["command"]
    assert command[0] == str(Path(sys.executable).resolve())
    assert str(context.runner_root / "npubench_runner.py") in command
    assert str(context.request_path) in command
    assert "bwrap" not in command
    assert "sandbox-exec" not in command
    assert report["execution_isolation"] == "process_boundary"
    assert report["tamper_protection"] == "post_run_hash_check"
    assert "execution_security" not in report
    assert captured["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_candidate_root_rejects_a_symlink(tmp_path: Path) -> None:
    from npubench.npubench_runner import _candidate_root
    real = tmp_path / "real-candidate"
    real.mkdir()
    alias = tmp_path / "candidate-alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(runner.NpuBenchRunnerError, match="real directory"):
        _candidate_root(alias)


def test_isolated_child_uses_configured_a5_npu_python(tmp_path: Path, monkeypatch) -> None:
    from npubench.npubench_runner import (
        _base_report, _cleanup_execution_context, _create_execution_context, _run_isolated_context,
    )
    npu_bin = tmp_path / "npu-python-bin"
    npu_bin.mkdir()
    npu_python = npu_bin / "python3"
    npu_python.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(npu_python, 0o700)
    monkeypatch.setenv("A5_NPU_PYTHON_BIN", str(npu_bin))
    workspace, _, _ = _workspace(tmp_path, monkeypatch)
    bundle = runner.resolve_staged_bundle(workspace)
    binding = runner.build_evaluation_binding(workspace, bundle=bundle)
    context = _create_execution_context(
        workspace, bundle=bundle, candidate_dir=None, binding=binding, verb="preflight"
    )
    captured: dict[str, object] = {}

    def fake_subprocess(command, **kwargs):
        captured["command"] = list(command)
        captured["env"] = kwargs["env"]
        child = _base_report(
            "preflight", status="PASS", binding=binding, run_id=context.run_id
        )
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(child), stderr="")

    try:
        _run_isolated_context(context, subprocess_run=fake_subprocess)
    finally:
        _cleanup_execution_context(context)

    assert captured["command"][0] == str(npu_python.resolve())
    assert captured["env"]["PATH"].split(os.pathsep)[0] == str(npu_bin)


def test_isolated_child_reads_npu_python_from_ascendc_env(tmp_path: Path, monkeypatch) -> None:
    """The normal launcher need not shell-source config for native evaluation."""
    from npubench.npubench_runner import (
        _base_report, _cleanup_execution_context, _create_execution_context, _run_isolated_context,
    )
    for key in ("A5_NPU_PYTHON_BIN", "NPU_PYTHON_BIN", "A5_HOST_PYTHON"):
        monkeypatch.delenv(key, raising=False)
    npu_bin = tmp_path / "configured-npu-python"
    npu_bin.mkdir()
    npu_python = npu_bin / "python3"
    npu_python.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(npu_python, 0o700)
    config = tmp_path / ".ascendc_env"
    config.write_text(f"A5_NPU_PYTHON_BIN={npu_bin}\n", encoding="utf-8")
    monkeypatch.setenv("ASCENDC_ENV_PATH", str(config))
    workspace, _, _ = _workspace(tmp_path, monkeypatch)
    bundle = runner.resolve_staged_bundle(workspace)
    binding = runner.build_evaluation_binding(workspace, bundle=bundle)
    context = _create_execution_context(
        workspace, bundle=bundle, candidate_dir=None, binding=binding, verb="preflight"
    )
    captured: dict[str, object] = {}

    def fake_subprocess(command, **kwargs):
        captured["command"] = list(command)
        child = _base_report(
            "preflight", status="PASS", binding=binding, run_id=context.run_id
        )
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(child), stderr="")

    try:
        _run_isolated_context(context, subprocess_run=fake_subprocess)
    finally:
        _cleanup_execution_context(context)

    assert captured["command"][0] == str(npu_python.resolve())


def test_isolated_child_rejects_unresolvable_npu_python_from_ascendc_env(
    tmp_path: Path, monkeypatch
) -> None:
    """A configured target interpreter must not fall back to controller Python."""
    from npubench.npubench_runner import _resolve_target_python
    for key in ("A5_NPU_PYTHON_BIN", "NPU_PYTHON_BIN", "A5_HOST_PYTHON"):
        monkeypatch.delenv(key, raising=False)
    config = tmp_path / ".ascendc_env"
    config.write_text("NPU_PYTHON_BIN=/does/not/exist\n", encoding="utf-8")
    monkeypatch.setenv("ASCENDC_ENV_PATH", str(config))

    with pytest.raises(runner.NpuBenchRunnerError, match="NPU_PYTHON_BIN"):
        _resolve_target_python()


def test_isolated_child_rejects_unresolvable_npu_python_for_remote_container(
    tmp_path: Path, monkeypatch
) -> None:
    from npubench.npubench_runner import _resolve_target_python
    for key in ("A5_NPU_PYTHON_BIN", "NPU_PYTHON_BIN", "A5_HOST_PYTHON"):
        monkeypatch.delenv(key, raising=False)
    config = tmp_path / ".ascendc_env"
    config.write_text(
        "A5_CONTAINER=remote_a5\nNPU_PYTHON_BIN=/does/not/exist\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ASCENDC_ENV_PATH", str(config))

    with pytest.raises(runner.NpuBenchRunnerError, match="NPU_PYTHON_BIN"):
        _resolve_target_python()


def test_isolated_child_rejects_remote_container_without_target_python(
    tmp_path: Path, monkeypatch
) -> None:
    from npubench.npubench_runner import _resolve_target_python
    for key in ("A5_NPU_PYTHON_BIN", "NPU_PYTHON_BIN", "A5_HOST_PYTHON"):
        monkeypatch.delenv(key, raising=False)
    config = tmp_path / ".ascendc_env"
    config.write_text("A5_CONTAINER=remote_a5\n", encoding="utf-8")
    monkeypatch.setenv("ASCENDC_ENV_PATH", str(config))

    with pytest.raises(runner.NpuBenchRunnerError, match="non-local target"):
        _resolve_target_python()


def test_preflight_rejects_staging_only_ready_status(tmp_path: Path, monkeypatch) -> None:
    from npubench.npubench_runner import _base_report
    workspace, _, _ = _workspace(tmp_path, monkeypatch)
    bundle = runner.resolve_staged_bundle(workspace)
    binding = runner.build_evaluation_binding(workspace, bundle=bundle)
    monkeypatch.setattr(
        runner,
        "_preflight_workspace_in_process",
        lambda _workspace: _base_report(
            "preflight", status="READY", binding=binding
        ),
    )

    report = runner.preflight_workspace(workspace, isolated=False)

    assert report["status"] == "ERROR"
    assert report["reason"] == "preflight returned staging-only READY status"


def _native_perf_manifest_stub() -> dict:
    """Minimal single-case native perf manifest that satisfies the runner contract."""
    return {
        "manifest_sha256": "m" * 64,
        "fixture_sha256": "f" * 64,
        "fixture_schema": runner.NATIVE_PERF_FIXTURE_SCHEMA,
        "input_adapter": {
            "kind": "task_provider",
            "provider": "get_input_groups",
            "contract": runner.INPUT_ADAPTER_CONTRACT_VERSION,
        },
        "case_count": 1,
        "valid_case_indices": [0],
        "empty_case_indices": [],
        "init_call_style": "args",
        "case_fixtures": [
            {
                "case": 0,
                "path": "native_perf_cases/case_000000.pt",
                "sha256": "c" * 64,
                "empty_tensor": False,
            }
        ],
    }


def _install_perf_process_boundary_doubles(monkeypatch, workspace: Path) -> types.SimpleNamespace:
    """Stub freeze / isolated-run / manifest-load so run_performance_workspace stays hermetic."""
    from npubench.npubench_runner import _base_report, _make_tree_read_only

    native = _native_perf_manifest_stub()
    contexts: list[object] = []
    manifest_checks: list[Path] = []
    frozen = workspace / runner.EXECUTION_DIRNAME / ".frozen" / "fixture-unit"

    def fake_freeze(_workspace, _source, *, binding, token):
        del binding, token
        frozen.mkdir(parents=True)
        (frozen / runner.NATIVE_PERF_MANIFEST_FILENAME).write_text("{}\n", encoding="utf-8")
        _make_tree_read_only(frozen)
        return frozen, native

    def fake_run(context, **_kwargs):
        contexts.append(context)
        child = _base_report(
            context.verb, status="PASS", binding=context.binding, run_id=context.run_id
        )
        child["child_returncode"] = 0
        return child

    def fake_load_manifest(path, *, binding, verify_case_payloads=True):
        del binding, verify_case_payloads
        manifest_checks.append(Path(path))
        return native

    monkeypatch.setattr(runner, "_freeze_native_perf_fixture", fake_freeze)
    monkeypatch.setattr(runner, "_run_isolated_context", fake_run)
    monkeypatch.setattr(runner, "_load_native_perf_manifest", fake_load_manifest)
    monkeypatch.setattr(runner, "_load_quick_profiler_summary", lambda *_args, **_kwargs: {"per_case": []})
    monkeypatch.setattr(
        runner,
        "_archive_retained_profiles",
        lambda *_args, **_kwargs: ("npubench_evidence/profiles/unit", "d" * 64),
    )
    return types.SimpleNamespace(contexts=contexts, manifest_checks=manifest_checks, frozen=frozen)


def _quick_profile_dirs(profile_root: Path, run_id: str) -> tuple[Path, Path]:
    """Create and return the raw ref/asc profiler directories the runner expects for case 0."""
    from npubench.npubench_runner import _expected_profile_dir

    ref = _expected_profile_dir(profile_root, run_id, 0, "ref") / "PROF_ref"
    asc = _expected_profile_dir(profile_root, run_id, 0, "asc") / "PROF_asc"
    for raw in (ref, asc):
        raw.mkdir(parents=True)
    return ref, asc


def _quick_summary_payload(ref: Path, asc: Path, *, total: int, valid: int, **extra) -> dict:
    """Build a one-case quick-profiler performance.json payload."""
    payload = {
        "profiling_mode": "quick",
        "warmup": 3,
        "repeats": 5,
        "n_cases_total": total,
        "n_cases_valid": valid,
        "per_case": [
            {
                "case": 0,
                "ref_us": 10.0,
                "asc_us": 5.0,
                "speedup": 2.0,
                "ref_error": None,
                "asc_error": None,
                "ref_prof_dir": str(ref),
                "asc_prof_dir": str(asc),
            }
        ],
    }
    payload.update(extra)
    return payload


def _timeout_probe_context(tmp_path: Path) -> types.SimpleNamespace:
    """Minimal execution context accepted by _run_isolated_context."""
    return types.SimpleNamespace(
        verb="precision",
        target_python=Path(sys.executable),
        runner_root=tmp_path,
        scratch=tmp_path,
        request_path=tmp_path / "request.json",
    )


def _install_skip_perf_doubles(monkeypatch, binding: dict) -> list:
    """Stub precision success and make any performance run a hard failure."""
    monkeypatch.setattr(
        runner,
        "run_precision_workspace",
        lambda *args, **kwargs: {
            "schema": "cannbot.npubench.precision/v1",
            "status": "PASS",
            "binding_sha256": binding["binding_sha256"],
            "evaluation_binding": binding,
            "pass_a": {"status": "PASS", "tier1_pass": 3, "total": 3},
        },
    )
    ran_perf: list = []

    def fake_perf(*args, **kwargs):
        ran_perf.append(1)
        raise AssertionError("performance must not run in skip-perf mode")

    monkeypatch.setattr(runner, "run_performance_workspace", fake_perf)
    monkeypatch.setenv("CANNBOT_NPUBENCH_SKIP_PERF", "1")
    return ran_perf


def test_performance_process_boundary_revalidates_fixture_before_publish(
    tmp_path: Path, monkeypatch
) -> None:
    workspace, _, _ = _workspace(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path)
    doubles = _install_perf_process_boundary_doubles(monkeypatch, workspace)
    contexts = doubles.contexts
    manifest_checks = doubles.manifest_checks
    frozen = doubles.frozen

    report = runner.run_performance_workspace(workspace, candidate, device=0)

    assert report["status"] == "PASS"
    assert report["execution_isolation"] == "process_boundary"
    assert report["tamper_protection"] == "post_run_hash_check"
    assert report["fixture_execution_isolation"] == "process_boundary"
    assert report["fixture_tamper_protection"] == "post_run_hash_check"
    assert "execution_security" not in report
    assert len(contexts) == 2
    assert contexts[0].verb == "fixture"
    assert contexts[0].candidate_root == candidate.resolve()
    assert contexts[1].verb == "performance"
    assert contexts[1].candidate_root == candidate.resolve()
    assert frozen in manifest_checks
    assert (
        contexts[1].scratch
        / ".npubench_adapter"
        / contexts[1].binding["binding_sha256"]
        / contexts[1].run_id
        / "native_fixture"
        in manifest_checks
    )
    assert not frozen.exists()


def test_exec_context_is_excluded_from_candidate_scope_and_cleaned(tmp_path: Path, monkeypatch) -> None:
    from npubench.npubench_runner import _cleanup_execution_context, _create_execution_context
    workspace, _, _ = _workspace(tmp_path, monkeypatch)
    (workspace / "model_new_ascendc.py").write_text("VALUE = 1\n", encoding="utf-8")
    # Candidate is the workspace in the O5 flow, so its source digest must not
    # drift merely because the evaluator creates a private execution context.
    before = runner.candidate_tree_sha256(workspace)
    bundle = runner.resolve_staged_bundle(workspace)
    binding = runner.build_evaluation_binding(workspace, workspace, bundle=bundle)
    context = _create_execution_context(
        workspace,
        bundle=bundle,
        candidate_dir=workspace,
        binding=binding,
        verb="preflight",
    )
    try:
        assert runner.candidate_tree_sha256(workspace) == before
    finally:
        _cleanup_execution_context(context)
    assert not context.root.exists()


def _install_precision_pipeline_doubles(monkeypatch, *, reference, candidate, order, calls) -> None:
    """Stub every _run_precision collaborator so only the init-input plumbing is exercised."""
    monkeypatch.setattr(
        runner,
        "load_task_module",
        lambda _path, _root, *, role: reference if role == "reference" else candidate,
    )
    monkeypatch.setattr(runner, "_validate_reference_api", lambda _module: {})
    monkeypatch.setattr(runner, "_resolve_model_constructor", lambda *_args, **_kwargs: object)
    monkeypatch.setattr(runner, "_import_torch", lambda: object())
    monkeypatch.setattr(
        runner, "seed_everything", lambda *_args, **_kwargs: order.append("seed") or ["seed"]
    )
    monkeypatch.setattr(
        runner,
        "_construct_model",
        lambda _ctor, args, role: (
            calls.append((role, tuple(args))), order.append(role), object()
        )[-1],
    )
    monkeypatch.setattr(runner, "_resolve_device", lambda *_args: "cpu")
    monkeypatch.setattr(runner, "_move_model", lambda *_args: None)
    monkeypatch.setattr(runner, "_set_eval", lambda *_args: None)
    monkeypatch.setattr(runner, "_get_input_groups", lambda _module: order.append("inputs") or [])
    monkeypatch.setattr(runner, "_candidate_tree_sha256", lambda _root: "c" * 64)


def test_precision_uses_candidate_init_inputs_with_reference_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    from npubench.npubench_runner import _run_precision
    reference = types.ModuleType("reference")
    candidate = types.ModuleType("candidate")
    order: list[str] = []

    def reference_init_inputs():
        return [1, {"alpha": 2}, "reference"]

    def candidate_init_inputs():
        order.append("candidate_init")
        return [7, {"alpha": 9}, "candidate"]

    reference.get_init_inputs = reference_init_inputs
    candidate.get_init_inputs = candidate_init_inputs
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "model_new_ascendc.py").write_text("# placeholder\n", encoding="utf-8")
    task = tmp_path / "task.py"
    task.write_text("# placeholder\n", encoding="utf-8")
    bundle = types.SimpleNamespace(task_path=task, root=tmp_path)
    calls: list[tuple[str, tuple[object, ...]]] = []

    _install_precision_pipeline_doubles(
        monkeypatch, reference=reference, candidate=candidate, order=order, calls=calls
    )

    report = _run_precision(
        bundle,
        candidate_dir,
        device=0,
        seed=0,
        binding={"binding_sha256": "b" * 64, "candidate_tree_sha256": "c" * 64},
    )

    assert report["status"] == "PASS"
    assert calls == [
        ("reference", (7, {"alpha": 9}, "candidate")),
        ("candidate", (7, {"alpha": 9}, "candidate")),
    ]
    assert order == ["seed", "candidate_init", "inputs", "reference", "candidate"]


def test_quick_summary_rejects_partial_coverage_and_wrong_profile_tag(tmp_path: Path) -> None:
    from npubench.npubench_runner import _load_quick_profiler_summary
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    profile_root = tmp_path / "prof"
    run_id = "strict-run"
    ref, asc = _quick_profile_dirs(profile_root, run_id)
    partial = _quick_summary_payload(
        ref, asc, total=2, valid=1, gate_eligible=True, profiler_artifacts_retained=True
    )
    (adapter / "performance.json").write_text(json.dumps(partial), encoding="utf-8")

    with pytest.raises(runner.NpuBenchRunnerError, match="valid case coverage"):
        _load_quick_profiler_summary(
            adapter,
            expected_case_count=2,
            expected_valid_case_indices=[0, 1],
            run_id=run_id,
            profile_root=profile_root,
        )

    wrong_tag = dict(partial)
    wrong_tag["n_cases_total"] = 1
    wrong_tag["n_cases_valid"] = 1
    wrong_tag["per_case"] = [dict(partial["per_case"][0], ref_prof_dir=str(profile_root / "msprof_anything"))]
    (adapter / "performance.json").write_text(json.dumps(wrong_tag), encoding="utf-8")
    with pytest.raises(runner.NpuBenchRunnerError, match="tag/path mismatch"):
        _load_quick_profiler_summary(
            adapter,
            expected_case_count=1,
            expected_valid_case_indices=[0],
            run_id=run_id,
            profile_root=profile_root,
        )


def test_quick_summary_accepts_direct_prof_child_from_shared_engine(tmp_path: Path) -> None:
    from npubench.npubench_runner import _load_quick_profiler_summary
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    profile_root = tmp_path / "prof"
    run_id = "shared-engine"
    ref, asc = _quick_profile_dirs(profile_root, run_id)
    (adapter / "performance.json").write_text(
        json.dumps(_quick_summary_payload(ref, asc, total=1, valid=1)),
        encoding="utf-8",
    )

    summary = _load_quick_profiler_summary(
        adapter,
        expected_case_count=1,
        expected_valid_case_indices=[0],
        run_id=run_id,
        profile_root=profile_root,
    )

    assert summary["per_case"] == [
        {
            "case": 0,
            "ref_us": 10.0,
            "asc_us": 5.0,
            "speedup": 2.0,
            "ref_prof_dir": str(ref),
            "asc_prof_dir": str(asc),
        }
    ]


def test_native_fixture_rejects_repeated_tensor_and_shared_storage() -> None:
    from npubench.npubench_runner import _native_reject_tensor_aliases
    torch = _torch_or_skip()
    tensor = torch.arange(8)
    with pytest.raises(runner.NpuBenchRunnerError, match="repeated tensor identity"):
        _native_reject_tensor_aliases([tensor, tensor], torch, label="test")
    with pytest.raises(runner.NpuBenchRunnerError, match="shared tensor storage"):
        _native_reject_tensor_aliases([tensor, tensor.view(2, 4)], torch, label="test")


def test_native_reject_tensor_aliases_ignores_torch_dtype_and_device() -> None:
    from npubench.npubench_runner import _native_reject_tensor_aliases, _native_restricted_tree
    torch = _torch_or_skip()
    group = [{"output_dtype": torch.float16, "device": torch.device("cpu")}]
    _native_reject_tensor_aliases(group, torch, label="test")
    tree = _native_restricted_tree(group, torch)
    assert tree[0]["output_dtype"] is torch.float16
    assert tree[0]["device"] == torch.device("cpu")


def test_native_perf_fixture_accepts_torch_dtype_in_input_group(tmp_path: Path, monkeypatch) -> None:
    """Golden tasks may carry ``output_dtype`` as a ``torch.dtype`` in a group."""
    from npubench.npubench_runner import _materialize_native_perf_fixture
    torch = _torch_or_skip()
    workspace, bundle, task = _workspace(tmp_path, monkeypatch, sidecar_suffix="_perf_cases.jsonl")
    task.write_text(
        "import torch\n"
        "class Model:\n"
        "    def __init__(self, *args): pass\n"
        "    def state_dict(self): return {}\n"
        "    def load_state_dict(self, state, strict=True): return None\n"
        "    def to(self, device): return self\n"
        "    def __call__(self, *args, **kwargs): return args\n"
        "def get_input_groups():\n"
        "    return [[torch.tensor([1.0]), {'output_dtype': torch.float16, 'device': torch.device('cpu')}]]\n"
        "def get_init_inputs(): return []\n",
        encoding="utf-8",
    )
    candidate = _candidate(tmp_path)
    binding = runner.build_evaluation_binding(workspace, candidate)
    adapter = runner.prepare_adapter_view(
        workspace, candidate, bundle=runner.resolve_staged_bundle(workspace), binding=binding, run_id="dtype-test"
    )
    manifest = _materialize_native_perf_fixture(
        adapter, runner.resolve_staged_bundle(workspace), candidate, binding=binding, seed=0
    )
    case = torch.load(
        adapter / manifest["case_fixtures"][0]["path"], map_location="cpu", weights_only=True
    )
    group = case["input_group"]
    assert group[1]["output_dtype"] is torch.float16
    assert group[1]["device"] == torch.device("cpu")


def test_verify_evidence_report_bubbles_failed_subreport_reason(tmp_path: Path, monkeypatch) -> None:
    from npubench.npubench_runner import _base_report
    workspace, _, _ = _workspace(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path)
    report = _base_report("performance", status="ERROR", run_id="lane-run")
    report["reason"] = "native performance fixture cannot safely inspect dtype for aliasing"
    ok, reason = runner.verify_evidence_report(
        workspace, report, expected_verb="performance", candidate_dir=candidate
    )
    assert not ok
    assert reason == report["reason"]


def test_verify_evidence_report_keeps_digest_message_for_successful_report(
    tmp_path: Path, monkeypatch
) -> None:
    from npubench.npubench_runner import _base_report
    workspace, _, _ = _workspace(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path)
    report = _base_report("performance", status="PASS", run_id="lane-run")
    report["binding_sha256"] = "0" * 64
    ok, reason = runner.verify_evidence_report(
        workspace, report, expected_verb="performance", candidate_dir=candidate
    )
    assert not ok
    assert reason == "npubench evidence binding digest mismatch"


def test_evaluate_bubbles_lane_reason_when_binding_is_missing(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    candidate = _candidate(tmp_path)
    binding = {"binding_sha256": "c" * 64, "source": "npubench"}
    monkeypatch.setattr(
        runner,
        "run_precision_workspace",
        lambda *args, **kwargs: {
            "schema": "cannbot.npubench.precision/v1",
            "status": "ERROR",
            "reason": "candidate module has no PyInit",
        },
    )
    monkeypatch.setattr(
        runner,
        "run_performance_workspace",
        lambda *args, **kwargs: {
            "schema": "cannbot.npubench.performance/v1",
            "status": "PASS",
            "binding_sha256": binding["binding_sha256"],
            "evaluation_binding": binding,
        },
    )

    report = runner.evaluate_workspace(workspace, candidate, precision_device=0, performance_device=0)

    assert report["status"] == "ERROR"
    assert report["reason"] == "candidate module has no PyInit"


def test_json_safe_non_finite_replaces_only_non_finite_floats() -> None:
    from npubench.npubench_runner import _json_safe_non_finite
    payload = {
        "MERE": float("inf"),
        "neg": float("-inf"),
        "missing": float("nan"),
        "finite": 1.25,
        "count": 3,
        "flag": True,
        "nested": {"children": [{"max_abs_diff": float("inf")}]},
        "binding_sha256": "a" * 64,
    }

    safe = _json_safe_non_finite(payload)

    assert safe["MERE"] == "inf"
    assert safe["neg"] == "-inf"
    assert safe["missing"] == "nan"
    assert safe["finite"] == 1.25
    assert safe["count"] == 3
    assert safe["flag"] is True
    assert safe["nested"]["children"][0]["max_abs_diff"] == "inf"
    assert safe["binding_sha256"] == "a" * 64
    # The original payload is not mutated and the result is strict-JSON safe.
    assert payload["MERE"] == float("inf")
    json.dumps(safe, ensure_ascii=False, sort_keys=True, allow_nan=False)


def test_precision_report_with_non_finite_metrics_still_persists_fail(
    tmp_path: Path, monkeypatch
) -> None:
    from npubench.npubench_runner import _base_report
    workspace, _, _ = _workspace(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path)

    def fake_run(context, **_kwargs):
        child = _base_report(
            context.verb, status="FAIL", binding=context.binding, run_id=context.run_id
        )
        child.update(
            {
                "case_count": 1,
                "passed_case_count": 0,
                "failed_case_count": 1,
                "cases": [
                    {
                        "case": 0,
                        "status": "FAIL",
                        "metrics": {
                            "MERE": float("inf"),
                            "max_abs_diff": float("inf"),
                            "matched_ratio": float("nan"),
                        },
                        "reason": "output: NPU benchmark accuracy checks failed",
                    }
                ],
                "child_returncode": 0,
            }
        )
        return child

    monkeypatch.setattr(runner, "_run_isolated_context", fake_run)

    report = runner.run_precision_workspace(workspace, candidate, device=0)

    assert report["status"] == "FAIL"
    persisted_path = workspace / runner.EVIDENCE_DIRNAME / runner.PRECISION_REPORT_FILENAME
    assert persisted_path.is_file() and not persisted_path.is_symlink()
    persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "FAIL"
    assert persisted["passed_case_count"] == 0
    assert persisted["failed_case_count"] == 1
    case = persisted["cases"][0]
    assert case["status"] == "FAIL"
    assert case["metrics"]["MERE"] == "inf"
    assert case["metrics"]["max_abs_diff"] == "inf"
    assert case["metrics"]["matched_ratio"] == "nan"
    assert case["reason"] == "output: NPU benchmark accuracy checks failed"
    # Binding digests are hex strings and must pass through untouched.
    assert persisted["binding_sha256"] == report["binding_sha256"]
    # The returned report must be identical to the persisted file: finalize
    # (npubench_target._verify_reports) rejects any memory/disk divergence.
    assert persisted == report


def test_evaluate_report_with_non_finite_lane_metrics_persists_fail(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    candidate = _candidate(tmp_path)
    binding = {"binding_sha256": "c" * 64, "source": "npubench"}
    precision_report = {
        "schema": "cannbot.npubench.precision/v1",
        "status": "FAIL",
        "binding_sha256": binding["binding_sha256"],
        "evaluation_binding": binding,
        "case_count": 1,
        "passed_case_count": 0,
        "failed_case_count": 1,
        "cases": [
            {
                "case": 0,
                "status": "FAIL",
                "metrics": {"MERE": float("inf"), "matched_ratio": float("nan")},
                "reason": "output: NPU benchmark accuracy checks failed",
            }
        ],
    }
    performance_report = {
        "schema": "cannbot.npubench.performance/v1",
        "status": "PASS",
        "binding_sha256": binding["binding_sha256"],
        "evaluation_binding": binding,
    }
    monkeypatch.setattr(
        runner, "run_precision_workspace", lambda *args, **kwargs: dict(precision_report)
    )
    monkeypatch.setattr(
        runner, "run_performance_workspace", lambda *args, **kwargs: dict(performance_report)
    )

    report = runner.evaluate_workspace(workspace, candidate, precision_device=0, performance_device=0)

    assert report["status"] == "FAIL"
    persisted_path = workspace / runner.EVIDENCE_DIRNAME / runner.EVALUATE_REPORT_FILENAME
    assert persisted_path.is_file() and not persisted_path.is_symlink()
    persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "FAIL"
    metrics = persisted["precision"]["cases"][0]["metrics"]
    assert metrics == {"MERE": "inf", "matched_ratio": "nan"}
    # An inf error metric still means FAIL: pass/fail derives from per-case
    # status, never from the sanitized metric values.
    assert persisted["precision"]["status"] == "FAIL"
    assert persisted["precision"]["failed_case_count"] == 1
    # The aggregate returned to the caller matches the on-disk evidence.
    assert persisted == report


def test_task_execution_timeout_defaults_without_env(monkeypatch) -> None:
    from npubench.npubench_runner import _resolve_task_execution_timeout
    monkeypatch.delenv("CANNBOT_NPUBENCH_TASK_TIMEOUT_SEC", raising=False)
    assert _resolve_task_execution_timeout(None) == 3600
    assert _resolve_task_execution_timeout(None) == runner.TASK_EXECUTION_TIMEOUT_SECONDS


def test_task_execution_timeout_env_override(monkeypatch) -> None:
    from npubench.npubench_runner import _resolve_task_execution_timeout
    monkeypatch.setenv("CANNBOT_NPUBENCH_TASK_TIMEOUT_SEC", "1200")
    assert _resolve_task_execution_timeout(None) == 1200


def test_task_execution_timeout_explicit_value_wins_over_env(monkeypatch) -> None:
    from npubench.npubench_runner import _resolve_task_execution_timeout
    monkeypatch.setenv("CANNBOT_NPUBENCH_TASK_TIMEOUT_SEC", "1200")
    assert _resolve_task_execution_timeout(7) == 7


@pytest.mark.parametrize("raw", ["", "   ", "abc", "1.5", "0", "-30"])
def test_task_execution_timeout_rejects_invalid_env(monkeypatch, raw: str) -> None:
    from npubench.npubench_runner import _resolve_task_execution_timeout
    monkeypatch.setenv("CANNBOT_NPUBENCH_TASK_TIMEOUT_SEC", raw)
    with pytest.raises(runner.NpuBenchRunnerError, match="CANNBOT_NPUBENCH_TASK_TIMEOUT_SEC"):
        _resolve_task_execution_timeout(None)


def test_isolated_context_uses_env_timeout_for_child_process(tmp_path: Path, monkeypatch) -> None:
    from npubench.npubench_runner import _run_isolated_context
    monkeypatch.setenv("CANNBOT_NPUBENCH_TASK_TIMEOUT_SEC", "1200")
    context = _timeout_probe_context(tmp_path)
    captured: dict = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(returncode=0, stdout='{"status": "PASS"}', stderr="")

    report = _run_isolated_context(context, subprocess_run=fake_run)

    assert captured["timeout"] == 1200
    assert report["status"] == "PASS"


def test_isolated_context_default_timeout_is_3600(tmp_path: Path, monkeypatch) -> None:
    from npubench.npubench_runner import _run_isolated_context
    monkeypatch.delenv("CANNBOT_NPUBENCH_TASK_TIMEOUT_SEC", raising=False)
    context = _timeout_probe_context(tmp_path)
    captured: dict = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(returncode=0, stdout='{"status": "PASS"}', stderr="")

    _run_isolated_context(context, subprocess_run=fake_run)

    assert captured["timeout"] == 3600


def test_evaluate_skip_perf_produces_deferred_report(tmp_path: Path, monkeypatch) -> None:
    """CANNBOT_NPUBENCH_SKIP_PERF=1 banks precision-only evidence (P3 精度优先)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    candidate = _candidate(tmp_path)
    binding = {"binding_sha256": "d" * 64, "source": "npubench", "input_adapter": {}}
    ran_perf = _install_skip_perf_doubles(monkeypatch, binding)

    report = runner.evaluate_workspace(workspace, candidate, precision_device=0, performance_device=0)

    assert ran_perf == []
    assert report["status"] == "PASS"
    assert "deferred" in report["reason"]
    perf = report["performance"]
    assert perf["status"] == "DEFERRED"
    assert perf["perf_deferred"] is True
    assert perf["binding_sha256"] == binding["binding_sha256"]
    assert perf["profile_archive"] is None


def test_evaluate_skip_perf_wins_over_parallel_two_lane(tmp_path: Path, monkeypatch) -> None:
    """CANNBOT_NPUBENCH_SKIP_PERF=1 must defer perf even with two distinct leases.

    codex review F6 (2026-08-25): the parallel two-lane branch must not run
    performance when skip-perf is requested.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    candidate = _candidate(tmp_path)
    lease_manifest = tmp_path / "lease.json"
    lease_manifest.write_text(json.dumps({"leases": {
        "precision": {"device": 0, "token": "p"},
        "performance": {"device": 1, "token": "q"},
    }}))
    binding = {"binding_sha256": "d" * 64, "source": "npubench", "input_adapter": {}}
    ran_perf = _install_skip_perf_doubles(monkeypatch, binding)

    report = runner.evaluate_workspace(
        workspace,
        candidate,
        precision_device=0,
        performance_device=1,
        lease_manifest=lease_manifest,
    )

    assert ran_perf == []
    assert report["parallelism"] == "precision_only"
    assert report["status"] == "PASS"
    perf = report["performance"]
    assert perf["status"] == "DEFERRED"
    assert perf["perf_deferred"] is True
    assert perf["device"] == 1
