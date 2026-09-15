# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Child-side internal-exec verbs for the NPUKernelBench runner.

This module owns the child-only half of one isolated execution: pinning the
assigned device before any torch import, routing an already-verified
internal-exec request to its preflight/precision/fixture/performance verb
handler, and turning any child failure into an ERROR report the parent can
attribute to the request's binding.

It imports the runner's sibling modules directly.  The two runner-owned steps
the verbs drive (``_run_precision`` and ``_materialize_native_perf_fixture``)
are reached through lazy runner imports at call time, because the runner
itself imports this module at load time.  ``npubench_runner`` re-exports this
module's public surface, so importers keep using the runner module path.
"""
from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from npubench_core import (
    DEFAULT_SEED,
    NpuBenchRunnerError,
    StagedBundle,
    _base_report,
    _output_tail,
    _request_scratch,
    _require_regular,
    _resolve_execution_request,
    _safe_prof_tag,
    build_evaluation_binding,
    resolve_staged_bundle,
)
from npubench_fixture import (
    _assert_input_adapter_api,
    _prepare_native_quick_adapter,
    _validate_sidecar_descriptors,
    prepare_adapter_view,
)
from npubench_precision import (
    _validate_reference_api,
    load_task_module,
)
from npubench_profile import (
    REPEATS,
    WARM_UP,
    _default_profiler_summary,
)


def _run_precision_impl() -> Callable[..., dict[str, Any]]:
    """Return the runner-owned ``_run_precision`` without an import cycle.

    Package-qualified first — that is the module instance the UT suite
    monkeypatches — with a flat fallback for the staged runner copy, which
    has no package layout (mirrors ``npubench_core._load_inputs_provider``).
    """
    try:
        from npubench.npubench_runner import _run_precision
    except ModuleNotFoundError as exc:
        if exc.name != "npubench":
            raise
        from npubench_runner import _run_precision
    return _run_precision


def _materialize_native_perf_fixture_impl() -> Callable[..., Mapping[str, Any]]:
    """Return the runner-owned ``_materialize_native_perf_fixture``; see above."""
    try:
        from npubench.npubench_runner import _materialize_native_perf_fixture
    except ModuleNotFoundError as exc:
        if exc.name != "npubench":
            raise
        from npubench_runner import _materialize_native_perf_fixture
    return _materialize_native_perf_fixture


def _preflight_workspace_in_process(workspace: Path) -> dict[str, Any]:
    """Internal child-only implementation that imports the untrusted task."""
    try:
        bundle = resolve_staged_bundle(workspace)
        module = load_task_module(bundle.task_path, bundle.root, role="reference")
        api = _validate_reference_api(module)
        binding = build_evaluation_binding(workspace, bundle=bundle)
        _assert_input_adapter_api(api, binding)
        if api.get("input_provider") == "sidecar_descriptor":
            _validate_sidecar_descriptors(bundle.sidecar_cases)
        result = _base_report("preflight", status="PASS", binding=binding)
        result.update(
            {
                "task_path": str(bundle.task_path),
                "sidecar_path": str(bundle.sidecar_path),
                "sidecar_encoding": bundle.sidecar_encoding,
                "case_count": len(bundle.sidecar_cases),
                "task_api": api,
            }
        )
    except (NpuBenchRunnerError, OSError, ValueError, SyntaxError, ImportError) as exc:
        result = _base_report("preflight", status="ERROR")
        result["reason"] = str(exc)
    return result


def _apply_assigned_device(request: dict[str, Any]) -> None:
    """Pin the child to its ONE assigned NPU before any torch/torch_npu import.

    Full 8-card visibility makes torch_npu negotiate an HCCL collective world
    at first device use; on shared-mode cards that init faults with acl error
    507035 (2026-08-22 BAM/SDPA/CoT on lanes 1-3, while single-card
    ASCEND_RT_VISIBLE_DEVICES runs work fine — stack traceback shows
    libhccl.so during the first tensor move).
    """
    assigned_device = request.get("device")
    if isinstance(assigned_device, int) and assigned_device >= 0:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(assigned_device)
        request["device"] = 0


def _execute_preflight_verb(
    bundle: StagedBundle, binding: Mapping[str, Any], run_id: str | None
) -> dict[str, Any]:
    """Child-side ``preflight``: import the task and validate its input API."""
    module = load_task_module(bundle.task_path, bundle.root, role="reference")
    api = _validate_reference_api(module)
    _assert_input_adapter_api(api, binding)
    if api.get("input_provider") == "sidecar_descriptor":
        _validate_sidecar_descriptors(bundle.sidecar_cases)
    result = _base_report("preflight", status="PASS", binding=binding, run_id=run_id)
    result.update(
        {
            "sidecar_encoding": bundle.sidecar_encoding,
            "case_count": len(bundle.sidecar_cases),
            "task_api": api,
        }
    )
    return result


def _execute_precision_verb(
    bundle: StagedBundle,
    candidate: Path | None,
    binding: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Child-side ``precision``: run the three checks against the candidate."""
    if candidate is None:
        raise NpuBenchRunnerError("isolated precision execution requires a candidate snapshot")
    result = _run_precision_impl()(
        bundle,
        candidate,
        device=request.get("device"),
        seed=int(request.get("seed", DEFAULT_SEED)),
        binding=binding,
    )
    result["run_id"] = str(request.get("run_id") or result.get("run_id"))
    return result


def _execute_fixture_verb(
    bundle: StagedBundle,
    candidate: Path | None,
    binding: Mapping[str, Any],
    request: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Child-side ``fixture``: materialize the shared native perf fixture."""
    if candidate is None:
        raise NpuBenchRunnerError("isolated native fixture requires a candidate snapshot")
    fixture_dir = _request_scratch(request) / "native_fixture"
    if fixture_dir.exists():
        raise NpuBenchRunnerError("isolated native fixture output already exists")
    fixture_dir.mkdir(mode=0o700)
    native_manifest = _materialize_native_perf_fixture_impl()(
        fixture_dir,
        bundle,
        candidate,
        binding=binding,
        seed=int(request.get("seed", DEFAULT_SEED)),
        write_adapter_manifest=False,
    )
    result = _base_report("fixture", status="PASS", binding=binding, run_id=run_id)
    result.update(
        {
            "fixture_relative": "native_fixture",
            "fixture_sha256": native_manifest["fixture_sha256"],
            "case_count": native_manifest["case_count"],
        }
    )
    return result


def _execute_performance_verb(
    bundle: StagedBundle,
    candidate: Path | None,
    binding: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    fixture_root: Path | None,
    run_id: str,
) -> dict[str, Any]:
    """Child-side ``performance``: drive the bundled quick profiler once."""
    if candidate is None:
        raise NpuBenchRunnerError("isolated performance execution requires a candidate snapshot")
    if fixture_root is None:
        raise NpuBenchRunnerError("isolated performance requires a parent-frozen native fixture")
    scratch = _request_scratch(request)
    adapter = prepare_adapter_view(
        scratch,
        candidate,
        bundle=bundle,
        binding=binding,
        run_id=run_id,
        allow_existing_native_fixture=True,
    )
    device = request.get("device")
    if isinstance(device, bool) or not isinstance(device, int):
        raise NpuBenchRunnerError("isolated performance request has invalid device")
    native = _prepare_native_quick_adapter(
        adapter,
        fixture_root,
        binding=binding,
    )
    return _child_profiler_report(
        adapter,
        scratch,
        binding,
        native,
        device=device,
        run_id=run_id,
        profiler_script=Path(str(request.get("profiler_script", ""))),
    )


def build_performance_command(
    adapter_dir: Path,
    *,
    device: int,
    run_id: str,
    profiler_script: Path | None = None,
) -> list[str]:
    """Build the fixed quick W3/R5/keep-profile command without executing it.

    The NPUKernelBench adapter supplies frozen inputs through generated local
    shims, then delegates timing and CSV parsing to the existing repository
    quick profiler engine.  The generic shell launcher is intentionally not
    used here: its current quick branch drops ``--keep-prof`` before it
    reaches that engine.
    """
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise NpuBenchRunnerError("performance device must be a non-negative integer")
    if not _safe_prof_tag(run_id):
        raise NpuBenchRunnerError("run_id is not safe for --prof-tag")
    script = Path(profiler_script) if profiler_script is not None else _default_profiler_summary()
    _require_regular(script, "msprof profiler summary script")
    command = [
        str(sys.executable),
        str(script),
        "--quick",
        "--warmup",
        str(WARM_UP),
        "--device",
        str(device),
        "--keep-prof",
        "--repeats",
        str(REPEATS),
        "--output-dir",
        str(Path(adapter_dir)),
        "--prof-tag",
        run_id,
    ]
    return command


def _child_profiler_report(
    adapter: Path,
    scratch: Path,
    binding: Mapping[str, Any],
    native: Mapping[str, Any],
    *,
    device: int,
    run_id: str,
    profiler_script: Path,
) -> dict[str, Any]:
    """Invoke the repository quick profiler once and report what it returned."""
    command = build_performance_command(
        adapter,
        device=device,
        run_id=run_id,
        profiler_script=profiler_script,
    )
    completed = subprocess.run(command, cwd=str(adapter), text=True, capture_output=True, check=False)
    result = _base_report(
        "performance",
        status="PASS" if completed.returncode == 0 else "FAIL",
        binding=binding,
        run_id=run_id,
    )
    result.update(
        {
            "returncode": int(completed.returncode),
            "command": command,
            "adapter_relative": str(adapter.relative_to(scratch)),
            "native_fixture_sha256": native["fixture_sha256"],
            "stdout_tail": _output_tail(completed.stdout),
            "stderr_tail": _output_tail(completed.stderr),
        }
    )
    return result


def _resolve_child_run_id(request: Mapping[str, Any], verb: str) -> str:
    """Return the run id this child reports under, generating one when allowed."""
    if verb in {"fixture", "performance"}:
        return str(request.get("run_id") or uuid.uuid4().hex)
    return str(request.get("run_id") or None)


def _dispatch_execution_verb(
    verb: str,
    bundle: StagedBundle,
    candidate: Path | None,
    binding: Mapping[str, Any],
    request: dict[str, Any],
    *,
    fixture_root: Path | None,
    run_id: str,
) -> dict[str, Any]:
    """Route one already-verified request to its child-side verb handler."""
    _apply_assigned_device(request)
    if verb == "preflight":
        return _execute_preflight_verb(bundle, binding, run_id)
    if verb == "precision":
        return _execute_precision_verb(bundle, candidate, binding, request)
    if verb == "fixture":
        return _execute_fixture_verb(bundle, candidate, binding, request, run_id)
    if verb == "performance":
        return _execute_performance_verb(
            bundle, candidate, binding, request, fixture_root=fixture_root, run_id=run_id
        )
    raise NpuBenchRunnerError(f"unsupported internal execution verb: {verb}")


def _internal_execute_request(request_path: Path, *, verb: str) -> dict[str, Any]:
    """Child-only task/candidate executor; it never opens state or evidence."""
    binding: Mapping[str, Any] | None = None
    run_id: str | None = None
    try:
        bundle, candidate, binding, request, fixture_root = _resolve_execution_request(request_path)
        run_id = _resolve_child_run_id(request, verb)
        return _dispatch_execution_verb(
            verb, bundle, candidate, binding, dict(request), fixture_root=fixture_root, run_id=run_id
        )
    except Exception as exc:
        # A task import/fixture error is still a response to one verified
        # parent request.  Preserve that binding when it is already available
        # so the parent can surface the actionable child reason rather than
        # reporting a misleading "binding differs" infrastructure failure.
        # The catch is deliberately ``Exception`` and not ``BaseException``:
        # the interpreter's own shutdown signals are not task failures and
        # must escape this catch-all unchanged, which they do by not being
        # ``Exception`` subclasses.
        # 2026-08-22 (BAM 507035): candidate kernels can FAULT the NPU device
        # (acl error 507035) and torch surfaces that as RuntimeError — outside
        # the historical catch tuple — so the child died with a bare traceback
        # and the parent reported "no machine-readable report", classifying a
        # worker-fixable kernel fault as infra-terminal.  Catch EVERYTHING and
        # emit an ERROR report: the parent then records a measured FAIL and
        # the FSM routes the fix back to the worker.
        result = _base_report(verb, status="ERROR", binding=binding, run_id=run_id)
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result
