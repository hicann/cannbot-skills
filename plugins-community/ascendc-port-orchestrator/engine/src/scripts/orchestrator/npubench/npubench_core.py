# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Shared contracts and primitives for the NPUKernelBench native runner.

This module owns everything the runner's phase modules need in common and
nothing that depends on them: the error type, the staged-bundle and execution
dataclasses, every contract constant, the safe path/tree/digest primitives,
the candidate-scope snapshot rules, the evaluation binding, the parent-authored
execution request, the report envelope, target-interpreter resolution, and the
bounded child-process plumbing.

It is the bottom of the runner's import DAG: ``npubench_precision``,
``npubench_fixture``, ``npubench_profile`` and ``npubench_runner`` all import
from it and it imports from none of them.  ``npubench_runner`` re-exports its
public surface, so importers keep using the runner module path.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


NPUBENCH_SOURCE = "npubench"
RUNNER_CONTRACT_VERSION = "npubench/v1"
# Digest-scheme version for the candidate scope exclusion list
# (_candidate_excluded / _CANDIDATE_RUNTIME_TOP_LEVEL).  Bump whenever the
# exclusion semantics change so finalize can distinguish "evidence frozen
# under an older exclusion list" (harness digest-scheme drift — re-freeze and
# re-evaluate) from genuine candidate-tree drift.  v1: original list;
# v2: +5 finalize runtime markers (fix 11); v3: + GE op_host delivery trio
# (fix 15).
CANDIDATE_DIGEST_SCHEME = "npubench-candidate-scope/v3"
PRECISION_CONTRACT_VERSION = "npubench-precision/v1"
PERFORMANCE_CONTRACT_VERSION = "npubench-performance-quick/v1"
EVIDENCE_DIRNAME = "npubench_evidence"
PRECISION_REPORT_FILENAME = "precision_report.json"
PERFORMANCE_REPORT_FILENAME = "performance_report.json"
EVALUATE_REPORT_FILENAME = "evaluate_report.json"
PREFLIGHT_REPORT_FILENAME = "preflight_report.json"
SNAPSHOT_DIRNAME = ".npubench_candidate"
NATIVE_PERF_FIXTURE_FILENAME = "native_perf_common.pt"
NATIVE_PERF_MANIFEST_FILENAME = "native_perf_manifest.json"
NATIVE_PERF_CASE_DIRNAME = "native_perf_cases"
NATIVE_PERF_FIXTURE_SCHEMA = "cannbot.npubench.native_perf_fixture/v1"
NATIVE_PERF_CASE_SCHEMA = "cannbot.npubench.native_perf_case/v1"
NATIVE_PERF_MANIFEST_SCHEMA = "cannbot.npubench.native_perf_manifest/v1"
INPUT_ADAPTER_CONTRACT_VERSION = "npubench-input-adapter/v1"
SIDECAR_DESCRIPTOR_SCHEMA = "cannbot.npubench.sidecar_descriptor/v1"
SIDECAR_DESCRIPTOR_ADAPTER = "sidecar_descriptor/v1"
_SIDECAR_DTYPE_ALIASES = {
    "float16": "float16",
    "float32": "float32",
    "bfloat16": "bfloat16",
}
_SIDECAR_FLOAT_DTYPES = frozenset({"float16", "float32", "bfloat16"})
_SIDECAR_INT_DTYPES = frozenset()
_SIDECAR_MAX_DIM = 1 << 20
_SIDECAR_MAX_ELEMENTS = 1 << 30
# These are descriptor-time limits.  They bound allocations without imposing
# the old, unsafe "materialize every case" policy.  The real level1/3_Add
# fixture is ~256 MiB for its largest case and ~1.2 GiB across 50 cases.
_SIDECAR_MAX_TENSOR_BYTES = 512 * 1024 * 1024
_SIDECAR_MAX_CASE_BYTES = 768 * 1024 * 1024
_SIDECAR_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
_SIDECAR_DTYPE_ITEMSIZE = {"float16": 2, "float32": 4, "bfloat16": 2}
DEFAULT_SEED = 0
WARM_UP = 3
REPEATS = 5
TASK_EXECUTION_TIMEOUT_SECONDS = 3600
PERF_SKIP_ENV = "CANNBOT_NPUBENCH_SKIP_PERF"
TASK_EXECUTION_TIMEOUT_ENV = "CANNBOT_NPUBENCH_TASK_TIMEOUT_SEC"
EXECUTION_DIRNAME = ".npubench_exec"

# Precision semantics are independently reimplemented from the following
# locally reviewed source -- it is deliberately *not* imported at runtime:
#
#   external verifier: verification_ascendc.py
#   sha256: 855649269af085ee34d093375bf67567b2c1d936ab9253977a163a29f4b6e9a9
#   reviewed semantic range: lines 62-873
#
# The source is relative to the cannbot-skills-shushu checkout supplied for
# this change.  Retaining the provenance here makes a future contract update
# auditable without adding a runtime dependency or copying that verifier.
PRECISION_SEMANTICS_SOURCE: Mapping[str, str] = {
    "relative_path": (
        "external verifier: verification_ascendc.py"
    ),
    "sha256": "855649269af085ee34d093375bf67567b2c1d936ab9253977a163a29f4b6e9a9",
    "reviewed_range": "62-873",
}

# This table is the executable table in that reviewed verifier.  Keeping the
# data local makes the contract auditable and avoids a runtime dependency on a
# user checkout of that plugin.
ALLCLOSE_TOLERANCES: Mapping[str, tuple[float, float]] = {
    "float32": (1.0e-3, 2 ** -13),
    "float16": (9.0e-2, 2 ** -10),
    "bfloat16": (1.0e-1, 2 ** -7),
}
NPU_LIMITS: Mapping[str, tuple[float, float, float]] = {
    "float16": (2 ** -11, 2 ** -16, 2 ** -10),
    "bfloat16": (2 ** -8, 2 ** -16, 2 ** -7),
    "float32": (2 ** -14, 2 ** -30, 2 ** -13),
    "hifloat32": (2 ** -12, 2 ** -28, 2 ** -11),
    "float8_e4m3fn": (2 ** -4, 2 ** -6, 2 ** -3),
    "float8_e4m3": (2 ** -4, 2 ** -6, 2 ** -3),
    "fp8_e4m3": (2 ** -4, 2 ** -6, 2 ** -3),
    "float8_e5m2fn": (2 ** -3, 2 ** -5, 2 ** -2),
    "float8_e5m2": (2 ** -3, 2 ** -5, 2 ** -2),
    "fp8_e5m2": (2 ** -3, 2 ** -5, 2 ** -2),
}
INT_LSB_TOLERANCE: Mapping[str, int] = {"int8": 1, "int16": 1}
REQUIRED_MATCHED_RATIO = 0.9


class NpuBenchRunnerError(RuntimeError):
    """A staged task, candidate, or evaluation contract is unsafe to use."""


def _resolve_task_execution_timeout(timeout_seconds: int | None) -> int:
    """Resolve one task timeout: explicit value, env override, then default.

    ``CANNBOT_NPUBENCH_TASK_TIMEOUT_SEC`` widens/narrows the 3600s default.
    The default was raised from 300s (2026-08-22 A5 campaign): attention-op
    performance runs (W3 + R5 + msprof) legitimately take 35-60 minutes and
    the 300s cap silently killed every perf phase; 1800s was then tried and
    also proved too tight, so the default is 3600s — a healthy card finishes
    inside it while a wedged lane costs at most one hour.  An invalid value
    is a configuration error and fails fast instead of being silently
    ignored.
    """
    if timeout_seconds is not None:
        return timeout_seconds
    raw = os.environ.get(TASK_EXECUTION_TIMEOUT_ENV)
    if raw is None:
        return TASK_EXECUTION_TIMEOUT_SECONDS
    value = raw.strip()
    if not value:
        raise NpuBenchRunnerError(
            f"{TASK_EXECUTION_TIMEOUT_ENV} must be a positive integer, got an empty value"
        )
    try:
        timeout = int(value)
    except ValueError as exc:
        raise NpuBenchRunnerError(
            f"{TASK_EXECUTION_TIMEOUT_ENV} must be a positive integer, got {raw!r}"
        ) from exc
    if timeout <= 0:
        raise NpuBenchRunnerError(
            f"{TASK_EXECUTION_TIMEOUT_ENV} must be greater than zero, got {raw!r}"
        )
    return timeout


@dataclass(frozen=True)
class StagedBundle:
    """Resolved, immutable old-format task bundle."""

    workspace: Path
    reference: Mapping[str, Any]
    manifest: Mapping[str, Any]
    manifest_path: Path
    root: Path
    task_path: Path
    sidecar_path: Path
    sidecar_encoding: str
    sidecar_cases: tuple[Any, ...]


@dataclass(frozen=True)
class _ExecutionContext:
    """Parent-owned paths for one independent native-evaluation child."""

    root: Path
    request_path: Path
    runner_root: Path
    scratch: Path
    tmp: Path
    bundle: StagedBundle
    candidate_root: Path | None
    binding: Mapping[str, Any]
    run_id: str
    verb: str
    device: int | str | None
    seed: int | None
    target_python: Path
    native_fixture_source: Path | None = None
    native_fixture_mount: Path | None = None


def tree_sha256(
    path: Path,
    *,
    exclude: Callable[[Path], bool] | None = None,
) -> str:
    """Hash a regular-file tree deterministically, rejecting links and devices.

    The digest includes each POSIX relative path and content digest, so moving
    a source file or replacing it with identical bytes changes the result.
    It is intentionally used for candidate provenance, not as an execution
    sandbox.
    """
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise NpuBenchRunnerError(f"tree root must be a real directory: {root}")
    entries: list[tuple[str, str]] = []
    # ``Path.rglob`` descends before a caller can exclude a directory.  Walk
    # explicitly so operational evidence cannot make a candidate digest drift
    # merely because O5 wrote a report under the same workspace.
    for directory_text, dirs, files in os.walk(root, topdown=True, followlinks=False):
        directory = Path(directory_text)
        kept_dirs: list[str] = []
        for name in sorted(dirs):
            item = directory / name
            relative = item.relative_to(root)
            if item.is_symlink():
                raise NpuBenchRunnerError(f"tree must not contain symlink: {item}")
            if exclude is not None and exclude(relative):
                continue
            kept_dirs.append(name)
        dirs[:] = kept_dirs
        for name in sorted(files):
            entry = directory / name
            relative = entry.relative_to(root)
            if entry.is_symlink():
                raise NpuBenchRunnerError(f"tree must not contain symlink: {entry}")
            if exclude is not None and exclude(relative):
                continue
            if not entry.is_file():
                raise NpuBenchRunnerError(f"tree must contain only regular files: {entry}")
            entries.append((relative.as_posix(), _file_sha256(entry)))
    return _canonical_sha256({"files": entries})


def profile_tree_sha256(path: Path) -> str:
    """Hash an archived profile with the finalize contract's byte protocol.

    This intentionally differs from the candidate tree digest for historical
    compatibility: finalization uses ``relative-path + NUL + file-sha + LF``.
    Both producer and verifier reject links/special files rather than silently
    omitting them from an apparently complete archive digest.
    """
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise NpuBenchRunnerError("profile archive root must be a real directory")
    entries: list[tuple[str, str]] = []
    for directory_text, dirs, files in os.walk(root, topdown=True, followlinks=False):
        directory = Path(directory_text)
        for name in sorted(dirs):
            item = directory / name
            if item.is_symlink():
                raise NpuBenchRunnerError(f"profile archive contains symlink: {item}")
        for name in sorted(files):
            item = directory / name
            if item.is_symlink() or not item.is_file():
                raise NpuBenchRunnerError(f"profile archive contains non-regular file: {item}")
            entries.append((item.relative_to(root).as_posix(), _file_sha256(item)))
    encoded = "".join(f"{relative}\0{digest}\n" for relative, digest in sorted(entries))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def materialize_candidate_snapshot(
    workspace: Path,
    candidate_dir: Path | None = None,
) -> Path:
    """Freeze candidate-relevant files into a read-only content-addressed tree.

    O5 commonly passes the workspace itself as ``candidate_dir``.  That tree is
    also where state, profiler adapters and evidence are written, so hashing it
    directly makes an evaluator race its own outputs.  This function freezes
    only the candidate scope *before* evaluation into
    ``.npubench_candidate/<digest>/``; callers should then pass that returned
    directory to precision/performance/evaluate.  The snapshot excludes only
    operational paths, not source files, and rejects links/special files.
    """
    supplied_workspace = Path(workspace)
    if supplied_workspace.is_symlink():
        raise NpuBenchRunnerError("workspace must be a real non-symlink directory")
    workspace = supplied_workspace.resolve()
    _require_real_directory(workspace, "workspace")
    source = _candidate_root(candidate_dir if candidate_dir is not None else workspace)
    _candidate_entry(source)
    source_digest = _candidate_tree_sha256(source)
    snapshot_parent = _workspace_runtime_directory(
        workspace, SNAPSHOT_DIRNAME, "candidate snapshot root"
    )
    destination = snapshot_parent / source_digest
    if destination.exists():
        _require_snapshot(destination, source_digest)
        return destination
    incoming_parent = _ensure_real_child_directory(
        snapshot_parent, ".incoming", "candidate snapshot incoming root"
    )
    incoming = _create_real_child_directory(
        incoming_parent, uuid.uuid4().hex, "candidate snapshot incoming"
    )
    try:
        _copy_candidate_scope(source, incoming)
        if _candidate_tree_sha256(incoming) != source_digest:
            raise NpuBenchRunnerError("candidate snapshot digest differs after copy")
        if _candidate_tree_sha256(source) != source_digest:
            raise NpuBenchRunnerError("candidate source changed while snapshot was copied")
        try:
            os.replace(incoming, destination)
        except FileExistsError:
            _require_snapshot(destination, source_digest)
        else:
            _make_tree_read_only(destination)
        _require_snapshot(destination, source_digest)
        return destination
    finally:
        if incoming.exists():
            shutil.rmtree(incoming, ignore_errors=True)


def stage_workspace(
    workspace: Path,
    *,
    task_path: Path,
    root: Path | None = None,
) -> dict[str, Any]:
    """Delegate immutable staging to P1 without mutating durable state here.

    ``npubench_inputs`` owns all state transaction semantics.  This runner
    merely provides the CLI verb and a friendly error if an incomplete install
    tries to invoke it before that provider module is available.
    """
    try:
        inputs = _load_inputs_provider()
        stager = getattr(inputs, "stage_npubench_inputs", None)
        if not callable(stager):
            raise NpuBenchRunnerError(
                "npubench_inputs.stage_npubench_inputs is unavailable; "
                "install the npubench provider before using runner stage"
            )
        stage = stager(
            Path(workspace),
            npubench_task=Path(task_path),
            npubench_root=Path(root) if root is not None else None,
        )
        payload = {
            "schema": "cannbot.npubench.stage/v1",
            "status": "PASS",
            "runner_contract_version": RUNNER_CONTRACT_VERSION,
            "stage_root": str(getattr(stage, "root", "")),
            "bundle_sha256": getattr(stage, "bundle_sha256", None),
            "manifest_sha256": getattr(stage, "manifest_sha256", None),
        }
    except (NpuBenchRunnerError, OSError, ValueError) as exc:
        payload = _base_report("stage", status="ERROR")
        payload["reason"] = str(exc)
    return payload


def resolve_staged_bundle(workspace: Path) -> StagedBundle:
    """Resolve a P1-validated bundle into real paths and sidecar records."""
    workspace = Path(workspace)
    state = _read_state(workspace)
    reference = state.get("reference")
    if not isinstance(reference, Mapping) or reference.get("source") != NPUBENCH_SOURCE:
        raise NpuBenchRunnerError("durable state does not select npubench")

    # P1 is the authority for immutable stage validation.  Keep the import
    # lazy so reading this module never brings in provider/FSM/A3 code.
    inputs = _load_inputs_provider()
    verifier = getattr(inputs, "verify_npubench_stage", None)
    if not callable(verifier):
        raise NpuBenchRunnerError(
            "npubench_inputs.verify_npubench_stage is unavailable; "
            "cannot trust a staged bundle"
        )
    verified = verifier(workspace, reference)
    if not isinstance(verified, tuple) or len(verified) != 3:
        raise NpuBenchRunnerError("npubench stage verifier returned an invalid result")
    ok, reason, manifest = verified
    if not ok:
        raise NpuBenchRunnerError("staged npubench bundle rejected: " + str(reason))
    if not isinstance(manifest, Mapping):
        raise NpuBenchRunnerError("staged npubench bundle has no manifest object")

    manifest_path = _resolve_manifest_path(workspace, reference)
    root = manifest_path.parent
    task_relative = _required_relative_path(
        reference.get("task_relative_path", manifest.get("task_relative_path")),
        "task_relative_path",
    )
    sidecar_relative = _required_relative_path(
        reference.get("sidecar_relative_path", manifest.get("sidecar_relative_path")),
        "sidecar_relative_path",
    )
    task_path = _resolve_under(root, task_relative, "staged task")
    sidecar_path = _resolve_under(root, sidecar_relative, "staged sidecar")
    if task_path.suffix != ".py":
        raise NpuBenchRunnerError("staged task must be a .py file")
    if sidecar_path.suffix not in {".json", ".jsonl"}:
        raise NpuBenchRunnerError("staged sidecar must be .json or .jsonl")
    _require_regular(task_path, "staged task")
    _require_regular(sidecar_path, "staged sidecar")
    encoding, cases = parse_sidecar(sidecar_path)
    return StagedBundle(
        workspace=workspace,
        reference=reference,
        manifest=manifest,
        manifest_path=manifest_path,
        root=root,
        task_path=task_path,
        sidecar_path=sidecar_path,
        sidecar_encoding=encoding,
        sidecar_cases=tuple(cases),
    )


def _read_execution_request_payload(path: Path) -> Mapping[str, Any]:
    """Read and schema-check the parent-authored request document."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NpuBenchRunnerError(f"execution request is unreadable: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise NpuBenchRunnerError("execution request schema is invalid")
    if payload.get("schema") != "cannbot.npubench.execution_request/v1":
        raise NpuBenchRunnerError("execution request schema is invalid")
    return payload


def _resolve_request_manifest(bundle_data: Mapping[str, Any]) -> tuple[Path, Path, Mapping[str, Any]]:
    """Resolve the frozen bundle root and its digest-checked manifest object."""
    root = Path(str(bundle_data.get("root", "")))
    manifest_path = Path(str(bundle_data.get("manifest_path", "")))
    _require_real_read_only_tree(root, "isolated frozen bundle")
    _require_regular(manifest_path, "isolated bundle manifest")
    if _file_sha256(manifest_path) != bundle_data.get("manifest_sha256"):
        raise NpuBenchRunnerError("isolated bundle manifest digest differs from request")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NpuBenchRunnerError("isolated bundle manifest is unreadable") from exc
    if not isinstance(manifest, Mapping):
        raise NpuBenchRunnerError("isolated bundle manifest is not an object")
    return root, manifest_path, manifest


def _resolve_request_task_paths(
    root: Path, bundle_data: Mapping[str, Any]
) -> tuple[Path, Path, str, list[Any]]:
    """Resolve and digest-check the frozen task/sidecar named by the request."""
    task_relative = _required_relative_path(bundle_data.get("task_relative_path"), "request task_relative_path")
    sidecar_relative = _required_relative_path(
        bundle_data.get("sidecar_relative_path"),
        "request sidecar_relative_path",
    )
    task_path = _resolve_under(root, task_relative, "isolated task")
    sidecar_path = _resolve_under(root, sidecar_relative, "isolated sidecar")
    _require_regular(task_path, "isolated task")
    _require_regular(sidecar_path, "isolated sidecar")
    if _file_sha256(task_path) != bundle_data.get("task_sha256"):
        raise NpuBenchRunnerError("isolated task digest differs from request")
    if _file_sha256(sidecar_path) != bundle_data.get("sidecar_sha256"):
        raise NpuBenchRunnerError("isolated sidecar digest differs from request")
    encoding, cases = parse_sidecar(sidecar_path)
    if encoding != bundle_data.get("sidecar_encoding"):
        raise NpuBenchRunnerError("isolated sidecar encoding differs from request")
    return task_path, sidecar_path, encoding, cases


def _assert_request_binding(binding: Mapping[str, Any], task_path: Path, sidecar_path: Path) -> None:
    """Reject a request whose binding is unsigned or unbound to the frozen input."""
    if binding.get("source") != NPUBENCH_SOURCE:
        raise NpuBenchRunnerError("isolated execution binding source is not npubench")
    without_digest = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if binding.get("binding_sha256") != _canonical_sha256(without_digest):
        raise NpuBenchRunnerError("isolated execution binding digest is invalid")
    task_matches = binding.get("task_sha256") == _file_sha256(task_path)
    sidecar_matches = binding.get("sidecar_sha256") == _file_sha256(sidecar_path)
    if not task_matches or not sidecar_matches:
        raise NpuBenchRunnerError("isolated execution binding does not match frozen input")


def _resolve_request_bundle(
    bundle_data: Mapping[str, Any], binding: Mapping[str, Any], scratch: Path
) -> StagedBundle:
    """Rebuild the staged bundle from a request without reading durable state."""
    root, manifest_path, manifest = _resolve_request_manifest(bundle_data)
    task_path, sidecar_path, encoding, cases = _resolve_request_task_paths(root, bundle_data)
    _assert_request_binding(binding, task_path, sidecar_path)
    return StagedBundle(
        workspace=scratch,
        reference={
            "source": NPUBENCH_SOURCE,
            "bundle_manifest_sha256": _file_sha256(manifest_path),
            "bundle_sha256": binding.get("bundle_sha256"),
        },
        manifest=manifest,
        manifest_path=manifest_path,
        root=root,
        task_path=task_path,
        sidecar_path=sidecar_path,
        sidecar_encoding=encoding,
        sidecar_cases=tuple(cases),
    )


def _resolve_request_candidate(candidate_data: Any, binding: Mapping[str, Any]) -> Path:
    """Resolve the request's candidate snapshot and bind it to the digest."""
    if not isinstance(candidate_data, Mapping):
        raise NpuBenchRunnerError("isolated candidate request is invalid")
    candidate = _candidate_root(Path(str(candidate_data.get("root", ""))))
    _require_snapshot(candidate, str(candidate_data.get("tree_sha256", "")))
    entry = _candidate_entry(candidate)
    if _file_sha256(entry) != candidate_data.get("entry_sha256"):
        raise NpuBenchRunnerError("isolated candidate entry digest differs from request")
    if binding.get("candidate_tree_sha256") != candidate_data.get("tree_sha256"):
        raise NpuBenchRunnerError("isolated candidate binding digest differs from request")
    return candidate


def _resolve_request_fixture(fixture_data: Any) -> Path:
    """Resolve the parent-frozen native performance fixture named by a request."""
    if not isinstance(fixture_data, Mapping):
        raise NpuBenchRunnerError("isolated native fixture request is invalid")
    fixture_root = Path(str(fixture_data.get("root", "")))
    _require_real_read_only_tree(fixture_root, "isolated native performance fixture")
    expected_tree = fixture_data.get("tree_sha256")
    expected_manifest = fixture_data.get("manifest_sha256")
    if not isinstance(expected_tree, str) or tree_sha256(fixture_root) != expected_tree:
        raise NpuBenchRunnerError("isolated native fixture tree digest differs from request")
    fixture_manifest_path = fixture_root / NATIVE_PERF_MANIFEST_FILENAME
    _require_regular(fixture_manifest_path, "isolated native fixture manifest")
    if not isinstance(expected_manifest, str) or _file_sha256(fixture_manifest_path) != expected_manifest:
        raise NpuBenchRunnerError("isolated native fixture manifest digest differs from request")
    return fixture_root


def _resolve_execution_request(
    path: Path,
) -> tuple[StagedBundle, Path | None, Mapping[str, Any], Mapping[str, Any], Path | None]:
    """Validate a parent-authored, state-free request in the evaluation child."""
    payload = _read_execution_request_payload(path)
    scratch = _request_scratch(payload)
    bundle_data = payload.get("bundle")
    binding = payload.get("binding")
    if not isinstance(bundle_data, Mapping) or not isinstance(binding, Mapping):
        raise NpuBenchRunnerError("execution request lacks bundle or binding")
    bundle = _resolve_request_bundle(bundle_data, binding, scratch)
    candidate_data = payload.get("candidate")
    candidate: Path | None = None
    if candidate_data is not None:
        candidate = _resolve_request_candidate(candidate_data, binding)
    fixture_data = payload.get("native_fixture")
    fixture_root: Path | None = None
    if fixture_data is not None:
        fixture_root = _resolve_request_fixture(fixture_data)
    return bundle, candidate, dict(binding), payload, fixture_root


def _request_scratch(request: Mapping[str, Any]) -> Path:
    """Return the parent-authored execution scratch directory from a request."""
    value = request.get("scratch")
    if not isinstance(value, str) or not value:
        raise NpuBenchRunnerError("execution request has no scratch directory")
    path = Path(value)
    _require_real_directory(path, "execution request scratch")
    return path


def _validate_parent_published_child_report(
    report: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    verb: str,
    expected_run_id: str | None = None,
) -> None:
    """Reject child output unless it is bound to the parent's frozen request."""
    if not isinstance(report, Mapping):
        raise NpuBenchRunnerError("evaluation child returned a non-object report")
    if report.get("schema") != f"cannbot.npubench.{verb}/v1":
        raise NpuBenchRunnerError("evaluation child returned the wrong report schema")
    if report.get("binding_sha256") != binding.get("binding_sha256"):
        raise NpuBenchRunnerError("evaluation child report binding digest differs from parent binding")
    if report.get("evaluation_binding") != dict(binding):
        raise NpuBenchRunnerError("evaluation child report binding payload differs from parent binding")
    if report.get("status") not in {"PASS", "FAIL", "ERROR"}:
        raise NpuBenchRunnerError("evaluation child report status is invalid")
    if expected_run_id is not None and report.get("run_id") != expected_run_id:
        raise NpuBenchRunnerError("evaluation child report run_id differs from parent request")


def _verify_parent_binding_unchanged(
    workspace: Path, candidate_dir: Path | None, expected: Mapping[str, Any]
) -> None:
    """Recompute from parent-visible immutable paths before publishing evidence."""
    bundle = resolve_staged_bundle(workspace)
    actual = build_evaluation_binding(workspace, candidate_dir, bundle=bundle)
    if actual != dict(expected):
        raise NpuBenchRunnerError("frozen bundle or candidate changed during isolated execution")


def parse_sidecar(path: Path) -> tuple[str, list[Any]]:
    """Strictly parse a sidecar, accepting JSONL even under a ``.json`` name."""
    path = Path(path)
    _require_regular(path, "sidecar")
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise NpuBenchRunnerError(f"sidecar is not UTF-8: {path.name}") from exc
    if not raw.strip():
        raise NpuBenchRunnerError(f"sidecar contains no cases: {path.name}")

    if path.suffix == ".jsonl":
        return "jsonl", _parse_jsonl(raw, path)
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        # Old benchmark tasks commonly call their JSONL sidecar ``.json``.
        return "jsonl", _parse_jsonl(raw, path)
    if isinstance(document, list):
        cases = document
    else:
        cases = [document]
    if not cases:
        raise NpuBenchRunnerError(f"sidecar contains no cases: {path.name}")
    return "json", cases


def build_evaluation_binding(
    workspace: Path,
    candidate_dir: Path | None = None,
    *,
    bundle: StagedBundle | None = None,
) -> dict[str, Any]:
    """Return the immutable input/candidate/contract binding and its digest."""
    bundle = bundle or resolve_staged_bundle(Path(workspace))
    manifest_sha256 = _file_sha256(bundle.manifest_path)
    expected_manifest = bundle.reference.get("bundle_manifest_sha256")
    if isinstance(expected_manifest, str) and expected_manifest != manifest_sha256:
        raise NpuBenchRunnerError("bundle manifest digest differs from durable binding")
    binding: dict[str, Any] = {
        "source": NPUBENCH_SOURCE,
        "bundle_manifest_sha256": manifest_sha256,
        "bundle_sha256": bundle.reference.get("bundle_sha256", bundle.manifest.get("bundle_sha256")),
        "task_relative_path": str(bundle.task_path.relative_to(bundle.root)),
        "task_sha256": _file_sha256(bundle.task_path),
        "sidecar_relative_path": str(bundle.sidecar_path.relative_to(bundle.root)),
        "sidecar_sha256": _file_sha256(bundle.sidecar_path),
        "precision_contract": _precision_contract(),
        "precision_contract_sha256": _canonical_sha256(_precision_contract()),
        "performance_contract": _performance_contract(),
        "performance_contract_sha256": _canonical_sha256(_performance_contract()),
        "input_adapter_contract": _input_adapter_contract(),
        "input_adapter_contract_sha256": _canonical_sha256(_input_adapter_contract()),
        "input_adapter": _input_adapter_binding(bundle),
        "runner_contract_version": RUNNER_CONTRACT_VERSION,
        "candidate_digest_scheme": CANDIDATE_DIGEST_SCHEME,
    }
    if candidate_dir is not None:
        candidate_root = _candidate_root(candidate_dir)
        frozen_snapshot = _is_frozen_candidate_snapshot(Path(workspace), candidate_root)
        if frozen_snapshot:
            # A content-addressed snapshot proves only that its bytes match its
            # directory name.
            _require_snapshot(candidate_root, candidate_root.name)
        candidate_entry = _candidate_entry(candidate_root)
        binding.update(
            {
                "candidate_tree_sha256": _candidate_tree_sha256(candidate_root),
                "candidate_entry": candidate_entry.name,
                "candidate_entry_sha256": _file_sha256(candidate_entry),
            }
        )
    binding["binding_sha256"] = _canonical_sha256(binding)
    return binding


def _copy_regular_tree(source: Path, destination: Path) -> None:
    """Copy a verified tree without following links or preserving write bits."""
    source, destination = Path(source), Path(destination)
    _require_real_directory(source, "tree copy source")
    for directory_text, dirs, files in os.walk(source, topdown=True, followlinks=False):
        directory = Path(directory_text)
        relative_dir = directory.relative_to(source)
        target_dir = destination / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        kept_dirs: list[str] = []
        for name in sorted(dirs):
            item = directory / name
            if item.is_symlink() or not item.is_dir():
                raise NpuBenchRunnerError("tree copy source contains unsafe directory entry")
            kept_dirs.append(name)
        dirs[:] = kept_dirs
        for name in sorted(files):
            item = directory / name
            if item.is_symlink() or not item.is_file():
                raise NpuBenchRunnerError("tree copy source contains unsafe file entry")
            target = target_dir / name
            shutil.copyfile(item, target)


def _perf_below_threshold(summary: Any) -> bool:
    """True when a completed measurement reports a geomean speedup below 1.0.

    Perf-status semantics (2026-08-23 SDPA perf backfill, C26): the quick
    profiler status=PASS means "measurement completed", NOT "perf met a
    threshold".  Callers stamp this annotation so a below-1.0 speedup is never
    misread as a perf pass.
    """
    geomean = summary.get("geomean_speedup") if isinstance(summary, Mapping) else None
    numeric = isinstance(geomean, (int, float)) and not isinstance(geomean, bool)
    return numeric and float(geomean) < 1.0


def _performance_adapter_path(context: _ExecutionContext) -> Path:
    adapter = context.scratch / ".npubench_adapter" / str(context.binding["binding_sha256"]) / context.run_id
    _require_real_directory(adapter, "isolated performance adapter")
    return adapter


def _parent_performance_command(*, device: int, run_id: str) -> list[str]:
    """Publish a semantic command record without child-controlled host paths."""
    return [
        "python3",
        "ops/ops-profiling/scripts/msprof_perf_summary.py",
        "--quick",
        "--warmup",
        str(WARM_UP),
        "--device",
        str(device),
        "--keep-prof",
        "--repeats",
        str(REPEATS),
        "--prof-tag",
        run_id,
        "--output-dir=<native-profiler-adapter>",
    ]


def _parent_performance_report_fields(
    *,
    device: int,
    lease_manifest: Path | None,
    command: Sequence[str],
    returncode: int,
) -> dict[str, Any]:
    fields = {
        "device": device,
        "lease_manifest": str(lease_manifest) if lease_manifest else None,
        "profiling_mode": "quick",
        "warm_up": WARM_UP,
        "repeats": REPEATS,
        "keep_prof": True,
        "command": list(command),
        "returncode": returncode,
        "child_returncode": returncode,
        # This is intentionally a process boundary plus parent-side digest
        # verification, not an adversarial OS sandbox claim.  The fixture
        # phase is also a separate process and its bytes are frozen/checked
        # before B, then rechecked after B returns.
        "execution_isolation": "process_boundary",
        "tamper_protection": "post_run_hash_check",
        "fixture_execution_isolation": "process_boundary",
        "fixture_tamper_protection": "post_run_hash_check",
    }
    return fields


def _safe_child_failure_reason(report: Mapping[str, Any]) -> str:
    reason = report.get("reason")
    if isinstance(reason, str) and reason:
        return _output_tail(reason, limit=2048)
    return "isolated performance child did not complete successfully"


def _cleanup_execution_context(context: _ExecutionContext) -> None:
    """Remove transient fixture/adapter/profile data after parent publication."""
    root = Path(context.root).resolve()
    expected_parent = (context.bundle.workspace / EXECUTION_DIRNAME).resolve()
    try:
        root.relative_to(expected_parent)
    except ValueError:
        return
    if root.is_symlink() or not root.is_dir():
        return
    try:
        # The child-facing runner/fixture trees are deliberately read-only;
        # restore ownership-local permissions only after evidence publication
        # so their transient context can be removed without retaining 3_Add
        # sized chunks between evaluations.
        for entry in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if entry.is_symlink():
                return
            os.chmod(entry, 0o700 if entry.is_dir() else 0o600)
        shutil.rmtree(root)
    except OSError:
        # Runtime leftovers are excluded from the candidate digest and never
        # accepted as evidence.  A later housekeeping pass may remove them.
        pass


def _subreport_failure_reason(report: Mapping[str, Any]) -> str | None:
    """Return an errored/failed sub-report's own reason, when it carries one.

    A lane report that already failed often lacks the complete binding fields,
    so surfacing its own reason keeps the real cause (missing PyInit, fixture
    rejection, ...) from being masked by a generic binding-digest message.
    """
    reason = report.get("reason")
    if report.get("status") in {"ERROR", "FAIL"} and isinstance(reason, str) and reason:
        return reason
    return None


def _deferred_performance_report(
    *, binding: Mapping[str, Any], run_id: str, device: int
) -> dict[str, Any]:
    """Placeholder performance evidence for the precision-first mode.

    CANNBOT_NPUBENCH_SKIP_PERF=1 defers the slow W3/R5 msprof profiling (each
    fresh profiler process pays a full device init, which on a degraded host
    takes minutes) while banking precision evidence.  The report keeps the
    canonical schema + binding so the finalize contract can verify it, and
    carries perf_deferred=True so the contract skips the perf-specific gates.
    """
    report = _base_report("performance", status="DEFERRED", binding=binding, run_id=run_id)
    report.update({
        "reason": (
            "performance deferred: CANNBOT_NPUBENCH_SKIP_PERF=1 "
            "(precision-first mode; perf to be backfilled later)"
        ),
        "device": device,
        "profiling_mode": "quick",
        "warm_up": WARM_UP,
        "repeats": REPEATS,
        "keep_prof": True,
        "perf_deferred": True,
        "native_fixture": {},
        "profile_archive": None,
        "profile_tree_sha256": None,
        "profiler_summary": None,
        "command": [],
    })
    return report


def _evaluate_lane_verdict(
    precision: Mapping[str, Any], performance: Mapping[str, Any], binding_sha: Any
) -> tuple[str, str]:
    """Combine two lane reports into the aggregate status and reason."""
    if binding_sha is None or performance.get("binding_sha256") != binding_sha:
        lane_reason = _subreport_failure_reason(precision) or _subreport_failure_reason(performance)
        reason = lane_reason or "precision and performance evidence do not share a binding digest"
        return "ERROR", reason
    if precision.get("status") != "PASS":
        return "FAIL", "one or more evaluation lanes did not pass"
    if performance.get("status") == "PASS":
        return "PASS", ""
    if performance.get("status") == "DEFERRED":
        return "PASS", "performance deferred (precision-first mode)"
    return "FAIL", "one or more evaluation lanes did not pass"


def verify_evidence_report(
    workspace: Path,
    report: Mapping[str, Any],
    *,
    expected_verb: str,
    candidate_dir: Path | None = None,
) -> tuple[bool, str]:
    """Recompute binding and validate the small common report contract for finalize."""
    if not isinstance(report, Mapping):
        return False, "npubench evidence is not an object"
    if report.get("schema") != f"cannbot.npubench.{expected_verb}/v1":
        return False, "npubench evidence schema mismatch"
    if report.get("runner_contract_version") != RUNNER_CONTRACT_VERSION:
        return False, "npubench runner contract mismatch"
    if report.get("status") not in {"PASS", "FAIL", "ERROR", "PLANNED", "DEFERRED"}:
        return False, "npubench evidence status is invalid"
    try:
        expected = build_evaluation_binding(Path(workspace), candidate_dir)
    except (NpuBenchRunnerError, OSError, ValueError) as exc:
        return False, f"cannot recompute npubench binding: {exc}"
    if report.get("binding_sha256") != expected["binding_sha256"]:
        lane_reason = _subreport_failure_reason(report)
        if lane_reason is not None:
            # A failed lane report does not carry the complete binding fields;
            # bubble its own reason so the real cause is not masked by the
            # generic digest message.
            return False, lane_reason
        return False, "npubench evidence binding digest mismatch"
    if report.get("evaluation_binding") != expected:
        return False, "npubench evidence binding payload mismatch"
    return True, "npubench evidence binding verified"


def _load_flat_inputs_provider() -> types.ModuleType:
    """Load the sibling input provider used by an isolated flat runner copy."""
    try:
        return importlib.import_module("npubench_inputs")
    except ModuleNotFoundError as exc:
        if exc.name == "npubench_inputs":
            raise NpuBenchRunnerError(
                "npubench input provider is not installed; cannot validate immutable task input"
            ) from exc
        raise exc


def _load_inputs_provider() -> types.ModuleType:
    try:
        return importlib.import_module("npubench.npubench_inputs")
    except ModuleNotFoundError as exc:
        if exc.name == "npubench":
            # Staged flat runner copy: no package layout there, the sibling
            # files sit directly on sys.path (the runner script's directory).
            return _load_flat_inputs_provider()
        if exc.name == "npubench.npubench_inputs":
            raise NpuBenchRunnerError(
                "npubench input provider is not installed; cannot validate immutable task input"
            ) from exc
        raise exc


def _resolve_manifest_path(workspace: Path, reference: Mapping[str, Any]) -> Path:
    # P1 intentionally permits an absolute controller-side value in durable
    # state so the same content-addressed stage can be rebased onto an A5
    # target workspace.  Do not dereference that untrusted absolute string:
    # P1 has already checked its fixed suffix, and the runner derives the only
    # executable path from *this* workspace plus the bound bundle digest.
    bundle_sha256 = reference.get("bundle_sha256")
    if (
        not isinstance(bundle_sha256, str)
        or len(bundle_sha256) != 64
        or any(character not in "0123456789abcdef" for character in bundle_sha256)
    ):
        raise NpuBenchRunnerError("bundle_sha256 is invalid")
    path = (
        Path(workspace)
        / "reference_inputs"
        / NPUBENCH_SOURCE
        / bundle_sha256
        / "bundle_manifest.json"
    )
    # Test doubles and old development workspaces may use a non-content
    # addressed suffix.  Never use an arbitrary state path for that fallback:
    # only accept the manifest P1 returned after verifying the entire stage.
    if not path.exists():
        raw = reference.get("bundle_manifest_path")
        if isinstance(raw, str) and not Path(raw).is_absolute():
            relative = _required_relative_path(raw, "bundle_manifest_path")
            path = _resolve_under(workspace, relative, "bundle manifest")
    _require_regular(path, "bundle manifest")
    # Keep the manifest root in one canonical spelling.  On macOS `/var` is
    # commonly an alias of `/private/var`; `_resolve_under()` canonicalizes
    # task/sidecar descendants, so returning the raw manifest parent here
    # makes a legitimate task look outside its own bundle during `relative_to`.
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NpuBenchRunnerError(f"cannot resolve bundle manifest: {exc}") from exc


def _required_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise NpuBenchRunnerError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise NpuBenchRunnerError(f"{label} must be a containment-safe relative path")
    return path


def _resolve_under(root: Path, relative: Path, label: str) -> Path:
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise NpuBenchRunnerError(f"{label} escapes its staged root") from exc
    return candidate


def _require_regular(path: Path, label: str) -> None:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise NpuBenchRunnerError(f"{label} must be a regular non-symlink file: {path}")


def _require_real_directory(path: Path, label: str) -> None:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise NpuBenchRunnerError(f"{label} must be a real non-symlink directory: {path}")


def _assert_safe_child_name(name: Any, label: str) -> None:
    """Reject a child name that could escape or re-enter its parent directory."""
    named = isinstance(name, str) and bool(name)
    traversal = not named or name in {".", ".."}
    if traversal or "/" in name or "\\" in name:
        raise NpuBenchRunnerError(f"{label} child name is unsafe")


def _ensure_real_child_directory(parent: Path, name: str, label: str, *, mode: int = 0o700) -> Path:
    """Create/reuse one direct real directory child without following links.

    Runtime roots live below a user-selected workspace, so ``mkdir(parents,
    exist_ok=True)`` is not safe here: a preexisting ``npubench_evidence`` or
    ``.npubench_exec`` symlink would redirect writes outside the workspace.
    Each hop is checked before it is used as the parent of the next one.
    """
    parent = Path(parent)
    _require_real_directory(parent, f"{label} parent")
    _assert_safe_child_name(name, label)
    child = parent / name
    if child.exists() or child.is_symlink():
        _require_real_directory(child, label)
        return child
    try:
        os.mkdir(child, mode=mode)
    except FileExistsError:
        _require_real_directory(child, label)
    _require_real_directory(child, label)
    return child


def _create_real_child_directory(parent: Path, name: str, label: str, *, mode: int = 0o700) -> Path:
    """Create a new direct child, rejecting collisions and symlink redirection."""
    parent = Path(parent)
    _require_real_directory(parent, f"{label} parent")
    _assert_safe_child_name(name, label)
    child = parent / name
    try:
        os.mkdir(child, mode=mode)
    except FileExistsError as exc:
        raise NpuBenchRunnerError(f"{label} already exists") from exc
    _require_real_directory(child, label)
    return child


def _workspace_runtime_directory(workspace: Path, name: str, label: str) -> Path:
    """Return a trusted direct runtime root below a real workspace."""
    supplied = Path(workspace)
    if supplied.is_symlink():
        raise NpuBenchRunnerError("workspace must be a real non-symlink directory")
    root = supplied.resolve()
    _require_real_directory(root, "workspace")
    return _ensure_real_child_directory(root, name, label)


def _parse_jsonl(raw: str, path: Path) -> list[Any]:
    cases: list[Any] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise NpuBenchRunnerError(
                f"sidecar JSONL is invalid at line {number}: {exc.msg}"
            ) from exc
    if not cases:
        raise NpuBenchRunnerError(f"sidecar contains no JSONL cases: {path.name}")
    return cases


def _input_adapter_contract() -> dict[str, Any]:
    """Describe the evaluator-owned sidecar input materializer."""
    return {
        "version": INPUT_ADAPTER_CONTRACT_VERSION,
        "sidecar_schema": SIDECAR_DESCRIPTOR_SCHEMA,
        "adapter": SIDECAR_DESCRIPTOR_ADAPTER,
        "supported_tensor_dtypes": sorted(_SIDECAR_FLOAT_DTYPES),
        "tensor_shape": (
            "positive integer list; each dimension <= 1048576 and "
            "elements <= 1073741824"
        ),
        "tensor_allocation_limits": {
            "max_tensor_bytes": _SIDECAR_MAX_TENSOR_BYTES,
            "max_case_bytes": _SIDECAR_MAX_CASE_BYTES,
            "max_total_descriptor_bytes": _SIDECAR_MAX_TOTAL_BYTES,
        },
        "tensor_range": "finite floating uniform bounds [low, high]",
        "supported_attr": {
            "dtype": "float|int|bool|str",
            "value": "finite scalar matching dtype",
        },
        "ordering": "inputs list order",
        "seed": "case-local CPU generator seeded by global seed and case index",
    }


def _input_adapter_binding(bundle: StagedBundle) -> Mapping[str, Any]:
    """Classify the immutable task without executing it for binding creation."""
    try:
        tree = ast.parse(bundle.task_path.read_text(encoding="utf-8"), filename=str(bundle.task_path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise NpuBenchRunnerError(f"cannot inspect task input API for binding: {exc}") from exc
    # The runtime API resolves attributes exported by the module itself.  Do
    # not let nested functions, conditional dead code, or class methods alter
    # the immutable binding: only unconditional module-level bindings can be
    # classified without executing an untrusted task.
    names: set[str] = set()
    for node in tree.body:
        names.update(_input_provider_names_bound_by(node))
    if "get_input_groups" in names:
        selected = "get_input_groups"
    elif "get_inputs" in names:
        selected = "get_inputs"
    else:
        selected = "sidecar_descriptor"
    return _input_adapter_identity(
        selected,
        case_count=len(bundle.sidecar_cases) if selected == "sidecar_descriptor" else None,
    )


def _input_provider_names_bound_by(node: ast.stmt) -> set[str]:
    """Return the input-provider names one module-level statement binds.

    Only unconditional module-level bindings can be classified without
    executing an untrusted task, so nested functions, conditional dead code
    and class methods are deliberately invisible here.
    """
    wanted = {"get_input_groups", "get_inputs"}
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {node.name} & wanted
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        bound = {target.id for target in targets if isinstance(target, ast.Name)}
        return bound & wanted
    if isinstance(node, (ast.ImportFrom, ast.Import)):
        names: set[str] = set()
        for alias in node.names:
            imported = alias.name.rsplit(".", 1)[-1]
            names.add(alias.asname or imported)
        return names & wanted
    return set()


def _positive_case_count(case_count: Any) -> bool:
    """True when ``case_count`` is a positive, non-boolean integer."""
    numeric = not isinstance(case_count, bool) and isinstance(case_count, int)
    return numeric and case_count > 0


def _input_adapter_identity(provider: str, *, case_count: int | None) -> dict[str, Any]:
    """Return the canonical adapter identity used at every evaluation phase."""
    if provider == "sidecar_descriptor" and not _positive_case_count(case_count):
        raise NpuBenchRunnerError("input adapter case_count must be a positive integer")
    if case_count is not None and not _positive_case_count(case_count):
        raise NpuBenchRunnerError("input adapter case_count must be null or a positive integer")
    if provider == "sidecar_descriptor":
        kind = SIDECAR_DESCRIPTOR_ADAPTER
        provider_value = None
        schema = SIDECAR_DESCRIPTOR_SCHEMA
    elif provider in {"get_input_groups", "get_inputs"}:
        kind = provider
        provider_value = provider
        schema = None
    else:
        raise NpuBenchRunnerError("unsupported input adapter provider")
    return {
        "kind": kind,
        "provider": provider_value,
        "contract": INPUT_ADAPTER_CONTRACT_VERSION,
        "schema": schema,
        "case_count": case_count,
    }


def _candidate_root(candidate_dir: Path) -> Path:
    supplied = Path(candidate_dir)
    if supplied.is_symlink() or not supplied.is_dir():
        raise NpuBenchRunnerError("candidate_dir must be a real directory")
    try:
        return supplied.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NpuBenchRunnerError("candidate_dir cannot be resolved") from exc


def _is_frozen_candidate_snapshot(workspace: Path, candidate_root: Path) -> bool:
    """Recognize only the canonical content-addressed candidate snapshot."""
    workspace = Path(workspace).resolve()
    candidate_root = Path(candidate_root).resolve()
    snapshot_parent = (workspace / SNAPSHOT_DIRNAME).resolve()
    if candidate_root.parent != snapshot_parent:
        return False
    name = candidate_root.name
    return len(name) == 64 and all(character in "0123456789abcdef" for character in name)


_CANDIDATE_RUNTIME_TOP_LEVEL = frozenset(
    {
        "reference_inputs",
        "npubench_evidence",
        ".npubench_adapter",
        EXECUTION_DIRNAME,
        SNAPSHOT_DIRNAME,
        ".incoming",
        ".git",
        ".opgen_state.json",
        # O5 publishes this controller-owned summary after the candidate has
        # been frozen.  It is evidence, not candidate source; including it
        # would make the current-scope digest drift before finalization.
        "verification.json",
        ".npubench_candidate_repair.json",
        "LEASES.json",
        "leases.json",
        "logs",
        ".lingxi_verify_logs",
        ".harness",
        ".source_arch22",
        ".graybox_plugin_runtime",
        "output",
        "reports",
        "__pycache__",
        # 2026-08-21 (batch campaign, MUSE loop-break): these controller/audit
        # files are written AFTER the O5 candidate freeze.  Including them in
        # the candidate scope made the frozen digest drift on every finalize
        # round ("current candidate scope differs"), so every provenance gate
        # failed identically and DEBT-192 loop-break fired despite a 50/50
        # precision PASS.  They are runtime/evidence, not candidate source.
        "audit_self_critic_post_worker.md",
        "finalize_precheck.md",
        "orchestrator_events.jsonl",
        "construction_manifest.json",
        ".delegation_scan_violations.json",
        "user_decision.md",
        "state_transitions.jsonl",
        # 2026-08-22 (MUSE finalize drift): controller-written runtime
        # markers at the workspace root.  ``.delegation_scan_passed`` is
        # written by the finalize delegation scan AFTER the O5 freeze and
        # drifted the current candidate scope ("current candidate scope
        # differs from the frozen evaluation snapshot"), tripping the
        # DEBT-192 loop-break on an otherwise-PASS finalize.  The other two
        # are finalize/rollback ledgers appended on the same path.
        ".delegation_scan_passed",
        ".finalize_loop_nonconvergent",
        ".rollback_history.jsonl",
        ".user_decision_consumed.md",
        ".cba_required_routes.json",
        # Worker-authored pipeline documentation, edited across iterations
        # (post-O5 edits to these drifted the frozen candidate digest).
        "knowledge_update.md",
        "failures_ledger.md",
        "analysis.md",
        "PROGRESS.md",
        "self_critic_report.md",
        "a_tier_manifest.json",
        "op_classification.json",
        "reference_manifest.jsonl",
        ".cc_envelope_log.jsonl",
        ".critic_invoke_log.jsonl",
        ".opgen.log",
    }
)
_CANDIDATE_RUNTIME_SUFFIXES = (".log", ".stdout", ".stderr")


def _candidate_excluded(relative: Path) -> bool:
    """Whether a workspace path is evaluator-owned rather than candidate code."""
    if not relative.parts:
        return False
    top = relative.parts[0]
    if top in _CANDIDATE_RUNTIME_TOP_LEVEL:
        return True
    if relative.name.startswith("audit_self_critic_post_worker"):
        # Includes the .STALE_<ts> archived variants the audit rotates in place.
        return True
    if relative.name.startswith(".cc_stream_log"):
        return True
    if relative.name.startswith(".finalized-"):
        # Finalize success markers written by the harness after promotion;
        # a later resume/second finalize must not drift the scope digest on
        # them (audit M5 — same class as the fix-11 runtime markers).
        return True
    if len(relative.parts) >= 2 and relative.parts[0] == "op_host":
        # GE op_host delivery trio + FA-class shared headers: the finalize GE
        # assembler injects them AFTER the O5 freeze (they need the GE
        # framework and would break the kernel build if staged earlier).
        # They are delivery-archive source verified separately by the
        # GE_OPHOST_RAW_CANN_COPY gate; keeping them in the evaluation scope
        # made the finalize fail with "current candidate scope differs from
        # the frozen evaluation snapshot" (3_FusionAttention 2026-08-22,
        # finalize after a PROVISIONAL O5).
        name = relative.name
        if name in {"wp_fa_host_tiling.h", "ge_host_shim.h", "wp_fa_host_cache.h"}:
            return True
        if name.endswith(("_def.cpp", "_infershape.cpp", "_tiling.cpp", "_tiling_common.h")):
            return True
    if len(relative.parts) >= 2 and relative.parts[0] == "kernel" and relative.parts[1] == "build":
        # Controlled-build artifacts are evaluator-owned outputs; they are
        # rebuilt (and differ byte-wise) on every O5 round, so including them
        # in the scope digest made the frozen candidate digest drift between
        # rounds.  They stay out of the DIGEST only: _copy_candidate_scope
        # still carries the built extension modules (.so) into the frozen
        # snapshot because the isolated evaluator imports the extension from
        # the candidate's kernel/build.
        return True
    return relative.name.endswith(_CANDIDATE_RUNTIME_SUFFIXES)


def _candidate_tree_sha256(root: Path) -> str:
    return tree_sha256(root, exclude=_candidate_excluded)


def candidate_tree_sha256(candidate_dir: Path) -> str:
    """Return the canonical candidate scope digest used in evidence bindings."""
    root = _candidate_root(candidate_dir)
    _candidate_entry(root)
    return _candidate_tree_sha256(root)


_CANDIDATE_BUILD_RELATIVE = Path("kernel") / "build"


def _copy_candidate_scope(source: Path, destination: Path) -> None:
    """Copy the candidate scope, dropping links, detritus and excluded paths."""
    for directory_text, dirs, files in os.walk(source, topdown=True, followlinks=False):
        directory = Path(directory_text)
        relative_dir = directory.relative_to(source)
        if len(relative_dir.parts) >= 2 and relative_dir.parts[:2] == ("kernel", "build"):
            _copy_candidate_build_directory(directory, destination, relative_dir, dirs, files)
            continue
        if relative_dir != Path(".") and _candidate_excluded(relative_dir):
            dirs[:] = []
            continue
        destination_dir = destination / relative_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        dirs[:] = _kept_candidate_subdirectories(source, directory, dirs)
        _copy_candidate_files(source, directory, destination, files)


def _copy_candidate_build_directory(
    directory: Path, destination: Path, relative_dir: Path, dirs: list[str], files: list[str]
) -> None:
    """Copy only the built extension modules out of ``kernel/build``.

    The isolated evaluator imports the ``_<op>_ext`` module from the
    candidate's ``kernel/build`` (kw_brief contract: the candidate snapshot
    includes its own built extension artifacts).  Keep the directory layout
    but copy ONLY the built extension modules (``.so`` files, at any depth —
    some CMake projects place them under ``kernel/build/lib``).  Object files,
    CMake caches and makefiles are build detritus that must not enter the
    snapshot (and stay out of the scope digest — see ``_candidate_excluded``,
    which excludes ``kernel/build`` from the digest but not from this copy).
    Symlinks inside the build tree are build-system detritus: prune them.
    """
    destination_dir = destination / relative_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    dirs[:] = [name for name in sorted(dirs) if not (directory / name).is_symlink()]
    for name in sorted(files):
        item = directory / name
        if item.is_symlink() or item.suffix != ".so":
            continue
        if not item.is_file():
            raise NpuBenchRunnerError(f"candidate source contains non-regular file: {item}")
        shutil.copyfile(item, destination / relative_dir / name)


def _kept_candidate_subdirectories(source: Path, directory: Path, dirs: list[str]) -> list[str]:
    """Return the subdirectory names that stay inside the candidate scope."""
    kept_dirs: list[str] = []
    for name in sorted(dirs):
        item = directory / name
        relative = item.relative_to(source)
        if item.is_symlink():
            raise NpuBenchRunnerError(f"candidate source contains symlink: {item}")
        if relative == _CANDIDATE_BUILD_RELATIVE:
            # Traverse into kernel/build; its extension modules are copied by
            # the dedicated branch above, everything else is dropped.
            kept_dirs.append(name)
            continue
        if _candidate_excluded(relative):
            continue
        kept_dirs.append(name)
    return kept_dirs


def _copy_candidate_files(
    source: Path, directory: Path, destination: Path, files: list[str]
) -> None:
    """Copy one directory's in-scope regular files, rejecting links/specials."""
    for name in sorted(files):
        item = directory / name
        relative = item.relative_to(source)
        if item.is_symlink():
            raise NpuBenchRunnerError(f"candidate source contains symlink: {item}")
        if _candidate_excluded(relative):
            continue
        if not item.is_file():
            raise NpuBenchRunnerError(f"candidate source contains non-regular file: {item}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, target)


def _make_tree_read_only(root: Path) -> None:
    for entry in sorted(root.rglob("*"), key=lambda item: item.as_posix(), reverse=True):
        if entry.is_symlink():
            raise NpuBenchRunnerError(f"candidate snapshot contains symlink: {entry}")
        if entry.is_dir():
            os.chmod(entry, 0o500)
        elif entry.is_file():
            os.chmod(entry, 0o400)
        else:
            raise NpuBenchRunnerError(f"candidate snapshot contains non-regular file: {entry}")
    os.chmod(root, 0o500)


def _require_snapshot(path: Path, expected_digest: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise NpuBenchRunnerError("candidate snapshot path is not a real directory")
    if _candidate_tree_sha256(path) != expected_digest:
        raise NpuBenchRunnerError("candidate snapshot digest differs from its content address")
    for entry in [path, *path.rglob("*")]:
        if entry.is_symlink():
            raise NpuBenchRunnerError("candidate snapshot contains symlink")
        if entry.stat().st_mode & 0o222:
            raise NpuBenchRunnerError("candidate snapshot must be read-only")


def _candidate_entry(candidate_dir: Path) -> Path:
    entry = Path(candidate_dir) / "model_new_ascendc.py"
    _require_regular(entry, "candidate model_new_ascendc.py")
    return entry.resolve()


def _precision_contract() -> dict[str, Any]:
    return {
        "version": PRECISION_CONTRACT_VERSION,
        "seed": DEFAULT_SEED,
        "semantics_source": dict(PRECISION_SEMANTICS_SOURCE),
        "checks": [
            "allclose",
            "matched_ratio",
            "MERE",
            "nan_inf_masks_and_inf_signs",
            "quantized_integer_lsb",
            "nested_structure",
        ],
        "required_matched_ratio": REQUIRED_MATCHED_RATIO,
        "allclose_tolerances": {key: list(value) for key, value in ALLCLOSE_TOLERANCES.items()},
        "npu_limits": {key: list(value) for key, value in NPU_LIMITS.items()},
        "integer_lsb_tolerance": dict(INT_LSB_TOLERANCE),
        "float_dtype_policy": "candidate_cast_to_reference_before_three_checks",
        "complex_policy": "real_imag_separate_three_checks",
        "initialization": "candidate_get_init_inputs_or_reference_fallback",
        "input_generation": "once_then_deep_clone",
    }


def _performance_contract() -> dict[str, Any]:
    return {
        "version": PERFORMANCE_CONTRACT_VERSION,
        "profiling_mode": "quick",
        "warm_up": WARM_UP,
        "repeats": REPEATS,
        "keep_prof": True,
        "profiler_engine": "ops/ops-profiling/scripts/msprof_perf_summary.py",
        "input_adapter": "plugin_native_fixture_shim",
    }


def _base_report(
    verb: str,
    *,
    status: str,
    binding: Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": f"cannbot.npubench.{verb}/v1",
        "status": status,
        "runner_contract_version": RUNNER_CONTRACT_VERSION,
        "run_id": run_id or uuid.uuid4().hex,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if binding is not None:
        report["binding_sha256"] = binding["binding_sha256"]
        report["evaluation_binding"] = dict(binding)
    return report


def _json_safe_non_finite(value: Any) -> Any:
    """Recursively replace non-finite floats with JSON-safe string markers.

    Precision metrics can legitimately be inf/nan when a candidate output
    diverges (for example a float32 subtraction overflow inside the benchmark
    accuracy computation).  Reports are persisted with ``allow_nan=False``,
    so without this replacement one diverged case aborts the ``json.dumps``
    in ``_atomic_json`` and the report never lands on disk — a real FAIL is
    then misreported upstream as missing evidence.  Pass/fail verdicts derive
    from per-case booleans and status strings decided before publication, and
    binding digests are hex strings, so converting only the non-finite metric
    values to "inf"/"-inf"/"nan" preserves semantics while keeping the report
    JSON-compliant.  Finite numbers and all other values pass through
    unchanged.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, Mapping):
        return {key: _json_safe_non_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_non_finite(item) for item in value]
    return value


def _try_write_report(workspace: Path, filename: str, report: Mapping[str, Any]) -> None:
    try:
        evidence_root = _workspace_runtime_directory(
            workspace, EVIDENCE_DIRNAME, "NPUKernelBench evidence root"
        )
        _atomic_json(evidence_root / filename, _json_safe_non_finite(dict(report)))
    except (NpuBenchRunnerError, OSError):
        # The caller gets the actual validation result even in a deliberately
        # read-only smoke environment.  O5/finalize requires the file and will
        # reject missing evidence, so this is never a path to a false PASS.
        pass


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    _require_real_directory(path.parent, "atomic JSON parent")
    if path.is_symlink():
        raise NpuBenchRunnerError("atomic JSON target must not be a symlink")
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise NpuBenchRunnerError("precision requires PyTorch in the target environment") from exc


def runner_module_path() -> Path:
    """Absolute path of the runner entry module, which sits beside this one.

    The evaluator is executed and loaded by absolute path -- by its own child
    processes and by the generated profiler shims -- so its location must be
    derived from a module that is always copied next to it, never from the
    calling module's ``__file__``.
    """
    return Path(__file__).resolve().with_name("npubench_runner.py")


def _runner_ascendc_env_path() -> Path:
    """Return the runner configuration path without importing engine helpers.

    The execution child is a copied single-file runner, deliberately not the
    entire engine source tree.  Resolve the target interpreter in the parent
    while this original module still knows the canonical engine layout.  An
    explicit ``ASCENDC_ENV_PATH`` keeps subprocess tests and deployed wrappers
    able to select a different config file.
    """
    override = str(os.environ.get("ASCENDC_ENV_PATH") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[4] / "workspace" / ".ascendc_env"


def _configured_target_python_values() -> Mapping[str, str]:
    """Read only the interpreter keys from the normal shell-style config.

    This parser intentionally does not source or evaluate ``.ascendc_env``:
    it accepts the same simple ``KEY=VALUE`` form as the engine config reader
    and only exposes the three non-secret interpreter selectors needed here.
    Missing config is valid for local runner unit tests and falls back to the
    current Python interpreter.
    """
    config_path = _runner_ascendc_env_path()
    if not config_path.exists():
        return {}
    try:
        contents = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise NpuBenchRunnerError(
            f"cannot read NPU Python configuration at {config_path}: {exc}"
        ) from exc
    selected = {"A5_NPU_PYTHON_BIN", "NPU_PYTHON_BIN", "A5_HOST_PYTHON", "A5_CONTAINER"}
    values: dict[str, str] = {}
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key not in selected:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value:
            values[key] = value
    return values


def _resolve_configured_python(key: str, raw: str) -> Path:
    """Resolve one configured interpreter/bin selector to an executable."""
    supplied = Path(raw).expanduser()
    executable = supplied / "python3" if supplied.is_dir() else supplied
    if not executable.is_file() and len(supplied.parts) == 1:
        resolved_from_path = shutil.which(raw)
        if resolved_from_path:
            executable = Path(resolved_from_path)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise NpuBenchRunnerError(
            f"{key}={raw} does not name an executable Python interpreter "
            "(No such file or directory)"
        )
    try:
        return executable.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NpuBenchRunnerError(
            f"{key}={raw} cannot be resolved as a Python interpreter: {exc}"
        ) from exc


def _resolve_target_python() -> Path:
    """Choose the A5 Python interpreter that owns torch/torch_npu.

    The native runner executes benchmark task code in a child process.  On a
    controller machine that interpreter is often not ``sys.executable``;
    respect the same A5-specific/generic NPU Python contract used by the
    existing O5 verifier.  A configured value may name either a bin directory
    or an executable.  No configured value preserves local-development
    behaviour.
    """
    # A controller-side `.ascendc_env` may describe a remote A5 image, but any
    # interpreter selector it contains is still an explicit runtime contract.
    # Do not fall back to the controller interpreter when that contract cannot
    # be resolved; a green result from the wrong runtime is not valid evidence.
    config_values = _configured_target_python_values()
    configured_container = str(
        os.environ.get("A5_CONTAINER") or config_values.get("A5_CONTAINER") or ""
    ).strip()
    for key in ("A5_NPU_PYTHON_BIN", "NPU_PYTHON_BIN", "A5_HOST_PYTHON"):
        explicit = str(os.environ.get(key) or "").strip()
        raw = explicit or str(config_values.get(key) or "").strip()
        if not raw:
            continue
        # A configured interpreter is an explicit target contract even when it
        # came from .ascendc_env, so an unresolvable value propagates: falling
        # back silently would run the benchmark with the controller Python and
        # produce a green result from the wrong runtime.
        return _resolve_configured_python(key, raw)
    if configured_container and configured_container.lower() != "local":
        raise NpuBenchRunnerError(
            "A5_CONTAINER requests a non-local target but no target Python "
            "interpreter is configured"
        )
    return Path(sys.executable).resolve()


def _is_a3_routing_variable(key: str) -> bool:
    """True for the environment names that carry A3/private routing."""
    upper = key.upper()
    prefixed = upper.startswith(("A3_", "ASCEND_A3", "CANNBOT_A3"))
    return prefixed or "_A3_" in upper


def _scrubbed_task_environment(*, target_python: Path | None = None) -> dict[str, str]:
    """Return a child environment with A3/private routing and proxies removed.

    This is a minimum process boundary, not a substitute for a target's mount
    namespace/firewall policy.  The report identifies it as such so an O5
    deployment that requires OS-level network isolation can fail closed before
    accepting evidence.
    """
    environment = dict(os.environ)
    for key in list(environment):
        if _is_a3_routing_variable(key):
            environment.pop(key, None)
    for key in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "SSH_AUTH_SOCK",
    ):
        environment.pop(key, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CANNBOT_NPUBENCH_CHILD"] = "1"
    if target_python is not None:
        current_path = environment.get("PATH", "")
        environment["PATH"] = (
            str(Path(target_python).parent)
            if not current_path
            else str(Path(target_python).parent) + os.pathsep + current_path
        )
    return environment


def _require_real_read_only_tree(root: Path, label: str) -> None:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise NpuBenchRunnerError(f"{label} must be a real directory")
    for item in [root, *root.rglob("*")]:
        if item.is_symlink() or item.stat().st_mode & 0o222:
            raise NpuBenchRunnerError(f"{label} must be read-only and symlink-free")


def _run_isolated_context(
    context: _ExecutionContext,
    *,
    subprocess_run: Callable[..., Any] = subprocess.run,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Run one request in a scrubbed, short-lived normal child process.

    This is deliberately a *process boundary*, not a host-security sandbox.
    The parent owns the staged input/candidate binding, freezes phase-A
    fixture bytes before phase B, and checks the relevant digests again after
    the child exits.  A malicious same-UID program can still attempt runtime
    tampering; callers must not interpret this as adversarial OS isolation.
    """
    timeout_seconds = _resolve_task_execution_timeout(timeout_seconds)
    if timeout_seconds <= 0:
        raise NpuBenchRunnerError("task execution timeout must be positive")
    internal = {
        "preflight": "internal-exec-preflight",
        "precision": "internal-exec-precision",
        "fixture": "internal-exec-fixture",
        "performance": "internal-exec-performance",
    }.get(context.verb)
    if internal is None:
        raise NpuBenchRunnerError(f"unsupported isolated execution verb: {context.verb}")
    target_python = context.target_python
    command = [
        str(target_python),
        str(context.runner_root / "npubench_runner.py"),
        internal,
        "--execution-request",
        str(context.request_path),
    ]
    completed = _run_child_process_group(
        command,
        cwd=context.scratch,
        env=_scrubbed_task_environment(target_python=target_python),
        timeout_seconds=timeout_seconds,
        subprocess_run=subprocess_run,
    )
    report = _parse_child_report(getattr(completed, "stdout", ""))
    if report is None:
        raise NpuBenchRunnerError(
            "isolated task runner produced no machine-readable report "
            f"(returncode={getattr(completed, 'returncode', None)}): "
            f"{_output_tail(getattr(completed, 'stderr', ''), limit=2048)}"
        )
    report["child_returncode"] = int(getattr(completed, "returncode", 1))
    report["execution_isolation"] = "process_boundary"
    report["tamper_protection"] = "post_run_hash_check"
    return report


def _run_child_process_group(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
    subprocess_run: Callable[..., Any],
) -> Any:
    if subprocess_run is not subprocess.run:
        return subprocess_run(
            list(command), cwd=str(cwd), text=True, capture_output=True,
            check=False, env=dict(env), timeout=timeout_seconds,
        )
    process = subprocess.Popen(
        list(command), cwd=str(cwd), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=dict(env), start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_child_process_group(process)
        stdout, stderr = process.communicate()
        raise NpuBenchRunnerError(f"isolated task timed out after {timeout_seconds}s") from exc
    # ``communicate`` reaps only the direct child.  If a normal descendant
    # inherited stdout/stderr it forces the timeout path above; otherwise
    # best-effort kill the fresh process group after the child exits.
    _terminate_child_process_group(process)
    return types.SimpleNamespace(returncode=process.returncode, stdout=stdout, stderr=stderr)


def _terminate_child_process_group(process: subprocess.Popen[Any]) -> None:
    """Best-effort kill of the fresh launcher session created by this runner."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # The normal no-descendant case has no remaining process group.  A
        # direct kill is only a fallback for platforms that deny killpg.
        try:
            if process.poll() is None:
                process.kill()
        except (ProcessLookupError, PermissionError):
            pass


def _run_runner_child(
    verb: str,
    *,
    workspace: Path,
    candidate_dir: Path | None = None,
    device: int | str | None = None,
    seed: int | None = None,
    subprocess_run: Callable[..., Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    """Execute an internal task-importing verb in a bounded scrubbed process."""
    if timeout_seconds <= 0:
        raise NpuBenchRunnerError("task execution timeout must be positive")
    target_python = _resolve_target_python()
    runner_script = str(runner_module_path())
    command = [str(target_python), runner_script, verb, "--workspace", str(Path(workspace).resolve())]
    if candidate_dir is not None:
        command.extend(["--candidate-dir", str(Path(candidate_dir).resolve())])
    if device is not None:
        command.extend(["--device", str(device)])
    if seed is not None:
        command.extend(["--seed", str(seed)])
    completed = subprocess_run(
        command,
        cwd=str(Path(workspace).resolve()),
        text=True,
        capture_output=True,
        check=False,
        env=_scrubbed_task_environment(target_python=target_python),
        timeout=timeout_seconds,
    )
    report = _parse_child_report(getattr(completed, "stdout", ""))
    if report is None:
        stderr = _output_tail(getattr(completed, "stderr", ""), limit=2048)
        raise NpuBenchRunnerError(
            "isolated task runner produced no machine-readable report "
            f"(returncode={getattr(completed, 'returncode', None)}): {stderr}"
        )
    expected_verb = "preflight" if verb == "internal-preflight" else "precision"
    if report.get("schema") != f"cannbot.npubench.{expected_verb}/v1":
        raise NpuBenchRunnerError("isolated task runner returned an unexpected report schema")
    if report.get("status") not in {"PASS", "FAIL", "ERROR"}:
        raise NpuBenchRunnerError("isolated task runner returned an invalid status")
    report["child_returncode"] = int(getattr(completed, "returncode", 1))
    report["execution_isolation"] = "process_boundary"
    report["tamper_protection"] = "post_run_hash_check"
    return report


def _parse_child_report(stdout: Any) -> dict[str, Any] | None:
    for line in reversed(str(stdout or "").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _safe_prof_tag(value: str) -> bool:
    return bool(value) and all(char.isascii() and (char.isalnum() or char in "_.-") for char in value)


def _output_tail(value: Any, *, limit: int = 8192) -> str:
    text = str(value or "")
    return text[-limit:]


def _read_state(workspace: Path) -> Mapping[str, Any]:
    state_path = Path(workspace) / ".opgen_state.json"
    if state_path.is_symlink():
        raise NpuBenchRunnerError("durable state must be a regular non-symlink file")
    try:
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise NpuBenchRunnerError("durable state is missing") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise NpuBenchRunnerError(f"durable state is unreadable: {exc}") from exc
    if not isinstance(state, Mapping):
        raise NpuBenchRunnerError("durable state must be a JSON object")
    return state


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NPUKernelBench old-format runner")
    sub = parser.add_subparsers(dest="verb", required=True)
    stage = sub.add_parser("stage")
    stage.add_argument("--workspace", required=True, type=Path)
    stage.add_argument("--task", required=True, type=Path)
    stage.add_argument("--root", type=Path)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--workspace", required=True, type=Path)
    precision = sub.add_parser("precision")
    precision.add_argument("--workspace", required=True, type=Path)
    precision.add_argument("--candidate-dir", required=True, type=Path)
    precision.add_argument("--device", required=True, type=int)
    performance = sub.add_parser("performance")
    performance.add_argument("--workspace", required=True, type=Path)
    performance.add_argument("--candidate-dir", required=True, type=Path)
    performance.add_argument("--device", required=True, type=int)
    performance.add_argument("--lease-manifest", type=Path)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--workspace", required=True, type=Path)
    evaluate.add_argument("--candidate-dir", required=True, type=Path)
    evaluate.add_argument("--precision-device", required=True, type=int)
    evaluate.add_argument("--performance-device", required=True, type=int)
    evaluate.add_argument("--lease-manifest", type=Path)
    # Internal verbs are deliberately not documented.  They are the only
    # code paths that import user task/candidate modules; public callers use
    # the parent wrappers which scrub A3/proxy environment and enforce timeout.
    internal_preflight = sub.add_parser("internal-preflight", help=argparse.SUPPRESS)
    internal_preflight.add_argument("--workspace", required=True, type=Path)
    internal_precision = sub.add_parser("internal-precision", help=argparse.SUPPRESS)
    internal_precision.add_argument("--workspace", required=True, type=Path)
    internal_precision.add_argument("--candidate-dir", required=True, type=Path)
    internal_precision.add_argument("--device", required=True)
    internal_precision.add_argument("--seed", type=int, default=DEFAULT_SEED)
    for internal_verb in (
        "internal-exec-preflight",
        "internal-exec-precision",
        "internal-exec-fixture",
        "internal-exec-performance",
    ):
        parser_internal_exec = sub.add_parser(internal_verb, help=argparse.SUPPRESS)
        parser_internal_exec.add_argument("--execution-request", required=True, type=Path)
    return parser

