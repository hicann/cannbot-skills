# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Frozen input fixtures and adapter views for the NPUKernelBench runner.

This module owns everything that turns an immutable task into something the
repository's fixed-filename quick profiler can consume without ever rewriting
the task into a ``model.py``/``test.py`` pair: the sidecar descriptor subset
and its lazy materializer, the input-adapter shims, the restricted
``weights_only`` fixture tree, the per-case fixture chunks and their manifest,
and the parent-owned freeze/copy/cleanup of that fixture.

It imports only ``npubench_core``.  ``npubench_runner`` re-exports its public
surface, so importers keep using the runner module path.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from npubench_core import (
    EXECUTION_DIRNAME,
    INPUT_ADAPTER_CONTRACT_VERSION,
    NATIVE_PERF_CASE_DIRNAME,
    NATIVE_PERF_CASE_SCHEMA,
    NATIVE_PERF_FIXTURE_FILENAME,
    NATIVE_PERF_FIXTURE_SCHEMA,
    NATIVE_PERF_MANIFEST_FILENAME,
    NATIVE_PERF_MANIFEST_SCHEMA,
    NpuBenchRunnerError,
    SIDECAR_DESCRIPTOR_ADAPTER,
    SIDECAR_DESCRIPTOR_SCHEMA,
    StagedBundle,
    _SIDECAR_DTYPE_ALIASES,
    _SIDECAR_DTYPE_ITEMSIZE,
    _SIDECAR_INT_DTYPES,
    _SIDECAR_MAX_CASE_BYTES,
    _SIDECAR_MAX_DIM,
    _SIDECAR_MAX_ELEMENTS,
    _SIDECAR_MAX_TENSOR_BYTES,
    _SIDECAR_MAX_TOTAL_BYTES,
    _atomic_json,
    _candidate_entry,
    _candidate_root,
    _copy_regular_tree,
    _create_real_child_directory,
    _ensure_real_child_directory,
    _file_sha256,
    _import_torch,
    _input_adapter_identity,
    _make_tree_read_only,
    _require_real_directory,
    _require_real_read_only_tree,
    _require_regular,
    _safe_prof_tag,
    _workspace_runtime_directory,
    build_evaluation_binding,
    resolve_staged_bundle,
    runner_module_path,
    tree_sha256,
)


# Scalar leaf classes the frozen fixture accepts.  Membership is tested
# against the *exact* type of a value, never with ``isinstance``: a subclass
# of one of these is not on PyTorch's ``weights_only`` allowlist.
_NATIVE_PRIMITIVE_TYPES = frozenset({bool, int, float, complex, str, bytes})


def prepare_adapter_view(
    workspace: Path,
    candidate_dir: Path,
    *,
    bundle: StagedBundle | None = None,
    binding: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    allow_existing_native_fixture: bool = False,
) -> Path:
    """Create a profiler-only fixed-name view without copying/relabeling task code."""
    workspace = Path(workspace)
    bundle = bundle or resolve_staged_bundle(workspace)
    binding = binding or build_evaluation_binding(workspace, candidate_dir, bundle=bundle)
    candidate_root = _candidate_root(candidate_dir)
    candidate_entry = _candidate_entry(candidate_root)
    token = run_id or uuid.uuid4().hex
    adapter = workspace / ".npubench_adapter" / str(binding["binding_sha256"]) / token
    created_adapter = False
    if adapter.exists():
        if not allow_existing_native_fixture or adapter.is_symlink() or not adapter.is_dir():
            raise NpuBenchRunnerError(f"adapter path already exists: {adapter}")
        allowed = {"native_fixture"}
        existing = {entry.name for entry in adapter.iterdir()}
        if existing - allowed:
            raise NpuBenchRunnerError("precreated adapter contains unexpected writable content")
    else:
        adapter.mkdir(parents=True, mode=0o700)
        created_adapter = True
    try:
        _write_adapter_proxy(
            adapter / "model.py",
            target=bundle.task_path,
            root=bundle.root,
            role="reference",
        )
        _write_adapter_proxy(
            adapter / "model_new_ascendc.py",
            target=candidate_entry,
            root=candidate_root,
            role="candidate",
        )
        shutil.copyfile(bundle.sidecar_path, adapter / bundle.sidecar_path.name)
        os.chmod(adapter / bundle.sidecar_path.name, 0o400)
        metadata = {
            "schema": "cannbot.npubench.adapter/v1",
            "binding_sha256": binding["binding_sha256"],
            "reference_task": str(bundle.task_path),
            "reference_root": str(bundle.root),
            "candidate_entry": str(candidate_entry),
            "candidate_root": str(candidate_root),
            "sidecar_name": bundle.sidecar_path.name,
        }
        _atomic_json(adapter / "adapter_manifest.json", metadata)
    except Exception:
        if created_adapter:
            shutil.rmtree(adapter, ignore_errors=True)
        raise
    return adapter


def _prepare_native_quick_adapter(
    adapter: Path,
    fixture_root: Path,
    *,
    binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Expose a frozen native fixture through the unchanged quick profiler API.

    The repository profiler predates NPUKernelBench's task-pair format.  Its
    quick mode asks ``model.py`` for a case by index, so this adapter replaces
    only its temporary profiler views with shims backed by the frozen fixture.
    The original task and sidecar remain staged byte-for-byte and are never
    converted into a user-facing ``model.py``/``test.py`` pair.
    """
    adapter = Path(adapter)
    fixture_root = Path(fixture_root)
    _require_real_directory(adapter, "native quick profiler adapter")
    _require_real_directory(fixture_root, "native quick profiler fixture")
    native = _load_native_perf_manifest(fixture_root, binding=binding, verify_case_payloads=True)
    metadata = _read_native_quick_adapter_manifest(adapter)
    sidecar_name = _native_quick_sidecar_name(metadata)
    reference_task, reference_root, candidate_entry, candidate_root = _native_quick_shim_targets(metadata)

    # The shared profiler scans ``*_perf_cases.jsonl`` before arbitrary
    # ``*.jsonl``.  An old benchmark may already use that suffix, so remove
    # the temporary copied sidecar before supplying the shim-only index.  The
    # immutable staged source remains untouched.
    sidecar_view = adapter / sidecar_name
    if sidecar_view.suffix == ".jsonl":
        _require_regular(sidecar_view, "native quick adapter copied sidecar")
        sidecar_view.unlink()

    _write_native_quick_model_shim(
        adapter / "model.py",
        class_name="Model",
        role="reference",
        target=reference_task,
        root=reference_root,
        fixture_root=fixture_root,
        expose_inputs=True,
    )
    _write_native_quick_model_shim(
        adapter / "model_new_ascendc.py",
        class_name="ModelNew",
        role="candidate",
        target=candidate_entry,
        root=candidate_root,
        fixture_root=fixture_root,
        expose_inputs=False,
    )
    _write_native_quick_case_index(adapter / "npubench_perf_cases.jsonl", native)
    return native


def _read_native_quick_adapter_manifest(adapter: Path) -> Mapping[str, Any]:
    """Read the adapter manifest that names the shim targets."""
    manifest_path = adapter / "adapter_manifest.json"
    _require_regular(manifest_path, "native quick adapter manifest")
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NpuBenchRunnerError("native quick adapter manifest is unreadable") from exc
    if not isinstance(metadata, Mapping):
        raise NpuBenchRunnerError("native quick adapter manifest is invalid")
    return metadata


def _native_quick_sidecar_name(metadata: Mapping[str, Any]) -> str:
    """Return the manifest's sidecar file name, rejecting any path component."""
    sidecar_name = metadata.get("sidecar_name")
    named = isinstance(sidecar_name, str) and bool(sidecar_name)
    if not named or Path(str(sidecar_name)).name != sidecar_name:
        raise NpuBenchRunnerError("native quick adapter sidecar name is invalid")
    return str(sidecar_name)


def _native_quick_shim_targets(metadata: Mapping[str, Any]) -> tuple[Path, Path, Path, Path]:
    """Resolve the reference/candidate entry points the shims will import."""
    reference_task = Path(str(metadata.get("reference_task", "")))
    reference_root = Path(str(metadata.get("reference_root", "")))
    candidate_entry = Path(str(metadata.get("candidate_entry", "")))
    candidate_root = Path(str(metadata.get("candidate_root", "")))
    _require_regular(reference_task, "native quick reference task")
    _require_real_directory(reference_root, "native quick reference root")
    _require_regular(candidate_entry, "native quick candidate entry")
    _require_real_directory(candidate_root, "native quick candidate root")
    try:
        reference_task.resolve().relative_to(reference_root.resolve())
        candidate_entry.resolve().relative_to(candidate_root.resolve())
    except ValueError as exc:
        raise NpuBenchRunnerError("native quick adapter task is outside its declared root") from exc
    return reference_task, reference_root, candidate_entry, candidate_root


def _write_native_quick_case_index(path: Path, native: Mapping[str, Any]) -> None:
    """Write only case-control records; benchmark payloads stay in the fixture."""
    count = native.get("case_count")
    empty = native.get("empty_case_indices")
    positive_count = not isinstance(count, bool) and isinstance(count, int) and count > 0
    if not positive_count or not isinstance(empty, list):
        raise NpuBenchRunnerError("native quick fixture has invalid case metadata")
    empty_indices = set(empty)
    records: list[str] = []
    for index in range(count):
        # If the shared profiler cannot import this adapter's reference shim,
        # it falls back to JSONL-generated kwargs.  This sentinel makes that
        # fallback fail closed instead of benchmarking a synthetic no-input
        # invocation.  The normal path always supplies the frozen envelope.
        record: dict[str, Any] = {
            "npubench_case": index,
            "inputs": [
                {
                    "name": "__npubench_fixture_required__",
                    "type": "attr",
                    "dtype": "bool",
                    "value": True,
                }
            ],
        }
        if index in empty_indices:
            record["inputs"] = [{"type": "tensor", "shape": [0], "dtype": "float32"}]
        records.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    if path.is_symlink():
        raise NpuBenchRunnerError("native quick case index must not be a symlink")
    path.write_text("\n".join(records) + "\n", encoding="utf-8")
    os.chmod(path, 0o400)


# Static half of the generated quick-profiler shim: fixture loading, device
# movement and the frozen input-group sequence.  It carries no interpolation,
# so it lives next to the two interpolated fragments below rather than inside
# the writer function.
_NATIVE_QUICK_SHIM_HELPERS = '''

def _load_payload(path):
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("PyTorch weights_only fixture loading is required") from exc
    if not isinstance(value, dict):
        raise RuntimeError("frozen NPUKernelBench fixture payload is invalid")
    return value


def _common_payload():
    value = _load_payload(_COMMON_PATH)
    if value.get("schema") != "cannbot.npubench.native_perf_fixture/v1":
        raise RuntimeError("frozen NPUKernelBench common fixture schema is invalid")
    return value


def _case_payload(index):
    value = _load_payload(_CASE_ROOT / ("case_%06d.pt" % index))
    if value.get("schema") != "cannbot.npubench.native_perf_case/v1" or value.get("case") != index:
        raise RuntimeError("frozen NPUKernelBench case fixture identity is invalid")
    return value


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _model_device(model):
    for parameter in model.parameters():
        return parameter.device
    return torch.device("npu")


class _NpuBenchInput:
    _npubench_frozen_input = True

    def __init__(self, value):
        self.value = value
        self._moved = {}

    def for_device(self, device):
        key = str(device)
        if key not in self._moved:
            self._moved[key] = _move(self.value, device)
        return self._moved[key]


class _FrozenInputGroups:
    def __init__(self):
        self._count = int(_common_payload().get("case_count", 0))

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(self._count))]
        if not isinstance(index, int):
            raise TypeError("NPUKernelBench case index must be an integer")
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(index)
        payload = _case_payload(index)
        return [_NpuBenchInput(payload["input_group"])]
'''

# Optional tail: only the reference shim publishes the frozen input groups.
_NATIVE_QUICK_SHIM_INPUTS = "\n\ndef get_input_groups():\n    return _FrozenInputGroups()\n"


def _native_quick_shim_header(
    *, class_name: str, role: str, target: Path, root: Path, fixture_root: Path
) -> str:
    """Return the shim prologue that binds this runner and the frozen fixture."""
    runner_path = runner_module_path()
    return f'''# Auto-generated NPUKernelBench quick-profiler shim; not benchmark truth.
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_RUNNER_PATH = Path({str(runner_path)!r})
_TARGET_PATH = Path({str(Path(target).resolve())!r})
_ROOT_PATH = Path({str(Path(root).resolve())!r})
_FIXTURE_ROOT = Path({str(Path(fixture_root).resolve())!r})
_COMMON_PATH = _FIXTURE_ROOT / "native_perf_common.pt"
_CASE_ROOT = _FIXTURE_ROOT / "native_perf_cases"

# The runner is loaded by absolute path, so its sibling modules are not on
# ``sys.path`` in the profiler process.  Append (never prepend) its directory
# so the split runner imports without shadowing the profiler's own modules.
if str(_RUNNER_PATH.parent) not in sys.path:
    sys.path.append(str(_RUNNER_PATH.parent))

_spec = importlib.util.spec_from_file_location("_npubench_quick_runner", _RUNNER_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load NPUKernelBench quick runner")
_runner = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _runner
_spec.loader.exec_module(_runner)
_task = _runner.load_task_module(_TARGET_PATH, _ROOT_PATH, role={role!r})
_ctor = _runner._resolve_model_constructor(_task, preferred={class_name!r}, role={role!r})
'''


def _native_quick_shim_model(*, class_name: str, role: str) -> str:
    """Return the shim body that adapts the frozen envelope to the profiler."""
    return f'''

def _build_impl():
    common = _common_payload()
    values = common.get("init_values")
    if not isinstance(values, list):
        raise RuntimeError("frozen NPUKernelBench init fixture is invalid")
    impl = _runner._construct_model(_ctor, tuple(values), {role!r})
    return impl


class {class_name}(nn.Module):
    def __init__(self):
        super().__init__()
        self.impl = _build_impl()

    def forward(self, *args, **kwargs):
        if "__npubench_fixture_required__" in kwargs:
            raise RuntimeError("NPUKernelBench profiler must use the frozen fixture input")
        frozen = args[0] if len(args) == 1 and not kwargs else None
        if (
            getattr(frozen, "_npubench_frozen_input", False) is True
            and callable(getattr(frozen, "for_device", None))
        ):
            group = frozen.for_device(_model_device(self.impl))
            if isinstance(group, dict):
                return self.impl(**group)
            if isinstance(group, (list, tuple)):
                return self.impl(*group)
            return self.impl(group)
        return self.impl(*args, **kwargs)
'''


def _write_native_quick_model_shim(
    path: Path,
    *,
    class_name: str,
    role: str,
    target: Path,
    root: Path,
    fixture_root: Path,
    expose_inputs: bool,
) -> None:
    """Write a profiler-only nn.Module shim over frozen input/init fixtures."""
    if class_name not in {"Model", "ModelNew"} or role not in {"reference", "candidate"}:
        raise NpuBenchRunnerError("native quick shim role is invalid")
    if path.is_symlink():
        raise NpuBenchRunnerError("native quick shim path must not be a symlink")
    # ``prepare_adapter_view`` deliberately creates read-only proxy files.
    # Replace those exact local views rather than changing their mode in
    # place: unlinking is safe inside the evaluator-owned writable adapter
    # directory and cannot modify a possible hard-linked source file.
    if path.exists():
        if not path.is_file():
            raise NpuBenchRunnerError("native quick shim path must be a regular file")
        path.unlink()
    source = (
        _native_quick_shim_header(
            class_name=class_name, role=role, target=target, root=root, fixture_root=fixture_root
        )
        + _NATIVE_QUICK_SHIM_HELPERS
        + _native_quick_shim_model(class_name=class_name, role=role)
    )
    if expose_inputs:
        source += _NATIVE_QUICK_SHIM_INPUTS
    path.write_text(source, encoding="utf-8")
    os.chmod(path, 0o500)


def _freeze_native_case_chunks(
    adapter: Path, groups: Any, torch: Any, binding: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[int], int]:
    """Serialize one immutable ``.pt`` chunk per benchmark case.

    Keep the common construction payload small.  Each benchmark case gets its
    own immutable archive so a 50-case task such as level1/3_Add does not make
    every ref/asc profiler process load ~1.2 GiB of inputs.  The temporary
    list slot is cleared after each atomic write to cap additional runner
    retention; the task's ``get_input_groups`` API necessarily created the
    list once, but the fixture itself is streaming/chunked.
    """
    empty_case_indices: list[int] = []
    case_records: list[dict[str, Any]] = []
    case_root = adapter / NATIVE_PERF_CASE_DIRNAME
    case_root.mkdir(mode=0o700)
    case_count = len(groups)
    for index in range(case_count):
        # ``groups`` is a lazy sidecar sequence for native tasks.  Fetch one
        # case, serialize it, and drop the only strong reference before the
        # next case is materialized.
        group = groups[index]
        is_empty = _native_value_has_empty_tensor(group, torch)
        if is_empty:
            empty_case_indices.append(index)
        relative = f"{NATIVE_PERF_CASE_DIRNAME}/case_{index:06d}.pt"
        case_payload_raw = {
                "schema": NATIVE_PERF_CASE_SCHEMA,
                "binding_sha256": binding["binding_sha256"],
                "case": index,
                "empty_tensor": is_empty,
                "input_group": group,
            }
        _native_reject_tensor_aliases(case_payload_raw, torch, label=f"case {index}")
        case_payload = _native_restricted_tree(case_payload_raw, torch)
        case_path = adapter / relative
        _atomic_torch_fixture(case_path, case_payload, torch)
        case_records.append(
            {
                "case": index,
                "path": relative,
                "sha256": _file_sha256(case_path),
                "empty_tensor": is_empty,
            }
        )
        del case_payload, case_payload_raw, group
        # ``get_input_groups`` may return an eagerly materialized list.  Drop
        # its slot as well; deleting only the loop-local name leaves every
        # tensor-backed case retained until the entire fixture is serialized.
        try:
            groups[index] = None
        except (TypeError, AttributeError):
            # Tuples and custom lazy sequences do not expose mutable slots;
            # their per-case value is already released by ``del group``.
            pass
    return case_records, empty_case_indices, case_count


def _write_native_common_fixture(
    adapter: Path,
    torch: Any,
    *,
    binding: Mapping[str, Any],
    seed: int,
    seed_events: Any,
    input_adapter: Mapping[str, Any],
    init_args: Sequence[Any],
    case_count: int,
    empty_case_indices: list[int],
    valid_case_indices: list[int],
) -> str:
    """Write the shared construction payload and return its digest."""
    fixture_raw = {
        "schema": NATIVE_PERF_FIXTURE_SCHEMA,
        "binding_sha256": binding["binding_sha256"],
        "task_sha256": binding["task_sha256"],
        "candidate_tree_sha256": binding.get("candidate_tree_sha256"),
        "candidate_entry_sha256": binding.get("candidate_entry_sha256"),
        "seed": seed,
        "seed_events": seed_events,
        "input_adapter": input_adapter,
        "init_call_style": "args",
        "init_values": list(init_args),
        "case_count": case_count,
        "empty_case_indices": empty_case_indices,
        "valid_case_indices": valid_case_indices,
    }
    _native_reject_tensor_aliases(fixture_raw, torch, label="common init")
    fixture = _native_restricted_tree(fixture_raw, torch)
    fixture_path = adapter / NATIVE_PERF_FIXTURE_FILENAME
    _atomic_torch_fixture(fixture_path, fixture, torch)
    return _file_sha256(fixture_path)


def _decorate_adapter_manifest(adapter: Path, manifest_path: Path, fixture_sha256: str) -> None:
    """Record the native bridge in the adapter manifest without changing identity.

    Fixture-stage generation has no adapter proxy yet, so callers there
    deliberately skip this decoration.
    """
    adapter_manifest_path = adapter / "adapter_manifest.json"
    try:
        adapter_manifest = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NpuBenchRunnerError(f"adapter manifest is unreadable: {exc}") from exc
    if not isinstance(adapter_manifest, Mapping):
        raise NpuBenchRunnerError("adapter manifest is not an object")
    adapter_manifest = dict(adapter_manifest)
    adapter_manifest["native_performance"] = {
        "manifest_path": NATIVE_PERF_MANIFEST_FILENAME,
        "manifest_sha256": _file_sha256(manifest_path),
        "fixture_sha256": fixture_sha256,
    }
    _atomic_json(adapter_manifest_path, adapter_manifest)


def _native_value_has_empty_tensor(
    value: Any,
    torch: Any,
    active: set[int] | None = None,
) -> bool:
    tensor_type = getattr(torch, "Tensor", ())
    if isinstance(value, tensor_type):
        return int(value.numel()) == 0
    active = active if active is not None else set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise NpuBenchRunnerError("native performance fixture does not support cyclic containers")
        active.add(identity)
        try:
            return any(_native_value_has_empty_tensor(item, torch, active) for item in value.values())
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise NpuBenchRunnerError("native performance fixture does not support cyclic containers")
        active.add(identity)
        try:
            return any(_native_value_has_empty_tensor(item, torch, active) for item in value)
        finally:
            active.remove(identity)
    return False


def _is_torch_descriptor_type(value_type: type, torch: Any) -> bool:
    """True for ``torch.dtype``/``torch.device``: immutable, non-aliasing tags.

    Exact-type comparison, not ``isinstance``: a subclass of either descriptor
    is not on the ``weights_only`` unpickler allowlist and must be rejected.
    """
    dtype_type = getattr(torch, "dtype", None)
    device_type = getattr(torch, "device", None)
    is_dtype = dtype_type is not None and value_type is dtype_type
    is_device = device_type is not None and value_type is device_type
    return is_dtype or is_device


def _native_restricted_tensor(value: Any, torch: Any, memo: dict[int, Any]) -> Any:
    """Accept one exact ``torch.Tensor`` leaf without disturbing its storage."""
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if getattr(value, "layout", None) != getattr(torch, "strided", None):
        raise NpuBenchRunnerError("native performance fixture does not support non-strided tensors")
    if bool(getattr(value, "is_quantized", False)):
        raise NpuBenchRunnerError("native performance fixture does not support quantized tensors")
    # Do not clone, detach or make contiguous: torch.save retains the
    # original storage/view graph (including tied tensors), and the
    # consumer's ``map_location='cpu'`` performs the device relocation.
    memo[identity] = value
    return value


def _native_restricted_mapping(
    value: Mapping[Any, Any],
    torch: Any,
    memo: dict[int, Any],
    active: set[int],
    identity: int,
) -> dict[Any, Any]:
    """Copy an exact ``dict`` node, rejecting keys that survive as unhashable."""
    result_dict: dict[Any, Any] = {}
    memo[identity] = result_dict
    active.add(identity)
    try:
        for key, item in value.items():
            saved_key = _native_restricted_tree(key, torch, memo, active)
            try:
                hash(saved_key)
            except TypeError as exc:
                raise NpuBenchRunnerError("native performance fixture mapping key is not hashable") from exc
            result_dict[saved_key] = _native_restricted_tree(item, torch, memo, active)
    finally:
        active.remove(identity)
    return result_dict


def _native_restricted_tree(
    value: Any,
    torch: Any,
    memo: dict[int, Any] | None = None,
    active: set[int] | None = None,
) -> Any:
    """Validate/copy a non-executable fixture tree while preserving aliases.

    The output uses only classes accepted by PyTorch's ``weights_only``
    unpickler.  We intentionally preserve repeated references through ``memo``
    so tied parameters and input aliases remain representable by ``torch.save``.
    Tensor subclasses, sparse/quantized layouts and custom containers are
    rejected rather than degraded into an imprecise projection.

    Every type test below compares the *exact* type, never ``isinstance``:
    rejecting tensor and container subclasses is this function's contract, and
    ``isinstance`` would silently admit them.  ``type(...)`` is hoisted into
    ``value_type`` so the exact-type intent stays explicit and local.
    """
    memo = memo if memo is not None else {}
    active = active if active is not None else set()
    value_type = type(value)
    tensor_type = getattr(torch, "Tensor", None)
    if tensor_type is not None and value_type is tensor_type:
        return _native_restricted_tensor(value, torch, memo)
    if value is None or value_type in _NATIVE_PRIMITIVE_TYPES:
        return value
    if _is_torch_descriptor_type(value_type, torch):
        # ``torch.dtype``/``torch.device`` are immutable descriptors: they
        # cannot alias tensor data and are on the ``weights_only`` unpickler
        # allowlist, so they pass through unchanged.
        return value
    identity = id(value)
    if identity in active:
        raise NpuBenchRunnerError("native performance fixture does not support cyclic containers")
    if identity in memo:
        return memo[identity]
    if value_type is list:
        result_list: list[Any] = []
        memo[identity] = result_list
        active.add(identity)
        try:
            result_list.extend(_native_restricted_tree(item, torch, memo, active) for item in value)
        finally:
            active.remove(identity)
        return result_list
    if value_type is tuple:
        active.add(identity)
        try:
            result_tuple = tuple(_native_restricted_tree(item, torch, memo, active) for item in value)
        finally:
            active.remove(identity)
        memo[identity] = result_tuple
        return result_tuple
    if value_type is dict:
        return _native_restricted_mapping(value, torch, memo, active, identity)
    raise NpuBenchRunnerError(
        f"native performance fixture cannot safely serialize {value_type.__name__}; "
        "only exact torch.Tensor, torch.dtype/torch.device, list/tuple/dict, and "
        "primitive scalars are supported"
    )


def _native_reject_tensor_aliases(value: Any, torch: Any, *, label: str) -> None:
    """Fail closed when the profiler's clone protocol would change aliases.

    The native wrapper intentionally gives reference/candidate independent
    clone trees.  Its current per-tensor ``detach().clone()`` operation cannot
    preserve repeated tensor identity or view/shared-storage relationships, so
    accepting them would falsely claim a lossless benchmark input.  Rather
    than silently alter semantics, reject such tasks until the consumer grows
    a graph-preserving clone implementation.
    """
    tensor_type = getattr(torch, "Tensor", None)
    object_paths: dict[int, str] = {}
    storage_paths: dict[tuple[str, int], str] = {}
    active: set[int] = set()

    def visit_tensor(item: Any, path: str) -> None:
        """Record one tensor leaf, rejecting repeated identity/shared storage."""
        identity = id(item)
        if identity in object_paths:
            raise NpuBenchRunnerError(
                f"native performance fixture {label} has repeated tensor identity "
                f"at {path} and {object_paths[identity]}; graph-preserving clone is required"
            )
        object_paths[identity] = path
        if int(item.numel()) == 0:
            return
        key = _native_tensor_storage_key(item)
        if key is None:
            return
        previous = storage_paths.get(key)
        if previous is not None:
            raise NpuBenchRunnerError(
                f"native performance fixture {label} has shared tensor storage "
                f"at {path} and {previous}; graph-preserving clone is required"
            )
        storage_paths[key] = path

    def visit(item: Any, path: str) -> None:
        """Walk one fixture node; exact-type tests reject every subclass."""
        item_type = type(item)
        if tensor_type is not None and item_type is tensor_type:
            visit_tensor(item, path)
            return
        if item is None or item_type in _NATIVE_PRIMITIVE_TYPES:
            return
        if _is_torch_descriptor_type(item_type, torch):
            # ``torch.dtype``/``torch.device`` are immutable descriptors that
            # cannot alias tensor data, so there is nothing to inspect.
            return
        identity = id(item)
        if identity in active:
            raise NpuBenchRunnerError("native performance fixture does not support cyclic containers")
        if item_type is list or item_type is tuple:
            active.add(identity)
            try:
                for index, child in enumerate(item):
                    visit(child, f"{path}[{index}]")
            finally:
                active.remove(identity)
            return
        if item_type is dict:
            active.add(identity)
            try:
                for index, (key_item, child) in enumerate(item.items()):
                    visit(key_item, f"{path}.key[{index}]")
                    visit(child, f"{path}[{index}]")
            finally:
                active.remove(identity)
            return
        raise NpuBenchRunnerError(
            f"native performance fixture cannot safely inspect {item_type.__name__} for aliasing"
        )

    visit(value, "root")


def _native_tensor_storage_key(value: Any) -> tuple[str, int] | None:
    try:
        storage = value.untyped_storage()
        identity = int(getattr(storage, "_cdata"))
        return str(value.device), identity
    except Exception as exc:
        raise NpuBenchRunnerError(
            f"native performance fixture cannot inspect tensor storage aliasing: {type(exc).__name__}: {exc}"
        ) from exc


def _atomic_torch_fixture(path: Path, payload: Mapping[str, Any], torch: Any) -> None:
    """Write a restricted torch archive atomically without a pickle fallback."""
    path = Path(path)
    _require_real_directory(path.parent, "native performance fixture parent")
    if path.is_symlink():
        raise NpuBenchRunnerError("native performance fixture target must not be a symlink")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(dict(payload), temporary)
        with Path(temporary).open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o400)
    except Exception as exc:
        raise NpuBenchRunnerError(
            f"native performance fixture serialization failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_native_perf_manifest(
    adapter: Path,
    *,
    binding: Mapping[str, Any],
    verify_case_payloads: bool = True,
) -> Mapping[str, Any]:
    """Validate the parent-visible native bridge handoff after child exit."""
    adapter = Path(adapter)
    manifest_path = adapter / NATIVE_PERF_MANIFEST_FILENAME
    manifest = _read_native_perf_manifest(manifest_path, binding)
    adapter_identity = _validated_native_adapter_identity(manifest, binding)
    fixture_path = adapter / NATIVE_PERF_FIXTURE_FILENAME
    _require_regular(fixture_path, "native performance fixture")
    fixture_sha256 = _file_sha256(fixture_path)
    if manifest.get("fixture_sha256") != fixture_sha256:
        raise NpuBenchRunnerError("native performance fixture digest differs from manifest")
    torch = _import_torch()
    fixture = _load_native_common_fixture(fixture_path, torch, binding, adapter_identity)
    case_count, valid_indices, empty_indices = _validated_native_case_partition(manifest, fixture)
    sanitized_cases = _validated_native_case_fixtures(
        adapter,
        manifest,
        torch=torch,
        binding=binding,
        case_count=case_count,
        empty_indices=empty_indices,
        verify_case_payloads=verify_case_payloads,
    )
    return {
        "manifest_sha256": _file_sha256(manifest_path),
        "fixture_sha256": fixture_sha256,
        "fixture_schema": NATIVE_PERF_FIXTURE_SCHEMA,
        "case_count": case_count,
        "valid_case_indices": valid_indices,
        "empty_case_indices": empty_indices,
        "init_call_style": manifest.get("init_call_style"),
        "case_fixtures": sanitized_cases,
    }


def _read_native_perf_manifest(manifest_path: Path, binding: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read the native manifest and bind it to the parent's frozen digests."""
    _require_regular(manifest_path, "native performance manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NpuBenchRunnerError(f"native performance manifest is unreadable: {exc}") from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema") != NATIVE_PERF_MANIFEST_SCHEMA:
        raise NpuBenchRunnerError("native performance manifest schema is invalid")
    for field in ("binding_sha256", "task_sha256", "candidate_tree_sha256", "candidate_entry_sha256"):
        if manifest.get(field) != binding.get(field):
            raise NpuBenchRunnerError(f"native performance manifest {field} differs from binding")
    if manifest.get("fixture_schema") != NATIVE_PERF_FIXTURE_SCHEMA:
        raise NpuBenchRunnerError("native performance fixture schema is invalid")
    if manifest.get("fixture_path") != NATIVE_PERF_FIXTURE_FILENAME:
        raise NpuBenchRunnerError("native performance fixture path is invalid")
    return manifest


def _validated_native_adapter_identity(
    manifest: Mapping[str, Any], binding: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Return the manifest's input-adapter identity once it matches the binding."""
    adapter_identity = manifest.get("input_adapter")
    if not isinstance(adapter_identity, Mapping):
        raise NpuBenchRunnerError("native performance manifest input adapter identity is invalid")
    expected_adapter = binding.get("input_adapter")
    if not isinstance(expected_adapter, Mapping):
        raise NpuBenchRunnerError("evaluation binding has no input adapter identity")
    if adapter_identity != expected_adapter:
        raise NpuBenchRunnerError("native performance input adapter differs from binding")
    if adapter_identity.get("contract") != INPUT_ADAPTER_CONTRACT_VERSION:
        raise NpuBenchRunnerError("native performance input adapter contract is invalid")
    is_sidecar = adapter_identity.get("kind") == SIDECAR_DESCRIPTOR_ADAPTER
    if is_sidecar and adapter_identity.get("schema") != SIDECAR_DESCRIPTOR_SCHEMA:
        raise NpuBenchRunnerError("native performance sidecar adapter schema is invalid")
    return adapter_identity


def _load_native_common_fixture(
    fixture_path: Path,
    torch: Any,
    binding: Mapping[str, Any],
    adapter_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Load the shared construction payload under the ``weights_only`` contract."""
    try:
        # ``weights_only`` is an explicit part of the fixture contract.  Do
        # not add a compatibility fallback to ordinary pickle loading.
        fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise NpuBenchRunnerError(
            "installed PyTorch lacks required weights_only native fixture loading"
        ) from exc
    except Exception as exc:
        raise NpuBenchRunnerError(
            f"native performance fixture cannot be loaded with weights_only: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(fixture, Mapping) or fixture.get("schema") != NATIVE_PERF_FIXTURE_SCHEMA:
        raise NpuBenchRunnerError("native performance fixture payload schema is invalid")
    _native_restricted_tree(dict(fixture), torch)
    _native_reject_tensor_aliases(dict(fixture), torch, label="validated common init")
    if fixture.get("binding_sha256") != binding.get("binding_sha256"):
        raise NpuBenchRunnerError("native performance fixture binding differs from parent")
    if fixture.get("input_adapter") != adapter_identity:
        raise NpuBenchRunnerError("native performance fixture input adapter differs from manifest")
    return fixture


def _validated_native_case_partition(
    manifest: Mapping[str, Any], fixture: Mapping[str, Any]
) -> tuple[int, list[int], list[int]]:
    """Check that valid/empty case indices exactly partition the case range."""
    case_count = manifest.get("case_count")
    valid = manifest.get("valid_case_indices")
    empty = manifest.get("empty_case_indices")
    positive_count = not isinstance(case_count, bool) and isinstance(case_count, int) and case_count > 0
    if not positive_count:
        raise NpuBenchRunnerError("native performance manifest has no cases")
    if not isinstance(valid, list) or not isinstance(empty, list):
        raise NpuBenchRunnerError("native performance manifest case index lists are invalid")
    valid_indices = _validated_case_indices(valid, case_count, "valid")
    empty_indices = _validated_case_indices(empty, case_count, "empty")
    valid_set = set(valid_indices)
    empty_set = set(empty_indices)
    if valid_set & empty_set or valid_set | empty_set != set(range(case_count)):
        raise NpuBenchRunnerError("native performance manifest case partitions are incomplete")
    if (
        fixture.get("case_count") != case_count
        or fixture.get("valid_case_indices") != valid_indices
        or fixture.get("empty_case_indices") != empty_indices
    ):
        raise NpuBenchRunnerError("native performance fixture case metadata differs from manifest")
    return case_count, valid_indices, empty_indices


def _validated_native_case_record(
    record: Any, case_count: int, empty_index_set: set[int], seen_cases: set[int]
) -> tuple[int, str, str, bool]:
    """Validate one manifest case record and return its normalized fields."""
    if not isinstance(record, Mapping):
        raise NpuBenchRunnerError("native performance manifest has an invalid case fixture record")
    index = record.get("case")
    numeric_index = not isinstance(index, bool) and isinstance(index, int)
    in_range = numeric_index and 0 <= index < case_count
    if not in_range or index in seen_cases:
        raise NpuBenchRunnerError("native performance manifest case fixture index is invalid")
    expected_relative = f"{NATIVE_PERF_CASE_DIRNAME}/case_{index:06d}.pt"
    if record.get("path") != expected_relative or not isinstance(record.get("sha256"), str):
        raise NpuBenchRunnerError("native performance manifest case fixture path/digest is invalid")
    expected_empty = index in empty_index_set
    if record.get("empty_tensor") is not expected_empty:
        raise NpuBenchRunnerError("native performance manifest case fixture empty marker is invalid")
    return index, expected_relative, str(record["sha256"]), expected_empty


def _validated_native_case_fixtures(
    adapter: Path,
    manifest: Mapping[str, Any],
    *,
    torch: Any,
    binding: Mapping[str, Any],
    case_count: int,
    empty_indices: list[int],
    verify_case_payloads: bool,
) -> list[dict[str, Any]]:
    """Validate every per-case chunk the manifest claims, in manifest order."""
    case_fixtures = manifest.get("case_fixtures")
    if not isinstance(case_fixtures, list) or len(case_fixtures) != case_count:
        raise NpuBenchRunnerError("native performance manifest case fixture coverage is incomplete")
    empty_index_set = set(empty_indices)
    sanitized_cases: list[dict[str, Any]] = []
    seen_cases: set[int] = set()
    for record in case_fixtures:
        index, expected_relative, digest, expected_empty = _validated_native_case_record(
            record, case_count, empty_index_set, seen_cases
        )
        path = adapter / expected_relative
        _require_regular(path, "native performance case fixture")
        if _file_sha256(path) != digest:
            raise NpuBenchRunnerError("native performance case fixture digest differs from manifest")
        if verify_case_payloads:
            _validate_native_case_fixture_payload(
                path,
                torch=torch,
                binding=binding,
                case_index=index,
                empty_tensor=expected_empty,
            )
        seen_cases.add(index)
        sanitized_cases.append(
            {
                "case": index,
                "path": expected_relative,
                "sha256": digest,
                "empty_tensor": expected_empty,
            }
        )
    if seen_cases != set(range(case_count)):
        raise NpuBenchRunnerError("native performance manifest omitted a case fixture")
    return sanitized_cases


def _validate_native_case_fixture_payload(
    path: Path,
    *,
    torch: Any,
    binding: Mapping[str, Any],
    case_index: int,
    empty_tensor: bool,
) -> None:
    """Safely load one bounded input chunk before parent freezes it."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise NpuBenchRunnerError(
            "installed PyTorch lacks required weights_only native case loading"
        ) from exc
    except Exception as exc:
        raise NpuBenchRunnerError(
            f"native performance case fixture cannot be loaded with weights_only: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != NATIVE_PERF_CASE_SCHEMA:
        raise NpuBenchRunnerError("native performance case fixture payload schema is invalid")
    if payload.get("binding_sha256") != binding.get("binding_sha256"):
        raise NpuBenchRunnerError("native performance case fixture binding differs from parent")
    if payload.get("case") != case_index or payload.get("empty_tensor") is not empty_tensor:
        raise NpuBenchRunnerError("native performance case fixture identity differs from manifest")
    if "input_group" not in payload:
        raise NpuBenchRunnerError("native performance case fixture lacks input_group")
    _native_restricted_tree(dict(payload), torch)
    _native_reject_tensor_aliases(dict(payload), torch, label=f"validated case {case_index}")


def _validated_case_indices(value: Sequence[Any], case_count: int, label: str) -> list[int]:
    """Return the case indices in ``value`` once each is unique and in range."""
    result: list[int] = []
    for item in value:
        numeric = not isinstance(item, bool) and isinstance(item, int)
        if not numeric or not 0 <= item < case_count:
            raise NpuBenchRunnerError(f"native performance manifest {label} case index is invalid")
        if item in result:
            raise NpuBenchRunnerError(f"native performance manifest {label} case index is duplicated")
        result.append(item)
    return result


def _freeze_native_perf_fixture(
    workspace: Path,
    source: Path,
    *,
    binding: Mapping[str, Any],
    token: str,
) -> tuple[Path, Mapping[str, Any]]:
    """Parent-validate and publish a read-only fixture tree for phase B.

    Phase A's scratch is controlled by code that imported the task/candidate.
    Never hand that tree directly to the profiler.  Validate its manifest,
    common payload and every bounded case with ``weights_only=True``, copy it
    into a parent-owned content-addressed-ish directory, verify again, then
    chmod it read-only before phase B receives a mount of *only* that tree.
    """
    supplied_workspace = Path(workspace)
    if supplied_workspace.is_symlink():
        raise NpuBenchRunnerError("workspace must be a real non-symlink directory")
    workspace = supplied_workspace.resolve()
    _require_real_directory(workspace, "workspace")
    source = Path(source).resolve()
    if not _safe_prof_tag(token):
        raise NpuBenchRunnerError("native fixture freeze token is unsafe")
    _require_real_directory(source, "phase-A native fixture output")
    before = tree_sha256(source)
    source_info = _load_native_perf_manifest(source, binding=binding, verify_case_payloads=True)
    destination, incoming = _frozen_native_fixture_slots(workspace, token)
    published = False
    try:
        _copy_regular_tree(source, incoming)
        if tree_sha256(source) != before:
            raise NpuBenchRunnerError("phase-A native fixture changed while parent copied it")
        if tree_sha256(incoming) != before:
            raise NpuBenchRunnerError("parent-frozen native fixture copy digest differs from source")
        copied_info = _load_native_perf_manifest(incoming, binding=binding, verify_case_payloads=True)
        if copied_info != source_info:
            raise NpuBenchRunnerError("parent-frozen native fixture metadata differs after copy")
        os.replace(incoming, destination)
        _make_tree_read_only(destination)
        _require_real_read_only_tree(destination, "parent-frozen native performance fixture")
        final_info = _load_native_perf_manifest(destination, binding=binding, verify_case_payloads=True)
        if final_info != source_info:
            raise NpuBenchRunnerError("parent-frozen native fixture changed after publication")
        published = True
        return destination, final_info
    except Exception:
        # Do not leave a half-published fixture below a later evaluator-owned
        # root if the post-copy validation fails.  The helper is deliberately
        # scoped to exactly ``.npubench_exec/.frozen/<token>``.
        if not published:
            _cleanup_frozen_native_fixture(workspace, destination)
        raise
    finally:
        if incoming.exists():
            shutil.rmtree(incoming, ignore_errors=True)


def _frozen_native_fixture_slots(workspace: Path, token: str) -> tuple[Path, Path]:
    """Reserve the parent-owned publication and staging directories for a token."""
    execution_parent = _workspace_runtime_directory(
        workspace, EXECUTION_DIRNAME, "execution root"
    )
    parent = _ensure_real_child_directory(
        execution_parent, ".frozen", "parent-frozen native fixture root"
    )
    destination = parent / token
    incoming_parent = _ensure_real_child_directory(
        parent, ".incoming", "parent-frozen native fixture incoming root"
    )
    if destination.exists():
        raise NpuBenchRunnerError("parent-frozen native fixture destination already exists")
    incoming = _create_real_child_directory(
        incoming_parent, uuid.uuid4().hex, "parent-frozen native fixture incoming"
    )
    return destination, incoming


def _copy_frozen_native_fixture(source: Path, destination: Path) -> None:
    """Copy a parent-frozen fixture into the child adapter-local path."""
    source = Path(source).resolve()
    destination = Path(destination)
    _require_real_read_only_tree(source, "parent-frozen native performance fixture")
    _require_real_directory(destination, "native fixture mount target")
    if any(destination.iterdir()):
        raise NpuBenchRunnerError("native fixture mount target must start empty")
    _copy_regular_tree(source, destination)
    if tree_sha256(destination) != tree_sha256(source):
        raise NpuBenchRunnerError("isolated native fixture copy digest differs from frozen source")
    _make_tree_read_only(destination)


def _require_parent_frozen_native_fixture(workspace: Path, fixture: Path) -> Path:
    """Require an exact published fixture child, never phase-A scratch/input."""
    supplied_workspace = Path(workspace)
    if supplied_workspace.is_symlink():
        raise NpuBenchRunnerError("workspace must be a real non-symlink directory")
    workspace_root = supplied_workspace.resolve()
    _require_real_directory(workspace_root, "workspace")
    supplied_fixture = Path(fixture)
    if supplied_fixture.is_symlink():
        raise NpuBenchRunnerError("parent-frozen native fixture must not be a symlink")
    execution_parent = workspace_root / EXECUTION_DIRNAME
    _require_real_directory(execution_parent, "execution root")
    frozen_parent_path = execution_parent / ".frozen"
    _require_real_directory(frozen_parent_path, "parent-frozen native fixture root")
    fixture_root = supplied_fixture.resolve()
    frozen_parent = frozen_parent_path.resolve()
    if fixture_root.parent != frozen_parent or fixture_root.name in {"", ".incoming"}:
        raise NpuBenchRunnerError(
            "native fixture must be an exact parent-published .npubench_exec/.frozen child"
        )
    _require_real_read_only_tree(fixture_root, "parent-frozen native performance fixture")
    return fixture_root


def _cleanup_frozen_native_fixture(workspace: Path, frozen_fixture: Path) -> None:
    """Remove exactly one parent-owned published fixture tree.

    This is intentionally narrower than the regular execution-context
    cleanup: a caller cannot turn a stale/fabricated path into a recursive
    workspace deletion.  The child never receives a writable handle to this
    tree, but its files are read-only by construction, so restore mode bits
    only after the report/evidence publisher has finished with them.
    """
    supplied_workspace = Path(workspace)
    if supplied_workspace.is_symlink():
        return
    workspace_root = supplied_workspace.resolve()
    execution_parent = workspace_root / EXECUTION_DIRNAME
    if execution_parent.is_symlink() or not execution_parent.is_dir():
        return
    frozen_parent_path = execution_parent / ".frozen"
    if frozen_parent_path.is_symlink() or not frozen_parent_path.is_dir():
        return
    frozen_parent = frozen_parent_path.resolve()
    supplied = Path(frozen_fixture)
    if supplied.is_symlink():
        return
    target = supplied.resolve()
    if target.parent != frozen_parent or target == frozen_parent:
        return
    if target.is_symlink() or not target.is_dir():
        return
    try:
        for entry in sorted(target.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if entry.is_symlink():
                return
            if entry.is_dir():
                os.chmod(entry, 0o700)
            elif entry.is_file():
                os.chmod(entry, 0o600)
            else:
                return
        os.chmod(target, 0o700)
        shutil.rmtree(target)
    except OSError:
        # This transient root is never evidence; leave a failed cleanup for
        # an explicit housekeeping pass rather than widening deletion scope.
        pass


class _SidecarInputGroups:
    """Lazy, repeatable case sequence for evaluator-owned sidecar inputs.

    The sidecar JSON itself is metadata and is validated eagerly.  Tensor
    storage is created only when one case is requested, so precision and
    fixture generation never retain the complete benchmark (level1/3_Add is
    roughly 1.2 GiB if all 50 cases are materialized at once).
    """

    def __init__(self, descriptors: Sequence[Sequence[Mapping[str, Any]]], torch: Any, seed: int):
        self._descriptors = tuple(tuple(item for item in case) for case in descriptors)
        self._torch = torch
        self._seed = seed

    def __len__(self) -> int:
        return len(self._descriptors)

    def __iter__(self) -> Iterator[list[Any]]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int | slice) -> list[Any] | list[list[Any]]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("sidecar case index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return _materialize_sidecar_case(
            self._descriptors[index], torch=self._torch, seed=self._seed, case_index=index
        )


def _sidecar_case_generator(torch: Any, seed: int, case_index: int) -> Any:
    """Return a deterministic CPU generator for one sidecar case."""
    generator_type = getattr(torch, "Generator", None)
    if not callable(generator_type):
        raise NpuBenchRunnerError("PyTorch does not expose a CPU random generator")
    try:
        generator = generator_type(device="cpu")
    except TypeError:
        generator = generator_type()
    manual_seed = getattr(generator, "manual_seed", None)
    if not callable(manual_seed):
        raise NpuBenchRunnerError("PyTorch CPU generator cannot be seeded")
    # Keep the mapping stable across processes and independent of the number
    # or order of other cases.  The mask also handles very large signed seeds.
    manual_seed((int(seed) + 0x9E3779B9 * (case_index + 1)) & ((1 << 63) - 1))
    return generator


def _materialize_sidecar_case(
    descriptors: Sequence[Mapping[str, Any]], *, torch: Any, seed: int, case_index: int
) -> list[Any]:
    generator = _sidecar_case_generator(torch, seed, case_index)
    group: list[Any] = []
    for descriptor in descriptors:
        if descriptor["type"] == "tensor":
            dtype_name = str(descriptor["dtype"])
            dtype = _sidecar_torch_dtype(torch, dtype_name)
            shape = tuple(descriptor["shape"])
            bounds = descriptor.get("range")
            if bounds is None:
                group.append(torch.randn(shape, dtype=dtype, generator=generator))
            else:
                low, high = bounds
                # PyTorch builds differ in whether ``uniform_`` is implemented
                # for bfloat16 CPU tensors.  Sample in float32, whose CPU
                # generator semantics are stable, then cast to preserve the
                # descriptor's declared dtype.
                sample_dtype = torch.float32 if dtype_name == "bfloat16" else dtype
                sampled = torch.empty(shape, dtype=sample_dtype).uniform_(
                    float(low), float(high), generator=generator
                )
                group.append(sampled.to(dtype) if sample_dtype is not dtype else sampled)
        else:
            group.append(descriptor["value"])
    return group


def _assert_input_adapter_binding(
    adapter: Mapping[str, Any], binding: Mapping[str, Any]
) -> None:
    expected = binding.get("input_adapter")
    if expected is None:
        # A few internal unit tests intentionally provide a reduced binding;
        # production bindings always carry the immutable adapter identity.
        return
    if not isinstance(expected, Mapping) or dict(adapter) != dict(expected):
        raise NpuBenchRunnerError(
            "resolved input adapter identity differs from evaluation binding"
        )


def _assert_input_adapter_api(
    api: Mapping[str, Any], binding: Mapping[str, Any]
) -> None:
    provider = api.get("input_provider")
    if provider not in {"get_input_groups", "get_inputs", "sidecar_descriptor"}:
        raise NpuBenchRunnerError("reference task reported an invalid input adapter")
    expected = binding.get("input_adapter")
    if not isinstance(expected, Mapping):
        return
    expected_case_count = expected.get("case_count")
    actual = _input_adapter_identity(provider, case_count=expected_case_count)
    _assert_input_adapter_binding(actual, {"input_adapter": expected})


def _validate_sidecar_descriptors(cases: Sequence[Any]) -> list[list[dict[str, Any]]]:
    """Validate and normalize the supported native sidecar subset.

    This pass allocates metadata only.  It computes declared tensor bytes and
    rejects oversized one-tensor, one-case, or whole-sidecar workloads before
    the lazy materializer can allocate storage.
    """
    if not cases:
        raise NpuBenchRunnerError("sidecar descriptor adapter requires at least one case")
    normalized: list[list[dict[str, Any]]] = []
    total_bytes = 0
    for case_index, case in enumerate(cases):
        normalized_case, total_bytes = _validate_sidecar_case(case, case_index, total_bytes)
        normalized.append(normalized_case)
    return normalized


def _validate_sidecar_case(
    case: Any, case_index: int, total_bytes: int
) -> tuple[list[dict[str, Any]], int]:
    """Validate one sidecar case, returning its descriptors and running total."""
    if not isinstance(case, Mapping) or set(case) != {"inputs"}:
        raise NpuBenchRunnerError(
            f"sidecar descriptor case {case_index} must contain only an inputs list"
        )
    inputs = case.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise NpuBenchRunnerError(
            f"sidecar descriptor case {case_index} inputs must be a non-empty list"
        )
    normalized_case: list[dict[str, Any]] = []
    case_bytes = 0
    names: set[str] = set()
    for input_index, raw in enumerate(inputs):
        if not isinstance(raw, Mapping):
            raise NpuBenchRunnerError(
                f"sidecar descriptor case {case_index} input {input_index} must be an object"
            )
        label = f"{case_index}:{input_index}"
        kind = raw.get("type")
        if kind == "tensor":
            dtype, shape, tensor_bytes = _sidecar_tensor_extent(raw, label, names)
            case_bytes += tensor_bytes
            total_bytes += tensor_bytes
            if case_bytes > _SIDECAR_MAX_CASE_BYTES:
                raise NpuBenchRunnerError(
                    f"sidecar descriptor case {case_index} exceeds case byte limit"
                )
            if total_bytes > _SIDECAR_MAX_TOTAL_BYTES:
                raise NpuBenchRunnerError("sidecar descriptors exceed total byte limit")
            descriptor = _sidecar_tensor_descriptor(raw, label, dtype=dtype, shape=shape)
        elif kind == "attr":
            descriptor = _sidecar_attr_descriptor(raw, label, names)
        else:
            raise NpuBenchRunnerError(
                f"sidecar descriptor case {case_index} input {input_index} has unsupported type {kind!r}"
            )
        normalized_case.append(descriptor)
        names.add(descriptor["name"])
    return normalized_case, total_bytes


def _sidecar_descriptor_fields_ok(
    raw: Mapping[str, Any], allowed: set[str], names: set[str]
) -> bool:
    """Shared field checks: no unknown keys, a fresh name, a boolean ``required``."""
    if set(raw) - allowed:
        return False
    name = raw.get("name")
    if not isinstance(name, str) or not name or name in names:
        return False
    return "required" not in raw or isinstance(raw["required"], bool)


def _sidecar_shape_ok(shape: Any) -> bool:
    """True for a non-empty list of positive, non-boolean integer dimensions."""
    if not isinstance(shape, list) or not shape:
        return False
    return all(not isinstance(dim, bool) and isinstance(dim, int) and dim > 0 for dim in shape)


def _sidecar_range_ok(bounds: Any) -> bool:
    """True for an ordered finite two-element numeric range."""
    if not isinstance(bounds, list) or len(bounds) != 2:
        return False
    numeric = all(not isinstance(value, bool) and isinstance(value, (int, float)) for value in bounds)
    if not numeric:
        return False
    if not all(math.isfinite(float(value)) for value in bounds):
        return False
    return bounds[0] <= bounds[1]


def _sidecar_tensor_extent(
    raw: Mapping[str, Any], label: str, names: set[str]
) -> tuple[str, list[int], int]:
    """Validate a tensor descriptor's identity/dtype/shape and size it in bytes."""
    allowed = {"name", "type", "required", "dtype", "shape", "range"}
    fields_ok = _sidecar_descriptor_fields_ok(raw, allowed, names)
    if not fields_ok or raw.get("required", True) is False:
        raise NpuBenchRunnerError(f"sidecar tensor descriptor {label} has invalid fields")
    dtype = _SIDECAR_DTYPE_ALIASES.get(raw.get("dtype"))
    if dtype is None:
        raise NpuBenchRunnerError(f"sidecar tensor descriptor {label} has unsupported dtype")
    shape = raw.get("shape")
    if not _sidecar_shape_ok(shape):
        raise NpuBenchRunnerError(f"sidecar tensor descriptor {label} shape is invalid")
    elements = 1
    for dimension in shape:
        if dimension > _SIDECAR_MAX_DIM:
            raise NpuBenchRunnerError(
                f"sidecar tensor descriptor {label} shape dimension is too large"
            )
        elements *= max(1, dimension)
        if elements > _SIDECAR_MAX_ELEMENTS:
            raise NpuBenchRunnerError(
                f"sidecar tensor descriptor {label} shape has too many elements"
            )
    tensor_bytes = elements * _SIDECAR_DTYPE_ITEMSIZE[dtype]
    if tensor_bytes > _SIDECAR_MAX_TENSOR_BYTES:
        raise NpuBenchRunnerError(f"sidecar tensor descriptor {label} exceeds tensor byte limit")
    return dtype, list(shape), tensor_bytes


def _sidecar_tensor_descriptor(
    raw: Mapping[str, Any], label: str, *, dtype: str, shape: list[int]
) -> dict[str, Any]:
    """Validate the optional value range and return the normalized descriptor."""
    bounds = raw.get("range")
    normalized_range = None
    if bounds is not None:
        if not _sidecar_range_ok(bounds):
            raise NpuBenchRunnerError(f"sidecar tensor descriptor {label} range is invalid")
        if dtype in _SIDECAR_INT_DTYPES and any(not isinstance(value, int) for value in bounds):
            raise NpuBenchRunnerError(
                f"sidecar integer tensor descriptor {label} range must be integral"
            )
        if dtype == "bool":
            raise NpuBenchRunnerError(f"sidecar bool tensor descriptor {label} range is unsupported")
        normalized_range = list(bounds)
    return {
        "name": raw["name"],
        "type": "tensor",
        "dtype": dtype,
        "shape": list(shape),
        **({"range": normalized_range} if normalized_range is not None else {}),
    }


def _sidecar_attr_value_ok(value: Any, attr_dtype: str) -> bool:
    """True when a scalar attribute value exactly matches its declared dtype.

    Exact-type comparison is required here, not ``isinstance``: ``True`` is an
    ``int`` instance, so an ``isinstance`` check would silently accept a bool
    where the descriptor declares ``int``.  ``type(...)`` is hoisted into
    ``value_type`` so that intent stays explicit.
    """
    value_type = type(value)
    if attr_dtype == "float":
        return value_type in {int, float} and math.isfinite(float(value))
    if attr_dtype == "int":
        return value_type is int
    if attr_dtype == "bool":
        return value_type is bool
    return value_type is str


def _sidecar_attr_descriptor(
    raw: Mapping[str, Any], label: str, names: set[str]
) -> dict[str, Any]:
    """Validate one scalar attribute descriptor and return its normal form."""
    allowed = {"name", "type", "required", "dtype", "value"}
    fields_ok = _sidecar_descriptor_fields_ok(raw, allowed, names)
    if not fields_ok or not isinstance(raw.get("dtype"), str) or "value" not in raw:
        raise NpuBenchRunnerError(f"sidecar attr descriptor {label} has invalid fields")
    attr_dtype = raw["dtype"]
    if attr_dtype not in {"float", "int", "bool", "str"}:
        raise NpuBenchRunnerError(f"sidecar attr descriptor {label} has unsupported dtype")
    value = raw["value"]
    if not _sidecar_attr_value_ok(value, attr_dtype):
        raise NpuBenchRunnerError(f"sidecar attr descriptor {label} value does not match dtype")
    return {
        "name": raw["name"],
        "type": "attr",
        "dtype": attr_dtype,
        "value": float(value) if attr_dtype == "float" else value,
    }


def _sidecar_torch_dtype(torch: Any, name: str) -> Any:
    value = getattr(torch, name, None)
    if value is None:
        raise NpuBenchRunnerError(f"PyTorch does not expose sidecar dtype {name}")
    return value


def _write_adapter_proxy(path: Path, *, target: Path, root: Path, role: str) -> None:
    # The generated adapter is intentionally tiny and self-contained.  It
    # delegates module loading to the shipped runner by absolute path, which
    # preserves the actual task's __file__ and synthetic-package treatment.
    runner_path = runner_module_path()
    source = (
        "# Auto-generated NPUKernelBench profiler adapter; not benchmark truth.\n"
        "import importlib.util\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        f"_RUNNER = Path({str(runner_path)!r})\n"
        "# The runner is loaded by absolute path; append (never prepend) its\n"
        "# directory so its sibling modules import without shadowing others.\n"
        "if str(_RUNNER.parent) not in sys.path:\n"
        "    sys.path.append(str(_RUNNER.parent))\n"
        "_spec = importlib.util.spec_from_file_location(\"_npubench_runner_adapter\", _RUNNER)\n"
        "if _spec is None or _spec.loader is None:\n"
        "    raise RuntimeError(\"cannot load npubench runner adapter\")\n"
        "_runner = importlib.util.module_from_spec(_spec)\n"
        "sys.modules[_spec.name] = _runner\n"
        "_spec.loader.exec_module(_runner)\n"
        f"_task = _runner.load_task_module(Path({str(target)!r}), Path({str(root)!r}), role={role!r})\n"
        "for _name in (\"Model\", \"ModelNew\", \"get_input_groups\", \"get_inputs\", \"get_init_inputs\"):\n"
        "    if hasattr(_task, _name):\n"
        "        globals()[_name] = getattr(_task, _name)\n"
    )
    path.write_text(source, encoding="utf-8")
    os.chmod(path, 0o500)

