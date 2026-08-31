# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Task loading and precision semantics for the NPUKernelBench runner.

This module owns the two halves of one precision case: bringing an untrusted
benchmark task into the process (synthetic package installation, module
execution, model construction/placement, input cloning and movement) and
comparing a reference output against a candidate output under the reviewed
verifier's NaN/Inf, integer, complex and float tolerance rules.

It imports only ``npubench_core``.  ``npubench_runner`` re-exports its public
surface, so importers keep using the runner module path.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import importlib.machinery
import importlib.util
import random
import sys
import types
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from npubench_core import (
    ALLCLOSE_TOLERANCES,
    DEFAULT_SEED,
    INT_LSB_TOLERANCE,
    NPU_LIMITS,
    NpuBenchRunnerError,
    REQUIRED_MATCHED_RATIO,
    _file_sha256,
    _require_regular,
)


def load_task_module(task_path: Path, root: Path, *, role: str = "reference") -> types.ModuleType:
    """Load a task from its real path, preserving ``__file__`` and packages.

    A relative import in a non-package task is rejected before execution.
    Package tasks are installed under a fresh synthetic top-level namespace,
    preventing collisions between benchmark task names while preserving their
    actual package directory layout.
    """
    task_path = Path(task_path).resolve()
    root = Path(root).resolve()
    _require_regular(task_path, f"{role} module")
    try:
        task_path.relative_to(root)
    except ValueError as exc:
        raise NpuBenchRunnerError(f"{role} module is outside its declared root") from exc
    syntax = _parse_task_syntax(task_path, role)
    package_parts, package_root = _package_layout(task_path, root)
    _assert_task_relative_imports(syntax, task_path, package_parts, package_root, role)
    module_name = _task_module_name(task_path, package_parts, package_root, role)
    module = _exec_task_module(module_name, task_path, role)
    if Path(str(getattr(module, "__file__", ""))).resolve() != task_path:
        raise NpuBenchRunnerError(f"{role} module did not retain the staged task __file__")
    return module


def _parse_task_syntax(task_path: Path, role: str) -> ast.AST:
    """Parse the frozen task without executing it, rejecting non-UTF-8 sources."""
    try:
        source = task_path.read_text(encoding="utf-8")
        return ast.parse(source, filename=str(task_path))
    except UnicodeDecodeError as exc:
        raise NpuBenchRunnerError(f"{role} module is not UTF-8 Python") from exc


def _assert_task_relative_imports(
    syntax: ast.AST,
    task_path: Path,
    package_parts: Sequence[str],
    package_root: Path | None,
    role: str,
) -> None:
    """Reject relative imports that have no package, or that escape the bundle."""
    relative_imports = any(
        isinstance(node, ast.ImportFrom) and node.level > 0 for node in ast.walk(syntax)
    )
    if relative_imports and package_root is None:
        raise NpuBenchRunnerError(
            f"{role} task uses a relative import but is not inside a package: {task_path.name}"
        )
    if package_root is None:
        return
    max_relative_level = len(package_parts) + 1
    for node in ast.walk(syntax):
        if isinstance(node, ast.ImportFrom) and node.level > max_relative_level:
            raise NpuBenchRunnerError(
                f"{role} task relative import escapes its frozen bundle root"
            )


def _task_module_name(
    task_path: Path, package_parts: Sequence[str], package_root: Path | None, role: str
) -> str:
    """Reserve a collision-free synthetic module name for one frozen task."""
    digest = _file_sha256(task_path).encode("ascii")
    token = hashlib.sha256(str(task_path).encode("utf-8") + digest).hexdigest()[:16]
    namespace = f"_npubench_{role}_{token}"
    if package_root is None:
        return namespace
    _install_synthetic_packages(namespace, package_parts, package_root)
    return ".".join((namespace, *package_parts, task_path.stem))


def _exec_task_module(module_name: str, task_path: Path, role: str) -> types.ModuleType:
    """Execute the untrusted task body, normalising any import-time failure."""
    old_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(module_name, task_path)
        if spec is None or spec.loader is None:
            raise NpuBenchRunnerError(f"cannot create import spec for {role} task")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except NpuBenchRunnerError:
        raise
    except Exception as exc:
        raise NpuBenchRunnerError(
            f"{role} task import failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.dont_write_bytecode = old_dont_write
    return module


def _run_precision_cases(
    groups: Any,
    torch: Any,
    reference_model: Any,
    candidate_model: Any,
    device_value: Any,
) -> list[dict[str, Any]]:
    """Compare both models over every case, one independently cloned input each.

    The task owns generation.  It is intentionally called exactly once; both
    model invocations receive recursively cloned copies of that one result.
    """
    case_reports: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        reference_inputs = _move_value(_clone_value(group, torch), device_value, torch)
        candidate_inputs = _move_value(_clone_value(group, torch), device_value, torch)
        reference_output = None
        candidate_output = None
        try:
            reference_output = _invoke_model(reference_model, reference_inputs, "reference", index)
            candidate_output = _invoke_model(candidate_model, candidate_inputs, "candidate", index)
            input_type, input_dtype = _infer_input_type(group, torch)
            passed, metrics, reason = compare_outputs(
                reference_output,
                candidate_output,
                torch,
                input_type=input_type,
                input_dtype=input_dtype,
            )
        except Exception as exc:
            passed, metrics, reason = False, {}, f"case execution failed: {type(exc).__name__}: {exc}"
        case_reports.append(
            {
                "case": index,
                "status": "PASS" if passed else "FAIL",
                "metrics": metrics,
                "reason": reason if not passed else "",
            }
        )
        # Release CPU and device copies before the next potentially large case.
        del reference_inputs, candidate_inputs, reference_output, candidate_output, group
    return case_reports


def seed_everything(
    seed: int = DEFAULT_SEED,
    *,
    torch_module: Any | None = None,
    existing: Sequence[str] | None = None,
) -> list[str]:
    """Seed Python, NumPy (when installed), PyTorch CPU and NPU deterministically."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise NpuBenchRunnerError("seed must be an integer")
    events = list(existing or [])
    random.seed(seed)
    events.append("python")
    try:
        numpy = importlib.import_module("numpy")
        numpy.random.seed(seed)
        events.append("numpy")
    except ModuleNotFoundError:
        events.append("numpy_unavailable")
    torch = torch_module
    if torch is not None:
        manual_seed = getattr(torch, "manual_seed", None)
        if callable(manual_seed):
            manual_seed(seed)
            events.append("torch")
        npu = getattr(torch, "npu", None)
        for name in ("manual_seed", "manual_seed_all"):
            function = getattr(npu, name, None)
            if callable(function):
                function(seed)
                events.append(f"torch.npu.{name}")
    return events


def compare_outputs(
    reference: Any,
    candidate: Any,
    torch: Any,
    *,
    input_type: str = "no_tensor",
    input_dtype: Any | None = None,
) -> tuple[bool, dict[str, Any], str]:
    """Compare outputs with the reviewed NPU benchmark verifier semantics.

    The comparator intentionally does not require output dtype equality:
    floating candidate values are cast to the reference dtype before the three
    benchmark checks, as the reviewed verifier does.  ``input_type`` is needed
    to distinguish quantization output from integer computation; the runner
    derives it from each old-format task input group before calling this API.
    """
    return _compare_output_leaf(
        reference,
        candidate,
        torch,
        path="output",
        input_type=input_type,
        input_dtype=input_dtype,
    )


def _compare_output_leaf(
    reference: Any,
    candidate: Any,
    torch: Any,
    *,
    path: str,
    input_type: str,
    input_dtype: Any | None,
) -> tuple[bool, dict[str, Any], str]:
    tensor_type = getattr(torch, "Tensor", ())
    if isinstance(reference, tensor_type):
        if not isinstance(candidate, tensor_type):
            return False, {}, f"{path}: candidate output is not a tensor"
        return _compare_tensor(
            reference,
            candidate,
            torch,
            path=path,
            input_type=input_type,
            input_dtype=input_dtype,
        )
    if type(reference) is not type(candidate):
        return False, {}, f"{path}: output type differs"
    if isinstance(reference, Mapping):
        return _compare_output_mapping(
            reference,
            candidate,
            torch,
            path=path,
            input_type=input_type,
            input_dtype=input_dtype,
        )
    if isinstance(reference, (list, tuple)):
        return _compare_output_sequence(
            reference,
            candidate,
            torch,
            path=path,
            input_type=input_type,
            input_dtype=input_dtype,
        )
    try:
        equal = reference == candidate
    except Exception as exc:
        return False, {}, f"{path}: scalar comparison failed: {type(exc).__name__}: {exc}"
    if isinstance(equal, bool):
        return equal, {"kind": "scalar", "exact": equal}, "" if equal else f"{path}: scalar values differ"
    return False, {}, f"{path}: unsupported output type {type(reference).__name__}"


def _compare_output_mapping(
    reference: Mapping[Any, Any],
    candidate: Mapping[Any, Any],
    torch: Any,
    *,
    path: str,
    input_type: str,
    input_dtype: Any | None,
) -> tuple[bool, dict[str, Any], str]:
    """Compare two mappings key by key, reporting the first differing child."""
    # ``dict_keys`` equality is order independent, matching the reviewed
    # verifier rather than requiring the original insertion order.
    if reference.keys() != candidate.keys():
        return False, {}, f"{path}: mapping keys differ"
    all_metrics: dict[str, Any] = {"children": {}}
    for key in reference:
        ok, metrics, reason = _compare_output_leaf(
            reference[key],
            candidate[key],
            torch,
            path=f"{path}.{key}",
            input_type=input_type,
            input_dtype=input_dtype,
        )
        all_metrics["children"][str(key)] = metrics
        if not ok:
            return False, all_metrics, reason
    return True, all_metrics, ""


def _compare_output_sequence(
    reference: Sequence[Any],
    candidate: Sequence[Any],
    torch: Any,
    *,
    path: str,
    input_type: str,
    input_dtype: Any | None,
) -> tuple[bool, dict[str, Any], str]:
    """Compare two sequences element by element, first difference wins."""
    if len(reference) != len(candidate):
        return False, {}, f"{path}: sequence length differs"
    metrics: dict[str, Any] = {"children": []}
    for index, (left, right) in enumerate(zip(reference, candidate)):
        ok, child_metrics, reason = _compare_output_leaf(
            left,
            right,
            torch,
            path=f"{path}[{index}]",
            input_type=input_type,
            input_dtype=input_dtype,
        )
        metrics["children"].append(child_metrics)
        if not ok:
            return False, metrics, reason
    return True, metrics, ""


def _compare_tensor(
    reference: Any,
    candidate: Any,
    torch: Any,
    *,
    path: str,
    input_type: str,
    input_dtype: Any | None,
) -> tuple[bool, dict[str, Any], str]:
    """Apply tensor semantics after the verifier's detach-to-CPU normalization."""
    reference = _detach_to_cpu(reference)
    candidate = _detach_to_cpu(candidate)
    if tuple(reference.shape) != tuple(candidate.shape):
        return False, {}, f"{path}: tensor shape differs"

    nonfinite_ok, nonfinite_reason, nonfinite_metrics = _check_nan_inf(
        reference, candidate, torch, path
    )
    if not nonfinite_ok:
        return False, nonfinite_metrics, nonfinite_reason

    finite_mask = torch.isfinite(reference) & torch.isfinite(candidate)
    finite_count = int(finite_mask.sum().item())
    total_numel = int(reference.numel())
    common = {
        "input_type": input_type,
        "input_dtype": _dtype_name(input_dtype) if input_dtype is not None else None,
        "finite_count": finite_count,
        "total_numel": total_numel,
        **nonfinite_metrics,
    }
    # After equal NaN/Inf masks and Inf signs are checked, the source verifier
    # accepts an all-nonfinite tensor without a numeric comparison.
    if finite_count == 0:
        return True, {"kind": "all_nonfinite", "checks": {"nonfinite": True}, **common}, ""

    reference_finite = reference[finite_mask]
    candidate_finite = candidate[finite_mask]
    if _is_bool_dtype(reference_finite.dtype):
        equal = bool(torch.equal(reference_finite, candidate_finite))
        metrics = {
            "kind": "bool",
            "exact": equal,
            "checks": {"exact": equal, "nonfinite": True},
            **common,
        }
        return equal, metrics, "" if equal else f"{path}: bool values differ"
    passed, metrics, reason = _compare_finite_tensor(
        reference,
        candidate,
        reference_finite,
        candidate_finite,
        torch,
        path=path,
        input_type=input_type,
    )
    metrics.update(common)
    metrics.setdefault("checks", {})["nonfinite"] = True
    return passed, metrics, reason


def _compare_finite_tensor(
    reference: Any,
    candidate: Any,
    reference_finite: Any,
    candidate_finite: Any,
    torch: Any,
    *,
    path: str,
    input_type: str,
) -> tuple[bool, dict[str, Any], str]:
    """Route the finite elements to the integer/complex/float comparison."""
    if _is_integer_dtype(reference_finite.dtype):
        return _compare_integer_tensor(
            reference_finite,
            candidate_finite,
            torch,
            path=path,
            input_type=input_type,
        )
    if _tensor_is_complex(reference) or _tensor_is_complex(candidate):
        return _compare_complex_tensor(reference, candidate, torch, path=path)
    return _compare_float_tensor(
        reference_finite,
        candidate_finite,
        torch,
        path=path,
    )


def _check_nan_inf(
    reference: Any,
    candidate: Any,
    torch: Any,
    path: str,
) -> tuple[bool, str, dict[str, Any]]:
    """Match NaN masks, Inf masks, and Inf signs before numeric comparison."""
    reference_nan = torch.isnan(reference)
    candidate_nan = torch.isnan(candidate)
    reference_nan_count = int(reference_nan.sum().item())
    candidate_nan_count = int(candidate_nan.sum().item())
    metrics = {
        "reference_nan_count": reference_nan_count,
        "candidate_nan_count": candidate_nan_count,
    }
    if _mask_any(reference_nan ^ candidate_nan):
        return (
            False,
            f"{path}: NaN mask mismatch: ref={reference_nan_count}/{reference.numel()}, "
            f"cand={candidate_nan_count}/{candidate.numel()}",
            metrics,
        )

    reference_inf = torch.isinf(reference)
    candidate_inf = torch.isinf(candidate)
    reference_inf_count = int(reference_inf.sum().item())
    candidate_inf_count = int(candidate_inf.sum().item())
    metrics.update(
        {
            "reference_inf_count": reference_inf_count,
            "candidate_inf_count": candidate_inf_count,
        }
    )
    if _mask_any(reference_inf ^ candidate_inf):
        return (
            False,
            f"{path}: Inf mask mismatch: ref={reference_inf_count}/{reference.numel()}, "
            f"cand={candidate_inf_count}/{candidate.numel()}",
            metrics,
        )
    if _mask_any(reference_inf):
        reference_sign = _tensor_sign(torch, reference[reference_inf])
        candidate_sign = _tensor_sign(torch, candidate[reference_inf])
        if not bool(torch.equal(reference_sign, candidate_sign)):
            return False, f"{path}: Inf sign mismatch", metrics
    return True, "", metrics


def _compare_integer_tensor(
    reference: Any,
    candidate: Any,
    torch: Any,
    *,
    path: str,
    input_type: str,
) -> tuple[bool, dict[str, Any], str]:
    """Match the verifier's quantized-versus-integer output decision table."""
    if input_type == "float":
        difference = (reference.to(torch.int64) - candidate.to(torch.int64)).abs()
        violation_count = int((difference > 1).sum().item())
        max_abs_diff = int(difference.max().item()) if difference.numel() else 0
        passed = violation_count == 0
        metrics = {
            "kind": "integer_quantized",
            "dtype": _dtype_name(reference.dtype),
            "lsb_tolerance": 1,
            "max_abs_diff": max_abs_diff,
            "violation_count": violation_count,
            "checks": {"lsb_tolerance": passed},
        }
        reason = "" if passed else f"{path}: quantized output exceeds ±1 LSB"
        return passed, metrics, reason

    dtype_name = _dtype_name(reference.dtype)
    tolerance = INT_LSB_TOLERANCE.get(dtype_name) if reference.dtype == candidate.dtype else None
    if tolerance is not None:
        difference = (candidate.to(torch.int32) - reference.to(torch.int32)).abs()
        max_abs_diff = int(difference.max().item()) if difference.numel() else 0
        passed = max_abs_diff <= tolerance
        metrics = {
            "kind": "integer_lsb",
            "dtype": dtype_name,
            "lsb_tolerance": tolerance,
            "max_abs_diff": max_abs_diff,
            "checks": {"lsb_tolerance": passed},
        }
        reason = "" if passed else f"{path}: integer output exceeds ±{tolerance} LSB"
        return passed, metrics, reason

    passed = bool(torch.equal(reference, candidate))
    metrics = {
        "kind": "integer_exact",
        "dtype": dtype_name,
        "exact": passed,
        "checks": {"exact": passed},
    }
    return passed, metrics, "" if passed else f"{path}: integer values differ"


def _compare_complex_tensor(
    reference: Any,
    candidate: Any,
    torch: Any,
    *,
    path: str,
) -> tuple[bool, dict[str, Any], str]:
    """Run the benchmark checks independently on complex real and imag parts."""
    # Deliberately preserve the reviewed verifier's direction: the candidate
    # component is supplied as ``golden`` and reference as ``actual``.  This
    # affects its denominator-based MERE and allclose bound.
    real_ok, real_metrics = _benchmark_accuracy(
        candidate.real, reference.real, candidate.real.dtype, torch
    )
    imag_ok, imag_metrics = _benchmark_accuracy(
        candidate.imag, reference.imag, candidate.imag.dtype, torch
    )
    passed = real_ok and imag_ok
    metrics = {
        "kind": "complex",
        "real": real_metrics,
        "imag": imag_metrics,
        "checks": {"real": real_ok, "imag": imag_ok},
    }
    reason = "" if passed else f"{path}: complex real/imag accuracy checks failed"
    return passed, metrics, reason


def _compare_float_tensor(
    reference: Any,
    candidate: Any,
    torch: Any,
    *,
    path: str,
) -> tuple[bool, dict[str, Any], str]:
    """Run the benchmark float checks after the verifier's candidate cast."""
    if candidate.dtype != reference.dtype:
        candidate = candidate.to(reference.dtype)
    # The reviewed implementation calls _check_accuracy_npu_benchmark(rhs,
    # lhs, lhs.dtype), not the more intuitive lhs/rhs order.  Preserve that
    # behavior exactly for compatibility with its MERE/allclose semantics.
    passed, metrics = _benchmark_accuracy(candidate, reference, reference.dtype, torch)
    metrics["kind"] = "floating"
    reason = "" if passed else f"{path}: NPU benchmark accuracy checks failed"
    return passed, metrics, reason


def _benchmark_accuracy(
    golden: Any,
    actual: Any,
    data_type: Any,
    torch: Any,
) -> tuple[bool, dict[str, Any]]:
    """Independent, compact implementation of the verifier's three checks."""
    golden_f = golden.float()
    actual_f = actual.float()
    dtype_name = _dtype_name(data_type)
    small_value_threshold, small_value_error, relative_threshold = NPU_LIMITS.get(
        dtype_name, NPU_LIMITS["float32"]
    )
    atol, rtol = ALLCLOSE_TOLERANCES.get(dtype_name, ALLCLOSE_TOLERANCES["float32"])

    absolute_difference = (actual_f - golden_f).abs()
    absolute_golden = golden_f.abs()
    small_mask = absolute_golden < small_value_threshold
    normal_mask = ~small_mask
    small_ok = absolute_difference <= small_value_error
    relative_error = absolute_difference / (absolute_golden + 1.0e-7)
    normal_ok = relative_error <= relative_threshold
    matched = torch.where(small_mask, small_ok, normal_ok)
    total_finite = int(matched.numel())
    matched_count = int(matched.sum().item())
    matched_ratio = matched_count / total_finite if total_finite else 1.0
    allclose = absolute_difference <= (atol + rtol * absolute_golden)
    allclose_violation_count = int((~allclose).sum().item()) if total_finite else 0
    mere = float(relative_error.mean().item()) if total_finite else None
    max_abs_diff = float(absolute_difference.max().item()) if total_finite else 0.0
    normal_count = int(normal_mask.sum().item())
    small_count = int(small_mask.sum().item())
    checks = {
        "max_error_cap": allclose_violation_count == 0,
        "required_matched_ratio": matched_ratio >= REQUIRED_MATCHED_RATIO,
        "MERE": mere is None or mere < relative_threshold,
    }
    return all(checks.values()), {
        "dtype": dtype_name,
        "matched_ratio": matched_ratio,
        "max_abs_diff": max_abs_diff,
        "MERE": mere,
        "rel_threshold": relative_threshold,
        "small_value_threshold": small_value_threshold,
        "small_value_error": small_value_error,
        "atol": atol,
        "rtol": rtol,
        "max_error_cap_violation_count": allclose_violation_count,
        "required_matched_ratio": REQUIRED_MATCHED_RATIO,
        "total_finite": total_finite,
        "matched_count": matched_count,
        "small_count": small_count,
        "normal_count": normal_count,
        "checks": checks,
    }


def _infer_input_type(inputs: Any, torch: Any) -> tuple[str, Any | None]:
    """Infer the reviewed verifier's float/int/no-tensor input classification."""
    tensor_type = getattr(torch, "Tensor", ())
    if isinstance(inputs, Mapping):
        values = list(inputs.values())
    elif isinstance(inputs, (list, tuple)):
        values = list(inputs)
    else:
        values = [inputs]
    tensors = [value for value in values if isinstance(value, tensor_type)]
    if tensors:
        dtype = max((value.dtype for value in tensors), key=_dtype_rank)
        return ("int" if _is_int_like_dtype(dtype) else "float"), dtype
    for value in values:
        if not isinstance(value, (list, tuple)) or not value:
            continue
        if not all(isinstance(element, tensor_type) for element in value):
            continue
        dtype = value[0].dtype
        return ("int" if _is_int_like_dtype(dtype) else "float"), dtype
    return "no_tensor", None


def _dtype_rank(dtype: Any) -> int:
    return {
        "float64": 100,
        "float32": 90,
        "float16": 80,
        "bfloat16": 70,
        "float8_e4m3fn": 60,
        "float8_e4m3": 60,
        "float8_e5m2fn": 60,
        "float8_e5m2": 60,
        "int64": 50,
        "int32": 40,
        "int16": 30,
        "int8": 20,
        "uint8": 20,
        "bool": 10,
    }.get(_dtype_name(dtype), 0)


def _is_int_like_dtype(dtype: Any) -> bool:
    return _is_bool_dtype(dtype) or not _dtype_flag(dtype, "is_floating_point") and not _dtype_flag(
        dtype, "is_complex"
    )


def _is_integer_dtype(dtype: Any) -> bool:
    return not _is_bool_dtype(dtype) and not _dtype_flag(dtype, "is_floating_point") and not _dtype_flag(
        dtype, "is_complex"
    )


def _is_bool_dtype(dtype: Any) -> bool:
    return _dtype_name(dtype) == "bool"


def _dtype_flag(dtype: Any, name: str) -> bool:
    value = getattr(dtype, name, False)
    return bool(value() if callable(value) else value)


def _tensor_is_complex(value: Any) -> bool:
    flag = getattr(value, "is_complex", False)
    return bool(flag() if callable(flag) else flag)


def _detach_to_cpu(value: Any) -> Any:
    value = value.detach()
    cpu = getattr(value, "cpu", None)
    return cpu() if callable(cpu) else value


def _mask_any(mask: Any) -> bool:
    value = mask.any()
    return bool(value.item() if hasattr(value, "item") else value)


def _tensor_sign(torch: Any, value: Any) -> Any:
    try:
        return torch.sign(value)
    except (AttributeError, RuntimeError):
        # Some PyTorch versions expose complex sign only as ``sgn``.  The
        # normal float/int path still takes the reviewed verifier's sign call.
        return torch.sgn(value)


def _package_layout(task_path: Path, root: Path) -> tuple[list[str], Path | None]:
    """Return package components wholly contained by ``root``.

    The old implementation walked one directory past ``root`` when the
    bundle root itself had an ``__init__.py``.  That made the synthetic
    package's search path a parent of the frozen bundle.  Keep the namespace
    package rooted *at* ``root`` instead: a root-level task can still use
    ``from .helper import ...`` as ``_npubench_x.helper``, but no import can
    bind an on-disk package outside the immutable bundle.
    """
    parts: list[str] = []
    cursor = task_path.parent
    while cursor != root:
        init = cursor / "__init__.py"
        if not init.is_file() or init.is_symlink():
            return [], None
        parts.insert(0, cursor.name)
        cursor = cursor.parent
    if parts:
        return parts, root
    root_init = root / "__init__.py"
    if root_init.is_file() and not root_init.is_symlink():
        return [], root
    return [], None


def _install_synthetic_packages(namespace: str, parts: Sequence[str], root: Path) -> None:
    _install_package(namespace, root)
    current = root
    name = namespace
    for part in parts:
        current = current / part
        name = f"{name}.{part}"
        _install_package(name, current)


def _install_package(name: str, directory: Path) -> None:
    existing = sys.modules.get(name)
    if existing is not None:
        return
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(directory)]  # type: ignore[attr-defined]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(directory)]
    module.__spec__ = spec
    module.__file__ = str(directory / "__init__.py")
    sys.modules[name] = module


def _validate_reference_api(module: types.ModuleType) -> dict[str, Any]:
    _resolve_model_constructor(module, preferred="Model", role="reference")
    groups_present = "get_input_groups" in vars(module)
    inputs_present = "get_inputs" in vars(module)
    groups_provider = getattr(module, "get_input_groups", None)
    inputs_provider = getattr(module, "get_inputs", None)
    if callable(groups_provider):
        input_provider = "get_input_groups"
    elif callable(inputs_provider):
        input_provider = "get_inputs"
    elif not groups_present and not inputs_present:
        # Native NPUKernelBench tasks may intentionally keep input generation
        # in the same-stem sidecar.  The resolver below only permits the
        # narrow, versioned descriptor schema; it never guesses from task code.
        input_provider = "sidecar_descriptor"
    else:
        raise NpuBenchRunnerError(
            "reference task input provider must be callable when present"
        )
    return {
        "model": "Model",
        "input_provider": input_provider,
        "init_provider": "get_init_inputs" if callable(getattr(module, "get_init_inputs", None)) else None,
    }


def _resolve_model_constructor(module: types.ModuleType, *, preferred: str, role: str) -> Callable[..., Any]:
    constructor = getattr(module, preferred, None)
    if callable(constructor):
        return constructor
    # The old verifier accepts another nn.Module subclass as a compatibility
    # fallback.  Restrict this to a single unambiguous public Model-like name.
    alternatives = [
        value
        for name, value in vars(module).items()
        if name in {"Model", "ModelNew"} and callable(value)
    ]
    if len(alternatives) == 1:
        return alternatives[0]
    raise NpuBenchRunnerError(f"{role} module must expose callable {preferred}")


def _get_init_args(
    module: types.ModuleType,
    *,
    fallback_module: types.ModuleType | None = None,
) -> tuple[Any, ...]:
    """Get constructor arguments, preferring candidate then frozen reference.

    ``fallback_module`` is deliberately only used when the preferred module
    has no ``get_init_inputs`` attribute.  A present but malformed candidate
    provider is an evaluation error rather than an invitation to silently use
    a different constructor contract.
    """
    provider = getattr(module, "get_init_inputs", None)
    if provider is None and fallback_module is not None:
        provider = getattr(fallback_module, "get_init_inputs", None)
    if provider is None:
        return ()
    if not callable(provider):
        raise NpuBenchRunnerError("get_init_inputs must be callable when present")
    value = provider()
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _get_input_groups(module: types.ModuleType) -> list[Any]:
    provider = getattr(module, "get_input_groups", None)
    if provider is None:
        provider = getattr(module, "get_inputs", None)
        if not callable(provider):
            raise NpuBenchRunnerError("task has no input provider")
        value = [provider()]
    else:
        if not callable(provider):
            raise NpuBenchRunnerError("get_input_groups must be callable")
        value = provider()
    if not isinstance(value, (list, tuple)) or not value:
        raise NpuBenchRunnerError("get_input_groups must return a non-empty list or tuple")
    return list(value)


def _construct_model(constructor: Callable[..., Any], init_args: Sequence[Any], role: str) -> Any:
    try:
        return constructor(*_clone_pythonish(init_args))
    except Exception as exc:
        raise NpuBenchRunnerError(f"{role} model construction failed: {type(exc).__name__}: {exc}") from exc


def _resolve_device(torch: Any, device: int | str | None) -> Any:
    if isinstance(device, str):
        return torch.device(device)
    if device is None:
        return torch.device("npu")
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise NpuBenchRunnerError("precision device must be a non-negative integer, 'cpu', or None")
    npu = getattr(torch, "npu", None)
    set_device = getattr(npu, "set_device", None)
    if callable(set_device):
        set_device(device)
    try:
        return torch.device("npu", device)
    except TypeError:
        return torch.device(f"npu:{device}")


def _move_model(model: Any, device: Any, role: str) -> None:
    move = getattr(model, "to", None)
    if not callable(move):
        raise NpuBenchRunnerError(f"{role} model has no .to(device) method")
    try:
        move(device)
    except Exception as exc:
        raise NpuBenchRunnerError(f"{role} model cannot move to {device}: {exc}") from exc


def _set_eval(model: Any) -> None:
    evaluate = getattr(model, "eval", None)
    if callable(evaluate):
        evaluate()


def _clone_value(value: Any, torch: Any) -> Any:
    tensor_type = getattr(torch, "Tensor", ())
    if isinstance(value, tensor_type):
        detached = value.detach() if callable(getattr(value, "detach", None)) else value
        return detached.clone()
    if isinstance(value, list):
        return [_clone_value(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item, torch) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item, torch) for key, item in value.items()}
    return _clone_pythonish(value)


def _clone_pythonish(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise NpuBenchRunnerError(
            f"input/state value cannot be cloned deterministically: {type(exc).__name__}: {exc}"
        ) from exc


def _drain_npu_stream(torch: Any) -> bool:
    """Best-effort stream drain before one bounded host-to-device copy retry.

    A failed or unavailable ``synchronize`` is not actionable at this point:
    the caller immediately retries the copy and surfaces the real
    ``RuntimeError`` if that retry also fails.  The outcome is returned rather
    than discarded inside the handler so the suppression stays explicit.
    """
    synchronize = getattr(getattr(torch, "npu", None), "synchronize", None)
    if not callable(synchronize):
        return False
    try:
        synchronize()
    except Exception:
        return False
    return True


def _move_value(value: Any, device: Any, torch: Any) -> Any:
    tensor_type = getattr(torch, "Tensor", ())
    if isinstance(value, tensor_type):
        try:
            return value.to(device)
        except RuntimeError as exc:
            # Host-level transient device fault (2026-08-22 A5 campaign:
            # TBE Slice kernel in the H2D copy path raises 507035 "vector core
            # execution is abnormal" intermittently, on multiple cards, even
            # for a bare torch copy).  One bounded retry after draining the
            # stream; the O5 infra retry loop absorbs persistent windows.
            if "507035" not in str(exc):
                raise
            _drain_npu_stream(torch)
            return value.to(device)
    if isinstance(value, list):
        return [_move_value(item, device, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value(item, device, torch) for item in value)
    if isinstance(value, dict):
        return {key: _move_value(item, device, torch) for key, item in value.items()}
    return value


def _invoke_model(model: Any, inputs: Any, role: str, index: int) -> Any:
    try:
        if isinstance(inputs, Mapping):
            return model(**inputs)
        if isinstance(inputs, (list, tuple)):
            return model(*inputs)
        return model(inputs)
    except Exception as exc:
        raise NpuBenchRunnerError(
            f"{role} invocation failed for case {index}: {type(exc).__name__}: {exc}"
        ) from exc


def _is_floating_tensor(value: Any) -> bool:
    return bool(getattr(value, "is_floating_point", lambda: False)()) or bool(
        getattr(value, "is_complex", lambda: False)()
    )


def _dtype_name(dtype: Any) -> str:
    name = str(dtype)
    return name.split(".")[-1].lower()

