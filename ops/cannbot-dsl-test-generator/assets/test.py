#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Fixed CANNBotDSL correctness runner for active testcase.csv rows."""

import argparse
import copy
import csv
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

import torch
from cannbotdsl_case_cache import CaseCache, canonical_output, load_golden, zero_buffers
from cannbotdsl_case_contract import (
    case_calls, check_inputs, validate_assertions,
    validate_tolerance, staged_policy,
)

TEST_DIR = Path(__file__).resolve().parent
OP_NAME = "__CANNBOTDSL_OP_NAME__"  # Bound from spec.op.name by prepare_golden.py.
CASEBOOK = TEST_DIR / "testcase.csv"


def _result_path(sheet, case_id, grep=None):
    suffix = f"_{case_id}" if case_id else ""
    if grep is not None:
        suffix = "_grep_" + hashlib.sha256(grep.encode("utf-8")).hexdigest()[:16]
    output_dir = TEST_DIR / "testcase_output" / sheet
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"verify_result_{sheet}{suffix}.csv"


def _golden():
    golden = load_golden(TEST_DIR / f"{OP_NAME}_golden.py")
    staged = {
        dtype: limits
        for dtype, limits in getattr(golden, "TOLERANCE", {}).items()
        if isinstance(limits, dict) and limits.get("metric") == "staged_relative"
    }
    if staged:
        # Validate even when selected cases only check behavior or empty outputs.
        validate_tolerance(staged)
    return golden


def _rows(sheet, case_id, grep=None):
    try:
        pattern = re.compile(grep) if grep is not None else None
    except re.error as exc:
        raise ValueError(f"invalid --grep: {exc}") from exc
    if not CASEBOOK.is_file():
        raise RuntimeError(f"missing casebook: {CASEBOOK}")
    with CASEBOOK.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "sheet" not in (reader.fieldnames or []):
            raise ValueError("testcase.csv has no sheet column")
        all_rows = list(reader)
    if any(row.get("sheet") not in ("blackbox", "whitebox", "tdd", "performance") for row in all_rows):
        raise ValueError("testcase.csv contains an unsupported logical sheet")
    found = []
    for row in all_rows:
        if (sheet is not None and row.get("sheet") != sheet) or row.get("status") != "active":
            continue
        if case_id is not None and row.get("case_id") != case_id:
            continue
        if pattern is not None and not pattern.search(row["case_id"]):
            continue
        found.append(row)
    if not found:
        raise ValueError(
            f"no cases selected: sheet={sheet!r} case_id={case_id!r} grep={grep!r}"
        )
    keys = [(row["sheet"], row["case_id"]) for row in all_rows]
    if len(keys) != len(set(keys)) or any(not row.get("inputs_json") for row in found):
        raise ValueError("duplicate case or missing inputs_json in testcase.csv")
    return found


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _assert_device(value, device, path="output"):
    if isinstance(value, torch.Tensor):
        if value.device.type != torch.device(device).type:
            raise AssertionError(
                f"{path}: output on {value.device}, requested {device}"
            )
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_device(item, device, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_device(item, device, f"{path}[{index}]")


def _synchronize(device):
    device_type = torch.device(device).type
    if device_type == "cpu":
        return
    backend = getattr(torch, device_type, None)
    if backend is None or not hasattr(backend, "synchronize"):
        raise RuntimeError(
            f"cannot synchronize target device {device_type}; repeated-call evidence is not reliable"
        )
    backend.synchronize()


def _assert_nonnumeric(actual, inputs, assertions, device):
    if not isinstance(assertions, list) or not assertions:
        raise ValueError("nonnumeric TDD case has no assertions")
    records = []
    for assertion in assertions:
        records.append(_observe_assertion(actual, inputs, assertion, device))
    return records


def _compare(actual, expected, tolerance, path="output", equal_nan=False, native=None):
    if isinstance(expected, dict):
        if not expected:
            raise AssertionError(f"{path}: no outputs to compare")
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{path}: output structure mismatch")
        measurements = []
        for key in expected:
            measurements.extend(
                _compare(actual[key], expected[key], tolerance, f"{path}.{key}", equal_nan,
                         None if native is None else lambda key=key: native()[key])
            )
        return measurements
    if isinstance(expected, (tuple, list)):
        if not expected:
            raise AssertionError(f"{path}: no outputs to compare")
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            raise AssertionError(f"{path}: output count mismatch")
        measurements = []
        for index, (a, e) in enumerate(zip(actual, expected)):
            measurements.extend(
                _compare(a, e, tolerance, f"{path}[{index}]", equal_nan,
                         None if native is None else lambda index=index: native()[index])
            )
        return measurements
    return _compare_tensor(actual, expected, tolerance, path, equal_nan, native)


def _compare_outputs(actual, expected, cache, inputs, output_map, equal_nan=False):
    expected = cache.validate_output(expected, inputs, output_map)
    actual = cache.validate_output(actual, inputs, output_map)
    records = []
    native_result = []

    def get_native():
        if not native_result:
            callback = getattr(cache.golden, "NATIVE_REFERENCE_FUNCTION", None)
            if not callable(callback):
                raise ValueError("NATIVE_REFERENCE_FUNCTION must bind an independent CPU reference")
            kwargs = {name: inputs[name] for name in cache.golden.INPUT_NAMES if name in inputs}
            kwargs.update(cache.golden.ATTR_DEFAULTS)
            for name in list(cache.golden.ATTR_DEFAULTS) + cache.golden.REQUIRED_ATTR_NAMES:
                if name in inputs:
                    kwargs[name] = inputs[name]
            result = callback(**copy.deepcopy(kwargs))
            checked = cache.validate_output(result, inputs, output_map)
            _assert_device(checked, "cpu")
            native_result.append(checked)
        return native_result[0]
    has_native = getattr(cache.golden, "NATIVE_REFERENCE_FUNCTION", None) is not None
    for name, reference in expected.items():
        value = actual[name]
        path = f"output.{name}"
        if reference is None:
            if value is not None:
                raise AssertionError(f"{path}: optional output presence mismatch")
            records.append({"path": path, "metric": "output_presence", "equal": True})
        elif isinstance(reference, (tuple, list)) and not reference:
            if not isinstance(value, (tuple, list)) or value:
                raise AssertionError(f"{path}: empty TensorList mismatch")
            records.append({"path": path, "metric": "output_list_length", "equal": True})
        else:
            records.extend(_compare(value, reference, cache.golden.TOLERANCE, path, equal_nan,
                                    (lambda name=name: get_native()[name]) if has_native else None))
    if not records:
        raise AssertionError("no output checks performed")
    return records


def _has_nan(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isnan(value).any())
    if isinstance(value, dict):
        return any(_has_nan(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_has_nan(item) for item in value)
    return False


def _error_matches(exc, code):
    if code in (getattr(exc, "code", None), getattr(exc, "error_code", None)):
        return True
    module_name = os.environ.get("CANNBOTDSL_ERROR_MATCHER")
    if not module_name:
        return False
    matcher = getattr(importlib.import_module(module_name), "match_error")
    return matcher(exc, code) is True


class Invocation(NamedTuple):
    run: Any
    golden: Any
    device: str


def _invoke(
    invocation,
    inputs,
    output_map,
    expected_error_code=None,
    unchanged=(),
    prepared_outputs=None,
    output_spec=None,
    dst_args=False,
):
    run, golden, device = invocation
    inputs = _to_device(inputs, device)
    if device != "cpu":
        module = sys.modules.get(run.__module__)
        if module and hasattr(module, "_aicpu_preprocess"):
            inputs = module._aicpu_preprocess(inputs)
    snapshots = {}
    for name in unchanged:
        value = inputs.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"input_unchanged requires Tensor input: {name}")
        snapshots[name] = (_clone_cpu(value), value.device, value.stride())
    outputs = {}
    dst_args = dst_args or bool(output_map)
    if dst_args:
        declared = (output_spec or {}).get("outputs") or [{"name": name} for name in golden.OUTPUT_NAMES]
        required = {item["name"] for item in declared if not item.get("optional", False)}
        if not isinstance(output_map, dict) or set(output_map) - set(golden.OUTPUT_NAMES):
            raise ValueError("output_json must use declared output names")
        supplied = {name: value for name, value in output_map.items() if value is not None}
        if required - set(supplied):
            raise ValueError("destination buffers missing required outputs")
        outputs = _to_device(
            prepared_outputs
            if prepared_outputs is not None
            else golden.make_inputs(supplied, seed=0) if supplied else {},
            device,
        )
        if set(outputs) != set(supplied):
            raise ValueError("destination buffers disagree with output_json")
        zero_buffers(outputs)
    try:
        result = run(**inputs, **outputs)
    except Exception as target_error:
        if expected_error_code is None:
            raise
        if not _error_matches(target_error, expected_error_code):
            raise AssertionError(
                f"target error does not match spec code {expected_error_code}: {target_error}"
            ) from target_error
        return None
    finally:
        for name, (before, original_device, original_stride) in snapshots.items():
            value = inputs[name]
            try:
                _bitwise_equal(_clone_cpu(value), before)
                if value.device != original_device or value.stride() != original_stride:
                    raise AssertionError("input metadata changed")
            except AssertionError as error:
                raise AssertionError(f"input_unchanged failed: {name}") from error
    if expected_error_code is not None:
        raise AssertionError(
            f"target accepted an input that should raise {expected_error_code}"
        )
    if not dst_args:
        return result
    return (
        outputs.get(golden.OUTPUT_NAMES[0])
        if len(golden.OUTPUT_NAMES) == 1
        else tuple(outputs.get(name) for name in golden.OUTPUT_NAMES)
    )


def _load_invocation(golden, device):
    module_name = os.environ.get("CANNBOTDSL_OP_MODULE")
    if not module_name:
        raise RuntimeError("CANNBOTDSL_OP_MODULE is unset")
    entry_name = os.environ.get("CANNBOTDSL_OP_ENTRY", "run")
    run = getattr(importlib.import_module(module_name), entry_name)
    return Invocation(run, golden, device)


def _case_result(row, device, invocation=None, load_error=None, cache=None):
    try:
        if invocation is None:
            raise RuntimeError(load_error)
        status, detail = _run_case(row, invocation, cache)
    except Exception as exc:
        status, detail = "failed", str(exc)
    return {
        "sheet": row["sheet"],
        "case_id": row["case_id"],
        "device": device,
        "status": status,
        "detail": detail,
    }


def _run_rows(rows, golden, device):
    invocation, load_error = None, None
    cache = CaseCache(TEST_DIR, golden)
    try:
        invocation = _load_invocation(golden, device)
    except Exception as exc:
        load_error = str(exc)
    return [_case_result(row, device, invocation, load_error, cache) for row in rows]


def _write_results(result_path, results):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            dir=result_path.parent,
            prefix=".verify-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(
                handle, fieldnames=("sheet", "case_id", "device", "status", "detail")
            )
            writer.writeheader()
            writer.writerows(results)
        temporary.replace(result_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(
        f"wrote {result_path} total={len(results)} failed={sum(row['status'] != 'passed' for row in results)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sheets = ("blackbox", "whitebox", "tdd", "performance")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--sheet", choices=sheets)
    scope.add_argument("--all", action="store_true", help="run all active cases across sheets")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case-id")
    selection.add_argument(
        "--grep", help="regular expression matching case_id within the sheet"
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--require-accelerator", action="store_true")
    args = parser.parse_args()
    if args.all and (args.case_id is not None or args.grep is not None):
        parser.error("--all cannot be combined with --case-id or --grep")
    selected_sheets = sheets if args.all else (args.sheet,)
    result_paths = {sheet: _result_path(sheet, args.case_id, args.grep) for sheet in selected_sheets}
    for result_path in result_paths.values():
        result_path.unlink(missing_ok=True)
    if args.require_accelerator and torch.device(args.device).type == "cpu":
        print(
            "precision verification requires a non-CPU target device", file=sys.stderr
        )
        return 2
    try:
        rows = _rows(args.sheet, args.case_id, args.grep)
        golden = _golden()
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    results = _run_rows(rows, golden, args.device)
    for sheet, result_path in result_paths.items():
        sheet_results = [row for row in results if row["sheet"] == sheet]
        if sheet_results:
            _write_results(result_path, sheet_results)
        else:
            print(f"skipped {sheet}: no active cases")
    return 1 if any(row["status"] != "passed" for row in results) else 0


def _compare_tensor(actual, expected, tolerance, path, equal_nan, native=None):
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise AssertionError(f"{path}: expected tensor output")
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"{path}: shape/dtype mismatch: {actual.shape}/{actual.dtype} vs {expected.shape}/{expected.dtype}"
        )
    dtype = str(expected.dtype).removeprefix("torch.")
    limits = tolerance.get(dtype)
    if not isinstance(limits, dict):
        raise AssertionError(f"{path}: no tolerance for {dtype}")
    validate_tolerance({dtype: limits})
    metric = limits["metric"]
    if metric == "staged_relative":
        return _compare_staged_relative(actual, expected, limits, path, equal_nan, native)
    if metric == "mean_relative":
        return _compare_mean_relative(actual, expected, limits, path, equal_nan)
    if metric == "bitwise_equal":
        # Numeric equality would treat +0.0 and -0.0 as equal; compare storage bits.
        passed = torch.equal(
            actual.contiguous().reshape(-1).view(torch.uint8),
            expected.contiguous().reshape(-1).view(torch.uint8),
        )
        detail = {"path": path, "dtype": dtype, "metric": metric, "equal": passed}
        if not passed:
            raise AssertionError(json.dumps(detail, ensure_ascii=False))
        return [detail]
    rtol, atol = limits.get("rtol"), limits.get("atol")
    numeric_limits = isinstance(rtol, (int, float)) and isinstance(atol, (int, float))
    if not numeric_limits or rtol < 0 or atol < 0:
        raise AssertionError(f"{path}: invalid rtol/atol")
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    if bool(finite.any()):
        reference = expected[finite].double()
        absolute_error = (actual[finite].double() - reference).abs()
        allowance = atol + rtol * reference.abs()
        normalized = torch.where(
            allowance > 0,
            absolute_error / allowance,
            torch.where(
                absolute_error == 0,
                torch.zeros_like(absolute_error),
                torch.full_like(absolute_error, float("inf")),
            ),
        )
        max_abs = float(absolute_error.max())
        max_normalized = float(normalized.max())
    else:
        max_abs = max_normalized = 0.0
    detail = {
        "path": path,
        "dtype": dtype,
        "metric": metric,
        "rtol": rtol,
        "atol": atol,
        "max_abs": max_abs,
        "max_normalized_error": max_normalized,
    }
    try:
        torch.testing.assert_close(
            actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
    except AssertionError as exc:
        raise AssertionError(
            f"{json.dumps(detail, ensure_ascii=False)}; {exc}"
        ) from exc
    return [detail]


def _compare_mean_relative(actual, expected, limits, path, equal_nan):
    """Joint strict mean and maximum relative errors from the confirmed policy."""
    dtype = str(expected.dtype).removeprefix("torch.")
    detail = {
        "path": path, "dtype": dtype, "metric": "mean_relative",
        "backend": limits.get("backend", "native"),
        "rtol": limits["rtol"], "max_relative_rtol": limits["max_relative_rtol"],
        "epsilon": limits.get("epsilon", 1e-7),
    }
    # Matching infinities are exact; NaNs only match for explicit NaN cases.
    actual, expected = actual.double(), expected.double()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    matching = (actual == expected) | (
        torch.isnan(actual) & torch.isnan(expected) & equal_nan
    )
    if not bool((finite | matching).all()):
        detail.update(passed=False, reason="NaN/Inf positions or signs mismatch")
        raise AssertionError(json.dumps(detail, ensure_ascii=False))
    if bool(finite.any()):
        errors = (actual[finite] - expected[finite]).abs() / (
            expected[finite].abs() + detail["epsilon"]
        )
        mere, mare = float(errors.mean()), float(errors.max())
    else:
        mere = mare = 0.0
    detail.update(mere=mere, mare=mare, finite_count=int(finite.sum()))
    detail["passed"] = mere < limits["rtol"] and mare < limits["max_relative_rtol"]
    if not detail["passed"]:
        raise AssertionError(json.dumps(detail, ensure_ascii=False))
    return [detail]


def _staged_relative_errors(actual, expected, epsilon):
    """Preserve ordinary arithmetic; rescale intermediate fp64 overflows only."""
    difference = (actual - expected).abs()
    denominator = expected.abs() + epsilon
    errors = difference / denominator
    overflow = (torch.isfinite(actual) & torch.isfinite(expected)
                & (~torch.isfinite(difference) | ~torch.isfinite(denominator)))
    if bool(overflow.any()):
        a, g = actual[overflow], expected[overflow]
        scale = torch.maximum(g.abs(), torch.full_like(g, epsilon))
        finite_difference = difference[overflow]
        numerator = finite_difference / scale
        lost_difference = ~torch.isfinite(finite_difference)
        if bool(lost_difference.any()):
            # Only reconstruct a difference when the original subtraction overflowed.
            # A finite difference retains more precision than subtracting a/scale and g/scale.
            numerator[lost_difference] = (
                a[lost_difference] / scale[lost_difference]
                - g[lost_difference] / scale[lost_difference]
            ).abs()
        errors[overflow] = numerator / (g.abs() / scale + epsilon / scale)
    return errors


def _compare_staged_relative(actual, expected, limits, path, equal_nan, native=None):
    """Complete staged decision, used by all fixed-runner execution modes."""
    dtype = str(expected.dtype).removeprefix("torch.")
    policy = staged_policy(dtype, limits)
    a, g = actual.double().clone(), expected.double().clone()
    if not torch.equal(torch.isnan(a), torch.isnan(g)):
        raise AssertionError(f"{path}: NaN positions or policy mismatch")
    both_inf = torch.isinf(a) & torch.isinf(g)
    if bool((both_inf & (torch.sign(a) != torch.sign(g))).any()):
        raise AssertionError(f"{path}: Inf signs mismatch")
    one_inf = torch.isinf(a) ^ torch.isinf(g)
    maximum = torch.finfo(expected.dtype).max
    a[one_inf & torch.isinf(a)] = torch.sign(a[one_inf & torch.isinf(a)]) * maximum
    g[one_inf & torch.isinf(g)] = torch.sign(g[one_inf & torch.isinf(g)]) * maximum
    valid = ~(torch.isnan(a) | torch.isnan(g) | both_inf)
    diff = (a - g).abs()
    errors = _staged_relative_errors(a, g, limits.get("epsilon", 1e-7))
    if bool(torch.isnan(errors[valid]).any()):
        raise ValueError(f"{path}: relative error computation produced NaN")
    error = errors[valid]
    mere = float(error.mean()) if error.numel() else 0.0
    mare = float(error.max()) if error.numel() else 0.0
    detail = dict(path=path, dtype=dtype, metric="staged_relative", mere=mere, mare=mare,
                  finite_count=int(valid.sum()), stage=1, passed=True, policy=policy)
    if mere < limits["rtol"] and mare < limits["max_relative_rtol"]:
        return [detail]
    mode = policy["reference_mode"]
    if mode == "native" and native is None:
        raise ValueError("staged_relative requires NATIVE_REFERENCE_FUNCTION")
    if mode != "truncated" and native is not None:
        reference = native()
        if not isinstance(
                reference, torch.Tensor) or reference.shape != expected.shape or reference.dtype != expected.dtype:
            raise ValueError(f"{path}: native reference shape/dtype mismatch")
        n = reference.to(device=g.device, dtype=torch.float64)
        if not torch.equal(torch.isnan(n), torch.isnan(expected)):
            raise ValueError(f"{path}: native reference NaN positions mismatch")
        detail["reference"] = "native"
    else:
        # Preserve the original Golden for the explicit truncated-reference path,
        # including one-sided Inf saturation semantics.
        n = expected.double()
        detail["reference"] = "truncated"
    native_diff = (n - g).abs()
    native_error = _staged_relative_errors(n, g, limits.get("epsilon", 1e-7))
    small = valid & (g.abs() < policy["small_value_threshold"])
    cancel = valid & (
        a.abs() < policy["cancel_zero_threshold"]) & (
        g.abs() >= policy["small_value_threshold"]) & (
            g.abs() < policy["cancel_boundary"])
    normal = valid & ~small & ~cancel
    candidate_bad = errors > limits["max_relative_rtol"]
    native_bad = native_error > limits["max_relative_rtol"]
    detail.update(stage=2, passed=True)
    for name, mask, bad, refbad in (
        ("small_value", small, diff > policy["small_value_error"], native_diff > policy["small_value_error"]),
        ("cancel", cancel, candidate_bad, native_bad),
        ("normal", normal, candidate_bad, native_bad),
    ):
        count, refcount = int((mask & bad).sum()), int((mask & refbad).sum())
        passed = (
            (count == 0 or (refcount > 0 and count / refcount <= policy["error_ratio_limit"]))
            if name == "normal"
            else count / max(refcount, 1) <= policy["error_ratio_limit"]
        )
        detail.update({name + "_error_count": count, name + "_reference_error_count": refcount,
                       name + "_total_count": int(mask.sum()), name + "_passed": passed})
        detail["passed"] = detail["passed"] and passed
    if not detail["passed"]:
        raise AssertionError(json.dumps(detail, ensure_ascii=False))
    return [detail]


def _run_case(row, invocation, cache):
    _, golden, device = invocation
    spec_map = json.loads(row["inputs_json"])
    if not isinstance(spec_map, dict):
        raise ValueError("inputs_json must be a mapping")
    public_names = (
        set(golden.INPUT_NAMES)
        | set(golden.ATTR_DEFAULTS)
        | set(golden.REQUIRED_ATTR_NAMES)
    )
    if set(spec_map) - public_names:
        raise ValueError("inputs_json contains names outside spec")
    expectation = row.get("expected")
    output_map = json.loads(row.get("output_json") or "{}")
    assertions = json.loads(row.get("assertions_json") or "[]")
    validate_assertions(assertions)
    if expectation and expectation.startswith("raises_error:"):
        code = expectation.removeprefix("raises_error:")
        data = cache.get(
            row,
            {
                "inputs": spec_map,
                "seed": int(row.get("seed") or 42),
                "assertions": assertions,
            },
            0,
        )
        inputs = copy.deepcopy(data["inputs"])
        _invoke(
            invocation,
            inputs,
            output_map,
            expected_error_code=code,
            unchanged=_unchanged_names(assertions),
            prepared_outputs=copy.deepcopy(data["outputs"]),
            output_spec=cache.spec, dst_args=row.get("note") == "dst_args",
        )
        return "passed", json.dumps({"expected_error_code": code}, ensure_ascii=False)
    if expectation not in ("match_golden", "match_golden_nan", "assertions"):
        raise ValueError(f"unsupported expected outcome: {expectation}")
    repeat, bitwise, calls = case_calls(
        spec_map, int(row.get("seed") or 42), assertions
    )
    baselines = {}
    measurements = []
    for repetition in range(repeat):
        for index, call in enumerate(calls):
            if set(call["inputs"]) - public_names:
                raise ValueError("sequence inputs contain names outside spec")
            data = cache.get(row, call, index)
            inputs = copy.deepcopy(data["inputs"])
            checks = [
                c
                for a in call["assertions"]
                if a["kind"] == "input_conditions"
                for c in a["checks"]
            ]
            records = check_inputs(inputs, checks) if checks else []
            numeric = expectation != "assertions"
            expected = None
            if numeric:
                expected = data["expected"]
                if expectation == "match_golden_nan" and not _has_nan(expected):
                    raise AssertionError("spec expects NaN but Golden returned no NaN")
            actual = _invoke(
                invocation,
                inputs,
                output_map,
                unchanged=_unchanged_names(call["assertions"]),
                prepared_outputs=copy.deepcopy(data["outputs"]),
                output_spec=cache.spec, dst_args=row.get("note") == "dst_args",
            )
            _assert_device(actual, device)
            # All successful calls obey the same public output contract, even
            # when this case requests behavioral assertions without a Golden.
            named_actual = canonical_output(
                cache.validate_output(actual, data["inputs"], output_map)
            )
            if bitwise:
                _synchronize(device)
            if numeric:
                records.extend(
                    _compare_outputs(
                        _to_device(named_actual, "cpu"),
                        expected,
                        cache, data["inputs"], output_map,
                        equal_nan=expectation == "match_golden_nan",
                    )
                )
            observable = [
                a for a in call["assertions"] if a["kind"] != "input_conditions"
            ]
            if observable:
                records.extend(_assert_nonnumeric(actual, inputs, observable, device))
            elif not numeric:
                raise ValueError("assertions case lacks observable behavior")
            if bitwise:
                snapshot = _clone_cpu(named_actual)
                if repetition == 0:
                    baselines[index] = snapshot
                else:
                    _bitwise_equal(snapshot, baselines[index])
            measurements.append(
                {"repetition": repetition, "call": index, "records": records}
            )
    return "passed", json.dumps(measurements, ensure_ascii=False)


def _clone_cpu(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_cpu(item) for item in value)
    raise ValueError("bitwise repeat requires tensor outputs")


def _bitwise_equal(actual, expected):
    if expected is None:
        if actual is not None:
            raise AssertionError("repeated call output presence changed")
    elif isinstance(expected, torch.Tensor):
        if (
            not isinstance(actual, torch.Tensor)
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or not torch.equal(
                actual.contiguous().reshape(-1).view(torch.uint8),
                expected.contiguous().reshape(-1).view(torch.uint8),
            )
        ):
            raise AssertionError("repeated call output changed bitwise")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError("repeated call output structure changed")
        for key in expected:
            _bitwise_equal(actual[key], expected[key])
    else:
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError("repeated call output structure changed")
        for left, right in zip(actual, expected):
            _bitwise_equal(left, right)


def _unchanged_names(assertions):
    return {
        name
        for item in assertions
        if item["kind"] == "input_unchanged"
        for name in item["inputs"]
    }


def _assertion_targets(actual, assertion):
    """Resolve the tensor targets an output assertion applies to.

    Operators may return a tuple/list of outputs or a dict of named outputs.
    An assertion can target one element via ``index`` (tuple/list) or
    ``output`` (dict name); without targeting it applies to every element.
    Returns ``(targets, label)`` where ``label`` describes the selection.
    """
    index = assertion.get("index")
    name = assertion.get("output")
    if isinstance(actual, (tuple, list)):
        if index is not None:
            if type(index) is not int or not 0 <= index < len(actual):
                raise AssertionError(
                    f"output index {index} outside {len(actual)} outputs"
                )
            return [actual[index]], f"[{index}]"
        return list(actual), ""
    if isinstance(actual, dict):
        if name is not None:
            if name not in actual:
                raise AssertionError(
                    f"output {name!r} not among {sorted(actual)}"
                )
            return [actual[name]], f".{name}"
        return list(actual.values()), ""
    return [actual], ""


def _observe_assertion(actual, inputs, assertion, device):
    kind = assertion.get("kind")
    if kind == "input_unchanged":
        return {"kind": kind, "observed": True}  # Checked around the actual device invocation in _invoke.
    if kind not in (
        "output_shape",
        "output_dtype",
        "output_device",
        "output_equals_input",
    ):
        raise ValueError(f"unsupported or inapplicable assertion: {assertion!r}")
    targets, label = _assertion_targets(actual, assertion)
    observed = []
    for target in targets:
        if not isinstance(target, torch.Tensor):
            raise AssertionError(f"output{label} is not a Tensor")
        if kind == "output_shape":
            value = list(target.shape)
            if value != assertion.get("shape"):
                raise AssertionError(
                    f"output{label} shape {value} != {assertion.get('shape')}"
                )
        elif kind == "output_dtype":
            value = str(target.dtype).removeprefix("torch.")
            if value != assertion.get("dtype"):
                raise AssertionError(
                    f"output{label} dtype {value} != {assertion.get('dtype')}"
                )
        elif kind == "output_device":
            value = target.device.type
            if value != torch.device(device).type:
                raise AssertionError(f"output{label} device {value} != {device}")
        else:
            source = inputs.get(assertion.get("input"))
            if not isinstance(source, torch.Tensor) or not torch.equal(
                target, source.to(target.device)
            ):
                raise AssertionError(
                    f"output{label} differs from specified input"
                )
            value = True
        observed.append(value)
    return {
        "kind": kind,
        "observed": observed[0] if len(observed) == 1 else observed,
    }


if __name__ == "__main__":
    sys.exit(main())
