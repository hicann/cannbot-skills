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

"""Fixed CANNBotDSL runner for testcase.csv blackbox/whitebox/TDD rows."""

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
from cannbotdsl_case_cache import CaseCache, load_golden
from cannbotdsl_case_contract import case_calls, check_inputs, validate_assertions

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
    return load_golden(TEST_DIR / f"{OP_NAME}_golden.py")


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
    if any(row.get("sheet") not in ("blackbox", "whitebox", "tdd") for row in all_rows):
        raise ValueError("testcase.csv contains an unsupported logical sheet")
    found = []
    for row in all_rows:
        if row.get("sheet") != sheet or row.get("status") != "active":
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


def _compare(actual, expected, tolerance, path="output", equal_nan=False):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{path}: output structure mismatch")
        measurements = []
        for key in expected:
            measurements.extend(
                _compare(
                    actual[key], expected[key], tolerance, f"{path}.{key}", equal_nan
                )
            )
        return measurements
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            raise AssertionError(f"{path}: output count mismatch")
        measurements = []
        for index, (a, e) in enumerate(zip(actual, expected)):
            measurements.extend(
                _compare(a, e, tolerance, f"{path}[{index}]", equal_nan)
            )
        return measurements
    return _compare_tensor(actual, expected, tolerance, path, equal_nan)


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
    if output_map:
        if not isinstance(output_map, dict) or set(output_map) != set(
            golden.OUTPUT_NAMES
        ):
            raise ValueError("output_json must contain exactly the spec outputs")
        outputs = _to_device(
            prepared_outputs
            if prepared_outputs is not None
            else golden.make_inputs(output_map, seed=0),
            device,
        )
        for tensor in outputs.values():
            if isinstance(tensor, torch.Tensor):
                tensor.zero_()
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
    if not outputs:
        return result
    return (
        outputs[golden.OUTPUT_NAMES[0]]
        if len(outputs) == 1
        else tuple(outputs[name] for name in golden.OUTPUT_NAMES)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sheet", required=True, choices=("blackbox", "whitebox", "tdd")
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case-id")
    selection.add_argument(
        "--grep", help="regular expression matching case_id within the sheet"
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--require-accelerator", action="store_true")
    args = parser.parse_args()
    result_path = _result_path(args.sheet, args.case_id, args.grep)
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
    return 1 if any(row["status"] != "passed" for row in results) else 0


def _compare_tensor(actual, expected, tolerance, path, equal_nan):
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
    metric = limits.get("metric")
    if metric not in ("max_relative", "bitwise_equal"):
        raise AssertionError(
            f"{path}: unsupported metric {metric}; do not substitute assert_close"
        )
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
            )
            _assert_device(actual, device)
            if bitwise:
                _synchronize(device)
            if numeric:
                records.extend(
                    _compare(
                        _to_device(actual, "cpu"),
                        expected,
                        golden.TOLERANCE,
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
                snapshot = _clone_cpu(actual)
                if repetition == 0:
                    baselines[index] = snapshot
                else:
                    _bitwise_equal(snapshot, baselines[index])
            measurements.append(
                {"repetition": repetition, "call": index, "records": records}
            )
    return "passed", json.dumps(measurements, ensure_ascii=False)


def _clone_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_cpu(item) for item in value)
    raise ValueError("bitwise repeat requires tensor outputs")


def _bitwise_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
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
