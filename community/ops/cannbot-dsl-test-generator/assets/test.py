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
from typing import NamedTuple, Any
import csv
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

from cannbotdsl_case_contract import case_calls, check_inputs, check_probe

TEST_DIR = Path(__file__).resolve().parent
OP_NAME = "__CANNBOTDSL_OP_NAME__"  # Bound from spec.op.name by prepare_golden.py.
CASEBOOK = TEST_DIR / "testcase.csv"


def _result_path(sheet, case_id):
    suffix = f"_{case_id}" if case_id else ""
    output_dir = TEST_DIR / "testcase_output" / sheet
    output_dir.mkdir(exist_ok=True)
    return output_dir / f"verify_result_{sheet}{suffix}.csv"


def _golden():
    path = TEST_DIR / f"{OP_NAME}_golden.py"
    spec = importlib.util.spec_from_file_location("cannbotdsl_golden", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"missing golden: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(sheet, case_id):
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
        if case_id is None or row.get("case_id") == case_id:
            found.append(row)
    if not found:
        raise ValueError(f"no cases selected: sheet={sheet!r} case_id={case_id!r}")
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
            f"cannot synchronize target device {device_type}; probe evidence is not reliable"
        )
    backend.synchronize()


def _assert_nonnumeric(actual, inputs, assertions, device, probe):
    if not isinstance(assertions, list) or not assertions:
        raise ValueError("nonnumeric TDD case has no assertions")
    records = []
    for assertion in assertions:
        records.append(_observe_assertion(actual, inputs, assertion, device, probe))
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


def _invoke(invocation, inputs, output_map, expected_error_code=None):
    run, golden, device = invocation
    inputs = _to_device(inputs, device)
    outputs = {}
    if output_map:
        if not isinstance(output_map, dict) or set(output_map) != set(
            golden.OUTPUT_NAMES
        ):
            raise ValueError("output_json must contain exactly the spec outputs")
        outputs = _to_device(golden.make_inputs(output_map, seed=0), device)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sheet", required=True, choices=("blackbox", "whitebox", "tdd")
    )
    parser.add_argument("--case-id")
    parser.add_argument("--device", required=True)
    parser.add_argument("--require-accelerator", action="store_true")
    args = parser.parse_args()
    if args.require_accelerator and torch.device(args.device).type == "cpu":
        print(
            "precision verification requires a non-CPU target device", file=sys.stderr
        )
        return 2
    try:
        rows = _rows(args.sheet, args.case_id)
        golden = _golden()
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    module_name = os.environ.get("CANNBOTDSL_OP_MODULE")
    entry_name = os.environ.get("CANNBOTDSL_OP_ENTRY", "run")
    run = None
    load_error = "CANNBOTDSL_OP_MODULE is unset"
    if module_name:
        try:
            run = getattr(importlib.import_module(module_name), entry_name)
        except (ImportError, AttributeError) as exc:
            load_error = str(exc)
    results = []
    for row in rows:
        try:
            if run is None:
                raise RuntimeError(load_error)
            status, detail = _run_case(row, Invocation(run, golden, args.device))
        except Exception as exc:
            status, detail = "failed", str(exc)
        results.append(
            {
                "sheet": row["sheet"],
                "case_id": row["case_id"],
                "device": args.device,
                "status": status,
                "detail": detail,
            }
        )
    result_path = _result_path(args.sheet, args.case_id)
    with result_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sheet", "case_id", "device", "status", "detail")
        )
        writer.writeheader()
        writer.writerows(results)
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


def _run_case(row, invocation):
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
    if expectation and expectation.startswith("raises_error:"):
        code = expectation.removeprefix("raises_error:")
        inputs = golden.make_inputs(spec_map, seed=int(row.get("seed") or 42))
        _invoke(invocation, inputs, output_map, expected_error_code=code)
        return "passed", json.dumps({"expected_error_code": code}, ensure_ascii=False)
    if expectation not in ("match_golden", "match_golden_nan", "assertions"):
        raise ValueError(f"unsupported expected outcome: {expectation}")
    assertions = json.loads(row.get("assertions_json") or "[]")
    repeat, bitwise, calls = case_calls(
        spec_map, int(row.get("seed") or 42), assertions
    )
    baselines = {}
    measurements = []
    for repetition in range(repeat):
        for index, call in enumerate(calls):
            if set(call["inputs"]) - public_names:
                raise ValueError("sequence inputs contain names outside spec")
            inputs = golden.make_inputs(call["inputs"], seed=call["seed"])
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
                _, expected = golden.simulate(call["inputs"], seed=call["seed"])
                if expectation == "match_golden_nan" and not _has_nan(expected):
                    raise AssertionError("spec expects NaN but Golden returned no NaN")
            probe_assertions = any(
                a["kind"] in {"probe_events", "probe_predicates"}
                for a in call["assertions"]
            )
            probe = None
            if probe_assertions:
                module = os.environ.get("CANNBOTDSL_PROBE_MODULE")
                probe = importlib.import_module(module) if module else None
                if (
                    probe is None
                    or not hasattr(probe, "reset")
                    or not hasattr(probe, "snapshot")
                ):
                    raise RuntimeError("probe assertions require reset()/snapshot()")
                probe.reset()
            actual = _invoke(invocation, inputs, output_map)
            _assert_device(actual, device)
            if probe_assertions or bitwise:
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
                records.extend(
                    _assert_nonnumeric(actual, inputs, observable, device, probe)
                )
            elif not numeric:
                raise ValueError("assertions case lacks observable output or probe")
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


def _observe_assertion(actual, inputs, assertion, device, probe):
    kind = assertion.get("kind")
    if kind == "probe_predicates":
        if probe is None or not hasattr(probe, "snapshot"):
            raise RuntimeError("probe predicates require snapshot()")
        observed = check_probe(probe.snapshot(), assertion["checks"])
    elif kind == "probe_events":
        if probe is None or not hasattr(probe, "snapshot"):
            raise RuntimeError(
                "probe_events requires CANNBOTDSL_PROBE_MODULE.snapshot()"
            )
        observed = probe.snapshot()
        if observed != assertion.get("events"):
            raise AssertionError(f"probe events mismatch: {observed!r}")
    elif kind == "output_shape" and isinstance(actual, torch.Tensor):
        observed = list(actual.shape)
        if observed != assertion.get("shape"):
            raise AssertionError(f"output shape {observed} != {assertion.get('shape')}")
    elif kind == "output_dtype" and isinstance(actual, torch.Tensor):
        observed = str(actual.dtype).removeprefix("torch.")
        if observed != assertion.get("dtype"):
            raise AssertionError(f"output dtype {observed} != {assertion.get('dtype')}")
    elif kind == "output_device" and isinstance(actual, torch.Tensor):
        observed = actual.device.type
        if observed != torch.device(device).type:
            raise AssertionError(f"output device {observed} != {device}")
    elif kind == "output_equals_input" and isinstance(actual, torch.Tensor):
        source = inputs.get(assertion.get("input"))
        if not isinstance(source, torch.Tensor) or not torch.equal(
            actual, source.to(actual.device)
        ):
            raise AssertionError("output differs from specified input")
        observed = True
    else:
        raise ValueError(f"unsupported or inapplicable assertion: {assertion!r}")
    return {"kind": kind, "observed": observed}


if __name__ == "__main__":
    sys.exit(main())
