#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import argparse
import csv
import json
import logging
import math
import random
import re
from pathlib import Path

_logger = logging.getLogger(__name__)


KERNEL_COLUMNS = [
    "testcase_name", "network_name", "op_name",
    "input_shapes", "input_dtypes", "input_formats",
    "output_shapes", "output_dtypes", "output_formats",
    "input_ori_shapes", "input_ori_formats",
    "output_ori_shapes", "output_ori_formats",
    "attributes", "input_data_ranges", "precision_tolerances",
    "absolute_precision",
    "output_inplace_indexes", "output_shape_unknown_indexes",
    "is_enabled", "remark", "soc_series", "priority",
    "dump_file_prefix", "manual_input_binaries", "manual_golden_binaries",
]

DTYPE_MAX = {"float16": 65504.0, "bfloat16": 3.3895e38, "float32": 3.4e38}
PRECISION_TOLERANCE = {
    "float16": (0.001, 0.001),
    "bfloat16": (0.001, 0.001),
    "float32": (0.0001, 0.0001),
}


class KernelCsvError(RuntimeError):
    pass


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cases(path):
    data = load_json(path)
    if not isinstance(data, list):
        raise KernelCsvError(f"{path} must be a Step 5 final case list")
    return data


def load_attribute_names(operator_model_path):
    model = load_json(operator_model_path)
    attrs = model.get("attributes", [])
    names = []
    for attr in attrs:
        if isinstance(attr, dict) and isinstance(attr.get("name"), str):
            names.append(attr["name"])
    return set(names)


def strip_cpp_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def parse_def_cpp_symbols(def_cpp_path):
    if not def_cpp_path:
        return set(), set()
    text = Path(def_cpp_path).read_text(encoding="utf-8")
    text = strip_cpp_comments(text)
    attr_names = set(re.findall(r'\.Attr\s*\(\s*"([^"]+)"\s*\)', text))
    input_names = []
    for match in re.finditer(r'\.Input\s*\(\s*"([^"]+)"\s*\)', text):
        input_names.append((match.start(), match.group(1)))

    value_depend_inputs = set()
    for match in re.finditer(r'\.ValueDepend\s*\(', text):
        previous_inputs = [item for item in input_names if item[0] < match.start()]
        if previous_inputs:
            value_depend_inputs.add(previous_inputs[-1][1])
    return attr_names, value_depend_inputs


def py_literal(value):
    if isinstance(value, str):
        return repr(value)
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if math.isnan(value):
            return "float('nan')"
        if math.isinf(value):
            return "float('inf')" if value > 0 else "float('-inf')"
        return repr(value)
    if isinstance(value, int):
        return repr(value)
    if isinstance(value, tuple):
        return tuple_literal(value)
    if isinstance(value, list):
        return tuple_literal(tuple(value))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return repr(value)


def tuple_literal(values):
    values = tuple(values)
    if not values:
        return "()"
    inner = ", ".join(py_literal(v) for v in values)
    if len(values) == 1:
        inner += ","
    return f"({inner})"


def shape_tuple(shape, field_name):
    if not isinstance(shape, list):
        raise KernelCsvError(f"{field_name}.shape must be a list")
    return tuple(shape)


def get_tensor_entries(container, case_id, field_name):
    if not isinstance(container, dict):
        raise KernelCsvError(f"case {case_id}: {field_name} must be an object")
    return list(container.items())


def is_present_output(spec):
    return spec.get("present", True) is not False


def tensor_shape_entry(spec, name):
    kind = spec.get("kind")
    if kind == "tensor_list":
        tensors = spec.get("tensors", [])
        if not isinstance(tensors, list):
            raise KernelCsvError(f"{name}.tensors must be a list")
        return tuple(shape_tuple(sub.get("shape"), f"{name}.tensors[]") for sub in tensors)
    if kind == "tensor":
        return shape_tuple(spec.get("shape"), name)
    raise KernelCsvError(f"{name}.kind must be tensor or tensor_list")


def tensor_dtype_entry(spec, name):
    kind = spec.get("kind")
    if kind == "tensor_list":
        dtype = spec.get("dtype")
        if dtype is None:
            tensors = spec.get("tensors", [])
            if not tensors:
                raise KernelCsvError(f"{name}.dtype is missing")
            dtype = tensors[0].get("dtype")
        return (dtype,)
    if kind == "tensor":
        dtype = spec.get("dtype")
        if dtype is None:
            raise KernelCsvError(f"{name}.dtype is missing")
        return dtype
    raise KernelCsvError(f"{name}.kind must be tensor or tensor_list")


def tensor_format_entry(spec, name):
    fmt = spec.get("format")
    if not isinstance(fmt, str) or not fmt:
        raise KernelCsvError(f"{name}.format must be a non-empty string")
    return fmt


def _range_normal_bounds(value_domain):
    lo = value_domain.get("min") if value_domain else None
    hi = value_domain.get("max") if value_domain else None
    return (lo if lo is not None else -10.0, hi if hi is not None else 10.0)


def _range_normal(value_domain, rng):
    if value_domain:
        domain_type = value_domain.get("type")
        if domain_type == "positive":
            return (0.01, 10.0)
        if domain_type == "non_negative":
            return (0.0, 10.0)
        if domain_type == "non_zero":
            a, b = rng.uniform(-10, 10), rng.uniform(-10, 10)
            a, b = min(a, b), max(a, b)
            if abs(a) < 0.1:
                a = -10.0
            if abs(b) < 0.1:
                b = 10.0
            return (round(a, 2), round(b, 2))
        if domain_type == "range":
            return _range_normal_bounds(value_domain)
    a, b = rng.uniform(-10, 10), rng.uniform(-10, 10)
    return (round(min(a, b), 2), round(max(a, b), 2))


def _range_negative(value_domain, rng):
    if value_domain and value_domain.get("type") == "range":
        lo = value_domain.get("min")
        hi = value_domain.get("max")
        eff_lo = lo if lo is not None else -100.0
        eff_hi = min(0, hi) if hi is not None else -0.01
        if eff_lo < eff_hi:
            return (eff_lo, eff_hi)
        return _range_normal_bounds(value_domain)
    a, b = rng.uniform(-100, -0.01), rng.uniform(-100, -0.01)
    return (round(min(a, b), 2), round(max(a, b), 2))


def _range_tiny_pos(value_domain):
    if value_domain and value_domain.get("type") == "range":
        lo = value_domain.get("min")
        hi = value_domain.get("max")
        tp_lo = max(lo, 1e-7) if lo is not None else 1e-7
        tp_hi = min(hi, 1e-5) if hi is not None else 1e-5
        if tp_lo < tp_hi:
            return (tp_lo, tp_hi)
        return _range_normal_bounds(value_domain)
    return (1e-7, 1e-5)


def _range_near_zero(value_domain):
    if value_domain and value_domain.get("type") == "range":
        lo = value_domain.get("min")
        hi = value_domain.get("max")
        nz_lo = max(lo, -0.01) if lo is not None else -0.01
        nz_hi = min(hi, 0.01) if hi is not None else 0.01
        if nz_lo < nz_hi:
            return (nz_lo, nz_hi)
        return _range_normal_bounds(value_domain)
    return (-0.01, 0.01)


_DATA_RANGE_DISPATCH = {
    "zero": lambda vd, dt, rng: (0, 0),
    "all_ones": lambda vd, dt, rng: (1, 1),
    "with_inf": lambda vd, dt, rng: (1, float("inf")),
    "with_nan": lambda vd, dt, rng: (float("nan"), float("nan")),
}


def _data_range_to_ttk(data_range, dtype_str, value_domain, rng):
    if data_range == "normal":
        return _range_normal(value_domain, rng)
    if data_range == "extreme":
        mx = DTYPE_MAX.get(dtype_str, 3.4e38)
        return (mx * 0.9, mx)
    if data_range == "negative":
        return _range_negative(value_domain, rng)
    if data_range == "tiny_pos":
        return _range_tiny_pos(value_domain)
    if data_range == "near_zero":
        return _range_near_zero(value_domain)
    handler = _DATA_RANGE_DISPATCH.get(data_range)
    if handler is not None:
        return handler(value_domain, dtype_str, rng)
    return (-2, 2)


def input_range_entry(spec, name, rng):
    kind = spec.get("kind")
    if kind == "tensor_list":
        tensors = spec.get("tensors", [])
        data_range = spec.get("data_range", "normal")
        dtype = spec.get("dtype") or (tensors[0].get("dtype") if tensors else None)
        value_domain = spec.get("value_domain")
        item = _data_range_to_ttk(data_range, dtype, value_domain, rng)
        return tuple(item for _ in tensors)
    if kind == "tensor":
        return _data_range_to_ttk(spec.get("data_range", "normal"), spec.get("dtype"), spec.get("value_domain"), rng)
    raise KernelCsvError(f"{name}.kind must be tensor or tensor_list")


def tolerance_entry(spec, name):
    kind = spec.get("kind")
    if kind == "tensor_list":
        tensors = spec.get("tensors", [])
        dtype = spec.get("dtype") or (tensors[0].get("dtype") if tensors else None)
        tol = PRECISION_TOLERANCE.get(dtype, (0.001, 0.001))
        return tuple(tol for _ in tensors)
    if kind == "tensor":
        dtype = spec.get("dtype")
        return PRECISION_TOLERANCE.get(dtype, (0.001, 0.001))
    raise KernelCsvError(f"{name}.kind must be tensor or tensor_list")


def build_remark(case):
    meta = case.get("meta", {}) if isinstance(case.get("meta", {}), dict) else {}
    parts = [f"original_id={case.get('id', '')}", f"source={case.get('source', '')}"]
    for key in ("source_kind", "path", "tiling_key", "network_name", "variant_kind"):
        if key in meta:
            parts.append(f"{key}={meta[key]}")
    return "; ".join(parts)


def build_row(case, index, op_name, attribute_names, rng):
    case_id = case.get("id", f"case{index:05d}")
    inputs = get_tensor_entries(case.get("inputs"), case_id, "inputs")
    outputs = [
        (name, spec)
        for name, spec in get_tensor_entries(case.get("outputs"), case_id, "outputs")
        if is_present_output(spec)
    ]

    params = case.get("params", {})
    if not isinstance(params, dict):
        raise KernelCsvError(f"case {case_id}: params must be an object")
    attributes = {name: params[name] for name in attribute_names if name in params}

    input_shapes = tuple(tensor_shape_entry(spec, name) for name, spec in inputs)
    input_dtypes = tuple(tensor_dtype_entry(spec, name) for name, spec in inputs)
    input_formats = tuple(tensor_format_entry(spec, name) for name, spec in inputs)
    output_shapes = tuple(tensor_shape_entry(spec, name) for name, spec in outputs)
    output_dtypes = tuple(tensor_dtype_entry(spec, name) for name, spec in outputs)
    output_formats = tuple(tensor_format_entry(spec, name) for name, spec in outputs)
    input_data_ranges = tuple(input_range_entry(spec, name, rng) for name, spec in inputs)
    precision_tolerances = tuple(tolerance_entry(spec, name) for name, spec in outputs)

    meta = case.get("meta", {}) if isinstance(case.get("meta", {}), dict) else {}
    row = {
        "testcase_name": f"case{index:05d}",
        "network_name": str(meta.get("network_name", "")),
        "op_name": op_name,
        "input_shapes": tuple_literal(input_shapes),
        "input_dtypes": tuple_literal(input_dtypes),
        "input_formats": tuple_literal(input_formats),
        "output_shapes": tuple_literal(output_shapes),
        "output_dtypes": tuple_literal(output_dtypes),
        "output_formats": tuple_literal(output_formats),
        "input_ori_shapes": "",
        "input_ori_formats": "",
        "output_ori_shapes": "",
        "output_ori_formats": "",
        "attributes": py_literal(attributes),
        "input_data_ranges": tuple_literal(input_data_ranges),
        "precision_tolerances": tuple_literal(precision_tolerances),
        "absolute_precision": "1e-8",
        "output_inplace_indexes": "()",
        "output_shape_unknown_indexes": "()",
        "is_enabled": "True",
        "remark": build_remark(case),
        "soc_series": "",
        "priority": "0",
        "dump_file_prefix": "",
        "manual_input_binaries": "()",
        "manual_golden_binaries": "()",
    }
    return row


def write_csv(cases, output_path, op_name, attribute_names):
    rng = random.Random(42)
    rows = [build_row(case, idx, op_name, attribute_names, rng) for idx, case in enumerate(cases)]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KERNEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate TTK kernel CSV from Step 5 final case files.")
    parser.add_argument("--op-name", required=True, help="Normalized operator name, e.g. add_rms_norm")
    parser.add_argument("--operator-model", required=True, help="Path to S2P1_operator_model.json")
    parser.add_argument(
        "--op-def-cpp", default=None,
        help="Path to *_def.cpp. When provided, Attr and ValueDepend names are parsed from it.",
    )
    parser.add_argument("--low-cases", required=True, help="Path to S5_cases_low.json")
    parser.add_argument("--high-cases", required=True, help="Path to S5_cases_high.json")
    parser.add_argument("--low-csv", required=True, help="Output low CSV path")
    parser.add_argument("--high-csv", required=True, help="Output high CSV path")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    model_attr_names = load_attribute_names(args.operator_model)
    def_attr_names, value_depend_input_names = parse_def_cpp_symbols(args.op_def_cpp)
    attribute_names = set(def_attr_names or model_attr_names) | value_depend_input_names
    low_cases = load_cases(args.low_cases)
    high_cases = load_cases(args.high_cases)
    low_count = write_csv(low_cases, args.low_csv, args.op_name, attribute_names)
    high_count = write_csv(high_cases, args.high_csv, args.op_name, attribute_names)
    _logger.info(f"INFO: attribute_names={sorted(def_attr_names or model_attr_names)}")
    _logger.info(f"INFO: value_depend_input_names={sorted(value_depend_input_names)}")
    _logger.info(f"INFO: attribute_csv_keys={sorted(attribute_names)}")
    _logger.info(f"PASS: wrote {low_count} low rows to {Path(args.low_csv)}")
    _logger.info(f"PASS: wrote {high_count} high rows to {Path(args.high_csv)}")


if __name__ == "__main__":
    main()
