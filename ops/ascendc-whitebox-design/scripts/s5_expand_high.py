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
"""Expand Step 5.4 final low cases into high data_range cases."""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


NON_NORMAL_DATA_RANGES = (
    "zero",
    "extreme",
    "negative",
    "tiny_pos",
    "all_ones",
    "near_zero",
    "with_inf",
    "with_nan",
)


def validate_data_ranges(ranges: Any, label: str) -> list[str]:
    if not isinstance(ranges, (list, tuple)):
        raise ValueError(f"{label}: supported must be a list or tuple")
    result: list[str] = []
    seen: set[str] = set()
    for range_name in ranges:
        if not isinstance(range_name, str):
            raise ValueError(f"{label}: data_range must be a string, got {range_name!r}")
        if range_name == "normal":
            raise ValueError(f"{label}: supported must not include normal")
        if range_name not in NON_NORMAL_DATA_RANGES:
            raise ValueError(f"{label}: unsupported data_range {range_name!r}")
        if range_name in seen:
            raise ValueError(f"{label}: duplicate data_range {range_name!r}")
        seen.add(range_name)
        result.append(range_name)
    return result


def validate_data_range_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise ValueError("S5_data_range_policy.json: root must be an object")
    if policy.get("version") != 1:
        raise ValueError("S5_data_range_policy.json: version must be 1")
    if policy.get("mode") != "per_input_cyclic":
        raise ValueError("S5_data_range_policy.json: mode must be per_input_cyclic")
    inputs = policy.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("S5_data_range_policy.json: inputs must be a non-empty object")

    normalized_inputs: dict[str, dict[str, Any]] = {}
    for input_name, spec in inputs.items():
        if not isinstance(input_name, str) or not input_name:
            raise ValueError("S5_data_range_policy.json: input names must be non-empty strings")
        if not isinstance(spec, dict):
            raise ValueError(f"S5_data_range_policy.json: input {input_name} spec must be an object")
        if set(spec) != {"participates", "supported"}:
            raise ValueError(
                f"S5_data_range_policy.json: input {input_name} must only contain participates and supported"
            )
        participates = spec.get("participates")
        if not isinstance(participates, bool):
            raise ValueError(f"S5_data_range_policy.json: input {input_name}.participates must be bool")
        supported = validate_data_ranges(spec.get("supported"), f"S5_data_range_policy.json: input {input_name}")
        if not participates and supported:
            raise ValueError(f"S5_data_range_policy.json: input {input_name}.supported must be []")
        normalized_inputs[input_name] = {"participates": participates, "supported": supported}

    return {"version": 1, "mode": "per_input_cyclic", "inputs": normalized_inputs}


def load_json(root: Path, path: str) -> Any:
    with (root / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(root: Path, path: str, data: Any) -> None:
    with (root / path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_data_range_policy(root: Path) -> dict[str, Any]:
    return validate_data_range_policy(load_json(root, "S5_data_range_policy.json"))


def clone_case(case: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(case)


def mapped_inputs(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    input_dict = case.get("inputs")
    if not isinstance(input_dict, dict):
        raise ValueError(f"{case.get('id', '<unknown>')}: inputs must be an object")
    return input_dict


def _reset_tensor_range(t: dict[str, Any]) -> None:
    t["data_range"] = "normal"
    if t.get("param_type") == "DYNAMIC":
        for sub_t in t.get("tensors", []):
            if isinstance(sub_t, dict):
                sub_t["data_range"] = "normal"


def normalize_input_ranges(case: dict[str, Any]) -> dict[str, Any]:
    norm_case = clone_case(case)
    for item in mapped_inputs(norm_case).values():
        _reset_tensor_range(item)
    return norm_case


def set_input_range(case: dict[str, Any], input_name: str, range_name: str) -> None:
    tensor = mapped_inputs(case).get(input_name)
    if not isinstance(tensor, dict):
        raise ValueError(f"{case.get('id', '<unknown>')}: input {input_name!r} does not exist")
    tensor["data_range"] = range_name
    if tensor.get("param_type") == "DYNAMIC":
        for child in tensor.get("tensors", []):
            if isinstance(child, dict):
                child["data_range"] = range_name


def derive_case(case: dict[str, Any], case_id: str, source: str, **meta_updates: Any) -> dict[str, Any]:
    derived = normalize_input_ranges(case)
    derived["id"] = case_id
    derived["source"] = source
    meta = derived.setdefault("meta", {})
    if not isinstance(meta, dict):
        raise ValueError(f"{case.get('id', '<unknown>')}: meta must be an object")
    meta.update({"base_id": case["id"], **meta_updates})
    return derived


def policy_participants(case: dict[str, Any], policy: dict[str, Any]) -> list[tuple[str, list[str]]]:
    case_inputs = mapped_inputs(case)
    policy_inputs = policy["inputs"]
    missing = set(case_inputs) - set(policy_inputs)
    extra = set(policy_inputs) - set(case_inputs)
    if missing:
        raise ValueError(f"{case.get('id', '<unknown>')}: policy missing inputs {sorted(missing)}")
    if extra:
        raise ValueError(f"{case.get('id', '<unknown>')}: policy has unknown inputs {sorted(extra)}")

    participants: list[tuple[str, list[str]]] = []
    for input_name, spec in policy_inputs.items():
        if spec["participates"] and spec["supported"]:
            participants.append((input_name, spec["supported"]))
    return participants


def make_cyclic_range_variant(
    case: dict[str, Any],
    index: int,
    range_by_input: dict[str, str],
) -> dict[str, Any]:
    variant = derive_case(
        case,
        f"{case['id']}_range{index:02d}",
        "range",
        range_mode="per_input_cyclic",
        range_index=index,
        range_by_input=range_by_input,
        variant_kind="range",
    )
    for input_name, range_name in range_by_input.items():
        set_input_range(variant, input_name, range_name)
    return variant


def make_range_variants(case: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    participants = policy_participants(case, policy)
    if not participants:
        return []

    variant_count = max(len(ranges) for _, ranges in participants)
    variants: list[dict[str, Any]] = []
    for index in range(variant_count):
        range_by_input = {
            input_name: ranges[index % len(ranges)]
            for input_name, ranges in participants
        }
        variants.append(make_cyclic_range_variant(case, index, range_by_input))
    return variants


def _validate_dynamic_input_child(name: str, child_index: int, child: Any, label: str) -> None:
    if not isinstance(child, dict):
        raise ValueError(f"{label}: dynamic input {name} child {child_index} must be an object")
    for field in ("kind", "dtype", "format", "shape", "data_range"):
        if field not in child:
            raise ValueError(f"{label}: dynamic input {name} child {child_index} missing {field}")
    if not isinstance(child.get("format"), str) or not child.get("format"):
        raise ValueError(
            f"{label}: dynamic input {name} child {child_index}"
            ".format must be a non-empty string"
        )
    if child.get("kind") != "tensor":
        raise ValueError(f"{label}: dynamic input {name} child {child_index} kind must be tensor")


def _validate_dynamic_input(name: str, tensor: dict[str, Any], label: str) -> None:
    if tensor.get("kind") != "tensor_list" or tensor.get("param_type") != "DYNAMIC":
        raise ValueError(f"{label}: dynamic input {name} must have kind tensor_list and param_type DYNAMIC")
    children = tensor.get("tensors")
    if not isinstance(children, list):
        raise ValueError(f"{label}: dynamic input {name} tensors must be a list")
    if tensor.get("tensor_count") != len(children):
        raise ValueError(f"{label}: dynamic input {name} tensor_count must equal len(tensors)")
    for child_index, child in enumerate(children):
        _validate_dynamic_input_child(name, child_index, child, label)


def _validate_case_input(name: str, tensor: dict[str, Any], label: str) -> None:
    if not isinstance(tensor, dict):
        raise ValueError(f"{label}: input {name} must be an object")
    for field in ("kind", "dtype", "format", "param_type", "data_range"):
        if field not in tensor:
            raise ValueError(f"{label}: input {name} missing {field}")
    if not isinstance(tensor.get("format"), str) or not tensor.get("format"):
        raise ValueError(f"{label}: input {name}.format must be a non-empty string")
    if tensor.get("kind") == "tensor_list" or tensor.get("param_type") == "DYNAMIC":
        _validate_dynamic_input(name, tensor, label)
    elif tensor.get("kind") != "tensor":
        raise ValueError(f"{label}: input {name} kind must be tensor")


def _validate_case_outputs(outputs: dict[str, Any], label: str) -> None:
    for name, tensor in outputs.items():
        if not isinstance(tensor, dict):
            raise ValueError(f"{label}: output {name} must be an object")
        for field in ("kind", "dtype", "format", "shape", "param_type", "present"):
            if field not in tensor:
                raise ValueError(f"{label}: output {name} missing {field}")
        if not isinstance(tensor.get("format"), str) or not tensor.get("format"):
            raise ValueError(f"{label}: output {name}.format must be a non-empty string")
        if tensor.get("kind") != "tensor":
            raise ValueError(f"{label}: output {name} kind must be tensor")


def validate_case_schema(case: dict[str, Any], label: str) -> None:
    for field in ("id", "source", "params", "inputs", "outputs", "meta"):
        if field not in case:
            raise ValueError(f"{label}: missing {field}")
    if not isinstance(case["id"], str):
        raise ValueError(f"{label}: id must be a string")
    if not isinstance(case["params"], dict) or not isinstance(case["meta"], dict):
        raise ValueError(f"{label}: params and meta must be objects")
    outputs = case.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"{label}: outputs must be an object")
    for name, tensor in mapped_inputs(case).items():
        _validate_case_input(name, tensor, label)
    _validate_case_outputs(outputs, label)


def assert_unique_ids(cases: list[dict[str, Any]], label: str) -> None:
    seen: set[str] = set()
    for case in cases:
        validate_case_schema(case, label)
        case_id = case["id"]
        if case_id in seen:
            raise ValueError(f"duplicate id in {label}: {case_id}")
        seen.add(case_id)


def load_low_cases(root: Path) -> list[dict[str, Any]]:
    return [normalize_input_ranges(case) for case in load_json(root, "S5_cases_low.json")]


def build_high_cases(root: Path, low_cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    policy = load_data_range_policy(root)
    high_cases = [clone_case(case) for case in low_cases]
    range_count = 0
    for case in low_cases:
        variants = make_range_variants(case, policy)
        high_cases.extend(variants)
        range_count += len(variants)
    return high_cases, range_count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Expand Step 5.4 final low cases into high data_range cases.")
    parser.add_argument(
        "--whitebox-dir", required=True,
        help="Directory containing S5_cases_low.json and S5_data_range_policy.json.",
    )
    args = parser.parse_args()

    whitebox_dir = Path(args.whitebox_dir).resolve()
    low_cases = load_low_cases(whitebox_dir)
    high_cases, range_count = build_high_cases(whitebox_dir, low_cases)
    assert_unique_ids(low_cases, "low")
    assert_unique_ids(high_cases, "high")
    dump_json(whitebox_dir, "S5_cases_high.json", high_cases)
    _logger.info("PASS: expanded high data_range cases")
    _logger.info(f"low_cases={len(low_cases)} range_cases={range_count} high_cases={len(high_cases)}")


if __name__ == "__main__":
    main()
