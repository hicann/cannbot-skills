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
"""Append Step 5.4 empty low cases and write final low cases."""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


INPUT_FILE = "S5_mapped_cases_low_shape.json"
OUTPUT_FILE = "S5_cases_low.json"


def load_json(root: Path, path: str) -> Any:
    with (root / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(root: Path, path: str, data: Any) -> None:
    with (root / path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def clone_case(case: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(case)


def mapped_inputs(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    inputs = case.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"{case.get('id', '<unknown>')}: inputs must be an object")
    return inputs


def _validate_input_tensor(name: str, tensor: dict[str, Any], label: str) -> None:
    if not isinstance(tensor.get("format"), str) or not tensor.get("format"):
        raise ValueError(f"{label}: input {name}.format must be a non-empty string")
    kind = tensor.get("kind")
    if kind == "tensor":
        shape = tensor.get("shape")
        if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
            raise ValueError(f"{label}: input {name}.shape must be list[int]")
    elif kind == "tensor_list":
        tensors = tensor.get("tensors")
        if not isinstance(tensor.get("tensor_count"), int) or not isinstance(tensors, list):
            raise ValueError(f"{label}: input {name} tensor_list must contain tensor_count and tensors")
        if tensor["tensor_count"] != len(tensors):
            raise ValueError(f"{label}: input {name}.tensor_count must equal len(tensors)")
        for child_index, child in enumerate(tensors):
            if not isinstance(child, dict):
                raise ValueError(f"{label}: input {name} child {child_index} must be an object")
            if not isinstance(child.get("format"), str) or not child.get("format"):
                raise ValueError(
                    f"{label}: input {name} child {child_index}.format must be a non-empty string"
                )
    else:
        raise ValueError(f"{label}: input {name}.kind must be tensor or tensor_list")


def _validate_output_tensor(name: str, tensor: dict[str, Any], label: str) -> None:
    if not isinstance(tensor, dict):
        raise ValueError(f"{label}: output {name} must be an object")
    if not isinstance(tensor.get("format"), str) or not tensor.get("format"):
        raise ValueError(f"{label}: output {name}.format must be a non-empty string")


def validate_case_schema(case: Any, label: str) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise ValueError(f"{label}: case must be an object")
    for field in ("id", "source", "params", "inputs", "outputs", "meta"):
        if field not in case:
            raise ValueError(f"{label}: missing {field}")
    if not isinstance(case["id"], str) or not isinstance(case["source"], str):
        raise ValueError(f"{label}: id and source must be strings")
    if not isinstance(case["params"], dict) or not isinstance(case["outputs"], dict) \
            or not isinstance(case["meta"], dict):
        raise ValueError(f"{label}: params, outputs, and meta must be objects")
    for name, tensor in mapped_inputs(case).items():
        if not isinstance(tensor, dict):
            raise ValueError(f"{label}: input {name} must be an object")
        _validate_input_tensor(name, tensor, label)
    for name, tensor in case["outputs"].items():
        _validate_output_tensor(name, tensor, label)
    return case


def assert_unique_ids(cases: list[dict[str, Any]], label: str) -> None:
    seen: set[str] = set()
    for case in cases:
        case_id = case["id"]
        if case_id in seen:
            raise ValueError(f"duplicate id in {label}: {case_id}")
        seen.add(case_id)


def select_empty_seeds(shape_cases: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    seeds: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for case in shape_cases:
        for input_name, tensor in mapped_inputs(case).items():
            if input_name in seeds:
                continue
            if tensor.get("kind") == "tensor_list":
                seeds[input_name] = case
                order.append(input_name)
                continue
            shape = tensor.get("shape")
            if tensor.get("kind") == "tensor" and isinstance(shape, list) and shape:
                seeds[input_name] = case
                order.append(input_name)
    return [(input_name, seeds[input_name]) for input_name in order]


def make_empty_case(seed: dict[str, Any], input_name: str, index: int) -> dict[str, Any]:
    case = clone_case(seed)
    case["id"] = f"low_case_empty_{index:02d}"
    case["source"] = "empty"
    meta = case.setdefault("meta", {})
    if not isinstance(meta, dict):
        raise ValueError(f"{seed['id']}: meta must be an object")
    meta.update({"base_id": seed["id"], "variant_kind": "empty", "empty_input": input_name})

    target = mapped_inputs(case)[input_name]
    if target.get("kind") == "tensor_list":
        target["tensor_count"] = 0
        target["tensors"] = []
        meta["empty_mode"] = "tensor_list_empty"
    else:
        shape = target.get("shape")
        if not isinstance(shape, list) or not shape:
            raise ValueError(f"{seed['id']}: input {input_name} has no non-empty shape")
        target["shape"] = list(shape)
        target["shape"][0] = 0
        meta.update({"empty_mode": "shape_dim_0", "empty_axis": 0})
    return validate_case_schema(case, case["id"])


def build_final_low_cases(shape_cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if not shape_cases:
        raise ValueError(f"{INPUT_FILE}: must contain at least one shape case")
    normalized = [validate_case_schema(case, str(case.get("id", "<unknown>"))) for case in shape_cases]
    non_shape = [case.get("id", "<unknown>") for case in normalized if case.get("source") != "shape"]
    if non_shape:
        raise ValueError(f"{INPUT_FILE}: expected only source='shape', got non-shape cases {non_shape}")

    seeds = select_empty_seeds(normalized)
    empty_cases = [make_empty_case(seed, input_name, index) for index, (input_name, seed) in enumerate(seeds)]
    final_cases = normalized + empty_cases
    assert_unique_ids(final_cases, OUTPUT_FILE)
    return final_cases, len(empty_cases)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Append empty low cases and write final Step 5 low cases.")
    parser.add_argument("--whitebox-dir", required=True, help=f"Directory containing {INPUT_FILE}.")
    parser.add_argument("--force", action="store_true", help=f"Overwrite existing {OUTPUT_FILE}.")
    args = parser.parse_args()

    whitebox_dir = Path(args.whitebox_dir).resolve()
    output_path = whitebox_dir / OUTPUT_FILE
    if output_path.exists() and not args.force:
        raise SystemExit(f"{OUTPUT_FILE} already exists; use --force to regenerate it")

    raw_cases = load_json(whitebox_dir, INPUT_FILE)
    if not isinstance(raw_cases, list):
        raise SystemExit(f"{INPUT_FILE}: root must be a JSON list")
    final_cases, empty_count = build_final_low_cases(raw_cases)
    dump_json(whitebox_dir, OUTPUT_FILE, final_cases)
    _logger.info("PASS: appended empty cases")
    _logger.info(
        f"input={INPUT_FILE} output={OUTPUT_FILE} shape_cases={len(raw_cases)}"
        f" empty_cases_added={empty_count} low_cases={len(final_cases)}"
    )


if __name__ == "__main__":
    main()
