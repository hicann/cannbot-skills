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
"""Validate Step 5.2 mapped case JSON schema."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


OUTPUT_JSON_FILES = (
    "S5_mapped_cases_path.json",
    "S5_mapped_cases_network.json",
    "S5_mapped_cases_low_shape.json",
)

TOP_LEVEL_FIELDS = ("id", "source", "params", "inputs", "outputs", "meta")
INPUT_TENSOR_FIELDS = ("kind", "dtype", "format", "shape", "param_type", "data_range")
INPUT_TENSOR_LIST_FIELDS = ("kind", "dtype", "format", "tensor_count", "tensors", "param_type", "data_range")
TENSOR_LIST_CHILD_FIELDS = ("kind", "dtype", "format", "shape", "data_range")
OUTPUT_FIELDS = ("kind", "dtype", "format", "shape", "param_type", "present")


@dataclass
class ErrorCtx:
    errors: list[str]
    file_name: str
    case_id: str


def add_error(errors: list[str], file_name: str, case_id: str, field_path: str, message: str) -> None:
    errors.append(f"{file_name}: {case_id}: {field_path}: {message}")


def load_json(path: Path, errors: list[str]) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001 - report parse/open failure as validation error
        add_error(errors, path.name, "<file>", "$", f"failed to parse JSON: {exc}")
        return None


def require_fields(
    obj: dict[str, Any],
    fields: tuple[str, ...],
    ctx: ErrorCtx,
    field_path: str,
) -> None:
    for field in fields:
        if field not in obj:
            add_error(ctx.errors, ctx.file_name, ctx.case_id, f"{field_path}.{field}", "missing field")


def check_shape(value: Any, errors: list[str], file_name: str, case_id: str, field_path: str) -> None:
    if not isinstance(value, list) or not all(isinstance(dim, int) for dim in value):
        add_error(errors, file_name, case_id, field_path, "shape must be list[int]")


def check_string(value: Any, errors: list[str], file_name: str, case_id: str, field_path: str) -> None:
    if not isinstance(value, str):
        add_error(errors, file_name, case_id, field_path, "must be a string")


def check_non_empty_string(value: Any, errors: list[str], file_name: str, case_id: str, field_path: str) -> None:
    if not isinstance(value, str) or not value:
        add_error(errors, file_name, case_id, field_path, "must be a non-empty string")


def check_tensor_input(
    tensor: dict[str, Any],
    errors: list[str],
    file_name: str,
    case_id: str,
    field_path: str,
) -> None:
    require_fields(tensor, INPUT_TENSOR_FIELDS, ErrorCtx(errors, file_name, case_id), field_path)
    if tensor.get("kind") != "tensor":
        add_error(errors, file_name, case_id, f"{field_path}.kind", "expected 'tensor'")
    for field in ("dtype", "param_type", "data_range"):
        if field in tensor:
            check_string(tensor[field], errors, file_name, case_id, f"{field_path}.{field}")
    if "format" in tensor:
        check_non_empty_string(tensor["format"], errors, file_name, case_id, f"{field_path}.format")
    if "shape" in tensor:
        check_shape(tensor["shape"], errors, file_name, case_id, f"{field_path}.shape")


def check_tensor_list_child(
    child: Any,
    errors: list[str],
    file_name: str,
    case_id: str,
    field_path: str,
) -> None:
    if not isinstance(child, dict):
        add_error(errors, file_name, case_id, field_path, "child tensor must be an object")
        return
    require_fields(child, TENSOR_LIST_CHILD_FIELDS, ErrorCtx(errors, file_name, case_id), field_path)
    if child.get("kind") != "tensor":
        add_error(errors, file_name, case_id, f"{field_path}.kind", "expected 'tensor'")
    for field in ("dtype", "data_range"):
        if field in child:
            check_string(child[field], errors, file_name, case_id, f"{field_path}.{field}")
    if "format" in child:
        check_non_empty_string(child["format"], errors, file_name, case_id, f"{field_path}.format")
    if "shape" in child:
        check_shape(child["shape"], errors, file_name, case_id, f"{field_path}.shape")


def check_tensor_list_input(
    tensor: dict[str, Any],
    errors: list[str],
    file_name: str,
    case_id: str,
    field_path: str,
) -> None:
    require_fields(tensor, INPUT_TENSOR_LIST_FIELDS, ErrorCtx(errors, file_name, case_id), field_path)
    if tensor.get("kind") != "tensor_list":
        add_error(errors, file_name, case_id, f"{field_path}.kind", "expected 'tensor_list'")
    for field in ("dtype", "param_type", "data_range"):
        if field in tensor:
            check_string(tensor[field], errors, file_name, case_id, f"{field_path}.{field}")
    if "format" in tensor:
        check_non_empty_string(tensor["format"], errors, file_name, case_id, f"{field_path}.format")
    tensor_count = tensor.get("tensor_count")
    tensors = tensor.get("tensors")
    if not isinstance(tensor_count, int):
        add_error(errors, file_name, case_id, f"{field_path}.tensor_count", "must be an int")
    if not isinstance(tensors, list):
        add_error(errors, file_name, case_id, f"{field_path}.tensors", "must be a list")
        return
    if isinstance(tensor_count, int) and tensor_count != len(tensors):
        add_error(errors, file_name, case_id, f"{field_path}.tensor_count", "must equal len(tensors)")
    for index, child in enumerate(tensors):
        check_tensor_list_child(child, errors, file_name, case_id, f"{field_path}.tensors[{index}]")


def check_input_descriptor(
    tensor: Any,
    errors: list[str],
    file_name: str,
    case_id: str,
    field_path: str,
) -> None:
    if not isinstance(tensor, dict):
        add_error(errors, file_name, case_id, field_path, "input descriptor must be an object")
        return
    kind = tensor.get("kind")
    if kind == "tensor":
        check_tensor_input(tensor, errors, file_name, case_id, field_path)
    elif kind == "tensor_list":
        check_tensor_list_input(tensor, errors, file_name, case_id, field_path)
    else:
        add_error(errors, file_name, case_id, f"{field_path}.kind", "expected 'tensor' or 'tensor_list'")


def check_output_descriptor(
    tensor: Any,
    errors: list[str],
    file_name: str,
    case_id: str,
    field_path: str,
) -> None:
    if not isinstance(tensor, dict):
        add_error(errors, file_name, case_id, field_path, "output descriptor must be an object")
        return
    require_fields(tensor, OUTPUT_FIELDS, ErrorCtx(errors, file_name, case_id), field_path)
    if tensor.get("kind") != "tensor":
        add_error(errors, file_name, case_id, f"{field_path}.kind", "expected 'tensor'")
    for field in ("dtype", "param_type"):
        if field in tensor:
            check_string(tensor[field], errors, file_name, case_id, f"{field_path}.{field}")
    if "format" in tensor:
        check_non_empty_string(tensor["format"], errors, file_name, case_id, f"{field_path}.format")
    if "shape" in tensor:
        check_shape(tensor["shape"], errors, file_name, case_id, f"{field_path}.shape")
    if "present" in tensor and not isinstance(tensor["present"], bool):
        add_error(errors, file_name, case_id, f"{field_path}.present", "must be a bool")


def attribute_names_from_operator_model(whitebox_dir: Path, errors: list[str]) -> set[str]:
    model_path = whitebox_dir / "S2P1_operator_model.json"
    model = load_json(model_path, errors)
    if not isinstance(model, dict):
        return set()
    attrs = model.get("attributes", [])
    if not isinstance(attrs, list):
        return set()

    names: set[str] = set()
    for index, attr in enumerate(attrs):
        if not isinstance(attr, dict):
            add_error(errors, model_path.name, "<file>", f"attributes[{index}]", "must be an object")
            continue
        name = attr.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def check_mapped_case_schema(
    case: Any, index: int, file_name: str, errors: list[str], attribute_names: set[str],
) -> None:
    if not isinstance(case, dict):
        add_error(errors, file_name, f"[{index}]", "$", "case must be an object")
        return

    case_id = str(case.get("id", f"[{index}]"))
    require_fields(case, TOP_LEVEL_FIELDS, ErrorCtx(errors, file_name, case_id), "$")

    if "id" in case:
        check_string(case["id"], errors, file_name, case_id, "id")
    if "source" in case:
        check_string(case["source"], errors, file_name, case_id, "source")
    for field in ("params", "inputs", "outputs", "meta"):
        if field in case and not isinstance(case[field], dict):
            add_error(errors, file_name, case_id, field, "must be an object")

    params = case.get("params")
    meta = case.get("meta")
    if isinstance(params, dict) and isinstance(meta, dict):
        duplicated_non_attrs = sorted((set(params) & set(meta)) - attribute_names)
        if duplicated_non_attrs:
            add_error(
                errors,
                file_name,
                case_id,
                "params",
                "fields duplicated in params and meta must be operator attributes;"
                " move audit/router fields to meta only: "
                + ", ".join(duplicated_non_attrs),
            )

    inputs = case.get("inputs")
    if isinstance(inputs, dict):
        for input_name, tensor in inputs.items():
            check_input_descriptor(tensor, errors, file_name, case_id, f"inputs.{input_name}")

    outputs = case.get("outputs")
    if isinstance(outputs, dict):
        for output_name, tensor in outputs.items():
            check_output_descriptor(tensor, errors, file_name, case_id, f"outputs.{output_name}")


def check_json_file(path: Path, errors: list[str], attribute_names: set[str]) -> int:
    if not path.is_file():
        add_error(errors, path.name, "<file>", "$", "required output json is missing")
        return 0

    data = load_json(path, errors)
    if data is None:
        return 0
    if not isinstance(data, list):
        add_error(errors, path.name, "<file>", "$", "must be a JSON list")
        return 0

    for index, case in enumerate(data):
        check_mapped_case_schema(case, index, path.name, errors, attribute_names)
    return len(data)


def check_outputs(whitebox_dir: Path) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    attribute_names = attribute_names_from_operator_model(whitebox_dir, errors)
    stats = {name: check_json_file(whitebox_dir / name, errors, attribute_names) for name in OUTPUT_JSON_FILES}
    return errors, stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Validate Step 5.2 mapped case JSON schema.")
    parser.add_argument("--whitebox-dir", required=True, help="Directory containing Step 5.2 mapped case JSON files.")
    args = parser.parse_args()

    whitebox_dir = Path(args.whitebox_dir).resolve()
    errors, stats = check_outputs(whitebox_dir)
    if errors:
        _logger.info("FAIL: mapper schema validation errors")
        for error in errors:
            _logger.info(error)
        raise SystemExit(1)

    _logger.info("PASS: mapper schema accepted")
    for name in OUTPUT_JSON_FILES:
        _logger.info(f"{name} cases={stats[name]}")


if __name__ == "__main__":
    main()
