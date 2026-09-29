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

"""Validate executable tensor recipes against the public DSL spec."""

CPU_DTYPES = {
    "float16",
    "float32",
    "float64",
    "bfloat16",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bool",
}
FILLS = {
    "normal",
    "zeros",
    "ones",
    "constant",
    "nan_one",
    "pos_inf_one",
    "neg_inf_one",
    "pos_inf",
    "neg_inf",
}


def fail(message):
    raise ValueError(message)


def validate_tensor(case_id, name, value, meta, error_code):
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("shape"), list)
        or not isinstance(value.get("dtype"), str)
    ):
        fail(f"{case_id}.{name} must have shape and dtype")
    shape, dtype = value["shape"], value["dtype"]
    if not all(type(dim) is int and dim >= 0 for dim in shape):
        fail(f"{case_id}.{name}.shape must be nonnegative integers")
    if dtype not in CPU_DTYPES:
        fail(
            f"{case_id}.{name} dtype {dtype} cannot be constructed by the fixed runner"
        )
    if (
        dtype not in (meta.get("dtype_set") or [])
        and error_code != "dtype_not_supported"
    ):
        fail(f"{case_id}.{name} dtype {dtype} is outside spec")
    bounds = meta.get("rank_range")
    has_bounds = isinstance(bounds, list) and len(bounds) == 2
    if (
        has_bounds
        and not bounds[0] <= len(shape) <= bounds[1]
        and error_code != "shape_mismatch"
    ):
        fail(f"{case_id}.{name} rank is outside spec")
    validate_fill(case_id, name, value)


def validate_tensor_list(case_id, name, value, meta, error_code):
    tensors = value.get("tensors") if isinstance(value, dict) else None
    if not isinstance(tensors, list):
        fail(f"{case_id}.{name} tensor_list needs tensors: [...] recipe")
    length = meta.get("list_length") or {}
    kind = length.get("kind")
    if kind == "fixed" and len(tensors) != length.get("value"):
        fail(f"{case_id}.{name} tensor_list length differs from spec")
    if kind == "range" and not length.get("min", 0) <= len(tensors) <= length.get(
        "max", float("inf")
    ):
        fail(f"{case_id}.{name} tensor_list length outside spec")
    if kind not in ("fixed", "range", "unconstrained"):
        fail(f"{case_id}.{name} tensor_list length rule {kind} needs explicit resolver")
    for index, member in enumerate(tensors):
        validate_tensor(case_id, f"{name}[{index}]", member, meta, error_code)


def validate_fill(case_id, name, value):
    shape, dtype = value["shape"], value["dtype"]
    fill = value.get("fill", "normal")
    if fill not in FILLS or (
        fill == "constant" and not isinstance(value.get("value"), (int, float))
    ):
        fail(f"{case_id}.{name} has an unsupported fill recipe")
    if fill in (
        "nan_one",
        "pos_inf_one",
        "neg_inf_one",
        "pos_inf",
        "neg_inf",
    ) and not dtype.startswith(("float", "bfloat")):
        fail(f"{case_id}.{name} nonfinite fill requires floating dtype")
    if fill in ("nan_one", "pos_inf_one", "neg_inf_one") and 0 in shape:
        fail(f"{case_id}.{name} single-position fill needs a nonempty tensor")
