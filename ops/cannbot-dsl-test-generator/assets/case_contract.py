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

"""Shared contracts for generated inputs and observable operator behavior."""

import math

RELATIONS = {"eq", "ne", "ge", "gt", "le", "lt"}
RUNTIME_KINDS = {
    "output_shape",
    "output_dtype",
    "output_device",
    "output_equals_input",
    "input_unchanged",
    "input_conditions",
}


def compare(left, relation, right):
    return {
        "eq": lambda: left == right,
        "ne": lambda: left != right,
        "ge": lambda: left >= right,
        "gt": lambda: left > right,
        "le": lambda: left <= right,
        "lt": lambda: left < right,
    }[relation]()


def validate_checks(checks):
    if not isinstance(checks, list) or not checks:
        raise ValueError("checks must be a nonempty list")
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("check must be a mapping")
        if not isinstance(check.get("input"), str):
            raise ValueError("input condition requires input name")
        if check.get("metric") not in {
            "dimension",
            "numel",
            "unique_count",
            "count",
        }:
            raise ValueError("unknown input metric")
        if check.get("metric") == "dimension" or "axis" in check:
            if type(check.get("axis")) is not int:
                raise ValueError("axis must be an integer")
        where = check.get("where", {})
        if not isinstance(where, dict) or set(where) - {"min", "max", "eq"}:
            raise ValueError("input filter supports min/max/eq")
        for bound in where.values():
            if isinstance(bound, dict):
                if (
                    set(bound) != {"input", "dimension"}
                    or not isinstance(bound["input"], str)
                    or type(bound["dimension"]) is not int
                ):
                    raise ValueError("filter bound must reference an input dimension")
            elif not isinstance(bound, (int, float)) or not math.isfinite(bound):
                raise ValueError("filter bound must be finite")
        if (
            check.get("relation") not in RELATIONS
            or not isinstance(check.get("value"), (int, float))
            or not math.isfinite(check["value"])
        ):
            raise ValueError(
                "predicate requires a finite value and comparison relation"
            )


def validate_assertions(assertions):
    if not isinstance(assertions, list):
        raise ValueError("assertions must be a list")
    executions = 0
    for assertion in assertions:
        if not isinstance(assertion, dict):
            raise ValueError("assertion must be a mapping")
        kind = assertion.get("kind")
        if kind in {"probe_events", "probe_predicates"}:
            raise ValueError(
                f"{kind} is removed; migrate structural checks to review_evidence and public behavior assertions"
            )
        if kind == "execution":
            executions += 1
            repeat = assertion.get("repeat", 1)
            if type(repeat) is not int or not 1 <= repeat <= 100:
                raise ValueError("execution repeat must be in [1,100]")
            if type(assertion.get("bitwise", False)) is not bool:
                raise ValueError("execution bitwise must be boolean")
            if assertion.get("bitwise") and repeat < 2:
                raise ValueError("bitwise repeat requires at least two repetitions")
            calls = assertion.get("calls", [])
            if not isinstance(calls, list) or len(calls) > 32:
                raise ValueError("execution calls must be a list of at most 32 entries")
            for call in calls:
                if not isinstance(call, dict) or not isinstance(
                    call.get("inputs"), dict
                ):
                    raise ValueError("sequence call requires public input recipes")
                if "assertions" in call:
                    validate_assertions(call["assertions"])
                    if any(item["kind"] == "execution" for item in call["assertions"]):
                        raise ValueError("nested execution is unsupported")
                if "seed" in call and type(call["seed"]) is not int:
                    raise ValueError("call seed must be an integer")
        elif kind not in RUNTIME_KINDS:
            raise ValueError("unknown assertion kind")
        elif kind == "input_conditions":
            validate_checks(assertion.get("checks"))
        elif kind == "input_unchanged":
            names = assertion.get("inputs")
            if (
                not isinstance(names, list)
                or not names
                or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(names) != len(set(names))
            ):
                raise ValueError(
                    "input_unchanged requires unique nonempty Tensor input names"
                )
        else:
            index = assertion.get("index")
            if index is not None and (type(index) is not int or index < 0):
                raise ValueError("output index must be a nonnegative integer")
            output = assertion.get("output")
            if output is not None and (
                not isinstance(output, str) or not output.strip()
            ):
                raise ValueError("output name must be a nonempty string")
    if executions > 1:
        raise ValueError("one execution control per case is allowed")


def check_inputs(inputs, checks):
    import torch

    validate_checks(checks)
    records = []
    for check in checks:
        tensor = inputs.get(check["input"])
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("input condition requires a tensor")
        metric = check["metric"]
        axis = check.get("axis")
        if axis is not None and not -tensor.ndim <= axis < tensor.ndim:
            raise ValueError("condition axis outside tensor rank")
        if metric == "dimension":
            observed = [tensor.shape[axis]]
        elif metric == "numel":
            observed = [tensor.numel()]
        else:
            if axis is None:
                rows = [tensor.reshape(-1)]
            else:
                moved = tensor.movedim(axis, -1)
                row_count = math.prod(moved.shape[:-1])
                rows = moved.reshape(row_count, moved.shape[-1])
            observed = []
            for row in rows:
                valid = torch.ones_like(row, dtype=torch.bool)
                for key, bound in check.get("where", {}).items():
                    if isinstance(bound, dict):
                        source = inputs.get(bound["input"])
                        if not isinstance(source, torch.Tensor):
                            raise ValueError("filter dimension source is not a tensor")
                        bound = source.shape[bound["dimension"]]
                    valid &= {
                        "min": lambda: row >= bound,
                        "max": lambda: row < bound,
                        "eq": lambda: row == bound,
                    }[key]()
                selected = row[valid]
                observed.append(
                    int(
                        selected.numel()
                        if metric == "count"
                        else torch.unique(selected).numel()
                    )
                )
        if not observed or not all(
            compare(value, check["relation"], check["value"]) for value in observed
        ):
            raise AssertionError(
                f"input condition not triggered: {check!r}; observed={observed}"
            )
        records.append({"check": check, "observed": observed})
    return records


def case_calls(inputs, seed, assertions):
    validate_assertions(assertions)
    control = next((a for a in assertions if a["kind"] == "execution"), {})
    runtime = [a for a in assertions if a["kind"] != "execution"]
    calls = [{"inputs": inputs, "seed": seed, "assertions": runtime}]
    for call in control.get("calls", []):
        calls.append(
            {
                "inputs": call["inputs"],
                "seed": call.get("seed", seed),
                "assertions": call.get("assertions", runtime),
            }
        )
    for call in calls:
        for assertion in call["assertions"]:
            if assertion["kind"] == "input_unchanged":
                missing = set(assertion["inputs"]) - set(call["inputs"])
                if missing:
                    raise ValueError(
                        f"input_unchanged names missing from inputs: {sorted(missing)}"
                    )
    return control.get("repeat", 1), control.get("bitwise", False), calls
