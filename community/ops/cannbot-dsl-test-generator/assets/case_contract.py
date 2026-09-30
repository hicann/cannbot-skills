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

"""Shared, bounded predicates for generated CPU inputs and observed probe events."""

import math
from collections import Counter, defaultdict


RELATIONS = {"eq", "ne", "ge", "gt", "le", "lt"}
RUNTIME_KINDS = {
    "output_shape",
    "output_dtype",
    "output_device",
    "output_equals_input",
    "probe_events",
    "probe_predicates",
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


def validate_checks(checks, probe=False):
    if not isinstance(checks, list) or not checks:
        raise ValueError("checks must be a nonempty list")
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("check must be a mapping")
        if probe:
            op = check.get("op")
            if op not in {"count", "paired", "before"}:
                raise ValueError("unknown probe predicate")
            selectors = ["where"] if op == "count" else ["first", "second"]
            if any(
                not isinstance(check.get(key), dict) or not check[key]
                for key in selectors
            ):
                raise ValueError("probe selectors must be nonempty mappings")
            if op != "count":
                keys = check.get("keys")
                if (
                    not isinstance(keys, list)
                    or not keys
                    or any(not isinstance(k, str) for k in keys)
                    or len(keys) != len(set(keys))
                ):
                    raise ValueError("pairing requires unique key names")
                continue
        else:
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
                        raise ValueError(
                            "filter bound must reference an input dimension"
                        )
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
        elif kind in {"input_conditions", "probe_predicates"}:
            validate_checks(assertion.get("checks"), probe=kind == "probe_predicates")
        elif kind == "probe_events":
            if not isinstance(assertion.get("events"), list) or not assertion["events"]:
                raise ValueError("probe_events requires nonempty events")
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


def check_probe(events, checks):
    validate_checks(checks, probe=True)
    if not isinstance(events, list) or any(
        not isinstance(event, dict) for event in events
    ):
        raise ValueError("probe predicates require mapping events")
    records = []
    for check in checks:

        def selected(selector):
            return [
                (index, event)
                for index, event in enumerate(events)
                if all(k in event and event[k] == v for k, v in selector.items())
            ]

        if check["op"] == "count":
            count = len(selected(check["where"]))
            if not compare(count, check["relation"], check["value"]):
                raise AssertionError(f"probe count failed: {check!r}; observed={count}")
            records.append({"check": check, "observed": count})
            continue
        first, second = selected(check["first"]), selected(check["second"])

        def grouped(items):
            result = defaultdict(list)
            for index, event in items:
                if any(key not in event for key in check["keys"]):
                    raise AssertionError("probe pairing key missing")
                key = tuple(event[name] for name in check["keys"])
                try:
                    result[key].append(index)
                except TypeError as exc:
                    raise ValueError("probe pairing keys must be scalar") from exc
            return result

        left, right = grouped(first), grouped(second)
        if not left or (
            Counter({k: len(v) for k, v in left.items()})
            != Counter({k: len(v) for k, v in right.items()})
        ):
            raise AssertionError(f"probe pairing failed: {check!r}")
        if check["op"] == "before" and any(
            a >= b for key in left for a, b in zip(left[key], right[key])
        ):
            raise AssertionError(f"probe order failed: {check!r}")
        records.append({"check": check, "pairs": len(first)})
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
    return control.get("repeat", 1), control.get("bitwise", False), calls
