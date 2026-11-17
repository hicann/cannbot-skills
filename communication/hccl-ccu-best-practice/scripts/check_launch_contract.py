#!/usr/bin/env python3
# Copyright (c) 2026 CANNBot contributors
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check a declared CCU launch ABI; does not parse C++ or prove runtime safety."""

import argparse
import json
import sys
from pathlib import Path


class ContractError(ValueError):
    """Invalid or internally inconsistent launch contract."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def object_keys(value, required, optional, where):
    require(isinstance(value, dict), f"{where}: expected object")
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    require(not missing, f"{where}: missing keys {sorted(missing)}")
    require(not extra, f"{where}: unknown keys {sorted(extra)}")


def nonempty(value, where):
    require(
        isinstance(value, str) and bool(value.strip()),
        f"{where}: expected nonempty string",
    )


def names(value, where):
    require(isinstance(value, list), f"{where}: expected array")
    for item in value:
        nonempty(item, where)
    require(len(set(value)) == len(value), f"{where}: duplicate names")
    return value


def count(value, where):
    require(
        type(value) is int and 0 <= value <= 0xFFFFFFFF,
        f"{where}: expected unsigned 32-bit integer slot count",
    )
    return value


def reasons(value, where):
    require(isinstance(value, dict), f"{where}: expected name-to-reason object")
    for name, reason in value.items():
        nonempty(name, where)
        nonempty(reason, f"{where}.{name}")
    return set(value)


def mappings(value, slots, where, unique=False):
    require(isinstance(value, list), f"{where}: expected array")
    seen = set()
    for i, item in enumerate(value):
        loc = f"{where}[{i}]"
        object_keys(item, {"index", "name"}, set(), loc)
        index = count(item["index"], f"{loc}.index")
        require(index < len(slots), f"{loc}: index {index} outside {len(slots)} slots")
        require(
            item["name"] == slots[index],
            f"{loc}: slot {index} must be {slots[index]!r}",
        )
        if unique:
            require(index not in seen, f"{loc}: duplicate update index {index}")
        seen.add(index)
    return {slots[i] for i in seen}


def validate_contract(document):
    object_keys(document, {"schema_version", "kernels"}, {"source"}, "contract")
    require(
        type(document["schema_version"]) is int and document["schema_version"] == 1,
        "contract: unsupported schema_version (expected 1)",
    )
    if "source" in document:
        nonempty(document["source"], "contract.source")
    kernels = document["kernels"]
    require(
        isinstance(kernels, list) and bool(kernels), "kernels: expected nonempty array"
    )
    kernel_names = set()
    for i, kernel in enumerate(kernels):
        loc = f"kernels[{i}]"
        object_keys(
            kernel,
            {"name", "launch_args", "launch_count", "loads", "fast_launch"},
            {"unused_args"},
            loc,
        )
        nonempty(kernel["name"], f"{loc}.name")
        require(kernel["name"] not in kernel_names, f"{loc}: duplicate kernel name")
        kernel_names.add(kernel["name"])
        args = names(kernel["launch_args"], f"{loc}.launch_args")
        size = count(kernel["launch_count"], f"{loc}.launch_count")
        require(
            size == len(args),
            f"{loc}: launch_count must equal number of 64-bit slots ({len(args)})",
        )
        loaded = mappings(kernel["loads"], args, f"{loc}.loads")
        unused = reasons(kernel.get("unused_args", {}), f"{loc}.unused_args")
        require(
            unused == set(args) - loaded,
            f"{loc}: unconsumed slots need exact unused_args reasons",
        )

        fast = kernel["fast_launch"]
        require(isinstance(fast, dict), f"{loc}.fast_launch: expected object")
        require(
            type(fast.get("enabled")) is bool,
            f"{loc}.fast_launch.enabled: expected boolean",
        )
        if not fast["enabled"]:
            object_keys(fast, {"enabled", "reason"}, set(), f"{loc}.fast_launch")
            nonempty(fast["reason"], f"{loc}.fast_launch.reason")
            continue
        object_keys(
            fast,
            {
                "enabled",
                "cache_capacity",
                "cached_args",
                "launch_count",
                "required_updates",
                "updates",
                "invariant_args",
            },
            {"cache_launch_offset"},
            f"{loc}.fast_launch",
        )
        cached = names(fast["cached_args"], f"{loc}.fast_launch.cached_args")
        capacity = count(fast["cache_capacity"], f"{loc}.fast_launch.cache_capacity")
        require(len(cached) <= capacity, f"{loc}: cached_args exceed cache_capacity")
        offset = count(
            fast.get("cache_launch_offset", 0), f"{loc}.fast_launch.cache_launch_offset"
        )
        require(
            offset <= len(cached) and size <= len(cached) - offset,
            f"{loc}: cache launch slice exceeds cached_args bounds",
        )
        cache_end = offset + size
        require(
            cached[offset:cache_end] == args,
            f"{loc}: cache launch slice must preserve launch_args order",
        )
        require(
            count(fast["launch_count"], f"{loc}.fast_launch.launch_count") == size,
            f"{loc}: FastLaunch must send only the launch slice, not cache metadata",
        )
        required = set(
            names(fast["required_updates"], f"{loc}.fast_launch.required_updates")
        )
        invariant = reasons(fast["invariant_args"], f"{loc}.fast_launch.invariant_args")
        require(
            not (required & invariant),
            f"{loc}: fields cannot be both updated and invariant",
        )
        require(
            required | invariant == set(args),
            f"{loc}: every launch field must be classified as required_updates or invariant_args",
        )
        updated = mappings(
            fast["updates"], cached, f"{loc}.fast_launch.updates", unique=True
        )
        require(
            required <= updated,
            f"{loc}: FastLaunch missing updates {sorted(required - updated)}",
        )
        require(
            not (updated & invariant),
            f"{loc}: updated fields contradict invariant_args",
        )
    return len(kernels)


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "contract", type=Path, help="JSON declarations; compare separately against C++"
    )
    args = parser.parse_args(argv)
    try:
        data = json.loads(
            args.contract.read_text(encoding="utf-8"),
            object_pairs_hook=no_duplicate_keys,
        )
        checked = validate_contract(data)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"invalid launch contract: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "consistent",
                "kernels_checked": checked,
                "scope": "declared ABI only; C++ and runtime behavior not verified",
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
