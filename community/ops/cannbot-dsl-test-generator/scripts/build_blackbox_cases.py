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

"""Validate spec-traceable blackbox rows in testcase.csv."""

from __future__ import annotations

import logging
import sys
import argparse
import ast
import json
import re

from casebook import read_casebook
from recipe_validation import CPU_DTYPES, validate_tensor, validate_tensor_list
from generator_support import (
    add_io_arguments,
    artifact_paths,
    spec_path as input_spec_path,
    load_yaml_mapping,
)

CASE_ID = re.compile(r"^BB-[A-Za-z0-9][A-Za-z0-9-]{0,63}$")
NAN_CHECKS = {"nan_propagates", "produces_nan"}


LOGGER = logging.getLogger(__name__)


def fail(message):
    raise ValueError(message)


def required_refs(spec):
    refs = {"math_semantics": spec.get("math_semantics") or {}}
    for section in (
        "dtype_policy.supported_combinations",
        "semantic_cases",
        "boundary_conditions",
        "extreme_inputs",
    ):
        value = spec
        for key in section.split("."):
            value = value.get(key, {}) if isinstance(value, dict) else {}
        for index, item in enumerate(value or []):
            refs[f"{section}[{index}]"] = item
    return refs


def factor_targets(spec):
    targets = {}
    for item in spec.get("inputs") or []:
        name = item["name"]
        if item.get("optional"):
            targets[f"{name}.presence"] = ["absent", "present"]
        if item.get("role", "tensor") != "tensor":
            continue
        targets[f"{name}.dtype"] = list(item.get("dtype_set") or [])
        bounds = item.get("rank_range")
        if isinstance(bounds, list) and len(bounds) == 2:
            targets[f"{name}.rank"] = [str(rank) for rank in sorted(set(bounds))]
    return targets


def inventory(spec):
    refs = required_refs(spec)
    return {
        "operator": spec["op"]["name"],
        "inputs": spec.get("inputs") or [],
        "attributes": spec.get("attributes") or [],
        "outputs": spec.get("outputs") or [],
        "required_refs": refs,
        "factor_targets": factor_targets(spec),
        "levels": {
            "L0": "核心合法路径",
            "L1": "规格组合、边界与极值",
            "L2": "规格声明的异常路径",
        },
    }


def validate_pattern(case_id, name, pattern, inputs):
    value = inputs.get(name)
    if not isinstance(value, dict):
        fail(f"{case_id}: pattern target {name} is not a tensor recipe")
    expected = {
        "all_zero": ("zeros", None),
        "inject_nan_one_element": ("nan_one", None),
        "single_pos_inf": ("pos_inf_one", None),
        "single_neg_inf": ("neg_inf_one", None),
    }.get(pattern)
    same = re.fullmatch(r"all_same\((-?(?:\d+(?:\.\d*)?|\.\d+))\)", str(pattern))
    if same:
        expected = ("constant", float(same.group(1)))
    if expected is None:
        fail(f"{case_id}: unsupported synthesis pattern {pattern}")
    fill, constant = expected
    if value.get("fill") != fill or (
        constant is not None and value.get("value") != constant
    ):
        fail(f"{case_id}: pattern {pattern} is not reflected in {name} input")


def validate_literal_synthesis(case_id, ref, item, inputs):
    synth = item.get("synthesize") or {}
    if not isinstance(synth, dict):
        fail(f"{case_id}: {ref}.synthesize must be a mapping")
    validate_synthesis_shapes(case_id, ref, synth, inputs)
    validate_synthesis_dtypes(case_id, ref, synth, inputs)
    for key, pattern in synth.items():
        if isinstance(key, str) and key.endswith(".pattern"):
            validate_pattern(case_id, key[:-8], pattern, inputs)
    for item_pattern in synth.get("patterns") or []:
        if (
            not isinstance(item_pattern, dict)
            or item_pattern.get("target") not in inputs
        ):
            fail(f"{case_id}: {ref} has invalid pattern target")
        name = item_pattern["target"]
        pattern = item_pattern.get("pattern")
        validate_pattern(case_id, name, pattern, inputs)


def validate_case(raw, spec, refs, seen):
    if not isinstance(raw, dict):
        fail("each blackbox case must be a mapping")
    case_id = raw.get("id")
    if (
        not isinstance(case_id, str)
        or not CASE_ID.fullmatch(case_id)
        or case_id in seen
    ):
        fail(f"invalid or duplicate blackbox case ID: {case_id}")
    seen.add(case_id)
    level = raw.get("level")
    if level not in ("L0", "L1", "L2"):
        fail(f"{case_id}: level must be L0, L1 or L2")
    purpose = raw.get("purpose")
    if not isinstance(purpose, str) or not purpose.strip():
        fail(f"{case_id}: purpose is required")
    case_refs = raw.get("spec_refs")
    if not isinstance(case_refs, list) or not case_refs:
        fail(f"{case_id}: spec_refs must name unique required spec paths")
    if any(ref not in refs for ref in case_refs) or len(set(case_refs)) != len(
        case_refs
    ):
        fail(f"{case_id}: spec_refs must name unique required spec paths")
    expected, error_code = validate_case_expectation(
        case_id, raw, spec, refs, case_refs
    )
    validate_case_inputs(case_id, raw, spec, refs, error_code)
    outputs = raw.get("output_tensors") or {}
    if outputs:
        output_names = {item["name"] for item in spec.get("outputs") or []}
        if not isinstance(outputs, dict) or set(outputs) != output_names:
            fail(f"{case_id}: output_tensors must contain exactly the spec outputs")
        for name, value in outputs.items():
            validate_tensor(
                case_id, name, value, {"dtype_set": [value.get("dtype")]}, None
            )
    seed = raw.get("seed", 42)
    if type(seed) is not int or seed < 0:
        fail(f"{case_id}: seed must be a nonnegative integer")
    clean = dict(raw)
    clean["seed"] = seed
    return clean


def factor_coverage(spec, cases):
    """Report observed executable factors without treating path references as value coverage."""
    from collections import defaultdict
    from itertools import combinations

    observed = defaultdict(lambda: defaultdict(list))
    pairs = defaultdict(list)
    for case in cases:
        if case["level"] == "L2":
            continue
        factors = case_factors(spec, case)
        for key, value in factors.items():
            observed[key][value].append(case["id"])
        for (left, left_value), (right, right_value) in combinations(
            sorted(factors.items()), 2
        ):
            pairs[f"{left}={left_value} | {right}={right_value}"].append(case["id"])
    targets = factor_targets(spec)
    missing = {
        key: [value for value in values if value not in observed.get(key, {})]
        for key, values in targets.items()
    }
    return {
        "scope": "L0/L1 executable cases only",
        "observed_values": {
            key: dict(values) for key, values in sorted(observed.items())
        },
        "observed_pairs": dict(sorted(pairs.items())),
        "declared_targets": targets,
        "missing_declared_targets": {
            key: values for key, values in missing.items() if values
        },
        "note": "Observed pairs are evidence, not a claim of exhaustive pairwise coverage.",
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    add_io_arguments(parser, design=False, units=False)
    parser.add_argument(
        "--inventory", action="store_true", help="print spec-derived factor inventory"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="print coverage report without writing files",
    )
    args = parser.parse_args()
    work = artifact_paths(args)
    spec_path = input_spec_path(work)
    spec = load_yaml_mapping(spec_path)
    op = (spec.get("op") or {}).get("name")
    if not isinstance(op, str) or not op:
        fail("spec.op.name is required")
    unsupported = set()
    for item in spec.get("inputs") or []:
        unsupported.update(set(item.get("dtype_set") or []) - CPU_DTYPES)
    if unsupported:
        fail(
            f"fixed CPU case constructor cannot represent spec dtypes {sorted(unsupported)}; "
            "add an exact recipe before claiming coverage"
        )
    factors = inventory(spec)
    if args.inventory:
        print(json.dumps(factors, ensure_ascii=False, indent=2))
        return
    rows = [row for row in read_casebook(work, op) if row["sheet"] == "blackbox"]
    refs = factors["required_refs"]
    raw_cases, exclusions = parse_blackbox_rows(rows)
    if not raw_cases:
        fail("blackbox testcase.csv needs active cases")
    seen = set()
    cases = [validate_case(raw, spec, refs, seen) for raw in raw_cases]
    if not any(case["level"] == "L0" for case in cases):
        fail("blackbox plan needs at least one L0 core case")
    report = coverage_report(op, spec, refs, cases, exclusions)
    if args.coverage:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    LOGGER.info(f"validated {len(cases)} blackbox cases")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def validate_synthesis_shapes(case_id, ref, synth, inputs):
    shaped = {}
    for key, value in synth.items():
        if isinstance(key, str) and key.endswith(".shape"):
            shaped[key[:-6]] = value
    shaped.update(synth.get("shapes") or {})
    for name, raw_shape in shaped.items():
        if name not in inputs or not isinstance(inputs[name], dict):
            fail(f"{case_id}: {ref} requires tensor input {name}")
        try:
            shape = (
                ast.literal_eval(raw_shape) if isinstance(raw_shape, str) else raw_shape
            )
        except (ValueError, SyntaxError):
            continue  # Symbolic shapes are reviewed against spec constraints by the design agent.
        if (
            isinstance(shape, list)
            and all(type(dim) is int for dim in shape)
            and inputs[name].get("shape") != shape
        ):
            fail(f"{case_id}: {ref} literal shape for {name} does not match case")


def validate_synthesis_dtypes(case_id, ref, synth, inputs):
    dtype_spec = synth.get("dtype")
    if dtype_spec is None:
        dtype_spec = {}
    elif isinstance(dtype_spec, str):
        if dtype_spec not in CPU_DTYPES:
            fail(f"{case_id}: {ref}.synthesize.dtype is unsupported: {dtype_spec!r}")
        # Scalar dtype applies to supplied tensor recipes, not scalar attributes
        # or absent optional inputs. TensorList members share the declared dtype.
        dtype_spec = {
            name: dtype_spec
            for name, value in inputs.items()
            if isinstance(value, dict) and ("shape" in value or "tensors" in value)
        }
    elif not isinstance(dtype_spec, dict):
        fail(f"{case_id}: {ref}.synthesize.dtype must be a string or mapping")
    for name, dtype in dtype_spec.items():
        if not isinstance(dtype, str) or dtype not in CPU_DTYPES:
            fail(
                f"{case_id}: {ref}.synthesize.dtype for {name} is unsupported: {dtype!r}"
            )
        value = inputs.get(name)
        recipes = (
            value.get("tensors")
            if isinstance(value, dict) and "tensors" in value
            else [value]
        )
        if not isinstance(recipes, list) or any(
            not isinstance(recipe, dict) or recipe.get("dtype") != dtype
            for recipe in recipes
        ):
            fail(f"{case_id}: {ref} dtype for {name} does not match case")


def validate_case_expectation(case_id, raw, spec, refs, case_refs):
    level = raw.get("level")
    expected = raw.get("expected")
    error_code = (
        expected.removeprefix("raises_error:")
        if isinstance(expected, str) and expected.startswith("raises_error:")
        else None
    )
    if expected not in ("match_golden", "match_golden_nan") and not error_code:
        fail(f"{case_id}: unsupported expected outcome {expected}")
    if error_code and error_code not in (
        (spec.get("op") or {}).get("error_codes") or []
    ):
        fail(
            f"{case_id}: error code {error_code} is not declared in spec.op.error_codes"
        )
    if bool(error_code) != (level == "L2"):
        fail(f"{case_id}: L2 is reserved for expected errors")
    checks = [
        ((refs[ref].get("machine_check") or {}).get("kind"))
        for ref in case_refs
        if isinstance(refs[ref], dict)
    ]
    if error_code and "raises_error" not in checks:
        fail(f"{case_id}: expected error must trace to a raises_error spec case")
    if expected == "match_golden_nan" and not any(
        kind in NAN_CHECKS for kind in checks
    ):
        fail(f"{case_id}: match_golden_nan needs a NaN spec case")
    validate_reference_expectations(case_id, refs, case_refs, expected, error_code)
    return expected, error_code


def validate_case_inputs(case_id, raw, spec, refs, error_code):
    case_refs = raw["spec_refs"]
    expected = raw["expected"]
    inputs = raw.get("inputs")
    if not isinstance(inputs, dict):
        fail(f"{case_id}: inputs must be a mapping")
    input_meta = validate_input_names(case_id, inputs, spec, error_code)
    for ref in case_refs:
        if not ref.startswith("semantic_cases["):
            continue
        semantic = refs[ref].get("inputs") or {}
        if set(semantic.get("required") or []) - set(inputs) or set(
            semantic.get("forbidden") or []
        ) & set(inputs):
            fail(f"{case_id}: {ref} required/forbidden inputs do not match case")
    validate_input_recipes(case_id, inputs, input_meta, error_code)
    if expected == "match_golden_nan" and not any(
        isinstance(value, dict)
        and value.get("fill")
        in ("nan_one", "pos_inf_one", "neg_inf_one", "pos_inf", "neg_inf")
        for value in inputs.values()
    ):
        fail(f"{case_id}: NaN case needs a nonfinite input recipe")
    for ref in case_refs:
        if ref.startswith(("boundary_conditions[", "extreme_inputs[")):
            validate_literal_synthesis(case_id, ref, refs[ref], inputs)
    for ref in case_refs:
        if not ref.startswith("dtype_policy.supported_combinations["):
            continue
        combo = refs[ref]
        for name, dtype in (combo.get("inputs") or {}).items():
            actual = inputs.get(name)
            if not isinstance(actual, dict) or actual.get("dtype") != dtype:
                fail(f"{case_id}: {ref} input dtype combination does not match case")


def case_factors(spec, case):
    factors = {}
    for item in spec.get("inputs") or []:
        name = item["name"]
        recipe = case["inputs"].get(name)
        if isinstance(recipe, dict) and "shape" in recipe:
            shape = recipe["shape"]
            factors[f"{name}.dtype"] = recipe["dtype"]
            factors[f"{name}.rank"] = str(len(shape))
            factors[f"{name}.shape_class"] = (
                "empty" if 0 in shape else "scalar" if not shape else "nonempty"
            )
            factors[f"{name}.fill"] = recipe.get("fill", "normal")
        elif isinstance(recipe, dict) and "tensors" in recipe:
            factors[f"{name}.list_length"] = str(len(recipe["tensors"]))
        elif recipe is None and item.get("optional"):
            factors[f"{name}.presence"] = "absent"
        elif name in case["inputs"]:
            factors[f"{name}.value"] = str(recipe)
        if item.get("optional"):
            factors[f"{name}.presence"] = (
                "present" if name in case["inputs"] else "absent"
            )
    for item in spec.get("attributes") or []:
        name = item["name"]
        if name in case["inputs"]:
            factors[f"{name}.value"] = str(case["inputs"][name])
    return factors


def parse_blackbox_rows(rows):
    raw_cases = []
    exclusions = []
    for row in rows:
        paths = [item.strip() for item in row["design_ref"].split(",") if item.strip()]
        if row["status"] == "excluded":
            if not CASE_ID.fullmatch(row["case_id"]):
                fail(f"{row['case_id']}: invalid excluded blackbox ID")
            if len(paths) != 1:
                fail(f"{row['case_id']}: excluded blackbox row needs one spec path")
            exclusions.append({"spec_ref": paths[0], "reason": row["exclude_reason"]})
            continue
        try:
            inputs = json.loads(row["inputs_json"])
            outputs = json.loads(row["output_json"] or "{}")
            seed = int(row["seed"] or "42")
        except (ValueError, TypeError) as exc:
            fail(f"{row['case_id']}: invalid JSON or seed: {exc}")
        raw_cases.append(
            {
                "id": row["case_id"],
                "level": row["condition"],
                "purpose": row["note"],
                "spec_refs": paths,
                "inputs": inputs,
                "output_tensors": outputs,
                "seed": seed,
                "expected": row["expected"],
            }
        )
    return raw_cases, exclusions


def coverage_report(op, spec, refs, cases, exclusions):
    covered = {ref: [] for ref in refs}
    for case in cases:
        for ref in case["spec_refs"]:
            covered[ref].append(case["id"])
    excluded = {}
    for item in exclusions:
        if (
            not isinstance(item, dict)
            or item.get("spec_ref") not in refs
            or not str(item.get("reason") or "").strip()
        ):
            fail("each exclusion needs a required spec_ref and nonempty reason")
        ref = item["spec_ref"]
        if ref in excluded or covered[ref]:
            fail(f"duplicate or already covered exclusion: {ref}")
        excluded[ref] = item["reason"]
    missing = sorted(ref for ref in refs if not covered[ref] and ref not in excluded)
    if missing:
        fail(f"uncovered blackbox spec paths: {missing}")
    if not any(
        "math_semantics" in case["spec_refs"] and case["level"] != "L2"
        for case in cases
    ):
        fail("math_semantics needs a positive executable blackbox case")
    report = {
        "operator": op,
        "covered": covered,
        "exclusions": excluded,
        "case_count": len(cases),
        "factor_coverage": factor_coverage(spec, cases),
    }
    return report


def validate_reference_expectations(case_id, refs, case_refs, expected, error_code):
    for ref in case_refs:
        item = refs[ref]
        check = (item.get("machine_check") or {}) if isinstance(item, dict) else {}
        kind = check.get("kind")
        if kind == "raises_error" and error_code != check.get("error_type"):
            fail(f"{case_id}: {ref} requires raises_error:{check.get('error_type')}")
        if kind and kind != "raises_error" and error_code:
            fail(f"{case_id}: {ref} is not an error case")
        if kind in NAN_CHECKS and expected != "match_golden_nan":
            fail(f"{case_id}: {ref} needs match_golden_nan")


def validate_input_names(case_id, inputs, spec, error_code):
    input_meta = {item["name"]: item for item in spec.get("inputs") or []}
    attributes = {item["name"] for item in spec.get("attributes") or []}
    if set(inputs) - (set(input_meta) | attributes):
        fail(f"{case_id}: inputs contain names outside spec")
    required_inputs = {
        name for name, meta in input_meta.items() if not meta.get("optional")
    }
    if required_inputs - set(inputs) and error_code != "null_input":
        fail(f"{case_id}: inputs do not cover all required spec inputs")
    required_attrs = {
        item["name"] for item in spec.get("attributes") or [] if "default" not in item
    }
    if required_attrs - set(inputs):
        fail(
            f"{case_id}: inputs omit required attributes {sorted(required_attrs - set(inputs))}"
        )
    return input_meta


def validate_input_recipes(case_id, inputs, input_meta, error_code):
    for name, value in inputs.items():
        if name in input_meta and input_meta[name].get("role", "tensor") == "tensor":
            if value is None and error_code == "null_input":
                continue
            validate_tensor(case_id, name, value, input_meta[name], error_code)
        elif name in input_meta and input_meta[name].get("role") == "tensor_list":
            validate_tensor_list(case_id, name, value, input_meta[name], error_code)
        elif isinstance(value, (dict, list)):
            fail(f"{case_id}.{name}: scalar or attribute must be a JSON scalar")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
