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

"""Validate TDD rows against caller-supplied Unit obligations."""

from __future__ import annotations

import logging
import sys
import argparse
from dataclasses import dataclass
import json
import re
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets"))
from case_contract import validate_assertions, validate_checks, check_inputs, case_calls

from casebook import read_casebook
from recipe_validation import validate_tensor, validate_tensor_list
from generator_support import (
    add_io_arguments,
    artifact_paths,
    spec_path as input_spec_path,
    units_dir,
    golden_path,
    load_yaml_mapping,
)
from build_whitebox_cases import validated_branches

UNIT_ID = re.compile(r"^[a-z][a-z0-9-]{0,63}$")


LOGGER = logging.getLogger(__name__)


def fail(message):
    raise ValueError(message)


def split_ids(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def validated_units(work):
    spec = load_yaml_mapping(input_spec_path(work))
    op = spec["op"]["name"]
    rows = read_casebook(work, op)
    tdd = [row for row in rows if row["sheet"] == "tdd"]
    if not tdd or any(row["status"] != "active" for row in tdd):
        fail("TDD needs active rows; exclusions belong in native Unit design")
    whitebox = [row for row in rows if row["sheet"] == "whitebox"]
    runnable = None
    if whitebox:
        branches = validated_branches(work)
        runnable = {
            item["branch_id"]
            for item in branches["branches"]
            if item["status"] == "draft"
        }
    native = load_native_units(work)
    validate_test_contracts(native)
    if any(row["unit_id"] not in {uid.lower() for uid in native} for row in tdd):
        fail("TDD row references unknown native Unit")
    adapted = adapt_native_units(native, tdd, runnable)
    return {"operator": op, "units": adapted}


@dataclass(frozen=True)
class CaseContext:
    input_names: set
    public_names: set
    input_meta: dict
    draft: dict | None
    branches: dict


def validate_test_contracts(native):
    all_ids = set()
    for uid, unit in native.items():
        obligations = unit.get("verification_obligations")
        if not isinstance(obligations, list) or not obligations:
            fail(f"{uid}: verification_obligations must be a nonempty list")
        for item in obligations:
            if not isinstance(item, dict):
                fail(f"{uid}: obligation must be a mapping")
            ref = item.get("id")
            if not isinstance(ref, str) or not ref.strip() or ref in all_ids:
                fail(f"{uid}: obligation IDs must be nonempty and globally unique")
            all_ids.add(ref)
            if not isinstance(item.get("kind"), str) or item["kind"] not in {
                "numerical",
                "structural",
                "semantic",
                "device",
            }:
                fail(
                    f"{uid}: unsupported verification obligation kind: {item.get('kind')}"
                )
            conditions = item.get("input_conditions", [])
            if conditions != []:
                validate_checks(conditions)
    validate_dependency_graph(
        [
            {"id": uid, "depends_on": unit.get("depends_on", [])}
            for uid, unit in native.items()
        ]
    )


def validate_materialized_conditions(work, units, op):
    native = load_native_units(work)
    module = None
    for unit in units:
        obligations = native[unit["native_id"]].get("verification_obligations") or []
        required = {
            item["id"]: item.get("input_conditions") or [] for item in obligations
        }
        covered = set()
        for case in unit["cases"]:
            _, _, calls = case_calls(case["inputs"], case["seed"], case["assertions"])
            for call in calls:
                checks = [
                    c
                    for a in call["assertions"]
                    if a["kind"] == "input_conditions"
                    for c in a["checks"]
                ]
                if checks:
                    if module is None:
                        spec = importlib.util.spec_from_file_location(
                            "condition_golden", golden_path(work, op)
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                    check_inputs(
                        module.make_inputs(call["inputs"], call["seed"]), checks
                    )
                for ref in case["obligation_ids"]:
                    if required[ref] and all(
                        check in checks for check in required[ref]
                    ):
                        covered.add(ref)
        missing = {ref for ref, checks in required.items() if checks} - covered
        if missing:
            fail(f"{unit['id']}: input conditions not bound to TDD: {sorted(missing)}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    add_io_arguments(parser, design=True, units=True)
    args = parser.parse_args()
    work = artifact_paths(args)
    spec = load_yaml_mapping(input_spec_path(work))
    op = (spec.get("op") or {}).get("name")
    units_doc = validated_units(work)
    rows = [row for row in read_casebook(work, op) if row["sheet"] == "whitebox"]
    draft = validated_branches(work) if rows else None
    golden = golden_path(work, op)
    if not golden.is_file():
        fail(f"missing Golden: {golden}")
    branches = (
        {item["branch_id"]: item for item in draft.get("branches") or []}
        if draft
        else {}
    )
    input_names = {
        item["name"] for item in spec.get("inputs") or [] if not item.get("optional")
    }
    input_names |= {
        item["name"] for item in spec.get("attributes") or [] if "default" not in item
    }
    public_names = input_names | {item["name"] for item in spec.get("attributes") or []}
    public_names |= {item["name"] for item in spec.get("inputs") or []}
    input_meta = {item["name"]: item for item in spec.get("inputs") or []}
    units = units_doc.get("units") or []
    if not isinstance(units, list) or not units:
        fail("native Unit needs nonempty units")
    context = CaseContext(input_names, public_names, input_meta, draft, branches)
    seen, cases = set(), []
    for unit in units:
        unit_id = unit.get("id")
        if (
            not isinstance(unit_id, str)
            or not UNIT_ID.fullmatch(unit_id)
            or unit_id in seen
        ):
            fail(f"invalid or duplicate unit id: {unit_id}")
        seen.add(unit_id)
        cases.extend(validate_unit_cases(unit, context))
    validate_dependency_graph(units)
    validate_materialized_conditions(work, units, op)
    if len({case["case_id"] for case in cases}) != len(cases):
        fail("duplicate TDD case ID")
    LOGGER.info(f"validated {len(cases)} TDD rows")


def validate_unit_evidence(row, obligations, refs):
    try:
        inputs = json.loads(row["inputs_json"])
        outputs = json.loads(row["output_json"] or "{}")
        assertions = json.loads(row["assertions_json"] or "[]")
    except ValueError as exc:
        fail(f"{row['case_id']}: invalid JSON: {exc}")
    if (
        not isinstance(inputs, dict)
        or not isinstance(outputs, dict)
        or not isinstance(assertions, list)
    ):
        fail(f"{row['case_id']}: invalid input, output or assertions")
    expected = row["expected"]
    validate_assertions(assertions)
    runtime = [item for item in assertions if item["kind"] != "execution"]
    if expected == "assertions":
        if not any(item["kind"] != "input_conditions" for item in runtime):
            fail(f"{row['case_id']}: assertions needs an observable output or probe")
    elif expected != "match_golden":
        fail(f"{row['case_id']}: expected must be match_golden or assertions")
    validate_obligation_evidence(row, obligations, refs, expected, assertions)
    return inputs, outputs, assertions, expected


def validate_dependency_graph(units):
    graph = {}
    for unit in units:
        deps = unit.get("depends_on", [])
        if (
            not isinstance(deps, list)
            or any(not isinstance(dep, str) or not dep for dep in deps)
            or len(deps) != len(set(deps))
        ):
            fail(f"{unit['id']}: depends_on must contain unique unit IDs")
        graph[unit["id"]] = deps
    visiting, visited = set(), set()

    def visit(unit_id):
        if unit_id in visiting:
            fail(f"unit dependency cycle includes {unit_id}")
        if unit_id in visited:
            return
        visiting.add(unit_id)
        for dep in graph[unit_id]:
            if dep not in graph:
                fail(f"unknown unit dependency: {dep}")
            visit(dep)
        visiting.remove(unit_id)
        visited.add(unit_id)

    for unit_id in graph:
        visit(unit_id)


def load_native_units(work):
    native = {}
    for path in sorted(units_dir(work).glob("U*.yaml")):
        unit = load_yaml_mapping(path).get("unit")
        if not isinstance(unit, dict) or unit.get("unit_id") != path.stem:
            fail(f"invalid native Unit: {path}")
        native[path.stem] = unit
    if not native:
        fail("no native U*.yaml in --units-dir")
    return native


def adapt_native_units(native, tdd, runnable):
    adapted = []
    for native_id, unit in native.items():
        uid = native_id.lower()
        obligations = {
            item["id"]: item for item in unit.get("verification_obligations") or []
        }
        if not obligations:
            fail(f"{native_id}: missing verification obligations")
        for obligation in obligations.values():
            if obligation.get("kind") not in {
                "numerical",
                "structural",
                "semantic",
                "device",
            }:
                fail(
                    f"{native_id}: unsupported verification obligation kind: {obligation.get('kind')}; "
                    "performance requirements belong in the performance validation stage"
                )
        deps = unit.get("depends_on") or []
        if any(dep not in native or dep == native_id for dep in deps):
            fail(f"{native_id}: invalid dependency")
        cases, covered = [], set()
        for row in (row for row in tdd if row["unit_id"] == uid):
            case, refs = validate_unit_row(row, uid, obligations, runnable)
            cases.append(case)
            covered.update(refs)
        if not cases or covered != set(obligations):
            fail(f"{native_id}: uncovered verification obligations")
        adapted.append(
            {
                "id": uid,
                "native_id": native_id,
                "scope": unit.get("title", native_id),
                "depends_on": [dep.lower() for dep in deps],
                "branch_ids": sorted(
                    {bid for case in cases for bid in case["branch_ids"]}
                ),
                "cases": cases,
            }
        )
    return adapted


def validate_unit_row(row, uid, obligations, runnable):
    prefix = f"TD-{uid}-"
    if not row["case_id"].startswith(prefix) or not re.fullmatch(
        r"[a-z][a-z0-9-]*", row["case_id"][len(prefix) :]
    ):
        fail(f"{row['case_id']}: invalid TDD case ID")
    refs = split_ids(row["obligation_ids"])
    if not refs or len(refs) != len(set(refs)) or not set(refs) <= set(obligations):
        fail(f"{row['case_id']}: invalid obligation IDs")
    branch_ids = split_ids(row["branch_ids"])
    allowed_branches = runnable if runnable is not None else set()
    if (
        len(branch_ids) != len(set(branch_ids))
        or not set(branch_ids) <= allowed_branches
    ):
        fail(f"{row['case_id']}: invalid runnable branch IDs")
    inputs, outputs, assertions, expected = validate_unit_evidence(
        row, obligations, refs
    )
    case = {
        "id": row["case_id"][len(prefix) :],
        "inputs": inputs,
        "output_tensors": outputs,
        "obligation_ids": refs,
        "branch_ids": branch_ids,
        "expected": expected,
        "assertions": assertions,
        "seed": int(row.get("seed") or 42),
    }
    _, _, calls = case_calls(inputs, int(row.get("seed") or 42), assertions)
    for call in calls[1:]:
        validate_obligation_evidence(
            row, obligations, refs, expected, call["assertions"]
        )
    return case, refs


def validate_unit_cases(unit, context):
    unit_id = unit["id"]
    draft, branches = context.draft, context.branches
    cases = []
    branch_ids = unit.get("branch_ids") or []
    if not isinstance(branch_ids, list):
        fail(f"{unit_id}: branch_ids must be a list")
    if draft is None and branch_ids:
        fail(f"{unit_id}: branch_ids require a whitebox branch map")
    if draft is not None and any(
        branch_id not in branches or branches[branch_id]["status"] != "draft"
        for branch_id in branch_ids
    ):
        fail(f"{unit_id}: branch_ids must reference runnable whitebox branches")
    deps = unit.get("depends_on") or []
    if not isinstance(deps, list):
        fail(f"{unit_id}: depends_on must be a list")
    examples = unit.get("cases")
    if examples is None and branch_ids:
        example = unit.get("example") or branches[branch_ids[0]].get("suggested_input")
        examples = [{"id": "main", "inputs": example, "branch_ids": branch_ids}]
    if not isinstance(examples, list) or not examples:
        fail(f"{unit_id}: cases must be nonempty")
    for index, item in enumerate(examples, start=1):
        cases.append(validate_execution_case(item, index, unit, context))
    return cases


def validate_execution_case(item, index, unit, context):
    unit_id = unit["id"]
    branch_ids = unit.get("branch_ids") or []
    draft, branches = context.draft, context.branches
    validate_execution_inputs((unit_id, index), item, context)
    case_branches = item.get("branch_ids", branch_ids)
    if not isinstance(case_branches, list) or any(
        branch_id not in branch_ids for branch_id in case_branches
    ):
        fail(f"{unit_id}: case {index} references an unrelated branch")
    suffix = item.get("id") or f"case-{index}"
    if not UNIT_ID.fullmatch(suffix):
        fail(f"{unit_id}: invalid case id suffix {suffix}")
    case_id = f"TD-{unit_id}-{suffix}"
    _, _, calls = case_calls(
        item["inputs"], item.get("seed", 42), item.get("assertions") or []
    )
    for call in calls[1:]:
        validate_execution_inputs((unit_id, index), {"inputs": call["inputs"]}, context)
        validate_obligation_evidence(
            {"case_id": case_id},
            {},
            [],
            item.get("expected", "match_golden"),
            call["assertions"],
        )
        if item.get("expected") == "assertions" and not any(
            a["kind"] != "input_conditions" for a in call["assertions"]
        ):
            fail(f"{case_id}: sequence assertion call lacks observable evidence")
    outputs = item.get("output_tensors") or {}
    if draft and draft.get("call_style") == "dst_args" and not outputs:
        if not case_branches:
            fail(f"{case_id}: dst_args requires output_tensors or a branch")
        outputs = branches[case_branches[0]].get("output_tensors") or {}
    return {
        "case_id": case_id,
        "unit_id": unit_id,
        "branch_ids": case_branches,
        "scope": str(unit.get("scope") or ""),
        "inputs": item["inputs"],
        "output_tensors": outputs,
        "expected": item.get("expected", "match_golden"),
        "assertions": item.get("assertions") or [],
        "obligation_ids": item.get("obligation_ids") or [],
        "seed": item.get("seed", 42),
    }


def validate_obligation_evidence(row, obligations, refs, expected, assertions):
    if any(obligations[ref]["kind"] == "structural" for ref in refs):
        if not any(
            item["kind"] in {"probe_events", "probe_predicates"} for item in assertions
        ):
            fail(
                f"{row['case_id']}: structural obligation requires probe_events or probe_predicates"
            )
    if (
        any(obligations[ref]["kind"] == "numerical" for ref in refs)
        and expected != "match_golden"
    ):
        fail(f"{row['case_id']}: numerical obligation requires Golden")
    assertion_kinds = {item.get("kind") for item in assertions}
    if any(obligations[ref]["kind"] == "device" for ref in refs):
        if "output_device" not in assertion_kinds:
            fail(
                f"{row['case_id']}: device obligation requires output_device assertion"
            )
    if any(obligations[ref]["kind"] == "semantic" for ref in refs):
        if not assertion_kinds.intersection(
            {
                "output_shape",
                "output_dtype",
                "output_equals_input",
                "probe_events",
                "probe_predicates",
            }
        ):
            fail(
                f"{row['case_id']}: semantic obligation requires an observable behavior assertion"
            )


def validate_execution_inputs(label, item, context):
    input_names, public_names, input_meta = (
        context.input_names,
        context.public_names,
        context.input_meta,
    )
    unit_id, index = label
    if not isinstance(item, dict) or not isinstance(item.get("inputs"), dict):
        fail(f"{unit_id}: case {index} needs inputs")
    if not input_names.issubset(item["inputs"]):
        fail(f"{unit_id}: case {index} is missing spec inputs")
    if set(item["inputs"]) - public_names:
        fail(f"{unit_id}: case {index} has names outside spec")
    for name, recipe in item["inputs"].items():
        if name not in input_meta:
            if isinstance(recipe, (dict, list)):
                fail(f"{unit_id}: case {index} attribute {name} must be a JSON scalar")
            continue
        meta = input_meta[name]
        if meta.get("role", "tensor") == "tensor":
            validate_tensor(f"{unit_id}: case {index}", name, recipe, meta, None)
        elif meta.get("role") == "tensor_list":
            validate_tensor_list(f"{unit_id}: case {index}", name, recipe, meta, None)


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
