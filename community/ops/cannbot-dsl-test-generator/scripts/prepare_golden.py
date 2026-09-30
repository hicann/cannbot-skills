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

"""Validate a CPU reference, then package operator-named Golden and test runner."""

from __future__ import annotations

import logging
import sys
import argparse
from dataclasses import dataclass
import ast
import re
from pathlib import Path

from generator_support import (
    add_io_arguments,
    artifact_paths,
    design_path as input_design_path,
    spec_path as input_spec_path,
    golden_path,
    test_dir,
)

try:
    import yaml
except ImportError:
    yaml = None

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SUPPORTED_DTYPES = {
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
SUPPORT = '''

def _seed_for(name, seed):
    value = int(seed) & 0x7FFFFFFF
    for character in str(name):
        value = (value * 131 + ord(character)) & 0x7FFFFFFF
    return value or 1


def make_inputs(spec_map, seed=42):
    """Create deterministic, independent CPU inputs for one testcase.csv row."""
    import math
    dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    result = {}
    for name, item in spec_map.items():
        if isinstance(item, dict) and "tensors" in item:
            if not isinstance(item["tensors"], list):
                raise ValueError(f"{name}.tensors must be a list")
            result[name] = [
                make_inputs({f"{name}[{index}]": member}, seed)[f"{name}[{index}]"]
                for index, member in enumerate(item["tensors"])
            ]
            continue
        if not isinstance(item, dict) or "shape" not in item:
            result[name] = item
            continue
        dtype = dtypes[item["dtype"]]
        shape = item["shape"]
        generator = torch.Generator().manual_seed(_seed_for(name, seed))
        fill = item.get("fill", "normal")
        if fill == "values":
            def flatten(values):
                return [number for value in values for number in (flatten(value) if isinstance(value, list) else [value])]
            values = torch.tensor(flatten(item["values"]), dtype=dtype)
            result[name] = values.reshape(shape)
        elif fill == "arange":
            axis = item.get("axis")
            count = math.prod(shape) if axis is None else shape[axis]
            values = (torch.arange(count, dtype=torch.float64) * item.get("step", 1) + item.get("start", 0)).to(dtype)
            if axis is None:
                result[name] = values.reshape(shape)
            else:
                view_shape = [1] * len(shape)
                view_shape[axis] = count
                result[name] = values.reshape(view_shape).expand(shape).clone()
        elif fill == "zeros":
            result[name] = torch.zeros(shape, dtype=dtype)
        elif fill == "ones":
            result[name] = torch.ones(shape, dtype=dtype)
        elif fill == "constant":
            result[name] = torch.full(shape, item["value"], dtype=dtype)
        elif fill in ("nan_one", "pos_inf_one", "neg_inf_one") and dtype.is_floating_point:
            result[name] = torch.zeros(shape, dtype=dtype)
            result[name].reshape(-1)[0] = {
                "nan_one": float("nan"),
                "pos_inf_one": float("inf"),
                "neg_inf_one": -float("inf"),
            }[fill]
        elif fill in ("pos_inf", "neg_inf") and dtype.is_floating_point:
            result[name] = torch.full(
                shape, float("inf") if fill == "pos_inf" else -float("inf"), dtype=dtype
            )
        elif fill != "normal":
            raise ValueError(f"unsupported fill for {name}: {fill}")
        elif dtype.is_floating_point:
            result[name] = torch.randn(shape, dtype=dtype, generator=generator)
        elif dtype == torch.bool:
            result[name] = torch.randint(0, 2, shape, generator=generator).bool()
        else:
            low, high = int(item.get("low", 0)), int(item.get("high", 3))
            if high <= low:
                raise ValueError(f"invalid integer range for {name}")
            result[name] = torch.randint(low, high, shape, dtype=dtype, generator=generator)
    return result


def simulate(spec_map, seed=42):
    """Return CPU inputs and the independent expected output."""
    inputs = make_inputs(spec_map, seed)
    kwargs = {name: inputs[name] for name in INPUT_NAMES if name in inputs}
    kwargs.update(ATTR_DEFAULTS)
    for name in list(ATTR_DEFAULTS) + REQUIRED_ATTR_NAMES:
        if name in inputs:
            kwargs[name] = inputs[name]
    return inputs, GOLDEN_FUNCTION(**kwargs)
'''


def fail(message):
    raise ValueError(message)


@dataclass(frozen=True)
class ReferenceContract:
    op: str
    inputs: list[str]
    attrs: list[str]
    optional_inputs: list[str]


def checked_reference(source: str, contract: ReferenceContract, design: str) -> None:
    op, inputs, attrs, optional_inputs = (
        contract.op,
        contract.inputs,
        contract.attrs,
        contract.optional_inputs,
    )
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        fail(f"final Golden is invalid Python: {exc}")
    refs, functions = reference_declarations(tree)
    if (
        not isinstance(refs, (list, tuple))
        or not refs
        or not all(
            isinstance(ref, str) and ref.strip() and ref in design for ref in refs
        )
    ):
        fail("DESIGN_REFS must point to text in the supplied design document")
    name = f"{op}_golden"
    matched = [node for node in functions if node.name == name]
    if len(matched) != 1 or isinstance(matched[0], ast.AsyncFunctionDef):
        fail(f"final Golden must define exactly one synchronous {name} function")
    actual = [arg.arg for arg in matched[0].args.posonlyargs + matched[0].args.args]
    variadic = any(
        (matched[0].args.vararg, matched[0].args.kwarg, matched[0].args.kwonlyargs)
    )
    if actual != inputs + attrs or variadic:
        fail(
            f"{name} parameters must be exactly spec inputs then attributes: {inputs + attrs}"
        )
    if matched[0].decorator_list:
        fail("golden reference must not use decorators")
    defaulted = (
        set(actual[-len(matched[0].args.defaults) :])
        if matched[0].args.defaults
        else set()
    )
    if set(optional_inputs) - defaulted:
        fail(
            f"optional spec inputs need defaults in {name}: {sorted(set(optional_inputs) - defaulted)}"
        )
    if re.search(
        r"\bcannbotdsl\b|\btorch_npu\b|\.npu\s*\(|\bCANNBOTDSL_OP_MODULE\b", source
    ):
        fail("golden reference must be CPU-only and independent of the implementation")


def main():
    if yaml is None:
        logging.basicConfig(
            level=logging.ERROR, format="%(message)s", stream=sys.stderr
        )
        LOGGER.error("PyYAML is required")
        return 2
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    add_io_arguments(parser, design=True, units=False)
    args = parser.parse_args()
    work = artifact_paths(args)
    spec_path, design_path = input_spec_path(work), input_design_path(work)
    for path in (spec_path, design_path):
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            fail(f"required nonempty input missing: {path}")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        fail("spec.yaml must be a mapping")
    op = (spec.get("op") or {}).get("name")
    if not isinstance(op, str) or not NAME.fullmatch(op):
        fail("spec.op.name must be a Python identifier")
    reference_path = golden_path(work, op)
    if (
        not reference_path.is_file()
        or not reference_path.read_text(encoding="utf-8").strip()
    ):
        fail(f"required nonempty input missing: {reference_path}")
    inputs, optional_inputs, outputs, attrs, defaults, required_attrs = (
        golden_spec_fields(spec)
    )
    design = design_path.read_text(encoding="utf-8")
    source = reference_path.read_text(encoding="utf-8")
    checked_reference(
        source, ReferenceContract(op, inputs, attrs, optional_inputs), design
    )
    tolerance = (spec.get("numerical_tolerance") or {}).get("per_dtype") or {}
    unsupported = {
        dtype: limits.get("metric") if isinstance(limits, dict) else limits
        for dtype, limits in tolerance.items()
        if not isinstance(limits, dict)
        or limits.get("metric") not in ("max_relative", "bitwise_equal")
    }
    if unsupported:
        fail(f"fixed runner cannot execute these tolerance metrics: {unsupported}")
    required = {
        "OP_NAME": op,
        "INPUT_NAMES": inputs,
        "OPTIONAL_INPUT_NAMES": optional_inputs,
        "OUTPUT_NAMES": outputs,
        "ATTR_DEFAULTS": defaults,
        "REQUIRED_ATTR_NAMES": required_attrs,
        "TOLERANCE": tolerance,
    }
    validate_reference_metadata(source, op, required)
    compile(source, str(reference_path), "exec")
    tests = test_dir(work, op)
    tests.mkdir(parents=True, exist_ok=True)
    runner = (ROOT / "assets" / "test.py").read_text(encoding="utf-8")
    runner = runner.replace('"__CANNBOTDSL_OP_NAME__"', repr(op))
    (tests / f"test_{op}.py").write_text(runner, encoding="utf-8")
    helper = (ROOT / "assets" / "case_contract.py").read_text(encoding="utf-8")
    (tests / "cannbotdsl_case_contract.py").write_text(helper, encoding="utf-8")
    for sheet in ("tdd", "whitebox", "blackbox"):
        (tests / "testcase_output" / sheet).mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"validated {reference_path}")
    LOGGER.info(f"wrote {tests / f'test_{op}.py'}")


def reference_declarations(tree):
    refs = None
    functions = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(
                name != "torch"
                and not name.startswith("torch.")
                and name not in ("math", "typing")
                for name in modules
            ):
                fail(f"final Golden imports unsupported module: {modules}")
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DESIGN_REFS":
                    try:
                        refs = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        fail("DESIGN_REFS must be a literal list/tuple")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)
    return refs, functions


def validate_reference_metadata(source, op, required):
    tree = ast.parse(source)
    metadata = {}
    golden_binding = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            key = node.targets[0].id
            if key in (
                "OP_NAME",
                "INPUT_NAMES",
                "OPTIONAL_INPUT_NAMES",
                "OUTPUT_NAMES",
                "ATTR_DEFAULTS",
                "REQUIRED_ATTR_NAMES",
                "TOLERANCE",
            ):
                try:
                    metadata[key] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    fail(f"Golden metadata {key} must be a literal")
            if key == "GOLDEN_FUNCTION" and isinstance(node.value, ast.Name):
                golden_binding = node.value.id
    for key, value in required.items():
        if metadata.get(key) != value:
            fail(f"Golden metadata {key} disagrees with spec")
    if golden_binding != f"{op}_golden":
        fail("GOLDEN_FUNCTION must bind the named Golden")
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if not {"make_inputs", "simulate"} <= functions:
        fail("final Golden needs make_inputs and simulate for the fixed runner")


def golden_spec_fields(spec):
    inputs = [item["name"] for item in spec.get("inputs") or []]
    unsupported_dtypes = set()
    for item in spec.get("inputs") or []:
        unsupported_dtypes.update(set(item.get("dtype_set") or []) - SUPPORTED_DTYPES)
    if unsupported_dtypes:
        fail(
            f"fixed CPU Golden input constructor cannot represent spec dtypes: {sorted(unsupported_dtypes)}"
        )
    optional_inputs = [
        item["name"] for item in spec.get("inputs") or [] if item.get("optional")
    ]
    outputs = [item["name"] for item in spec.get("outputs") or []]
    attrs = [item["name"] for item in spec.get("attributes") or []]
    if (
        not inputs
        or not outputs
        or not all(
            isinstance(name, str) and NAME.fullmatch(name)
            for name in inputs + attrs + outputs
        )
    ):
        fail("spec inputs/attributes/outputs must have valid names")
    if len(set(inputs + attrs)) != len(inputs + attrs) or len(set(outputs)) != len(
        outputs
    ):
        fail("duplicate spec input, attribute or output name")
    defaults = {}
    for item in spec.get("attributes") or []:
        if "default" in item:
            defaults[item["name"]] = item["default"]
    required_attrs = [
        item["name"] for item in spec.get("attributes") or [] if "default" not in item
    ]
    return inputs, optional_inputs, outputs, attrs, defaults, required_attrs


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
