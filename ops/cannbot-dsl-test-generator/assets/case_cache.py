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

"""Reusable CPU input and reference data; target execution is never cached."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import tempfile
from pathlib import Path

import torch

CACHE_VERSION = 4


def _output_spec(spec=None):
    if spec is None:
        spec = {}  # __CANNBOTDSL_OUTPUT_SPEC__
    return {
        key: spec.get(key) or default
        for key, default in (
            ("inputs", []), ("outputs", []), ("attributes", []),
            ("dtype_policy", {}), ("semantic_cases", []),
        )
    }


def _contract_expression(expression, inputs, attributes, input_metadata=None):
    """Interpret only the existing presence/length DSL, never Python eval."""
    def visit(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in ("true", "false"):
            return node.id == "true"
        if isinstance(node, (ast.List, ast.Tuple)):
            return [visit(item) for item in node.elts]
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "attr":
                return attributes[node.attr]
            parent = node.value
            if (isinstance(parent, ast.Attribute) and isinstance(parent.value, ast.Name)
                    and parent.value.id == "input"):
                value = inputs.get(parent.attr)
                if node.attr == "is_present":
                    return value is not None
                if node.attr == "dtype" and isinstance(value, torch.Tensor):
                    return str(value.dtype).removeprefix("torch.")
                if node.attr == "dtype" and isinstance(value, (tuple, list)):
                    if not all(isinstance(item, torch.Tensor) for item in value):
                        raise ValueError(f"input.{parent.attr}.dtype requires TensorList members")
                    dtypes = {str(item.dtype).removeprefix("torch.") for item in value}
                    if len(dtypes) == 1:
                        return next(iter(dtypes))
                    if not value:
                        declared = (input_metadata or {}).get(parent.attr, {}).get("dtype_set") or []
                        if len(declared) == 1:
                            return declared[0]
                    raise ValueError(f"cannot determine one dtype for input.{parent.attr} TensorList")
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name) and node.value.value.id == "shape"):
            axis = visit(node.slice)
            if type(axis) is int:
                return inputs[node.value.attr].shape[axis]
        if isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            if isinstance(node.op, ast.Not):
                return not value
            if isinstance(node.op, ast.USub) and type(value) is int:
                return -value
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(visit(value) for value in node.values)
            return any(visit(value) for value in node.values)
        if isinstance(node, ast.Compare):
            left = visit(node.left)
            for op, right_node in zip(node.ops, node.comparators):
                right = visit(right_node)
                if isinstance(op, ast.Eq):
                    passed = left == right
                elif isinstance(op, ast.NotEq):
                    passed = left != right
                elif isinstance(op, ast.In):
                    passed = left in right
                elif isinstance(op, ast.NotIn):
                    passed = left not in right
                else:
                    raise ValueError("unsupported output contract comparison")
                if not passed:
                    return False
                left = right
            return True
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if type(left) is int and type(right) is int:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "len" and len(node.args) == 1 and not node.keywords
                and isinstance(node.args[0], ast.Attribute)
                and isinstance(node.args[0].value, ast.Name) and node.args[0].value.id == "attr"):
            return len(visit(node.args[0]))
        raise ValueError(f"unsupported output contract expression: {expression}")

    try:
        return visit(ast.parse(expression, mode="eval").body)
    except (SyntaxError, KeyError, IndexError, AttributeError, TypeError) as exc:
        raise ValueError(f"cannot resolve output contract expression: {expression}") from exc


def _output_presence(declared, inputs, spec):
    attributes = {item["name"]: item["default"] for item in spec["attributes"] if "default" in item}
    attributes.update({name: inputs[name] for name in attributes if name in inputs})
    for item in spec["attributes"]:
        if item["name"] in inputs:
            attributes[item["name"]] = inputs[item["name"]]
    required = {item["name"] for item in declared if not item.get("optional", False)}
    forbidden = set()
    input_metadata = {item["name"]: item for item in spec["inputs"]}
    for case in spec["semantic_cases"]:
        outputs = case.get("outputs") or {}
        if not (outputs.get("present") or outputs.get("absent")):
            continue
        if _contract_expression(case["when"], inputs, attributes, input_metadata):
            required.update(outputs.get("present") or [])
            forbidden.update(outputs.get("absent") or [])
    if required & forbidden:
        raise ValueError("conflicting output presence requirements")
    return required, forbidden


def validate_outputs(value, names, inputs, spec, output_map=None):
    """Return named outputs after checking the existing public output contract."""
    declared = spec["outputs"] or [{"name": name} for name in names]
    if not names or [item["name"] for item in declared] != list(names):
        raise ValueError("output contract names disagree with Golden metadata")
    metadata = {item["name"]: item for item in declared}
    if isinstance(value, dict):
        if set(value) - set(names):
            raise ValueError("output contract contains undeclared output names")
        result = {name: value.get(name) for name in names}
    elif len(names) == 1:
        item = declared[0]
        if (item.get("role", "tensor") == "tensor" and isinstance(value, (tuple, list))
                and len(value) == 1):
            value = value[0]
        result = {names[0]: value}
    elif isinstance(value, (tuple, list)) and len(value) == len(names):
        result = dict(zip(names, value))
    else:
        raise ValueError(f"output contract requires {len(names)} ordered outputs")
    required, forbidden = _output_presence(declared, inputs, spec)
    attributes = {item["name"]: inputs.get(item["name"], item.get("default"))
                  for item in spec["attributes"]}
    for name, item in metadata.items():
        output = result[name]
        if output is None:
            if name in required:
                raise ValueError(f"output.{name}: missing required output")
            continue
        if name in forbidden:
            raise ValueError(f"output.{name}: output must be absent for this case")
        role = item.get("role", "tensor")
        if role == "tensor":
            tensors = [output]
        elif role == "tensor_list" and isinstance(output, (tuple, list)):
            tensors = output
            length = item.get("list_length") or {}
            kind = length.get("kind")
            if kind == "fixed":
                allowed = len(output) == length["value"]
            elif kind == "range":
                allowed = length["min"] <= len(output) <= length["max"]
            elif kind == "unconstrained":
                allowed = True
            elif kind == "same_as":
                side, ref = length["ref"].split(".", 1)
                source = inputs if side == "input" else result
                referenced = source.get(ref)
                if not isinstance(referenced, (tuple, list)):
                    raise ValueError(f"output.{name}: list_length reference is not a list")
                allowed = len(output) == len(referenced)
            elif kind == "expression":
                count = _contract_expression(length["value"], inputs, attributes)
                if type(count) is not int or count < 0:
                    raise ValueError(f"output.{name}: list_length must resolve to a nonnegative integer")
                allowed = len(output) == count
            else:
                raise ValueError(f"output.{name}: unsupported or missing list_length contract")
            if not allowed:
                raise ValueError(f"output.{name}: list_length contract mismatch")
        else:
            raise ValueError(f"output.{name}: unsupported output role {role!r}")
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"output.{name}: expected Tensor for declared {role}")
            dtype = str(tensor.dtype).removeprefix("torch.")
            if item.get("dtype_set") and dtype not in item["dtype_set"]:
                raise ValueError(f"output.{name}: dtype outside declared dtype_set")
            rank = (item.get("shape_bounds") or {}).get("output_rank")
            if rank is not None and tensor.ndim != rank:
                raise ValueError(f"output.{name}: output_rank contract mismatch")
        recipe = (output_map or {}).get(name)
        if recipe:
            recipes = recipe.get("tensors") if role == "tensor_list" else [recipe]
            if not isinstance(recipes, list) or len(recipes) != len(tensors):
                raise ValueError(f"output.{name}: output buffer count mismatch")
            for tensor, buffer in zip(tensors, recipes):
                if list(tensor.shape) != buffer["shape"] or str(tensor.dtype).removeprefix("torch.") != buffer["dtype"]:
                    raise ValueError(f"output.{name}: output buffer shape/dtype mismatch")
    combinations = (spec["dtype_policy"] or {}).get("supported_combinations") or []
    if combinations:
        def matches(mapping, values):
            for name, dtype in mapping.items():
                value = values.get(name)
                members = value if isinstance(value, (tuple, list)) else [value]
                if any(
                    isinstance(member, torch.Tensor) and str(member.dtype).removeprefix("torch.") != dtype
                    for member in members
                ):
                    return False
            return True
        candidates = [combo for combo in combinations if matches(combo.get("inputs") or {}, inputs)]
        if not candidates or not any(
            set(combo.get("outputs") or {}) == set(names)
            and matches(combo["outputs"], result) for combo in candidates
        ):
            raise ValueError("output dtype combination violates the declared input/output contract")
    return result


def canonical_output(value):
    """Keep Tensor values while removing nonportable/equivalent containers."""
    if value is None or isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        return {name: canonical_output(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(canonical_output(item) for item in value)
    raise ValueError("output contract requires Tensor leaves or optional None")


def zero_buffers(value):
    """Reset destination Tensor leaves, including TensorList members."""
    if isinstance(value, torch.Tensor):
        value.zero_()
    elif isinstance(value, dict):
        for item in value.values():
            zero_buffers(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            zero_buffers(item)
    elif value is not None:
        raise ValueError("output buffers must contain only Tensors, containers or None")


def load_golden(path):
    path = Path(path)
    source = path.read_bytes()
    spec = importlib.util.spec_from_file_location("cannbotdsl_golden", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"missing golden: {path}")
    module = importlib.util.module_from_spec(spec)
    # The code and cache identity must use the same source, even for edits with
    # identical size/mtime that Python's timestamp bytecode cache can miss.
    exec(compile(source, str(path), "exec"), module.__dict__)
    policies = getattr(module, "TOLERANCE", {})
    native = getattr(module, "NATIVE_REFERENCE_FUNCTION", None)
    needs_native = any(isinstance(value, dict) and value.get("metric") == "staged_relative"
                       and value.get("fallback", {}).get("reference_mode", "native") == "native"
                       for value in policies.values())
    if needs_native or hasattr(module, "NATIVE_REFERENCE_FUNCTION"):
        if not callable(native) or native is getattr(module, "GOLDEN_FUNCTION", None):
            raise ValueError("NATIVE_REFERENCE_FUNCTION must bind a separate independent reference")
    module.__cannbotdsl_source_hash__ = hashlib.sha256(source).hexdigest()
    return module


def _check_cpu_data(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("case cache requires CPU tensors")
    elif isinstance(value, dict):
        if not all(type(key) is str for key in value):
            raise ValueError("case cache mapping keys must be strings")
        for item in value.values():
            _check_cpu_data(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _check_cpu_data(item)
    elif value is not None and type(value) not in (bool, int, float, str):
        raise ValueError(f"unsupported case cache value: {type(value).__name__}")


class CaseCache:
    def __init__(self, test_dir, golden, spec=None):
        cache_dir = os.environ.get("CANNBOTDSL_CASE_CACHE_DIR")
        self.root = Path(cache_dir).expanduser() if cache_dir else Path(test_dir) / ".case_cache"
        self.golden = golden
        self.spec = _output_spec(spec)
        self.identity = {
            "version": CACHE_VERSION,
            "golden": golden.__cannbotdsl_source_hash__,
            "torch": str(torch.__version__),
        }
        self.hits = 0
        self.built = 0

    def get(self, row, call, index):
        recipe = {
            **self.identity,
            "sheet": row["sheet"],
            "case_id": row["case_id"],
            "call": index,
            "inputs": call["inputs"],
            "seed": call["seed"],
            "expected": row.get("expected"),
            "assertions": call["assertions"],
            "outputs": json.loads(row.get("output_json") or "{}"),
        }
        if not isinstance(recipe["outputs"], dict) or set(recipe["outputs"]) - set(self.golden.OUTPUT_NAMES):
            raise ValueError("output_json must use declared output names")
        dst_args = bool(recipe["outputs"]) or row.get("note") == "dst_args"
        key = hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\0" + json.dumps(self.spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        path = self.root / f"{key}.pt"
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
            self._validate(data, key, recipe, dst_args)
            data["expected"] = canonical_output(data["expected"])
            zero_buffers(data["outputs"])
        except (
            OSError,
            RuntimeError,
            EOFError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            pickle.UnpicklingError,
        ):
            data = self._build(recipe, key, dst_args)
            self._write(path, data)
            self.built += 1
        else:
            self.hits += 1
        return data

    def _validate(self, data, key, recipe, dst_args=False):
        if (
            not isinstance(data, dict)
            or set(data) != {"version", "key", "inputs", "expected", "outputs"}
            or data["version"] != CACHE_VERSION
            or data["key"] != key
            or not isinstance(data["inputs"], dict)
            or not isinstance(data["outputs"], dict)
        ):
            raise ValueError("invalid case cache entry")
        _check_cpu_data(data)
        reference = None
        if recipe["expected"] in ("match_golden", "match_golden_nan"):
            reference = self.validate_output(data["expected"], data["inputs"], recipe["outputs"])
        if dst_args:
            supplied = {name for name, value in recipe["outputs"].items() if value is not None}
            if set(data["outputs"]) != supplied:
                raise ValueError("destination buffers disagree with output_json")
            declared = self.spec["outputs"] or [{"name": name} for name in self.golden.OUTPUT_NAMES]
            required = {item["name"] for item in declared if not item.get("optional", False)}
            if required - supplied:
                raise ValueError("destination buffers missing required outputs")
            # Error cases deliberately violate input preconditions. Do not turn
            # output dtype/presence checks into new prerequisites for that call.
            if not (recipe["expected"] or "").startswith("raises_error:"):
                buffers = self.validate_output(data["outputs"], data["inputs"], recipe["outputs"])
                if reference is not None and any(
                    (reference[name] is None) != (buffers[name] is None) for name in reference
                ):
                    raise ValueError("destination buffers must match reference output presence")

    def validate_output(self, value, inputs, output_map=None):
        return validate_outputs(value, self.golden.OUTPUT_NAMES, inputs, self.spec, output_map)

    def _build(self, recipe, key, dst_args=False):
        inputs = self.golden.make_inputs(recipe["inputs"], seed=recipe["seed"])
        expected = None
        if recipe["expected"] in ("match_golden", "match_golden_nan"):
            primary = getattr(self.golden, "GOLDEN_FUNCTION", None)
            if not callable(primary):
                raise ValueError("GOLDEN_FUNCTION must bind a callable primary reference")
            kwargs = {name: inputs[name] for name in self.golden.INPUT_NAMES if name in inputs}
            kwargs.update(self.golden.ATTR_DEFAULTS)
            for name in list(self.golden.ATTR_DEFAULTS) + self.golden.REQUIRED_ATTR_NAMES:
                if name in inputs:
                    kwargs[name] = inputs[name]
            # Reference and target use the same materialization. Isolate reference
            # mutations without losing aliases within its input/attribute mapping.
            expected = primary(**copy.deepcopy(kwargs))
        outputs = {}
        if recipe["outputs"]:
            supplied = {name: value for name, value in recipe["outputs"].items() if value is not None}
            if supplied:
                outputs = self.golden.make_inputs(supplied, seed=0)
            zero_buffers(outputs)
        data = {
            "version": CACHE_VERSION,
            "key": key,
            "inputs": inputs,
            "expected": expected,
            "outputs": outputs,
        }
        self._validate(data, key, recipe, dst_args)
        data["expected"] = canonical_output(data["expected"])
        return data

    def _write(self, path, data):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.root, prefix=".case-", suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                torch.save(data, handle)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def prepare(self, rows, expand_calls):
        for row in rows:
            if row["status"] != "active":
                continue
            inputs = json.loads(row["inputs_json"])
            seed = int(row.get("seed") or 42)
            assertions = json.loads(row.get("assertions_json") or "[]")
            if (row.get("expected") or "").startswith("raises_error:"):
                calls = [{"inputs": inputs, "seed": seed, "assertions": assertions}]
            else:
                _, _, calls = expand_calls(inputs, seed, assertions)
            for index, call in enumerate(calls):
                self.get(row, call, index)
        return {"built": self.built, "hits": self.hits}
