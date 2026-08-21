#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Run a CATLASS DSL solution against NPU-KernelBench-style inputs."""

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path


os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"


DEFAULT_WARMUP = 1
DEFAULT_TRIALS = 2
ANTI_HACK_POLICY = "single-fused-catlass-kernel-v1"


ALLOWED_TORCH_CALLS = {
    "torch.device",
    "torch.empty",
    "torch.empty_like",
    "torch.empty_strided",
    "torch.is_tensor",
}
TENSOR_METADATA_METHODS = {
    "dim",
    "is_contiguous",
    "ndimension",
    "numel",
    "size",
    "stride",
}
TENSOR_METADATA_ATTRIBUTES = {
    "device",
    "dtype",
    "layout",
    "ndim",
    "requires_grad",
    "shape",
}


DTYPES = {
    "bool",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
}
INPUT_TYPES = {
    "random", "zeros", "ones", "scalar", "tensor", "null",
    "custom", "safetensors", "tensor_list",
}
AXIS_TYPES = {"const", "var", "expr"}
SOLUTION_LANGUAGES = {"python", "ascendc", "triton", "tilelang", "catlass"}
TARGET_HARDWARE = {"ascend910b", "ascend910_93", "ascend950", "LOCAL"}


class BenchError(ValueError):
    def __init__(self, category, message, details=None):
        super().__init__(message)
        self.category = category
        self.details = details


def _not_applicable_anti_hack(reason="cpu_device"):
    return {
        "status": "not_applicable",
        "policy": ANTI_HACK_POLICY,
        "declared_kernel_names": [],
        "observed_kernel_names": [],
        "profiled_iterations": 0,
        "observed_launches": 0,
        "launches_per_iteration": None,
        "reason": reason,
    }


def _qualified_name(node, aliases):
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value, aliases)
        return "{}.{}".format(parent, node.attr) if parent else node.attr
    return None


def _module_name(path):
    value = Path(path).with_suffix("").as_posix().strip("/")
    return value.replace("/", ".")


def _module_aliases(tree):
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".", 1)[0]] = item.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                if item.name != "*":
                    aliases[item.asname or item.name] = "{}.{}".format(
                        node.module, item.name
                    )
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            name = _qualified_name(value, aliases) if value is not None else None
            if not name or not (name == "torch" or name.startswith("torch.")):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and aliases.get(target.id) != name:
                    aliases[target.id] = name
                    changed = True
    return aliases


def _is_catlass_kernel(function, aliases):
    for decorator in function.decorator_list:
        name = _qualified_name(
            decorator.func if isinstance(decorator, ast.Call) else decorator,
            aliases,
        )
        if name in {"catlass.kernel", "catlass.tla.kernel"}:
            return True
    return False


def _source_location(path, node, operation):
    return {
        "path": path,
        "line": getattr(node, "lineno", None),
        "operation": operation,
    }


def _torch_source_violations(path, tree, aliases):
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _qualified_name(node.func, aliases)
        if name and (name == "torch" or name.startswith("torch.")):
            if name not in ALLOWED_TORCH_CALLS:
                violations.append(_source_location(path, node, name))
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and node.args
        ):
            owner = _qualified_name(node.args[0], aliases)
            if owner and (owner == "torch" or owner.startswith("torch.")):
                violations.append(_source_location(path, node, "getattr({})".format(owner)))
    return violations


def _function_arguments(function):
    positional = list(function.args.posonlyargs) + list(function.args.args)
    return [item.arg for item in positional]


def _contains_tainted_tensor(node, tainted):
    if isinstance(node, ast.Name):
        return node.id in tainted
    if isinstance(node, ast.Attribute):
        if node.attr in TENSOR_METADATA_ATTRIBUTES:
            return False
        return _contains_tainted_tensor(node.value, tainted)
    if isinstance(node, ast.Subscript):
        return _contains_tainted_tensor(node.value, tainted)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_contains_tainted_tensor(item, tainted) for item in node.elts)
    if isinstance(node, ast.Dict):
        return any(
            _contains_tainted_tensor(item, tainted)
            for item in list(node.keys) + list(node.values)
            if item is not None
        )
    return False


def _assignment_names(target):
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for item in target.elts:
            names.extend(_assignment_names(item))
        return names
    return []


def _resolve_helper(module, name, functions):
    if not name:
        return None
    local = "{}.{}".format(module, name) if "." not in name else name
    if local in functions:
        return local
    matches = [key for key in functions if key.endswith("." + name)]
    return matches[0] if len(matches) == 1 else None


def _host_tensor_violations(modules, functions, entry_key):
    """Track candidate tensors through host helpers and reject framework math."""
    tainted_parameters = {key: set() for key in functions}
    tainted_parameters[entry_key].update(_function_arguments(functions[entry_key][2]))
    tainted_locals = {key: set() for key in functions}
    return_tainted = {key: False for key in functions}
    violations = []
    seen_violations = set()

    def add_violation(path, node, operation):
        key = (path, getattr(node, "lineno", None), operation)
        if key not in seen_violations:
            seen_violations.add(key)
            violations.append(_source_location(path, node, operation))

    changed = True
    while changed:
        changed = False
        for key, (path, module, function) in functions.items():
            if _is_catlass_kernel(function, modules[module][2]):
                continue
            tainted = tainted_locals[key]
            before_parameters = len(tainted)
            tainted.update(tainted_parameters[key])
            if len(tainted) != before_parameters:
                changed = True
            aliases = modules[module][2]

            def expression_tainted(value):
                if _contains_tainted_tensor(value, tainted):
                    return True
                if isinstance(value, ast.BinOp):
                    return expression_tainted(value.left) or expression_tainted(
                        value.right
                    )
                if isinstance(value, ast.UnaryOp):
                    return expression_tainted(value.operand)
                if isinstance(value, ast.IfExp):
                    return expression_tainted(value.body) or expression_tainted(
                        value.orelse
                    )
                if isinstance(value, ast.BoolOp):
                    return any(expression_tainted(item) for item in value.values)
                if isinstance(value, ast.Compare):
                    return expression_tainted(value.left) or any(
                        expression_tainted(item) for item in value.comparators
                    )
                if isinstance(value, ast.Call):
                    called = _qualified_name(value.func, aliases)
                    if called in {
                        "torch.empty", "torch.empty_like", "torch.empty_strided"
                    }:
                        return True
                    helper = _resolve_helper(module, called, functions)
                    return bool(helper and return_tainted[helper])
                return False

            for node in ast.walk(function):
                if isinstance(node, ast.Call):
                    called = _qualified_name(node.func, aliases)
                    helper = _resolve_helper(module, called, functions)
                    if helper:
                        helper_args = _function_arguments(functions[helper][2])
                        for index, argument in enumerate(node.args):
                            if index < len(helper_args) and expression_tainted(argument):
                                if helper_args[index] not in tainted_parameters[helper]:
                                    tainted_parameters[helper].add(helper_args[index])
                                    changed = True
                    if isinstance(node.func, ast.Attribute):
                        owner = node.func.value
                        if (
                            _contains_tainted_tensor(owner, tainted)
                            and node.func.attr not in TENSOR_METADATA_METHODS
                        ):
                            add_violation(path, node, "Tensor.{}".format(node.func.attr))
                elif isinstance(node, ast.BinOp) and (
                    expression_tainted(node.left) or expression_tainted(node.right)
                ):
                    add_violation(path, node, type(node.op).__name__)
                elif isinstance(node, ast.UnaryOp) and expression_tainted(node.operand):
                    add_violation(path, node, type(node.op).__name__)
                elif isinstance(node, ast.BoolOp) and any(
                    expression_tainted(item) for item in node.values
                ):
                    add_violation(path, node, type(node.op).__name__)
                elif isinstance(node, ast.Compare) and (
                    expression_tainted(node.left)
                    or any(expression_tainted(item) for item in node.comparators)
                ):
                    add_violation(path, node, type(node.ops[0]).__name__)
                elif isinstance(node, ast.AugAssign) and (
                    expression_tainted(node.target) or expression_tainted(node.value)
                ):
                    add_violation(path, node, type(node.op).__name__)
                elif (
                    isinstance(node, ast.Attribute)
                    and node.attr not in TENSOR_METADATA_ATTRIBUTES
                    and node.attr not in TENSOR_METADATA_METHODS
                    and _contains_tainted_tensor(node.value, tainted)
                ):
                    add_violation(path, node, "Tensor.{}".format(node.attr))
                elif isinstance(node, ast.Subscript) and _contains_tainted_tensor(
                    node.value, tainted
                ):
                    add_violation(path, node, "Tensor.__getitem__")
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    value = node.value
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if value is not None and expression_tainted(value):
                        for target in targets:
                            for name in _assignment_names(target):
                                if name not in tainted:
                                    tainted.add(name)
                                    changed = True
                elif isinstance(node, ast.Return) and node.value is not None:
                    if expression_tainted(node.value) and not return_tainted[key]:
                        return_tainted[key] = True
                        changed = True
    return violations


def audit_solution_sources(solution):
    modules = {}
    functions = {}
    declared = set()
    violations = []
    for source in solution["sources"]:
        path = source["path"]
        try:
            tree = ast.parse(source["content"], filename=path)
        except SyntaxError as exc:
            raise BenchError(
                "configuration", "solution source 不是合法 Python: {}".format(path)
            ) from exc
        module = _module_name(path)
        aliases = _module_aliases(tree)
        modules[module] = (path, tree, aliases)
        violations.extend(_torch_source_violations(path, tree, aliases))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = "{}.{}".format(module, node.name)
                functions[key] = (path, module, node)
                if _is_catlass_kernel(node, aliases):
                    declared.add(node.name)

    entry_file, entry_symbol = solution["spec"]["entry_point"].split("::")
    entry_key = "{}.{}".format(_module_name(entry_file), entry_symbol)
    if entry_key not in functions:
        raise BenchError("configuration", "solution entry_point 函数未在源码中定义")
    violations.extend(_host_tensor_violations(modules, functions, entry_key))
    violations = sorted(
        violations,
        key=lambda item: (item["path"], item.get("line") or 0, item["operation"]),
    )
    reason = None
    if violations:
        reason = "torch_computation"
    elif not declared:
        reason = "no_catlass_kernel"
    status = "failed" if reason else "passed"
    return {
        "status": status,
        "policy": ANTI_HACK_POLICY,
        "declared_kernel_names": sorted(declared),
        "observed_kernel_names": [],
        "profiled_iterations": 0,
        "observed_launches": 0,
        "launches_per_iteration": None,
        "reason": reason,
        "source_violations": violations,
    }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchError("configuration", "无法读取 {}: {}".format(label, exc)) from exc
    if not isinstance(value, dict):
        raise BenchError("configuration", "{} 必须是 JSON 对象".format(label))
    return value


def _keys(value, required, optional, label):
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise BenchError(
            "configuration", "{} 缺少字段: {}".format(label, sorted(missing))
        )
    if unknown:
        raise BenchError(
            "configuration", "{} 包含未知字段: {}".format(label, sorted(unknown))
        )


def _string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise BenchError("configuration", "{} 必须是非空字符串".format(label))
    return value


def _safe_relative(value, label):
    _string(value, label)
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BenchError("configuration", "{} 必须是安全相对路径".format(label))
    return value


def _number(value, label, minimum=0.0):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise BenchError("configuration", "{} 必须是有限数值".format(label))
    return float(value)


def _tensor_specs(value, axes, label):
    if not isinstance(value, dict) or not value:
        raise BenchError("configuration", "{} 必须是非空对象".format(label))
    result = {}
    for name, spec in value.items():
        _string(name, "{} name".format(label))
        if not isinstance(spec, dict):
            raise BenchError("configuration", "{}.{} 必须是对象".format(label, name))
        _keys(
            spec,
            set(),
            {"shape", "dtype", "description"},
            "{}.{}".format(label, name),
        )
        shape = spec.get("shape")
        if shape is not None:
            if not isinstance(shape, list) or any(
                not isinstance(item, str)
                or not item
                or (not item.isdigit() and item not in axes)
                for item in shape
            ):
                raise BenchError(
                    "configuration", "{}.{}.shape 非法".format(label, name)
                )
        dtype = spec.get("dtype")
        if dtype is not None and dtype not in DTYPES:
            raise BenchError(
                "configuration", "{}.{}.dtype 不受支持".format(label, name)
            )
        result[name] = dict(spec)
    return result


def load_definition(path):
    value = _read_json(path, "definition")
    _keys(
        value,
        {"name", "axes", "inputs", "outputs", "reference"},
        {"op_type", "description", "custom_inputs_entrypoint", "kernel_reference"},
        "definition",
    )
    _string(value["name"], "definition.name")
    axes = value["axes"]
    if not isinstance(axes, dict):
        raise BenchError("configuration", "definition.axes 必须是对象")
    normalized_axes = {}
    for name, spec in axes.items():
        _string(name, "axis name")
        if not isinstance(spec, dict) or spec.get("type") not in AXIS_TYPES:
            raise BenchError("configuration", "axis type 必须是 const/var/expr")
        axis_type = spec["type"]
        optional = {"description"}
        if axis_type == "const":
            optional.add("value")
            if type(spec.get("value")) is not int or spec["value"] < 0:
                raise BenchError(
                    "configuration", "const axis value 必须是非负整数"
                )
        elif axis_type == "expr":
            optional.add("expression")
            _string(spec.get("expression"), "expr axis expression")
        _keys(spec, {"type"}, optional, "axis {}".format(name))
        normalized_axes[name] = dict(spec)
    inputs = _tensor_specs(value["inputs"], normalized_axes, "inputs")
    outputs = _tensor_specs(value["outputs"], normalized_axes, "outputs")
    if set(inputs) & set(outputs):
        raise BenchError("configuration", "definition 输入输出名称不能重叠")
    reference = _string(value["reference"], "definition.reference")
    try:
        tree = ast.parse(reference)
    except SyntaxError as exc:
        raise BenchError("configuration", "definition.reference 不是合法 Python") from exc
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run"
        for node in tree.body
    ):
        raise BenchError("configuration", "definition.reference 必须定义顶层 run")
    custom = value.get("custom_inputs_entrypoint")
    if custom is not None:
        _string(custom, "definition.custom_inputs_entrypoint")
    result = dict(value)
    result.update({"axes": normalized_axes, "inputs": inputs, "outputs": outputs})
    result["_path"] = str(Path(path).resolve())
    result["_sha256"] = _sha256(path)
    return result


def load_solution(path):
    value = _read_json(path, "solution")
    _keys(
        value,
        {"name", "definition", "author", "spec", "sources"},
        {"description"},
        "solution",
    )
    for field in ("name", "definition", "author"):
        _string(value[field], "solution.{}".format(field))
    spec = value["spec"]
    if not isinstance(spec, dict):
        raise BenchError("configuration", "solution.spec 必须是对象")
    _keys(
        spec,
        {"languages", "target_hardware", "entry_point"},
        {
            "dependencies",
            "destination_passing_style",
            "build_method",
            "bisheng_cflags",
            "ld_flags",
        },
        "solution.spec",
    )
    for field in ("languages", "target_hardware"):
        if (
            not isinstance(spec[field], list)
            or not spec[field]
            or any(not isinstance(item, str) or not item for item in spec[field])
        ):
            raise BenchError(
                "configuration", "solution.spec.{} 必须是非空字符串列表".format(field)
            )
    if set(spec["languages"]) - SOLUTION_LANGUAGES:
        raise BenchError("configuration", "solution.spec.languages 包含未知值")
    if set(spec["target_hardware"]) - TARGET_HARDWARE:
        raise BenchError("configuration", "solution.spec.target_hardware 包含未知值")
    entry = _string(spec["entry_point"], "solution.spec.entry_point")
    if entry.count("::") != 1:
        raise BenchError("configuration", "entry_point 必须是 <file>::<function>")
    entry_file, entry_symbol = entry.split("::")
    _safe_relative(entry_file, "entry_point file")
    _string(entry_symbol, "entry_point function")
    dps = spec.get("destination_passing_style", True)
    if type(dps) is not bool:
        raise BenchError("configuration", "destination_passing_style 必须是 bool")
    sources = value["sources"]
    if not isinstance(sources, list) or not sources:
        raise BenchError("configuration", "solution.sources 必须是非空列表")
    normalized_sources = []
    seen = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise BenchError("configuration", "solution source 必须是对象")
        _keys(source, {"path", "content"}, set(), "solution source")
        source_path = _safe_relative(source["path"], "solution source path")
        content = _string(source["content"], "solution source content")
        if source_path in seen:
            raise BenchError("configuration", "solution source path 重复")
        seen.add(source_path)
        normalized_sources.append({"path": source_path, "content": content})
    if entry_file not in seen:
        raise BenchError("configuration", "entry_point 文件不在 solution.sources")
    result = dict(value)
    result["spec"] = dict(spec)
    result["spec"]["destination_passing_style"] = dps
    result["sources"] = normalized_sources
    result["_path"] = str(Path(path).resolve())
    result["_sha256"] = _sha256(path)
    return result


def _validate_tolerance(value):
    if not isinstance(value, dict):
        raise BenchError("configuration", "workload.tolerance 必须是对象")
    _keys(
        value,
        set(),
        {
            "max_atol",
            "max_rtol",
            "required_matched_ratio",
            "max_error_cap",
            "allow_negative_inf",
        },
        "workload.tolerance",
    )
    result = {
        "max_atol": _number(value.get("max_atol", 1e-2), "max_atol"),
        "max_rtol": _number(value.get("max_rtol", 1e-2), "max_rtol"),
        "required_matched_ratio": _number(
            value.get("required_matched_ratio", 0.99),
            "required_matched_ratio",
        ),
        "max_error_cap": value.get("max_error_cap"),
        "allow_negative_inf": value.get("allow_negative_inf", False),
    }
    if not 0 < result["required_matched_ratio"] <= 1:
        raise BenchError(
            "configuration", "required_matched_ratio 必须在 (0, 1] 内"
        )
    if result["max_error_cap"] is not None:
        result["max_error_cap"] = _number(
            result["max_error_cap"], "max_error_cap"
        )
    if type(result["allow_negative_inf"]) is not bool:
        raise BenchError("configuration", "allow_negative_inf 必须是 bool")
    return result


def _validate_input(name, value):
    if value is None:
        return {"type": "null"}
    if not isinstance(value, dict) or value.get("type") not in INPUT_TYPES:
        raise BenchError(
            "configuration", "workload input {} type 不受支持".format(name)
        )
    input_type = value["type"]
    allowed = {
        "random": {"dtype", "shape", "range"},
        "zeros": {"dtype", "shape"},
        "ones": {"dtype", "shape"},
        "scalar": {"value"},
        "tensor": {"value", "dtype"},
        "null": set(),
        "custom": set(),
        "safetensors": {"path", "tensor_key"},
        "tensor_list": {"dtype", "shapes"},
    }[input_type]
    required = {
        "zeros": {"dtype", "shape"},
        "ones": {"dtype", "shape"},
        "scalar": {"value"},
        "tensor": {"value"},
        "safetensors": {"path", "tensor_key"},
        "tensor_list": {"dtype", "shapes"},
    }.get(input_type, set())
    _keys(value, {"type"} | required, allowed - required, "workload input")
    if "dtype" in value and value["dtype"] not in DTYPES:
        raise BenchError("configuration", "workload input dtype 不受支持")
    if "shape" in value and (
        not isinstance(value["shape"], list)
        or any(type(item) is not int or item < 0 for item in value["shape"])
    ):
        raise BenchError("configuration", "tensor shape 必须是非负整数列表")
    if "range" in value and (
        not isinstance(value["range"], list)
        or len(value["range"]) != 2
        or any(type(item) is not int for item in value["range"])
        or value["range"][0] >= value["range"][1]
    ):
        raise BenchError("configuration", "random range 必须是 [low, high]")
    if input_type == "safetensors":
        _string(value["path"], "safetensors path")
        _string(value["tensor_key"], "safetensors tensor_key")
    if input_type == "tensor_list":
        shapes = value["shapes"]
        if not isinstance(shapes, list) or any(
            not isinstance(shape, list)
            or any(type(item) is not int or item < 0 for item in shape)
            for shape in shapes
        ):
            raise BenchError(
                "configuration", "tensor_list.shapes 必须是 shape 列表"
            )
    return dict(value)


def load_workloads(path):
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BenchError("configuration", "无法读取 workload.jsonl") from exc
    workloads = []
    seen = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as exc:
            raise BenchError(
                "configuration", "workload 第 {} 行不是合法 JSON".format(line_number)
            ) from exc
        if not isinstance(value, dict):
            raise BenchError("configuration", "workload 每行必须是对象")
        _keys(value, {"uuid", "inputs"}, {"axes", "tolerance"}, "workload")
        workload_id = _string(value["uuid"], "workload.uuid")
        if workload_id in seen:
            raise BenchError("configuration", "workload.uuid 重复")
        axes = value.get("axes")
        if axes is not None and (
            not isinstance(axes, dict)
            or any(type(item) is not int or item < 0 for item in axes.values())
        ):
            raise BenchError("configuration", "workload.axes 必须是非负整数对象或 null")
        if not isinstance(value["inputs"], dict) or not value["inputs"]:
            raise BenchError("configuration", "workload.inputs 必须是非空对象")
        inputs = {
            name: _validate_input(name, spec)
            for name, spec in value["inputs"].items()
        }
        custom_count = sum(spec["type"] == "custom" for spec in inputs.values())
        if custom_count not in (0, len(inputs)):
            raise BenchError("configuration", "custom input 不能与其他 input type 混用")
        workloads.append(
            {
                "uuid": workload_id,
                "axes": axes,
                "inputs": inputs,
                "tolerance": _validate_tolerance(value.get("tolerance", {})),
                "_line": line_number,
            }
        )
        seen.add(workload_id)
    if not workloads:
        raise BenchError("configuration", "workload.jsonl 不能为空")
    return workloads, str(path.resolve()), _sha256(path)


def _evaluate_axis(expression, values):
    operations = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.FloorDiv: lambda left, right: left // right,
        ast.Mod: lambda left, right: left % right,
        ast.Pow: lambda left, right: left**right,
    }

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](visit(node.left), visit(node.right))
        raise BenchError("configuration", "axis expression 包含不支持的语法")

    try:
        result = visit(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ArithmeticError) as exc:
        raise BenchError("configuration", "axis expression 求值失败") from exc
    if type(result) is not int or result < 0:
        raise BenchError("configuration", "axis expression 结果必须是非负整数")
    return result


def resolve_axes(definition, workload):
    values = {
        name: spec["value"]
        for name, spec in definition["axes"].items()
        if spec["type"] == "const"
    }
    supplied = workload.get("axes") or {}
    unknown = set(supplied) - {
        name for name, spec in definition["axes"].items() if spec["type"] == "var"
    }
    if unknown:
        raise BenchError("configuration", "workload.axes 包含未知 var axis")
    values.update(supplied)
    for name, tensor_spec in definition["inputs"].items():
        input_spec = workload["inputs"].get(name, {})
        concrete = input_spec.get("shape")
        symbolic = tensor_spec.get("shape")
        if concrete is None or symbolic is None or len(concrete) != len(symbolic):
            continue
        for axis, size in zip(symbolic, concrete):
            if axis in definition["axes"] and definition["axes"][axis]["type"] == "var":
                if axis in values and values[axis] != size:
                    raise BenchError("configuration", "var axis 推断值不一致")
                values[axis] = size
    missing = {
        name
        for name, spec in definition["axes"].items()
        if spec["type"] == "var" and name not in values
    }
    if missing:
        raise BenchError(
            "configuration", "workload 缺少 var axes: {}".format(sorted(missing))
        )
    pending = {
        name: spec["expression"]
        for name, spec in definition["axes"].items()
        if spec["type"] == "expr"
    }
    while pending:
        progressed = False
        for name, expression in list(pending.items()):
            references = {
                node.id
                for node in ast.walk(ast.parse(expression, mode="eval"))
                if isinstance(node, ast.Name)
            }
            if references <= set(values):
                values[name] = _evaluate_axis(expression, values)
                del pending[name]
                progressed = True
        if not progressed:
            raise BenchError("configuration", "expr axis 存在未知引用或循环依赖")
    return values


def resolve_shape(spec, axes):
    shape = spec.get("shape")
    if shape is None:
        return None
    return [int(item) if item.isdigit() else axes[item] for item in shape]


def _torch_dtype(torch, name):
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise BenchError("environment_missing", "Torch 不支持 dtype {}".format(name))
    return dtype


def _load_safetensor(spec, root, device):
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise BenchError(
            "environment_missing", "safetensors input 需要 safetensors 包"
        ) from exc
    source = Path(spec["path"])
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    if not source.is_file():
        raise BenchError("configuration", "safetensors 文件不存在")
    values = load_file(str(source), device=str(device))
    if spec["tensor_key"] not in values:
        raise BenchError("configuration", "safetensors tensor_key 不存在")
    return values[spec["tensor_key"]]


def generate_inputs(definition, workload, reference_module, device, seed, workload_root):
    import torch

    axes = resolve_axes(definition, workload)
    random.seed(seed)
    torch.manual_seed(seed)
    backend_name = str(device).split(":", 1)[0]
    backend = getattr(torch, backend_name, None)
    if backend_name != "cpu" and backend is not None and hasattr(backend, "manual_seed_all"):
        backend.manual_seed_all(seed)

    if all(spec["type"] == "custom" for spec in workload["inputs"].values()):
        if set(workload["inputs"]) != set(definition["inputs"]):
            raise BenchError(
                "configuration", "custom workload inputs 必须完整匹配 definition inputs"
            )
        entry = definition.get("custom_inputs_entrypoint")
        generator = getattr(reference_module, entry, None) if entry else None
        if not callable(generator):
            raise BenchError(
                "configuration", "custom input 需要可调用的 custom_inputs_entrypoint"
            )
        values = generator(axes, str(device))
        if not isinstance(values, dict) or set(values) != set(definition["inputs"]):
            raise BenchError(
                "runtime", "custom_inputs_entrypoint 返回值必须完整匹配 definition inputs"
            )
        return [values[name] for name in definition["inputs"]], axes

    if set(workload["inputs"]) != set(definition["inputs"]):
        raise BenchError(
            "configuration", "workload.inputs 必须与 definition.inputs 一致"
        )
    inputs = []
    for name, tensor_spec in definition["inputs"].items():
        input_spec = workload["inputs"][name]
        input_type = input_spec["type"]
        if input_type == "scalar":
            inputs.append(input_spec["value"])
            continue
        if input_type == "null":
            inputs.append(None)
            continue
        if input_type == "tensor":
            dtype_name = input_spec.get("dtype")
            dtype = _torch_dtype(torch, dtype_name) if dtype_name else None
            inputs.append(torch.tensor(input_spec["value"], dtype=dtype, device=device))
            continue
        if input_type == "safetensors":
            inputs.append(_load_safetensor(input_spec, workload_root, device))
            continue
        if input_type == "tensor_list":
            dtype = _torch_dtype(torch, input_spec["dtype"])
            inputs.append(
                [
                    torch.randn(shape, dtype=dtype, device=device)
                    if dtype.is_floating_point
                    else torch.randint(0, 2, shape, dtype=dtype, device=device)
                    for shape in input_spec["shapes"]
                ]
            )
            continue
        if input_type == "custom":
            raise BenchError("configuration", "custom input 不能与其他类型混用")
        shape = input_spec.get("shape")
        if shape is None:
            shape = resolve_shape(tensor_spec, axes)
        dtype_name = input_spec.get("dtype") or tensor_spec.get("dtype")
        if shape is None or dtype_name is None:
            raise BenchError(
                "configuration", "random input {} 缺少 shape 或 dtype".format(name)
            )
        dtype = _torch_dtype(torch, dtype_name)
        if input_type == "zeros":
            value = torch.zeros(shape, dtype=dtype, device=device)
        elif input_type == "ones":
            value = torch.ones(shape, dtype=dtype, device=device)
        elif dtype.is_floating_point:
            value = torch.randn(shape, dtype=dtype, device=device)
        else:
            low, high = input_spec.get(
                "range", [0, 2] if dtype_name == "bool" else [-16, 16]
            )
            value = torch.randint(low, high, shape, dtype=dtype, device=device)
        inputs.append(value)
    return inputs, axes


def _load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise BenchError("build", "无法加载 Python 模块 {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise BenchError("build", "Python 模块加载失败: {}".format(exc)) from exc
    return module


def stage_solution(solution, output):
    root = Path(output) / "sources"
    root.mkdir(parents=True, exist_ok=True)
    for source in solution["sources"]:
        target = root / source["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, source["content"])
    entry_file, entry_symbol = solution["spec"]["entry_point"].split("::")
    sys.path.insert(0, str(root.resolve()))
    module = _load_module(
        root / entry_file, "catlass_dsl_solution_{}".format(uuid.uuid4().hex)
    )
    entry = getattr(module, entry_symbol, None)
    if not callable(entry):
        raise BenchError("build", "solution entry_point 不可调用")
    return entry, root


def load_reference(definition, output):
    path = Path(output) / "reference.py"
    _atomic_write(path, definition["reference"])
    module = _load_module(
        path, "catlass_dsl_reference_{}".format(uuid.uuid4().hex)
    )
    run = getattr(module, "run", None)
    if not callable(run):
        raise BenchError("build", "definition.reference.run 不可调用")
    return module, run, path


def clone_value(value, torch):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone_value(item, torch) for item in value)
    if isinstance(value, list):
        return [clone_value(item, torch) for item in value]
    if isinstance(value, dict):
        return {key: clone_value(item, torch) for key, item in value.items()}
    return value


def normalize_outputs(value, names, torch):
    if torch.is_tensor(value):
        values = [value]
    elif isinstance(value, (tuple, list)):
        values = list(value)
    else:
        raise BenchError("runtime", "输出必须是 Tensor 或 Tensor tuple/list")
    if len(values) != len(names) or any(not torch.is_tensor(item) for item in values):
        raise BenchError("runtime", "输出数量或类型与 definition.outputs 不一致")
    return dict(zip(names, values))


def allocate_outputs(definition, axes, inputs, torch):
    tensor = next((item for item in inputs if torch.is_tensor(item)), None)
    device = tensor.device if tensor is not None else torch.device("cpu")
    outputs = []
    for name, spec in definition["outputs"].items():
        shape = resolve_shape(spec, axes)
        dtype = spec.get("dtype")
        if shape is None or dtype is None:
            raise BenchError(
                "configuration", "DPS output {} 必须声明 shape 和 dtype".format(name)
            )
        outputs.append(
            torch.empty(shape, dtype=_torch_dtype(torch, dtype), device=device)
        )
    return outputs


def compare_outputs(candidate, reference, definition, axes, tolerance, torch):
    records = []
    passed = True
    for name in definition["outputs"]:
        actual = candidate[name]
        expected = reference[name]
        record = {
            "name": name,
            "shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "dtype": str(actual.dtype).split(".")[-1],
            "expected_dtype": str(expected.dtype).split(".")[-1],
        }
        expected_shape = resolve_shape(definition["outputs"][name], axes)
        expected_dtype = definition["outputs"][name].get("dtype")
        shape_ok = actual.shape == expected.shape and (
            expected_shape is None or list(actual.shape) == expected_shape
        )
        dtype_ok = actual.dtype == expected.dtype and (
            expected_dtype is None or record["dtype"] == expected_dtype
        )
        record.update({"shape_passed": shape_ok, "dtype_passed": dtype_ok})
        if not shape_ok or not dtype_ok:
            record.update({"passed": False, "reason": "shape 或 dtype 不一致"})
            records.append(record)
            passed = False
            continue
        actual_float = actual.to(torch.float32)
        expected_float = expected.to(torch.float32)
        both_negative_inf = (
            (actual_float == float("-inf"))
            & (expected_float == float("-inf"))
            & tolerance["allow_negative_inf"]
        )
        nonfinite = (
            (~torch.isfinite(actual_float) | ~torch.isfinite(expected_float))
            & ~both_negative_inf
        )
        if bool(nonfinite.any()):
            record.update(
                {
                    "passed": False,
                    "reason": "输出包含不允许的 NaN 或 Inf",
                    "has_nan": bool(
                        torch.isnan(actual_float).any()
                        or torch.isnan(expected_float).any()
                    ),
                    "has_inf": bool(
                        torch.isinf(actual_float).any()
                        or torch.isinf(expected_float).any()
                    ),
                }
            )
            records.append(record)
            passed = False
            continue
        finite = ~both_negative_inf
        actual_float = actual_float[finite]
        expected_float = expected_float[finite]
        if actual_float.numel() == 0:
            max_absolute_error = 0.0
            max_relative_error = 0.0
            matched_ratio = 1.0
        else:
            absolute_error = torch.abs(actual_float - expected_float)
            bound = (
                tolerance["max_atol"]
                + tolerance["max_rtol"] * torch.abs(expected_float)
            )
            matched_ratio = float((absolute_error <= bound).float().mean().item())
            max_absolute_error = float(absolute_error.max().item())
            denominator = torch.clamp(
                torch.abs(expected_float),
                min=tolerance["max_atol"] or torch.finfo(torch.float32).tiny,
            )
            max_relative_error = float((absolute_error / denominator).max().item())
        numerical_ok = matched_ratio >= tolerance["required_matched_ratio"]
        cap = tolerance["max_error_cap"]
        if cap is not None and max_absolute_error > cap:
            numerical_ok = False
        record.update(
            {
                "passed": numerical_ok,
                "matched_ratio": matched_ratio,
                "required_matched_ratio": tolerance["required_matched_ratio"],
                "max_absolute_error": max_absolute_error,
                "max_relative_error": max_relative_error,
                "max_error_cap": cap,
            }
        )
        if not numerical_ok:
            record["reason"] = "数值误差超过 workload 容差"
            passed = False
        records.append(record)
    return passed, records


def configure_device(device, torch):
    if not isinstance(device, str) or not device:
        raise BenchError("configuration", "device 必须是非空字符串")
    backend_name = device.split(":", 1)[0]
    if backend_name == "cpu":
        return lambda: None
    backend = getattr(torch, backend_name, None)
    if backend_name == "npu" and backend is None:
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise BenchError(
                "environment_missing", "npu device 需要 torch_npu"
            ) from exc
        backend = getattr(torch, "npu", None)
    if backend is None or not hasattr(backend, "is_available") or not backend.is_available():
        raise BenchError("environment_missing", "{} backend 不可用".format(backend_name))
    try:
        backend.set_device(device)
    except Exception as exc:
        raise BenchError(
            "environment_missing", "无法选择设备 {}: {}".format(device, exc)
        ) from exc
    return backend.synchronize


def measure(function, inputs, warmup, trials, synchronize):
    for _ in range(warmup):
        function(*inputs)
        synchronize()
    durations = []
    for _ in range(trials):
        synchronize()
        started = time.perf_counter()
        function(*inputs)
        synchronize()
        durations.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(durations)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "mean_ms": statistics.mean(durations),
        "median_ms": statistics.median(durations),
        "min_ms": min(durations),
        "p95_ms": ordered[p95_index],
        "std_ms": statistics.pstdev(durations),
        "trials": durations,
        "measurement_source": "host_wall_clock",
    }


def _profiler_csv_total_us(csv_path, columns):
    total_us = 0.0
    try:
        with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                for column in columns:
                    value = row.get(column, "")
                    if value:
                        try:
                            total_us += float(value.strip())
                        except (TypeError, ValueError):
                            pass
                        break
    except OSError as exc:
        raise BenchError(
            "runtime", "无法读取 torch_npu profiler 数据: {}".format(exc)
        ) from exc
    return total_us or None


def parse_profile_kernel_mean_ms(trace_dir, trials):
    """Return mean NPU kernel time from torch_npu profiler text exports."""
    trace_dir = Path(trace_dir)
    step_files = sorted(trace_dir.glob("**/step_trace_time.csv"))
    if step_files:
        total_us = _profiler_csv_total_us(step_files[0], ["Computing"])
        if total_us is not None:
            return total_us / trials / 1000.0, "step_trace_time.Computing"

    kernel_files = sorted(trace_dir.glob("**/kernel_details.csv"))
    if kernel_files:
        total_us = _profiler_csv_total_us(
            kernel_files[0], ["Duration(us)", "Duration"]
        )
        if total_us is not None:
            return total_us / trials / 1000.0, "kernel_details.Duration"

    raise BenchError(
        "runtime", "torch_npu profiler 未生成可用的 kernel 性能数据"
    )


def _anti_hack_failure(reason, declared, names, trials, launches):
    details = {
        "status": "failed",
        "policy": ANTI_HACK_POLICY,
        "declared_kernel_names": sorted(declared),
        "observed_kernel_names": sorted(set(names)),
        "profiled_iterations": trials,
        "observed_launches": launches,
        "launches_per_iteration": launches / trials if trials else None,
        "reason": reason,
    }
    raise BenchError(
        "hack", "candidate anti-hack 校验失败: {}".format(reason), details=details
    )


def parse_profile_anti_hack(trace_dir, trials, declared_kernel_names):
    """Prove that each profiled candidate iteration launched one CATLASS kernel."""
    trace_dir = Path(trace_dir)
    declared = set(declared_kernel_names)
    kernel_files = sorted(trace_dir.glob("**/kernel_details.csv"))
    if len(kernel_files) != 1:
        _anti_hack_failure(
            "kernel_details_file_count", declared, [], trials, 0
        )
    names = []
    try:
        with kernel_files[0].open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required_columns = {"Name", "Type", "OP State"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                _anti_hack_failure(
                    "kernel_details_missing_provenance", declared, [], trials, 0
                )
            for row in reader:
                name = (row.get("Name") or "").strip()
                if not name:
                    _anti_hack_failure(
                        "kernel_details_empty_name", declared, names, trials, len(names)
                    )
                names.append(name)
                kernel_type = (row.get("Type") or "").strip()
                op_state = (row.get("OP State") or "").strip()
                if kernel_type != name or op_state not in {"", "N/A"}:
                    _anti_hack_failure(
                        "kernel_not_catlass_profile",
                        declared,
                        names,
                        trials,
                        len(names),
                    )
    except OSError as exc:
        _anti_hack_failure("kernel_details_unreadable", declared, names, trials, len(names))

    if len(names) != trials:
        _anti_hack_failure(
            "launch_count_mismatch", declared, names, trials, len(names)
        )
    observed = set(names)
    if len(observed) != 1:
        _anti_hack_failure(
            "multiple_kernel_names", declared, names, trials, len(names)
        )
    if not observed.issubset(declared):
        _anti_hack_failure(
            "kernel_not_declared_catlass", declared, names, trials, len(names)
        )
    return {
        "status": "passed",
        "policy": ANTI_HACK_POLICY,
        "declared_kernel_names": sorted(declared),
        "observed_kernel_names": sorted(observed),
        "profiled_iterations": trials,
        "observed_launches": len(names),
        "launches_per_iteration": 1.0,
        "reason": None,
    }


def _merge_profile_csv(iteration_dirs, name, target):
    fieldnames = None
    rows = []
    for iteration_dir in iteration_dirs:
        matches = sorted(Path(iteration_dir).glob("**/{}".format(name)))
        if len(matches) != 1:
            continue
        with matches[0].open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            if fieldnames is None:
                fieldnames = list(reader.fieldnames)
            if list(reader.fieldnames) != fieldnames:
                raise BenchError(
                    "runtime", "profiler iteration CSV 表头不一致: {}".format(name)
                )
            rows.extend(reader)
    if fieldnames is None:
        return None
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def measure_npu(
    function,
    inputs,
    warmup,
    trials,
    synchronize,
    trace_dir,
    declared_kernel_names=None,
):
    """Measure average kernel time using torch_npu profiler text exports."""
    try:
        import torch_npu
    except ImportError as exc:
        raise BenchError(
            "environment_missing", "NPU 性能测量需要 torch_npu profiler"
        ) from exc

    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    )
    # References are outside the candidate policy and retain normal priming.
    # Candidates must never run outside a profiler window: the active trials
    # are their only invocations, and one captured output is reused for the
    # correctness gate by run_suite.
    if declared_kernel_names is None:
        function(*inputs)
        synchronize()
        for _ in range(warmup):
            function(*inputs)
            synchronize()
    iteration_dirs = []
    durations = []
    sources = []
    anti_hack_iterations = []
    last_output = None
    try:
        for iteration in range(trials):
            iteration_dir = trace_dir / "iteration-{:04d}".format(iteration)
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                schedule=torch_npu.profiler.schedule(
                    wait=0,
                    warmup=0,
                    active=1,
                    repeat=1,
                    skip_first=0,
                ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(iteration_dir)
                ),
                profile_memory=False,
                with_modules=False,
                experimental_config=experimental_config,
            ) as profiler:
                last_output = function(*inputs)
                synchronize()
                profiler.step()
            mean_ms, source = parse_profile_kernel_mean_ms(iteration_dir, 1)
            durations.append(mean_ms)
            sources.append(source)
            iteration_dirs.append(iteration_dir)
            if declared_kernel_names is not None:
                anti_hack_iterations.append(
                    parse_profile_anti_hack(
                        iteration_dir, 1, declared_kernel_names
                    )
                )
    except Exception as exc:
        if isinstance(exc, BenchError):
            raise
        raise BenchError(
            "runtime", "torch_npu profiler 性能测量失败: {}".format(exc)
        ) from exc
    _merge_profile_csv(iteration_dirs, "kernel_details.csv", trace_dir / "kernel_details.csv")
    _merge_profile_csv(iteration_dirs, "step_trace_time.csv", trace_dir / "step_trace_time.csv")
    ordered = sorted(durations)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    result = {
        "mean_ms": statistics.mean(durations),
        "median_ms": statistics.median(durations),
        "min_ms": min(durations),
        "p95_ms": ordered[p95_index],
        "std_ms": statistics.pstdev(durations),
        "trials": durations,
        "profiled_iterations": trials,
        "measurement_source": (
            sources[0] if len(set(sources)) == 1 else "per_iteration_profile"
        ),
        "trace_dir": str(trace_dir.resolve()),
    }
    if declared_kernel_names is not None:
        manifest_iterations = []
        for index, (iteration_dir, item) in enumerate(
            zip(iteration_dirs, anti_hack_iterations)
        ):
            matches = sorted(Path(iteration_dir).glob("**/kernel_details.csv"))
            if len(matches) != 1:
                _anti_hack_failure(
                    "kernel_details_file_count",
                    declared_kernel_names,
                    [],
                    1,
                    0,
                )
            snapshot = (
                trace_dir
                / "anti_hack"
                / "iteration-{:04d}".format(index)
                / "kernel_details.csv"
            )
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(matches[0], snapshot)
            manifest_iterations.append(
                {
                    "iteration": index,
                    "kernel_details": str(snapshot.relative_to(trace_dir)),
                    "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                    "observed_kernel_names": item["observed_kernel_names"],
                    "observed_launches": item["observed_launches"],
                }
            )
        manifest = {
            "schema_version": 1,
            "policy": ANTI_HACK_POLICY,
            "profiled_iterations": trials,
            "iterations": manifest_iterations,
        }
        _atomic_write(
            trace_dir / "anti_hack_manifest.json",
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        observed_names = sorted(
            {
                name
                for item in anti_hack_iterations
                for name in item["observed_kernel_names"]
            }
        )
        result["anti_hack"] = {
            "status": "passed",
            "policy": ANTI_HACK_POLICY,
            "declared_kernel_names": sorted(declared_kernel_names),
            "observed_kernel_names": observed_names,
            "profiled_iterations": trials,
            "observed_launches": trials,
            "launches_per_iteration": 1.0,
            "reason": None,
        }
        result["_last_output"] = last_output
    return result


def _git_head(path):
    result = subprocess.run(
        ["git", "-C", str(Path(path).parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _device_name(device, torch):
    backend_name = device.split(":", 1)[0]
    if backend_name == "cpu":
        return platform.processor() or platform.machine()
    backend = getattr(torch, backend_name, None)
    getter = getattr(backend, "get_device_name", None)
    if callable(getter):
        try:
            return getter(device)
        except TypeError:
            return getter()
        except Exception:
            pass
    return device


def _atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".bench.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _reference_profile_cache_root(output, definition, requested=None):
    """Return the shared reference-profile root, outside an individual run."""
    if requested is not None:
        return Path(requested).resolve()
    resolved_output = Path(output).resolve()
    for parent in (resolved_output, *resolved_output.parents):
        if parent.name == ".catlass-dsl":
            return parent / "profiles" / "reference"
    return Path(definition).resolve().parent / ".catlass-dsl" / "profiles" / "reference"


def _reference_profile_key(definition, workload, environment):
    identity = {
        "definition_sha256": definition["_sha256"],
        "workload_sha256": workload,
        "device": environment.get("device"),
        "arch": environment.get("arch"),
        "seed": environment.get("seed"),
        "warmup": environment.get("warmup"),
        "trials": environment.get("trials"),
        "torch": environment.get("torch"),
        "catlass": environment.get("catlass"),
        "cann": environment.get("cann"),
        "measurement": "torch_npu.profiler-level1-text-per-iteration-v2",
    }
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), identity


def _load_reference_profile_cache(entry, cache_key, workload_ids):
    manifest_path = Path(entry) / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "catlass.dsl.reference-profile.v1":
            raise ValueError("schema 不匹配")
        if manifest.get("cache_key") != cache_key:
            raise ValueError("cache key 不匹配")
        timings = manifest["workloads"]
        if set(timings) != set(workload_ids):
            raise ValueError("workload 集合不匹配")
        for workload_id in workload_ids:
            timing = timings[workload_id]
            if timing.get("mean_ms", 0) <= 0:
                raise ValueError("缺少有效 mean_ms")
        return manifest
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise BenchError(
            "configuration", "reference profile cache 已损坏: {}".format(exc)
        ) from exc


def _persist_reference_profile_cache(entry, cache_key, identity, timings):
    manifest_path = Path(entry) / "manifest.json"
    manifest = {
        "schema": "catlass.dsl.reference-profile.v1",
        "cache_key": cache_key,
        "identity": identity,
        "workloads": timings,
    }
    _atomic_write(
        manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest_path


def _aggregate_performance(records):
    passed = [
        record["performance"]
        for record in records
        if record["performance"]["status"] == "passed"
    ]
    if not passed:
        return {"status": "not_run"}
    candidate_means = [item["candidate"]["mean_ms"] for item in passed]
    reference_means = [item["reference"]["mean_ms"] for item in passed]
    candidate_stds = [item["candidate"]["std_ms"] for item in passed]
    reference_stds = [item["reference"]["std_ms"] for item in passed]
    candidate_mean = statistics.mean(candidate_means)
    reference_mean = statistics.mean(reference_means)
    return {
        "status": "passed" if len(passed) == len(records) else "partial",
        "candidate": {
            "mean_ms": candidate_mean,
            "std_ms": statistics.mean(candidate_stds),
            "workload_mean_ms": candidate_means,
            "workload_std_ms": candidate_stds,
        },
        "reference": {
            "mean_ms": reference_mean,
            "std_ms": statistics.mean(reference_stds),
            "workload_mean_ms": reference_means,
            "workload_std_ms": reference_stds,
        },
        "speedup": reference_mean / candidate_mean if candidate_mean > 0 else None,
    }


def _report(result):
    lines = [
        "# CATLASS DSL Benchmark",
        "",
        "- Status: `{}`".format(result["status"]),
        "- Definition: `{}`".format(result.get("definition", {}).get("name")),
        "- Solution: `{}`".format(result.get("solution", {}).get("name")),
        "- Correctness: `{}/{}`".format(
            result["correctness"]["passed"], result["correctness"]["total"]
        ),
        "- Anti-hack: `{}`".format(
            result.get("anti_hack", {}).get("status", "not_run")
        ),
    ]
    performance = result.get("performance", {})
    if performance.get("status") in {"passed", "partial"}:
        lines.extend(
            [
                "- Candidate mean: `{:.6f} ms`".format(
                    performance["candidate"]["mean_ms"]
                ),
                "- Reference mean: `{:.6f} ms`".format(
                    performance["reference"]["mean_ms"]
                ),
                "- Speedup: `{:.4f}x`".format(performance["speedup"]),
            ]
        )
    reference_profile = result.get("profiling", {}).get("reference", {})
    if reference_profile:
        lines.extend(
            [
                "- Reference profile: `{}`".format(
                    reference_profile.get("cache_status")
                ),
                "- Reference profile manifest: `{}`".format(
                    reference_profile.get("manifest")
                ),
            ]
        )
    lines.extend(["", "## Workloads", ""])
    for record in result.get("workloads", []):
        lines.append(
            "- `{}`: `{}`; anti-hack=`{}`".format(
                record["uuid"],
                record["status"],
                record.get("anti_hack", {}).get("status", "not_run"),
            )
        )
    if result.get("error"):
        lines.extend(
            [
                "",
                "## Error",
                "",
                "- Category: `{}`".format(result["error"]["category"]),
                "- Message: {}".format(result["error"]["message"]),
            ]
        )
    return "\n".join(lines) + "\n"


def _persist(result, output):
    output = Path(output)
    result_path = output / "result.json"
    report_path = output / "report.md"
    result.setdefault("artifacts", {}).update(
        {
            "result_json": str(result_path.resolve()),
            "report_md": str(report_path.resolve()),
        }
    )
    _atomic_write(
        result_path, json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    _atomic_write(report_path, _report(result))


def run_suite(
    solution,
    workload,
    definition,
    output,
    *,
    device="cpu",
    seed=1,
    warmup=DEFAULT_WARMUP,
    trials=DEFAULT_TRIALS,
    reference_profile_cache=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 3,
        "status": "failed",
        "definition": {},
        "solution": {},
        "workload": {},
        "environment": {
            "device": device,
            "seed": seed,
            "warmup": warmup,
            "trials": trials,
            "commit": _git_head(solution),
            "config": str(Path(workload).resolve()),
            "arch": None,
            "catlass": os.environ.get("CATLASS_VERSION"),
            "cann": os.environ.get("ASCEND_TOOLKIT_VERSION"),
            "python": platform.python_version(),
        },
        "correctness": {"status": "failed", "passed": 0, "total": 0},
        "performance": {"status": "not_run"},
        "profiling": {"status": "not_run"},
        "anti_hack": _not_applicable_anti_hack(),
        "workloads": [],
        "artifacts": {},
    }
    try:
        if type(seed) is not int:
            raise BenchError("configuration", "seed 必须是整数")
        if type(warmup) is not int or warmup < 0:
            raise BenchError("configuration", "warmup 必须是非负整数")
        if type(trials) is not int or trials < 1:
            raise BenchError("configuration", "trials 必须是正整数")
        import torch

        synchronize = configure_device(device, torch)
        result["environment"].update(
            {
                "arch": _device_name(device, torch),
                "torch": torch.__version__,
            }
        )
        definition_data = load_definition(definition)
        solution_data = load_solution(solution)
        workloads, workload_path, workload_sha = load_workloads(workload)
        is_npu = device.split(":", 1)[0] == "npu"
        source_anti_hack = None
        if is_npu:
            source_anti_hack = audit_solution_sources(solution_data)
            result["anti_hack"] = source_anti_hack
            if source_anti_hack["status"] != "passed":
                raise BenchError(
                    "hack",
                    "candidate source anti-hack 校验失败: {}".format(
                        source_anti_hack["reason"]
                    ),
                    details=source_anti_hack,
                )
        if solution_data["definition"] != definition_data["name"]:
            raise BenchError(
                "configuration", "solution.definition 与 definition.name 不一致"
            )
        reference_module, reference_run, reference_path = load_reference(
            definition_data, output
        )
        solution_run, sources_root = stage_solution(solution_data, output)
        result.update(
            {
                "definition": {
                    "name": definition_data["name"],
                    "path": definition_data["_path"],
                    "sha256": definition_data["_sha256"],
                },
                "solution": {
                    "name": solution_data["name"],
                    "path": solution_data["_path"],
                    "sha256": solution_data["_sha256"],
                    "source_sha256": (
                        hashlib.sha256(
                            solution_data["sources"][0]["content"].encode("utf-8")
                        ).hexdigest()
                        if len(solution_data["sources"]) == 1
                        else None
                    ),
                    "author": solution_data["author"],
                },
                "workload": {"path": workload_path, "sha256": workload_sha},
            }
        )
        result["artifacts"].update(
            {
                "sources": str(sources_root.resolve()),
                "reference": str(reference_path.resolve()),
            }
        )
        workload_root = Path(workload_path).parent
        profile_root = output / "profiling"
        reference_cache = None
        reference_cache_entry = None
        reference_cache_key = None
        reference_cache_identity = None
        reference_timings = {}
        if is_npu:
            profile_root.mkdir(parents=True, exist_ok=True)
            result["artifacts"]["profiling"] = str(profile_root.resolve())
            reference_cache_root = _reference_profile_cache_root(
                output, definition, reference_profile_cache
            )
            reference_cache_key, reference_cache_identity = _reference_profile_key(
                definition_data, workload_sha, result["environment"]
            )
            reference_cache_entry = reference_cache_root / reference_cache_key
            reference_cache = _load_reference_profile_cache(
                reference_cache_entry,
                reference_cache_key,
                [item["uuid"] for item in workloads],
            )
            if reference_cache is not None:
                reference_timings.update(reference_cache["workloads"])
            result["artifacts"]["reference_profile"] = str(
                reference_cache_entry.resolve()
            )
        for workload_index, workload_data in enumerate(workloads):
            record = {
                "uuid": workload_data["uuid"],
                "line": workload_data["_line"],
                "status": "failed",
                "axes": {},
                "correctness": {"status": "failed", "outputs": []},
                "performance": {"status": "not_run"},
                "anti_hack": (
                    {
                        "status": "failed",
                        "policy": ANTI_HACK_POLICY,
                        "declared_kernel_names": source_anti_hack[
                            "declared_kernel_names"
                        ],
                        "observed_kernel_names": [],
                        "profiled_iterations": 0,
                        "observed_launches": 0,
                        "launches_per_iteration": None,
                        "reason": "not_run",
                    }
                    if is_npu
                    else _not_applicable_anti_hack()
                ),
            }
            try:
                inputs, axes = generate_inputs(
                    definition_data,
                    workload_data,
                    reference_module,
                    device,
                    seed,
                    workload_root,
                )
                record["axes"] = axes
                reference_inputs = [clone_value(item, torch) for item in inputs]
                candidate_inputs = [clone_value(item, torch) for item in inputs]
                reference_raw = reference_run(*reference_inputs)
                synchronize()
                reference_outputs = normalize_outputs(
                    reference_raw, definition_data["outputs"], torch
                )
                candidate_timing = None
                if is_npu:
                    case_profile_dir = profile_root / "case-{:04d}".format(
                        workload_index
                    )
                    if solution_data["spec"]["destination_passing_style"]:
                        candidate_buffers = allocate_outputs(
                            definition_data, axes, candidate_inputs, torch
                        )

                        def timed_solution(*args):
                            return solution_run(*args, *candidate_buffers)

                    else:
                        timed_solution = solution_run
                    candidate_timing = measure_npu(
                        timed_solution,
                        candidate_inputs,
                        warmup,
                        trials,
                        synchronize,
                        case_profile_dir / "candidate",
                        declared_kernel_names=source_anti_hack[
                            "declared_kernel_names"
                        ],
                    )
                    record["anti_hack"] = candidate_timing.pop("anti_hack")
                    captured_output = candidate_timing.pop("_last_output")
                    candidate_raw = (
                        candidate_buffers
                        if solution_data["spec"]["destination_passing_style"]
                        else captured_output
                    )
                elif solution_data["spec"]["destination_passing_style"]:
                    candidate_buffers = allocate_outputs(
                        definition_data, axes, candidate_inputs, torch
                    )
                    solution_run(*candidate_inputs, *candidate_buffers)
                    synchronize()
                    candidate_raw = candidate_buffers
                else:
                    candidate_raw = solution_run(*candidate_inputs)
                    synchronize()
                candidate_outputs = normalize_outputs(
                    candidate_raw, definition_data["outputs"], torch
                )
                correct, output_records = compare_outputs(
                    candidate_outputs,
                    reference_outputs,
                    definition_data,
                    axes,
                    workload_data["tolerance"],
                    torch,
                )
                record["correctness"] = {
                    "status": "passed" if correct else "failed",
                    "outputs": output_records,
                }
                if not correct:
                    record["error"] = {
                        "category": "incorrect",
                        "message": "solution 未通过正确性门禁",
                    }
                    result["workloads"].append(record)
                    continue

                reference_timing_inputs = [
                    clone_value(item, torch) for item in inputs
                ]
                if is_npu:
                    if workload_data["uuid"] in reference_timings:
                        reference_timing = dict(
                            reference_timings[workload_data["uuid"]]
                        )
                        reference_timing["cache_status"] = "reused"
                    else:
                        reference_timing = measure_npu(
                            reference_run,
                            reference_timing_inputs,
                            warmup,
                            trials,
                            synchronize,
                            reference_cache_entry
                            / "profiling"
                            / "case-{:04d}".format(workload_index),
                        )
                        reference_timing["cache_status"] = "collected"
                        reference_timings[workload_data["uuid"]] = dict(
                            reference_timing
                        )
                else:
                    timing_inputs = [clone_value(item, torch) for item in inputs]
                    if solution_data["spec"]["destination_passing_style"]:
                        timing_outputs = allocate_outputs(
                            definition_data, axes, timing_inputs, torch
                        )

                        def timed_solution(*args):
                            return solution_run(*args, *timing_outputs)

                    else:
                        timed_solution = solution_run
                    candidate_timing = measure(
                        timed_solution, timing_inputs, warmup, trials, synchronize
                    )
                    reference_timing = measure(
                        reference_run,
                        reference_timing_inputs,
                        warmup,
                        trials,
                        synchronize,
                    )
                record.update(
                    {
                        "status": "passed",
                        "performance": {
                            "status": "passed",
                            "candidate": candidate_timing,
                            "reference": reference_timing,
                            "speedup": (
                                reference_timing["mean_ms"]
                                / candidate_timing["mean_ms"]
                                if candidate_timing["mean_ms"] > 0
                                else None
                            ),
                        },
                    }
                )
            except Exception as exc:
                if getattr(exc, "category", None) == "hack" and getattr(
                    exc, "details", None
                ):
                    record["anti_hack"] = exc.details
                record["error"] = {
                    "category": getattr(exc, "category", "runtime"),
                    "message": str(exc),
                    "type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                }
            result["workloads"].append(record)

        passed = sum(
            item["correctness"]["status"] == "passed"
            for item in result["workloads"]
        )
        total = len(result["workloads"])
        result["correctness"] = {
            "status": "passed" if passed == total else "failed",
            "passed": passed,
            "total": total,
        }
        result["performance"] = _aggregate_performance(result["workloads"])
        if is_npu:
            profiled = sum(
                item["performance"]["status"] == "passed"
                for item in result["workloads"]
            )
            result["profiling"] = {
                "status": (
                    "passed"
                    if profiled == total
                    else "partial" if profiled else "not_run"
                ),
                "profiled": profiled,
                "total": total,
                "artifact": str(profile_root.resolve()),
                "candidate": {"artifact": str(profile_root.resolve())},
                "reference": {
                    "cache_status": (
                        "reused"
                        if reference_cache
                        else "collected" if profiled == total else "incomplete"
                    ),
                    "cache_key": reference_cache_key,
                    "artifact": str(reference_cache_entry.resolve()),
                    "manifest": str(
                        (reference_cache_entry / "manifest.json").resolve()
                    ),
                },
            }
            if profiled == total and reference_cache is None:
                manifest_path = _persist_reference_profile_cache(
                    reference_cache_entry,
                    reference_cache_key,
                    reference_cache_identity,
                    reference_timings,
                )
                result["profiling"]["reference"]["manifest"] = str(manifest_path.resolve())
            anti_hack_records = [item["anti_hack"] for item in result["workloads"]]
            anti_hack_passed = (
                len(anti_hack_records) == total
                and all(item.get("status") == "passed" for item in anti_hack_records)
            )
            observed_names = sorted(
                {
                    name
                    for item in anti_hack_records
                    for name in item.get("observed_kernel_names", [])
                }
            )
            observed_launches = sum(
                item.get("observed_launches", 0) for item in anti_hack_records
            )
            profiled_iterations = sum(
                item.get("profiled_iterations", 0) for item in anti_hack_records
            )
            failed_reason = next(
                (
                    item.get("reason")
                    for item in anti_hack_records
                    if item.get("status") != "passed"
                ),
                None,
            )
            result["anti_hack"] = {
                "status": "passed" if anti_hack_passed else "failed",
                "policy": ANTI_HACK_POLICY,
                "declared_kernel_names": source_anti_hack[
                    "declared_kernel_names"
                ],
                "observed_kernel_names": observed_names,
                "profiled_iterations": profiled_iterations,
                "observed_launches": observed_launches,
                "launches_per_iteration": (
                    observed_launches / profiled_iterations
                    if profiled_iterations
                    else None
                ),
                "reason": failed_reason,
                "source_violations": source_anti_hack["source_violations"],
            }
            if not anti_hack_passed:
                for record in result["workloads"]:
                    record["performance"] = {
                        "status": "not_run",
                        "reason": "suite_anti_hack_failed",
                    }
                result["performance"] = {
                    "status": "not_run",
                    "reason": "suite_anti_hack_failed",
                }
        result["status"] = (
            "passed"
            if all(item["status"] == "passed" for item in result["workloads"])
            else "failed"
        )
    except Exception as exc:
        result["error"] = {
            "category": getattr(exc, "category", "runtime"),
            "message": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
    _persist(result, output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="以 solution、workload 和 definition 运行 CATLASS DSL benchmark",
        epilog="三个输入都会触发用户信任的 Python 代码执行；本工具不是安全沙箱。",
    )
    parser.add_argument("--solution", required=True, type=Path)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--definition", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--reference-profile-cache",
        type=Path,
        help=(
            "共享 reference profile 根目录；默认使用当前项目 "
            ".catlass-dsl/profiles/reference"
        ),
    )
    args = parser.parse_args(argv)
    result = run_suite(
        args.solution,
        args.workload,
        args.definition,
        args.output,
        device=args.device,
        seed=args.seed,
        warmup=args.warmup,
        trials=args.trials,
        reference_profile_cache=args.reference_profile_cache,
    )
    if result["status"] == "passed":
        return 0
    if result.get("error", {}).get("category") == "environment_missing" or any(
        item.get("error", {}).get("category") == "environment_missing"
        for item in result.get("workloads", [])
    ):
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
