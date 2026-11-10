#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared helpers for ops-perf-routing-plan.

The module is intentionally dependency-free.  It provides:
  * safe numeric expression evaluation for golden flow / tiling / route predicates
  * case normalization
  * hardware/profile loading
  * performance.json normalization
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional


class RouteError(Exception):
    """Fatal, user-facing route failure."""


logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
LOGGER = logging.getLogger("routing_plan")


def emit(message: object = "") -> None:
    """Write one CLI result line to stdout."""
    os.write(1, (str(message) + "\n").encode("utf-8"))


def _excepthook(exc_type, exc, traceback) -> None:
    if exc_type is RouteError:
        LOGGER.error("[routing-plan][ERROR] %s", exc)
        return
    _DEFAULT_EXCEPTHOOK(exc_type, exc, traceback)


_DEFAULT_EXCEPTHOOK = sys.excepthook
sys.excepthook = _excepthook


def die(msg: str) -> None:
    raise RouteError(msg)


def warn(msg: str) -> None:
    LOGGER.warning(f"[routing-plan][WARN] {msg}")


def load_json(path: str, default: Any = None) -> Any:
    if not path or not os.path.isfile(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        die(f"无法解析 JSON {path}: {exc}")
    return default


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def rel(path: str, base: str) -> str:
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return path


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------
DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int32": 4,
    "uint32": 4,
    "float32": 4,
    "fp32": 4,
    "float": 4,
    "int64": 8,
    "uint64": 8,
    "float64": 8,
    "fp64": 8,
}


def load_text(path: str, default: str = "") -> str:
    if not path or not os.path.isfile(path):
        return default
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        die(f"无法读取文件 {path}: {exc}")
    return default


def dtype_bytes(dtype: Any) -> int:
    s = str(dtype or "float32").strip().lower()
    s = s.replace("torch.", "")
    if s in DTYPE_BYTES:
        return DTYPE_BYTES[s]
    if s.endswith("_t"):
        s = s[:-2]
    if s in DTYPE_BYTES:
        return DTYPE_BYTES[s]
    # Last-resort alias used by several CANN case files.
    alias = {
        "bf16": 2,
        "bfloat16": 2,
        "half": 2,
        "float16": 2,
        "float": 4,
        "fp32": 4,
        "float32": 4,
        "double": 8,
        "fp64": 8,
    }
    if s in alias:
        return alias[s]
    die(f"不支持的 dtype: {dtype}")
    return 1


def canonical_dtype(dtype: Any) -> str:
    s = str(dtype or "float32").strip().lower().replace("torch.", "")
    return {
        "half": "float16",
        "float": "float32",
        "double": "float64",
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
    }.get(s, s)


# ---------------------------------------------------------------------------
# safe expression evaluator
# ---------------------------------------------------------------------------
class ExprError(ValueError):
    pass


_ALLOWED_FUNCS = {
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "min": min,
    "max": max,
    "round": round,
    "sum": sum,
    "len": len,
    "prod": math.prod,
    "int": int,
    "float": float,
}


_CMP_FUNCS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_BIN_FUNCS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}


def _eval_constant(node):
    if isinstance(node.value, (int, float, bool)):
        return node.value
    raise ExprError("只允许数值常量")


def _eval_name(node, ns):
    if node.id in ("true", "false"):
        return node.id == "true"
    if node.id not in ns:
        raise ExprError(f"未知变量: {node.id}")
    return ns[node.id]


def _eval_unary(node, ns):
    value = _safe_eval_node(node.operand, ns)
    if isinstance(node.op, ast.UAdd):
        return +value
    if isinstance(node.op, ast.USub):
        return -value
    if isinstance(node.op, ast.Not):
        return not value
    raise ExprError("不支持的一元运算")


def _eval_binop(node, ns):
    left = _safe_eval_node(node.left, ns)
    right = _safe_eval_node(node.right, ns)
    func = _BIN_FUNCS.get(type(node.op))
    if func is None:
        raise ExprError("不支持的二元运算")
    if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
        raise ExprError("除零")
    if isinstance(node.op, ast.Pow) and (
        not isinstance(right, (int, float)) or abs(right) > 64
    ):
        raise ExprError(f"幂指数超出范围: {right!r}")
    return func(left, right)


def _eval_boolop(node, ns):
    if isinstance(node.op, ast.And):
        for operand in node.values:
            if not _safe_eval_node(operand, ns):
                return False
        return True
    if isinstance(node.op, ast.Or):
        for operand in node.values:
            if _safe_eval_node(operand, ns):
                return True
        return False
    raise ExprError("不支持的布尔运算")


def _eval_compare(node, ns):
    left = _safe_eval_node(node.left, ns)
    for op, comparator in zip(node.ops, node.comparators):
        right = _safe_eval_node(comparator, ns)
        func = _CMP_FUNCS.get(type(op))
        if func is None:
            raise ExprError("不支持的比较运算")
        if not func(left, right):
            return False
        left = right
    return True


def _eval_call(node, ns):
    if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
        raise ExprError("不支持的函数调用")
    if node.keywords:
        raise ExprError("不支持关键字参数")
    args = []
    for arg in node.args:
        args.append(_safe_eval_node(arg, ns))
    return _ALLOWED_FUNCS[node.func.id](*args)


def _eval_container(node, ns):
    if isinstance(node, ast.List):
        values = []
        for element in node.elts:
            values.append(_safe_eval_node(element, ns))
        return values
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_node(e, ns) for e in node.elts)
    return {
        _safe_eval_node(k, ns): _safe_eval_node(v, ns)
        for k, v in zip(node.keys, node.values)
    }


def _eval_subscript(node, ns):
    value = _safe_eval_node(node.value, ns)
    index = _safe_eval_node(node.slice, ns)
    try:
        return value[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise ExprError(f"下标访问失败: {exc}") from exc


def _safe_eval_node(node: ast.AST, ns: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body, ns)
    if isinstance(node, ast.Constant):
        return _eval_constant(node)
    if isinstance(node, ast.Name):
        return _eval_name(node, ns)
    if isinstance(node, ast.UnaryOp):
        return _eval_unary(node, ns)
    if isinstance(node, ast.BinOp):
        return _eval_binop(node, ns)
    if isinstance(node, ast.BoolOp):
        return _eval_boolop(node, ns)
    if isinstance(node, ast.Compare):
        return _eval_compare(node, ns)
    if isinstance(node, ast.IfExp):
        branch = node.body if _safe_eval_node(node.test, ns) else node.orelse
        return _safe_eval_node(branch, ns)
    if isinstance(node, ast.Call):
        return _eval_call(node, ns)
    if isinstance(node, (ast.List, ast.Tuple, ast.Dict)):
        return _eval_container(node, ns)
    if isinstance(node, ast.Subscript):
        return _eval_subscript(node, ns)
    raise ExprError(f"不支持的表达式节点: {type(node).__name__}")


def eval_expr(expr: Any, variables: Dict[str, Any]) -> Any:
    """Evaluate a numeric expression.  Numeric values pass through."""
    if expr is None:
        return 0
    if isinstance(expr, (int, float, bool)):
        return expr
    if isinstance(expr, (list, tuple, dict)):
        return expr
    if not isinstance(expr, str):
        raise ExprError(f"不支持的表达式类型: {type(expr).__name__}")
    text = expr.strip()
    if not text:
        return 0
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"表达式语法错误 {expr!r}: {exc}") from exc
    return _safe_eval_node(tree, variables)


def eval_number(expr: Any, variables: Dict[str, Any], default: float = 0.0) -> float:
    v = eval_expr(expr, variables)
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    raise ExprError(f"表达式不是数值: {expr!r} -> {type(v).__name__}")


def eval_derived(derived: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(variables)
    for name, expr in (derived or {}).items():
        try:
            out[name] = eval_expr(expr, out)
        except ExprError as exc:
            raise ExprError(f"derived[{name}] 求值失败: {exc}") from exc
    return out


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def _json_or_value(v: Any, default: Any = None) -> Any:
    if v is None or v == "":
        return default
    if isinstance(v, (list, tuple, dict, int, float, bool)):
        return v
    try:
        return json.loads(str(v))
    except Exception:
        return default


def _shape_list(value: Any) -> List[List[int]]:
    v = _json_or_value(value, [])
    if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
        shapes = []
        for shape in v:
            shapes.append(list(map(int, shape)))
        return shapes
    if isinstance(v, list) and v and isinstance(v[0], (int, float)):
        return [list(map(int, v))]
    if isinstance(v, tuple):
        return [list(map(int, v))]
    return []


def _dtype_list(value: Any) -> List[str]:
    v = _json_or_value(value, None)
    if isinstance(v, list):
        return list(map(str, v))
    if isinstance(v, str):
        return [v]
    return []


def _shapes_from_inputs(inputs: Any) -> List[List[int]]:
    shapes = []
    if not isinstance(inputs, list):
        return shapes
    for item in inputs:
        shape = item.get("shape") if isinstance(item, dict) else item
        try:
            shapes.append(list(map(int, shape or [])))
        except (TypeError, ValueError):
            continue
    return shapes


def _dtypes_from_inputs(inputs: Any) -> List[str]:
    if not isinstance(inputs, list):
        return []
    dtypes = []
    for item in inputs:
        if isinstance(item, dict) and item.get("dtype"):
            dtypes.append(str(item["dtype"]))
    return dtypes


def normalize_case(
    raw: Dict[str, Any], index: int = 0, operator: str = ""
) -> Dict[str, Any]:
    cid = raw.get("case_id", raw.get("case", raw.get("id", f"case_{index:03d}")))
    shapes = _shape_list(raw.get("input_shape", raw.get("shape", raw.get("dims"))))
    inputs = raw.get("inputs")
    if not shapes:
        shapes = _shapes_from_inputs(inputs)
    dtypes = _dtype_list(raw.get("dtype", raw.get("dtypes")))
    if not dtypes:
        dtypes = _dtypes_from_inputs(inputs)
    attrs = _json_or_value(raw.get("attrs"), {})
    if not isinstance(attrs, dict):
        attrs = {}
    return {
        "operator": raw.get("operator", operator),
        "case_id": str(cid),
        "input_shape": shapes,
        "dtype": dtypes,
        "attrs": attrs,
        "value_range": raw.get("value_range", ""),
        "note": raw.get("note", ""),
    }


def _read_csv_cases(path: str, operator: str) -> List[Dict[str, Any]]:
    import csv

    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for index, raw in enumerate(csv.DictReader(handle)):
            rows.append(normalize_case(raw, index, operator))
    return rows


def _read_jsonl_cases(path: str, operator: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    for index, line in enumerate(lines):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        rows.append(normalize_case(json.loads(text), index, operator))
    return rows


def _read_json_cases(path: str, operator: str) -> List[Dict[str, Any]]:
    text = load_text(path)
    data = json.loads(text)
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        data = data.get("cases")
    if not isinstance(data, list):
        die(f"cases JSON 必须是数组或 {{'cases': [...]}}: {path}")
    rows = []
    for index, raw in enumerate(data):
        rows.append(normalize_case(raw, index, operator))
    return rows


def read_cases(path: str, operator: str = "") -> List[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        die(f"cases 文件不存在: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        rows = _read_csv_cases(path, operator)
    elif ext in (".jsonl", ".ndjson"):
        rows = _read_jsonl_cases(path, operator)
    else:
        rows = _read_json_cases(path, operator)
    if not rows:
        die(f"cases 为空: {path}")
    ids = []
    for row in rows:
        ids.append(row["case_id"])
    if len(set(ids)) != len(ids):
        die("cases 中存在重复 case_id")
    return rows


def write_cases(path: str, cases: List[Dict[str, Any]]) -> None:
    import csv

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "operator",
        "case_id",
        "input_shape",
        "dtype",
        "attrs",
        "value_range",
        "note",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in cases:
            row = dict(c)
            row["input_shape"] = json.dumps(
                row.get("input_shape", []), ensure_ascii=False
            )
            row["dtype"] = json.dumps(row.get("dtype", []), ensure_ascii=False)
            row["attrs"] = json.dumps(row.get("attrs", {}), ensure_ascii=False)
            w.writerow({k: row.get(k, "") for k in fields})


def _shape_product(shape: Any) -> int:
    product = 1
    for value in shape or []:
        product *= int(value)
    return product


def _input_bytes(shapes: List[List[int]], dtypes: List[str], first_dtype: str) -> int:
    total = 0
    for index, shape in enumerate(shapes):
        dtype = dtypes[index] if index < len(dtypes) else first_dtype
        total += _shape_product(shape) * dtype_bytes(dtype)
    return total


def _apply_hardware(ns: Dict[str, Any], hardware: Dict[str, Any]) -> None:
    for key, value in (hardware or {}).items():
        if isinstance(value, (int, float, bool, str)):
            ns[key] = value
        elif key in (
            "dtype_bytes",
            "vector_elements_per_cycle",
            "vector_instruction_costs",
        ):
            ns[key] = value
    ns.setdefault(
        "aiv_core_num", hardware.get("aiv_core_num", hardware.get("vector_core", 48))
    )
    ns.setdefault(
        "aic_core_num", hardware.get("aic_core_num", hardware.get("cube_core", 24))
    )
    ns.setdefault(
        "frequency_hz", hardware.get("frequency_hz", hardware.get("frequency", 1.8e9))
    )
    ns.setdefault("freq_hz", ns["frequency_hz"])
    ns.setdefault(
        "ub_size", hardware.get("ub_size_bytes", hardware.get("ub_size", 196608))
    )
    ns.setdefault("ub_size_bytes", ns["ub_size"])
    ns.setdefault(
        "l2_capacity_bytes", hardware.get("l2_capacity_bytes", 192 * 1024 * 1024)
    )


def case_variables(case: Dict[str, Any], hardware: Dict[str, Any]) -> Dict[str, Any]:
    shapes = case.get("input_shape") or []
    dtypes = case.get("dtype") or []
    if not isinstance(shapes, list):
        shapes = []
    if isinstance(dtypes, str):
        dtypes = [dtypes]
    if not dtypes:
        dtypes = ["float32"]
    numels = list(map(_shape_product, shapes))
    first_shape = shapes[0] if shapes else []
    first_dtype = dtypes[0] if dtypes else "float32"
    ns: Dict[str, Any] = {
        "dims": shapes,
        "shape": first_shape,
        "shapes": shapes,
        "numel": numels[0] if numels else 0,
        "total_elems": numels[0] if numels else 0,
        "numels": numels,
        "elem_bytes": dtype_bytes(first_dtype),
        "dtype": canonical_dtype(first_dtype),
        "dtypes": list(map(canonical_dtype, dtypes)),
        "input_bytes": _input_bytes(shapes, dtypes, first_dtype),
        "attrs": case.get("attrs") or {},
        "case_id": case.get("case_id", ""),
    }
    _apply_hardware(ns, hardware or {})
    ns["is_fp16"] = canonical_dtype(first_dtype) in ("float16",)
    ns["is_bf16"] = canonical_dtype(first_dtype) in ("bfloat16",)
    ns["is_fp32"] = canonical_dtype(first_dtype) in ("float32",)
    return ns


# ---------------------------------------------------------------------------
# performance.json normalization
# ---------------------------------------------------------------------------
def _norm_case_id(raw: Any) -> str:
    s = str(raw)
    tail = s.rsplit("_", 1)[-1] if "_" in s else s.rsplit("/", 1)[-1]
    return tail if tail.isdigit() else s


def _dict_rows(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _list_field(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("cases", "per_case", "results"):
        rows = _dict_rows(data.get(key))
        if rows:
            return rows
    return []


def _flat_perf_cases(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return _dict_rows(data)
    if not isinstance(data, dict):
        return []
    rows = _list_field(data)
    if rows:
        return rows
    operators = data.get("operators")
    if isinstance(operators, list):
        rows = []
        for operator in operators:
            if isinstance(operator, dict):
                rows.extend(_list_field(operator))
        return rows
    operator = data.get("operator")
    return _flat_perf_cases(operator) if isinstance(operator, dict) else []


def _as_float(v: Any) -> Optional[float]:
    try:
        number = float(v)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pick_float(row: Dict[str, Any], keys: tuple) -> Optional[float]:
    for key in keys:
        if row.get(key) is not None:
            value = _as_float(row.get(key))
            if value is not None:
                return value
    return None


def _perf_row(row: Dict[str, Any]) -> Optional[tuple]:
    raw_id = row.get("case_id", row.get("case", row.get("id", row.get("name", ""))))
    if raw_id is None or raw_id == "":
        return None
    candidate = _pick_float(
        row,
        (
            "t_cand_us",
            "elapsed_us",
            "task_us",
            "asc_us",
            "duration_us",
            "kernel_us",
            "kernel_time_us",
            "time_us",
        ),
    )
    speedup = _pick_float(row, ("speedup", "speedup_ratio", "加速比"))
    baseline = _pick_float(
        row, ("baseline_perf_us", "baseline_us", "ref_us", "t_base_us")
    )
    if candidate is None and baseline is not None and speedup:
        candidate = baseline / speedup
    if candidate is None:
        return None
    return _norm_case_id(raw_id), {
        "t_candidate_us": candidate,
        "speedup": speedup or 0.0,
        "t_baseline_us": baseline or 0.0,
    }


def load_performance(path: str) -> Dict[str, Dict[str, float]]:
    data = load_json(path)
    output: Dict[str, Dict[str, float]] = {}
    for row in _flat_perf_cases(data):
        parsed = _perf_row(row)
        if parsed is not None:
            output[str(parsed[0])] = parsed[1]
    return output


# ---------------------------------------------------------------------------
# paths / names
# ---------------------------------------------------------------------------
def sanitize_name(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "template"
    if s[0].isdigit():
        s = "t_" + s
    return s


def find_case(rows: Iterable[Dict[str, Any]], case_id: str) -> Optional[Dict[str, Any]]:
    for r in rows:
        if str(r.get("case_id")) == str(case_id):
            return r
    return None


def parse_shape_text(s: Any) -> List[int]:
    if isinstance(s, list):
        return list(map(int, s))
    try:
        v = json.loads(str(s))
        return list(map(int, v))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# tiling plan resolution / validation
# ---------------------------------------------------------------------------
def _get_plan(
    template: Dict[str, Any], explicit_plan: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if explicit_plan:
        return explicit_plan
    plan = template.get("tiling_plan")
    if isinstance(plan, dict) and plan:
        return plan
    # Degrade tiling_intent to a minimal plan.  This is intentionally explicit:
    # the caller sees tiling_source="intent_defaults" in the validation result.
    intent = template.get("tiling_intent") or {}
    double_buffer = int(intent.get("double_buffer", 2) or 2)
    # ub_bytes_per_elem already counts input+output; divide by double_buffer again
    # so the resulting plan stays within 0.8 * UB after buffering.
    default_tile = (
        "min(total_elems, max(1, floor(ub_size * 0.8 / "
        f"({2 * max(1, double_buffer)} * elem_bytes))))"
    )
    return {
        "core_split": {
            "units": "total_elems",
            "num_cores": intent.get(
                "num_cores", "min(aiv_core_num, ceil(total_elems / 1024))"
            ),
        },
        "tile": {
            "axis": intent.get("split_axis", "flat"),
            "elems_per_tile": intent.get("elems_per_tile", default_tile),
            "double_buffer": double_buffer,
            "ub_bytes_per_elem": intent.get("ub_bytes_per_elem", "2 * elem_bytes"),
        },
        "tail_policy": intent.get("tail_policy", "mask"),
        "alignment": {"ub_block_bytes": 32, "require_inner_align": False},
        "tiling_source": "intent_defaults",
    }


def _resolve_numbers(core_cfg, tile_cfg, ns):
    try:
        num_cores = int(
            round(eval_number(core_cfg.get("num_cores", "aiv_core_num"), ns))
        )
        tile_elems = int(
            round(eval_number(tile_cfg.get("elems_per_tile", "total_elems"), ns))
        )
    except ExprError as exc:
        raise ValueError(f"tiling 表达式求值失败: {exc}") from exc
    return num_cores, tile_elems


def _core_errors(num_cores, tile_elems, cores):
    errors = []
    if num_cores < 1:
        errors.append(f"num_cores={num_cores} 必须 >= 1")
    if num_cores > int(cores):
        errors.append(f"num_cores={num_cores} 超过硬件核数 {int(cores)}")
    if tile_elems < 1:
        errors.append(f"tile_elems={tile_elems} 必须 >= 1")
    return errors


def _ub_metrics(tile_cfg, ns, ub_size, tile_elems, errors, warnings):
    ub_budget = ub_size * 0.8 if ub_size > 0 else None
    expression = tile_cfg.get("ub_bytes_per_elem")
    if expression is None:
        warnings.append("未提供 ub_bytes_per_elem，跳过 UB 容量校验")
        return None, ub_budget
    try:
        per_elem = eval_number(expression, ns)
    except ExprError as exc:
        raise ValueError(f"ub_bytes_per_elem 求值失败: {exc}") from exc
    double_buffer = int(tile_cfg.get("double_buffer", 2) or 1)
    ub_used = tile_elems * per_elem * max(1, double_buffer)
    if ub_budget is not None and ub_used > ub_budget:
        errors.append(f"UB 超限: used={ub_used:.0f} > budget={ub_budget:.0f}")
    return ub_used, ub_budget


def _alignment(case, ns, plan, errors):
    align_bytes = int((plan.get("alignment") or {}).get("ub_block_bytes", 32) or 32)
    require_align = bool(
        (plan.get("alignment") or {}).get("require_inner_align", False)
    )
    shapes = case.get("input_shape") or []
    inner = shapes[0][-1] if shapes and shapes[0] else 0
    inner_bytes = int(inner) * int(ns.get("elem_bytes", 1) or 1)
    align_ok = inner_bytes % align_bytes == 0
    if require_align and not align_ok:
        errors.append(
            f"inner dim {inner} * {ns.get('elem_bytes')}B 未按 {align_bytes}B 对齐"
        )
    return inner_bytes, align_ok, require_align


_TAIL_POLICIES = {"mask", "pad", "handle", "skip", "dynamic"}
_UNHANDLED_TAIL_POLICIES = {"", "none", "forbid", "forbidden", "error"}


def _tail_check(ns, plan, tile_elems, errors, warnings):
    total_elems = int(ns.get("total_elems", 0) or 0)
    policy = str(plan.get("tail_policy", "mask") or "mask").strip().lower()
    tail_elems = total_elems % tile_elems if tile_elems > 0 else total_elems
    if tail_elems > 0:
        if policy in _UNHANDLED_TAIL_POLICIES:
            errors.append(
                f"存在尾块 tail_elems={tail_elems}，tail_policy={policy!r} 未处理尾块"
            )
        elif policy not in _TAIL_POLICIES:
            warnings.append(
                f"未知 tail_policy={policy!r}，请确认可处理 tail_elems={tail_elems}"
            )
    return tail_elems, policy


def resolve_tiling(
    case: Dict[str, Any],
    template: Dict[str, Any],
    hardware: Dict[str, Any],
    explicit_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    plan = _get_plan(template, explicit_plan)
    ns = eval_derived(template.get("derived") or {}, case_variables(case, hardware))
    cores = float(ns.get("aiv_core_num", 1) or 1)
    ub_size = float(ns.get("ub_size", ns.get("ub_size_bytes", 0)) or 0)
    core_cfg = plan.get("core_split") or {}
    tile_cfg = plan.get("tile") or {}
    num_cores, tile_elems = _resolve_numbers(core_cfg, tile_cfg, ns)
    errors = _core_errors(num_cores, tile_elems, cores)
    warnings: List[str] = []
    ub_used, ub_budget = _ub_metrics(
        tile_cfg, ns, ub_size, tile_elems, errors, warnings
    )
    inner_bytes, align_ok, require_align = _alignment(case, ns, plan, errors)
    tail_elems, tail_policy = _tail_check(ns, plan, tile_elems, errors, warnings)
    return {
        "case_id": str(case.get("case_id")),
        "template": template.get("name", ""),
        "tiling_key": template.get("tiling_key", 0),
        "num_cores": num_cores,
        "tile_elems": tile_elems,
        "tail_elems": tail_elems,
        "double_buffer": int(tile_cfg.get("double_buffer", 2) or 1),
        "tail_policy": tail_policy,
        "tiling_source": plan.get("tiling_source", "tiling_plan"),
        "ub_used_bytes": round(ub_used, 3) if ub_used is not None else None,
        "ub_budget_bytes": round(ub_budget, 3) if ub_budget is not None else None,
        "inner_bytes": inner_bytes,
        "align_ok": align_ok,
        "required_inner_align": require_align,
        "warnings": warnings,
        "errors": errors,
        "ok": not errors,
    }


def validate_template(
    template: Dict[str, Any],
    cases: List[Dict[str, Any]],
    hardware: Dict[str, Any],
    explicit_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rows = []
    for case in cases:
        rows.append(resolve_tiling(case, template, hardware, explicit_plan))
    errors = []
    for row in rows:
        for message in row["errors"]:
            errors.append(f"{row['case_id']}: {message}")
    return {
        "schema_version": 1,
        "template": template.get("name", ""),
        "n_cases": len(rows),
        "ok": not errors,
        "errors": errors,
        "cases": rows,
    }
