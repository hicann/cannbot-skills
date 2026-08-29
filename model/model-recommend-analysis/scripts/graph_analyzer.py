#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
图结构分析脚本 — 支持 GE Build 图 (pbtxt) 和 PyTorch FX Graph (runnable.py)

设计原则：
  - 不硬编码任何 IR 算子范围，所有非结构性算子都参与分析
  - 不硬编码优化建议，所有建议基于分析结果动态生成

功能：
  1. 自动识别文件格式（.pbtxt -> 纯文本解析；.py -> AST 解析），无需 protobuf 库
  2. 提取 node 信息 (name, op_type, input, output, shape/dtype 或 输入参数/源码行号)
  3. 统计算子类型分布、图拓扑结构（深度/宽度）
  4. 识别重复子图，给出子图范围；结合 Profiling 给出耗时
  5. 排序：有 Profiling 按总耗时降序；无 Profiling 按算子数×重复次数降序
  6. 输出 Markdown 格式报告及优化建议

用法:
  python graph_analyzer.py <graph_file> [--output report.md] [--profiling-dir DIR]

依赖: 仅 Python 标准库
"""

import os
import re
import ast
import csv
import glob
import logging
import argparse
from collections import defaultdict, deque, Counter
from typing import Optional, List, Dict, Any, Tuple, Set

logging.basicConfig(level=logging.INFO, format='%(message)s')


# ============================================================
#  统一节点数据结构
# ============================================================

class GraphNode:
    """统一的图节点，兼容 pbtxt 和 fxgraph"""

    def __init__(self):
        self.name: str = ""
        self.op_type: str = ""
        self.raw_op_type: str = ""
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        # pbtxt 专属
        self.input_shapes: List[str] = []
        self.input_dtypes: List[str] = []
        self.output_shapes: List[str] = []
        self.output_dtypes: List[str] = []
        self.is_subgraph: bool = False
        self.subgraph_nodes: List["GraphNode"] = []
        # fxgraph 专属
        self.source_line: int = 0
        self.raw_call: str = ""


# ============================================================
#  格式自动识别
# ============================================================

def detect_graph_type(filepath: str) -> str:
    """根据文件扩展名和内容自动识别图类型"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pbtxt":
        return "pbtxt"
    elif ext == ".py":
        return "fxgraph"
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(2000)
        if "ir_version:" in head or "graph {" in head or "op_type:" in head:
            return "pbtxt"
        if "class Repro" in head or "torch.ops" in head:
            return "fxgraph"
        raise ValueError(f"无法识别文件格式: {filepath} (支持 .pbtxt 和 .py)")


# ============================================================
#  结构性算子识别（非硬编码 IR 范围，而是基于算子语义角色）
# ============================================================

def get_trivial_types(graph_type: str) -> Set[str]:
    """
    返回结构性/辅助算子集合。
    这些算子不参与重复子图分析，因为它们是图框架的结构性节点而非计算算子。

    判断依据：这些算子在所有 IR 中都是图结构的组织节点，
    不代表实际的计算操作（如数据输入、常量、输出、子图容器、元组取元素等）。
    """
    if graph_type == "pbtxt":
        # GE 图中的结构性/输入节点：数据输入、常量、变量权重、图输出、子图容器、
        # 通信控制节点(Send/Recv)、空操作(NoOp)、恒等映射(Identity)、
        # 动态常量(dynamic_const)、合并节点(Merge)等
        return {
            "Data", "Const", "Constant", "Variable", "Placeholder", "NetOutput", "subgraph",
            "Send", "Recv", "NoOp", "Identity",
        }
    else:
        # PyTorch FX 图中的辅助操作：元组取元素、标量提取、
        # reinterpret_tensor/empty_strided 等内存布局操作
        return {"getitem", "item", "reinterpret_tensor", "empty_strided",
                "empty_strided_cpu", "empty_strided_cpu_pinned", "alloc_from_pool"}


def is_compute_op(op_type: str, trivial_types: Set[str]) -> bool:
    """判断是否为计算算子：非结构性算子即为计算算子"""
    return op_type not in trivial_types and op_type != ""


# ============================================================
#  Pbtxt 解析器
# ============================================================

class PbtxtParser:
    """纯文本解析 GE pbtxt 文件，输出 GraphNode 列表"""

    RE_INPUT = r'^\s*input:\s*"?(.*?)"?\s*$'

    RE_OUTPUT = r'^\s*output:\s*"?(.*?)"?\s*$'

    RE_NAME = r'^\s*name:\s*"?(.*?)"?\s*$'

    RE_OP_TYPE = r'^\s*op_type:\s*"?(.*?)"?\s*$'

    def __init__(self):
        self.ir_version: str = ""
        self.producer_name: str = ""

    @staticmethod
    def _apply_shape_attr(target_list, attr):
        if "_ints" in attr:
            target_list.append("[" + ", ".join(str(v) for v in attr["_ints"]) + "]")
        elif "_value" in attr:
            target_list.append(str(attr["_value"]))

    @staticmethod
    def _process_brace_line(stripped, brace_depth):
        """处理括号行，返回 (depth_delta, should_break, should_skip)"""
        if stripped == "{":
            return 1, False, True
        if stripped == "}":
            new_depth = brace_depth - 1
            if new_depth == 0:
                return -1, True, True
            return -1, False, True
        return 0, False, False

    @staticmethod
    def _parse_ints_list(stripped, attr):
        for v in stripped[5:].strip().split(", "):
            v = v.strip()
            if v:
                try:
                    attr.setdefault("_ints", []).append(int(v))
                except ValueError:
                    logging.warning("无法解析整数值: %s", v)

    @staticmethod
    def _match_node_field(line, brace_depth, node, regexes):
        m = regexes["input"].match(line)
        if m:
            val = m.group(1).strip().strip('"')
            if val:
                node.inputs.append(val)
            return True
        m = regexes["output"].match(line)
        if m:
            val = m.group(1).strip().strip('"')
            if val:
                node.outputs.append(val)
            return True
        if brace_depth == 1:
            m = regexes["name"].match(line)
            if m:
                node.name = m.group(1).strip().strip('"')
                return True
            m = regexes["op_type"].match(line)
            if m:
                raw = m.group(1).strip().strip('"')
                node.raw_op_type = raw
                node.op_type = raw[3:] if raw.startswith("ge:") else raw
                return True
        return False

    @staticmethod
    def _apply_dtype_attr(node, attr, attr_name):
        if attr_name.startswith("input_desc_dtype:"):
            node.input_dtypes.append(str(attr.get("_value", "")))
        elif attr_name.startswith("output_desc_dtype:"):
            node.output_dtypes.append(str(attr.get("_value", "")))

    @staticmethod
    def _apply_brace_line(stripped, brace_depth, i):
        """处理括号行并更新 depth/i，返回 (new_brace_depth, new_i, action)
        action: 'break', 'continue', 或 None
        """
        delta, should_break, should_skip = PbtxtParser._process_brace_line(stripped, brace_depth)
        if should_skip:
            new_depth = brace_depth + delta
            new_i = i + 1
            if should_break:
                return new_depth, new_i, 'break'
            return new_depth, new_i, 'continue'
        return brace_depth, i, None

    @staticmethod
    def _update_subgraph_depth(sl):
        if sl == "{":
            return 1
        if sl == "}":
            return -1
        return 0

    @staticmethod
    def _parse_attr_scalar(stripped, attr):
        if stripped.startswith("s:"):
            attr["_value"] = stripped[2:].strip().strip('"')
            return True
        if stripped.startswith("i:"):
            try:
                attr["_value"] = int(stripped[2:].strip())
            except ValueError:
                logging.warning("无法解析整数值: %s", stripped)
            return True
        return False

    def parse_file(self, filepath: str) -> List[GraphNode]:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return self._parse_lines(lines)

    def _parse_lines(self, lines: List[str]) -> List[GraphNode]:
        re_input = re.compile(self.RE_INPUT)
        re_output = re.compile(self.RE_OUTPUT)
        re_name = re.compile(self.RE_NAME)
        re_op_type = re.compile(self.RE_OP_TYPE)
        re_attr_type = re.compile(r'type:\s*(\w+)')
        regexes = {
            "input": re_input,
            "output": re_output,
            "name": re_name,
            "op_type": re_op_type,
            "attr_type": re_attr_type,
        }

        i = 0
        n = len(lines)
        top_nodes: List[GraphNode] = []

        while i < n:
            line = lines[i].strip()
            if line.startswith("ir_version:"):
                self.ir_version = line.split(":", 1)[1].strip()
            elif line.startswith("producer_name:"):
                self.producer_name = line.split(":", 1)[1].strip().strip('"')
            elif line == "graph {":
                i += 1
                break
            i += 1

        while i < n:
            line = lines[i].strip()
            if line == "}":
                break
            if line.startswith("node {"):
                node, i = self._parse_node(lines, i, regexes)
                if node:
                    top_nodes.append(node)
            else:
                i += 1
        return top_nodes

    def _apply_node_attr(self, node, attr):
        attr_name = attr.get("_name", "")
        self._apply_dtype_attr(node, attr, attr_name)
        self._apply_shape_graph_attr(node, attr, attr_name)

    def _apply_shape_graph_attr(self, node, attr, attr_name):
        if attr_name.startswith("input_desc_shape:"):
            self._apply_shape_attr(node.input_shapes, attr)
        elif attr_name.startswith("output_desc_shape:"):
            self._apply_shape_attr(node.output_shapes, attr)
        elif attr_name == "graph":
            node.is_subgraph = True
            node.subgraph_nodes = attr.get("_subgraph_nodes", [])

    def _handle_node_attr(self, lines, i, regexes, node):
        attr, new_i = self._parse_attribute(lines, i, regexes)
        if attr:
            self._apply_node_attr(node, attr)
        return new_i

    def _parse_node(self, lines, start_idx, regexes):
        node = GraphNode()
        i = start_idx + 1
        n = len(lines)
        brace_depth = 1

        while i < n and brace_depth > 0:
            line = lines[i]
            stripped = line.strip()

            brace_depth, i, action = self._apply_brace_line(stripped, brace_depth, i)
            if action == 'break':
                break
            elif action == 'continue':
                continue

            if self._match_node_field(line, brace_depth, node, regexes):
                i += 1
                continue

            if stripped.startswith("attribute {") and brace_depth == 1:
                i = self._handle_node_attr(lines, i, regexes, node)
                continue

            i += 1
        return node, i

    def _build_sub_regexes(self):
        return {
            "input": re.compile(self.RE_INPUT),
            "output": re.compile(self.RE_OUTPUT),
            "name": re.compile(self.RE_NAME),
            "op_type": re.compile(self.RE_OP_TYPE),
            "attr_type": re.compile(r'type:\s*(\w+)'),
        }

    def _parse_subgraph_nodes(self, lines, start_idx, regexes):
        j = start_idx + 1
        n = len(lines)
        sub_depth = 1
        subgraph_nodes: List[GraphNode] = []
        while j < n and sub_depth > 0:
            sl = lines[j].strip()
            delta = self._update_subgraph_depth(sl)
            if delta != 0:
                sub_depth += delta
            elif sl.startswith("node {") and sub_depth == 1:
                sn, j = self._parse_node(lines, j, regexes)
                if sn:
                    subgraph_nodes.append(sn)
                continue
            j += 1
        return subgraph_nodes, j

    def _parse_attr_list(self, stripped, attr):
        if stripped.startswith("strings:"):
            attr.setdefault("_strings", []).append(stripped[8:].strip().strip('"'))
            return True
        if stripped.startswith("ints:"):
            self._parse_ints_list(stripped, attr)
            return True
        return False

    def _parse_attribute(self, lines, start_idx, regexes):
        attr: Dict[str, Any] = {}
        i = start_idx + 1
        n = len(lines)
        brace_depth = 1
        re_name = regexes["name"]
        re_attr_type = regexes["attr_type"]

        while i < n and brace_depth > 0:
            line = lines[i]
            stripped = line.strip()

            brace_depth, i, action = self._apply_brace_line(stripped, brace_depth, i)
            if action == 'break':
                break
            elif action == 'continue':
                continue

            if brace_depth == 1:
                m = re_name.match(line)
                if m:
                    attr["_name"] = m.group(1).strip().strip('"')
                    i += 1
                    continue
                m = re_attr_type.search(line)
                if m:
                    attr["_type"] = m.group(1)
                    i += 1
                    continue
                if self._parse_attr_scalar(stripped, attr):
                    i += 1
                    continue

            if self._parse_attr_list(stripped, attr):
                i += 1
                continue

            if stripped.startswith("g {") and brace_depth == 1:
                sub_regexes = self._build_sub_regexes()
                subgraph_nodes, i = self._parse_subgraph_nodes(lines, i, sub_regexes)
                attr["_subgraph_nodes"] = subgraph_nodes
                continue

            i += 1
        return attr, i


# ============================================================
#  FX Graph AST 解析器
# ============================================================

class FXGraphParser:
    """使用 AST 解析 PyTorch fxgraph 文件，输出 GraphNode 列表

    支持两种 inductor output_code 格式：
    1. class Repro / def forward (torch.compile 标准输出)
    2. class Runner / def call (inductor partition 输出)

    同时支持：
    - torch.ops.aten.* / torch.ops.prims.* 标准算子调用
    - autofused_* 自定义融合算子函数调用
    - buf alias 追踪 (buf5 = buf4; del buf4)
    - buf[N] = ... 形式的赋值
    - buf_tuple[0] (getitem) 形式的元组取元素
    """

    _METHOD_NAMES = ("forward", "call", "run")

    _CLASS_NAMES = ("Repro", "Runner", "Callable")

    _FUSED_PREFIXES = ("autofused", "triton_poi_fused", "triton_per_fused",
                       "triton_unk_fused", "dvm_")

    def __init__(self):
        self.source_lines: List[str] = []
        self._alias_map: Dict[str, str] = {}

    @staticmethod
    def _extract_target_names_from_tuple(tup: ast.Tuple) -> List[str]:
        return [elt.id for elt in tup.elts if isinstance(elt, ast.Name)]

    @staticmethod
    def _extract_attribute_chain(node) -> List[str]:
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()
        return parts

    @staticmethod
    def _format_constant(val) -> str:
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, str):
            return f"'{val}'"
        if val is None:
            return "None"
        return "<const>"

    @staticmethod
    def _find_entry_method(class_body) -> Optional[ast.AST]:
        """在 class body 中查找入口方法（已知方法名优先，否则取第一个非 __init__）"""
        entry_method = None
        for item in class_body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in FXGraphParser._METHOD_NAMES:
                entry_method = item
                break
        if entry_method is None:
            for item in class_body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name != "__init__":
                    entry_method = item
                    break
        return entry_method

    @classmethod
    def _is_fused_kernel(cls, func_name: str) -> bool:
        """判断函数名是否为融合算子（Inductor 各后端生成的融合 kernel）"""
        return any(func_name.startswith(prefix) for prefix in cls._FUSED_PREFIXES)

    def parse_file(self, filepath: str) -> List[GraphNode]:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        self.source_lines = source.splitlines()
        tree = ast.parse(source, filename=filepath)
        return self._parse_ast(tree)

    def _parse_ast(self, tree: ast.Module) -> List[GraphNode]:
        nodes: List[GraphNode] = []
        target_class = None
        # 优先查找已知类名
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name in self._CLASS_NAMES:
                    target_class = node
                    break
                if target_class is None:
                    target_class = node

        if target_class is not None:
            entry_method = self._find_entry_method(target_class.body)
            if entry_method is not None:
                for stmt in entry_method.body:
                    self._parse_stmt(stmt, nodes)
                return nodes

        # 无 class 定义时，查找模块级 def call/def forward/def run
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in self._METHOD_NAMES:
                for stmt in node.body:
                    self._parse_stmt(stmt, nodes)
                return nodes

        return nodes

    def _handle_assign_value(self, value, target_name, lineno, nodes):
        if isinstance(value, ast.Name):
            self._alias_map[target_name] = self._resolve_alias(value.id)
            return
        if isinstance(value, ast.Subscript):
            node = self._parse_getitem(value, target_name, lineno)
            if node:
                nodes.append(node)
            else:
                base = self._extract_name(value)
                if base:
                    self._alias_map[target_name] = self._resolve_alias(base)
            return
        if isinstance(value, ast.Call):
            node = self._parse_call(value, target_name, lineno)
            if node:
                node.inputs = [
                    self._resolve_alias(inp)
                    if not inp.startswith("'") and not inp.startswith("[")
                    else inp
                    for inp in node.inputs
                ]
                nodes.append(node)

    def _handle_assign_node(self, stmt, nodes):
        target_names = self._extract_target_names(stmt.targets)
        if not target_names:
            return
        target_name = target_names[0]
        self._handle_assign_value(stmt.value, target_name, stmt.lineno, nodes)

    def _handle_call_node(self, stmt, nodes):
        node = self._parse_call(stmt.value, f"__expr_{stmt.lineno}", stmt.lineno)
        if node:
            nodes.append(node)

    def _handle_other_stmt(self, stmt, nodes):
        if isinstance(stmt, ast.Delete):
            for tgt in stmt.targets:
                name = self._extract_name(tgt)
                if name and name in self._alias_map:
                    del self._alias_map[name]
        elif isinstance(stmt, (ast.With, ast.AsyncWith, ast.For)):
            for inner_stmt in stmt.body:
                self._parse_stmt(inner_stmt, nodes)
        elif isinstance(stmt, ast.If):
            for inner_stmt in stmt.body:
                self._parse_stmt(inner_stmt, nodes)
            for inner_stmt in stmt.orelse:
                self._parse_stmt(inner_stmt, nodes)

    def _parse_stmt(self, stmt: ast.stmt, nodes: List[GraphNode]):
        """解析单条语句，可能生成 0 或 1 个节点"""
        if isinstance(stmt, ast.Assign):
            self._handle_assign_node(stmt, nodes)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            self._handle_call_node(stmt, nodes)
        else:
            self._handle_other_stmt(stmt, nodes)

    def _resolve_alias(self, name: str) -> str:
        """递归解析别名链，找到最终来源"""
        seen = set()
        while name in self._alias_map and name not in seen:
            seen.add(name)
            name = self._alias_map[name]
        return name

    def _parse_getitem(self, expr: ast.Subscript, target_name: str, lineno: int) -> Optional[GraphNode]:
        """解析 buf = tuple[0] 形式的 getitem"""
        base = self._extract_name(expr.value)
        if not base:
            return None
        node = GraphNode()
        node.name = target_name
        node.raw_op_type = "getitem"
        node.op_type = "getitem"
        node.outputs = [target_name]
        node.source_line = lineno
        node.inputs = [self._resolve_alias(base)]
        if 0 < lineno <= len(self.source_lines):
            node.raw_call = self.source_lines[lineno - 1].strip()
        return node

    def _create_call_node(self, target_name: str, raw_op_type: str, op_type: str,
                          lineno: int, expr: ast.Call) -> GraphNode:
        """创建 GraphNode 并填充通用字段"""
        node = GraphNode()
        node.name = target_name
        node.raw_op_type = raw_op_type
        node.op_type = op_type
        node.outputs = [target_name]
        node.source_line = lineno
        node.inputs = self._extract_input_names(expr.args)
        if 0 < lineno <= len(self.source_lines):
            node.raw_call = self.source_lines[lineno - 1].strip()
        return node

    def _parse_call(self, expr: ast.Call, target_name: str, lineno: int) -> Optional[GraphNode]:
        if not isinstance(expr, ast.Call):
            return None

        # 尝试1: torch.ops.<ns>.<op>.<overload>(...) 标准调用
        op_info = self._extract_torch_op(expr.func)
        if op_info is not None:
            full = f"{op_info[0]}.{op_info[1]}.{op_info[2]}" if op_info[2] else f"{op_info[0]}.{op_info[1]}"
            return self._create_call_node(target_name, full, f"{op_info[0]}.{op_info[1]}", lineno, expr)

        # 尝试2: 融合算子函数调用
        # 支持的融合后端前缀：
        #   autofused_*       — Inductor + AscendC (AutoFuse)
        #   triton_poi_fused_* — Inductor + Triton (pointwise)
        #   triton_per_fused_* — Inductor + Triton (persistent/reduction)
        #   dvm_*              — Inductor + DVM
        func_name = self._extract_func_name(expr.func)
        if func_name and self._is_fused_kernel(func_name):
            return self._create_call_node(target_name, func_name, func_name, lineno, expr)

        # 尝试3: reinterpret_tensor / empty_strided 等辅助函数 — 记为 structural
        if func_name in ("reinterpret_tensor", "empty_strided", "empty_strided_cpu",
                         "empty_strided_cpu_pinned", "alloc_from_pool"):
            return self._create_call_node(target_name, func_name, func_name, lineno, expr)

        return None

    def _extract_func_name(self, func: ast.expr) -> str:
        """提取函数调用的名称（支持直接 Name 和 Attribute 链）"""
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts = self._extract_attribute_chain(func)
            return ".".join(parts) if parts else ""
        return ""

    def _extract_name(self, expr: ast.expr) -> str:
        """从表达式中提取变量名（用于别名追踪）"""
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Subscript):
            return self._extract_name(expr.value)
        if isinstance(expr, ast.Attribute):
            return self._extract_name(expr.value)
        return ""

    def _extract_torch_op(self, func: ast.expr) -> Optional[Tuple[str, str, str]]:
        parts = self._extract_attribute_chain(func)
        if len(parts) >= 4 and parts[0] == "torch" and parts[1] == "ops":
            return (parts[2], parts[3], parts[4] if len(parts) > 4 else "")
        return None

    def _extract_target_names(self, targets: List[ast.expr]) -> List[str]:
        names = []
        for target in targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, ast.Tuple):
                names.extend(self._extract_target_names_from_tuple(target))
        return names

    def _resolve_input_name(self, arg) -> Optional[str]:
        if isinstance(arg, ast.Name):
            return arg.id
        if isinstance(arg, ast.Attribute):
            return ".".join(self._extract_attribute_chain(arg))
        if isinstance(arg, ast.Constant):
            return self._format_constant(arg.value)
        if isinstance(arg, ast.List):
            return "[...]"
        if isinstance(arg, ast.Starred) and isinstance(arg.value, ast.Name):
            return f"*{arg.value.id}"
        return None

    def _extract_input_names(self, args: List[ast.expr]) -> List[str]:
        names = []
        for arg in args:
            resolved = self._resolve_input_name(arg)
            if resolved is not None:
                names.append(resolved)
            else:
                names.append(f"<{type(arg).__name__}>")
        return names


# ============================================================
#  Profiling 数据加载器
# ============================================================

def _safe_float(val) -> float:
    if val is None:
        return 0.0
    val = str(val).strip()
    if val in ("", "N/A", "n/a", "NA", "None", "null"):
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


class ProfilingDataLoader:
    """从 MindStudio Profiler 输出目录加载算子耗时数据"""

    def __init__(self, profiling_dir: str):
        self.profiling_dir = profiling_dir
        self.op_duration_map: Dict[str, float] = {}
        self.op_type_avg_map: Dict[str, float] = {}
        self.total_op_time_us: float = 0.0
        self.iteration_time_us: float = 0.0
        self.loaded: bool = False

    @staticmethod
    def _try_score_encoding(path: str, encoding: str) -> int:
        try:
            with open(path, "r", encoding=encoding) as f:
                header = next(csv.reader(f), None)
                if not header:
                    return 0
                return 10 if "Op Name" in header else 0
        except (UnicodeDecodeError, csv.Error, OSError):
            return -1

    @staticmethod
    def _score_from_header(path: str) -> int:
        try:
            for encoding in ["utf-8-sig", "utf-8", "gbk"]:
                result = ProfilingDataLoader._try_score_encoding(path, encoding)
                if result >= 0:
                    return result
        except Exception:
            logging.warning("评分文件 header 时发生异常: %s", path)
        return 0

    @staticmethod
    def _score_op_summary(path: str) -> int:
        score = 0
        base = os.path.basename(path).lower()
        if "_output_" in base:
            score -= 100
        if "_no_op_name" in base:
            score -= 50
        score += ProfilingDataLoader._score_from_header(path)
        return score

    @staticmethod
    def _process_api_row(row, state):
        name = str(row.get("API Name", "")).strip()
        level = str(row.get("Level", "")).strip().lower()
        total = _safe_float(row.get("Time(us)", 0))
        count = int(_safe_float(row.get("Count", 0)))
        if level == "model" and "ModelExecute" in name and count > 1:
            state[0] = total
            state[1] = count
        elif name == "RunGraphAsync":
            state[2] = total
        elif name == "StreamSynchronize" and count > 1:
            state[3] = total
            state[4] = count
        elif "Synchronize" in name and count > state[5]:
            state[5] = count

    @staticmethod
    def _filter_api_rows_by_iter(rows):
        state = [0, 0, 0, 0, 0, 0]
        for row in rows:
            ProfilingDataLoader._process_api_row(row, state)
        return state[0], state[1], state[2], state[3], state[4]

    @staticmethod
    def _read_csv(filepath: str) -> List[Dict[str, str]]:
        _, rows = ProfilingDataLoader._read_csv_with_header(filepath)
        return rows

    @staticmethod
    def _try_read_with_encoding(filepath: str, encoding: str):
        with open(filepath, "r", encoding=encoding, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return [], []
            header_len = len(header)

            def _normalize(row):
                if not row or all(c.strip() == "" for c in row):
                    return None
                if len(row) < header_len:
                    row.extend([""] * (header_len - len(row)))
                elif len(row) > header_len:
                    row = row[:header_len]
                return dict(zip(header, row))

            rows = [r for r in (_normalize(row) for row in reader) if r is not None]
            return header, rows

    @staticmethod
    def _read_csv_with_header(filepath: str) -> Tuple[List[str], List[Dict[str, str]]]:
        for encoding in ["utf-8-sig", "utf-8", "gbk", "latin-1"]:
            try:
                return ProfilingDataLoader._try_read_with_encoding(filepath, encoding)
            except (UnicodeDecodeError, csv.Error, OSError):
                logging.warning("读取 CSV 失败 (编码 %s): %s", encoding, filepath)
                continue
        return [], []

    def load(self):
        for patterns, loader in [
            (["*op_summary*.csv", "kernel_details.csv"], self._load_op_summary),
            (["*op_statistic*.csv", "op_statistic.csv"], self._load_op_statistic),
            (["step_trace_*.csv", "step_trace_time.csv"], self._load_step_trace),
            (["api_statistic_*.csv", "api_statistic.csv"], self._load_api_for_iter),
        ]:
            f = self._find_file(patterns)
            if f:
                loader(f)
        self.loaded = len(self.op_duration_map) > 0 or len(self.op_type_avg_map) > 0

    def get_node_duration(self, node_name: str, op_type: str = "", node: Any = None) -> float:
        if node_name in self.op_duration_map:
            return self.op_duration_map[node_name]
        dur = self._match_by_fuzzy_name(node_name)
        if dur > 0:
            return dur
        dur = self._match_by_aclnn(node_name)
        if dur > 0:
            return dur
        if op_type and op_type in self.op_type_avg_map:
            return self.op_type_avg_map[op_type]
        if node is not None:
            dur = self._match_by_neighbor(node)
            if dur > 0:
                return dur
            dur = self._match_by_shape(node)
            if dur > 0:
                return dur
        return 0.0

    def _find_file(self, patterns: List[str]) -> Optional[str]:
        for pattern in patterns:
            matches = glob.glob(os.path.join(self.profiling_dir, "**", pattern), recursive=True)
            if matches:
                if "op_summary" in pattern and len(matches) > 1:
                    return self._pick_best_op_summary(matches)
                return matches[0]
        return None

    def _pick_best_op_summary(self, candidates: List[str]) -> str:
        if not candidates:
            return candidates[0] if candidates else ""
        best_path = candidates[0]
        best_score = self._score_op_summary(best_path)
        for candidate in candidates[1:]:
            candidate_score = self._score_op_summary(candidate)
            if candidate_score > best_score:
                best_path = candidate
                best_score = candidate_score
        return best_path

    def _load_op_summary(self, filepath: str):
        header, rows = self._read_csv_with_header(filepath)
        if not header or not rows:
            return
        is_kernel_details = "Name" in header and "Accelerator Core" in header
        name_col = "Name" if is_kernel_details else "Op Name"
        type_col = "Type" if is_kernel_details else "OP Type"
        dur_col = "Duration(us)" if is_kernel_details else "Task Duration(us)"

        type_durations: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            op_name = str(row.get(name_col, "")).strip()
            op_type = str(row.get(type_col, "")).strip()
            dur = _safe_float(row.get(dur_col, 0))
            if op_name and dur > 0:
                if op_name not in self.op_duration_map or dur > self.op_duration_map[op_name]:
                    self.op_duration_map[op_name] = dur
                self.total_op_time_us += dur
            if op_type and dur > 0:
                type_durations[op_type].append(dur)
        for op_type, durs in type_durations.items():
            self.op_type_avg_map[op_type] = sum(durs) / len(durs)

    def _load_op_statistic(self, filepath: str):
        rows = self._read_csv(filepath)
        if not rows:
            return
        for row in rows:
            op_type = str(row.get("OP Type", "")).strip()
            avg = _safe_float(row.get("Avg Time(us)", row.get("Avg(us)", 0)))
            if op_type and avg > 0:
                self.op_type_avg_map[op_type] = avg

    def _load_step_trace(self, filepath: str):
        rows = self._read_csv(filepath)
        if not rows:
            return
        iter_times = []
        for row in rows:
            # Skip eager summary row (step_trace_time.csv)
            if "Stage" in row or "Computing" in row:
                continue
            for col in ["Iteration Time(us)", "Iteration Time", "iteration_time"]:
                if col in row and str(row[col]).strip() not in ("", "N/A"):
                    iter_times.append(_safe_float(row[col]))
                    break
        if iter_times:
            self.iteration_time_us = sum(iter_times) / len(iter_times)

    def _load_api_for_iter(self, filepath: str):
        """Infer iteration time from api_statistic when step_trace is unavailable."""
        if self.iteration_time_us > 0:
            return
        rows = self._read_csv(filepath)
        if not rows:
            return
        me_total, me_count, rg_total, ss_total, ss_count = self._filter_api_rows_by_iter(rows)
        if me_count > 1 and me_total > 0:
            self.iteration_time_us = me_total / me_count
        elif ss_count > 1 and (rg_total + ss_total) > 0:
            self.iteration_time_us = (rg_total + ss_total) / ss_count

    def _match_by_fuzzy_name(self, node_name: str) -> float:
        if len(node_name) >= 3:
            for prof_name, dur in self.op_duration_map.items():
                name_long_enough = len(prof_name) >= 3
                name_contains = node_name in prof_name or prof_name in node_name
                if name_long_enough and name_contains:
                    return dur
        return 0.0

    def _match_by_aclnn(self, node_name: str) -> float:
        aclnn_key = f"aclnn{node_name}" if not node_name.startswith("aclnn") else node_name
        if aclnn_key in self.op_duration_map:
            return self.op_duration_map[aclnn_key]
        for prof_name, dur in self.op_duration_map.items():
            if prof_name.startswith("aclnn"):
                prof_op = prof_name[5:]
                names_valid = prof_op and node_name
                name_match = (prof_op == node_name
                              or prof_op in node_name
                              or node_name in prof_op)
                if names_valid and name_match:
                    return dur
        return 0.0

    def _match_by_neighbor(self, node: Any) -> float:
        neighbor_names = set()
        for inp in node.inputs:
            neighbor_names.add(inp.split(":")[0] if ":" in inp else inp)
        for out in node.outputs:
            neighbor_names.add(out.split(":")[0] if ":" in out else out)
        for prof_name, dur in self.op_duration_map.items():
            for nb in neighbor_names:
                if nb and nb in prof_name:
                    return dur
        return 0.0

    def _match_by_shape(self, node: Any) -> float:
        if not (hasattr(node, 'input_shapes') and node.input_shapes):
            return 0.0
        node_shape_sig = ";".join(node.input_shapes)
        for prof_name, dur in self.op_duration_map.items():
            if hasattr(self, '_op_shape_map') and prof_name in self._op_shape_map:
                prof_shape = self._op_shape_map[prof_name]
                if prof_shape and prof_shape == node_shape_sig:
                    return dur
        return 0.0


# ============================================================
#  统一图结构分析器
# ============================================================

class GraphAnalyzer:
    """
    分析计算图结构，识别重复子图和优化建议。支持 pbtxt 和 fxgraph。

    设计原则：
    - 不硬编码任何 IR 算子范围，通过 get_trivial_types() 动态识别结构性算子
    - 不硬编码优化建议，所有建议基于分析结果（重复子图、算子分布、拓扑）动态生成
    """

    def __init__(self, nodes: List[GraphNode], graph_type: str, profiling_dir: str = ""):
        self.graph_type = graph_type
        self.all_nodes = nodes
        self.flat_nodes: List[GraphNode] = []
        self._flatten_nodes(nodes)
        self.node_map: Dict[str, GraphNode] = {n.name: n for n in self.flat_nodes}
        self._cache: Dict[str, Any] = {}
        self.ir_version: str = ""
        self.producer_name: str = ""
        self.profiling_loader: Optional[ProfilingDataLoader] = None
        if profiling_dir and os.path.isdir(profiling_dir):
            self.profiling_loader = ProfilingDataLoader(profiling_dir)
            self.profiling_loader.load()
        self.trivial_types = get_trivial_types(graph_type)

    @staticmethod
    def _format_node_range(node_names):
        if len(node_names) <= 3:
            return ", ".join(node_names)
        return f"{node_names[0]} ... {node_names[-1]} ({len(node_names)} nodes)"

    @staticmethod
    def _format_first_instance_nodes(sg):
        if not sg.get("instances"):
            return ""
        ns = sg["instances"][0].get("nodes", [])
        if len(ns) <= 5:
            return ", ".join(ns)
        return ", ".join(ns[:3]) + " ... " + ns[-1]

    @staticmethod
    def _select_maximal_patterns(candidates):
        by_len: Dict[int, List[Tuple[Tuple[str, ...], List[Tuple[int, int]]]]] = defaultdict(list)
        for ngram, occ in candidates.items():
            by_len[len(ngram)].append((ngram, occ))
        deduped: Dict[Tuple[str, ...], List[Tuple[int, int]]] = {}
        for length in sorted(by_len.keys(), reverse=True):
            group = by_len[length]
            group.sort(key=lambda x: len(x[1]), reverse=True)
            kept_sigs: List[frozenset] = []
            for ngram, occ in group:
                sig = frozenset(Counter(ngram).items())
                if not any(sig == ks for ks in kept_sigs):
                    deduped[ngram] = occ
                    kept_sigs.append(sig)
        maximal: Dict[Tuple[str, ...], List[Tuple[int, int]]] = {}
        # covered_records stores (start, end) intervals of already-covered instances
        covered_records: List[Tuple[int, int]] = []
        for ngram, occurrences in sorted(deduped.items(), key=lambda x: len(x[0]), reverse=True):
            n = len(ngram)
            uncovered = []
            for s, e in occurrences:
                # Check if this instance [s, s+n) is fully contained in any covered interval
                if not any(cs <= s and ce >= s + n for cs, ce in covered_records):
                    uncovered.append((s, e))
            if len(uncovered) >= 2:
                maximal[ngram] = uncovered
                for s, _ in uncovered:
                    covered_records.append((s, s + n))
        return maximal

    def analyze_op_distribution(self) -> Dict[str, int]:
        if "op_distribution" in self._cache:
            return self._cache["op_distribution"]
        counter = defaultdict(int)
        for node in self.flat_nodes:
            counter[node.op_type] += 1
        result = dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
        self._cache["op_distribution"] = result
        return result

    def analyze_graph_topology(self) -> Dict[str, Any]:
        if "graph_topology" in self._cache:
            return self._cache["graph_topology"]
        children, in_degree = self._build_children_indegree()
        original_in_degree = dict(in_degree)
        root_nodes = [n.name for n in self.flat_nodes if original_in_degree.get(n.name, 0) == 0]
        leaf_nodes = [n.name for n in self.flat_nodes if not children.get(n.name)]
        max_depth, max_width, processed = self._compute_depth_width(children, in_degree)
        has_cycle = processed < len(self.flat_nodes)
        avg_branch = round(sum(len(v) for v in children.values()) / max(len(children), 1), 2)
        result = {
            "total_nodes": len(self.flat_nodes),
            "max_depth": max_depth,
            "max_width": max_width,
            "avg_branching_factor": avg_branch,
            "root_node_count": len(root_nodes),
            "leaf_node_count": len(leaf_nodes),
            "processed": processed,
            "has_cycle": has_cycle,
        }
        self._cache["graph_topology"] = result
        return result

    def find_repeated_subgraphs(self, min_len=2, max_len=20, min_repeat=2) -> List[Dict[str, Any]]:
        if "repeated_subgraphs" in self._cache:
            return self._cache["repeated_subgraphs"]
        topo_order = self._topological_sort()
        compute_entries = self._build_compute_entries(topo_order)
        compute_names = [e[0] for e in compute_entries]
        op_sequence = [e[1] for e in compute_entries]
        all_patterns = self._find_ngram_patterns(op_sequence, compute_names, min_len, max_len)
        candidates = {k: v for k, v in all_patterns.items() if len(v) >= min_repeat}
        maximal = self._select_maximal_patterns(candidates)
        has_prof = bool(self.profiling_loader and self.profiling_loader.loaded)
        iter_time = self.profiling_loader.iteration_time_us if has_prof else 0.0
        results = []
        for ngram, occurrences in maximal.items():
            results.append(self._build_pattern_result(ngram, occurrences, compute_names, has_prof, iter_time))
        results.sort(key=lambda x: x["sort_score"], reverse=True)
        self._cache["repeated_subgraphs"] = results
        return results

    def format_markdown_report(self) -> str:
        self.analyze_op_distribution()
        topo = self.analyze_graph_topology()
        subgraphs = self.find_repeated_subgraphs()
        has_prof = bool(self.profiling_loader and self.profiling_loader.loaded)
        lines = []
        lines.extend(self._format_basic_info(topo, has_prof))
        lines.extend(self._format_subgraph_table(subgraphs, has_prof))
        lines.extend(self._format_subgraph_details(subgraphs, has_prof))
        return "\n".join(lines)

    def _flatten_nodes(self, nodes: List[GraphNode]):
        for node in nodes:
            self.flat_nodes.append(node)
            if node.is_subgraph and node.subgraph_nodes:
                self._flatten_nodes(node.subgraph_nodes)

    def _topological_sort(self) -> List[str]:
        children, in_degree = self._build_children_indegree()
        queue = deque()
        for node in self.flat_nodes:
            if in_degree.get(node.name, 0) == 0:
                queue.append(node.name)
        result = []
        while queue:
            current = queue.popleft()
            result.append(current)
            for child in children.get(current, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return result

    def _build_adjacency(self) -> Dict[str, Set[str]]:
        """构建邻接表：producer_name -> set(consumer_names)，用于连通性校验"""
        if "adjacency" in self._cache:
            return self._cache["adjacency"]
        adj: Dict[str, Set[str]] = defaultdict(set)
        for node in self.flat_nodes:
            for inp in node.inputs:
                if not inp:
                    continue
                src = inp.split(":")[0] if self.graph_type == "pbtxt" else inp
                if src in self.node_map:
                    adj[src].add(node.name)
        self._cache["adjacency"] = dict(adj)
        return dict(adj)

    def _compute_node_depths(self) -> Dict[str, int]:
        """计算每个节点在图中的拓扑深度（从根节点的最长路径）"""
        if "node_depths" in self._cache:
            return self._cache["node_depths"]
        children, in_degree = self._build_children_indegree()
        depths: Dict[str, int] = {}
        queue = deque()
        for node in self.flat_nodes:
            if in_degree.get(node.name, 0) == 0:
                queue.append(node.name)
                depths[node.name] = 0
        while queue:
            current = queue.popleft()
            d = depths.get(current, 0)
            for child in children.get(current, []):
                depths[child] = max(depths.get(child, 0), d + 1)
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        self._cache["node_depths"] = depths
        return depths

    def _is_connected_subgraph(self, node_names: List[str]) -> bool:
        """
        校验一组节点在数据依赖图上是否构成有意义的子图。

        两种有效子图模式：
        1. 串行子图：节点间通过数据依赖边连通（如 MatMul→Sigmoid→MatMul）
        2. 并行子图：所有计算算子同类型、同深度，无内部边但属于同一并行组
           （如 20 个并行 ClipByValue，是批量融合的目标）

        无效模式：窗口内混合了不相关的算子（拓扑排序恰好相邻但无结构关系）
        """
        if len(node_names) <= 1:
            return True
        # 提取计算算子（排除结构性算子）
        compute_ops = [nn for nn in node_names if is_compute_op(self.node_map[nn].op_type, self.trivial_types)]
        if not compute_ops:
            return False
        # 并行组模式：所有计算算子同类型且同深度（同一层级并行执行），允许无内部边
        compute_types = [self.node_map[nn].op_type for nn in compute_ops]
        if len(set(compute_types)) == 1:
            depths = self._compute_node_depths()
            compute_depths = [depths.get(nn, -1) for nn in compute_ops]
            if len(set(compute_depths)) <= 2:  # 允许微小深度差（±1）
                return True
        # 串行子图模式：要求通过数据依赖边连通
        node_set = set(node_names)
        adj = self._build_adjacency()
        internal_edges = 0
        for nn in node_names:
            consumers = adj.get(nn, set())
            internal_edges += len(consumers & node_set)
        if internal_edges >= len(compute_ops) - 1:
            return True
        return self._bfs_connected(node_names, node_set, adj)

    def _process_bfs_reverse_edges(self, current: str, node_set: Set[str],
                                    visited: Set[str], queue: deque):
        """处理 BFS 反向边：检查谁指向 current，加入未访问的源节点"""
        node = self.node_map.get(current)
        if not node:
            return
        for inp in node.inputs:
            if not inp:
                continue
            src = inp.split(":")[0] if self.graph_type == "pbtxt" else inp
            if src in node_set and src not in visited:
                visited.add(src)
                queue.append(src)

    def _bfs_connected(self, node_names: List[str], node_set: Set[str], adj: Dict[str, Set[str]]) -> bool:
        """BFS 验证窗口内节点是否通过内部边连通"""
        if not node_names:
            return True
        visited: Set[str] = set()
        queue = deque([node_names[0]])
        visited.add(node_names[0])
        while queue:
            current = queue.popleft()
            # 正向边
            for consumer in adj.get(current, set()):
                if consumer in node_set and consumer not in visited:
                    visited.add(consumer)
                    queue.append(consumer)
            # 反向边：检查谁指向 current
            self._process_bfs_reverse_edges(current, node_set, visited, queue)
        return len(visited) == len(node_set)

    def _build_node_detail(self, node: GraphNode) -> Dict[str, Any]:
        detail = {"name": node.name, "op_type": node.op_type}
        if self.graph_type == "pbtxt":
            detail["input_shapes"] = "; ".join(s for s in node.input_shapes if s)
            detail["input_dtypes"] = "; ".join(s for s in node.input_dtypes if s)
            detail["output_shapes"] = "; ".join(s for s in node.output_shapes if s)
            detail["output_dtypes"] = "; ".join(s for s in node.output_dtypes if s)
        else:
            detail["inputs"] = ", ".join(node.inputs[:6])
            detail["source_line"] = node.source_line
            detail["raw_call"] = node.raw_call
        return detail

    def _build_children_indegree(self):
        children: Dict[str, List[str]] = defaultdict(list)
        in_degree: Dict[str, int] = defaultdict(int)
        for node in self.flat_nodes:
            in_degree.setdefault(node.name, 0)
            for inp in node.inputs:
                if not inp:
                    continue
                src = inp.split(":")[0] if self.graph_type == "pbtxt" else inp
                if src in self.node_map:
                    children[src].append(node.name)
                    in_degree[node.name] += 1
        return children, in_degree

    def _compute_depth_width(self, children, in_degree):
        queue = deque()
        depth: Dict[str, int] = {}
        for node in self.flat_nodes:
            if in_degree.get(node.name, 0) == 0:
                queue.append(node.name)
                depth[node.name] = 0
        max_depth = max_width = 0
        width_at_depth: Dict[int, int] = defaultdict(int)
        processed = 0
        while queue:
            current = queue.popleft()
            d = depth.get(current, 0)
            max_depth = max(max_depth, d)
            width_at_depth[d] += 1
            for child in children.get(current, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    depth[child] = d + 1
                    queue.append(child)
            processed += 1
        for cnt in width_at_depth.values():
            max_width = max(max_width, cnt)
        return max_depth, max_width, processed

    def _build_compute_entries(self, topo_order):
        return [
            (name, self.node_map[name].op_type)
            for name in topo_order
            if name in self.node_map and is_compute_op(self.node_map[name].op_type, self.trivial_types)
        ]

    def _find_ngram_patterns(self, op_sequence, compute_names, min_len, max_len):
        all_patterns: Dict[Tuple[str, ...], List[Tuple[int, int]]] = defaultdict(list)
        actual_max = min(max_len, len(op_sequence))
        for n in range(min_len, actual_max + 1):
            for i in range(len(op_sequence) - n + 1):
                ngram = tuple(op_sequence[i:i + n])
                window_nodes = compute_names[i:i + n]
                if not self._is_connected_subgraph(window_nodes):
                    continue
                all_patterns[ngram].append((i, i + n))
        return all_patterns

    def _compute_instance_time(self, start_idx, end_idx, compute_names, has_prof):
        node_names = compute_names[start_idx:end_idx]
        inst_time = 0.0
        if has_prof:
            for nn in node_names:
                node = self.node_map.get(nn)
                if node:
                    inst_time += self.profiling_loader.get_node_duration(nn, node.op_type, node)
        return node_names, inst_time

    def _build_instance(self, node_names, inst_time):
        node_details = [self._build_node_detail(self.node_map[nn]) for nn in node_names if nn in self.node_map]
        return {
            "nodes": node_names, "node_range": self._format_node_range(node_names),
            "first_node": node_names[0] if node_names else "",
            "last_node": node_names[-1] if node_names else "",
            "instance_time_us": round(inst_time, 2),
            "node_details": node_details,
        }

    def _build_pattern_result(self, ngram, occurrences, compute_names, has_prof, iter_time):
        instances = []
        instance_times = []
        for start_idx, end_idx in occurrences:
            node_names, inst_time = self._compute_instance_time(start_idx, end_idx, compute_names, has_prof)
            instance_times.append(inst_time)
            instances.append(self._build_instance(node_names, inst_time))
        repeat_count = len(occurrences)
        pattern_len = len(ngram)
        avg_inst_time = sum(instance_times) / repeat_count if instance_times else 0.0
        total_time = avg_inst_time * repeat_count
        ratio = (total_time / iter_time * 100) if iter_time > 0 else 0.0
        sort_score = total_time if has_prof else pattern_len * repeat_count
        sep = " -> " if self.graph_type == "pbtxt" else " → "
        return {
            "pattern": sep.join(ngram), "op_types": list(ngram),
            "pattern_len": pattern_len, "repeat_count": repeat_count,
            "instances": instances[:20],
            "avg_instance_time_us": round(avg_inst_time, 2),
            "total_time_us": round(total_time, 2),
            "total_time_ratio_pct": round(ratio, 2),
            "has_profiling": has_prof, "sort_score": round(sort_score, 2),
        }

    def _format_basic_info(self, topo, has_prof):
        title = "GE Build 图 (pbtxt)" if self.graph_type == "pbtxt" else "PyTorch FX Graph"
        lines = []
        lines.append(f"# {title} 结构分析报告\n")
        lines.append("## 1. 基本信息\n")
        if self.graph_type == "pbtxt":
            lines.append(f"- **顶级节点数**: {len(self.all_nodes)}")
            lines.append(f"- **展平后节点数**: {len(self.flat_nodes)}")
            lines.append(f"- **IR 版本**: {self.ir_version}")
            lines.append(f"- **Producer**: {self.producer_name}")
        else:
            lines.append(f"- **节点总数**: {len(self.all_nodes)}")
        if has_prof:
            lines.append("- **Profiling 关联**: 已加载")
            iter_us = self.profiling_loader.iteration_time_us
            lines.append(f"- **迭代耗时**: {iter_us:.1f} us")
            total_us = self.profiling_loader.total_op_time_us
            lines.append(f"- **算子总耗时**: {total_us:.1f} us")
        else:
            lines.append("- **Profiling 关联**: 未加载 (按算子数×重复次数排序)")
        if topo.get("has_cycle"):
            unprocessed = len(self.flat_nodes) - topo['processed']
            lines.append(f"- **拓扑结构**: 检测到环路 ({unprocessed} 个节点未完成拓扑排序)")
        else:
            d = topo['max_depth']
            w = topo['max_width']
            bf = topo['avg_branching_factor']
            rc = topo['root_node_count']
            lc = topo['leaf_node_count']
            lines.append(f"- **拓扑结构**: 最大深度 {d}, 最大宽度 {w}, 平均分支因子 {bf}, 根节点 {rc}, 叶节点 {lc}")
        lines.append("")
        return lines

    def _format_subgraph_table(self, subgraphs, has_prof):
        lines = []
        sort_hint = "按总耗时排序" if has_prof else "按算子数×重复次数排序"
        lines.append(f"## 2. 重复子图分析 (共 {len(subgraphs)} 种模式, {sort_hint})\n")
        if not subgraphs:
            lines.append("(未发现重复子图)\n")
            return lines
        if has_prof:
            lines.append("| # | Pattern | Repeat | Len | 首个实例算子名 | Avg(us) | Total(us) | Ratio% |")
            lines.append("|---|---------|--------|-----|----------------|---------|-----------|--------|")
        else:
            lines.append("| # | Pattern | Repeat | Len | 首个实例算子名 | Score |")
            lines.append("|---|---------|--------|-----|----------------|-------|")
        for i, sg in enumerate(subgraphs[:30], 1):
            pat = sg["pattern"].replace(" -> ", " → ")[:50]
            first_inst_nodes = self._format_first_instance_nodes(sg)
            if has_prof:
                rc = sg['repeat_count']
                pl = sg['pattern_len']
                avg = sg['avg_instance_time_us']
                tot = sg['total_time_us']
                rat = sg['total_time_ratio_pct']
                lines.append(f"| {i} | {pat} | {rc} | {pl} | {first_inst_nodes} | {avg:.1f} | {tot:.1f} | {rat:.1f} |")
            else:
                rc = sg['repeat_count']
                pl = sg['pattern_len']
                sc = sg['sort_score']
                lines.append(f"| {i} | {pat} | {rc} | {pl} | {first_inst_nodes} | {sc:.1f} |")
        lines.append("")
        return lines

    def _format_subgraph_details(self, subgraphs, has_prof):
        lines = []
        if not subgraphs:
            return lines
        lines.append("### Top 5 子图实例详情\n")
        for i, sg in enumerate(subgraphs[:5], 1):
            rc = sg['repeat_count']
            lines.append(f"**[{i}]** `{sg['pattern']}` (重复 {rc} 次)\n")
            if sg.get("instances"):
                inst = sg["instances"][0]
                inst_time_str = f" (耗时 {inst['instance_time_us']:.1f}us)" if has_prof else ""
                lines.append(f"首个实例算子详情{inst_time_str}:\n")
                lines.extend(self._format_node_detail_table(inst))
                lines.append("")
            lines.append("所有实例算子名:\n")
            for j, inst in enumerate(sg.get("instances", [])[:5], 1):
                inst_time_str = f", 耗时 {inst['instance_time_us']:.1f}us" if has_prof else ""
                lines.append(f"- 实例{j}: `{inst['node_range']}`{inst_time_str}")
            if sg["repeat_count"] > 5:
                lines.append(f"- ... (共 {sg['repeat_count']} 个实例)")
            lines.append("")
        return lines

    def _format_node_detail_table(self, inst):
        lines = []
        if self.graph_type == "pbtxt":
            lines.append("| # | 算子名 | OP Type | Input Shape | Input Dtype | Output Shape | Output Dtype |")
            lines.append("|---|--------|---------|-------------|-------------|--------------|--------------|")
            for j, nd in enumerate(inst.get("node_details", []), 1):
                name = nd['name']
                ot = nd['op_type']
                ish = nd.get('input_shapes') or '-'
                idt = nd.get('input_dtypes') or '-'
                osh = nd.get('output_shapes') or '-'
                odt = nd.get('output_dtypes') or '-'
                lines.append(f"| {j} | {name} | {ot} | {ish} | {idt} | {osh} | {odt} |")
        else:
            lines.append("| # | 算子名 | OP Type | 输入参数 | 源码行号 | 原始调用 |")
            lines.append("|---|--------|---------|----------|----------|----------|")
            for j, nd in enumerate(inst.get("node_details", []), 1):
                raw = nd.get("raw_call", "").replace("|", "\\|")[:60]
                name = nd['name']
                ot = nd['op_type']
                inp = nd.get('inputs', '-')
                sl = nd.get('source_line', 0)
                lines.append(f"| {j} | {name} | {ot} | {inp} | L{sl} | `{raw}` |")
        return lines


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="图结构分析脚本 (支持 GE pbtxt 和 PyTorch FX Graph)")
    parser.add_argument("graph_file", help="图文件路径 (.pbtxt 或 *_runnable.py)")
    parser.add_argument("--output", "-o", default=None, help="Markdown 报告输出路径")
    parser.add_argument("--profiling-dir", default="", help="MindStudio Profiler 输出目录路径")
    args = parser.parse_args()

    if not os.path.exists(args.graph_file):
        logging.error(f"错误: 文件不存在: {args.graph_file}")
        return 1

    graph_type = detect_graph_type(args.graph_file)
    logging.info(f"解析文件: {args.graph_file}")
    logging.info(f"文件类型: {graph_type}")
    file_size_mb = os.path.getsize(args.graph_file) / (1024 * 1024)
    logging.info(f"文件大小: {file_size_mb:.2f} MB")
    if args.profiling_dir:
        logging.info(f"Profiling: {args.profiling_dir}")
    logging.info("")

    if graph_type == "pbtxt":
        p = PbtxtParser()
        nodes = p.parse_file(args.graph_file)
        logging.info(f"解析完成: {len(nodes)} 个顶级节点")
        analyzer = GraphAnalyzer(nodes, "pbtxt", profiling_dir=args.profiling_dir)
        analyzer.ir_version = p.ir_version
        analyzer.producer_name = p.producer_name
    else:
        p = FXGraphParser()
        nodes = p.parse_file(args.graph_file)
        logging.info(f"解析完成: {len(nodes)} 个算子节点")
        analyzer = GraphAnalyzer(nodes, "fxgraph", profiling_dir=args.profiling_dir)

    if analyzer.profiling_loader and analyzer.profiling_loader.loaded:
        msg = f"Profiling 加载成功: {len(analyzer.profiling_loader.op_duration_map)} 个算子耗时记录"
        logging.info(msg)
    elif args.profiling_dir:
        logging.warning(f"警告: Profiling 目录未找到有效数据文件: {args.profiling_dir}")
    else:
        logging.info("未指定 Profiling 目录，将按算子数×重复次数排序")
    logging.info("")

    try:
        md_report = analyzer.format_markdown_report()
    except Exception as e:
        logging.error(f"错误: 生成报告时发生异常: {e}")
        import traceback
        traceback.print_exc()
        return 1
    logging.info(md_report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as out_f:
            out_f.write(md_report)
        logging.info(f"\n报告已写入: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
