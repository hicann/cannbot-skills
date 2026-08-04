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
"""GE/CANN 融合 pass 输入适配（内部里程碑 R3，不是版本号）。

把用户提供的模型/dump 识别成统一图表达，产出需求分析与开发所需的证据产物：
  - input-inventory.json   输入类型、文件 hash、节点/边/子图概览、缺失信息
  - normalized-graph.json  统一节点/数据边/控制边/端口/属性/shape/dtype/format（schema 见 references/normalized-graph-schema.md）
  - artifacts/repro/       skill 生成的最小复现脚本/模型，绝不覆盖用户 data/
  - provenance.json        来源、转换命令、工具版本、假设、复现级别

子命令：
  detect <file>            检测输入类型 + 文件完整性 + hash → input-inventory.json
  inventory <file>         解析图 → normalized-graph.json + input-inventory.json
  repro <file>             生成最小复现到 artifacts/repro/（含 provenance）
  compile <file>           触发 atc/pyatc，记录 OM、dump、完整日志
  provenance <file>        产/补 provenance.json

宿主中立：CANN 路径从 env 展开（ASCEND_HOME_PATH/ASCEND_OPP_PATH），不硬编码默认安装路径。
降级铁律：onnx 库不在 / atc 不在 / 文件解析失败 → 如实标 NOT_RUN + 原因，不伪造。
"""
import argparse
import hashlib
import importlib.util
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = "1.0"
LOGGER = logging.getLogger(__name__)
OUTPUT_LOGGER = logging.getLogger(f"{__name__}.stdout")
OUTPUT_LOGGER.setLevel(logging.INFO)
OUTPUT_LOGGER.propagate = False


class CliError(RuntimeError):
    """Expected command-line failure with a stable exit code."""

    def __init__(self, message, exit_code=1):
        super().__init__(message)
        self.exit_code = exit_code


def _emit_json(value):
    """Write one machine-readable JSON value to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    OUTPUT_LOGGER.handlers = [handler]
    OUTPUT_LOGGER.info("%s", json.dumps(value, ensure_ascii=False, indent=2))


def die(msg, code=1):
    raise CliError(msg, code)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _pass_load_snapshot(root):
    """Hash the currently visible custom pass files without modifying them."""
    root = Path(root).resolve()
    if not root.is_dir():
        return {"status": "NOT_RUN", "root": str(root), "reason": "加载根不存在", "files": []}
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".so", ".py"}:
            files.append({"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size})
    return {"status": "PASSED", "root": str(root), "files": files}


def _pass_load_delta(before, after):
    before_files = {item["path"]: item["sha256"] for item in before.get("files", [])}
    after_files = {item["path"]: item["sha256"] for item in after.get("files", [])}
    return {
        "added": sorted(set(after_files) - set(before_files)),
        "removed": sorted(set(before_files) - set(after_files)),
        "changed": sorted(path for path in set(before_files) & set(after_files)
                          if before_files[path] != after_files[path]),
    }


def detect_input_type(path):
    """按扩展名 + 魔数判输入类型。返回 (input_type, reason)。"""
    p = Path(path)
    if not p.exists():
        return None, f"文件不存在: {path}"
    suf = p.suffix.lower()
    if suf == ".onnx":
        # 校验是否真 onnx
        try:
            import onnx  # noqa: F401
            return "onnx", "扩展名 .onnx + onnx 库可用"
        except ImportError:
            return "onnx", "扩展名 .onnx（onnx 库不可用，解析将降级）"
    if suf == ".air":
        return "air", "扩展名 .air（MindSpore IR）"
    if suf in (".pbtxt", ".pb.txt") or (suf == ".txt" and "pbtxt" in p.name.lower()):
        return "pbtxt", "扩展名 .pbtxt（GE dump 文本）"
    if suf in (".py",):
        text = p.read_text(errors="ignore")
        if "torch" in text and "npu" in text.lower():
            return "torch_script", "Torch/GE IR 脚本"
        if "tensorflow" in text or "tf." in text:
            return "tf_script", "TensorFlow 脚本"
        return "script", "Python 脚本（未识别框架）"
    if suf == ".pb":
        return "tf_pb", "TensorFlow PB（可用 ATC framework=3 直接编译；脚本生成/在线运行仍需 TF1 runtime）"
    # 无扩展名或未知：尝试由 onnx 直接解析。
    try:
        import onnx
        onnx.load(path)
        return "onnx", "内容可被 onnx.load 解析"
    except Exception as exc:
        LOGGER.debug("unknown input is not ONNX: %s", exc)
    # 试 pbtxt（文本，含 op_type:）
    try:
        text = p.read_text(errors="ignore")
        if "op_type:" in text or "node {" in text:
            return "pbtxt", "内容含 GE pbtxt 标记"
    except (OSError, UnicodeError) as exc:
        LOGGER.debug("unknown input cannot be read as pbtxt: %s", exc)
    return None, f"无法识别输入类型: {path}"


# ── ONNX 解析 ──────────────────────────────────────────────
def _load_onnx_model(path):
    try:
        import onnx
        from onnx import shape_inference
    except ImportError as exc:
        raise RuntimeError(f"onnx 库不可用：{exc}") from exc
    model = onnx.load(path)
    try:
        return shape_inference.infer_shapes(model)
    except Exception as exc:
        LOGGER.debug("ONNX shape inference failed; shapes remain partial: %s", exc)
        return model


def _onnx_data_nodes(graph, tensor_sources):
    nodes = []
    for index, input_value in enumerate(graph.input):
        node_id = f"n_in{index}"
        tensor_sources[input_value.name] = (node_id, 0)
        dtype = _onnx_dtype(input_value.type)
        shape = _onnx_shape(input_value.type)
        nodes.append({
            "id": node_id,
            "op_type": "ge:Data",
            "name": input_value.name,
            "outputs": [{"port": 0, "name": input_value.name}],
            "dtype": {input_value.name: dtype} if dtype else {},
            "shape": {input_value.name: shape} if shape else {},
        })
    return nodes


def _onnx_initializer_dtype(initializer):
    try:
        from onnx import TensorProto

        return TensorProto.DataType.Name(initializer.data_type)
    except Exception as exc:
        LOGGER.debug("cannot resolve initializer dtype for %s: %s", initializer.name, exc)
        return None


def _onnx_initializer_nodes(graph, tensor_sources):
    nodes = []
    # 较新的 ONNX 模型通常不把 initializer 同时列作 graph.input。把它们作为
    # Const 节点保留，才能表达 initializer → 算子的依赖边。
    for index, initializer in enumerate(graph.initializer):
        if initializer.name in tensor_sources:
            continue
        node_id = f"n_const{index}"
        tensor_sources[initializer.name] = (node_id, 0)
        dtype = _onnx_initializer_dtype(initializer)
        nodes.append({
            "id": node_id,
            "op_type": "ge:Const",
            "original_op_type": "Constant",
            "name": initializer.name,
            "outputs": [{"port": 0, "name": initializer.name}],
            "dtype": {initializer.name: dtype} if dtype else {},
            "shape": {initializer.name: list(initializer.dims)} if initializer.dims else {},
            "unrepresentable": [],
        })
    return nodes


def _onnx_input_references(node, index, node_id, tensor_sources, missing):
    inputs = []
    edges = []
    display_name = node.name or f"{node.op_type}_{index}"
    for port, source_name in enumerate(node.input):
        if not source_name:
            continue
        producer = tensor_sources.get(source_name)
        if producer is None:
            missing.append({
                "element": f"{display_name}.input[{port}]",
                "reason": f"ONNX tensor {source_name!r} 没有 producer",
                "assumption": "作为不完整输入处理，不伪造连接",
            })
            continue
        inputs.append({
            "port": port,
            "from_node": producer[0],
            "from_port": producer[1],
        })
        edges.append({
            "from_node": producer[0],
            "from_port": producer[1],
            "to_node": node_id,
            "to_port": port,
        })
    return inputs, edges


def _onnx_compute_nodes(graph, tensor_sources, missing):
    nodes = []
    edges = []
    for index, node in enumerate(graph.node):
        node_id = f"n{index}"
        inputs, node_edges = _onnx_input_references(
            node,
            index,
            node_id,
            tensor_sources,
            missing,
        )
        outputs = [{"port": port, "name": name} for port, name in enumerate(node.output)]
        attrs = {attribute.name: _onnx_attr(attribute) for attribute in node.attribute}
        nodes.append({
            "id": node_id,
            "op_type": node.op_type,
            "original_op_type": node.op_type,
            "name": node.name or f"{node.op_type}_{index}",
            "inputs": inputs,
            "outputs": outputs,
            "attrs": attrs,
            "optional_inputs_present": [],
            "unrepresentable": [],
        })
        edges.extend(node_edges)
        for port, output_name in enumerate(node.output):
            if output_name:
                tensor_sources[output_name] = (node_id, port)
    return nodes, edges


def _onnx_graph_outputs(graph, tensor_sources, missing):
    outputs = []
    for output in graph.output:
        producer = tensor_sources.get(output.name)
        if producer is None:
            missing.append({
                "element": f"graph.output:{output.name}",
                "reason": "ONNX graph output 没有 producer",
                "assumption": "输出签名不完整，不伪造来源端口",
            })
            continue
        outputs.append({
            "name": output.name,
            "from_node": producer[0],
            "from_port": producer[1],
        })
    return outputs


def parse_onnx(path):
    """ONNX → normalized-graph dict。onnx 库不可用 → 抛 RuntimeError 让上层降级。"""
    model = _load_onnx_model(path)
    tensor_sources = {}
    missing = []
    nodes = _onnx_data_nodes(model.graph, tensor_sources)
    nodes.extend(_onnx_initializer_nodes(model.graph, tensor_sources))
    compute_nodes, edges = _onnx_compute_nodes(model.graph, tensor_sources, missing)
    nodes.extend(compute_nodes)
    graph_outputs = _onnx_graph_outputs(model.graph, tensor_sources, missing)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "input_type": "onnx",
            "file": str(path),
            "sha256": sha256_file(path),
            "reproduction_level": "semantic",
        },
        "nodes": nodes,
        "data_edges": edges,
        "control_edges": [],
        "outputs": graph_outputs,
        "unrepresentable": [],
        "missing": missing,
    }


def _onnx_dtype(type_proto):
    try:
        from onnx import TensorProto
        return TensorProto.DataType.Name(type_proto.tensor_type.elem_type)
    except Exception as exc:
        LOGGER.debug("cannot decode ONNX dtype: %s", exc)
        return None


def _onnx_shape(type_proto):
    try:
        dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in type_proto.tensor_type.shape.dim]
        return dims if dims else None
    except Exception as exc:
        LOGGER.debug("cannot decode ONNX shape: %s", exc)
        return None


def _onnx_attr(a):
    try:
        from onnx import AttributeProto
        t = a.type
        if t == AttributeProto.INT:
            return a.i
        if t == AttributeProto.FLOAT:
            return a.f
        if t == AttributeProto.STRING:
            return a.s.decode(errors="ignore")
        if t == AttributeProto.INTS:
            return list(a.ints)
        if t == AttributeProto.FLOATS:
            return list(a.floats)
    except Exception as exc:
        LOGGER.debug("cannot decode ONNX attribute %s: %s", getattr(a, "name", "<unknown>"), exc)
    return f"<attr type={a.type}>"


_NODE_BLOCK_START = re.compile(r"^\s*node\s*\{")
_QUOTED_FIELD = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"((?:\\.|[^"\\])*)"')
_ATTRIBUTE_BLOCK_START = re.compile(r"^\s*attribute\s*\{")
_GRAPH_BLOCK_START = re.compile(r"^\s*graph\s*\{")
_GRAPH_OUTPUT_BLOCK_START = re.compile(r"^\s*output\s*\{")
_ATTRIBUTE_QUOTED_FIELD = re.compile(r'^\s*(name|s)\s*:\s*"((?:\\.|[^"\\])*)"')
_ATTRIBUTE_INT_FIELD = re.compile(r"^\s*(i|type)\s*:\s*([^\s]+)")
_ATTRIBUTE_REPEATED_INT = re.compile(r"^\s*ints\s*:\s*(-?\d+)")
_ATTRIBUTE_REPEATED_STRING = re.compile(r'^\s*strings\s*:\s*"((?:\\.|[^"\\])*)"')
_DESCRIPTOR_FIELD = re.compile(
    r"^(input|output)_desc_"
    r"(shape|shape_range|dtype|layout|origin_shape|origin_shape_range|origin_dtype|origin_layout):(\d+)$"
)


def _unescape_proto_string(value):
    """Decode the small protobuf string subset used by GE text dumps."""
    if "\\" not in value:
        return value
    try:
        return json.loads(f'"{value}"')
    except (TypeError, json.JSONDecodeError):
        return value


def _attribute_value(attribute):
    if attribute.get("strings"):
        values = attribute["strings"]
        return values[0] if len(values) == 1 else values
    if attribute.get("ints"):
        values = attribute["ints"]
        return values[0] if len(values) == 1 else values
    if attribute.get("s") is not None:
        return attribute["s"]
    if attribute.get("i") is not None:
        return attribute["i"]
    return None


@dataclass
class _GraphOutputState:
    depth: int = 0
    in_graph: bool = False
    output_depth: Optional[int] = None
    current_name: Optional[str] = None
    done: bool = False


def _consume_graph_output_line(state, line):
    completed_name = None
    if not state.in_graph:
        if _GRAPH_BLOCK_START.match(line):
            state.in_graph = True
            state.depth = _brace_delta(line)
        return None
    if state.output_depth is None and state.depth == 1:
        if _GRAPH_OUTPUT_BLOCK_START.match(line):
            state.output_depth = state.depth + _brace_delta(line)
            state.current_name = None
    elif state.output_depth is not None:
        match = _QUOTED_FIELD.match(line)
        if match and match.group(1) == "name" and state.current_name is None:
            state.current_name = _unescape_proto_string(match.group(2))
        if line.strip() == "}" and state.depth == state.output_depth:
            completed_name = state.current_name
            state.output_depth = None
            state.current_name = None
    state.depth += _brace_delta(line)
    state.done = state.in_graph and state.depth == 0
    return completed_name


def _proto_graph_outputs(path):
    """Read top-level GraphDef output names without loading a large dump."""
    outputs = []
    state = _GraphOutputState()
    with Path(path).open(errors="ignore") as handle:
        for line in handle:
            output_name = _consume_graph_output_line(state, line)
            if output_name:
                outputs.append(output_name)
            if state.done:
                break
    return outputs


@dataclass
class _NodeRecordState:
    record: Optional[dict] = None
    depth: int = 0
    current_attribute: Optional[dict] = None


def _new_node_record():
    return {
        "names": [],
        "op_types": [],
        "inputs": [],
        "outputs": [],
        "attributes": [],
        "has_attribute": False,
    }


def _start_node_record(line):
    match = _NODE_BLOCK_START.match(line)
    if not match:
        return None, 0
    open_index = line.find("{", match.start(), match.end())
    return _new_node_record(), _brace_delta(line[open_index:])


def _new_attribute_record():
    return {
        "name": None,
        "s": None,
        "i": None,
        "strings": [],
        "ints": [],
        "type": None,
    }


def _append_record_field(record, line):
    match = _QUOTED_FIELD.match(line)
    if not match:
        return
    field, raw_value = match.groups()
    value = _unescape_proto_string(raw_value)
    destinations = {
        "name": "names",
        "op_type": "op_types",
        "type": "op_types",
        "input": "inputs",
        "output": "outputs",
    }
    destination = destinations.get(field)
    if destination:
        record.get(destination, []).append(value)


def _update_attribute(attribute, line):
    quoted = _ATTRIBUTE_QUOTED_FIELD.match(line)
    if quoted:
        field, raw_value = quoted.groups()
        attribute[field] = _unescape_proto_string(raw_value)
    integer = _ATTRIBUTE_INT_FIELD.match(line)
    if integer:
        field, value = integer.groups()
        attribute[field] = int(value) if field == "i" else value
    repeated_int = _ATTRIBUTE_REPEATED_INT.match(line)
    if repeated_int:
        attribute.get("ints", []).append(int(repeated_int.group(1)))
    repeated_string = _ATTRIBUTE_REPEATED_STRING.match(line)
    if repeated_string:
        value = _unescape_proto_string(repeated_string.group(1))
        attribute.get("strings", []).append(value)


def _finish_attribute(record, attribute):
    if attribute is not None and attribute.get("name"):
        record.get("attributes", []).append(attribute)


def _consume_node_line(state, line):
    if state.record is None:
        state.record, state.depth = _start_node_record(line)
        if state.record is not None and state.depth == 0:
            completed = state.record
            state.record = None
            return completed
        return None
    if state.depth == 1:
        if _ATTRIBUTE_BLOCK_START.match(line):
            state.current_attribute = _new_attribute_record()
            state.record.update({"has_attribute": True})
        _append_record_field(state.record, line)
    elif state.depth == 2 and state.current_attribute is not None:
        _update_attribute(state.current_attribute, line)
        if line.strip() == "}":
            _finish_attribute(state.record, state.current_attribute)
            state.current_attribute = None
    state.depth += _brace_delta(line)
    if state.depth != 0:
        return None
    _finish_attribute(state.record, state.current_attribute)
    completed = state.record
    state.record = None
    state.current_attribute = None
    return completed


def _proto_node_records(path):
    """单次流式抽取 GE node 顶层字段，适用于数百 MB 的 dump。"""
    state = _NodeRecordState()
    with Path(path).open(errors="ignore") as handle:
        for line in handle:
            completed = _consume_node_line(state, line)
            if completed is not None:
                yield completed


def _brace_delta(text):
    """统计 protobuf 单行中的花括号增量，忽略字符串内花括号。"""
    if "{" not in text and "}" not in text:
        return 0
    if '"' not in text:
        return text.count("{") - text.count("}")
    delta = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            delta += 1
        elif char == "}":
            delta -= 1
    return delta


def _parse_pbtxt_input(value):
    """解析 GE/GraphDef input 的 node[:out_port] 或 ^node 控制边写法。"""
    control = value.startswith("^")
    raw = value[1:] if control else value
    match = re.match(r"^(.*):(-?\d+)$", raw)
    if match:
        node_name = match.group(1)
        output_port = int(match.group(2))
        if output_port < 0:
            control = True
    else:
        node_name = raw
        output_port = 0
    return node_name, output_port, control


def _parse_pbtxt_port(value):
    match = re.match(r"^(.*):(-?\d+)$", value)
    if match:
        return _unescape_proto_string(match.group(1)), int(match.group(2))
    return _unescape_proto_string(value), 0


_OPTIONAL_INPUT_HINTS = {"bias", "offset", "offset_w", "scale", "mean", "variance", "gamma", "beta"}
_BOOKKEEPING_ATTRIBUTES = {
    "id", "index", "stream_id", "workspace", "workspace_bytes", "input_i", "output_i", "is_input_const",
}


def _append_semantic_attribute(attributes, name, value):
    if value is None:
        return
    if name not in attributes:
        attributes[name] = value
        return
    previous = attributes[name]
    attributes[name] = previous if isinstance(previous, list) else [previous]
    attributes[name].append(value)


def _is_bookkeeping_attribute(name):
    prefixes = ("_", "input_desc_", "output_desc_", "dst_", "src_")
    return name.startswith(prefixes) or name in _BOOKKEEPING_ATTRIBUTES


def _collect_descriptor_attributes(record):
    descriptors = {}
    attributes = {}
    optional_keys = []
    optional_values = []
    for attribute in record.get("attributes", []):
        name = attribute.get("name")
        if not name:
            continue
        if name == "_input_name_key":
            optional_keys = attribute.get("strings", [])
            continue
        if name == "_input_name_value":
            optional_values = attribute.get("ints", [])
            continue
        match = _DESCRIPTOR_FIELD.match(name)
        value = _attribute_value(attribute)
        if match:
            direction, descriptor_name, index = match.groups()
            descriptors[(direction, descriptor_name, int(index))] = value
        elif not _is_bookkeeping_attribute(name):
            _append_semantic_attribute(attributes, name, value)
    return descriptors, attributes, optional_keys, optional_values


def _store_descriptor(maps, descriptors, direction, descriptor_name, index):
    key = f"{direction}:{index}"
    value = descriptors[(direction, descriptor_name, index)]
    if value in (None, []):
        return
    if descriptor_name in {"shape", "shape_range"}:
        maps[descriptor_name][key] = value if isinstance(value, list) else [value]
    elif descriptor_name == "dtype":
        maps["dtype"][key] = value
    elif descriptor_name == "layout":
        layout = {"storage": value}
        origin = descriptors.get((direction, "origin_layout", index))
        if origin is not None:
            layout["origin"] = origin
        maps["format"][key] = layout
    elif descriptor_name.startswith("origin_"):
        target = {"origin_shape": "shape", "origin_shape_range": "shape_range", "origin_dtype": "dtype",
                  "origin_layout": "format"}[descriptor_name]
        if key not in maps[target]:
            if target == "format":
                maps[target][key] = {"origin": value}
            elif target in {"shape", "shape_range"}:
                maps[target][key] = value if isinstance(value, list) else [value]
            else:
                maps[target][key] = value


def _descriptor_maps(descriptors):
    maps = {"shape": {}, "shape_range": {}, "dtype": {}, "format": {}}
    for direction, descriptor_name, index in sorted(descriptors):
        _store_descriptor(maps, descriptors, direction, descriptor_name, index)
    return maps


def _present_optional_inputs(record, aliases, positions):
    inputs = record.get("inputs", [])
    present = []
    for alias, position in zip(aliases, positions):
        valid_position = 0 <= position < len(inputs)
        if alias.lower() in _OPTIONAL_INPUT_HINTS and valid_position and inputs[position]:
            present.append(alias)
    return sorted(set(present))


def _descriptor_fields(record):
    """Extract stable tensor descriptors and semantic attributes from a GE node."""
    descriptors, attributes, aliases, positions = _collect_descriptor_attributes(record)
    fields = _descriptor_maps(descriptors)
    fields["attrs"] = attributes
    fields["optional_inputs_present"] = _present_optional_inputs(record, aliases, positions)
    return fields


# ── pbtxt 解析（GE dump 文本，保守的结构抽取）──────────────
@dataclass
class _PbtxtState:
    nodes: list = dataclass_field(default_factory=list)
    name_to_id: dict = dataclass_field(default_factory=dict)
    raw_inputs: dict = dataclass_field(default_factory=dict)
    raw_outputs: dict = dataclass_field(default_factory=dict)
    raw_attributes: dict = dataclass_field(default_factory=dict)
    data_edges: list = dataclass_field(default_factory=list)
    control_edges: list = dataclass_field(default_factory=list)
    unrepresentable: list = dataclass_field(default_factory=list)
    missing: list = dataclass_field(default_factory=list)
    has_attributes: bool = False


def _add_pbtxt_node(state, record, index):
    names = record.get("names", [])
    op_types = record.get("op_types", [])
    name = names[0] if names else f"node_{index}"
    op_type = op_types[0] if op_types else "Unknown"
    op_type = op_type if op_type.startswith("ge:") else f"ge:{op_type}"
    node_id = f"n{index}"
    if name in state.name_to_id:
        state.unrepresentable.append({
            "element": name,
            "reason": "pbtxt 存在重名节点，无法建立唯一结构映射",
            "degradation": "保留原 pbtxt，使用 ES/GE IR 复现",
        })
    state.name_to_id[name] = node_id
    descriptors = _descriptor_fields(record)
    state.nodes.append({
        "id": node_id,
        "op_type": op_type,
        "name": name,
        "inputs": [],
        "outputs": [],
        "attrs": descriptors.get("attrs", {}),
        "shape": descriptors.get("shape", {}),
        "shape_range": descriptors.get("shape_range", {}),
        "dtype": descriptors.get("dtype", {}),
        "format": descriptors.get("format", {}),
        "optional_inputs_present": descriptors.get("optional_inputs_present", []),
        "unrepresentable": [],
    })
    state.raw_inputs[node_id] = [_parse_pbtxt_input(value) for value in record.get("inputs", [])]
    state.raw_outputs[node_id] = [_parse_pbtxt_port(value) for value in record.get("outputs", [])]
    state.raw_attributes[node_id] = record.get("attributes", [])
    state.has_attributes = state.has_attributes or record.get("has_attribute", False)


def _load_pbtxt_nodes(path):
    state = _PbtxtState()
    for index, record in enumerate(_proto_node_records(path)):
        _add_pbtxt_node(state, record, index)
    return state


def _pbtxt_output_ports(state):
    output_ports = {}
    for node in state.nodes:
        node_id = node.get("id")
        ports = {port for _, port in state.raw_outputs.get(node_id, [])}
        output_ports[node_id] = ports or {0}
    return output_ports


def _connect_pbtxt_node(state, node, output_ports):
    node_id = node.get("id")
    data_port = 0
    for source_name, source_port, is_control in state.raw_inputs.get(node_id, []):
        if not source_name:
            continue
        source_id = state.name_to_id.get(source_name)
        if source_id is None:
            state.missing.append({
                "element": source_name,
                "reason": "输入来源节点不在 pbtxt 片段内",
                "assumption": "仅结构复现时作为外部边界，不伪造内部连接",
            })
        elif is_control:
            state.control_edges.append({"from_node": source_id, "to_node": node_id})
        else:
            node.get("inputs", []).append({
                "port": data_port, "from_node": source_id, "from_port": source_port,
            })
            state.data_edges.append({
                "from_node": source_id, "from_port": source_port, "to_node": node_id, "to_port": data_port,
            })
            output_ports.get(source_id, set()).add(source_port)
            data_port += 1


def _pbtxt_edges(state):
    output_ports = _pbtxt_output_ports(state)
    for node in state.nodes:
        _connect_pbtxt_node(state, node, output_ports)
    for node in state.nodes:
        node_id = node.get("id")
        outputs = [
            {"port": port, "name": f"{node.get('name')}:{port}"}
            for port in sorted(output_ports.get(node_id, {0}))
        ]
        node.update({"outputs": outputs})


def _pbtxt_output_aliases(state, net_output_id):
    aliases = {}
    pattern = re.compile(r"^input_desc_attr__origin_output_tensor_name:(\d+)$")
    for attribute in state.raw_attributes.get(net_output_id, []):
        match = pattern.match(attribute.get("name", ""))
        value = _attribute_value(attribute) if match else None
        if value:
            aliases[int(match.group(1))] = value
    return aliases


def _pbtxt_graph_outputs(path, state):
    output_names = _proto_graph_outputs(path)
    net_output = next((node for node in state.nodes if node.get("op_type") == "ge:NetOutput"), None)
    if not output_names or net_output is None:
        return []
    net_output_id = net_output.get("id")
    net_inputs = []
    for item in state.raw_inputs.get(net_output_id, []):
        if item[0] and not item[2]:
            net_inputs.append(item)
    aliases = _pbtxt_output_aliases(state, net_output_id)
    outputs = []
    for index, output_name in enumerate(output_names[:len(net_inputs)]):
        source_name, source_port, _ = net_inputs[index]
        source_id = state.name_to_id.get(source_name)
        if source_id is not None:
            outputs.append({
                "name": aliases.get(index, output_name), "from_node": source_id, "from_port": source_port,
            })
    return outputs


def parse_pbtxt(path):
    """GE pbtxt → normalized-graph，保留数据端口、控制边和不完整信息。"""
    state = _load_pbtxt_nodes(path)
    if not state.nodes:
        raise RuntimeError("pbtxt 未找到 node { ... } 块，无法提取结构")
    _pbtxt_edges(state)
    if not state.has_attributes:
        state.missing.append({
            "element": "graph", "reason": "pbtxt 未含可解析属性值", "assumption": "结构复现，不承诺语义",
        })
    outputs = _pbtxt_graph_outputs(path, state)
    if not outputs:
        state.missing.append({
            "element": "graph.outputs",
            "reason": "pbtxt 片段未提供可可靠识别的 graph output 顺序和来源端口",
            "assumption": "验证证据阶段的输出签名检查标 NOT_RUN；不以末端节点猜测图输出",
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "input_type": "pbtxt", "file": str(path), "sha256": sha256_file(path),
            "reproduction_level": "structural",
        },
        "nodes": state.nodes,
        "data_edges": state.data_edges,
        "control_edges": state.control_edges,
        "outputs": outputs,
        "unrepresentable": state.unrepresentable,
        "missing": state.missing,
    }


# ── AIR：环境约束降级 ──────────────────────────────────────
def parse_air(path):
    """AIR 需 MindSpore，环境未装时降级。"""
    try:
        import mindspore  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "AIR 解析需 MindSpore，当前环境未安装。降级：AIR 走 atc --mode=0 --framework=1 直接编译触发 pass，"
            "不在此解析为 normalized-graph。atc 编译 AIR 生成 OM 同样需 .air 文件，无样例则 NOT_RUN。"
        ) from exc
    raise RuntimeError("MindSpore 在位但 AIR 解析未实现（输入适配阶段范围外，走 atc 直接编译）")


PARSERS = {"onnx": parse_onnx, "air": parse_air, "pbtxt": parse_pbtxt}


# ── 子命令 ─────────────────────────────────────────────────
def cmd_detect(args):
    itype, reason = detect_input_type(args.file)
    inv = {
        "input_type": itype,
        "file": str(args.file),
        "sha256": sha256_file(args.file) if Path(args.file).exists() else None,
        "exists": Path(args.file).exists(),
        "detection_reason": reason,
        "size_bytes": Path(args.file).stat().st_size if Path(args.file).exists() else None,
        "node_count": None, "edge_count": None, "missing": [],
        "status": "DETECTED" if itype else "NOT_RUN",
    }
    _write(args.out or "input-inventory.json", inv)
    _emit_json(inv)
    return 0


def cmd_inventory(args):
    itype, reason = detect_input_type(args.file)
    if not itype:
        _write(args.out_inv or "input-inventory.json",
               {"input_type": None, "status": "NOT_RUN", "reason": reason})
        die(f"无法识别输入：{reason}")
    parser = PARSERS.get(itype)
    try:
        ng = parser(args.file)
    except RuntimeError as exc:
        # 降级：记 inventory，normalized-graph 标 NOT_RUN
        inv = {"input_type": itype, "file": str(args.file), "sha256": sha256_file(args.file),
               "status": "NOT_RUN", "reason": str(exc), "node_count": None, "edge_count": None, "missing": []}
        _write(args.out_inv or "input-inventory.json", inv)
        _write(args.out_ng or "normalized-graph.json",
               {"schema_version": SCHEMA_VERSION, "source": {"input_type": itype, "file": str(args.file)},
                "status": "NOT_RUN", "reason": str(exc)})
        _emit_json(inv)
        return 0
    _write(args.out_ng or "normalized-graph.json", ng)
    source = ng.get("source", {})
    inv = {
        "input_type": itype, "file": str(args.file), "sha256": source.get("sha256"),
        "detection_reason": reason, "node_count": len(ng.get("nodes", [])),
        "edge_count": len(ng.get("data_edges", [])) + len(ng.get("control_edges", [])),
        "control_edge_count": len(ng.get("control_edges", [])),
        "missing": ng.get("missing", []), "unrepresentable": ng.get("unrepresentable", []),
        "reproduction_level": source.get("reproduction_level"),
        "status": "OK",
    }
    _write(args.out_inv or "input-inventory.json", inv)
    _emit_json(inv)
    return 0


def cmd_repro(args):
    """生成最小复现到 artifacts/repro/。绝不覆盖用户 data/。"""
    itype, _ = detect_input_type(args.file)
    repro_dir = _prepare_repro_directory(args.repro_dir)
    prov = _repro_provenance(args.file, itype, repro_dir)
    handlers = {"onnx": _repro_onnx, "pbtxt": _repro_pbtxt, "air": _repro_air}
    handler = handlers.get(itype, _repro_unsupported)
    handler(args, repro_dir, prov)
    _write(args.out_prov or "provenance.json", prov)
    _emit_json(prov)
    return 0


def _prepare_repro_directory(value):
    repro_dir = Path(value or "artifacts/repro")
    user_data = Path("data")
    if user_data.exists() and repro_dir.resolve().is_relative_to(user_data.resolve()):
        die("artifacts/repro/ 不得落在用户 data/ 内（文件所有权规则）")
    repro_dir.mkdir(parents=True, exist_ok=True)
    return repro_dir


def _repro_provenance(source_file, input_type, repro_dir):
    repro_command = ["python3", str(Path(__file__).resolve()), "repro", str(Path(source_file).resolve()),
                     "--repro-dir", str(repro_dir.resolve())]
    return {
        "source_file": str(source_file), "input_type": input_type,
        "sha256": sha256_file(source_file), "reproduction_level": "structural",
        "tool": "adapt_input.py", "tool_version": SCHEMA_VERSION,
        "transform_commands": [shlex.join(repro_command)], "tool_versions": _tool_versions(),
        "compile_command_templates": [], "artifacts": [],
        "assumptions": [], "status": "OK",
    }


def _repro_onnx(args, repro_dir, provenance):
    destination = repro_dir / "repro.onnx"
    shutil.copy2(args.file, destination)
    copy_command = ["cp", str(Path(args.file).resolve()), str(destination.resolve())]
    provenance.get("transform_commands", []).append(shlex.join(copy_command))
    provenance.get("compile_command_templates", []).append(_compile_template(destination, "onnx", "cpp"))
    provenance.get("artifacts", []).append(str(destination.resolve()))
    provenance.update({"reproduction_level": "semantic"})
    provenance.get("assumptions", []).append("ONNX 最小复现直接复用原模型结构（目标片段抽取在需求阶段框定后做）")


def _repro_pbtxt(args, repro_dir, provenance):
    graph = parse_pbtxt(args.file)
    provenance.get("assumptions", []).append(
        "pbtxt 缺权重/属性，复现用随机权重，仅验目标片段同构，不承诺语义"
    )
    for missing in graph.get("missing", []):
        provenance.get("assumptions", []).append(
            f"{missing.get('element')}: {missing.get('reason')} → {missing.get('assumption')}"
        )
    repro_onnx, reason = _build_minimal_onnx(graph, repro_dir)
    if repro_onnx is None:
        provenance.update({"status": "NOT_RUN", "reason": reason})
        return
    iso_path = Path(args.out_iso or repro_dir / "structural-isomorphism.json")
    isomorphism = _verify_structural_isomorphism(graph, repro_onnx)
    _write(iso_path, isomorphism)
    provenance.get("artifacts", []).extend([str(repro_onnx.resolve()), str(iso_path.resolve())])
    provenance.get("compile_command_templates", []).append(_compile_template(repro_onnx, "onnx", "cpp"))
    provenance["structural_isomorphism"] = {
        "status": isomorphism.get("status"), "evidence": str(iso_path.resolve()),
    }
    if isomorphism.get("status") != "PASSED":
        provenance.update({
            "status": "FAILED", "reason": "生成 ONNX 与原 pbtxt 目标片段未通过结构同构核验",
        })


def _repro_air(args, _repro_dir, provenance):
    provenance.update({
        "status": "NOT_RUN", "reason": "AIR 不转换为其他格式；请用 compile 子命令直接走 atc/pyatc",
    })
    provenance.get("compile_command_templates", []).append(_compile_template(Path(args.file), "air", "cpp"))


def _repro_unsupported(_args, _repro_dir, provenance):
    provenance.update({
        "status": "NOT_RUN", "reason": f"输入类型 {provenance.get('input_type')} 的最小复现未实现",
    })


def _normalized_op_type(node):
    return node.get("original_op_type") or node.get("op_type", "").removeprefix("ge:")


def _node_output_ports(node):
    ports = {item.get("port", 0) for item in node.get("outputs", [])}
    return sorted(ports or {0})


def _tensor_name(node, port):
    return node["name"] if port == 0 and node.get("op_type") == "ge:Data" else f"{node['name']}__out{port}"


_STANDARD_ONNX_OPS = {
    "Abs", "Add", "BatchNormalization", "Cast", "Concat", "Conv", "Div", "Exp", "Flatten",
    "Gemm", "Identity", "MatMul", "Max", "Mul", "Relu", "Reshape", "Sigmoid", "Softmax",
    "Split", "Sub", "Tanh", "Transpose",
}


class _ReproBuildError(RuntimeError):
    """A structural reproduction cannot be represented faithfully."""


def _validate_repro_graph(graph):
    if graph.get("control_edges"):
        raise _ReproBuildError("pbtxt 含控制边，ONNX 前端无法保持该结构；请改用 ES/GE IR 复现")
    if graph.get("unrepresentable"):
        raise _ReproBuildError("pbtxt 含无法建立唯一映射的结构；请保留原 pbtxt 并使用 ES/GE IR")
    missing_source = any(
        item.get("reason") == "输入来源节点不在 pbtxt 片段内" for item in graph.get("missing", [])
    )
    if missing_source:
        raise _ReproBuildError("pbtxt 片段缺少外部边界节点，无法证明生成模型与原片段同构")


def _onnx_inputs(nodes, helper, tensor_proto):
    input_infos = []
    tensors = {}
    for node in nodes:
        for port in _node_output_ports(node):
            tensor = _tensor_name(node, port)
            input_infos.append(helper.make_tensor_value_info(tensor, tensor_proto.FLOAT, [2, 2]))
            tensors[(node.get("id"), port)] = tensor
    return input_infos, tensors


def _resolve_node_inputs(node, tensors):
    input_tensors = []
    refs = sorted(node.get("inputs", []), key=lambda item: item.get("port", 0))
    for expected_port, ref in enumerate(refs):
        if ref.get("port") != expected_port:
            raise _ReproBuildError(f"节点 {node.get('name')} 的输入端口不连续，不能无损生成 ONNX")
        tensor = tensors.get((ref.get("from_node"), ref.get("from_port")))
        if tensor is None:
            return None
        input_tensors.append(tensor)
    return input_tensors


def _append_ready_onnx_nodes(remaining, tensors, helper):
    generated = []
    custom_domain_used = False
    for node in list(remaining):
        input_tensors = _resolve_node_inputs(node, tensors)
        if input_tensors is None:
            continue
        output_ports = _node_output_ports(node)
        if output_ports != list(range(len(output_ports))):
            raise _ReproBuildError(f"节点 {node.get('name')} 的输出端口不连续，不能无损生成 ONNX")
        outputs = [_tensor_name(node, port) for port in output_ports]
        op_type = _normalized_op_type(node)
        domain = "" if op_type in _STANDARD_ONNX_OPS else "ge.structural"
        generated.append(helper.make_node(op_type, input_tensors, outputs, name=node.get("name"), domain=domain))
        custom_domain_used = custom_domain_used or bool(domain)
        for port, tensor in zip(output_ports, outputs):
            tensors[(node.get("id"), port)] = tensor
        remaining.remove(node)
    return generated, custom_domain_used


def _build_onnx_compute_nodes(compute_nodes, tensors, helper):
    generated = []
    custom_domain_used = False
    remaining = list(compute_nodes)
    while remaining:
        ready, uses_custom_domain = _append_ready_onnx_nodes(remaining, tensors, helper)
        if not ready:
            unresolved = ", ".join(node.get("name", "<unknown>") for node in remaining)
            raise _ReproBuildError(f"pbtxt 数据边存在环或未解析 producer，无法生成 ONNX：{unresolved}")
        generated.extend(ready)
        custom_domain_used = custom_domain_used or uses_custom_domain
    return generated, custom_domain_used


def _build_onnx_graph_outputs(graph, compute_nodes, tensors, helper, tensor_proto):
    consumed = {
        (edge.get("from_node"), edge.get("from_port")) for edge in graph.get("data_edges", [])
    }
    outputs = []
    for node in compute_nodes:
        for port in _node_output_ports(node):
            key = (node.get("id"), port)
            if key not in consumed:
                outputs.append(helper.make_tensor_value_info(tensors.get(key), tensor_proto.FLOAT, [2, 2]))
    if not outputs:
        raise _ReproBuildError("pbtxt 未找到可作为最小复现输出的计算端口")
    return outputs


@dataclass
class _StructuralOnnxGraph:
    nodes: list
    input_infos: list
    outputs: list
    custom_domain_used: bool


def _save_structural_onnx(repro_dir, helper, graph_spec):
    import onnx

    opsets = [helper.make_opsetid("", 13)]
    if graph_spec.custom_domain_used:
        opsets.append(helper.make_opsetid("ge.structural", 1))
    graph = helper.make_graph(
        graph_spec.nodes,
        "pbtxt_structural_repro",
        graph_spec.input_infos,
        graph_spec.outputs,
    )
    model = helper.make_model(graph, opset_imports=opsets)
    model.ir_version = 9
    destination = repro_dir / "repro.onnx"
    onnx.save(model, destination)
    return destination


def _build_minimal_onnx(graph, repro_dir):
    """由完整、无控制边的 pbtxt 数据子图生成 ONNX；不能证明时明确降级。"""
    try:
        from onnx import helper, TensorProto
    except ImportError:
        return None, "onnx 库不可用，无法生成并核验 pbtxt 结构复现"
    try:
        _validate_repro_graph(graph)
        nodes = graph.get("nodes", [])
        data_nodes = [node for node in nodes if node.get("op_type") == "ge:Data"]
        compute_nodes = [node for node in nodes if node.get("op_type") != "ge:Data"]
        if not compute_nodes:
            raise _ReproBuildError("pbtxt 不含可复现的计算节点")
        input_infos, tensors = _onnx_inputs(data_nodes, helper, TensorProto)
        onnx_nodes, custom_domain_used = _build_onnx_compute_nodes(compute_nodes, tensors, helper)
        outputs = _build_onnx_graph_outputs(graph, compute_nodes, tensors, helper, TensorProto)
        graph_spec = _StructuralOnnxGraph(
            nodes=onnx_nodes,
            input_infos=input_infos,
            outputs=outputs,
            custom_domain_used=custom_domain_used,
        )
        destination = _save_structural_onnx(repro_dir, helper, graph_spec)
        return destination, None
    except _ReproBuildError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"生成 ONNX 失败：{exc}"


def _structural_signature(graph):
    node_by_id = {node["id"]: node for node in graph.get("nodes", [])}
    nodes = sorted({(node["name"], _normalized_op_type(node)) for node in node_by_id.values()})
    edges = sorted({
        (node_by_id[edge["from_node"]]["name"], edge["from_port"],
         node_by_id[edge["to_node"]]["name"], edge["to_port"])
        for edge in graph.get("data_edges", [])
        if edge["from_node"] in node_by_id and edge["to_node"] in node_by_id
    })
    return {"nodes": [{"name": name, "op_type": op_type} for name, op_type in nodes],
            "data_edges": [
                {"from_node": source, "from_port": source_port,
                 "to_node": target, "to_port": target_port}
                for source, source_port, target, target_port in edges
            ]}


def _verify_structural_isomorphism(source_graph, repro_onnx):
    """以节点名、op type 和带端口的数据边验证 pbtxt 与生成 ONNX 的结构。"""
    try:
        generated_graph = parse_onnx(repro_onnx)
        source_signature = _structural_signature(source_graph)
        generated_signature = _structural_signature(generated_graph)
    except Exception as exc:
        return {"status": "FAILED", "reason": f"无法解析生成 ONNX 进行同构核验：{exc}"}
    status = "PASSED" if source_signature == generated_signature else "FAILED"
    result = {
        "status": status,
        "comparison": "节点 name/op_type 和数据边 src:port → dst:port 精确比较",
        "source": source_signature,
        "generated": generated_signature,
    }
    if status != "PASSED":
        result["reason"] = "节点或带端口数据边不一致，不能声称结构同构"
    return result


def _compile_template(model, input_type, pass_language):
    compiler = "pyatc" if pass_language == "python" else "atc"
    framework = "5" if input_type == "onnx" else "1"
    command = [compiler, f"--model={Path(model).resolve()}", f"--framework={framework}"]
    if input_type == "air":
        command.append("--mode=0")
    command.extend(["--soc_version=<完整 soc_version>", "--output=<输出前缀>"])
    return shlex.join(command)


def _file_snapshot(root):
    if not root.exists():
        return {}
    return {path.resolve(): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def _changed_files(before, root):
    after = _file_snapshot(root)
    return sorted(path for path, signature in after.items() if before.get(path) != signature)


def _compile_not_run(args, result, reason):
    result.update({"status": "NOT_RUN", "reason": reason})
    _write(args.out_evidence or Path(args.work_dir or "artifacts/compile") / "compile-evidence.json", result)
    _emit_json(result)
    return 0


def _find_effect_records(value, path=""):
    records = []
    if isinstance(value, dict):
        match = value.get("match_times")
        effect = value.get("effect_times")
        if match is not None or effect is not None:
            name = value.get("pass") or value.get("pass_name") or value.get("name")
            records.append({
                "pass": str(name or path.rsplit("/", 1)[-1] or "unknown_pass"),
                "match_times": match, "effect_times": effect, "path": path,
            })
        for key, child in value.items():
            records.extend(_find_effect_records(child, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            records.extend(_find_effect_records(child, f"{path}/{index}"))
    return records


def _positive_effect_times(value):
    try:
        return int(value or 0) > 0
    except (TypeError, ValueError):
        return False


def _event_counts(events_path, pass_name):
    counts = {
        "candidate": 0, "matched": 0, "guard_passed": 0, "applied": 0, "replacement_failed": 0, "skip": 0,
    }
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_name = event.get("event")
        if event.get("pass") == pass_name and event_name in counts:
            counts[event_name] += 1
    return counts


def _add_event_evidence(args, evidence):
    if not args.events:
        return False
    events_path = Path(args.events).resolve()
    if not events_path.is_file():
        return False
    counts = _event_counts(events_path, args.pass_name)
    evidence.get("sources", []).append(str(events_path))
    evidence.get("records", []).append({"kind": "events", "counts": counts})
    if counts.get("applied", 0) <= 0:
        return False
    evidence.update({"status": "PASSED", "reason": "目标 pass 有 applied 事件"})
    return True


def _fusion_records(path, pass_name):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    records = []
    for record in _find_effect_records(value):
        record_name = record.get("pass", "")
        if record_name == pass_name or pass_name in record_name:
            records.append(record)
    return records


def _add_fusion_evidence(args, evidence):
    if not args.fusion_result:
        return False
    fusion_path = Path(args.fusion_result).resolve()
    if not fusion_path.is_file():
        return False
    records = _fusion_records(fusion_path, args.pass_name)
    evidence.get("sources", []).append(str(fusion_path))
    evidence.get("records", []).extend({"kind": "fusion_result", **record} for record in records)
    effect_found = any(_positive_effect_times(record.get("effect_times")) for record in records)
    if effect_found:
        evidence.update({
            "status": "PASSED", "reason": "fusion_result.json 记录目标 pass effect_times > 0",
        })
    return effect_found


def _add_pass_log(evidence, log_path, pass_name):
    if not log_path.is_file():
        return
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    evidence["log"] = {
        "path": str(log_path), "pass_name_found": pass_name in log_text,
        "bytes": len(log_text.encode("utf-8")),
    }


def _pass_effect_evidence(args, log_path):
    """Read structured effect evidence; logs alone never prove replacement."""
    pass_name = args.pass_name
    evidence = {"status": "NOT_RUN", "pass_name": pass_name, "sources": [], "records": []}
    if not pass_name:
        evidence["reason"] = "未提供 --pass-name，无法把 effect 归属到目标 pass"
        return evidence
    if _add_event_evidence(args, evidence) or _add_fusion_evidence(args, evidence):
        return evidence
    if evidence.get("sources"):
        evidence.update({
            "status": "FAILED",
            "reason": "已提供结构化 pass 证据，但未发现目标 pass 的 applied/effect_times > 0",
        })
    else:
        evidence["reason"] = "未提供 events 或 fusion_result.json；日志和 dump 存在不等于 pass 命中"
    _add_pass_log(evidence, log_path, pass_name)
    return evidence


class _CompileUnavailable(RuntimeError):
    """Compilation cannot run because a required input or runtime is absent."""


def _compile_result(args, input_type):
    return {
        "input_file": str(args.file),
        "input_type": input_type,
        "pass_language": args.pass_language,
        "status": "NOT_RUN",
        "tool_versions": _tool_versions(),
        "env": _env_snapshot(),
    }


def _resolve_compiler(args):
    compiler_kind = "pyatc" if args.pass_language == "python" else "atc"
    compiler_path = shutil.which(compiler_kind)
    compiler = shlex.split(args.compiler) if args.compiler else ([compiler_path] if compiler_path else [])
    if not compiler:
        raise _CompileUnavailable(f"未找到 {compiler_kind}；请设置 PATH 或传 --compiler")
    if args.pass_language == "python" and not args.compiler:
        runtime = _python_pass_runtime()
        missing = [name for name, value in runtime.items() if value == "NOT_AVAILABLE"]
        if missing:
            raise _CompileUnavailable("Python pass runtime 缺少：" + ", ".join(missing))
    return compiler_kind, compiler


def _compile_environment(args):
    environment = os.environ.copy()
    environment.setdefault("DUMP_GE_GRAPH", "1")
    environment.setdefault("DUMP_GRAPH_LEVEL", "1")
    if args.pass_language == "python":
        py_pass_path = args.py_pass_path or environment.get("ASCEND_GE_PY_PASS_PATH")
        if not py_pass_path:
            raise _CompileUnavailable("Python pass 缺 ASCEND_GE_PY_PASS_PATH 或 --py-pass-path")
        environment["ASCEND_GE_PY_PASS_PATH"] = py_pass_path
    return environment


def _compile_command(args, input_type, compiler):
    work_dir = Path(args.work_dir or "artifacts/compile").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else work_dir / "model"
    output = output if output.is_absolute() else work_dir / output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    framework = {"onnx": "5", "air": "1", "tf_pb": "3"}.get(input_type)
    command = compiler + [f"--model={Path(args.file).resolve()}", f"--framework={framework}"]
    if input_type == "air":
        command.append("--mode=0")
    command.extend([f"--soc_version={args.soc_version}", f"--output={output}"])
    return work_dir, command


def _run_compiler(command, work_dir, environment, compiler_kind):
    try:
        completed = subprocess.run(
            command, cwd=work_dir, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
    except OSError as exc:
        raise _CompileUnavailable(f"无法启动 {compiler_kind}：{exc}") from exc
    log_path = work_dir / "compile.log"
    log_path.write_text(f"command: {shlex.join(command)}\n\n{completed.stdout or ''}", encoding="utf-8")
    return completed, log_path


def _compile_artifacts(before, work_dir):
    artifacts = {"om": [], "pre_run_begin": [], "custom_pass_dump": []}
    for path in _changed_files(before, work_dir):
        lower_name = path.name.lower()
        if path.suffix == ".om":
            artifacts.get("om", []).append(path)
        if path.suffix in (".pbtxt", ".txt") and "prerunbegin" in lower_name:
            artifacts.get("pre_run_begin", []).append(path)
        if path.suffix in (".pbtxt", ".txt") and "runcustompass" in lower_name:
            artifacts.get("custom_pass_dump", []).append(path)
    return artifacts


def _artifact_record(paths, missing_status):
    return {"status": "PASSED" if paths else missing_status, "paths": [str(path) for path in paths]}


@dataclass
class _CompileRecord:
    compiler_kind: str
    command: list
    completed: subprocess.CompletedProcess
    work_dir: Path
    log_path: Path
    artifacts: dict


def _add_compile_records(result, record):
    result.update({
        "compiler_kind": record.compiler_kind,
        "command": shlex.join(record.command),
        "command_argv": record.command,
        "returncode": record.completed.returncode,
        "work_dir": str(record.work_dir),
        "log": str(record.log_path),
        "om": _artifact_record(record.artifacts.get("om", []), "FAILED"),
        "pre_run_begin": _artifact_record(record.artifacts.get("pre_run_begin", []), "NOT_RUN"),
        "custom_pass_dump": _artifact_record(record.artifacts.get("custom_pass_dump", []), "NOT_RUN"),
    })


def _add_pass_load_record(result, pass_load_root, before):
    if pass_load_root is None:
        return
    after = _pass_load_snapshot(pass_load_root)
    result["pass_load"] = {
        "root": str(pass_load_root),
        "before": before,
        "after": after,
        "delta": _pass_load_delta(before, after),
        "isolation": "UNCONFIRMED",
        "reason": "GE 官方独立 C++ pass 加载根尚未确认；本记录只证明共享加载目录的实际内容",
    }


def _add_python_pass_record(args, result):
    if args.pass_language != "python" or not args.py_pass_path:
        return
    py_path = Path(args.py_pass_path).resolve()
    result["python_pass"] = {
        "path": str(py_path), "sha256": sha256_file(py_path) if py_path.is_file() else None,
    }


def _finalize_compile_status(args, result, compiler_kind, completed, artifacts):
    pass_effect = result.get("pass_effect", {})
    if completed.returncode != 0:
        result.update({"status": "FAILED", "reason": f"{compiler_kind} 退出码为 {completed.returncode}"})
    elif not artifacts.get("om"):
        result.update({"status": "FAILED", "reason": "编译命令成功但未生成本轮 OM"})
    elif not artifacts.get("pre_run_begin"):
        result.update({"status": "FAILED", "reason": "未采集到本轮 PreRunBegin dump"})
    elif not artifacts.get("custom_pass_dump") and not args.allow_missing_pass_dump:
        result.update({"status": "FAILED", "reason": "未采集到本轮自定义 pass 阶段 dump"})
    elif args.require_pass_effect and pass_effect.get("status") != "PASSED":
        result.update({
            "status": pass_effect.get("status", "NOT_RUN"),
            "reason": "目标 pass 的结构化命中/生效证据不可用或 effect_times 为 0："
                      + pass_effect.get("reason", "unknown"),
        })
    else:
        result["status"] = "PASSED"


def cmd_compile(args):
    """直接编译 ONNX/AIR，并记录本轮新生成的 OM、dump 和完整日志。"""
    input_type, reason = detect_input_type(args.file)
    result = _compile_result(args, input_type)
    if input_type not in ("onnx", "air", "tf_pb"):
        message = f"{reason}；仅 ONNX/AIR/TensorFlow PB 可直接编译，pbtxt 先执行 repro"
        return _compile_not_run(args, result, message)
    if not args.soc_version:
        return _compile_not_run(args, result, "缺 --soc-version，无法构造可复跑的 ATC/pyatc 命令")
    try:
        compiler_kind, compiler = _resolve_compiler(args)
        environment = _compile_environment(args)
        work_dir, command = _compile_command(args, input_type, compiler)
        before = _file_snapshot(work_dir)
        pass_load_root = Path(args.pass_load_root).resolve() if args.pass_load_root else None
        pass_load_before = _pass_load_snapshot(pass_load_root) if pass_load_root else None
        completed, log_path = _run_compiler(command, work_dir, environment, compiler_kind)
    except _CompileUnavailable as exc:
        return _compile_not_run(args, result, str(exc))
    artifacts = _compile_artifacts(before, work_dir)
    compile_record = _CompileRecord(
        compiler_kind=compiler_kind,
        command=command,
        completed=completed,
        work_dir=work_dir,
        log_path=log_path,
        artifacts=artifacts,
    )
    _add_compile_records(result, compile_record)
    _add_pass_load_record(result, pass_load_root, pass_load_before)
    _add_python_pass_record(args, result)
    result["pass_effect"] = _pass_effect_evidence(args, log_path)
    _finalize_compile_status(args, result, compiler_kind, completed, artifacts)
    _write(args.out_evidence or work_dir / "compile-evidence.json", result)
    _emit_json(result)
    return 0


def cmd_provenance(args):
    itype, _ = detect_input_type(args.file)
    prov = {
        "source_file": str(args.file), "input_type": itype,
        "sha256": sha256_file(args.file), "tool_versions": _tool_versions(),
        "env": _env_snapshot(), "status": "OK",
    }
    _write(args.out or "provenance.json", prov)
    _emit_json(prov)
    return 0


def _python_pass_runtime():
    """探测 Python pass runtime，避免把 pyatc 在位误报成 bridge 可用。"""
    result = {
        "ge.passes": "NOT_AVAILABLE",
        "python_pass_bridge": "NOT_AVAILABLE",
    }
    try:
        module_spec = importlib.util.find_spec("ge.passes")
        module_path = module_spec.origin if module_spec is not None else None
        if not module_path:
            return result
        module_path = Path(module_path).resolve()
        result["ge.passes"] = str(module_path)
        artifact_root = module_path.parent / "python_pass_artifacts"
        bridge = next(
            (path for path in artifact_root.rglob("libge_python_pass_bridge.so") if path.is_file()),
            None,
        ) if artifact_root.is_dir() else None
        if bridge is not None:
            result["python_pass_bridge"] = str(bridge.resolve())
    except Exception as exc:
        # 能力探测失败也必须落成 NOT_AVAILABLE，不影响其他输入路线的证据生成。
        LOGGER.debug("cannot inspect Python pass runtime: %s", exc)
    return result


def _tool_versions():
    v = {"python": sys.version.split()[0]}
    try:
        import onnx
        v["onnx"] = onnx.__version__
    except ImportError:
        v["onnx"] = "NOT_AVAILABLE"
    try:
        import mindspore
        v["mindspore"] = mindspore.__version__
    except ImportError:
        v["mindspore"] = "NOT_AVAILABLE"
    for tool in ("atc", "pyatc"):
        path = shutil.which(tool)
        v[tool] = path or "NOT_AVAILABLE"
    v.update(_python_pass_runtime())
    return v


def _env_snapshot():
    snapshot = {}
    names = ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_TOOLKIT_HOME", "GE_REPO_PATH", "GE_ES_API_ROOT")
    for name in names:
        value = os.environ.get(name)
        if value:
            snapshot[name] = value
    return snapshot


def _write(path, obj):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("adapt_input: 写出 %s", destination)


def _build_cli_parser():
    parser = argparse.ArgumentParser(description="GE/CANN 融合 pass 输入适配（内部里程碑 R3，非版本号）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("detect")
    p.add_argument("file")
    p.add_argument("--out")
    p = sub.add_parser("inventory")
    p.add_argument("file")
    p.add_argument("--out-ng")
    p.add_argument("--out-inv")
    p = sub.add_parser("repro")
    p.add_argument("file")
    p.add_argument("--repro-dir")
    p.add_argument("--out-prov")
    p.add_argument("--out-iso")
    p = sub.add_parser("compile")
    p.add_argument("file")
    p.add_argument("--pass-language", choices=("cpp", "python"), default="cpp")
    p.add_argument("--soc-version")
    p.add_argument("--work-dir")
    p.add_argument("--output")
    p.add_argument("--py-pass-path")
    p.add_argument("--compiler", help="测试或自定义包装器；默认按 pass language 选择 atc/pyatc")
    p.add_argument("--allow-missing-pass-dump", action="store_true")
    p.add_argument("--pass-name", help="目标 pass 名称，用于关联结构化命中/生效证据")
    p.add_argument("--events", help="自有 JSONL pass events 证据")
    p.add_argument("--fusion-result", help="CANN fusion_result.json 官方统计证据")
    p.add_argument("--require-pass-effect", action="store_true",
                   help="没有目标 pass applied/effect_times > 0 时不把 compile 记为 PASSED")
    p.add_argument("--pass-load-root", help="C++ custom_fusion_passes 共享加载根；只读快照并记录变化")
    p.add_argument("--out-evidence")
    p = sub.add_parser("provenance")
    p.add_argument("file")
    p.add_argument("--out")
    return parser


def _run_command(args):
    commands = {
        "detect": cmd_detect,
        "inventory": cmd_inventory,
        "repro": cmd_repro,
        "compile": cmd_compile,
        "provenance": cmd_provenance,
    }
    return commands.get(args.cmd)(args)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_cli_parser().parse_args()
    try:
        exit_code = _run_command(args)
    except CliError as exc:
        LOGGER.error("adapt_input: %s", exc)
        exit_code = exc.exit_code
    if exit_code:
        raise SystemExit(exit_code)
    return 0


if __name__ == "__main__":
    main()
