#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate evidence for GE/CANN fusion passes.

This tool implements the skill's internal validation-evidence milestone (R4);
R4 is a workflow milestone, not a version identifier.

The tool is intentionally offline. It compares artifacts produced by a real
baseline/optimized run and records PASSED, FAILED, or NOT_RUN without inventing
missing device evidence.
"""
import argparse
import csv
import hashlib
import json
import logging
import math
import shlex
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


EVENT_NAMES = {
    "candidate", "matched", "guard_passed", "guard_rejected", "applied",
    "replacement_failed", "skip", "pass_begin", "pass_end",
}
OUTPUT_CONTEXT_KEYS = (
    "source_model_sha256", "input_sha256", "seed", "preprocess", "soc_version",
    "compile_parameters", "run_parameters", "environment",
)
PERFORMANCE_CONTEXT_KEYS = OUTPUT_CONTEXT_KEYS + ("warmup", "runs")
OUTPUT_LOGGER = logging.getLogger(f"{__name__}.stdout")
OUTPUT_LOGGER.setLevel(logging.INFO)
OUTPUT_LOGGER.propagate = False


def _emit_json(value):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    OUTPUT_LOGGER.handlers = [handler]
    OUTPUT_LOGGER.info("%s", json.dumps(value, ensure_ascii=False))


def _write_json(path, value):
    if isinstance(value, dict) and "status" in value and "command" not in value:
        value = dict(value)
        value["command"] = shlex.join(sys.argv)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path, title, status, sections):
    lines = [f"# {title}", "", f"status: `{status}`", ""]
    for heading, content in sections:
        lines.extend([f"## {heading}", ""])
        if isinstance(content, str):
            lines.extend([content, ""])
        else:
            lines.extend(["```json", json.dumps(content, ensure_ascii=False, indent=2), "```", ""])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _status_from_issues(issues):
    return "FAILED" if issues else "PASSED"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_hash_candidates(input_files):
    if not all(isinstance(item, dict) for item in input_files):
        return set()
    indices = [item.get("index") for item in input_files]
    if indices != list(range(len(input_files))):
        return set()
    hashes = [item.get("sha256") for item in input_files]
    if any(not isinstance(value, str) or not value for value in hashes):
        return set()
    canonical = "\n".join(f"{index}:{value}" for index, value in enumerate(hashes)).encode("utf-8")
    candidates = {hashlib.sha256(canonical).hexdigest()}
    if len(hashes) == 1:
        candidates.add(hashes[0])
    return candidates


def _source_provenance_issues(execution, label):
    issues = []
    missing = []
    source_path = execution.get("source_model")
    source_hash = execution.get("source_model_sha256")
    if not isinstance(source_path, str) or not source_path:
        missing.append(f"{label}: execution.source_model is missing")
    elif not isinstance(source_hash, str) or not source_hash:
        missing.append(f"{label}: execution.source_model_sha256 is missing")
    elif not Path(source_path).is_file():
        missing.append(f"{label}: source model is unavailable: {source_path}")
    else:
        try:
            source_matches = _sha256_file(source_path) == source_hash
        except OSError as exc:
            missing.append(f"{label}: source model cannot be hashed: {exc}")
            source_matches = True
        if not source_matches:
            issues.append(f"{label}: source model hash does not match execution record")
    return source_hash, issues, missing


def _input_file_issues(item, index, label):
    issues = []
    missing = []
    if not isinstance(item, dict):
        return [f"{label}: execution.input_files[{index}] is not an object"], missing
    path = item.get("path")
    expected = item.get("sha256")
    valid_record = item.get("index") == index and isinstance(path, str) and isinstance(expected, str)
    if not valid_record:
        return [f"{label}: execution.input_files[{index}] lacks a valid index/path/sha256"], missing
    input_path = Path(path)
    if not input_path.is_file():
        missing.append(f"{label}: input file is unavailable: {path}")
        return issues, missing
    try:
        input_matches = _sha256_file(input_path) == expected
        size_matches = item.get("size") == input_path.stat().st_size
    except OSError as exc:
        missing.append(f"{label}: input file cannot be hashed: {exc}")
        return issues, missing
    if not input_matches:
        issues.append(f"{label}: input file hash does not match execution record: {path}")
    if item.get("size") is None:
        missing.append(f"{label}: execution.input_files[{index}].size is missing")
    elif not size_matches:
        issues.append(f"{label}: input file size does not match execution record: {path}")
    return issues, missing


def _input_provenance_issues(execution, label):
    issues = []
    missing = []
    input_files = execution.get("input_files")
    actual_input_hash = execution.get("input_sha256")
    input_hash_scheme = execution.get("input_hash_scheme")
    if not isinstance(input_files, list) or not input_files:
        missing.append(f"{label}: execution.input_files is missing")
        return input_files, actual_input_hash, issues, missing
    if not isinstance(actual_input_hash, str) or not actual_input_hash:
        missing.append(f"{label}: execution.input_sha256 is missing")
        return input_files, actual_input_hash, issues, missing
    expected_scheme = "single-file-sha256-v1" if len(input_files) == 1 else "ordered-index-sha256-v1"
    if not isinstance(input_hash_scheme, str) or not input_hash_scheme:
        missing.append(f"{label}: execution.input_hash_scheme is missing")
    elif input_hash_scheme != expected_scheme:
        issues.append(f"{label}: input_hash_scheme is {input_hash_scheme!r}, expected {expected_scheme!r}")
    if actual_input_hash not in _input_hash_candidates(input_files):
        issues.append(f"{label}: execution input fingerprint is internally inconsistent")
    for index, item in enumerate(input_files):
        item_issues, item_missing = _input_file_issues(item, index, label)
        issues.extend(item_issues)
        missing.extend(item_missing)
    return input_files, actual_input_hash, issues, missing


def _context_provenance_issues(manifest, label, source_hash, input_hash, input_files):
    context = manifest.get("context")
    if not isinstance(context, dict):
        return [f"{label}: context is not an object"]
    issues = []
    if isinstance(source_hash, str) and context.get("source_model_sha256") != source_hash:
        issues.append(f"{label}: context.source_model_sha256 differs from execution")
    if isinstance(input_hash, str):
        candidates = {input_hash}
        if isinstance(input_files, list) and len(input_files) == 1 and isinstance(input_files[0], dict):
            only_hash = input_files[0].get("sha256")
            if only_hash:
                candidates.add(only_hash)
        if context.get("input_sha256") not in candidates:
            issues.append(f"{label}: context.input_sha256 differs from execution")
    return issues


def _provenance_issues(manifest, label):
    """Validate that run_om used the source model and inputs declared by its manifest."""
    execution = manifest.get("execution")
    if not isinstance(execution, dict):
        return [], [f"{label}: execution provenance is missing"]
    source_hash, source_issues, source_missing = _source_provenance_issues(execution, label)
    input_files, input_hash, input_issues, input_missing = _input_provenance_issues(execution, label)
    issues = source_issues + input_issues
    missing = source_missing + input_missing
    issues.extend(_context_provenance_issues(manifest, label, source_hash, input_hash, input_files))
    return issues, missing


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value):
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").replace("us", "").replace("ms", "").strip())
    except (TypeError, ValueError):
        return None


def _find_fusion_results(value, path=""):
    """Find CANN fusion_result.json records regardless of session nesting."""
    records = []
    if isinstance(value, dict):
        if "match_times" in value and "effect_times" in value:
            name = path.rsplit("/", 1)[-1] or "unknown_pass"
            records.append({
                "pass": name,
                "match_times": _as_int(value.get("match_times")),
                "effect_times": _as_int(value.get("effect_times")),
                "source_path": path,
            })
        for key, child in value.items():
            records.extend(_find_fusion_results(child, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            records.extend(_find_fusion_results(child, f"{path}/{index}"))
    return records


def _read_self_events(events_path):
    by_pass = defaultdict(Counter)
    reasons = Counter()
    issues = []
    event_count = 0
    if events_path is None:
        return by_pass, reasons, issues, event_count
    for line_no, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"events line {line_no} is not JSON: {exc.msg}")
            continue
        event_name = event.get("event")
        pass_name = event.get("pass")
        if event_name not in EVENT_NAMES:
            issues.append(f"events line {line_no} has unsupported event: {event_name!r}")
            continue
        if not isinstance(pass_name, str) or not pass_name:
            issues.append(f"events line {line_no} lacks pass")
            continue
        by_pass[pass_name][event_name] += 1
        if event_name in {"skip", "guard_rejected", "replacement_failed"}:
            reason = event.get("reason")
            if not isinstance(reason, str) or not reason:
                issues.append(f"events line {line_no} needs reason for {event_name}")
            else:
                reasons[reason] += 1
        event_count += 1
    return by_pass, reasons, issues, event_count


def _self_event_records(by_pass, issues):
    records = []
    for pass_name, counts in sorted(by_pass.items()):
        candidate = counts.get("candidate", 0)
        matched = counts.get("matched", 0)
        guard_passed = counts.get("guard_passed", 0)
        applied = counts.get("applied", 0)
        replacement_failed = counts.get("replacement_failed", 0)
        if matched > candidate:
            issues.append(f"{pass_name}: matched ({matched}) exceeds candidate ({candidate})")
        if guard_passed > matched:
            issues.append(f"{pass_name}: guard_passed ({guard_passed}) exceeds matched ({matched})")
        if applied + replacement_failed > guard_passed:
            issues.append(f"{pass_name}: applied + replacement_failed exceeds guard_passed")
        record = {"pass": pass_name}
        record.update({key: counts.get(key, 0) for key in sorted(counts)})
        records.append(record)
    return records


def _official_record_issues(records):
    issues = []
    for record in records:
        match_times = record.get("match_times")
        effect_times = record.get("effect_times")
        if match_times is None or effect_times is None:
            issues.append(f"{record.get('pass')}: count is not an integer")
        elif effect_times > match_times:
            issues.append(f"{record.get('pass')}: effect_times exceeds match_times")
    return issues


def _official_fusion_result(value):
    if not value:
        return {"status": "NOT_RUN", "records": []}, []
    fusion_path = Path(value)
    if not fusion_path.is_file():
        result = {"status": "NOT_RUN", "reason": f"fusion_result.json 不存在: {fusion_path}", "records": []}
        return result, []
    try:
        records = _find_fusion_results(_load_json(fusion_path))
    except (OSError, json.JSONDecodeError) as exc:
        reason = f"无法解析 fusion_result.json: {exc}"
        return {"status": "FAILED", "reason": reason, "records": []}, [reason]
    issues = _official_record_issues(records)
    status = "FAILED" if issues else "PASSED" if records else "NOT_RUN"
    result = {"status": status, "records": records, "issues": issues}
    if not records:
        result["reason"] = "fusion_result.json 未含 match_times/effect_times 记录"
    return result, issues


def _target_event_issues(args, event_count, self_records, official):
    issues = []
    if args.pass_name and (event_count or official.get("records")):
        self_matches = [record for record in self_records if record.get("pass") == args.pass_name]
        official_matches = []
        for record in official.get("records", []):
            record_name = record.get("pass", "")
            if record_name == args.pass_name or args.pass_name in record_name:
                official_matches.append(record)
        if not self_matches and not official_matches:
            issues.append(f"未找到目标 pass 的结构化证据: {args.pass_name}")
        if args.require_applied:
            applied = sum(record.get("applied", 0) for record in self_matches)
            effect_times = sum(int(record.get("effect_times") or 0) for record in official_matches)
            if applied <= 0 and effect_times <= 0:
                issues.append(f"目标 pass 未产生 applied/effect_times > 0: {args.pass_name}")
    return issues


def _event_status(args, event_count, official, issues):
    if not event_count and official.get("status") == "NOT_RUN":
        status = "NOT_RUN"
        reason = "未提供 pass events 或 fusion_result.json"
        if args.require_applied:
            reason += "；--require-applied 要求结构化命中/生效证据"
        return status, reason
    return _status_from_issues(issues), None


def _write_event_result(args, result):
    _write_json(args.out_json, result)
    if args.out_md:
        sections = [
            ("Self Counters", result.get("self_counters", [])),
            ("Skip And Failure Reasons", result.get("skip_or_failure_reasons", {})),
            ("Official Fusion Result", result.get("official_fusion_result", {})),
            ("Consistency Issues", result.get("consistency_issues", [])),
        ]
        _write_markdown(args.out_md, "Pass Event Summary", result.get("status"), sections)


def cmd_events(args):
    events_path = Path(args.events) if args.events else None
    if events_path is not None and not events_path.is_file():
        result = {"status": "NOT_RUN", "reason": f"事件文件不存在: {events_path}"}
        _write_json(args.out_json, result)
        if args.out_md:
            _write_markdown(args.out_md, "Pass Event Summary", result.get("status"), [("Reason", result.get("reason"))])
        return 0
    by_pass, reasons, issues, event_count = _read_self_events(events_path)
    self_records = _self_event_records(by_pass, issues)
    official, official_issues = _official_fusion_result(args.fusion_result)
    issues.extend(official_issues)
    issues.extend(_target_event_issues(args, event_count, self_records, official))
    status, reason = _event_status(args, event_count, official, issues)
    result = {
        "status": status, "reason": reason, "event_schema_version": "1.0",
        "input_artifacts": {
            "events": str(Path(args.events).resolve()) if args.events else None,
            "fusion_result": str(Path(args.fusion_result).resolve()) if args.fusion_result else None,
        },
        "event_count": event_count, "self_counters": self_records,
        "skip_or_failure_reasons": dict(sorted(reasons.items())),
        "official_fusion_result": official, "consistency_issues": issues,
        "pass_name": args.pass_name, "require_applied": args.require_applied,
    }
    _write_event_result(args, result)
    return 0


def _load_graph(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        graph = _load_json(path)
    else:
        try:
            import adapt_input
        except ImportError as exc:
            raise RuntimeError(f"无法加载 adapt_input.py: {exc}") from exc
        input_type, reason = adapt_input.detect_input_type(path)
        parser = adapt_input.PARSERS.get(input_type)
        if parser is None:
            raise RuntimeError(f"{reason}；仅 normalized JSON、ONNX 或 pbtxt 可用于图 diff")
        graph = parser(path)
    if graph.get("status") == "NOT_RUN":
        raise RuntimeError(graph.get("reason", "normalized graph is NOT_RUN"))
    if not isinstance(graph.get("nodes"), list):
        raise RuntimeError("不是有效 normalized-graph.json：缺 nodes[]")
    return graph


def _node_by_name(graph):
    nodes = {}
    duplicates = []
    for node in graph.get("nodes", []):
        name = node.get("name")
        if not isinstance(name, str) or not name:
            duplicates.append("node without name")
        elif name in nodes:
            duplicates.append(f"duplicate node name: {name}")
        else:
            nodes[name] = node
    return nodes, duplicates


def _edge_signature(graph, key):
    node_by_id = {node.get("id"): node.get("name") for node in graph.get("nodes", [])}
    result = set()
    for edge in graph.get(key, []):
        source = node_by_id.get(edge.get("from_node"))
        target = node_by_id.get(edge.get("to_node"))
        if source is None or target is None:
            result.add(("<dangling>", json.dumps(edge, sort_keys=True)))
        elif key == "data_edges":
            result.add((source, edge.get("from_port"), target, edge.get("to_port")))
        else:
            result.add((source, target))
    return result


def _output_signature(graph):
    node_by_id = {node.get("id"): node.get("name") for node in graph.get("nodes", [])}
    return [
        (output.get("name"), node_by_id.get(output.get("from_node")), output.get("from_port"))
        for output in graph.get("outputs", [])
    ]


def _matches_node(spec, name, node):
    if isinstance(spec, str):
        return spec == name or spec == node.get("op_type") or spec == node.get("original_op_type")
    if not isinstance(spec, dict):
        return False
    return all(node.get(key) == value for key, value in spec.items())


def _contains_spec(specs, name, node):
    return any(_matches_node(spec, name, node) for spec in specs)


def _node_changes(base, optimized):
    fields = (
        "op_type", "original_op_type", "attrs", "shape", "shape_range", "dtype", "format",
        "optional_inputs_present",
    )
    return [field for field in fields if base.get(field, {}) != optimized.get(field, {})]


def _node_field_change_issues(name, changes, allowed_fields):
    issues = []
    for changed_field in changes:
        exact = f"{name}.{changed_field}"
        wildcard = f"*.{changed_field}"
        if exact not in allowed_fields and wildcard not in allowed_fields:
            issues.append(f"unallowed node field change: {exact}")
    return issues


def _graph_integrity(graph):
    node_ids = {node.get("id") for node in graph.get("nodes", [])}
    issues = []
    for edge in graph.get("data_edges", []) + graph.get("control_edges", []):
        if edge.get("from_node") not in node_ids or edge.get("to_node") not in node_ids:
            issues.append(f"dangling edge: {edge}")
    for output in graph.get("outputs", []):
        if output.get("from_node") not in node_ids:
            issues.append(f"dangling graph output: {output}")
    return issues


def _graph_node_changes(baseline, optimized, allow):
    issues = []
    base_nodes, base_duplicates = _node_by_name(baseline)
    opt_nodes, opt_duplicates = _node_by_name(optimized)
    issues.extend(base_duplicates + opt_duplicates)
    removed_names = sorted(set(base_nodes) - set(opt_nodes))
    added_names = sorted(set(opt_nodes) - set(base_nodes))
    common_names = sorted(set(base_nodes) & set(opt_nodes))
    unexpected_removed = [
        name for name in removed_names
        if not _contains_spec(allow.get("removed_nodes", []), name, base_nodes.get(name, {}))
    ]
    unexpected_added = [
        name for name in added_names
        if not _contains_spec(allow.get("added_nodes", []), name, opt_nodes.get(name, {}))
    ]
    if unexpected_removed:
        issues.append(f"unallowed removed nodes: {unexpected_removed}")
    if unexpected_added:
        issues.append(f"unallowed added nodes: {unexpected_added}")
    changed_nodes = []
    allowed_fields = set(allow.get("node_field_changes", []))
    for name in common_names:
        changes = _node_changes(base_nodes.get(name, {}), opt_nodes.get(name, {}))
        if not changes:
            continue
        changed_nodes.append({"name": name, "fields": changes})
        issues.extend(_node_field_change_issues(name, changes, allowed_fields))
    names = {"removed": removed_names, "added": added_names, "changed": changed_nodes}
    return base_nodes, opt_nodes, names, issues


def _graph_edge_changes(baseline, optimized, allow, base_nodes, opt_nodes):
    issues = []
    base_data = _edge_signature(baseline, "data_edges")
    opt_data = _edge_signature(optimized, "data_edges")
    base_control = _edge_signature(baseline, "control_edges")
    opt_control = _edge_signature(optimized, "control_edges")
    if base_data != opt_data and not allow.get("data_edge_changes", False):
        issues.append("unallowed data-edge changes")
    if base_control != opt_control and not allow.get("control_edge_changes", False):
        issues.append("unallowed control-edge changes")
    unexpected_consumers = []
    if not allow.get("unexpected_consumers", False):
        for edge in opt_data - base_data:
            source = edge[0]
            if source in base_nodes and source in opt_nodes and edge[2] in base_nodes:
                unexpected_consumers.append(edge)
        if unexpected_consumers:
            issues.append(f"unexpected new consumers: {sorted(unexpected_consumers)}")
    edges = {
        "data": {"removed": sorted(base_data - opt_data), "added": sorted(opt_data - base_data)},
        "control": {
            "removed": sorted(base_control - opt_control), "added": sorted(opt_control - base_control),
        },
    }
    return edges, sorted(unexpected_consumers), issues


def _matches_any_node(spec, names, nodes):
    for name in names:
        if _matches_node(spec, name, nodes.get(name, {})):
            return True
    return False


def _required_node_issues(require, base_nodes, opt_nodes, removed_names, added_names):
    issues = []
    for spec in require.get("keep_nodes", []):
        if not _matches_any_node(spec, opt_nodes, opt_nodes):
            issues.append(f"required kept node missing: {spec}")
    for spec in require.get("remove_nodes", []):
        if not _matches_any_node(spec, removed_names, base_nodes):
            issues.append(f"required removed node absent: {spec}")
    for spec in require.get("add_nodes", []):
        if not _matches_any_node(spec, added_names, opt_nodes):
            issues.append(f"required added node absent: {spec}")
    return issues


def _graph_output_check(baseline, optimized, preserve_outputs, issues):
    base_outputs = _output_signature(baseline)
    opt_outputs = _output_signature(optimized)
    if not baseline.get("outputs") or not optimized.get("outputs"):
        return {"status": "NOT_RUN", "reason": "一侧 normalized graph 缺可靠 outputs 签名"}
    if base_outputs != opt_outputs:
        if preserve_outputs:
            issues.append("graph outputs changed")
            return {"status": "FAILED", "baseline": base_outputs, "optimized": opt_outputs}
        return {
            "status": "PASSED", "allowed_change": True,
            "baseline": base_outputs, "optimized": opt_outputs,
        }
    return {"status": "PASSED", "outputs": base_outputs}


def _write_graph_result(args, result):
    _write_json(args.out_json, result)
    if args.out_md:
        sections = [
            ("Checks", result.get("checks", {})), ("Node Changes", result.get("nodes", {})),
            ("Data Edge Changes", result.get("data_edges", {})),
            ("Control Edge Changes", result.get("control_edges", {})), ("Issues", result.get("issues", [])),
        ]
        _write_markdown(args.out_md, "Graph Comparison", result.get("status"), sections)


def cmd_graph(args):
    try:
        baseline = _load_graph(args.baseline)
        optimized = _load_graph(args.optimized)
    except RuntimeError as exc:
        result = {"status": "NOT_RUN", "reason": str(exc)}
        _write_json(args.out_json, result)
        if args.out_md:
            _write_markdown(args.out_md, "Graph Comparison", result.get("status"), [("Reason", result.get("reason"))])
        return 0
    rules = _load_json(args.rules) if args.rules else {}
    allow = rules.get("allow", {})
    require = rules.get("require", {})
    integrity_issues = _graph_integrity(baseline) + _graph_integrity(optimized)
    base_nodes, opt_nodes, nodes, node_issues = _graph_node_changes(baseline, optimized, allow)
    edges, unexpected_consumers, edge_issues = _graph_edge_changes(
        baseline, optimized, allow, base_nodes, opt_nodes,
    )
    issues = integrity_issues + node_issues + edge_issues
    issues.extend(_required_node_issues(
        require, base_nodes, opt_nodes, nodes.get("removed", []), nodes.get("added", []),
    ))
    output_check = _graph_output_check(baseline, optimized, require.get("preserve_outputs", True), issues)

    status = _status_from_issues(issues)
    if status == "PASSED" and output_check.get("status") == "NOT_RUN":
        status = "NOT_RUN"
    result = {
        "status": status, "rules": rules,
        "input_graphs": {"baseline": str(Path(args.baseline).resolve()),
                         "optimized": str(Path(args.optimized).resolve())},
        "checks": {
            "outputs": output_check, "graph_integrity": "PASSED" if not integrity_issues else "FAILED",
        },
        "nodes": nodes,
        "data_edges": edges.get("data", {}),
        "control_edges": edges.get("control", {}),
        "unexpected_consumers": unexpected_consumers,
        "issues": issues,
    }
    _write_graph_result(args, result)
    return 0


def _load_manifest(path):
    path = Path(path)
    manifest = _load_json(path)
    if not isinstance(manifest, dict):
        raise RuntimeError("manifest must be a JSON object")
    manifest.update({"_path": path.resolve()})
    return manifest


def _resolve_artifact(manifest, item):
    value = Path(item.get("path"))
    manifest_path = manifest.get("_path")
    return value if value.is_absolute() else manifest_path.parent / value


def _context_issues(baseline, optimized, required_keys):
    base_context = baseline.get("context", {})
    opt_context = optimized.get("context", {})
    if not isinstance(base_context, dict) or not isinstance(opt_context, dict):
        return ["context must be an object in both manifests"]
    issues = []
    for key in required_keys:
        if key not in base_context or key not in opt_context:
            issues.append(f"context missing required key: {key}")
    keys = sorted(set(base_context) | set(opt_context))
    issues.extend(f"context mismatch: {key}" for key in keys if base_context.get(key) != opt_context.get(key))
    return issues


def _float_tolerance(dtype, atol, rtol):
    if atol is not None and rtol is not None:
        return float(atol), float(rtol)
    name = str(dtype).lower()
    if "float16" in name or "bfloat16" in name:
        return 1e-3, 1e-3
    if "float64" in name:
        return 1e-8, 1e-8
    return 1e-5, 1e-5


def _load_output_map(path):
    value = _load_json(path)
    if isinstance(value, dict) and "mapping" in value:
        value = value.get("mapping")
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(target, str) and key and target
        for key, target in value.items()
    ):
        raise RuntimeError("output map must be a JSON object: {baseline_name: optimized_name}")
    return value


def _manifest_provenance(baseline, optimized):
    issues = []
    missing = []
    for label, manifest in (("baseline", baseline), ("optimized", optimized)):
        current_issues, current_missing = _provenance_issues(manifest, label)
        issues.extend(current_issues)
        missing.extend(current_missing)
    return issues, missing


def _output_names(outputs):
    return [item.get("name") if isinstance(item, dict) else None for item in outputs]


def _validate_output_lists(baseline, optimized, issues):
    base_outputs = baseline.get("outputs", [])
    opt_outputs = optimized.get("outputs", [])
    if not isinstance(base_outputs, list) or not isinstance(opt_outputs, list):
        issues.append("baseline/optimized outputs must be arrays")
        return [], [], [], []
    if len(base_outputs) != len(opt_outputs):
        issues.append(f"output count differs: {len(base_outputs)} != {len(opt_outputs)}")
    base_names = _output_names(base_outputs)
    opt_names = _output_names(opt_outputs)
    if any(not isinstance(name, str) or not name for name in base_names + opt_names):
        issues.append("every output must have a non-empty name")
    if len(set(base_names)) != len(base_names):
        issues.append("baseline output names must be unique")
    if len(set(opt_names)) != len(opt_names):
        issues.append("optimized output names must be unique")
    return base_outputs, opt_outputs, base_names, opt_names


def _validate_output_map(output_map, base_names, opt_names, issues):
    unknown_baseline = sorted(set(output_map) - set(base_names))
    unknown_optimized = sorted(set(output_map.values()) - set(opt_names))
    target_counts = Counter(output_map.values())
    duplicate_targets = sorted(target for target, count in target_counts.items() if count > 1)
    if unknown_baseline:
        issues.append(f"output map has unknown baseline names: {unknown_baseline}")
    if unknown_optimized:
        issues.append(f"output map has unknown optimized names: {unknown_optimized}")
    if duplicate_targets:
        issues.append(f"output map is not one-to-one: {duplicate_targets}")


def _read_output_map(args, base_names, opt_names, issues):
    output_map = {}
    mapping_supplied = args.output_map is not None
    if args.output_map:
        try:
            output_map = _load_output_map(args.output_map)
        except (OSError, json.JSONDecodeError, RuntimeError) as exc:
            issues.append(f"cannot read output map: {exc}")
    if mapping_supplied:
        _validate_output_map(output_map, base_names, opt_names, issues)
    return output_map, mapping_supplied


def _output_pairs(base_outputs, opt_outputs, output_map, mapping_supplied, issues):
    if not mapping_supplied:
        return list(zip(base_outputs, opt_outputs))
    pairs = []
    optimized_by_name = {item.get("name"): item for item in opt_outputs}
    for base_item in base_outputs:
        base_name = base_item.get("name")
        target_name = output_map.get(base_name, base_name)
        opt_item = optimized_by_name.get(target_name)
        if opt_item is None:
            issues.append(f"output map has no optimized target for {base_name!r}")
        else:
            pairs.append((base_item, opt_item))
    return pairs


def _floating_output_result(np, base, optimized, args):
    atol, rtol = _float_tolerance(base.dtype, args.atol, args.rtol)
    finite = np.isfinite(base) & np.isfinite(optimized)
    if np.any(finite):
        abs_error = np.abs(base[finite] - optimized[finite])
        rel_error = abs_error / np.maximum(np.abs(base[finite]), atol)
    else:
        abs_error = np.array([], dtype=float)
        rel_error = np.array([], dtype=float)
    equal = bool(np.allclose(base, optimized, atol=atol, rtol=rtol, equal_nan=True))
    close_count = np.isclose(base, optimized, atol=atol, rtol=rtol, equal_nan=True).sum()
    result = {
        "status": "PASSED" if equal else "FAILED", "atol": atol, "rtol": rtol,
        "max_abs_error": float(abs_error.max()) if abs_error.size else 0.0,
        "max_rel_error": float(rel_error.max()) if rel_error.size else 0.0,
        "nan_counts": {"baseline": int(np.isnan(base).sum()), "optimized": int(np.isnan(optimized).sum())},
        "inf_counts": {"baseline": int(np.isinf(base).sum()), "optimized": int(np.isinf(optimized).sum())},
        "failed_elements": int(np.size(base) - close_count),
    }
    return result, equal


def _exact_output_result(np, base, optimized):
    equal = bool(np.array_equal(base, optimized))
    result = {
        "status": "PASSED" if equal else "FAILED", "comparison": "exact",
        "failed_elements": int(np.size(base) - np.equal(base, optimized).sum()),
    }
    return result, equal


@dataclass
class _OutputComparisonContext:
    np: object
    args: argparse.Namespace
    baseline: dict
    optimized: dict
    mapping_supplied: bool
    issues: list
    warnings: list


def _compare_output_tensors(context, name, base, optimized):
    if base.shape != optimized.shape or base.dtype != optimized.dtype:
        context.issues.append(f"{name}: shape or dtype differs")
        return {"status": "FAILED", "reason": "shape or dtype differs"}
    if context.np.issubdtype(base.dtype, context.np.floating):
        comparison, equal = _floating_output_result(context.np, base, optimized, context.args)
        message = f"{name}: float allclose failed"
    else:
        comparison, equal = _exact_output_result(context.np, base, optimized)
        message = f"{name}: integer/bool exact comparison failed"
    if not equal:
        context.issues.append(message)
    return comparison


def _compare_output_pair(context, pair, index):
    base_item, opt_item = pair
    name = base_item.get("name")
    optimized_name = opt_item.get("name")
    if name != optimized_name:
        message = f"output mapping differs at {index}: {name!r} != {optimized_name!r}"
        if not context.mapping_supplied:
            suffix = "; --output-map is required for semantic alignment" if context.args.allow_output_name_diff else ""
            context.issues.append(message + suffix)
            return None
        context.warnings.append(message)
    base_path = _resolve_artifact(context.baseline, base_item)
    optimized_path = _resolve_artifact(context.optimized, opt_item)
    try:
        base = context.np.load(base_path, allow_pickle=False)
        optimized_value = context.np.load(optimized_path, allow_pickle=False)
    except OSError as exc:
        context.issues.append(f"{name}: cannot read output tensor: {exc}")
        return None
    item_result = {
        "name": name, "optimized_name": optimized_name,
        "baseline_path": str(base_path.resolve()), "optimized_path": str(optimized_path.resolve()),
        "baseline_shape": list(base.shape), "optimized_shape": list(optimized_value.shape),
        "baseline_dtype": str(base.dtype), "optimized_dtype": str(optimized_value.dtype),
    }
    item_result.update(_compare_output_tensors(context, name, base, optimized_value))
    return item_result


def _compare_output_pairs(context, pairs):
    comparisons = []
    for index, pair in enumerate(pairs):
        result = _compare_output_pair(context, pair, index)
        if result is not None:
            comparisons.append(result)
    return comparisons, context.warnings


def _write_output_result(args, result):
    _write_json(args.out_json, result)
    if args.out_md:
        sections = [
            ("Provenance", result.get("provenance", {})),
            ("Tensor Comparisons", result.get("comparisons", [])),
            ("Issues", result.get("issues", [])), ("Warnings", result.get("warnings", [])),
        ]
        _write_markdown(args.out_md, "Output Comparison", result.get("status"), sections)


def _write_missing_outputs(args, issues, provenance_missing, input_manifests):
    status = "FAILED" if issues else "NOT_RUN"
    result = {
        "status": status, "reason": "baseline/optimized 均需至少一个最终输出张量",
        "issues": issues, "provenance_missing": provenance_missing, "input_manifests": input_manifests,
    }
    _write_json(args.out_json, result)
    if args.out_md:
        sections = [("Reason", result.get("reason")), ("Issues", issues)]
        _write_markdown(args.out_md, "Output Comparison", status, sections)
    return 0


def cmd_outputs(args):
    input_manifests = {
        "baseline": str(Path(args.baseline).resolve()), "optimized": str(Path(args.optimized).resolve()),
    }
    try:
        import numpy as np
        baseline = _load_manifest(args.baseline)
        optimized = _load_manifest(args.optimized)
    except (ImportError, OSError, json.JSONDecodeError, RuntimeError) as exc:
        result = {"status": "NOT_RUN", "reason": str(exc), "input_manifests": input_manifests}
        _write_json(args.out_json, result)
        if args.out_md:
            _write_markdown(args.out_md, "Output Comparison", result.get("status"), [("Reason", result.get("reason"))])
        return 0
    issues = _context_issues(baseline, optimized, OUTPUT_CONTEXT_KEYS)
    provenance_issues, provenance_missing = _manifest_provenance(baseline, optimized)
    issues.extend(provenance_issues)
    base_outputs, opt_outputs, base_names, opt_names = _validate_output_lists(baseline, optimized, issues)
    if not base_outputs or not opt_outputs:
        return _write_missing_outputs(args, issues, provenance_missing, input_manifests)
    output_map, mapping_supplied = _read_output_map(args, base_names, opt_names, issues)
    pairs = _output_pairs(base_outputs, opt_outputs, output_map, mapping_supplied, issues)
    comparison_context = _OutputComparisonContext(
        np=np,
        args=args,
        baseline=baseline,
        optimized=optimized,
        mapping_supplied=mapping_supplied,
        issues=issues,
        warnings=[],
    )
    comparisons, warnings = _compare_output_pairs(comparison_context, pairs)

    status = "FAILED" if issues else "NOT_RUN" if provenance_missing else "PASSED"
    provenance_status = "FAILED" if provenance_issues else "NOT_RUN" if provenance_missing else "PASSED"
    result = {
        "status": status, "context": {
            "baseline": baseline.get("context"), "optimized": optimized.get("context"),
        },
        "context_issues": _context_issues(baseline, optimized, OUTPUT_CONTEXT_KEYS),
        "provenance": {
            "status": provenance_status, "issues": provenance_issues, "missing": provenance_missing,
        },
        "comparisons": comparisons, "issues": issues, "warnings": warnings,
        "output_map": output_map or None, "input_manifests": input_manifests,
    }
    _write_output_result(args, result)
    return 0


def _percentile_95(values):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _csv_field(row, *names):
    normalized = {key.lower().replace(" ", ""): value for key, value in row.items() if key}
    for name in names:
        value = normalized.get(name.lower().replace(" ", ""))
        if value is not None:
            return value
    return None


def _read_profile_rows(path):
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _add_op_summary(rows, op_summary):
    for row in rows:
        duration = _as_float(_csv_field(row, "Task Duration(us)", "Task Duration", "Duration(us)"))
        if duration is not None:
            op_summary.append(duration)


def _add_op_statistics(rows, op_statistic):
    for row in rows:
        op_type = _csv_field(row, "OP Type", "Op Type", "Type")
        count = _as_float(_csv_field(row, "Count"))
        total = _as_float(_csv_field(row, "Total Time(us)", "Total Time", "Task Duration(us)"))
        if op_type and total is not None:
            statistic = op_statistic.setdefault(op_type, {"count": 0.0, "total_us": 0.0})
            statistic.update({
                "count": statistic.get("count", 0.0) + (count or 0.0),
                "total_us": statistic.get("total_us", 0.0) + total,
            })


def _profile_summary(directory):
    if not directory:
        return {"status": "NOT_RUN", "reason": "未提供 profiling 目录"}
    root = Path(directory)
    if not root.is_dir():
        return {"status": "NOT_RUN", "path": str(root), "reason": f"profiling 目录不存在: {root}"}
    op_summary = []
    op_statistic = defaultdict(lambda: {"count": 0.0, "total_us": 0.0})
    for path in root.rglob("*.csv"):
        rows = _read_profile_rows(path)
        lower_name = path.name.lower()
        if "op_summary" in lower_name:
            _add_op_summary(rows, op_summary)
        elif "op_statistic" in lower_name:
            _add_op_statistics(rows, op_statistic)
    if not op_summary and not op_statistic:
        return {"status": "NOT_RUN", "path": str(root.resolve()), "reason": "未找到可解析的 op_summary/op_statistic CSV"}
    return {"status": "PASSED", "path": str(root.resolve()), "op_summary_total_us": sum(op_summary),
            "op_summary_rows": len(op_summary), "op_statistic": dict(sorted(op_statistic.items()))}


def _operator_group_summary(profile, op_types):
    valid_group = isinstance(op_types, list) and op_types
    if not valid_group or not all(isinstance(value, str) and value for value in op_types):
        return {"status": "NOT_RUN", "reason": "未提供非空 operator group"}
    if profile.get("status") != "PASSED":
        return {"status": "NOT_RUN", "reason": "profiling 不可用"}
    statistic = profile.get("op_statistic", {})
    missing = [op_type for op_type in op_types if op_type not in statistic]
    if missing:
        return {"status": "FAILED", "op_types": op_types, "missing_op_types": missing}
    return {
        "status": "PASSED", "op_types": op_types,
        "count": sum(statistic[op_type]["count"] for op_type in op_types),
        "total_us": sum(statistic[op_type]["total_us"] for op_type in op_types),
    }


def _operator_group_comparison(profiles, groups):
    if not groups:
        return {"status": "NOT_RUN", "reason": "未提供 --operator-groups"}
    if not isinstance(groups, dict):
        return {"status": "FAILED", "reason": "operator groups 必须是 JSON object"}
    baseline = _operator_group_summary(profiles["baseline"], groups.get("baseline"))
    optimized = _operator_group_summary(profiles["optimized"], groups.get("optimized"))
    statuses = {baseline["status"], optimized["status"]}
    status = "FAILED" if "FAILED" in statuses else "NOT_RUN" if "NOT_RUN" in statuses else "PASSED"
    result = {"status": status, "baseline": baseline, "optimized": optimized}
    if status == "PASSED":
        absolute = optimized["total_us"] - baseline["total_us"]
        result["delta"] = {
            "absolute_us": absolute,
            "relative_percent": absolute / baseline["total_us"] * 100 if baseline["total_us"] else None,
        }
    return result


def _validate_latency_context(label, manifest, values, issues):
    context = manifest.get("context", {})
    warmup = _as_int(context.get("warmup"))
    runs = _as_int(context.get("runs"))
    if warmup is None or warmup < 0:
        issues.append(f"{label}: warmup must be a non-negative integer")
    if runs is None or runs < 1:
        issues.append(f"{label}: runs must be a positive integer")
    elif len(values) != runs:
        issues.append(f"{label}: latencies_ms samples ({len(values)}) differs from runs ({runs})")


def _latency_values(baseline, optimized, issues):
    try:
        base_times = [float(value) for value in baseline.get("latencies_ms", [])]
        opt_times = [float(value) for value in optimized.get("latencies_ms", [])]
    except (TypeError, ValueError):
        issues.append("latencies_ms contains non-numeric value")
        return [], []
    if any(value < 0 for value in base_times + opt_times):
        issues.append("latencies_ms must be non-negative")
    _validate_latency_context("baseline", baseline, base_times, issues)
    _validate_latency_context("optimized", optimized, opt_times, issues)
    return base_times, opt_times


def _latency_stats(values):
    return {"samples": len(values), "median_ms": statistics.median(values), "p95_ms": _percentile_95(values)}


def _latency_delta(base_stats, opt_stats):
    if not base_stats or not opt_stats:
        return None
    base_median = base_stats.get("median_ms")
    absolute = opt_stats.get("median_ms") - base_median
    return {
        "median_absolute_ms": absolute,
        "median_relative_percent": absolute / base_median * 100 if base_median else None,
    }


def _performance_result(baseline, optimized, issues):
    raw_base = baseline.get("latencies_ms", [])
    raw_optimized = optimized.get("latencies_ms", [])
    if not raw_base or not raw_optimized:
        status = "NOT_RUN" if not issues else "FAILED"
        return {"status": status, "reason": "baseline/optimized 均需非空 latencies_ms", "issues": issues}
    base_times, opt_times = _latency_values(baseline, optimized, issues)
    base_stats = _latency_stats(base_times) if base_times else None
    opt_stats = _latency_stats(opt_times) if opt_times else None
    return {
        "status": _status_from_issues(issues),
        "context": {"baseline": baseline.get("context"), "optimized": optimized.get("context")},
        "baseline": base_stats, "optimized": opt_stats,
        "delta": _latency_delta(base_stats, opt_stats), "issues": issues,
    }


def _add_performance_profiles(args, result):
    profiles = {
        "baseline": _profile_summary(args.baseline_profile),
        "optimized": _profile_summary(args.optimized_profile),
    }
    result["profiling"] = profiles
    try:
        groups = _load_json(args.operator_groups) if args.operator_groups else None
    except (OSError, json.JSONDecodeError) as exc:
        comparison = {"status": "FAILED", "reason": f"无法读取 operator groups: {exc}"}
    else:
        comparison = _operator_group_comparison(profiles, groups)
    result["operator_group_comparison"] = comparison
    if args.operator_groups:
        result["operator_groups_path"] = str(Path(args.operator_groups).resolve())


def _merge_operator_group_status(result):
    status = result.get("status")
    group_status = result.get("operator_group_comparison", {}).get("status")
    if status == "PASSED" and group_status != "PASSED":
        result["status"] = group_status
    elif status == "NOT_RUN" and group_status == "FAILED":
        result["status"] = "FAILED"


def _write_performance_result(args, result):
    _write_json(args.out_json, result)
    if args.out_md:
        sections = [
            ("End To End", {key: result.get(key) for key in ("baseline", "optimized", "delta")}),
            ("Profiling", result.get("profiling", {})),
            ("Operator Group", result.get("operator_group_comparison", {})),
            ("Issues", result.get("issues", [])),
        ]
        _write_markdown(args.out_md, "Performance Comparison", result.get("status"), sections)


def cmd_performance(args):
    try:
        baseline = _load_manifest(args.baseline)
        optimized = _load_manifest(args.optimized)
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        result = {"status": "NOT_RUN", "reason": str(exc)}
        _write_json(args.out_json, result)
        if args.out_md:
            _write_markdown(
                args.out_md, "Performance Comparison", result.get("status"), [("Reason", result.get("reason"))],
            )
        return 0
    issues = _context_issues(baseline, optimized, PERFORMANCE_CONTEXT_KEYS)
    provenance_issues, provenance_missing = _manifest_provenance(baseline, optimized)
    issues.extend(provenance_issues)
    result = _performance_result(baseline, optimized, issues)
    if not issues and provenance_missing:
        result["status"] = "NOT_RUN"
    result["provenance"] = {
        "status": "FAILED" if provenance_issues else "NOT_RUN" if provenance_missing else "PASSED",
        "issues": provenance_issues, "missing": provenance_missing,
    }
    result["input_manifests"] = {"baseline": str(Path(args.baseline).resolve()),
                                 "optimized": str(Path(args.optimized).resolve())}
    _add_performance_profiles(args, result)
    _merge_operator_group_status(result)
    _write_performance_result(args, result)
    return 0


def cmd_report(args):
    items = []
    for path in args.items:
        try:
            content = _load_json(path)
            items.append({"path": str(Path(path).resolve()), "status": content.get("status", "NOT_RUN"),
                          "command": content.get("command"), "reason": content.get("reason")})
        except (OSError, json.JSONDecodeError) as exc:
            items.append({"path": str(path), "status": "NOT_RUN", "reason": str(exc)})
    statuses = {item["status"] for item in items}
    overall = "FAILED" if "FAILED" in statuses else "NOT_RUN" if "NOT_RUN" in statuses else "PASSED"
    # G3 availability is distinct from the validation conclusion: a failed run is
    # still an executed run and must remain visible as overall_status=FAILED.
    gate = "DEGRADED" if "NOT_RUN" in statuses else "PASS"
    result = {"status": overall, "gate_status": gate, "items": items,
              "artifacts": [item["path"] for item in items]}
    _write_json(args.out_json, result)
    if args.out_md:
        _write_markdown(args.out_md, "Validation Report", overall, [
            ("G3 Gate", f"gate_status: `{gate}`"), ("Evidence", items),
        ])
    return 0


def cmd_normalize(args):
    try:
        graph = _load_graph(args.input)
        result = {"status": "PASSED", "graph": graph}
    except RuntimeError as exc:
        result = {"status": "NOT_RUN", "reason": str(exc)}
    _write_json(args.out_json, result.get("graph") if result.get("status") == "PASSED" else result)
    _emit_json({key: value for key, value in result.items() if key != "graph"})
    return 0


def _add_report_paths(command):
    command.add_argument("--out-json", required=True)
    command.add_argument("--out-md")


def _add_pair_paths(command):
    command.add_argument("--baseline", required=True)
    command.add_argument("--optimized", required=True)


def _add_normalize_command(subparsers):
    command = subparsers.add_parser("normalize")
    command.add_argument("input")
    command.add_argument("--out-json", required=True)
    command.set_defaults(func=cmd_normalize)


def _add_events_command(subparsers):
    command = subparsers.add_parser("events")
    command.add_argument("--events")
    command.add_argument("--fusion-result")
    command.add_argument("--pass-name", help="只验收指定 pass 的结构化证据")
    command.add_argument(
        "--require-applied", action="store_true", help="要求指定 pass 存在 applied/effect_times > 0",
    )
    _add_report_paths(command)
    command.set_defaults(func=cmd_events)


def _add_graph_command(subparsers):
    command = subparsers.add_parser("graph")
    _add_pair_paths(command)
    command.add_argument("--rules")
    _add_report_paths(command)
    command.set_defaults(func=cmd_graph)


def _add_outputs_command(subparsers):
    command = subparsers.add_parser("outputs")
    _add_pair_paths(command)
    command.add_argument("--atol", type=float)
    command.add_argument("--rtol", type=float)
    command.add_argument(
        "--allow-output-name-diff", action="store_true",
        help="兼容旧调用；名称变化时仍必须提供 --output-map",
    )
    command.add_argument(
        "--output-map", help="JSON 文件，显式声明 {baseline_output_name: optimized_output_name} 映射",
    )
    _add_report_paths(command)
    command.set_defaults(func=cmd_outputs)


def _add_performance_command(subparsers):
    command = subparsers.add_parser("performance")
    _add_pair_paths(command)
    command.add_argument("--baseline-profile")
    command.add_argument("--optimized-profile")
    command.add_argument("--operator-groups", help="JSON: {baseline:[...], optimized:[...]} 的算子类型组")
    _add_report_paths(command)
    command.set_defaults(func=cmd_performance)


def _add_report_command(subparsers):
    command = subparsers.add_parser("report")
    command.add_argument("items", nargs="+")
    _add_report_paths(command)
    command.set_defaults(func=cmd_report)


def _build_parser():
    parser = argparse.ArgumentParser(description="Validate GE fusion pass evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for register in (
        _add_normalize_command, _add_events_command, _add_graph_command,
        _add_outputs_command, _add_performance_command, _add_report_command,
    ):
        register(subparsers)
    return parser


def main():
    args = _build_parser().parse_args()
    exit_code = args.func(args)
    if exit_code:
        raise SystemExit(exit_code)
    return 0


if __name__ == "__main__":
    main()
