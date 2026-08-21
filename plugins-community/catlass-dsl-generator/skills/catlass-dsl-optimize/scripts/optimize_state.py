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

"""Compact, evidence-backed optimization state for one CATLASS DSL kernel."""

import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SCHEMA = "catlass.dsl.optimization.v4"
SAFE_ID = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
RUN_ID = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*-(?P<timestamp>\d{8}-\d{6})$")
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
PROFILE_FILES = ("kernel_details.csv", "step_trace_time.csv")
ANTI_HACK_POLICY = "single-fused-catlass-kernel-v1"


class OptimizeError(ValueError):
    pass


def _validate_anti_hack(value, label, single_observed=False):
    if not isinstance(value, dict) or value.get("status") != "passed":
        raise OptimizeError("{} anti_hack.status=passed 是必需的".format(label))
    if value.get("policy") != ANTI_HACK_POLICY:
        raise OptimizeError("{} anti_hack.policy 非法".format(label))
    declared = value.get("declared_kernel_names")
    observed = value.get("observed_kernel_names")
    if (
        not isinstance(declared, list)
        or not declared
        or any(not isinstance(item, str) or not item for item in declared)
        or len(set(declared)) != len(declared)
    ):
        raise OptimizeError("{} declared_kernel_names 非法".format(label))
    if (
        not isinstance(observed, list)
        or not observed
        or any(not isinstance(item, str) or not item for item in observed)
        or not set(observed).issubset(declared)
        or (single_observed and len(observed) != 1)
    ):
        raise OptimizeError("{} observed_kernel_names 非法".format(label))
    iterations = value.get("profiled_iterations")
    launches = value.get("observed_launches")
    if (
        type(iterations) is not int
        or iterations < 1
        or type(launches) is not int
        or launches != iterations
        or value.get("launches_per_iteration") != 1.0
        or value.get("reason") is not None
    ):
        raise OptimizeError("{} anti_hack launch 证据非法".format(label))
    return value


def _profile_kernel_evidence(trace_root, anti_hack, label):
    trace_root = Path(trace_root)
    direct = trace_root / "kernel_details.csv"
    if not direct.is_file() or direct.is_symlink():
        raise OptimizeError("{} 必须有合并 kernel_details.csv".format(label))
    names = []
    try:
        with direct.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"Name", "Type", "OP State"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise OptimizeError("{} kernel_details 缺少来源字段".format(label))
            for row in reader:
                name = (row.get("Name") or "").strip()
                if (
                    not name
                    or (row.get("Type") or "").strip() != name
                    or (row.get("OP State") or "").strip() not in {"", "N/A"}
                ):
                    raise OptimizeError("{} kernel_details 非 CATLASS 证据".format(label))
                names.append(name)
    except OSError as exc:
        raise OptimizeError("{} kernel_details 不可读".format(label)) from exc
    if (
        len(names) != anti_hack["profiled_iterations"]
        or set(names) != set(anti_hack["observed_kernel_names"])
    ):
        raise OptimizeError("{} kernel_details 与 anti_hack 不一致".format(label))
    manifest_path = trace_root / "anti_hack_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise OptimizeError("{} 缺少 anti_hack_manifest.json".format(label))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OptimizeError("{} anti_hack manifest 非法".format(label)) from exc
    entries = manifest.get("iterations") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("policy") != ANTI_HACK_POLICY
        or manifest.get("profiled_iterations") != anti_hack["profiled_iterations"]
        or not isinstance(entries, list)
        or len(entries) != anti_hack["profiled_iterations"]
    ):
        raise OptimizeError("{} anti_hack manifest 与结果不一致".format(label))
    iteration_files = []
    iteration_names = []
    for expected, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("iteration") != expected:
            raise OptimizeError("{} iteration 顺序非法".format(label))
        relative = entry.get("kernel_details")
        if not isinstance(relative, str) or not relative:
            raise OptimizeError("{} iteration kernel_details 非法".format(label))
        expected_relative = Path("anti_hack") / "iteration-{:04d}".format(expected) / "kernel_details.csv"
        if Path(relative) != expected_relative:
            raise OptimizeError("{} iteration kernel_details 路径非 canonical".format(label))
        unresolved = trace_root / relative
        relative_parts = Path(relative).parts
        if any(
            (trace_root.joinpath(*relative_parts[:index])).is_symlink()
            for index in range(1, len(relative_parts) + 1)
        ):
            raise OptimizeError("{} iteration 证据不能是符号链接".format(label))
        path = unresolved.resolve()
        try:
            path.relative_to(trace_root.resolve())
        except ValueError as exc:
            raise OptimizeError("{} iteration 证据越界".format(label)) from exc
        if path.is_symlink() or not path.is_file():
            raise OptimizeError("{} iteration 证据缺失".format(label))
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
            raise OptimizeError("{} iteration 证据哈希不一致".format(label))
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise OptimizeError("{} 每个 iteration 必须恰好一次 launch".format(label))
        row = rows[0]
        name = (row.get("Name") or "").strip()
        if (
            not name
            or (row.get("Type") or "").strip() != name
            or (row.get("OP State") or "").strip() not in {"", "N/A"}
            or entry.get("observed_launches") != 1
            or entry.get("observed_kernel_names") != [name]
        ):
            raise OptimizeError("{} iteration 非 CATLASS 单 launch 证据".format(label))
        iteration_files.append(path)
        iteration_names.append(name)
    if set(iteration_names) != set(anti_hack["observed_kernel_names"]):
        raise OptimizeError("{} iteration 名称与 anti_hack 不一致".format(label))
    return {
        "kernel_details": direct,
        "manifest": manifest_path,
        "iterations": iteration_files,
    }


def _duplicates_rejected(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise OptimizeError("JSON 包含重复字段：{}".format(key))
        value[key] = item
    return value


def _load_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise OptimizeError("JSON 必须是无符号链接的普通文件：{}".format(path))
    if path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise OptimizeError("JSON 超过大小上限：{}".format(path))
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_duplicates_rejected)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OptimizeError("JSON 非法：{}".format(path)) from exc


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".opt.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path, value, exclusive=False):
    path = Path(path)
    if exclusive and path.exists():
        raise OptimizeError("state 已存在")
    content = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    _atomic_bytes(path, content)


def _is_within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _reject_symlink_chain(path, stop=None):
    current = Path(path).absolute()
    stop = Path(stop).absolute() if stop is not None else None
    while True:
        if current.exists() and current.is_symlink():
            raise OptimizeError("路径不得经过符号链接：{}".format(current))
        if current == current.parent or current == stop:
            return
        current = current.parent


def _validate_run_id(value):
    match = RUN_ID.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise OptimizeError("run_id 必须以 YYYYMMDD-HHMMSS 时间戳结尾")
    try:
        datetime.datetime.strptime(match.group("timestamp"), "%Y%m%d-%H%M%S")
    except ValueError as exc:
        raise OptimizeError("run_id 时间戳不是有效的 UTC 日期时间") from exc


def _policy(metric_path, direction, threshold, max_iterations, stall_threshold,
            min_improvement_fraction, profiling_required):
    if not isinstance(metric_path, str) or not SAFE_ID.fullmatch(metric_path):
        raise OptimizeError("metric_path 必须是安全的点分字段路径")
    if direction not in {"lower", "higher"}:
        raise OptimizeError("direction 必须为 lower 或 higher")
    threshold = float(threshold)
    minimum = float(min_improvement_fraction)
    if not math.isfinite(threshold) or threshold <= 0:
        raise OptimizeError("threshold 必须为有限正数")
    if not 1 <= int(max_iterations) <= 100:
        raise OptimizeError("max_iterations 必须位于 1..100")
    if not 1 <= int(stall_threshold) <= int(max_iterations):
        raise OptimizeError("stall_threshold 必须位于 1..max_iterations")
    if not math.isfinite(minimum) or not 0 <= minimum < 1:
        raise OptimizeError("min_improvement_fraction 必须位于 [0, 1)")
    if profiling_required not in {True, False, "yes", "no"}:
        raise OptimizeError("profiling_required 必须为 yes 或 no")
    return {
        "metric_path": metric_path, "direction": direction, "threshold": threshold,
        "max_iterations": int(max_iterations), "stall_threshold": int(stall_threshold),
        "min_improvement_fraction": minimum,
        "profiling_required": profiling_required in {True, "yes"},
    }


def _config_digest(state):
    keys = ("run_id", "repository_root", "evidence_root", "kernel", "policy", "required_commands")
    value = {key: state.get(key) for key in keys}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_config(state):
    if state.get("config_digest") != _config_digest(state):
        raise OptimizeError("state 固化配置摘要不一致")


def _evidence_root_from_state(state):
    return Path(state["evidence_root"]).resolve()


def _working_tree_state(repository):
    """Fingerprint dirty paths outside generated optimization evidence."""
    completed = subprocess.run(
        [
            "git", "-C", str(repository), "status", "--porcelain=v1", "-z",
            "--untracked-files=all",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise OptimizeError(completed.stderr.decode(errors="replace").strip() or "无法检查工作区修改")
    records = completed.stdout.split(b"\0")
    state = {}
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4:
            raise OptimizeError("无法解析 Git status")
        status = record[:2].decode("ascii", errors="replace")
        path = record[3:].decode("utf-8")
        if "R" in status or "C" in status:
            if index >= len(records) or not records[index]:
                raise OptimizeError("无法解析 Git rename status")
            path = records[index].decode("utf-8")
            index += 1
        if path == ".catlass-dsl" or path.startswith(".catlass-dsl/"):
            continue
        resolved = Path(repository) / path
        state[path] = {
            "status": status,
            "sha256": _sha256(resolved) if resolved.is_file() and not resolved.is_symlink() else None,
        }
    return state


def _verify_only_kernel_changed(state, active):
    repository = _repository_root_from_state(state)
    before = active["working_tree_state"]
    after = _working_tree_state(repository)
    kernel = state["kernel"]["relative_path"]
    paths = set(before) | set(after)
    changed = {path for path in paths if before.get(path) != after.get(path)}
    if changed - {kernel}:
        raise OptimizeError("候选修改越出唯一 kernel path：{}".format(sorted(changed - {kernel})))


def _approved_file(path, evidence_root, label, owner=None):
    resolved = Path(path).resolve()
    if not _is_within(resolved, evidence_root):
        raise OptimizeError("{} 必须位于批准 evidence_directory".format(label))
    if owner is not None and not _is_within(resolved, owner):
        raise OptimizeError("{} 必须位于当前 evidence 目录".format(label))
    _reject_symlink_chain(resolved, evidence_root)
    if resolved.is_symlink() or not resolved.is_file():
        raise OptimizeError("{} 必须是无符号链接的普通文件".format(label))
    if resolved.stat().st_size > MAX_EVIDENCE_BYTES:
        raise OptimizeError("{} 超过大小上限".format(label))
    return resolved


def _record_file(path, repository):
    path = Path(path).resolve()
    return {
        "path": str(path.relative_to(repository)),
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }


def _json_path(value, dotted):
    current = value
    for component in dotted.split("."):
        if not isinstance(current, dict) or component not in current:
            raise OptimizeError("benchmark 缺少 metric_path：{}".format(dotted))
        current = current[component]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise OptimizeError("metric_path 必须指向数值")
    current = float(current)
    if not math.isfinite(current) or current <= 0:
        raise OptimizeError("metric_path 必须指向有限正数")
    return current


def _benchmark_point(path, policy, evidence_root, kernel_sha256, baseline=None, owner=None):
    resolved = _approved_file(path, evidence_root, "benchmark result", owner)
    value = _load_json(resolved)
    if value.get("status") != "passed" or value.get("correctness", {}).get("status") != "passed":
        raise OptimizeError("benchmark 或其 correctness 未通过")
    environment = value.get("environment")
    if not isinstance(environment, dict):
        raise OptimizeError("benchmark environment 非法")
    device = environment.get("device", "")
    if isinstance(device, str) and device.split(":", 1)[0] == "npu":
        _validate_anti_hack(value.get("anti_hack"), "NPU benchmark")
        workloads = value.get("workloads")
        if (
            not isinstance(workloads, list)
            or not workloads
            or any(
                not isinstance(item, dict)
                for item in workloads
            )
        ):
            raise OptimizeError("NPU benchmark workloads 非法")
        for index, item in enumerate(workloads):
            _validate_anti_hack(
                item.get("anti_hack"),
                "NPU workload {}".format(index),
                single_observed=True,
            )
    source_sha = value.get("solution", {}).get("source_sha256")
    if source_sha != kernel_sha256:
        raise OptimizeError("benchmark solution source 与候选 kernel SHA-256 不一致")
    if policy["profiling_required"] and value.get("profiling", {}).get("status") != "passed":
        raise OptimizeError("当前优化配置要求 profiling passed")
    candidate = value.get("performance", {}).get("candidate")
    if not isinstance(candidate, dict):
        raise OptimizeError("benchmark 缺少 candidate performance")
    mean, std = candidate.get("mean_ms"), candidate.get("std_ms")
    if (
        isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(float(mean)) or float(mean) <= 0
        or isinstance(std, bool) or not isinstance(std, (int, float)) or not math.isfinite(float(std)) or float(std) < 0
    ):
        raise OptimizeError("benchmark mean_ms/std_ms 非法")
    config = Path(environment.get("config", ""))
    if config.is_symlink() or not config.is_file():
        raise OptimizeError("benchmark config 不可访问")
    point = {
        "kernel_sha256": kernel_sha256,
        "metric_value": _json_path(value, policy["metric_path"]),
        "mean_ms": float(mean),
        "std_ms": float(std),
        "benchmark_sha256": _sha256(resolved),
        "config": {"path": str(config.resolve()), "sha256": _sha256(config)},
        "environment": {key: environment.get(key) for key in ("device", "arch", "catlass", "cann", "warmup", "trials")},
        "anti_hack": value.get("anti_hack"),
        "source": value,
    }
    if baseline is None:
        point["speedup_vs_baseline"] = 1.0
    else:
        if point["config"]["sha256"] != baseline["config"]["sha256"]:
            raise OptimizeError("candidate benchmark config 与 baseline 不同")
        if point["environment"] != baseline["environment"]:
            raise OptimizeError("candidate benchmark environment 与 baseline 不同")
        point["speedup_vs_baseline"] = baseline["mean_ms"] / point["mean_ms"]
    return point


def _compact_point(point):
    return {key: value for key, value in point.items() if key != "source"}


def _compact_profiles(benchmark, destination, required):
    destination = Path(destination)
    records = []
    found = 0
    workloads = benchmark.get("workloads", [])
    device = benchmark.get("environment", {}).get("device", "")
    anti_hack_required = (
        isinstance(device, str) and device.split(":", 1)[0] == "npu"
    )
    for index, workload in enumerate(workloads):
        timing = workload.get("performance", {}).get("candidate", {})
        trace = timing.get("trace_dir")
        case_record = {"case": index, "uuid": workload.get("uuid")}
        if trace:
            trace_root = Path(trace)
            for name in PROFILE_FILES:
                direct = trace_root / name
                matches = (
                    [direct]
                    if direct.is_file()
                    else sorted(trace_root.glob("**/{}".format(name)))
                    if trace_root.is_dir()
                    else []
                )
                if matches:
                    target = destination / "case-{:04d}".format(index) / name
                    _atomic_bytes(target, matches[0].read_bytes())
                    case_record[name] = str(target.relative_to(destination.parent))
                    found += 1
                else:
                    case_record[name] = None
        else:
            for name in PROFILE_FILES:
                case_record[name] = None
        records.append(case_record)
        if anti_hack_required:
            if not trace:
                raise OptimizeError(
                    "NPU workload {} 缺少 candidate trace_dir".format(index)
                )
            anti_hack = _validate_anti_hack(
                workload.get("anti_hack"),
                "NPU workload {}".format(index),
                single_observed=True,
            )
            _profile_kernel_evidence(
                timing.get("trace_dir"), anti_hack, "NPU workload {}".format(index)
            )
    if anti_hack_required and any(
        not item.get("kernel_details.csv") for item in records
    ):
        raise OptimizeError(
            "NPU anti-hack 要求每个 workload 保留 kernel_details.csv"
        )
    if required and found == 0:
        raise OptimizeError("profiling 未生成 kernel_details.csv 或 step_trace_time.csv")
    return records


def _prune_directory(root, allowed):
    root = Path(root)
    allowed = {Path(item).resolve() for item in allowed}
    if not root.is_dir() or root.is_symlink():
        return
    for child in list(root.iterdir()):
        resolved = child.resolve()
        keep = any(resolved == item or _is_within(item, resolved) for item in allowed)
        if keep:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def _required_correctness_ids(state):
    return set(state["required_commands"])


def _command_evidence(values, state, evidence_root, owner):
    if not isinstance(values, dict) or set(values) != _required_correctness_ids(state):
        raise OptimizeError("command_results 必须恰好覆盖 required commands")
    records = {}
    for command_id in sorted(values):
        item = values[command_id]
        if not isinstance(item, dict) or set(item) != {"status", "evidence"} or item["status"] != "passed":
            raise OptimizeError("required command 未通过：{}".format(command_id))
        path = _approved_file(item["evidence"], evidence_root, "command evidence", owner)
        records[command_id] = {"status": "passed", "sha256": _sha256(path), "size": path.stat().st_size}
    return records


def _baseline_evidence(values, state, evidence_root, owner):
    supplied = {}
    for value in values:
        if not isinstance(value, str) or "=" not in value:
            raise OptimizeError("baseline correctness evidence 必须为 command_id=path")
        command_id, path = value.split("=", 1)
        if command_id in supplied:
            raise OptimizeError("baseline correctness command_id 重复")
        supplied[command_id] = path
    if set(supplied) != _required_correctness_ids(state):
        raise OptimizeError("baseline correctness evidence 必须恰好覆盖 required commands")
    return _command_evidence(
        {key: {"status": "passed", "evidence": value} for key, value in supplied.items()},
        state,
        evidence_root,
        owner,
    )


def _target_met(point, policy):
    return point["metric_value"] <= policy["threshold"] if policy["direction"] == "lower" else point["metric_value"] >= policy["threshold"]


def _improvement(candidate, best, policy):
    best_value = best["metric_value"]
    fraction = ((best_value - candidate["metric_value"]) / best_value if policy["direction"] == "lower" else (candidate["metric_value"] - best_value) / best_value)
    required = max(policy["min_improvement_fraction"], best["std_ms"] / best["mean_ms"], candidate["std_ms"] / candidate["mean_ms"])
    return fraction, required


def _state_view(state):
    keys = ("status", "run_id", "config_digest", "revision", "kernel", "baseline", "best_correct", "active_round", "finalization", "iterations", "stall_count", "reprofile_required", "target_met", "evidence", "next_action")
    view = {key: state.get(key) for key in keys}
    best = state.get("best_correct") or {}
    kernel = state.get("kernel") or {}
    view["best_kernel_path"] = kernel.get("path")
    view["best_kernel_sha256"] = best.get("kernel_sha256")
    return view


def _md(value):
    if value is None:
        return "—"
    return str(value).replace("|", "&#124;").replace("\n", "<br>")


def _iterations_markdown(state):
    lines = [
        "# Iteration Log",
        "",
        "Append a Summary row for every labeled bench. `state.json` is the machine-readable source of truth.",
        "",
        "## Summary",
        "",
        "| Iter | Title | Score | Passed | Notes |",
        "|------|-------|-------|--------|-------|",
    ]
    for item in state.get("iterations", []):
        benchmark = item.get("benchmark")
        if benchmark is None:
            continue
        lines.append(
            "| {} | {} | {:.4f}x | {} | {}: {} |".format(
                item["round"], _md(item["axis_id"]), benchmark["speedup_vs_baseline"],
                "yes" if item["correctness_status"] == "passed" else "no",
                _md(item["decision"]), _md(item["reason"]),
            )
        )
    lines.extend(["", "## Notes", ""])
    for item in state.get("iterations", []):
        proposal = item.get("proposal", {})
        lines.append("- Iter {} Expected: {} {} Falsified when: {}".format(
            item["round"], _md(proposal.get("hypothesis")), _md(proposal.get("expected_effect")), _md(proposal.get("falsification_condition"))
        ))
    active = state.get("active_round")
    if active:
        proposal = active["proposal"]
        lines.append("- Iter {} Expected: {} {} Falsified when: {}".format(
            active["round"], _md(proposal["hypothesis"]), _md(proposal["expected_effect"]), _md(proposal["falsification_condition"])
        ))
    final = state.get("finalization") or {}
    verification = final.get("verification")
    if verification:
        accepted = [item["axis_id"] for item in state["iterations"] if item["decision"] == "accepted"]
        rejected = [item["axis_id"] for item in state["iterations"] if item["decision"] != "accepted"]
        benchmark = verification["benchmark"]
        lines.extend([
            "",
            "### Session synthesis",
            "",
            "- Final kernel SHA-256: `{}`".format(state["best_correct"]["kernel_sha256"]),
            "- Baseline/final mean: {} ms → {} ms ({:.4f}x).".format(state["baseline"]["mean_ms"], benchmark["mean_ms"], benchmark["speedup_vs_baseline"]),
            "- Effective directions: {}.".format(", ".join(accepted) or "none"),
            "- Dead ends to skip: {}.".format(", ".join(rejected) or "none"),
            "- Remaining bottlenecks: {}".format(_md(verification["remaining_bottlenecks"])),
            "- Next session: {}".format(_md(verification["next_steps"])),
        ])
    return "\n".join(lines) + "\n"


def _write_iterations(state):
    _atomic_bytes(state["iterations_path"], _iterations_markdown(state).encode())


class LockedState:
    def __init__(self, path, expected_revision):
        self.path = Path(path)
        self.expected_revision = expected_revision
        self.lock = None

    def __enter__(self):
        if self.path.is_symlink() or not self.path.is_file():
            raise OptimizeError("state 不存在或 identity 非法")
        self.lock = os.open(str(self.path.parent), os.O_RDONLY)
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        state = _load_json(self.path)
        if state.get("schema") != SCHEMA:
            raise OptimizeError("state schema 非法")
        _validate_config(state)
        if state.get("revision") != self.expected_revision:
            raise OptimizeError("stale expected revision")
        self.state = state
        return state

    def __exit__(self, exc_type, exc, traceback):
        if self.lock is not None:
            os.close(self.lock)


def _save(state_path, state):
    state["revision"] += 1
    _write_iterations(state)
    _atomic_json(state_path, state)
    return _state_view(state)


def initialize(state_path, run_id, repository_root, kernel_path, baseline_path, evidence,
               expected_revision, metric_path="performance.candidate.mean_ms", direction="lower",
               threshold=1.0, max_iterations=30, stall_threshold=3,
               min_improvement_fraction=0.0, profiling_required="no",
               required_commands=("full-test",)):
    if expected_revision != 0:
        raise OptimizeError("initialize expected_revision 必须为 0")
    _validate_run_id(run_id)
    repository_root = Path(repository_root).resolve()
    if repository_root.is_symlink() or not repository_root.is_dir():
        raise OptimizeError("repository_root 必须是无符号链接的目录")
    completed = subprocess.run(["git", "-C", str(repository_root), "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
    if completed.returncode or Path(completed.stdout.strip()).resolve() != repository_root:
        raise OptimizeError("repository_root 必须是 Git 工作区根目录")
    evidence_root = Path(state_path).resolve().parent
    expected_root = repository_root / ".catlass-dsl/optimize-runs" / run_id
    if evidence_root != expected_root or evidence_root.name != run_id:
        raise OptimizeError("evidence_directory 必须为 .catlass-dsl/optimize-runs/<run-id>")
    state_path = Path(state_path).resolve()
    if state_path != evidence_root / "state.json":
        raise OptimizeError("state 必须位于 run 根目录的 state.json")
    if state_path.exists():
        raise OptimizeError("state 已存在")
    _reject_symlink_chain(evidence_root, repository_root)
    evidence_root.mkdir(parents=True, exist_ok=True)
    kernel_path = Path(kernel_path)
    kernel_path = (repository_root / kernel_path).resolve() if not kernel_path.is_absolute() else kernel_path.resolve()
    if not _is_within(kernel_path, repository_root):
        raise OptimizeError("kernel path 越出 repository_root")
    _reject_symlink_chain(kernel_path, repository_root)
    if kernel_path.is_symlink() or not kernel_path.is_file():
        raise OptimizeError("kernel 必须是无符号链接的普通文件")
    kernel_relative = str(kernel_path.relative_to(repository_root))
    commands = list(required_commands)
    if not commands or len(commands) != len(set(commands)) or not all(isinstance(item, str) and SAFE_ID.fullmatch(item) for item in commands):
        raise OptimizeError("required_commands 必须是非空、不重复的安全 command ID")
    policy = _policy(metric_path, direction, threshold, max_iterations, stall_threshold, min_improvement_fraction, profiling_required)
    kernel_sha = _sha256(kernel_path)
    baseline_dir = evidence_root / "baseline"
    seed = {"required_commands": commands}
    records = _baseline_evidence(evidence, seed, evidence_root, baseline_dir)
    point = _benchmark_point(baseline_path, policy, evidence_root, kernel_sha, owner=baseline_dir)
    point = _compact_point(point)
    baseline_kernel = baseline_dir / "kernel.py"
    _atomic_bytes(baseline_kernel, kernel_path.read_bytes())
    point.update({"kernel_snapshot": str(baseline_kernel), "trace_id": "baseline", "correctness_evidence": records})
    _atomic_json(baseline_dir / "result.json", point)
    _prune_directory(baseline_dir, [baseline_kernel, baseline_dir / "result.json"])
    state = {
        "schema": SCHEMA, "status": "running", "revision": 0, "run_id": run_id,
        "repository_root": str(repository_root), "evidence_root": str(evidence_root),
        "kernel": {"path": str(kernel_path), "relative_path": kernel_relative},
        "required_commands": commands,
        "policy": policy, "baseline": point, "best_correct": dict(point),
        "active_round": None, "iterations": [], "stall_count": 0,
        "reprofile_required": False, "reprofile_used": False, "post_reprofile_round": False,
        "target_met": _target_met(point, policy), "finalization": None,
        "evidence": [], "next_action": "begin_finalization" if _target_met(point, policy) else "begin_round",
        "iterations_path": str(evidence_root / "ITERATIONS.md"),
    }
    state["config_digest"] = _config_digest(state)
    _write_iterations(state)
    _atomic_json(state_path, state, exclusive=True)
    return _state_view(state)


def _load_proposal(path, evidence_root, state):
    proposal = _load_json(path)
    required = {"hypothesis", "axis_id", "expected_effect", "falsification_condition", "knowledge_sources", "profile_evidence"}
    if set(proposal) != required:
        raise OptimizeError("proposal 字段集合非法")
    for key in ("hypothesis", "expected_effect", "falsification_condition"):
        if not isinstance(proposal[key], str) or not proposal[key].strip():
            raise OptimizeError("proposal {} 不能为空".format(key))
    if not isinstance(proposal["axis_id"], str) or not SAFE_ID.fullmatch(proposal["axis_id"]):
        raise OptimizeError("axis_id 非法")
    if not isinstance(proposal["knowledge_sources"], list) or not all(isinstance(item, str) and item for item in proposal["knowledge_sources"]):
        raise OptimizeError("knowledge_sources 必须是字符串列表")
    if not isinstance(proposal["profile_evidence"], list):
        raise OptimizeError("profile_evidence 必须是列表")
    profile_records = []
    for value in proposal["profile_evidence"]:
        path = _approved_file(value, evidence_root, "profile evidence")
        profile_records.append(_record_file(path, _repository_root_from_state(state)))
    if not proposal["knowledge_sources"] and not profile_records:
        raise OptimizeError("hypothesis 必须引用 profile 或 knowledge source")
    proposal = dict(proposal)
    proposal["profile_evidence"] = profile_records
    return proposal


def _repository_root_from_state(state):
    return Path(state["repository_root"]).resolve()


def begin_round(state_path, expected_revision, proposal_path):
    with LockedState(state_path, expected_revision) as state:
        evidence_root = _evidence_root_from_state(state)
        if state["active_round"] is not None or state["next_action"] not in {"begin_round", "reprofile"}:
            raise OptimizeError("当前状态不能开始 round")
        proposal = _load_proposal(proposal_path, evidence_root, state)
        profiles = proposal["profile_evidence"]
        if state["reprofile_required"]:
            if not profiles:
                raise OptimizeError("停滞后必须提供 fresh profiling")
            known = {item["sha256"] for item in state["evidence"]}
            if any(item["sha256"] in known for item in profiles):
                raise OptimizeError("停滞后必须提供未使用过的 fresh profiling")
            if state["iterations"] and proposal["axis_id"] == state["iterations"][-1]["axis_id"]:
                raise OptimizeError("reprofile 后必须切换 axis_id")
            state["reprofile_required"] = False
            state["reprofile_used"] = True
            state["post_reprofile_round"] = True
            state["stall_count"] = 0
        number = len(state["iterations"]) + 1
        if number > state["policy"]["max_iterations"]:
            raise OptimizeError("优化预算已耗尽")
        trace_dir = evidence_root / "traces" / "iter-{:03d}-{}".format(number, proposal["axis_id"])
        if trace_dir.exists():
            raise OptimizeError("trace 目录已存在")
        trace_dir.mkdir(parents=True)
        best_snapshot = Path(state["best_correct"]["kernel_snapshot"])
        kernel_path = Path(state["kernel"]["path"])
        _atomic_bytes(kernel_path, best_snapshot.read_bytes())
        _atomic_json(trace_dir / "proposal.json", proposal)
        proposal_input = Path(proposal_path).resolve()
        if _is_within(proposal_input, evidence_root) and proposal_input != trace_dir / "proposal.json":
            proposal_input.unlink()
        state["active_round"] = {
            "round": number, "trace_id": "iter-{:03d}-{}".format(number, proposal["axis_id"]),
            "trace_dir": str(trace_dir), "parent_kernel_sha256": state["best_correct"]["kernel_sha256"],
            "proposal": proposal, "working_tree_state": _working_tree_state(_repository_root_from_state(state)),
        }
        state["evidence"].extend(profiles)
        state["next_action"] = "run_candidate"
        return _save(state_path, state)


def record_round(state_path, expected_revision, result_path):
    submission = _load_json(result_path)
    if set(submission) != {"round", "correctness_status", "command_results", "benchmark_result"}:
        raise OptimizeError("round result 字段集合非法")
    with LockedState(state_path, expected_revision) as state:
        evidence_root = _evidence_root_from_state(state)
        active = state["active_round"]
        if active is None or state["next_action"] != "run_candidate" or submission["round"] != active["round"]:
            raise OptimizeError("没有匹配的 active round")
        trace_dir = Path(active["trace_dir"])
        _approved_file(result_path, evidence_root, "round submission", trace_dir)
        status = submission["correctness_status"]
        if status not in {"passed", "failed", "blocked", "not_run"}:
            raise OptimizeError("correctness_status 非法")
        kernel_path = Path(state["kernel"]["path"])
        if kernel_path.is_symlink() or not kernel_path.is_file():
            raise OptimizeError("kernel identity 非法")
        _verify_only_kernel_changed(state, active)
        candidate_sha = _sha256(kernel_path)
        if candidate_sha == active["parent_kernel_sha256"]:
            raise OptimizeError("候选 kernel 没有发生变化")
        kernel_snapshot = trace_dir / "kernel.py"
        _atomic_bytes(kernel_snapshot, kernel_path.read_bytes())
        command_results, candidate, profiles = {}, None, []
        decision, reason, accepted = "rejected", status, False
        if status == "passed":
            command_results = _command_evidence(submission["command_results"], state, evidence_root, trace_dir)
            if not submission["benchmark_result"]:
                raise OptimizeError("正确候选必须提供 benchmark result")
            candidate = _benchmark_point(submission["benchmark_result"], state["policy"], evidence_root, candidate_sha, state["baseline"], trace_dir)
            profiles = _compact_profiles(candidate["source"], trace_dir / "profile", state["policy"]["profiling_required"])
            candidate = _compact_point(candidate)
            fraction, required = _improvement(candidate, state["best_correct"], state["policy"])
            accepted = fraction >= required
            decision = "accepted" if accepted else "rejected"
            reason = "improvement {:.8f} {} required {:.8f}".format(fraction, ">=" if accepted else "<", required)
        else:
            if submission["benchmark_result"] is not None:
                raise OptimizeError("未通过正确性时不得提供 benchmark")
            if submission["command_results"] not in ({}, None):
                raise OptimizeError("未通过正确性时 command_results 必须为空")
        iteration = {
            "round": active["round"], "trace_id": active["trace_id"], "axis_id": active["proposal"]["axis_id"],
            "parent_kernel_sha256": active["parent_kernel_sha256"], "candidate_kernel_sha256": candidate_sha,
            "kernel_snapshot": str(kernel_snapshot), "correctness_status": status,
            "command_results": command_results, "benchmark": candidate, "profile": profiles,
            "decision": decision, "reason": reason, "proposal": active["proposal"],
        }
        _atomic_json(trace_dir / "result.json", iteration)
        _prune_directory(trace_dir, [trace_dir / "proposal.json", kernel_snapshot, trace_dir / "result.json", trace_dir / "profile"])
        state["iterations"].append(iteration)
        state["active_round"] = None
        if accepted:
            best = dict(candidate)
            best.update({"kernel_snapshot": str(kernel_snapshot), "trace_id": active["trace_id"]})
            state["best_correct"] = best
            state["stall_count"] = 0
        else:
            state["stall_count"] += 1
            _atomic_bytes(kernel_path, Path(state["best_correct"]["kernel_snapshot"]).read_bytes())
        state["post_reprofile_round"] = False
        state["target_met"] = _target_met(state["best_correct"], state["policy"])
        if state["target_met"]:
            state["next_action"] = "begin_finalization"
        elif len(state["iterations"]) >= state["policy"]["max_iterations"]:
            state["status"] = "failed"
            state["next_action"] = "begin_finalization"
        elif state["stall_count"] >= state["policy"]["stall_threshold"]:
            state["reprofile_required"] = True
            state["next_action"] = "reprofile"
        else:
            state["next_action"] = "begin_round"
        return _save(state_path, state)


def begin_finalization(state_path, expected_revision):
    with LockedState(state_path, expected_revision) as state:
        evidence_root = _evidence_root_from_state(state)
        if state["next_action"] != "begin_finalization" or state["finalization"] is not None:
            raise OptimizeError("当前状态不能开始 finalization")
        kernel_path = Path(state["kernel"]["path"])
        snapshot = Path(state["best_correct"]["kernel_snapshot"])
        _atomic_bytes(kernel_path, snapshot.read_bytes())
        final_dir = evidence_root / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_kernel = final_dir / "kernel.py"
        _atomic_bytes(final_kernel, snapshot.read_bytes())
        state["finalization"] = {
            "kernel_sha256": state["best_correct"]["kernel_sha256"],
            "kernel_snapshot": str(final_kernel), "evidence_dir": str(final_dir),
        }
        state["next_action"] = "verify_final"
        return _save(state_path, state)


def finalize(state_path, expected_revision, result_path):
    submission = _load_json(result_path)
    required = {"command_results", "benchmark_result", "remaining_bottlenecks", "next_steps"}
    if set(submission) != required:
        raise OptimizeError("final result 字段集合非法")
    if not all(isinstance(submission[key], str) and submission[key].strip() for key in ("remaining_bottlenecks", "next_steps")):
        raise OptimizeError("final synthesis 字段不能为空")
    with LockedState(state_path, expected_revision) as state:
        evidence_root = _evidence_root_from_state(state)
        final = state["finalization"]
        if final is None or state["next_action"] != "verify_final":
            raise OptimizeError("没有待验证的 finalization")
        final_dir = Path(final["evidence_dir"])
        _approved_file(result_path, evidence_root, "final submission", final_dir)
        kernel_path = Path(state["kernel"]["path"])
        if _sha256(kernel_path) != state["best_correct"]["kernel_sha256"]:
            raise OptimizeError("真实 kernel 与 best snapshot 不一致")
        command_results = _command_evidence(submission["command_results"], state, evidence_root, final_dir)
        point = _benchmark_point(submission["benchmark_result"], state["policy"], evidence_root, state["best_correct"]["kernel_sha256"], state["baseline"], final_dir)
        if point["benchmark_sha256"] in {
            item["benchmark"].get("benchmark_sha256")
            for item in state["iterations"]
            if item.get("benchmark")
        }:
            raise OptimizeError("finalization 必须使用 fresh benchmark result")
        profiles = _compact_profiles(point["source"], final_dir / "profile", state["policy"]["profiling_required"])
        point = _compact_point(point)
        target_met = _target_met(point, state["policy"])
        verification = {
            "command_results": command_results, "benchmark": point, "profile": profiles,
            "target_met": target_met, "remaining_bottlenecks": submission["remaining_bottlenecks"],
            "next_steps": submission["next_steps"],
        }
        _atomic_json(final_dir / "result.json", verification)
        _prune_directory(final_dir, [final_dir / "kernel.py", final_dir / "result.json", final_dir / "profile"])
        state["target_met"] = target_met
        state["status"] = "passed" if target_met else "failed"
        state["next_action"] = "complete"
        state["finalization"]["verification"] = verification
        return _save(state_path, state)


def status(state_path):
    state = _load_json(state_path)
    if state.get("schema") != SCHEMA:
        raise OptimizeError("state schema 非法")
    _validate_config(state)
    view = _state_view(state)
    kernel = Path(state["kernel"]["path"])
    actual = _sha256(kernel) if kernel.is_file() and not kernel.is_symlink() else None
    expected = (state["active_round"] or {}).get("parent_kernel_sha256", state["best_correct"]["kernel_sha256"])
    view["kernel_status"] = {"actual_sha256": actual, "expected_sha256": expected, "matches_expected": actual == expected}
    return view


def _expected(parser):
    parser.add_argument("--state", required=True)
    parser.add_argument("--expected-revision", required=True, type=int)


def build_parser():
    parser = argparse.ArgumentParser(description="CATLASS DSL compact single-kernel optimizer")
    commands = parser.add_subparsers(dest="command", required=True)
    initialize_parser = commands.add_parser("initialize")
    _expected(initialize_parser)
    initialize_parser.add_argument("--run-id", required=True)
    initialize_parser.add_argument("--repository-root", required=True)
    initialize_parser.add_argument("--kernel", required=True)
    initialize_parser.add_argument("--metric-path", default="performance.candidate.mean_ms")
    initialize_parser.add_argument("--direction", choices=("lower", "higher"), default="lower")
    initialize_parser.add_argument("--threshold", required=True, type=float)
    initialize_parser.add_argument("--max-iterations", type=int, default=30)
    initialize_parser.add_argument("--stall-threshold", type=int, default=3)
    initialize_parser.add_argument("--min-improvement-fraction", type=float, default=0.0)
    initialize_parser.add_argument("--profiling-required", choices=("yes", "no"), default="no")
    initialize_parser.add_argument("--required-command", action="append", required=True)
    initialize_parser.add_argument("--baseline-result", required=True)
    initialize_parser.add_argument("--correctness-evidence", action="append", default=[], required=True)
    begin = commands.add_parser("begin-round")
    _expected(begin)
    begin.add_argument("--proposal", required=True)
    record = commands.add_parser("record-round")
    _expected(record)
    record.add_argument("--result", required=True)
    final_begin = commands.add_parser("begin-finalization")
    _expected(final_begin)
    final = commands.add_parser("finalize")
    _expected(final)
    final.add_argument("--result", required=True)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--state", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "initialize":
            output = initialize(
                args.state, args.run_id, args.repository_root, args.kernel,
                args.baseline_result, args.correctness_evidence, args.expected_revision,
                args.metric_path, args.direction, args.threshold, args.max_iterations,
                args.stall_threshold, args.min_improvement_fraction,
                args.profiling_required, args.required_command,
            )
        elif args.command == "begin-round":
            output = begin_round(args.state, args.expected_revision, args.proposal)
        elif args.command == "record-round":
            output = record_round(args.state, args.expected_revision, args.result)
        elif args.command == "begin-finalization":
            output = begin_finalization(args.state, args.expected_revision)
        elif args.command == "finalize":
            output = finalize(args.state, args.expected_revision, args.result)
        else:
            output = status(args.state)
    except (OSError, OptimizeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
